# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validation-only worker: replay each real V1 batch through the SC lifecycle.

Use the existing worker-extension hook; production sources stay unchanged.
The launch driver must register this actor with the AutoModel interpreter.
"""

import copy
import hashlib
import inspect
import json
import os
import random
from pathlib import Path
from unittest.mock import patch

import numpy as np
import ray
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torch.utils._pytree import tree_map
from transformer_engine.pytorch.optimizers import FusedAdam

from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker
from nemo_rl.models.policy.workers import automodel_policy_worker as module
from nemo_rl.models.policy.workers.automodel_policy_worker import (
    AutomodelPolicyWorkerImpl,
)


def local(value):
    return value.to_local() if isinstance(value, DTensor) else value


def cpu_copy(value):
    return tree_map(
        lambda x: local(x).detach().cpu().clone()
        if isinstance(x, torch.Tensor)
        else copy.deepcopy(x),
        value,
    )


def rng_state():
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }


def restore_rng(state):
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])


def model_tensors(worker):
    return dict(
        list(worker.model.named_parameters()) + list(worker.model.named_buffers())
    )


def gates(worker):
    return {
        name: item
        for name, item in worker.model.named_modules()
        if isinstance(item, module.Gate)
    }


def optimizer_params(worker):
    return [p for group in worker.optimizer.param_groups for p in group["params"]]


def snapshot(worker):
    """Capture raw BF16 moments and int16 master remainders without conversion."""
    optimizer = worker.optimizer
    params = optimizer_params(worker)
    return {
        "model": cpu_copy(model_tensors(worker)),
        "optimizer": {
            "state": {
                i: cpu_copy(optimizer.state[p])
                for i, p in enumerate(params)
                if p in optimizer.state
            },
            "scales": {
                i: cpu_copy(optimizer._scales[p])
                for i, p in enumerate(params)
                if p in optimizer._scales
            },
            "groups": [
                cpu_copy({k: v for k, v in group.items() if k != "params"})
                for group in optimizer.param_groups
            ],
            "overflow": cpu_copy(optimizer._dummy_overflow_buf),
        },
        "scheduler": copy.deepcopy(worker.scheduler.state_dict()),
        "router_loads": {
            name: cpu_copy(gate._cumulative_expert_load)
            for name, gate in gates(worker).items()
        },
    }


@torch.no_grad()
def restore(worker, saved):
    for name, value in model_tensors(worker).items():
        local(value).copy_(saved["model"][name])
    optimizer = worker.optimizer
    params = optimizer_params(worker)
    # Reuse already allocated DTensor state from the V1 update, copying only
    # local storage. Preserve the original mesh, placements, dtype and layout.
    for index, param in enumerate(params):
        if index not in saved["optimizer"]["state"]:
            optimizer.state.pop(param, None)
        else:
            current = optimizer.state[param]
            expected = saved["optimizer"]["state"][index]
            assert current.keys() == expected.keys()
            for name, value in expected.items():
                local(current[name]).copy_(value)
        if index not in saved["optimizer"]["scales"]:
            optimizer._scales.pop(param, None)
        else:
            for name, value in saved["optimizer"]["scales"][index].items():
                optimizer._scales[param][name].copy_(value)
    for current, expected in zip(
        optimizer.param_groups, saved["optimizer"]["groups"], strict=True
    ):
        parameters = current["params"]
        current.clear()
        current.update(copy.deepcopy(expected))
        current["params"] = parameters
    optimizer._dummy_overflow_buf.copy_(saved["optimizer"]["overflow"])
    worker.scheduler.load_state_dict(copy.deepcopy(saved["scheduler"]))
    for name, gate in gates(worker).items():
        value = saved["router_loads"][name]
        gate._cumulative_expert_load = None if value is None else value.to("cuda")
    optimizer.zero_grad()


def compare(actual, expected, path="", *, exact=False):
    """Visit every leaf, recording errors without letting one rank exit early."""
    result = {"max_abs": 0.0, "different": 0, "elements": 0, "failures": []}

    def visit(a, b, name):
        if isinstance(b, (torch.Tensor, np.ndarray)):
            a = (
                local(a).detach().cpu()
                if isinstance(a, torch.Tensor)
                else torch.as_tensor(a)
            )
            b = (
                local(b).detach().cpu()
                if isinstance(b, torch.Tensor)
                else torch.as_tensor(b)
            )
            if a.shape != b.shape or a.dtype != b.dtype:
                result["failures"].append(name + ": shape/dtype")
                return
            if a.is_floating_point() and (
                not torch.isfinite(a).all() or not torch.isfinite(b).all()
            ):
                result["failures"].append(name + ": nonfinite")
                return
            result["elements"] += b.numel()
            if torch.equal(a, b):
                return
            # Chunk on CPU so comparisons don't materialize full FP64 copies of
            # large embeddings or MoE tensors.
            failed_tolerance = False
            for av, bv in zip(
                a.reshape(-1).split(1_000_000),
                b.reshape(-1).split(1_000_000),
                strict=True,
            ):
                result["different"] += int((av != bv).sum())
                if av.numel():
                    result["max_abs"] = max(
                        result["max_abs"],
                        float((av.double() - bv.double()).abs().max()),
                    )
                rtol, atol = (
                    (0, 0) if exact or not b.is_floating_point() else (1e-5, 1e-7)
                )
                if (
                    not torch.allclose(av, bv, rtol=rtol, atol=atol)
                    and not failed_tolerance
                ):
                    result["failures"].append(name + ": tolerance")
                    failed_tolerance = True
        elif isinstance(b, dict):
            if not isinstance(a, dict) or a.keys() != b.keys():
                result["failures"].append(name + ": keys")
                return
            for key in b:
                visit(a[key], b[key], f"{name}/{key}")
        elif isinstance(b, (tuple, list)):
            if len(a) != len(b):
                result["failures"].append(name + ": length")
                return
            for index, (av, bv) in enumerate(zip(a, b, strict=True)):
                visit(av, bv, f"{name}/{index}")
        elif isinstance(b, (float, np.floating)):
            visit(
                torch.tensor(a, dtype=torch.float64),
                torch.tensor(b, dtype=torch.float64),
                name,
            )
        elif a != b:
            result["failures"].append(name + ": value")

    visit(actual, expected, path)
    result["failures"] = result["failures"][:20]
    return result


def digest_batch(batch):
    digest = hashlib.sha256()
    for name, value in sorted(batch.items()):
        digest.update(name.encode())
        if isinstance(value, torch.Tensor):
            tensor = local(value).detach().cpu().contiguous()
            digest.update(str((tensor.dtype, tensor.shape)).encode())
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        else:
            digest.update(repr(value).encode())
    return digest.hexdigest()


class ReplayWorkerImpl(AutomodelPolicyWorkerImpl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.replay_step = 0
        self.replay_output = Path(os.environ["SC_REPLAY_OUTPUT"])
        self.replay_rank = dist.get_rank()
        self.replay_output.mkdir(parents=True, exist_ok=True)
        assert isinstance(self.optimizer, FusedAdam)
        assert self.optimizer.master_weights and self.optimizer.store_param_remainders
        assert self.optimizer.exp_avg_dtype == torch.bfloat16
        assert self.optimizer.exp_avg_sq_dtype == torch.bfloat16
        assert not self.optimizer.capturable
        self.replay_strict = os.environ.get("SC_REPLAY_STRICT") == "1"
        if self.replay_strict:
            assert self.cfg["automodel_cfg"]["deterministic"]
            assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
            assert os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] == "0"
            # NeMo RL's warn-only flag permits nondeterministic cuDNN attention.
            # This is a validation-only override, shared by V1 and SC.
            torch.use_deterministic_algorithms(True, warn_only=False)
        if self.replay_rank == 0:
            (self.replay_output / "fused_adam_runtime.py").write_text(
                inspect.getsource(FusedAdam)
            )

    def train(
        self,
        data,
        loss_fn,
        eval_mode=False,
        gbs=None,
        mbs=None,
        check_dim_skip_keys=None,
    ):
        if eval_mode:
            return super().train(
                data, loss_fn, eval_mode, gbs, mbs, check_dim_skip_keys
            )
        self.replay_step += 1
        label = f"step-{self.replay_step:03d}-rank-{self.replay_rank:02d}"
        print(f"REPLAY_BEGIN {label}", flush=True)
        # Persist exactly what Policy.train delivered, before either path mutates it.
        recorded = copy.deepcopy(data).to("cpu")
        before_rng = rng_state()
        before = snapshot(self)
        initial_hash = digest_batch(recorded)
        torch.save(
            {
                "batch": recorded,
                "rng": before_rng,
                "gbs": gbs,
                "mbs": mbs,
                "loss_fn": loss_fn,
                "check_dim_skip_keys": check_dim_skip_keys,
            },
            self.replay_output / f"{label}-input.pt",
        )
        if self.replay_step == 1:
            torch.save(
                {
                    "state": before,
                    "config": self.cfg,
                    "parameter_names": list(model_tensors(self)),
                },
                self.replay_output / f"initial-rank-{self.replay_rank:02d}.pt",
            )

        original_clip = module.scale_grads_and_clip_grad_norm
        observed = {}
        active = "v1"

        def clip(*args, **kwargs):
            observed[active] = cpu_copy(
                {name: param.grad for name, param in self.model.named_parameters()}
            )
            return original_clip(*args, **kwargs)

        # Restore RNG after capture/serialization, immediately before each call.
        restore_rng(before_rng)
        with patch.object(module, "scale_grads_and_clip_grad_norm", side_effect=clip):
            v1_metrics = super().train(
                copy.deepcopy(recorded), loss_fn, False, gbs, mbs, check_dim_skip_keys
            )
        v1_after = snapshot(self)
        v1_rng = rng_state()
        restore(self, before)
        restore_check = compare(snapshot(self), before, "restore", exact=True)
        restore_rng(before_rng)
        active = "sc"
        with patch.object(module, "scale_grads_and_clip_grad_norm", side_effect=clip):
            super().begin_train_step(loss_fn, gbs=gbs, mbs=mbs)
            super().train_microbatch(copy.deepcopy(recorded))
            sc_metrics = super().finish_train_step()
        sc_rng = rng_state()
        sc_after = snapshot(self)
        comparisons = {
            "restore": restore_check,
            "gradients": compare(observed["sc"], observed["v1"], "gradients"),
            "metrics": compare(sc_metrics, v1_metrics, "metrics"),
            "model": compare(sc_after["model"], v1_after["model"], "model"),
            "optimizer": compare(
                sc_after["optimizer"], v1_after["optimizer"], "optimizer"
            ),
            "scheduler": compare(
                sc_after["scheduler"], v1_after["scheduler"], "scheduler", exact=True
            ),
            "router_loads": compare(
                sc_after["router_loads"], v1_after["router_loads"], "router_loads"
            ),
            "rng": compare(sc_rng, v1_rng, "rng", exact=True),
        }
        assert digest_batch(recorded) == initial_hash
        # Verify this was substantive training, including initialized moments.
        assert sc_after["optimizer"]["state"]
        changed_parameters = sum(
            int(torch.count_nonzero(v1_after["model"][name] != before["model"][name]))
            for name, _ in self.model.named_parameters()
        )
        record = {
            "step": self.replay_step,
            "rank": self.replay_rank,
            "input_sha256": initial_hash,
            "rows": recorded.size,
            "keys": sorted(recorded.keys()),
            "rtol": 1e-5,
            "atol": 1e-7,
            "masked_rows": int((recorded["sample_mask"] == 0).sum()),
            "nonzero_advantages": int(torch.count_nonzero(recorded["advantages"])),
            "parameter_elements_changed": changed_parameters,
            "strict_determinism": self.replay_strict,
            "comparisons": comparisons,
            "passed": not any(item["failures"] for item in comparisons.values()),
        }
        torch.save(
            {"v1": v1_metrics, "sc": sc_metrics},
            self.replay_output / f"{label}-metrics.pt",
        )
        (self.replay_output / f"{label}.json").write_text(json.dumps(record, indent=2))
        failed = torch.tensor(int(not record["passed"]), device="cuda")
        dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=self.dp_mesh.get_group())
        if failed.item():
            torch.save(
                {"state": before}, self.replay_output / f"{label}-failed-prestate.pt"
            )
            raise AssertionError(
                f"Worker numerical replay failed: {label}; see per-rank JSON"
            )
        print(
            f"REPLAY_PASS {label} "
            + json.dumps({key: value["max_abs"] for key, value in comparisons.items()}),
            flush=True,
        )
        # Continue the V1 producer trajectory exactly, even if SC was only close.
        restore(self, v1_after)
        restore_rng(v1_rng)
        # Standard metrics retain their names. Add clearly identified test results.
        for name, comparison in comparisons.items():
            v1_metrics["all_mb_metrics"][f"replay_{name}_max_abs"] = [
                comparison["max_abs"]
            ]
        v1_metrics["all_mb_metrics"]["replay_passed"] = [1.0]
        return v1_metrics


@ray.remote(runtime_env=get_runtime_env_for_policy_worker("automodel_policy_worker"))
class ReplayWorker(ReplayWorkerImpl):  # pragma: no cover
    pass
