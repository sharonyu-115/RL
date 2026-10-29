# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replay a saved real batch through V1 twice and SC to diagnose differences."""

import argparse
import copy
import json
import os
from pathlib import Path
from unittest.mock import patch

import ray
import torch
import torch.distributed as dist

from automodel_sc_real_replay_worker import (
    compare,
    cpu_copy,
    restore,
    restore_rng,
    rng_state,
    snapshot,
)
from nemo_rl.models.policy.workers import automodel_policy_worker as module

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--deterministic", action="store_true")
parser.add_argument("--strict-deterministic", action="store_true")
args = parser.parse_args()
if args.strict_deterministic:
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
    assert os.environ.get("NVTE_ALLOW_NONDETERMINISTIC_ALGO") == "0"
    original_enable_deterministic = module.enable_deterministic_algorithms

    def enable_strict_determinism():
        original_enable_deterministic()
        torch.use_deterministic_algorithms(True, warn_only=False)

    module.enable_deterministic_algorithms = enable_strict_determinism
rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
initial = torch.load(args.input / f"initial-rank-{rank:02d}.pt", weights_only=False)
batch = torch.load(
    args.input / f"step-001-rank-{rank:02d}-input.pt", weights_only=False
)
assert len(batch["rng"]["cuda"]) == 1, "Recorded Ray actors each owned one GPU"
# torchrun exposes all local GPUs; map the actor's CUDA RNG to each device.
batch["rng"]["cuda"] = [
    batch["rng"]["cuda"][0].clone() for _ in range(torch.cuda.device_count())
]
cfg = initial["config"]
cfg["automodel_cfg"]["deterministic"] = args.deterministic or args.strict_deterministic
with patch.object(ray, "get_gpu_ids", return_value=[local_rank]):
    worker = module.AutomodelPolicyWorkerImpl(cfg, init_reference_model=False)
worker.prepare_for_training()
before = initial["state"]
restore(worker, before)
assert not compare(snapshot(worker), before, exact=True)["failures"]
args.output.mkdir(parents=True, exist_ok=True)
reference = None
reports = {}
original_clip = module.scale_grads_and_clip_grad_norm
for mode in ("v1", "v1_repeat", "sc"):
    restore(worker, before)
    assert not compare(snapshot(worker), before, exact=True)["failures"]
    restore_rng(batch["rng"])
    gradients = {}

    def clip(*clip_args, **clip_kwargs):
        gradients.update(
            cpu_copy({name: p.grad for name, p in worker.model.named_parameters()})
        )
        return original_clip(*clip_args, **clip_kwargs)

    with patch.object(module, "scale_grads_and_clip_grad_norm", side_effect=clip):
        if mode == "sc":
            worker.begin_train_step(
                batch["loss_fn"], gbs=batch["gbs"], mbs=batch["mbs"]
            )
            worker.train_microbatch(copy.deepcopy(batch["batch"]))
            metrics = worker.finish_train_step()
        else:
            metrics = worker.train(
                copy.deepcopy(batch["batch"]),
                batch["loss_fn"],
                gbs=batch["gbs"],
                mbs=batch["mbs"],
                check_dim_skip_keys=batch["check_dim_skip_keys"],
            )
    result = {
        "gradients": gradients,
        "metrics": metrics,
        "state": snapshot(worker),
        "rng": rng_state(),
    }
    if reference is None:
        reference = result
    else:
        reports[mode] = {
            key: compare(result[key], reference[key], key) for key in result
        }
        del result
        torch.cuda.empty_cache()
report = {
    "rank": rank,
    "deterministic": cfg["automodel_cfg"]["deterministic"],
    "strict_deterministic": args.strict_deterministic,
    "comparisons": reports,
}
(args.output / f"rank-{rank:02d}.json").write_text(json.dumps(report, indent=2))
print("RECORDED_CONTROL_RESULT " + json.dumps(report), flush=True)
dist.barrier()
dist.destroy_process_group()
