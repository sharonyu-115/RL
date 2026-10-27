# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Stage 1 ownership tests with real CPU gradients and a small optimizer."""

from unittest.mock import Mock

import pytest
import torch

pytest.importorskip("nemo_automodel")
from nemo_rl.models.policy.workers.automodel_policy_worker import (
    AutomodelPolicyWorkerImpl,
)

pytestmark = pytest.mark.automodel


@pytest.fixture
def worker(monkeypatch):
    worker = object.__new__(AutomodelPolicyWorkerImpl)
    worker._train_step_state = None
    worker.cfg = {
        "train_global_batch_size": 4,
        "train_micro_batch_size": 1,
        "dynamic_batching": {"enabled": False},
        "automodel_cfg": {},
    }
    worker.dp_size = 2
    worker.tp_size = worker.cp_size = 1
    worker.cpu_offload = worker.is_vlm = worker.is_moe_model = False
    worker.lora_enabled = worker.enable_seq_packing = False
    worker.draft_model = None
    worker.model = torch.nn.Linear(2, 1, bias=False)
    worker.optimizer = torch.optim.SGD(worker.model.parameters(), lr=0.1)
    worker.scheduler = torch.optim.lr_scheduler.StepLR(worker.optimizer, 1, gamma=0.5)
    worker._step_optimizer = Mock(
        side_effect=lambda **kw: (worker.optimizer.step(), (torch.tensor(2.0), None))[1]
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda.nvtx, "range_push", lambda _: None)
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda: None)
    return worker


def test_lifecycle_and_single_commit(worker):
    before = worker.model.weight.detach().clone()
    worker.begin_train_step(Mock())
    with pytest.raises(RuntimeError, match="already open"):
        worker.begin_train_step(Mock())
    with pytest.raises(RuntimeError, match="successfully completed"):
        worker.finish_train_step()
    with pytest.raises(RuntimeError, match="cannot run"):
        worker.train(None, Mock())
    worker.model(torch.ones(1, 2)).sum().backward()
    worker._train_step_state.dispatched = True
    worker._train_step_state.metrics = {"global_loss": torch.tensor([1.0])}
    with pytest.raises(RuntimeError, match="exactly one"):
        worker.train_microbatch(None)
    torch.testing.assert_close(before, worker.model.weight)
    result = worker.finish_train_step()
    assert result["grad_norm"] == 2
    assert worker.scheduler.last_epoch == 1
    assert worker.optimizer.param_groups[0]["lr"] == 0.05
    assert worker._train_step_state is None
    assert worker.model.weight.grad is None
    assert not torch.equal(before, worker.model.weight)
    with pytest.raises(RuntimeError, match="successfully completed"):
        worker.finish_train_step()
    worker._step_optimizer.assert_called_once()


@pytest.mark.parametrize("backward", [False, True])
def test_abort_does_not_update(worker, backward):
    before = worker.model.weight.detach().clone()
    epoch = worker.scheduler.last_epoch
    worker.begin_train_step(Mock())
    if backward:
        worker.model(torch.ones(1, 2)).sum().backward()
    worker.abort_train_step()
    worker.abort_train_step()
    torch.testing.assert_close(before, worker.model.weight)
    assert worker.model.weight.grad is None
    assert worker.scheduler.last_epoch == epoch
    worker._step_optimizer.assert_not_called()
    worker.begin_train_step(Mock())
    worker.abort_train_step()


def test_failed_commit_cannot_be_retried(worker):
    worker.begin_train_step(Mock())
    worker._train_step_state.metrics = {}
    worker._step_optimizer.side_effect = RuntimeError("optimizer failed")
    with pytest.raises(RuntimeError, match="optimizer failed"):
        worker.finish_train_step()
    assert worker._train_step_state is None
    with pytest.raises(RuntimeError, match="successfully completed"):
        worker.finish_train_step()


@pytest.mark.parametrize("gbs,mbs", [(0, 1), (4, 0), (3, 1), (4, 3)])
def test_invalid_sizes(worker, gbs, mbs):
    with pytest.raises(ValueError, match="divisible"):
        worker.begin_train_step(Mock(), gbs=gbs, mbs=mbs)
    assert worker._train_step_state is None


@pytest.mark.parametrize(
    "flag",
    ["cpu_offload", "is_vlm", "is_moe_model", "lora_enabled", "enable_seq_packing"],
)
def test_unsupported_worker_modes(worker, flag):
    setattr(worker, flag, True)
    with pytest.raises(ValueError, match="Stage 1"):
        worker.begin_train_step(Mock())


def test_dispatch_requires_begin(worker):
    with pytest.raises(RuntimeError, match="begin_train_step"):
        worker.train_microbatch(None)


@pytest.mark.parametrize("minimum,maximum", [(1, 2), (2, 3), (0, 0)])
def test_rejects_incomplete_or_unequal_dp_batch(worker, monkeypatch, minimum, maximum):
    from types import SimpleNamespace

    worker.dp_mesh = Mock()
    worker._train_full_batches = Mock()
    tensor_factory = torch.tensor
    monkeypatch.setattr(torch, "tensor", lambda *a, **kw: tensor_factory(*a))

    def reduce_sizes(sizes, **kwargs):
        sizes.copy_(tensor_factory([minimum, -maximum]))

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce_sizes)
    worker.begin_train_step(Mock())
    with pytest.raises(ValueError, match="every DP rank"):
        worker.train_microbatch(SimpleNamespace(size=2))
    worker._train_full_batches.assert_not_called()
    with pytest.raises(RuntimeError, match="successfully completed"):
        worker.finish_train_step()
    worker.abort_train_step()


def test_failed_backward_can_abort_and_start_again(worker, monkeypatch):
    from types import SimpleNamespace
    from nemo_rl.utils.timer import Timer

    worker.timer = Timer()
    worker.dp_mesh = Mock()
    tensor_factory = torch.tensor
    monkeypatch.setattr(torch, "tensor", lambda *a, **kw: tensor_factory(*a))
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *a, **kw: None)
    worker._train_full_batches = Mock(side_effect=RuntimeError("backward failed"))
    worker.begin_train_step(Mock())
    with pytest.raises(RuntimeError, match="backward failed"):
        worker.train_microbatch(SimpleNamespace(size=2))
    worker.abort_train_step()
    worker._train_full_batches = Mock(return_value={})
    worker.begin_train_step(Mock())
    worker.train_microbatch(SimpleNamespace(size=2))
    worker.abort_train_step()
