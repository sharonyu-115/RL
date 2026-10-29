# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check the replay harness against real FusedAdam before launching training."""

from types import SimpleNamespace

import torch
from transformer_engine.pytorch.optimizers import FusedAdam

from automodel_sc_real_replay_worker import compare, restore, snapshot

model = torch.nn.Linear(16, 8, dtype=torch.bfloat16, device="cuda")
optimizer = FusedAdam(
    model.parameters(),
    lr=2e-6,
    weight_decay=0.1,
    master_weights=True,
    store_param_remainders=True,
    exp_avg_dtype=torch.bfloat16,
    exp_avg_sq_dtype=torch.bfloat16,
)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
worker = SimpleNamespace(model=model, optimizer=optimizer, scheduler=scheduler)
for step in range(3):
    before = snapshot(worker)
    for param in model.parameters():
        param.grad = torch.full_like(param, 0.25)
    optimizer.step()
    scheduler.step()
    after = snapshot(worker)
    restore(worker, before)
    assert not compare(snapshot(worker), before, exact=True)["failures"]
    for param in model.parameters():
        param.grad = torch.full_like(param, 0.25)
    optimizer.step()
    scheduler.step()
    assert not compare(snapshot(worker), after, exact=True)["failures"]
    assert compare(snapshot(worker), before, exact=True)["failures"]
    assert compare(torch.tensor([float("nan")]), torch.tensor([float("nan")]))[
        "failures"
    ]
    assert compare(torch.tensor([1.0]), torch.tensor([2.0]))["failures"]
print("REPLAY_HARNESS_SELFCHECK_PASS", flush=True)
