# Qwen3 NVFP4 score-centering campaign

This campaign uses the existing upstream 8-node, 4-GPU-per-node recipe without
changing training batch size, response length, TP2/EP8 layout, R3, or TIS bounds.
The launchers contain local paths but no credentials. `campaign.sh` defines the
campaign identity; do not change it between allocations of the same campaign.
Keep the mounted source worktree unchanged while jobs are running; use a separate
worktree and campaign when testing a different implementation.

## Gates and commands

1. Refresh cluster credentials if `klist -s` or `sinfo` fails. Inspect every
   requested partition's node/time/resource/account constraints. The launcher
   also runs `sbatch --test-only` for each partition before submitting. The
   verified list is retained for later allocations. Backfill preemption is
   acceptable because training is checkpointed; an interrupted runtime build
   needs a new build attempt rather than being treated as a valid image.
2. `bash launch/score_centering/submit.sh runtime` requests one GB200 node to
   synchronize the pinned driver, Megatron, and synchronous vLLM environments
   inside a new image. It checks versions, imports, CPU regressions, and changed
   allow-listed source types before publishing the image and readiness marker.
   The old image and old experiments are never overwritten.
3. After runtime success, `bash launch/score_centering/submit.sh smoke` requests
   the real 8-node layout. It trains to step 3 with SC/K32, checks finalized model,
   optimizer, and dataloader state, then restarts the driver and advances to step
   6. It uses the original 20,480-token response and 22,528-token sequence limits.
   Validation is reduced to 16 prompts only for this disposable smoke campaign.
4. Only after `smoke/passed.json` exists, submit the comparison arms with
   `bash launch/score_centering/submit.sh nvfp4-control`, `nvfp4-sc`, and `bf16`.
   The matched BF16 arm retains R3 and optimizer settings but disables training
   FP4/TE precision overrides and rollout quantization, using the Triton MoE
   backend. Historical curves remain supplementary, not a matched-code control.

Append `dry-run` to inspect a submission without queueing it. A dry run is not
proof that partitions, runtime, or smoke gates pass. Append `resume` after
inspecting a terminal allocation to continue the same run. No existing
checkpoints, W&B runs, or runtime images are deleted by these scripts.

## Continue rather than restart

All comparison arms use `grpo.max_num_steps=1500` and an unlimited epoch count
from their first allocation. Step 100 is an analysis checkpoint, not a different
training horizon. A finalized checkpoint is saved every 10 steps and at the
four-hour training deadline in a five-hour allocation. Subsequent allocations
reuse the same checkpoint directory and W&B ID. The normal GRPO resume path
restores model/optimizer, scheduler, counters, and dataloader state. The separate
3-to-6-step smoke is not used to initialize comparison arms.

The active agent monitors Slurm and driver logs, diagnoses failures, and submits
continuations only after inspecting exit state and finalized checkpoints. There
is no unattended retry loop hiding repeated training failures. Failed smoke
attempts may be resumed after diagnosis; the complete metric history is retained.

## Readiness and analysis

Local scalar JSONL files accompany W&B and TensorBoard. The smoke gate checks
head mass and finite loss/gradient/KL diagnostics after resume. Successful real
training also exercises multi-node initialization, candidate generation,
TP-aware backward, router replay, and repeated IPC refit. It is not an HF/BF16
logprob-parity certificate: the stock smoke dispatcher targets Automodel and its
multi-GPU KL check rejects multi-node recipes, so it cannot certify this Megatron
NVFP4 configuration. Before long continuation, inspect initial generation-policy
KL against the fresh matched control/BF16 arms rather than requiring NVFP4 KL to
be zero. Short generations alone do not prove worst-case full-length memory use.

At roughly 100 matched steps, compare validation accuracy/reward, KL and
importance-weight tails, gradient spikes, SC head mass, GPU/host memory, and step
time. Preserve these runs for continuation if healthy; do not reset their
optimizer/scheduler or silently change objective-affecting settings to recover
from failures. Multi-seed evidence is needed before claiming a stability or
alignment improvement.
