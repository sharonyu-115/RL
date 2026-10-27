# AutoModel single-controller: Stage 1

## Scope and baseline

Base: upstream `main` at `4e9a597af9e5c84f21c4762408c6d1eac90342bc`
(2026-10-27). This stage enables text GRPO with disaggregated async vLLM,
`in_order`, lookahead zero, and one complete rollout batch per training step.
The first validation model is `google/gemma-4-E2B-it`.

The acceptance criterion is numerical parity with the existing AutoModel
`train()` for identical data and initial model/optimizer/scheduler state.
Independent rollout curves are integration evidence, not numerical parity.

Stage 1 supports dense full-parameter FSDP2, TP=CP=EP=1, no CPU offload,
and fixed-size microbatches. Packing, dynamic batching, LoRA, draft training,
MoE, VLM, PPO, distillation, nonzero lookahead, other samplers, and multiple
chunks per step remain follow-up work. Configuration validation rejects
unsupported explicit settings before worker allocation. Worker validation checks
model-dependent properties (such as MoE) and also protects direct split-API
callers. Existing synchronous training retains its current support.

## Existing boundaries

`setup_single_controller` constructs a `TQPolicy`. The driver shards metadata;
`TQWorkerMixin.train_microbatch_presharded` fetches its rows, reconstructs the
batch, and delegates to `AutomodelPolicyWorkerImpl.train_microbatch`.
The mixin already delegates begin, finish, and abort. No new controller actor,
sampler, metadata schema, or data-plane implementation is required.

`train_microbatch` names a controller dispatch, not one forward pass. A full
batch dispatch still runs several backend forward/backward microbatches.

## Worker state and transitions

Introduce an internal dataclass holding the loss function, global batch size,
microbatch size, whether a dispatch was attempted, and completed batch metrics.
The worker initializes its optional state to `None` in the constructor.

| Call | Preconditions | Effects |
| --- | --- | --- |
| begin | Idle; supported backend and sizes | Enter training mode, clear gradients, retain loss/sizes; no optimizer or scheduler step. |
| microbatch | Open; no previous dispatch | Mark dispatch attempted, validate every DP rank has exactly GBS/DP rows, execute the complete batch's forward/backward, retain gradients and metrics. |
| finish | Open; batch completed successfully | Clip gradients, update optimizer once, advance scheduler once, return metrics and clear state/gradients. |
| abort | Any state | Drop gradients and state; advance neither optimizer nor scheduler. Idempotent. |

Reject nested begin, finish without a completed batch, repeated dispatch,
legacy `train()` during an open step, and invalid batch sizes. A failed dispatch
cannot be retried in place: abort first. Validation of per-rank row counts uses
a DP collective so all ranks fail together before forward collectives.
Once finish starts committing, it cannot be retried; cleanup runs even if
optimizer/scheduler execution fails. Abort cannot roll back an optimizer step
that has already begun; such a failure requires recovery from a checkpoint.

## Reusing the established numerical path

Keep the public synchronous `train()` signature. Move its body into a private
shared implementation with an explicit `defer_update` argument. The synchronous
entry passes false. The split dispatch passes true, with exactly one global
batch. Deferred execution skips clipping, optimizer updates, scheduler updates,
and final gradient clearing. It returns the same aggregated metrics, with the
gradient norm filled in by finish. Factor the existing clipping/update code into
one helper shared by the two paths, preserving draft behavior on the legacy path.

Both paths continue to use `process_global_batch`, `get_microbatch_iterator`,
`LossPostProcessor`, and `automodel_forward_backward`. Thus token/sequence masks,
global valid counts, FSDP gradient averaging compensation, precision context,
microbatch ordering, and metric aggregation have one implementation. Since the
whole batch is present, global normalization is known before backward; Stage 1
does not need deferred rescaling or modified FSDP synchronization hooks.

The fixed batching restriction guarantees equal row counts and microbatch
participation across DP ranks. Packing and dynamic batching can assign unequal
row counts and need a different full-batch extraction contract before enabling.

## Phase transitions and integration

The normal controller computes policy/reference logprobs before the full batch
dispatch, then calls finish before refit/checkpoint. Existing
`prepare_for_lp_inference(keep_train_buffers=True)` leaves gradients intact;
tests cover a logprob phase between backward and finish to detect accidental
clearing. There is no cross-chunk streaming claim in this stage.

Admission is `in_order`/0, the streaming threshold equals prompts per step,
and prompts × generations equals train GBS. Disable prompt-drop budgets so
rollout failures cannot silently produce short batches. Both global and local
batch checks reject short or oversized dispatches rather than truncate data.

Use the existing full-weight refit and checkpoint facilities. The Gemma validation
launcher overrides the existing Gemma recipe to set disaggregated async vLLM,
full-batch selection, fixed microbatches, and W&B logging. Environment paths and
credentials belong in launch overrides, never committed files.

## Validation plan

1. Lifecycle tests: illegal transitions, one update, abort before/after backward,
   failed dispatch, and refusal of partial or repeated batches.
2. Distributed fixed-batch parity on Gemma4 E2B from identical states, with
   token-level and sequence-level loss, unequal valid lengths and masked rows.
   Compare loss, pre-clipping gradients, gradient norm, updated parameter shards,
   optimizer state, scheduler state, and per-microbatch metrics over multiple
   updates. Use deterministic seeds and record tolerances and observed errors.
3. Exercise the metadata fetch/shard path, not just direct tensor dispatch.
4. Run a short live SC rollout/train/refit job with W&B; require finite metrics,
   advancing versions and exactly one dispatch/update per logical step.
5. Run focused configuration/lifecycle tests and formatting checks. Record actual
   results, environment versions, job IDs, and W&B URLs below after execution.

No throughput or learning-quality claim follows from Stage 1. Streaming
normalization, overlap, and scheduling experiments remain stages 2–4.

## Validation results

Executed on 2026-10-27 against worker/config implementation commit `d9e11db46`,
based on the upstream commit above. The numerical run used four H100 80 GB GPUs,
FSDP DP=4, GBS=8, MBS=1,
sequence length 64, BF16 compute, and FP32 AdamW state. It exercised the real
local metadata data plane, not the network TransferQueue service.

| Check | Result |
| --- | --- |
| Worker lifecycle unit tests | 18 passed |
| Stage 1 configuration and existing resiliency configuration tests | 134 passed |
| Token-level loss, updates 1 and 2 | Exact gradients, metrics, parameters, and optimizer state |
| Sequence-level loss, updates 1 and 2 | Exact gradients, metrics, parameters, and optimizer state |
| Scheduler state, LP transition with retained gradients, abort after backward | Passed on all four updates |
| Ruff, formatting, shell syntax, whitespace | Passed |

Every reported maximum absolute error was **0.0**, with acceptance tolerances
`rtol=1e-5`, `atol=1e-6` (scheduler and abort preservation checked exactly).
Slurm job `20910205`; [W&B parity run](https://wandb.ai/ys_fishcool-nvidia/nemorl-gemma4/runs/3o38bes2).
The preceding two-GPU attempt matched its first update exactly but ran out of
memory on a later backward pass with the FP32 AdamW validation setup; it is not
counted as a passing run.

The validation image was `rl.pipeline73042345-x86.sqsh`: PyTorch
`2.13.0+cu130`, Transformers `5.12.1`, and Python 3.13. AutoModel source was the
main-pinned submodule `b916107a5ce8e77a3f18682f8b0eaa46a4b54389`.
The temporary container installed main's pinned Lens
`4d045c03186b90afdc815db46b671f6a21bc7e98` and
`multi-storage-client[boto3]==0.51.0` to satisfy newer source imports. This was
not a freshly rebuilt image from the complete main lockfile; no dependency pins
were changed in the repository.

The live smoke completed five GRPO steps on eight H100 GPUs (four training,
four generation), with async vLLM `0.29.0`, the main-pinned TransferQueue
`05d4cc3377bee9b0f2897bbb90914ee20c0aea39`, and the existing Gemma recipe's
FusedAdam optimizer. Slurm job `20909968`; [W&B live run](https://wandb.ai/ys_fishcool-nvidia/nemorl-gemma4/runs/ou1umtre).
Each step closed on exactly one chunk containing eight prompt groups / 32
samples. Trainer versions advanced from 1 through 5, version lag remained zero,
and every step completed weight refit. The run exited with code 0. The initial
launcher attempt exposed a missing Hydra add-key prefix for the recipe's absent
`async_rl` block; the corrected launcher is
`tests/functional/automodel_sc_stage1_smoke.sh`.

All five training records were verified through the W&B API:

| Step | Loss | Gradient norm | Mean reward |
| --- | --- | --- | --- |
| 1 | 0.020320 | 0.580403 | 0.34375 |
| 2 | 0.015364 | 0.877944 | 0.40625 |
| 3 | 0.037431 | 0.909428 | 0.50000 |
| 4 | 0.012519 | 0.711080 | 0.21875 |
| 5 | 0.007353 | 1.139169 | 0.50000 |

The W&B SDK emitted a teardown `BrokenPipeError` after reporting upload completion;
the remote run is marked finished with all five records present. This did not
interrupt training or refit.
Checkpoint/restart, long-run learning quality, other models, and performance
are not established by these checks.
