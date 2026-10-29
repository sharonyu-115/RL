# AutoModel full-batch SC worker numerical validation

Status: complete. Under strict deterministic execution, the AutoModel full-batch
SC worker is numerically equivalent to V1 for the tested configurations and
recorded real inputs. Every compared value matched exactly.

## Scope

Determine whether the AutoModel full-batch SC worker matches V1 training for
identical recorded real inputs and pre-update state. This excludes controller,
sampler, transport, rollout/refit and end-to-end trajectory equivalence.

Production PR #4963 at 8ec0e19ba28e0c6c7474529af37786cf81a3b809 is unchanged.
Validation harness is on
[validation/automodel-sc-real-replay](https://github.com/sharonyu-115/RL/tree/validation/automodel-sc-real-replay)
at fae8934804c25f59c6cebd6c387373ba793b8ec6.

The V1 train() source is identical to upstream base
0e36f2e6a944a2e617fbc1941d57041715a4ac3f (source SHA-256
cba82803676d1f259792463ca300cd5e0765320e6cb21157774cb37b5b00b4a8).

## Results

| Model | Updates | DP ranks | Rank/update comparisons | Maximum absolute difference | Job exit |
| --- | --- | --- | --- | --- | --- |
| Gemma4 E2B IT | 6 | 4 | 24 | 0 | 0 |
| Qwen3 30B-A3B Base | 6 | 16 | 96 | 0 | 0 |

Gemma job 20983726 completed with exit 0 in 12m47s. All 24 rank/update records
passed. Maximum absolute error and differing element counts were zero for
gradients, returned metrics, model parameters/buffers, optimizer, restoration,
scheduler and RNG. Gemma has no MoE router-load state, so that category is empty.

All six batches had 32 rows and nonzero advantages; no zero-weight rows occurred.
Updates 2–6 changed 2,587,673; 6,349,264; 8,619,308; 10,851,929; and 13,759,952
parameter elements, identically in both paths. Update 1's tiny warmup LR did not
change BF16 model parameters, but optimizer state was still compared.

[Gemma W&B run](https://wandb.ai/ys_fishcool-nvidia/nemo-rl-automodel-sc/runs/85c4bff7)
and [all-metrics workspace](https://wandb.ai/ys_fishcool-nvidia/nemo-rl-automodel-sc?nw=myuos0f8jrr)
(371 panels; no populated non-bookkeeping metric keys omitted).

Qwen job 20983727 completed with exit 0 in 15m35s. All 96 rank/update records
passed with zero maximum absolute error and zero differing elements across all
compared categories. Update 1 had zero advantages; updates 2–6 had nonzero
advantages and changed 43,296,389; 84,429,402; 142,207,927; 190,620,314; and
229,392,208 parameter elements, identically in both paths. Update 4 retained
32 physical rows with four zero-weight rows and 28 valid-loss sequences.

[Qwen W&B run](https://wandb.ai/ys_fishcool-nvidia/nemo-rl-automodel-sc/runs/1f8d3435)
and [all-metrics workspace](https://wandb.ai/ys_fishcool-nvidia/nemo-rl-automodel-sc?nw=iql83f2l0qk)
(399 panels; no populated non-bookkeeping metric keys omitted).

Both models have empty Gate cumulative-load state in this configuration. Router
parameters and registered buffers are included in the model comparison, but the
empty router-load category does not validate auxiliary-loss-free gate-bias
updates for other architectures.

## Inputs and method

The regular V1 GRPO driver generates real rollouts with vLLM and computes the
normal training inputs. The worker extension records the exact local batch
(tokens, lengths, masks, advantages, previous/generation/reference logprob
fields), preserving dtypes and local ordering. The initial full state and every
batch/RNG snapshot are saved. Later pre-update states are captured in memory.

For each batch, unchanged V1 train() runs first. The harness then restores model
parameters/buffers, FusedAdam raw moments/master remainders/scales/counters,
scheduler, MoE router cumulative loads and RNG, checks exact restoration, and
runs begin_train_step → train_microbatch → finish_train_step. It compares every
gradient before clipping, every returned numerical metric and complete resulting
model/optimizer/scheduler/router/RNG state on every DP rank. It restores the V1
post-update state to produce the next real batch.

Floating tolerance is rtol=1e-5, atol=1e-7. Restoration, scheduler/RNG and integer
state are exact. Non-finite values, missing ranks/steps, differing keys/dtypes/
shapes, and failed comparisons prevent a passing result.

## Determinism controls

| Control | Job | Result |
| --- | --- | --- |
| Initial real Gemma V1/SC replay, default kernels | 20982624 | Strict comparison failed on update 1 |
| Recorded batch, V1/V1 and V1/SC, default kernels | 20983158 | Both comparisons failed |
| Recorded batch, warning-only deterministic mode | 20983158 | Both comparisons failed |
| Recorded batch, strict deterministic mode | 20983411 | Both comparisons exactly equal on all 4 ranks |

The default and warning-only controls showed nondeterministic gradients and
optimizer state. PyTorch explicitly warned that cuDNN attention backward remains
nondeterministic with warn_only=True. The strict control set
CUBLAS_WORKSPACE_CONFIG=:4096:8, NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 and
torch.use_deterministic_algorithms(True, warn_only=False), obtaining zero
differences in gradients, metrics, full state and RNG for both comparisons.

The six-update validation therefore uses policy.automodel_cfg.deterministic=true
and a validation-only strict PyTorch override. It makes no claim of strict
repeatability under the default nondeterministic kernels.

## Six-update matrix

Both use seed 42, GBS=32, MBS=1, TP=CP=EP=1, full-parameter FSDP2, disabled packing
and dynamic batching, and the recipe's FusedAdam and LR scheduler. BF16 optimizer
moments and master-weight remainders are preserved; no AdamW replacement,
synthetic logprobs, synthetic advantages or forced on-policy ratio is used.
Both recipes use token-level loss, importance-sampling correction, and zero
reference-policy KL penalty. The full local dispatch contains 8 backend
microbatches on each Gemma rank and 2 on each Qwen rank.

| Model | Layout | Recipe and overrides | Job |
| --- | --- | --- | --- |
| Gemma4 E2B IT | 4 training + 4 rollout GPUs | dapo-gemma4-e2b-it-1n8g-automodel.yaml; 2048 input / 2048 generation / 4096 total | 20983726 |
| Qwen3 30B-A3B Base | 16 training + 8 rollout GPUs | grpo-qwen3-30ba3b-2507-instruct-4n8g-automodel.yaml with Base checkpoint; torch dispatcher; 1024 / 2048 / 3072 tokens | 20983727 |

The previous 100-step recipe overrides are retained: 8 prompts × 4 completions,
dynamic sampling/reward scaling/reward shaping disabled, disaggregated async
vLLM. Checkpointing and evaluation are disabled for this worker replay.

All runs use project ys_fishcool-nvidia/nemo-rl-automodel-sc. Training metrics keep
their original names; replay_* identifies additional test diagnostics. Timing
includes capture, restoration and two training paths and is not performance data.
The standard GRPO logger sums the per-rank replay_passed values: the expected
dashboard value is 4 for Gemma and 16 for Qwen on each step. Per-rank JSON records
are the authoritative comparison evidence; every error is independently zero
before aggregation.

## Artifacts

Each result directory contains the resolved config, command/source manifest,
initial per-rank state, exact inputs and RNG, raw V1/SC metrics, per-rank comparison
JSON, and runtime FusedAdam source. The saved inputs support a no-rollout replay
with tests/functional/automodel_sc_recorded_control.py.

The launch manifest's legacy variant label "upstream" selects the regular V1
GRPO entrypoint. Both numerical paths actually execute the recorded validation
source SHA in one worker, and the V1 method is verified unchanged from upstream.
There are not two independently sampled training runs per model.

The earlier Qwen SC step-60 rollout/training logprob anomaly remains outside this
worker-only validation and is not resolved by a passing worker comparison.

## Replaying the captured first-batch control

The standalone control runs V1, repeats V1, and runs SC from the saved initial
state and batch. Use the same AutoModel environment and DP layout as capture.
For the four-rank Gemma capture:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
uv run --no-project --python "$AUTOMODEL_PYTHON" python -m torch.distributed.run \
  --standalone --nproc-per-node=4 \
  tests/functional/automodel_sc_recorded_control.py \
  --input "$CAPTURE_DIRECTORY" --output "$CONTROL_OUTPUT_DIRECTORY" \
  --strict-deterministic
```

The strict CLI flag incorporates the identical override used by the validated
external strict_control.py shim in job 20983411. The six-update jobs used the
same strict setting through SC_REPLAY_STRICT=1 in the worker extension. The CLI
packaging change after those jobs does not change their tested production code.

Raw artifacts are retained under the validation directory
automodel-sc-worker-replay-20261028, including failed/default controls. The full
per-rank tensor packages are local artifacts; only summaries and harness source
are committed to the validation branch.
