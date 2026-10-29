# AutoModel real-batch worker replay

Validation-only branch based on PR #4963 at 8ec0e19ba28e0c6c7474529af37786cf81a3b809.
No production sources are changed. V1 train() remains the reference implementation.

## Claim and boundary

Test whether the AutoModel full-batch SC worker is numerically equivalent to V1
training for the recorded inputs and configurations. This is not an end-to-end SC
controller, sampler, data-plane, or rollout/refit equivalence test.

## Method

Use the existing worker_extension_cls_fqn hook from the regular GRPO entrypoint.
For each incoming real rollout batch on every DP rank:

1. Save the original prepared batch, loss function and RNG state. Preserve every
   input field, dtype, padding, mask, advantage, logprob and local row ordering.
2. Capture model parameters/buffers, raw optimizer state (BF16 moments, int16
   master remainders, scales, parameter groups/counters), scheduler and router
   cumulative loads. Save the initial full state once per rank; later pre-update
   snapshots remain in memory. Ordered inputs plus the initial state describe
   the trajectory without writing a full model checkpoint every update.
3. Run unchanged V1 train(). Capture all gradients immediately before clipping,
   returned metrics, post-update model/optimizer/scheduler/router and RNG state.
4. Restore the pre-update state and assert exact restoration on every leaf.
   Restore RNG immediately before SC execution.
5. Run the production begin_train_step → train_microbatch → finish_train_step
   methods on an independent copy of the same full local batch.
6. Compare all leaves on every rank. Require finite values and matching keys,
   shapes and dtypes. Floating tolerance is rtol=1e-5, atol=1e-7; integer state,
   scheduler, RNG and restore checks are exact. Report maximum absolute error,
   unequal element counts and failures separately for each category. Any failed
   rank fails the job collectively and saves its pre-update state.
7. Restore the V1 post-update state/RNG to generate the next real batch. Training
   metrics retain NeMo RL names; replay_* metrics identify validation diagnostics.

The helper self-check uses real GPU FusedAdam, checks initial and populated
optimizer restoration, verifies repeated updates match, and checks that the
comparator rejects different values and non-finite values.

## Validation matrix

Six consecutive real updates per model, GBS=32, MBS=1, TP=CP=EP=1, full parameter
FSDP2, no packing or dynamic batching, seed 42, normal stochastic vLLM rollouts.
No forced on-policy ratios, synthetic logprobs, synthetic advantages, or optimizer
replacement. Keep the prior 100-step validation recipe overrides and scheduler.

* Gemma4 E2B IT: 4 training GPUs + 4 rollout GPUs, one H100 node.
* Qwen3 30B-A3B Base: 16 training GPUs + 8 rollout GPUs, three H100 nodes; torch
  MoE dispatcher, 1024 input / 2048 generation / 3072 total tokens.

Results belong to W&B project ys_fishcool-nvidia/nemo-rl-automodel-sc.
Realized coverage (nonzero advantages, masked rows and changed parameters) must
be reported from the artifacts, not assumed. This test does not resolve the
earlier Qwen SC step-60 rollout/training logprob discrepancy.

## Determinism qualification from the recorded control

The first real Gemma batch failed strict V1/SC comparison with the default kernel
settings. Replaying the saved batch through V1 twice also failed, including
gradients and optimizer state. NeMo RL's deterministic flag calls PyTorch with
warn_only=True; cuDNN attention explicitly warns that this still allows a
nondeterministic backward algorithm. The warning-only control also failed.

A separate control with CUBLAS_WORKSPACE_CONFIG=:4096:8,
NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 and
torch.use_deterministic_algorithms(True, warn_only=False) produced zero differences
for V1/V1 and V1/SC on all four ranks (job 20983411). The six-update runs therefore
use policy.automodel_cfg.deterministic=true plus SC_REPLAY_STRICT=1, which enforces
strict PyTorch determinism in the validation worker only. All production methods
remain unchanged. Equivalence results must explicitly carry this qualification.

## Completed evidence

See [the result report](automodel_sc_real_replay_results.md) and
[the machine-readable summary](automodel_sc_real_replay_results.json).
