#!/usr/bin/env bash
set -euo pipefail
source /opt/nemo-rl/launch/score_centering/campaign.sh
STAGE=${1:?Missing stage}
cd -P /opt/nemo-rl
export PYTHONPATH="$PWD"
export UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv NEMO_RL_VENV_DIR=/opt/ray_venvs
export UV_LOCKED=true PYTHONUNBUFFERED=1 UV_HTTP_TIMEOUT=900
export HF_HOME=/lustre/fsw/general_sa/shuangy/hf
export CUDA_DEVICE_MAX_CONNECTIONS=1 NVTE_WITH_NCCL_EP=0
export TORCH_CUDA_ARCH_LIST=10.0
unset NRL_FORCE_REBUILD_VENVS
uv run --locked --no-sync python tools/generate_fingerprint.py > "$CAMPAIGN/fingerprint.${SLURM_JOB_ID}.json"
cmp "$CAMPAIGN/runtime-fingerprint.json" "$CAMPAIGN/fingerprint.${SLURM_JOB_ID}.json"
RUN_ROOT="$CAMPAIGN/$STAGE"
mkdir -p "$RUN_ROOT/checkpoints" "$RUN_ROOT/training" "$RUN_ROOT/megatron-base"
export SC_METRICS_PATH="$RUN_ROOT/metrics.${SLURM_JOB_ID}.jsonl"
COMMON=(--config "$RECIPE" "policy.model_name=$MODEL" "policy.tokenizer.name=$MODEL"
    grpo.max_num_epochs=null checkpointing.enabled=true checkpointing.save_optimizer=true
    checkpointing.save_period=10 checkpointing.checkpoint_must_save_by=00:04:00:00
    "checkpointing.checkpoint_dir=$RUN_ROOT/checkpoints" "logger.log_dir=$RUN_ROOT/training"
    "++policy.megatron_cfg.env_vars.NRL_MEGATRON_CHECKPOINT_DIR=$RUN_ROOT/megatron-base"
    'logger.extra_backends=[launch.score_centering.metrics.ScalarJsonlLogger]')
START_STEP=0
for CHECKPOINT in "$RUN_ROOT"/checkpoints/step_*; do
    [[ -f "$CHECKPOINT/training_info.json" ]] || continue
    CHECKPOINT_STEP=${CHECKPOINT##*/step_}
    [[ "$CHECKPOINT_STEP" =~ ^[0-9]+$ ]] || continue
    if (( CHECKPOINT_STEP > START_STEP )); then START_STEP=$CHECKPOINT_STEP; fi
done
if [[ "$STAGE" == smoke ]]; then
    COMMON+=(loss_fn.score_centering=true policy.generation.top_logprobs=32
        checkpointing.save_period=1 logger.wandb_enabled=false
        grpo.max_val_samples=16 grpo.val_batch_size=16 grpo.val_at_end=false)
    if (( START_STEP < 3 )); then
        export SC_METRICS_PATH="$RUN_ROOT/phase1.metrics.${SLURM_JOB_ID}.jsonl"
        uv run --locked --no-sync examples/run_grpo.py "${COMMON[@]}" grpo.max_num_steps=3 \
            2>&1 | tee "$RUN_ROOT/phase1.${SLURM_JOB_ID}.log"
        START_STEP=3
    fi
    uv run --no-project --python /opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python python launch/score_centering/check_smoke.py checkpoint "$RUN_ROOT" "$START_STEP"
    export SC_METRICS_PATH="$RUN_ROOT/phase2.metrics.${SLURM_JOB_ID}.jsonl"
    uv run --locked --no-sync examples/run_grpo.py "${COMMON[@]}" grpo.max_num_steps=6 \
        2>&1 | tee "$RUN_ROOT/phase2.${SLURM_JOB_ID}.log"
    uv run --no-project --python /opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python python launch/score_centering/check_smoke.py complete "$RUN_ROOT" 6
    exit 0
fi
[[ -f "$CAMPAIGN/smoke/passed.json" ]] || fail "Smoke gate has not passed"
if [[ "$RUN_MODE" == resume ]]; then
    (( START_STEP > 0 )) || fail "Resume must not restart training from scratch"
    [[ -f "$RUN_ROOT/wandb-id" ]] || fail "Resume is missing its original W&B identity"
    uv run --no-project --python /opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python python launch/score_centering/check_smoke.py checkpoint "$RUN_ROOT" "$START_STEP"
    WANDB_RESUME_MODE=must
else
    (( START_STEP == 0 )) || fail "Fresh run found an unexpected checkpoint"
    WANDB_RESUME_MODE=allow
fi
set +x
[[ -f "$SECRETS_FILE" ]] || fail "Missing credentials file"
set -a
source "$SECRETS_FILE"
set +a
: "${WANDB_API_KEY:?Missing WANDB_API_KEY}"
IDENTITY="$RUN_ROOT/wandb-id"
if [[ ! -f "$IDENTITY" ]]; then
    printf '%s' "$CAMPAIGN/$STAGE" | sha256sum | cut -c1-16 > "$IDENTITY"
fi
COMMON+=(grpo.max_num_steps=1500 "logger.wandb.project=$PROJECT" "++logger.wandb.entity=$ENTITY"
    "logger.wandb.name=$(basename "$CAMPAIGN")-$STAGE" "++logger.wandb.id=$(cat "$IDENTITY")"
    "++logger.wandb.resume=$WANDB_RESUME_MODE" logger.wandb_enabled=true)
case "$STAGE" in
    nvfp4-sc) COMMON+=(loss_fn.score_centering=true policy.generation.top_logprobs=32) ;;
    nvfp4-control) COMMON+=(loss_fn.score_centering=false policy.generation.top_logprobs=0) ;;
    bf16) COMMON+=(loss_fn.score_centering=false policy.generation.top_logprobs=0
        policy.megatron_cfg.fp4_cfg.enabled=false '~policy.megatron_cfg.te_precision_config_file'
        policy.generation.nvfp4_pertoken_rollout.enabled=false policy.generation.vllm_kwargs.moe_backend=triton) ;;
    *) fail "Unknown training arm: $STAGE" ;;
esac
uv run --locked --no-sync examples/run_grpo.py "${COMMON[@]}"
