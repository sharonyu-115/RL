#!/usr/bin/env bash
set -euo pipefail
source /opt/nemo-rl/launch/score_centering/campaign.sh
cd -P /opt/nemo-rl
export PYTHONPATH="$PWD"
export UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv
export NEMO_RL_VENV_DIR=/opt/ray_venvs
export UV_LOCKED=true NRL_FORCE_REBUILD_VENVS=true UV_HTTP_TIMEOUT=900
export MAX_JOBS=16 NVTE_BUILD_THREADS_PER_JOB=2
sha256sum pyproject.toml uv.lock > "$CAMPAIGN/dependencies.before.sha256"
uv sync --locked --inexact
uv run --locked python -m nemo_rl.utils.prefetch_venvs MegatronPolicyWorker VllmGenerationWorker
uv run --locked --no-sync python launch/score_centering/runtime_audit.py driver
uv run --no-project --python /opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python python launch/score_centering/runtime_audit.py policy
uv run --no-project --python /opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker/bin/python python launch/score_centering/runtime_audit.py generation
uv run --locked --group test python -m pytest --noconftest \
    tests/unit/algorithms/test_sync_score_centering.py \
    tests/unit/algorithms/test_score_centering_campaign.py \
    tests/unit/algorithms/test_score_centering.py \
    tests/unit/models/generation/test_score_centering.py -q --tb=short
RAYON_NUM_THREADS=2 uv run --locked --group dev pyrefly check --config "$PWD/pyrefly.toml" \
    nemo_rl/algorithms/loss/loss_input.py nemo_rl/algorithms/single_controller_utils/config.py \
    nemo_rl/experience/rollouts.py nemo_rl/models/generation/score_centering.py \
    nemo_rl/models/generation/interfaces.py
sha256sum -c "$CAMPAIGN/dependencies.before.sha256"
uv run --locked --no-sync python tools/generate_fingerprint.py > /opt/nemo_rl_container_fingerprint
cp /opt/nemo_rl_container_fingerprint "$CAMPAIGN/runtime-fingerprint.json"
uv run --locked --no-sync python launch/score_centering/runtime_audit.py finalize
