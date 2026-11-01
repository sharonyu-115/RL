#!/usr/bin/env bash
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CAMPAIGN=${CAMPAIGN:-/lustre/fsw/general_sa/shuangy/experiments/qwen3-score-centering-20261101-r2}
BASE_IMAGE=${BASE_IMAGE:-/lustre/fsw/general_sa/shuangy/images/nemo-rl-v0.8.0.rc0-5293408cb209-arm64.sqsh}
RUNTIME_IMAGE=${RUNTIME_IMAGE:-${CAMPAIGN}/runtime.sqsh}
MODEL=${MODEL:-/lustre/fsw/general_sa/shuangy/models/Qwen/Qwen3-30B-A3B-Base}
SECRETS_FILE=${SECRETS_FILE:-/lustre/fsw/general_sa/shuangy/tmp/pr3566-longrun/secrets.env}
RECIPE=examples/configs/recipes/llm/grpo-qwen3-30ba3b-base-8n4g-megatron-te-nvfp4-pertoken.yaml
PARTITIONS=${PARTITIONS:-batch,backfill,36x2-a01r,tcpo,tcpo-backfill}
if [[ -f "$CAMPAIGN/partitions" ]]; then
    PARTITIONS=$(cat "$CAMPAIGN/partitions")
fi
ACCOUNT=${ACCOUNT:-general_sa}
PROJECT=${PROJECT:-qwen3-30b-nvfp4-sc}
ENTITY=${ENTITY:-nv-welcome}
export REPO CAMPAIGN RUNTIME_IMAGE MODEL SECRETS_FILE RECIPE PROJECT ENTITY PARTITIONS

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}
