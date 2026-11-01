#!/usr/bin/env bash
set -euo pipefail
source "${SLURM_SUBMIT_DIR:?}/launch/score_centering/campaign.sh"
STAGE=${1:?Missing stage}
MODE=${2:?Missing mode}
[[ "$(git -C "$REPO" rev-parse HEAD)" == "${SC_SOURCE_REVISION:?}" ]] || fail "Source revision changed while the job was queued"
[[ -z "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]] || fail "Source changed while the job was queued"
export RUN_MODE=$MODE
export ENROOT_ROOTFS_WRITABLE=1 PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
export GPUS_PER_NODE=4 DEDICATED_RAY_HEAD=0
export MOUNTS="$REPO:/opt/nemo-rl,/lustre:/lustre"
export BASE_LOG_DIR="$CAMPAIGN/logs" RAY_LOG_SYNC_FREQUENCY=30
if [[ "$STAGE" == runtime ]]; then
    PARTIAL="$CAMPAIGN/runtime.${SLURM_JOB_ID}.partial.sqsh"
    srun --nodes=1 --ntasks=1 --gres=gpu:4 --no-container-mount-home \
        --container-image="$BASE_IMAGE" --container-mounts="$MOUNTS" \
        --container-workdir=/opt/nemo-rl --container-writable --container-save="$PARTIAL" \
        bash /opt/nemo-rl/launch/score_centering/build_runtime.sh
    [[ -f "$CAMPAIGN/runtime-audit.json" && -s "$PARTIAL" ]] || fail "Runtime build did not produce its audited image"
    [[ ! -e "$RUNTIME_IMAGE" ]] || fail "Refusing to replace an existing runtime image"
    mv "$PARTIAL" "$RUNTIME_IMAGE"
    cp "$CAMPAIGN/runtime-audit.json" "$CAMPAIGN/runtime-ready.json"
else
    export CONTAINER="$RUNTIME_IMAGE"
    export COMMAND="cd /opt/nemo-rl && bash launch/score_centering/run.sh $STAGE"
    cd "$REPO"
    source ray.sub
fi
