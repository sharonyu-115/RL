#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/campaign.sh"
cd "$REPO"

STAGE=${1:?Usage: submit.sh runtime|smoke|nvfp4-control|nvfp4-sc|bf16 [fresh|resume|dry-run]}
MODE=${2:-fresh}
case "$STAGE" in runtime|smoke|nvfp4-control|nvfp4-sc|bf16) ;; *) fail "Unknown stage: $STAGE" ;; esac
case "$MODE" in fresh|resume|dry-run) ;; *) fail "Unknown mode: $MODE" ;; esac
[[ -f "$MODEL/config.json" ]] || fail "Missing model: $MODEL"
[[ -f "$REPO/$RECIPE" ]] || fail "Missing recipe: $RECIPE"
if [[ "$STAGE" == runtime ]]; then
    NODES=1
    IMAGE=$BASE_IMAGE
else
    NODES=8
    IMAGE=$RUNTIME_IMAGE
fi
OPTIONS=(--parsable --account="$ACCOUNT" --partition="$PARTITIONS" --nodes="$NODES"
    --ntasks-per-node=1 --exclusive --time=05:00:00
    --job-name="general_sa-qwen3-sc.${STAGE}" --chdir="$REPO"
    --output="$CAMPAIGN/logs/${STAGE}-%j.out" --error="$CAMPAIGN/logs/${STAGE}-%j.err")
if [[ "$MODE" == dry-run ]]; then
    printf 'Image: %s\n' "$IMAGE"
    printf 'Live partition, runtime, checkpoint, and smoke gates have NOT been checked.\n'
    printf '%q ' sbatch "${OPTIONS[@]}" "$REPO/launch/score_centering/job.sh" "$STAGE" "$MODE"
    printf '\n'
    exit 0
fi
[[ -f "$IMAGE" ]] || fail "Missing container: $IMAGE"
[[ -z "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]] || fail "Commit tracked changes before submission"
if git -C "$REPO" submodule status --recursive | grep -Eq '^[-+U]'; then
    fail "Submodules do not match the pinned revisions"
fi
mkdir -p "$CAMPAIGN/logs" "$CAMPAIGN/submissions"
exec 9>"$CAMPAIGN/submissions/${STAGE}.lock"
flock -n 9 || fail "Another submission for $STAGE is in progress"
if [[ "$STAGE" == runtime ]]; then
    [[ ! -e "$RUNTIME_IMAGE" ]] || fail "A runtime image already exists; preserve it and choose a new campaign"
else
    [[ -f "$CAMPAIGN/runtime-ready.json" ]] || fail "Runtime refresh has not completed"
fi
if [[ "$STAGE" != runtime && "$STAGE" != smoke ]]; then
    [[ -f "$CAMPAIGN/smoke/passed.json" ]] || fail "GPU save/resume smoke gate has not passed"
    grep -Fq "\"source_revision\": \"$(git -C "$REPO" rev-parse HEAD)\"" "$CAMPAIGN/smoke/passed.json" || fail "Smoke result belongs to a different source revision"
fi
if [[ "$MODE" == fresh ]]; then
    [[ ! -e "$CAMPAIGN/submissions/${STAGE}.latest" ]] || fail "Stage already submitted; inspect the previous job before using resume"
    [[ ! -d "$CAMPAIGN/$STAGE/checkpoints" ]] || fail "Existing checkpoints must not be overwritten"
else
    [[ "$STAGE" != runtime ]] || fail "Runtime builds are not resumable; use a new campaign after diagnosing a failed build"
    if [[ -f "$CAMPAIGN/submissions/${STAGE}.latest" ]]; then
        PREVIOUS=$(cat "$CAMPAIGN/submissions/${STAGE}.latest")
        STATE=$(sacct -X -n -P -j "$PREVIOUS" --format=State | head -1)
        case "$STATE" in COMPLETED*|FAILED*|CANCELLED*|TIMEOUT*|PREEMPTED*|NODE_FAIL*|OUT_OF_MEMORY*) ;;
            *) fail "Previous job $PREVIOUS is active or unverified: $STATE" ;;
        esac
    fi
    if [[ "$STAGE" != smoke ]]; then
        find "$CAMPAIGN/$STAGE/checkpoints" -maxdepth 2 -name training_info.json -print -quit | grep -q . || fail "No finalized checkpoint to resume"
    fi
fi
IFS=',' read -ra REQUESTED_PARTITIONS <<< "$PARTITIONS"
for PARTITION in "${REQUESTED_PARTITIONS[@]}"; do
    scontrol show partition "$PARTITION" -o >> "$CAMPAIGN/submissions/partition-audit.log"
    sbatch --test-only "${OPTIONS[@]}" --partition="$PARTITION" "$REPO/launch/score_centering/job.sh" "$STAGE" "$MODE"
done
printf '%s\n' "$PARTITIONS" > "$CAMPAIGN/partitions"
export SC_SOURCE_REVISION
SC_SOURCE_REVISION=$(git -C "$REPO" rev-parse HEAD)
JOB_ID=$(sbatch "${OPTIONS[@]}" "$REPO/launch/score_centering/job.sh" "$STAGE" "$MODE")
JOB_ID=${JOB_ID%%;*}
printf '%s\n' "$JOB_ID" > "$CAMPAIGN/submissions/${STAGE}.latest"
printf '%s\t%s\t%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$STAGE" "$MODE" "$JOB_ID" "$(git -C "$REPO" rev-parse HEAD)" >> "$CAMPAIGN/submissions/history.tsv"
printf 'Submitted %s job %s\n' "$STAGE" "$JOB_ID"
