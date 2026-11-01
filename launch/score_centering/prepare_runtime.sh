#!/usr/bin/env bash
set -euo pipefail
SOURCE_REPO=${1:?Missing source worktree}
[[ "${NRL_CONTAINER:-}" == 1 && -d /opt/nemo_rl_venv ]] || {
    printf 'Runtime preparation must run inside the NeMo-RL container.\n' >&2
    exit 1
}
[[ -f "$SOURCE_REPO/pyproject.toml" ]] || exit 1
if [[ -d /opt/nemo-rl && ! -L /opt/nemo-rl ]]; then
    [[ ! -e /opt/nemo-rl.image-source ]] || exit 1
    mv /opt/nemo-rl /opt/nemo-rl.image-source
fi
ln -sfn "$SOURCE_REPO" /opt/nemo-rl
git -C /opt/nemo-rl submodule status --recursive
exec bash /opt/nemo-rl/launch/score_centering/build_runtime.sh
