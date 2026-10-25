#!/usr/bin/env bash
set -euo pipefail
: "${PART_SELECTED_TIGRIS_OUTPUT:?Missing campaign}"
: "${PART_SELECTED_TIGRIS_PHASE:?Missing phase}"
: "${SELECTED_WALLTIME_RECOVERY:?Missing recovery}"
[[ "$(uname -m)" == aarch64 ]] || exit 2
source "$PART_SELECTED_TIGRIS_OUTPUT/launch.env"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$PART_SELECTED_TIGRIS_OUTPUT/source"
selected_bin="$(dirname -- "$SELECTED_PYTHON")"
export PATH="$selected_bin:$PATH" LD_LIBRARY_PATH="$selected_bin/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MAX_JOBS=2 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 WORLD_SIZE=1
(cd -- "$SELECTED_WALLTIME_RECOVERY/source"; sha256sum --quiet -c ../source.sha256)
cd -- "$PART_SELECTED_TIGRIS_OUTPUT/source"
sha256sum --quiet -c "$PART_SELECTED_TIGRIS_OUTPUT/source.sha256"
args=(--phase "$PART_SELECTED_TIGRIS_PHASE" --output "$PART_SELECTED_TIGRIS_OUTPUT" --recovery "$SELECTED_WALLTIME_RECOVERY")
case "$PART_SELECTED_TIGRIS_PHASE" in
  train|evaluate)
    [[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-3]$ ]] || exit 2
    args+=(--task-index "$SLURM_ARRAY_TASK_ID") ;;
  gate|lock|report) ;;
  *) echo 'Recovery cannot prepare or benchmark' >&2; exit 2 ;;
esac
exec "$SELECTED_PYTHON" -u "$SELECTED_WALLTIME_RECOVERY/source/scripts/recover_part_selected_tigris_walltime.py" "${args[@]}"
