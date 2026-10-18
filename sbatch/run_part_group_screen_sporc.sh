#!/usr/bin/env bash
set -euo pipefail
: "${PART_GROUP_OUTPUT:?Missing campaign}"
: "${PART_GROUP_PHASE:?Missing phase}"
[[ "$PART_GROUP_OUTPUT" == /* ]] || exit 2
[[ "$(uname -m)" == x86_64 ]] || { echo 'Requires donor-matched SPORC x86_64' >&2; exit 2; }
source "$PART_GROUP_OUTPUT/launch.env"
group_source="$PART_GROUP_OUTPUT/source"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$group_source"
group_bin="$(dirname -- "$GROUP_PYTHON")"
export PATH="$group_bin:$PATH" LD_LIBRARY_PATH="$group_bin/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 WORLD_SIZE=1
cd -- "$group_source"
sha256sum --quiet -c "$PART_GROUP_OUTPUT/source.sha256"
args=(--phase "$PART_GROUP_PHASE" --output "$PART_GROUP_OUTPUT")
case "$PART_GROUP_PHASE" in
  benchmark|train|evaluate)
    [[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-9]+$ ]] || { echo 'Need numeric new-model array index' >&2; exit 2; }
    args+=(--task-index "$SLURM_ARRAY_TASK_ID") ;;
  prepare|lock|report|refresh-controls) ;;
  *) echo 'Invalid worker phase' >&2; exit 2 ;;
esac
exec "$GROUP_PYTHON" -u "$group_source/scripts/run_part_group_screen.py" "${args[@]}"
