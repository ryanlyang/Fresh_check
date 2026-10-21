#!/usr/bin/env bash
set -euo pipefail
: "${PART_SELECTED_OUTPUT:?Missing campaign}"
: "${PART_SELECTED_PHASE:?Missing phase}"
[[ "$PART_SELECTED_OUTPUT" == /* && "$(uname -m)" == x86_64 ]] || exit 2
source "$PART_SELECTED_OUTPUT/launch.env"
selected_source="$PART_SELECTED_OUTPUT/source"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$selected_source"
selected_bin="$(dirname -- "$SELECTED_PYTHON")"
export PATH="$selected_bin:$PATH" LD_LIBRARY_PATH="$selected_bin/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MAX_JOBS=2
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 WORLD_SIZE=1
cd -- "$selected_source"
sha256sum --quiet -c "$PART_SELECTED_OUTPUT/source.sha256"
args=(--phase "$PART_SELECTED_PHASE" --output "$PART_SELECTED_OUTPUT")
case "$PART_SELECTED_PHASE" in
  benchmark|train|evaluate)
    [[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-3]$ ]] || { echo 'Need new-model array index 0..3' >&2; exit 2; }
    args+=(--task-index "$SLURM_ARRAY_TASK_ID") ;;
  prepare|gate|lock|report) ;;
  *) echo 'Invalid worker phase' >&2; exit 2 ;;
esac
exec "$SELECTED_PYTHON" -u "$selected_source/scripts/run_part_selected_screen.py" "${args[@]}"
