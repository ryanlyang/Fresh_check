#!/usr/bin/env bash
set -euo pipefail
: "${PART_LAYERWISE_OUTPUT:?Missing campaign}"
: "${PART_LAYERWISE_PHASE:?Missing phase}"
[[ "$PART_LAYERWISE_OUTPUT" == /* && "$(uname -m)" == x86_64 ]] || exit 2
source "$PART_LAYERWISE_OUTPUT/launch.env"
layer_source="$PART_LAYERWISE_OUTPUT/source"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$layer_source"
layer_bin="$(dirname -- "$LAYER_PYTHON")"
export PATH="$layer_bin:$PATH" LD_LIBRARY_PATH="$layer_bin/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 WORLD_SIZE=1
cd -- "$layer_source"
sha256sum --quiet -c "$PART_LAYERWISE_OUTPUT/source.sha256"
args=(--phase "$PART_LAYERWISE_PHASE" --output "$PART_LAYERWISE_OUTPUT")
case "$PART_LAYERWISE_PHASE" in
  train|evaluate)
    [[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^[01]$ ]] || { echo 'Need new-model array index 0 or 1' >&2; exit 2; }
    args+=(--task-index "$SLURM_ARRAY_TASK_ID") ;;
  lock|report) ;;
  *) echo 'Invalid worker phase' >&2; exit 2 ;;
esac
exec "$LAYER_PYTHON" -u "$layer_source/scripts/run_part_layerwise_screen.py" "${args[@]}"
