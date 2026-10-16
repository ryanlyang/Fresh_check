#!/usr/bin/env bash
set -euo pipefail
: "${PART_EV_SCREEN_OUTPUT:?Missing campaign}"
: "${PART_EV_SCREEN_PHASE:?Missing phase}"
[[ "$PART_EV_SCREEN_OUTPUT" == /* ]] || exit 2
[[ "$(uname -m)" == x86_64 ]] || { echo 'Requires SPORC x86_64' >&2; exit 2; }
source "$PART_EV_SCREEN_OUTPUT/launch.env"
screen_source="$PART_EV_SCREEN_OUTPUT/source"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$screen_source"
screen_bin="$(dirname -- "$SCREEN_PYTHON")"
export PATH="$screen_bin:$PATH" LD_LIBRARY_PATH="$screen_bin/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 WORLD_SIZE=1
cd -- "$screen_source"
sha256sum --quiet -c "$PART_EV_SCREEN_OUTPUT/source.sha256"
screen_args=(--phase "$PART_EV_SCREEN_PHASE" --output "$PART_EV_SCREEN_OUTPUT")
case "$PART_EV_SCREEN_PHASE" in
  benchmark|train|evaluate)
    [[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-6]$ ]] || { echo 'Need array index 0..6' >&2; exit 2; }
    screen_args+=(--task-index "$SLURM_ARRAY_TASK_ID") ;;
  prepare|report) ;;
  *) echo 'Invalid worker phase' >&2; exit 2 ;;
esac
# exec preserves Slurm warning/termination signals for the checkpoint handler.
exec "$SCREEN_PYTHON" -u "$screen_source/scripts/run_part_ev_screen.py" "${screen_args[@]}"
