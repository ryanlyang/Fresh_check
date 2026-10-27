#!/usr/bin/env bash
set -euo pipefail
: "${PART_EV_STATE_CAPACITY_OUTPUT:?Missing campaign}"
: "${PART_EV_STATE_CAPACITY_PHASE:?Missing phase}"
[[ "$PART_EV_STATE_CAPACITY_OUTPUT" == /* ]] || exit 2
[[ "$(uname -m)" == x86_64 ]] || { echo 'Requires SPORC x86_64' >&2; exit 2; }
(cd -- "$PART_EV_STATE_CAPACITY_OUTPUT"; sha256sum --quiet -c launch.sha256)
source "$PART_EV_STATE_CAPACITY_OUTPUT/launch.env"
screen_source="$PART_EV_STATE_CAPACITY_OUTPUT/source"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$screen_source"
screen_bin="$(dirname -- "$STATE_CAPACITY_PYTHON")"
export PATH="$screen_bin:$PATH" LD_LIBRARY_PATH="$screen_bin/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 WORLD_SIZE=1
cd -- "$screen_source"
sha256sum --quiet -c "$PART_EV_STATE_CAPACITY_OUTPUT/source.sha256"
screen_args=(--phase "$PART_EV_STATE_CAPACITY_PHASE" --output "$PART_EV_STATE_CAPACITY_OUTPUT")
case "$PART_EV_STATE_CAPACITY_PHASE" in
  benchmark|train|evaluate)
    [[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-9]$ ]] || { echo 'Need array index 0..9' >&2; exit 2; }
    if [[ "$PART_EV_STATE_CAPACITY_PHASE" == train && "$SLURM_ARRAY_TASK_ID" -gt 5 ]]; then
      echo 'Controls 6..9 must never be trained by this campaign' >&2; exit 2
    fi
    screen_args+=(--task-index "$SLURM_ARRAY_TASK_ID") ;;
  prepare|lock|report) ;;
  *) echo 'Invalid worker phase' >&2; exit 2 ;;
esac
# exec preserves Slurm warning/termination signals for the checkpoint handler.
exec "$STATE_CAPACITY_PYTHON" -u "$screen_source/scripts/run_part_ev_state_capacity.py" "${screen_args[@]}"




