#!/usr/bin/env bash
set -euo pipefail
[[ $# == 2 && "$1" == /* ]] || { echo 'Need absolute recovery directory and phase' >&2; exit 2; }
output="$(realpath -e -- "$1")"
phase=$2
[[ "$(uname -m)" == x86_64 ]] || { echo 'Requires SPORC x86_64' >&2; exit 2; }
cd -- "$output"
sha256sum --quiet -c diagnostic.sha256
source ./launch.env
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
unset PYTHONPATH
recovery_bin="$(dirname -- "$SCREEN_PYTHON")"
export PATH="$recovery_bin:$PATH" LD_LIBRARY_PATH="$recovery_bin/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 WORLD_SIZE=1
args=(--phase "$phase" --output "$output")
case "$phase" in
  run)
    [[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-6]$ ]] || { echo 'Need task index 0..6' >&2; exit 2; }
    args+=(--task-index "$SLURM_ARRAY_TASK_ID") ;;
  summary) ;;
  *) echo 'Invalid recovery phase' >&2; exit 2 ;;
esac
exec "$SCREEN_PYTHON" -u "$output/check_part_ev_screen_amp_recovery.py" "${args[@]}"
