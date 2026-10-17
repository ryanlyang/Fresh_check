#!/usr/bin/env bash
set -euo pipefail
[[ $# == 3 && "$1" == /* && "$3" =~ ^[01]$ ]] || { echo 'Need approval directory, phase, resume flag' >&2; exit 2; }
approval="$(realpath -e -- "$1")"
phase=$2
resume=$3
[[ "$(uname -m)" == x86_64 ]] || { echo 'Requires SPORC x86_64' >&2; exit 2; }
cd -- "$approval"
sha256sum --quiet -c approval.sha256
source ./launch.env
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
unset PYTHONPATH
screen_bin="$(dirname -- "$SCREEN_PYTHON")"
export PATH="$screen_bin:$PATH" LD_LIBRARY_PATH="$screen_bin/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 WORLD_SIZE=1
campaign="$("$SCREEN_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["campaign"])' "$approval/approval.json")"
(cd -- "$campaign/source"; sha256sum --quiet -c "$campaign/source.sha256")
args=(--phase "$phase" --approval "$approval")
case "$phase" in
  train|evaluate)
    [[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-6]$ ]] || { echo 'Need task index 0..6' >&2; exit 2; }
    args+=(--task-index "$SLURM_ARRAY_TASK_ID") ;;
  report) ;;
  *) echo 'Invalid approved production phase' >&2; exit 2 ;;
esac
(( ! resume )) || args+=(--resume)
exec "$SCREEN_PYTHON" -u "$approval/run_part_ev_screen_after_recovery.py" "${args[@]}"
