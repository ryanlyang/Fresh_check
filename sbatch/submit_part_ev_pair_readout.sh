#!/usr/bin/env bash
set -euo pipefail
readout_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
readout_python=/home/ryreu/miniconda3/envs/part_ev_screen_sporc/bin/python
readout_args=("$@")
while (( $# )); do
  if [[ "$1" == --python ]]; then
    (( $# >= 2 )) || { echo 'Missing --python value' >&2; exit 2; }
    readout_python=$2; shift 2
  else shift; fi
done
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PATH="$(dirname -- "$readout_python"):$PATH"
export LD_LIBRARY_PATH="$(dirname -- "$readout_python")/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$readout_python" -B "$readout_repo/scripts/submit_part_ev_pair_readout.py" "${readout_args[@]}"
