#!/usr/bin/env bash
set -euo pipefail
support_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
support_python=/home/ryreu/miniconda3/envs/part_ev_screen_sporc/bin/python
support_args=("$@")
while (( $# )); do
  if [[ "$1" == --python ]]; then
    (( $# >= 2 )) || { echo 'Missing --python value' >&2; exit 2; }
    support_python=$2; shift 2
  else shift; fi
done
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PATH="$(dirname -- "$support_python"):$PATH"
export LD_LIBRARY_PATH="$(dirname -- "$support_python")/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$support_python" -B "$support_repo/scripts/submit_part_ev_support.py" "${support_args[@]}"
