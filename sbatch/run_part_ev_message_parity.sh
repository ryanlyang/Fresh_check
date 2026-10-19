#!/usr/bin/env bash
set -euo pipefail
[[ $# == 1 && "$1" == /* ]] || { echo 'Need absolute diagnostic directory' >&2; exit 2; }
output="$(realpath -e -- "$1")"
[[ "$(uname -m)" == x86_64 ]] || { echo 'Requires SPORC x86_64' >&2; exit 2; }
cd -- "$output"
sha256sum --quiet -c diagnostic.sha256
source ./launch.env
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
unset PYTHONPATH
parity_bin="$(dirname -- "$MESSAGE_PYTHON")"
export PATH="$parity_bin:$PATH" LD_LIBRARY_PATH="$parity_bin/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 WORLD_SIZE=1
campaign="$("$MESSAGE_PYTHON" -c 'import json; print(json.load(open("diagnostic_plan.json"))["campaign"])')"
(cd -- "$campaign"; sha256sum --quiet -c launch.sha256)
(cd -- "$campaign/source"; sha256sum --quiet -c "$campaign/source.sha256")
exec "$MESSAGE_PYTHON" -u "$output/diagnose_part_ev_message_parity.py" --phase run --output "$output"
