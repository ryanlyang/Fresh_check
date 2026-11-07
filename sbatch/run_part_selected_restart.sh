#!/usr/bin/env bash
set -euo pipefail
[[ $# == 1 && "$1" == /* && "$(uname -m)" == aarch64 ]] || exit 2
restart_output="$(realpath -e -- "$1")"
cd -- "$restart_output"
sha256sum --quiet -c diagnostic.sha256
restart_python="$(python3 -c 'import json; print(json.load(open("plan.json"))["python"])')"
restart_source="$(python3 -c 'import json; print(json.load(open("plan.json"))["campaign"]+"/source")')"
restart_bin="$(dirname -- "$restart_python")"
export PATH="$restart_bin:$PATH" LD_LIBRARY_PATH="$restart_bin/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$restart_source" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 MAX_JOBS=2 WORLD_SIZE=1
exec "$restart_python" -B -u "$restart_output/diagnose_part_selected_restart.py" --phase run --output "$restart_output"
