#!/usr/bin/env bash
set -euo pipefail
[[ $# == 3 && "$1" == /* && "$(uname -m)" == aarch64 ]] || exit 2
production_output="$(realpath -e -- "$1")"
production_phase="$2"
cd -- "$production_output"
sha256sum --quiet -c snapshot.sha256
production_python="$(python3 -c 'import json; print(json.load(open("production.json"))["python"])')"
production_science="$(python3 -c 'import json; print(json.load(open("production.json"))["benchmark"]+"/source")')"
production_bin="$(dirname -- "$production_python")"
export PATH="$production_bin:$PATH" LD_LIBRARY_PATH="$production_bin/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$production_science" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 MAX_JOBS=2 WORLD_SIZE=1
unset RANK LOCAL_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
production_args=(--phase "$production_phase" --output "$production_output")
case "$production_phase" in
  train|evaluate)
    [[ "$3" =~ ^[0-6]$ ]] || exit 2
    production_args+=(--task-index "$3") ;;
  report) [[ "$3" == - ]] || exit 2 ;;
  *) exit 2 ;;
esac
exec "$production_python" -B -u "$production_output/source/scripts/run_part_selected_production.py" "${production_args[@]}"
