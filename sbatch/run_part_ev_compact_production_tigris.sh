#!/usr/bin/env bash
set -euo pipefail
[[ $# == 3 && "$1" == /* && "$(uname -m)" == aarch64 ]] || exit 2
compact_output="$(realpath -e -- "$1")"
compact_phase="$2"
cd -- "$compact_output"
sha256sum --quiet -c snapshot.sha256
compact_python="$(python3 -c 'import json; print(json.load(open("production.json"))["python"])')"
compact_science="$(python3 -c 'import json; print(json.load(open("production.json"))["benchmark"]+"/source")')"
compact_bin="$(dirname -- "$compact_python")"
export PATH="$compact_bin:$PATH" LD_LIBRARY_PATH="$compact_bin/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$compact_science" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 MAX_JOBS=2 WORLD_SIZE=1
unset RANK LOCAL_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
compact_args=(--phase "$compact_phase" --output "$compact_output")
case "$compact_phase" in
  train|evaluate) [[ "$3" =~ ^[01]$ ]] || exit 2; compact_args+=(--task-index "$3") ;;
  report) [[ "$3" == - ]] || exit 2 ;;
  *) exit 2 ;;
esac
exec "$compact_python" -B -u "$compact_output/source/scripts/run_part_ev_compact_production.py" "${compact_args[@]}"
