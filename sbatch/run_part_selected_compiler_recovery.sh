#!/usr/bin/env bash
set -euo pipefail
[[ $# == 3 && "$1" == /* && "$(uname -m)" == aarch64 ]] || exit 2
recovery_output="$(realpath -e -- "$1")"
cd -- "$recovery_output"
sha256sum --quiet -c snapshot.sha256
recovery_python="$(python3 -c 'import json; print(json.load(open("recovery.json"))["python"])')"
recovery_bin="$(dirname -- "$recovery_python")"
export PATH="$recovery_bin:$PATH" LD_LIBRARY_PATH="$recovery_bin/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$recovery_output/source" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 MAX_JOBS=2 WORLD_SIZE=1
unset RANK LOCAL_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
recovery_args=(--phase "$2" --output "$recovery_output")
case "$2" in
  train|evaluate)
    [[ "$3" =~ ^(2|5|6)$ ]] || exit 2
    recovery_args+=(--task-index "$3") ;;
  check|report) [[ "$3" == - ]] || exit 2 ;;
  *) exit 2 ;;
esac
exec "$recovery_python" -B -u "$recovery_output/source/scripts/run_part_selected_compiler_recovery.py" "${recovery_args[@]}"
