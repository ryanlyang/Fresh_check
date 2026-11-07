#!/usr/bin/env bash
set -euo pipefail
[[ $# == 2 && "$1" == /* && "$(uname -m)" == aarch64 ]] || exit 2
retry_output="$(realpath -e -- "$1")"
retry_phase="$2"
cd -- "$retry_output"
sha256sum --quiet -c retry.sha256
retry_python="$(python3 -c 'import json; print(json.load(open("retry_plan.json"))["python"])')"
retry_source="$(python3 -c 'import json; print(json.load(open("retry_plan.json"))["campaign"]+"/source")')"
retry_bin="$(dirname -- "$retry_python")"
export PATH="$retry_bin:$PATH" LD_LIBRARY_PATH="$retry_bin/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$retry_source" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 MAX_JOBS=2 WORLD_SIZE=1
retry_args=(--mode "$retry_phase" --output "$retry_output")
case "$retry_phase" in
  benchmark)
    [[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0134]$ ]] || exit 2
    retry_args+=(--task-index "$SLURM_ARRAY_TASK_ID") ;;
  summary|review) ;;
  *) exit 2 ;;
esac
exec "$retry_python" -B -u "$retry_output/retry_part_selected_fullscale.py" "${retry_args[@]}"
