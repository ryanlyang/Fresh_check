#!/usr/bin/env bash
set -euo pipefail
selcompact_output=${1:?Missing campaign}
selcompact_phase=${2:?Missing phase}
[[ "$selcompact_output" == /* && "$(uname -m)" == aarch64 ]] || exit 2
selcompact_source="$selcompact_output/source"
selcompact_python="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["python"])' "$selcompact_output/launch.json")"
selcompact_bin="$(dirname -- "$selcompact_python")"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$selcompact_source"
export PATH="$selcompact_bin:$PATH" LD_LIBRARY_PATH="$selcompact_bin/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 WORLD_SIZE=1
unset RANK LOCAL_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
cd -- "$selcompact_source"
selcompact_args=(--phase "$selcompact_phase" --output "$selcompact_output")
case "$selcompact_phase" in
  benchmark)
    [[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^[01]$ ]] || exit 2
    selcompact_args+=(--task-index "$SLURM_ARRAY_TASK_ID") ;;
  prepare|summary) ;;
  *) echo 'Benchmark-only worker; no production phase.' >&2; exit 2 ;;
esac
exec "$selcompact_python" -B -u "$selcompact_source/scripts/run_part_selected_compact.py" "${selcompact_args[@]}"
