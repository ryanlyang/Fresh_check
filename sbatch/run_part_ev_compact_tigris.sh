#!/usr/bin/env bash
set -euo pipefail
compact_output=${1:?Missing campaign}
compact_phase=${2:?Missing phase}
[[ "$compact_output" == /* && "$(uname -m)" == aarch64 ]] || exit 2
compact_source="$compact_output/source"
compact_python="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["python"])' "$compact_output/launch.json")"
compact_bin="$(dirname -- "$compact_python")"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$compact_source"
export PATH="$compact_bin:$PATH" LD_LIBRARY_PATH="$compact_bin/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 WORLD_SIZE=1
unset RANK LOCAL_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
cd -- "$compact_source"
compact_args=(--phase "$compact_phase" --output "$compact_output")
case "$compact_phase" in
  benchmark)
    [[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^[01]$ ]] || exit 2
    compact_args+=(--task-index "$SLURM_ARRAY_TASK_ID") ;;
  prepare|summary) ;;
  *) echo 'Benchmark-only worker; no production phase.' >&2; exit 2 ;;
esac
exec "$compact_python" -B -u "$compact_source/scripts/run_part_ev_compact.py" "${compact_args[@]}"
