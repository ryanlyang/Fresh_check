#!/usr/bin/env bash
set -euo pipefail
: "${PART_SELECTED_FAST_OUTPUT:?Missing campaign}"
: "${PART_SELECTED_FAST_PHASE:?Missing phase}"
[[ "$PART_SELECTED_FAST_OUTPUT" == /* && "$(uname -m)" == aarch64 ]] || exit 2
fast_source="$PART_SELECTED_FAST_OUTPUT/source"
# launch.json is written by our standard-library submitter, never shell-evaluated.
fast_python="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["python"])' "$PART_SELECTED_FAST_OUTPUT/launch.json")"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$fast_source"
fast_bin="$(dirname -- "$fast_python")"
export PATH="$fast_bin:$PATH" LD_LIBRARY_PATH="$fast_bin/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MAX_JOBS=2 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 WORLD_SIZE=1
cd -- "$fast_source"
"$fast_python" - "$PART_SELECTED_FAST_OUTPUT" <<'PY'
import sys
from pathlib import Path
from scripts.submit_part_selected_fast import verify_snapshot
verify_snapshot(Path(sys.argv[1]))
PY
fast_args=(--phase "$PART_SELECTED_FAST_PHASE" --output "$PART_SELECTED_FAST_OUTPUT")
case "$PART_SELECTED_FAST_PHASE" in
  profile|benchmark|train|evaluate)
    [[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-3]$ ]] || exit 2
    fast_args+=(--task-index "$SLURM_ARRAY_TASK_ID") ;;
  prepare|gate|lock|report) ;;
  *) echo 'Invalid phase' >&2; exit 2 ;;
esac
exec "$fast_python" -u "$fast_source/scripts/run_part_selected_fast.py" "${fast_args[@]}"
