#!/usr/bin/env bash
set -euo pipefail
: "${SELECTED_FULL_OUTPUT:?Missing campaign}"
: "${SELECTED_FULL_PHASE:?Missing phase}"
[[ "$SELECTED_FULL_OUTPUT" == /* && "$(uname -m)" == aarch64 ]] || exit 2
seven_source="$SELECTED_FULL_OUTPUT/source"
seven_python="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["python"])' "$SELECTED_FULL_OUTPUT/launch.json")"
seven_bin="$(dirname -- "$seven_python")"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$seven_source"
export PATH="$seven_bin:$PATH" LD_LIBRARY_PATH="$seven_bin/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MAX_JOBS=2 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 WORLD_SIZE=1
cd -- "$seven_source"
"$seven_python" - "$SELECTED_FULL_OUTPUT" <<'PY'
import sys
from pathlib import Path
from scripts.submit_part_selected_fast import verify_snapshot
verify_snapshot(Path(sys.argv[1]))
PY
seven_args=(--phase "$SELECTED_FULL_PHASE" --output "$SELECTED_FULL_OUTPUT")
case "$SELECTED_FULL_PHASE" in
  profile|benchmark)
    [[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-6]$ ]] || exit 2
    seven_args+=(--task-index "$SLURM_ARRAY_TASK_ID") ;;
  prepare|summary) ;;
  *) echo 'Only benchmark preparation/profile/measurement/summary phases are allowed.' >&2; exit 2 ;;
esac
exec "$seven_python" -u "$seven_source/scripts/run_part_selected_fullscale.py" "${seven_args[@]}"
