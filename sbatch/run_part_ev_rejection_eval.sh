#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
# Worker re-execs the frozen donor Python; report needs only stdlib Python.
exec python3 -B "$repo/scripts/run_part_ev_rejection_eval.py" "$@"
