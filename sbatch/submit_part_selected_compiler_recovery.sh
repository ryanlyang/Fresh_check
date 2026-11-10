#!/usr/bin/env bash
set -euo pipefail
recovery_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
exec python3 -B "$recovery_repo/scripts/submit_part_selected_compiler_recovery.py" "$@"
