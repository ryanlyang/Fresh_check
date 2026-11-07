#!/usr/bin/env bash
set -euo pipefail
restart_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
exec python3 "$restart_repo/scripts/submit_part_selected_restart.py" "$@"
