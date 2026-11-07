#!/usr/bin/env bash
set -euo pipefail
production_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
exec python3 -B "$production_repo/scripts/submit_part_selected_production.py" "$@"
