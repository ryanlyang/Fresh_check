#!/usr/bin/env bash
set -euo pipefail
compact_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
exec python3 -B "$compact_repo/scripts/submit_part_ev_compact_production.py" "$@"
