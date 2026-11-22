#!/usr/bin/env bash
set -euo pipefail
evtri_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
exec python3 -B "$evtri_repo/scripts/submit_part_ev_triangles_restart.py" "$@"
