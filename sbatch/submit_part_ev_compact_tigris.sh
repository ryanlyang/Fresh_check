#!/usr/bin/env bash
set -euo pipefail
compact_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
# Submitter uses stdlib only; GPU imports occur in the frozen batch workers.
exec python3 -B "$compact_repo/scripts/submit_part_ev_compact.py" "$@"
