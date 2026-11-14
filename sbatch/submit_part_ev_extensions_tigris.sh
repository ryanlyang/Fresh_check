#!/usr/bin/env bash
set -euo pipefail
evext_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
exec python3 -B "${evext_repo}/scripts/submit_part_ev_extensions_tigris.py" "$@"
