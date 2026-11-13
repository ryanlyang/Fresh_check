#!/usr/bin/env bash
set -euo pipefail
evprod_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 -B "${evprod_repo}/scripts/submit_part_ev_state_production.py" "$@"
