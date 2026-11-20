#!/usr/bin/env bash
set -euo pipefail
evrts_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 -B "${evrts_repo}/scripts/submit_part_ev_rts_tigris.py" "$@"
