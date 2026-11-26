#!/usr/bin/env bash
set -euo pipefail
speed_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 -B "${speed_repo}/scripts/submit_part_ev_masked_speed.py" "$@"
