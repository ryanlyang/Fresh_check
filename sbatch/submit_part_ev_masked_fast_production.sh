#!/usr/bin/env bash
set -euo pipefail
fast_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 -B "${fast_repo}/scripts/submit_part_ev_masked_fast_production.py" "$@"
