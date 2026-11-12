#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
exec python3 -B "$repo/scripts/submit_part_ev_rejection_eval.py" "$@"
