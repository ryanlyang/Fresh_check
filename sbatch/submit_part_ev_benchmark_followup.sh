#!/usr/bin/env bash
set -euo pipefail
evfollow_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
exec python3 -B "$evfollow_repo/scripts/submit_part_ev_benchmark_followup.py" "$@"
