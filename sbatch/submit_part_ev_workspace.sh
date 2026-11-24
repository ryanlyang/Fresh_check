#!/usr/bin/env bash
set -euo pipefail
evws_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 -B "${evws_repo}/scripts/submit_part_ev_workspace.py" "$@"
