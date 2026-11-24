#!/usr/bin/env bash
set -euo pipefail
evms_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 -B "${evms_repo}/scripts/submit_part_ev_multiscale.py" "$@"

