#!/usr/bin/env bash
set -euo pipefail
evmp_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 -B "${evmp_repo}/scripts/submit_part_ev_masked_predictive.py" "$@"
