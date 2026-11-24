#!/usr/bin/env bash
set -euo pipefail
evex_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 -B "${evex_repo}/scripts/submit_part_ev_ffn_experts.py" "$@"
