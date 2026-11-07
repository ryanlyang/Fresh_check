#!/usr/bin/env bash
set -euo pipefail
retry_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
exec python3 "$retry_repo/scripts/retry_part_selected_fullscale.py" "$@"
