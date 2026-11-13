#!/usr/bin/env bash
# Invoke from the transferred tool snapshot. Installs only into a NEW scratch venv.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH
exec python3 -B "$script_dir/part_ev_oscar.py" setup-env "$@"
