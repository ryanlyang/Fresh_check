#!/usr/bin/env bash
set -euo pipefail
fast_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
fast_reference=/home/ryreu/atlas/Fresh_check/checkpoints/part_selected_tigris/selected_gh200_20261023T073044Z
fast_python=
fast_args=("$@")
while (( $# )); do
  case "$1" in
    --help|-h) exec python3 "$fast_repo/scripts/submit_part_selected_fast.py" --help ;;
    --python) fast_python=${2:?Missing Python}; shift 2 ;;
    --reference) fast_reference=${2:?Missing reference}; shift 2 ;;
    *) shift ;;
  esac
done
if [[ -z "$fast_python" ]]; then
  [[ -f "$fast_reference/launch.env" ]] || { echo 'Use --python or an existing --reference campaign.' >&2; exit 2; }
  source "$fast_reference/launch.env"
  fast_python=${SELECTED_PYTHON:?Original Python missing}
fi
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
exec "$fast_python" "$fast_repo/scripts/submit_part_selected_fast.py" "${fast_args[@]}"
