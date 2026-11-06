#!/usr/bin/env bash
set -euo pipefail
seven_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
seven_reference=/home/ryreu/atlas/Fresh_check/checkpoints/part_selected_tigris/selected_gh200_20261023T073044Z
seven_python=
seven_args=("$@")
while (( $# )); do
  case "$1" in
    --help|-h) exec python3 "$seven_repo/scripts/submit_part_selected_fullscale.py" --help ;;
    --python) seven_python=${2:?Missing Python}; shift 2 ;;
    --reference) seven_reference=${2:?Missing reference}; shift 2 ;;
    *) shift ;;
  esac
done
if [[ -z "$seven_python" ]]; then
  [[ -f "$seven_reference/launch.env" ]] || { echo 'Specify --python for the existing Tigris environment.' >&2; exit 2; }
  source "$seven_reference/launch.env"
  seven_python=${SELECTED_PYTHON:?Original Python missing}
fi
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
exec "$seven_python" "$seven_repo/scripts/submit_part_selected_fullscale.py" "${seven_args[@]}" --python "$seven_python"
