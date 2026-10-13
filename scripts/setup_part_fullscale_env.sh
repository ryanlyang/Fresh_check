#!/usr/bin/env bash
# Run once ON TIGRIS. Never upgrades the old atlas environment in place.
set -euo pipefail
part_conda=/home/ryreu/miniforge3-aarch64/bin/conda
part_donor=/home/ryreu/miniforge3-aarch64/envs/atlas_kd_tigris
part_env=/home/ryreu/miniforge3-aarch64/envs/part_ev_fullscale
part_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
[[ "$(uname -m)" == aarch64 ]] || { echo 'Run this on Tigris (aarch64), not SPORC.' >&2; exit 1; }
[[ -x "${part_conda}" && -x "${part_donor}/bin/python" ]] || { echo 'Missing donor CUDA environment.' >&2; exit 1; }
if [[ -e "${part_env}" ]]; then
  echo "Environment already exists: ${part_env}. Not modifying it."
  echo 'Run prepare to verify it; resolve any environment issue before submitting training.'
  exit 0
fi
"${part_conda}" create --yes --prefix "${part_env}" --clone "${part_donor}"
export PATH="${part_env}/bin:${PATH}" LD_LIBRARY_PATH="${part_env}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
"${part_env}/bin/python" -m pip install --no-cache-dir -r "${part_repo}/requirements-part-fullscale.txt"
PYTHONPATH="${part_repo}" PYTHONNOUSERSITE=1 "${part_env}/bin/python" -c \
  'import json; from part_ev_fullscale.protocol import environment; print(json.dumps(environment(), indent=2))'
echo "Prepared ${part_env}; GPU operation is checked separately by the benchmark job."
