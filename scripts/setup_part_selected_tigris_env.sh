#!/usr/bin/env bash
# Optional one-time ARM environment creation. Never modify full-scale/SPORC environments.
set -euo pipefail
[[ "$(uname -m)" == aarch64 ]] || { echo 'Run on Tigris (ARM), not SPORC.' >&2; exit 2; }
selected_conda=/home/ryreu/miniforge3-aarch64/bin/conda
selected_donor=/home/ryreu/miniforge3-aarch64/envs/part_ev_fullscale
selected_env=/home/ryreu/miniforge3-aarch64/envs/part_selected_tigris
selected_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
[[ -x "$selected_conda" && -x "$selected_donor/bin/python" ]] || {
  echo 'Missing ARM full-scale donor environment; no existing environment modified.' >&2; exit 2;
}
if [[ -e "$selected_env" || -L "$selected_env" ]]; then
  echo "Existing $selected_env left unchanged; checking dependencies."
else
  "$selected_conda" create --yes --prefix "$selected_env" --clone "$selected_donor"
  # No torch/numpy/weaver upgrade. Ninja is needed only for the C++ tree build.
  "$selected_env/bin/python" -m pip install --no-cache-dir --no-deps --only-binary=:all: ninja==1.13.2
fi
export PATH="$selected_env/bin:$PATH"
export LD_LIBRARY_PATH="$selected_env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
PYTHONPATH="$selected_repo" "$selected_env/bin/python" - <<'PY'
import shutil, torch
from part_ev_fullscale.model import reference_metadata
reference_metadata()
if torch.version.cuda is None:
    raise RuntimeError('The cloned environment is not a CUDA PyTorch build')
for command in ('ninja', 'c++'):
    if not shutil.which(command):
        raise RuntimeError(f'Missing {command}; nothing installed into an existing environment')
print('ARM dependencies available. Actual GPU operation is checked by the benchmark jobs.')
PY
