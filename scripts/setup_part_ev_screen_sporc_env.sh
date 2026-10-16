#!/usr/bin/env bash
# Create a new isolated x86 environment; never alter the donor or active jobs.
set -euo pipefail
screen_conda=/home/ryreu/miniconda3/bin/conda
screen_donor=/home/ryreu/miniconda3/envs/atlas_kd_sporc
screen_env=/home/ryreu/miniconda3/envs/part_ev_screen_sporc
screen_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
[[ "$(uname -m)" == x86_64 ]] || { echo 'Run on sporcsubmit, not Tigris.' >&2; exit 1; }
[[ -x "$screen_conda" && -x "$screen_donor/bin/python" ]] || { echo 'Missing x86 donor environment.' >&2; exit 1; }
[[ ! -e "$screen_env" ]] || { echo "Environment exists; leaving it unchanged: $screen_env"; exit 0; }
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
"$screen_conda" create --yes --prefix "$screen_env" --clone "$screen_donor"
export PATH="$screen_env/bin:$PATH" LD_LIBRARY_PATH="$screen_env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Preserve the CUDA-enabled donor PyTorch version; the rest is isolated here.
screen_torch_version="$("$screen_env/bin/python" -c 'import torch; assert torch.version.cuda; print(torch.__version__)')"
"$screen_env/bin/python" -m pip install --no-cache-dir "torch==$screen_torch_version" \
  -r "$screen_repo/requirements-part-fullscale.txt" 'matplotlib>=3.7,<4'
PYTHONPATH="$screen_repo" "$screen_env/bin/python" -c \
  'import json, matplotlib; from part_ev_fullscale.protocol import environment; print(json.dumps(environment(), indent=2))'
echo 'Environment prepared. GPU execution and memory are checked by the seven benchmark jobs.'
