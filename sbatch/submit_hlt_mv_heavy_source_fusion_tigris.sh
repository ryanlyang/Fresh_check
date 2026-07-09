#!/usr/bin/env bash
# TIGRIS/GH200 defaults for the heavy HLT-MV source/fusion graph.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"

: "${HLT_MV_SBATCH_PARTITION:=tigris}"
: "${HLT_MV_GPU_GRES:=gpu:gh200:1}"
: "${HLT_MV_GPU_CPUS_PER_TASK:=16}"
: "${HLT_MV_GPU_MEM:=300G}"
: "${HLT_MV_CPU_CPUS_PER_TASK:=16}"
: "${HLT_MV_CPU_MEM:=220G}"
: "${DEVICE:=cuda}"
: "${CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
: "${CONDA_ENV:=atlas_kd_tigris}"
: "${PYTHONNOUSERSITE:=1}"
: "${HLT_MV_TIGRIS_PREFLIGHT:=1}"

export HLT_MV_SBATCH_PARTITION HLT_MV_GPU_GRES
export HLT_MV_GPU_CPUS_PER_TASK HLT_MV_GPU_MEM HLT_MV_GPU_TIME
export HLT_MV_CPU_CPUS_PER_TASK HLT_MV_CPU_MEM HLT_MV_CPU_TIME
export DEVICE CONDA_BASE CONDA_ENV PYTHONNOUSERSITE

if [[ "${HLT_MV_TIGRIS_PREFLIGHT}" != "0" ]]; then
  conda_sh="${CONDA_BASE}/etc/profile.d/conda.sh"
  if [[ ! -f "${conda_sh}" ]]; then
    echo "TIGRIS preflight failed: missing conda.sh at ${conda_sh}" >&2
    exit 2
  fi
  # shellcheck disable=SC1090
  source "${conda_sh}"
  conda activate "${CONDA_ENV}"
  python - <<'PY'
import platform
import torch
from weaver.nn.model.ParticleTransformer import ParticleTransformer

if platform.machine() != "aarch64":
    raise SystemExit(f"TIGRIS preflight expected aarch64, got {platform.machine()}")
if not torch.backends.cuda.is_built():
    raise SystemExit("TIGRIS preflight requires a CUDA-built PyTorch.")
print("TIGRIS preflight ok:", platform.machine(), "torch", torch.__version__, "cuda", torch.version.cuda)
print("weaver ParticleTransformer ok:", ParticleTransformer.__name__)
PY
fi

exec bash "${SCRIPT_DIR}/submit_hlt_mv_heavy_source_fusion.sh"
