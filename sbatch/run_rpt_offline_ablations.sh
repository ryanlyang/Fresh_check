#!/usr/bin/env bash
# The submitter sets resources; this path is safe when Slurm spools the script.
set -euo pipefail
: "${RPT_ABLATION_ROOT:?Missing supplemental campaign root}"
: "${RPT_ABLATION_PHASE:?Missing phase}"
: "${CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
: "${CONDA_ENV:=atlas_kd_tigris}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
cd -- "${RPT_ABLATION_ROOT}/source"
python -s scripts/run_relational_part_offline_ablations.py \
  "${RPT_ABLATION_PHASE}" --output "${RPT_ABLATION_ROOT}"
