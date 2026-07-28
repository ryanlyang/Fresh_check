#!/usr/bin/env bash
#SBATCH --job-name=retb_select_shapes
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail
IFS=$'\n\t'
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
: "${CONDA_ENV:=atlas_kd_tigris}"
: "${CAMPAIGN_ROOT:?CAMPAIGN_ROOT is required}"
: "${RETB_UNIFORM_SHAPE_METRICS:?RETB_UNIFORM_SHAPE_METRICS is required}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cd "${PROJECT_DIR}"

arguments=(
  --campaign-root "${CAMPAIGN_ROOT}"
  --uniform-metrics "${RETB_UNIFORM_SHAPE_METRICS}"
)
if [[ -n "${RETB_BASELINE_MEAN_ACCURACY:-}" ]]; then
  arguments+=(--baseline-mean-accuracy "${RETB_BASELINE_MEAN_ACCURACY}")
fi
if [[ "${RETB_DRY_RUN:-0}" == "1" ]]; then
  arguments+=(--dry-run)
fi
python scripts/select_retb_offline_shapes.py "${arguments[@]}"
