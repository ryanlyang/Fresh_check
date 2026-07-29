#!/usr/bin/env bash
#SBATCH --job-name=retb_predictor_spec
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -euo pipefail
IFS=$'\n\t'
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
: "${CONDA_ENV:=atlas_kd_tigris}"
: "${CAMPAIGN_ROOT:?CAMPAIGN_ROOT is required}"
: "${RETB_PREDICTOR_CONFIGURATION:?RETB_PREDICTOR_CONFIGURATION is required}"
: "${RETB_PREDICTOR_RUN_OUTPUT:?RETB_PREDICTOR_RUN_OUTPUT is required}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cd "${PROJECT_DIR}"

python scripts/materialize_retb_predictor_run.py \
  --campaign-root "${CAMPAIGN_ROOT}" \
  --configuration "${RETB_PREDICTOR_CONFIGURATION}" \
  --output "${RETB_PREDICTOR_RUN_OUTPUT}"
