#!/usr/bin/env bash
#SBATCH --job-name=retb_target_norm
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail
IFS=$'\n\t'
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
: "${CONDA_ENV:=atlas_kd_tigris}"
: "${CAMPAIGN_ROOT:?CAMPAIGN_ROOT is required}"
: "${RETB_TARGET_CACHE_MANIFEST:?RETB_TARGET_CACHE_MANIFEST is required}"
: "${RETB_PIPELINE_SEED:?RETB_PIPELINE_SEED is required}"
: "${RETB_TARGET_SPECIFICATION_SHA256:?RETB_TARGET_SPECIFICATION_SHA256 is required}"
: "${RETB_TARGET_NORMALIZER_OUTPUT:?RETB_TARGET_NORMALIZER_OUTPUT is required}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cd "${PROJECT_DIR}"

python scripts/fit_retb_target_normalizers.py \
  --campaign-root "${CAMPAIGN_ROOT}" \
  --target-cache-manifest "${RETB_TARGET_CACHE_MANIFEST}" \
  --pipeline-seed "${RETB_PIPELINE_SEED}" \
  --specification-sha256 "${RETB_TARGET_SPECIFICATION_SHA256}" \
  --output-dir "${RETB_TARGET_NORMALIZER_OUTPUT}"
