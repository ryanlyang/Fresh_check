#!/usr/bin/env bash
#SBATCH --job-name=retb_consumer_cache
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=08:00:00

set -euo pipefail
IFS=$'\n\t'
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
: "${CONDA_ENV:=atlas_kd_tigris}"
: "${CAMPAIGN_ROOT:?CAMPAIGN_ROOT is required}"
: "${RETB_FINAL_CONSUMER_RUN:?RETB_FINAL_CONSUMER_RUN is required}"
: "${RETB_PREPARED_CONSUMER_DATASET:?RETB_PREPARED_CONSUMER_DATASET is required}"
: "${RETB_PREPARED_CONSUMER_DATASET_SHA256:?RETB_PREPARED_CONSUMER_DATASET_SHA256 is required}"
: "${RETB_CONSUMER_DATASET_PARENTS:?RETB_CONSUMER_DATASET_PARENTS is required}"
: "${RETB_CONSUMER_DATASET_OUTPUT:?RETB_CONSUMER_DATASET_OUTPUT is required}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cd "${PROJECT_DIR}"
python scripts/register_retb_final_consumer_dataset.py \
  --campaign-root "${CAMPAIGN_ROOT}" --run "${RETB_FINAL_CONSUMER_RUN}" \
  --prepared-dataset "${RETB_PREPARED_CONSUMER_DATASET}" \
  --prepared-dataset-sha256 "${RETB_PREPARED_CONSUMER_DATASET_SHA256}" \
  --parent-hashes "${RETB_CONSUMER_DATASET_PARENTS}" \
  --output-dir "${RETB_CONSUMER_DATASET_OUTPUT}"
