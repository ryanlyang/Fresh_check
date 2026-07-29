#!/usr/bin/env bash
#SBATCH --job-name=retb_500k_seed
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
: "${RETB_STAGE_L_GRAPH_REGISTRY:?RETB_STAGE_L_GRAPH_REGISTRY is required}"
: "${RETB_CLASSIFICATION_METRICS:?RETB_CLASSIFICATION_METRICS is required}"
: "${RETB_PAIRED_STATISTICS:?RETB_PAIRED_STATISTICS is required}"
: "${RETB_SEED_CONFIRMATION_CONFIGURATION:?RETB_SEED_CONFIRMATION_CONFIGURATION is required}"
: "${RETB_SEED_CONFIRMATION_OUTPUT:?RETB_SEED_CONFIRMATION_OUTPUT is required}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cd "${PROJECT_DIR}"
arguments=(
  --campaign-root "${CAMPAIGN_ROOT}"
  --graph-registry "${RETB_STAGE_L_GRAPH_REGISTRY}"
  --classification-metrics "${RETB_CLASSIFICATION_METRICS}"
  --paired-statistics "${RETB_PAIRED_STATISTICS}"
  --configuration "${RETB_SEED_CONFIRMATION_CONFIGURATION}"
  --output "${RETB_SEED_CONFIRMATION_OUTPUT}"
)
if [[ "${RETB_DRY_RUN:-0}" == "1" ]]; then arguments+=(--dry-run); fi
python scripts/register_retb_500k_seed_confirmation.py "${arguments[@]}"
