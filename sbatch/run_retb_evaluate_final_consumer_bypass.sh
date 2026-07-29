#!/usr/bin/env bash
#SBATCH --job-name=retb_bypass
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00

set -euo pipefail
IFS=$'\n\t'
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
: "${CONDA_ENV:=atlas_kd_tigris}"
: "${CAMPAIGN_ROOT:?CAMPAIGN_ROOT is required}"
: "${RETB_FINAL_CONSUMER_RUN:?RETB_FINAL_CONSUMER_RUN is required}"
: "${RETB_FINAL_CONSUMER_TEMPLATE:?RETB_FINAL_CONSUMER_TEMPLATE is required}"
: "${RETB_FINAL_CONSUMER_REGISTRATION:?RETB_FINAL_CONSUMER_REGISTRATION is required}"
: "${RETB_FINAL_CONSUMER_CHECKPOINT:?RETB_FINAL_CONSUMER_CHECKPOINT is required}"
: "${RETB_CONSUMER_VAL_DESIGN_CACHE:?RETB_CONSUMER_VAL_DESIGN_CACHE is required}"
: "${RETB_BYPASS_OUTPUT:?RETB_BYPASS_OUTPUT is required}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cd "${PROJECT_DIR}"
python scripts/evaluate_retb_final_consumer_bypass_controls.py \
  --campaign-root "${CAMPAIGN_ROOT}" --run "${RETB_FINAL_CONSUMER_RUN}" \
  --template "${RETB_FINAL_CONSUMER_TEMPLATE}" \
  --registration "${RETB_FINAL_CONSUMER_REGISTRATION}" \
  --checkpoint "${RETB_FINAL_CONSUMER_CHECKPOINT}" \
  --val-design-cache "${RETB_CONSUMER_VAL_DESIGN_CACHE}" \
  --output-dir "${RETB_BYPASS_OUTPUT}"
