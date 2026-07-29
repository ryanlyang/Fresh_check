#!/usr/bin/env bash
#SBATCH --job-name=retb_consumer
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00

set -euo pipefail
IFS=$'\n\t'
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
: "${CONDA_ENV:=atlas_kd_tigris}"
: "${CAMPAIGN_ROOT:?CAMPAIGN_ROOT is required}"
: "${RETB_FINAL_CONSUMER_RUN:?RETB_FINAL_CONSUMER_RUN is required}"
: "${RETB_FINAL_CONSUMER_TEMPLATE:?RETB_FINAL_CONSUMER_TEMPLATE is required}"
: "${RETB_CONSUMER_MODEL_TRAIN_CACHE:?RETB_CONSUMER_MODEL_TRAIN_CACHE is required}"
: "${RETB_CONSUMER_VAL_STOP_CACHE:?RETB_CONSUMER_VAL_STOP_CACHE is required}"
: "${RETB_CONSUMER_VAL_DESIGN_CACHE:?RETB_CONSUMER_VAL_DESIGN_CACHE is required}"
: "${RETB_FINAL_CONSUMER_OUTPUT:?RETB_FINAL_CONSUMER_OUTPUT is required}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cd "${PROJECT_DIR}"
arguments=(
  --campaign-root "${CAMPAIGN_ROOT}" --run "${RETB_FINAL_CONSUMER_RUN}"
  --template "${RETB_FINAL_CONSUMER_TEMPLATE}"
  --model-train-cache "${RETB_CONSUMER_MODEL_TRAIN_CACHE}"
  --val-stop-cache "${RETB_CONSUMER_VAL_STOP_CACHE}"
  --val-design-cache "${RETB_CONSUMER_VAL_DESIGN_CACHE}"
  --output-dir "${RETB_FINAL_CONSUMER_OUTPUT}"
  --microbatch-size "${RETB_CONSUMER_MICROBATCH_SIZE:-0}"
  --gradient-accumulation-steps "${RETB_CONSUMER_ACCUMULATION_STEPS:-1}"
)
if [[ "${RETB_DRY_RUN:-0}" == "1" ]]; then arguments+=(--dry-run); fi
python scripts/train_retb_final_consumer.py "${arguments[@]}"
