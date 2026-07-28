#!/usr/bin/env bash
#SBATCH --job-name=retb_offline_fusion
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --time=2-00:00:00

set -euo pipefail
IFS=$'\n\t'
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
: "${CONDA_ENV:=atlas_kd_tigris}"
: "${CAMPAIGN_ROOT:?CAMPAIGN_ROOT is required}"
: "${RETB_RUN_ID:?RETB_RUN_ID is required}"
: "${RETB_MODEL_TRAIN_CACHE:?RETB_MODEL_TRAIN_CACHE is required}"
: "${RETB_VAL_STOP_CACHE:?RETB_VAL_STOP_CACHE is required}"
: "${RETB_FUSION_BATCH_SIZE:=512}"
: "${RETB_DEVICE:=auto}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cd "${PROJECT_DIR}"

arguments=(
  --campaign-root "${CAMPAIGN_ROOT}"
  --run-id "${RETB_RUN_ID}"
  --model-train-cache "${RETB_MODEL_TRAIN_CACHE}"
  --val-stop-cache "${RETB_VAL_STOP_CACHE}"
  --batch-size "${RETB_FUSION_BATCH_SIZE}"
  --device "${RETB_DEVICE}"
)
if [[ -n "${RETB_OUTPUT_DIR:-}" ]]; then
  arguments+=(--output-dir "${RETB_OUTPUT_DIR}")
fi
if [[ "${RETB_DRY_RUN:-0}" == "1" ]]; then
  arguments+=(--dry-run)
fi
python scripts/train_retb_offline_fusion.py "${arguments[@]}"
