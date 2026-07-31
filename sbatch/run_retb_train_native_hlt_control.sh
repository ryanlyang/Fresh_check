#!/usr/bin/env bash
#SBATCH --job-name=retb_native_hlt_control
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
: "${RETB_TRAIN_LABELS:?RETB_TRAIN_LABELS is required}"
: "${RETB_VAL_STOP_LABELS:?RETB_VAL_STOP_LABELS is required}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cd "${PROJECT_DIR}"
arguments=(
  --campaign-root "${CAMPAIGN_ROOT}"
  --run-id "${RETB_RUN_ID}"
  --train-labels "${RETB_TRAIN_LABELS}"
  --val-stop-labels "${RETB_VAL_STOP_LABELS}"
  --device "${RETB_DEVICE:-auto}"
)
if [[ -n "${RETB_CONFIRMATION_REGISTRY:-}" ]]; then
  arguments+=(--confirmation-registry "${RETB_CONFIRMATION_REGISTRY}")
fi
for replica in 0 1 2 3; do
  train_variable="RETB_TRAIN_CACHE_R${replica}"
  val_variable="RETB_VAL_STOP_CACHE_R${replica}"
  if [[ -n "${!train_variable:-}" ]]; then
    arguments+=(--train-cache "${replica}=${!train_variable}")
  fi
  if [[ -n "${!val_variable:-}" ]]; then
    arguments+=(--val-stop-cache "${replica}=${!val_variable}")
  fi
done
if [[ "${RETB_DRY_RUN:-0}" == "1" ]]; then arguments+=(--dry-run); fi
python scripts/train_retb_native_hlt_control.py "${arguments[@]}"
