#!/usr/bin/env bash
#SBATCH --job-name=retb_stack_infer
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --time=12:00:00
set -euo pipefail
IFS=$'\n\t'
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
: "${CONDA_ENV:=atlas_kd_tigris}"
: "${CAMPAIGN_ROOT:?CAMPAIGN_ROOT is required}"
: "${RETB_LOCKED_SCALE_SHORTLIST:?RETB_LOCKED_SCALE_SHORTLIST is required}"
: "${RETB_SCALE_COMPLETION:?RETB_SCALE_COMPLETION is required}"
: "${RETB_STACK_INFERENCE_NPZ:?RETB_STACK_INFERENCE_NPZ is required}"
: "${RETB_STACK_INFERENCE_CONFIGURATION:?RETB_STACK_INFERENCE_CONFIGURATION is required}"
: "${RETB_STACK_PREDICTION_OUTPUT_DIR:?RETB_STACK_PREDICTION_OUTPUT_DIR is required}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cd "${PROJECT_DIR}"
arguments=(--campaign-root "${CAMPAIGN_ROOT}" --locked-scale-shortlist "${RETB_LOCKED_SCALE_SHORTLIST}" --scale-completion "${RETB_SCALE_COMPLETION}" --inference-output-npz "${RETB_STACK_INFERENCE_NPZ}" --configuration "${RETB_STACK_INFERENCE_CONFIGURATION}" --output-dir "${RETB_STACK_PREDICTION_OUTPUT_DIR}")
if [[ "${RETB_DRY_RUN:-0}" == "1" ]]; then arguments+=(--dry-run); fi
python scripts/infer_retb_scale_stack_val.py "${arguments[@]}"
