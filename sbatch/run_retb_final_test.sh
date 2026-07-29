#!/usr/bin/env bash
#SBATCH --job-name=retb_final_test
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --time=24:00:00
set -euo pipefail
IFS=$'\n\t'
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
: "${CONDA_ENV:=atlas_kd_tigris}"
: "${CAMPAIGN_ROOT:?CAMPAIGN_ROOT is required}"
: "${RETB_FINAL_EXECUTION_LOCK:?RETB_FINAL_EXECUTION_LOCK is required}"
: "${RETB_FINAL_LABELS_NPZ:?RETB_FINAL_LABELS_NPZ is required}"
: "${RETB_FINAL_LABELS_ARTIFACT_SHA256:?RETB_FINAL_LABELS_ARTIFACT_SHA256 is required}"
: "${RETB_FINAL_PREDICTION_INDEX:?RETB_FINAL_PREDICTION_INDEX is required}"
: "${RETB_FINAL_EVALUATION_OUTPUT_DIR:?RETB_FINAL_EVALUATION_OUTPUT_DIR is required}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cd "${PROJECT_DIR}"
arguments=(--campaign-root "${CAMPAIGN_ROOT}" --execution-lock "${RETB_FINAL_EXECUTION_LOCK}" --final-labels-npz "${RETB_FINAL_LABELS_NPZ}" --final-labels-artifact-sha256 "${RETB_FINAL_LABELS_ARTIFACT_SHA256}" --prediction-index "${RETB_FINAL_PREDICTION_INDEX}" --output-dir "${RETB_FINAL_EVALUATION_OUTPUT_DIR}")
if [[ "${RETB_DRY_RUN:-0}" == "1" ]]; then arguments+=(--dry-run); fi
python scripts/evaluate_retb_final_test.py "${arguments[@]}"
