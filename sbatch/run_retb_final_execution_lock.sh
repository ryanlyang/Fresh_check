#!/usr/bin/env bash
#SBATCH --job-name=retb_final_lock
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
set -euo pipefail
IFS=$'\n\t'
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
: "${CONDA_ENV:=atlas_kd_tigris}"
: "${CAMPAIGN_ROOT:?CAMPAIGN_ROOT is required}"
: "${RETB_LOCKED_SCALE_FINALISTS:?RETB_LOCKED_SCALE_FINALISTS is required}"
: "${RETB_FINALIST_CONTROLS:?RETB_FINALIST_CONTROLS is required}"
: "${RETB_PRELOCK_FINAL_INPUTS:?RETB_PRELOCK_FINAL_INPUTS is required}"
: "${RETB_FINAL_EXECUTION_CONFIGURATION:?RETB_FINAL_EXECUTION_CONFIGURATION is required}"
: "${RETB_FINAL_EXECUTION_LOCK_OUTPUT:?RETB_FINAL_EXECUTION_LOCK_OUTPUT is required}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cd "${PROJECT_DIR}"
arguments=(--campaign-root "${CAMPAIGN_ROOT}" --locked-scale-finalists "${RETB_LOCKED_SCALE_FINALISTS}" --finalist-controls "${RETB_FINALIST_CONTROLS}" --prelock-final-inputs "${RETB_PRELOCK_FINAL_INPUTS}" --configuration "${RETB_FINAL_EXECUTION_CONFIGURATION}" --output "${RETB_FINAL_EXECUTION_LOCK_OUTPUT}")
IFS=':' read -r -a target_paths <<< "${RETB_POSTLOCK_TARGET_PATHS:?RETB_POSTLOCK_TARGET_PATHS is required}"
for path in "${target_paths[@]}"; do arguments+=(--postlock-target "${path}"); done
if [[ "${RETB_DRY_RUN:-0}" == "1" ]]; then arguments+=(--dry-run); fi
python scripts/write_retb_final_test_execution_lock.py "${arguments[@]}"
