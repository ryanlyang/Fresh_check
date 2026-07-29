#!/usr/bin/env bash
#SBATCH --job-name=retb_final_inputs
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=24:00:00
set -euo pipefail
IFS=$'\n\t'
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
: "${CONDA_ENV:=atlas_kd_tigris}"
: "${CAMPAIGN_ROOT:?CAMPAIGN_ROOT is required}"
: "${RETB_FINAL_INPUT_CONFIGURATION:?RETB_FINAL_INPUT_CONFIGURATION is required}"
: "${RETB_PRELOCK_FINAL_INPUTS_OUTPUT:?RETB_PRELOCK_FINAL_INPUTS_OUTPUT is required}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cd "${PROJECT_DIR}"
arguments=(--campaign-root "${CAMPAIGN_ROOT}" --configuration "${RETB_FINAL_INPUT_CONFIGURATION}" --output "${RETB_PRELOCK_FINAL_INPUTS_OUTPUT}")
if [[ "${RETB_DRY_RUN:-0}" == "1" ]]; then arguments+=(--dry-run); fi
python scripts/prepare_retb_final_test_inputs.py "${arguments[@]}"
