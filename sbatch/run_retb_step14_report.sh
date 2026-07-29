#!/usr/bin/env bash
#SBATCH --job-name=retb_mn_report
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
: "${RETB_SCALE_COMPLETION:?RETB_SCALE_COMPLETION is required}"
: "${RETB_LOCKED_SCALE_FINALISTS:?RETB_LOCKED_SCALE_FINALISTS is required}"
: "${RETB_FINAL_EXECUTION_LOCK:?RETB_FINAL_EXECUTION_LOCK is required}"
: "${RETB_FINAL_EVALUATION:?RETB_FINAL_EVALUATION is required}"
: "${RETB_STEP14_REPORT_OUTPUT_DIR:?RETB_STEP14_REPORT_OUTPUT_DIR is required}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cd "${PROJECT_DIR}"
arguments=(--campaign-root "${CAMPAIGN_ROOT}" --scale-completion "${RETB_SCALE_COMPLETION}" --locked-scale-finalists "${RETB_LOCKED_SCALE_FINALISTS}" --execution-lock "${RETB_FINAL_EXECUTION_LOCK}" --final-evaluation "${RETB_FINAL_EVALUATION}" --output-dir "${RETB_STEP14_REPORT_OUTPUT_DIR}")
if [[ "${RETB_DRY_RUN:-0}" == "1" ]]; then arguments+=(--dry-run); fi
python scripts/write_retb_step14_report.py "${arguments[@]}"
