#!/usr/bin/env bash
#SBATCH --job-name=retb_scale_refit
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
: "${RETB_LOCKED_SCALE_SHORTLIST:?RETB_LOCKED_SCALE_SHORTLIST is required}"
: "${RETB_SCALE_REFIT_CONFIGURATION:?RETB_SCALE_REFIT_CONFIGURATION is required}"
: "${RETB_SCALE_REFIT_OUTPUT:?RETB_SCALE_REFIT_OUTPUT is required}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cd "${PROJECT_DIR}"
arguments=(--campaign-root "${CAMPAIGN_ROOT}" --locked-scale-shortlist "${RETB_LOCKED_SCALE_SHORTLIST}" --configuration "${RETB_SCALE_REFIT_CONFIGURATION}" --output "${RETB_SCALE_REFIT_OUTPUT}")
if [[ "${RETB_DRY_RUN:-0}" == "1" ]]; then arguments+=(--dry-run); fi
python scripts/register_retb_scale_refits.py "${arguments[@]}"
