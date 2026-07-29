#!/usr/bin/env bash
#SBATCH --job-name=retb_finalists
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
set -euo pipefail
IFS=$'\n\t'
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
: "${CONDA_ENV:=atlas_kd_tigris}"
: "${CAMPAIGN_ROOT:?CAMPAIGN_ROOT is required}"
: "${RETB_LOCKED_SCALE_SHORTLIST:?RETB_LOCKED_SCALE_SHORTLIST is required}"
: "${RETB_SCALE_COMPLETION:?RETB_SCALE_COMPLETION is required}"
: "${RETB_FINAL_SELECT_LABEL_MANIFEST:?RETB_FINAL_SELECT_LABEL_MANIFEST is required}"
: "${RETB_FINALIST_LINEAGE_HASHES:?RETB_FINALIST_LINEAGE_HASHES is required}"
: "${RETB_FINALIST_SHAPE_ASSIGNMENTS:?RETB_FINALIST_SHAPE_ASSIGNMENTS is required}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cd "${PROJECT_DIR}"
arguments=(--campaign-root "${CAMPAIGN_ROOT}" --locked-scale-shortlist "${RETB_LOCKED_SCALE_SHORTLIST}" --scale-completion "${RETB_SCALE_COMPLETION}" --final-select-label-manifest "${RETB_FINAL_SELECT_LABEL_MANIFEST}" --lineage-hashes "${RETB_FINALIST_LINEAGE_HASHES}" --shape-assignments "${RETB_FINALIST_SHAPE_ASSIGNMENTS}")
IFS=':' read -r -a prediction_paths <<< "${RETB_STACK_PREDICTION_MANIFESTS:?RETB_STACK_PREDICTION_MANIFESTS is required}"
for path in "${prediction_paths[@]}"; do arguments+=(--prediction-manifest "${path}"); done
if [[ "${RETB_DRY_RUN:-0}" == "1" ]]; then arguments+=(--dry-run); fi
python scripts/select_retb_scale_finalists.py "${arguments[@]}"
