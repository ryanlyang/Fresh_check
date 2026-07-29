#!/usr/bin/env bash
#SBATCH --job-name=retb_shortlist
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
: "${RETB_STAGE_L_GRAPH_REGISTRY:?RETB_STAGE_L_GRAPH_REGISTRY is required}"
: "${RETB_CONFIRMATION_SUMMARY:?RETB_CONFIRMATION_SUMMARY is required}"
: "${RETB_BRIDGE_SHAPE_SELECTION:?RETB_BRIDGE_SHAPE_SELECTION is required}"
: "${RETB_SCALE_SHORTLIST_PARENTS:?RETB_SCALE_SHORTLIST_PARENTS is required}"
: "${RETB_SCALE_SHORTLIST_OUTPUT:?RETB_SCALE_SHORTLIST_OUTPUT is required}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cd "${PROJECT_DIR}"
arguments=(
  --campaign-root "${CAMPAIGN_ROOT}"
  --graph-registry "${RETB_STAGE_L_GRAPH_REGISTRY}"
  --confirmation-summary "${RETB_CONFIRMATION_SUMMARY}"
  --bridge-shape-selection "${RETB_BRIDGE_SHAPE_SELECTION}"
  --parent-hashes "${RETB_SCALE_SHORTLIST_PARENTS}"
  --output "${RETB_SCALE_SHORTLIST_OUTPUT}"
)
if [[ "${RETB_DRY_RUN:-0}" == "1" ]]; then arguments+=(--dry-run); fi
python scripts/select_retb_scale_shortlist.py "${arguments[@]}"
