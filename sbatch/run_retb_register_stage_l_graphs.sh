#!/usr/bin/env bash
#SBATCH --job-name=retb_l_graphs
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
: "${RETB_BRIDGE_SHAPE_SELECTION:?RETB_BRIDGE_SHAPE_SELECTION is required}"
: "${RETB_STAGE_L_DEFINITIONS:?RETB_STAGE_L_DEFINITIONS is required}"
: "${RETB_STAGE_L_GRAPH_REGISTRY:?RETB_STAGE_L_GRAPH_REGISTRY is required}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cd "${PROJECT_DIR}"
arguments=(
  --campaign-root "${CAMPAIGN_ROOT}"
  --bridge-shape-selection "${RETB_BRIDGE_SHAPE_SELECTION}"
  --definitions "${RETB_STAGE_L_DEFINITIONS}"
  --output "${RETB_STAGE_L_GRAPH_REGISTRY}"
)
if [[ "${RETB_DRY_RUN:-0}" == "1" ]]; then arguments+=(--dry-run); fi
python scripts/register_retb_stage_l_graphs.py "${arguments[@]}"
