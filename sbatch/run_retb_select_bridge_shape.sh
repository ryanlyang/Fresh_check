#!/usr/bin/env bash
#SBATCH --job-name=retb_bridge_shape
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail
IFS=$'\n\t'
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
: "${CONDA_ENV:=atlas_kd_tigris}"
: "${CAMPAIGN_ROOT:?CAMPAIGN_ROOT is required}"
: "${RETB_BRIDGE_SHAPE_CONFIGURATION:?RETB_BRIDGE_SHAPE_CONFIGURATION is required}"
: "${RETB_BRIDGE_SHAPE_OUTPUT:?RETB_BRIDGE_SHAPE_OUTPUT is required}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cd "${PROJECT_DIR}"
arguments=(
  --campaign-root "${CAMPAIGN_ROOT}"
  --configuration "${RETB_BRIDGE_SHAPE_CONFIGURATION}"
  --output "${RETB_BRIDGE_SHAPE_OUTPUT}"
)
if [[ "${RETB_DRY_RUN:-0}" == "1" ]]; then arguments+=(--dry-run); fi
python scripts/select_retb_bridge_shape.py "${arguments[@]}"
