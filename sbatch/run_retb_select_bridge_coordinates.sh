#!/usr/bin/env bash
#SBATCH --job-name=retb_bridge_select
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail
IFS=$'\n\t'
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
: "${CONDA_ENV:=atlas_kd_tigris}"
: "${CAMPAIGN_ROOT:?CAMPAIGN_ROOT is required}"
: "${RETB_BRIDGE_SCORE_TABLE:?RETB_BRIDGE_SCORE_TABLE is required}"
: "${RETB_BRIDGE_ELIGIBILITY_FILE:?newline-delimited EXPERT:MODE=JSON file is required}"
: "${RETB_BRIDGE_SELECTION_OUTPUT:?RETB_BRIDGE_SELECTION_OUTPUT is required}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cd "${PROJECT_DIR}"

arguments=(
  --campaign-root "${CAMPAIGN_ROOT}"
  --score-table "${RETB_BRIDGE_SCORE_TABLE}"
  --output "${RETB_BRIDGE_SELECTION_OUTPUT}"
)
while IFS= read -r row; do
  [[ -z "${row}" ]] || arguments+=(--eligibility "${row}")
done < "${RETB_BRIDGE_ELIGIBILITY_FILE}"
python scripts/select_retb_bridge_coordinates.py "${arguments[@]}"
