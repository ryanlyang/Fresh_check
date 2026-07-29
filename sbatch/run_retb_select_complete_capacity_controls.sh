#!/usr/bin/env bash
#SBATCH --job-name=retb_mono_match
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
: "${RETB_CAPACITY_ARTIFACT:?RETB_CAPACITY_ARTIFACT is required}"
: "${RETB_MONOLITHIC_CANDIDATE_GRID:?RETB_MONOLITHIC_CANDIDATE_GRID is required}"
: "${RETB_CAPACITY_DOMAIN:?RETB_CAPACITY_DOMAIN is required}"
: "${RETB_CAPACITY_SELECTION_OUTPUT:?RETB_CAPACITY_SELECTION_OUTPUT is required}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cd "${PROJECT_DIR}"
python scripts/select_retb_complete_graph_capacity_controls.py \
  --campaign-root "${CAMPAIGN_ROOT}" \
  --capacity-artifact "${RETB_CAPACITY_ARTIFACT}" \
  --candidate-grid "${RETB_MONOLITHIC_CANDIDATE_GRID}" \
  --domain "${RETB_CAPACITY_DOMAIN}" \
  --output "${RETB_CAPACITY_SELECTION_OUTPUT}"
