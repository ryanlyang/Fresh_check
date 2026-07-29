#!/usr/bin/env bash
#SBATCH --job-name=retb_bundle_select
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00

set -euo pipefail
IFS=$'\n\t'
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
: "${CONDA_ENV:=atlas_kd_tigris}"
: "${CAMPAIGN_ROOT:?CAMPAIGN_ROOT is required}"
: "${RETB_BUNDLE_SELECTOR_CONFIGURATION:?RETB_BUNDLE_SELECTOR_CONFIGURATION is required}"
: "${RETB_BUNDLE_SELECTOR_OUTPUT:?RETB_BUNDLE_SELECTOR_OUTPUT is required}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cd "${PROJECT_DIR}"

arguments=(
  --campaign-root "${CAMPAIGN_ROOT}"
  --configuration "${RETB_BUNDLE_SELECTOR_CONFIGURATION}"
  --output-dir "${RETB_BUNDLE_SELECTOR_OUTPUT}"
)
if [[ "${RETB_DRY_RUN:-0}" == "1" ]]; then
  arguments+=(--dry-run)
fi
python scripts/select_retb_joint_predictor_bundle.py "${arguments[@]}"
