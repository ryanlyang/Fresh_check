#!/usr/bin/env bash
#SBATCH --job-name=retb_confirm
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail
IFS=$'\n\t'
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
: "${CONDA_ENV:=atlas_kd_tigris}"
: "${CAMPAIGN_ROOT:?CAMPAIGN_ROOT is required}"
: "${RETB_STAGE_L_GRAPH_REGISTRY:?RETB_STAGE_L_GRAPH_REGISTRY is required}"
: "${RETB_VAL_DESIGN_LABEL_MANIFEST_SHA256:?RETB_VAL_DESIGN_LABEL_MANIFEST_SHA256 is required}"
: "${RETB_CONFIRMATION_SUMMARY_OUTPUT:?RETB_CONFIRMATION_SUMMARY_OUTPUT is required}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cd "${PROJECT_DIR}"
arguments=(
  --campaign-root "${CAMPAIGN_ROOT}"
  --graph-registry "${RETB_STAGE_L_GRAPH_REGISTRY}"
  --val-design-label-manifest-sha256 "${RETB_VAL_DESIGN_LABEL_MANIFEST_SHA256}"
  --output "${RETB_CONFIRMATION_SUMMARY_OUTPUT}"
)
if [[ -n "${RETB_SEED_CONFIRMATION_PATHS:-}" ]]; then
  IFS=':' read -r -a confirmation_paths <<< "${RETB_SEED_CONFIRMATION_PATHS}"
  for path in "${confirmation_paths[@]}"; do
    arguments+=(--seed-confirmation "${path}")
  done
fi
if [[ "${RETB_DRY_RUN:-0}" == "1" ]]; then arguments+=(--dry-run); fi
python scripts/aggregate_retb_confirmation.py "${arguments[@]}"
