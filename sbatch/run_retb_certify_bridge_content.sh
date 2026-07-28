#!/usr/bin/env bash
#SBATCH --job-name=retb_bridge_cert
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail
IFS=$'\n\t'
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
: "${CONDA_ENV:=atlas_kd_tigris}"
: "${CAMPAIGN_ROOT:?CAMPAIGN_ROOT is required}"
: "${RETB_CANDIDATE_REGISTRATION:?RETB_CANDIDATE_REGISTRATION is required}"
: "${RETB_T0_REGISTRATION:?RETB_T0_REGISTRATION is required}"
: "${RETB_T0_NORMALIZER:?RETB_T0_NORMALIZER is required}"
: "${RETB_IDENTITY_MANIFEST:?RETB_IDENTITY_MANIFEST is required}"
: "${RETB_CERTIFICATION_ARRAYS:?RETB_CERTIFICATION_ARRAYS is required}"
: "${RETB_CERTIFICATION_OUTPUT:?RETB_CERTIFICATION_OUTPUT is required}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cd "${PROJECT_DIR}"

arguments=(
  --campaign-root "${CAMPAIGN_ROOT}"
  --candidate-registration "${RETB_CANDIDATE_REGISTRATION}"
  --t0-registration "${RETB_T0_REGISTRATION}"
  --t0-normalizer "${RETB_T0_NORMALIZER}"
  --identity-manifest "${RETB_IDENTITY_MANIFEST}"
  --arrays "${RETB_CERTIFICATION_ARRAYS}"
  --output "${RETB_CERTIFICATION_OUTPUT}"
)
if [[ -n "${RETB_BRIDGE_NORMALIZER:-}" ]]; then
  arguments+=(--bridge-normalizer "${RETB_BRIDGE_NORMALIZER}")
fi
python scripts/certify_retb_bridge_content.py "${arguments[@]}"
