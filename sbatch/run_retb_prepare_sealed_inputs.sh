#!/usr/bin/env bash
#SBATCH --job-name=retb_sealed_inputs
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail
IFS=$'\n\t'
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
: "${CONDA_ENV:=atlas_kd_tigris}"
: "${CAMPAIGN_ROOT:?CAMPAIGN_ROOT is required}"
: "${RETB_SEALED_SPLIT:?RETB_SEALED_SPLIT is required}"
: "${RETB_IDENTITY_MANIFEST_SHA256:?RETB_IDENTITY_MANIFEST_SHA256 is required}"
: "${RETB_RAW_INPUT_MANIFEST_SHA256:?RETB_RAW_INPUT_MANIFEST_SHA256 is required}"
: "${RETB_DEGRADED_HLT_INPUT_MANIFEST_SHA256:?RETB_DEGRADED_HLT_INPUT_MANIFEST_SHA256 is required}"
: "${RETB_RELATION_SIDECAR_MANIFEST_SHA256:?RETB_RELATION_SIDECAR_MANIFEST_SHA256 is required}"
: "${RETB_REGION_SIDECAR_MANIFEST_SHA256:?RETB_REGION_SIDECAR_MANIFEST_SHA256 is required}"
: "${RETB_SEALED_INPUT_OUTPUT:?RETB_SEALED_INPUT_OUTPUT is required}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cd "${PROJECT_DIR}"

python scripts/prepare_retb_sealed_inputs.py \
  --campaign-root "${CAMPAIGN_ROOT}" \
  --split "${RETB_SEALED_SPLIT}" \
  --identity-manifest-sha256 "${RETB_IDENTITY_MANIFEST_SHA256}" \
  --raw-input-manifest-sha256 "${RETB_RAW_INPUT_MANIFEST_SHA256}" \
  --degraded-hlt-input-manifest-sha256 "${RETB_DEGRADED_HLT_INPUT_MANIFEST_SHA256}" \
  --relation-sidecar-manifest-sha256 "${RETB_RELATION_SIDECAR_MANIFEST_SHA256}" \
  --region-sidecar-manifest-sha256 "${RETB_REGION_SIDECAR_MANIFEST_SHA256}" \
  --output "${RETB_SEALED_INPUT_OUTPUT}"
