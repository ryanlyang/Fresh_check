#!/usr/bin/env bash
#SBATCH --job-name=retb_target_spec
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
: "${RETB_TARGET_SPEC_INPUTS:?RETB_TARGET_SPEC_INPUTS is required}"
: "${RETB_TARGET_SPLIT:?RETB_TARGET_SPLIT is required}"
: "${RETB_PIPELINE_SEED:?RETB_PIPELINE_SEED is required}"
: "${RETB_TARGET_SHAPE_ID:?RETB_TARGET_SHAPE_ID is required}"
: "${RETB_COORDINATE_CONTRACT_SHA256:?RETB_COORDINATE_CONTRACT_SHA256 is required}"
: "${RETB_TARGET_LINEAGE_OUTPUT:?RETB_TARGET_LINEAGE_OUTPUT is required}"
: "${RETB_TARGET_SPECIFICATION_OUTPUT:?RETB_TARGET_SPECIFICATION_OUTPUT is required}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cd "${PROJECT_DIR}"

python scripts/materialize_retb_target_cache_spec.py \
  --campaign-root "${CAMPAIGN_ROOT}" \
  --inputs "${RETB_TARGET_SPEC_INPUTS}" \
  --split "${RETB_TARGET_SPLIT}" \
  --pipeline-seed "${RETB_PIPELINE_SEED}" \
  --shape-id "${RETB_TARGET_SHAPE_ID}" \
  --coordinate-contract-sha256 "${RETB_COORDINATE_CONTRACT_SHA256}" \
  --output-lineage "${RETB_TARGET_LINEAGE_OUTPUT}" \
  --output-specification "${RETB_TARGET_SPECIFICATION_OUTPUT}"
