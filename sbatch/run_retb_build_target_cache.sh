#!/usr/bin/env bash
#SBATCH --job-name=retb_target_cache
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail
IFS=$'\n\t'
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
: "${CONDA_ENV:=atlas_kd_tigris}"
: "${CAMPAIGN_ROOT:?CAMPAIGN_ROOT is required}"
: "${RETB_TARGET_SPECIFICATION:?RETB_TARGET_SPECIFICATION is required}"
: "${RETB_GENERATED_TARGETS:?RETB_GENERATED_TARGETS is required}"
: "${RETB_TARGET_CHECKPOINT_MAP:?RETB_TARGET_CHECKPOINT_MAP is required}"
: "${RETB_TARGET_CACHE_OUTPUT:?RETB_TARGET_CACHE_OUTPUT is required}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cd "${PROJECT_DIR}"

python scripts/build_retb_target_cache.py \
  --campaign-root "${CAMPAIGN_ROOT}" \
  --specification "${RETB_TARGET_SPECIFICATION}" \
  --generated-targets "${RETB_GENERATED_TARGETS}" \
  --checkpoint-map "${RETB_TARGET_CHECKPOINT_MAP}" \
  --output-dir "${RETB_TARGET_CACHE_OUTPUT}" \
  --shard-size "${RETB_TARGET_SHARD_SIZE:-2048}"
