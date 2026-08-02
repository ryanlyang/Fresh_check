#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

: "${CAMPAIGN_ROOT:?CAMPAIGN_ROOT is required}"
: "${PINNED_SOURCE_ROOT:?PINNED_SOURCE_ROOT is required}"
: "${FUSION_AGGREGATOR:=${CAMPAIGN_ROOT}/aggregate_relational_part_supplemental_fusion.py}"
: "${CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
: "${CONDA_ENV:=atlas_kd_tigris}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

python "${FUSION_AGGREGATOR}" \
  --source-root "${PINNED_SOURCE_ROOT}" \
  --campaign-root "${CAMPAIGN_ROOT}"
