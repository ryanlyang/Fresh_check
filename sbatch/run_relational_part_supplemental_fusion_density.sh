#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

: "${CAMPAIGN_ROOT:?CAMPAIGN_ROOT is required}"
: "${PINNED_SOURCE_ROOT:?PINNED_SOURCE_ROOT is required}"
: "${FUSION_SCRIPT:=${CAMPAIGN_ROOT}/evaluate_relational_part_supplemental_fusion.py}"
: "${CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
: "${CONDA_ENV:=atlas_kd_tigris}"

seeds=(101 202 303)
task_index="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
if (( task_index < 0 || task_index >= ${#seeds[@]} )); then
  echo "Fusion array index is outside 0..2: ${task_index}" >&2
  exit 2
fi
seed="${seeds[task_index]}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

python "${FUSION_SCRIPT}" \
  --source-root "${PINNED_SOURCE_ROOT}" \
  --campaign-root "${CAMPAIGN_ROOT}" \
  --comparison-id track_charge_density \
  --seed "${seed}" \
  --device auto \
  --allow-source-status-drift
