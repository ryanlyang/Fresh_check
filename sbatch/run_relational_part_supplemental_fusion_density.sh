#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

: "${CAMPAIGN_ROOT:?CAMPAIGN_ROOT is required}"
: "${PINNED_SOURCE_ROOT:?PINNED_SOURCE_ROOT is required}"
: "${PROJECT_DIR:=${PINNED_SOURCE_ROOT}}"
: "${FUSION_SCRIPT:=${CAMPAIGN_ROOT}/evaluate_relational_part_supplemental_fusion.py}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=relational_part_common.sh
source "${SCRIPT_DIR}/relational_part_common.sh"
rpt_setup

seeds=(101 202 303)
task_index="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
if (( task_index < 0 || task_index >= ${#seeds[@]} )); then
  echo "Fusion array index is outside 0..2: ${task_index}" >&2
  exit 2
fi
seed="${seeds[task_index]}"

python "${FUSION_SCRIPT}" \
  --source-root "${PINNED_SOURCE_ROOT}" \
  --campaign-root "${CAMPAIGN_ROOT}" \
  --comparison-id track_charge_density \
  --seed "${seed}" \
  --device auto \
  --allow-source-status-drift
