#!/usr/bin/env bash
#SBATCH --time=3-00:00:00
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/home/ryreu/atlas/Fresh_check}"
source "${PROJECT_DIR}/sbatch/retb_common.sh"
retb_activate
: "${RETB_SUPPLEMENTAL_ROOT:?RETB_SUPPLEMENTAL_ROOT is required}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
seeds=(101 202 303 404 505 606 707)
seed="${seeds[SLURM_ARRAY_TASK_ID]}"
python scripts/train_retb_supplemental_obase7_member.py \
  --plan "${RETB_SUPPLEMENTAL_ROOT}/registry/supplemental_plan.json" \
  --seed "${seed}" \
  --output-root "${RETB_SUPPLEMENTAL_ROOT}/runs/obase7" \
  --device auto
