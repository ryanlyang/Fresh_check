#!/usr/bin/env bash
#SBATCH --time=2-00:00:00
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/home/ryreu/atlas/Fresh_check}"
source "${PROJECT_DIR}/sbatch/retb_common.sh"
retb_activate
: "${RETB_SUPPLEMENTAL_ROOT:?RETB_SUPPLEMENTAL_ROOT is required}"
: "${RETB_SUPPLEMENTAL_PLAN_ROLE:?RETB_SUPPLEMENTAL_PLAN_ROLE is required}"
: "${RETB_SUPPLEMENTAL_BANKS:?RETB_SUPPLEMENTAL_BANKS is required}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
IFS=':' read -r -a banks <<< "${RETB_SUPPLEMENTAL_BANKS}"
bank="${banks[SLURM_ARRAY_TASK_ID]}"
python scripts/run_retb_supplemental_offline_fusion_bank.py \
  --plan "${RETB_SUPPLEMENTAL_ROOT}/registry/${RETB_SUPPLEMENTAL_PLAN_ROLE}_plan.json" \
  --bank-id "${bank}" \
  --output-dir "${RETB_SUPPLEMENTAL_ROOT}/runs/fusion_banks/${bank}" \
  --device auto
