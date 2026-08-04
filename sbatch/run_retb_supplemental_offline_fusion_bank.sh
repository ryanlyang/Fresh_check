#!/usr/bin/env bash
#SBATCH --time=2-00:00:00
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/home/ryreu/atlas/Fresh_check}"
source "${PROJECT_DIR}/sbatch/retb_common.sh"
retb_activate
: "${RETB_SUPPLEMENTAL_ROOT:?RETB_SUPPLEMENTAL_ROOT is required}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
banks=(CE4 CE7 KD3 KD4 MIXED7)
bank="${banks[SLURM_ARRAY_TASK_ID]}"
python scripts/run_retb_supplemental_offline_fusion_bank.py \
  --plan "${RETB_SUPPLEMENTAL_ROOT}/registry/supplemental_plan.json" \
  --bank-id "${bank}" \
  --output-dir "${RETB_SUPPLEMENTAL_ROOT}/runs/fusion_banks/${bank}" \
  --device auto
