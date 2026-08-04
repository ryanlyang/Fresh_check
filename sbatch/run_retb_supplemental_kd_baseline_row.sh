#!/usr/bin/env bash
#SBATCH --time=2-00:00:00
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/home/ryreu/atlas/Fresh_check}"
source "${PROJECT_DIR}/sbatch/retb_common.sh"
retb_activate
: "${RETB_SUPP_KD_ROOT:?RETB_SUPP_KD_ROOT is required}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
architectures=(O_BASE O_BASE O_BASE O_FULLREL O_FULLREL O_FULLREL)
seeds=(101 202 303 101 202 303)
python scripts/train_retb_supplemental_kd_baseline.py \
  --plan "${RETB_SUPP_KD_ROOT}/registry/kd_baseline_plan.json" \
  --architecture "${architectures[SLURM_ARRAY_TASK_ID]}" \
  --seed "${seeds[SLURM_ARRAY_TASK_ID]}" \
  --output-root "${RETB_SUPP_KD_ROOT}/runs" \
  --device auto
