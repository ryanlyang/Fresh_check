#!/usr/bin/env bash
#SBATCH --job-name=retb_internal_phase
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --time=3-00:00:00
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/home/ryreu/atlas/Fresh_check}"
source "${PROJECT_DIR}/sbatch/retb_common.sh"
retb_setup
: "${RETB_INTERNAL_PHASE_PLAN:?RETB_INTERNAL_PHASE_PLAN is required}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
python scripts/run_retb_internal_phase_task.py \
  --campaign-root "${CAMPAIGN_ROOT}" \
  --phase-plan "${RETB_INTERNAL_PHASE_PLAN}" \
  --task-index "${SLURM_ARRAY_TASK_ID}"
