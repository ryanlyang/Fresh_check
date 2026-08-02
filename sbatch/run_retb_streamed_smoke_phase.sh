#!/usr/bin/env bash
#SBATCH --job-name=retb_streamed_smoke
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --time=02:00:00
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/home/ryreu/atlas/Fresh_check}"
source "${PROJECT_DIR}/sbatch/retb_common.sh"
retb_setup
: "${RETB_SMOKE_PHASE_ID:?RETB_SMOKE_PHASE_ID is required}"
python scripts/run_retb_streamed_smoke_phase.py \
  --campaign-root "${CAMPAIGN_ROOT}" \
  --phase-id "${RETB_SMOKE_PHASE_ID}"
