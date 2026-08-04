#!/usr/bin/env bash
#SBATCH --time=00:30:00
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/home/ryreu/atlas/Fresh_check}"
source "${PROJECT_DIR}/sbatch/retb_common.sh"
retb_activate
: "${RETB_SUPP_KD_ROOT:?RETB_SUPP_KD_ROOT is required}"
: "${RETB_SUPP_KD_ID:?RETB_SUPP_KD_ID is required}"
: "${RETB_PARENT_CAMPAIGN_ROOT:?RETB_PARENT_CAMPAIGN_ROOT is required}"
python scripts/bootstrap_retb_supplemental_kd_baselines.py \
  --parent-campaign-root "${RETB_PARENT_CAMPAIGN_ROOT}" \
  --supplemental-id "${RETB_SUPP_KD_ID}" \
  --output "${RETB_SUPP_KD_ROOT}/registry/kd_baseline_plan.json"
