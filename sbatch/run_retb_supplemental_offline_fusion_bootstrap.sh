#!/usr/bin/env bash
#SBATCH --time=00:30:00
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/home/ryreu/atlas/Fresh_check}"
source "${PROJECT_DIR}/sbatch/retb_common.sh"
retb_activate
: "${RETB_SUPPLEMENTAL_ROOT:?RETB_SUPPLEMENTAL_ROOT is required}"
: "${RETB_PARENT_CAMPAIGN_ROOT:?RETB_PARENT_CAMPAIGN_ROOT is required}"
: "${RETB_SUPPLEMENTAL_ID:?RETB_SUPPLEMENTAL_ID is required}"
python scripts/bootstrap_retb_supplemental_offline_fusion.py \
  --parent-campaign-root "${RETB_PARENT_CAMPAIGN_ROOT}" \
  --supplemental-id "${RETB_SUPPLEMENTAL_ID}" \
  --output "${RETB_SUPPLEMENTAL_ROOT}/registry/supplemental_plan.json"
