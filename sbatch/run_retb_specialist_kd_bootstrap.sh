#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/retb_common.sh"
retb_activate
python scripts/bootstrap_retb_specialist_kd.py \
  --parent-root "${RETB_PARENT_CAMPAIGN_ROOT}" \
  --common-fusion-root "${RETB_COMMON_FUSION_ROOT}" \
  --supplemental-id "${RETB_SPECIALIST_KD_ID}" \
  --output "${RETB_SPECIALIST_KD_ROOT}/registry/specialist_kd_plan.json"
