#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/retb_common.sh"
retb_activate
python scripts/finalize_retb_specialist_kd.py \
  --plan "${RETB_SPECIALIST_KD_ROOT}/registry/specialist_kd_plan.json" \
  --student-root "${RETB_SPECIALIST_KD_ROOT}/runs/students" \
  --output-root "${RETB_SPECIALIST_KD_ROOT}/reports"
