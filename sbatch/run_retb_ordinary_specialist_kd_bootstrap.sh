#!/usr/bin/env bash
set -euo pipefail
: "${PROJECT_DIR:?PROJECT_DIR must name the frozen RETB source worktree}"
source "${PROJECT_DIR}/sbatch/retb_common.sh"
retb_activate
python scripts/bootstrap_retb_ordinary_specialist_kd.py \
  --compact-specialist-root "${RETB_COMPACT_SPECIALIST_KD_ROOT}" \
  --supplemental-id "${RETB_ORDINARY_SPECIALIST_KD_ID}" \
  --output "${RETB_ORDINARY_SPECIALIST_KD_ROOT}/registry/ordinary_specialist_kd_plan.json"
