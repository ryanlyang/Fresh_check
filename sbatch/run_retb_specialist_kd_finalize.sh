#!/usr/bin/env bash
set -euo pipefail
: "${PROJECT_DIR:?PROJECT_DIR must name the frozen RETB source worktree}"
source "${PROJECT_DIR}/sbatch/retb_common.sh"
retb_activate
python scripts/finalize_retb_specialist_kd.py \
  --plan "${RETB_SPECIALIST_KD_ROOT}/registry/specialist_kd_plan.json" \
  --student-root "${RETB_SPECIALIST_KD_ROOT}/runs/students" \
  --output-root "${RETB_SPECIALIST_KD_ROOT}/reports"
