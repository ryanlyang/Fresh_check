#!/usr/bin/env bash
set -euo pipefail
: "${PROJECT_DIR:?PROJECT_DIR must name the frozen RETB source worktree}"
source "${PROJECT_DIR}/sbatch/retb_common.sh"
retb_activate
recovery_args=()
if [[ "${RETB_ORDINARY_SPECIALIST_KD_FINALIZER_RECOVERY:-0}" == "1" ]]; then
  recovery_args+=(--corrective-source-recovery)
fi
python scripts/finalize_retb_ordinary_specialist_kd.py \
  --plan "${RETB_ORDINARY_SPECIALIST_KD_ROOT}/registry/ordinary_specialist_kd_plan.json" \
  --student-root "${RETB_ORDINARY_SPECIALIST_KD_ROOT}/runs/students" \
  --output-root "${RETB_ORDINARY_SPECIALIST_KD_ROOT}/reports" \
  "${recovery_args[@]}"
