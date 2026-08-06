#!/usr/bin/env bash
set -euo pipefail
: "${PROJECT_DIR:?PROJECT_DIR must name the frozen RETB source worktree}"
source "${PROJECT_DIR}/sbatch/retb_common.sh"
retb_activate
conditions=(MATCHED_KD MATCHED_KD MATCHED_KD MATCHED_KD HYBRID_KD HYBRID_KD HYBRID_KD HYBRID_KD)
experts=(BASE4 PT TRACK REGION BASE4 PT TRACK REGION)
index="${SLURM_ARRAY_TASK_ID:?}"
python scripts/train_retb_ordinary_specialist_kd_student.py \
  --plan "${RETB_ORDINARY_SPECIALIST_KD_ROOT}/registry/ordinary_specialist_kd_plan.json" \
  --condition "${conditions[${index}]}" \
  --expert "${experts[${index}]}" \
  --output-root "${RETB_ORDINARY_SPECIALIST_KD_ROOT}/runs/students" \
  --device cuda
