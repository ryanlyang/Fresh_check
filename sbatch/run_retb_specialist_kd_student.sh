#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/retb_common.sh"
retb_activate
conditions=(MATCHED_KD MATCHED_KD MATCHED_KD MATCHED_KD HYBRID_KD HYBRID_KD HYBRID_KD HYBRID_KD)
experts=(BASE4 PT TRACK REGION BASE4 PT TRACK REGION)
index="${SLURM_ARRAY_TASK_ID:?}"
python scripts/train_retb_specialist_kd_student.py \
  --plan "${RETB_SPECIALIST_KD_ROOT}/registry/specialist_kd_plan.json" \
  --condition "${conditions[${index}]}" \
  --expert "${experts[${index}]}" \
  --teacher-root "${RETB_SPECIALIST_KD_ROOT}/runs/teachers" \
  --output-root "${RETB_SPECIALIST_KD_ROOT}/runs/students" \
  --device cuda
