#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/retb_common.sh"
retb_activate
experts=(PT TRACK REGION)
expert="${experts[${SLURM_ARRAY_TASK_ID:?}]}"
python scripts/train_retb_specialist_teacher.py \
  --plan "${RETB_SPECIALIST_KD_ROOT}/registry/specialist_kd_plan.json" \
  --expert "${expert}" \
  --output-root "${RETB_SPECIALIST_KD_ROOT}/runs/teachers" \
  --device cuda
