#!/usr/bin/env bash
#SBATCH --time=00:30:00
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/home/ryreu/atlas/Fresh_check}"
source "${PROJECT_DIR}/sbatch/retb_common.sh"
retb_activate
: "${RETB_SUPP_KD_ROOT:?RETB_SUPP_KD_ROOT is required}"
python scripts/finalize_retb_supplemental_kd_baselines.py \
  --plan "${RETB_SUPP_KD_ROOT}/registry/kd_baseline_plan.json" \
  --supplemental-root "${RETB_SUPP_KD_ROOT}"
