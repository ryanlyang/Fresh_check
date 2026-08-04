#!/usr/bin/env bash
#SBATCH --time=01:00:00
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/home/ryreu/atlas/Fresh_check}"
source "${PROJECT_DIR}/sbatch/retb_common.sh"
retb_activate
: "${RETB_SUPPLEMENTAL_ROOT:?RETB_SUPPLEMENTAL_ROOT is required}"
python scripts/finalize_retb_supplemental_offline_fusion.py \
  --plan "${RETB_SUPPLEMENTAL_ROOT}/registry/supplemental_plan.json" \
  --supplemental-root "${RETB_SUPPLEMENTAL_ROOT}"
