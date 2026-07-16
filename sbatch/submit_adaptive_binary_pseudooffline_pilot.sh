#!/usr/bin/env bash
# Queue the complete model-validation pilot from an empty campaign root.
set -euo pipefail
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
export ABPH_CAMPAIGN_MODE=pilot ABPH_STAGE_MODE=full CONFIRM_FINAL_TEST=0
exec bash "${PROJECT_DIR}/sbatch/submit_adaptive_binary_pseudooffline.sh" "$@"
