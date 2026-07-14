#!/usr/bin/env bash
# Rebuild the strict report from existing predictions and fusion output.
set -euo pipefail
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
CONSTRAINED_C2F_STAGE_MODE=report_only bash "${PROJECT_DIR}/sbatch/submit_constrained_coarse_to_fine_experiment.sh"
