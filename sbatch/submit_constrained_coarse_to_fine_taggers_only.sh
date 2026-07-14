#!/usr/bin/env bash
# Queue implemented staged taggers and predictions from existing reconstructors.
set -euo pipefail
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
CONSTRAINED_C2F_STAGE_MODE=taggers_only bash "${PROJECT_DIR}/sbatch/submit_constrained_coarse_to_fine_experiment.sh"
