#!/usr/bin/env bash
# Queue the multi-depth D8 tagger from existing C5-B1/B2/B3 checkpoints.
set -euo pipefail
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
CONSTRAINED_C2F_STAGE_MODE=d8_only bash "${PROJECT_DIR}/sbatch/submit_constrained_coarse_to_fine_experiment.sh"
