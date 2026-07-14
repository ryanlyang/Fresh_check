#!/usr/bin/env bash
# Queue D5 and its B1/B2/B3 depth-matched comparisons from existing checkpoints.
set -euo pipefail
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
CONSTRAINED_C2F_STAGE_MODE=depth_d5 bash "${PROJECT_DIR}/sbatch/submit_constrained_coarse_to_fine_experiment.sh"
