#!/usr/bin/env bash
# Queue the B/C sweep from an existing validated target cache.
set -euo pipefail
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
CONSTRAINED_C2F_STAGE_MODE=reconstructors_only bash "${PROJECT_DIR}/sbatch/submit_constrained_coarse_to_fine_experiment.sh"
