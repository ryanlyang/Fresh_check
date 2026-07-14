#!/usr/bin/env bash
# Build split/HLT/offline/target caches without queueing models.
set -euo pipefail
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
CONSTRAINED_C2F_STAGE_MODE=targets_only bash "${PROJECT_DIR}/sbatch/submit_constrained_coarse_to_fine_experiment.sh"
