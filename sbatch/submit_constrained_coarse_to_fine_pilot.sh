#!/usr/bin/env bash
# Submit one pilot campaign graph.
set -euo pipefail
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
CONSTRAINED_C2F_CAMPAIGN_MODE=pilot bash "${PROJECT_DIR}/sbatch/submit_constrained_coarse_to_fine_experiment.sh"
