#!/usr/bin/env bash
# Submit one high-data campaign graph.
set -euo pipefail
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONSTRAINED_C2F_RUNTIME_PROFILE:=accelerated_approved_v1}"
export CONSTRAINED_C2F_RUNTIME_PROFILE
CONSTRAINED_C2F_CAMPAIGN_MODE=highdata bash "${PROJECT_DIR}/sbatch/submit_constrained_coarse_to_fine_experiment.sh"
