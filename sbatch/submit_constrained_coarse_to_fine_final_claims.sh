#!/usr/bin/env bash
# Append one immutable final-test claim to a completed high-data campaign.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONSTRAINED_C2F_ROOT:?Set CONSTRAINED_C2F_ROOT to the selected completed high-data campaign}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=constrained_coarse_to_fine_claim_contract.sh
source "${SCRIPT_DIR}/constrained_coarse_to_fine_claim_contract.sh"

export CONSTRAINED_C2F_STAGE_MODE=final_claims
export CONSTRAINED_C2F_CAMPAIGN_MODE=highdata
export CONSTRAINED_C2F_RUNTIME_PROFILE=accelerated_approved_v1
export CONSTRAINED_C2F_RECON_RUN_IDS="${C2F_FROZEN_RECON_RUN_IDS}"
export CONSTRAINED_C2F_TAGGER_RUN_IDS="${C2F_FROZEN_TAGGER_RUN_IDS}"
export CONSTRAINED_C2F_PREDICT_RUN_IDS="${C2F_FROZEN_TAGGER_RUN_IDS}"
export CONSTRAINED_C2F_REPORT_RECON_RUN_IDS="${C2F_FROZEN_RECON_RUN_IDS}"
export CONSTRAINED_C2F_REPORT_TAGGER_RUN_IDS="${C2F_FROZEN_TAGGER_RUN_IDS}"
export CONSTRAINED_C2F_FUSION_GROUPS="${C2F_FROZEN_FUSION_GROUPS}"
export CONSTRAINED_C2F_REQUIRED_FUSION_GROUPS="${C2F_FROZEN_REQUIRED_FUSION_GROUPS}"
export CONSTRAINED_C2F_PREDICT_SPLITS=final_test
export CONSTRAINED_C2F_OFFLINE_SPLITS=final_test
export CONFIRM_FINAL_TEST=1
export SKIP_EXISTING=0
export OVERWRITE=0

exec bash "${PROJECT_DIR}/sbatch/submit_constrained_coarse_to_fine_experiment.sh"
