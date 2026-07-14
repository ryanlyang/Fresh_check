#!/usr/bin/env bash
# Submit pilot and high-data graphs independently so both can run concurrently.
set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
fresh_prepare_submitter

: "${CONSTRAINED_C2F_PAIR_STAMP:=$(date +%Y%m%d_%H%M%S)}"
: "${CONSTRAINED_C2F_PILOT_ROOT:=${OUTPUT_ROOT}/constrained_coarse_to_fine_pseudooffline_hltv2_s2p5_pilot_${CONSTRAINED_C2F_PAIR_STAMP}}"
: "${CONSTRAINED_C2F_HIGHDATA_ROOT:=${OUTPUT_ROOT}/constrained_coarse_to_fine_pseudooffline_hltv2_s2p5_highdata_${CONSTRAINED_C2F_PAIR_STAMP}}"
: "${CONSTRAINED_C2F_PILOT_HLT_WARM_START_CHECKPOINT:=}"
: "${CONSTRAINED_C2F_HIGHDATA_HLT_WARM_START_CHECKPOINT:=}"

if [[ "${CONSTRAINED_C2F_STAGE_MODE:-full}" != "targets_only" && "${CONSTRAINED_C2F_STAGE_MODE:-full}" != "reconstructors_only" ]]; then
  [[ -n "${CONSTRAINED_C2F_PILOT_HLT_WARM_START_CHECKPOINT}" ]] || {
    echo "Set CONSTRAINED_C2F_PILOT_HLT_WARM_START_CHECKPOINT to the pilot split's A0 checkpoint." >&2
    exit 2
  }
  [[ -n "${CONSTRAINED_C2F_HIGHDATA_HLT_WARM_START_CHECKPOINT}" ]] || {
    echo "Set CONSTRAINED_C2F_HIGHDATA_HLT_WARM_START_CHECKPOINT to the high-data split's A0 checkpoint." >&2
    exit 2
  }
fi

CONSTRAINED_C2F_CAMPAIGN_MODE=pilot \
CONSTRAINED_C2F_ROOT="${CONSTRAINED_C2F_PILOT_ROOT}" \
CONSTRAINED_C2F_HLT_WARM_START_CHECKPOINT="${CONSTRAINED_C2F_PILOT_HLT_WARM_START_CHECKPOINT}" \
  bash "${SCRIPT_DIR}/submit_constrained_coarse_to_fine_experiment.sh"

CONSTRAINED_C2F_CAMPAIGN_MODE=highdata \
CONSTRAINED_C2F_ROOT="${CONSTRAINED_C2F_HIGHDATA_ROOT}" \
CONSTRAINED_C2F_HLT_WARM_START_CHECKPOINT="${CONSTRAINED_C2F_HIGHDATA_HLT_WARM_START_CHECKPOINT}" \
  bash "${SCRIPT_DIR}/submit_constrained_coarse_to_fine_experiment.sh"

cat <<SUMMARY
constrained_c2f_pair_submission_complete:
  pilot_root: ${CONSTRAINED_C2F_PILOT_ROOT}
  highdata_root: ${CONSTRAINED_C2F_HIGHDATA_ROOT}
SUMMARY
