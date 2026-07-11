#!/usr/bin/env bash
# Submit both pilot and high-data Canonical Multi-Scale Jet State campaigns.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

fresh_prepare_submitter

: "${CANONICAL_STATE_PAIR_STAMP:=$(date +%Y%m%d_%H%M%S)}"
: "${CANONICAL_STATE_PILOT_ROOT:=${OUTPUT_ROOT}/canonical_multi_scale_jet_state_hltv2_s2p5_pilot_${CANONICAL_STATE_PAIR_STAMP}}"
: "${CANONICAL_STATE_HIGHDATA_ROOT:=${OUTPUT_ROOT}/canonical_multi_scale_jet_state_hltv2_s2p5_highdata_${CANONICAL_STATE_PAIR_STAMP}}"

echo "canonical_state_pilot_and_highdata_submission_start:"
echo "  pilot_root: ${CANONICAL_STATE_PILOT_ROOT}"
echo "  highdata_root: ${CANONICAL_STATE_HIGHDATA_ROOT}"

CANONICAL_STATE_CAMPAIGN_MODE=pilot \
CANONICAL_STATE_ROOT="${CANONICAL_STATE_PILOT_ROOT}" \
  bash "${SCRIPT_DIR}/submit_canonical_state_experiment.sh"

CANONICAL_STATE_CAMPAIGN_MODE=highdata \
CANONICAL_STATE_ROOT="${CANONICAL_STATE_HIGHDATA_ROOT}" \
  bash "${SCRIPT_DIR}/submit_canonical_state_experiment.sh"

cat <<SUMMARY
canonical_state_pilot_and_highdata_submission_complete:
  pilot_root: ${CANONICAL_STATE_PILOT_ROOT}
  highdata_root: ${CANONICAL_STATE_HIGHDATA_ROOT}
SUMMARY
