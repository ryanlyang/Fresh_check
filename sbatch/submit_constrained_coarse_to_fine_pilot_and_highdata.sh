#!/usr/bin/env bash
# Stage the campaign: pilot first, high-data only after explicit pilot approval.
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
: "${CONSTRAINED_C2F_CAMPAIGN_STAGE:=pilot}"
: "${CONSTRAINED_C2F_APPROVE_HIGHDATA:=0}"

case "${CONSTRAINED_C2F_CAMPAIGN_STAGE}" in
  pilot)
    CONSTRAINED_C2F_CAMPAIGN_MODE=pilot \
    CONSTRAINED_C2F_ROOT="${CONSTRAINED_C2F_PILOT_ROOT}" \
      bash "${SCRIPT_DIR}/submit_constrained_coarse_to_fine_experiment.sh"
    ;;
  highdata)
    fresh_bool_enabled "${CONSTRAINED_C2F_APPROVE_HIGHDATA}" || {
      echo "Set CONSTRAINED_C2F_APPROVE_HIGHDATA=1 only after reviewing the pilot diagnostics." >&2
      exit 2
    }
    pilot_report="${CONSTRAINED_C2F_PILOT_ROOT}/final_report/final_report.json"
    fresh_require_file "${pilot_report}"
    "${PYTHON_BIN}" - "${pilot_report}" <<'PY'
import json
import sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
if not bool(report.get("ok")):
    raise SystemExit("pilot final report is not ok; refusing high-data submission")
PY
    CONSTRAINED_C2F_CAMPAIGN_MODE=highdata \
    CONSTRAINED_C2F_ROOT="${CONSTRAINED_C2F_HIGHDATA_ROOT}" \
      bash "${SCRIPT_DIR}/submit_constrained_coarse_to_fine_experiment.sh"
    ;;
  *) echo "CONSTRAINED_C2F_CAMPAIGN_STAGE must be pilot or highdata" >&2; exit 2 ;;
esac

cat <<SUMMARY
constrained_c2f_pair_submission_complete:
  pilot_root: ${CONSTRAINED_C2F_PILOT_ROOT}
  requested_stage: ${CONSTRAINED_C2F_CAMPAIGN_STAGE}
  highdata_root: ${CONSTRAINED_C2F_HIGHDATA_ROOT}
SUMMARY
