#!/usr/bin/env bash
# Queue immutable approved final-test predictions/fusions/reports only.
set -euo pipefail
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
[[ "${ABPH_APPROVE_FINAL_TEST:-0}" == "1" ]] || { echo "Set ABPH_APPROVE_FINAL_TEST=1" >&2; exit 2; }
[[ "${CONFIRM_FINAL_TEST:-0}" == "1" ]] || { echo "Set CONFIRM_FINAL_TEST=1" >&2; exit 2; }
[[ -n "${ABPH_SELECTION_REPORT_PATH:-}" ]] || { echo "Set ABPH_SELECTION_REPORT_PATH" >&2; exit 2; }
[[ -n "${ABPH_FINAL_CLAIM_CONTRACT:-}" ]] || { echo "Set ABPH_FINAL_CLAIM_CONTRACT" >&2; exit 2; }
export ABPH_CAMPAIGN_MODE=highdata ABPH_STAGE_MODE=final_claims
exec bash "${PROJECT_DIR}/sbatch/submit_adaptive_binary_pseudooffline.sh" "$@"
