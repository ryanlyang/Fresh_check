#!/usr/bin/env bash
# Write strict model-selection or final-claim campaign report.

#SBATCH --job-name=abph_report
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

set -euo pipefail
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
source "${PROJECT_DIR}/sbatch/common.sh"
MODE="${1:?Usage: run_adaptive_binary_report.sh <selection|final_claim>}"
: "${ABPH_ROOT:=${OUTPUT_ROOT}/adaptive_binary_pseudooffline}"
export PYTHONNOUSERSITE=1
fresh_setup
cmd=("${PYTHON_BIN}" -u scripts/write_adaptive_binary_pseudooffline_report.py --campaign-root "${ABPH_ROOT}")
if [[ "${MODE}" == "final_claim" ]]; then
  [[ -n "${ABPH_FINAL_CLAIM_CONTRACT:-}" ]] || { echo "final report lacks frozen claim contract" >&2; exit 2; }
  [[ -n "${ABPH_SELECTION_REPORT_PATH:-}" ]] || { echo "final report lacks frozen selection report" >&2; exit 2; }
  cmd+=(--confirm-final-test --output-dir "${ABPH_ROOT}/final_claim_report"
    --selection-report "${ABPH_SELECTION_REPORT_PATH}"
    --final-claim-contract "${ABPH_FINAL_CLAIM_CONTRACT}")
else
  cmd+=(--output-dir "${ABPH_ROOT}/report")
fi
fresh_run "${cmd[@]}"
