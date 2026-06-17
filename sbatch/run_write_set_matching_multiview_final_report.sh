#!/usr/bin/env bash
# Write a compact final report for the set-matching multi-view branch.

#SBATCH --job-name=setmatch_report
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=debug
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

fresh_setup "$@"
fresh_require_file "scripts/write_set_matching_multiview_final_report.py"
fresh_require_file "${SET_MATCHING_ABLATION_DIR}/run_report.json"
fresh_claim_new_dir "${SET_MATCHING_FINAL_REPORT_DIR}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/write_set_matching_multiview_final_report.py"
  --output-dir "${SET_MATCHING_FINAL_REPORT_DIR}"
  --experiment-dir "${SET_MATCHING_ROOT}"
  --reconstructor-dir "${SET_MATCHING_RECONSTRUCTOR_DIR}"
  --reconstructed-view-dir "${SET_MATCHING_RECONSTRUCTED_VIEW_DIR}"
  --tagger-root "${SET_MATCHING_TAGGER_ROOT}"
  --ablation-dir "${SET_MATCHING_ABLATION_DIR}"
)
fresh_append_flag_if_enabled cmd --confirm-final-test "${SET_MATCHING_CONFIRM_FINAL_TEST}"

fresh_write_run_config "${SET_MATCHING_FINAL_REPORT_DIR}" "set_matching_multiview_final_report" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${SET_MATCHING_FINAL_REPORT_DIR}/final_report.json"
  fresh_require_file "${SET_MATCHING_FINAL_REPORT_DIR}/final_report.md"
fi
