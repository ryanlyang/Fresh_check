#!/usr/bin/env bash
# Write the PDV3 AV10-adapter privileged distillation final report.

#SBATCH --job-name=pdv3_report
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=debug
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

fresh_setup "$@"
fresh_require_dir "${PDV3_STUDENTS_DIR}"
fresh_claim_new_dir "${PDV3_FINAL_REPORT_DIR}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/write_pdv3_report.py"
  --output-dir "${PDV3_FINAL_REPORT_DIR}"
  --students-dir "${PDV3_STUDENTS_DIR}"
  --baseline-variant pdv3_hlt_part_ce
  --confirm-final-test
)

old_ifs="${IFS}"
IFS=' '
read -r -a variants <<< "${PDV3_REPORT_STUDENT_VARIANTS}"
IFS="${old_ifs}"
if [[ "${#variants[@]}" -gt 0 ]]; then
  cmd+=(--student-variants "${variants[@]}")
fi
fresh_append_flag_if_enabled cmd --allow-missing-students "${PDV3_REPORT_ALLOW_MISSING_STUDENTS}"

fresh_write_run_config "${PDV3_FINAL_REPORT_DIR}" "pdv3_report" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${PDV3_FINAL_REPORT_DIR}/pdv3_report.json"
  fresh_require_file "${PDV3_FINAL_REPORT_DIR}/pdv3_report.md"
  fresh_require_file "${PDV3_FINAL_REPORT_DIR}/student_metrics.csv"
  fresh_require_file "${PDV3_FINAL_REPORT_DIR}/comparison_table.csv"
  fresh_require_file "${PDV3_FINAL_REPORT_DIR}/confusion_matrix.csv"
  fresh_assert_json_ok "${PDV3_FINAL_REPORT_DIR}/pdv3_report.json"
fi
