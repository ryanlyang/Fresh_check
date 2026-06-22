#!/usr/bin/env bash
# Write a subtoken Particle Transformer final comparison report from child run reports.

#SBATCH --job-name=subtoken_report
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=debug
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${SUBTOKEN_PART_REPORT_EXPERIMENT_DIR:?Set SUBTOKEN_PART_REPORT_EXPERIMENT_DIR}"
: "${SUBTOKEN_PART_REPORT_OUTPUT_DIR:=${SUBTOKEN_PART_REPORT_EXPERIMENT_DIR}/final_report}"
: "${SUBTOKEN_PART_REPORT_VARIANTS:=}"
: "${SUBTOKEN_PART_REPORT_BASELINE_VARIANT:=hlt_part_baseline}"
: "${SUBTOKEN_PART_REPORT_PRIMARY_METRIC:=}"
: "${SUBTOKEN_PART_REPORT_COMPARISON_SPLIT:=}"
: "${SUBTOKEN_PART_REPORT_SKIP_PARAMETER_COUNTS:=0}"
: "${SUBTOKEN_PART_REPORT_CONFIRM_FINAL_TEST:=1}"

fresh_setup "$@"
fresh_require_file "scripts/write_subtoken_part_report.py"
fresh_require_file "${SUBTOKEN_PART_REPORT_EXPERIMENT_DIR}/run_report.json"
fresh_claim_new_dir "${SUBTOKEN_PART_REPORT_OUTPUT_DIR}"

fresh_split_words variant_args "${SUBTOKEN_PART_REPORT_VARIANTS}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/write_subtoken_part_report.py"
  --experiment-dir "${SUBTOKEN_PART_REPORT_EXPERIMENT_DIR}"
  --output-dir "${SUBTOKEN_PART_REPORT_OUTPUT_DIR}"
  --baseline-variant "${SUBTOKEN_PART_REPORT_BASELINE_VARIANT}"
)
if [[ "${#variant_args[@]}" -gt 0 ]]; then
  cmd+=(--variants "${variant_args[@]}")
fi
fresh_append_optional_arg cmd --primary-metric "${SUBTOKEN_PART_REPORT_PRIMARY_METRIC}"
fresh_append_optional_arg cmd --comparison-split "${SUBTOKEN_PART_REPORT_COMPARISON_SPLIT}"
fresh_append_flag_if_enabled cmd --skip-parameter-counts "${SUBTOKEN_PART_REPORT_SKIP_PARAMETER_COUNTS}"
fresh_append_flag_if_enabled cmd --confirm-final-test "${SUBTOKEN_PART_REPORT_CONFIRM_FINAL_TEST}"

fresh_write_run_config "${SUBTOKEN_PART_REPORT_OUTPUT_DIR}" "subtoken_part_final_report" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${SUBTOKEN_PART_REPORT_OUTPUT_DIR}/subtoken_part_final_report.json"
  fresh_require_file "${SUBTOKEN_PART_REPORT_OUTPUT_DIR}/subtoken_part_final_report.md"
  fresh_require_file "${SUBTOKEN_PART_REPORT_OUTPUT_DIR}/metric_table.csv"
  fresh_require_file "${SUBTOKEN_PART_REPORT_OUTPUT_DIR}/runtime_summary.csv"
fi
