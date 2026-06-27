#!/usr/bin/env bash
# Write a dual-view ParT real-vs-shuffled comparison report.

#SBATCH --job-name=dualview_report
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

: "${DUALVIEW_PART_REPORT_OUTPUT_DIR:=${DUALVIEW_PART_FINAL_REPORT_DIR}}"

fresh_setup "$@"
fresh_require_file "scripts/write_dualview_part_report.py"
fresh_split_words variant_args "${DUALVIEW_PART_REPORT_VARIANTS}"
for variant in "${variant_args[@]}"; do
  fresh_require_file "${DUALVIEW_PART_TAGGER_ROOT}/${variant}/run_report.json"
done
fresh_claim_new_dir "${DUALVIEW_PART_REPORT_OUTPUT_DIR}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/write_dualview_part_report.py"
  --experiment-dir "${DUALVIEW_PART_EXPERIMENT_DIR}"
  --output-dir "${DUALVIEW_PART_REPORT_OUTPUT_DIR}"
  --tagger-root "${DUALVIEW_PART_TAGGER_ROOT}"
  --variants "${variant_args[@]}"
  --real-variant "${DUALVIEW_PART_REPORT_REAL_VARIANT}"
  --shuffled-variant "${DUALVIEW_PART_REPORT_SHUFFLED_VARIANT}"
  --primary-metric "${DUALVIEW_PART_SELECTION_METRIC}"
  --comparison-split "${DUALVIEW_PART_REPORT_COMPARISON_SPLIT}"
)
fresh_append_flag_if_enabled cmd --confirm-final-test "${DUALVIEW_PART_CONFIRM_FINAL_TEST}"
if ! fresh_bool_enabled "${DUALVIEW_PART_REPORT_REQUIRE_REAL_BEATS_SHUFFLED}"; then
  cmd+=(--allow-real-not-better)
fi

fresh_write_run_config "${DUALVIEW_PART_REPORT_OUTPUT_DIR}" "dualview_part_real_vs_shuffled_report" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${DUALVIEW_PART_REPORT_OUTPUT_DIR}/dualview_part_report.json"
  fresh_require_file "${DUALVIEW_PART_REPORT_OUTPUT_DIR}/dualview_part_report.md"
  fresh_require_file "${DUALVIEW_PART_REPORT_OUTPUT_DIR}/metric_table.csv"
fi
