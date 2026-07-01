#!/usr/bin/env bash
# Write an Architecture-View Residual ParT comparison report.

#SBATCH --job-name=archview_report
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=debug
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_ENV:=atlas_kd}"
export CONDA_ENV
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${ARCHITECTURE_VIEW_PART_ROOT:=${OUTPUT_ROOT}/architecture_view_part_qcd_hgg_binary_hlt0p6}"
: "${ARCHITECTURE_VIEW_PART_TAGGER_ROOT:=${ARCHITECTURE_VIEW_PART_ROOT}/taggers}"
: "${ARCHITECTURE_VIEW_PART_FINAL_REPORT_DIR:=${ARCHITECTURE_VIEW_PART_ROOT}/final_report}"
: "${ARCHITECTURE_VIEW_PART_REPORT_VARIANTS:=av_baseline_recheck av_all_views av_pn_only av_pfn_only av_pcnn_only av_random_view_control av_context_mlp_control}"
: "${ARCHITECTURE_VIEW_PART_REPORT_BASELINE_VARIANT:=av_baseline_recheck}"
: "${ARCHITECTURE_VIEW_PART_REPORT_PRIMARY_METRIC:=fpr_at_signal_eff_0p50}"
: "${ARCHITECTURE_VIEW_PART_REPORT_COMPARISON_SPLIT:=final_test}"
: "${ARCHITECTURE_VIEW_PART_REPORT_CONFIRM_FINAL_TEST:=1}"

fresh_setup "$@"
fresh_require_file "scripts/write_architecture_view_part_report.py"
fresh_split_words variant_args "${ARCHITECTURE_VIEW_PART_REPORT_VARIANTS}"
for variant in "${variant_args[@]}"; do
  fresh_require_file "${ARCHITECTURE_VIEW_PART_TAGGER_ROOT}/${variant}/run_report.json"
done
fresh_claim_new_dir "${ARCHITECTURE_VIEW_PART_FINAL_REPORT_DIR}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/write_architecture_view_part_report.py"
  --experiment-dir "${ARCHITECTURE_VIEW_PART_TAGGER_ROOT}"
  --output-dir "${ARCHITECTURE_VIEW_PART_FINAL_REPORT_DIR}"
  --baseline-variant "${ARCHITECTURE_VIEW_PART_REPORT_BASELINE_VARIANT}"
  --variants "${variant_args[@]}"
)
fresh_append_optional_arg cmd --primary-metric "${ARCHITECTURE_VIEW_PART_REPORT_PRIMARY_METRIC}"
fresh_append_optional_arg cmd --comparison-split "${ARCHITECTURE_VIEW_PART_REPORT_COMPARISON_SPLIT}"
fresh_append_flag_if_enabled cmd --confirm-final-test "${ARCHITECTURE_VIEW_PART_REPORT_CONFIRM_FINAL_TEST}"

fresh_write_run_config "${ARCHITECTURE_VIEW_PART_FINAL_REPORT_DIR}" "architecture_view_part_final_report" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${ARCHITECTURE_VIEW_PART_FINAL_REPORT_DIR}/architecture_view_part_final_report.json"
  fresh_require_file "${ARCHITECTURE_VIEW_PART_FINAL_REPORT_DIR}/architecture_view_part_final_report.md"
  fresh_require_file "${ARCHITECTURE_VIEW_PART_FINAL_REPORT_DIR}/run_report.json"
  fresh_require_file "${ARCHITECTURE_VIEW_PART_FINAL_REPORT_DIR}/metric_table.csv"
  fresh_require_file "${ARCHITECTURE_VIEW_PART_FINAL_REPORT_DIR}/baseline_comparison.csv"
  fresh_require_file "${ARCHITECTURE_VIEW_PART_FINAL_REPORT_DIR}/diagnostics.csv"
  fresh_require_file "${ARCHITECTURE_VIEW_PART_FINAL_REPORT_DIR}/runtime_summary.csv"
fi
