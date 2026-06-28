#!/usr/bin/env bash
# Write a multi-scale subjet Particle Transformer comparison report.

#SBATCH --job-name=multiscale_report
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

: "${MULTISCALE_SUBJET_PART_ROOT:=${OUTPUT_ROOT}/multiscale_subjet_part_qcd_hgg_binary_hlt0p6}"
: "${MULTISCALE_SUBJET_PART_TAGGER_ROOT:=${MULTISCALE_SUBJET_PART_ROOT}/taggers}"
: "${MULTISCALE_SUBJET_PART_FINAL_REPORT_DIR:=${MULTISCALE_SUBJET_PART_ROOT}/final_report}"
: "${MULTISCALE_SUBJET_PART_REPORT_VARIANTS:=hlt_part_baseline multiscale_subjet_residual_part_adapter pure_perceiver_latent_control part_plus_random_subjet_control}"
: "${MULTISCALE_SUBJET_PART_REPORT_BASELINE_VARIANT:=hlt_part_baseline}"
: "${MULTISCALE_SUBJET_PART_REPORT_PRIMARY_VARIANT:=multiscale_subjet_residual_part_adapter}"
: "${MULTISCALE_SUBJET_PART_REPORT_PRIMARY_METRIC:=fpr_at_signal_eff_0p50}"
: "${MULTISCALE_SUBJET_PART_REPORT_COMPARISON_SPLIT:=final_test}"
: "${MULTISCALE_SUBJET_PART_REPORT_SKIP_PARAMETER_COUNTS:=0}"
: "${MULTISCALE_SUBJET_PART_REPORT_CONFIRM_FINAL_TEST:=1}"
: "${MULTISCALE_SUBJET_PART_REPORT_ALLOW_NON_PROTOCOL:=0}"
: "${MULTISCALE_SUBJET_PART_REPORT_ALLOW_MISSING_DEFAULT_CONTROLS:=0}"
: "${MULTISCALE_SUBJET_PART_REPORT_REQUIRE_HLT_DEGRADATION_SLICES:=0}"

fresh_setup "$@"
fresh_require_file "scripts/write_multiscale_subjet_part_report.py"
fresh_split_words variant_args "${MULTISCALE_SUBJET_PART_REPORT_VARIANTS}"
for variant in "${variant_args[@]}"; do
  fresh_require_file "${MULTISCALE_SUBJET_PART_TAGGER_ROOT}/${variant}/run_report.json"
done
fresh_claim_new_dir "${MULTISCALE_SUBJET_PART_FINAL_REPORT_DIR}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/write_multiscale_subjet_part_report.py"
  --experiment-dir "${MULTISCALE_SUBJET_PART_TAGGER_ROOT}"
  --output-dir "${MULTISCALE_SUBJET_PART_FINAL_REPORT_DIR}"
  --baseline-variant "${MULTISCALE_SUBJET_PART_REPORT_BASELINE_VARIANT}"
  --primary-variant "${MULTISCALE_SUBJET_PART_REPORT_PRIMARY_VARIANT}"
  --variants "${variant_args[@]}"
)
fresh_append_optional_arg cmd --primary-metric "${MULTISCALE_SUBJET_PART_REPORT_PRIMARY_METRIC}"
fresh_append_optional_arg cmd --comparison-split "${MULTISCALE_SUBJET_PART_REPORT_COMPARISON_SPLIT}"
fresh_append_flag_if_enabled cmd --skip-parameter-counts "${MULTISCALE_SUBJET_PART_REPORT_SKIP_PARAMETER_COUNTS}"
fresh_append_flag_if_enabled cmd --confirm-final-test "${MULTISCALE_SUBJET_PART_REPORT_CONFIRM_FINAL_TEST}"
fresh_append_flag_if_enabled cmd --allow-non-protocol-report "${MULTISCALE_SUBJET_PART_REPORT_ALLOW_NON_PROTOCOL}"
fresh_append_flag_if_enabled cmd --allow-missing-default-controls "${MULTISCALE_SUBJET_PART_REPORT_ALLOW_MISSING_DEFAULT_CONTROLS}"
fresh_append_flag_if_enabled cmd --require-hlt-degradation-slices "${MULTISCALE_SUBJET_PART_REPORT_REQUIRE_HLT_DEGRADATION_SLICES}"

fresh_write_run_config "${MULTISCALE_SUBJET_PART_FINAL_REPORT_DIR}" "multiscale_subjet_part_final_report" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${MULTISCALE_SUBJET_PART_FINAL_REPORT_DIR}/multiscale_subjet_part_report.json"
  fresh_require_file "${MULTISCALE_SUBJET_PART_FINAL_REPORT_DIR}/multiscale_subjet_part_report.md"
  fresh_require_file "${MULTISCALE_SUBJET_PART_FINAL_REPORT_DIR}/run_report.json"
  fresh_require_file "${MULTISCALE_SUBJET_PART_FINAL_REPORT_DIR}/metric_table.csv"
  fresh_require_file "${MULTISCALE_SUBJET_PART_FINAL_REPORT_DIR}/diagnostics.csv"
  fresh_require_file "${MULTISCALE_SUBJET_PART_FINAL_REPORT_DIR}/parameter_counts.csv"
  fresh_require_file "${MULTISCALE_SUBJET_PART_FINAL_REPORT_DIR}/runtime.csv"
  fresh_require_file "${MULTISCALE_SUBJET_PART_FINAL_REPORT_DIR}/hlt_degradation.csv"
fi
