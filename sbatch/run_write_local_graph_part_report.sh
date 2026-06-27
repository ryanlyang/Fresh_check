#!/usr/bin/env bash
# Write a local-graph Particle Transformer comparison report.

#SBATCH --job-name=localgraph_report
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

: "${LOCAL_GRAPH_PART_ROOT:=${OUTPUT_ROOT}/local_graph_part_qcd_hgg_binary_hlt0p6}"
: "${LOCAL_GRAPH_PART_TAGGER_ROOT:=${LOCAL_GRAPH_PART_ROOT}/taggers}"
: "${LOCAL_GRAPH_PART_FINAL_REPORT_DIR:=${LOCAL_GRAPH_PART_ROOT}/final_report}"
: "${LOCAL_GRAPH_PART_REPORT_VARIANTS:=hlt_part_baseline local_edgeconv_adapter local_point_attention_adapter local_point_attention_adapter_warmstart}"
: "${LOCAL_GRAPH_PART_REPORT_BASELINE_VARIANT:=hlt_part_baseline}"
: "${LOCAL_GRAPH_PART_REPORT_PRIMARY_METRIC:=fpr_at_signal_eff_0p50}"
: "${LOCAL_GRAPH_PART_REPORT_COMPARISON_SPLIT:=final_test}"
: "${LOCAL_GRAPH_PART_REPORT_SKIP_PARAMETER_COUNTS:=0}"
: "${LOCAL_GRAPH_PART_REPORT_CONFIRM_FINAL_TEST:=1}"

fresh_setup "$@"
fresh_require_file "scripts/write_local_graph_part_report.py"
fresh_split_words variant_args "${LOCAL_GRAPH_PART_REPORT_VARIANTS}"
for variant in "${variant_args[@]}"; do
  fresh_require_file "${LOCAL_GRAPH_PART_TAGGER_ROOT}/${variant}/run_report.json"
done
fresh_claim_new_dir "${LOCAL_GRAPH_PART_FINAL_REPORT_DIR}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/write_local_graph_part_report.py"
  --experiment-dir "${LOCAL_GRAPH_PART_TAGGER_ROOT}"
  --output-dir "${LOCAL_GRAPH_PART_FINAL_REPORT_DIR}"
  --baseline-variant "${LOCAL_GRAPH_PART_REPORT_BASELINE_VARIANT}"
  --variants "${variant_args[@]}"
)
fresh_append_optional_arg cmd --primary-metric "${LOCAL_GRAPH_PART_REPORT_PRIMARY_METRIC}"
fresh_append_optional_arg cmd --comparison-split "${LOCAL_GRAPH_PART_REPORT_COMPARISON_SPLIT}"
fresh_append_flag_if_enabled cmd --skip-parameter-counts "${LOCAL_GRAPH_PART_REPORT_SKIP_PARAMETER_COUNTS}"
fresh_append_flag_if_enabled cmd --confirm-final-test "${LOCAL_GRAPH_PART_REPORT_CONFIRM_FINAL_TEST}"

fresh_write_run_config "${LOCAL_GRAPH_PART_FINAL_REPORT_DIR}" "local_graph_part_final_report" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${LOCAL_GRAPH_PART_FINAL_REPORT_DIR}/local_graph_part_report.json"
  fresh_require_file "${LOCAL_GRAPH_PART_FINAL_REPORT_DIR}/local_graph_part_report.md"
  fresh_require_file "${LOCAL_GRAPH_PART_FINAL_REPORT_DIR}/metric_table.csv"
  fresh_require_file "${LOCAL_GRAPH_PART_FINAL_REPORT_DIR}/adapter_diagnostics.csv"
  fresh_require_file "${LOCAL_GRAPH_PART_FINAL_REPORT_DIR}/parameter_counts.csv"
  fresh_require_file "${LOCAL_GRAPH_PART_FINAL_REPORT_DIR}/runtime_summary.csv"
  fresh_require_file "${LOCAL_GRAPH_PART_FINAL_REPORT_DIR}/hlt_degradation_summary.csv"
fi
