#!/usr/bin/env bash
# Write the AV10 architecture-view final comparison report.

#SBATCH --job-name=av10_report
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

: "${ARCHITECTURE_VIEW_10CLASS_ROOT:=${OUTPUT_ROOT}/architecture_view_10class_hlt0p6}"
: "${ARCHITECTURE_VIEW_10CLASS_TAGGER_ROOT:=${ARCHITECTURE_VIEW_10CLASS_ROOT}/taggers}"
: "${ARCHITECTURE_VIEW_10CLASS_FUSION_DIR:=${ARCHITECTURE_VIEW_10CLASS_ROOT}/fusion}"
: "${ARCHITECTURE_VIEW_10CLASS_REPORT_DIR:=${ARCHITECTURE_VIEW_10CLASS_ROOT}/final_report}"
: "${ARCHITECTURE_VIEW_10CLASS_VARIANTS:=av10_baseline_recheck av10_pn_context_to_part av10_pfn_context_to_part av10_pcnn_context_to_part av10_all_views_to_part av10_random_view_control av10_context_mlp_control}"
: "${ARCHITECTURE_VIEW_10CLASS_BASELINE_VARIANT:=av10_baseline_recheck}"
: "${ARCHITECTURE_VIEW_10CLASS_STANDALONE_FUSION_REPORT:=}"
: "${ARCHITECTURE_VIEW_10CLASS_CONFIRM_FINAL_TEST:=1}"

fresh_setup "$@"
fresh_require_file "scripts/write_architecture_view_10class_report.py"
fresh_split_words variant_args "${ARCHITECTURE_VIEW_10CLASS_VARIANTS}"
for variant in "${variant_args[@]}"; do
  fresh_require_file "${ARCHITECTURE_VIEW_10CLASS_TAGGER_ROOT}/${variant}/run_report.json"
done
fresh_require_file "${ARCHITECTURE_VIEW_10CLASS_FUSION_DIR}/fusion_report.json"
fresh_claim_new_dir "${ARCHITECTURE_VIEW_10CLASS_REPORT_DIR}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/write_architecture_view_10class_report.py"
  --tagger-root "${ARCHITECTURE_VIEW_10CLASS_TAGGER_ROOT}"
  --output-dir "${ARCHITECTURE_VIEW_10CLASS_REPORT_DIR}"
  --variants "${variant_args[@]}"
  --baseline-variant "${ARCHITECTURE_VIEW_10CLASS_BASELINE_VARIANT}"
  --fusion-report "${ARCHITECTURE_VIEW_10CLASS_FUSION_DIR}/fusion_report.json"
)
fresh_append_flag_if_enabled cmd --confirm-final-test "${ARCHITECTURE_VIEW_10CLASS_CONFIRM_FINAL_TEST}"
fresh_append_optional_arg cmd --standalone-fusion-report "${ARCHITECTURE_VIEW_10CLASS_STANDALONE_FUSION_REPORT}"

fresh_write_run_config "${ARCHITECTURE_VIEW_10CLASS_REPORT_DIR}" "architecture_view_10class_report" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${ARCHITECTURE_VIEW_10CLASS_REPORT_DIR}/architecture_view_10class_report.json"
  fresh_require_file "${ARCHITECTURE_VIEW_10CLASS_REPORT_DIR}/architecture_view_10class_report.md"
  fresh_require_file "${ARCHITECTURE_VIEW_10CLASS_REPORT_DIR}/individual_model_table.csv"
  fresh_require_file "${ARCHITECTURE_VIEW_10CLASS_REPORT_DIR}/fusion_metric_table.csv"
  fresh_require_file "${ARCHITECTURE_VIEW_10CLASS_REPORT_DIR}/binary_projection_table.csv"
  fresh_require_file "${ARCHITECTURE_VIEW_10CLASS_REPORT_DIR}/per_class_accuracy.csv"
  fresh_require_file "${ARCHITECTURE_VIEW_10CLASS_REPORT_DIR}/run_report.json"
fi
