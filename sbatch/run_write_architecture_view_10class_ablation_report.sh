#!/usr/bin/env bash
# Write the AV10 ablation report across HLT, fusion, and offline-transfer runs.

#SBATCH --job-name=av10_ablate_report
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

: "${ARCHITECTURE_VIEW_10CLASS_ABLATION_ROOT:=${OUTPUT_ROOT}/architecture_view_10class_ablation}"
: "${ARCHITECTURE_VIEW_10CLASS_ABLATION_TAGGER_ROOT:=${ARCHITECTURE_VIEW_10CLASS_ABLATION_ROOT}/taggers}"
: "${ARCHITECTURE_VIEW_10CLASS_ABLATION_REPORT_DIR:=${ARCHITECTURE_VIEW_10CLASS_ABLATION_ROOT}/final_report}"
: "${ARCHITECTURE_VIEW_10CLASS_ABLATION_VARIANTS:=av10_hlt_baseline_recheck av10_larger_part av10_extra_part_block av10_part_only_adapter av10_feature_mlp_adapter av10_lc_mlp_delta_features av10_feature_mlp_adapter_wide av10_frozen_part_feature_adapter av10_shuffled_feature_adapter av10_pcnn_context_repeat av10_pfn_context_repeat}"
: "${ARCHITECTURE_VIEW_10CLASS_ABLATION_BASELINE_VARIANT:=av10_hlt_baseline_recheck}"
: "${ARCHITECTURE_VIEW_10CLASS_ABLATION_FUSION_REPORT:=}"
: "${ARCHITECTURE_VIEW_10CLASS_ABLATION_REQUIRE_FUSION:=0}"
: "${ARCHITECTURE_VIEW_10CLASS_OFFLINE_TAGGER_ROOT:=}"
: "${ARCHITECTURE_VIEW_10CLASS_OFFLINE_VARIANTS:=av10_offline_part_baseline av10_offline_feature_mlp_adapter av10_offline_pcnn_context}"
: "${ARCHITECTURE_VIEW_10CLASS_OFFLINE_BASELINE_VARIANT:=av10_offline_part_baseline}"
: "${ARCHITECTURE_VIEW_10CLASS_ABLATION_REQUIRE_OFFLINE_TRANSFER:=0}"
: "${ARCHITECTURE_VIEW_10CLASS_CONFIRM_FINAL_TEST:=1}"

fresh_setup "$@"
fresh_require_file "scripts/write_architecture_view_10class_ablation_report.py"
fresh_split_words hlt_variant_args "${ARCHITECTURE_VIEW_10CLASS_ABLATION_VARIANTS}"
for variant in "${hlt_variant_args[@]}"; do
  fresh_require_file "${ARCHITECTURE_VIEW_10CLASS_ABLATION_TAGGER_ROOT}/${variant}/run_report.json"
done
fresh_split_words offline_variant_args "${ARCHITECTURE_VIEW_10CLASS_OFFLINE_VARIANTS}"
if [[ -n "${ARCHITECTURE_VIEW_10CLASS_OFFLINE_TAGGER_ROOT}" ]]; then
  for variant in "${offline_variant_args[@]}"; do
    fresh_require_file "${ARCHITECTURE_VIEW_10CLASS_OFFLINE_TAGGER_ROOT}/${variant}/run_report.json"
  done
fi
if [[ -n "${ARCHITECTURE_VIEW_10CLASS_ABLATION_FUSION_REPORT}" ]]; then
  fresh_require_file "${ARCHITECTURE_VIEW_10CLASS_ABLATION_FUSION_REPORT}"
fi
fresh_claim_new_dir "${ARCHITECTURE_VIEW_10CLASS_ABLATION_REPORT_DIR}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/write_architecture_view_10class_ablation_report.py"
  --hlt-tagger-root "${ARCHITECTURE_VIEW_10CLASS_ABLATION_TAGGER_ROOT}"
  --output-dir "${ARCHITECTURE_VIEW_10CLASS_ABLATION_REPORT_DIR}"
  --hlt-variants "${hlt_variant_args[@]}"
  --hlt-baseline-variant "${ARCHITECTURE_VIEW_10CLASS_ABLATION_BASELINE_VARIANT}"
  --offline-variants "${offline_variant_args[@]}"
  --offline-baseline-variant "${ARCHITECTURE_VIEW_10CLASS_OFFLINE_BASELINE_VARIANT}"
)
fresh_append_optional_arg cmd --fusion-report "${ARCHITECTURE_VIEW_10CLASS_ABLATION_FUSION_REPORT}"
fresh_append_optional_arg cmd --offline-tagger-root "${ARCHITECTURE_VIEW_10CLASS_OFFLINE_TAGGER_ROOT}"
fresh_append_flag_if_enabled cmd --require-fusion "${ARCHITECTURE_VIEW_10CLASS_ABLATION_REQUIRE_FUSION}"
fresh_append_flag_if_enabled cmd --require-offline-transfer "${ARCHITECTURE_VIEW_10CLASS_ABLATION_REQUIRE_OFFLINE_TRANSFER}"
fresh_append_flag_if_enabled cmd --confirm-final-test "${ARCHITECTURE_VIEW_10CLASS_CONFIRM_FINAL_TEST}"

fresh_write_run_config "${ARCHITECTURE_VIEW_10CLASS_ABLATION_REPORT_DIR}" "architecture_view_10class_ablation_report" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${ARCHITECTURE_VIEW_10CLASS_ABLATION_REPORT_DIR}/architecture_view_10class_ablation_report.json"
  fresh_require_file "${ARCHITECTURE_VIEW_10CLASS_ABLATION_REPORT_DIR}/architecture_view_10class_ablation_report.md"
  fresh_require_file "${ARCHITECTURE_VIEW_10CLASS_ABLATION_REPORT_DIR}/hlt_ablation_table.csv"
  fresh_require_file "${ARCHITECTURE_VIEW_10CLASS_ABLATION_REPORT_DIR}/fusion_complementarity_table.csv"
  fresh_require_file "${ARCHITECTURE_VIEW_10CLASS_ABLATION_REPORT_DIR}/offline_transfer_table.csv"
  fresh_require_file "${ARCHITECTURE_VIEW_10CLASS_ABLATION_REPORT_DIR}/parameter_accounting.csv"
  fresh_require_file "${ARCHITECTURE_VIEW_10CLASS_ABLATION_REPORT_DIR}/diagnostics.csv"
  fresh_require_file "${ARCHITECTURE_VIEW_10CLASS_ABLATION_REPORT_DIR}/decision_summary.txt"
  fresh_require_file "${ARCHITECTURE_VIEW_10CLASS_ABLATION_REPORT_DIR}/run_report.json"
fi
