#!/usr/bin/env bash
# Write the local-graph residual expert final report.

#SBATCH --job-name=localgraph_resid_report
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=1-00:00:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${LOCAL_GRAPH_RESIDUAL_ROOT:=${OUTPUT_ROOT}/local_graph_residual_expert_qcd_hgg_binary_hlt0p6}"
: "${LOCAL_GRAPH_RESIDUAL_HLT_CACHE_DIR:=${LOCAL_GRAPH_RESIDUAL_ROOT}/binary_inputs/hlt_cache}"
: "${LOCAL_GRAPH_RESIDUAL_BASELINE_LOGIT_CACHE_DIR:=${LOCAL_GRAPH_RESIDUAL_ROOT}/baseline_logits}"
: "${LOCAL_GRAPH_RESIDUAL_EXPERT_ROOT:=${LOCAL_GRAPH_RESIDUAL_ROOT}/residual_experts}"
: "${LOCAL_GRAPH_RESIDUAL_TAGGER_ROOT:=${LOCAL_GRAPH_RESIDUAL_ROOT}/taggers}"
: "${LOCAL_GRAPH_RESIDUAL_FINAL_REPORT_DIR:=${LOCAL_GRAPH_RESIDUAL_ROOT}/final_report}"
: "${LOCAL_GRAPH_RESIDUAL_REPORT_VARIANTS:=}"
: "${LOCAL_GRAPH_RESIDUAL_REPORT_STANDALONE_VARIANTS:=}"
: "${LOCAL_GRAPH_RESIDUAL_SCORE_FUSION_REPORT_PATH:=}"
: "${LOCAL_GRAPH_RESIDUAL_REPORT_PRIMARY_METRIC:=fpr_at_signal_eff_0p50}"
: "${LOCAL_GRAPH_RESIDUAL_REPORT_COMPARISON_SPLIT:=final_test}"
: "${LOCAL_GRAPH_RESIDUAL_REPORT_BATCH_SIZE:=128}"
: "${LOCAL_GRAPH_RESIDUAL_REPORT_NUM_WORKERS:=${NUM_WORKERS}}"
: "${LOCAL_GRAPH_RESIDUAL_REPORT_DEVICE:=${DEVICE}}"
: "${LOCAL_GRAPH_RESIDUAL_REPORT_AMP:=0}"
: "${LOCAL_GRAPH_RESIDUAL_REPORT_MODEL_VAL_SIZE:=150000}"
: "${LOCAL_GRAPH_RESIDUAL_REPORT_STACK_VAL_SIZE:=150000}"
: "${LOCAL_GRAPH_RESIDUAL_REPORT_FINAL_TEST_SIZE:=500000}"
: "${LOCAL_GRAPH_RESIDUAL_REPORT_CONFIRM_FINAL_TEST:=1}"
: "${LOCAL_GRAPH_RESIDUAL_REPORT_SKIP_CHECKPOINT_EVALUATION:=0}"
: "${LOCAL_GRAPH_RESIDUAL_REPORT_ALLOW_PRECOMPUTED_EVALUATIONS:=0}"
: "${LOCAL_GRAPH_RESIDUAL_REPORT_ALLOW_MISSING_VARIANTS:=0}"
: "${LOCAL_GRAPH_RESIDUAL_REPORT_SKIP_HLT_HASH_CHECK:=0}"
: "${LOCAL_GRAPH_RESIDUAL_REPORT_SKIP_HLT_PARAMS_CHECK:=0}"
: "${LOCAL_GRAPH_RESIDUAL_EXPECTED_HLT_DEGRADATION_STRENGTH:=0.6}"

fresh_setup "$@"
fresh_require_file "scripts/write_local_graph_residual_expert_report.py"
fresh_require_file "${LOCAL_GRAPH_RESIDUAL_BASELINE_LOGIT_CACHE_DIR}/baseline_logit_manifest.json"
for split in model_train model_val stack_val final_test; do
  fresh_require_file "${LOCAL_GRAPH_RESIDUAL_BASELINE_LOGIT_CACHE_DIR}/${split}_baseline_logits.npz"
  fresh_require_file "${LOCAL_GRAPH_RESIDUAL_BASELINE_LOGIT_CACHE_DIR}/${split}_baseline_logits_metadata.json"
done
if [[ -n "${LOCAL_GRAPH_RESIDUAL_REPORT_VARIANTS}" ]]; then
  fresh_split_words residual_variant_args "${LOCAL_GRAPH_RESIDUAL_REPORT_VARIANTS}"
  for variant in "${residual_variant_args[@]}"; do
    fresh_require_file "${LOCAL_GRAPH_RESIDUAL_EXPERT_ROOT}/${variant}/run_report.json"
    if ! fresh_bool_enabled "${LOCAL_GRAPH_RESIDUAL_REPORT_SKIP_CHECKPOINT_EVALUATION}"; then
      fresh_require_file "${LOCAL_GRAPH_RESIDUAL_EXPERT_ROOT}/${variant}/best_model_val.pt"
    fi
  done
else
  residual_variant_args=()
fi
if [[ -n "${LOCAL_GRAPH_RESIDUAL_REPORT_STANDALONE_VARIANTS}" ]]; then
  fresh_split_words standalone_variant_args "${LOCAL_GRAPH_RESIDUAL_REPORT_STANDALONE_VARIANTS}"
else
  standalone_variant_args=()
fi
fresh_claim_new_dir "${LOCAL_GRAPH_RESIDUAL_FINAL_REPORT_DIR}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/write_local_graph_residual_expert_report.py"
  --output-dir "${LOCAL_GRAPH_RESIDUAL_FINAL_REPORT_DIR}"
  --hlt-cache-dir "${LOCAL_GRAPH_RESIDUAL_HLT_CACHE_DIR}"
  --baseline-logit-cache-dir "${LOCAL_GRAPH_RESIDUAL_BASELINE_LOGIT_CACHE_DIR}"
  --residual-expert-root "${LOCAL_GRAPH_RESIDUAL_EXPERT_ROOT}"
  --primary-metric "${LOCAL_GRAPH_RESIDUAL_REPORT_PRIMARY_METRIC}"
  --comparison-split "${LOCAL_GRAPH_RESIDUAL_REPORT_COMPARISON_SPLIT}"
  --batch-size "${LOCAL_GRAPH_RESIDUAL_REPORT_BATCH_SIZE}"
  --num-workers "${LOCAL_GRAPH_RESIDUAL_REPORT_NUM_WORKERS}"
  --device "${LOCAL_GRAPH_RESIDUAL_REPORT_DEVICE}"
  --max-model-val-jets "${LOCAL_GRAPH_RESIDUAL_REPORT_MODEL_VAL_SIZE}"
  --max-stack-val-jets "${LOCAL_GRAPH_RESIDUAL_REPORT_STACK_VAL_SIZE}"
  --max-final-test-jets "${LOCAL_GRAPH_RESIDUAL_REPORT_FINAL_TEST_SIZE}"
  --expected-hlt-degradation-strength "${LOCAL_GRAPH_RESIDUAL_EXPECTED_HLT_DEGRADATION_STRENGTH}"
)
if [[ ${#residual_variant_args[@]} -gt 0 ]]; then
  cmd+=(--residual-variants "${residual_variant_args[@]}")
fi
if [[ ${#standalone_variant_args[@]} -gt 0 ]]; then
  cmd+=(--standalone-tagger-root "${LOCAL_GRAPH_RESIDUAL_TAGGER_ROOT}" --standalone-variants "${standalone_variant_args[@]}")
fi
fresh_append_optional_arg cmd --score-fusion-report-path "${LOCAL_GRAPH_RESIDUAL_SCORE_FUSION_REPORT_PATH}"
fresh_append_flag_if_enabled cmd --confirm-final-test "${LOCAL_GRAPH_RESIDUAL_REPORT_CONFIRM_FINAL_TEST}"
fresh_append_flag_if_enabled cmd --amp "${LOCAL_GRAPH_RESIDUAL_REPORT_AMP}"
fresh_append_flag_if_enabled cmd --skip-checkpoint-evaluation "${LOCAL_GRAPH_RESIDUAL_REPORT_SKIP_CHECKPOINT_EVALUATION}"
fresh_append_flag_if_enabled cmd --allow-precomputed-evaluations "${LOCAL_GRAPH_RESIDUAL_REPORT_ALLOW_PRECOMPUTED_EVALUATIONS}"
fresh_append_flag_if_enabled cmd --allow-missing-residual-variants "${LOCAL_GRAPH_RESIDUAL_REPORT_ALLOW_MISSING_VARIANTS}"
fresh_append_flag_if_enabled cmd --skip-hlt-hash-check "${LOCAL_GRAPH_RESIDUAL_REPORT_SKIP_HLT_HASH_CHECK}"
fresh_append_flag_if_enabled cmd --skip-hlt-params-check "${LOCAL_GRAPH_RESIDUAL_REPORT_SKIP_HLT_PARAMS_CHECK}"

fresh_write_run_config "${LOCAL_GRAPH_RESIDUAL_FINAL_REPORT_DIR}" "local_graph_residual_expert_final_report" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${LOCAL_GRAPH_RESIDUAL_FINAL_REPORT_DIR}/local_graph_residual_expert_report.json"
  fresh_require_file "${LOCAL_GRAPH_RESIDUAL_FINAL_REPORT_DIR}/local_graph_residual_expert_report.md"
  fresh_require_file "${LOCAL_GRAPH_RESIDUAL_FINAL_REPORT_DIR}/residual_metric_table.csv"
  fresh_require_file "${LOCAL_GRAPH_RESIDUAL_FINAL_REPORT_DIR}/residual_diagnostics.csv"
  fresh_require_file "${LOCAL_GRAPH_RESIDUAL_FINAL_REPORT_DIR}/baseline_comparison.csv"
  fresh_require_file "${LOCAL_GRAPH_RESIDUAL_FINAL_REPORT_DIR}/run_report.json"
fi
