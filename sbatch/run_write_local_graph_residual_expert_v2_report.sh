#!/usr/bin/env bash
# Write the local-graph residual expert V2 Step 12 report.

#SBATCH --job-name=lgresidv2_report
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${LOCAL_GRAPH_RESIDUAL_V2_ROOT:=${OUTPUT_ROOT}/local_graph_residual_expert_v2_qcd_hgg_binary_hlt0p6}"
: "${LOCAL_GRAPH_RESIDUAL_V2_HLT_CACHE_DIR:=${LOCAL_GRAPH_RESIDUAL_V2_ROOT}/binary_inputs/hlt_cache}"
: "${LOCAL_GRAPH_RESIDUAL_V2_EMBEDDING_CACHE_DIR:=${LOCAL_GRAPH_RESIDUAL_V2_ROOT}/baseline_embeddings}"
: "${LOCAL_GRAPH_RESIDUAL_V2_EXPERT_ROOT:=${LOCAL_GRAPH_RESIDUAL_V2_ROOT}/residual_experts}"
: "${LOCAL_GRAPH_RESIDUAL_V2_FINAL_REPORT_DIR:=${LOCAL_GRAPH_RESIDUAL_V2_ROOT}/final_report}"
: "${LOCAL_GRAPH_RESIDUAL_V2_REPORT_VARIANTS:=}"
: "${LOCAL_GRAPH_RESIDUAL_V2_REPORT_PRIMARY_METRIC:=fpr_at_signal_eff_0p50}"
: "${LOCAL_GRAPH_RESIDUAL_V2_REPORT_COMPARISON_SPLIT:=final_test}"
: "${LOCAL_GRAPH_RESIDUAL_V2_REPORT_CONFIRM_FINAL_TEST:=1}"
: "${LOCAL_GRAPH_RESIDUAL_V2_REPORT_ALLOW_MISSING_VARIANTS:=0}"
: "${LOCAL_GRAPH_RESIDUAL_V2_REPORT_BATCH_SIZE:=128}"
: "${LOCAL_GRAPH_RESIDUAL_V2_REPORT_NUM_WORKERS:=${SLURM_CPUS_PER_TASK:-2}}"
: "${LOCAL_GRAPH_RESIDUAL_V2_REPORT_AMP:=1}"
: "${LOCAL_GRAPH_RESIDUAL_V2_REPORT_DISABLE_CALIBRATION_CONTROL:=0}"
: "${LOCAL_GRAPH_RESIDUAL_V2_EXPECTED_HLT_DEGRADATION_STRENGTH:=0.6}"
: "${LOCAL_GRAPH_RESIDUAL_V2_MODEL_VAL_SIZE:=}"
: "${LOCAL_GRAPH_RESIDUAL_V2_STACK_TRAIN_SIZE:=}"
: "${LOCAL_GRAPH_RESIDUAL_V2_STACK_VAL_SIZE:=}"
: "${LOCAL_GRAPH_RESIDUAL_V2_FINAL_TEST_SIZE:=}"
: "${LOCAL_GRAPH_RESIDUAL_V2_V1_REPORT_PATH:=}"
: "${LOCAL_GRAPH_RESIDUAL_V2_SCORE_FUSION_REPORT_PATH:=}"
: "${LOCAL_GRAPH_RESIDUAL_V2_STANDALONE_REPORT_PATH:=}"

fresh_setup "$@"
fresh_require_file "scripts/write_local_graph_residual_expert_v2_report.py"
for split in model_val stack_val final_test; do
  fresh_require_file "${LOCAL_GRAPH_RESIDUAL_V2_HLT_CACHE_DIR}/${split}_fixed_hlt.npz"
  fresh_require_file "${LOCAL_GRAPH_RESIDUAL_V2_HLT_CACHE_DIR}/${split}_fixed_hlt_metadata.json"
  fresh_require_file "${LOCAL_GRAPH_RESIDUAL_V2_EMBEDDING_CACHE_DIR}/${split}_baseline_embedding_cache.npz"
  fresh_require_file "${LOCAL_GRAPH_RESIDUAL_V2_EMBEDDING_CACHE_DIR}/${split}_baseline_embedding_cache_metadata.json"
done
fresh_require_file "${LOCAL_GRAPH_RESIDUAL_V2_EMBEDDING_CACHE_DIR}/model_train_baseline_embedding_cache.npz"
fresh_require_file "${LOCAL_GRAPH_RESIDUAL_V2_EMBEDDING_CACHE_DIR}/model_train_baseline_embedding_cache_metadata.json"
if [[ -n "${LOCAL_GRAPH_RESIDUAL_V2_REPORT_VARIANTS}" ]]; then
  fresh_split_words residual_variant_args "${LOCAL_GRAPH_RESIDUAL_V2_REPORT_VARIANTS}"
  for variant in "${residual_variant_args[@]}"; do
    fresh_require_file "${LOCAL_GRAPH_RESIDUAL_V2_EXPERT_ROOT}/${variant}/run_report.json"
  done
else
  residual_variant_args=()
fi
fresh_claim_new_dir "${LOCAL_GRAPH_RESIDUAL_V2_FINAL_REPORT_DIR}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/write_local_graph_residual_expert_v2_report.py"
  --output-dir "${LOCAL_GRAPH_RESIDUAL_V2_FINAL_REPORT_DIR}"
  --hlt-cache-dir "${LOCAL_GRAPH_RESIDUAL_V2_HLT_CACHE_DIR}"
  --baseline-embedding-cache-dir "${LOCAL_GRAPH_RESIDUAL_V2_EMBEDDING_CACHE_DIR}"
  --residual-expert-root "${LOCAL_GRAPH_RESIDUAL_V2_EXPERT_ROOT}"
  --primary-metric "${LOCAL_GRAPH_RESIDUAL_V2_REPORT_PRIMARY_METRIC}"
  --comparison-split "${LOCAL_GRAPH_RESIDUAL_V2_REPORT_COMPARISON_SPLIT}"
  --batch-size "${LOCAL_GRAPH_RESIDUAL_V2_REPORT_BATCH_SIZE}"
  --num-workers "${LOCAL_GRAPH_RESIDUAL_V2_REPORT_NUM_WORKERS}"
  --device "${DEVICE}"
  --expected-hlt-degradation-strength "${LOCAL_GRAPH_RESIDUAL_V2_EXPECTED_HLT_DEGRADATION_STRENGTH}"
)
if [[ ${#residual_variant_args[@]} -gt 0 ]]; then
  cmd+=(--residual-variants "${residual_variant_args[@]}")
fi
fresh_append_flag_if_enabled cmd --confirm-final-test "${LOCAL_GRAPH_RESIDUAL_V2_REPORT_CONFIRM_FINAL_TEST}"
fresh_append_flag_if_enabled cmd --allow-missing-residual-variants "${LOCAL_GRAPH_RESIDUAL_V2_REPORT_ALLOW_MISSING_VARIANTS}"
fresh_append_flag_if_enabled cmd --amp "${LOCAL_GRAPH_RESIDUAL_V2_REPORT_AMP}"
fresh_append_flag_if_enabled cmd --disable-calibration-control "${LOCAL_GRAPH_RESIDUAL_V2_REPORT_DISABLE_CALIBRATION_CONTROL}"
if [[ -n "${LOCAL_GRAPH_RESIDUAL_V2_MODEL_VAL_SIZE}" ]]; then
  cmd+=(--max-model-val-jets "${LOCAL_GRAPH_RESIDUAL_V2_MODEL_VAL_SIZE}")
fi
if [[ -n "${LOCAL_GRAPH_RESIDUAL_V2_STACK_TRAIN_SIZE}" ]]; then
  cmd+=(--max-stack-train-jets "${LOCAL_GRAPH_RESIDUAL_V2_STACK_TRAIN_SIZE}")
fi
if [[ -n "${LOCAL_GRAPH_RESIDUAL_V2_STACK_VAL_SIZE}" ]]; then
  cmd+=(--max-stack-val-jets "${LOCAL_GRAPH_RESIDUAL_V2_STACK_VAL_SIZE}")
fi
if [[ -n "${LOCAL_GRAPH_RESIDUAL_V2_FINAL_TEST_SIZE}" ]]; then
  cmd+=(--max-final-test-jets "${LOCAL_GRAPH_RESIDUAL_V2_FINAL_TEST_SIZE}")
fi
if [[ -n "${LOCAL_GRAPH_RESIDUAL_V2_V1_REPORT_PATH}" ]]; then
  cmd+=(--v1-residual-report-path "${LOCAL_GRAPH_RESIDUAL_V2_V1_REPORT_PATH}")
fi
if [[ -n "${LOCAL_GRAPH_RESIDUAL_V2_SCORE_FUSION_REPORT_PATH}" ]]; then
  cmd+=(--score-fusion-report-path "${LOCAL_GRAPH_RESIDUAL_V2_SCORE_FUSION_REPORT_PATH}")
fi
if [[ -n "${LOCAL_GRAPH_RESIDUAL_V2_STANDALONE_REPORT_PATH}" ]]; then
  cmd+=(--standalone-report-path "${LOCAL_GRAPH_RESIDUAL_V2_STANDALONE_REPORT_PATH}")
fi

fresh_write_run_config "${LOCAL_GRAPH_RESIDUAL_V2_FINAL_REPORT_DIR}" "local_graph_residual_v2_report" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${LOCAL_GRAPH_RESIDUAL_V2_FINAL_REPORT_DIR}/local_graph_residual_expert_v2_report.json"
  fresh_require_file "${LOCAL_GRAPH_RESIDUAL_V2_FINAL_REPORT_DIR}/metric_table.csv"
fi
