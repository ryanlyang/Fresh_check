#!/usr/bin/env bash
# Cache frozen HLT ParT baseline logits for local-graph residual experts.

#SBATCH --job-name=localgraph_logits
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=12:00:00
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
: "${LOCAL_GRAPH_BASELINE_LOGIT_CACHE_OUTPUT_DIR:=${LOCAL_GRAPH_RESIDUAL_BASELINE_LOGIT_CACHE_DIR:-${LOCAL_GRAPH_RESIDUAL_ROOT}/baseline_logits}}"
: "${LOCAL_GRAPH_BASELINE_LOGIT_CACHE_HLT_CACHE_DIR:=${LOCAL_GRAPH_RESIDUAL_HLT_CACHE_DIR:-${LOCAL_GRAPH_RESIDUAL_ROOT}/binary_inputs/hlt_cache}}"
: "${LOCAL_GRAPH_BASELINE_LOGIT_CACHE_CHECKPOINT:=${LOCAL_GRAPH_RESIDUAL_BASELINE_CHECKPOINT:-${LOCAL_GRAPH_RESIDUAL_ROOT}/taggers/hlt_part_baseline/best_model_val.pt}}"
: "${LOCAL_GRAPH_BASELINE_LOGIT_CACHE_EXPECTED_CHECKPOINT_VARIANT:=hlt_part_baseline}"
: "${LOCAL_GRAPH_BASELINE_LOGIT_CACHE_SPLITS:=model_train model_val stack_train stack_val final_test}"
: "${LOCAL_GRAPH_BASELINE_LOGIT_CACHE_METRIC_SPLITS:=model_train model_val stack_train stack_val}"
: "${LOCAL_GRAPH_BASELINE_LOGIT_CACHE_BATCH_SIZE:=256}"
: "${LOCAL_GRAPH_BASELINE_LOGIT_CACHE_NUM_WORKERS:=${NUM_WORKERS}}"
: "${LOCAL_GRAPH_BASELINE_LOGIT_CACHE_DEVICE:=${DEVICE}}"
: "${LOCAL_GRAPH_BASELINE_LOGIT_CACHE_SEED:=7717}"
: "${LOCAL_GRAPH_BASELINE_LOGIT_CACHE_EXPECTED_HLT_DEGRADATION_STRENGTH:=0.6}"
: "${LOCAL_GRAPH_BASELINE_LOGIT_CACHE_MODEL_TRAIN_SIZE:=500000}"
: "${LOCAL_GRAPH_BASELINE_LOGIT_CACHE_MODEL_VAL_SIZE:=150000}"
: "${LOCAL_GRAPH_BASELINE_LOGIT_CACHE_STACK_TRAIN_SIZE:=500000}"
: "${LOCAL_GRAPH_BASELINE_LOGIT_CACHE_STACK_VAL_SIZE:=150000}"
: "${LOCAL_GRAPH_BASELINE_LOGIT_CACHE_FINAL_TEST_SIZE:=500000}"
: "${LOCAL_GRAPH_BASELINE_LOGIT_CACHE_SKIP_HLT_HASH_CHECK:=0}"
: "${LOCAL_GRAPH_BASELINE_LOGIT_CACHE_SKIP_HLT_PARAMS_CHECK:=0}"
: "${LOCAL_GRAPH_BASELINE_LOGIT_CACHE_OVERWRITE:=0}"

fresh_setup "$@"
fresh_require_file "scripts/cache_local_graph_baseline_logits.py"
fresh_require_file "${LOCAL_GRAPH_BASELINE_LOGIT_CACHE_CHECKPOINT}"
fresh_split_words split_args "${LOCAL_GRAPH_BASELINE_LOGIT_CACHE_SPLITS}"
fresh_split_words metric_split_args "${LOCAL_GRAPH_BASELINE_LOGIT_CACHE_METRIC_SPLITS}"
for split in "${split_args[@]}"; do
  fresh_require_file "${LOCAL_GRAPH_BASELINE_LOGIT_CACHE_HLT_CACHE_DIR}/${split}_fixed_hlt.npz"
  fresh_require_file "${LOCAL_GRAPH_BASELINE_LOGIT_CACHE_HLT_CACHE_DIR}/${split}_fixed_hlt_metadata.json"
done
if fresh_bool_enabled "${LOCAL_GRAPH_BASELINE_LOGIT_CACHE_OVERWRITE}"; then
  mkdir -p "${LOCAL_GRAPH_BASELINE_LOGIT_CACHE_OUTPUT_DIR}"
else
  fresh_claim_new_dir "${LOCAL_GRAPH_BASELINE_LOGIT_CACHE_OUTPUT_DIR}"
fi

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/cache_local_graph_baseline_logits.py"
  --output-dir "${LOCAL_GRAPH_BASELINE_LOGIT_CACHE_OUTPUT_DIR}"
  --hlt-cache-dir "${LOCAL_GRAPH_BASELINE_LOGIT_CACHE_HLT_CACHE_DIR}"
  --checkpoint "${LOCAL_GRAPH_BASELINE_LOGIT_CACHE_CHECKPOINT}"
  --expected-checkpoint-variant "${LOCAL_GRAPH_BASELINE_LOGIT_CACHE_EXPECTED_CHECKPOINT_VARIANT}"
  --splits "${split_args[@]}"
  --metric-splits "${metric_split_args[@]}"
  --batch-size "${LOCAL_GRAPH_BASELINE_LOGIT_CACHE_BATCH_SIZE}"
  --num-workers "${LOCAL_GRAPH_BASELINE_LOGIT_CACHE_NUM_WORKERS}"
  --device "${LOCAL_GRAPH_BASELINE_LOGIT_CACHE_DEVICE}"
  --seed "${LOCAL_GRAPH_BASELINE_LOGIT_CACHE_SEED}"
  --expected-hlt-degradation-strength "${LOCAL_GRAPH_BASELINE_LOGIT_CACHE_EXPECTED_HLT_DEGRADATION_STRENGTH}"
  --max-model-train-jets "${LOCAL_GRAPH_BASELINE_LOGIT_CACHE_MODEL_TRAIN_SIZE}"
  --max-model-val-jets "${LOCAL_GRAPH_BASELINE_LOGIT_CACHE_MODEL_VAL_SIZE}"
  --max-stack-train-jets "${LOCAL_GRAPH_BASELINE_LOGIT_CACHE_STACK_TRAIN_SIZE}"
  --max-stack-val-jets "${LOCAL_GRAPH_BASELINE_LOGIT_CACHE_STACK_VAL_SIZE}"
  --max-final-test-jets "${LOCAL_GRAPH_BASELINE_LOGIT_CACHE_FINAL_TEST_SIZE}"
)
fresh_append_flag_if_enabled cmd --skip-hlt-hash-check "${LOCAL_GRAPH_BASELINE_LOGIT_CACHE_SKIP_HLT_HASH_CHECK}"
fresh_append_flag_if_enabled cmd --skip-hlt-params-check "${LOCAL_GRAPH_BASELINE_LOGIT_CACHE_SKIP_HLT_PARAMS_CHECK}"
fresh_append_flag_if_enabled cmd --overwrite "${LOCAL_GRAPH_BASELINE_LOGIT_CACHE_OVERWRITE}"

fresh_write_run_config "${LOCAL_GRAPH_BASELINE_LOGIT_CACHE_OUTPUT_DIR}" "local_graph_baseline_logit_cache" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${LOCAL_GRAPH_BASELINE_LOGIT_CACHE_OUTPUT_DIR}/baseline_logit_manifest.json"
  fresh_require_file "${LOCAL_GRAPH_BASELINE_LOGIT_CACHE_OUTPUT_DIR}/run_report.json"
  for split in "${split_args[@]}"; do
    fresh_require_file "${LOCAL_GRAPH_BASELINE_LOGIT_CACHE_OUTPUT_DIR}/${split}_baseline_logits.npz"
    fresh_require_file "${LOCAL_GRAPH_BASELINE_LOGIT_CACHE_OUTPUT_DIR}/${split}_baseline_logits_metadata.json"
  done
fi
