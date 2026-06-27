#!/usr/bin/env bash
# Train one local-graph HLT ParT comparison variant.

#SBATCH --job-name=localgraph_part
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=2-12:00:00
#SBATCH --mem=160G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

REQUESTED_VARIANT="${1:?Usage: sbatch run_train_local_graph_part_tagger.sh <hlt_part_baseline|local_edgeconv_adapter|local_point_attention_adapter|local_point_attention_adapter_warmstart>}"

: "${LOCAL_GRAPH_PART_ROOT:=${OUTPUT_ROOT}/local_graph_part_qcd_hgg_binary_hlt0p6}"
: "${LOCAL_GRAPH_PART_TAGGER_ROOT:=${LOCAL_GRAPH_PART_ROOT}/taggers}"
: "${LOCAL_GRAPH_PART_HLT_CACHE_DIR:=${LOCAL_GRAPH_PART_ROOT}/binary_inputs/hlt_cache}"
: "${LOCAL_GRAPH_PART_SEED:=3107}"
: "${LOCAL_GRAPH_PART_BATCH_SIZE:=64}"
: "${LOCAL_GRAPH_PART_EVAL_BATCH_SIZE:=128}"
: "${LOCAL_GRAPH_PART_EPOCHS:=45}"
: "${LOCAL_GRAPH_PART_LR:=0.0003}"
: "${LOCAL_GRAPH_PART_WEIGHT_DECAY:=0.0001}"
: "${LOCAL_GRAPH_PART_NUM_WORKERS:=${NUM_WORKERS}}"
: "${LOCAL_GRAPH_PART_DEVICE:=${DEVICE}}"
: "${LOCAL_GRAPH_PART_GRAD_CLIP_NORM:=${GRAD_CLIP_NORM}}"
: "${LOCAL_GRAPH_PART_EARLY_STOP_PATIENCE:=6}"
: "${LOCAL_GRAPH_PART_MAX_TRAIN_BATCHES:=}"
: "${LOCAL_GRAPH_PART_MAX_VAL_BATCHES:=}"
: "${LOCAL_GRAPH_PART_MAX_STACK_VAL_BATCHES:=}"
: "${LOCAL_GRAPH_PART_MAX_FINAL_TEST_BATCHES:=}"
: "${LOCAL_GRAPH_PART_MODEL_TRAIN_SIZE:=500000}"
: "${LOCAL_GRAPH_PART_MODEL_VAL_SIZE:=150000}"
: "${LOCAL_GRAPH_PART_STACK_VAL_SIZE:=150000}"
: "${LOCAL_GRAPH_PART_FINAL_TEST_SIZE:=500000}"
: "${LOCAL_GRAPH_PART_SELECTION_METRIC:=fpr_at_signal_eff_0p50}"
: "${LOCAL_GRAPH_PART_MODEL_SIZE:=base}"
: "${LOCAL_GRAPH_PART_MAX_CONSTITS:=128}"
: "${LOCAL_GRAPH_PART_K:=16}"
: "${LOCAL_GRAPH_PART_LOCAL_EMBED_DIM:=128}"
: "${LOCAL_GRAPH_PART_LOCAL_HEADS:=8}"
: "${LOCAL_GRAPH_PART_LOCAL_HIDDEN_DIM:=}"
: "${LOCAL_GRAPH_PART_DROPOUT:=0.05}"
: "${LOCAL_GRAPH_PART_ATTENTION_DROPOUT:=0.05}"
: "${LOCAL_GRAPH_PART_RESIDUAL_GAMMA_INIT:=}"
: "${LOCAL_GRAPH_PART_WARM_START_RESIDUAL_GAMMA_INIT:=0.01}"
: "${LOCAL_GRAPH_PART_WEIGHT_THRESHOLD:=0.0}"
: "${LOCAL_GRAPH_PART_NO_AMP:=0}"
: "${LOCAL_GRAPH_PART_COMPILE_MODEL:=0}"
: "${LOCAL_GRAPH_PART_SKIP_HLT_HASH_CHECK:=0}"
: "${LOCAL_GRAPH_PART_SKIP_HLT_PARAMS_CHECK:=0}"
: "${LOCAL_GRAPH_PART_EXPECTED_HLT_DEGRADATION_STRENGTH:=0.6}"
: "${LOCAL_GRAPH_PART_CONFIRM_FINAL_TEST:=1}"
: "${LOCAL_GRAPH_PART_WARM_START_CHECKPOINT:=${LOCAL_GRAPH_PART_TAGGER_ROOT}/hlt_part_baseline/best_model_val.pt}"
: "${LOCAL_GRAPH_PART_WARM_START_FREEZE_EPOCHS:=0}"

TRAIN_VARIANT="${REQUESTED_VARIANT}"
OUTPUT_VARIANT="${REQUESTED_VARIANT}"
WARM_START_ENABLED=0
case "${REQUESTED_VARIANT}" in
  hlt_part_baseline|local_edgeconv_adapter|local_point_attention_adapter)
    ;;
  local_point_attention_adapter_warmstart)
    TRAIN_VARIANT="local_point_attention_adapter"
    WARM_START_ENABLED=1
    ;;
  *)
    echo "Unknown local graph variant: ${REQUESTED_VARIANT}" >&2
    exit 2
    ;;
esac

OUTPUT_DIR="${LOCAL_GRAPH_PART_TAGGER_ROOT}/${OUTPUT_VARIANT}"
if fresh_bool_enabled "${WARM_START_ENABLED}"; then
  : "${LOCAL_GRAPH_PART_RESIDUAL_GAMMA_INIT:=${LOCAL_GRAPH_PART_WARM_START_RESIDUAL_GAMMA_INIT}}"
else
  : "${LOCAL_GRAPH_PART_RESIDUAL_GAMMA_INIT:=0.0}"
fi

fresh_setup "$@"
fresh_require_file "scripts/train_local_graph_part_tagger.py"
for split in model_train model_val stack_val final_test; do
  fresh_require_file "${LOCAL_GRAPH_PART_HLT_CACHE_DIR}/${split}_fixed_hlt.npz"
  fresh_require_file "${LOCAL_GRAPH_PART_HLT_CACHE_DIR}/${split}_fixed_hlt_metadata.json"
done
if fresh_bool_enabled "${WARM_START_ENABLED}"; then
  fresh_require_file "${LOCAL_GRAPH_PART_WARM_START_CHECKPOINT}"
fi
fresh_claim_new_dir "${OUTPUT_DIR}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/train_local_graph_part_tagger.py"
  --output-dir "${OUTPUT_DIR}"
  --hlt-cache-dir "${LOCAL_GRAPH_PART_HLT_CACHE_DIR}"
  --variant "${TRAIN_VARIANT}"
  --train-split model_train
  --val-split model_val
  --stack-val-split stack_val
  --final-test-split final_test
  --confirm-split-settings
  --seed "${LOCAL_GRAPH_PART_SEED}"
  --batch-size "${LOCAL_GRAPH_PART_BATCH_SIZE}"
  --eval-batch-size "${LOCAL_GRAPH_PART_EVAL_BATCH_SIZE}"
  --epochs "${LOCAL_GRAPH_PART_EPOCHS}"
  --lr "${LOCAL_GRAPH_PART_LR}"
  --weight-decay "${LOCAL_GRAPH_PART_WEIGHT_DECAY}"
  --num-workers "${LOCAL_GRAPH_PART_NUM_WORKERS}"
  --device "${LOCAL_GRAPH_PART_DEVICE}"
  --grad-clip-norm "${LOCAL_GRAPH_PART_GRAD_CLIP_NORM}"
  --early-stop-patience "${LOCAL_GRAPH_PART_EARLY_STOP_PATIENCE}"
  --max-train-jets "${LOCAL_GRAPH_PART_MODEL_TRAIN_SIZE}"
  --max-val-jets "${LOCAL_GRAPH_PART_MODEL_VAL_SIZE}"
  --max-stack-val-jets "${LOCAL_GRAPH_PART_STACK_VAL_SIZE}"
  --max-final-test-jets "${LOCAL_GRAPH_PART_FINAL_TEST_SIZE}"
  --selection-metric "${LOCAL_GRAPH_PART_SELECTION_METRIC}"
  --expected-hlt-degradation-strength "${LOCAL_GRAPH_PART_EXPECTED_HLT_DEGRADATION_STRENGTH}"
  --model-size "${LOCAL_GRAPH_PART_MODEL_SIZE}"
  --max-constits "${LOCAL_GRAPH_PART_MAX_CONSTITS}"
  --k "${LOCAL_GRAPH_PART_K}"
  --local-embed-dim "${LOCAL_GRAPH_PART_LOCAL_EMBED_DIM}"
  --local-heads "${LOCAL_GRAPH_PART_LOCAL_HEADS}"
  --dropout "${LOCAL_GRAPH_PART_DROPOUT}"
  --attention-dropout "${LOCAL_GRAPH_PART_ATTENTION_DROPOUT}"
  --residual-gamma-init "${LOCAL_GRAPH_PART_RESIDUAL_GAMMA_INIT}"
  --weight-threshold "${LOCAL_GRAPH_PART_WEIGHT_THRESHOLD}"
)
fresh_append_flag_if_enabled cmd --confirm-final-test "${LOCAL_GRAPH_PART_CONFIRM_FINAL_TEST}"
fresh_append_flag_if_enabled cmd --no-amp "${LOCAL_GRAPH_PART_NO_AMP}"
fresh_append_flag_if_enabled cmd --compile-model "${LOCAL_GRAPH_PART_COMPILE_MODEL}"
fresh_append_flag_if_enabled cmd --skip-hlt-hash-check "${LOCAL_GRAPH_PART_SKIP_HLT_HASH_CHECK}"
fresh_append_flag_if_enabled cmd --skip-hlt-params-check "${LOCAL_GRAPH_PART_SKIP_HLT_PARAMS_CHECK}"
fresh_append_optional_arg cmd --local-hidden-dim "${LOCAL_GRAPH_PART_LOCAL_HIDDEN_DIM}"
fresh_append_optional_arg cmd --max-train-batches "${LOCAL_GRAPH_PART_MAX_TRAIN_BATCHES}"
fresh_append_optional_arg cmd --max-val-batches "${LOCAL_GRAPH_PART_MAX_VAL_BATCHES}"
fresh_append_optional_arg cmd --max-stack-val-batches "${LOCAL_GRAPH_PART_MAX_STACK_VAL_BATCHES}"
fresh_append_optional_arg cmd --max-final-test-batches "${LOCAL_GRAPH_PART_MAX_FINAL_TEST_BATCHES}"
if fresh_bool_enabled "${WARM_START_ENABLED}"; then
  cmd+=(
    --warm-start-checkpoint "${LOCAL_GRAPH_PART_WARM_START_CHECKPOINT}"
    --require-warm-start
    --freeze-part-epochs "${LOCAL_GRAPH_PART_WARM_START_FREEZE_EPOCHS}"
  )
fi

fresh_write_run_config "${OUTPUT_DIR}" "local_graph_part_${OUTPUT_VARIANT}" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${OUTPUT_DIR}/best_model_val.pt"
  fresh_require_file "${OUTPUT_DIR}/last.pt"
  fresh_require_file "${OUTPUT_DIR}/training_curves.json"
  fresh_require_file "${OUTPUT_DIR}/config.json"
  fresh_require_file "${OUTPUT_DIR}/model_val_report.json"
  fresh_require_file "${OUTPUT_DIR}/run_report.json"
  fresh_require_file "${OUTPUT_DIR}/diagnostics/summary_metrics.csv"
  fresh_require_file "${OUTPUT_DIR}/diagnostics/best_metrics.csv"
  if fresh_bool_enabled "${WARM_START_ENABLED}"; then
    fresh_require_file "${OUTPUT_DIR}/diagnostics/warm_start_report.json"
  fi
fi
