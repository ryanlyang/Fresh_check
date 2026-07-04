#!/usr/bin/env bash
# Train one AV10 Architecture-View Residual ParT variant.

#SBATCH --job-name=archview10
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=3-00:00:00
#SBATCH --mem=160G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_ENV:=atlas_kd}"
export CONDA_ENV
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

REQUESTED_VARIANT="${1:?Usage: sbatch run_train_architecture_view_10class_part.sh <av10-variant>}"

: "${ARCHITECTURE_VIEW_10CLASS_ROOT:=${OUTPUT_ROOT}/architecture_view_10class_hlt0p6}"
: "${ARCHITECTURE_VIEW_10CLASS_TAGGER_ROOT:=${ARCHITECTURE_VIEW_10CLASS_ROOT}/taggers}"
: "${ARCHITECTURE_VIEW_10CLASS_MANIFEST_PATH:=${ARCHITECTURE_VIEW_10CLASS_ROOT}/inputs/split_manifest.json.gz}"
: "${ARCHITECTURE_VIEW_10CLASS_HLT_CACHE_DIR:=${ARCHITECTURE_VIEW_10CLASS_ROOT}/inputs/hlt_cache}"
: "${ARCHITECTURE_VIEW_10CLASS_BASELINE_CHECKPOINT:?Set ARCHITECTURE_VIEW_10CLASS_BASELINE_CHECKPOINT to the 10-class HLT ParT checkpoint}"
: "${ARCHITECTURE_VIEW_10CLASS_LABEL_NAMES:=QCD Hbb Hcc Hgg H4q Hqql Zqq Wqq Tbqq Tbl}"
: "${ARCHITECTURE_VIEW_10CLASS_LABEL_FILTER_NAMES:=${ARCHITECTURE_VIEW_10CLASS_LABEL_NAMES}}"
: "${ARCHITECTURE_VIEW_10CLASS_SEED:=6207}"
: "${ARCHITECTURE_VIEW_10CLASS_BATCH_SIZE:=64}"
: "${ARCHITECTURE_VIEW_10CLASS_EVAL_BATCH_SIZE:=128}"
: "${ARCHITECTURE_VIEW_10CLASS_EPOCHS:=45}"
: "${ARCHITECTURE_VIEW_10CLASS_ADAPTER_LR:=0.0003}"
: "${ARCHITECTURE_VIEW_10CLASS_PART_LR:=0.00001}"
: "${ARCHITECTURE_VIEW_10CLASS_WEIGHT_DECAY:=0.0001}"
: "${ARCHITECTURE_VIEW_10CLASS_NUM_WORKERS:=${NUM_WORKERS}}"
: "${ARCHITECTURE_VIEW_10CLASS_DEVICE:=${DEVICE}}"
: "${ARCHITECTURE_VIEW_10CLASS_GRAD_CLIP_NORM:=${GRAD_CLIP_NORM}}"
: "${ARCHITECTURE_VIEW_10CLASS_EARLY_STOP_PATIENCE:=6}"
: "${ARCHITECTURE_VIEW_10CLASS_MODEL_TRAIN_SIZE:=500000}"
: "${ARCHITECTURE_VIEW_10CLASS_MODEL_VAL_SIZE:=150000}"
: "${ARCHITECTURE_VIEW_10CLASS_STACK_VAL_SIZE:=150000}"
: "${ARCHITECTURE_VIEW_10CLASS_FINAL_TEST_SIZE:=500000}"
: "${ARCHITECTURE_VIEW_10CLASS_SELECTION_METRIC:=accuracy}"
: "${ARCHITECTURE_VIEW_10CLASS_EXPECTED_HLT_DEGRADATION_STRENGTH:=0.6}"
: "${ARCHITECTURE_VIEW_10CLASS_REQUIRE_BASELINE_SPLIT_MANIFEST_HASH:=1}"
: "${ARCHITECTURE_VIEW_10CLASS_CONFIRM_FINAL_TEST:=1}"
: "${ARCHITECTURE_VIEW_10CLASS_NO_AMP:=0}"
: "${ARCHITECTURE_VIEW_10CLASS_COMPILE_MODEL:=0}"
: "${ARCHITECTURE_VIEW_10CLASS_SKIP_HLT_HASH_CHECK:=0}"
: "${ARCHITECTURE_VIEW_10CLASS_SKIP_HLT_PARAMS_CHECK:=0}"
: "${ARCHITECTURE_VIEW_10CLASS_VIEW_DIM:=32}"
: "${ARCHITECTURE_VIEW_10CLASS_HIDDEN_DIM:=64}"
: "${ARCHITECTURE_VIEW_10CLASS_PN_K:=16}"
: "${ARCHITECTURE_VIEW_10CLASS_PN_LAYERS:=2}"
: "${ARCHITECTURE_VIEW_10CLASS_PFN_HIDDEN_DIM:=64}"
: "${ARCHITECTURE_VIEW_10CLASS_PCNN_CHANNELS:=64}"
: "${ARCHITECTURE_VIEW_10CLASS_PCNN_LAYERS:=2}"
: "${ARCHITECTURE_VIEW_10CLASS_FUSION_HIDDEN_DIM:=96}"
: "${ARCHITECTURE_VIEW_10CLASS_PART_EMBED_DIM:=128}"
: "${ARCHITECTURE_VIEW_10CLASS_DROPOUT:=0.05}"
: "${ARCHITECTURE_VIEW_10CLASS_ATTENTION_DROPOUT:=0.05}"
: "${ARCHITECTURE_VIEW_10CLASS_GATE_BIAS_INIT:=-5.0}"
: "${ARCHITECTURE_VIEW_10CLASS_RANDOM_CONTROL_SEED:=2907}"
: "${ARCHITECTURE_VIEW_10CLASS_DELTA_L2_WEIGHT:=0.0001}"
: "${ARCHITECTURE_VIEW_10CLASS_FREEZE_PART_EPOCHS:=2}"
: "${ARCHITECTURE_VIEW_10CLASS_INPUT_DELTA_SCALE:=1.0}"
: "${ARCHITECTURE_VIEW_10CLASS_DISABLE_FEATURE_WISE_INPUT_DELTA_SCALES:=0}"
: "${ARCHITECTURE_VIEW_10CLASS_FREEZE_INPUT_DELTA_PID:=0}"
: "${ARCHITECTURE_VIEW_10CLASS_FREEZE_INPUT_DELTA_GEOMETRY:=0}"
: "${ARCHITECTURE_VIEW_10CLASS_MAX_TRAIN_BATCHES:=}"
: "${ARCHITECTURE_VIEW_10CLASS_MAX_VAL_BATCHES:=}"
: "${ARCHITECTURE_VIEW_10CLASS_MAX_STACK_VAL_BATCHES:=}"
: "${ARCHITECTURE_VIEW_10CLASS_MAX_FINAL_TEST_BATCHES:=}"

OUTPUT_DIR="${ARCHITECTURE_VIEW_10CLASS_TAGGER_ROOT}/${REQUESTED_VARIANT}"

fresh_setup "$@"
fresh_require_file "scripts/train_architecture_view_part_tagger.py"
fresh_require_file "${ARCHITECTURE_VIEW_10CLASS_MANIFEST_PATH}"
fresh_require_file "${ARCHITECTURE_VIEW_10CLASS_BASELINE_CHECKPOINT}"
for split in model_train model_val stack_val final_test; do
  fresh_require_file "${ARCHITECTURE_VIEW_10CLASS_HLT_CACHE_DIR}/${split}_fixed_hlt.npz"
  fresh_require_file "${ARCHITECTURE_VIEW_10CLASS_HLT_CACHE_DIR}/${split}_fixed_hlt_metadata.json"
done
fresh_claim_new_dir "${OUTPUT_DIR}"
fresh_split_words label_name_args "${ARCHITECTURE_VIEW_10CLASS_LABEL_NAMES}"
fresh_split_words label_filter_args "${ARCHITECTURE_VIEW_10CLASS_LABEL_FILTER_NAMES}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/train_architecture_view_part_tagger.py"
  --output-dir "${OUTPUT_DIR}"
  --manifest-path "${ARCHITECTURE_VIEW_10CLASS_MANIFEST_PATH}"
  --hlt-cache-dir "${ARCHITECTURE_VIEW_10CLASS_HLT_CACHE_DIR}"
  --baseline-checkpoint "${ARCHITECTURE_VIEW_10CLASS_BASELINE_CHECKPOINT}"
  --label-names "${label_name_args[@]}"
  --label-filter-names "${label_filter_args[@]}"
  --num-classes 10
  --train-split model_train
  --val-split model_val
  --stack-val-split stack_val
  --final-test-split final_test
  --confirm-split-settings
  --seed "${ARCHITECTURE_VIEW_10CLASS_SEED}"
  --batch-size "${ARCHITECTURE_VIEW_10CLASS_BATCH_SIZE}"
  --eval-batch-size "${ARCHITECTURE_VIEW_10CLASS_EVAL_BATCH_SIZE}"
  --epochs "${ARCHITECTURE_VIEW_10CLASS_EPOCHS}"
  --adapter-lr "${ARCHITECTURE_VIEW_10CLASS_ADAPTER_LR}"
  --part-lr "${ARCHITECTURE_VIEW_10CLASS_PART_LR}"
  --weight-decay "${ARCHITECTURE_VIEW_10CLASS_WEIGHT_DECAY}"
  --num-workers "${ARCHITECTURE_VIEW_10CLASS_NUM_WORKERS}"
  --device "${ARCHITECTURE_VIEW_10CLASS_DEVICE}"
  --grad-clip-norm "${ARCHITECTURE_VIEW_10CLASS_GRAD_CLIP_NORM}"
  --early-stop-patience "${ARCHITECTURE_VIEW_10CLASS_EARLY_STOP_PATIENCE}"
  --max-train-jets "${ARCHITECTURE_VIEW_10CLASS_MODEL_TRAIN_SIZE}"
  --max-val-jets "${ARCHITECTURE_VIEW_10CLASS_MODEL_VAL_SIZE}"
  --max-stack-val-jets "${ARCHITECTURE_VIEW_10CLASS_STACK_VAL_SIZE}"
  --max-final-test-jets "${ARCHITECTURE_VIEW_10CLASS_FINAL_TEST_SIZE}"
  --selection-metric "${ARCHITECTURE_VIEW_10CLASS_SELECTION_METRIC}"
  --expected-hlt-degradation-strength "${ARCHITECTURE_VIEW_10CLASS_EXPECTED_HLT_DEGRADATION_STRENGTH}"
  --variant "${REQUESTED_VARIANT}"
  --view-dim "${ARCHITECTURE_VIEW_10CLASS_VIEW_DIM}"
  --hidden-dim "${ARCHITECTURE_VIEW_10CLASS_HIDDEN_DIM}"
  --pn-k "${ARCHITECTURE_VIEW_10CLASS_PN_K}"
  --pn-layers "${ARCHITECTURE_VIEW_10CLASS_PN_LAYERS}"
  --pfn-hidden-dim "${ARCHITECTURE_VIEW_10CLASS_PFN_HIDDEN_DIM}"
  --pcnn-channels "${ARCHITECTURE_VIEW_10CLASS_PCNN_CHANNELS}"
  --pcnn-layers "${ARCHITECTURE_VIEW_10CLASS_PCNN_LAYERS}"
  --fusion-hidden-dim "${ARCHITECTURE_VIEW_10CLASS_FUSION_HIDDEN_DIM}"
  --part-embed-dim "${ARCHITECTURE_VIEW_10CLASS_PART_EMBED_DIM}"
  --dropout "${ARCHITECTURE_VIEW_10CLASS_DROPOUT}"
  --attention-dropout "${ARCHITECTURE_VIEW_10CLASS_ATTENTION_DROPOUT}"
  --gate-bias-init "${ARCHITECTURE_VIEW_10CLASS_GATE_BIAS_INIT}"
  --random-control-seed "${ARCHITECTURE_VIEW_10CLASS_RANDOM_CONTROL_SEED}"
  --delta-l2-weight "${ARCHITECTURE_VIEW_10CLASS_DELTA_L2_WEIGHT}"
  --freeze-part-epochs "${ARCHITECTURE_VIEW_10CLASS_FREEZE_PART_EPOCHS}"
  --input-delta-scale "${ARCHITECTURE_VIEW_10CLASS_INPUT_DELTA_SCALE}"
)
fresh_append_flag_if_enabled cmd --confirm-final-test "${ARCHITECTURE_VIEW_10CLASS_CONFIRM_FINAL_TEST}"
fresh_append_flag_if_enabled cmd --no-amp "${ARCHITECTURE_VIEW_10CLASS_NO_AMP}"
fresh_append_flag_if_enabled cmd --compile-model "${ARCHITECTURE_VIEW_10CLASS_COMPILE_MODEL}"
fresh_append_flag_if_enabled cmd --skip-hlt-hash-check "${ARCHITECTURE_VIEW_10CLASS_SKIP_HLT_HASH_CHECK}"
fresh_append_flag_if_enabled cmd --skip-hlt-params-check "${ARCHITECTURE_VIEW_10CLASS_SKIP_HLT_PARAMS_CHECK}"
fresh_append_flag_if_enabled cmd --disable-feature-wise-input-delta-scales "${ARCHITECTURE_VIEW_10CLASS_DISABLE_FEATURE_WISE_INPUT_DELTA_SCALES}"
fresh_append_flag_if_enabled cmd --freeze-input-delta-pid "${ARCHITECTURE_VIEW_10CLASS_FREEZE_INPUT_DELTA_PID}"
fresh_append_flag_if_enabled cmd --freeze-input-delta-geometry "${ARCHITECTURE_VIEW_10CLASS_FREEZE_INPUT_DELTA_GEOMETRY}"
if fresh_bool_enabled "${ARCHITECTURE_VIEW_10CLASS_REQUIRE_BASELINE_SPLIT_MANIFEST_HASH}"; then
  cmd+=(--require-baseline-split-manifest-hash)
else
  cmd+=(--allow-missing-baseline-split-manifest-hash)
fi
fresh_append_optional_arg cmd --max-train-batches "${ARCHITECTURE_VIEW_10CLASS_MAX_TRAIN_BATCHES}"
fresh_append_optional_arg cmd --max-val-batches "${ARCHITECTURE_VIEW_10CLASS_MAX_VAL_BATCHES}"
fresh_append_optional_arg cmd --max-stack-val-batches "${ARCHITECTURE_VIEW_10CLASS_MAX_STACK_VAL_BATCHES}"
fresh_append_optional_arg cmd --max-final-test-batches "${ARCHITECTURE_VIEW_10CLASS_MAX_FINAL_TEST_BATCHES}"

fresh_write_run_config "${OUTPUT_DIR}" "architecture_view_10class_${REQUESTED_VARIANT}" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${OUTPUT_DIR}/best_model_val.pt"
  fresh_require_file "${OUTPUT_DIR}/run_report.json"
  fresh_require_file "${OUTPUT_DIR}/model_val_report.json"
  fresh_require_file "${OUTPUT_DIR}/stack_val_report.json"
  fresh_require_file "${OUTPUT_DIR}/final_test_report.json"
fi
