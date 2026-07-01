#!/usr/bin/env bash
# Train one local-compression feature-adapter HLT ParT variant.

#SBATCH --job-name=localcomp_part
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
: "${CONDA_ENV:=atlas_kd}"
export CONDA_ENV
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

REQUESTED_VARIANT="${1:?Usage: sbatch run_train_local_compression_part.sh <local-compression-variant>}"

: "${LOCAL_COMPRESSION_PART_ROOT:=${OUTPUT_ROOT}/local_compression_part_qcd_hgg_binary_hlt0p6}"
: "${LOCAL_COMPRESSION_PART_TAGGER_ROOT:=${LOCAL_COMPRESSION_PART_ROOT}/taggers}"
: "${LOCAL_COMPRESSION_PART_MANIFEST_PATH:=${LOCAL_COMPRESSION_PART_ROOT}/binary_inputs/split_manifest.json.gz}"
: "${LOCAL_COMPRESSION_PART_HLT_CACHE_DIR:=${LOCAL_COMPRESSION_PART_ROOT}/binary_inputs/hlt_cache}"
: "${LOCAL_COMPRESSION_PART_BASELINE_CHECKPOINT:?Set LOCAL_COMPRESSION_PART_BASELINE_CHECKPOINT to an FPR@50-selected QCD/Hgg HLT0.6 ParT checkpoint}"
: "${LOCAL_COMPRESSION_PART_SEED:=5207}"
: "${LOCAL_COMPRESSION_PART_BATCH_SIZE:=64}"
: "${LOCAL_COMPRESSION_PART_EVAL_BATCH_SIZE:=128}"
: "${LOCAL_COMPRESSION_PART_EPOCHS:=45}"
: "${LOCAL_COMPRESSION_PART_ADAPTER_LR:=0.0003}"
: "${LOCAL_COMPRESSION_PART_PART_LR:=0.00003}"
: "${LOCAL_COMPRESSION_PART_WEIGHT_DECAY:=0.0001}"
: "${LOCAL_COMPRESSION_PART_NUM_WORKERS:=${NUM_WORKERS}}"
: "${LOCAL_COMPRESSION_PART_DEVICE:=${DEVICE}}"
: "${LOCAL_COMPRESSION_PART_GRAD_CLIP_NORM:=${GRAD_CLIP_NORM}}"
: "${LOCAL_COMPRESSION_PART_EARLY_STOP_PATIENCE:=6}"
: "${LOCAL_COMPRESSION_PART_MAX_TRAIN_BATCHES:=}"
: "${LOCAL_COMPRESSION_PART_MAX_VAL_BATCHES:=}"
: "${LOCAL_COMPRESSION_PART_MAX_STACK_VAL_BATCHES:=}"
: "${LOCAL_COMPRESSION_PART_MAX_FINAL_TEST_BATCHES:=}"
: "${LOCAL_COMPRESSION_PART_MODEL_TRAIN_SIZE:=500000}"
: "${LOCAL_COMPRESSION_PART_MODEL_VAL_SIZE:=150000}"
: "${LOCAL_COMPRESSION_PART_STACK_VAL_SIZE:=150000}"
: "${LOCAL_COMPRESSION_PART_FINAL_TEST_SIZE:=500000}"
: "${LOCAL_COMPRESSION_PART_SELECTION_METRIC:=fpr_at_signal_eff_0p50}"
: "${LOCAL_COMPRESSION_PART_EXPECTED_HLT_DEGRADATION_STRENGTH:=0.6}"
: "${LOCAL_COMPRESSION_PART_LABEL_NAMES:=QCD Hgg}"
: "${LOCAL_COMPRESSION_PART_LABEL_FILTER_NAMES:=QCD Hgg}"
: "${LOCAL_COMPRESSION_PART_NO_AMP:=0}"
: "${LOCAL_COMPRESSION_PART_COMPILE_MODEL:=0}"
: "${LOCAL_COMPRESSION_PART_SKIP_HLT_HASH_CHECK:=0}"
: "${LOCAL_COMPRESSION_PART_SKIP_HLT_PARAMS_CHECK:=0}"
: "${LOCAL_COMPRESSION_PART_REQUIRE_BASELINE_SPLIT_MANIFEST_HASH:=1}"
: "${LOCAL_COMPRESSION_PART_CONFIRM_FINAL_TEST:=1}"
: "${LOCAL_COMPRESSION_PART_EMBED_DIM:=96}"
: "${LOCAL_COMPRESSION_PART_LOCAL_LAYERS:=2}"
: "${LOCAL_COMPRESSION_PART_LOCAL_HEADS:=4}"
: "${LOCAL_COMPRESSION_PART_CONTEXT_LAYERS:=1}"
: "${LOCAL_COMPRESSION_PART_CONTEXT_HEADS:=4}"
: "${LOCAL_COMPRESSION_PART_MLP_RATIO:=2.0}"
: "${LOCAL_COMPRESSION_PART_DROPOUT:=0.05}"
: "${LOCAL_COMPRESSION_PART_ATTENTION_DROPOUT:=0.05}"
: "${LOCAL_COMPRESSION_PART_POOL_MODE:=learned_query}"
: "${LOCAL_COMPRESSION_PART_GATE_MODE:=}"
: "${LOCAL_COMPRESSION_PART_DELTA_SCALE:=1.0}"
: "${LOCAL_COMPRESSION_PART_FREEZE_PID_DELTAS:=0}"
: "${LOCAL_COMPRESSION_PART_FREEZE_GEOMETRY_DELTAS:=0}"
: "${LOCAL_COMPRESSION_PART_FREEZE_PART_EPOCHS:=1}"
: "${LOCAL_COMPRESSION_PART_RANDOM_GROUPING_SEED:=2907}"

OUTPUT_DIR="${LOCAL_COMPRESSION_PART_TAGGER_ROOT}/${REQUESTED_VARIANT}"

fresh_setup "$@"
fresh_require_file "scripts/train_local_compression_part_tagger.py"
fresh_require_file "${LOCAL_COMPRESSION_PART_MANIFEST_PATH}"
fresh_require_file "${LOCAL_COMPRESSION_PART_BASELINE_CHECKPOINT}"
for split in model_train model_val stack_val final_test; do
  fresh_require_file "${LOCAL_COMPRESSION_PART_HLT_CACHE_DIR}/${split}_fixed_hlt.npz"
  fresh_require_file "${LOCAL_COMPRESSION_PART_HLT_CACHE_DIR}/${split}_fixed_hlt_metadata.json"
done
fresh_claim_new_dir "${OUTPUT_DIR}"
fresh_split_words label_name_args "${LOCAL_COMPRESSION_PART_LABEL_NAMES}"
fresh_split_words label_filter_args "${LOCAL_COMPRESSION_PART_LABEL_FILTER_NAMES}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/train_local_compression_part_tagger.py"
  --output-dir "${OUTPUT_DIR}"
  --manifest-path "${LOCAL_COMPRESSION_PART_MANIFEST_PATH}"
  --hlt-cache-dir "${LOCAL_COMPRESSION_PART_HLT_CACHE_DIR}"
  --baseline-checkpoint "${LOCAL_COMPRESSION_PART_BASELINE_CHECKPOINT}"
  --label-names "${label_name_args[@]}"
  --label-filter-names "${label_filter_args[@]}"
  --train-split model_train
  --val-split model_val
  --stack-val-split stack_val
  --final-test-split final_test
  --confirm-split-settings
  --seed "${LOCAL_COMPRESSION_PART_SEED}"
  --batch-size "${LOCAL_COMPRESSION_PART_BATCH_SIZE}"
  --eval-batch-size "${LOCAL_COMPRESSION_PART_EVAL_BATCH_SIZE}"
  --epochs "${LOCAL_COMPRESSION_PART_EPOCHS}"
  --adapter-lr "${LOCAL_COMPRESSION_PART_ADAPTER_LR}"
  --part-lr "${LOCAL_COMPRESSION_PART_PART_LR}"
  --weight-decay "${LOCAL_COMPRESSION_PART_WEIGHT_DECAY}"
  --num-workers "${LOCAL_COMPRESSION_PART_NUM_WORKERS}"
  --device "${LOCAL_COMPRESSION_PART_DEVICE}"
  --grad-clip-norm "${LOCAL_COMPRESSION_PART_GRAD_CLIP_NORM}"
  --early-stop-patience "${LOCAL_COMPRESSION_PART_EARLY_STOP_PATIENCE}"
  --max-train-jets "${LOCAL_COMPRESSION_PART_MODEL_TRAIN_SIZE}"
  --max-val-jets "${LOCAL_COMPRESSION_PART_MODEL_VAL_SIZE}"
  --max-stack-val-jets "${LOCAL_COMPRESSION_PART_STACK_VAL_SIZE}"
  --max-final-test-jets "${LOCAL_COMPRESSION_PART_FINAL_TEST_SIZE}"
  --selection-metric "${LOCAL_COMPRESSION_PART_SELECTION_METRIC}"
  --expected-hlt-degradation-strength "${LOCAL_COMPRESSION_PART_EXPECTED_HLT_DEGRADATION_STRENGTH}"
  --variant "${REQUESTED_VARIANT}"
  --random-grouping-seed "${LOCAL_COMPRESSION_PART_RANDOM_GROUPING_SEED}"
  --embed-dim "${LOCAL_COMPRESSION_PART_EMBED_DIM}"
  --local-layers "${LOCAL_COMPRESSION_PART_LOCAL_LAYERS}"
  --local-heads "${LOCAL_COMPRESSION_PART_LOCAL_HEADS}"
  --context-layers "${LOCAL_COMPRESSION_PART_CONTEXT_LAYERS}"
  --context-heads "${LOCAL_COMPRESSION_PART_CONTEXT_HEADS}"
  --mlp-ratio "${LOCAL_COMPRESSION_PART_MLP_RATIO}"
  --dropout "${LOCAL_COMPRESSION_PART_DROPOUT}"
  --attention-dropout "${LOCAL_COMPRESSION_PART_ATTENTION_DROPOUT}"
  --pool-mode "${LOCAL_COMPRESSION_PART_POOL_MODE}"
  --delta-scale "${LOCAL_COMPRESSION_PART_DELTA_SCALE}"
  --freeze-part-epochs "${LOCAL_COMPRESSION_PART_FREEZE_PART_EPOCHS}"
)
fresh_append_flag_if_enabled cmd --confirm-final-test "${LOCAL_COMPRESSION_PART_CONFIRM_FINAL_TEST}"
fresh_append_flag_if_enabled cmd --no-amp "${LOCAL_COMPRESSION_PART_NO_AMP}"
fresh_append_flag_if_enabled cmd --compile-model "${LOCAL_COMPRESSION_PART_COMPILE_MODEL}"
fresh_append_flag_if_enabled cmd --skip-hlt-hash-check "${LOCAL_COMPRESSION_PART_SKIP_HLT_HASH_CHECK}"
fresh_append_flag_if_enabled cmd --skip-hlt-params-check "${LOCAL_COMPRESSION_PART_SKIP_HLT_PARAMS_CHECK}"
if fresh_bool_enabled "${LOCAL_COMPRESSION_PART_REQUIRE_BASELINE_SPLIT_MANIFEST_HASH}"; then
  cmd+=(--require-baseline-split-manifest-hash)
else
  cmd+=(--allow-missing-baseline-split-manifest-hash)
fi
fresh_append_flag_if_enabled cmd --freeze-pid-deltas "${LOCAL_COMPRESSION_PART_FREEZE_PID_DELTAS}"
fresh_append_flag_if_enabled cmd --freeze-geometry-deltas "${LOCAL_COMPRESSION_PART_FREEZE_GEOMETRY_DELTAS}"
fresh_append_optional_arg cmd --max-train-batches "${LOCAL_COMPRESSION_PART_MAX_TRAIN_BATCHES}"
fresh_append_optional_arg cmd --max-val-batches "${LOCAL_COMPRESSION_PART_MAX_VAL_BATCHES}"
fresh_append_optional_arg cmd --max-stack-val-batches "${LOCAL_COMPRESSION_PART_MAX_STACK_VAL_BATCHES}"
fresh_append_optional_arg cmd --max-final-test-batches "${LOCAL_COMPRESSION_PART_MAX_FINAL_TEST_BATCHES}"
fresh_append_optional_arg cmd --gate-mode "${LOCAL_COMPRESSION_PART_GATE_MODE}"

fresh_write_run_config "${OUTPUT_DIR}" "local_compression_part_${REQUESTED_VARIANT}" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${OUTPUT_DIR}/best_model_val.pt"
  fresh_require_file "${OUTPUT_DIR}/last.pt"
  fresh_require_file "${OUTPUT_DIR}/training_curves.json"
  fresh_require_file "${OUTPUT_DIR}/config.json"
  fresh_require_file "${OUTPUT_DIR}/model_val_report.json"
  fresh_require_file "${OUTPUT_DIR}/run_report.json"
  fresh_require_file "${OUTPUT_DIR}/diagnostics/baseline_load_report.json"
  fresh_require_file "${OUTPUT_DIR}/diagnostics/init_logit_diff_vs_baseline.json"
  fresh_require_file "${OUTPUT_DIR}/diagnostics/epoch_metrics.csv"
fi
