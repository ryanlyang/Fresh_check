#!/usr/bin/env bash
# Train one frozen-HLT-ParT architecture residual expert.

#SBATCH --job-name=archres_part
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

REQUESTED_ARCH="${1:?Usage: sbatch run_train_arch_residual_part.sh <pfn|pcnn|pn>}"

: "${ARCH_RESIDUAL_PART_ROOT:=${OUTPUT_ROOT}/arch_residual_part_qcd_hgg_binary_hlt0p6_500k}"
: "${ARCH_RESIDUAL_PART_TAGGER_ROOT:=${ARCH_RESIDUAL_PART_ROOT}/taggers}"
: "${ARCH_RESIDUAL_PART_MANIFEST_PATH:=${ARCH_RESIDUAL_PART_ROOT}/binary_inputs/split_manifest.json.gz}"
: "${ARCH_RESIDUAL_PART_HLT_CACHE_DIR:=${ARCH_RESIDUAL_PART_ROOT}/binary_inputs/hlt_cache}"
: "${ARCH_RESIDUAL_PART_BASELINE_CHECKPOINT:?Set ARCH_RESIDUAL_PART_BASELINE_CHECKPOINT to the frozen HLT ParT baseline}"
: "${ARCH_RESIDUAL_PART_SEED:=7307}"
: "${ARCH_RESIDUAL_PART_BATCH_SIZE:=64}"
: "${ARCH_RESIDUAL_PART_EVAL_BATCH_SIZE:=128}"
: "${ARCH_RESIDUAL_PART_EPOCHS:=30}"
: "${ARCH_RESIDUAL_PART_LR:=0.0003}"
: "${ARCH_RESIDUAL_PART_WEIGHT_DECAY:=0.0001}"
: "${ARCH_RESIDUAL_PART_NUM_WORKERS:=${NUM_WORKERS}}"
: "${ARCH_RESIDUAL_PART_DEVICE:=${DEVICE}}"
: "${ARCH_RESIDUAL_PART_GRAD_CLIP_NORM:=${GRAD_CLIP_NORM}}"
: "${ARCH_RESIDUAL_PART_EARLY_STOP_PATIENCE:=6}"
: "${ARCH_RESIDUAL_PART_MAX_TRAIN_BATCHES:=}"
: "${ARCH_RESIDUAL_PART_MAX_VAL_BATCHES:=}"
: "${ARCH_RESIDUAL_PART_MAX_STACK_VAL_BATCHES:=}"
: "${ARCH_RESIDUAL_PART_MAX_FINAL_TEST_BATCHES:=}"
: "${ARCH_RESIDUAL_PART_MODEL_TRAIN_SIZE:=500000}"
: "${ARCH_RESIDUAL_PART_MODEL_VAL_SIZE:=150000}"
: "${ARCH_RESIDUAL_PART_STACK_VAL_SIZE:=150000}"
: "${ARCH_RESIDUAL_PART_FINAL_TEST_SIZE:=500000}"
: "${ARCH_RESIDUAL_PART_SELECTION_METRIC:=fpr_at_signal_eff_0p50}"
: "${ARCH_RESIDUAL_PART_EXPECTED_HLT_DEGRADATION_STRENGTH:=0.6}"
: "${ARCH_RESIDUAL_PART_LABEL_NAMES:=QCD Hgg}"
: "${ARCH_RESIDUAL_PART_LABEL_FILTER_NAMES:=QCD Hgg}"
: "${ARCH_RESIDUAL_PART_NO_AMP:=0}"
: "${ARCH_RESIDUAL_PART_SKIP_HLT_HASH_CHECK:=0}"
: "${ARCH_RESIDUAL_PART_SKIP_HLT_PARAMS_CHECK:=0}"
: "${ARCH_RESIDUAL_PART_REQUIRE_BASELINE_SPLIT_MANIFEST_HASH:=1}"
: "${ARCH_RESIDUAL_PART_CONFIRM_FINAL_TEST:=1}"
: "${ARCH_RESIDUAL_PART_HIDDEN_DIM:=128}"
: "${ARCH_RESIDUAL_PART_PARTICLE_LAYERS:=3}"
: "${ARCH_RESIDUAL_PART_GLOBAL_LAYERS:=2}"
: "${ARCH_RESIDUAL_PART_EDGE_K:=16}"
: "${ARCH_RESIDUAL_PART_DROPOUT:=0.05}"
: "${ARCH_RESIDUAL_PART_NO_BASELINE_CONDITIONING:=0}"
: "${ARCH_RESIDUAL_PART_GAMMA_INIT:=1.0}"
: "${ARCH_RESIDUAL_PART_RESIDUAL_SCALE:=1.0}"
: "${ARCH_RESIDUAL_PART_RESIDUAL_L2_WEIGHT:=0.0001}"

OUTPUT_DIR="${ARCH_RESIDUAL_PART_TAGGER_ROOT}/${REQUESTED_ARCH}_residual"

fresh_setup "$@"
fresh_require_file "scripts/train_arch_residual_part_tagger.py"
fresh_require_file "${ARCH_RESIDUAL_PART_MANIFEST_PATH}"
fresh_require_file "${ARCH_RESIDUAL_PART_BASELINE_CHECKPOINT}"
for split in model_train model_val stack_val final_test; do
  fresh_require_file "${ARCH_RESIDUAL_PART_HLT_CACHE_DIR}/${split}_fixed_hlt.npz"
  fresh_require_file "${ARCH_RESIDUAL_PART_HLT_CACHE_DIR}/${split}_fixed_hlt_metadata.json"
done
fresh_claim_new_dir "${OUTPUT_DIR}"
fresh_split_words label_name_args "${ARCH_RESIDUAL_PART_LABEL_NAMES}"
fresh_split_words label_filter_args "${ARCH_RESIDUAL_PART_LABEL_FILTER_NAMES}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/train_arch_residual_part_tagger.py"
  --output-dir "${OUTPUT_DIR}"
  --manifest-path "${ARCH_RESIDUAL_PART_MANIFEST_PATH}"
  --hlt-cache-dir "${ARCH_RESIDUAL_PART_HLT_CACHE_DIR}"
  --baseline-checkpoint "${ARCH_RESIDUAL_PART_BASELINE_CHECKPOINT}"
  --label-names "${label_name_args[@]}"
  --label-filter-names "${label_filter_args[@]}"
  --train-split model_train
  --val-split model_val
  --stack-val-split stack_val
  --final-test-split final_test
  --confirm-split-settings
  --seed "${ARCH_RESIDUAL_PART_SEED}"
  --batch-size "${ARCH_RESIDUAL_PART_BATCH_SIZE}"
  --eval-batch-size "${ARCH_RESIDUAL_PART_EVAL_BATCH_SIZE}"
  --epochs "${ARCH_RESIDUAL_PART_EPOCHS}"
  --lr "${ARCH_RESIDUAL_PART_LR}"
  --weight-decay "${ARCH_RESIDUAL_PART_WEIGHT_DECAY}"
  --num-workers "${ARCH_RESIDUAL_PART_NUM_WORKERS}"
  --device "${ARCH_RESIDUAL_PART_DEVICE}"
  --grad-clip-norm "${ARCH_RESIDUAL_PART_GRAD_CLIP_NORM}"
  --early-stop-patience "${ARCH_RESIDUAL_PART_EARLY_STOP_PATIENCE}"
  --max-train-jets "${ARCH_RESIDUAL_PART_MODEL_TRAIN_SIZE}"
  --max-val-jets "${ARCH_RESIDUAL_PART_MODEL_VAL_SIZE}"
  --max-stack-val-jets "${ARCH_RESIDUAL_PART_STACK_VAL_SIZE}"
  --max-final-test-jets "${ARCH_RESIDUAL_PART_FINAL_TEST_SIZE}"
  --selection-metric "${ARCH_RESIDUAL_PART_SELECTION_METRIC}"
  --expected-hlt-degradation-strength "${ARCH_RESIDUAL_PART_EXPECTED_HLT_DEGRADATION_STRENGTH}"
  --architecture "${REQUESTED_ARCH}"
  --hidden-dim "${ARCH_RESIDUAL_PART_HIDDEN_DIM}"
  --particle-layers "${ARCH_RESIDUAL_PART_PARTICLE_LAYERS}"
  --global-layers "${ARCH_RESIDUAL_PART_GLOBAL_LAYERS}"
  --edge-k "${ARCH_RESIDUAL_PART_EDGE_K}"
  --dropout "${ARCH_RESIDUAL_PART_DROPOUT}"
  --gamma-init "${ARCH_RESIDUAL_PART_GAMMA_INIT}"
  --residual-scale "${ARCH_RESIDUAL_PART_RESIDUAL_SCALE}"
  --residual-l2-weight "${ARCH_RESIDUAL_PART_RESIDUAL_L2_WEIGHT}"
)
fresh_append_flag_if_enabled cmd --confirm-final-test "${ARCH_RESIDUAL_PART_CONFIRM_FINAL_TEST}"
fresh_append_flag_if_enabled cmd --no-amp "${ARCH_RESIDUAL_PART_NO_AMP}"
fresh_append_flag_if_enabled cmd --skip-hlt-hash-check "${ARCH_RESIDUAL_PART_SKIP_HLT_HASH_CHECK}"
fresh_append_flag_if_enabled cmd --skip-hlt-params-check "${ARCH_RESIDUAL_PART_SKIP_HLT_PARAMS_CHECK}"
if fresh_bool_enabled "${ARCH_RESIDUAL_PART_REQUIRE_BASELINE_SPLIT_MANIFEST_HASH}"; then
  cmd+=(--require-baseline-split-manifest-hash)
else
  cmd+=(--allow-missing-baseline-split-manifest-hash)
fi
fresh_append_flag_if_enabled cmd --no-baseline-conditioning "${ARCH_RESIDUAL_PART_NO_BASELINE_CONDITIONING}"
fresh_append_optional_arg cmd --max-train-batches "${ARCH_RESIDUAL_PART_MAX_TRAIN_BATCHES}"
fresh_append_optional_arg cmd --max-val-batches "${ARCH_RESIDUAL_PART_MAX_VAL_BATCHES}"
fresh_append_optional_arg cmd --max-stack-val-batches "${ARCH_RESIDUAL_PART_MAX_STACK_VAL_BATCHES}"
fresh_append_optional_arg cmd --max-final-test-batches "${ARCH_RESIDUAL_PART_MAX_FINAL_TEST_BATCHES}"

fresh_write_run_config "${OUTPUT_DIR}" "arch_residual_part_${REQUESTED_ARCH}" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${OUTPUT_DIR}/best_model_val.pt"
  fresh_require_file "${OUTPUT_DIR}/last.pt"
  fresh_require_file "${OUTPUT_DIR}/training_curves.json"
  fresh_require_file "${OUTPUT_DIR}/config.json"
  fresh_require_file "${OUTPUT_DIR}/model_val_report.json"
  fresh_require_file "${OUTPUT_DIR}/run_report.json"
  fresh_require_file "${OUTPUT_DIR}/diagnostics/baseline_load_report.json"
  fresh_require_file "${OUTPUT_DIR}/diagnostics/epoch_metrics.csv"
fi
