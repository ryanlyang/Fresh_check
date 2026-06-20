#!/usr/bin/env bash
# Train/evaluate a fresh offline-only ParT reference for the Hbb-vs-QCD set-matching split.

#SBATCH --job-name=offline_bin_ref
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=1-00:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${BINARY_OFFLINE_TEACHER_TAG:=$(date +%Y%m%d_%H%M%S)}"
: "${BINARY_OFFLINE_TEACHER_DIR:=${SET_MATCHING_ROOT}/offline_teacher_reference/fresh_binary_part_${BINARY_OFFLINE_TEACHER_TAG}}"
: "${BINARY_OFFLINE_TEACHER_SEED:=2405}"
: "${BINARY_OFFLINE_TEACHER_MODEL_SIZE:=base}"
: "${BINARY_OFFLINE_TEACHER_EPOCHS:=30}"
: "${BINARY_OFFLINE_TEACHER_BATCH_SIZE:=64}"
: "${BINARY_OFFLINE_TEACHER_EVAL_BATCH_SIZE:=128}"
: "${BINARY_OFFLINE_TEACHER_LR:=0.0003}"
: "${BINARY_OFFLINE_TEACHER_WEIGHT_DECAY:=0.0001}"
: "${BINARY_OFFLINE_TEACHER_EARLY_STOP_PATIENCE:=5}"
: "${BINARY_OFFLINE_TEACHER_NUM_WORKERS:=2}"
: "${BINARY_OFFLINE_TEACHER_DEVICE:=${DEVICE}}"
: "${BINARY_OFFLINE_TEACHER_MAX_TRAIN_JETS:=${SET_MATCHING_MODEL_TRAIN_SIZE}}"
: "${BINARY_OFFLINE_TEACHER_MAX_VAL_JETS:=${SET_MATCHING_MODEL_VAL_SIZE}}"
: "${BINARY_OFFLINE_TEACHER_MAX_STACK_VAL_JETS:=${SET_MATCHING_STACK_VAL_SIZE}}"
: "${BINARY_OFFLINE_TEACHER_MAX_FINAL_TEST_JETS:=${SET_MATCHING_FINAL_TEST_SIZE}}"
: "${BINARY_OFFLINE_TEACHER_MAX_TRAIN_BATCHES:=}"
: "${BINARY_OFFLINE_TEACHER_MAX_VAL_BATCHES:=}"
: "${BINARY_OFFLINE_TEACHER_MAX_EVAL_BATCHES:=}"
: "${BINARY_OFFLINE_TEACHER_NO_AMP:=0}"
: "${BINARY_OFFLINE_TEACHER_COMPILE_MODEL:=0}"
: "${VERIFY_LABEL_BRANCHES:=0}"
: "${READ_CHUNK_SIZE:=50000}"

fresh_setup "$@"
fresh_require_data_dir
fresh_require_file "scripts/train_eval_set_matching_binary_offline_teacher.py"
fresh_require_file "${SET_MATCHING_MANIFEST_PATH}"
fresh_claim_new_dir "${BINARY_OFFLINE_TEACHER_DIR}"

fresh_split_words label_filter_args "${SET_MATCHING_LABEL_FILTER_NAMES:-QCD Hbb}"
fresh_split_words label_name_args "${SET_MATCHING_LABEL_NAMES:-QCD Hbb}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/train_eval_set_matching_binary_offline_teacher.py"
  --output-dir "${BINARY_OFFLINE_TEACHER_DIR}"
  --manifest-path "${SET_MATCHING_MANIFEST_PATH}"
  --data-dir "${DATA_DIR}"
  --label-filter-names "${label_filter_args[@]}"
  --label-names "${label_name_args[@]}"
  --seed "${BINARY_OFFLINE_TEACHER_SEED}"
  --batch-size "${BINARY_OFFLINE_TEACHER_BATCH_SIZE}"
  --eval-batch-size "${BINARY_OFFLINE_TEACHER_EVAL_BATCH_SIZE}"
  --epochs "${BINARY_OFFLINE_TEACHER_EPOCHS}"
  --lr "${BINARY_OFFLINE_TEACHER_LR}"
  --weight-decay "${BINARY_OFFLINE_TEACHER_WEIGHT_DECAY}"
  --num-workers "${BINARY_OFFLINE_TEACHER_NUM_WORKERS}"
  --device "${BINARY_OFFLINE_TEACHER_DEVICE}"
  --grad-clip-norm "${GRAD_CLIP_NORM}"
  --early-stop-patience "${BINARY_OFFLINE_TEACHER_EARLY_STOP_PATIENCE}"
  --max-train-jets "${BINARY_OFFLINE_TEACHER_MAX_TRAIN_JETS}"
  --max-val-jets "${BINARY_OFFLINE_TEACHER_MAX_VAL_JETS}"
  --max-stack-val-jets "${BINARY_OFFLINE_TEACHER_MAX_STACK_VAL_JETS}"
  --max-final-test-jets "${BINARY_OFFLINE_TEACHER_MAX_FINAL_TEST_JETS}"
  --model-size "${BINARY_OFFLINE_TEACHER_MODEL_SIZE}"
  --read-chunk-size "${READ_CHUNK_SIZE}"
  --confirm-final-test
)
fresh_append_flag_if_enabled cmd --no-amp "${BINARY_OFFLINE_TEACHER_NO_AMP}"
fresh_append_flag_if_enabled cmd --compile-model "${BINARY_OFFLINE_TEACHER_COMPILE_MODEL}"
fresh_append_flag_if_enabled cmd --verify-label-branches "${VERIFY_LABEL_BRANCHES}"
fresh_append_optional_arg cmd --max-train-batches "${BINARY_OFFLINE_TEACHER_MAX_TRAIN_BATCHES}"
fresh_append_optional_arg cmd --max-val-batches "${BINARY_OFFLINE_TEACHER_MAX_VAL_BATCHES}"
fresh_append_optional_arg cmd --max-eval-batches "${BINARY_OFFLINE_TEACHER_MAX_EVAL_BATCHES}"

fresh_write_run_config "${BINARY_OFFLINE_TEACHER_DIR}" "set_matching_binary_offline_teacher_reference" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${BINARY_OFFLINE_TEACHER_DIR}/best_model_val.pt"
  fresh_require_file "${BINARY_OFFLINE_TEACHER_DIR}/last.pt"
  fresh_require_file "${BINARY_OFFLINE_TEACHER_DIR}/training_curves.json"
  fresh_require_file "${BINARY_OFFLINE_TEACHER_DIR}/model_val_report.json"
  fresh_require_file "${BINARY_OFFLINE_TEACHER_DIR}/run_report.json"
  fresh_require_file "${BINARY_OFFLINE_TEACHER_DIR}/diagnostics/summary.csv"
  fresh_require_file "${BINARY_OFFLINE_TEACHER_DIR}/diagnostics/per_class_metrics.csv"
fi
