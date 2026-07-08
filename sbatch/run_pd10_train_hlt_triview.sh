#!/usr/bin/env bash
# Train the deployable PD10 HLT tri-view particle fusion model.

#SBATCH --job-name=pd10_hlt_triview
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=debug
#SBATCH --time=12:00:00
#SBATCH --mem=240G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_ENV:=atlas_kd}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${PD10_HLT_TRIVIEW_ROOT:=${PD10_ROOT}/hlt_triview_debug}"
: "${PD10_HLT_TRIVIEW_SOURCE_MODELS_DIR:=${PD10_HLT_TRIVIEW_ROOT}/source_models}"
: "${PD10_HLT_TRIVIEW_MODELS_DIR:=${PD10_HLT_TRIVIEW_ROOT}/models}"
: "${PD10_HLT_TRIVIEW_MODEL_NAME:=tri_hlt_hlt2_s0p35_s1p00}"
: "${PD10_HLT_TRIVIEW_HLT2_S0P35_CACHE_DIR:=$(fresh_pd10_hlt_sdv_hlt2_cache_dir 0.35)}"
: "${PD10_HLT_TRIVIEW_HLT2_S1P00_CACHE_DIR:=$(fresh_pd10_hlt_sdv_hlt2_cache_dir 1.00)}"
: "${PD10_HLT_TRIVIEW_HLT_SOURCE_CHECKPOINT:=${PD10_HLT_TRIVIEW_SOURCE_MODELS_DIR}/hlt_source/best_model_val.pt}"
: "${PD10_HLT_TRIVIEW_HLT2_S0P35_SOURCE_CHECKPOINT:=${PD10_HLT_TRIVIEW_SOURCE_MODELS_DIR}/hlt2_s0p35_source/best_model_val.pt}"
: "${PD10_HLT_TRIVIEW_HLT2_S1P00_SOURCE_CHECKPOINT:=${PD10_HLT_TRIVIEW_SOURCE_MODELS_DIR}/hlt2_s1p00_source/best_model_val.pt}"
: "${PD10_HLT_TRIVIEW_SEED:=9201}"
: "${PD10_HLT_TRIVIEW_EPOCHS:=10}"
: "${PD10_HLT_TRIVIEW_HEAD_WARMUP_EPOCHS:=1}"
: "${PD10_HLT_TRIVIEW_BATCH_SIZE:=64}"
: "${PD10_HLT_TRIVIEW_EVAL_BATCH_SIZE:=96}"
: "${PD10_HLT_TRIVIEW_HEAD_WARMUP_LR:=0.001}"
: "${PD10_HLT_TRIVIEW_BRANCH_LR:=0.00002}"
: "${PD10_HLT_TRIVIEW_HEAD_LR:=0.0003}"
: "${PD10_HLT_TRIVIEW_WEIGHT_DECAY:=0.0001}"
: "${PD10_HLT_TRIVIEW_DROPOUT:=0.05}"
: "${PD10_HLT_TRIVIEW_FUSION_HIDDEN_DIM:=512}"
: "${PD10_HLT_TRIVIEW_REPRESENTATION_DIM:=256}"
: "${PD10_HLT_TRIVIEW_EARLY_STOP_PATIENCE:=3}"
: "${PD10_HLT_TRIVIEW_NUM_WORKERS:=${NUM_WORKERS}}"
: "${PD10_HLT_TRIVIEW_DEVICE:=${DEVICE}}"
: "${PD10_HLT_TRIVIEW_MODEL_SIZE:=base}"
: "${PD10_HLT_TRIVIEW_NO_AMP:=0}"
: "${PD10_HLT_TRIVIEW_COMPILE_MODEL:=0}"
: "${PD10_HLT_TRIVIEW_SKIP_MODEL_VAL_PREDICTIONS:=0}"
: "${PD10_HLT_TRIVIEW_SKIP_FINAL_TEST:=0}"
: "${PD10_HLT_TRIVIEW_MAX_TRAIN_BATCHES:=}"
: "${PD10_HLT_TRIVIEW_MAX_VAL_BATCHES:=}"
: "${PD10_HLT_TRIVIEW_MAX_FINAL_TEST_BATCHES:=}"
: "${PD10_HLT_TRIVIEW_TRAIN_SIZE:=1000000}"
: "${PD10_HLT_TRIVIEW_VAL_SIZE:=250000}"
: "${PD10_HLT_TRIVIEW_FINAL_TEST_SIZE:=500000}"

OUTPUT_DIR="${PD10_HLT_TRIVIEW_MODELS_DIR}/${PD10_HLT_TRIVIEW_MODEL_NAME}"

fresh_setup "$@"
fresh_require_dir "${PD10_HLT_CACHE_DIR}"
fresh_require_dir "${PD10_HLT_TRIVIEW_HLT2_S0P35_CACHE_DIR}"
fresh_require_dir "${PD10_HLT_TRIVIEW_HLT2_S1P00_CACHE_DIR}"
for cache_dir in "${PD10_HLT_CACHE_DIR}" "${PD10_HLT_TRIVIEW_HLT2_S0P35_CACHE_DIR}" "${PD10_HLT_TRIVIEW_HLT2_S1P00_CACHE_DIR}"; do
  fresh_require_file "${cache_dir}/model_train_fixed_hlt_metadata.json"
  fresh_require_file "${cache_dir}/model_val_fixed_hlt_metadata.json"
  fresh_require_file "${cache_dir}/final_test_fixed_hlt_metadata.json"
done
fresh_require_file "${PD10_HLT_TRIVIEW_HLT_SOURCE_CHECKPOINT}"
fresh_require_file "${PD10_HLT_TRIVIEW_HLT2_S0P35_SOURCE_CHECKPOINT}"
fresh_require_file "${PD10_HLT_TRIVIEW_HLT2_S1P00_SOURCE_CHECKPOINT}"
if ! fresh_bool_enabled "${PD10_HLT_TRIVIEW_SKIP_FINAL_TEST}" && ! fresh_bool_enabled "${CONFIRM_FINAL_TEST}"; then
  echo "Refusing HLT tri-view final-test evaluation without CONFIRM_FINAL_TEST=1." >&2
  exit 2
fi
fresh_claim_new_dir "${OUTPUT_DIR}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/train_pd10_hlt_triview.py"
  --output-dir "${OUTPUT_DIR}"
  --hlt-cache-dir "${PD10_HLT_CACHE_DIR}"
  --hlt2-s0p35-cache-dir "${PD10_HLT_TRIVIEW_HLT2_S0P35_CACHE_DIR}"
  --hlt2-s1p00-cache-dir "${PD10_HLT_TRIVIEW_HLT2_S1P00_CACHE_DIR}"
  --hlt-source-checkpoint "${PD10_HLT_TRIVIEW_HLT_SOURCE_CHECKPOINT}"
  --hlt2-s0p35-source-checkpoint "${PD10_HLT_TRIVIEW_HLT2_S0P35_SOURCE_CHECKPOINT}"
  --hlt2-s1p00-source-checkpoint "${PD10_HLT_TRIVIEW_HLT2_S1P00_SOURCE_CHECKPOINT}"
  --model-name "${PD10_HLT_TRIVIEW_MODEL_NAME}"
  --epochs "${PD10_HLT_TRIVIEW_EPOCHS}"
  --head-warmup-epochs "${PD10_HLT_TRIVIEW_HEAD_WARMUP_EPOCHS}"
  --batch-size "${PD10_HLT_TRIVIEW_BATCH_SIZE}"
  --eval-batch-size "${PD10_HLT_TRIVIEW_EVAL_BATCH_SIZE}"
  --head-warmup-lr "${PD10_HLT_TRIVIEW_HEAD_WARMUP_LR}"
  --branch-lr "${PD10_HLT_TRIVIEW_BRANCH_LR}"
  --head-lr "${PD10_HLT_TRIVIEW_HEAD_LR}"
  --weight-decay "${PD10_HLT_TRIVIEW_WEIGHT_DECAY}"
  --dropout "${PD10_HLT_TRIVIEW_DROPOUT}"
  --fusion-hidden-dim "${PD10_HLT_TRIVIEW_FUSION_HIDDEN_DIM}"
  --representation-dim "${PD10_HLT_TRIVIEW_REPRESENTATION_DIM}"
  --early-stop-patience "${PD10_HLT_TRIVIEW_EARLY_STOP_PATIENCE}"
  --num-workers "${PD10_HLT_TRIVIEW_NUM_WORKERS}"
  --device "${PD10_HLT_TRIVIEW_DEVICE}"
  --seed "${PD10_HLT_TRIVIEW_SEED}"
  --model-size "${PD10_HLT_TRIVIEW_MODEL_SIZE}"
  --max-train-jets "${PD10_HLT_TRIVIEW_TRAIN_SIZE}"
  --max-val-jets "${PD10_HLT_TRIVIEW_VAL_SIZE}"
  --max-final-test-jets "${PD10_HLT_TRIVIEW_FINAL_TEST_SIZE}"
)
fresh_append_optional_arg cmd --max-train-batches "${PD10_HLT_TRIVIEW_MAX_TRAIN_BATCHES}"
fresh_append_optional_arg cmd --max-val-batches "${PD10_HLT_TRIVIEW_MAX_VAL_BATCHES}"
fresh_append_optional_arg cmd --max-final-test-batches "${PD10_HLT_TRIVIEW_MAX_FINAL_TEST_BATCHES}"
fresh_append_flag_if_enabled cmd --no-amp "${PD10_HLT_TRIVIEW_NO_AMP}"
fresh_append_flag_if_enabled cmd --compile-model "${PD10_HLT_TRIVIEW_COMPILE_MODEL}"
fresh_append_flag_if_enabled cmd --skip-model-val-predictions "${PD10_HLT_TRIVIEW_SKIP_MODEL_VAL_PREDICTIONS}"
fresh_append_flag_if_enabled cmd --skip-final-test "${PD10_HLT_TRIVIEW_SKIP_FINAL_TEST}"
fresh_append_flag_if_enabled cmd --confirm-final-test "${CONFIRM_FINAL_TEST}"
fresh_append_flag_if_enabled cmd --overwrite "${OVERWRITE}"

fresh_write_run_config "${OUTPUT_DIR}" "pd10_hlt_triview_${PD10_HLT_TRIVIEW_MODEL_NAME}" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${OUTPUT_DIR}/best_model_val.pt"
  fresh_require_file "${OUTPUT_DIR}/last.pt"
  fresh_require_file "${OUTPUT_DIR}/run_report.json"
  fresh_require_file "${OUTPUT_DIR}/model_val_report.json"
  fresh_require_file "${OUTPUT_DIR}/config.json"
  fresh_require_file "${OUTPUT_DIR}/training_curves.json"
  fresh_assert_json_ok "${OUTPUT_DIR}/run_report.json"
  if ! fresh_bool_enabled "${PD10_HLT_TRIVIEW_SKIP_FINAL_TEST}"; then
    fresh_require_file "${OUTPUT_DIR}/final_test_report.json"
  fi
fi
