#!/usr/bin/env bash
# Train one single-view source branch for the deployable PD10 HLT tri-view test.

#SBATCH --job-name=pd10_hlt_tri_src
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=debug
#SBATCH --time=12:00:00
#SBATCH --mem=220G
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
: "${PD10_HLT_TRIVIEW_SOURCE_WARM_START_CHECKPOINT:=}"
: "${PD10_HLT_TRIVIEW_SOURCE_SEED:=9101}"
: "${PD10_HLT_TRIVIEW_SOURCE_EPOCHS:=10}"
: "${PD10_HLT_TRIVIEW_SOURCE_BATCH_SIZE:=128}"
: "${PD10_HLT_TRIVIEW_SOURCE_EVAL_BATCH_SIZE:=128}"
: "${PD10_HLT_TRIVIEW_SOURCE_LR:=0.001}"
: "${PD10_HLT_TRIVIEW_SOURCE_WEIGHT_DECAY:=0.0001}"
: "${PD10_HLT_TRIVIEW_SOURCE_EARLY_STOP_PATIENCE:=3}"
: "${PD10_HLT_TRIVIEW_SOURCE_NUM_WORKERS:=${NUM_WORKERS}}"
: "${PD10_HLT_TRIVIEW_SOURCE_DEVICE:=${DEVICE}}"
: "${PD10_HLT_TRIVIEW_SOURCE_MODEL_SIZE:=base}"
: "${PD10_HLT_TRIVIEW_SOURCE_NO_AMP:=1}"
: "${PD10_HLT_TRIVIEW_SOURCE_COMPILE_MODEL:=0}"
: "${PD10_HLT_TRIVIEW_SOURCE_SKIP_MODEL_VAL_PREDICTIONS:=0}"
: "${PD10_HLT_TRIVIEW_SOURCE_SKIP_FINAL_TEST:=0}"
: "${PD10_HLT_TRIVIEW_SOURCE_MAX_TRAIN_BATCHES:=}"
: "${PD10_HLT_TRIVIEW_SOURCE_MAX_VAL_BATCHES:=}"
: "${PD10_HLT_TRIVIEW_SOURCE_MAX_FINAL_TEST_BATCHES:=}"
: "${PD10_HLT_TRIVIEW_TRAIN_SIZE:=1000000}"
: "${PD10_HLT_TRIVIEW_VAL_SIZE:=250000}"
: "${PD10_HLT_TRIVIEW_FINAL_TEST_SIZE:=500000}"

SOURCE_NAME="${1:?source name is required, e.g. hlt_source}"
CACHE_DIR="${2:?cache dir is required}"
SOURCE_VIEW="${3:?source view is required: fixed_hlt or hlt2}"
OUTPUT_DIR="${PD10_HLT_TRIVIEW_SOURCE_MODELS_DIR}/${SOURCE_NAME}"

fresh_setup "$@"
fresh_require_dir "${CACHE_DIR}"
fresh_require_file "${CACHE_DIR}/model_train_fixed_hlt_metadata.json"
fresh_require_file "${CACHE_DIR}/model_val_fixed_hlt_metadata.json"
fresh_require_file "${CACHE_DIR}/final_test_fixed_hlt_metadata.json"
if ! fresh_bool_enabled "${PD10_HLT_TRIVIEW_SOURCE_SKIP_FINAL_TEST}" && ! fresh_bool_enabled "${CONFIRM_FINAL_TEST}"; then
  echo "Refusing HLT tri-view source final-test evaluation without CONFIRM_FINAL_TEST=1." >&2
  exit 2
fi
fresh_claim_new_dir "${OUTPUT_DIR}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/train_pd10_hlt_triview_source.py"
  --output-dir "${OUTPUT_DIR}"
  --cache-dir "${CACHE_DIR}"
  --source-name "${SOURCE_NAME}"
  --source-view "${SOURCE_VIEW}"
  --epochs "${PD10_HLT_TRIVIEW_SOURCE_EPOCHS}"
  --batch-size "${PD10_HLT_TRIVIEW_SOURCE_BATCH_SIZE}"
  --eval-batch-size "${PD10_HLT_TRIVIEW_SOURCE_EVAL_BATCH_SIZE}"
  --lr "${PD10_HLT_TRIVIEW_SOURCE_LR}"
  --weight-decay "${PD10_HLT_TRIVIEW_SOURCE_WEIGHT_DECAY}"
  --early-stop-patience "${PD10_HLT_TRIVIEW_SOURCE_EARLY_STOP_PATIENCE}"
  --num-workers "${PD10_HLT_TRIVIEW_SOURCE_NUM_WORKERS}"
  --device "${PD10_HLT_TRIVIEW_SOURCE_DEVICE}"
  --seed "${PD10_HLT_TRIVIEW_SOURCE_SEED}"
  --model-size "${PD10_HLT_TRIVIEW_SOURCE_MODEL_SIZE}"
  --max-train-jets "${PD10_HLT_TRIVIEW_TRAIN_SIZE}"
  --max-val-jets "${PD10_HLT_TRIVIEW_VAL_SIZE}"
  --max-final-test-jets "${PD10_HLT_TRIVIEW_FINAL_TEST_SIZE}"
)
if [[ -n "${PD10_HLT_TRIVIEW_SOURCE_WARM_START_CHECKPOINT}" && -f "${PD10_HLT_TRIVIEW_SOURCE_WARM_START_CHECKPOINT}" ]]; then
  cmd+=(--warm-start-checkpoint "${PD10_HLT_TRIVIEW_SOURCE_WARM_START_CHECKPOINT}")
elif [[ -n "${PD10_HLT_TRIVIEW_SOURCE_WARM_START_CHECKPOINT}" ]]; then
  echo "Warm-start checkpoint not found; training ${SOURCE_NAME} from scratch: ${PD10_HLT_TRIVIEW_SOURCE_WARM_START_CHECKPOINT}" >&2
fi
fresh_append_optional_arg cmd --max-train-batches "${PD10_HLT_TRIVIEW_SOURCE_MAX_TRAIN_BATCHES}"
fresh_append_optional_arg cmd --max-val-batches "${PD10_HLT_TRIVIEW_SOURCE_MAX_VAL_BATCHES}"
fresh_append_optional_arg cmd --max-final-test-batches "${PD10_HLT_TRIVIEW_SOURCE_MAX_FINAL_TEST_BATCHES}"
fresh_append_flag_if_enabled cmd --no-amp "${PD10_HLT_TRIVIEW_SOURCE_NO_AMP}"
fresh_append_flag_if_enabled cmd --compile-model "${PD10_HLT_TRIVIEW_SOURCE_COMPILE_MODEL}"
fresh_append_flag_if_enabled cmd --skip-model-val-predictions "${PD10_HLT_TRIVIEW_SOURCE_SKIP_MODEL_VAL_PREDICTIONS}"
fresh_append_flag_if_enabled cmd --skip-final-test "${PD10_HLT_TRIVIEW_SOURCE_SKIP_FINAL_TEST}"
fresh_append_flag_if_enabled cmd --confirm-final-test "${CONFIRM_FINAL_TEST}"
fresh_append_flag_if_enabled cmd --overwrite "${OVERWRITE}"

fresh_write_run_config "${OUTPUT_DIR}" "pd10_hlt_triview_source_${SOURCE_NAME}" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${OUTPUT_DIR}/best_model_val.pt"
  fresh_require_file "${OUTPUT_DIR}/last.pt"
  fresh_require_file "${OUTPUT_DIR}/run_report.json"
  fresh_require_file "${OUTPUT_DIR}/model_val_report.json"
  fresh_require_file "${OUTPUT_DIR}/config.json"
  fresh_require_file "${OUTPUT_DIR}/training_curves.json"
  fresh_assert_json_ok "${OUTPUT_DIR}/run_report.json"
  if ! fresh_bool_enabled "${PD10_HLT_TRIVIEW_SOURCE_SKIP_FINAL_TEST}"; then
    fresh_require_file "${OUTPUT_DIR}/final_test_report.json"
  fi
fi
