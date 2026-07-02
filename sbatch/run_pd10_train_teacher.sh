#!/usr/bin/env bash
#SBATCH --job-name=pd10_teacher
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=4-00:00:00
#SBATCH --mem=220G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

TEACHER="${1:?Usage: sbatch run_pd10_train_teacher.sh <hlt|offline>}"
MODEL_NAME="$(fresh_pd10_teacher_model_name "${TEACHER}")"
OUTPUT_DIR="${PD10_TEACHERS_DIR}/${MODEL_NAME}"
TEACHER_SEED="$(fresh_pd10_teacher_seed "${TEACHER}")"

: "${NO_AMP:=0}"
: "${COMPILE_MODEL:=0}"
: "${VERIFY_LABEL_BRANCHES:=0}"
: "${READ_CHUNK_SIZE:=50000}"
: "${MAX_TRAIN_BATCHES:=}"
: "${MAX_VAL_BATCHES:=}"
: "${MAX_FINAL_TEST_BATCHES:=}"

DATA_DIR="${PD10_DATA_DIR}"

fresh_setup "$@"
fresh_require_file "${PD10_MANIFEST_PATH}"
fresh_claim_new_dir "${OUTPUT_DIR}"

source_checkpoint="$(fresh_pd10_teacher_source_checkpoint "${TEACHER}")"
source_report="$(fresh_pd10_teacher_source_report "${TEACHER}")"
source_final_test_report="$(fresh_pd10_teacher_source_final_test_report "${TEACHER}")"
if [[ -n "${source_checkpoint}" ]]; then
  fresh_require_file "${source_checkpoint}"
  if [[ -n "${source_report}" ]]; then
    fresh_require_file "${source_report}"
  fi
  if [[ -n "${source_final_test_report}" ]]; then
    fresh_require_file "${source_final_test_report}"
  fi
else
  if [[ "${TEACHER}" == "hlt" ]]; then
    fresh_require_file "${PD10_HLT_CACHE_DIR}/model_train_fixed_hlt_metadata.json"
    fresh_require_file "${PD10_HLT_CACHE_DIR}/model_val_fixed_hlt_metadata.json"
    fresh_require_file "${PD10_HLT_CACHE_DIR}/final_test_fixed_hlt_metadata.json"
  else
    fresh_require_data_dir
  fi
fi

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/train_or_register_pd10_teacher.py"
  --teacher "${TEACHER}"
  --manifest "${PD10_MANIFEST_PATH}"
  --hlt-cache-dir "${PD10_HLT_CACHE_DIR}"
  --data-dir "${PD10_DATA_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --seed "${TEACHER_SEED}"
  --batch-size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --num-workers "${NUM_WORKERS}"
  --device "${DEVICE}"
  --grad-clip-norm "${GRAD_CLIP_NORM}"
  --early-stop-patience "${EARLY_STOP_PATIENCE}"
  --max-train-jets "${PD10_MODEL_TRAIN_SIZE}"
  --max-val-jets "${PD10_MODEL_VAL_SIZE}"
  --max-final-test-jets "${PD10_FINAL_TEST_SIZE}"
  --model-size "${PD10_TEACHER_MODEL_SIZE}"
  --read-chunk-size "${READ_CHUNK_SIZE}"
  --confirm-final-test
)
fresh_append_flag_if_enabled cmd --no-amp "${NO_AMP}"
fresh_append_flag_if_enabled cmd --compile-model "${COMPILE_MODEL}"
fresh_append_flag_if_enabled cmd --verify-label-branches "${VERIFY_LABEL_BRANCHES}"
fresh_append_optional_arg cmd --max-train-batches "${MAX_TRAIN_BATCHES}"
fresh_append_optional_arg cmd --max-val-batches "${MAX_VAL_BATCHES}"
fresh_append_optional_arg cmd --max-final-test-batches "${MAX_FINAL_TEST_BATCHES}"
fresh_append_optional_arg cmd --register-checkpoint "${source_checkpoint}"
fresh_append_optional_arg cmd --register-source-report "${source_report}"
fresh_append_optional_arg cmd --register-source-final-test-report "${source_final_test_report}"

fresh_write_run_config "${OUTPUT_DIR}" "pd10_teacher_${TEACHER}" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${OUTPUT_DIR}/best_model_val.pt"
  fresh_require_file "${OUTPUT_DIR}/run_report.json"
  fresh_require_file "${OUTPUT_DIR}/model_val_report.json"
  fresh_require_file "${OUTPUT_DIR}/final_test_report.json"
  fresh_require_file "${OUTPUT_DIR}/source_metadata.json"
  fresh_require_file "${OUTPUT_DIR}/config.json"
fi
