#!/usr/bin/env bash
#SBATCH --job-name=pd10_teacher_logits
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=1-12:00:00
#SBATCH --mem=220G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

TEACHER="${1:?Usage: sbatch run_pd10_cache_teacher_logits.sh <hlt|offline>}"
MODEL_NAME="$(fresh_pd10_teacher_model_name "${TEACHER}")"
CHECKPOINT="${PD10_TEACHERS_DIR}/${MODEL_NAME}/best_model_val.pt"

: "${VERIFY_LABEL_BRANCHES:=0}"
: "${READ_CHUNK_SIZE:=50000}"

DATA_DIR="${PD10_DATA_DIR}"
if [[ -z "${PD10_OFFLINE_CACHE_DIR:-}" && -d "${PD10_ROOT}/inputs/offline_cache" ]]; then
  PD10_OFFLINE_CACHE_DIR="${PD10_ROOT}/inputs/offline_cache"
fi

fresh_setup "$@"
fresh_require_file "${CHECKPOINT}"
fresh_require_file "${PD10_MANIFEST_PATH}"
if [[ "${TEACHER}" == "hlt" ]]; then
  fresh_require_dir "${PD10_HLT_CACHE_DIR}"
else
  if [[ -n "${PD10_OFFLINE_CACHE_DIR:-}" ]]; then
    fresh_require_file "${PD10_OFFLINE_CACHE_DIR}/model_train_offline_metadata.json"
    fresh_require_file "${PD10_OFFLINE_CACHE_DIR}/model_val_offline_metadata.json"
    fresh_require_file "${PD10_OFFLINE_CACHE_DIR}/final_test_offline_metadata.json"
  else
    fresh_require_data_dir
  fi
fi
if fresh_bool_enabled "${PD10_TEACHER_LOGIT_NO_SKIP_EXISTING}"; then
  fresh_refuse_existing_dir "${PD10_TEACHER_LOGITS_DIR}/${MODEL_NAME}"
fi

fresh_split_words split_args "${PD10_TEACHER_LOGIT_SPLITS}"
cmd=(
  "${PYTHON_BIN}" "-u" "scripts/cache_pd10_teacher_logits.py"
  --teacher "${TEACHER}"
  --checkpoint "${CHECKPOINT}"
  --output-dir "${PD10_TEACHER_LOGITS_DIR}"
  --manifest "${PD10_MANIFEST_PATH}"
  --hlt-cache-dir "${PD10_HLT_CACHE_DIR}"
  --data-dir "${PD10_DATA_DIR}"
  --splits "${split_args[@]}"
  --batch-size "${PD10_TEACHER_LOGIT_BATCH_SIZE}"
  --num-workers "${PD10_TEACHER_LOGIT_NUM_WORKERS}"
  --device "${PD10_TEACHER_LOGIT_DEVICE}"
  --max-model-train-jets "${PD10_MODEL_TRAIN_SIZE}"
  --max-model-val-jets "${PD10_MODEL_VAL_SIZE}"
  --max-final-test-jets "${PD10_FINAL_TEST_SIZE}"
  --control-seed "${PD10_TEACHER_LOGIT_CONTROL_SEED}"
  --read-chunk-size "${READ_CHUNK_SIZE}"
  --confirm-final-test
)
if [[ "${TEACHER}" == "offline" ]]; then
  fresh_append_optional_arg cmd --offline-cache-dir "${PD10_OFFLINE_CACHE_DIR:-}"
fi
fresh_append_flag_if_enabled cmd --overwrite "${OVERWRITE}"
fresh_append_flag_if_enabled cmd --no-skip-existing "${PD10_TEACHER_LOGIT_NO_SKIP_EXISTING}"
fresh_append_flag_if_enabled cmd --verify-label-branches "${VERIFY_LABEL_BRANCHES}"

fresh_write_run_config "${PD10_TEACHER_LOGITS_DIR}/${MODEL_NAME}" "pd10_teacher_logits_${TEACHER}" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${PD10_TEACHER_LOGITS_DIR}/${MODEL_NAME}/teacher_logit_manifest.json"
  fresh_assert_json_ok "${PD10_TEACHER_LOGITS_DIR}/${MODEL_NAME}/teacher_logit_manifest.json"
  for split in "${split_args[@]}"; do
    fresh_require_file "${PD10_TEACHER_LOGITS_DIR}/${MODEL_NAME}/${split}_predictions.npz"
    fresh_require_file "${PD10_TEACHER_LOGITS_DIR}/${MODEL_NAME}/${split}_predictions_metadata.json"
  done
fi
