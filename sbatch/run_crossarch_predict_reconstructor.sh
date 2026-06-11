#!/usr/bin/env bash
# Collect prediction blocks for one cross-architecture teacher-logit reconstructor.

#SBATCH --job-name=crossarch_reco_pred
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=05:00:00
#SBATCH --mem=160G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

RECO_ARCHITECTURE="${1:?Usage: sbatch run_crossarch_predict_reconstructor.sh <gt|pn|pfn|pcnn> <part|pn|pfn|pcnn>}"
TEACHER_ARCHITECTURE="${2:?Usage: sbatch run_crossarch_predict_reconstructor.sh <gt|pn|pfn|pcnn> <part|pn|pfn|pcnn>}"

MODEL_NAME="$(fresh_crossarch_reco_model_name "${RECO_ARCHITECTURE}" "${TEACHER_ARCHITECTURE}")"
PREDICT_SCRIPT="$(fresh_crossarch_reco_predict_script "${RECO_ARCHITECTURE}")"
RECONSTRUCTOR_CHECKPOINT="${CROSSARCH_RECO_MODEL_DIR}/${RECO_ARCHITECTURE}/${TEACHER_ARCHITECTURE}/best_model_val.pt"
TEACHER_CHECKPOINT="${CROSSARCH_OFFLINE_TEACHER_DIR}/${TEACHER_ARCHITECTURE}/best_model_val.pt"
RUN_OUTPUT_DIR="${CROSSARCH_RECO_PREDICTION_RUN_DIR}/${MODEL_NAME}"
SOURCE_PREDICTION_DIR="${CROSSARCH_PREDICTION_DIR}/${MODEL_NAME}"

: "${NO_AMP:=0}"
: "${MAX_CONSTITS:=128}"
: "${TEACHER_WEIGHT_THRESHOLD:=0.0}"
: "${NON_STRICT_CHECKPOINT:=0}"

fresh_setup "$@"
fresh_require_file "${PREDICT_SCRIPT}"
fresh_require_file "${CROSSARCH_HLT_CACHE_DIR}/stack_train_fixed_hlt_metadata.json"
fresh_require_file "${CROSSARCH_HLT_CACHE_DIR}/stack_val_fixed_hlt_metadata.json"
fresh_require_file "${CROSSARCH_HLT_CACHE_DIR}/final_test_fixed_hlt_metadata.json"
fresh_require_file "${RECONSTRUCTOR_CHECKPOINT}"
fresh_require_file "${TEACHER_CHECKPOINT}"
fresh_refuse_existing_dir "${SOURCE_PREDICTION_DIR}"
fresh_claim_new_dir "${RUN_OUTPUT_DIR}"
if ! fresh_is_dry_run; then
  mkdir -p "${CROSSARCH_PREDICTION_DIR}"
fi

fresh_split_words split_args "${CROSSARCH_RECO_PREDICT_SPLITS}"
cmd=(
  "${PYTHON_BIN}" "-u" "${PREDICT_SCRIPT}"
  --output-dir "${RUN_OUTPUT_DIR}"
  --prediction-dir "${CROSSARCH_PREDICTION_DIR}"
  --hlt-cache-dir "${CROSSARCH_HLT_CACHE_DIR}"
  --reconstructor-checkpoint "${RECONSTRUCTOR_CHECKPOINT}"
  --teacher-checkpoint "${TEACHER_CHECKPOINT}"
  --teacher-architecture "${TEACHER_ARCHITECTURE}"
  --model-name "${MODEL_NAME}"
  --splits "${split_args[@]}"
  --batch-size "${CROSSARCH_RECO_PREDICT_BATCH_SIZE}"
  --num-workers "${CROSSARCH_RECO_PREDICT_NUM_WORKERS}"
  --device "${CROSSARCH_RECO_PREDICT_DEVICE}"
  --max-constits "${MAX_CONSTITS}"
  --teacher-weight-threshold "${TEACHER_WEIGHT_THRESHOLD}"
  --confirm-final-test
)
fresh_append_flag_if_enabled cmd --no-amp "${NO_AMP}"
fresh_append_flag_if_enabled cmd --overwrite-predictions "${OVERWRITE}"
fresh_append_flag_if_enabled cmd --non-strict-checkpoint "${NON_STRICT_CHECKPOINT}"
fresh_append_optional_arg cmd --max-jets-per-split "${CROSSARCH_RECO_PREDICT_MAX_JETS_PER_SPLIT}"

fresh_write_run_config "${RUN_OUTPUT_DIR}" "crossarch_reco_predict_${MODEL_NAME}" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${RUN_OUTPUT_DIR}/prediction_collection_report.json"
  for split in "${split_args[@]}"; do
    fresh_require_file "${SOURCE_PREDICTION_DIR}/${split}_predictions.npz"
    fresh_require_file "${SOURCE_PREDICTION_DIR}/${split}_predictions_metadata.json"
  done
fi
