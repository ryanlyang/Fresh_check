#!/usr/bin/env bash
#SBATCH --job-name=crossarch_hlt_pred
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

ARCHITECTURE="${1:?Usage: sbatch run_crossarch_predict_hlt_baseline.sh <part|pn|pfn|pcnn>}"
MODEL_NAME="$(fresh_crossarch_hlt_model_name "${ARCHITECTURE}")"
CHECKPOINT="${CROSSARCH_HLT_BASELINE_DIR}/${ARCHITECTURE}/best_model_val.pt"
RUN_OUTPUT_DIR="${CROSSARCH_HLT_PREDICTION_RUN_DIR}/${MODEL_NAME}"
SOURCE_PREDICTION_DIR="${CROSSARCH_PREDICTION_DIR}/${MODEL_NAME}"

fresh_setup "$@"
fresh_require_file "${CHECKPOINT}"
fresh_require_file "${CROSSARCH_HLT_CACHE_DIR}/stack_train_fixed_hlt_metadata.json"
fresh_require_file "${CROSSARCH_HLT_CACHE_DIR}/stack_val_fixed_hlt_metadata.json"
fresh_require_file "${CROSSARCH_HLT_CACHE_DIR}/final_test_fixed_hlt_metadata.json"
fresh_claim_new_dir "${RUN_OUTPUT_DIR}"
if [[ -d "${SOURCE_PREDICTION_DIR}" ]] && ! fresh_bool_enabled "${OVERWRITE}" && ! fresh_is_dry_run; then
  echo "Refusing to reuse existing crossarch HLT prediction directory without OVERWRITE=1: ${SOURCE_PREDICTION_DIR}" >&2
  exit 2
fi

fresh_split_words split_args "${CROSSARCH_HLT_PREDICT_SPLITS}"
cmd=(
  "${PYTHON_BIN}" "-u" "scripts/predict_crossarch_hlt_baseline.py"
  --architecture "${ARCHITECTURE}"
  --checkpoint "${CHECKPOINT}"
  --cache-dir "${CROSSARCH_HLT_CACHE_DIR}"
  --prediction-dir "${CROSSARCH_PREDICTION_DIR}"
  --output-dir "${RUN_OUTPUT_DIR}"
  --splits "${split_args[@]}"
  --batch-size "${CROSSARCH_HLT_PREDICT_BATCH_SIZE}"
  --num-workers "${CROSSARCH_HLT_PREDICT_NUM_WORKERS}"
  --device "${CROSSARCH_HLT_PREDICT_DEVICE}"
  --stack-train-size "${CROSSARCH_STACK_TRAIN_SIZE}"
  --stack-val-size "${CROSSARCH_STACK_VAL_SIZE}"
  --final-test-size "${CROSSARCH_FINAL_TEST_SIZE}"
  --control-seed "${CROSSARCH_HLT_PREDICT_CONTROL_SEED}"
  --confirm-final-test
)
fresh_append_flag_if_enabled cmd --overwrite-predictions "${OVERWRITE}"

fresh_write_run_config "${RUN_OUTPUT_DIR}" "crossarch_hlt_predict_${ARCHITECTURE}" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${RUN_OUTPUT_DIR}/prediction_collection_report.json"
  for split in "${split_args[@]}"; do
    fresh_require_file "${SOURCE_PREDICTION_DIR}/${split}_predictions.npz"
    fresh_require_file "${SOURCE_PREDICTION_DIR}/${split}_predictions_metadata.json"
  done
fi
