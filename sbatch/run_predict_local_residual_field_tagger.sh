#!/usr/bin/env bash
# Cache prediction logits for one legacy tagger or deployable curriculum run.

#SBATCH --job-name=lprf_pred
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=12:00:00
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

RUN_ID="${1:?Usage: sbatch run_predict_local_residual_field_tagger.sh <tagger run_id>}"

: "${LOCAL_RESIDUAL_FIELD_ROOT:=${OUTPUT_ROOT}/local_particle_residual_field}"
: "${LOCAL_RESIDUAL_FIELD_MANIFEST_PATH:=${LOCAL_RESIDUAL_FIELD_ROOT}/inputs/split_manifest.json.gz}"
: "${LOCAL_RESIDUAL_FIELD_HLT_CACHE_DIR:=${LOCAL_RESIDUAL_FIELD_ROOT}/inputs/hlt_cache}"
: "${LOCAL_RESIDUAL_FIELD_TARGET_CACHE_DIR:=${LOCAL_RESIDUAL_FIELD_ROOT}/targets}"
: "${LOCAL_RESIDUAL_FIELD_TAGGER_ROOT:=${LOCAL_RESIDUAL_FIELD_ROOT}/taggers}"
: "${LOCAL_RESIDUAL_FIELD_PREDICT_MODEL_ROOT:=${LOCAL_RESIDUAL_FIELD_TAGGER_ROOT}}"
: "${LOCAL_RESIDUAL_FIELD_PREDICT_CHECKPOINT:=}"
: "${LOCAL_RESIDUAL_FIELD_PREDICTION_DIR:=${LOCAL_RESIDUAL_FIELD_ROOT}/predictions}"
: "${LOCAL_RESIDUAL_FIELD_PREDICT_SPLITS:=stack_train stack_val final_test}"
: "${LOCAL_RESIDUAL_FIELD_PREDICT_BATCH_SIZE:=128}"
: "${LOCAL_RESIDUAL_FIELD_PREDICT_NUM_WORKERS:=4}"
: "${LOCAL_RESIDUAL_FIELD_PREDICT_MAX_JETS:=}"
: "${LOCAL_RESIDUAL_FIELD_PREDICT_DISABLE_AMP:=0}"
: "${LOCAL_RESIDUAL_FIELD_ALLOW_ORACLE_FINAL_TEST:=0}"

CHECKPOINT="${LOCAL_RESIDUAL_FIELD_PREDICT_CHECKPOINT:-${LOCAL_RESIDUAL_FIELD_PREDICT_MODEL_ROOT}/${RUN_ID}/best_model_val.pt}"

fresh_setup "$@"
fresh_require_file "${CHECKPOINT}"
fresh_require_file "${LOCAL_RESIDUAL_FIELD_MANIFEST_PATH}"
fresh_require_dir "${LOCAL_RESIDUAL_FIELD_HLT_CACHE_DIR}"
fresh_require_dir "${LOCAL_RESIDUAL_FIELD_TARGET_CACHE_DIR}"
mkdir -p "${LOCAL_RESIDUAL_FIELD_PREDICTION_DIR}"
fresh_split_words split_args "${LOCAL_RESIDUAL_FIELD_PREDICT_SPLITS}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/predict_local_residual_field_tagger.py"
  --checkpoint "${CHECKPOINT}"
  --prediction-dir "${LOCAL_RESIDUAL_FIELD_PREDICTION_DIR}"
  --model-name "${RUN_ID}"
  --hlt-cache-dir "${LOCAL_RESIDUAL_FIELD_HLT_CACHE_DIR}"
  --target-cache-dir "${LOCAL_RESIDUAL_FIELD_TARGET_CACHE_DIR}"
  --manifest-path "${LOCAL_RESIDUAL_FIELD_MANIFEST_PATH}"
  --splits "${split_args[@]}"
  --batch-size "${LOCAL_RESIDUAL_FIELD_PREDICT_BATCH_SIZE}"
  --num-workers "${LOCAL_RESIDUAL_FIELD_PREDICT_NUM_WORKERS}"
  --device "${DEVICE}"
)
fresh_append_flag_if_enabled cmd --confirm-final-test "${CONFIRM_FINAL_TEST}"
fresh_append_flag_if_enabled cmd --allow-oracle-final-test "${LOCAL_RESIDUAL_FIELD_ALLOW_ORACLE_FINAL_TEST}"
fresh_append_flag_if_enabled cmd --overwrite "${OVERWRITE}"
fresh_append_flag_if_enabled cmd --disable-amp "${LOCAL_RESIDUAL_FIELD_PREDICT_DISABLE_AMP}"
if [[ -n "${LOCAL_RESIDUAL_FIELD_PREDICT_MAX_JETS}" ]]; then cmd+=(--max-jets "${LOCAL_RESIDUAL_FIELD_PREDICT_MAX_JETS}"); fi

fresh_write_run_config "${LOCAL_RESIDUAL_FIELD_PREDICTION_DIR}/${RUN_ID}" "local_residual_field_predictions_${RUN_ID}" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  for split in "${split_args[@]}"; do
    fresh_require_file "${LOCAL_RESIDUAL_FIELD_PREDICTION_DIR}/${RUN_ID}/${split}_predictions.npz"
    fresh_require_file "${LOCAL_RESIDUAL_FIELD_PREDICTION_DIR}/${RUN_ID}/${split}_predictions_metadata.json"
  done
fi
