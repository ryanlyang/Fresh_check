#!/usr/bin/env bash
# Cache HLT-only predictions for one selected Step 8 tagger.

#SBATCH --job-name=c2f_predict
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=1-00:00:00
#SBATCH --mem=160G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

RUN_ID="${1:?Usage: sbatch run_cache_constrained_coarse_to_fine_predictions.sh <tagger-run-id>}"
: "${CONSTRAINED_C2F_ROOT:=${OUTPUT_ROOT}/constrained_coarse_to_fine}"
: "${CONSTRAINED_C2F_MANIFEST_PATH:=${CONSTRAINED_C2F_ROOT}/inputs/split_manifest/split_manifest.json.gz}"
: "${CONSTRAINED_C2F_HLT_CACHE_DIR:=${CONSTRAINED_C2F_ROOT}/inputs/hlt_cache}"
: "${CONSTRAINED_C2F_TAGGER_ROOT:=${CONSTRAINED_C2F_ROOT}/taggers}"
: "${CONSTRAINED_C2F_PREDICTION_DIR:=${CONSTRAINED_C2F_ROOT}/predictions}"
: "${CONSTRAINED_C2F_PREDICT_SPLITS:=model_val stack_train stack_val}"
: "${CONSTRAINED_C2F_PREDICT_BATCH_SIZE:=128}"
: "${CONSTRAINED_C2F_PREDICT_MAX_JETS_PER_SPLIT:=}"

CHECKPOINT="${CONSTRAINED_C2F_TAGGER_ROOT}/${RUN_ID}/best_model_val.pt"
OUTPUT_DIR="${CONSTRAINED_C2F_PREDICTION_DIR}/${RUN_ID}"
fresh_setup "$@"
fresh_require_file "${CONSTRAINED_C2F_MANIFEST_PATH}"
fresh_require_dir "${CONSTRAINED_C2F_HLT_CACHE_DIR}"
fresh_require_file "${CHECKPOINT}"
if fresh_bool_enabled "${CONFIRM_FINAL_TEST}"; then
  fresh_require_dir "${OUTPUT_DIR}"
else
  fresh_claim_new_dir "${OUTPUT_DIR}"
fi
fresh_split_words split_args "${CONSTRAINED_C2F_PREDICT_SPLITS}"

cmd=(
  "${PYTHON_BIN}" -u scripts/cache_constrained_coarse_to_fine_predictions.py
  --prediction-dir "${CONSTRAINED_C2F_PREDICTION_DIR}"
  --model-name "${RUN_ID}"
  --manifest "${CONSTRAINED_C2F_MANIFEST_PATH}"
  --hlt-cache-dir "${CONSTRAINED_C2F_HLT_CACHE_DIR}"
  --checkpoint "${CHECKPOINT}"
  --splits "${split_args[@]}"
  --batch-size "${CONSTRAINED_C2F_PREDICT_BATCH_SIZE}"
  --device "${DEVICE}"
)
fresh_append_flag_if_enabled cmd --overwrite "${OVERWRITE}"
fresh_append_flag_if_enabled cmd --confirm-final-test "${CONFIRM_FINAL_TEST}"
if [[ -n "${CONSTRAINED_C2F_PREDICT_MAX_JETS_PER_SPLIT}" ]]; then cmd+=(--max-jets-per-split "${CONSTRAINED_C2F_PREDICT_MAX_JETS_PER_SPLIT}"); fi

fresh_write_run_config "${OUTPUT_DIR}" "constrained_c2f_predictions_${RUN_ID}" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${OUTPUT_DIR}/prediction_run_report.json"
  for split in "${split_args[@]}"; do
    fresh_require_file "${OUTPUT_DIR}/${split}_predictions.npz"
    fresh_require_file "${OUTPUT_DIR}/${split}_predictions_metadata.json"
  done
fi
