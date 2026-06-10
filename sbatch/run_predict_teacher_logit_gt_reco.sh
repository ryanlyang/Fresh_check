#!/usr/bin/env bash
# Step 7 helper: collect prediction blocks for one trained teacher-logit GT reconstructor.

#SBATCH --job-name=tlogit_pred
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

ARCHITECTURE="${1:?Usage: sbatch/run_predict_teacher_logit_gt_reco.sh <part|pn|pfn|pcnn>}"
case "${ARCHITECTURE}" in
  part|pn|pfn|pcnn) ;;
  *)
    echo "Unknown teacher-logit GT architecture ${ARCHITECTURE}; expected part pn pfn pcnn" >&2
    exit 2
    ;;
esac

MODEL_NAME="$(fresh_teacher_logit_gt_model_name "${ARCHITECTURE}")"
OUTPUT_DIR="${TEACHER_LOGIT_GT_PREDICTION_RUN_ROOT}/${ARCHITECTURE}"
RECONSTRUCTOR_CHECKPOINT="${TEACHER_LOGIT_GT_RECO_ROOT}/${ARCHITECTURE}/best_model_val.pt"
TEACHER_CHECKPOINT="$(fresh_teacher_logit_gt_teacher_checkpoint "${ARCHITECTURE}")"

fresh_setup "$@"
fresh_require_file "${HLT_CACHE_DIR}/stack_train_fixed_hlt_metadata.json"
fresh_require_file "${HLT_CACHE_DIR}/stack_val_fixed_hlt_metadata.json"
fresh_require_file "${HLT_CACHE_DIR}/final_test_fixed_hlt_metadata.json"
fresh_require_file "${RECONSTRUCTOR_CHECKPOINT}"
fresh_require_file "${TEACHER_CHECKPOINT}"
fresh_refuse_existing_dir "${TEACHER_LOGIT_GT_PREDICTION_DIR}/${MODEL_NAME}"
fresh_claim_new_dir "${OUTPUT_DIR}"
if ! fresh_is_dry_run; then
  mkdir -p "${TEACHER_LOGIT_GT_PREDICTION_DIR}"
fi

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/predict_teacher_logit_global_transformer_reco.py"
  --output-dir "${OUTPUT_DIR}"
  --prediction-dir "${TEACHER_LOGIT_GT_PREDICTION_DIR}"
  --hlt-cache-dir "${HLT_CACHE_DIR}"
  --reconstructor-checkpoint "${RECONSTRUCTOR_CHECKPOINT}"
  --teacher-checkpoint "${TEACHER_CHECKPOINT}"
  --teacher-architecture "${ARCHITECTURE}"
  --model-name "${MODEL_NAME}"
  --splits stack_train stack_val final_test
  --batch-size "${TEACHER_LOGIT_GT_PREDICT_BATCH_SIZE}"
  --num-workers "${TEACHER_LOGIT_GT_PREDICT_NUM_WORKERS}"
  --device "${TEACHER_LOGIT_GT_PREDICT_DEVICE}"
  --confirm-final-test
)
fresh_append_flag_if_enabled cmd --no-amp "${NO_AMP:-0}"
fresh_append_flag_if_enabled cmd --overwrite-predictions "${OVERWRITE}"
fresh_append_optional_arg cmd --max-jets-per-split "${TEACHER_LOGIT_GT_MAX_JETS_PER_SPLIT}"

fresh_write_run_config "${OUTPUT_DIR}" "predict_teacher_logit_gt_${ARCHITECTURE}" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${OUTPUT_DIR}/prediction_collection_report.json"
  fresh_require_file "${TEACHER_LOGIT_GT_PREDICTION_DIR}/${MODEL_NAME}/stack_train_predictions.npz"
  fresh_require_file "${TEACHER_LOGIT_GT_PREDICTION_DIR}/${MODEL_NAME}/stack_val_predictions.npz"
  fresh_require_file "${TEACHER_LOGIT_GT_PREDICTION_DIR}/${MODEL_NAME}/final_test_predictions.npz"
fi
