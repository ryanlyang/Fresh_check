#!/usr/bin/env bash
# Train one PD10 HLT-only student condition from a pipe-delimited spec.
# Spec format: init|teacher|target_mode|temperature|kd_alpha|top_k|variant_name[|representation_beta|representation_mode|representation_dim]
# Example: warm_start|dual_view|full_logits|2.0|0.5|3|pd10_student_warm_start_dual_view_full_logits_t2_a0p5

#SBATCH --job-name=pd10_student
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=3-00:00:00
#SBATCH --mem=220G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

SPEC="${1:?Usage: sbatch run_pd10_train_student.sh <init|teacher|target_mode|temperature|kd_alpha|top_k|variant_name[|representation_beta|representation_mode|representation_dim]>}"

old_ifs="${IFS}"
IFS='|'
read -r STUDENT_INIT TEACHER_TARGET TARGET_MODE TEMPERATURE KD_ALPHA TOP_K VARIANT_NAME REPRESENTATION_BETA REPRESENTATION_MODE REPRESENTATION_DIM <<< "${SPEC}"
IFS="${old_ifs}"

if [[ -z "${STUDENT_INIT}" || -z "${TEACHER_TARGET}" || -z "${TARGET_MODE}" || -z "${VARIANT_NAME}" ]]; then
  echo "Malformed PD10 student spec: ${SPEC}" >&2
  exit 2
fi

OUTPUT_DIR="${PD10_STUDENTS_DIR}/${VARIANT_NAME}"
BASELINE_CHECKPOINT="${PD10_STUDENT_WARM_START_BASELINE_CHECKPOINT}"
REPRESENTATION_BETA="${REPRESENTATION_BETA:-0.0}"
REPRESENTATION_MODE="${REPRESENTATION_MODE:-none}"
REPRESENTATION_DIM="${REPRESENTATION_DIM:-256}"
STUDENT_EPOCHS="${PD10_STUDENT_EPOCHS}"
STUDENT_LR="${PD10_STUDENT_SCRATCH_LR}"
KD_WARMUP_EPOCHS="${PD10_STUDENT_SCRATCH_KD_WARMUP_EPOCHS}"
if [[ "${STUDENT_INIT}" == "warm_start" ]]; then
  STUDENT_LR="${PD10_STUDENT_WARM_START_LR}"
  KD_WARMUP_EPOCHS="${PD10_STUDENT_WARM_START_KD_WARMUP_EPOCHS}"
  if [[ -z "${STUDENT_EPOCHS}" ]]; then
    STUDENT_EPOCHS="${PD10_STUDENT_WARM_START_EPOCHS}"
  fi
else
  if [[ -z "${STUDENT_EPOCHS}" ]]; then
    STUDENT_EPOCHS="${PD10_STUDENT_SCRATCH_EPOCHS}"
  fi
fi

fresh_setup "$@"
fresh_require_dir "${PD10_HLT_CACHE_DIR}"
fresh_require_file "${PD10_HLT_CACHE_DIR}/model_train_fixed_hlt_metadata.json"
fresh_require_file "${PD10_HLT_CACHE_DIR}/model_val_fixed_hlt_metadata.json"
fresh_require_file "${PD10_HLT_CACHE_DIR}/final_test_fixed_hlt_metadata.json"
fresh_claim_new_dir "${OUTPUT_DIR}"

if [[ "${STUDENT_INIT}" == "warm_start" ]]; then
  fresh_require_file "${BASELINE_CHECKPOINT}"
elif [[ "${STUDENT_INIT}" != "scratch" ]]; then
  echo "Unknown PD10 student init mode: ${STUDENT_INIT}" >&2
  exit 2
fi

case "${TARGET_MODE}" in
  full_logits|top3|confidence_weighted|full_logits_plus_rep|top3_plus_rep|confidence_weighted_plus_rep)
    NEED_TEACHER_LOGITS=1
    ;;
  *)
    NEED_TEACHER_LOGITS=0
    ;;
esac

if [[ "${TEACHER_TARGET}" != "none" && "${NEED_TEACHER_LOGITS}" == "1" ]]; then
  TEACHER_MODEL_NAME="$(fresh_pd10_teacher_model_name "${TEACHER_TARGET}")"
  fresh_require_file "${PD10_TEACHER_LOGITS_DIR}/${TEACHER_MODEL_NAME}/teacher_logit_manifest.json"
fi
case "${TARGET_MODE}" in
  rep_only|full_logits_plus_rep|top3_plus_rep|confidence_weighted_plus_rep)
    TEACHER_MODEL_NAME="$(fresh_pd10_teacher_model_name "${TEACHER_TARGET}")"
    fresh_require_file "${PD10_TEACHER_REPRESENTATIONS_DIR}/${TEACHER_MODEL_NAME}/teacher_representation_manifest.json"
    ;;
esac

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/train_pd10_student.py"
  --student-init "${STUDENT_INIT}"
  --teacher-target "${TEACHER_TARGET}"
  --target-mode "${TARGET_MODE}"
  --temperature "${TEMPERATURE}"
  --kd-alpha "${KD_ALPHA}"
  --kd-warmup-epochs "${KD_WARMUP_EPOCHS}"
  --top-k "${TOP_K}"
  --representation-beta "${REPRESENTATION_BETA}"
  --representation-mode "${REPRESENTATION_MODE}"
  --representation-dim "${REPRESENTATION_DIM}"
  --hlt-cache-dir "${PD10_HLT_CACHE_DIR}"
  --teacher-logit-cache "${PD10_TEACHER_LOGITS_DIR}"
  --teacher-representation-cache "${PD10_TEACHER_REPRESENTATIONS_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --seed "${PD10_STUDENT_SEED}"
  --batch-size "${PD10_STUDENT_BATCH_SIZE}"
  --epochs "${STUDENT_EPOCHS}"
  --lr "${STUDENT_LR}"
  --weight-decay "${PD10_STUDENT_WEIGHT_DECAY}"
  --num-workers "${PD10_STUDENT_NUM_WORKERS}"
  --device "${PD10_STUDENT_DEVICE}"
  --grad-clip-norm "${PD10_STUDENT_GRAD_CLIP_NORM}"
  --early-stop-patience "${PD10_STUDENT_EARLY_STOP_PATIENCE}"
  --max-train-jets "${PD10_MODEL_TRAIN_SIZE}"
  --max-val-jets "${PD10_MODEL_VAL_SIZE}"
  --max-final-test-jets "${PD10_FINAL_TEST_SIZE}"
  --model-size "${PD10_STUDENT_MODEL_SIZE}"
  --confirm-final-test
)
if [[ "${STUDENT_INIT}" == "warm_start" ]]; then
  fresh_append_optional_arg cmd --baseline-checkpoint "${BASELINE_CHECKPOINT}"
fi
fresh_append_flag_if_enabled cmd --no-amp "${PD10_STUDENT_NO_AMP}"
fresh_append_flag_if_enabled cmd --compile-model "${PD10_STUDENT_COMPILE_MODEL}"
fresh_append_flag_if_enabled cmd --align-prediction-to-teacher-cache "${PD10_STUDENT_ALIGN_PREDICTION_TO_TEACHER_CACHE}"
fresh_append_flag_if_enabled cmd --skip-final-test "${PD10_STUDENT_SKIP_FINAL_TEST}"
fresh_append_flag_if_enabled cmd --overwrite "${OVERWRITE}"
fresh_append_optional_arg cmd --max-train-batches "${PD10_STUDENT_MAX_TRAIN_BATCHES}"
fresh_append_optional_arg cmd --max-val-batches "${PD10_STUDENT_MAX_VAL_BATCHES}"
fresh_append_optional_arg cmd --max-final-test-batches "${PD10_STUDENT_MAX_FINAL_TEST_BATCHES}"

fresh_write_run_config "${OUTPUT_DIR}" "pd10_student_${VARIANT_NAME}" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${OUTPUT_DIR}/best_model_val.pt"
  fresh_require_file "${OUTPUT_DIR}/last.pt"
  fresh_require_file "${OUTPUT_DIR}/run_report.json"
  fresh_require_file "${OUTPUT_DIR}/model_val_report.json"
  fresh_require_file "${OUTPUT_DIR}/config.json"
  fresh_require_file "${OUTPUT_DIR}/training_curves.json"
  fresh_assert_json_ok "${OUTPUT_DIR}/run_report.json"
  if ! fresh_bool_enabled "${PD10_STUDENT_SKIP_FINAL_TEST}"; then
    fresh_require_file "${OUTPUT_DIR}/final_test_report.json"
  fi
fi
