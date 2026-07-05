#!/usr/bin/env bash
# Train one PDV3 AV10-adapter student variant.

#SBATCH --job-name=pdv3_student
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

STUDENT_VARIANT="${1:?Usage: sbatch run_pdv3_train_student.sh <pdv3_student_variant>}"
OUTPUT_DIR="${PDV3_STUDENTS_DIR}/${STUDENT_VARIANT}"

fresh_setup "$@"
fresh_require_file "${PDV3_MANIFEST_PATH}"
fresh_require_dir "${PDV3_HLT_CACHE_DIR}"
fresh_require_file "${PDV3_HLT_CACHE_DIR}/model_train_fixed_hlt_metadata.json"
fresh_require_file "${PDV3_HLT_CACHE_DIR}/model_val_fixed_hlt_metadata.json"
fresh_require_file "${PDV3_HLT_CACHE_DIR}/final_test_fixed_hlt_metadata.json"
if [[ -f "${PDV3_STUDENT_BASELINE_CHECKPOINT}" ]]; then
  echo "using PDV3 warm-start baseline checkpoint: ${PDV3_STUDENT_BASELINE_CHECKPOINT}"
elif [[ "${STUDENT_VARIANT}" == "pdv3_hlt_part_ce" ]] && fresh_bool_enabled "${PDV3_STUDENT_ALLOW_BASELINE_FROM_SCRATCH}"; then
  echo "PDV3 baseline checkpoint is missing; pdv3_hlt_part_ce will train from scratch"
else
  fresh_require_file "${PDV3_STUDENT_BASELINE_CHECKPOINT}"
fi
fresh_claim_new_dir "${OUTPUT_DIR}"

case "${STUDENT_VARIANT}" in
  *v1_dual_logit_kd*)
    fresh_require_file "${PDV3_TEACHER_LOGITS_DIR}/dual_view_logit_teacher_10class/teacher_logit_manifest.json"
    ;;
  *v2_logit_rep_kd*)
    fresh_require_file "${PDV3_TEACHER_LOGITS_DIR}/particle_dual_view_teacher_10class/particle_dual_view_cache_manifest.json"
    fresh_require_file "${PDV3_TEACHER_REPRESENTATIONS_DIR}/particle_dual_view_teacher_10class/teacher_representation_manifest.json"
    ;;
esac

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/train_pdv3_student.py"
  --output-dir "${OUTPUT_DIR}"
  --manifest-path "${PDV3_MANIFEST_PATH}"
  --hlt-cache-dir "${PDV3_HLT_CACHE_DIR}"
  --baseline-checkpoint "${PDV3_STUDENT_BASELINE_CHECKPOINT}"
  --student-variant "${STUDENT_VARIANT}"
  --teacher-logit-root "${PDV3_TEACHER_LOGITS_DIR}"
  --teacher-representation-root "${PDV3_TEACHER_REPRESENTATIONS_DIR}"
  --train-split model_train
  --val-split model_val
  --final-test-split final_test
  --confirm-split-settings
  --confirm-final-test
  --seed "${PDV3_STUDENT_SEED}"
  --batch-size "${PDV3_STUDENT_BATCH_SIZE}"
  --eval-batch-size "${PDV3_STUDENT_EVAL_BATCH_SIZE}"
  --epochs "${PDV3_STUDENT_EPOCHS}"
  --num-workers "${PDV3_STUDENT_NUM_WORKERS}"
  --device "${PDV3_STUDENT_DEVICE}"
  --grad-clip-norm "${PDV3_STUDENT_GRAD_CLIP_NORM}"
  --early-stop-patience "${PDV3_STUDENT_EARLY_STOP_PATIENCE}"
  --max-train-jets "${PDV3_MODEL_TRAIN_SIZE}"
  --max-val-jets "${PDV3_MODEL_VAL_SIZE}"
  --max-final-test-jets "${PDV3_FINAL_TEST_SIZE}"
  --selection-metric "${PDV3_STUDENT_SELECTION_METRIC}"
  --expected-hlt-degradation-strength "${PDV3_HLT_DEGRADATION_STRENGTH}"
  --delta-l2-weight "${PDV3_STUDENT_DELTA_L2_WEIGHT}"
  --representation-dim "${PDV3_STUDENT_REPRESENTATION_DIM}"
)

if fresh_bool_enabled "${PDV3_STUDENT_REQUIRE_BASELINE_SPLIT_MANIFEST_HASH}"; then
  cmd+=(--require-baseline-split-manifest-hash)
else
  cmd+=(--allow-missing-baseline-split-manifest-hash)
fi
fresh_append_flag_if_enabled cmd --no-amp "${PDV3_STUDENT_NO_AMP}"
fresh_append_flag_if_enabled cmd --compile-model "${PDV3_STUDENT_COMPILE_MODEL}"
if ! fresh_bool_enabled "${PDV3_STUDENT_ALLOW_BASELINE_FROM_SCRATCH}"; then
  cmd+=(--disable-baseline-from-scratch)
fi
fresh_append_flag_if_enabled cmd --final-test-teacher-diagnostics "${PDV3_STUDENT_FINAL_TEST_TEACHER_DIAGNOSTICS}"
fresh_append_flag_if_enabled cmd --overwrite "${OVERWRITE}"
fresh_append_optional_arg cmd --max-train-batches "${PDV3_STUDENT_MAX_TRAIN_BATCHES}"
fresh_append_optional_arg cmd --max-val-batches "${PDV3_STUDENT_MAX_VAL_BATCHES}"
fresh_append_optional_arg cmd --max-final-test-batches "${PDV3_STUDENT_MAX_FINAL_TEST_BATCHES}"

fresh_write_run_config "${OUTPUT_DIR}" "pdv3_student_${STUDENT_VARIANT}" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${OUTPUT_DIR}/best_model_val.pt"
  fresh_require_file "${OUTPUT_DIR}/last.pt"
  fresh_require_file "${OUTPUT_DIR}/run_report.json"
  fresh_require_file "${OUTPUT_DIR}/model_val_report.json"
  fresh_require_file "${OUTPUT_DIR}/final_test_report.json"
  fresh_require_file "${OUTPUT_DIR}/config.json"
  fresh_require_file "${OUTPUT_DIR}/training_curves.json"
  fresh_assert_json_ok "${OUTPUT_DIR}/run_report.json"
fi
