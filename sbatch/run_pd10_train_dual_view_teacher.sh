#!/usr/bin/env bash
#SBATCH --job-name=pd10_dual_view
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=12:00:00
#SBATCH --mem=220G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${MAX_TRAIN_BATCHES:=}"
: "${MAX_VAL_BATCHES:=}"

OUTPUT_DIR="${PD10_DUAL_VIEW_TEACHER_DIR}"
MODEL_NAME="$(fresh_pd10_teacher_model_name dual_view)"

fresh_setup "$@"
fresh_claim_new_dir "${OUTPUT_DIR}"
fresh_require_file "${PD10_TEACHER_LOGITS_DIR}/hlt_part_teacher_10class/teacher_logit_manifest.json"
fresh_require_file "${PD10_TEACHER_LOGITS_DIR}/offline_part_teacher_10class/teacher_logit_manifest.json"
if fresh_bool_enabled "${PD10_DUAL_VIEW_NO_SKIP_EXISTING_PREDICTIONS}"; then
  fresh_refuse_existing_dir "${PD10_TEACHER_LOGITS_DIR}/${MODEL_NAME}"
fi

fresh_split_words split_args "${PD10_DUAL_VIEW_PREDICT_SPLITS}"
cmd=(
  "${PYTHON_BIN}" "-u" "scripts/train_pd10_dual_view_logit_teacher.py"
  --teacher-logit-dir "${PD10_TEACHER_LOGITS_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --prediction-output-dir "${PD10_TEACHER_LOGITS_DIR}"
  --splits "${split_args[@]}"
  --seed "${PD10_DUAL_VIEW_TEACHER_SEED}"
  --batch-size "${PD10_DUAL_VIEW_BATCH_SIZE}"
  --eval-batch-size "${PD10_DUAL_VIEW_EVAL_BATCH_SIZE}"
  --num-workers "${PD10_DUAL_VIEW_NUM_WORKERS}"
  --epochs "${PD10_DUAL_VIEW_EPOCHS}"
  --lr "${PD10_DUAL_VIEW_LR}"
  --weight-decay "${PD10_DUAL_VIEW_WEIGHT_DECAY}"
  --hidden-dim "${PD10_DUAL_VIEW_HIDDEN_DIM}"
  --dropout "${PD10_DUAL_VIEW_DROPOUT}"
  --early-stop-patience "${PD10_DUAL_VIEW_EARLY_STOP_PATIENCE}"
  --grad-clip-norm "${PD10_DUAL_VIEW_GRAD_CLIP_NORM}"
  --device "${PD10_DUAL_VIEW_DEVICE}"
  --max-train-jets "${PD10_MODEL_TRAIN_SIZE}"
  --max-val-jets "${PD10_MODEL_VAL_SIZE}"
  --max-final-test-jets "${PD10_FINAL_TEST_SIZE}"
  --confirm-final-test
)
fresh_append_flag_if_enabled cmd --overwrite "${OVERWRITE}"
fresh_append_flag_if_enabled cmd --no-skip-existing-predictions "${PD10_DUAL_VIEW_NO_SKIP_EXISTING_PREDICTIONS}"
fresh_append_optional_arg cmd --max-train-batches "${MAX_TRAIN_BATCHES}"
fresh_append_optional_arg cmd --max-val-batches "${MAX_VAL_BATCHES}"

fresh_write_run_config "${OUTPUT_DIR}" "pd10_dual_view_logit_teacher" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${OUTPUT_DIR}/best_model_val.pt"
  fresh_require_file "${OUTPUT_DIR}/last.pt"
  fresh_require_file "${OUTPUT_DIR}/run_report.json"
  fresh_require_file "${OUTPUT_DIR}/model_val_report.json"
  fresh_require_file "${OUTPUT_DIR}/source_metadata.json"
  fresh_require_file "${OUTPUT_DIR}/config.json"
  fresh_require_file "${PD10_TEACHER_LOGITS_DIR}/${MODEL_NAME}/teacher_logit_manifest.json"
  fresh_assert_json_ok "${OUTPUT_DIR}/run_report.json"
  fresh_assert_json_ok "${PD10_TEACHER_LOGITS_DIR}/${MODEL_NAME}/teacher_logit_manifest.json"
  for split in "${split_args[@]}"; do
    fresh_require_file "${PD10_TEACHER_LOGITS_DIR}/${MODEL_NAME}/${split}_predictions.npz"
    fresh_require_file "${PD10_TEACHER_LOGITS_DIR}/${MODEL_NAME}/${split}_predictions_metadata.json"
  done
fi
