#!/usr/bin/env bash
# Evaluate the offline-only teacher upper reference on balanced held-out splits.

#SBATCH --job-name=offline_ref
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=08:00:00
#SBATCH --mem=160G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${OFFLINE_REFERENCE_EVAL_DIR:=${OUTPUT_ROOT}/jetclass_offline_teacher_reference_eval}"
: "${OFFLINE_REFERENCE_STACK_VAL_SIZE:=50000}"
: "${OFFLINE_REFERENCE_FINAL_TEST_SIZE:=300000}"
: "${OFFLINE_REFERENCE_BATCH_SIZE:=128}"
: "${OFFLINE_REFERENCE_NUM_WORKERS:=4}"
: "${OFFLINE_REFERENCE_DEVICE:=${DEVICE}}"

fresh_setup "$@"
fresh_require_file "${MANIFEST_PATH}"
fresh_require_file "${OFFLINE_TEACHER_DIR}/best_model_val.pt"
fresh_claim_new_dir "${OFFLINE_REFERENCE_EVAL_DIR}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/evaluate_offline_teacher_reference.py"
  --manifest-path "${MANIFEST_PATH}"
  --data-dir "${DATA_DIR}"
  --checkpoint "${OFFLINE_TEACHER_DIR}/best_model_val.pt"
  --output-dir "${OFFLINE_REFERENCE_EVAL_DIR}"
  --splits stack_val final_test
  --stack-val-size "${OFFLINE_REFERENCE_STACK_VAL_SIZE}"
  --final-test-size "${OFFLINE_REFERENCE_FINAL_TEST_SIZE}"
  --batch-size "${OFFLINE_REFERENCE_BATCH_SIZE}"
  --num-workers "${OFFLINE_REFERENCE_NUM_WORKERS}"
  --device "${OFFLINE_REFERENCE_DEVICE}"
  --control-seed "${HETERO_HLT4_CONTROL_SEED}"
  --confirm-final-test
)
fresh_append_flag_if_enabled cmd --overwrite-predictions "${OVERWRITE}"
fresh_append_flag_if_enabled cmd --verify-label-branches "${VERIFY_LABEL_BRANCHES:-0}"

fresh_write_run_config "${OFFLINE_REFERENCE_EVAL_DIR}" "evaluate_offline_teacher_reference" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${OFFLINE_REFERENCE_EVAL_DIR}/offline_teacher_reference_report.json"
  fresh_require_file "${OFFLINE_REFERENCE_EVAL_DIR}/predictions/offline_teacher/stack_val_predictions.npz"
  fresh_require_file "${OFFLINE_REFERENCE_EVAL_DIR}/predictions/offline_teacher/final_test_predictions.npz"
fi
