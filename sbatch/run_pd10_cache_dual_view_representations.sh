#!/usr/bin/env bash
# Cache hidden representations from the existing PD10 dual-view logit-fusion teacher.

#SBATCH --job-name=pd10_dual_rep
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_ENV:=atlas_kd}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

MODEL_NAME="dual_view_logit_teacher_10class"
CHECKPOINT="${PD10_DUAL_VIEW_TEACHER_DIR}/best_model_val.pt"

fresh_setup "$@"
fresh_require_file "${CHECKPOINT}"
fresh_require_file "${PD10_TEACHER_LOGITS_DIR}/hlt_part_teacher_10class/teacher_logit_manifest.json"
fresh_require_file "${PD10_TEACHER_LOGITS_DIR}/offline_part_teacher_10class/teacher_logit_manifest.json"
if fresh_bool_enabled "${PD10_V2_REPRESENTATION_CACHE_NO_SKIP_EXISTING}"; then
  fresh_refuse_existing_dir "${PD10_V2_TEACHER_REPRESENTATIONS_DIR}/${MODEL_NAME}"
fi

fresh_split_words split_args "${PD10_V2_PARTICLE_DUAL_VIEW_CACHE_SPLITS}"
cmd=(
  "${PYTHON_BIN}" "-u" "scripts/cache_pd10_dual_view_representations.py"
  --checkpoint "${CHECKPOINT}"
  --teacher-logit-dir "${PD10_TEACHER_LOGITS_DIR}"
  --output-dir "${PD10_V2_TEACHER_REPRESENTATIONS_DIR}"
  --splits "${split_args[@]}"
  --batch-size "${PD10_DUAL_VIEW_EVAL_BATCH_SIZE}"
  --device "${PD10_DUAL_VIEW_DEVICE}"
  --max-model-train-jets "${PD10_MODEL_TRAIN_SIZE}"
  --max-model-val-jets "${PD10_MODEL_VAL_SIZE}"
  --max-final-test-jets "${PD10_FINAL_TEST_SIZE}"
  --confirm-final-test
)
fresh_append_flag_if_enabled cmd --overwrite "${OVERWRITE}"
fresh_append_flag_if_enabled cmd --no-skip-existing "${PD10_V2_REPRESENTATION_CACHE_NO_SKIP_EXISTING}"

fresh_write_run_config "${PD10_V2_TEACHER_REPRESENTATIONS_DIR}/${MODEL_NAME}" "pd10_v2_dual_view_representations" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${PD10_V2_TEACHER_REPRESENTATIONS_DIR}/${MODEL_NAME}/teacher_representation_manifest.json"
  fresh_require_file "${PD10_V2_TEACHER_REPRESENTATIONS_DIR}/${MODEL_NAME}/teacher_representation_cache_report.json"
  fresh_assert_json_ok "${PD10_V2_TEACHER_REPRESENTATIONS_DIR}/${MODEL_NAME}/teacher_representation_manifest.json"
  for split in "${split_args[@]}"; do
    fresh_require_file "${PD10_V2_TEACHER_REPRESENTATIONS_DIR}/${MODEL_NAME}/${split}_representations.npz"
    fresh_require_file "${PD10_V2_TEACHER_REPRESENTATIONS_DIR}/${MODEL_NAME}/${split}_representations_metadata.json"
  done
fi
