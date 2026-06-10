#!/usr/bin/env bash
# Step 8 helper: fuse saved teacher-logit PN prediction blocks.

#SBATCH --job-name=tlogit_pn_fuse
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=05:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

fresh_setup "$@"
fresh_split_words arch_args "${TEACHER_LOGIT_PN_TEACHERS}"
if [[ "${#arch_args[@]}" -eq 0 ]]; then
  echo "TEACHER_LOGIT_PN_TEACHERS is empty" >&2
  exit 2
fi

model_names=()
for architecture in "${arch_args[@]}"; do
  model_name="$(fresh_teacher_logit_pn_model_name "${architecture}")"
  model_names+=("${model_name}")
  fresh_require_file "${TEACHER_LOGIT_PN_PREDICTION_DIR}/${model_name}/stack_train_predictions.npz"
  fresh_require_file "${TEACHER_LOGIT_PN_PREDICTION_DIR}/${model_name}/stack_val_predictions.npz"
  fresh_require_file "${TEACHER_LOGIT_PN_PREDICTION_DIR}/${model_name}/final_test_predictions.npz"
done
fresh_claim_new_dir "${TEACHER_LOGIT_PN_FUSION_DIR}"
fresh_split_words feature_mode_args "${TEACHER_LOGIT_PN_FEATURE_MODES}"
group_models="$(fresh_join_by_comma "${model_names[@]}")"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/run_independent_fusion_from_predictions.py"
  --prediction-dir "${TEACHER_LOGIT_PN_PREDICTION_DIR}"
  --output-dir "${TEACHER_LOGIT_PN_FUSION_DIR}"
  --model-names "${model_names[@]}"
  --group "teacher_logit_pn:${group_models}"
  --feature-modes "${feature_mode_args[@]}"
  --max-iter "${TEACHER_LOGIT_PN_MAX_ITER}"
  --control-seed "${TEACHER_LOGIT_PN_CONTROL_SEED}"
  --confirm-final-test
)
fresh_append_flag_if_enabled cmd --skip-controls "${TEACHER_LOGIT_PN_SKIP_CONTROLS}"
if [[ -n "${TEACHER_LOGIT_PN_C_GRID}" ]]; then
  fresh_split_words c_grid_args "${TEACHER_LOGIT_PN_C_GRID}"
  cmd+=(--c-grid "${c_grid_args[@]}")
fi

fresh_write_run_config "${TEACHER_LOGIT_PN_FUSION_DIR}" "fuse_teacher_logit_pn" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${TEACHER_LOGIT_PN_FUSION_DIR}/fusion_report.json"
  fresh_require_file "${TEACHER_LOGIT_PN_FUSION_DIR}/group_fusion_metrics.csv"
  fresh_require_file "${TEACHER_LOGIT_PN_FUSION_DIR}/singleton_stacker_metrics.csv"
  if ! fresh_bool_enabled "${TEACHER_LOGIT_PN_SKIP_CONTROLS}"; then
    fresh_require_file "${TEACHER_LOGIT_PN_FUSION_DIR}/controls.json"
  fi
fi
