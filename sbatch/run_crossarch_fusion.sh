#!/usr/bin/env bash
# Fit cross-architecture fusion groups, fusers, controls, and audits.

#SBATCH --job-name=crossarch_fusion
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=1-00:00:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=8

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

fresh_setup "$@"
fresh_require_file "scripts/run_crossarch_fusion.py"

fresh_split_words reco_args "${CROSSARCH_RECO_ARCHITECTURES}"
fresh_split_words teacher_args "${CROSSARCH_RECO_TEACHERS}"
fresh_split_words hlt_arch_args "${CROSSARCH_HLT_BASELINE_ARCHITECTURES}"
fresh_split_words split_args "${CROSSARCH_RECO_PREDICT_SPLITS}"
fresh_split_words feature_mode_args "${CROSSARCH_FUSION_FEATURE_MODES}"
fresh_split_words fuser_args "${CROSSARCH_FUSERS}"
fresh_split_words control_feature_mode_args "${CROSSARCH_FUSION_CONTROL_FEATURE_MODES}"

if ! fresh_is_dry_run; then
  for architecture in "${hlt_arch_args[@]}"; do
    model_name="$(fresh_crossarch_hlt_model_name "${architecture}")"
    for split in "${split_args[@]}"; do
      fresh_require_file "${CROSSARCH_PREDICTION_DIR}/${model_name}/${split}_predictions.npz"
      fresh_require_file "${CROSSARCH_PREDICTION_DIR}/${model_name}/${split}_predictions_metadata.json"
    done
  done
  for reco_architecture in "${reco_args[@]}"; do
    for teacher_architecture in "${teacher_args[@]}"; do
      model_name="$(fresh_crossarch_reco_model_name "${reco_architecture}" "${teacher_architecture}")"
      for split in "${split_args[@]}"; do
        fresh_require_file "${CROSSARCH_PREDICTION_DIR}/${model_name}/${split}_predictions.npz"
        fresh_require_file "${CROSSARCH_PREDICTION_DIR}/${model_name}/${split}_predictions_metadata.json"
      done
    done
  done
fi

fresh_claim_new_dir "${CROSSARCH_FUSION_DIR}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/run_crossarch_fusion.py"
  --fit-fusers
  --prediction-dir "${CROSSARCH_PREDICTION_DIR}"
  --output-dir "${CROSSARCH_FUSION_DIR}"
  --splits stack_train stack_val final_test
  --feature-modes "${feature_mode_args[@]}"
  --fusers "${fuser_args[@]}"
  --max-iter "${CROSSARCH_FUSION_MAX_ITER}"
  --min-bin-train-rows "${CROSSARCH_FUSION_MIN_BIN_TRAIN_ROWS}"
  --control-seed "${CROSSARCH_FUSION_CONTROL_SEED}"
  --control-feature-modes "${control_feature_mode_args[@]}"
  --control-warning-min-accuracy "${CROSSARCH_FUSION_CONTROL_WARNING_MIN_ACCURACY}"
  --control-warning-chance-margin "${CROSSARCH_FUSION_CONTROL_WARNING_CHANCE_MARGIN}"
  --confirm-final-test
)
fresh_append_flag_if_enabled cmd --include-optional-groups "${CROSSARCH_FUSION_INCLUDE_OPTIONAL_GROUPS}"
fresh_append_flag_if_enabled cmd --skip-controls "${CROSSARCH_FUSION_SKIP_CONTROLS}"
if [[ -n "${CROSSARCH_FUSION_C_GRID}" ]]; then
  fresh_split_words c_grid_args "${CROSSARCH_FUSION_C_GRID}"
  cmd+=(--c-grid "${c_grid_args[@]}")
fi

fresh_write_run_config "${CROSSARCH_FUSION_DIR}" "crossarch_fusion" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${CROSSARCH_FUSION_DIR}/fusion_report.json"
  fresh_assert_json_ok "${CROSSARCH_FUSION_DIR}/fusion_report.json"
fi
