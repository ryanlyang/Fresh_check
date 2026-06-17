#!/usr/bin/env bash
# Run neural conditional-evidence fusers anchored on HLT4.

#SBATCH --job-name=crossarch_cond_neural
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=debug
#SBATCH --time=06:00:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=8

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

fresh_setup "$@"
fresh_require_file "scripts/run_crossarch_conditional_evidence_fusers.py"

fresh_split_words hlt_arch_args "${CROSSARCH_HLT_BASELINE_ARCHITECTURES}"
fresh_split_words reco_args "${CROSSARCH_RECO_ARCHITECTURES}"
fresh_split_words teacher_args "${CROSSARCH_RECO_TEACHERS}"
fresh_split_words residual_penalty_args "${CROSSARCH_CONDITIONAL_FUSER_RESIDUAL_PENALTIES}"
fresh_split_words weight_decay_args "${CROSSARCH_CONDITIONAL_FUSER_WEIGHT_DECAYS}"
fresh_split_words hidden_dim_args "${CROSSARCH_CONDITIONAL_FUSER_NEURAL_HIDDEN_DIMS}"

hlt_model_args=()
for architecture in "${hlt_arch_args[@]}"; do
  hlt_model_args+=("$(fresh_crossarch_hlt_model_name "${architecture}")")
done

adapted_model_args=()
for reco_architecture in "${reco_args[@]}"; do
  for teacher_architecture in "${teacher_args[@]}"; do
    adapted_model_args+=("$(fresh_crossarch_reco_domain_tagger_model_name "${reco_architecture}" "${teacher_architecture}")")
  done
done

if ! fresh_is_dry_run; then
  for model_name in "${hlt_model_args[@]}" "${adapted_model_args[@]}"; do
    for split in stack_train stack_val final_test; do
      fresh_require_file "${CROSSARCH_PREDICTION_DIR}/${model_name}/${split}_predictions.npz"
      fresh_require_file "${CROSSARCH_PREDICTION_DIR}/${model_name}/${split}_predictions_metadata.json"
    done
  done
fi

OUTPUT_DIR="${CROSSARCH_CONDITIONAL_FUSER_DIR}/neural"
fresh_claim_new_dir "${OUTPUT_DIR}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/run_crossarch_conditional_evidence_fusers.py"
  --prediction-dir "${CROSSARCH_PREDICTION_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --suite neural
  --hlt-models "${hlt_model_args[@]}"
  --adapted-models "${adapted_model_args[@]}"
  --max-iter "${CROSSARCH_CONDITIONAL_FUSER_MAX_ITER}"
  --residual-penalties "${residual_penalty_args[@]}"
  --weight-decays "${weight_decay_args[@]}"
  --neural-epochs "${CROSSARCH_CONDITIONAL_FUSER_NEURAL_EPOCHS}"
  --neural-batch-size "${CROSSARCH_CONDITIONAL_FUSER_NEURAL_BATCH_SIZE}"
  --neural-lr "${CROSSARCH_CONDITIONAL_FUSER_NEURAL_LR}"
  --neural-hidden-dims "${hidden_dim_args[@]}"
  --neural-dropout "${CROSSARCH_CONDITIONAL_FUSER_NEURAL_DROPOUT}"
  --neural-device "${CROSSARCH_CONDITIONAL_FUSER_NEURAL_DEVICE}"
  --neural-patience "${CROSSARCH_CONDITIONAL_FUSER_NEURAL_PATIENCE}"
  --control-seed "${CROSSARCH_CONDITIONAL_FUSER_CONTROL_SEED}"
  --skip-controls
  --confirm-final-test
)
if [[ -n "${CROSSARCH_CONDITIONAL_FUSER_C_GRID}" ]]; then
  fresh_split_words c_grid_args "${CROSSARCH_CONDITIONAL_FUSER_C_GRID}"
  cmd+=(--c-grid "${c_grid_args[@]}")
fi

fresh_write_run_config "${OUTPUT_DIR}" "crossarch_conditional_evidence_fusers_neural" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${OUTPUT_DIR}/conditional_fuser_report.json"
  fresh_require_file "${OUTPUT_DIR}/method_summary.csv"
  fresh_require_file "${OUTPUT_DIR}/per_class_final_test.csv"
fi
