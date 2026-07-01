#!/usr/bin/env bash
# Run AV10 architecture-view ensemble/fusion from cached predictions.

#SBATCH --job-name=av10_fusion
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_ENV:=atlas_kd}"
export CONDA_ENV
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${ARCHITECTURE_VIEW_10CLASS_ROOT:=${OUTPUT_ROOT}/architecture_view_10class_hlt0p6}"
: "${ARCHITECTURE_VIEW_10CLASS_PREDICTION_ROOT:=${ARCHITECTURE_VIEW_10CLASS_ROOT}/prediction_cache}"
: "${ARCHITECTURE_VIEW_10CLASS_PREDICTION_DIR:=${ARCHITECTURE_VIEW_10CLASS_PREDICTION_ROOT}/predictions}"
: "${ARCHITECTURE_VIEW_10CLASS_FUSION_DIR:=${ARCHITECTURE_VIEW_10CLASS_ROOT}/fusion}"
: "${ARCHITECTURE_VIEW_10CLASS_VARIANTS:=av10_baseline_recheck av10_pn_context_to_part av10_pfn_context_to_part av10_pcnn_context_to_part av10_all_views_to_part av10_random_view_control av10_context_mlp_control}"
: "${ARCHITECTURE_VIEW_10CLASS_FUSION_MODES:=uniform_logit_mean uniform_probability_mean temperature_scaled_logit_mean scalar_weighted_logit_mean classwise_weighted_logit_mean ridge_logit_stacker binary_projection_weighted}"
: "${ARCHITECTURE_VIEW_10CLASS_FUSION_GROUPS:=}"
: "${ARCHITECTURE_VIEW_10CLASS_TEMPERATURE_GRID:=}"
: "${ARCHITECTURE_VIEW_10CLASS_C_GRID:=}"
: "${ARCHITECTURE_VIEW_10CLASS_SCALAR_WEIGHT_TRIALS:=256}"
: "${ARCHITECTURE_VIEW_10CLASS_BINARY_WEIGHT_TRIALS:=256}"
: "${ARCHITECTURE_VIEW_10CLASS_CLASSWISE_UNIFORM_MIX:=0.25}"
: "${ARCHITECTURE_VIEW_10CLASS_FUSION_CONTROL_SEED:=7207}"
: "${ARCHITECTURE_VIEW_10CLASS_CONFIRM_FINAL_TEST:=1}"

fresh_setup "$@"
fresh_require_file "scripts/run_architecture_view_10class_fusion.py"
fresh_split_words variant_args "${ARCHITECTURE_VIEW_10CLASS_VARIANTS}"
fresh_split_words mode_args "${ARCHITECTURE_VIEW_10CLASS_FUSION_MODES}"
for variant in "${variant_args[@]}"; do
  for split in stack_train stack_val final_test; do
    fresh_require_file "${ARCHITECTURE_VIEW_10CLASS_PREDICTION_DIR}/${variant}/${split}_logits.npz"
    fresh_require_file "${ARCHITECTURE_VIEW_10CLASS_PREDICTION_DIR}/${variant}/${split}_metadata.json"
  done
done
fresh_claim_new_dir "${ARCHITECTURE_VIEW_10CLASS_FUSION_DIR}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/run_architecture_view_10class_fusion.py"
  --prediction-dir "${ARCHITECTURE_VIEW_10CLASS_PREDICTION_DIR}"
  --output-dir "${ARCHITECTURE_VIEW_10CLASS_FUSION_DIR}"
  --model-names "${variant_args[@]}"
  --fusion-modes "${mode_args[@]}"
  --scalar-weight-trials "${ARCHITECTURE_VIEW_10CLASS_SCALAR_WEIGHT_TRIALS}"
  --binary-weight-trials "${ARCHITECTURE_VIEW_10CLASS_BINARY_WEIGHT_TRIALS}"
  --classwise-uniform-mix "${ARCHITECTURE_VIEW_10CLASS_CLASSWISE_UNIFORM_MIX}"
  --control-seed "${ARCHITECTURE_VIEW_10CLASS_FUSION_CONTROL_SEED}"
)
fresh_append_flag_if_enabled cmd --confirm-final-test "${ARCHITECTURE_VIEW_10CLASS_CONFIRM_FINAL_TEST}"
if [[ -n "${ARCHITECTURE_VIEW_10CLASS_TEMPERATURE_GRID}" ]]; then
  fresh_split_words temperature_grid_args "${ARCHITECTURE_VIEW_10CLASS_TEMPERATURE_GRID}"
  cmd+=(--temperature-grid "${temperature_grid_args[@]}")
fi
if [[ -n "${ARCHITECTURE_VIEW_10CLASS_C_GRID}" ]]; then
  fresh_split_words c_grid_args "${ARCHITECTURE_VIEW_10CLASS_C_GRID}"
  cmd+=(--c-grid "${c_grid_args[@]}")
fi
if [[ -n "${ARCHITECTURE_VIEW_10CLASS_FUSION_GROUPS}" ]]; then
  fresh_split_words group_args "${ARCHITECTURE_VIEW_10CLASS_FUSION_GROUPS}"
  for group in "${group_args[@]}"; do
    cmd+=(--group "${group}")
  done
fi

fresh_write_run_config "${ARCHITECTURE_VIEW_10CLASS_FUSION_DIR}" "architecture_view_10class_fusion" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${ARCHITECTURE_VIEW_10CLASS_FUSION_DIR}/fusion_report.json"
  fresh_require_file "${ARCHITECTURE_VIEW_10CLASS_FUSION_DIR}/fusion_metric_table.csv"
  fresh_require_file "${ARCHITECTURE_VIEW_10CLASS_FUSION_DIR}/run_report.json"
fi
