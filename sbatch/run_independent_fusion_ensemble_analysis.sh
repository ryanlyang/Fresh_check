#!/usr/bin/env bash
# Analyze diversity and uncertainty features from an existing independent-fusion run.

#SBATCH --job-name=ifuse_ens
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${RUN_ROOT:?Set RUN_ROOT to an existing run root containing predictions/, e.g. checkpoints/jetclass_fresh_independent_fusion_handoff/large_...}"
: "${ENSEMBLE_ANALYSIS_OUTPUT_DIR:=${RUN_ROOT}/ensemble_analysis}"
: "${ENSEMBLE_ANALYSIS_FEATURE_MODES:=uncertainty mean_uncertainty logits_probs_uncertainty}"
: "${ENSEMBLE_ANALYSIS_MAX_ITER:=2000}"
: "${ENSEMBLE_ANALYSIS_C_GRID:=}"
: "${ENSEMBLE_ANALYSIS_MODEL_NAMES:=}"

PREDICTION_DIR="${RUN_ROOT}/predictions"
DIVERSITY_DIR="${ENSEMBLE_ANALYSIS_OUTPUT_DIR}/diversity"
UNCERTAINTY_DIR="${ENSEMBLE_ANALYSIS_OUTPUT_DIR}/uncertainty_stackers"

fresh_setup "$@"
fresh_require_dir "${PREDICTION_DIR}"
fresh_claim_new_dir "${ENSEMBLE_ANALYSIS_OUTPUT_DIR}"

fresh_split_words feature_mode_args "${ENSEMBLE_ANALYSIS_FEATURE_MODES}"
fresh_split_words model_args "${ENSEMBLE_ANALYSIS_MODEL_NAMES}"

diversity_cmd=(
  "${PYTHON_BIN}" "-u" "scripts/run_diversity_audit.py"
  --prediction-dir "${PREDICTION_DIR}"
  --output-dir "${DIVERSITY_DIR}"
  --confirm-final-test
)

uncertainty_cmd=(
  "${PYTHON_BIN}" "-u" "scripts/run_uncertainty_feature_stacker.py"
  --prediction-dir "${PREDICTION_DIR}"
  --output-dir "${UNCERTAINTY_DIR}"
  --feature-modes "${feature_mode_args[@]}"
  --max-iter "${ENSEMBLE_ANALYSIS_MAX_ITER}"
  --confirm-final-test
)

if ((${#model_args[@]})); then
  diversity_cmd+=(--model-names "${model_args[@]}")
  uncertainty_cmd+=(--model-names "${model_args[@]}")
fi

if [[ -n "${ENSEMBLE_ANALYSIS_C_GRID}" ]]; then
  fresh_split_words c_grid_args "${ENSEMBLE_ANALYSIS_C_GRID}"
  uncertainty_cmd+=(--c-grid "${c_grid_args[@]}")
fi

fresh_write_run_config \
  "${ENSEMBLE_ANALYSIS_OUTPUT_DIR}" \
  "independent_fusion_ensemble_analysis" \
  "run_diversity_audit.py then run_uncertainty_feature_stacker.py"

fresh_run "${diversity_cmd[@]}"
fresh_run "${uncertainty_cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${DIVERSITY_DIR}/diversity_report.json"
  fresh_require_file "${DIVERSITY_DIR}/pairwise_diversity.csv"
  fresh_require_file "${DIVERSITY_DIR}/group_oracle_summary.csv"
  fresh_require_file "${UNCERTAINTY_DIR}/uncertainty_stacker_report.json"
  fresh_require_file "${UNCERTAINTY_DIR}/uncertainty_stacker_metrics.csv"
  fresh_require_file "${UNCERTAINTY_DIR}/feature_columns.json"
fi
