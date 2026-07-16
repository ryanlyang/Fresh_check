#!/usr/bin/env bash
# Generate deployable pseudo/logit predictions for declared splits.

#SBATCH --job-name=abph_prediction
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=3-00:00:00
#SBATCH --mem=300G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16

set -euo pipefail
IFS=$'\n\t'
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
source "${PROJECT_DIR}/sbatch/common.sh"
VARIANT="${1:?Usage: run_adaptive_binary_prediction.sh <variant> <splits...>}"; shift
(( $# > 0 )) || { echo "At least one prediction split is required" >&2; exit 2; }
: "${ABPH_ROOT:=${OUTPUT_ROOT}/adaptive_binary_pseudooffline}"
: "${ABPH_PREDICTION_EXECUTOR:=${PROJECT_DIR}/scripts/predict_adaptive_binary_pseudooffline.py}"
export PYTHONNOUSERSITE=1
fresh_setup
fresh_require_file "${ABPH_PREDICTION_EXECUTOR}"
if [[ " $* " == *" final_test "* ]]; then
  [[ "${ABPH_CONFIRM_FINAL_TEST:-0}" == "1" ]] || { echo "final_test prediction lacks approval" >&2; exit 2; }
  [[ -n "${ABPH_FINAL_CLAIM_CONTRACT:-}" ]] || { echo "final_test prediction lacks frozen claim contract" >&2; exit 2; }
  [[ -n "${ABPH_SELECTION_REPORT_PATH:-}" ]] || { echo "final_test prediction lacks selection report" >&2; exit 2; }
  "${PYTHON_BIN}" scripts/validate_adaptive_binary_orchestration.py final-claim \
    --path "${ABPH_FINAL_CLAIM_CONTRACT}" --selection-report "${ABPH_SELECTION_REPORT_PATH}" \
    --member "${VARIANT}"
fi
cmd=("${PYTHON_BIN}" -u "${ABPH_PREDICTION_EXECUTOR}" --variant "${VARIANT}" --campaign-root "${ABPH_ROOT}" --splits "$@" --device "${DEVICE}")
fresh_run "${cmd[@]}"
