#!/usr/bin/env bash
# Score one exact pseudo-source family and persist only fusion-ready logits.
#SBATCH --job-name=abph_score_bundle
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tigris
#SBATCH --time=2-00:00:00
#SBATCH --mem=300G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16

set -euo pipefail
IFS=$'\n\t'
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
source "${PROJECT_DIR}/sbatch/common.sh"
: "${ABPH_ROOT:=${OUTPUT_ROOT}/adaptive_binary_pseudooffline}"
: "${ABPH_PREDICTION_EXECUTOR:=${PROJECT_DIR}/scripts/predict_adaptive_binary_pseudooffline.py}"
: "${ABPH_SCORING_SPLITS:=model_val stack_train stack_val}"
(( $# > 0 )) || { echo "Usage: run_adaptive_binary_bundled_scoring.sh <member...>" >&2; exit 2; }
members=("$@")
read -r -a splits <<< "${ABPH_SCORING_SPLITS}"
[[ " ${splits[*]} " != *" final_test "* ]] || {
  echo "bundled scoring is forbidden on final_test" >&2
  exit 2
}
export PYTHONNOUSERSITE=1
fresh_setup
fresh_require_file "${ABPH_PREDICTION_EXECUTOR}"
cmd=("${PYTHON_BIN}" -u "${ABPH_PREDICTION_EXECUTOR}"
  --members "${members[@]}"
  --campaign-root "${ABPH_ROOT}"
  --splits "${splits[@]}"
  --device "${DEVICE}")
fresh_run bash "${PROJECT_DIR}/sbatch/run_with_adaptive_binary_ram_workspace.sh" "${cmd[@]}"
