#!/usr/bin/env bash
# Fit/evaluate the frozen-prediction F0-F5 fusion groups.

#SBATCH --job-name=c2f_fusion
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=08:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${CONSTRAINED_C2F_ROOT:=${OUTPUT_ROOT}/constrained_coarse_to_fine}"
: "${CONSTRAINED_C2F_PREDICTION_DIR:=${CONSTRAINED_C2F_ROOT}/predictions}"
: "${CONSTRAINED_C2F_FUSION_DIR:=${CONSTRAINED_C2F_ROOT}/fusion}"
: "${CONSTRAINED_C2F_FUSION_GROUPS:=F0:mean_logits:A0,BEST_D F1:simplex_logits:A0,BEST_D F2:representation_stacker:D3,D4,D5,D6,D8 F3:simplex_logits:A0,BEST_D F4:mean_logits:BEST_D,BEST_D_SEED1,BEST_D_SEED2 F5:linear_stacker:D8,D6,BEST_D,BEST_D_SEED1,BEST_D_SEED2}"
: "${CONSTRAINED_C2F_REQUIRED_FUSION_GROUPS:=F0 F1 F2 F3 F4 F5}"

fresh_setup "$@"
fresh_require_dir "${CONSTRAINED_C2F_PREDICTION_DIR}"
fresh_claim_new_dir "${CONSTRAINED_C2F_FUSION_DIR}"
fresh_split_words group_args "${CONSTRAINED_C2F_FUSION_GROUPS}"
fresh_split_words required_args "${CONSTRAINED_C2F_REQUIRED_FUSION_GROUPS}"

cmd=(
  "${PYTHON_BIN}" -u scripts/run_constrained_coarse_to_fine_fusion.py
  --prediction-dir "${CONSTRAINED_C2F_PREDICTION_DIR}"
  --output-dir "${CONSTRAINED_C2F_FUSION_DIR}"
  --required-groups "${required_args[@]}"
)
for group in "${group_args[@]}"; do cmd+=(--group "${group}"); done
fresh_append_flag_if_enabled cmd --overwrite-predictions "${OVERWRITE}"
fresh_append_flag_if_enabled cmd --confirm-final-test "${CONFIRM_FINAL_TEST}"

fresh_write_run_config "${CONSTRAINED_C2F_FUSION_DIR}" constrained_c2f_fusion "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  if fresh_bool_enabled "${CONFIRM_FINAL_TEST}"; then
    fresh_require_file "${CONSTRAINED_C2F_FUSION_DIR}/fusion_final_claim_report.json"
  else
    fresh_require_file "${CONSTRAINED_C2F_FUSION_DIR}/fusion_report.json"
  fi
  fresh_require_file "${CONSTRAINED_C2F_FUSION_DIR}/fusion_metrics.csv"
fi
