#!/usr/bin/env bash
# Materialize an explicit prediction alias for a shared checkpoint row.

#SBATCH --job-name=c2f_alias
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

SOURCE_NAME="${1:?Usage: sbatch run_alias_constrained_coarse_to_fine_predictions.sh <source> <alias>}"
ALIAS_NAME="${2:?Usage: sbatch run_alias_constrained_coarse_to_fine_predictions.sh <source> <alias>}"
: "${CONSTRAINED_C2F_ROOT:=${OUTPUT_ROOT}/constrained_coarse_to_fine}"
: "${CONSTRAINED_C2F_PREDICTION_DIR:=${CONSTRAINED_C2F_ROOT}/predictions}"
: "${CONSTRAINED_C2F_PREDICT_SPLITS:=model_val stack_train stack_val final_test}"

OUTPUT_DIR="${CONSTRAINED_C2F_PREDICTION_DIR}/${ALIAS_NAME}"
fresh_setup "$@"
fresh_require_dir "${CONSTRAINED_C2F_PREDICTION_DIR}/${SOURCE_NAME}"
fresh_claim_new_dir "${OUTPUT_DIR}"
fresh_split_words split_args "${CONSTRAINED_C2F_PREDICT_SPLITS}"

cmd=(
  "${PYTHON_BIN}" -u scripts/cache_constrained_coarse_to_fine_prediction_alias.py
  --prediction-dir "${CONSTRAINED_C2F_PREDICTION_DIR}"
  --source-name "${SOURCE_NAME}"
  --alias-name "${ALIAS_NAME}"
  --splits "${split_args[@]}"
)
fresh_append_flag_if_enabled cmd --overwrite "${OVERWRITE}"
fresh_write_run_config "${OUTPUT_DIR}" "constrained_c2f_prediction_alias_${ALIAS_NAME}" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${OUTPUT_DIR}/prediction_run_report.json"
fi
