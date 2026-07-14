#!/usr/bin/env bash
# Materialize a provenance-explicit selected-tagger alias without copying weights.

#SBATCH --job-name=c2f_tagalias
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

SOURCE_NAME="${1:?Usage: sbatch run_alias_constrained_coarse_to_fine_tagger.sh <source> <alias>}"
ALIAS_NAME="${2:?Usage: sbatch run_alias_constrained_coarse_to_fine_tagger.sh <source> <alias>}"
: "${CONSTRAINED_C2F_ROOT:=${OUTPUT_ROOT}/constrained_coarse_to_fine}"
: "${CONSTRAINED_C2F_TAGGER_ROOT:=${CONSTRAINED_C2F_ROOT}/taggers}"
SOURCE_DIR="${CONSTRAINED_C2F_TAGGER_ROOT}/${SOURCE_NAME}"
OUTPUT_DIR="${CONSTRAINED_C2F_TAGGER_ROOT}/${ALIAS_NAME}"

fresh_setup "$@"
fresh_require_file "${SOURCE_DIR}/best_model_val.pt"
fresh_require_file "${SOURCE_DIR}/run_report.json"
fresh_claim_new_dir "${OUTPUT_DIR}"
cmd=(
  "${PYTHON_BIN}" -u scripts/alias_constrained_coarse_to_fine_tagger.py
  --source-dir "${SOURCE_DIR}"
  --alias-dir "${OUTPUT_DIR}"
  --source-name "${SOURCE_NAME}"
  --alias-name "${ALIAS_NAME}"
)
fresh_append_flag_if_enabled cmd --overwrite "${OVERWRITE}"
fresh_write_run_config "${OUTPUT_DIR}" "constrained_c2f_tagger_alias_${ALIAS_NAME}" "${cmd[@]}"
fresh_run "${cmd[@]}"
if ! fresh_is_dry_run; then
  fresh_require_file "${OUTPUT_DIR}/best_model_val.pt"
  fresh_require_file "${OUTPUT_DIR}/run_report.json"
fi
