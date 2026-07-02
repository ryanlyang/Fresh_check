#!/usr/bin/env bash
#SBATCH --job-name=pd10_splits
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${ROOT_PATTERN:=*.root}"
: "${TREE_NAME:=tree}"
: "${MAX_CONSTITS:=128}"

DATA_DIR="${PD10_DATA_DIR}"

fresh_setup "$@"
fresh_require_data_dir
fresh_refuse_existing_path "${PD10_MANIFEST_PATH}"

cmd=(
  "${PYTHON_BIN}" "scripts/build_jetclass_splits.py"
  --data-dir "${PD10_DATA_DIR}"
  --out "${PD10_MANIFEST_PATH}"
  --pattern "${ROOT_PATTERN}"
  --tree-name "${TREE_NAME}"
  --max-constits "${MAX_CONSTITS}"
  --model-train "${PD10_MODEL_TRAIN_SIZE}"
  --model-val "${PD10_MODEL_VAL_SIZE}"
  --stack-train "${PD10_STACK_TRAIN_SIZE}"
  --stack-val "${PD10_STACK_VAL_SIZE}"
  --final-test "${PD10_FINAL_TEST_SIZE}"
  --pretty
)

fresh_write_run_config "${PD10_SPLIT_MANIFEST_DIR}" "pd10_build_splits" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${PD10_MANIFEST_PATH}"
fi
