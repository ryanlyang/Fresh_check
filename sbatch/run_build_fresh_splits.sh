#!/usr/bin/env bash
#SBATCH --job-name=fresh_build_splits
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${MODEL_TRAIN_SIZE:=500000}"
: "${MODEL_VAL_SIZE:=150000}"
: "${STACK_TRAIN_SIZE:=250000}"
: "${STACK_VAL_SIZE:=50000}"
: "${FINAL_TEST_SIZE:=500000}"
: "${ROOT_PATTERN:=*.root}"
: "${TREE_NAME:=tree}"
: "${MAX_CONSTITS:=128}"

fresh_setup "$@"
fresh_require_data_dir
fresh_refuse_existing_path "${MANIFEST_PATH}"

cmd=(
  "${PYTHON_BIN}" "scripts/build_jetclass_splits.py"
  --data-dir "${DATA_DIR}"
  --out "${MANIFEST_PATH}"
  --pattern "${ROOT_PATTERN}"
  --tree-name "${TREE_NAME}"
  --max-constits "${MAX_CONSTITS}"
  --model-train "${MODEL_TRAIN_SIZE}"
  --model-val "${MODEL_VAL_SIZE}"
  --stack-train "${STACK_TRAIN_SIZE}"
  --stack-val "${STACK_VAL_SIZE}"
  --final-test "${FINAL_TEST_SIZE}"
  --pretty
)

fresh_write_run_config "$(dirname "${MANIFEST_PATH}")" "build_splits" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${MANIFEST_PATH}"
fi

