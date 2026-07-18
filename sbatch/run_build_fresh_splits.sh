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

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
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
: "${SKIP_UNREADABLE_ROOT_FILES:=0}"
: "${DATA_DIRS:=}"

fresh_setup "$@"
data_dir_args=()
if [[ -n "${DATA_DIRS}" ]]; then
  fresh_split_words data_dir_args "${DATA_DIRS}"
else
  fresh_require_data_dir
  data_dir_args=("${DATA_DIR}")
fi
if ! fresh_is_dry_run; then
  for data_dir_arg in "${data_dir_args[@]}"; do
    if [[ ! -d "${data_dir_arg}" ]]; then
      echo "JetClass data directory does not exist on this machine: ${data_dir_arg}" >&2
      exit 2
    fi
  done
fi
fresh_refuse_existing_path "${MANIFEST_PATH}"

cmd=(
  "${PYTHON_BIN}" "scripts/build_jetclass_splits.py"
  --data-dir "${data_dir_args[@]}"
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
fresh_append_flag_if_enabled cmd --skip-unreadable-files "${SKIP_UNREADABLE_ROOT_FILES}"

fresh_write_run_config "$(dirname "${MANIFEST_PATH}")" "build_splits" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${MANIFEST_PATH}"
fi
