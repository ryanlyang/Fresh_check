#!/usr/bin/env bash
# Build fresh balanced splits after selecting labels directly from JetClass files.

#SBATCH --job-name=build_label_fresh
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${LABEL_FILTER_OUTPUT_MANIFEST_PATH:?Set LABEL_FILTER_OUTPUT_MANIFEST_PATH}"
: "${LABEL_FILTER_NAMES:?Set LABEL_FILTER_NAMES, e.g. 'QCD Tbqq'}"
: "${LABEL_FILTER_MODEL_TRAIN_SIZE:=${SET_MATCHING_MODEL_TRAIN_SIZE:-500000}}"
: "${LABEL_FILTER_MODEL_VAL_SIZE:=${SET_MATCHING_MODEL_VAL_SIZE:-150000}}"
: "${LABEL_FILTER_STACK_TRAIN_SIZE:=${SET_MATCHING_STACK_TRAIN_SIZE:-500000}}"
: "${LABEL_FILTER_STACK_VAL_SIZE:=${SET_MATCHING_STACK_VAL_SIZE:-150000}}"
: "${LABEL_FILTER_FINAL_TEST_SIZE:=${SET_MATCHING_FINAL_TEST_SIZE:-500000}}"
: "${LABEL_FILTER_ROOT_PATTERN:=*.root}"
: "${LABEL_FILTER_TREE_NAME:=tree}"
: "${LABEL_FILTER_MAX_CONSTITS:=128}"
: "${LABEL_FILTER_PRETTY:=0}"
: "${LABEL_FILTER_BASE_SEED:=52}"

fresh_setup "$@"
fresh_require_data_dir
fresh_refuse_existing_path "${LABEL_FILTER_OUTPUT_MANIFEST_PATH}"

OUTPUT_DIR="$(dirname "${LABEL_FILTER_OUTPUT_MANIFEST_PATH}")"
REPORT_PATH="${OUTPUT_DIR}/filtered_manifest_report.json"
fresh_split_words label_args "${LABEL_FILTER_NAMES}"

cmd=(
  "${PYTHON_BIN}" "scripts/build_label_filtered_fresh_splits.py"
  --data-dir "${DATA_DIR}"
  --output-manifest "${LABEL_FILTER_OUTPUT_MANIFEST_PATH}"
  --label-names "${label_args[@]}"
  --output-report "${REPORT_PATH}"
  --pattern "${LABEL_FILTER_ROOT_PATTERN}"
  --tree-name "${LABEL_FILTER_TREE_NAME}"
  --max-constits "${LABEL_FILTER_MAX_CONSTITS}"
  --model-train "${LABEL_FILTER_MODEL_TRAIN_SIZE}"
  --model-val "${LABEL_FILTER_MODEL_VAL_SIZE}"
  --stack-train "${LABEL_FILTER_STACK_TRAIN_SIZE}"
  --stack-val "${LABEL_FILTER_STACK_VAL_SIZE}"
  --final-test "${LABEL_FILTER_FINAL_TEST_SIZE}"
  --base-seed "${LABEL_FILTER_BASE_SEED}"
)
fresh_append_flag_if_enabled cmd --pretty "${LABEL_FILTER_PRETTY}"

fresh_write_run_config "${OUTPUT_DIR}" "build_label_filtered_fresh_splits" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${LABEL_FILTER_OUTPUT_MANIFEST_PATH}"
  fresh_require_file "${REPORT_PATH}"
fi
