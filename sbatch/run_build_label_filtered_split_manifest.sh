#!/usr/bin/env bash
#SBATCH --job-name=build_label_manifest
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=debug
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${LABEL_FILTER_SOURCE_MANIFEST_PATH:=${MANIFEST_PATH}}"
: "${LABEL_FILTER_OUTPUT_MANIFEST_PATH:?Set LABEL_FILTER_OUTPUT_MANIFEST_PATH}"
: "${LABEL_FILTER_NAMES:?Set LABEL_FILTER_NAMES, e.g. 'QCD Hbb'}"
: "${LABEL_FILTER_PRETTY:=0}"

fresh_setup "$@"
fresh_require_file "${LABEL_FILTER_SOURCE_MANIFEST_PATH}"

OUTPUT_DIR="$(dirname "${LABEL_FILTER_OUTPUT_MANIFEST_PATH}")"
REPORT_PATH="${OUTPUT_DIR}/filtered_manifest_report.json"
fresh_split_words label_args "${LABEL_FILTER_NAMES}"

cmd=(
  "${PYTHON_BIN}" "scripts/build_label_filtered_split_manifest.py"
  --source-manifest "${LABEL_FILTER_SOURCE_MANIFEST_PATH}"
  --output-manifest "${LABEL_FILTER_OUTPUT_MANIFEST_PATH}"
  --label-names "${label_args[@]}"
  --output-report "${REPORT_PATH}"
)
fresh_append_flag_if_enabled cmd --pretty "${LABEL_FILTER_PRETTY}"

fresh_write_run_config "${OUTPUT_DIR}" "build_label_filtered_split_manifest" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${LABEL_FILTER_OUTPUT_MANIFEST_PATH}"
  fresh_require_file "${REPORT_PATH}"
fi
