#!/usr/bin/env bash
#SBATCH --job-name=build_binary_hlt
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${LABEL_FILTER_MANIFEST_PATH:?Set LABEL_FILTER_MANIFEST_PATH}"
: "${LABEL_FILTER_HLT_CACHE_DIR:?Set LABEL_FILTER_HLT_CACHE_DIR}"
: "${LABEL_FILTER_HLT_SPLITS:=model_train model_val stack_train stack_val final_test}"
: "${LABEL_FILTER_READ_CHUNK_SIZE:=50000}"
: "${LABEL_FILTER_VERIFY_LABEL_BRANCHES:=0}"
: "${LABEL_FILTER_SHOW_PROGRESS:=0}"

fresh_setup "$@"
fresh_require_data_dir
fresh_require_file "${LABEL_FILTER_MANIFEST_PATH}"
if [[ -d "${LABEL_FILTER_HLT_CACHE_DIR}" ]] && ! fresh_bool_enabled "${OVERWRITE}" && ! fresh_is_dry_run; then
  echo "Refusing to reuse existing label-filtered HLT cache directory without OVERWRITE=1: ${LABEL_FILTER_HLT_CACHE_DIR}" >&2
  exit 2
fi

fresh_split_words split_args "${LABEL_FILTER_HLT_SPLITS}"
cmd=(
  "${PYTHON_BIN}" "scripts/build_fixed_hlt_cache.py"
  --manifest "${LABEL_FILTER_MANIFEST_PATH}"
  --data-dir "${DATA_DIR}"
  --cache-dir "${LABEL_FILTER_HLT_CACHE_DIR}"
  --splits "${split_args[@]}"
  --read-chunk-size "${LABEL_FILTER_READ_CHUNK_SIZE}"
)
fresh_append_flag_if_enabled cmd --overwrite "${OVERWRITE}"
fresh_append_flag_if_enabled cmd --verify-label-branches "${LABEL_FILTER_VERIFY_LABEL_BRANCHES}"
fresh_append_flag_if_enabled cmd --show-progress "${LABEL_FILTER_SHOW_PROGRESS}"

fresh_write_run_config "${LABEL_FILTER_HLT_CACHE_DIR}" "build_label_filtered_hlt_cache" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  for split in "${split_args[@]}"; do
    fresh_require_file "${LABEL_FILTER_HLT_CACHE_DIR}/${split}_fixed_hlt.npz"
    fresh_require_file "${LABEL_FILTER_HLT_CACHE_DIR}/${split}_fixed_hlt_metadata.json"
  done
fi
