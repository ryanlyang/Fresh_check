#!/usr/bin/env bash
# Cache offline JetClass views for AV10 offline-transfer runs.

#SBATCH --job-name=av10_offcache
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=1-00:00:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=8

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_ENV:=atlas_kd}"
export CONDA_ENV
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${ARCHITECTURE_VIEW_10CLASS_OFFLINE_ROOT:=${OUTPUT_ROOT}/architecture_view_10class_offline_transfer}"
: "${ARCHITECTURE_VIEW_10CLASS_OFFLINE_MANIFEST_PATH:=${ARCHITECTURE_VIEW_10CLASS_OFFLINE_ROOT}/inputs/split_manifest.json.gz}"
: "${ARCHITECTURE_VIEW_10CLASS_OFFLINE_CACHE_DIR:=${ARCHITECTURE_VIEW_10CLASS_OFFLINE_ROOT}/inputs/offline_cache}"
: "${ARCHITECTURE_VIEW_10CLASS_OFFLINE_SPLITS:=model_train model_val stack_val final_test}"
: "${ARCHITECTURE_VIEW_10CLASS_OFFLINE_DATA_DIRS:=}"
: "${ARCHITECTURE_VIEW_10CLASS_OFFLINE_TREE_NAME:=tree}"
: "${ARCHITECTURE_VIEW_10CLASS_OFFLINE_MAX_CONSTITS:=}"
: "${ARCHITECTURE_VIEW_10CLASS_OFFLINE_READ_CHUNK_SIZE:=50000}"
: "${ARCHITECTURE_VIEW_10CLASS_OFFLINE_VERIFY_LABEL_BRANCHES:=0}"
: "${ARCHITECTURE_VIEW_10CLASS_OFFLINE_OVERWRITE:=0}"

fresh_setup "$@"
fresh_require_file "scripts/cache_architecture_view_offline_inputs.py"
fresh_require_file "${ARCHITECTURE_VIEW_10CLASS_OFFLINE_MANIFEST_PATH}"
fresh_claim_new_dir "${ARCHITECTURE_VIEW_10CLASS_OFFLINE_CACHE_DIR}"
fresh_split_words split_args "${ARCHITECTURE_VIEW_10CLASS_OFFLINE_SPLITS}"
fresh_split_words data_dir_args "${ARCHITECTURE_VIEW_10CLASS_OFFLINE_DATA_DIRS}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/cache_architecture_view_offline_inputs.py"
  --manifest-path "${ARCHITECTURE_VIEW_10CLASS_OFFLINE_MANIFEST_PATH}"
  --output-dir "${ARCHITECTURE_VIEW_10CLASS_OFFLINE_CACHE_DIR}"
  --splits "${split_args[@]}"
  --tree-name "${ARCHITECTURE_VIEW_10CLASS_OFFLINE_TREE_NAME}"
  --read-chunk-size "${ARCHITECTURE_VIEW_10CLASS_OFFLINE_READ_CHUNK_SIZE}"
)
if ((${#data_dir_args[@]})); then
  cmd+=(--data-dir "${data_dir_args[@]}")
fi
fresh_append_optional_arg cmd --max-constits "${ARCHITECTURE_VIEW_10CLASS_OFFLINE_MAX_CONSTITS}"
fresh_append_flag_if_enabled cmd --verify-label-branches "${ARCHITECTURE_VIEW_10CLASS_OFFLINE_VERIFY_LABEL_BRANCHES}"
fresh_append_flag_if_enabled cmd --overwrite "${ARCHITECTURE_VIEW_10CLASS_OFFLINE_OVERWRITE}"

fresh_write_run_config "${ARCHITECTURE_VIEW_10CLASS_OFFLINE_CACHE_DIR}" "architecture_view_10class_offline_cache" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${ARCHITECTURE_VIEW_10CLASS_OFFLINE_CACHE_DIR}/offline_cache_report.json"
  for split in "${split_args[@]}"; do
    fresh_require_file "${ARCHITECTURE_VIEW_10CLASS_OFFLINE_CACHE_DIR}/${split}_offline.npz"
    fresh_require_file "${ARCHITECTURE_VIEW_10CLASS_OFFLINE_CACHE_DIR}/${split}_offline_metadata.json"
  done
fi
