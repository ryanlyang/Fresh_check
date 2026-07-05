#!/usr/bin/env bash
#SBATCH --job-name=pdv3_offcache
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=1-00:00:00
#SBATCH --mem=180G
#SBATCH --cpus-per-task=8

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${READ_CHUNK_SIZE:=50000}"
: "${VERIFY_LABEL_BRANCHES:=0}"
: "${TREE_NAME:=tree}"
: "${MAX_CONSTITS:=128}"

fresh_setup "$@"
DATA_DIR="${PDV3_DATA_DIR}"
fresh_require_data_dir
fresh_require_file "${PDV3_MANIFEST_PATH}"
if [[ -d "${PDV3_OFFLINE_CACHE_DIR}" ]] && ! fresh_bool_enabled "${OVERWRITE}" && ! fresh_is_dry_run; then
  echo "Refusing to reuse existing PDV3 offline cache directory without OVERWRITE=1: ${PDV3_OFFLINE_CACHE_DIR}" >&2
  exit 2
fi

fresh_split_words split_args "${PDV3_OFFLINE_SPLITS}"
cmd=(
  "${PYTHON_BIN}" "-u" "scripts/cache_architecture_view_offline_inputs.py"
  --manifest-path "${PDV3_MANIFEST_PATH}"
  --output-dir "${PDV3_OFFLINE_CACHE_DIR}"
  --splits "${split_args[@]}"
  --data-dir "${PDV3_DATA_DIR}"
  --tree-name "${TREE_NAME}"
  --max-constits "${MAX_CONSTITS}"
  --read-chunk-size "${READ_CHUNK_SIZE}"
)
fresh_append_flag_if_enabled cmd --overwrite "${OVERWRITE}"
fresh_append_flag_if_enabled cmd --verify-label-branches "${VERIFY_LABEL_BRANCHES}"

fresh_write_run_config "${PDV3_OFFLINE_CACHE_DIR}" "pdv3_cache_offline_inputs" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${PDV3_OFFLINE_CACHE_DIR}/offline_cache_report.json"
  for split in "${split_args[@]}"; do
    fresh_require_file "${PDV3_OFFLINE_CACHE_DIR}/${split}_offline.npz"
    fresh_require_file "${PDV3_OFFLINE_CACHE_DIR}/${split}_offline_metadata.json"
  done
fi
