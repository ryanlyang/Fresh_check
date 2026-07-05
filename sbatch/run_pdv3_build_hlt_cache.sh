#!/usr/bin/env bash
#SBATCH --job-name=pdv3_hlt_cache
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=2-00:00:00
#SBATCH --mem=220G
#SBATCH --cpus-per-task=8

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${READ_CHUNK_SIZE:=50000}"
: "${VERIFY_LABEL_BRANCHES:=0}"
: "${SHOW_PROGRESS:=0}"

fresh_setup "$@"
DATA_DIR="${PDV3_DATA_DIR}"
fresh_require_data_dir
fresh_require_file "${PDV3_MANIFEST_PATH}"
if [[ -d "${PDV3_HLT_CACHE_DIR}" ]] && ! fresh_bool_enabled "${OVERWRITE}" && ! fresh_is_dry_run; then
  echo "Refusing to reuse existing PDV3 HLT cache directory without OVERWRITE=1: ${PDV3_HLT_CACHE_DIR}" >&2
  exit 2
fi

fresh_split_words split_args "${PDV3_HLT_SPLITS}"
cmd=(
  "${PYTHON_BIN}" "scripts/build_fixed_hlt_cache.py"
  --manifest "${PDV3_MANIFEST_PATH}"
  --data-dir "${PDV3_DATA_DIR}"
  --cache-dir "${PDV3_HLT_CACHE_DIR}"
  --splits "${split_args[@]}"
  --read-chunk-size "${READ_CHUNK_SIZE}"
  --hlt-profile "${PDV3_HLT_PROFILE}"
  --hlt-degradation-strength "${PDV3_HLT_DEGRADATION_STRENGTH}"
)
fresh_append_flag_if_enabled cmd --overwrite "${OVERWRITE}"
fresh_append_flag_if_enabled cmd --verify-label-branches "${VERIFY_LABEL_BRANCHES}"
fresh_append_flag_if_enabled cmd --show-progress "${SHOW_PROGRESS}"

fresh_write_run_config "${PDV3_HLT_CACHE_DIR}" "pdv3_build_fixed_hlt_cache" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  for split in "${split_args[@]}"; do
    fresh_require_file "${PDV3_HLT_CACHE_DIR}/${split}_fixed_hlt.npz"
    fresh_require_file "${PDV3_HLT_CACHE_DIR}/${split}_fixed_hlt_metadata.json"
  done
fi
