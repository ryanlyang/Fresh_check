#!/usr/bin/env bash
#SBATCH --job-name=fresh_build_hlt_cache
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=1-00:00:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=8

set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${HLT_SPLITS:=model_train model_val stack_train stack_val final_test}"
: "${READ_CHUNK_SIZE:=50000}"
: "${VERIFY_LABEL_BRANCHES:=0}"
: "${SHOW_PROGRESS:=0}"

fresh_setup "$@"
fresh_require_data_dir
fresh_require_file "${MANIFEST_PATH}"
if [[ -d "${HLT_CACHE_DIR}" ]] && ! fresh_bool_enabled "${OVERWRITE}" && ! fresh_is_dry_run; then
  echo "Refusing to reuse existing HLT cache directory without OVERWRITE=1: ${HLT_CACHE_DIR}" >&2
  exit 2
fi

fresh_split_words split_args "${HLT_SPLITS}"
cmd=(
  "${PYTHON_BIN}" "scripts/build_fixed_hlt_cache.py"
  --manifest "${MANIFEST_PATH}"
  --data-dir "${DATA_DIR}"
  --cache-dir "${HLT_CACHE_DIR}"
  --splits "${split_args[@]}"
  --read-chunk-size "${READ_CHUNK_SIZE}"
)
fresh_append_flag_if_enabled cmd --overwrite "${OVERWRITE}"
fresh_append_flag_if_enabled cmd --verify-label-branches "${VERIFY_LABEL_BRANCHES}"
fresh_append_flag_if_enabled cmd --show-progress "${SHOW_PROGRESS}"

fresh_write_run_config "${HLT_CACHE_DIR}" "build_fixed_hlt_cache" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  for split in "${split_args[@]}"; do
    fresh_require_file "${HLT_CACHE_DIR}/${split}_fixed_hlt.npz"
    fresh_require_file "${HLT_CACHE_DIR}/${split}_fixed_hlt_metadata.json"
  done
fi
