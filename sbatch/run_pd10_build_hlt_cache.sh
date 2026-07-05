#!/usr/bin/env bash
#SBATCH --job-name=pd10_hlt_cache
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

DATA_DIR="${PD10_DATA_DIR}"

fresh_setup "$@"
fresh_require_data_dir
fresh_require_file "${PD10_MANIFEST_PATH}"
if [[ -d "${PD10_HLT_CACHE_DIR}" ]] && ! fresh_bool_enabled "${OVERWRITE}" && ! fresh_is_dry_run; then
  echo "Refusing to reuse existing PD10 HLT cache directory without OVERWRITE=1: ${PD10_HLT_CACHE_DIR}" >&2
  exit 2
fi

fresh_split_words split_args "${PD10_HLT_SPLITS}"
cmd=(
  "${PYTHON_BIN}" "scripts/build_fixed_hlt_cache.py"
  --manifest "${PD10_MANIFEST_PATH}"
  --data-dir "${PD10_DATA_DIR}"
  --cache-dir "${PD10_HLT_CACHE_DIR}"
  --splits "${split_args[@]}"
  --read-chunk-size "${READ_CHUNK_SIZE}"
  --hlt-profile "${PD10_HLT_PROFILE}"
  --hlt-degradation-strength "${PD10_HLT_DEGRADATION_STRENGTH}"
)
fresh_append_flag_if_enabled cmd --overwrite "${OVERWRITE}"
fresh_append_flag_if_enabled cmd --verify-label-branches "${VERIFY_LABEL_BRANCHES}"
fresh_append_flag_if_enabled cmd --show-progress "${SHOW_PROGRESS}"

fresh_write_run_config "${PD10_HLT_CACHE_DIR}" "pd10_build_fixed_hlt_cache" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  for split in "${split_args[@]}"; do
    fresh_require_file "${PD10_HLT_CACHE_DIR}/${split}_fixed_hlt.npz"
    fresh_require_file "${PD10_HLT_CACHE_DIR}/${split}_fixed_hlt_metadata.json"
  done
fi
