#!/usr/bin/env bash
# Cache sharded constrained coarse-to-fine hierarchy targets.

#SBATCH --job-name=c2f_targets
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=2-00:00:00
#SBATCH --mem=300G
#SBATCH --cpus-per-task=16

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${CONSTRAINED_C2F_ROOT:=${OUTPUT_ROOT}/constrained_coarse_to_fine}"
: "${CONSTRAINED_C2F_MANIFEST_PATH:=${CONSTRAINED_C2F_ROOT}/inputs/split_manifest/split_manifest.json.gz}"
: "${CONSTRAINED_C2F_HLT_CACHE_DIR:=${CONSTRAINED_C2F_ROOT}/inputs/hlt_cache}"
: "${CONSTRAINED_C2F_OFFLINE_CACHE_DIR:=${CONSTRAINED_C2F_ROOT}/inputs/offline_cache}"
: "${CONSTRAINED_C2F_TARGET_CACHE_DIR:=${CONSTRAINED_C2F_ROOT}/targets}"
: "${CONSTRAINED_C2F_TARGET_SPLITS:=model_train model_val stack_val}"
: "${CONSTRAINED_C2F_TARGET_CHUNK_SIZE:=8192}"
: "${CONSTRAINED_C2F_TARGET_DTYPE:=float32}"
: "${CONSTRAINED_C2F_RADIAL_FIT_CHUNK_SIZE:=8192}"

fresh_setup "$@"
fresh_require_file "${CONSTRAINED_C2F_MANIFEST_PATH}"
fresh_require_dir "${CONSTRAINED_C2F_HLT_CACHE_DIR}"
fresh_require_dir "${CONSTRAINED_C2F_OFFLINE_CACHE_DIR}"
fresh_claim_new_dir "${CONSTRAINED_C2F_TARGET_CACHE_DIR}"
fresh_split_words split_args "${CONSTRAINED_C2F_TARGET_SPLITS}"

cmd=(
  "${PYTHON_BIN}" -u scripts/cache_constrained_hierarchy_targets.py
  --manifest "${CONSTRAINED_C2F_MANIFEST_PATH}"
  --hlt-cache-dir "${CONSTRAINED_C2F_HLT_CACHE_DIR}"
  --offline-cache-dir "${CONSTRAINED_C2F_OFFLINE_CACHE_DIR}"
  --output-cache-dir "${CONSTRAINED_C2F_TARGET_CACHE_DIR}"
  --splits "${split_args[@]}"
  --chunk-size "${CONSTRAINED_C2F_TARGET_CHUNK_SIZE}"
  --target-dtype "${CONSTRAINED_C2F_TARGET_DTYPE}"
  --radial-fit-chunk-size "${CONSTRAINED_C2F_RADIAL_FIT_CHUNK_SIZE}"
)
fresh_append_flag_if_enabled cmd --overwrite "${OVERWRITE}"

fresh_write_run_config "${CONSTRAINED_C2F_TARGET_CACHE_DIR}" constrained_c2f_targets "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${CONSTRAINED_C2F_TARGET_CACHE_DIR}/hierarchy_target_cache_manifest.json"
  for split in "${split_args[@]}"; do
    fresh_require_dir "${CONSTRAINED_C2F_TARGET_CACHE_DIR}/${split}_hierarchy_targets"
    fresh_require_file "${CONSTRAINED_C2F_TARGET_CACHE_DIR}/${split}_hierarchy_targets_metadata.json"
  done
fi
