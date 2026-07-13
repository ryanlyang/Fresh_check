#!/usr/bin/env bash
# Cache local per-particle residual-field targets.

#SBATCH --job-name=lprf_targets
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

: "${LOCAL_RESIDUAL_FIELD_ROOT:=${OUTPUT_ROOT}/local_particle_residual_field}"
: "${LOCAL_RESIDUAL_FIELD_MANIFEST_PATH:=${LOCAL_RESIDUAL_FIELD_ROOT}/inputs/split_manifest.json.gz}"
: "${LOCAL_RESIDUAL_FIELD_HLT_CACHE_DIR:=${LOCAL_RESIDUAL_FIELD_ROOT}/inputs/hlt_cache}"
: "${LOCAL_RESIDUAL_FIELD_OFFLINE_CACHE_DIR:=${LOCAL_RESIDUAL_FIELD_ROOT}/inputs/offline_cache}"
: "${LOCAL_RESIDUAL_FIELD_TARGET_CACHE_DIR:=${LOCAL_RESIDUAL_FIELD_ROOT}/targets}"
: "${LOCAL_RESIDUAL_FIELD_TARGET_SPLITS:=model_train model_val stack_val}"
: "${LOCAL_RESIDUAL_FIELD_TARGET_RADII:=0.02 0.05 0.10}"
: "${LOCAL_RESIDUAL_FIELD_TARGET_CHUNK_SIZE:=1024}"
: "${LOCAL_RESIDUAL_FIELD_TARGET_DTYPE:=float16}"
: "${LOCAL_RESIDUAL_FIELD_INCLUDE_FINAL_TEST_TARGETS:=0}"

fresh_setup "$@"
fresh_require_file "${LOCAL_RESIDUAL_FIELD_MANIFEST_PATH}"
fresh_require_dir "${LOCAL_RESIDUAL_FIELD_HLT_CACHE_DIR}"
fresh_require_dir "${LOCAL_RESIDUAL_FIELD_OFFLINE_CACHE_DIR}"
fresh_claim_new_dir "${LOCAL_RESIDUAL_FIELD_TARGET_CACHE_DIR}"
fresh_split_words split_args "${LOCAL_RESIDUAL_FIELD_TARGET_SPLITS}"
fresh_split_words radius_args "${LOCAL_RESIDUAL_FIELD_TARGET_RADII}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/cache_local_particle_residual_fields.py"
  --manifest "${LOCAL_RESIDUAL_FIELD_MANIFEST_PATH}"
  --hlt-cache-dir "${LOCAL_RESIDUAL_FIELD_HLT_CACHE_DIR}"
  --offline-cache-dir "${LOCAL_RESIDUAL_FIELD_OFFLINE_CACHE_DIR}"
  --output-cache-dir "${LOCAL_RESIDUAL_FIELD_TARGET_CACHE_DIR}"
  --splits "${split_args[@]}"
  --radii "${radius_args[@]}"
  --chunk-size "${LOCAL_RESIDUAL_FIELD_TARGET_CHUNK_SIZE}"
  --target-dtype "${LOCAL_RESIDUAL_FIELD_TARGET_DTYPE}"
)
fresh_append_flag_if_enabled cmd --include-final-test-targets "${LOCAL_RESIDUAL_FIELD_INCLUDE_FINAL_TEST_TARGETS}"
fresh_append_flag_if_enabled cmd --overwrite "${OVERWRITE}"

fresh_write_run_config "${LOCAL_RESIDUAL_FIELD_TARGET_CACHE_DIR}" "local_residual_field_targets" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  for split in "${split_args[@]}"; do
    fresh_require_file "${LOCAL_RESIDUAL_FIELD_TARGET_CACHE_DIR}/${split}_local_particle_residual_fields.npz"
    fresh_require_file "${LOCAL_RESIDUAL_FIELD_TARGET_CACHE_DIR}/${split}_local_particle_residual_fields_metadata.json"
  done
fi
