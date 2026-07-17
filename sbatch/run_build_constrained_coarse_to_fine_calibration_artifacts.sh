#!/usr/bin/env bash
# Build the manifest-bound C2F runtime-calibration caches in one ordered job.

#SBATCH --job-name=c2f_calibration
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=1-00:00:00
#SBATCH --mem=300G
#SBATCH --cpus-per-task=16

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${CONSTRAINED_C2F_PARENT_MANIFEST_PATH:?CONSTRAINED_C2F_PARENT_MANIFEST_PATH is required}"
: "${CONSTRAINED_C2F_CALIBRATION_ROOT:=${OUTPUT_ROOT}/constrained_c2f_runtime_calibration}"
: "${CONSTRAINED_C2F_CALIBRATION_MANIFEST_PATH:=${CONSTRAINED_C2F_CALIBRATION_ROOT}/calibration_manifest.json.gz}"
: "${CONSTRAINED_C2F_CALIBRATION_HLT_CACHE_DIR:=${CONSTRAINED_C2F_CALIBRATION_ROOT}/hlt_cache}"
: "${CONSTRAINED_C2F_CALIBRATION_OFFLINE_CACHE_DIR:=${CONSTRAINED_C2F_CALIBRATION_ROOT}/offline_cache}"
: "${CONSTRAINED_C2F_CALIBRATION_TARGET_CACHE_DIR:=${CONSTRAINED_C2F_CALIBRATION_ROOT}/targets}"
: "${CONSTRAINED_C2F_CALIBRATION_MODEL_TRAIN_SIZE:=50000}"
: "${CONSTRAINED_C2F_CALIBRATION_MODEL_VAL_SIZE:=30000}"
: "${CONSTRAINED_C2F_CALIBRATION_SEED:=81173}"
: "${CONSTRAINED_C2F_CALIBRATION_TARGET_CHUNK_SIZE:=8192}"
: "${CONSTRAINED_C2F_CALIBRATION_TARGET_DTYPE:=float32}"
: "${CONSTRAINED_C2F_CALIBRATION_READ_CHUNK_SIZE:=50000}"
: "${CONSTRAINED_C2F_DATA_DIR:=${PD10_DATA_DIR:-${DATA_DIR}}}"

fresh_setup "$@"
fresh_require_conda_python_package uproot
fresh_require_file "${CONSTRAINED_C2F_PARENT_MANIFEST_PATH}"
if ! fresh_is_dry_run && [[ ! -d "${CONSTRAINED_C2F_DATA_DIR}" ]]; then
  echo "Calibration data directory does not exist: ${CONSTRAINED_C2F_DATA_DIR}" >&2
  exit 2
fi
fresh_claim_new_dir "${CONSTRAINED_C2F_CALIBRATION_ROOT}"

manifest_cmd=(
  "${PYTHON_BIN}" scripts/build_constrained_coarse_to_fine_calibration_slice.py
  --parent-manifest "${CONSTRAINED_C2F_PARENT_MANIFEST_PATH}"
  --output-manifest "${CONSTRAINED_C2F_CALIBRATION_MANIFEST_PATH}"
  --report "${CONSTRAINED_C2F_CALIBRATION_ROOT}/calibration_slice_report.json"
  --model-train "${CONSTRAINED_C2F_CALIBRATION_MODEL_TRAIN_SIZE}"
  --model-val "${CONSTRAINED_C2F_CALIBRATION_MODEL_VAL_SIZE}"
  --seed "${CONSTRAINED_C2F_CALIBRATION_SEED}"
  --pretty
)
fresh_write_run_config "${CONSTRAINED_C2F_CALIBRATION_ROOT}" c2f_calibration_manifest "${manifest_cmd[@]}"
fresh_run "${manifest_cmd[@]}"

hlt_cmd=(
  "${PYTHON_BIN}" scripts/build_fixed_hlt_cache.py
  --manifest "${CONSTRAINED_C2F_CALIBRATION_MANIFEST_PATH}"
  --data-dir "${CONSTRAINED_C2F_DATA_DIR}"
  --cache-dir "${CONSTRAINED_C2F_CALIBRATION_HLT_CACHE_DIR}"
  --splits model_train model_val
  --read-chunk-size "${CONSTRAINED_C2F_CALIBRATION_READ_CHUNK_SIZE}"
  --hlt-profile fixed_hlt_v2_realistic
  --hlt-degradation-strength 2.5
)
fresh_write_run_config "${CONSTRAINED_C2F_CALIBRATION_HLT_CACHE_DIR}" c2f_calibration_hlt "${hlt_cmd[@]}"
fresh_run "${hlt_cmd[@]}"

offline_cmd=(
  "${PYTHON_BIN}" -u scripts/cache_architecture_view_offline_inputs.py
  --manifest-path "${CONSTRAINED_C2F_CALIBRATION_MANIFEST_PATH}"
  --output-dir "${CONSTRAINED_C2F_CALIBRATION_OFFLINE_CACHE_DIR}"
  --splits model_train model_val
  --data-dir "${CONSTRAINED_C2F_DATA_DIR}"
  --read-chunk-size "${CONSTRAINED_C2F_CALIBRATION_READ_CHUNK_SIZE}"
)
fresh_write_run_config "${CONSTRAINED_C2F_CALIBRATION_OFFLINE_CACHE_DIR}" c2f_calibration_offline "${offline_cmd[@]}"
fresh_run "${offline_cmd[@]}"

target_cmd=(
  "${PYTHON_BIN}" -u scripts/cache_constrained_hierarchy_targets.py
  --manifest "${CONSTRAINED_C2F_CALIBRATION_MANIFEST_PATH}"
  --hlt-cache-dir "${CONSTRAINED_C2F_CALIBRATION_HLT_CACHE_DIR}"
  --offline-cache-dir "${CONSTRAINED_C2F_CALIBRATION_OFFLINE_CACHE_DIR}"
  --output-cache-dir "${CONSTRAINED_C2F_CALIBRATION_TARGET_CACHE_DIR}"
  --splits model_train model_val
  --chunk-size "${CONSTRAINED_C2F_CALIBRATION_TARGET_CHUNK_SIZE}"
  --target-dtype "${CONSTRAINED_C2F_CALIBRATION_TARGET_DTYPE}"
)
fresh_write_run_config "${CONSTRAINED_C2F_CALIBRATION_TARGET_CACHE_DIR}" c2f_calibration_targets "${target_cmd[@]}"
fresh_run "${target_cmd[@]}"

validate_cmd=(
  "${PYTHON_BIN}" scripts/validate_constrained_coarse_to_fine_calibration_slice.py
  --parent-manifest "${CONSTRAINED_C2F_PARENT_MANIFEST_PATH}"
  --calibration-manifest "${CONSTRAINED_C2F_CALIBRATION_MANIFEST_PATH}"
  --hlt-cache-dir "${CONSTRAINED_C2F_CALIBRATION_HLT_CACHE_DIR}"
  --offline-cache-dir "${CONSTRAINED_C2F_CALIBRATION_OFFLINE_CACHE_DIR}"
  --target-cache-dir "${CONSTRAINED_C2F_CALIBRATION_TARGET_CACHE_DIR}"
  --output "${CONSTRAINED_C2F_CALIBRATION_ROOT}/calibration_validation.json"
)
fresh_run "${validate_cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${CONSTRAINED_C2F_CALIBRATION_MANIFEST_PATH}"
  fresh_require_file "${CONSTRAINED_C2F_CALIBRATION_ROOT}/calibration_slice_report.json"
  fresh_require_file "${CONSTRAINED_C2F_CALIBRATION_ROOT}/calibration_validation.json"
fi
