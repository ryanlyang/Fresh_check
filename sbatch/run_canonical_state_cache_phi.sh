#!/usr/bin/env bash
# Cache canonical Phi state tokens for HLT or offline source views.

#SBATCH --job-name=cstate_phi
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=1-00:00:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=8

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

SOURCE_VIEW="${1:?Usage: sbatch run_canonical_state_cache_phi.sh <hlt|offline>}"

: "${CANONICAL_STATE_ROOT:=${OUTPUT_ROOT}/canonical_multi_scale_jet_state}"
: "${CANONICAL_STATE_MANIFEST_PATH:=${CANONICAL_STATE_ROOT}/inputs/split_manifest.json.gz}"
: "${CANONICAL_STATE_HLT_CACHE_DIR:=${CANONICAL_STATE_ROOT}/inputs/hlt_cache}"
: "${CANONICAL_STATE_OFFLINE_CACHE_DIR:=${CANONICAL_STATE_ROOT}/inputs/offline_cache}"
: "${CANONICAL_STATE_PHI_CACHE_DIR:=${CANONICAL_STATE_ROOT}/phi_cache}"
: "${CANONICAL_STATE_PHI_HLT_SPLITS:=model_train model_val stack_train stack_val final_test}"
: "${CANONICAL_STATE_PHI_OFFLINE_SPLITS:=model_train model_val stack_train stack_val}"
: "${CANONICAL_STATE_ALLOW_FINAL_TEST_OFFLINE_ORACLE:=0}"
: "${CANONICAL_STATE_MODEL_TRAIN_SIZE:=5000000}"
: "${CANONICAL_STATE_MODEL_VAL_SIZE:=1000000}"
: "${CANONICAL_STATE_STACK_TRAIN_SIZE:=3000000}"
: "${CANONICAL_STATE_STACK_VAL_SIZE:=1000000}"
: "${CANONICAL_STATE_FINAL_TEST_SIZE:=1000000}"
: "${CANONICAL_STATE_PHI_CHUNK_SIZE:=32768}"

fresh_setup "$@"
fresh_require_file "${CANONICAL_STATE_MANIFEST_PATH}"
case "${SOURCE_VIEW}" in
  hlt|fixed_hlt|phi_hlt)
    input_cache_dir="${CANONICAL_STATE_HLT_CACHE_DIR}"
    output_cache_dir="${CANONICAL_STATE_PHI_CACHE_DIR}/hlt"
    fresh_split_words split_args "${CANONICAL_STATE_PHI_HLT_SPLITS}"
    ;;
  offline|off|phi_off|phi_offline)
    input_cache_dir="${CANONICAL_STATE_OFFLINE_CACHE_DIR}"
    output_cache_dir="${CANONICAL_STATE_PHI_CACHE_DIR}/offline"
    fresh_split_words split_args "${CANONICAL_STATE_PHI_OFFLINE_SPLITS}"
    ;;
  *)
    echo "Unknown canonical Phi source view: ${SOURCE_VIEW}" >&2
    exit 2
    ;;
esac
fresh_require_dir "${input_cache_dir}"

cmd=(
  "${PYTHON_BIN}" "-u" "scripts/cache_canonical_state_phi.py"
  --source-view "${SOURCE_VIEW}"
  --input-cache-dir "${input_cache_dir}"
  --output-cache-dir "${output_cache_dir}"
  --manifest "${CANONICAL_STATE_MANIFEST_PATH}"
  --splits "${split_args[@]}"
  --expected-model-train "${CANONICAL_STATE_MODEL_TRAIN_SIZE}"
  --expected-model-val "${CANONICAL_STATE_MODEL_VAL_SIZE}"
  --expected-stack-train "${CANONICAL_STATE_STACK_TRAIN_SIZE}"
  --expected-stack-val "${CANONICAL_STATE_STACK_VAL_SIZE}"
  --expected-final-test "${CANONICAL_STATE_FINAL_TEST_SIZE}"
  --chunk-size "${CANONICAL_STATE_PHI_CHUNK_SIZE}"
)
fresh_append_flag_if_enabled cmd --overwrite "${OVERWRITE}"
fresh_append_flag_if_enabled cmd --allow-final-test-offline-oracle "${CANONICAL_STATE_ALLOW_FINAL_TEST_OFFLINE_ORACLE}"

fresh_write_run_config "${output_cache_dir}" "canonical_state_cache_phi_${SOURCE_VIEW}" "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  for split in "${split_args[@]}"; do
    fresh_require_file "${output_cache_dir}/${split}_phi_${SOURCE_VIEW}.npz"
    fresh_require_file "${output_cache_dir}/${split}_phi_${SOURCE_VIEW}_metadata.json"
  done
fi
