#!/usr/bin/env bash
# Build or audit ABPH split/cache inputs.

#SBATCH --job-name=abph_inputs
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=2-00:00:00
#SBATCH --mem=220G
#SBATCH --cpus-per-task=16

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
source "${PROJECT_DIR}/sbatch/common.sh"

ACTION="${1:?Usage: run_adaptive_binary_inputs.sh <splits|hlt_cache|offline_cache|audit>}"
: "${ABPH_ROOT:=${OUTPUT_ROOT}/adaptive_binary_pseudooffline}"
: "${ABPH_CAMPAIGN_MODE:=pilot}"
: "${ABPH_DATA_DIR:=/home/ryreu/atlas/PracticeTagging/data/jetclass_part1}"
: "${ABPH_MANIFEST_PATH:=${ABPH_ROOT}/inputs/split_manifest/split_manifest.json.gz}"
: "${ABPH_HLT_CACHE_DIR:=${ABPH_ROOT}/inputs/hlt_cache}"
: "${ABPH_OFFLINE_CACHE_DIR:=${ABPH_ROOT}/inputs/offline_cache}"

case "${ABPH_CAMPAIGN_MODE}" in
  pilot) sizes=(500000 150000 300000 150000 150000) ;;
  highdata) sizes=(5000000 1000000 2000000 1000000 1000000) ;;
  *) echo "Unknown ABPH_CAMPAIGN_MODE=${ABPH_CAMPAIGN_MODE}" >&2; exit 2 ;;
esac

export PYTHONNOUSERSITE=1
case "${ACTION}" in
  splits)
    export DATA_DIR="${ABPH_DATA_DIR}" MANIFEST_PATH="${ABPH_MANIFEST_PATH}"
    export MODEL_TRAIN_SIZE="${sizes[0]}" MODEL_VAL_SIZE="${sizes[1]}"
    export STACK_TRAIN_SIZE="${sizes[2]}" STACK_VAL_SIZE="${sizes[3]}" FINAL_TEST_SIZE="${sizes[4]}"
    exec bash "${PROJECT_DIR}/sbatch/run_build_fresh_splits.sh"
    ;;
  hlt_cache)
    export DATA_DIR="${ABPH_DATA_DIR}" MANIFEST_PATH="${ABPH_MANIFEST_PATH}" HLT_CACHE_DIR="${ABPH_HLT_CACHE_DIR}"
    export HLT_SPLITS="model_train model_val stack_train stack_val final_test"
    export HLT_PROFILE=fixed_hlt_v2_realistic HLT_DEGRADATION_STRENGTH=2.5
    exec bash "${PROJECT_DIR}/sbatch/run_build_fresh_hlt_cache.sh"
    ;;
  offline_cache)
    export ARCHITECTURE_VIEW_10CLASS_OFFLINE_MANIFEST_PATH="${ABPH_MANIFEST_PATH}"
    export ARCHITECTURE_VIEW_10CLASS_OFFLINE_CACHE_DIR="${ABPH_OFFLINE_CACHE_DIR}"
    export ARCHITECTURE_VIEW_10CLASS_OFFLINE_SPLITS="model_train model_val"
    export ARCHITECTURE_VIEW_10CLASS_OFFLINE_DATA_DIRS="${ABPH_DATA_DIR}"
    export ARCHITECTURE_VIEW_10CLASS_OFFLINE_OVERWRITE="${OVERWRITE:-0}"
    exec bash "${PROJECT_DIR}/sbatch/run_cache_architecture_view_offline_inputs.sh"
    ;;
  audit)
    fresh_setup
    output="${ABPH_ROOT}/audits/step1_input_audit.json"
    mkdir -p "$(dirname "${output}")"
    cmd=("${PYTHON_BIN}" -u scripts/audit_adaptive_binary_pseudooffline_step1_inputs.py
      --manifest "${ABPH_MANIFEST_PATH}" --hlt-cache-dir "${ABPH_HLT_CACHE_DIR}"
      --offline-cache-dir "${ABPH_OFFLINE_CACHE_DIR}" --campaign-mode "${ABPH_CAMPAIGN_MODE}" --output "${output}")
    fresh_run "${cmd[@]}"
    ;;
  *) echo "Unknown ABPH input action ${ACTION}" >&2; exit 2 ;;
esac
