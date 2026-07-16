#!/usr/bin/env bash
# Build hierarchy targets or execute the mandatory real-target hard gate.

#SBATCH --job-name=abph_targets
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=2-00:00:00
#SBATCH --mem=300G
#SBATCH --cpus-per-task=16

set -euo pipefail
IFS=$'\n\t'
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
source "${PROJECT_DIR}/sbatch/common.sh"
ACTION="${1:?Usage: run_adaptive_binary_targets.sh <cache|preflight>}"
: "${ABPH_ROOT:=${OUTPUT_ROOT}/adaptive_binary_pseudooffline}"
: "${ABPH_MANIFEST_PATH:=${ABPH_ROOT}/inputs/split_manifest/split_manifest.json.gz}"
: "${ABPH_HLT_CACHE_DIR:=${ABPH_ROOT}/inputs/hlt_cache}"
: "${ABPH_OFFLINE_CACHE_DIR:=${ABPH_ROOT}/inputs/offline_cache}"
: "${ABPH_TARGET_CACHE_DIR:=${ABPH_ROOT}/targets}"
: "${ABPH_TARGET_CHUNK_SIZE:=512}"
: "${ABPH_PREFLIGHT_JETS_PER_CLASS:=64}"
export PYTHONNOUSERSITE=1
fresh_setup
case "${ACTION}" in
  cache)
    cmd=("${PYTHON_BIN}" -u scripts/cache_adaptive_binary_hierarchy_targets.py
      --manifest "${ABPH_MANIFEST_PATH}" --hlt-cache-dir "${ABPH_HLT_CACHE_DIR}"
      --offline-cache-dir "${ABPH_OFFLINE_CACHE_DIR}" --output-cache-dir "${ABPH_TARGET_CACHE_DIR}"
      --splits model_train model_val --groupings exclusive_kt cambridge_aachen
      --chunk-size "${ABPH_TARGET_CHUNK_SIZE}" --report "${ABPH_TARGET_CACHE_DIR}/step2_target_cache_report.json")
    fresh_append_flag_if_enabled cmd --overwrite "${OVERWRITE:-0}"
    ;;
  preflight)
    mkdir -p "${ABPH_ROOT}/audits"
    cmd=("${PYTHON_BIN}" -u scripts/audit_adaptive_binary_step4_accounting.py
      --target-cache-dir "${ABPH_TARGET_CACHE_DIR}" --splits model_train model_val
      --groupings exclusive_kt cambridge_aachen --max-jets-per-class "${ABPH_PREFLIGHT_JETS_PER_CLASS}"
      --report "${ABPH_ROOT}/audits/actual_target_feasibility.json")
    ;;
  *) echo "Unknown ABPH target action ${ACTION}" >&2; exit 2 ;;
esac
fresh_run "${cmd[@]}"
