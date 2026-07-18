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
: "${ABPH_STORAGE_PROFILE:=cache_heavy_v1}"

case "${ABPH_CAMPAIGN_MODE}" in
  pilot) sizes=(500000 150000 300000 150000 150000) ;;
  highdata) sizes=(5000000 1000000 2000000 1000000 1000000) ;;
  *) echo "Unknown ABPH_CAMPAIGN_MODE=${ABPH_CAMPAIGN_MODE}" >&2; exit 2 ;;
esac

export PYTHONNOUSERSITE=1
fresh_setup
streaming=0
if [[ "${ABPH_STORAGE_PROFILE}" == "streaming_30gb_v1" ]]; then
  streaming=1
  source "${PROJECT_DIR}/sbatch/adaptive_binary_ram_workspace.sh"
  abph_setup_ram_workspace
fi

publish_tree() {
  local source_dir="$1"
  local destination_dir="$2"
  local role="$3"
  local publish_cmd=("${PYTHON_BIN}" -u scripts/publish_adaptive_binary_quota_tree.py \
    --campaign-root "${ABPH_ROOT}" \
    --source-dir "${source_dir}" \
    --destination-dir "${destination_dir}" \
    --artifact-role "${role}" \
    --run-id "inputs-${ACTION}-${SLURM_JOB_ID:-local}")
  fresh_append_flag_if_enabled publish_cmd --overwrite "${OVERWRITE:-0}"
  "${publish_cmd[@]}"
}

case "${ACTION}" in
  splits)
    manifest_path="${ABPH_MANIFEST_PATH}"
    if ((streaming)); then
      manifest_path="${ABPH_RAM_WORKSPACE}/split_manifest/split_manifest.json.gz"
    fi
    export DATA_DIR="${ABPH_DATA_DIR}" MANIFEST_PATH="${manifest_path}"
    export MODEL_TRAIN_SIZE="${sizes[0]}" MODEL_VAL_SIZE="${sizes[1]}"
    export STACK_TRAIN_SIZE="${sizes[2]}" STACK_VAL_SIZE="${sizes[3]}" FINAL_TEST_SIZE="${sizes[4]}"
    bash "${PROJECT_DIR}/sbatch/run_build_fresh_splits.sh"
    if ((streaming)); then
      publish_tree "$(dirname "${manifest_path}")" "$(dirname "${ABPH_MANIFEST_PATH}")" "input_split_manifest"
    fi
    ;;
  hlt_cache)
    hlt_cache_dir="${ABPH_HLT_CACHE_DIR}"
    if ((streaming)); then
      hlt_cache_dir="${ABPH_RAM_WORKSPACE}/hlt_cache"
      : "${ABPH_RAM_STAGE_RESERVATION_BYTES:?Streaming HLT staging requires a projected RAM reservation}"
      abph_reserve_ram_workspace "hlt_cache_stage" "${ABPH_RAM_STAGE_RESERVATION_BYTES}"
    fi
    export DATA_DIR="${ABPH_DATA_DIR}" MANIFEST_PATH="${ABPH_MANIFEST_PATH}" HLT_CACHE_DIR="${hlt_cache_dir}"
    export HLT_SPLITS="model_train model_val stack_train stack_val final_test"
    export HLT_PROFILE=fixed_hlt_v2_realistic HLT_DEGRADATION_STRENGTH=2.5
    bash "${PROJECT_DIR}/sbatch/run_build_fresh_hlt_cache.sh"
    if ((streaming)); then
      abph_commit_ram_workspace "${hlt_cache_dir}"
      publish_tree "${hlt_cache_dir}" "${ABPH_HLT_CACHE_DIR}" "input_hlt_cache"
      abph_release_ram_workspace
    fi
    ;;
  offline_cache)
    offline_cache_dir="${ABPH_OFFLINE_CACHE_DIR}"
    if ((streaming)); then
      offline_cache_dir="${ABPH_RAM_WORKSPACE}/offline_cache"
      : "${ABPH_RAM_STAGE_RESERVATION_BYTES:?Streaming offline staging requires a projected RAM reservation}"
      abph_reserve_ram_workspace "offline_cache_stage" "${ABPH_RAM_STAGE_RESERVATION_BYTES}"
    fi
    export ARCHITECTURE_VIEW_10CLASS_OFFLINE_MANIFEST_PATH="${ABPH_MANIFEST_PATH}"
    export ARCHITECTURE_VIEW_10CLASS_OFFLINE_CACHE_DIR="${offline_cache_dir}"
    export ARCHITECTURE_VIEW_10CLASS_OFFLINE_SPLITS="model_train model_val"
    export ARCHITECTURE_VIEW_10CLASS_OFFLINE_DATA_DIRS="${ABPH_DATA_DIR}"
    export ARCHITECTURE_VIEW_10CLASS_OFFLINE_OVERWRITE="${OVERWRITE:-0}"
    bash "${PROJECT_DIR}/sbatch/run_cache_architecture_view_offline_inputs.sh"
    if ((streaming)); then
      abph_commit_ram_workspace "${offline_cache_dir}"
      publish_tree "${offline_cache_dir}" "${ABPH_OFFLINE_CACHE_DIR}" "input_offline_cache"
      abph_release_ram_workspace
    fi
    ;;
  audit)
    output="${ABPH_ROOT}/audits/step1_input_audit.json"
    if ((streaming)); then
      output="${ABPH_RAM_WORKSPACE}/input_audit/step1_input_audit.json"
    fi
    mkdir -p "$(dirname "${output}")"
    cmd=("${PYTHON_BIN}" -u scripts/audit_adaptive_binary_pseudooffline_step1_inputs.py
      --manifest "${ABPH_MANIFEST_PATH}" --hlt-cache-dir "${ABPH_HLT_CACHE_DIR}"
      --offline-cache-dir "${ABPH_OFFLINE_CACHE_DIR}" --campaign-mode "${ABPH_CAMPAIGN_MODE}" --output "${output}")
    fresh_run "${cmd[@]}"
    if ((streaming)); then
      publish_tree "$(dirname "${output}")" "${ABPH_ROOT}/audits" "input_audit"
    fi
    ;;
  *) echo "Unknown ABPH input action ${ACTION}" >&2; exit 2 ;;
esac
