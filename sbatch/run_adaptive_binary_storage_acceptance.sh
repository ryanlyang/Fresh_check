#!/usr/bin/env bash
# Run Step-10 tests or compile the fail-closed real-data storage gate.
set -euo pipefail

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
CONDA_BASE="${ABPH_CONDA_BASE:-/home/ryreu/miniforge3-aarch64}"
CONDA_ENV="${ABPH_CONDA_ENV:-atlas_kd_tigris}"
export CONDA_BASE CONDA_ENV PYTHONNOUSERSITE=1
source "${PROJECT_DIR}/sbatch/common.sh"
fresh_setup
fresh_require_conda_python_package pytest

: "${ABPH_ROOT:?ABPH_ROOT is required}"
: "${ABPH_CAMPAIGN_MODE:=pilot}"
: "${ABPH_STORAGE_PROFILE:=streaming_30gb_v1}"
: "${ABPH_STORAGE_PROJECTION_PATH:?ABPH_STORAGE_PROJECTION_PATH is required}"
: "${ABPH_RUNTIME_ACCEPTANCE_PATH:?ABPH_RUNTIME_ACCEPTANCE_PATH is required}"

action="${1:?Usage: run_adaptive_binary_storage_acceptance.sh <tests|ram_lifecycle_smoke|compile>}"
evidence="${ABPH_ROOT}/storage/storage_acceptance_tests.json"
ram_evidence="${ABPH_ROOT}/storage/ram_lifecycle_smoke.json"
output="${ABPH_ROOT}/storage/storage_acceptance.json"

abph_acceptance_source_identity() {
  local commit status_hash
  commit="$(fresh_source_commit)"
  status_hash="$(fresh_source_status_hash)"
  if [[ -n "${ABPH_ACCEPTANCE_SOURCE_GIT_COMMIT:-}" && \
        "${commit}" != "${ABPH_ACCEPTANCE_SOURCE_GIT_COMMIT}" ]]; then
    echo "ABPH acceptance checkout changed: expected commit ${ABPH_ACCEPTANCE_SOURCE_GIT_COMMIT}, found ${commit}" >&2
    return 2
  fi
  if [[ -n "${ABPH_ACCEPTANCE_SOURCE_STATUS_HASH:-}" && \
        "${status_hash}" != "${ABPH_ACCEPTANCE_SOURCE_STATUS_HASH}" ]]; then
    echo "ABPH acceptance checkout status changed: expected hash ${ABPH_ACCEPTANCE_SOURCE_STATUS_HASH}, found ${status_hash}" >&2
    return 2
  fi
  ABPH_ACCEPTANCE_ACTIVE_COMMIT="${commit}"
  ABPH_ACCEPTANCE_ACTIVE_STATUS_HASH="${status_hash}"
  export ABPH_ACCEPTANCE_ACTIVE_COMMIT ABPH_ACCEPTANCE_ACTIVE_STATUS_HASH
}

abph_acceptance_source_identity

case "${action}" in
  tests)
    fresh_run "${PYTHON_BIN}" -u scripts/run_adaptive_binary_storage_acceptance_tests.py \
      --campaign-root "${ABPH_ROOT}" \
      --output "${evidence}"
    abph_acceptance_source_identity
    ;;
  ram_lifecycle_smoke)
    source "${PROJECT_DIR}/sbatch/adaptive_binary_ram_workspace.sh"
    abph_setup_ram_workspace
    abph_reserve_ram_workspace "storage_lifecycle_smoke" "$((2 * 1024 * 1024))"
    smoke_stage="${ABPH_RAM_WORKSPACE}/codec_scratch/ram_lifecycle_stage"
    fresh_run "${PYTHON_BIN}" -u scripts/run_adaptive_binary_ram_lifecycle_smoke.py prepare \
      --workspace "${ABPH_RAM_WORKSPACE}" \
      --campaign-root "${ABPH_ROOT}"
    abph_commit_ram_workspace "${smoke_stage}"
    fresh_run "${PYTHON_BIN}" -u scripts/publish_adaptive_binary_quota_tree.py \
      --campaign-root "${ABPH_ROOT}" \
      --source-dir "${smoke_stage}" \
      --destination-dir "${ABPH_ROOT}/storage/lifecycle_smoke_payload" \
      --artifact-role "ram_lifecycle_smoke_payload" \
      --run-id "ram-lifecycle-${SLURM_JOB_ID:-local}"
    abph_release_ram_workspace
    fresh_run "${PYTHON_BIN}" -u scripts/run_adaptive_binary_ram_lifecycle_smoke.py verify \
      --workspace "${ABPH_RAM_WORKSPACE}" \
      --campaign-root "${ABPH_ROOT}" \
      --output "${ram_evidence}" \
      --source-git-commit "${ABPH_ACCEPTANCE_ACTIVE_COMMIT}" \
      --source-status-hash "${ABPH_ACCEPTANCE_ACTIVE_STATUS_HASH}"
    abph_acceptance_source_identity
    ;;
  compile)
    abph_acceptance_source_identity
    fresh_run "${PYTHON_BIN}" -u scripts/write_adaptive_binary_storage_acceptance.py \
      --campaign-root "${ABPH_ROOT}" \
      --campaign-mode "${ABPH_CAMPAIGN_MODE}" \
      --storage-projection "${ABPH_ROOT}/storage/storage_projection.json" \
      --target-mode-selection "${ABPH_ROOT}/audits/target_mode_selection.json" \
      --target-feasibility "${ABPH_ROOT}/audits/actual_target_feasibility.json" \
      --wave-two-audit "${ABPH_ROOT}/storage/storage_audits/wave_2.json" \
      --artifact-manifest "${ABPH_ROOT}/storage/artifact_manifest.json" \
      --runtime-acceptance "${ABPH_RUNTIME_ACCEPTANCE_PATH}" \
      --test-evidence "${evidence}" \
      --ram-lifecycle-smoke "${ram_evidence}" \
      --source-git-commit "${ABPH_ACCEPTANCE_ACTIVE_COMMIT}" \
      --source-status-hash "${ABPH_ACCEPTANCE_ACTIVE_STATUS_HASH}" \
      --output "${output}"
    ;;
  *)
    echo "Unknown storage acceptance action ${action}" >&2
    exit 2
    ;;
esac
