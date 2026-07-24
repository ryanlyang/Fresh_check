#!/usr/bin/env bash
# Shared fail-closed allocation bootstrap for the prediction-anchored pilot.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${PREDICTION_ANCHORED_ARTIFACT_ROOT:=${PROJECT_DIR}/fresh_check_outputs/prediction_anchored_bridge}"
: "${PREDICTION_ANCHORED_GRAPH:?Set PREDICTION_ANCHORED_GRAPH to the immutable Step-10 graph}"
: "${PREDICTION_ANCHORED_NODE_ID:?The submitter must export PREDICTION_ANCHORED_NODE_ID}"
: "${PAB_PREFLIGHT_ROOT:=$(dirname -- "${PREDICTION_ANCHORED_GRAPH}")}"
: "${PAB_REPRESENTATIVE_RESOURCE_REFERENCE:=${PAB_PREFLIGHT_ROOT}/representative_architecture_resource_reference.json}"
: "${PAB_DRY_RUN:=0}"
: "${PAB_CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
: "${PAB_CONDA_ENV:=atlas_kd_tigris}"
CONDA_BASE="${PAB_CONDA_BASE}"
CONDA_ENV="${PAB_CONDA_ENV}"
export CONDA_BASE CONDA_ENV PAB_CONDA_BASE PAB_CONDA_ENV
export PAB_PREFLIGHT_ROOT PAB_REPRESENTATIVE_RESOURCE_REFERENCE
export PYTHONNOUSERSITE=1

# shellcheck source=common.sh
source "${PROJECT_DIR}/sbatch/common.sh"

pab_require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Required prediction-anchored variable is unset: ${name}" >&2
    return 2
  fi
}

pab_is_dry_run() {
  [[ "${PAB_DRY_RUN}" == "1" || "${PAB_DRY_RUN}" == "true" ]]
}

pab_bootstrap_allocation() {
  fresh_setup
  [[ "${SLURM_NNODES:-${SLURM_JOB_NUM_NODES:-0}}" == "1" ]] || {
    echo "Prediction-anchored allocations require exactly one node" >&2
    return 2
  }
  [[ "${SLURM_PROCID:-0}" == "0" ]] || {
    echo "Only allocation leader rank 0 may stage sources or create the ledger" >&2
    return 2
  }
  [[ "${PYTHONNOUSERSITE}" == "1" ]] || return 2
  : "${PAB_RAM_ROOT:=/dev/shm/prediction_anchored_bridge/${SLURM_JOB_ID:?missing SLURM_JOB_ID}}"
  case "${PAB_RAM_ROOT}" in
    /dev/shm/prediction_anchored_bridge/*) ;;
    *) echo "Unsafe allocation RAM root: ${PAB_RAM_ROOT}" >&2; return 2 ;;
  esac
  mkdir -p "${PAB_RAM_ROOT}"
  PAB_ALLOCATION_LEDGER_DIR="${PREDICTION_ANCHORED_ARTIFACT_ROOT}/allocation_ledgers/${PREDICTION_ANCHORED_NODE_ID}/${SLURM_JOB_ID}"
  local launch_args=(
    "${PYTHON_BIN}" -u scripts/run_prediction_anchored_bridge_allocation.py
    --graph "${PREDICTION_ANCHORED_GRAPH}"
    --execution-spec "${PAB_EXECUTION_SPEC:?missing PAB_EXECUTION_SPEC}"
    --reservations "${PAB_RESERVATIONS:?missing PAB_RESERVATIONS}"
    --representative-reference "${PAB_REPRESENTATIVE_RESOURCE_REFERENCE}"
    --node-id "${PREDICTION_ANCHORED_NODE_ID}"
    --ram-root "${PAB_RAM_ROOT}"
    --output "${PAB_ALLOCATION_LEDGER_DIR}/launch_manifest.json"
  )
  if [[ -f "${PREDICTION_ANCHORED_ARTIFACT_ROOT}/selection/selected_bridge_consumer.json" ]]; then
    launch_args+=(--selected-consumer "${PREDICTION_ANCHORED_ARTIFACT_ROOT}/selection/selected_bridge_consumer.json")
  fi
  if pab_is_dry_run; then
    launch_args+=(--dry-run)
  else
    mkdir -p "${PAB_ALLOCATION_LEDGER_DIR}"
  fi
  fresh_run "${launch_args[@]}"
  export PAB_RAM_ROOT PAB_ALLOCATION_LEDGER_DIR PAB_REPRESENTATIVE_RESOURCE_REFERENCE
}

pab_node_run_ids() {
  "${PYTHON_BIN}" scripts/inspect_prediction_anchored_bridge_graph.py \
    --graph "${PREDICTION_ANCHORED_GRAPH}" --node-id "${PREDICTION_ANCHORED_NODE_ID}" \
    --field configuration-run-ids
}

pab_run_executor() {
  local variable="$1"
  shift
  pab_require_env "${variable}"
  local executable="${!variable}"
  [[ -f "${executable}" && ! -L "${executable}" ]] || {
    echo "Unsafe or missing configured executor ${variable}: ${executable}" >&2
    return 2
  }
  fresh_run "${PYTHON_BIN}" -u "${executable}" "$@"
}

pab_cleanup_ram() {
  [[ -n "${PAB_RAM_ROOT:-}" ]] || return 0
  case "${PAB_RAM_ROOT}" in
    /dev/shm/prediction_anchored_bridge/*)
      if [[ -d "${PAB_RAM_ROOT}" ]]; then
        find "${PAB_RAM_ROOT}" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
        rmdir "${PAB_RAM_ROOT}" 2>/dev/null || true
      fi
      ;;
    *) echo "Refusing RAM cleanup outside the campaign prefix: ${PAB_RAM_ROOT}" >&2 ;;
  esac
}

# A preempted/failed allocation never resumes partial replicas.  It restarts the
# whole configuration pack in a fresh SLURM_JOB_ID workspace.
trap pab_cleanup_ram EXIT
