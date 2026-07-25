#!/usr/bin/env bash
# Refresh the immutable campaign graph and reuse successful B0-B3 allocations.
#SBATCH --job-name=pab_refresh_recovery
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=12

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${PREDICTION_ANCHORED_ARTIFACT_ROOT:?missing campaign artifact root}"
: "${PAB_SOURCE_PREFLIGHT:=${PREDICTION_ANCHORED_ARTIFACT_ROOT}/preflight}"
: "${PAB_REFRESHED_PREFLIGHT:=${PREDICTION_ANCHORED_ARTIFACT_ROOT}/preflight_clean_pair_v2}"
: "${PAB_B0_JOB_ID:?missing completed B0 job ID}"
: "${PAB_B1_JOB_ID:?missing completed B1 job ID}"
: "${PAB_B2_JOB_ID:?missing completed B2 job ID}"
: "${PAB_B3_CONSUMER_JOB_ID:?missing completed consumer-B3 job ID}"
: "${PAB_B3_L0_JOB_ID:?missing running/successful L0-B3 job ID}"
: "${PAB_B4_SELECT_JOB_ID:=}"
: "${PAB_B4_CONFIRM_JOB_ID:=}"
: "${PAB_CONDA_ENV:=atlas_kd_tigris}"
: "${PAB_CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
export PYTHONNOUSERSITE=1
export CONDA_ENV="${PAB_CONDA_ENV}"
export CONDA_BASE="${PAB_CONDA_BASE}"

for job_id in \
  "${PAB_B0_JOB_ID}" \
  "${PAB_B1_JOB_ID}" \
  "${PAB_B2_JOB_ID}" \
  "${PAB_B3_CONSUMER_JOB_ID}" \
  "${PAB_B3_L0_JOB_ID}"; do
  [[ "${job_id}" =~ ^[0-9]+$ ]] || {
    echo "Recovery received a non-numeric Slurm job ID: ${job_id}" >&2
    exit 2
  }
done
if [[ -n "${PAB_B4_SELECT_JOB_ID}" || -n "${PAB_B4_CONFIRM_JOB_ID}" ]]; then
  [[ "${PAB_B4_SELECT_JOB_ID}" =~ ^[0-9]+$ ]] || {
    echo "Recovery received an invalid completed B4-select job ID" >&2
    exit 2
  }
  [[ "${PAB_B4_CONFIRM_JOB_ID}" =~ ^[0-9]+$ ]] || {
    echo "Recovery received an invalid completed B4-confirm job ID" >&2
    exit 2
  }
fi

source "${PROJECT_DIR}/sbatch/common.sh"
fresh_setup
cd "${PROJECT_DIR}"

fresh_run "${PYTHON_BIN}" -u scripts/refresh_prediction_anchored_bridge_preflight.py \
  --source-preflight "${PAB_SOURCE_PREFLIGHT}" \
  --output-dir "${PAB_REFRESHED_PREFLIGHT}" \
  --artifact-root "${PREDICTION_ANCHORED_ARTIFACT_ROOT}" \
  --budget-gib 5

graph="${PAB_REFRESHED_PREFLIGHT}/prediction_anchored_tigris_graph.json"
registry="${PAB_REFRESHED_PREFLIGHT}/campaign_registry_step8.json"
reservations="${PAB_REFRESHED_PREFLIGHT}/campaign_reservations.json"
execution="${PAB_REFRESHED_PREFLIGHT}/prediction_anchored_execution_spec.json"
ledger="${PREDICTION_ANCHORED_ARTIFACT_ROOT}/job_ledgers/recovery_clean_pair_${SLURM_JOB_ID}.json"

reused_nodes=(
  --completed-job "b0_validate_preflight=${PAB_B0_JOB_ID}"
  --completed-job "b1_train_register_r0=${PAB_B1_JOB_ID}"
  --completed-job "b2_stage_recipes_scalers=${PAB_B2_JOB_ID}"
  --completed-job "b3_consumers_paired3=${PAB_B3_CONSUMER_JOB_ID}"
  --existing-job "b3_l0_paired3=${PAB_B3_L0_JOB_ID}"
)
if [[ -n "${PAB_B4_SELECT_JOB_ID}" ]]; then
  reused_nodes+=(
    --completed-job "b4_select_consumer=${PAB_B4_SELECT_JOB_ID}"
    --completed-job "b4_confirm_consumer=${PAB_B4_CONFIRM_JOB_ID}"
  )
fi

fresh_run "${PYTHON_BIN}" -u scripts/submit_prediction_anchored_bridge_graph.py \
  --graph "${graph}" \
  --registry "${registry}" \
  --reservations "${reservations}" \
  --execution-spec "${execution}" \
  --ledger-output "${ledger}" \
  "${reused_nodes[@]}" \
  --execute
