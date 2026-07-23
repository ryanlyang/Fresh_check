#!/usr/bin/env bash
# Train one packed paired3 reconstructor allocation from a shared source/teacher.
#SBATCH --job-name=pab_reco
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tigris
#SBATCH --account=reu-aisocial
#SBATCH --nodes=1
#SBATCH --time=3-00:00:00
#SBATCH --mem=512G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12

set -euo pipefail
IFS=$'\n\t'
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
source "${PROJECT_DIR}/sbatch/prediction_anchored_bridge_common.sh"
PACK_ID="${1:?Usage: run_train_prediction_anchored_bridge_reconstructor.sh <pack-id>}"
if [[ "${PACK_ID}" == "b3_l0_paired3" || "${PACK_ID}" == "b6_l0_postteacher_eval_paired3" ]]; then
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
fi
pab_bootstrap_allocation
pab_require_env PAB_EXECUTION_SPEC PAB_RESERVATIONS
: "${PAB_R0_CHECKPOINT:=${PREDICTION_ANCHORED_ARTIFACT_ROOT}/r0/r0_weights.pt}"
: "${PAB_R0_REGISTRATION:=${PREDICTION_ANCHORED_ARTIFACT_ROOT}/r0/r0_registration.json}"
: "${PAB_PHYSICAL45_SCALER:=${PREDICTION_ANCHORED_ARTIFACT_ROOT}/bridge_inputs/bridge_scalers_physical45.json}"
: "${PAB_ALL50_SCALER:=${PREDICTION_ANCHORED_ARTIFACT_ROOT}/bridge_inputs/bridge_scalers_all50.json}"
: "${PAB_ABSOLUTE_SCALER:=${PREDICTION_ANCHORED_ARTIFACT_ROOT}/bridge_inputs/bridge_absolute_scaler_physical45.json}"
: "${PAB_DEPLOYED_RESOURCE_REFERENCE:=${PREDICTION_ANCHORED_ARTIFACT_ROOT}/measurements/deployed_resource_reference.json}"

plan_args=("${PYTHON_BIN}" -u scripts/train_prediction_anchored_bridge_reconstructor.py
  --mode plan --scaler "${PAB_PHYSICAL45_SCALER}")
if pab_is_dry_run; then
  plan_args+=(--dry-run)
else
  plan_args+=(--output "${PAB_ALLOCATION_LEDGER_DIR}/reconstructor_plan.json")
fi
fresh_run "${plan_args[@]}"

if pab_is_dry_run; then
  exit 0
fi
fresh_run "${PYTHON_BIN}" -u scripts/train_prediction_anchored_bridge_reconstructor.py \
  --mode execute --execution-spec "${PAB_EXECUTION_SPEC}" \
  --graph "${PREDICTION_ANCHORED_GRAPH}" --node-id "${PREDICTION_ANCHORED_NODE_ID}" \
  --artifact-root "${PREDICTION_ANCHORED_ARTIFACT_ROOT}" \
  --r0-checkpoint "${PAB_R0_CHECKPOINT}" --r0-registration "${PAB_R0_REGISTRATION}" \
  --scaler "${PAB_PHYSICAL45_SCALER}" --all50-scaler "${PAB_ALL50_SCALER}" \
  --absolute-scaler "${PAB_ABSOLUTE_SCALER}" \
  --deployed-resource-reference "${PAB_DEPLOYED_RESOURCE_REFERENCE}" \
  --ram-root "${PAB_RAM_ROOT}" --allocation-id "${SLURM_JOB_ID}" \
  --replica-dir "${PAB_RAM_ROOT}/replicas" --device "${DEVICE}" \
  --execution-report "${PREDICTION_ANCHORED_ARTIFACT_ROOT}/telemetry/${PREDICTION_ANCHORED_NODE_ID}.json"
if [[ "${PACK_ID}" == "b6_l0_postteacher_eval_paired3" ]]; then
  run_ids=(D10_L0_bridge_only)
else
  mapfile -t run_ids < <(pab_node_run_ids)
  [[ "${#run_ids[@]}" -gt 0 ]] || { echo "Reconstructor graph node contains no configurations" >&2; exit 2; }
fi
for run_id in "${run_ids[@]}"; do
  publish_mode=publish
  publish_root="${PREDICTION_ANCHORED_ARTIFACT_ROOT}/reconstructors/${run_id}"
  if [[ "${PACK_ID}" == "b3_l0_paired3" && "${run_id}" == "D10_L0_bridge_only" ]]; then
    publish_mode=publish-l0-early
    publish_root="${PREDICTION_ANCHORED_ARTIFACT_ROOT}/l0_early/${run_id}"
  elif [[ "${PACK_ID}" == "b6_l0_postteacher_eval_paired3" ]]; then
    publish_mode=publish-l0-postteacher
  fi
  fresh_run "${PYTHON_BIN}" -u scripts/train_prediction_anchored_bridge_reconstructor.py \
    --mode "${publish_mode}" --run-id "${run_id}" --replica-dir "${PAB_RAM_ROOT}/replicas" \
    --output-dir "${publish_root}" --reservations "${PAB_RESERVATIONS}"
done
