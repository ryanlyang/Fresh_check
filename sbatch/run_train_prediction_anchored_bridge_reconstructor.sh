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
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12

set -euo pipefail
IFS=$'\n\t'
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
source "${PROJECT_DIR}/sbatch/prediction_anchored_bridge_common.sh"
PACK_ID="${1:?Usage: run_train_prediction_anchored_bridge_reconstructor.sh <pack-id>}"
pab_bootstrap_allocation
pab_require_env PAB_PHYSICAL45_SCALER

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
pab_run_executor PAB_RECONSTRUCTOR_EXECUTOR \
  --graph "${PREDICTION_ANCHORED_GRAPH}" --node-id "${PREDICTION_ANCHORED_NODE_ID}" \
  --ram-root "${PAB_RAM_ROOT}" --replica-output "${PAB_RAM_ROOT}/replicas"
mapfile -t run_ids < <(pab_node_run_ids)
[[ "${#run_ids[@]}" -gt 0 ]] || { echo "Reconstructor graph node contains no configurations" >&2; exit 2; }
for run_id in "${run_ids[@]}"; do
  fresh_run "${PYTHON_BIN}" -u scripts/train_prediction_anchored_bridge_reconstructor.py \
    --mode publish --run-id "${run_id}" --replica-dir "${PAB_RAM_ROOT}/replicas" \
    --output-dir "${PREDICTION_ANCHORED_ARTIFACT_ROOT}/reconstructors/${run_id}"
done
