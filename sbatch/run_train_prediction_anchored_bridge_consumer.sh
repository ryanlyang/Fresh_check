#!/usr/bin/env bash
# Train all paired3 upstream rows in one source-sharing allocation.
#SBATCH --job-name=pab_consumer
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
PACK_ID="${1:?Usage: run_train_prediction_anchored_bridge_consumer.sh <pack-id>}"
pab_bootstrap_allocation

plan_args=("${PYTHON_BIN}" -u scripts/train_prediction_anchored_bridge_consumer.py --mode plan)
if pab_is_dry_run; then
  plan_args+=(--dry-run)
else
  plan_args+=(--output "${PAB_ALLOCATION_LEDGER_DIR}/consumer_plan.json")
fi
fresh_run "${plan_args[@]}"

if pab_is_dry_run; then
  exit 0
fi
# The executor owns numerical training only. It must write three RAM-local
# checkpoint/metrics pairs per run; publication below retains median weights.
pab_run_executor PAB_CONSUMER_EXECUTOR \
  --graph "${PREDICTION_ANCHORED_GRAPH}" --node-id "${PREDICTION_ANCHORED_NODE_ID}" \
  --ram-root "${PAB_RAM_ROOT}" --replica-output "${PAB_RAM_ROOT}/replicas"
mapfile -t run_ids < <(pab_node_run_ids)
[[ "${#run_ids[@]}" -gt 0 ]] || { echo "Consumer graph node contains no configurations" >&2; exit 2; }
for run_id in "${run_ids[@]}"; do
  fresh_run "${PYTHON_BIN}" -u scripts/train_prediction_anchored_bridge_consumer.py \
    --mode publish --run-id "${run_id}" --replica-dir "${PAB_RAM_ROOT}/replicas" \
    --output-dir "${PREDICTION_ANCHORED_ARTIFACT_ROOT}/consumers/${run_id}"
done
