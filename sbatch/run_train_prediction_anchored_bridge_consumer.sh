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
#SBATCH --gres=gpu:1
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
# Numerical training and full model_val_select evidence are repository-owned.
# The 24 temporary checkpoint/metric pairs remain allocation-local; only the
# ordered median publications and compact selection evidence persist.
pab_require_env PAB_EXECUTION_SPEC PAB_RESERVATIONS
: "${PAB_R0_CHECKPOINT:=${PREDICTION_ANCHORED_ARTIFACT_ROOT}/r0/r0_weights.pt}"
: "${PAB_R0_REGISTRATION:=${PREDICTION_ANCHORED_ARTIFACT_ROOT}/r0/r0_registration.json}"
: "${PAB_PHYSICAL45_RECIPE:=${PREDICTION_ANCHORED_ARTIFACT_ROOT}/bridge_inputs/bridge_recipe_physical45.json}"
: "${PAB_ALL50_RECIPE:=${PREDICTION_ANCHORED_ARTIFACT_ROOT}/bridge_inputs/bridge_recipe_all50.json}"
fresh_run "${PYTHON_BIN}" -u scripts/execute_prediction_anchored_bridge_consumers.py \
  --execution-spec "${PAB_EXECUTION_SPEC}" \
  --r0-checkpoint "${PAB_R0_CHECKPOINT}" --r0-registration "${PAB_R0_REGISTRATION}" \
  --physical45-recipe "${PAB_PHYSICAL45_RECIPE}" --all50-recipe "${PAB_ALL50_RECIPE}" \
  --ram-root "${PAB_RAM_ROOT}" --allocation-id "${SLURM_JOB_ID}" \
  --replica-output "${PAB_RAM_ROOT}/replicas" \
  --evaluation-output "${PREDICTION_ANCHORED_ARTIFACT_ROOT}/consumer_evaluations" \
  --execution-report "${PREDICTION_ANCHORED_ARTIFACT_ROOT}/telemetry/b3_consumer_execution.json"
mapfile -t run_ids < <(pab_node_run_ids)
[[ "${#run_ids[@]}" -gt 0 ]] || { echo "Consumer graph node contains no configurations" >&2; exit 2; }
for run_id in "${run_ids[@]}"; do
  publish_args=("${PYTHON_BIN}" -u scripts/train_prediction_anchored_bridge_consumer.py \
    --mode publish --run-id "${run_id}" --replica-dir "${PAB_RAM_ROOT}/replicas" \
    --output-dir "${PREDICTION_ANCHORED_ARTIFACT_ROOT}/consumers/${run_id}" \
    --reservations "${PAB_RESERVATIONS}")
  case "${run_id}" in
    T10_clean|T10_robust|T10_all50_clean)
      publish_args+=(--selection-aggregate \
        "${PREDICTION_ANCHORED_ARTIFACT_ROOT}/consumer_evaluations/${run_id}/selection_aggregate.json")
      ;;
  esac
  fresh_run "${publish_args[@]}"
done
