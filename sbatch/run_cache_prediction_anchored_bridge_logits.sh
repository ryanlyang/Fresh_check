#!/usr/bin/env bash
# Generate and publish one immutable teacher-logit namespace.
#SBATCH --job-name=pab_logits
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tigris
#SBATCH --account=reu-aisocial
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=512G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12

set -euo pipefail
IFS=$'\n\t'
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
source "${PROJECT_DIR}/sbatch/prediction_anchored_bridge_common.sh"
NAMESPACE="${1:?Usage: run_cache_prediction_anchored_bridge_logits.sh <namespace>}"
pab_bootstrap_allocation
pab_require_env PAB_EXECUTION_SPEC
: "${PAB_R0_CHECKPOINT:=${PREDICTION_ANCHORED_ARTIFACT_ROOT}/r0/r0_weights.pt}"
: "${PAB_R0_REGISTRATION:=${PREDICTION_ANCHORED_ARTIFACT_ROOT}/r0/r0_registration.json}"
case "${NAMESPACE}" in
  physical45_selected_bridge_teacher|physical45_selected_teacher_on_f0_control)
    PAB_BINDING="${PREDICTION_ANCHORED_ARTIFACT_ROOT}/bindings/primary.json"
    PAB_RECIPE="${PREDICTION_ANCHORED_ARTIFACT_ROOT}/bridge_inputs/bridge_recipe_physical45.json"
    ;;
  all50_selected_bridge_teacher)
    PAB_BINDING="${PREDICTION_ANCHORED_ARTIFACT_ROOT}/bindings/all50.json"
    PAB_RECIPE="${PREDICTION_ANCHORED_ARTIFACT_ROOT}/bridge_inputs/bridge_recipe_all50.json"
    ;;
  physical45_alternate_bridge_teacher)
    PAB_BINDING="${PREDICTION_ANCHORED_ARTIFACT_ROOT}/bindings/alternate.json"
    PAB_RECIPE="${PREDICTION_ANCHORED_ARTIFACT_ROOT}/bridge_inputs/bridge_recipe_physical45.json"
    ;;
  *) echo "Unknown teacher-logit namespace: ${NAMESPACE}" >&2; exit 2 ;;
esac

cache_args=("${PYTHON_BIN}" -u scripts/cache_prediction_anchored_bridge_logits.py
  --execution-spec "${PAB_EXECUTION_SPEC}"
  --binding "${PAB_BINDING}" --namespace "${NAMESPACE}"
  --r0-checkpoint "${PAB_R0_CHECKPOINT}" --r0-registration "${PAB_R0_REGISTRATION}"
  --bridge-recipe "${PAB_RECIPE}"
  --ram-root "${PAB_RAM_ROOT}" --allocation-id "${SLURM_JOB_ID}" --device "${DEVICE}"
  --output-root "${PREDICTION_ANCHORED_ARTIFACT_ROOT}/teacher_logits"
)
if [[ "${NAMESPACE}" == "physical45_selected_bridge_teacher" || \
      "${NAMESPACE}" == "physical45_selected_teacher_on_f0_control" ]]; then
  cache_args+=(--selected-consumer "${PREDICTION_ANCHORED_ARTIFACT_ROOT}/selection/selected_bridge_consumer.json")
fi
if pab_is_dry_run; then
  cache_args+=(--dry-run)
fi
fresh_run "${cache_args[@]}"
