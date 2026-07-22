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
pab_require_env PAB_STACK_TRAIN_DISTILL_SHA256
pab_require_env PAB_CLASS_ORDER_JSON
case "${NAMESPACE}" in
  physical45_selected_bridge_teacher|physical45_selected_teacher_on_f0_control)
    PAB_BINDING="${PREDICTION_ANCHORED_ARTIFACT_ROOT}/bindings/primary.json"
    ;;
  all50_selected_bridge_teacher)
    PAB_BINDING="${PREDICTION_ANCHORED_ARTIFACT_ROOT}/bindings/all50.json"
    ;;
  alternate_selected_bridge_teacher)
    PAB_BINDING="${PREDICTION_ANCHORED_ARTIFACT_ROOT}/bindings/alternate.json"
    ;;
  *) echo "Unknown teacher-logit namespace: ${NAMESPACE}" >&2; exit 2 ;;
esac

cache_args=("${PYTHON_BIN}" -u scripts/cache_prediction_anchored_bridge_logits.py
  --binding "${PAB_BINDING}" --namespace "${NAMESPACE}"
  --output-root "${PREDICTION_ANCHORED_ARTIFACT_ROOT}/teacher_logits"
  --stack-train-distill-sha256 "${PAB_STACK_TRAIN_DISTILL_SHA256}"
  --class-order "${PAB_CLASS_ORDER_JSON}")
if [[ -f "${PREDICTION_ANCHORED_ARTIFACT_ROOT}/selection/selected_bridge_consumer.json" ]]; then
  cache_args+=(--selected-consumer "${PREDICTION_ANCHORED_ARTIFACT_ROOT}/selection/selected_bridge_consumer.json")
fi
if pab_is_dry_run; then
  cache_args+=(--dry-run)
else
  # Forward outputs stay in allocation RAM. Only detached logits are published.
  pab_run_executor PAB_TEACHER_FORWARD_EXECUTOR \
    --binding "${PAB_BINDING}" --namespace "${NAMESPACE}" --ram-root "${PAB_RAM_ROOT}" \
    --output "${PAB_RAM_ROOT}/teacher_outputs.npz"
  cache_args+=(--input-npz "${PAB_RAM_ROOT}/teacher_outputs.npz")
fi
fresh_run "${cache_args[@]}"
