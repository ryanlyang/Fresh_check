#!/usr/bin/env bash
# Execute one fail-closed B0/B1/B2/B4/B5/B6 policy or staging action.
#SBATCH --job-name=pab_prepare
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tigris
#SBATCH --account=reu-aisocial
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=512G
#SBATCH --cpus-per-task=24

set -euo pipefail
IFS=$'\n\t'
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
source "${PROJECT_DIR}/sbatch/prediction_anchored_bridge_common.sh"
ACTION="${1:?Usage: run_prepare_prediction_anchored_bridge_ram.sh <B0|B1|B2|B4_SELECT|B4_CONFIRM|B5_BIND|B5_RELEASE|B6_SELECT|DEPLOY_CONFIRM|REPORT_EXPORT|FINAL_TEST>}"
pab_bootstrap_allocation

dry_flag=()
pab_is_dry_run && dry_flag=(--dry-run)
selected="${PREDICTION_ANCHORED_ARTIFACT_ROOT}/selection/selected_bridge_consumer.json"

case "${ACTION}" in
  B0)
    pab_require_env PAB_REGISTRY
    pab_require_env PAB_RESERVATIONS
    pab_require_env PAB_EXECUTION_SPEC
    fresh_run "${PYTHON_BIN}" -u scripts/run_prediction_anchored_bridge_campaign.py \
      --campaign-action validate-production --registry "${PAB_REGISTRY}" \
      --reservations "${PAB_RESERVATIONS}" --execution-spec "${PAB_EXECUTION_SPEC}" \
      --artifact-root "${PREDICTION_ANCHORED_ARTIFACT_ROOT}" \
      --dry-run
    fresh_run "${PYTHON_BIN}" -u scripts/train_prediction_anchored_r0.py \
      --execution-spec "${PAB_EXECUTION_SPEC}" --output-dir "${PREDICTION_ANCHORED_ARTIFACT_ROOT}/r0" \
      --ram-root "${PAB_RAM_ROOT}" --dry-run
    ;;
  B1)
    pab_require_env PAB_EXECUTION_SPEC
    fresh_run "${PYTHON_BIN}" -u scripts/train_prediction_anchored_r0.py \
      --execution-spec "${PAB_EXECUTION_SPEC}" \
      --output-dir "${PREDICTION_ANCHORED_ARTIFACT_ROOT}/r0" \
      --ram-root "${PAB_RAM_ROOT}" --allocation-id "${SLURM_JOB_ID}" \
      --device "${DEVICE}" "${dry_flag[@]}"
    ;;
  B2)
    pab_require_env PAB_EXECUTION_SPEC
    : "${PAB_R0_CHECKPOINT:=${PREDICTION_ANCHORED_ARTIFACT_ROOT}/r0/r0_weights.pt}"
    : "${PAB_R0_REGISTRATION:=${PREDICTION_ANCHORED_ARTIFACT_ROOT}/r0/r0_registration.json}"
    fresh_run "${PYTHON_BIN}" -u scripts/prepare_prediction_anchored_bridge_inputs.py \
      --execution-spec "${PAB_EXECUTION_SPEC}" \
      --r0-checkpoint "${PAB_R0_CHECKPOINT}" --r0-registration "${PAB_R0_REGISTRATION}" \
      --ram-root "${PAB_RAM_ROOT}" --allocation-id "${SLURM_JOB_ID}" \
      --output-dir "${PREDICTION_ANCHORED_ARTIFACT_ROOT}/bridge_inputs" \
      --device "${DEVICE}" "${dry_flag[@]}"
    ;;
  B4_SELECT)
    : "${PAB_T10_CLEAN_AGGREGATE:=${PREDICTION_ANCHORED_ARTIFACT_ROOT}/consumer_evaluations/T10_clean/selection_aggregate.json}"
    : "${PAB_T10_ROBUST_AGGREGATE:=${PREDICTION_ANCHORED_ARTIFACT_ROOT}/consumer_evaluations/T10_robust/selection_aggregate.json}"
    : "${PAB_T10_CLEAN_CHECKPOINT:=${PREDICTION_ANCHORED_ARTIFACT_ROOT}/consumers/T10_clean/median_weights.pt}"
    : "${PAB_T10_ROBUST_CHECKPOINT:=${PREDICTION_ANCHORED_ARTIFACT_ROOT}/consumers/T10_robust/median_weights.pt}"
    : "${PAB_R0_CHECKPOINT:=${PREDICTION_ANCHORED_ARTIFACT_ROOT}/r0/r0_weights.pt}"
    : "${PAB_PHYSICAL45_RECIPE:=${PREDICTION_ANCHORED_ARTIFACT_ROOT}/bridge_inputs/bridge_recipe_physical45.json}"
    fresh_run "${PYTHON_BIN}" -u scripts/select_prediction_anchored_bridge_consumer.py select \
      --clean-aggregate "${PAB_T10_CLEAN_AGGREGATE}" --robust-aggregate "${PAB_T10_ROBUST_AGGREGATE}" \
      --clean-checkpoint "${PAB_T10_CLEAN_CHECKPOINT}" --robust-checkpoint "${PAB_T10_ROBUST_CHECKPOINT}" \
      --f0-checkpoint "${PAB_R0_CHECKPOINT}" --bridge-recipe "${PAB_PHYSICAL45_RECIPE}" \
      --output "${PREDICTION_ANCHORED_ARTIFACT_ROOT}/selection/consumer_preconfirmation.json" "${dry_flag[@]}"
    ;;
  B4_CONFIRM)
    pab_require_env PAB_EXECUTION_SPEC
    if pab_is_dry_run; then
      echo "B4_CONFIRM dry-run: sealed stack_val_consumer remains unopened"
      exit 0
    fi
    : "${PAB_R0_CHECKPOINT:=${PREDICTION_ANCHORED_ARTIFACT_ROOT}/r0/r0_weights.pt}"
    : "${PAB_R0_REGISTRATION:=${PREDICTION_ANCHORED_ARTIFACT_ROOT}/r0/r0_registration.json}"
    : "${PAB_PHYSICAL45_RECIPE:=${PREDICTION_ANCHORED_ARTIFACT_ROOT}/bridge_inputs/bridge_recipe_physical45.json}"
    fresh_run "${PYTHON_BIN}" -u scripts/confirm_prediction_anchored_bridge_consumer.py \
      --execution-spec "${PAB_EXECUTION_SPEC}" \
      --preconfirmation "${PREDICTION_ANCHORED_ARTIFACT_ROOT}/selection/consumer_preconfirmation.json" \
      --r0-checkpoint "${PAB_R0_CHECKPOINT}" --r0-registration "${PAB_R0_REGISTRATION}" \
      --physical45-recipe "${PAB_PHYSICAL45_RECIPE}" \
      --ram-root "${PAB_RAM_ROOT}" --allocation-id "${SLURM_JOB_ID}" \
      --output-dir "${PREDICTION_ANCHORED_ARTIFACT_ROOT}/selection" --device "${DEVICE}"
    ;;
  B5_BIND)
    [[ -f "${selected}" ]] || { echo "Stage B5 refuses a guessed consumer" >&2; exit 2; }
    pab_require_env PAB_EXECUTION_SPEC
    : "${PAB_PHYSICAL45_RECIPE:=${PREDICTION_ANCHORED_ARTIFACT_ROOT}/bridge_inputs/bridge_recipe_physical45.json}"
    : "${PAB_ALL50_RECIPE:=${PREDICTION_ANCHORED_ARTIFACT_ROOT}/bridge_inputs/bridge_recipe_all50.json}"
    : "${PAB_ALL50_SCALER:=${PREDICTION_ANCHORED_ARTIFACT_ROOT}/bridge_inputs/bridge_scalers_all50.json}"
    fresh_run "${PYTHON_BIN}" -u scripts/bind_prediction_anchored_bridge_teachers.py \
      --execution-spec "${PAB_EXECUTION_SPEC}" --selected-consumer "${selected}" \
      --physical45-recipe "${PAB_PHYSICAL45_RECIPE}" --all50-recipe "${PAB_ALL50_RECIPE}" \
      --all50-scaler "${PAB_ALL50_SCALER}" \
      --consumer-evaluation-root "${PREDICTION_ANCHORED_ARTIFACT_ROOT}/consumer_evaluations" \
      --consumer-publication-root "${PREDICTION_ANCHORED_ARTIFACT_ROOT}/consumers" \
      --output-dir "${PREDICTION_ANCHORED_ARTIFACT_ROOT}/bindings" "${dry_flag[@]}"
    ;;
  B5_RELEASE)
    pab_require_env PAB_EXECUTION_SPEC
    for pair in \
      "primary.json:physical45_selected_bridge_teacher" \
      "all50.json:all50_selected_bridge_teacher" \
      "primary.json:physical45_selected_teacher_on_f0_control"; do
      binding_name="${pair%%:*}"
      namespace="${pair#*:}"
      args=("${PYTHON_BIN}" -u scripts/validate_prediction_anchored_teacher_logits.py
        --binding "${PREDICTION_ANCHORED_ARTIFACT_ROOT}/bindings/${binding_name}"
        --namespace-dir "${PREDICTION_ANCHORED_ARTIFACT_ROOT}/teacher_logits/${namespace}"
        --execution-spec "${PAB_EXECUTION_SPEC}")
      [[ "${binding_name}" == "primary.json" ]] && args+=(--selected-consumer "${selected}")
      args+=("${dry_flag[@]}")
      fresh_run "${args[@]}"
    done
    if [[ -f "${PREDICTION_ANCHORED_ARTIFACT_ROOT}/bindings/alternate.json" ]]; then
      fresh_run "${PYTHON_BIN}" -u scripts/validate_prediction_anchored_teacher_logits.py \
        --binding "${PREDICTION_ANCHORED_ARTIFACT_ROOT}/bindings/alternate.json" \
        --namespace-dir "${PREDICTION_ANCHORED_ARTIFACT_ROOT}/teacher_logits/physical45_alternate_bridge_teacher" \
        --execution-spec "${PAB_EXECUTION_SPEC}" "${dry_flag[@]}"
    fi
    ;;
  B6_SELECT)
    pab_require_env PAB_REGISTRY
    pab_require_env PAB_EXECUTION_SPEC
    : "${PAB_R0_CHECKPOINT:=${PREDICTION_ANCHORED_ARTIFACT_ROOT}/r0/r0_weights.pt}"
    : "${PAB_SEMANTIC_EVIDENCE_ROOT:=${PREDICTION_ANCHORED_ARTIFACT_ROOT}/semantic_evidence}"
    if pab_is_dry_run; then
      echo "B6_SELECT dry-run: paired publications remain unopened"
      exit 0
    fi
    fresh_run "${PYTHON_BIN}" -u scripts/deploy_prediction_anchored_bridge.py select \
      --registry "${PAB_REGISTRY}" --artifact-root "${PREDICTION_ANCHORED_ARTIFACT_ROOT}" \
      --r0-checkpoint "${PAB_R0_CHECKPOINT}" --selected-consumer "${selected}" \
      --semantic-evidence-root "${PAB_SEMANTIC_EVIDENCE_ROOT}" \
      --evidence-output "${PREDICTION_ANCHORED_ARTIFACT_ROOT}/selection/deployable_replica_evidence.json" \
      --output "${PREDICTION_ANCHORED_ARTIFACT_ROOT}/selection/deployable_preconfirmation.json"
    ;;
  DEPLOY_CONFIRM)
    pab_require_env PAB_EXECUTION_SPEC
    : "${PAB_R0_CHECKPOINT:=${PREDICTION_ANCHORED_ARTIFACT_ROOT}/r0/r0_weights.pt}"
    : "${PAB_PHYSICAL45_SCALER:=${PREDICTION_ANCHORED_ARTIFACT_ROOT}/bridge_inputs/bridge_scalers_physical45.json}"
    if pab_is_dry_run; then
      echo "DEPLOY_CONFIRM dry-run: sealed stack_val_deploy remains unopened"
      exit 0
    fi
    fresh_run "${PYTHON_BIN}" -u scripts/deploy_prediction_anchored_bridge.py confirm \
      --execution-spec "${PAB_EXECUTION_SPEC}" \
      --preconfirmation "${PREDICTION_ANCHORED_ARTIFACT_ROOT}/selection/deployable_preconfirmation.json" \
      --r0-checkpoint "${PAB_R0_CHECKPOINT}" --physical45-scaler "${PAB_PHYSICAL45_SCALER}" \
      --selected-consumer "${selected}" --output-dir "${PREDICTION_ANCHORED_ARTIFACT_ROOT}/selection" \
      --device "${DEVICE}"
    ;;
  REPORT_EXPORT)
    pab_require_env PAB_REGISTRY
    pab_require_env PAB_EXECUTION_SPEC
    pab_require_env PAB_RESERVATIONS
    if pab_is_dry_run; then
      echo "REPORT_EXPORT dry-run: bundle export precedes automatic publication-derived reports"
      exit 0
    fi
    : "${PAB_R0_CHECKPOINT:=${PREDICTION_ANCHORED_ARTIFACT_ROOT}/r0/r0_weights.pt}"
    : "${PAB_PHYSICAL45_SCALER:=${PREDICTION_ANCHORED_ARTIFACT_ROOT}/bridge_inputs/bridge_scalers_physical45.json}"
    fresh_run "${PYTHON_BIN}" -u scripts/deploy_prediction_anchored_bridge.py export \
      --execution-spec "${PAB_EXECUTION_SPEC}" \
      --locked-deployable "${PREDICTION_ANCHORED_ARTIFACT_ROOT}/selection/locked_deployable.json" \
      --r0-checkpoint "${PAB_R0_CHECKPOINT}" --physical45-scaler "${PAB_PHYSICAL45_SCALER}" \
      --selected-consumer "${selected}" --output-dir "${PREDICTION_ANCHORED_ARTIFACT_ROOT}/deployable_bundle" \
      --reservations "${PAB_RESERVATIONS}" --device cpu
    fresh_run "${PYTHON_BIN}" -u scripts/evaluate_prediction_anchored_bridge_campaign.py reports \
      --registry "${PAB_REGISTRY}" --execution-spec "${PAB_EXECUTION_SPEC}" \
      --artifact-root "${PREDICTION_ANCHORED_ARTIFACT_ROOT}" --graph "${PREDICTION_ANCHORED_GRAPH}" \
      --evidence-output "${PREDICTION_ANCHORED_ARTIFACT_ROOT}/reports/automatic_report_evidence.json" \
      --output "${PREDICTION_ANCHORED_ARTIFACT_ROOT}/reports/campaign_reports.json"
    ;;
  FINAL_TEST)
    pab_require_env PAB_EXECUTION_SPEC
    pab_require_env PAB_FINAL_TEST_HLT_NPZ
    pab_require_env PAB_FINAL_TEST_HLT_METADATA
    pab_require_env PAB_PARENT_MANIFEST
    pab_require_env PAB_CHILD_MANIFEST
    final_args=("${PYTHON_BIN}" -u scripts/deploy_prediction_anchored_bridge.py final-test --hlt-only
      --bundle "${PREDICTION_ANCHORED_ARTIFACT_ROOT}/deployable_bundle/deployable_bundle.pt"
      --locked-deployable "${PREDICTION_ANCHORED_ARTIFACT_ROOT}/selection/locked_deployable.json"
      --clean-reload-audit "${PREDICTION_ANCHORED_ARTIFACT_ROOT}/deployable_bundle/clean_reload_audit.json"
      --child-manifest "${PAB_CHILD_MANIFEST}" --parent-manifest "${PAB_PARENT_MANIFEST}"
      --final-hlt-npz "${PAB_FINAL_TEST_HLT_NPZ}" --final-hlt-metadata "${PAB_FINAL_TEST_HLT_METADATA}"
      --output "${PREDICTION_ANCHORED_ARTIFACT_ROOT}/reports/final_test_hlt_only.json" --device "${DEVICE}")
    [[ -n "${PAB_FINAL_TEST_FLAGS_JSON:-}" ]] && final_args+=(--flags "${PAB_FINAL_TEST_FLAGS_JSON}")
    if pab_is_dry_run; then
      printf 'DRY RUN:'; printf ' %q' "${final_args[@]}"; printf '\n'
      exit 0
    fi
    fresh_run env \
      -u PAB_OFFLINE_NPZ -u PAB_OFFLINE_METADATA -u PAB_BINDING \
      -u PAB_PRIMARY_AGGREGATE -u PAB_ALL50_AGGREGATE -u PAB_ALTERNATE_AGGREGATE \
      -u PAB_STACK_TRAIN_DISTILL_SHA256 \
      -u PAB_PHYSICAL45_RECIPE_SHA256 -u PAB_ALL50_RECIPE_SHA256 \
      "${final_args[@]}"
    ;;
  *) echo "Unknown prediction-anchored preparation action: ${ACTION}" >&2; exit 2 ;;
esac
