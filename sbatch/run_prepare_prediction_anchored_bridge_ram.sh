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
    fresh_run "${PYTHON_BIN}" -u scripts/run_prediction_anchored_bridge_campaign.py \
      --campaign-action validate-production --registry "${PAB_REGISTRY}" \
      --reservations "${PAB_RESERVATIONS}" --artifact-root "${PREDICTION_ANCHORED_ARTIFACT_ROOT}" \
      --dry-run
    ;;
  B1)
    pab_require_env PAB_R0_CHECKPOINT
    pab_require_env PAB_PREPROCESSING_SHA256
    pab_require_env PAB_TARGET_SCHEMA_SHA256
    pab_require_env PAB_STACK_TRAIN_CONSUMER_SHA256
    pab_require_env PAB_R0_MATCHING_POLICY_JSON
    pab_require_env PAB_R0_VALIDATION_METRICS_JSON
    fresh_run "${PYTHON_BIN}" -u scripts/register_prediction_anchored_r0.py \
      --checkpoint "${PAB_R0_CHECKPOINT}" \
      --output "${PREDICTION_ANCHORED_ARTIFACT_ROOT}/r0/r0_registration.json" \
      --preprocessing-sha256 "${PAB_PREPROCESSING_SHA256}" \
      --target-schema-sha256 "${PAB_TARGET_SCHEMA_SHA256}" \
      --split-manifest-sha256 "${PAB_STACK_TRAIN_CONSUMER_SHA256}" \
      --matching-policy "${PAB_R0_MATCHING_POLICY_JSON}" \
      --validation-metrics "${PAB_R0_VALIDATION_METRICS_JSON}" "${dry_flag[@]}"
    ;;
  B2)
    for name in PAB_HLT_NPZ PAB_HLT_METADATA PAB_OFFLINE_NPZ PAB_OFFLINE_METADATA \
      PAB_R0_CHECKPOINT PAB_STACK_TRAIN_DISTILL_SHA256; do pab_require_env "${name}"; done
    fresh_run "${PYTHON_BIN}" -u scripts/audit_prediction_anchored_bridge_inputs.py \
      --hlt-npz "${PAB_HLT_NPZ}" --hlt-metadata "${PAB_HLT_METADATA}" \
      --offline-npz "${PAB_OFFLINE_NPZ}" --offline-metadata "${PAB_OFFLINE_METADATA}" \
      --r0-checkpoint "${PAB_R0_CHECKPOINT}" --ram-root "${PAB_RAM_ROOT}" \
      --allocation-id "${SLURM_JOB_ID}" --output-dir "${PREDICTION_ANCHORED_ARTIFACT_ROOT}/bridge_inputs" \
      --split-manifest-sha256 "${PAB_STACK_TRAIN_DISTILL_SHA256}" --device "${DEVICE}" \
      --world-size 1 "${dry_flag[@]}"
    ;;
  B4_SELECT)
    for name in PAB_T10_CLEAN_AGGREGATE PAB_T10_ROBUST_AGGREGATE PAB_T10_CLEAN_CHECKPOINT \
      PAB_T10_ROBUST_CHECKPOINT PAB_R0_CHECKPOINT_SHA256 PAB_PHYSICAL45_RECIPE_SHA256; do
      pab_require_env "${name}"
    done
    fresh_run "${PYTHON_BIN}" -u scripts/select_prediction_anchored_bridge_consumer.py select \
      --clean-aggregate "${PAB_T10_CLEAN_AGGREGATE}" --robust-aggregate "${PAB_T10_ROBUST_AGGREGATE}" \
      --clean-checkpoint "${PAB_T10_CLEAN_CHECKPOINT}" --robust-checkpoint "${PAB_T10_ROBUST_CHECKPOINT}" \
      --f0-checkpoint-sha256 "${PAB_R0_CHECKPOINT_SHA256}" \
      --bridge-recipe-sha256 "${PAB_PHYSICAL45_RECIPE_SHA256}" \
      --output "${PREDICTION_ANCHORED_ARTIFACT_ROOT}/selection/consumer_preconfirmation.json" "${dry_flag[@]}"
    ;;
  B4_CONFIRM)
    for name in PAB_CONSUMER_CONFIRMATION_METRICS PAB_STACK_VAL_CONSUMER_ACCESS_RECEIPT; do
      pab_require_env "${name}"
    done
    fresh_run "${PYTHON_BIN}" -u scripts/select_prediction_anchored_bridge_consumer.py confirm \
      --preconfirmation "${PREDICTION_ANCHORED_ARTIFACT_ROOT}/selection/consumer_preconfirmation.json" \
      --confirmation-metrics "${PAB_CONSUMER_CONFIRMATION_METRICS}" \
      --access-receipt "${PAB_STACK_VAL_CONSUMER_ACCESS_RECEIPT}" \
      --output-dir "${PREDICTION_ANCHORED_ARTIFACT_ROOT}/selection" "${dry_flag[@]}"
    ;;
  B5_BIND)
    [[ -f "${selected}" ]] || { echo "Stage B5 refuses a guessed consumer" >&2; exit 2; }
    for name in PAB_MODEL_VAL_SELECT_SHA256 PAB_STACK_VAL_CONSUMER_SHA256 \
      PAB_PHYSICAL45_RECIPE_SHA256 PAB_PRIMARY_AGGREGATE PAB_PRIMARY_RUN_ID \
      PAB_PRIMARY_CHECKPOINT PAB_PRIMARY_CHECKPOINT_SHA256 PAB_ALL50_AGGREGATE \
      PAB_ALL50_CHECKPOINT PAB_ALL50_CHECKPOINT_SHA256 PAB_ALL50_RECIPE_SHA256 \
      PAB_ALL50_SCALER; do pab_require_env "${name}"; done
    mkdir -p "${PREDICTION_ANCHORED_ARTIFACT_ROOT}/bindings"
    fresh_run "${PYTHON_BIN}" -u scripts/select_prediction_anchored_bridge_consumer.py bind \
      --kind primary --run-id "${PAB_PRIMARY_RUN_ID}" --aggregate "${PAB_PRIMARY_AGGREGATE}" \
      --checkpoint "${PAB_PRIMARY_CHECKPOINT}" --checkpoint-sha256 "${PAB_PRIMARY_CHECKPOINT_SHA256}" \
      --bridge-recipe-sha256 "${PAB_PHYSICAL45_RECIPE_SHA256}" \
      --model-val-select-sha256 "${PAB_MODEL_VAL_SELECT_SHA256}" \
      --stack-val-consumer-sha256 "${PAB_STACK_VAL_CONSUMER_SHA256}" \
      --selected-consumer "${selected}" --output "${PREDICTION_ANCHORED_ARTIFACT_ROOT}/bindings/primary.json" "${dry_flag[@]}"
    fresh_run "${PYTHON_BIN}" -u scripts/select_prediction_anchored_bridge_consumer.py bind \
      --kind all50 --run-id T10_all50_clean --aggregate "${PAB_ALL50_AGGREGATE}" \
      --checkpoint "${PAB_ALL50_CHECKPOINT}" --checkpoint-sha256 "${PAB_ALL50_CHECKPOINT_SHA256}" \
      --bridge-recipe-sha256 "${PAB_ALL50_RECIPE_SHA256}" --all50-scaler "${PAB_ALL50_SCALER}" \
      --model-val-select-sha256 "${PAB_MODEL_VAL_SELECT_SHA256}" \
      --stack-val-consumer-sha256 "${PAB_STACK_VAL_CONSUMER_SHA256}" \
      --output "${PREDICTION_ANCHORED_ARTIFACT_ROOT}/bindings/all50.json" "${dry_flag[@]}"
    if [[ -n "${PAB_ALTERNATE_AGGREGATE:-}" ]]; then
      for name in PAB_ALTERNATE_RUN_ID PAB_ALTERNATE_CHECKPOINT PAB_ALTERNATE_CHECKPOINT_SHA256; do pab_require_env "${name}"; done
      fresh_run "${PYTHON_BIN}" -u scripts/select_prediction_anchored_bridge_consumer.py bind \
        --kind alternate --run-id "${PAB_ALTERNATE_RUN_ID}" --aggregate "${PAB_ALTERNATE_AGGREGATE}" \
        --checkpoint "${PAB_ALTERNATE_CHECKPOINT}" --checkpoint-sha256 "${PAB_ALTERNATE_CHECKPOINT_SHA256}" \
        --bridge-recipe-sha256 "${PAB_PHYSICAL45_RECIPE_SHA256}" \
        --model-val-select-sha256 "${PAB_MODEL_VAL_SELECT_SHA256}" \
        --stack-val-consumer-sha256 "${PAB_STACK_VAL_CONSUMER_SHA256}" \
        --output "${PREDICTION_ANCHORED_ARTIFACT_ROOT}/bindings/alternate.json" "${dry_flag[@]}"
    fi
    ;;
  B5_RELEASE)
    pab_require_env PAB_STACK_TRAIN_DISTILL_SHA256
    for pair in \
      "primary.json:physical45_selected_bridge_teacher" \
      "all50.json:all50_selected_bridge_teacher" \
      "primary.json:physical45_selected_teacher_on_f0_control"; do
      binding_name="${pair%%:*}"
      namespace="${pair#*:}"
      args=("${PYTHON_BIN}" -u scripts/validate_prediction_anchored_teacher_logits.py
        --binding "${PREDICTION_ANCHORED_ARTIFACT_ROOT}/bindings/${binding_name}"
        --namespace-dir "${PREDICTION_ANCHORED_ARTIFACT_ROOT}/teacher_logits/${namespace}"
        --stack-train-distill-sha256 "${PAB_STACK_TRAIN_DISTILL_SHA256}")
      [[ "${binding_name}" == "primary.json" ]] && args+=(--selected-consumer "${selected}")
      args+=("${dry_flag[@]}")
      fresh_run "${args[@]}"
    done
    if [[ -f "${PREDICTION_ANCHORED_ARTIFACT_ROOT}/bindings/alternate.json" ]]; then
      fresh_run "${PYTHON_BIN}" -u scripts/validate_prediction_anchored_teacher_logits.py \
        --binding "${PREDICTION_ANCHORED_ARTIFACT_ROOT}/bindings/alternate.json" \
        --namespace-dir "${PREDICTION_ANCHORED_ARTIFACT_ROOT}/teacher_logits/alternate_selected_bridge_teacher" \
        --stack-train-distill-sha256 "${PAB_STACK_TRAIN_DISTILL_SHA256}" "${dry_flag[@]}"
    fi
    ;;
  B6_SELECT)
    pab_require_env PAB_REGISTRY
    pab_require_env PAB_DEPLOYABLE_REPLICA_EVIDENCE
    fresh_run "${PYTHON_BIN}" -u scripts/evaluate_prediction_anchored_bridge_campaign.py "${dry_flag[@]}" select \
      --registry "${PAB_REGISTRY}" --evidence "${PAB_DEPLOYABLE_REPLICA_EVIDENCE}" \
      --output "${PREDICTION_ANCHORED_ARTIFACT_ROOT}/selection/deployable_preconfirmation.json"
    ;;
  DEPLOY_CONFIRM)
    pab_require_env PAB_STACK_VAL_DEPLOY_ACCESS_RECEIPT
    pab_require_env PAB_DEPLOYABLE_CONFIRMATION_METRICS
    fresh_run "${PYTHON_BIN}" -u scripts/evaluate_prediction_anchored_bridge_campaign.py "${dry_flag[@]}" confirm \
      --preconfirmation "${PREDICTION_ANCHORED_ARTIFACT_ROOT}/selection/deployable_preconfirmation.json" \
      --access-receipt "${PAB_STACK_VAL_DEPLOY_ACCESS_RECEIPT}" \
      --metrics "${PAB_DEPLOYABLE_CONFIRMATION_METRICS}" \
      --output "${PREDICTION_ANCHORED_ARTIFACT_ROOT}/selection/locked_deployable.json"
    ;;
  REPORT_EXPORT)
    pab_require_env PAB_REGISTRY
    pab_require_env PAB_REPORT_EVIDENCE
    fresh_run "${PYTHON_BIN}" -u scripts/evaluate_prediction_anchored_bridge_campaign.py "${dry_flag[@]}" reports \
      --registry "${PAB_REGISTRY}" --evidence "${PAB_REPORT_EVIDENCE}" \
      --output "${PREDICTION_ANCHORED_ARTIFACT_ROOT}/reports/campaign_reports.json"
    if ! pab_is_dry_run; then
      # Export executor is model-factory specific, but its output must pass the
      # repository's HLT-only clean reload audit before final-test can unlock.
      pab_run_executor PAB_DEPLOYABLE_EXPORT_EXECUTOR \
        --locked-deployable "${PREDICTION_ANCHORED_ARTIFACT_ROOT}/selection/locked_deployable.json" \
        --output-dir "${PREDICTION_ANCHORED_ARTIFACT_ROOT}/deployable_bundle"
    fi
    ;;
  FINAL_TEST)
    pab_require_env PAB_CLEAN_RELOAD_AUDIT
    final_args=("${PYTHON_BIN}" -u scripts/evaluate_prediction_anchored_bridge_campaign.py "${dry_flag[@]}"
      validate-final-test --locked-deployable "${PREDICTION_ANCHORED_ARTIFACT_ROOT}/selection/locked_deployable.json"
      --clean-reload-audit "${PAB_CLEAN_RELOAD_AUDIT}")
    [[ -n "${PAB_FINAL_TEST_FLAGS_JSON:-}" ]] && final_args+=(--flags "${PAB_FINAL_TEST_FLAGS_JSON}")
    fresh_run "${final_args[@]}"
    if ! pab_is_dry_run; then
      pab_require_env PAB_FINAL_TEST_EXECUTOR
      [[ -f "${PAB_FINAL_TEST_EXECUTOR}" && ! -L "${PAB_FINAL_TEST_EXECUTOR}" ]] || {
        echo "Unsafe or missing configured final-test executor" >&2; exit 2;
      }
      # Do not merely pass an HLT flag: remove every campaign variable that
      # could name an offline/oracle/bridge/cache artifact from the process.
      fresh_run env \
        -u PAB_OFFLINE_NPZ -u PAB_OFFLINE_METADATA -u PAB_BINDING \
        -u PAB_PRIMARY_AGGREGATE -u PAB_ALL50_AGGREGATE -u PAB_ALTERNATE_AGGREGATE \
        -u PAB_TEACHER_FORWARD_EXECUTOR -u PAB_STACK_TRAIN_DISTILL_SHA256 \
        -u PAB_PHYSICAL45_RECIPE_SHA256 -u PAB_ALL50_RECIPE_SHA256 \
        "${PYTHON_BIN}" -u "${PAB_FINAL_TEST_EXECUTOR}" --hlt-only \
        --bundle "${PREDICTION_ANCHORED_ARTIFACT_ROOT}/deployable_bundle"
    fi
    ;;
  *) echo "Unknown prediction-anchored preparation action: ${ACTION}" >&2; exit 2 ;;
esac
