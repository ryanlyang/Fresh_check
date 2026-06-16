#!/usr/bin/env bash
# Submit a tiny aggressive crossarch smoke test for correctness only.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${UPSTREAM_DEPENDENCY:=}"
: "${FUSION_UPSTREAM_DEPENDENCY:=}"
: "${CROSSARCH_AGGRESSIVE_SMOKE_TAG:=$(date +%Y%m%d_%H%M%S)}"
: "${CROSSARCH_AGGRESSIVE_SMOKE_ROOT:=${OUTPUT_ROOT}/teacher_logit_reco_crossarch_aggressive_v1_smoke_${CROSSARCH_AGGRESSIVE_SMOKE_TAG}}"

export CROSSARCH_MODEL_TRAIN_SIZE="${CROSSARCH_AGGRESSIVE_SMOKE_MODEL_TRAIN_SIZE:-10000}"
export CROSSARCH_MODEL_VAL_SIZE="${CROSSARCH_AGGRESSIVE_SMOKE_MODEL_VAL_SIZE:-2000}"
export CROSSARCH_STACK_TRAIN_SIZE="${CROSSARCH_AGGRESSIVE_SMOKE_STACK_TRAIN_SIZE:-5000}"
export CROSSARCH_STACK_VAL_SIZE="${CROSSARCH_AGGRESSIVE_SMOKE_STACK_VAL_SIZE:-2000}"
export CROSSARCH_FINAL_TEST_SIZE="${CROSSARCH_AGGRESSIVE_SMOKE_FINAL_TEST_SIZE:-10000}"

export CROSSARCH_AGGRESSIVE_ROOT="${CROSSARCH_AGGRESSIVE_SMOKE_ROOT}"
export CROSSARCH_PREDICTION_DIR="${CROSSARCH_AGGRESSIVE_SMOKE_ROOT}/predictions"
export CROSSARCH_AGGRESSIVE_PREDICTION_DIR="${CROSSARCH_PREDICTION_DIR}"
export CROSSARCH_HLT_PREDICTION_RUN_DIR="${CROSSARCH_AGGRESSIVE_SMOKE_ROOT}/prediction_runs/hlt"
export CROSSARCH_AGGRESSIVE_RECO_MODEL_DIR="${CROSSARCH_AGGRESSIVE_SMOKE_ROOT}/reco_models"
export CROSSARCH_AGGRESSIVE_RECO_PREDICTION_RUN_DIR="${CROSSARCH_AGGRESSIVE_SMOKE_ROOT}/prediction_runs/reco"
export CROSSARCH_AGGRESSIVE_RECO_DOMAIN_TAGGER_DIR="${CROSSARCH_AGGRESSIVE_SMOKE_ROOT}/reco_domain_taggers"
export CROSSARCH_AGGRESSIVE_RECO_DOMAIN_TAGGER_PREDICTION_RUN_DIR="${CROSSARCH_AGGRESSIVE_SMOKE_ROOT}/prediction_runs/reco_domain_taggers"
export CROSSARCH_AGGRESSIVE_RECO_DOMAIN_FUSION_DIR="${CROSSARCH_AGGRESSIVE_SMOKE_ROOT}/fusion_reco_domain_taggers"
export CROSSARCH_AGGRESSIVE_FUSION_DIR="${CROSSARCH_AGGRESSIVE_SMOKE_ROOT}/fusion"
export CROSSARCH_AGGRESSIVE_AUDIT_DIR="${CROSSARCH_AGGRESSIVE_SMOKE_ROOT}/audits"

export CROSSARCH_AGGRESSIVE_RECO_EPOCHS="${CROSSARCH_AGGRESSIVE_SMOKE_RECO_EPOCHS:-2}"
export CROSSARCH_AGGRESSIVE_RECO_EARLY_STOP_PATIENCE="${CROSSARCH_AGGRESSIVE_SMOKE_RECO_EARLY_STOP_PATIENCE:-1}"
export CROSSARCH_AGGRESSIVE_RECO_MAX_TRAIN_JETS="${CROSSARCH_MODEL_TRAIN_SIZE}"
export CROSSARCH_AGGRESSIVE_RECO_MAX_VAL_JETS="${CROSSARCH_MODEL_VAL_SIZE}"
export CROSSARCH_AGGRESSIVE_RECO_DOMAIN_TAGGER_EPOCHS="${CROSSARCH_AGGRESSIVE_SMOKE_ADAPTED_EPOCHS:-2}"
export CROSSARCH_AGGRESSIVE_RECO_DOMAIN_TAGGER_EARLY_STOP_PATIENCE="${CROSSARCH_AGGRESSIVE_SMOKE_ADAPTED_EARLY_STOP_PATIENCE:-1}"
export CROSSARCH_AGGRESSIVE_RECO_DOMAIN_TAGGER_MAX_TRAIN_JETS="${CROSSARCH_MODEL_TRAIN_SIZE}"
export CROSSARCH_AGGRESSIVE_RECO_DOMAIN_TAGGER_MAX_VAL_JETS="${CROSSARCH_MODEL_VAL_SIZE}"
export CROSSARCH_AGGRESSIVE_RECO_BATCH_SIZE="${CROSSARCH_AGGRESSIVE_SMOKE_RECO_BATCH_SIZE:-64}"
export CROSSARCH_AGGRESSIVE_RECO_DOMAIN_TAGGER_BATCH_SIZE="${CROSSARCH_AGGRESSIVE_SMOKE_ADAPTED_BATCH_SIZE:-64}"
export CROSSARCH_HLT_PREDICT_BATCH_SIZE="${CROSSARCH_AGGRESSIVE_SMOKE_HLT_PREDICT_BATCH_SIZE:-256}"

export CROSSARCH_AGGRESSIVE_REQUIRE_HLT_PREDICTIONS_AT_SUBMIT=0
export CROSSARCH_AGGRESSIVE_AUDIT_REQUIRE_OK=1
export CROSSARCH_AGGRESSIVE_AUDIT_CHECK_PREDICTION_ARRAYS=1
export CROSSARCH_AGGRESSIVE_REQUIRE_FUSION_OK=0

fresh_prepare_submitter

submit_count=0
submit_job() {
  local label="$1"
  shift
  submit_count=$((submit_count + 1))
  if fresh_is_dry_run; then
    printf 'DRY_RUN sbatch %s: ' "${label}" >&2
    fresh_print_shell_command sbatch "$@" >&2
    printf '\n' >&2
    local clean_label="${label//[^A-Za-z0-9_]/_}"
    printf 'DRYRUN_%s\n' "${clean_label}"
    return 0
  fi
  local output
  output="$(sbatch "$@")"
  echo "${output}" >&2
  echo "${output}" | awk '{print $NF}'
}

afterok_args() {
  local dependency="$1"
  shift
  if [[ -n "${dependency}" ]]; then
    printf '%s\n' --dependency="afterok:${dependency}"
  fi
  printf '%s\n' "$@"
}

fresh_split_words hlt_arch_args "${CROSSARCH_HLT_BASELINE_ARCHITECTURES}"
fresh_refuse_existing_dir "${CROSSARCH_AGGRESSIVE_SMOKE_ROOT}"
fresh_claim_new_dir "${CROSSARCH_AGGRESSIVE_SMOKE_ROOT}/.smoke_submission_lock"

if ! fresh_is_dry_run; then
  {
    echo "created_at=$(date -Is)"
    echo "project_dir=${PROJECT_DIR}"
    echo "source_commit=$(fresh_source_commit)"
    echo "warning=smoke metrics are for pipeline correctness only, not physics interpretation"
    echo "smoke_root=${CROSSARCH_AGGRESSIVE_SMOKE_ROOT}"
    echo "prediction_dir=${CROSSARCH_PREDICTION_DIR}"
    echo "hlt_prediction_run_dir=${CROSSARCH_HLT_PREDICTION_RUN_DIR}"
    echo "aggressive_root=${CROSSARCH_AGGRESSIVE_ROOT}"
    echo "model_train_size=${CROSSARCH_MODEL_TRAIN_SIZE}"
    echo "model_val_size=${CROSSARCH_MODEL_VAL_SIZE}"
    echo "stack_train_size=${CROSSARCH_STACK_TRAIN_SIZE}"
    echo "stack_val_size=${CROSSARCH_STACK_VAL_SIZE}"
    echo "final_test_size=${CROSSARCH_FINAL_TEST_SIZE}"
  } > "${CROSSARCH_AGGRESSIVE_SMOKE_ROOT}/.smoke_submission_lock/metadata.txt"
fi

hlt_predict_job_ids=()
for architecture in "${hlt_arch_args[@]}"; do
  model_name="$(fresh_crossarch_hlt_model_name "${architecture}")"
  fresh_refuse_existing_dir "${CROSSARCH_HLT_PREDICTION_RUN_DIR}/${model_name}"
  fresh_refuse_existing_dir "${CROSSARCH_PREDICTION_DIR}/${model_name}"
  mapfile -t pred_args < <(
    afterok_args \
      "${UPSTREAM_DEPENDENCY}" \
      --time=02:00:00 \
      --mem=64G \
      "${SCRIPT_DIR}/run_crossarch_predict_hlt_baseline.sh" \
      "${architecture}"
  )
  pred_jid="$(submit_job "crossarch_aggressive_smoke_hlt_predict_${architecture}" "${pred_args[@]}")"
  hlt_predict_job_ids+=("${pred_jid}")
  echo "submitted crossarch_aggressive_smoke_hlt_predict_${architecture}=${pred_jid}"
done

fusion_dependencies=("${hlt_predict_job_ids[@]}")
if [[ -n "${FUSION_UPSTREAM_DEPENDENCY}" ]]; then
  fusion_dependencies+=("${FUSION_UPSTREAM_DEPENDENCY}")
fi
export FUSION_UPSTREAM_DEPENDENCY="$(fresh_join_by_colon "${fusion_dependencies[@]}")"

cat <<SUMMARY
crossarch_aggressive_smoke_hlt_submission:
  warning: smoke metrics are for pipeline correctness only, not physics interpretation
  hlt_predict_job_ids: $(fresh_join_by_space "${hlt_predict_job_ids[@]}")
  hlt_prediction_afterok_extra: ${UPSTREAM_DEPENDENCY:-none}
  aggressive_fusion_afterok_extra: ${FUSION_UPSTREAM_DEPENDENCY}
  smoke_sizes:
    model_train: ${CROSSARCH_MODEL_TRAIN_SIZE}
    model_val: ${CROSSARCH_MODEL_VAL_SIZE}
    stack_train: ${CROSSARCH_STACK_TRAIN_SIZE}
    stack_val: ${CROSSARCH_STACK_VAL_SIZE}
    final_test: ${CROSSARCH_FINAL_TEST_SIZE}
  smoke_output_dirs:
    smoke_root: ${CROSSARCH_AGGRESSIVE_SMOKE_ROOT}
    predictions: ${CROSSARCH_PREDICTION_DIR}
    hlt_prediction_runs: ${CROSSARCH_HLT_PREDICTION_RUN_DIR}
SUMMARY

bash "${SCRIPT_DIR}/submit_crossarch_aggressive_experiment.sh"
