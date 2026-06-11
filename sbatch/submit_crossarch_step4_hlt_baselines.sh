#!/usr/bin/env bash
# Submit four cross-architecture direct HLT baselines and prediction jobs.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${UPSTREAM_DEPENDENCY:=}"

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

dep_args() {
  local dependency="$1"
  shift
  if [[ -n "${dependency}" ]]; then
    printf '%s\n' --dependency="afterok:${dependency}"
  fi
  printf '%s\n' "$@"
}

fresh_split_words arch_args "${CROSSARCH_HLT_BASELINE_ARCHITECTURES}"
submitter_lock_dir="${CROSSARCH_ROOT}/.step4_hlt_baseline_submission_lock"
fresh_claim_new_dir "${submitter_lock_dir}"
if ! fresh_is_dry_run; then
  {
    echo "created_at=$(date -Is)"
    echo "project_dir=${PROJECT_DIR}"
    echo "source_commit=$(fresh_source_commit)"
    echo "architectures=$(fresh_join_by_space "${arch_args[@]}")"
    echo "hlt_cache_dir=${CROSSARCH_HLT_CACHE_DIR}"
    echo "hlt_baseline_dir=${CROSSARCH_HLT_BASELINE_DIR}"
    echo "prediction_dir=${CROSSARCH_PREDICTION_DIR}"
    echo "model_train_size=${CROSSARCH_MODEL_TRAIN_SIZE}"
    echo "model_val_size=${CROSSARCH_MODEL_VAL_SIZE}"
    echo "stack_train_size=${CROSSARCH_STACK_TRAIN_SIZE}"
    echo "stack_val_size=${CROSSARCH_STACK_VAL_SIZE}"
    echo "final_test_size=${CROSSARCH_FINAL_TEST_SIZE}"
  } > "${submitter_lock_dir}/metadata.txt"
fi

train_job_ids=()
predict_job_ids=()
for architecture in "${arch_args[@]}"; do
  model_name="$(fresh_crossarch_hlt_model_name "${architecture}")"
  fresh_refuse_existing_dir "${CROSSARCH_HLT_BASELINE_DIR}/${architecture}"
  mapfile -t train_args < <(dep_args "${UPSTREAM_DEPENDENCY}" "${SCRIPT_DIR}/run_crossarch_train_hlt_baseline.sh" "${architecture}")
  train_jid="$(submit_job "crossarch_hlt_train_${architecture}" "${train_args[@]}")"
  train_job_ids+=("${train_jid}")
  echo "submitted crossarch_hlt_train_${architecture}=${train_jid}"

  fresh_refuse_existing_dir "${CROSSARCH_PREDICTION_DIR}/${model_name}"
  pred_jid="$(submit_job "crossarch_hlt_predict_${architecture}" \
    --dependency="afterok:${train_jid}" \
    "${SCRIPT_DIR}/run_crossarch_predict_hlt_baseline.sh" "${architecture}")"
  predict_job_ids+=("${pred_jid}")
  echo "submitted crossarch_hlt_predict_${architecture}=${pred_jid}"
done

cat <<SUMMARY
crossarch_step4_hlt_baselines_submission:
  train_job_ids: $(fresh_join_by_space "${train_job_ids[@]}")
  predict_job_ids: $(fresh_join_by_space "${predict_job_ids[@]}")
  dependency_summary:
    train_afterok: ${UPSTREAM_DEPENDENCY:-none}
    each_predict_after_its_train: true
  architectures: $(fresh_join_by_space "${arch_args[@]}")
  split_sizes:
    model_train: ${CROSSARCH_MODEL_TRAIN_SIZE}
    model_val: ${CROSSARCH_MODEL_VAL_SIZE}
    stack_train: ${CROSSARCH_STACK_TRAIN_SIZE}
    stack_val: ${CROSSARCH_STACK_VAL_SIZE}
    final_test: ${CROSSARCH_FINAL_TEST_SIZE}
  output_dirs:
    root: ${CROSSARCH_ROOT}
    hlt_baselines: ${CROSSARCH_HLT_BASELINE_DIR}
    predictions: ${CROSSARCH_PREDICTION_DIR}
    prediction_runs: ${CROSSARCH_HLT_PREDICTION_RUN_DIR}
    logs: ${PROJECT_DIR}/fresh_check_logs
SUMMARY
