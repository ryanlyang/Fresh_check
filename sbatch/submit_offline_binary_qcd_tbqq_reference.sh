#!/usr/bin/env bash
# Submit an offline-only ParT reference for QCD-vs-Tbqq on a compact binary manifest.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${UPSTREAM_DEPENDENCY:=}"
: "${QCD_TBQQ_TAG:=$(date +%Y%m%d_%H%M%S)}"
: "${QCD_TBQQ_ROOT:=${OUTPUT_ROOT}/offline_binary_qcd_tbqq_${QCD_TBQQ_TAG}}"
: "${QCD_TBQQ_SOURCE_MANIFEST_PATH:=${MANIFEST_PATH}}"
: "${QCD_TBQQ_BINARY_INPUT_ROOT:=${QCD_TBQQ_ROOT}/binary_inputs}"
: "${QCD_TBQQ_MANIFEST_PATH:=${QCD_TBQQ_BINARY_INPUT_ROOT}/split_manifest.json.gz}"
: "${QCD_TBQQ_LABEL_NAMES:=QCD Tbqq}"
: "${QCD_TBQQ_MODEL_TRAIN_SIZE:=500000}"
: "${QCD_TBQQ_MODEL_VAL_SIZE:=150000}"
: "${QCD_TBQQ_STACK_VAL_SIZE:=150000}"
: "${QCD_TBQQ_FINAL_TEST_SIZE:=500000}"
: "${QCD_TBQQ_MANIFEST_TIME:=02:00:00}"
: "${QCD_TBQQ_MANIFEST_MEM:=16G}"
: "${QCD_TBQQ_MANIFEST_CPUS:=2}"
: "${QCD_TBQQ_OFFLINE_TIME:=1-00:00:00}"
: "${QCD_TBQQ_OFFLINE_MEM:=128G}"
: "${QCD_TBQQ_OFFLINE_CPUS:=4}"
: "${QCD_TBQQ_OFFLINE_EPOCHS:=30}"
: "${QCD_TBQQ_OFFLINE_BATCH_SIZE:=64}"
: "${QCD_TBQQ_OFFLINE_EVAL_BATCH_SIZE:=128}"
: "${QCD_TBQQ_OFFLINE_NUM_WORKERS:=2}"
: "${QCD_TBQQ_OFFLINE_MODEL_SIZE:=base}"

export SET_MATCHING_ROOT="${QCD_TBQQ_ROOT}"
export SET_MATCHING_MANIFEST_PATH="${QCD_TBQQ_MANIFEST_PATH}"
export SET_MATCHING_LABEL_FILTER_NAMES="${QCD_TBQQ_LABEL_NAMES}"
export SET_MATCHING_LABEL_NAMES="${QCD_TBQQ_LABEL_NAMES}"
export SET_MATCHING_NUM_CLASSES=2
export SET_MATCHING_MODEL_TRAIN_SIZE="${QCD_TBQQ_MODEL_TRAIN_SIZE}"
export SET_MATCHING_MODEL_VAL_SIZE="${QCD_TBQQ_MODEL_VAL_SIZE}"
export SET_MATCHING_STACK_VAL_SIZE="${QCD_TBQQ_STACK_VAL_SIZE}"
export SET_MATCHING_FINAL_TEST_SIZE="${QCD_TBQQ_FINAL_TEST_SIZE}"
export BINARY_OFFLINE_TEACHER_EPOCHS="${QCD_TBQQ_OFFLINE_EPOCHS}"
export BINARY_OFFLINE_TEACHER_BATCH_SIZE="${QCD_TBQQ_OFFLINE_BATCH_SIZE}"
export BINARY_OFFLINE_TEACHER_EVAL_BATCH_SIZE="${QCD_TBQQ_OFFLINE_EVAL_BATCH_SIZE}"
export BINARY_OFFLINE_TEACHER_NUM_WORKERS="${QCD_TBQQ_OFFLINE_NUM_WORKERS}"
export BINARY_OFFLINE_TEACHER_MODEL_SIZE="${QCD_TBQQ_OFFLINE_MODEL_SIZE}"

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

submitter_lock_dir="${QCD_TBQQ_ROOT}/.submission_lock"
fresh_refuse_existing_dir "${QCD_TBQQ_ROOT}"
fresh_claim_new_dir "${submitter_lock_dir}"
if ! fresh_is_dry_run; then
  {
    echo "created_at=$(date -Is)"
    echo "project_dir=${PROJECT_DIR}"
    echo "source_commit=$(fresh_source_commit)"
    echo "task=QCD_vs_Tbqq_binary_offline_part_reference"
    echo "root=${QCD_TBQQ_ROOT}"
    echo "source_manifest=${QCD_TBQQ_SOURCE_MANIFEST_PATH}"
    echo "binary_manifest=${QCD_TBQQ_MANIFEST_PATH}"
    echo "label_names=${QCD_TBQQ_LABEL_NAMES}"
    echo "model_train_size=${QCD_TBQQ_MODEL_TRAIN_SIZE}"
    echo "model_val_size=${QCD_TBQQ_MODEL_VAL_SIZE}"
    echo "stack_val_size=${QCD_TBQQ_STACK_VAL_SIZE}"
    echo "final_test_size=${QCD_TBQQ_FINAL_TEST_SIZE}"
    echo "manifest_time=${QCD_TBQQ_MANIFEST_TIME}"
    echo "manifest_mem=${QCD_TBQQ_MANIFEST_MEM}"
    echo "manifest_cpus=${QCD_TBQQ_MANIFEST_CPUS}"
    echo "offline_time=${QCD_TBQQ_OFFLINE_TIME}"
    echo "offline_mem=${QCD_TBQQ_OFFLINE_MEM}"
    echo "offline_cpus=${QCD_TBQQ_OFFLINE_CPUS}"
  } > "${submitter_lock_dir}/metadata.txt"
fi

export LABEL_FILTER_SOURCE_MANIFEST_PATH="${QCD_TBQQ_SOURCE_MANIFEST_PATH}"
export LABEL_FILTER_OUTPUT_MANIFEST_PATH="${QCD_TBQQ_MANIFEST_PATH}"
export LABEL_FILTER_NAMES="${QCD_TBQQ_LABEL_NAMES}"
export LABEL_FILTER_REMAP_LABELS=1

mapfile -t manifest_args < <(
  afterok_args \
    "${UPSTREAM_DEPENDENCY}" \
    --time="${QCD_TBQQ_MANIFEST_TIME}" \
    --cpus-per-task="${QCD_TBQQ_MANIFEST_CPUS}" \
    --mem="${QCD_TBQQ_MANIFEST_MEM}" \
    "${SCRIPT_DIR}/run_build_label_filtered_split_manifest.sh"
)
manifest_jid="$(submit_job "qcd_tbqq_binary_manifest" "${manifest_args[@]}")"
echo "submitted qcd_tbqq_binary_manifest=${manifest_jid}"

offline_jid="$(submit_job "qcd_tbqq_offline_part_reference" \
  --time="${QCD_TBQQ_OFFLINE_TIME}" \
  --cpus-per-task="${QCD_TBQQ_OFFLINE_CPUS}" \
  --mem="${QCD_TBQQ_OFFLINE_MEM}" \
  --dependency="afterok:${manifest_jid}" \
  "${SCRIPT_DIR}/run_train_eval_set_matching_binary_offline_teacher.sh")"
echo "submitted qcd_tbqq_offline_part_reference=${offline_jid}"

cat <<SUMMARY
qcd_tbqq_offline_binary_reference_submission:
  task: QCD_vs_Tbqq
  label_names: ${QCD_TBQQ_LABEL_NAMES}
  source_manifest: ${QCD_TBQQ_SOURCE_MANIFEST_PATH}
  binary_manifest: ${QCD_TBQQ_MANIFEST_PATH}
  root: ${QCD_TBQQ_ROOT}
  job_ids:
    binary_manifest: ${manifest_jid}
    offline_part_reference: ${offline_jid}
  expected_jobs:
    binary_manifest: 1
    offline_part_reference: 1
    total_submitted: ${submit_count}
  requested_split_caps:
    model_train: ${QCD_TBQQ_MODEL_TRAIN_SIZE}
    model_val: ${QCD_TBQQ_MODEL_VAL_SIZE}
    stack_val: ${QCD_TBQQ_STACK_VAL_SIZE}
    final_test: ${QCD_TBQQ_FINAL_TEST_SIZE}
  resources:
    binary_manifest: time=${QCD_TBQQ_MANIFEST_TIME} mem=${QCD_TBQQ_MANIFEST_MEM} cpus=${QCD_TBQQ_MANIFEST_CPUS}
    offline_part_reference: time=${QCD_TBQQ_OFFLINE_TIME} mem=${QCD_TBQQ_OFFLINE_MEM} cpus=${QCD_TBQQ_OFFLINE_CPUS}
  outputs:
    filtered_manifest_report: ${QCD_TBQQ_BINARY_INPUT_ROOT}/filtered_manifest_report.json
    offline_run_report: ${QCD_TBQQ_ROOT}/offline_teacher_reference/<run>/run_report.json
    offline_diagnostics: ${QCD_TBQQ_ROOT}/offline_teacher_reference/<run>/diagnostics/summary.csv
    logs: ${PROJECT_DIR}/fresh_check_logs
SUMMARY
