#!/usr/bin/env bash
# Submit a 10-class Step 21 subtoken Particle Transformer comparison.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${UPSTREAM_DEPENDENCY:=}"
: "${SUBTOKEN_PART_10CLASS_TAG:=$(date +%Y%m%d_%H%M%S)}"
: "${SUBTOKEN_PART_10CLASS_HLT_DEGRADATION_STRENGTH:=0.6}"
SUBTOKEN_PART_10CLASS_HLT_TAG="${SUBTOKEN_PART_10CLASS_HLT_DEGRADATION_STRENGTH//./p}"
: "${SUBTOKEN_PART_10CLASS_ROOT:=${OUTPUT_ROOT}/subtoken_part_10class_hlt${SUBTOKEN_PART_10CLASS_HLT_TAG}_${SUBTOKEN_PART_10CLASS_TAG}}"
: "${SUBTOKEN_PART_10CLASS_MANIFEST_PATH:=${SUBTOKEN_PART_10CLASS_ROOT}/split_manifest.json.gz}"
: "${SUBTOKEN_PART_10CLASS_HLT_CACHE_DIR:=${SUBTOKEN_PART_10CLASS_ROOT}/hlt_cache}"
: "${SUBTOKEN_PART_10CLASS_COMPAT_DIR:=${SUBTOKEN_PART_10CLASS_ROOT}/version_a_comparison}"
: "${SUBTOKEN_PART_10CLASS_FINAL_REPORT_DIR:=${SUBTOKEN_PART_10CLASS_COMPAT_DIR}/final_report}"
: "${SUBTOKEN_PART_10CLASS_LABEL_NAMES:=QCD Hbb Hcc Hgg H4q Hqql Zqq Wqq Tbqq Tbl}"
: "${SUBTOKEN_PART_10CLASS_VARIANTS:=hlt_part_baseline subtoken_no_gate subtoken_gate_local_only subtoken_gate_context}"

: "${SUBTOKEN_PART_10CLASS_MODEL_TRAIN_SIZE:=500000}"
: "${SUBTOKEN_PART_10CLASS_MODEL_VAL_SIZE:=150000}"
: "${SUBTOKEN_PART_10CLASS_STACK_TRAIN_SIZE:=500000}"
: "${SUBTOKEN_PART_10CLASS_STACK_VAL_SIZE:=150000}"
: "${SUBTOKEN_PART_10CLASS_FINAL_TEST_SIZE:=500000}"
: "${SUBTOKEN_PART_10CLASS_EPOCHS:=45}"
: "${SUBTOKEN_PART_10CLASS_SELECTION_METRIC:=accuracy}"

: "${SUBTOKEN_PART_10CLASS_SPLIT_TIME:=04:00:00}"
: "${SUBTOKEN_PART_10CLASS_HLT_CACHE_TIME:=1-00:00:00}"
: "${SUBTOKEN_PART_10CLASS_COMPAT_TIME:=4-00:00:00}"
: "${SUBTOKEN_PART_10CLASS_SPLIT_MEM:=32G}"
: "${SUBTOKEN_PART_10CLASS_HLT_CACHE_MEM:=160G}"
: "${SUBTOKEN_PART_10CLASS_COMPAT_MEM:=160G}"
: "${SUBTOKEN_PART_10CLASS_SPLIT_CPUS:=4}"
: "${SUBTOKEN_PART_10CLASS_HLT_CACHE_CPUS:=8}"
: "${SUBTOKEN_PART_10CLASS_COMPAT_CPUS:=8}"

export MANIFEST_PATH="${SUBTOKEN_PART_10CLASS_MANIFEST_PATH}"
export HLT_CACHE_DIR="${SUBTOKEN_PART_10CLASS_HLT_CACHE_DIR}"
export HLT_DEGRADATION_STRENGTH="${SUBTOKEN_PART_10CLASS_HLT_DEGRADATION_STRENGTH}"

export MODEL_TRAIN_SIZE="${SUBTOKEN_PART_10CLASS_MODEL_TRAIN_SIZE}"
export MODEL_VAL_SIZE="${SUBTOKEN_PART_10CLASS_MODEL_VAL_SIZE}"
export STACK_TRAIN_SIZE="${SUBTOKEN_PART_10CLASS_STACK_TRAIN_SIZE}"
export STACK_VAL_SIZE="${SUBTOKEN_PART_10CLASS_STACK_VAL_SIZE}"
export FINAL_TEST_SIZE="${SUBTOKEN_PART_10CLASS_FINAL_TEST_SIZE}"

export SUBTOKEN_PART_ROOT="${SUBTOKEN_PART_10CLASS_ROOT}"
export SUBTOKEN_PART_COMPAT_DIR="${SUBTOKEN_PART_10CLASS_COMPAT_DIR}"
export SUBTOKEN_PART_FINAL_REPORT_DIR="${SUBTOKEN_PART_10CLASS_FINAL_REPORT_DIR}"
export SUBTOKEN_PART_HLT_CACHE_DIR="${SUBTOKEN_PART_10CLASS_HLT_CACHE_DIR}"
export SUBTOKEN_PART_VARIANTS="${SUBTOKEN_PART_10CLASS_VARIANTS}"
export SUBTOKEN_PART_LABEL_FILTER_NAMES="${SUBTOKEN_PART_10CLASS_LABEL_NAMES}"
export SUBTOKEN_PART_LABEL_NAMES="${SUBTOKEN_PART_10CLASS_LABEL_NAMES}"
export SUBTOKEN_PART_NUM_CLASSES=10
export SUBTOKEN_PART_MODEL_TRAIN_SIZE="${SUBTOKEN_PART_10CLASS_MODEL_TRAIN_SIZE}"
export SUBTOKEN_PART_MODEL_VAL_SIZE="${SUBTOKEN_PART_10CLASS_MODEL_VAL_SIZE}"
export SUBTOKEN_PART_STACK_VAL_SIZE="${SUBTOKEN_PART_10CLASS_STACK_VAL_SIZE}"
export SUBTOKEN_PART_FINAL_TEST_SIZE="${SUBTOKEN_PART_10CLASS_FINAL_TEST_SIZE}"
export SUBTOKEN_PART_EPOCHS="${SUBTOKEN_PART_10CLASS_EPOCHS}"
export SUBTOKEN_PART_SELECTION_METRIC="${SUBTOKEN_PART_10CLASS_SELECTION_METRIC}"
export SUBTOKEN_PART_REPORT_PRIMARY_METRIC="${SUBTOKEN_PART_10CLASS_SELECTION_METRIC}"
export SUBTOKEN_PART_REPORT_COMPARISON_SPLIT="final_test"
export SUBTOKEN_PART_CONFIRM_FINAL_TEST=1

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

fresh_split_words variant_args "${SUBTOKEN_PART_10CLASS_VARIANTS}"
fresh_split_words label_name_args "${SUBTOKEN_PART_10CLASS_LABEL_NAMES}"

submitter_lock_dir="${SUBTOKEN_PART_10CLASS_ROOT}/.submission_lock"
fresh_refuse_existing_dir "${SUBTOKEN_PART_10CLASS_ROOT}"
fresh_claim_new_dir "${submitter_lock_dir}"
if ! fresh_is_dry_run; then
  {
    echo "created_at=$(date -Is)"
    echo "project_dir=${PROJECT_DIR}"
    echo "source_commit=$(fresh_source_commit)"
    echo "task=10class_subtoken_part_version_a"
    echo "root=${SUBTOKEN_PART_10CLASS_ROOT}"
    echo "manifest=${SUBTOKEN_PART_10CLASS_MANIFEST_PATH}"
    echo "hlt_cache=${SUBTOKEN_PART_10CLASS_HLT_CACHE_DIR}"
    echo "compat_dir=${SUBTOKEN_PART_10CLASS_COMPAT_DIR}"
    echo "final_report_dir=${SUBTOKEN_PART_10CLASS_FINAL_REPORT_DIR}"
    echo "hlt_degradation_strength=${SUBTOKEN_PART_10CLASS_HLT_DEGRADATION_STRENGTH}"
    echo "num_classes=10"
    echo "label_names=$(fresh_join_by_space "${label_name_args[@]}")"
    echo "variants=$(fresh_join_by_space "${variant_args[@]}")"
    echo "selection_metric=${SUBTOKEN_PART_10CLASS_SELECTION_METRIC}"
    echo "epochs=${SUBTOKEN_PART_10CLASS_EPOCHS}"
    echo "split_time=${SUBTOKEN_PART_10CLASS_SPLIT_TIME}"
    echo "hlt_cache_time=${SUBTOKEN_PART_10CLASS_HLT_CACHE_TIME}"
    echo "compat_time=${SUBTOKEN_PART_10CLASS_COMPAT_TIME}"
  } > "${submitter_lock_dir}/metadata.txt"
fi

mapfile -t split_args < <(
  afterok_args \
    "${UPSTREAM_DEPENDENCY}" \
    --time="${SUBTOKEN_PART_10CLASS_SPLIT_TIME}" \
    --cpus-per-task="${SUBTOKEN_PART_10CLASS_SPLIT_CPUS}" \
    --mem="${SUBTOKEN_PART_10CLASS_SPLIT_MEM}" \
    "${SCRIPT_DIR}/run_build_fresh_splits.sh"
)
split_jid="$(submit_job "subtoken_10class_splits" "${split_args[@]}")"
echo "submitted subtoken_10class_splits=${split_jid}"

hlt_cache_jid="$(submit_job "subtoken_10class_hlt_cache" \
  --time="${SUBTOKEN_PART_10CLASS_HLT_CACHE_TIME}" \
  --cpus-per-task="${SUBTOKEN_PART_10CLASS_HLT_CACHE_CPUS}" \
  --mem="${SUBTOKEN_PART_10CLASS_HLT_CACHE_MEM}" \
  --dependency="afterok:${split_jid}" \
  "${SCRIPT_DIR}/run_build_fresh_hlt_cache.sh")"
echo "submitted subtoken_10class_hlt_cache=${hlt_cache_jid}"

mapfile -t compat_args < <(
  afterok_args \
    "${hlt_cache_jid}" \
    --time="${SUBTOKEN_PART_10CLASS_COMPAT_TIME}" \
    --cpus-per-task="${SUBTOKEN_PART_10CLASS_COMPAT_CPUS}" \
    --mem="${SUBTOKEN_PART_10CLASS_COMPAT_MEM}" \
    "${SCRIPT_DIR}/run_subtoken_part_compat.sh"
)
compat_jid="$(submit_job "subtoken_10class_version_a_compat" "${compat_args[@]}")"
echo "submitted subtoken_10class_version_a_compat=${compat_jid}"

cat <<SUMMARY
subtoken_part_10class_submission:
  task: 10class_subtoken_part_version_a
  root: ${SUBTOKEN_PART_10CLASS_ROOT}
  hlt_degradation_strength: ${SUBTOKEN_PART_10CLASS_HLT_DEGRADATION_STRENGTH}
  num_classes: 10
  label_names: ${SUBTOKEN_PART_10CLASS_LABEL_NAMES}
  variants: $(fresh_join_by_space "${variant_args[@]}")
  selection_metric: ${SUBTOKEN_PART_10CLASS_SELECTION_METRIC}
  job_ids:
    split_manifest: ${split_jid}
    hlt_cache: ${hlt_cache_jid}
    version_a_compat: ${compat_jid}
  expected_jobs:
    split_manifest: 1
    hlt_cache: 1
    subtoken_part_compat: 1
    total_submitted: ${submit_count}
  requested_split_caps:
    model_train: ${SUBTOKEN_PART_10CLASS_MODEL_TRAIN_SIZE}
    model_val: ${SUBTOKEN_PART_10CLASS_MODEL_VAL_SIZE}
    stack_train: ${SUBTOKEN_PART_10CLASS_STACK_TRAIN_SIZE}
    stack_val: ${SUBTOKEN_PART_10CLASS_STACK_VAL_SIZE}
    final_test: ${SUBTOKEN_PART_10CLASS_FINAL_TEST_SIZE}
  resources:
    split_manifest: time=${SUBTOKEN_PART_10CLASS_SPLIT_TIME} mem=${SUBTOKEN_PART_10CLASS_SPLIT_MEM} cpus=${SUBTOKEN_PART_10CLASS_SPLIT_CPUS}
    hlt_cache: time=${SUBTOKEN_PART_10CLASS_HLT_CACHE_TIME} mem=${SUBTOKEN_PART_10CLASS_HLT_CACHE_MEM} cpus=${SUBTOKEN_PART_10CLASS_HLT_CACHE_CPUS}
    compat: time=${SUBTOKEN_PART_10CLASS_COMPAT_TIME} mem=${SUBTOKEN_PART_10CLASS_COMPAT_MEM} cpus=${SUBTOKEN_PART_10CLASS_COMPAT_CPUS}
  outputs:
    manifest: ${SUBTOKEN_PART_10CLASS_MANIFEST_PATH}
    hlt_cache: ${SUBTOKEN_PART_10CLASS_HLT_CACHE_DIR}
    comparison_run_report: ${SUBTOKEN_PART_10CLASS_COMPAT_DIR}/run_report.json
    comparison_csv: ${SUBTOKEN_PART_10CLASS_COMPAT_DIR}/diagnostics/comparison_metrics.csv
    final_report_json: ${SUBTOKEN_PART_10CLASS_FINAL_REPORT_DIR}/subtoken_part_final_report.json
    final_report_md: ${SUBTOKEN_PART_10CLASS_FINAL_REPORT_DIR}/subtoken_part_final_report.md
    final_metric_table: ${SUBTOKEN_PART_10CLASS_FINAL_REPORT_DIR}/metric_table.csv
    logs: ${PROJECT_DIR}/fresh_check_logs
SUMMARY
