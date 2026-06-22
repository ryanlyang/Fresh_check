#!/usr/bin/env bash
# Submit a QCD-vs-Hgg Version A subtoken Particle Transformer comparison.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${UPSTREAM_DEPENDENCY:=}"
: "${SUBTOKEN_PART_QCD_HGG_TAG:=$(date +%Y%m%d_%H%M%S)}"
: "${SUBTOKEN_PART_QCD_HGG_HLT_DEGRADATION_STRENGTH:=0.6}"
SUBTOKEN_PART_QCD_HGG_HLT_TAG="${SUBTOKEN_PART_QCD_HGG_HLT_DEGRADATION_STRENGTH//./p}"
: "${SUBTOKEN_PART_QCD_HGG_ROOT:=${OUTPUT_ROOT}/subtoken_part_qcd_hgg_binary_hlt${SUBTOKEN_PART_QCD_HGG_HLT_TAG}_${SUBTOKEN_PART_QCD_HGG_TAG}}"
: "${SUBTOKEN_PART_QCD_HGG_BUILD_BINARY_INPUTS:=1}"
: "${SUBTOKEN_PART_QCD_HGG_BUILD_DIRECT_BINARY_SPLITS:=1}"
: "${SUBTOKEN_PART_QCD_HGG_BINARY_INPUT_ROOT:=${SUBTOKEN_PART_QCD_HGG_ROOT}/binary_inputs}"
: "${SUBTOKEN_PART_QCD_HGG_SOURCE_MANIFEST_PATH:=${MANIFEST_PATH}}"
: "${SUBTOKEN_PART_QCD_HGG_BINARY_MANIFEST_PATH:=${SUBTOKEN_PART_QCD_HGG_BINARY_INPUT_ROOT}/split_manifest.json.gz}"
: "${SUBTOKEN_PART_QCD_HGG_BINARY_HLT_CACHE_DIR:=${SUBTOKEN_PART_QCD_HGG_BINARY_INPUT_ROOT}/hlt_cache}"
: "${SUBTOKEN_PART_QCD_HGG_SOURCE_LABEL_NAMES:=QCD Hgg}"
: "${SUBTOKEN_PART_QCD_HGG_BINARY_LABEL_FILTER:=0 1}"
: "${SUBTOKEN_PART_QCD_HGG_VARIANTS:=hlt_part_baseline subtoken_no_gate subtoken_gate_local_only subtoken_gate_context}"

: "${SUBTOKEN_PART_QCD_HGG_MODEL_TRAIN_SIZE:=500000}"
: "${SUBTOKEN_PART_QCD_HGG_MODEL_VAL_SIZE:=150000}"
: "${SUBTOKEN_PART_QCD_HGG_STACK_TRAIN_SIZE:=500000}"
: "${SUBTOKEN_PART_QCD_HGG_STACK_VAL_SIZE:=150000}"
: "${SUBTOKEN_PART_QCD_HGG_FINAL_TEST_SIZE:=500000}"
: "${SUBTOKEN_PART_QCD_HGG_EPOCHS:=45}"
: "${SUBTOKEN_PART_QCD_HGG_SELECTION_METRIC:=fpr_at_signal_eff_0p50}"

: "${SUBTOKEN_PART_QCD_HGG_BINARY_MANIFEST_MEM:=16G}"
: "${SUBTOKEN_PART_QCD_HGG_BINARY_HLT_CACHE_MEM:=128G}"
: "${SUBTOKEN_PART_QCD_HGG_COMPAT_MEM:=160G}"
: "${SUBTOKEN_PART_QCD_HGG_BINARY_MANIFEST_TIME:=02:00:00}"
: "${SUBTOKEN_PART_QCD_HGG_BINARY_HLT_CACHE_TIME:=12:00:00}"
: "${SUBTOKEN_PART_QCD_HGG_COMPAT_TIME:=2-12:00:00}"
: "${SUBTOKEN_PART_QCD_HGG_BINARY_MANIFEST_CPUS:=2}"
: "${SUBTOKEN_PART_QCD_HGG_BINARY_HLT_CACHE_CPUS:=4}"
: "${SUBTOKEN_PART_QCD_HGG_COMPAT_CPUS:=8}"

export SUBTOKEN_PART_ROOT="${SUBTOKEN_PART_QCD_HGG_ROOT}"
export SUBTOKEN_PART_COMPAT_DIR="${SUBTOKEN_PART_ROOT}/version_a_comparison"
export SUBTOKEN_PART_FINAL_REPORT_DIR="${SUBTOKEN_PART_COMPAT_DIR}/final_report"
: "${SUBTOKEN_PART_HLT_CACHE_DIR:=${SUBTOKEN_PART_QCD_HGG_BINARY_HLT_CACHE_DIR}}"
export SUBTOKEN_PART_HLT_CACHE_DIR
export SUBTOKEN_PART_VARIANTS="${SUBTOKEN_PART_QCD_HGG_VARIANTS}"
export SUBTOKEN_PART_LABEL_FILTER_NAMES="${SUBTOKEN_PART_QCD_HGG_BINARY_LABEL_FILTER}"
export SUBTOKEN_PART_LABEL_NAMES="${SUBTOKEN_PART_QCD_HGG_SOURCE_LABEL_NAMES}"
export SUBTOKEN_PART_NUM_CLASSES=2
export SUBTOKEN_PART_MODEL_TRAIN_SIZE="${SUBTOKEN_PART_QCD_HGG_MODEL_TRAIN_SIZE}"
export SUBTOKEN_PART_MODEL_VAL_SIZE="${SUBTOKEN_PART_QCD_HGG_MODEL_VAL_SIZE}"
export SUBTOKEN_PART_STACK_VAL_SIZE="${SUBTOKEN_PART_QCD_HGG_STACK_VAL_SIZE}"
export SUBTOKEN_PART_FINAL_TEST_SIZE="${SUBTOKEN_PART_QCD_HGG_FINAL_TEST_SIZE}"
export SUBTOKEN_PART_EPOCHS="${SUBTOKEN_PART_QCD_HGG_EPOCHS}"
export SUBTOKEN_PART_SELECTION_METRIC="${SUBTOKEN_PART_QCD_HGG_SELECTION_METRIC}"
export SUBTOKEN_PART_CONFIRM_FINAL_TEST=1
export HLT_DEGRADATION_STRENGTH="${SUBTOKEN_PART_QCD_HGG_HLT_DEGRADATION_STRENGTH}"

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

fresh_split_words variant_args "${SUBTOKEN_PART_VARIANTS}"

submitter_lock_dir="${SUBTOKEN_PART_ROOT}/.submission_lock"
fresh_refuse_existing_dir "${SUBTOKEN_PART_ROOT}"
fresh_claim_new_dir "${submitter_lock_dir}"
if ! fresh_is_dry_run; then
  {
    echo "created_at=$(date -Is)"
    echo "project_dir=${PROJECT_DIR}"
    echo "source_commit=$(fresh_source_commit)"
    echo "task=QCD_vs_Hgg_subtoken_part_version_a"
    echo "root=${SUBTOKEN_PART_ROOT}"
    echo "compat_dir=${SUBTOKEN_PART_COMPAT_DIR}"
    echo "final_report_dir=${SUBTOKEN_PART_FINAL_REPORT_DIR}"
    echo "source_label_names=${SUBTOKEN_PART_QCD_HGG_SOURCE_LABEL_NAMES}"
    echo "downstream_label_filter=${SUBTOKEN_PART_LABEL_FILTER_NAMES}"
    echo "label_names=${SUBTOKEN_PART_LABEL_NAMES}"
    echo "num_classes=${SUBTOKEN_PART_NUM_CLASSES}"
    echo "build_binary_inputs=${SUBTOKEN_PART_QCD_HGG_BUILD_BINARY_INPUTS}"
    echo "build_direct_binary_splits=${SUBTOKEN_PART_QCD_HGG_BUILD_DIRECT_BINARY_SPLITS}"
    echo "source_manifest=${SUBTOKEN_PART_QCD_HGG_SOURCE_MANIFEST_PATH}"
    echo "binary_manifest=${SUBTOKEN_PART_QCD_HGG_BINARY_MANIFEST_PATH}"
    echo "binary_hlt_cache=${SUBTOKEN_PART_QCD_HGG_BINARY_HLT_CACHE_DIR}"
    echo "hlt_degradation_strength=${SUBTOKEN_PART_QCD_HGG_HLT_DEGRADATION_STRENGTH}"
    echo "variants=$(fresh_join_by_space "${variant_args[@]}")"
    echo "selection_metric=${SUBTOKEN_PART_SELECTION_METRIC}"
    echo "epochs=${SUBTOKEN_PART_EPOCHS}"
    echo "binary_manifest_time=${SUBTOKEN_PART_QCD_HGG_BINARY_MANIFEST_TIME}"
    echo "binary_hlt_cache_time=${SUBTOKEN_PART_QCD_HGG_BINARY_HLT_CACHE_TIME}"
    echo "compat_time=${SUBTOKEN_PART_QCD_HGG_COMPAT_TIME}"
  } > "${submitter_lock_dir}/metadata.txt"
fi

binary_manifest_jid=""
binary_hlt_cache_jid=""
compat_jid=""
input_dependency="${UPSTREAM_DEPENDENCY}"

if fresh_bool_enabled "${SUBTOKEN_PART_QCD_HGG_BUILD_BINARY_INPUTS}"; then
  export LABEL_FILTER_OUTPUT_MANIFEST_PATH="${SUBTOKEN_PART_QCD_HGG_BINARY_MANIFEST_PATH}"
  export LABEL_FILTER_NAMES="${SUBTOKEN_PART_QCD_HGG_SOURCE_LABEL_NAMES}"
  export LABEL_FILTER_MANIFEST_PATH="${SUBTOKEN_PART_QCD_HGG_BINARY_MANIFEST_PATH}"
  export LABEL_FILTER_HLT_CACHE_DIR="${SUBTOKEN_PART_QCD_HGG_BINARY_HLT_CACHE_DIR}"
  export LABEL_FILTER_HLT_SPLITS="model_train model_val stack_train stack_val final_test"
  export LABEL_FILTER_MODEL_TRAIN_SIZE="${SUBTOKEN_PART_QCD_HGG_MODEL_TRAIN_SIZE}"
  export LABEL_FILTER_MODEL_VAL_SIZE="${SUBTOKEN_PART_QCD_HGG_MODEL_VAL_SIZE}"
  export LABEL_FILTER_STACK_TRAIN_SIZE="${SUBTOKEN_PART_QCD_HGG_STACK_TRAIN_SIZE}"
  export LABEL_FILTER_STACK_VAL_SIZE="${SUBTOKEN_PART_QCD_HGG_STACK_VAL_SIZE}"
  export LABEL_FILTER_FINAL_TEST_SIZE="${SUBTOKEN_PART_QCD_HGG_FINAL_TEST_SIZE}"

  if fresh_bool_enabled "${SUBTOKEN_PART_QCD_HGG_BUILD_DIRECT_BINARY_SPLITS}"; then
    mapfile -t manifest_args < <(
      afterok_args \
        "${UPSTREAM_DEPENDENCY}" \
        --time="${SUBTOKEN_PART_QCD_HGG_BINARY_MANIFEST_TIME}" \
        --cpus-per-task="${SUBTOKEN_PART_QCD_HGG_BINARY_MANIFEST_CPUS}" \
        --mem="${SUBTOKEN_PART_QCD_HGG_BINARY_MANIFEST_MEM}" \
        "${SCRIPT_DIR}/run_build_label_filtered_fresh_splits.sh"
    )
  else
    export LABEL_FILTER_SOURCE_MANIFEST_PATH="${SUBTOKEN_PART_QCD_HGG_SOURCE_MANIFEST_PATH}"
    export LABEL_FILTER_REMAP_LABELS=1
    mapfile -t manifest_args < <(
      afterok_args \
        "${UPSTREAM_DEPENDENCY}" \
        --time="${SUBTOKEN_PART_QCD_HGG_BINARY_MANIFEST_TIME}" \
        --cpus-per-task="${SUBTOKEN_PART_QCD_HGG_BINARY_MANIFEST_CPUS}" \
        --mem="${SUBTOKEN_PART_QCD_HGG_BINARY_MANIFEST_MEM}" \
        "${SCRIPT_DIR}/run_build_label_filtered_split_manifest.sh"
    )
  fi
  binary_manifest_jid="$(submit_job "subtoken_qcdhgg_binary_manifest" "${manifest_args[@]}")"
  echo "submitted subtoken_qcdhgg_binary_manifest=${binary_manifest_jid}"

  binary_hlt_cache_jid="$(submit_job "subtoken_qcdhgg_binary_hlt_cache" \
    --time="${SUBTOKEN_PART_QCD_HGG_BINARY_HLT_CACHE_TIME}" \
    --cpus-per-task="${SUBTOKEN_PART_QCD_HGG_BINARY_HLT_CACHE_CPUS}" \
    --mem="${SUBTOKEN_PART_QCD_HGG_BINARY_HLT_CACHE_MEM}" \
    --dependency="afterok:${binary_manifest_jid}" \
    "${SCRIPT_DIR}/run_build_label_filtered_hlt_cache.sh")"
  echo "submitted subtoken_qcdhgg_binary_hlt_cache=${binary_hlt_cache_jid}"
  input_dependency="${binary_hlt_cache_jid}"
fi

mapfile -t compat_args < <(
  afterok_args \
    "${input_dependency}" \
    --time="${SUBTOKEN_PART_QCD_HGG_COMPAT_TIME}" \
    --cpus-per-task="${SUBTOKEN_PART_QCD_HGG_COMPAT_CPUS}" \
    --mem="${SUBTOKEN_PART_QCD_HGG_COMPAT_MEM}" \
    "${SCRIPT_DIR}/run_subtoken_part_compat.sh"
)
compat_jid="$(submit_job "subtoken_qcdhgg_version_a_compat" "${compat_args[@]}")"
echo "submitted subtoken_qcdhgg_version_a_compat=${compat_jid}"

cat <<SUMMARY
subtoken_part_qcd_hgg_binary_submission:
  task: QCD_vs_Hgg_subtoken_part_version_a
  source_label_names: ${SUBTOKEN_PART_QCD_HGG_SOURCE_LABEL_NAMES}
  downstream_label_filter: ${SUBTOKEN_PART_LABEL_FILTER_NAMES}
  label_names: ${SUBTOKEN_PART_LABEL_NAMES}
  num_classes: ${SUBTOKEN_PART_NUM_CLASSES}
  root: ${SUBTOKEN_PART_ROOT}
  hlt_degradation_strength: ${SUBTOKEN_PART_QCD_HGG_HLT_DEGRADATION_STRENGTH}
  binary_inputs:
    build_enabled: ${SUBTOKEN_PART_QCD_HGG_BUILD_BINARY_INPUTS}
    direct_binary_splits: ${SUBTOKEN_PART_QCD_HGG_BUILD_DIRECT_BINARY_SPLITS}
    source_manifest: ${SUBTOKEN_PART_QCD_HGG_SOURCE_MANIFEST_PATH}
    filtered_manifest: ${SUBTOKEN_PART_QCD_HGG_BINARY_MANIFEST_PATH}
    filtered_hlt_cache: ${SUBTOKEN_PART_QCD_HGG_BINARY_HLT_CACHE_DIR}
    manifest_job_id: ${binary_manifest_jid:-none}
    hlt_cache_job_id: ${binary_hlt_cache_jid:-none}
  comparison_job_id: ${compat_jid}
  expected_jobs:
    binary_manifest: $([[ -n "${binary_manifest_jid}" ]] && echo 1 || echo 0)
    binary_hlt_cache: $([[ -n "${binary_hlt_cache_jid}" ]] && echo 1 || echo 0)
    subtoken_part_compat: 1
    total_submitted: ${submit_count}
  requested_split_caps:
    model_train: ${SUBTOKEN_PART_QCD_HGG_MODEL_TRAIN_SIZE}
    model_val: ${SUBTOKEN_PART_QCD_HGG_MODEL_VAL_SIZE}
    stack_train: ${SUBTOKEN_PART_QCD_HGG_STACK_TRAIN_SIZE}
    stack_val: ${SUBTOKEN_PART_QCD_HGG_STACK_VAL_SIZE}
    final_test: ${SUBTOKEN_PART_QCD_HGG_FINAL_TEST_SIZE}
  model:
    variants: $(fresh_join_by_space "${variant_args[@]}")
    epochs: ${SUBTOKEN_PART_EPOCHS}
    selection_metric: ${SUBTOKEN_PART_SELECTION_METRIC}
  resources:
    binary_manifest: time=${SUBTOKEN_PART_QCD_HGG_BINARY_MANIFEST_TIME} mem=${SUBTOKEN_PART_QCD_HGG_BINARY_MANIFEST_MEM} cpus=${SUBTOKEN_PART_QCD_HGG_BINARY_MANIFEST_CPUS}
    binary_hlt_cache: time=${SUBTOKEN_PART_QCD_HGG_BINARY_HLT_CACHE_TIME} mem=${SUBTOKEN_PART_QCD_HGG_BINARY_HLT_CACHE_MEM} cpus=${SUBTOKEN_PART_QCD_HGG_BINARY_HLT_CACHE_CPUS}
    compat: time=${SUBTOKEN_PART_QCD_HGG_COMPAT_TIME} mem=${SUBTOKEN_PART_QCD_HGG_COMPAT_MEM} cpus=${SUBTOKEN_PART_QCD_HGG_COMPAT_CPUS}
  outputs:
    comparison_run_report: ${SUBTOKEN_PART_COMPAT_DIR}/run_report.json
    comparison_csv: ${SUBTOKEN_PART_COMPAT_DIR}/diagnostics/comparison_metrics.csv
    final_report_json: ${SUBTOKEN_PART_FINAL_REPORT_DIR}/subtoken_part_final_report.json
    final_report_md: ${SUBTOKEN_PART_FINAL_REPORT_DIR}/subtoken_part_final_report.md
    final_metric_table: ${SUBTOKEN_PART_FINAL_REPORT_DIR}/metric_table.csv
    logs: ${PROJECT_DIR}/fresh_check_logs
SUMMARY
