#!/usr/bin/env bash
# Submit a QCD-vs-Tbqq DETR/free-slot reconstruction + five-view experiment.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${UPSTREAM_DEPENDENCY:=}"
: "${TAGGER_UPSTREAM_DEPENDENCY:=}"
: "${DETR_SLOT_QCD_TBQQ_TAG:=$(date +%Y%m%d_%H%M%S)}"
: "${DETR_SLOT_QCD_TBQQ_ROOT:=${OUTPUT_ROOT}/detr_slot_qcd_tbqq_binary_${DETR_SLOT_QCD_TBQQ_TAG}}"
: "${DETR_SLOT_BINARY_ROOT:=${DETR_SLOT_QCD_TBQQ_ROOT}}"
: "${DETR_SLOT_TASK_NAME:=QCD_vs_Tbqq_DETR_free_slot}"
: "${DETR_SLOT_SUBMISSION_NAME:=detr_slot_qcd_tbqq_binary_submission}"
: "${DETR_SLOT_BUILD_BINARY_INPUTS:=1}"
: "${DETR_SLOT_BUILD_DIRECT_BINARY_SPLITS:=1}"
: "${DETR_SLOT_BINARY_INPUT_ROOT:=${DETR_SLOT_BINARY_ROOT}/binary_inputs}"
: "${DETR_SLOT_SOURCE_MANIFEST_PATH:=${MANIFEST_PATH}}"
: "${DETR_SLOT_BINARY_MANIFEST_PATH:=${DETR_SLOT_BINARY_INPUT_ROOT}/split_manifest.json.gz}"
: "${DETR_SLOT_BINARY_HLT_CACHE_DIR:=${DETR_SLOT_BINARY_INPUT_ROOT}/hlt_cache}"
: "${DETR_SLOT_SOURCE_LABEL_NAMES:=QCD Tbqq}"
: "${DETR_SLOT_BINARY_LABEL_FILTER:=0 1}"
: "${DETR_SLOT_HLT_DEGRADATION_STRENGTH:=${HLT_DEGRADATION_STRENGTH:-1.0}}"
: "${DETR_SLOT_BINARY_MANIFEST_MEM:=16G}"
: "${DETR_SLOT_BINARY_HLT_CACHE_MEM:=128G}"
: "${DETR_SLOT_RECO_MEM:=160G}"
: "${DETR_SLOT_CACHE_MEM:=160G}"
: "${DETR_SLOT_TAGGER_MEM:=160G}"
: "${DETR_SLOT_AUDIT_MEM:=96G}"
: "${DETR_SLOT_REPORT_MEM:=16G}"
: "${DETR_SLOT_OFFLINE_TEACHER_MEM:=128G}"
: "${DETR_SLOT_BINARY_MANIFEST_TIME:=02:00:00}"
: "${DETR_SLOT_BINARY_HLT_CACHE_TIME:=12:00:00}"
: "${DETR_SLOT_RECO_TIME:=1-00:00:00}"
: "${DETR_SLOT_CACHE_TIME:=12:00:00}"
: "${DETR_SLOT_TAGGER_TIME:=1-00:00:00}"
: "${DETR_SLOT_AUDIT_TIME:=08:00:00}"
: "${DETR_SLOT_REPORT_TIME:=01:00:00}"
: "${DETR_SLOT_OFFLINE_TEACHER_TIME:=1-00:00:00}"
: "${DETR_SLOT_BINARY_MANIFEST_CPUS:=2}"
: "${DETR_SLOT_BINARY_HLT_CACHE_CPUS:=4}"
: "${DETR_SLOT_RECO_CPUS:=8}"
: "${DETR_SLOT_CACHE_CPUS:=8}"
: "${DETR_SLOT_TAGGER_CPUS:=8}"
: "${DETR_SLOT_AUDIT_CPUS:=8}"
: "${DETR_SLOT_REPORT_CPUS:=2}"
: "${DETR_SLOT_OFFLINE_TEACHER_CPUS:=4}"
: "${DETR_SLOT_SUBMIT_OFFLINE_TEACHER_REFERENCE:=1}"
: "${DETR_SLOT_OFFLINE_TEACHER_EPOCHS:=45}"

export DETR_SLOT_ROOT="${DETR_SLOT_BINARY_ROOT}"
export DETR_SLOT_RECONSTRUCTOR_DIR="${DETR_SLOT_ROOT}/reconstructors"
export DETR_SLOT_RECONSTRUCTED_VIEW_DIR="${DETR_SLOT_ROOT}/reconstructed_views"
export DETR_SLOT_TAGGER_ROOT="${DETR_SLOT_ROOT}/taggers"
export DETR_SLOT_AUDIT_DIR="${DETR_SLOT_ROOT}/ablations/five_view_ablation_eval"
export DETR_SLOT_FINAL_REPORT_DIR="${DETR_SLOT_ROOT}/final_report"
export DETR_SLOT_OFFLINE_REFERENCE_DIR="${DETR_SLOT_ROOT}/offline_teacher_reference"
export DETR_SLOT_LABEL_FILTER_NAMES="${DETR_SLOT_BINARY_LABEL_FILTER}"
export DETR_SLOT_LABEL_NAMES="${DETR_SLOT_SOURCE_LABEL_NAMES}"
export DETR_SLOT_NUM_CLASSES=2
export DETR_SLOT_CONFIRM_FINAL_TEST=1
export HLT_DEGRADATION_STRENGTH="${DETR_SLOT_HLT_DEGRADATION_STRENGTH}"

if fresh_bool_enabled "${DETR_SLOT_BUILD_BINARY_INPUTS}"; then
  export DETR_SLOT_MANIFEST_PATH="${DETR_SLOT_BINARY_MANIFEST_PATH}"
  export DETR_SLOT_HLT_CACHE_DIR="${DETR_SLOT_BINARY_HLT_CACHE_DIR}"
fi

export DETR_SLOT_MODEL_TRAIN_SIZE="${DETR_SLOT_BINARY_MODEL_TRAIN_SIZE:-${DETR_SLOT_QCD_TBQQ_MODEL_TRAIN_SIZE:-500000}}"
export DETR_SLOT_MODEL_VAL_SIZE="${DETR_SLOT_BINARY_MODEL_VAL_SIZE:-${DETR_SLOT_QCD_TBQQ_MODEL_VAL_SIZE:-150000}}"
export DETR_SLOT_STACK_TRAIN_SIZE="${DETR_SLOT_BINARY_STACK_TRAIN_SIZE:-${DETR_SLOT_QCD_TBQQ_STACK_TRAIN_SIZE:-500000}}"
export DETR_SLOT_STACK_VAL_SIZE="${DETR_SLOT_BINARY_STACK_VAL_SIZE:-${DETR_SLOT_QCD_TBQQ_STACK_VAL_SIZE:-150000}}"
export DETR_SLOT_FINAL_TEST_SIZE="${DETR_SLOT_BINARY_FINAL_TEST_SIZE:-${DETR_SLOT_QCD_TBQQ_FINAL_TEST_SIZE:-500000}}"
export DETR_SLOT_CACHE_MAX_JETS_PER_SPLIT="${DETR_SLOT_BINARY_CACHE_MAX_JETS_PER_SPLIT:-${DETR_SLOT_QCD_TBQQ_CACHE_MAX_JETS_PER_SPLIT:-${DETR_SLOT_FINAL_TEST_SIZE}}}"
export DETR_SLOT_RECO_EPOCHS="${DETR_SLOT_BINARY_RECO_EPOCHS:-${DETR_SLOT_QCD_TBQQ_RECO_EPOCHS:-25}}"
export DETR_SLOT_TAGGER_EPOCHS="${DETR_SLOT_BINARY_TAGGER_EPOCHS:-${DETR_SLOT_QCD_TBQQ_TAGGER_EPOCHS:-45}}"
export BINARY_OFFLINE_TEACHER_EPOCHS="${DETR_SLOT_OFFLINE_TEACHER_EPOCHS}"

# Shared parent-aligned utilities still read SET_MATCHING_* names. Alias them to
# the DETR layout only inside this submitter/job graph.
export SET_MATCHING_ROOT="${DETR_SLOT_ROOT}"
export SET_MATCHING_MANIFEST_PATH="${DETR_SLOT_MANIFEST_PATH}"
export SET_MATCHING_HLT_CACHE_DIR="${DETR_SLOT_HLT_CACHE_DIR}"
export SET_MATCHING_RECO_ARCHITECTURES="${DETR_SLOT_ARCHITECTURES}"
export SET_MATCHING_RECONSTRUCTOR_DIR="${DETR_SLOT_RECONSTRUCTOR_DIR}"
export SET_MATCHING_RECONSTRUCTED_VIEW_DIR="${DETR_SLOT_RECONSTRUCTED_VIEW_DIR}"
export SET_MATCHING_TAGGER_ROOT="${DETR_SLOT_TAGGER_ROOT}"
export SET_MATCHING_ABLATION_DIR="${DETR_SLOT_AUDIT_DIR}"
export SET_MATCHING_FINAL_REPORT_DIR="${DETR_SLOT_FINAL_REPORT_DIR}"
export SET_MATCHING_LABEL_FILTER_NAMES="${DETR_SLOT_LABEL_FILTER_NAMES}"
export SET_MATCHING_LABEL_NAMES="${DETR_SLOT_LABEL_NAMES}"
export SET_MATCHING_NUM_CLASSES="${DETR_SLOT_NUM_CLASSES}"
export SET_MATCHING_MODEL_TRAIN_SIZE="${DETR_SLOT_MODEL_TRAIN_SIZE}"
export SET_MATCHING_MODEL_VAL_SIZE="${DETR_SLOT_MODEL_VAL_SIZE}"
export SET_MATCHING_STACK_TRAIN_SIZE="${DETR_SLOT_STACK_TRAIN_SIZE}"
export SET_MATCHING_STACK_VAL_SIZE="${DETR_SLOT_STACK_VAL_SIZE}"
export SET_MATCHING_FINAL_TEST_SIZE="${DETR_SLOT_FINAL_TEST_SIZE}"
export SET_MATCHING_CACHE_SPLITS="${DETR_SLOT_CACHE_SPLITS}"
export SET_MATCHING_TAGGER_VARIANTS="${DETR_SLOT_TAGGER_VARIANTS}"
export SET_MATCHING_CONFIRM_FINAL_TEST="${DETR_SLOT_CONFIRM_FINAL_TEST}"
export SET_MATCHING_EVAL_BATCH_SIZE="${DETR_SLOT_EVAL_BATCH_SIZE}"
export SET_MATCHING_EVAL_NUM_WORKERS="${DETR_SLOT_EVAL_NUM_WORKERS}"
export SET_MATCHING_EVAL_DEVICE="${DETR_SLOT_EVAL_DEVICE}"
export SET_MATCHING_EVAL_REQUIRE_ALL_CANONICAL="${DETR_SLOT_EVAL_REQUIRE_ALL_CANONICAL}"
export SET_MATCHING_EVAL_MAX_VAL_BATCHES="${DETR_SLOT_EVAL_MAX_VAL_BATCHES}"
export SET_MATCHING_EVAL_MAX_FINAL_TEST_BATCHES="${DETR_SLOT_EVAL_MAX_FINAL_TEST_BATCHES}"
export SET_MATCHING_MAX_TOKENS_PER_VIEW="${DETR_SLOT_MAX_TOKENS_PER_VIEW}"
export SET_MATCHING_MIN_TOKENS_PER_VIEW="${DETR_SLOT_MIN_TOKENS_PER_VIEW}"
export SET_MATCHING_CONFIDENCE_THRESHOLD="${DETR_SLOT_CONFIDENCE_THRESHOLD}"
export SET_MATCHING_TAGGER_SEED="${DETR_SLOT_TAGGER_SEED}"

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

fresh_split_words reco_args "${DETR_SLOT_ARCHITECTURES}"
fresh_split_words tagger_args "${DETR_SLOT_TAGGER_VARIANTS}"

submitter_lock_dir="${DETR_SLOT_ROOT}/.submission_lock"
fresh_refuse_existing_dir "${DETR_SLOT_ROOT}"
fresh_claim_new_dir "${submitter_lock_dir}"
if ! fresh_is_dry_run; then
  {
    echo "created_at=$(date -Is)"
    echo "project_dir=${PROJECT_DIR}"
    echo "source_commit=$(fresh_source_commit)"
    echo "task=${DETR_SLOT_TASK_NAME}"
    echo "root=${DETR_SLOT_ROOT}"
    echo "source_label_names=${DETR_SLOT_SOURCE_LABEL_NAMES}"
    echo "downstream_label_filter=${DETR_SLOT_LABEL_FILTER_NAMES}"
    echo "label_names=${DETR_SLOT_LABEL_NAMES}"
    echo "num_classes=${DETR_SLOT_NUM_CLASSES}"
    echo "build_binary_inputs=${DETR_SLOT_BUILD_BINARY_INPUTS}"
    echo "build_direct_binary_splits=${DETR_SLOT_BUILD_DIRECT_BINARY_SPLITS}"
    echo "source_manifest=${DETR_SLOT_SOURCE_MANIFEST_PATH}"
    echo "binary_manifest=${DETR_SLOT_MANIFEST_PATH}"
    echo "binary_hlt_cache=${DETR_SLOT_HLT_CACHE_DIR}"
    echo "hlt_degradation_strength=${DETR_SLOT_HLT_DEGRADATION_STRENGTH}"
    echo "architectures=$(fresh_join_by_space "${reco_args[@]}")"
    echo "tagger_variants=$(fresh_join_by_space "${tagger_args[@]}")"
    echo "num_slots=${DETR_SLOT_NUM_SLOTS}"
    echo "export_max_tokens=${DETR_SLOT_EXPORT_MAX_TOKENS}"
    echo "submit_offline_teacher_reference=${DETR_SLOT_SUBMIT_OFFLINE_TEACHER_REFERENCE}"
  } > "${submitter_lock_dir}/metadata.txt"
fi

binary_manifest_jid=""
binary_hlt_cache_jid=""
offline_teacher_jid=""
input_dependency="${UPSTREAM_DEPENDENCY}"
reco_train_job_ids=()
cache_job_ids=()
tagger_job_ids=()
declare -A reco_train_job_id_by_arch=()

if fresh_bool_enabled "${DETR_SLOT_BUILD_BINARY_INPUTS}"; then
  export LABEL_FILTER_OUTPUT_MANIFEST_PATH="${DETR_SLOT_BINARY_MANIFEST_PATH}"
  export LABEL_FILTER_NAMES="${DETR_SLOT_SOURCE_LABEL_NAMES}"
  export LABEL_FILTER_MANIFEST_PATH="${DETR_SLOT_BINARY_MANIFEST_PATH}"
  export LABEL_FILTER_HLT_CACHE_DIR="${DETR_SLOT_BINARY_HLT_CACHE_DIR}"
  export LABEL_FILTER_HLT_SPLITS="model_train model_val stack_train stack_val final_test"
  export LABEL_FILTER_MODEL_TRAIN_SIZE="${DETR_SLOT_MODEL_TRAIN_SIZE}"
  export LABEL_FILTER_MODEL_VAL_SIZE="${DETR_SLOT_MODEL_VAL_SIZE}"
  export LABEL_FILTER_STACK_TRAIN_SIZE="${DETR_SLOT_STACK_TRAIN_SIZE}"
  export LABEL_FILTER_STACK_VAL_SIZE="${DETR_SLOT_STACK_VAL_SIZE}"
  export LABEL_FILTER_FINAL_TEST_SIZE="${DETR_SLOT_FINAL_TEST_SIZE}"

  if fresh_bool_enabled "${DETR_SLOT_BUILD_DIRECT_BINARY_SPLITS}"; then
    mapfile -t manifest_args < <(
      afterok_args \
        "${UPSTREAM_DEPENDENCY}" \
        --time="${DETR_SLOT_BINARY_MANIFEST_TIME}" \
        --cpus-per-task="${DETR_SLOT_BINARY_MANIFEST_CPUS}" \
        --mem="${DETR_SLOT_BINARY_MANIFEST_MEM}" \
        "${SCRIPT_DIR}/run_build_label_filtered_fresh_splits.sh"
    )
  else
    export LABEL_FILTER_SOURCE_MANIFEST_PATH="${DETR_SLOT_SOURCE_MANIFEST_PATH}"
    export LABEL_FILTER_REMAP_LABELS=1
    mapfile -t manifest_args < <(
      afterok_args \
        "${UPSTREAM_DEPENDENCY}" \
        --time="${DETR_SLOT_BINARY_MANIFEST_TIME}" \
        --cpus-per-task="${DETR_SLOT_BINARY_MANIFEST_CPUS}" \
        --mem="${DETR_SLOT_BINARY_MANIFEST_MEM}" \
        "${SCRIPT_DIR}/run_build_label_filtered_split_manifest.sh"
    )
  fi
  binary_manifest_jid="$(submit_job "detrslot_binary_manifest" "${manifest_args[@]}")"
  echo "submitted detrslot_binary_manifest=${binary_manifest_jid}"

  binary_hlt_cache_jid="$(submit_job "detrslot_binary_hlt_cache" \
    --time="${DETR_SLOT_BINARY_HLT_CACHE_TIME}" \
    --cpus-per-task="${DETR_SLOT_BINARY_HLT_CACHE_CPUS}" \
    --mem="${DETR_SLOT_BINARY_HLT_CACHE_MEM}" \
    --dependency="afterok:${binary_manifest_jid}" \
    "${SCRIPT_DIR}/run_build_label_filtered_hlt_cache.sh")"
  echo "submitted detrslot_binary_hlt_cache=${binary_hlt_cache_jid}"
  input_dependency="${binary_hlt_cache_jid}"
fi

if fresh_bool_enabled "${DETR_SLOT_SUBMIT_OFFLINE_TEACHER_REFERENCE}"; then
  offline_dependency="${UPSTREAM_DEPENDENCY}"
  if [[ -n "${binary_manifest_jid}" ]]; then
    offline_dependency="${binary_manifest_jid}"
  fi
  mapfile -t offline_args < <(
    afterok_args \
      "${offline_dependency}" \
      --time="${DETR_SLOT_OFFLINE_TEACHER_TIME}" \
      --cpus-per-task="${DETR_SLOT_OFFLINE_TEACHER_CPUS}" \
      --mem="${DETR_SLOT_OFFLINE_TEACHER_MEM}" \
      "${SCRIPT_DIR}/run_train_eval_set_matching_binary_offline_teacher.sh"
  )
  offline_teacher_jid="$(submit_job "detrslot_offline_teacher_reference" "${offline_args[@]}")"
  echo "submitted detrslot_offline_teacher_reference=${offline_teacher_jid}"
fi

for architecture in "${reco_args[@]}"; do
  mapfile -t train_args < <(
    afterok_args \
      "${input_dependency}" \
      --time="${DETR_SLOT_RECO_TIME}" \
      --cpus-per-task="${DETR_SLOT_RECO_CPUS}" \
      --mem="${DETR_SLOT_RECO_MEM}" \
      "${SCRIPT_DIR}/run_train_detr_slot_reconstructor.sh" \
      "${architecture}"
  )
  train_jid="$(submit_job "detrslot_train_${architecture}" "${train_args[@]}")"
  reco_train_job_ids+=("${train_jid}")
  reco_train_job_id_by_arch["${architecture}"]="${train_jid}"
  echo "submitted detrslot_train_${architecture}=${train_jid}"
done

for architecture in "${reco_args[@]}"; do
  train_jid="${reco_train_job_id_by_arch[$architecture]}"
  cache_jid="$(submit_job "detrslot_cache_${architecture}" \
    --time="${DETR_SLOT_CACHE_TIME}" \
    --cpus-per-task="${DETR_SLOT_CACHE_CPUS}" \
    --mem="${DETR_SLOT_CACHE_MEM}" \
    --dependency="afterok:${train_jid}" \
    "${SCRIPT_DIR}/run_cache_detr_slot_reco_views.sh" \
    "${architecture}")"
  cache_job_ids+=("${cache_jid}")
  echo "submitted detrslot_cache_${architecture}=${cache_jid}"
done

cache_dep="$(fresh_join_by_colon "${cache_job_ids[@]}")"
if [[ -n "${TAGGER_UPSTREAM_DEPENDENCY}" ]]; then
  cache_dep="${cache_dep}:${TAGGER_UPSTREAM_DEPENDENCY}"
fi
for variant in "${tagger_args[@]}"; do
  tagger_jid="$(submit_job "detrslot_tagger_${variant}" \
    --time="${DETR_SLOT_TAGGER_TIME}" \
    --cpus-per-task="${DETR_SLOT_TAGGER_CPUS}" \
    --mem="${DETR_SLOT_TAGGER_MEM}" \
    --dependency="afterok:${cache_dep}" \
    "${SCRIPT_DIR}/run_train_detr_slot_five_view_tagger.sh" \
    "${variant}")"
  tagger_job_ids+=("${tagger_jid}")
  echo "submitted detrslot_tagger_${variant}=${tagger_jid}"
done

tagger_dep="$(fresh_join_by_colon "${tagger_job_ids[@]}")"
audit_jid="$(submit_job "detrslot_audit" \
  --time="${DETR_SLOT_AUDIT_TIME}" \
  --cpus-per-task="${DETR_SLOT_AUDIT_CPUS}" \
  --mem="${DETR_SLOT_AUDIT_MEM}" \
  --dependency="afterok:${tagger_dep}" \
  "${SCRIPT_DIR}/run_audit_five_view_tagger.sh")"
echo "submitted detrslot_audit=${audit_jid}"

final_dep="${audit_jid}"
if [[ -n "${offline_teacher_jid}" ]]; then
  final_dep="${final_dep}:${offline_teacher_jid}"
fi
final_report_jid="$(submit_job "detrslot_final_report" \
  --time="${DETR_SLOT_REPORT_TIME}" \
  --cpus-per-task="${DETR_SLOT_REPORT_CPUS}" \
  --mem="${DETR_SLOT_REPORT_MEM}" \
  --dependency="afterok:${final_dep}" \
  "${SCRIPT_DIR}/run_write_detr_slot_final_report.sh")"
echo "submitted detrslot_final_report=${final_report_jid}"

cat <<SUMMARY
${DETR_SLOT_SUBMISSION_NAME}:
  task: ${DETR_SLOT_TASK_NAME}
  source_label_names: ${DETR_SLOT_SOURCE_LABEL_NAMES}
  downstream_label_filter: ${DETR_SLOT_LABEL_FILTER_NAMES}
  num_classes: ${DETR_SLOT_NUM_CLASSES}
  binary_inputs:
    build_enabled: ${DETR_SLOT_BUILD_BINARY_INPUTS}
    direct_binary_splits: ${DETR_SLOT_BUILD_DIRECT_BINARY_SPLITS}
    source_manifest: ${DETR_SLOT_SOURCE_MANIFEST_PATH}
    filtered_manifest: ${DETR_SLOT_MANIFEST_PATH}
    filtered_hlt_cache: ${DETR_SLOT_HLT_CACHE_DIR}
    hlt_degradation_strength: ${DETR_SLOT_HLT_DEGRADATION_STRENGTH}
    manifest_job_id: ${binary_manifest_jid:-none}
    hlt_cache_job_id: ${binary_hlt_cache_jid:-none}
  offline_teacher_reference_job_id: ${offline_teacher_jid:-none}
  reco_train_job_ids: $(fresh_join_by_space "${reco_train_job_ids[@]}")
  cache_job_ids: $(fresh_join_by_space "${cache_job_ids[@]}")
  tagger_job_ids: $(fresh_join_by_space "${tagger_job_ids[@]}")
  audit_job_id: ${audit_jid}
  final_report_job_id: ${final_report_jid}
  expected_jobs:
    binary_manifest: $([[ -n "${binary_manifest_jid}" ]] && echo 1 || echo 0)
    binary_hlt_cache: $([[ -n "${binary_hlt_cache_jid}" ]] && echo 1 || echo 0)
    offline_teacher_reference: $([[ -n "${offline_teacher_jid}" ]] && echo 1 || echo 0)
    detr_reco_train: ${#reco_train_job_ids[@]}
    detr_cache_reconstructed_views: ${#cache_job_ids[@]}
    detr_tagger_train: ${#tagger_job_ids[@]}
    five_view_audit: 1
    detr_final_report: 1
    total_submitted: ${submit_count}
  requested_split_caps:
    model_train: ${DETR_SLOT_MODEL_TRAIN_SIZE}
    model_val: ${DETR_SLOT_MODEL_VAL_SIZE}
    stack_train: ${DETR_SLOT_STACK_TRAIN_SIZE}
    stack_val: ${DETR_SLOT_STACK_VAL_SIZE}
    final_test: ${DETR_SLOT_FINAL_TEST_SIZE}
  detr_reco:
    architectures: $(fresh_join_by_space "${reco_args[@]}")
    num_slots: ${DETR_SLOT_NUM_SLOTS}
    export_max_tokens: ${DETR_SLOT_EXPORT_MAX_TOKENS}
    epochs: ${DETR_SLOT_RECO_EPOCHS}
  taggers:
    variants: $(fresh_join_by_space "${tagger_args[@]}")
    epochs: ${DETR_SLOT_TAGGER_EPOCHS}
    selection_metric: ${DETR_SLOT_TAGGER_SELECTION_METRIC}
  output_dirs:
    root: ${DETR_SLOT_ROOT}
    reconstructors: ${DETR_SLOT_RECONSTRUCTOR_DIR}
    reconstructed_views: ${DETR_SLOT_RECONSTRUCTED_VIEW_DIR}
    taggers: ${DETR_SLOT_TAGGER_ROOT}
    audit: ${DETR_SLOT_AUDIT_DIR}
    final_report: ${DETR_SLOT_FINAL_REPORT_DIR}
    offline_teacher_reference: ${DETR_SLOT_OFFLINE_REFERENCE_DIR}
    logs: ${PROJECT_DIR}/fresh_check_logs
SUMMARY
