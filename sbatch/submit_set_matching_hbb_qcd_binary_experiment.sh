#!/usr/bin/env bash
# Submit a compact Hbb-vs-QCD binary set-matching multi-view experiment.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${UPSTREAM_DEPENDENCY:=}"
: "${TAGGER_UPSTREAM_DEPENDENCY:=}"
: "${HBB_QCD_TAG:=$(date +%Y%m%d_%H%M%S)}"
: "${HBB_QCD_ROOT:=${OUTPUT_ROOT}/set_matching_hbb_qcd_binary_${HBB_QCD_TAG}}"
: "${HBB_QCD_TAGGER_VARIANTS:=hlt_only hlt_plus_gt hlt_plus_pn hlt_plus_pfn hlt_plus_pcnn five_view_plain five_view_geometry five_view_no_confidence view_label_shuffle_control}"
: "${HBB_QCD_BUILD_BINARY_INPUTS:=1}"
: "${HBB_QCD_BINARY_INPUT_ROOT:=${HBB_QCD_ROOT}/binary_inputs}"
: "${HBB_QCD_SOURCE_MANIFEST_PATH:=${MANIFEST_PATH}}"
: "${HBB_QCD_BINARY_MANIFEST_PATH:=${HBB_QCD_BINARY_INPUT_ROOT}/split_manifest.json.gz}"
: "${HBB_QCD_BINARY_HLT_CACHE_DIR:=${HBB_QCD_BINARY_INPUT_ROOT}/hlt_cache}"
: "${HBB_QCD_BINARY_MANIFEST_MEM:=8G}"
: "${HBB_QCD_BINARY_HLT_CACHE_MEM:=64G}"
: "${HBB_QCD_RECO_MEM:=64G}"
: "${HBB_QCD_CACHE_MEM:=64G}"
: "${HBB_QCD_TAGGER_MEM:=64G}"
: "${HBB_QCD_AUDIT_MEM:=48G}"
: "${HBB_QCD_REPORT_MEM:=8G}"
: "${HBB_QCD_OFFLINE_TEACHER_MEM:=64G}"
: "${HBB_QCD_BINARY_MANIFEST_TIME:=01:00:00}"
: "${HBB_QCD_BINARY_HLT_CACHE_TIME:=12:00:00}"
: "${HBB_QCD_RECO_TIME:=2-00:00:00}"
: "${HBB_QCD_CACHE_TIME:=12:00:00}"
: "${HBB_QCD_TAGGER_TIME:=2-00:00:00}"
: "${HBB_QCD_AUDIT_TIME:=08:00:00}"
: "${HBB_QCD_REPORT_TIME:=01:00:00}"
: "${HBB_QCD_OFFLINE_TEACHER_TIME:=1-00:00:00}"
: "${HBB_QCD_BINARY_MANIFEST_CPUS:=2}"
: "${HBB_QCD_BINARY_HLT_CACHE_CPUS:=4}"
: "${HBB_QCD_RECO_CPUS:=4}"
: "${HBB_QCD_CACHE_CPUS:=4}"
: "${HBB_QCD_TAGGER_CPUS:=4}"
: "${HBB_QCD_AUDIT_CPUS:=4}"
: "${HBB_QCD_REPORT_CPUS:=2}"
: "${HBB_QCD_OFFLINE_TEACHER_CPUS:=4}"
: "${HBB_QCD_SUBMIT_OFFLINE_TEACHER_REFERENCE:=1}"

export SET_MATCHING_ROOT="${HBB_QCD_ROOT}"
export SET_MATCHING_RECONSTRUCTOR_DIR="${SET_MATCHING_ROOT}/reconstructors"
export SET_MATCHING_RECONSTRUCTED_VIEW_DIR="${SET_MATCHING_ROOT}/reconstructed_views"
export SET_MATCHING_TAGGER_ROOT="${SET_MATCHING_ROOT}/taggers"
export SET_MATCHING_ABLATION_DIR="${SET_MATCHING_ROOT}/ablations/five_view_ablation_eval"
export SET_MATCHING_FINAL_REPORT_DIR="${SET_MATCHING_ROOT}/final_report"

export SET_MATCHING_LABEL_FILTER_NAMES="QCD Hbb"
export SET_MATCHING_LABEL_NAMES="QCD Hbb"
export SET_MATCHING_NUM_CLASSES=2
export SET_MATCHING_TAGGER_VARIANTS="${HBB_QCD_TAGGER_VARIANTS}"
export SET_MATCHING_EVAL_REQUIRE_ALL_CANONICAL=1

if fresh_bool_enabled "${HBB_QCD_BUILD_BINARY_INPUTS}"; then
  export SET_MATCHING_MANIFEST_PATH="${HBB_QCD_BINARY_MANIFEST_PATH}"
  export SET_MATCHING_HLT_CACHE_DIR="${HBB_QCD_BINARY_HLT_CACHE_DIR}"
fi

export SET_MATCHING_MODEL_TRAIN_SIZE="${HBB_QCD_MODEL_TRAIN_SIZE:-100000}"
export SET_MATCHING_MODEL_VAL_SIZE="${HBB_QCD_MODEL_VAL_SIZE:-30000}"
export SET_MATCHING_STACK_TRAIN_SIZE="${HBB_QCD_STACK_TRAIN_SIZE:-50000}"
export SET_MATCHING_STACK_VAL_SIZE="${HBB_QCD_STACK_VAL_SIZE:-10000}"
export SET_MATCHING_FINAL_TEST_SIZE="${HBB_QCD_FINAL_TEST_SIZE:-100000}"
export SET_MATCHING_CACHE_MAX_JETS_PER_SPLIT="${HBB_QCD_CACHE_MAX_JETS_PER_SPLIT:-${SET_MATCHING_FINAL_TEST_SIZE}}"

export SET_MATCHING_RECO_EPOCHS="${HBB_QCD_RECO_EPOCHS:-20}"
export SET_MATCHING_RECO_EARLY_STOP_PATIENCE="${HBB_QCD_RECO_EARLY_STOP_PATIENCE:-5}"
export SET_MATCHING_TAGGER_EPOCHS="${HBB_QCD_TAGGER_EPOCHS:-30}"
export SET_MATCHING_TAGGER_EARLY_STOP_PATIENCE="${HBB_QCD_TAGGER_EARLY_STOP_PATIENCE:-4}"
export SET_MATCHING_TAGGER_SELECTION_METRIC="${HBB_QCD_TAGGER_SELECTION_METRIC:-fpr_at_signal_eff_0p50}"
export SET_MATCHING_CONFIRM_FINAL_TEST=1

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

fresh_split_words reco_args "${SET_MATCHING_RECO_ARCHITECTURES}"
fresh_split_words tagger_args "${SET_MATCHING_TAGGER_VARIANTS}"

submitter_lock_dir="${SET_MATCHING_ROOT}/.submission_lock"
fresh_refuse_existing_dir "${SET_MATCHING_ROOT}"
fresh_claim_new_dir "${submitter_lock_dir}"
if ! fresh_is_dry_run; then
  {
    echo "created_at=$(date -Is)"
    echo "project_dir=${PROJECT_DIR}"
    echo "source_commit=$(fresh_source_commit)"
    echo "task=Hbb_vs_QCD_binary_set_matching_multiview"
    echo "label_filter=${SET_MATCHING_LABEL_FILTER_NAMES}"
    echo "label_names=${SET_MATCHING_LABEL_NAMES}"
    echo "num_classes=${SET_MATCHING_NUM_CLASSES}"
    echo "root=${SET_MATCHING_ROOT}"
    echo "build_binary_inputs=${HBB_QCD_BUILD_BINARY_INPUTS}"
    echo "source_manifest=${HBB_QCD_SOURCE_MANIFEST_PATH}"
    echo "binary_manifest=${SET_MATCHING_MANIFEST_PATH}"
    echo "binary_hlt_cache=${SET_MATCHING_HLT_CACHE_DIR}"
    echo "tagger_variants=$(fresh_join_by_space "${tagger_args[@]}")"
    echo "binary_manifest_mem=${HBB_QCD_BINARY_MANIFEST_MEM}"
    echo "binary_hlt_cache_mem=${HBB_QCD_BINARY_HLT_CACHE_MEM}"
    echo "reco_mem=${HBB_QCD_RECO_MEM}"
    echo "cache_mem=${HBB_QCD_CACHE_MEM}"
    echo "tagger_mem=${HBB_QCD_TAGGER_MEM}"
    echo "audit_mem=${HBB_QCD_AUDIT_MEM}"
    echo "report_mem=${HBB_QCD_REPORT_MEM}"
    echo "offline_teacher_mem=${HBB_QCD_OFFLINE_TEACHER_MEM}"
    echo "binary_manifest_time=${HBB_QCD_BINARY_MANIFEST_TIME}"
    echo "binary_hlt_cache_time=${HBB_QCD_BINARY_HLT_CACHE_TIME}"
    echo "reco_time=${HBB_QCD_RECO_TIME}"
    echo "cache_time=${HBB_QCD_CACHE_TIME}"
    echo "tagger_time=${HBB_QCD_TAGGER_TIME}"
    echo "audit_time=${HBB_QCD_AUDIT_TIME}"
    echo "report_time=${HBB_QCD_REPORT_TIME}"
    echo "offline_teacher_time=${HBB_QCD_OFFLINE_TEACHER_TIME}"
    echo "submit_offline_teacher_reference=${HBB_QCD_SUBMIT_OFFLINE_TEACHER_REFERENCE}"
    echo "tagger_selection_metric=${SET_MATCHING_TAGGER_SELECTION_METRIC}"
  } > "${submitter_lock_dir}/metadata.txt"
fi

reco_train_job_ids=()
cache_job_ids=()
tagger_job_ids=()
binary_manifest_jid=""
binary_hlt_cache_jid=""
offline_teacher_jid=""
input_dependency="${UPSTREAM_DEPENDENCY}"
declare -A reco_train_job_id_by_arch=()

if fresh_bool_enabled "${HBB_QCD_BUILD_BINARY_INPUTS}"; then
  export LABEL_FILTER_SOURCE_MANIFEST_PATH="${HBB_QCD_SOURCE_MANIFEST_PATH}"
  export LABEL_FILTER_OUTPUT_MANIFEST_PATH="${HBB_QCD_BINARY_MANIFEST_PATH}"
  export LABEL_FILTER_NAMES="${SET_MATCHING_LABEL_FILTER_NAMES}"
  export LABEL_FILTER_MANIFEST_PATH="${HBB_QCD_BINARY_MANIFEST_PATH}"
  export LABEL_FILTER_HLT_CACHE_DIR="${HBB_QCD_BINARY_HLT_CACHE_DIR}"
  export LABEL_FILTER_HLT_SPLITS="model_train model_val stack_train stack_val final_test"

  mapfile -t manifest_args < <(
    afterok_args \
      "${UPSTREAM_DEPENDENCY}" \
      --time="${HBB_QCD_BINARY_MANIFEST_TIME}" \
      --cpus-per-task="${HBB_QCD_BINARY_MANIFEST_CPUS}" \
      --mem="${HBB_QCD_BINARY_MANIFEST_MEM}" \
      "${SCRIPT_DIR}/run_build_label_filtered_split_manifest.sh"
  )
  binary_manifest_jid="$(submit_job "hbbqcd_binary_manifest" "${manifest_args[@]}")"
  echo "submitted hbbqcd_binary_manifest=${binary_manifest_jid}"

  binary_hlt_cache_jid="$(submit_job "hbbqcd_binary_hlt_cache" \
    --time="${HBB_QCD_BINARY_HLT_CACHE_TIME}" \
    --cpus-per-task="${HBB_QCD_BINARY_HLT_CACHE_CPUS}" \
    --mem="${HBB_QCD_BINARY_HLT_CACHE_MEM}" \
    --dependency="afterok:${binary_manifest_jid}" \
    "${SCRIPT_DIR}/run_build_label_filtered_hlt_cache.sh")"
  echo "submitted hbbqcd_binary_hlt_cache=${binary_hlt_cache_jid}"
  input_dependency="${binary_hlt_cache_jid}"
fi

if fresh_bool_enabled "${HBB_QCD_SUBMIT_OFFLINE_TEACHER_REFERENCE}"; then
  offline_dependency="${UPSTREAM_DEPENDENCY}"
  if [[ -n "${binary_manifest_jid}" ]]; then
    offline_dependency="${binary_manifest_jid}"
  fi
  mapfile -t offline_args < <(
    afterok_args \
      "${offline_dependency}" \
      --time="${HBB_QCD_OFFLINE_TEACHER_TIME}" \
      --cpus-per-task="${HBB_QCD_OFFLINE_TEACHER_CPUS}" \
      --mem="${HBB_QCD_OFFLINE_TEACHER_MEM}" \
      "${SCRIPT_DIR}/run_train_eval_set_matching_binary_offline_teacher.sh"
  )
  offline_teacher_jid="$(submit_job "hbbqcd_offline_teacher_reference" "${offline_args[@]}")"
  echo "submitted hbbqcd_offline_teacher_reference=${offline_teacher_jid}"
fi

for architecture in "${reco_args[@]}"; do
  mapfile -t train_args < <(
    afterok_args \
      "${input_dependency}" \
      --time="${HBB_QCD_RECO_TIME}" \
      --cpus-per-task="${HBB_QCD_RECO_CPUS}" \
      --mem="${HBB_QCD_RECO_MEM}" \
      "${SCRIPT_DIR}/run_train_set_matching_reconstructor.sh" \
      "${architecture}"
  )
  train_jid="$(submit_job "hbbqcd_setmatch_train_${architecture}" "${train_args[@]}")"
  reco_train_job_ids+=("${train_jid}")
  reco_train_job_id_by_arch["${architecture}"]="${train_jid}"
  echo "submitted hbbqcd_setmatch_train_${architecture}=${train_jid}"
done

for architecture in "${reco_args[@]}"; do
  train_jid="${reco_train_job_id_by_arch[$architecture]}"
  cache_jid="$(submit_job "hbbqcd_setmatch_cache_${architecture}" \
    --time="${HBB_QCD_CACHE_TIME}" \
    --cpus-per-task="${HBB_QCD_CACHE_CPUS}" \
    --mem="${HBB_QCD_CACHE_MEM}" \
    --dependency="afterok:${train_jid}" \
    "${SCRIPT_DIR}/run_cache_set_matching_multiview.sh" \
    "${architecture}")"
  cache_job_ids+=("${cache_jid}")
  echo "submitted hbbqcd_setmatch_cache_${architecture}=${cache_jid}"
done

cache_dep="$(fresh_join_by_colon "${cache_job_ids[@]}")"
if [[ -n "${TAGGER_UPSTREAM_DEPENDENCY}" ]]; then
  cache_dep="${cache_dep}:${TAGGER_UPSTREAM_DEPENDENCY}"
fi
for variant in "${tagger_args[@]}"; do
  tagger_jid="$(submit_job "hbbqcd_setmatch_tagger_${variant}" \
    --time="${HBB_QCD_TAGGER_TIME}" \
    --cpus-per-task="${HBB_QCD_TAGGER_CPUS}" \
    --mem="${HBB_QCD_TAGGER_MEM}" \
    --dependency="afterok:${cache_dep}" \
    "${SCRIPT_DIR}/run_train_five_view_tagger.sh" \
    "${variant}")"
  tagger_job_ids+=("${tagger_jid}")
  echo "submitted hbbqcd_setmatch_tagger_${variant}=${tagger_jid}"
done

audit_dep="$(fresh_join_by_colon "${tagger_job_ids[@]}")"
audit_jid="$(submit_job "hbbqcd_setmatch_audit" \
  --time="${HBB_QCD_AUDIT_TIME}" \
  --cpus-per-task="${HBB_QCD_AUDIT_CPUS}" \
  --mem="${HBB_QCD_AUDIT_MEM}" \
  --dependency="afterok:${audit_dep}" \
  "${SCRIPT_DIR}/run_audit_five_view_tagger.sh")"
final_report_jid="$(submit_job "hbbqcd_setmatch_final_report" \
  --time="${HBB_QCD_REPORT_TIME}" \
  --cpus-per-task="${HBB_QCD_REPORT_CPUS}" \
  --mem="${HBB_QCD_REPORT_MEM}" \
  --dependency="afterok:${audit_jid}" \
  "${SCRIPT_DIR}/run_write_set_matching_multiview_final_report.sh")"

cat <<SUMMARY
hbb_qcd_binary_set_matching_submission:
  task: Hbb_vs_QCD
  label_filter: ${SET_MATCHING_LABEL_FILTER_NAMES}
  num_classes: ${SET_MATCHING_NUM_CLASSES}
  binary_inputs:
    build_enabled: ${HBB_QCD_BUILD_BINARY_INPUTS}
    source_manifest: ${HBB_QCD_SOURCE_MANIFEST_PATH}
    filtered_manifest: ${SET_MATCHING_MANIFEST_PATH}
    filtered_hlt_cache: ${SET_MATCHING_HLT_CACHE_DIR}
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
    reco_train: ${#reco_train_job_ids[@]}
    cache_reconstructed_views: ${#cache_job_ids[@]}
    tagger_train: ${#tagger_job_ids[@]}
    audit: 1
    final_report: 1
    total_submitted: ${submit_count}
  split_sizes:
    model_train: ${SET_MATCHING_MODEL_TRAIN_SIZE}
    model_val: ${SET_MATCHING_MODEL_VAL_SIZE}
    stack_train: ${SET_MATCHING_STACK_TRAIN_SIZE}
    stack_val: ${SET_MATCHING_STACK_VAL_SIZE}
    final_test: ${SET_MATCHING_FINAL_TEST_SIZE}
  tagger_selection_metric: ${SET_MATCHING_TAGGER_SELECTION_METRIC}
  memory_overrides:
    binary_manifest: ${HBB_QCD_BINARY_MANIFEST_MEM}
    binary_hlt_cache: ${HBB_QCD_BINARY_HLT_CACHE_MEM}
    reco_train: ${HBB_QCD_RECO_MEM}
    cache_reconstructed_views: ${HBB_QCD_CACHE_MEM}
    tagger_train: ${HBB_QCD_TAGGER_MEM}
    audit: ${HBB_QCD_AUDIT_MEM}
    final_report: ${HBB_QCD_REPORT_MEM}
    offline_teacher_reference: ${HBB_QCD_OFFLINE_TEACHER_MEM}
  time_overrides:
    binary_manifest: ${HBB_QCD_BINARY_MANIFEST_TIME}
    binary_hlt_cache: ${HBB_QCD_BINARY_HLT_CACHE_TIME}
    reco_train: ${HBB_QCD_RECO_TIME}
    cache_reconstructed_views: ${HBB_QCD_CACHE_TIME}
    tagger_train: ${HBB_QCD_TAGGER_TIME}
    audit: ${HBB_QCD_AUDIT_TIME}
    final_report: ${HBB_QCD_REPORT_TIME}
    offline_teacher_reference: ${HBB_QCD_OFFLINE_TEACHER_TIME}
  cpu_overrides:
    binary_manifest: ${HBB_QCD_BINARY_MANIFEST_CPUS}
    binary_hlt_cache: ${HBB_QCD_BINARY_HLT_CACHE_CPUS}
    reco_train: ${HBB_QCD_RECO_CPUS}
    cache_reconstructed_views: ${HBB_QCD_CACHE_CPUS}
    tagger_train: ${HBB_QCD_TAGGER_CPUS}
    audit: ${HBB_QCD_AUDIT_CPUS}
    final_report: ${HBB_QCD_REPORT_CPUS}
    offline_teacher_reference: ${HBB_QCD_OFFLINE_TEACHER_CPUS}
  output_dirs:
    root: ${SET_MATCHING_ROOT}
    taggers: ${SET_MATCHING_TAGGER_ROOT}
    audit: ${SET_MATCHING_ABLATION_DIR}
    final_report: ${SET_MATCHING_FINAL_REPORT_DIR}
    offline_teacher_reference: ${SET_MATCHING_ROOT}/offline_teacher_reference
    logs: ${PROJECT_DIR}/fresh_check_logs
  key_metrics:
    tagger_run_reports: ${SET_MATCHING_TAGGER_ROOT}/<variant>/run_report.json
    audit_summary: ${SET_MATCHING_ABLATION_DIR}/summary.csv
    final_report: ${SET_MATCHING_FINAL_REPORT_DIR}/final_report.json
    offline_teacher_reference: ${SET_MATCHING_ROOT}/offline_teacher_reference/<run>/run_report.json
SUMMARY
