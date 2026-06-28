#!/usr/bin/env bash
# Submit the QCD-vs-Hgg multi-scale subjet HLT ParT comparison.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${UPSTREAM_DEPENDENCY:=}"
: "${MULTISCALE_SUBJET_PART_QCD_HGG_TAG:=$(date +%Y%m%d_%H%M%S)}"
: "${MULTISCALE_SUBJET_PART_QCD_HGG_HLT_DEGRADATION_STRENGTH:=0.6}"
MULTISCALE_SUBJET_PART_QCD_HGG_HLT_TAG="${MULTISCALE_SUBJET_PART_QCD_HGG_HLT_DEGRADATION_STRENGTH//./p}"
: "${MULTISCALE_SUBJET_PART_QCD_HGG_ROOT:=${OUTPUT_ROOT}/multiscale_subjet_part_qcd_hgg_binary_hlt${MULTISCALE_SUBJET_PART_QCD_HGG_HLT_TAG}_${MULTISCALE_SUBJET_PART_QCD_HGG_TAG}}"
: "${MULTISCALE_SUBJET_PART_QCD_HGG_BUILD_BINARY_INPUTS:=1}"
: "${MULTISCALE_SUBJET_PART_QCD_HGG_BUILD_DIRECT_BINARY_SPLITS:=1}"
: "${MULTISCALE_SUBJET_PART_QCD_HGG_BINARY_INPUT_ROOT:=${MULTISCALE_SUBJET_PART_QCD_HGG_ROOT}/binary_inputs}"
: "${MULTISCALE_SUBJET_PART_QCD_HGG_SOURCE_MANIFEST_PATH:=${MANIFEST_PATH}}"
: "${MULTISCALE_SUBJET_PART_QCD_HGG_BINARY_MANIFEST_PATH:=${MULTISCALE_SUBJET_PART_QCD_HGG_BINARY_INPUT_ROOT}/split_manifest.json.gz}"
: "${MULTISCALE_SUBJET_PART_QCD_HGG_BINARY_HLT_CACHE_DIR:=${MULTISCALE_SUBJET_PART_QCD_HGG_BINARY_INPUT_ROOT}/hlt_cache}"
: "${MULTISCALE_SUBJET_PART_QCD_HGG_SOURCE_LABEL_NAMES:=QCD Hgg}"
: "${MULTISCALE_SUBJET_PART_QCD_HGG_BINARY_LABEL_FILTER:=0 1}"
: "${MULTISCALE_SUBJET_PART_QCD_HGG_BASE_PROFILES:=hlt_part_baseline multiscale_subjet_residual_part_adapter pure_perceiver_latent_control part_plus_random_subjet_control subjet_branch_only}"
: "${MULTISCALE_SUBJET_PART_QCD_HGG_STEP14_PROFILES:=no_scale_bias one_scale_medium no_seeded_queries no_subjet_transformer no_particle_readback late_fusion cls_fusion cross_attention_branch_fusion few_subjets many_subjets physics_bias_removed large_hlt_part_control two_hlt_part_ensemble_control}"
: "${MULTISCALE_SUBJET_PART_QCD_HGG_INCLUDE_STEP14_ABLATIONS:=0}"
if [[ -z "${MULTISCALE_SUBJET_PART_QCD_HGG_VARIANTS:-}" ]]; then
  if fresh_bool_enabled "${MULTISCALE_SUBJET_PART_QCD_HGG_INCLUDE_STEP14_ABLATIONS}"; then
    MULTISCALE_SUBJET_PART_QCD_HGG_VARIANTS="${MULTISCALE_SUBJET_PART_QCD_HGG_BASE_PROFILES} ${MULTISCALE_SUBJET_PART_QCD_HGG_STEP14_PROFILES}"
  else
    MULTISCALE_SUBJET_PART_QCD_HGG_VARIANTS="${MULTISCALE_SUBJET_PART_QCD_HGG_BASE_PROFILES}"
  fi
fi

: "${MULTISCALE_SUBJET_PART_QCD_HGG_MODEL_TRAIN_SIZE:=500000}"
: "${MULTISCALE_SUBJET_PART_QCD_HGG_MODEL_VAL_SIZE:=150000}"
: "${MULTISCALE_SUBJET_PART_QCD_HGG_STACK_TRAIN_SIZE:=500000}"
: "${MULTISCALE_SUBJET_PART_QCD_HGG_STACK_VAL_SIZE:=150000}"
: "${MULTISCALE_SUBJET_PART_QCD_HGG_FINAL_TEST_SIZE:=500000}"
: "${MULTISCALE_SUBJET_PART_QCD_HGG_EPOCHS:=45}"
: "${MULTISCALE_SUBJET_PART_QCD_HGG_SELECTION_METRIC:=fpr_at_signal_eff_0p50}"

: "${MULTISCALE_SUBJET_PART_QCD_HGG_BINARY_MANIFEST_MEM:=16G}"
: "${MULTISCALE_SUBJET_PART_QCD_HGG_BINARY_HLT_CACHE_MEM:=128G}"
: "${MULTISCALE_SUBJET_PART_QCD_HGG_TRAIN_MEM:=160G}"
: "${MULTISCALE_SUBJET_PART_QCD_HGG_REPORT_MEM:=8G}"
: "${MULTISCALE_SUBJET_PART_QCD_HGG_BINARY_MANIFEST_TIME:=04:00:00}"
: "${MULTISCALE_SUBJET_PART_QCD_HGG_BINARY_HLT_CACHE_TIME:=1-00:00:00}"
: "${MULTISCALE_SUBJET_PART_QCD_HGG_TRAIN_TIME:=2-12:00:00}"
: "${MULTISCALE_SUBJET_PART_QCD_HGG_REPORT_TIME:=02:00:00}"
: "${MULTISCALE_SUBJET_PART_QCD_HGG_BINARY_MANIFEST_CPUS:=2}"
: "${MULTISCALE_SUBJET_PART_QCD_HGG_BINARY_HLT_CACHE_CPUS:=4}"
: "${MULTISCALE_SUBJET_PART_QCD_HGG_TRAIN_CPUS:=8}"
: "${MULTISCALE_SUBJET_PART_QCD_HGG_REPORT_CPUS:=2}"

export MULTISCALE_SUBJET_PART_ROOT="${MULTISCALE_SUBJET_PART_QCD_HGG_ROOT}"
export MULTISCALE_SUBJET_PART_TAGGER_ROOT="${MULTISCALE_SUBJET_PART_ROOT}/taggers"
export MULTISCALE_SUBJET_PART_FINAL_REPORT_DIR="${MULTISCALE_SUBJET_PART_ROOT}/final_report"
export MULTISCALE_SUBJET_PART_HLT_CACHE_DIR="${MULTISCALE_SUBJET_PART_QCD_HGG_BINARY_HLT_CACHE_DIR}"
export MULTISCALE_SUBJET_PART_MODEL_TRAIN_SIZE="${MULTISCALE_SUBJET_PART_QCD_HGG_MODEL_TRAIN_SIZE}"
export MULTISCALE_SUBJET_PART_MODEL_VAL_SIZE="${MULTISCALE_SUBJET_PART_QCD_HGG_MODEL_VAL_SIZE}"
export MULTISCALE_SUBJET_PART_STACK_VAL_SIZE="${MULTISCALE_SUBJET_PART_QCD_HGG_STACK_VAL_SIZE}"
export MULTISCALE_SUBJET_PART_FINAL_TEST_SIZE="${MULTISCALE_SUBJET_PART_QCD_HGG_FINAL_TEST_SIZE}"
export MULTISCALE_SUBJET_PART_EPOCHS="${MULTISCALE_SUBJET_PART_QCD_HGG_EPOCHS}"
export MULTISCALE_SUBJET_PART_SELECTION_METRIC="${MULTISCALE_SUBJET_PART_QCD_HGG_SELECTION_METRIC}"
export MULTISCALE_SUBJET_PART_CONFIRM_FINAL_TEST=1
export MULTISCALE_SUBJET_PART_REPORT_VARIANTS="${MULTISCALE_SUBJET_PART_QCD_HGG_VARIANTS}"
export MULTISCALE_SUBJET_PART_REPORT_PRIMARY_METRIC="${MULTISCALE_SUBJET_PART_QCD_HGG_SELECTION_METRIC}"
export MULTISCALE_SUBJET_PART_REPORT_COMPARISON_SPLIT="final_test"
export MULTISCALE_SUBJET_PART_REPORT_CONFIRM_FINAL_TEST=1
export MULTISCALE_SUBJET_PART_REPORT_BASELINE_VARIANT="hlt_part_baseline"
export MULTISCALE_SUBJET_PART_REPORT_PRIMARY_VARIANT="multiscale_subjet_residual_part_adapter"
export MULTISCALE_SUBJET_PART_EXPECTED_HLT_DEGRADATION_STRENGTH="${MULTISCALE_SUBJET_PART_QCD_HGG_HLT_DEGRADATION_STRENGTH}"
export HLT_DEGRADATION_STRENGTH="${MULTISCALE_SUBJET_PART_QCD_HGG_HLT_DEGRADATION_STRENGTH}"

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

safe_label() {
  local value="$1"
  value="${value//multiscale_subjet_/ms_}"
  value="${value//part_plus_/}"
  value="${value//_control/}"
  value="${value//[^A-Za-z0-9_]/_}"
  printf '%s' "${value}"
}

fresh_split_words variant_args "${MULTISCALE_SUBJET_PART_QCD_HGG_VARIANTS}"

submitter_lock_dir="${MULTISCALE_SUBJET_PART_ROOT}/.submission_lock"
fresh_refuse_existing_dir "${MULTISCALE_SUBJET_PART_ROOT}"
fresh_claim_new_dir "${submitter_lock_dir}"
if ! fresh_is_dry_run; then
  {
    echo "created_at=$(date -Is)"
    echo "project_dir=${PROJECT_DIR}"
    echo "source_commit=$(fresh_source_commit)"
    echo "task=QCD_vs_Hgg_multiscale_subjet_part"
    echo "root=${MULTISCALE_SUBJET_PART_ROOT}"
    echo "tagger_root=${MULTISCALE_SUBJET_PART_TAGGER_ROOT}"
    echo "final_report_dir=${MULTISCALE_SUBJET_PART_FINAL_REPORT_DIR}"
    echo "source_label_names=${MULTISCALE_SUBJET_PART_QCD_HGG_SOURCE_LABEL_NAMES}"
    echo "downstream_label_filter=${MULTISCALE_SUBJET_PART_QCD_HGG_BINARY_LABEL_FILTER}"
    echo "binary_manifest=${MULTISCALE_SUBJET_PART_QCD_HGG_BINARY_MANIFEST_PATH}"
    echo "binary_hlt_cache=${MULTISCALE_SUBJET_PART_QCD_HGG_BINARY_HLT_CACHE_DIR}"
    echo "hlt_degradation_strength=${MULTISCALE_SUBJET_PART_QCD_HGG_HLT_DEGRADATION_STRENGTH}"
    echo "profiles=$(fresh_join_by_space "${variant_args[@]}")"
    echo "selection_metric=${MULTISCALE_SUBJET_PART_SELECTION_METRIC}"
    echo "epochs=${MULTISCALE_SUBJET_PART_EPOCHS}"
  } > "${submitter_lock_dir}/metadata.txt"
fi

binary_manifest_jid=""
binary_hlt_cache_jid=""
final_report_jid=""
train_job_ids=()
input_dependency="${UPSTREAM_DEPENDENCY}"

if fresh_bool_enabled "${MULTISCALE_SUBJET_PART_QCD_HGG_BUILD_BINARY_INPUTS}"; then
  export LABEL_FILTER_OUTPUT_MANIFEST_PATH="${MULTISCALE_SUBJET_PART_QCD_HGG_BINARY_MANIFEST_PATH}"
  export LABEL_FILTER_NAMES="${MULTISCALE_SUBJET_PART_QCD_HGG_SOURCE_LABEL_NAMES}"
  export LABEL_FILTER_MANIFEST_PATH="${MULTISCALE_SUBJET_PART_QCD_HGG_BINARY_MANIFEST_PATH}"
  export LABEL_FILTER_HLT_CACHE_DIR="${MULTISCALE_SUBJET_PART_QCD_HGG_BINARY_HLT_CACHE_DIR}"
  export LABEL_FILTER_HLT_SPLITS="model_train model_val stack_train stack_val final_test"
  export LABEL_FILTER_MODEL_TRAIN_SIZE="${MULTISCALE_SUBJET_PART_QCD_HGG_MODEL_TRAIN_SIZE}"
  export LABEL_FILTER_MODEL_VAL_SIZE="${MULTISCALE_SUBJET_PART_QCD_HGG_MODEL_VAL_SIZE}"
  export LABEL_FILTER_STACK_TRAIN_SIZE="${MULTISCALE_SUBJET_PART_QCD_HGG_STACK_TRAIN_SIZE}"
  export LABEL_FILTER_STACK_VAL_SIZE="${MULTISCALE_SUBJET_PART_QCD_HGG_STACK_VAL_SIZE}"
  export LABEL_FILTER_FINAL_TEST_SIZE="${MULTISCALE_SUBJET_PART_QCD_HGG_FINAL_TEST_SIZE}"

  if fresh_bool_enabled "${MULTISCALE_SUBJET_PART_QCD_HGG_BUILD_DIRECT_BINARY_SPLITS}"; then
    mapfile -t manifest_args < <(
      afterok_args \
        "${UPSTREAM_DEPENDENCY}" \
        --time="${MULTISCALE_SUBJET_PART_QCD_HGG_BINARY_MANIFEST_TIME}" \
        --cpus-per-task="${MULTISCALE_SUBJET_PART_QCD_HGG_BINARY_MANIFEST_CPUS}" \
        --mem="${MULTISCALE_SUBJET_PART_QCD_HGG_BINARY_MANIFEST_MEM}" \
        "${SCRIPT_DIR}/run_build_label_filtered_fresh_splits.sh"
    )
  else
    export LABEL_FILTER_SOURCE_MANIFEST_PATH="${MULTISCALE_SUBJET_PART_QCD_HGG_SOURCE_MANIFEST_PATH}"
    export LABEL_FILTER_REMAP_LABELS=1
    mapfile -t manifest_args < <(
      afterok_args \
        "${UPSTREAM_DEPENDENCY}" \
        --time="${MULTISCALE_SUBJET_PART_QCD_HGG_BINARY_MANIFEST_TIME}" \
        --cpus-per-task="${MULTISCALE_SUBJET_PART_QCD_HGG_BINARY_MANIFEST_CPUS}" \
        --mem="${MULTISCALE_SUBJET_PART_QCD_HGG_BINARY_MANIFEST_MEM}" \
        "${SCRIPT_DIR}/run_build_label_filtered_split_manifest.sh"
    )
  fi
  binary_manifest_jid="$(submit_job "multiscale_binary_manifest" "${manifest_args[@]}")"
  echo "submitted multiscale_binary_manifest=${binary_manifest_jid}"

  binary_hlt_cache_jid="$(submit_job "multiscale_binary_hlt_cache" \
    --time="${MULTISCALE_SUBJET_PART_QCD_HGG_BINARY_HLT_CACHE_TIME}" \
    --cpus-per-task="${MULTISCALE_SUBJET_PART_QCD_HGG_BINARY_HLT_CACHE_CPUS}" \
    --mem="${MULTISCALE_SUBJET_PART_QCD_HGG_BINARY_HLT_CACHE_MEM}" \
    --dependency="afterok:${binary_manifest_jid}" \
    "${SCRIPT_DIR}/run_build_label_filtered_hlt_cache.sh")"
  echo "submitted multiscale_binary_hlt_cache=${binary_hlt_cache_jid}"
  input_dependency="${binary_hlt_cache_jid}"
fi

for variant in "${variant_args[@]}"; do
  label="$(safe_label "${variant}")"
  mapfile -t train_args < <(
    afterok_args \
      "${input_dependency}" \
      --time="${MULTISCALE_SUBJET_PART_QCD_HGG_TRAIN_TIME}" \
      --cpus-per-task="${MULTISCALE_SUBJET_PART_QCD_HGG_TRAIN_CPUS}" \
      --mem="${MULTISCALE_SUBJET_PART_QCD_HGG_TRAIN_MEM}" \
      "${SCRIPT_DIR}/run_train_multiscale_subjet_part_tagger.sh" \
      "${variant}"
  )
  train_jid="$(submit_job "multiscale_part_${label}" "${train_args[@]}")"
  train_job_ids+=("${train_jid}")
  echo "submitted multiscale_part_${label}=${train_jid}"
done

train_dep="$(fresh_join_by_colon "${train_job_ids[@]}")"
final_report_jid="$(submit_job "multiscale_part_report" \
  --time="${MULTISCALE_SUBJET_PART_QCD_HGG_REPORT_TIME}" \
  --cpus-per-task="${MULTISCALE_SUBJET_PART_QCD_HGG_REPORT_CPUS}" \
  --mem="${MULTISCALE_SUBJET_PART_QCD_HGG_REPORT_MEM}" \
  --dependency="afterok:${train_dep}" \
  "${SCRIPT_DIR}/run_write_multiscale_subjet_part_report.sh")"
echo "submitted multiscale_part_report=${final_report_jid}"

cat <<SUMMARY
multiscale_subjet_part_qcd_hgg_binary_submission:
  task: QCD_vs_Hgg_multiscale_subjet_part
  source_label_names: ${MULTISCALE_SUBJET_PART_QCD_HGG_SOURCE_LABEL_NAMES}
  downstream_label_filter: ${MULTISCALE_SUBJET_PART_QCD_HGG_BINARY_LABEL_FILTER}
  root: ${MULTISCALE_SUBJET_PART_ROOT}
  hlt_degradation_strength: ${MULTISCALE_SUBJET_PART_QCD_HGG_HLT_DEGRADATION_STRENGTH}
  binary_inputs:
    build_enabled: ${MULTISCALE_SUBJET_PART_QCD_HGG_BUILD_BINARY_INPUTS}
    direct_binary_splits: ${MULTISCALE_SUBJET_PART_QCD_HGG_BUILD_DIRECT_BINARY_SPLITS}
    source_manifest: ${MULTISCALE_SUBJET_PART_QCD_HGG_SOURCE_MANIFEST_PATH}
    filtered_manifest: ${MULTISCALE_SUBJET_PART_QCD_HGG_BINARY_MANIFEST_PATH}
    filtered_hlt_cache: ${MULTISCALE_SUBJET_PART_QCD_HGG_BINARY_HLT_CACHE_DIR}
    manifest_job_id: ${binary_manifest_jid:-none}
    hlt_cache_job_id: ${binary_hlt_cache_jid:-none}
  train_job_ids: $(fresh_join_by_space "${train_job_ids[@]}")
  final_report_job_id: ${final_report_jid}
  expected_jobs:
    binary_manifest: $([[ -n "${binary_manifest_jid}" ]] && echo 1 || echo 0)
    binary_hlt_cache: $([[ -n "${binary_hlt_cache_jid}" ]] && echo 1 || echo 0)
    multiscale_subjet_train: ${#train_job_ids[@]}
    multiscale_subjet_report: 1
    total_submitted: ${submit_count}
  requested_split_caps:
    model_train: ${MULTISCALE_SUBJET_PART_MODEL_TRAIN_SIZE}
    model_val: ${MULTISCALE_SUBJET_PART_MODEL_VAL_SIZE}
    stack_train: ${MULTISCALE_SUBJET_PART_QCD_HGG_STACK_TRAIN_SIZE}
    stack_val: ${MULTISCALE_SUBJET_PART_STACK_VAL_SIZE}
    final_test: ${MULTISCALE_SUBJET_PART_FINAL_TEST_SIZE}
  model:
    profiles: $(fresh_join_by_space "${variant_args[@]}")
    base_profiles: ${MULTISCALE_SUBJET_PART_QCD_HGG_BASE_PROFILES}
    step14_profiles: ${MULTISCALE_SUBJET_PART_QCD_HGG_STEP14_PROFILES}
    include_step14_ablations: ${MULTISCALE_SUBJET_PART_QCD_HGG_INCLUDE_STEP14_ABLATIONS}
    epochs: ${MULTISCALE_SUBJET_PART_EPOCHS}
    selection_metric: ${MULTISCALE_SUBJET_PART_SELECTION_METRIC}
  resources:
    binary_manifest: time=${MULTISCALE_SUBJET_PART_QCD_HGG_BINARY_MANIFEST_TIME} mem=${MULTISCALE_SUBJET_PART_QCD_HGG_BINARY_MANIFEST_MEM} cpus=${MULTISCALE_SUBJET_PART_QCD_HGG_BINARY_MANIFEST_CPUS}
    binary_hlt_cache: time=${MULTISCALE_SUBJET_PART_QCD_HGG_BINARY_HLT_CACHE_TIME} mem=${MULTISCALE_SUBJET_PART_QCD_HGG_BINARY_HLT_CACHE_MEM} cpus=${MULTISCALE_SUBJET_PART_QCD_HGG_BINARY_HLT_CACHE_CPUS}
    train: time=${MULTISCALE_SUBJET_PART_QCD_HGG_TRAIN_TIME} mem=${MULTISCALE_SUBJET_PART_QCD_HGG_TRAIN_MEM} cpus=${MULTISCALE_SUBJET_PART_QCD_HGG_TRAIN_CPUS}
    report: time=${MULTISCALE_SUBJET_PART_QCD_HGG_REPORT_TIME} mem=${MULTISCALE_SUBJET_PART_QCD_HGG_REPORT_MEM} cpus=${MULTISCALE_SUBJET_PART_QCD_HGG_REPORT_CPUS}
  outputs:
    tagger_root: ${MULTISCALE_SUBJET_PART_TAGGER_ROOT}
    final_report_json: ${MULTISCALE_SUBJET_PART_FINAL_REPORT_DIR}/multiscale_subjet_part_report.json
    final_report_md: ${MULTISCALE_SUBJET_PART_FINAL_REPORT_DIR}/multiscale_subjet_part_report.md
    final_metric_table: ${MULTISCALE_SUBJET_PART_FINAL_REPORT_DIR}/metric_table.csv
    diagnostics: ${MULTISCALE_SUBJET_PART_FINAL_REPORT_DIR}/diagnostics.csv
    hlt_degradation: ${MULTISCALE_SUBJET_PART_FINAL_REPORT_DIR}/hlt_degradation.csv
    logs: ${PROJECT_DIR}/fresh_check_logs
SUMMARY
