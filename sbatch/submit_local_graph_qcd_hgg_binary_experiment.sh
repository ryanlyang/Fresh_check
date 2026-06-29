#!/usr/bin/env bash
# Submit the QCD-vs-Hgg local-graph HLT ParT comparison.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${UPSTREAM_DEPENDENCY:=}"
: "${LOCAL_GRAPH_PART_QCD_HGG_TAG:=$(date +%Y%m%d_%H%M%S)}"
: "${LOCAL_GRAPH_PART_QCD_HGG_HLT_DEGRADATION_STRENGTH:=0.6}"
LOCAL_GRAPH_PART_QCD_HGG_HLT_TAG="${LOCAL_GRAPH_PART_QCD_HGG_HLT_DEGRADATION_STRENGTH//./p}"
: "${LOCAL_GRAPH_PART_QCD_HGG_ROOT:=${OUTPUT_ROOT}/local_graph_part_qcd_hgg_binary_hlt${LOCAL_GRAPH_PART_QCD_HGG_HLT_TAG}_${LOCAL_GRAPH_PART_QCD_HGG_TAG}}"
: "${LOCAL_GRAPH_PART_QCD_HGG_BUILD_BINARY_INPUTS:=1}"
: "${LOCAL_GRAPH_PART_QCD_HGG_BUILD_DIRECT_BINARY_SPLITS:=1}"
: "${LOCAL_GRAPH_PART_QCD_HGG_BINARY_INPUT_ROOT:=${LOCAL_GRAPH_PART_QCD_HGG_ROOT}/binary_inputs}"
: "${LOCAL_GRAPH_PART_QCD_HGG_SOURCE_MANIFEST_PATH:=${MANIFEST_PATH}}"
: "${LOCAL_GRAPH_PART_QCD_HGG_BINARY_MANIFEST_PATH:=${LOCAL_GRAPH_PART_QCD_HGG_BINARY_INPUT_ROOT}/split_manifest.json.gz}"
: "${LOCAL_GRAPH_PART_QCD_HGG_BINARY_HLT_CACHE_DIR:=${LOCAL_GRAPH_PART_QCD_HGG_BINARY_INPUT_ROOT}/hlt_cache}"
: "${LOCAL_GRAPH_PART_QCD_HGG_SOURCE_LABEL_NAMES:=QCD Hgg}"
: "${LOCAL_GRAPH_PART_QCD_HGG_BINARY_LABEL_FILTER:=0 1}"
: "${LOCAL_GRAPH_PART_QCD_HGG_VARIANTS:=hlt_part_baseline local_edgeconv_adapter local_point_attention_adapter local_point_attention_adapter_warmstart}"

: "${LOCAL_GRAPH_PART_QCD_HGG_MODEL_TRAIN_SIZE:=500000}"
: "${LOCAL_GRAPH_PART_QCD_HGG_MODEL_VAL_SIZE:=150000}"
: "${LOCAL_GRAPH_PART_QCD_HGG_STACK_TRAIN_SIZE:=500000}"
: "${LOCAL_GRAPH_PART_QCD_HGG_STACK_VAL_SIZE:=150000}"
: "${LOCAL_GRAPH_PART_QCD_HGG_FINAL_TEST_SIZE:=500000}"
: "${LOCAL_GRAPH_PART_QCD_HGG_EPOCHS:=45}"
: "${LOCAL_GRAPH_PART_QCD_HGG_SELECTION_METRIC:=fpr_at_signal_eff_0p50}"
: "${LOCAL_GRAPH_PART_QCD_HGG_K:=16}"
: "${LOCAL_GRAPH_PART_QCD_HGG_WARM_START_FREEZE_EPOCHS:=0}"
: "${LOCAL_GRAPH_PART_QCD_HGG_WARM_START_RESIDUAL_GAMMA_INIT:=0.01}"

: "${LOCAL_GRAPH_PART_QCD_HGG_BINARY_MANIFEST_MEM:=16G}"
: "${LOCAL_GRAPH_PART_QCD_HGG_BINARY_HLT_CACHE_MEM:=128G}"
: "${LOCAL_GRAPH_PART_QCD_HGG_TRAIN_MEM:=160G}"
: "${LOCAL_GRAPH_PART_QCD_HGG_REPORT_MEM:=8G}"
: "${LOCAL_GRAPH_PART_QCD_HGG_BINARY_MANIFEST_TIME:=04:00:00}"
: "${LOCAL_GRAPH_PART_QCD_HGG_BINARY_HLT_CACHE_TIME:=1-00:00:00}"
: "${LOCAL_GRAPH_PART_QCD_HGG_TRAIN_TIME:=2-12:00:00}"
: "${LOCAL_GRAPH_PART_QCD_HGG_REPORT_TIME:=02:00:00}"
: "${LOCAL_GRAPH_PART_QCD_HGG_SUBMIT_SCORE_FUSION:=0}"
: "${LOCAL_GRAPH_PART_QCD_HGG_SCORE_FUSION_DIR:=${LOCAL_GRAPH_PART_QCD_HGG_ROOT}/score_fusion}"
: "${LOCAL_GRAPH_PART_QCD_HGG_SCORE_FUSION_MEM:=96G}"
: "${LOCAL_GRAPH_PART_QCD_HGG_SCORE_FUSION_TIME:=08:00:00}"
: "${LOCAL_GRAPH_PART_QCD_HGG_SCORE_FUSION_CPUS:=4}"
: "${LOCAL_GRAPH_PART_QCD_HGG_SCORE_FUSION_REQUIRE_ALL_VARIANTS:=1}"
: "${LOCAL_GRAPH_PART_QCD_HGG_BINARY_MANIFEST_CPUS:=2}"
: "${LOCAL_GRAPH_PART_QCD_HGG_BINARY_HLT_CACHE_CPUS:=4}"
: "${LOCAL_GRAPH_PART_QCD_HGG_TRAIN_CPUS:=8}"
: "${LOCAL_GRAPH_PART_QCD_HGG_REPORT_CPUS:=2}"

export LOCAL_GRAPH_PART_ROOT="${LOCAL_GRAPH_PART_QCD_HGG_ROOT}"
export LOCAL_GRAPH_PART_TAGGER_ROOT="${LOCAL_GRAPH_PART_ROOT}/taggers"
export LOCAL_GRAPH_PART_FINAL_REPORT_DIR="${LOCAL_GRAPH_PART_ROOT}/final_report"
export LOCAL_GRAPH_PART_HLT_CACHE_DIR="${LOCAL_GRAPH_PART_QCD_HGG_BINARY_HLT_CACHE_DIR}"
export LOCAL_GRAPH_PART_MODEL_TRAIN_SIZE="${LOCAL_GRAPH_PART_QCD_HGG_MODEL_TRAIN_SIZE}"
export LOCAL_GRAPH_PART_MODEL_VAL_SIZE="${LOCAL_GRAPH_PART_QCD_HGG_MODEL_VAL_SIZE}"
export LOCAL_GRAPH_PART_STACK_VAL_SIZE="${LOCAL_GRAPH_PART_QCD_HGG_STACK_VAL_SIZE}"
export LOCAL_GRAPH_PART_FINAL_TEST_SIZE="${LOCAL_GRAPH_PART_QCD_HGG_FINAL_TEST_SIZE}"
export LOCAL_GRAPH_PART_EPOCHS="${LOCAL_GRAPH_PART_QCD_HGG_EPOCHS}"
export LOCAL_GRAPH_PART_SELECTION_METRIC="${LOCAL_GRAPH_PART_QCD_HGG_SELECTION_METRIC}"
export LOCAL_GRAPH_PART_K="${LOCAL_GRAPH_PART_QCD_HGG_K}"
export LOCAL_GRAPH_PART_CONFIRM_FINAL_TEST=1
export LOCAL_GRAPH_PART_REPORT_VARIANTS="${LOCAL_GRAPH_PART_QCD_HGG_VARIANTS}"
export LOCAL_GRAPH_PART_REPORT_PRIMARY_METRIC="${LOCAL_GRAPH_PART_QCD_HGG_SELECTION_METRIC}"
export LOCAL_GRAPH_PART_REPORT_COMPARISON_SPLIT="final_test"
export LOCAL_GRAPH_PART_REPORT_CONFIRM_FINAL_TEST=1
export LOCAL_GRAPH_PART_WARM_START_CHECKPOINT="${LOCAL_GRAPH_PART_TAGGER_ROOT}/hlt_part_baseline/best_model_val.pt"
export LOCAL_GRAPH_PART_WARM_START_FREEZE_EPOCHS="${LOCAL_GRAPH_PART_QCD_HGG_WARM_START_FREEZE_EPOCHS}"
export LOCAL_GRAPH_PART_WARM_START_RESIDUAL_GAMMA_INIT="${LOCAL_GRAPH_PART_QCD_HGG_WARM_START_RESIDUAL_GAMMA_INIT}"
export LOCAL_GRAPH_PART_EXPECTED_HLT_DEGRADATION_STRENGTH="${LOCAL_GRAPH_PART_QCD_HGG_HLT_DEGRADATION_STRENGTH}"
export HLT_DEGRADATION_STRENGTH="${LOCAL_GRAPH_PART_QCD_HGG_HLT_DEGRADATION_STRENGTH}"

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

variant_has() {
  local needle="$1"
  shift
  local item
  for item in "$@"; do
    if [[ "${item}" == "${needle}" ]]; then
      return 0
    fi
  done
  return 1
}

safe_label() {
  local value="$1"
  value="${value//local_/}"
  value="${value//_adapter/}"
  value="${value//[^A-Za-z0-9_]/_}"
  printf '%s' "${value}"
}

fresh_split_words variant_args "${LOCAL_GRAPH_PART_QCD_HGG_VARIANTS}"

submitter_lock_dir="${LOCAL_GRAPH_PART_ROOT}/.submission_lock"
fresh_refuse_existing_dir "${LOCAL_GRAPH_PART_ROOT}"
fresh_claim_new_dir "${submitter_lock_dir}"
if ! fresh_is_dry_run; then
  {
    echo "created_at=$(date -Is)"
    echo "project_dir=${PROJECT_DIR}"
    echo "source_commit=$(fresh_source_commit)"
    echo "task=QCD_vs_Hgg_local_graph_part"
    echo "root=${LOCAL_GRAPH_PART_ROOT}"
    echo "tagger_root=${LOCAL_GRAPH_PART_TAGGER_ROOT}"
    echo "final_report_dir=${LOCAL_GRAPH_PART_FINAL_REPORT_DIR}"
    echo "source_label_names=${LOCAL_GRAPH_PART_QCD_HGG_SOURCE_LABEL_NAMES}"
    echo "downstream_label_filter=${LOCAL_GRAPH_PART_QCD_HGG_BINARY_LABEL_FILTER}"
    echo "binary_manifest=${LOCAL_GRAPH_PART_QCD_HGG_BINARY_MANIFEST_PATH}"
    echo "binary_hlt_cache=${LOCAL_GRAPH_PART_QCD_HGG_BINARY_HLT_CACHE_DIR}"
    echo "hlt_degradation_strength=${LOCAL_GRAPH_PART_QCD_HGG_HLT_DEGRADATION_STRENGTH}"
    echo "variants=$(fresh_join_by_space "${variant_args[@]}")"
    echo "selection_metric=${LOCAL_GRAPH_PART_SELECTION_METRIC}"
    echo "epochs=${LOCAL_GRAPH_PART_EPOCHS}"
    echo "warm_start_checkpoint=${LOCAL_GRAPH_PART_WARM_START_CHECKPOINT}"
    echo "warm_start_freeze_epochs=${LOCAL_GRAPH_PART_WARM_START_FREEZE_EPOCHS}"
    echo "warm_start_residual_gamma_init=${LOCAL_GRAPH_PART_WARM_START_RESIDUAL_GAMMA_INIT}"
  } > "${submitter_lock_dir}/metadata.txt"
fi

binary_manifest_jid=""
binary_hlt_cache_jid=""
baseline_jid=""
final_report_jid=""
score_fusion_jid=""
train_job_ids=()
input_dependency="${UPSTREAM_DEPENDENCY}"

if fresh_bool_enabled "${LOCAL_GRAPH_PART_QCD_HGG_BUILD_BINARY_INPUTS}"; then
  export LABEL_FILTER_OUTPUT_MANIFEST_PATH="${LOCAL_GRAPH_PART_QCD_HGG_BINARY_MANIFEST_PATH}"
  export LABEL_FILTER_NAMES="${LOCAL_GRAPH_PART_QCD_HGG_SOURCE_LABEL_NAMES}"
  export LABEL_FILTER_MANIFEST_PATH="${LOCAL_GRAPH_PART_QCD_HGG_BINARY_MANIFEST_PATH}"
  export LABEL_FILTER_HLT_CACHE_DIR="${LOCAL_GRAPH_PART_QCD_HGG_BINARY_HLT_CACHE_DIR}"
  export LABEL_FILTER_HLT_SPLITS="model_train model_val stack_train stack_val final_test"
  export LABEL_FILTER_MODEL_TRAIN_SIZE="${LOCAL_GRAPH_PART_QCD_HGG_MODEL_TRAIN_SIZE}"
  export LABEL_FILTER_MODEL_VAL_SIZE="${LOCAL_GRAPH_PART_QCD_HGG_MODEL_VAL_SIZE}"
  export LABEL_FILTER_STACK_TRAIN_SIZE="${LOCAL_GRAPH_PART_QCD_HGG_STACK_TRAIN_SIZE}"
  export LABEL_FILTER_STACK_VAL_SIZE="${LOCAL_GRAPH_PART_QCD_HGG_STACK_VAL_SIZE}"
  export LABEL_FILTER_FINAL_TEST_SIZE="${LOCAL_GRAPH_PART_QCD_HGG_FINAL_TEST_SIZE}"

  if fresh_bool_enabled "${LOCAL_GRAPH_PART_QCD_HGG_BUILD_DIRECT_BINARY_SPLITS}"; then
    mapfile -t manifest_args < <(
      afterok_args \
        "${UPSTREAM_DEPENDENCY}" \
        --time="${LOCAL_GRAPH_PART_QCD_HGG_BINARY_MANIFEST_TIME}" \
        --cpus-per-task="${LOCAL_GRAPH_PART_QCD_HGG_BINARY_MANIFEST_CPUS}" \
        --mem="${LOCAL_GRAPH_PART_QCD_HGG_BINARY_MANIFEST_MEM}" \
        "${SCRIPT_DIR}/run_build_label_filtered_fresh_splits.sh"
    )
  else
    export LABEL_FILTER_SOURCE_MANIFEST_PATH="${LOCAL_GRAPH_PART_QCD_HGG_SOURCE_MANIFEST_PATH}"
    export LABEL_FILTER_REMAP_LABELS=1
    mapfile -t manifest_args < <(
      afterok_args \
        "${UPSTREAM_DEPENDENCY}" \
        --time="${LOCAL_GRAPH_PART_QCD_HGG_BINARY_MANIFEST_TIME}" \
        --cpus-per-task="${LOCAL_GRAPH_PART_QCD_HGG_BINARY_MANIFEST_CPUS}" \
        --mem="${LOCAL_GRAPH_PART_QCD_HGG_BINARY_MANIFEST_MEM}" \
        "${SCRIPT_DIR}/run_build_label_filtered_split_manifest.sh"
    )
  fi
  binary_manifest_jid="$(submit_job "localgraph_binary_manifest" "${manifest_args[@]}")"
  echo "submitted localgraph_binary_manifest=${binary_manifest_jid}"

  binary_hlt_cache_jid="$(submit_job "localgraph_binary_hlt_cache" \
    --time="${LOCAL_GRAPH_PART_QCD_HGG_BINARY_HLT_CACHE_TIME}" \
    --cpus-per-task="${LOCAL_GRAPH_PART_QCD_HGG_BINARY_HLT_CACHE_CPUS}" \
    --mem="${LOCAL_GRAPH_PART_QCD_HGG_BINARY_HLT_CACHE_MEM}" \
    --dependency="afterok:${binary_manifest_jid}" \
    "${SCRIPT_DIR}/run_build_label_filtered_hlt_cache.sh")"
  echo "submitted localgraph_binary_hlt_cache=${binary_hlt_cache_jid}"
  input_dependency="${binary_hlt_cache_jid}"
fi

if variant_has "local_point_attention_adapter_warmstart" "${variant_args[@]}" \
  && ! variant_has "hlt_part_baseline" "${variant_args[@]}"; then
  echo "local_point_attention_adapter_warmstart requires hlt_part_baseline in LOCAL_GRAPH_PART_QCD_HGG_VARIANTS" >&2
  exit 2
fi

if variant_has "hlt_part_baseline" "${variant_args[@]}"; then
  mapfile -t baseline_args < <(
    afterok_args \
      "${input_dependency}" \
      --time="${LOCAL_GRAPH_PART_QCD_HGG_TRAIN_TIME}" \
      --cpus-per-task="${LOCAL_GRAPH_PART_QCD_HGG_TRAIN_CPUS}" \
      --mem="${LOCAL_GRAPH_PART_QCD_HGG_TRAIN_MEM}" \
      "${SCRIPT_DIR}/run_train_local_graph_part_tagger.sh" \
      "hlt_part_baseline"
  )
  baseline_jid="$(submit_job "localgraph_part_baseline" "${baseline_args[@]}")"
  train_job_ids+=("${baseline_jid}")
  echo "submitted localgraph_part_baseline=${baseline_jid}"
fi

for variant in "${variant_args[@]}"; do
  case "${variant}" in
    hlt_part_baseline|local_point_attention_adapter_warmstart)
      continue
      ;;
  esac
  label="$(safe_label "${variant}")"
  mapfile -t train_args < <(
    afterok_args \
      "${input_dependency}" \
      --time="${LOCAL_GRAPH_PART_QCD_HGG_TRAIN_TIME}" \
      --cpus-per-task="${LOCAL_GRAPH_PART_QCD_HGG_TRAIN_CPUS}" \
      --mem="${LOCAL_GRAPH_PART_QCD_HGG_TRAIN_MEM}" \
      "${SCRIPT_DIR}/run_train_local_graph_part_tagger.sh" \
      "${variant}"
  )
  train_jid="$(submit_job "localgraph_part_${label}" "${train_args[@]}")"
  train_job_ids+=("${train_jid}")
  echo "submitted localgraph_part_${label}=${train_jid}"
done

if variant_has "local_point_attention_adapter_warmstart" "${variant_args[@]}"; then
  warmstart_dependency="${baseline_jid}"
  mapfile -t warmstart_args < <(
    afterok_args \
      "${warmstart_dependency}" \
      --time="${LOCAL_GRAPH_PART_QCD_HGG_TRAIN_TIME}" \
      --cpus-per-task="${LOCAL_GRAPH_PART_QCD_HGG_TRAIN_CPUS}" \
      --mem="${LOCAL_GRAPH_PART_QCD_HGG_TRAIN_MEM}" \
      "${SCRIPT_DIR}/run_train_local_graph_part_tagger.sh" \
      "local_point_attention_adapter_warmstart"
  )
  warmstart_jid="$(submit_job "localgraph_part_warmstart" "${warmstart_args[@]}")"
  train_job_ids+=("${warmstart_jid}")
  echo "submitted localgraph_part_warmstart=${warmstart_jid}"
fi

train_dep="$(fresh_join_by_colon "${train_job_ids[@]}")"
final_report_jid="$(submit_job "localgraph_part_report" \
  --time="${LOCAL_GRAPH_PART_QCD_HGG_REPORT_TIME}" \
  --cpus-per-task="${LOCAL_GRAPH_PART_QCD_HGG_REPORT_CPUS}" \
  --mem="${LOCAL_GRAPH_PART_QCD_HGG_REPORT_MEM}" \
  --dependency="afterok:${train_dep}" \
  "${SCRIPT_DIR}/run_write_local_graph_part_report.sh")"
echo "submitted localgraph_part_report=${final_report_jid}"

if fresh_bool_enabled "${LOCAL_GRAPH_PART_QCD_HGG_SUBMIT_SCORE_FUSION}"; then
  export LOCAL_GRAPH_SCORE_FUSION_TAGGER_ROOT="${LOCAL_GRAPH_PART_TAGGER_ROOT}"
  export LOCAL_GRAPH_SCORE_FUSION_HLT_CACHE_DIR="${LOCAL_GRAPH_PART_HLT_CACHE_DIR}"
  export LOCAL_GRAPH_SCORE_FUSION_OUTPUT_DIR="${LOCAL_GRAPH_PART_QCD_HGG_SCORE_FUSION_DIR}"
  export LOCAL_GRAPH_SCORE_FUSION_PREDICTION_DIR="${LOCAL_GRAPH_PART_QCD_HGG_SCORE_FUSION_DIR}/predictions"
  export LOCAL_GRAPH_SCORE_FUSION_VARIANTS="${LOCAL_GRAPH_PART_QCD_HGG_VARIANTS}"
  export LOCAL_GRAPH_SCORE_FUSION_PRIMARY_METRIC="${LOCAL_GRAPH_PART_QCD_HGG_SELECTION_METRIC}"
  export LOCAL_GRAPH_SCORE_FUSION_MAX_STACK_JETS="${LOCAL_GRAPH_PART_QCD_HGG_STACK_VAL_SIZE}"
  export LOCAL_GRAPH_SCORE_FUSION_MAX_FINAL_TEST_JETS="${LOCAL_GRAPH_PART_QCD_HGG_FINAL_TEST_SIZE}"
  export LOCAL_GRAPH_SCORE_FUSION_REQUIRE_ALL_VARIANTS="${LOCAL_GRAPH_PART_QCD_HGG_SCORE_FUSION_REQUIRE_ALL_VARIANTS}"
  export LOCAL_GRAPH_SCORE_FUSION_CONFIRM_FINAL_TEST=1
  score_fusion_jid="$(submit_job "localgraph_score_fusion" \
    --time="${LOCAL_GRAPH_PART_QCD_HGG_SCORE_FUSION_TIME}" \
    --cpus-per-task="${LOCAL_GRAPH_PART_QCD_HGG_SCORE_FUSION_CPUS}" \
    --mem="${LOCAL_GRAPH_PART_QCD_HGG_SCORE_FUSION_MEM}" \
    --dependency="afterok:${train_dep}" \
    "${SCRIPT_DIR}/run_local_graph_score_fusion.sh")"
  echo "submitted localgraph_score_fusion=${score_fusion_jid}"
fi

cat <<SUMMARY
local_graph_part_qcd_hgg_binary_submission:
  task: QCD_vs_Hgg_local_graph_part
  source_label_names: ${LOCAL_GRAPH_PART_QCD_HGG_SOURCE_LABEL_NAMES}
  downstream_label_filter: ${LOCAL_GRAPH_PART_QCD_HGG_BINARY_LABEL_FILTER}
  root: ${LOCAL_GRAPH_PART_ROOT}
  hlt_degradation_strength: ${LOCAL_GRAPH_PART_QCD_HGG_HLT_DEGRADATION_STRENGTH}
  binary_inputs:
    build_enabled: ${LOCAL_GRAPH_PART_QCD_HGG_BUILD_BINARY_INPUTS}
    direct_binary_splits: ${LOCAL_GRAPH_PART_QCD_HGG_BUILD_DIRECT_BINARY_SPLITS}
    source_manifest: ${LOCAL_GRAPH_PART_QCD_HGG_SOURCE_MANIFEST_PATH}
    filtered_manifest: ${LOCAL_GRAPH_PART_QCD_HGG_BINARY_MANIFEST_PATH}
    filtered_hlt_cache: ${LOCAL_GRAPH_PART_QCD_HGG_BINARY_HLT_CACHE_DIR}
    manifest_job_id: ${binary_manifest_jid:-none}
    hlt_cache_job_id: ${binary_hlt_cache_jid:-none}
  train_job_ids: $(fresh_join_by_space "${train_job_ids[@]}")
  final_report_job_id: ${final_report_jid}
  expected_jobs:
    binary_manifest: $([[ -n "${binary_manifest_jid}" ]] && echo 1 || echo 0)
    binary_hlt_cache: $([[ -n "${binary_hlt_cache_jid}" ]] && echo 1 || echo 0)
    local_graph_train: ${#train_job_ids[@]}
    local_graph_report: 1
    local_graph_score_fusion: $([[ -n "${score_fusion_jid}" ]] && echo 1 || echo 0)
    total_submitted: ${submit_count}
  requested_split_caps:
    model_train: ${LOCAL_GRAPH_PART_MODEL_TRAIN_SIZE}
    model_val: ${LOCAL_GRAPH_PART_MODEL_VAL_SIZE}
    stack_train: ${LOCAL_GRAPH_PART_QCD_HGG_STACK_TRAIN_SIZE}
    stack_val: ${LOCAL_GRAPH_PART_STACK_VAL_SIZE}
    final_test: ${LOCAL_GRAPH_PART_FINAL_TEST_SIZE}
  model:
    variants: $(fresh_join_by_space "${variant_args[@]}")
    epochs: ${LOCAL_GRAPH_PART_EPOCHS}
    selection_metric: ${LOCAL_GRAPH_PART_SELECTION_METRIC}
    k: ${LOCAL_GRAPH_PART_K}
    warm_start_checkpoint: ${LOCAL_GRAPH_PART_WARM_START_CHECKPOINT}
    warm_start_freeze_epochs: ${LOCAL_GRAPH_PART_WARM_START_FREEZE_EPOCHS}
    warm_start_residual_gamma_init: ${LOCAL_GRAPH_PART_WARM_START_RESIDUAL_GAMMA_INIT}
  resources:
    binary_manifest: time=${LOCAL_GRAPH_PART_QCD_HGG_BINARY_MANIFEST_TIME} mem=${LOCAL_GRAPH_PART_QCD_HGG_BINARY_MANIFEST_MEM} cpus=${LOCAL_GRAPH_PART_QCD_HGG_BINARY_MANIFEST_CPUS}
    binary_hlt_cache: time=${LOCAL_GRAPH_PART_QCD_HGG_BINARY_HLT_CACHE_TIME} mem=${LOCAL_GRAPH_PART_QCD_HGG_BINARY_HLT_CACHE_MEM} cpus=${LOCAL_GRAPH_PART_QCD_HGG_BINARY_HLT_CACHE_CPUS}
    train: time=${LOCAL_GRAPH_PART_QCD_HGG_TRAIN_TIME} mem=${LOCAL_GRAPH_PART_QCD_HGG_TRAIN_MEM} cpus=${LOCAL_GRAPH_PART_QCD_HGG_TRAIN_CPUS}
    report: time=${LOCAL_GRAPH_PART_QCD_HGG_REPORT_TIME} mem=${LOCAL_GRAPH_PART_QCD_HGG_REPORT_MEM} cpus=${LOCAL_GRAPH_PART_QCD_HGG_REPORT_CPUS}
    score_fusion: time=${LOCAL_GRAPH_PART_QCD_HGG_SCORE_FUSION_TIME} mem=${LOCAL_GRAPH_PART_QCD_HGG_SCORE_FUSION_MEM} cpus=${LOCAL_GRAPH_PART_QCD_HGG_SCORE_FUSION_CPUS}
  outputs:
    tagger_root: ${LOCAL_GRAPH_PART_TAGGER_ROOT}
    final_report_json: ${LOCAL_GRAPH_PART_FINAL_REPORT_DIR}/local_graph_part_report.json
    final_report_md: ${LOCAL_GRAPH_PART_FINAL_REPORT_DIR}/local_graph_part_report.md
    final_metric_table: ${LOCAL_GRAPH_PART_FINAL_REPORT_DIR}/metric_table.csv
    adapter_diagnostics: ${LOCAL_GRAPH_PART_FINAL_REPORT_DIR}/adapter_diagnostics.csv
    hlt_degradation_summary: ${LOCAL_GRAPH_PART_FINAL_REPORT_DIR}/hlt_degradation_summary.csv
    score_fusion_report: ${LOCAL_GRAPH_PART_QCD_HGG_SCORE_FUSION_DIR}/fusion_report.json
    logs: ${PROJECT_DIR}/fresh_check_logs
SUMMARY
