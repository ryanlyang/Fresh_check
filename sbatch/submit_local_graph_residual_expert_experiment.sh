#!/usr/bin/env bash
# Submit the QCD-vs-Hgg local-graph residual-expert ladder.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${UPSTREAM_DEPENDENCY:=}"
: "${LOCAL_GRAPH_RESIDUAL_TAG:=$(date +%Y%m%d_%H%M%S)}"
: "${LOCAL_GRAPH_RESIDUAL_HLT_DEGRADATION_STRENGTH:=0.6}"
LOCAL_GRAPH_RESIDUAL_HLT_TAG="${LOCAL_GRAPH_RESIDUAL_HLT_DEGRADATION_STRENGTH//./p}"
: "${LOCAL_GRAPH_RESIDUAL_ROOT:=${OUTPUT_ROOT}/local_graph_residual_expert_qcd_hgg_binary_hlt${LOCAL_GRAPH_RESIDUAL_HLT_TAG}_${LOCAL_GRAPH_RESIDUAL_TAG}}"

: "${LOCAL_GRAPH_RESIDUAL_BUILD_BINARY_INPUTS:=1}"
: "${LOCAL_GRAPH_RESIDUAL_BUILD_DIRECT_BINARY_SPLITS:=1}"
: "${LOCAL_GRAPH_RESIDUAL_BINARY_INPUT_ROOT:=${LOCAL_GRAPH_RESIDUAL_ROOT}/binary_inputs}"
: "${LOCAL_GRAPH_RESIDUAL_SOURCE_MANIFEST_PATH:=${MANIFEST_PATH}}"
: "${LOCAL_GRAPH_RESIDUAL_BINARY_MANIFEST_PATH:=${LOCAL_GRAPH_RESIDUAL_BINARY_INPUT_ROOT}/split_manifest.json.gz}"
: "${LOCAL_GRAPH_RESIDUAL_HLT_CACHE_DIR:=${LOCAL_GRAPH_RESIDUAL_BINARY_INPUT_ROOT}/hlt_cache}"
: "${LOCAL_GRAPH_RESIDUAL_SOURCE_LABEL_NAMES:=QCD Hgg}"
: "${LOCAL_GRAPH_RESIDUAL_BINARY_LABEL_FILTER:=0 1}"

: "${LOCAL_GRAPH_RESIDUAL_TRAIN_BASELINE:=1}"
: "${LOCAL_GRAPH_RESIDUAL_TAGGER_ROOT:=${LOCAL_GRAPH_RESIDUAL_ROOT}/taggers}"
: "${LOCAL_GRAPH_RESIDUAL_BASELINE_CHECKPOINT:=${LOCAL_GRAPH_RESIDUAL_TAGGER_ROOT}/hlt_part_baseline/best_model_val.pt}"
: "${LOCAL_GRAPH_RESIDUAL_BASELINE_LOGIT_CACHE_DIR:=${LOCAL_GRAPH_RESIDUAL_ROOT}/baseline_logits}"
: "${LOCAL_GRAPH_RESIDUAL_EXPERT_ROOT:=${LOCAL_GRAPH_RESIDUAL_ROOT}/residual_experts}"
: "${LOCAL_GRAPH_RESIDUAL_LOSS_MODES:=A B C D}"
: "${LOCAL_GRAPH_RESIDUAL_LOCAL_ADAPTER:=point_attention}"

: "${LOCAL_GRAPH_RESIDUAL_MODEL_TRAIN_SIZE:=500000}"
: "${LOCAL_GRAPH_RESIDUAL_MODEL_VAL_SIZE:=150000}"
: "${LOCAL_GRAPH_RESIDUAL_STACK_TRAIN_SIZE:=500000}"
: "${LOCAL_GRAPH_RESIDUAL_STACK_VAL_SIZE:=150000}"
: "${LOCAL_GRAPH_RESIDUAL_FINAL_TEST_SIZE:=500000}"
: "${LOCAL_GRAPH_RESIDUAL_BASELINE_EPOCHS:=45}"
: "${LOCAL_GRAPH_RESIDUAL_EPOCHS:=30}"
: "${LOCAL_GRAPH_RESIDUAL_SELECTION_METRIC:=fpr_at_signal_eff_0p50}"
: "${LOCAL_GRAPH_RESIDUAL_K:=16}"
: "${LOCAL_GRAPH_RESIDUAL_WARM_START_ENABLED:=1}"
: "${LOCAL_GRAPH_RESIDUAL_REQUIRE_WARM_START:=1}"
: "${LOCAL_GRAPH_RESIDUAL_FREEZE_PART_EPOCHS:=0}"
: "${LOCAL_GRAPH_RESIDUAL_RESIDUAL_GAMMA_INIT:=0.0}"

: "${LOCAL_GRAPH_RESIDUAL_BINARY_MANIFEST_TIME:=04:00:00}"
: "${LOCAL_GRAPH_RESIDUAL_BINARY_MANIFEST_MEM:=16G}"
: "${LOCAL_GRAPH_RESIDUAL_BINARY_MANIFEST_CPUS:=2}"
: "${LOCAL_GRAPH_RESIDUAL_BINARY_HLT_CACHE_TIME:=1-00:00:00}"
: "${LOCAL_GRAPH_RESIDUAL_BINARY_HLT_CACHE_MEM:=128G}"
: "${LOCAL_GRAPH_RESIDUAL_BINARY_HLT_CACHE_CPUS:=4}"
: "${LOCAL_GRAPH_RESIDUAL_BASELINE_TIME:=2-12:00:00}"
: "${LOCAL_GRAPH_RESIDUAL_BASELINE_MEM:=160G}"
: "${LOCAL_GRAPH_RESIDUAL_BASELINE_CPUS:=8}"
: "${LOCAL_GRAPH_RESIDUAL_BASELINE_LOGIT_CACHE_TIME:=12:00:00}"
: "${LOCAL_GRAPH_RESIDUAL_BASELINE_LOGIT_CACHE_MEM:=128G}"
: "${LOCAL_GRAPH_RESIDUAL_BASELINE_LOGIT_CACHE_CPUS:=4}"
: "${LOCAL_GRAPH_RESIDUAL_TRAIN_TIME:=2-12:00:00}"
: "${LOCAL_GRAPH_RESIDUAL_TRAIN_MEM:=160G}"
: "${LOCAL_GRAPH_RESIDUAL_TRAIN_CPUS:=8}"
: "${LOCAL_GRAPH_RESIDUAL_REPORT_TIME:=1-00:00:00}"
: "${LOCAL_GRAPH_RESIDUAL_REPORT_MEM:=128G}"
: "${LOCAL_GRAPH_RESIDUAL_REPORT_CPUS:=4}"

: "${LOCAL_GRAPH_RESIDUAL_SUBMIT_FINAL_REPORT:=1}"
: "${LOCAL_GRAPH_RESIDUAL_SUBMIT_SCORE_FUSION:=0}"
: "${LOCAL_GRAPH_RESIDUAL_FINAL_REPORT_DIR:=${LOCAL_GRAPH_RESIDUAL_ROOT}/final_report}"

export LOCAL_GRAPH_RESIDUAL_ROOT
export LOCAL_GRAPH_RESIDUAL_HLT_CACHE_DIR
export LOCAL_GRAPH_RESIDUAL_BASELINE_LOGIT_CACHE_DIR
export LOCAL_GRAPH_RESIDUAL_EXPERT_ROOT
export LOCAL_GRAPH_RESIDUAL_WARM_START_CHECKPOINT="${LOCAL_GRAPH_RESIDUAL_BASELINE_CHECKPOINT}"
export LOCAL_GRAPH_RESIDUAL_EXPECTED_HLT_DEGRADATION_STRENGTH="${LOCAL_GRAPH_RESIDUAL_HLT_DEGRADATION_STRENGTH}"
export HLT_DEGRADATION_STRENGTH="${LOCAL_GRAPH_RESIDUAL_HLT_DEGRADATION_STRENGTH}"

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
  local value="${1,,}"
  value="${value//residual_/}"
  value="${value//boundary_pairwise/bpair}"
  value="${value//soft_fpr_bce_anchor/sfpr_bce}"
  value="${value//[^A-Za-z0-9_]/_}"
  printf '%s' "${value}"
}

fresh_split_words loss_mode_args "${LOCAL_GRAPH_RESIDUAL_LOSS_MODES}"

submitter_lock_dir="${LOCAL_GRAPH_RESIDUAL_ROOT}/.submission_lock"
fresh_refuse_existing_dir "${LOCAL_GRAPH_RESIDUAL_ROOT}"
fresh_claim_new_dir "${submitter_lock_dir}"
if ! fresh_is_dry_run; then
  {
    echo "created_at=$(date -Is)"
    echo "project_dir=${PROJECT_DIR}"
    echo "source_commit=$(fresh_source_commit)"
    echo "task=QCD_vs_Hgg_local_graph_residual_expert"
    echo "root=${LOCAL_GRAPH_RESIDUAL_ROOT}"
    echo "hlt_degradation_strength=${LOCAL_GRAPH_RESIDUAL_HLT_DEGRADATION_STRENGTH}"
    echo "binary_manifest=${LOCAL_GRAPH_RESIDUAL_BINARY_MANIFEST_PATH}"
    echo "hlt_cache=${LOCAL_GRAPH_RESIDUAL_HLT_CACHE_DIR}"
    echo "train_baseline=${LOCAL_GRAPH_RESIDUAL_TRAIN_BASELINE}"
    echo "baseline_checkpoint=${LOCAL_GRAPH_RESIDUAL_BASELINE_CHECKPOINT}"
    echo "baseline_logit_cache=${LOCAL_GRAPH_RESIDUAL_BASELINE_LOGIT_CACHE_DIR}"
    echo "loss_modes=$(fresh_join_by_space "${loss_mode_args[@]}")"
    echo "alpha_shrinkage=reported_as_model_val_gamma_shrunk_rows"
    echo "selection_metric=${LOCAL_GRAPH_RESIDUAL_SELECTION_METRIC}"
  } > "${submitter_lock_dir}/metadata.txt"
fi

binary_manifest_jid=""
binary_hlt_cache_jid=""
baseline_jid=""
baseline_logit_cache_jid=""
final_report_jid=""
score_fusion_jid=""
residual_job_ids=()
residual_output_names=()
input_dependency="${UPSTREAM_DEPENDENCY}"

if fresh_bool_enabled "${LOCAL_GRAPH_RESIDUAL_BUILD_BINARY_INPUTS}"; then
  export LABEL_FILTER_OUTPUT_MANIFEST_PATH="${LOCAL_GRAPH_RESIDUAL_BINARY_MANIFEST_PATH}"
  export LABEL_FILTER_NAMES="${LOCAL_GRAPH_RESIDUAL_SOURCE_LABEL_NAMES}"
  export LABEL_FILTER_MANIFEST_PATH="${LOCAL_GRAPH_RESIDUAL_BINARY_MANIFEST_PATH}"
  export LABEL_FILTER_HLT_CACHE_DIR="${LOCAL_GRAPH_RESIDUAL_HLT_CACHE_DIR}"
  export LABEL_FILTER_HLT_SPLITS="model_train model_val stack_train stack_val final_test"
  export LABEL_FILTER_MODEL_TRAIN_SIZE="${LOCAL_GRAPH_RESIDUAL_MODEL_TRAIN_SIZE}"
  export LABEL_FILTER_MODEL_VAL_SIZE="${LOCAL_GRAPH_RESIDUAL_MODEL_VAL_SIZE}"
  export LABEL_FILTER_STACK_TRAIN_SIZE="${LOCAL_GRAPH_RESIDUAL_STACK_TRAIN_SIZE}"
  export LABEL_FILTER_STACK_VAL_SIZE="${LOCAL_GRAPH_RESIDUAL_STACK_VAL_SIZE}"
  export LABEL_FILTER_FINAL_TEST_SIZE="${LOCAL_GRAPH_RESIDUAL_FINAL_TEST_SIZE}"

  if fresh_bool_enabled "${LOCAL_GRAPH_RESIDUAL_BUILD_DIRECT_BINARY_SPLITS}"; then
    mapfile -t manifest_args < <(
      afterok_args \
        "${UPSTREAM_DEPENDENCY}" \
        --time="${LOCAL_GRAPH_RESIDUAL_BINARY_MANIFEST_TIME}" \
        --cpus-per-task="${LOCAL_GRAPH_RESIDUAL_BINARY_MANIFEST_CPUS}" \
        --mem="${LOCAL_GRAPH_RESIDUAL_BINARY_MANIFEST_MEM}" \
        "${SCRIPT_DIR}/run_build_label_filtered_fresh_splits.sh"
    )
  else
    export LABEL_FILTER_SOURCE_MANIFEST_PATH="${LOCAL_GRAPH_RESIDUAL_SOURCE_MANIFEST_PATH}"
    export LABEL_FILTER_REMAP_LABELS=1
    mapfile -t manifest_args < <(
      afterok_args \
        "${UPSTREAM_DEPENDENCY}" \
        --time="${LOCAL_GRAPH_RESIDUAL_BINARY_MANIFEST_TIME}" \
        --cpus-per-task="${LOCAL_GRAPH_RESIDUAL_BINARY_MANIFEST_CPUS}" \
        --mem="${LOCAL_GRAPH_RESIDUAL_BINARY_MANIFEST_MEM}" \
        "${SCRIPT_DIR}/run_build_label_filtered_split_manifest.sh"
    )
  fi
  binary_manifest_jid="$(submit_job "localgraph_residual_binary_manifest" "${manifest_args[@]}")"
  echo "submitted localgraph_residual_binary_manifest=${binary_manifest_jid}"

  binary_hlt_cache_jid="$(submit_job "localgraph_residual_binary_hlt_cache" \
    --time="${LOCAL_GRAPH_RESIDUAL_BINARY_HLT_CACHE_TIME}" \
    --cpus-per-task="${LOCAL_GRAPH_RESIDUAL_BINARY_HLT_CACHE_CPUS}" \
    --mem="${LOCAL_GRAPH_RESIDUAL_BINARY_HLT_CACHE_MEM}" \
    --dependency="afterok:${binary_manifest_jid}" \
    "${SCRIPT_DIR}/run_build_label_filtered_hlt_cache.sh")"
  echo "submitted localgraph_residual_binary_hlt_cache=${binary_hlt_cache_jid}"
  input_dependency="${binary_hlt_cache_jid}"
else
  fresh_require_file "${LOCAL_GRAPH_RESIDUAL_BINARY_MANIFEST_PATH}"
  for split in model_train model_val stack_train stack_val final_test; do
    fresh_require_file "${LOCAL_GRAPH_RESIDUAL_HLT_CACHE_DIR}/${split}_fixed_hlt.npz"
    fresh_require_file "${LOCAL_GRAPH_RESIDUAL_HLT_CACHE_DIR}/${split}_fixed_hlt_metadata.json"
  done
fi

export LOCAL_GRAPH_PART_ROOT="${LOCAL_GRAPH_RESIDUAL_ROOT}"
export LOCAL_GRAPH_PART_TAGGER_ROOT="${LOCAL_GRAPH_RESIDUAL_TAGGER_ROOT}"
export LOCAL_GRAPH_PART_HLT_CACHE_DIR="${LOCAL_GRAPH_RESIDUAL_HLT_CACHE_DIR}"
export LOCAL_GRAPH_PART_MODEL_TRAIN_SIZE="${LOCAL_GRAPH_RESIDUAL_MODEL_TRAIN_SIZE}"
export LOCAL_GRAPH_PART_MODEL_VAL_SIZE="${LOCAL_GRAPH_RESIDUAL_MODEL_VAL_SIZE}"
export LOCAL_GRAPH_PART_STACK_VAL_SIZE="${LOCAL_GRAPH_RESIDUAL_STACK_VAL_SIZE}"
export LOCAL_GRAPH_PART_FINAL_TEST_SIZE="${LOCAL_GRAPH_RESIDUAL_FINAL_TEST_SIZE}"
export LOCAL_GRAPH_PART_EPOCHS="${LOCAL_GRAPH_RESIDUAL_BASELINE_EPOCHS}"
export LOCAL_GRAPH_PART_SELECTION_METRIC="${LOCAL_GRAPH_RESIDUAL_SELECTION_METRIC}"
export LOCAL_GRAPH_PART_K="${LOCAL_GRAPH_RESIDUAL_K}"
export LOCAL_GRAPH_PART_CONFIRM_FINAL_TEST=1
export LOCAL_GRAPH_PART_EXPECTED_HLT_DEGRADATION_STRENGTH="${LOCAL_GRAPH_RESIDUAL_HLT_DEGRADATION_STRENGTH}"

if fresh_bool_enabled "${LOCAL_GRAPH_RESIDUAL_TRAIN_BASELINE}"; then
  mapfile -t baseline_args < <(
    afterok_args \
      "${input_dependency}" \
      --time="${LOCAL_GRAPH_RESIDUAL_BASELINE_TIME}" \
      --cpus-per-task="${LOCAL_GRAPH_RESIDUAL_BASELINE_CPUS}" \
      --mem="${LOCAL_GRAPH_RESIDUAL_BASELINE_MEM}" \
      "${SCRIPT_DIR}/run_train_local_graph_part_tagger.sh" \
      "hlt_part_baseline"
  )
  baseline_jid="$(submit_job "localgraph_residual_hlt_baseline" "${baseline_args[@]}")"
  echo "submitted localgraph_residual_hlt_baseline=${baseline_jid}"
else
  fresh_require_file "${LOCAL_GRAPH_RESIDUAL_BASELINE_CHECKPOINT}"
fi

cache_dependency="${baseline_jid:-${input_dependency}}"
export LOCAL_GRAPH_BASELINE_LOGIT_CACHE_OUTPUT_DIR="${LOCAL_GRAPH_RESIDUAL_BASELINE_LOGIT_CACHE_DIR}"
export LOCAL_GRAPH_BASELINE_LOGIT_CACHE_HLT_CACHE_DIR="${LOCAL_GRAPH_RESIDUAL_HLT_CACHE_DIR}"
export LOCAL_GRAPH_BASELINE_LOGIT_CACHE_CHECKPOINT="${LOCAL_GRAPH_RESIDUAL_BASELINE_CHECKPOINT}"
export LOCAL_GRAPH_BASELINE_LOGIT_CACHE_MODEL_TRAIN_SIZE="${LOCAL_GRAPH_RESIDUAL_MODEL_TRAIN_SIZE}"
export LOCAL_GRAPH_BASELINE_LOGIT_CACHE_MODEL_VAL_SIZE="${LOCAL_GRAPH_RESIDUAL_MODEL_VAL_SIZE}"
export LOCAL_GRAPH_BASELINE_LOGIT_CACHE_STACK_TRAIN_SIZE="${LOCAL_GRAPH_RESIDUAL_STACK_TRAIN_SIZE}"
export LOCAL_GRAPH_BASELINE_LOGIT_CACHE_STACK_VAL_SIZE="${LOCAL_GRAPH_RESIDUAL_STACK_VAL_SIZE}"
export LOCAL_GRAPH_BASELINE_LOGIT_CACHE_FINAL_TEST_SIZE="${LOCAL_GRAPH_RESIDUAL_FINAL_TEST_SIZE}"
export LOCAL_GRAPH_BASELINE_LOGIT_CACHE_EXPECTED_HLT_DEGRADATION_STRENGTH="${LOCAL_GRAPH_RESIDUAL_HLT_DEGRADATION_STRENGTH}"

mapfile -t cache_args < <(
  afterok_args \
    "${cache_dependency}" \
    --time="${LOCAL_GRAPH_RESIDUAL_BASELINE_LOGIT_CACHE_TIME}" \
    --cpus-per-task="${LOCAL_GRAPH_RESIDUAL_BASELINE_LOGIT_CACHE_CPUS}" \
    --mem="${LOCAL_GRAPH_RESIDUAL_BASELINE_LOGIT_CACHE_MEM}" \
    "${SCRIPT_DIR}/run_cache_local_graph_baseline_logits.sh"
)
baseline_logit_cache_jid="$(submit_job "localgraph_residual_baseline_logits" "${cache_args[@]}")"
echo "submitted localgraph_residual_baseline_logits=${baseline_logit_cache_jid}"

export LOCAL_GRAPH_RESIDUAL_MODEL_TRAIN_SIZE
export LOCAL_GRAPH_RESIDUAL_MODEL_VAL_SIZE
export LOCAL_GRAPH_RESIDUAL_EPOCHS
export LOCAL_GRAPH_RESIDUAL_SELECTION_METRIC
export LOCAL_GRAPH_RESIDUAL_K
export LOCAL_GRAPH_RESIDUAL_WARM_START_ENABLED
export LOCAL_GRAPH_RESIDUAL_REQUIRE_WARM_START
export LOCAL_GRAPH_RESIDUAL_FREEZE_PART_EPOCHS
export LOCAL_GRAPH_RESIDUAL_RESIDUAL_GAMMA_INIT

for loss_mode in "${loss_mode_args[@]}"; do
  label="$(safe_label "${loss_mode}")"
  export LOCAL_GRAPH_RESIDUAL_OUTPUT_NAME="${label}"
  mapfile -t residual_args < <(
    afterok_args \
      "${baseline_logit_cache_jid}" \
      --time="${LOCAL_GRAPH_RESIDUAL_TRAIN_TIME}" \
      --cpus-per-task="${LOCAL_GRAPH_RESIDUAL_TRAIN_CPUS}" \
      --mem="${LOCAL_GRAPH_RESIDUAL_TRAIN_MEM}" \
      "${SCRIPT_DIR}/run_train_local_graph_residual_expert.sh" \
      "${loss_mode}" \
      "${LOCAL_GRAPH_RESIDUAL_LOCAL_ADAPTER}"
  )
  residual_jid="$(submit_job "localgraph_residual_${label}" "${residual_args[@]}")"
  residual_job_ids+=("${residual_jid}")
  residual_output_names+=("${label}")
  echo "submitted localgraph_residual_${label}=${residual_jid}"
done

residual_dep="$(fresh_join_by_colon "${residual_job_ids[@]}")"
if fresh_bool_enabled "${LOCAL_GRAPH_RESIDUAL_SUBMIT_FINAL_REPORT}"; then
  export LOCAL_GRAPH_RESIDUAL_FINAL_REPORT_DIR
  export LOCAL_GRAPH_RESIDUAL_REPORT_VARIANTS
  LOCAL_GRAPH_RESIDUAL_REPORT_VARIANTS="$(fresh_join_by_space "${residual_output_names[@]}")"
  export LOCAL_GRAPH_RESIDUAL_REPORT_VARIANTS
  export LOCAL_GRAPH_RESIDUAL_REPORT_MODEL_VAL_SIZE="${LOCAL_GRAPH_RESIDUAL_MODEL_VAL_SIZE}"
  export LOCAL_GRAPH_RESIDUAL_REPORT_STACK_VAL_SIZE="${LOCAL_GRAPH_RESIDUAL_STACK_VAL_SIZE}"
  export LOCAL_GRAPH_RESIDUAL_REPORT_FINAL_TEST_SIZE="${LOCAL_GRAPH_RESIDUAL_FINAL_TEST_SIZE}"
  export LOCAL_GRAPH_RESIDUAL_REPORT_CONFIRM_FINAL_TEST=1
  final_report_jid="$(submit_job "localgraph_residual_final_report" \
    --time="${LOCAL_GRAPH_RESIDUAL_REPORT_TIME}" \
    --cpus-per-task="${LOCAL_GRAPH_RESIDUAL_REPORT_CPUS}" \
    --mem="${LOCAL_GRAPH_RESIDUAL_REPORT_MEM}" \
    --dependency="afterok:${residual_dep}" \
    "${SCRIPT_DIR}/run_write_local_graph_residual_expert_report.sh")"
  echo "submitted localgraph_residual_final_report=${final_report_jid}"
fi
if fresh_bool_enabled "${LOCAL_GRAPH_RESIDUAL_SUBMIT_SCORE_FUSION}"; then
  echo "LOCAL_GRAPH_RESIDUAL_SUBMIT_SCORE_FUSION=1 requested, but residual score-fusion export/reporting lands after Step 9." >&2
  exit 2
fi

cat <<SUMMARY
local_graph_residual_expert_submission:
  task: QCD_vs_Hgg_local_graph_residual_expert
  root: ${LOCAL_GRAPH_RESIDUAL_ROOT}
  hlt_degradation_strength: ${LOCAL_GRAPH_RESIDUAL_HLT_DEGRADATION_STRENGTH}
  binary_inputs:
    build_enabled: ${LOCAL_GRAPH_RESIDUAL_BUILD_BINARY_INPUTS}
    direct_binary_splits: ${LOCAL_GRAPH_RESIDUAL_BUILD_DIRECT_BINARY_SPLITS}
    source_manifest: ${LOCAL_GRAPH_RESIDUAL_SOURCE_MANIFEST_PATH}
    filtered_manifest: ${LOCAL_GRAPH_RESIDUAL_BINARY_MANIFEST_PATH}
    filtered_hlt_cache: ${LOCAL_GRAPH_RESIDUAL_HLT_CACHE_DIR}
    manifest_job_id: ${binary_manifest_jid:-none}
    hlt_cache_job_id: ${binary_hlt_cache_jid:-none}
  baseline:
    train_enabled: ${LOCAL_GRAPH_RESIDUAL_TRAIN_BASELINE}
    checkpoint: ${LOCAL_GRAPH_RESIDUAL_BASELINE_CHECKPOINT}
    train_job_id: ${baseline_jid:-none}
  baseline_logit_cache:
    output_dir: ${LOCAL_GRAPH_RESIDUAL_BASELINE_LOGIT_CACHE_DIR}
    job_id: ${baseline_logit_cache_jid}
  residual_experts:
    local_adapter: ${LOCAL_GRAPH_RESIDUAL_LOCAL_ADAPTER}
    loss_modes: $(fresh_join_by_space "${loss_mode_args[@]}")
    alpha_shrinkage: reported_as_model_val_gamma_shrunk_rows
    job_ids: $(fresh_join_by_space "${residual_job_ids[@]}")
    dependency: ${residual_dep}
  expected_jobs:
    binary_manifest: $([[ -n "${binary_manifest_jid}" ]] && echo 1 || echo 0)
    binary_hlt_cache: $([[ -n "${binary_hlt_cache_jid}" ]] && echo 1 || echo 0)
    hlt_baseline: $([[ -n "${baseline_jid}" ]] && echo 1 || echo 0)
    baseline_logit_cache: 1
    residual_experts: ${#residual_job_ids[@]}
    final_report: $([[ -n "${final_report_jid}" ]] && echo 1 || echo 0)
    score_fusion: $([[ -n "${score_fusion_jid}" ]] && echo 1 || echo 0)
    total_submitted: ${submit_count}
  requested_split_caps:
    model_train: ${LOCAL_GRAPH_RESIDUAL_MODEL_TRAIN_SIZE}
    model_val: ${LOCAL_GRAPH_RESIDUAL_MODEL_VAL_SIZE}
    stack_train: ${LOCAL_GRAPH_RESIDUAL_STACK_TRAIN_SIZE}
    stack_val: ${LOCAL_GRAPH_RESIDUAL_STACK_VAL_SIZE}
    final_test: ${LOCAL_GRAPH_RESIDUAL_FINAL_TEST_SIZE}
  model:
    baseline_epochs: ${LOCAL_GRAPH_RESIDUAL_BASELINE_EPOCHS}
    residual_epochs: ${LOCAL_GRAPH_RESIDUAL_EPOCHS}
    selection_metric: ${LOCAL_GRAPH_RESIDUAL_SELECTION_METRIC}
    k: ${LOCAL_GRAPH_RESIDUAL_K}
    warm_start_checkpoint: ${LOCAL_GRAPH_RESIDUAL_BASELINE_CHECKPOINT}
    warm_start_enabled: ${LOCAL_GRAPH_RESIDUAL_WARM_START_ENABLED}
    freeze_part_epochs: ${LOCAL_GRAPH_RESIDUAL_FREEZE_PART_EPOCHS}
  resources:
    binary_manifest: time=${LOCAL_GRAPH_RESIDUAL_BINARY_MANIFEST_TIME} mem=${LOCAL_GRAPH_RESIDUAL_BINARY_MANIFEST_MEM} cpus=${LOCAL_GRAPH_RESIDUAL_BINARY_MANIFEST_CPUS}
    binary_hlt_cache: time=${LOCAL_GRAPH_RESIDUAL_BINARY_HLT_CACHE_TIME} mem=${LOCAL_GRAPH_RESIDUAL_BINARY_HLT_CACHE_MEM} cpus=${LOCAL_GRAPH_RESIDUAL_BINARY_HLT_CACHE_CPUS}
    baseline: time=${LOCAL_GRAPH_RESIDUAL_BASELINE_TIME} mem=${LOCAL_GRAPH_RESIDUAL_BASELINE_MEM} cpus=${LOCAL_GRAPH_RESIDUAL_BASELINE_CPUS}
    baseline_logit_cache: time=${LOCAL_GRAPH_RESIDUAL_BASELINE_LOGIT_CACHE_TIME} mem=${LOCAL_GRAPH_RESIDUAL_BASELINE_LOGIT_CACHE_MEM} cpus=${LOCAL_GRAPH_RESIDUAL_BASELINE_LOGIT_CACHE_CPUS}
    residual_train: time=${LOCAL_GRAPH_RESIDUAL_TRAIN_TIME} mem=${LOCAL_GRAPH_RESIDUAL_TRAIN_MEM} cpus=${LOCAL_GRAPH_RESIDUAL_TRAIN_CPUS}
    final_report: time=${LOCAL_GRAPH_RESIDUAL_REPORT_TIME} mem=${LOCAL_GRAPH_RESIDUAL_REPORT_MEM} cpus=${LOCAL_GRAPH_RESIDUAL_REPORT_CPUS}
  outputs:
    baseline_tagger_root: ${LOCAL_GRAPH_RESIDUAL_TAGGER_ROOT}
    baseline_logit_cache: ${LOCAL_GRAPH_RESIDUAL_BASELINE_LOGIT_CACHE_DIR}/baseline_logit_manifest.json
    residual_expert_root: ${LOCAL_GRAPH_RESIDUAL_EXPERT_ROOT}
    final_report: ${LOCAL_GRAPH_RESIDUAL_FINAL_REPORT_DIR}/local_graph_residual_expert_report.json
    logs: ${PROJECT_DIR}/fresh_check_logs
SUMMARY
