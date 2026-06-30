#!/usr/bin/env bash
# Submit the QCD-vs-Hgg local-graph residual-expert V2 ladder.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${UPSTREAM_DEPENDENCY:=}"
: "${LOCAL_GRAPH_RESIDUAL_V2_TAG:=$(date +%Y%m%d_%H%M%S)}"
: "${LOCAL_GRAPH_RESIDUAL_V2_HLT_DEGRADATION_STRENGTH:=0.6}"
LOCAL_GRAPH_RESIDUAL_V2_HLT_TAG="${LOCAL_GRAPH_RESIDUAL_V2_HLT_DEGRADATION_STRENGTH//./p}"
: "${LOCAL_GRAPH_RESIDUAL_V2_ROOT:=${OUTPUT_ROOT}/local_graph_residual_expert_v2_qcd_hgg_binary_hlt${LOCAL_GRAPH_RESIDUAL_V2_HLT_TAG}_${LOCAL_GRAPH_RESIDUAL_V2_TAG}}"

: "${LOCAL_GRAPH_RESIDUAL_V2_EXISTING_INPUT_ROOT:=}"
: "${LOCAL_GRAPH_RESIDUAL_V2_HLT_CACHE_DIR:=${LOCAL_GRAPH_RESIDUAL_V2_EXISTING_INPUT_ROOT:+${LOCAL_GRAPH_RESIDUAL_V2_EXISTING_INPUT_ROOT}/binary_inputs/hlt_cache}}"
: "${LOCAL_GRAPH_RESIDUAL_V2_HLT_CACHE_DIR:=${LOCAL_GRAPH_RESIDUAL_V2_ROOT}/binary_inputs/hlt_cache}"
: "${LOCAL_GRAPH_RESIDUAL_V2_BASELINE_CHECKPOINT:=}"
: "${LOCAL_GRAPH_RESIDUAL_V2_EMBEDDING_CACHE_DIR:=${LOCAL_GRAPH_RESIDUAL_V2_ROOT}/baseline_embeddings}"
: "${LOCAL_GRAPH_RESIDUAL_V2_EXPERT_ROOT:=${LOCAL_GRAPH_RESIDUAL_V2_ROOT}/residual_experts}"
: "${LOCAL_GRAPH_RESIDUAL_V2_LOSS_MODES:=A C D}"
: "${LOCAL_GRAPH_RESIDUAL_V2_FINAL_REPORT_DIR:=${LOCAL_GRAPH_RESIDUAL_V2_ROOT}/final_report}"

: "${LOCAL_GRAPH_RESIDUAL_V2_MODEL_TRAIN_SIZE:=500000}"
: "${LOCAL_GRAPH_RESIDUAL_V2_MODEL_VAL_SIZE:=150000}"
: "${LOCAL_GRAPH_RESIDUAL_V2_STACK_TRAIN_SIZE:=500000}"
: "${LOCAL_GRAPH_RESIDUAL_V2_STACK_VAL_SIZE:=150000}"
: "${LOCAL_GRAPH_RESIDUAL_V2_FINAL_TEST_SIZE:=500000}"
: "${LOCAL_GRAPH_RESIDUAL_V2_EPOCHS:=30}"
: "${LOCAL_GRAPH_RESIDUAL_V2_SELECTION_METRIC:=fpr_at_signal_eff_0p50}"
: "${LOCAL_GRAPH_RESIDUAL_V2_K:=16}"
: "${LOCAL_GRAPH_RESIDUAL_V2_RESIDUAL_INPUT_MODE:=full}"
: "${LOCAL_GRAPH_RESIDUAL_V2_CONDITION_CONTROL_MODE:=normal}"
: "${LOCAL_GRAPH_RESIDUAL_V2_LABEL_CONTROL_MODE:=normal}"

: "${LOCAL_GRAPH_RESIDUAL_V2_CACHE_TIME:=12:00:00}"
: "${LOCAL_GRAPH_RESIDUAL_V2_CACHE_MEM:=128G}"
: "${LOCAL_GRAPH_RESIDUAL_V2_CACHE_CPUS:=4}"
: "${LOCAL_GRAPH_RESIDUAL_V2_TRAIN_TIME:=2-12:00:00}"
: "${LOCAL_GRAPH_RESIDUAL_V2_TRAIN_MEM:=160G}"
: "${LOCAL_GRAPH_RESIDUAL_V2_TRAIN_CPUS:=8}"
: "${LOCAL_GRAPH_RESIDUAL_V2_REPORT_TIME:=08:00:00}"
: "${LOCAL_GRAPH_RESIDUAL_V2_REPORT_MEM:=32G}"
: "${LOCAL_GRAPH_RESIDUAL_V2_REPORT_CPUS:=2}"

: "${LOCAL_GRAPH_RESIDUAL_V2_SUBMIT_REPORT:=1}"
: "${LOCAL_GRAPH_RESIDUAL_V2_REPORT_COMPARISON_SPLIT:=final_test}"
: "${LOCAL_GRAPH_RESIDUAL_V2_REPORT_CONFIRM_FINAL_TEST:=1}"

export LOCAL_GRAPH_RESIDUAL_V2_ROOT
export LOCAL_GRAPH_RESIDUAL_V2_HLT_CACHE_DIR
export LOCAL_GRAPH_RESIDUAL_V2_BASELINE_CHECKPOINT
export LOCAL_GRAPH_RESIDUAL_V2_EMBEDDING_CACHE_DIR
export LOCAL_GRAPH_RESIDUAL_V2_EXPERT_ROOT
export LOCAL_GRAPH_RESIDUAL_V2_EXPECTED_HLT_DEGRADATION_STRENGTH="${LOCAL_GRAPH_RESIDUAL_V2_HLT_DEGRADATION_STRENGTH}"
export HLT_DEGRADATION_STRENGTH="${LOCAL_GRAPH_RESIDUAL_V2_HLT_DEGRADATION_STRENGTH}"

fresh_prepare_submitter

if [[ -z "${LOCAL_GRAPH_RESIDUAL_V2_BASELINE_CHECKPOINT}" ]]; then
  echo "LOCAL_GRAPH_RESIDUAL_V2_BASELINE_CHECKPOINT must point at the existing hlt_part_baseline/best_model_val.pt" >&2
  exit 2
fi
fresh_require_file "${LOCAL_GRAPH_RESIDUAL_V2_BASELINE_CHECKPOINT}"
for split in model_train model_val stack_train stack_val final_test; do
  fresh_require_file "${LOCAL_GRAPH_RESIDUAL_V2_HLT_CACHE_DIR}/${split}_fixed_hlt.npz"
  fresh_require_file "${LOCAL_GRAPH_RESIDUAL_V2_HLT_CACHE_DIR}/${split}_fixed_hlt_metadata.json"
done

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

safe_v2_label() {
  local value="${1,,}"
  value="${value//residual_v2_/}"
  value="${value//boundary_pairwise/bpair}"
  value="${value//soft_fpr_bce_anchor/sfpr_bce}"
  value="${value//[^A-Za-z0-9_]/_}"
  printf '%s' "${value}"
}

fresh_split_words loss_mode_args "${LOCAL_GRAPH_RESIDUAL_V2_LOSS_MODES}"

submitter_lock_dir="${LOCAL_GRAPH_RESIDUAL_V2_ROOT}/.submission_lock"
fresh_refuse_existing_dir "${LOCAL_GRAPH_RESIDUAL_V2_ROOT}"
fresh_claim_new_dir "${submitter_lock_dir}"
if ! fresh_is_dry_run; then
  {
    echo "created_at=$(date -Is)"
    echo "project_dir=${PROJECT_DIR}"
    echo "source_commit=$(fresh_source_commit)"
    echo "task=QCD_vs_Hgg_local_graph_residual_expert_v2"
    echo "root=${LOCAL_GRAPH_RESIDUAL_V2_ROOT}"
    echo "hlt_degradation_strength=${LOCAL_GRAPH_RESIDUAL_V2_HLT_DEGRADATION_STRENGTH}"
    echo "hlt_cache=${LOCAL_GRAPH_RESIDUAL_V2_HLT_CACHE_DIR}"
    echo "baseline_checkpoint=${LOCAL_GRAPH_RESIDUAL_V2_BASELINE_CHECKPOINT}"
    echo "embedding_cache=${LOCAL_GRAPH_RESIDUAL_V2_EMBEDDING_CACHE_DIR}"
    echo "loss_modes=$(fresh_join_by_space "${loss_mode_args[@]}")"
    echo "gamma_shrinkage=reported_as_model_val_validation_shrunk_rows"
    echo "selection_metric=${LOCAL_GRAPH_RESIDUAL_V2_SELECTION_METRIC}"
  } > "${submitter_lock_dir}/metadata.txt"
fi

cache_jid=""
report_jid=""
residual_job_ids=()
residual_output_names=()

export LOCAL_GRAPH_RESIDUAL_V2_MODEL_TRAIN_SIZE
export LOCAL_GRAPH_RESIDUAL_V2_MODEL_VAL_SIZE
export LOCAL_GRAPH_RESIDUAL_V2_STACK_TRAIN_SIZE
export LOCAL_GRAPH_RESIDUAL_V2_STACK_VAL_SIZE
export LOCAL_GRAPH_RESIDUAL_V2_FINAL_TEST_SIZE

mapfile -t cache_args < <(
  afterok_args \
    "${UPSTREAM_DEPENDENCY}" \
    --time="${LOCAL_GRAPH_RESIDUAL_V2_CACHE_TIME}" \
    --cpus-per-task="${LOCAL_GRAPH_RESIDUAL_V2_CACHE_CPUS}" \
    --mem="${LOCAL_GRAPH_RESIDUAL_V2_CACHE_MEM}" \
    "${SCRIPT_DIR}/run_cache_local_graph_residual_v2_embeddings.sh"
)
cache_jid="$(submit_job "localgraph_residual_v2_embeddings" "${cache_args[@]}")"
echo "submitted localgraph_residual_v2_embeddings=${cache_jid}"

export LOCAL_GRAPH_RESIDUAL_V2_EPOCHS
export LOCAL_GRAPH_RESIDUAL_V2_SELECTION_METRIC
export LOCAL_GRAPH_RESIDUAL_V2_K
export LOCAL_GRAPH_RESIDUAL_V2_RESIDUAL_INPUT_MODE
export LOCAL_GRAPH_RESIDUAL_V2_CONDITION_CONTROL_MODE
export LOCAL_GRAPH_RESIDUAL_V2_LABEL_CONTROL_MODE

for loss_mode in "${loss_mode_args[@]}"; do
  label="$(safe_v2_label "${loss_mode}")"
  export LOCAL_GRAPH_RESIDUAL_V2_OUTPUT_NAME="${label}"
  mapfile -t train_args < <(
    afterok_args \
      "${cache_jid}" \
      --time="${LOCAL_GRAPH_RESIDUAL_V2_TRAIN_TIME}" \
      --cpus-per-task="${LOCAL_GRAPH_RESIDUAL_V2_TRAIN_CPUS}" \
      --mem="${LOCAL_GRAPH_RESIDUAL_V2_TRAIN_MEM}" \
      "${SCRIPT_DIR}/run_train_local_graph_residual_expert_v2.sh" \
      "${loss_mode}"
  )
  train_jid="$(submit_job "localgraph_residual_v2_${label}" "${train_args[@]}")"
  residual_job_ids+=("${train_jid}")
  residual_output_names+=("${label}")
  echo "submitted localgraph_residual_v2_${label}=${train_jid}"
done

residual_dep="$(fresh_join_by_colon "${residual_job_ids[@]}")"
if fresh_bool_enabled "${LOCAL_GRAPH_RESIDUAL_V2_SUBMIT_REPORT}"; then
  export LOCAL_GRAPH_RESIDUAL_V2_FINAL_REPORT_DIR
  LOCAL_GRAPH_RESIDUAL_V2_REPORT_VARIANTS="$(fresh_join_by_space "${residual_output_names[@]}")"
  export LOCAL_GRAPH_RESIDUAL_V2_REPORT_VARIANTS
  export LOCAL_GRAPH_RESIDUAL_V2_REPORT_COMPARISON_SPLIT
  export LOCAL_GRAPH_RESIDUAL_V2_REPORT_CONFIRM_FINAL_TEST
  report_jid="$(submit_job "localgraph_residual_v2_report" \
    --time="${LOCAL_GRAPH_RESIDUAL_V2_REPORT_TIME}" \
    --cpus-per-task="${LOCAL_GRAPH_RESIDUAL_V2_REPORT_CPUS}" \
    --mem="${LOCAL_GRAPH_RESIDUAL_V2_REPORT_MEM}" \
    --dependency="afterok:${residual_dep}" \
    "${SCRIPT_DIR}/run_write_local_graph_residual_expert_v2_report.sh")"
  echo "submitted localgraph_residual_v2_report=${report_jid}"
fi

cat <<SUMMARY
local_graph_residual_expert_v2_submission:
  task: QCD_vs_Hgg_local_graph_residual_expert_v2
  root: ${LOCAL_GRAPH_RESIDUAL_V2_ROOT}
  hlt_degradation_strength: ${LOCAL_GRAPH_RESIDUAL_V2_HLT_DEGRADATION_STRENGTH}
  inputs:
    hlt_cache: ${LOCAL_GRAPH_RESIDUAL_V2_HLT_CACHE_DIR}
    baseline_checkpoint: ${LOCAL_GRAPH_RESIDUAL_V2_BASELINE_CHECKPOINT}
  embedding_cache:
    output_dir: ${LOCAL_GRAPH_RESIDUAL_V2_EMBEDDING_CACHE_DIR}
    job_id: ${cache_jid}
  residual_experts:
    loss_modes: $(fresh_join_by_space "${loss_mode_args[@]}")
    output_names: $(fresh_join_by_space "${residual_output_names[@]}")
    gamma_shrinkage: reported_as_model_val_validation_shrunk_rows
    residual_input_mode: ${LOCAL_GRAPH_RESIDUAL_V2_RESIDUAL_INPUT_MODE}
    condition_control_mode: ${LOCAL_GRAPH_RESIDUAL_V2_CONDITION_CONTROL_MODE}
    label_control_mode: ${LOCAL_GRAPH_RESIDUAL_V2_LABEL_CONTROL_MODE}
    job_ids: $(fresh_join_by_space "${residual_job_ids[@]}")
    dependency: ${residual_dep}
  expected_jobs:
    embedding_cache: 1
    residual_experts: ${#residual_job_ids[@]}
    report: $([[ -n "${report_jid}" ]] && echo 1 || echo 0)
    total_submitted: ${submit_count}
  requested_split_caps:
    model_train: ${LOCAL_GRAPH_RESIDUAL_V2_MODEL_TRAIN_SIZE}
    model_val: ${LOCAL_GRAPH_RESIDUAL_V2_MODEL_VAL_SIZE}
    stack_train: ${LOCAL_GRAPH_RESIDUAL_V2_STACK_TRAIN_SIZE}
    stack_val: ${LOCAL_GRAPH_RESIDUAL_V2_STACK_VAL_SIZE}
    final_test: ${LOCAL_GRAPH_RESIDUAL_V2_FINAL_TEST_SIZE}
  model:
    residual_epochs: ${LOCAL_GRAPH_RESIDUAL_V2_EPOCHS}
    selection_metric: ${LOCAL_GRAPH_RESIDUAL_V2_SELECTION_METRIC}
    k: ${LOCAL_GRAPH_RESIDUAL_V2_K}
  resources:
    embedding_cache: time=${LOCAL_GRAPH_RESIDUAL_V2_CACHE_TIME} mem=${LOCAL_GRAPH_RESIDUAL_V2_CACHE_MEM} cpus=${LOCAL_GRAPH_RESIDUAL_V2_CACHE_CPUS}
    residual_train: time=${LOCAL_GRAPH_RESIDUAL_V2_TRAIN_TIME} mem=${LOCAL_GRAPH_RESIDUAL_V2_TRAIN_MEM} cpus=${LOCAL_GRAPH_RESIDUAL_V2_TRAIN_CPUS}
    report: time=${LOCAL_GRAPH_RESIDUAL_V2_REPORT_TIME} mem=${LOCAL_GRAPH_RESIDUAL_V2_REPORT_MEM} cpus=${LOCAL_GRAPH_RESIDUAL_V2_REPORT_CPUS}
  outputs:
    embedding_manifest: ${LOCAL_GRAPH_RESIDUAL_V2_EMBEDDING_CACHE_DIR}/baseline_embedding_manifest.json
    residual_expert_root: ${LOCAL_GRAPH_RESIDUAL_V2_EXPERT_ROOT}
    report_json: ${LOCAL_GRAPH_RESIDUAL_V2_FINAL_REPORT_DIR}/local_graph_residual_expert_v2_report.json
    metric_table: ${LOCAL_GRAPH_RESIDUAL_V2_FINAL_REPORT_DIR}/metric_table.csv
SUMMARY
