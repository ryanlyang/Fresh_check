#!/usr/bin/env bash
# Serious QCD-vs-Hgg HLT0.6 residual-expert V2 ablation suite on the 3M/1M/1M cache.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

fresh_prepare_submitter

: "${LOCAL_GRAPH_RESIDUAL_V2_SERIOUS_INPUT_ROOT:=/home/ryreu/atlas/Fresh_check/checkpoints/multiscale_subjet_part_qcd_hgg_binary_hlt0p6_qcd_hgg_hlt06_3m1m1m_full_20260628_194154}"
: "${LOCAL_GRAPH_RESIDUAL_V2_SERIOUS_LOCAL_GRAPH_ROOT:=/home/ryreu/atlas/Fresh_check/checkpoints/local_graph_part_qcd_hgg_hlt0p6_3m1m1m_20260629_015555}"
: "${LOCAL_GRAPH_RESIDUAL_V2_SERIOUS_BASELINE_CHECKPOINT:=${LOCAL_GRAPH_RESIDUAL_V2_SERIOUS_LOCAL_GRAPH_ROOT}/taggers/hlt_part_baseline/best_model_val.pt}"
: "${LOCAL_GRAPH_RESIDUAL_V2_ALLOW_BASELINE_SPLIT_MISMATCH:=0}"

: "${LOCAL_GRAPH_RESIDUAL_V2_EXISTING_INPUT_ROOT:=${LOCAL_GRAPH_RESIDUAL_V2_SERIOUS_INPUT_ROOT}}"
: "${LOCAL_GRAPH_RESIDUAL_V2_HLT_CACHE_DIR:=${LOCAL_GRAPH_RESIDUAL_V2_EXISTING_INPUT_ROOT}/binary_inputs/hlt_cache}"
: "${LOCAL_GRAPH_RESIDUAL_V2_BASELINE_CHECKPOINT:=${LOCAL_GRAPH_RESIDUAL_V2_SERIOUS_BASELINE_CHECKPOINT}}"
: "${LOCAL_GRAPH_RESIDUAL_V2_TAG:=$(date +%Y%m%d_%H%M%S)}"
: "${LOCAL_GRAPH_RESIDUAL_V2_ROOT:=${OUTPUT_ROOT}/local_graph_residual_expert_v2_qcd_hgg_binary_hlt0p6_3m1m1m_ablation_${LOCAL_GRAPH_RESIDUAL_V2_TAG}}"
: "${LOCAL_GRAPH_RESIDUAL_V2_EMBEDDING_CACHE_DIR:=${LOCAL_GRAPH_RESIDUAL_V2_ROOT}/baseline_embeddings}"
: "${LOCAL_GRAPH_RESIDUAL_V2_EXPERT_ROOT:=${LOCAL_GRAPH_RESIDUAL_V2_ROOT}/residual_experts}"
: "${LOCAL_GRAPH_RESIDUAL_V2_FINAL_REPORT_DIR:=${LOCAL_GRAPH_RESIDUAL_V2_ROOT}/final_report}"

: "${LOCAL_GRAPH_RESIDUAL_V2_HLT_DEGRADATION_STRENGTH:=0.6}"
: "${LOCAL_GRAPH_RESIDUAL_V2_MODEL_TRAIN_SIZE:=3000000}"
: "${LOCAL_GRAPH_RESIDUAL_V2_MODEL_VAL_SIZE:=1000000}"
: "${LOCAL_GRAPH_RESIDUAL_V2_STACK_TRAIN_SIZE:=3000000}"
: "${LOCAL_GRAPH_RESIDUAL_V2_STACK_VAL_SIZE:=1000000}"
: "${LOCAL_GRAPH_RESIDUAL_V2_FINAL_TEST_SIZE:=1000000}"
: "${LOCAL_GRAPH_RESIDUAL_V2_EPOCHS:=30}"
: "${LOCAL_GRAPH_RESIDUAL_V2_SELECTION_METRIC:=fpr_at_signal_eff_0p50}"
: "${LOCAL_GRAPH_RESIDUAL_V2_K:=16}"

: "${LOCAL_GRAPH_RESIDUAL_V2_CACHE_TIME:=1-12:00:00}"
: "${LOCAL_GRAPH_RESIDUAL_V2_CACHE_MEM:=192G}"
: "${LOCAL_GRAPH_RESIDUAL_V2_CACHE_CPUS:=8}"
: "${LOCAL_GRAPH_RESIDUAL_V2_TRAIN_TIME:=5-00:00:00}"
: "${LOCAL_GRAPH_RESIDUAL_V2_TRAIN_MEM:=192G}"
: "${LOCAL_GRAPH_RESIDUAL_V2_TRAIN_CPUS:=8}"
: "${LOCAL_GRAPH_RESIDUAL_V2_REPORT_TIME:=1-00:00:00}"
: "${LOCAL_GRAPH_RESIDUAL_V2_REPORT_MEM:=96G}"
: "${LOCAL_GRAPH_RESIDUAL_V2_REPORT_CPUS:=4}"

: "${LOCAL_GRAPH_RESIDUAL_V2_ABLATION_INCLUDE_MAINLINE:=1}"
: "${LOCAL_GRAPH_RESIDUAL_V2_ABLATION_INCLUDE_INPUT_CONTROLS:=1}"
: "${LOCAL_GRAPH_RESIDUAL_V2_ABLATION_INCLUDE_SHUFFLED_CONDITION:=1}"
: "${LOCAL_GRAPH_RESIDUAL_V2_ABLATION_INCLUDE_SHUFFLED_LABELS:=1}"

if [[ -z "${LOCAL_GRAPH_RESIDUAL_V2_STANDALONE_REPORT_PATH:-}" ]]; then
  candidate="${LOCAL_GRAPH_RESIDUAL_V2_SERIOUS_LOCAL_GRAPH_ROOT}/final_report/local_graph_part_report.json"
  if [[ -f "${candidate}" ]]; then
    LOCAL_GRAPH_RESIDUAL_V2_STANDALONE_REPORT_PATH="${candidate}"
  fi
fi
if [[ -z "${LOCAL_GRAPH_RESIDUAL_V2_SCORE_FUSION_REPORT_PATH:-}" ]]; then
  shopt -s nullglob
  score_fusion_candidates=("${LOCAL_GRAPH_RESIDUAL_V2_SERIOUS_LOCAL_GRAPH_ROOT}"/score_fusion_*/fusion_report.json)
  shopt -u nullglob
  if [[ ${#score_fusion_candidates[@]} -gt 0 ]]; then
    IFS=$'\n' sorted_score_fusion_candidates=($(printf '%s\n' "${score_fusion_candidates[@]}" | sort))
    unset IFS
    LOCAL_GRAPH_RESIDUAL_V2_SCORE_FUSION_REPORT_PATH="${sorted_score_fusion_candidates[$((${#sorted_score_fusion_candidates[@]} - 1))]}"
  fi
fi

fresh_require_file "${LOCAL_GRAPH_RESIDUAL_V2_BASELINE_CHECKPOINT}"
baseline_dir="$(dirname "${LOCAL_GRAPH_RESIDUAL_V2_BASELINE_CHECKPOINT}")"
fresh_require_file "${baseline_dir}/run_report.json"
if [[ "${LOCAL_GRAPH_RESIDUAL_V2_ALLOW_BASELINE_SPLIT_MISMATCH}" != "1" ]]; then
  if [[ "${baseline_dir}" != *"3m1m1m"* ]] && ! grep -Fq "3m1m1m" "${baseline_dir}/run_report.json"; then
    cat >&2 <<ERROR
LOCAL_GRAPH_RESIDUAL_V2_BASELINE_CHECKPOINT does not look like a 3M/1M/1M baseline:
  checkpoint: ${LOCAL_GRAPH_RESIDUAL_V2_BASELINE_CHECKPOINT}
  run_report: ${baseline_dir}/run_report.json

Set LOCAL_GRAPH_RESIDUAL_V2_SERIOUS_BASELINE_CHECKPOINT to the 3M hlt_part_baseline/best_model_val.pt,
or set LOCAL_GRAPH_RESIDUAL_V2_ALLOW_BASELINE_SPLIT_MISMATCH=1 only for deliberate debugging.
ERROR
    exit 2
  fi
fi
for split in model_train model_val stack_train stack_val final_test; do
  fresh_require_file "${LOCAL_GRAPH_RESIDUAL_V2_HLT_CACHE_DIR}/${split}_fixed_hlt.npz"
  fresh_require_file "${LOCAL_GRAPH_RESIDUAL_V2_HLT_CACHE_DIR}/${split}_fixed_hlt_metadata.json"
done

ablation_specs=()
if fresh_bool_enabled "${LOCAL_GRAPH_RESIDUAL_V2_ABLATION_INCLUDE_MAINLINE}"; then
  ablation_specs+=(
    "full_a|A|full|normal|normal"
    "full_c|C|full|normal|normal"
    "full_d|D|full|normal|normal"
  )
fi
if fresh_bool_enabled "${LOCAL_GRAPH_RESIDUAL_V2_ABLATION_INCLUDE_INPUT_CONTROLS}"; then
  ablation_specs+=(
    "embedding_only_d|D|embedding_only|normal|normal"
    "local_only_d|D|local_only|normal|normal"
  )
fi
if fresh_bool_enabled "${LOCAL_GRAPH_RESIDUAL_V2_ABLATION_INCLUDE_SHUFFLED_CONDITION}"; then
  ablation_specs+=("condition_shuffled_d|D|full|shuffled|normal")
fi
if fresh_bool_enabled "${LOCAL_GRAPH_RESIDUAL_V2_ABLATION_INCLUDE_SHUFFLED_LABELS}"; then
  ablation_specs+=("label_shuffled_d|D|full|normal|shuffled")
fi
if [[ ${#ablation_specs[@]} -eq 0 ]]; then
  echo "No V2 ablation specs selected." >&2
  exit 2
fi

export LOCAL_GRAPH_RESIDUAL_V2_ROOT
export LOCAL_GRAPH_RESIDUAL_V2_HLT_CACHE_DIR
export LOCAL_GRAPH_RESIDUAL_V2_BASELINE_CHECKPOINT
export LOCAL_GRAPH_RESIDUAL_V2_EMBEDDING_CACHE_DIR
export LOCAL_GRAPH_RESIDUAL_V2_EXPERT_ROOT
export LOCAL_GRAPH_RESIDUAL_V2_HLT_DEGRADATION_STRENGTH
export LOCAL_GRAPH_RESIDUAL_V2_EXPECTED_HLT_DEGRADATION_STRENGTH="${LOCAL_GRAPH_RESIDUAL_V2_HLT_DEGRADATION_STRENGTH}"
export HLT_DEGRADATION_STRENGTH="${LOCAL_GRAPH_RESIDUAL_V2_HLT_DEGRADATION_STRENGTH}"
export LOCAL_GRAPH_RESIDUAL_V2_MODEL_TRAIN_SIZE
export LOCAL_GRAPH_RESIDUAL_V2_MODEL_VAL_SIZE
export LOCAL_GRAPH_RESIDUAL_V2_STACK_TRAIN_SIZE
export LOCAL_GRAPH_RESIDUAL_V2_STACK_VAL_SIZE
export LOCAL_GRAPH_RESIDUAL_V2_FINAL_TEST_SIZE
export LOCAL_GRAPH_RESIDUAL_V2_EPOCHS
export LOCAL_GRAPH_RESIDUAL_V2_SELECTION_METRIC
export LOCAL_GRAPH_RESIDUAL_V2_K
export LOCAL_GRAPH_RESIDUAL_V2_CACHE_TIME
export LOCAL_GRAPH_RESIDUAL_V2_CACHE_MEM
export LOCAL_GRAPH_RESIDUAL_V2_CACHE_CPUS
export LOCAL_GRAPH_RESIDUAL_V2_TRAIN_TIME
export LOCAL_GRAPH_RESIDUAL_V2_TRAIN_MEM
export LOCAL_GRAPH_RESIDUAL_V2_TRAIN_CPUS
export LOCAL_GRAPH_RESIDUAL_V2_REPORT_TIME
export LOCAL_GRAPH_RESIDUAL_V2_REPORT_MEM
export LOCAL_GRAPH_RESIDUAL_V2_REPORT_CPUS
export LOCAL_GRAPH_RESIDUAL_V2_FINAL_REPORT_DIR
export LOCAL_GRAPH_RESIDUAL_V2_STANDALONE_REPORT_PATH="${LOCAL_GRAPH_RESIDUAL_V2_STANDALONE_REPORT_PATH:-}"
export LOCAL_GRAPH_RESIDUAL_V2_SCORE_FUSION_REPORT_PATH="${LOCAL_GRAPH_RESIDUAL_V2_SCORE_FUSION_REPORT_PATH:-}"
export LOCAL_GRAPH_RESIDUAL_V2_REPORT_COMPARISON_SPLIT="${LOCAL_GRAPH_RESIDUAL_V2_REPORT_COMPARISON_SPLIT:-final_test}"
export LOCAL_GRAPH_RESIDUAL_V2_REPORT_CONFIRM_FINAL_TEST="${LOCAL_GRAPH_RESIDUAL_V2_REPORT_CONFIRM_FINAL_TEST:-1}"

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

submitter_lock_dir="${LOCAL_GRAPH_RESIDUAL_V2_ROOT}/.submission_lock"
fresh_refuse_existing_dir "${LOCAL_GRAPH_RESIDUAL_V2_ROOT}"
fresh_claim_new_dir "${submitter_lock_dir}"
if ! fresh_is_dry_run; then
  {
    echo "created_at=$(date -Is)"
    echo "project_dir=${PROJECT_DIR}"
    echo "source_commit=$(fresh_source_commit)"
    echo "task=QCD_vs_Hgg_local_graph_residual_expert_v2_ablation_suite"
    echo "root=${LOCAL_GRAPH_RESIDUAL_V2_ROOT}"
    echo "hlt_degradation_strength=${LOCAL_GRAPH_RESIDUAL_V2_HLT_DEGRADATION_STRENGTH}"
    echo "hlt_cache=${LOCAL_GRAPH_RESIDUAL_V2_HLT_CACHE_DIR}"
    echo "baseline_checkpoint=${LOCAL_GRAPH_RESIDUAL_V2_BASELINE_CHECKPOINT}"
    echo "embedding_cache=${LOCAL_GRAPH_RESIDUAL_V2_EMBEDDING_CACHE_DIR}"
    printf 'ablation_specs=%s\n' "$(fresh_join_by_space "${ablation_specs[@]}")"
    echo "selection_metric=${LOCAL_GRAPH_RESIDUAL_V2_SELECTION_METRIC}"
  } > "${submitter_lock_dir}/metadata.txt"
fi

mapfile -t cache_args < <(
  afterok_args \
    "${UPSTREAM_DEPENDENCY:-}" \
    --time="${LOCAL_GRAPH_RESIDUAL_V2_CACHE_TIME}" \
    --cpus-per-task="${LOCAL_GRAPH_RESIDUAL_V2_CACHE_CPUS}" \
    --mem="${LOCAL_GRAPH_RESIDUAL_V2_CACHE_MEM}" \
    "${SCRIPT_DIR}/run_cache_local_graph_residual_v2_embeddings.sh"
)
cache_jid="$(submit_job "localgraph_residual_v2_embeddings" "${cache_args[@]}")"
echo "submitted localgraph_residual_v2_embeddings=${cache_jid}"

residual_job_ids=()
residual_output_names=()
for spec in "${ablation_specs[@]}"; do
  IFS='|' read -r output_name loss_mode residual_input_mode condition_control_mode label_control_mode <<< "${spec}"
  export LOCAL_GRAPH_RESIDUAL_V2_OUTPUT_NAME="${output_name}"
  export LOCAL_GRAPH_RESIDUAL_V2_RESIDUAL_INPUT_MODE="${residual_input_mode}"
  export LOCAL_GRAPH_RESIDUAL_V2_CONDITION_CONTROL_MODE="${condition_control_mode}"
  export LOCAL_GRAPH_RESIDUAL_V2_LABEL_CONTROL_MODE="${label_control_mode}"
  mapfile -t train_args < <(
    afterok_args \
      "${cache_jid}" \
      --time="${LOCAL_GRAPH_RESIDUAL_V2_TRAIN_TIME}" \
      --cpus-per-task="${LOCAL_GRAPH_RESIDUAL_V2_TRAIN_CPUS}" \
      --mem="${LOCAL_GRAPH_RESIDUAL_V2_TRAIN_MEM}" \
      "${SCRIPT_DIR}/run_train_local_graph_residual_expert_v2.sh" \
      "${loss_mode}"
  )
  train_jid="$(submit_job "localgraph_residual_v2_${output_name}" "${train_args[@]}")"
  residual_job_ids+=("${train_jid}")
  residual_output_names+=("${output_name}")
  echo "submitted localgraph_residual_v2_${output_name}=${train_jid}"
done

residual_dep="$(fresh_join_by_colon "${residual_job_ids[@]}")"
LOCAL_GRAPH_RESIDUAL_V2_REPORT_VARIANTS="$(fresh_join_by_space "${residual_output_names[@]}")"
export LOCAL_GRAPH_RESIDUAL_V2_REPORT_VARIANTS
report_jid="$(submit_job "localgraph_residual_v2_ablation_report" \
  --time="${LOCAL_GRAPH_RESIDUAL_V2_REPORT_TIME}" \
  --cpus-per-task="${LOCAL_GRAPH_RESIDUAL_V2_REPORT_CPUS}" \
  --mem="${LOCAL_GRAPH_RESIDUAL_V2_REPORT_MEM}" \
  --dependency="afterok:${residual_dep}" \
  "${SCRIPT_DIR}/run_write_local_graph_residual_expert_v2_report.sh")"
echo "submitted localgraph_residual_v2_ablation_report=${report_jid}"

cat <<SUMMARY
local_graph_residual_expert_v2_ablation_suite_submission:
  task: QCD_vs_Hgg_local_graph_residual_expert_v2_ablation_suite
  root: ${LOCAL_GRAPH_RESIDUAL_V2_ROOT}
  hlt_degradation_strength: ${LOCAL_GRAPH_RESIDUAL_V2_HLT_DEGRADATION_STRENGTH}
  inputs:
    hlt_cache: ${LOCAL_GRAPH_RESIDUAL_V2_HLT_CACHE_DIR}
    baseline_checkpoint: ${LOCAL_GRAPH_RESIDUAL_V2_BASELINE_CHECKPOINT}
    standalone_report: ${LOCAL_GRAPH_RESIDUAL_V2_STANDALONE_REPORT_PATH:-none}
    score_fusion_report: ${LOCAL_GRAPH_RESIDUAL_V2_SCORE_FUSION_REPORT_PATH:-none}
  embedding_cache:
    output_dir: ${LOCAL_GRAPH_RESIDUAL_V2_EMBEDDING_CACHE_DIR}
    job_id: ${cache_jid}
  residual_experts:
    specs: $(fresh_join_by_space "${ablation_specs[@]}")
    output_names: $(fresh_join_by_space "${residual_output_names[@]}")
    job_ids: $(fresh_join_by_space "${residual_job_ids[@]}")
    dependency: ${residual_dep}
  expected_jobs:
    embedding_cache: 1
    residual_experts: ${#residual_job_ids[@]}
    report: 1
    total_submitted: ${submit_count}
  requested_split_caps:
    model_train: ${LOCAL_GRAPH_RESIDUAL_V2_MODEL_TRAIN_SIZE}
    model_val: ${LOCAL_GRAPH_RESIDUAL_V2_MODEL_VAL_SIZE}
    stack_train: ${LOCAL_GRAPH_RESIDUAL_V2_STACK_TRAIN_SIZE}
    stack_val: ${LOCAL_GRAPH_RESIDUAL_V2_STACK_VAL_SIZE}
    final_test: ${LOCAL_GRAPH_RESIDUAL_V2_FINAL_TEST_SIZE}
  outputs:
    embedding_manifest: ${LOCAL_GRAPH_RESIDUAL_V2_EMBEDDING_CACHE_DIR}/baseline_embedding_manifest.json
    residual_expert_root: ${LOCAL_GRAPH_RESIDUAL_V2_EXPERT_ROOT}
    report_json: ${LOCAL_GRAPH_RESIDUAL_V2_FINAL_REPORT_DIR}/local_graph_residual_expert_v2_report.json
    metric_table: ${LOCAL_GRAPH_RESIDUAL_V2_FINAL_REPORT_DIR}/metric_table.csv
SUMMARY
