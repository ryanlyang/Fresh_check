#!/usr/bin/env bash
# Submit the QCD-vs-Hgg Architecture-View Residual ParT comparison.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_ENV:=atlas_kd}"
export CONDA_ENV
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${UPSTREAM_DEPENDENCY:=}"
: "${ARCHITECTURE_VIEW_PART_QCD_HGG_TAG:=$(date +%Y%m%d_%H%M%S)}"
: "${ARCHITECTURE_VIEW_PART_QCD_HGG_HLT_DEGRADATION_STRENGTH:=0.6}"
ARCHITECTURE_VIEW_PART_QCD_HGG_HLT_TAG="${ARCHITECTURE_VIEW_PART_QCD_HGG_HLT_DEGRADATION_STRENGTH//./p}"
: "${ARCHITECTURE_VIEW_PART_QCD_HGG_ROOT:=${OUTPUT_ROOT}/architecture_view_part_qcd_hgg_binary_hlt${ARCHITECTURE_VIEW_PART_QCD_HGG_HLT_TAG}_${ARCHITECTURE_VIEW_PART_QCD_HGG_TAG}}"
: "${ARCHITECTURE_VIEW_PART_QCD_HGG_INPUT_ROOT:=/home/ryreu/atlas/Fresh_check/checkpoints/multiscale_subjet_part_qcd_hgg_binary_hlt0p6_qcd_hgg_hlt06_500k_full_20260628_194154}"
: "${ARCHITECTURE_VIEW_PART_QCD_HGG_MANIFEST_PATH:=${ARCHITECTURE_VIEW_PART_QCD_HGG_INPUT_ROOT}/binary_inputs/split_manifest.json.gz}"
: "${ARCHITECTURE_VIEW_PART_QCD_HGG_HLT_CACHE_DIR:=${ARCHITECTURE_VIEW_PART_QCD_HGG_INPUT_ROOT}/binary_inputs/hlt_cache}"
: "${ARCHITECTURE_VIEW_PART_QCD_HGG_BASELINE_CHECKPOINT:=/home/ryreu/atlas/Fresh_check/checkpoints/local_graph_part_step10_qcd_hgg_binary_hlt0p6_20260627_075757/taggers/hlt_part_baseline/best_model_val.pt}"
: "${ARCHITECTURE_VIEW_PART_QCD_HGG_VARIANTS:=av_baseline_recheck av_all_views av_pn_only av_pfn_only av_pcnn_only av_random_view_control av_context_mlp_control}"
: "${ARCHITECTURE_VIEW_PART_QCD_HGG_SOURCE_LABEL_NAMES:=QCD Hgg}"
: "${ARCHITECTURE_VIEW_PART_QCD_HGG_LABEL_FILTER_NAMES:=QCD Hgg}"

: "${ARCHITECTURE_VIEW_PART_QCD_HGG_MODEL_TRAIN_SIZE:=500000}"
: "${ARCHITECTURE_VIEW_PART_QCD_HGG_MODEL_VAL_SIZE:=150000}"
: "${ARCHITECTURE_VIEW_PART_QCD_HGG_STACK_VAL_SIZE:=150000}"
: "${ARCHITECTURE_VIEW_PART_QCD_HGG_FINAL_TEST_SIZE:=500000}"
: "${ARCHITECTURE_VIEW_PART_QCD_HGG_EPOCHS:=45}"
: "${ARCHITECTURE_VIEW_PART_QCD_HGG_SELECTION_METRIC:=fpr_at_signal_eff_0p50}"
: "${ARCHITECTURE_VIEW_PART_QCD_HGG_RANDOM_CONTROL_SEED:=2907}"
: "${ARCHITECTURE_VIEW_PART_QCD_HGG_REQUIRE_BASELINE_SPLIT_MANIFEST_HASH:=1}"

: "${ARCHITECTURE_VIEW_PART_QCD_HGG_TRAIN_MEM:=160G}"
: "${ARCHITECTURE_VIEW_PART_QCD_HGG_REPORT_MEM:=8G}"
: "${ARCHITECTURE_VIEW_PART_QCD_HGG_TRAIN_TIME:=2-12:00:00}"
: "${ARCHITECTURE_VIEW_PART_QCD_HGG_REPORT_TIME:=02:00:00}"
: "${ARCHITECTURE_VIEW_PART_QCD_HGG_TRAIN_CPUS:=8}"
: "${ARCHITECTURE_VIEW_PART_QCD_HGG_REPORT_CPUS:=2}"

export ARCHITECTURE_VIEW_PART_ROOT="${ARCHITECTURE_VIEW_PART_QCD_HGG_ROOT}"
export ARCHITECTURE_VIEW_PART_TAGGER_ROOT="${ARCHITECTURE_VIEW_PART_ROOT}/taggers"
export ARCHITECTURE_VIEW_PART_FINAL_REPORT_DIR="${ARCHITECTURE_VIEW_PART_ROOT}/final_report"
export ARCHITECTURE_VIEW_PART_MANIFEST_PATH="${ARCHITECTURE_VIEW_PART_QCD_HGG_MANIFEST_PATH}"
export ARCHITECTURE_VIEW_PART_HLT_CACHE_DIR="${ARCHITECTURE_VIEW_PART_QCD_HGG_HLT_CACHE_DIR}"
export ARCHITECTURE_VIEW_PART_BASELINE_CHECKPOINT="${ARCHITECTURE_VIEW_PART_QCD_HGG_BASELINE_CHECKPOINT}"
export ARCHITECTURE_VIEW_PART_LABEL_NAMES="${ARCHITECTURE_VIEW_PART_QCD_HGG_SOURCE_LABEL_NAMES}"
export ARCHITECTURE_VIEW_PART_LABEL_FILTER_NAMES="${ARCHITECTURE_VIEW_PART_QCD_HGG_LABEL_FILTER_NAMES}"
export ARCHITECTURE_VIEW_PART_MODEL_TRAIN_SIZE="${ARCHITECTURE_VIEW_PART_QCD_HGG_MODEL_TRAIN_SIZE}"
export ARCHITECTURE_VIEW_PART_MODEL_VAL_SIZE="${ARCHITECTURE_VIEW_PART_QCD_HGG_MODEL_VAL_SIZE}"
export ARCHITECTURE_VIEW_PART_STACK_VAL_SIZE="${ARCHITECTURE_VIEW_PART_QCD_HGG_STACK_VAL_SIZE}"
export ARCHITECTURE_VIEW_PART_FINAL_TEST_SIZE="${ARCHITECTURE_VIEW_PART_QCD_HGG_FINAL_TEST_SIZE}"
export ARCHITECTURE_VIEW_PART_EPOCHS="${ARCHITECTURE_VIEW_PART_QCD_HGG_EPOCHS}"
export ARCHITECTURE_VIEW_PART_SELECTION_METRIC="${ARCHITECTURE_VIEW_PART_QCD_HGG_SELECTION_METRIC}"
export ARCHITECTURE_VIEW_PART_EXPECTED_HLT_DEGRADATION_STRENGTH="${ARCHITECTURE_VIEW_PART_QCD_HGG_HLT_DEGRADATION_STRENGTH}"
export ARCHITECTURE_VIEW_PART_RANDOM_CONTROL_SEED="${ARCHITECTURE_VIEW_PART_QCD_HGG_RANDOM_CONTROL_SEED}"
export ARCHITECTURE_VIEW_PART_REQUIRE_BASELINE_SPLIT_MANIFEST_HASH="${ARCHITECTURE_VIEW_PART_QCD_HGG_REQUIRE_BASELINE_SPLIT_MANIFEST_HASH}"
export ARCHITECTURE_VIEW_PART_CONFIRM_FINAL_TEST=1
export ARCHITECTURE_VIEW_PART_REPORT_VARIANTS="${ARCHITECTURE_VIEW_PART_QCD_HGG_VARIANTS}"
export ARCHITECTURE_VIEW_PART_REPORT_BASELINE_VARIANT="av_baseline_recheck"
export ARCHITECTURE_VIEW_PART_REPORT_PRIMARY_METRIC="${ARCHITECTURE_VIEW_PART_QCD_HGG_SELECTION_METRIC}"
export ARCHITECTURE_VIEW_PART_REPORT_COMPARISON_SPLIT="final_test"
export ARCHITECTURE_VIEW_PART_REPORT_CONFIRM_FINAL_TEST=1
export HLT_DEGRADATION_STRENGTH="${ARCHITECTURE_VIEW_PART_QCD_HGG_HLT_DEGRADATION_STRENGTH}"

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
  value="${value//av_/av_}"
  value="${value//_control/ctrl}"
  value="${value//[^A-Za-z0-9_]/_}"
  printf '%s' "${value}"
}

fresh_split_words variant_args "${ARCHITECTURE_VIEW_PART_QCD_HGG_VARIANTS}"

submitter_lock_dir="${ARCHITECTURE_VIEW_PART_ROOT}/.submission_lock"
fresh_refuse_existing_dir "${ARCHITECTURE_VIEW_PART_ROOT}"
fresh_claim_new_dir "${submitter_lock_dir}"
if ! fresh_is_dry_run; then
  fresh_require_file "${ARCHITECTURE_VIEW_PART_QCD_HGG_MANIFEST_PATH}"
  fresh_require_file "${ARCHITECTURE_VIEW_PART_QCD_HGG_BASELINE_CHECKPOINT}"
  for split in model_train model_val stack_train stack_val final_test; do
    fresh_require_file "${ARCHITECTURE_VIEW_PART_QCD_HGG_HLT_CACHE_DIR}/${split}_fixed_hlt.npz"
    fresh_require_file "${ARCHITECTURE_VIEW_PART_QCD_HGG_HLT_CACHE_DIR}/${split}_fixed_hlt_metadata.json"
  done
  {
    echo "created_at=$(date -Is)"
    echo "project_dir=${PROJECT_DIR}"
    echo "source_commit=$(fresh_source_commit)"
    echo "task=QCD_vs_Hgg_architecture_view_part"
    echo "root=${ARCHITECTURE_VIEW_PART_ROOT}"
    echo "tagger_root=${ARCHITECTURE_VIEW_PART_TAGGER_ROOT}"
    echo "final_report_dir=${ARCHITECTURE_VIEW_PART_FINAL_REPORT_DIR}"
    echo "source_label_names=${ARCHITECTURE_VIEW_PART_QCD_HGG_SOURCE_LABEL_NAMES}"
    echo "label_filter_names=${ARCHITECTURE_VIEW_PART_QCD_HGG_LABEL_FILTER_NAMES}"
    echo "reused_input_root=${ARCHITECTURE_VIEW_PART_QCD_HGG_INPUT_ROOT}"
    echo "manifest=${ARCHITECTURE_VIEW_PART_QCD_HGG_MANIFEST_PATH}"
    echo "hlt_cache=${ARCHITECTURE_VIEW_PART_QCD_HGG_HLT_CACHE_DIR}"
    echo "baseline_checkpoint=${ARCHITECTURE_VIEW_PART_QCD_HGG_BASELINE_CHECKPOINT}"
    echo "hlt_degradation_strength=${ARCHITECTURE_VIEW_PART_QCD_HGG_HLT_DEGRADATION_STRENGTH}"
    echo "variants=$(fresh_join_by_space "${variant_args[@]}")"
    echo "selection_metric=${ARCHITECTURE_VIEW_PART_SELECTION_METRIC}"
    echo "epochs=${ARCHITECTURE_VIEW_PART_EPOCHS}"
    echo "require_baseline_split_manifest_hash=${ARCHITECTURE_VIEW_PART_REQUIRE_BASELINE_SPLIT_MANIFEST_HASH}"
  } > "${submitter_lock_dir}/metadata.txt"
fi

train_job_ids=()
input_dependency="${UPSTREAM_DEPENDENCY}"
for variant in "${variant_args[@]}"; do
  label="$(safe_label "${variant}")"
  mapfile -t train_args < <(
    afterok_args \
      "${input_dependency}" \
      --time="${ARCHITECTURE_VIEW_PART_QCD_HGG_TRAIN_TIME}" \
      --cpus-per-task="${ARCHITECTURE_VIEW_PART_QCD_HGG_TRAIN_CPUS}" \
      --mem="${ARCHITECTURE_VIEW_PART_QCD_HGG_TRAIN_MEM}" \
      "${SCRIPT_DIR}/run_train_architecture_view_part.sh" \
      "${variant}"
  )
  train_jid="$(submit_job "archview_part_${label}" "${train_args[@]}")"
  train_job_ids+=("${train_jid}")
  echo "submitted archview_part_${label}=${train_jid}"
done

train_dep="$(fresh_join_by_colon "${train_job_ids[@]}")"
final_report_jid="$(submit_job "archview_part_report" \
  --time="${ARCHITECTURE_VIEW_PART_QCD_HGG_REPORT_TIME}" \
  --cpus-per-task="${ARCHITECTURE_VIEW_PART_QCD_HGG_REPORT_CPUS}" \
  --mem="${ARCHITECTURE_VIEW_PART_QCD_HGG_REPORT_MEM}" \
  --dependency="afterok:${train_dep}" \
  "${SCRIPT_DIR}/run_write_architecture_view_part_report.sh")"
echo "submitted archview_part_report=${final_report_jid}"

cat <<SUMMARY
architecture_view_part_qcd_hgg_hlt0p6_submission:
  task: QCD_vs_Hgg_architecture_view_part
  root: ${ARCHITECTURE_VIEW_PART_ROOT}
  reused_input_root: ${ARCHITECTURE_VIEW_PART_QCD_HGG_INPUT_ROOT}
  manifest: ${ARCHITECTURE_VIEW_PART_QCD_HGG_MANIFEST_PATH}
  hlt_cache: ${ARCHITECTURE_VIEW_PART_QCD_HGG_HLT_CACHE_DIR}
  baseline_checkpoint: ${ARCHITECTURE_VIEW_PART_QCD_HGG_BASELINE_CHECKPOINT}
  hlt_degradation_strength: ${ARCHITECTURE_VIEW_PART_QCD_HGG_HLT_DEGRADATION_STRENGTH}
  train_job_ids: $(fresh_join_by_space "${train_job_ids[@]}")
  final_report_job_id: ${final_report_jid}
  expected_jobs:
    architecture_view_train: ${#train_job_ids[@]}
    architecture_view_report: 1
    total_submitted: ${submit_count}
  requested_split_caps:
    model_train: ${ARCHITECTURE_VIEW_PART_MODEL_TRAIN_SIZE}
    model_val: ${ARCHITECTURE_VIEW_PART_MODEL_VAL_SIZE}
    stack_val: ${ARCHITECTURE_VIEW_PART_STACK_VAL_SIZE}
    final_test: ${ARCHITECTURE_VIEW_PART_FINAL_TEST_SIZE}
  model:
    variants: $(fresh_join_by_space "${variant_args[@]}")
    epochs: ${ARCHITECTURE_VIEW_PART_EPOCHS}
    selection_metric: ${ARCHITECTURE_VIEW_PART_SELECTION_METRIC}
    random_control_seed: ${ARCHITECTURE_VIEW_PART_RANDOM_CONTROL_SEED}
  resources:
    train: time=${ARCHITECTURE_VIEW_PART_QCD_HGG_TRAIN_TIME} mem=${ARCHITECTURE_VIEW_PART_QCD_HGG_TRAIN_MEM} cpus=${ARCHITECTURE_VIEW_PART_QCD_HGG_TRAIN_CPUS}
    report: time=${ARCHITECTURE_VIEW_PART_QCD_HGG_REPORT_TIME} mem=${ARCHITECTURE_VIEW_PART_QCD_HGG_REPORT_MEM} cpus=${ARCHITECTURE_VIEW_PART_QCD_HGG_REPORT_CPUS}
  outputs:
    tagger_root: ${ARCHITECTURE_VIEW_PART_TAGGER_ROOT}
    final_report_json: ${ARCHITECTURE_VIEW_PART_FINAL_REPORT_DIR}/architecture_view_part_final_report.json
    final_report_md: ${ARCHITECTURE_VIEW_PART_FINAL_REPORT_DIR}/architecture_view_part_final_report.md
    final_metric_table: ${ARCHITECTURE_VIEW_PART_FINAL_REPORT_DIR}/metric_table.csv
    diagnostics: ${ARCHITECTURE_VIEW_PART_FINAL_REPORT_DIR}/diagnostics.csv
    logs: ${PROJECT_DIR}/fresh_check_logs
SUMMARY
