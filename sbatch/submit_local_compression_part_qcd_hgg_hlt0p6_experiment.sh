#!/usr/bin/env bash
# Submit the QCD-vs-Hgg local-compression feature-adapter ParT comparison.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${UPSTREAM_DEPENDENCY:=}"
: "${LOCAL_COMPRESSION_PART_QCD_HGG_TAG:=$(date +%Y%m%d_%H%M%S)}"
: "${LOCAL_COMPRESSION_PART_QCD_HGG_HLT_DEGRADATION_STRENGTH:=0.6}"
LOCAL_COMPRESSION_PART_QCD_HGG_HLT_TAG="${LOCAL_COMPRESSION_PART_QCD_HGG_HLT_DEGRADATION_STRENGTH//./p}"
: "${LOCAL_COMPRESSION_PART_QCD_HGG_ROOT:=${OUTPUT_ROOT}/local_compression_part_qcd_hgg_binary_hlt${LOCAL_COMPRESSION_PART_QCD_HGG_HLT_TAG}_${LOCAL_COMPRESSION_PART_QCD_HGG_TAG}}"
: "${LOCAL_COMPRESSION_PART_QCD_HGG_INPUT_ROOT:=/home/ryreu/atlas/Fresh_check/checkpoints/multiscale_subjet_part_qcd_hgg_binary_hlt0p6_qcd_hgg_hlt06_500k_full_20260628_194154}"
: "${LOCAL_COMPRESSION_PART_QCD_HGG_MANIFEST_PATH:=${LOCAL_COMPRESSION_PART_QCD_HGG_INPUT_ROOT}/binary_inputs/split_manifest.json.gz}"
: "${LOCAL_COMPRESSION_PART_QCD_HGG_HLT_CACHE_DIR:=${LOCAL_COMPRESSION_PART_QCD_HGG_INPUT_ROOT}/binary_inputs/hlt_cache}"
: "${LOCAL_COMPRESSION_PART_QCD_HGG_BASELINE_CHECKPOINT:=/home/ryreu/atlas/Fresh_check/checkpoints/local_graph_part_step10_qcd_hgg_binary_hlt0p6_20260627_075757/taggers/hlt_part_baseline/best_model_val.pt}"
: "${LOCAL_COMPRESSION_PART_QCD_HGG_VARIANTS:=hlt_part_baseline_recheck lc_mlp_delta lc_local_compression_no_context lc_context_gated lc_context_delta_no_modalities lc_random_grouping}"
: "${LOCAL_COMPRESSION_PART_QCD_HGG_SOURCE_LABEL_NAMES:=QCD Hgg}"
: "${LOCAL_COMPRESSION_PART_QCD_HGG_LABEL_FILTER_NAMES:=QCD Hgg}"

: "${LOCAL_COMPRESSION_PART_QCD_HGG_MODEL_TRAIN_SIZE:=500000}"
: "${LOCAL_COMPRESSION_PART_QCD_HGG_MODEL_VAL_SIZE:=150000}"
: "${LOCAL_COMPRESSION_PART_QCD_HGG_STACK_TRAIN_SIZE:=500000}"
: "${LOCAL_COMPRESSION_PART_QCD_HGG_STACK_VAL_SIZE:=150000}"
: "${LOCAL_COMPRESSION_PART_QCD_HGG_FINAL_TEST_SIZE:=500000}"
: "${LOCAL_COMPRESSION_PART_QCD_HGG_EPOCHS:=45}"
: "${LOCAL_COMPRESSION_PART_QCD_HGG_SELECTION_METRIC:=fpr_at_signal_eff_0p50}"
: "${LOCAL_COMPRESSION_PART_QCD_HGG_RANDOM_GROUPING_SEED:=2907}"

: "${LOCAL_COMPRESSION_PART_QCD_HGG_TRAIN_MEM:=160G}"
: "${LOCAL_COMPRESSION_PART_QCD_HGG_REPORT_MEM:=8G}"
: "${LOCAL_COMPRESSION_PART_QCD_HGG_TRAIN_TIME:=2-12:00:00}"
: "${LOCAL_COMPRESSION_PART_QCD_HGG_REPORT_TIME:=02:00:00}"
: "${LOCAL_COMPRESSION_PART_QCD_HGG_TRAIN_CPUS:=8}"
: "${LOCAL_COMPRESSION_PART_QCD_HGG_REPORT_CPUS:=2}"

export LOCAL_COMPRESSION_PART_ROOT="${LOCAL_COMPRESSION_PART_QCD_HGG_ROOT}"
export LOCAL_COMPRESSION_PART_TAGGER_ROOT="${LOCAL_COMPRESSION_PART_ROOT}/taggers"
export LOCAL_COMPRESSION_PART_FINAL_REPORT_DIR="${LOCAL_COMPRESSION_PART_ROOT}/final_report"
export LOCAL_COMPRESSION_PART_MANIFEST_PATH="${LOCAL_COMPRESSION_PART_QCD_HGG_MANIFEST_PATH}"
export LOCAL_COMPRESSION_PART_HLT_CACHE_DIR="${LOCAL_COMPRESSION_PART_QCD_HGG_HLT_CACHE_DIR}"
export LOCAL_COMPRESSION_PART_BASELINE_CHECKPOINT="${LOCAL_COMPRESSION_PART_QCD_HGG_BASELINE_CHECKPOINT}"
export LOCAL_COMPRESSION_PART_LABEL_NAMES="${LOCAL_COMPRESSION_PART_QCD_HGG_SOURCE_LABEL_NAMES}"
export LOCAL_COMPRESSION_PART_LABEL_FILTER_NAMES="${LOCAL_COMPRESSION_PART_QCD_HGG_LABEL_FILTER_NAMES}"
export LOCAL_COMPRESSION_PART_MODEL_TRAIN_SIZE="${LOCAL_COMPRESSION_PART_QCD_HGG_MODEL_TRAIN_SIZE}"
export LOCAL_COMPRESSION_PART_MODEL_VAL_SIZE="${LOCAL_COMPRESSION_PART_QCD_HGG_MODEL_VAL_SIZE}"
export LOCAL_COMPRESSION_PART_STACK_VAL_SIZE="${LOCAL_COMPRESSION_PART_QCD_HGG_STACK_VAL_SIZE}"
export LOCAL_COMPRESSION_PART_FINAL_TEST_SIZE="${LOCAL_COMPRESSION_PART_QCD_HGG_FINAL_TEST_SIZE}"
export LOCAL_COMPRESSION_PART_EPOCHS="${LOCAL_COMPRESSION_PART_QCD_HGG_EPOCHS}"
export LOCAL_COMPRESSION_PART_SELECTION_METRIC="${LOCAL_COMPRESSION_PART_QCD_HGG_SELECTION_METRIC}"
export LOCAL_COMPRESSION_PART_EXPECTED_HLT_DEGRADATION_STRENGTH="${LOCAL_COMPRESSION_PART_QCD_HGG_HLT_DEGRADATION_STRENGTH}"
export LOCAL_COMPRESSION_PART_RANDOM_GROUPING_SEED="${LOCAL_COMPRESSION_PART_QCD_HGG_RANDOM_GROUPING_SEED}"
export LOCAL_COMPRESSION_PART_CONFIRM_FINAL_TEST=1
export LOCAL_COMPRESSION_PART_REPORT_VARIANTS="${LOCAL_COMPRESSION_PART_QCD_HGG_VARIANTS}"
export LOCAL_COMPRESSION_PART_REPORT_BASELINE_VARIANT="hlt_part_baseline_recheck"
export LOCAL_COMPRESSION_PART_REPORT_PRIMARY_METRIC="${LOCAL_COMPRESSION_PART_QCD_HGG_SELECTION_METRIC}"
export LOCAL_COMPRESSION_PART_REPORT_COMPARISON_SPLIT="final_test"
export LOCAL_COMPRESSION_PART_REPORT_CONFIRM_FINAL_TEST=1
export HLT_DEGRADATION_STRENGTH="${LOCAL_COMPRESSION_PART_QCD_HGG_HLT_DEGRADATION_STRENGTH}"

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
  value="${value//hlt_part_/hlt_}"
  value="${value//lc_/lc_}"
  value="${value//_compression/comp}"
  value="${value//_modalities/modal}"
  value="${value//[^A-Za-z0-9_]/_}"
  printf '%s' "${value}"
}

fresh_split_words variant_args "${LOCAL_COMPRESSION_PART_QCD_HGG_VARIANTS}"

submitter_lock_dir="${LOCAL_COMPRESSION_PART_ROOT}/.submission_lock"
fresh_refuse_existing_dir "${LOCAL_COMPRESSION_PART_ROOT}"
fresh_claim_new_dir "${submitter_lock_dir}"
if ! fresh_is_dry_run; then
  fresh_require_file "${LOCAL_COMPRESSION_PART_QCD_HGG_MANIFEST_PATH}"
  fresh_require_file "${LOCAL_COMPRESSION_PART_QCD_HGG_BASELINE_CHECKPOINT}"
  for split in model_train model_val stack_train stack_val final_test; do
    fresh_require_file "${LOCAL_COMPRESSION_PART_QCD_HGG_HLT_CACHE_DIR}/${split}_fixed_hlt.npz"
    fresh_require_file "${LOCAL_COMPRESSION_PART_QCD_HGG_HLT_CACHE_DIR}/${split}_fixed_hlt_metadata.json"
  done
  {
    echo "created_at=$(date -Is)"
    echo "project_dir=${PROJECT_DIR}"
    echo "source_commit=$(fresh_source_commit)"
    echo "task=QCD_vs_Hgg_local_compression_part"
    echo "root=${LOCAL_COMPRESSION_PART_ROOT}"
    echo "tagger_root=${LOCAL_COMPRESSION_PART_TAGGER_ROOT}"
    echo "final_report_dir=${LOCAL_COMPRESSION_PART_FINAL_REPORT_DIR}"
    echo "source_label_names=${LOCAL_COMPRESSION_PART_QCD_HGG_SOURCE_LABEL_NAMES}"
    echo "label_filter_names=${LOCAL_COMPRESSION_PART_QCD_HGG_LABEL_FILTER_NAMES}"
    echo "reused_input_root=${LOCAL_COMPRESSION_PART_QCD_HGG_INPUT_ROOT}"
    echo "manifest=${LOCAL_COMPRESSION_PART_QCD_HGG_MANIFEST_PATH}"
    echo "hlt_cache=${LOCAL_COMPRESSION_PART_QCD_HGG_HLT_CACHE_DIR}"
    echo "baseline_checkpoint=${LOCAL_COMPRESSION_PART_QCD_HGG_BASELINE_CHECKPOINT}"
    echo "hlt_degradation_strength=${LOCAL_COMPRESSION_PART_QCD_HGG_HLT_DEGRADATION_STRENGTH}"
    echo "variants=$(fresh_join_by_space "${variant_args[@]}")"
    echo "selection_metric=${LOCAL_COMPRESSION_PART_SELECTION_METRIC}"
    echo "epochs=${LOCAL_COMPRESSION_PART_EPOCHS}"
  } > "${submitter_lock_dir}/metadata.txt"
fi

train_job_ids=()
input_dependency="${UPSTREAM_DEPENDENCY}"
for variant in "${variant_args[@]}"; do
  label="$(safe_label "${variant}")"
  mapfile -t train_args < <(
    afterok_args \
      "${input_dependency}" \
      --time="${LOCAL_COMPRESSION_PART_QCD_HGG_TRAIN_TIME}" \
      --cpus-per-task="${LOCAL_COMPRESSION_PART_QCD_HGG_TRAIN_CPUS}" \
      --mem="${LOCAL_COMPRESSION_PART_QCD_HGG_TRAIN_MEM}" \
      "${SCRIPT_DIR}/run_train_local_compression_part.sh" \
      "${variant}"
  )
  train_jid="$(submit_job "localcomp_part_${label}" "${train_args[@]}")"
  train_job_ids+=("${train_jid}")
  echo "submitted localcomp_part_${label}=${train_jid}"
done

train_dep="$(fresh_join_by_colon "${train_job_ids[@]}")"
final_report_jid="$(submit_job "localcomp_part_report" \
  --time="${LOCAL_COMPRESSION_PART_QCD_HGG_REPORT_TIME}" \
  --cpus-per-task="${LOCAL_COMPRESSION_PART_QCD_HGG_REPORT_CPUS}" \
  --mem="${LOCAL_COMPRESSION_PART_QCD_HGG_REPORT_MEM}" \
  --dependency="afterok:${train_dep}" \
  "${SCRIPT_DIR}/run_write_local_compression_part_report.sh")"
echo "submitted localcomp_part_report=${final_report_jid}"

cat <<SUMMARY
local_compression_part_qcd_hgg_hlt0p6_submission:
  task: QCD_vs_Hgg_local_compression_part
  root: ${LOCAL_COMPRESSION_PART_ROOT}
  reused_input_root: ${LOCAL_COMPRESSION_PART_QCD_HGG_INPUT_ROOT}
  manifest: ${LOCAL_COMPRESSION_PART_QCD_HGG_MANIFEST_PATH}
  hlt_cache: ${LOCAL_COMPRESSION_PART_QCD_HGG_HLT_CACHE_DIR}
  baseline_checkpoint: ${LOCAL_COMPRESSION_PART_QCD_HGG_BASELINE_CHECKPOINT}
  hlt_degradation_strength: ${LOCAL_COMPRESSION_PART_QCD_HGG_HLT_DEGRADATION_STRENGTH}
  train_job_ids: $(fresh_join_by_space "${train_job_ids[@]}")
  final_report_job_id: ${final_report_jid}
  expected_jobs:
    local_compression_train: ${#train_job_ids[@]}
    local_compression_report: 1
    total_submitted: ${submit_count}
  requested_split_caps:
    model_train: ${LOCAL_COMPRESSION_PART_MODEL_TRAIN_SIZE}
    model_val: ${LOCAL_COMPRESSION_PART_MODEL_VAL_SIZE}
    stack_train: ${LOCAL_COMPRESSION_PART_QCD_HGG_STACK_TRAIN_SIZE}
    stack_val: ${LOCAL_COMPRESSION_PART_STACK_VAL_SIZE}
    final_test: ${LOCAL_COMPRESSION_PART_FINAL_TEST_SIZE}
  model:
    variants: $(fresh_join_by_space "${variant_args[@]}")
    epochs: ${LOCAL_COMPRESSION_PART_EPOCHS}
    selection_metric: ${LOCAL_COMPRESSION_PART_SELECTION_METRIC}
    random_grouping_seed: ${LOCAL_COMPRESSION_PART_RANDOM_GROUPING_SEED}
  resources:
    train: time=${LOCAL_COMPRESSION_PART_QCD_HGG_TRAIN_TIME} mem=${LOCAL_COMPRESSION_PART_QCD_HGG_TRAIN_MEM} cpus=${LOCAL_COMPRESSION_PART_QCD_HGG_TRAIN_CPUS}
    report: time=${LOCAL_COMPRESSION_PART_QCD_HGG_REPORT_TIME} mem=${LOCAL_COMPRESSION_PART_QCD_HGG_REPORT_MEM} cpus=${LOCAL_COMPRESSION_PART_QCD_HGG_REPORT_CPUS}
  outputs:
    tagger_root: ${LOCAL_COMPRESSION_PART_TAGGER_ROOT}
    final_report_json: ${LOCAL_COMPRESSION_PART_FINAL_REPORT_DIR}/local_compression_part_final_report.json
    final_report_md: ${LOCAL_COMPRESSION_PART_FINAL_REPORT_DIR}/local_compression_part_final_report.md
    final_metric_table: ${LOCAL_COMPRESSION_PART_FINAL_REPORT_DIR}/metric_table.csv
    diagnostics: ${LOCAL_COMPRESSION_PART_FINAL_REPORT_DIR}/diagnostics.csv
    logs: ${PROJECT_DIR}/fresh_check_logs
SUMMARY
