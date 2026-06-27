#!/usr/bin/env bash
# Submit the Step 11 reliability-gated dual-view ParT 500k QCD/Hgg HLT0.6 run.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${UPSTREAM_DEPENDENCY:=}"
: "${DUALVIEW_PART_500K_TAG:=$(date +%Y%m%d_%H%M%S)}"
: "${DUALVIEW_PART_500K_ROOT:=${OUTPUT_ROOT}/dualview_part_qcd_hgg_binary_hlt0p6_true500k_${DUALVIEW_PART_500K_TAG}}"
: "${DUALVIEW_PART_500K_SOURCE_ROOT:=${DUALVIEW_PART_EXPERIMENT_DIR:-${DUALVIEW_PART_ROOT}}}"
: "${DUALVIEW_PART_500K_HLT_CACHE_DIR:=${DUALVIEW_PART_HLT_CACHE_DIR}}"
: "${DUALVIEW_PART_500K_PN_RECONSTRUCTED_VIEW_DIR:=${DUALVIEW_PART_PN_RECONSTRUCTED_VIEW_DIR}}"
: "${DUALVIEW_PART_500K_HLT_ANCHOR_CHECKPOINT:=${DUALVIEW_PART_HLT_ANCHOR_CHECKPOINT}}"
: "${DUALVIEW_PART_500K_VARIANTS:=frozen_anchor_pn_residual frozen_anchor_shuffled_pn_control}"

: "${DUALVIEW_PART_500K_STACK_TRAIN_SIZE:=500000}"
: "${DUALVIEW_PART_500K_STACK_VAL_SIZE:=150000}"
: "${DUALVIEW_PART_500K_FINAL_TEST_SIZE:=500000}"
: "${DUALVIEW_PART_500K_EPOCHS:=45}"
: "${DUALVIEW_PART_500K_SELECTION_METRIC:=fpr_at_signal_eff_0p50}"
: "${DUALVIEW_PART_500K_EARLY_STOP_PATIENCE:=6}"
: "${DUALVIEW_PART_500K_BATCH_SIZE:=64}"
: "${DUALVIEW_PART_500K_EVAL_BATCH_SIZE:=128}"
: "${DUALVIEW_PART_500K_NUM_WORKERS:=4}"
: "${DUALVIEW_PART_500K_MAX_CASE_ROWS_PER_TYPE:=2000}"

: "${DUALVIEW_PART_500K_TIME:=2-12:00:00}"
: "${DUALVIEW_PART_500K_MEM:=128G}"
: "${DUALVIEW_PART_500K_CPUS:=4}"
: "${DUALVIEW_PART_500K_REPORT_TIME:=02:00:00}"
: "${DUALVIEW_PART_500K_REPORT_MEM:=8G}"
: "${DUALVIEW_PART_500K_REPORT_CPUS:=2}"

export DUALVIEW_PART_ROOT="${DUALVIEW_PART_500K_ROOT}"
export DUALVIEW_PART_EXPERIMENT_DIR="${DUALVIEW_PART_500K_SOURCE_ROOT}"
export DUALVIEW_PART_HLT_CACHE_DIR="${DUALVIEW_PART_500K_HLT_CACHE_DIR}"
export DUALVIEW_PART_PN_RECONSTRUCTED_VIEW_DIR="${DUALVIEW_PART_500K_PN_RECONSTRUCTED_VIEW_DIR}"
export DUALVIEW_PART_HLT_ANCHOR_CHECKPOINT="${DUALVIEW_PART_500K_HLT_ANCHOR_CHECKPOINT}"
export DUALVIEW_PART_TAGGER_ROOT="${DUALVIEW_PART_500K_ROOT}/taggers"
export DUALVIEW_PART_FINAL_REPORT_DIR="${DUALVIEW_PART_500K_ROOT}/final_report"

export DUALVIEW_PART_STACK_TRAIN_SIZE="${DUALVIEW_PART_500K_STACK_TRAIN_SIZE}"
export DUALVIEW_PART_STACK_VAL_SIZE="${DUALVIEW_PART_500K_STACK_VAL_SIZE}"
export DUALVIEW_PART_FINAL_TEST_SIZE="${DUALVIEW_PART_500K_FINAL_TEST_SIZE}"
export DUALVIEW_PART_EPOCHS="${DUALVIEW_PART_500K_EPOCHS}"
export DUALVIEW_PART_EARLY_STOP_PATIENCE="${DUALVIEW_PART_500K_EARLY_STOP_PATIENCE}"
export DUALVIEW_PART_BATCH_SIZE="${DUALVIEW_PART_500K_BATCH_SIZE}"
export DUALVIEW_PART_EVAL_BATCH_SIZE="${DUALVIEW_PART_500K_EVAL_BATCH_SIZE}"
export DUALVIEW_PART_NUM_WORKERS="${DUALVIEW_PART_500K_NUM_WORKERS}"
export DUALVIEW_PART_SELECTION_METRIC="${DUALVIEW_PART_500K_SELECTION_METRIC}"
export DUALVIEW_PART_CONFIRM_FINAL_TEST=1
export DUALVIEW_PART_INITIALIZATION_CHECK_BATCHES="${DUALVIEW_PART_500K_INITIALIZATION_CHECK_BATCHES:-1}"
export DUALVIEW_PART_SKIP_INITIALIZATION_CHECK="${DUALVIEW_PART_500K_SKIP_INITIALIZATION_CHECK:-0}"
export DUALVIEW_PART_MAX_CASE_ROWS_PER_TYPE="${DUALVIEW_PART_500K_MAX_CASE_ROWS_PER_TYPE}"
export DUALVIEW_PART_REPORT_VARIANTS="${DUALVIEW_PART_500K_VARIANTS}"
export DUALVIEW_PART_REPORT_REAL_VARIANT="${DUALVIEW_PART_500K_REAL_VARIANT:-frozen_anchor_pn_residual}"
export DUALVIEW_PART_REPORT_SHUFFLED_VARIANT="${DUALVIEW_PART_500K_SHUFFLED_VARIANT:-frozen_anchor_shuffled_pn_control}"
export DUALVIEW_PART_REPORT_COMPARISON_SPLIT="${DUALVIEW_PART_500K_REPORT_COMPARISON_SPLIT:-final_test}"
export DUALVIEW_PART_REPORT_REQUIRE_REAL_BEATS_SHUFFLED="${DUALVIEW_PART_500K_REQUIRE_REAL_BEATS_SHUFFLED:-1}"

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

fresh_split_words variant_args "${DUALVIEW_PART_500K_VARIANTS}"

submitter_lock_dir="${DUALVIEW_PART_500K_ROOT}/.submission_lock"
fresh_refuse_existing_dir "${DUALVIEW_PART_500K_ROOT}"
fresh_claim_new_dir "${submitter_lock_dir}"
if ! fresh_is_dry_run; then
  {
    echo "created_at=$(date -Is)"
    echo "project_dir=${PROJECT_DIR}"
    echo "source_commit=$(fresh_source_commit)"
    echo "task=QCD_vs_Hgg_HLT0p6_dualview_part_residual_step11_500k"
    echo "root=${DUALVIEW_PART_500K_ROOT}"
    echo "source_root=${DUALVIEW_PART_500K_SOURCE_ROOT}"
    echo "hlt_anchor_checkpoint=${DUALVIEW_PART_HLT_ANCHOR_CHECKPOINT}"
    echo "hlt_cache_dir=${DUALVIEW_PART_HLT_CACHE_DIR}"
    echo "pn_reconstructed_view_dir=${DUALVIEW_PART_PN_RECONSTRUCTED_VIEW_DIR}"
    echo "variants=$(fresh_join_by_space "${variant_args[@]}")"
    echo "stack_train=${DUALVIEW_PART_STACK_TRAIN_SIZE}"
    echo "stack_val=${DUALVIEW_PART_STACK_VAL_SIZE}"
    echo "final_test=${DUALVIEW_PART_FINAL_TEST_SIZE}"
    echo "epochs=${DUALVIEW_PART_EPOCHS}"
    echo "selection_metric=${DUALVIEW_PART_SELECTION_METRIC}"
    echo "max_case_rows_per_type=${DUALVIEW_PART_MAX_CASE_ROWS_PER_TYPE}"
    echo "report_real_variant=${DUALVIEW_PART_REPORT_REAL_VARIANT}"
    echo "report_shuffled_variant=${DUALVIEW_PART_REPORT_SHUFFLED_VARIANT}"
    echo "report_require_real_beats_shuffled=${DUALVIEW_PART_REPORT_REQUIRE_REAL_BEATS_SHUFFLED}"
  } > "${submitter_lock_dir}/metadata.txt"
fi

job_ids=()
for variant in "${variant_args[@]}"; do
  mapfile -t tagger_args < <(
    afterok_args \
      "${UPSTREAM_DEPENDENCY}" \
      --time="${DUALVIEW_PART_500K_TIME}" \
      --cpus-per-task="${DUALVIEW_PART_500K_CPUS}" \
      --mem="${DUALVIEW_PART_500K_MEM}" \
      "${SCRIPT_DIR}/run_train_dualview_part_residual.sh" \
      "${variant}"
  )
  tagger_jid="$(submit_job "dualview_part_500k_${variant}" "${tagger_args[@]}")"
  job_ids+=("${tagger_jid}")
  echo "submitted dualview_part_500k_${variant}=${tagger_jid}"
done

report_dep="$(fresh_join_by_colon "${job_ids[@]}")"
report_jid="$(submit_job "dualview_part_500k_report" \
  --time="${DUALVIEW_PART_500K_REPORT_TIME}" \
  --cpus-per-task="${DUALVIEW_PART_500K_REPORT_CPUS}" \
  --mem="${DUALVIEW_PART_500K_REPORT_MEM}" \
  --dependency="afterok:${report_dep}" \
  "${SCRIPT_DIR}/run_write_dualview_part_report.sh")"
echo "submitted dualview_part_500k_report=${report_jid}"

cat <<SUMMARY
dualview_part_residual_500k_submission:
  task: QCD_vs_Hgg_HLT0p6
  root: ${DUALVIEW_PART_500K_ROOT}
  source_root: ${DUALVIEW_PART_500K_SOURCE_ROOT}
  upstream_artifacts:
    hlt_anchor_checkpoint: ${DUALVIEW_PART_HLT_ANCHOR_CHECKPOINT}
    hlt_cache_dir: ${DUALVIEW_PART_HLT_CACHE_DIR}
    pn_reconstructed_view_dir: ${DUALVIEW_PART_PN_RECONSTRUCTED_VIEW_DIR}
  sizes:
    stack_train: ${DUALVIEW_PART_STACK_TRAIN_SIZE}
    stack_val: ${DUALVIEW_PART_STACK_VAL_SIZE}
    final_test: ${DUALVIEW_PART_FINAL_TEST_SIZE}
  training:
    epochs: ${DUALVIEW_PART_EPOCHS}
    selection_metric: ${DUALVIEW_PART_SELECTION_METRIC}
    max_case_rows_per_type: ${DUALVIEW_PART_MAX_CASE_ROWS_PER_TYPE}
    time: ${DUALVIEW_PART_500K_TIME}
    mem: ${DUALVIEW_PART_500K_MEM}
    cpus: ${DUALVIEW_PART_500K_CPUS}
  report:
    comparison_split: ${DUALVIEW_PART_REPORT_COMPARISON_SPLIT}
    real_variant: ${DUALVIEW_PART_REPORT_REAL_VARIANT}
    shuffled_variant: ${DUALVIEW_PART_REPORT_SHUFFLED_VARIANT}
    require_real_beats_shuffled: ${DUALVIEW_PART_REPORT_REQUIRE_REAL_BEATS_SHUFFLED}
  variants: $(fresh_join_by_space "${variant_args[@]}")
  job_ids: $(fresh_join_by_space "${job_ids[@]}")
  report_job_id: ${report_jid}
  expected_jobs:
    residual_real_pn: 1
    shuffled_pn_control: 1
    report: 1
    total_submitted: ${submit_count}
  output_dirs:
    taggers: ${DUALVIEW_PART_TAGGER_ROOT}
    final_report: ${DUALVIEW_PART_FINAL_REPORT_DIR}
    diagnostics_root: ${DIAGNOSTICS_ROOT}
    logs: ${PROJECT_DIR}/fresh_check_logs
SUMMARY
