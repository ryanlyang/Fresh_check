#!/usr/bin/env bash
# Submit a tiny reliability-gated dual-view ParT smoke test.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${UPSTREAM_DEPENDENCY:=}"
: "${DUALVIEW_PART_SMOKE_TAG:=$(date +%Y%m%d_%H%M%S)}"
: "${DUALVIEW_PART_SMOKE_ROOT:=${OUTPUT_ROOT}/dualview_part_residual_smoke_${DUALVIEW_PART_SMOKE_TAG}}"
: "${DUALVIEW_PART_SMOKE_SOURCE_ROOT:=${DUALVIEW_PART_EXPERIMENT_DIR:-${DUALVIEW_PART_ROOT}}}"
: "${DUALVIEW_PART_SMOKE_HLT_CACHE_DIR:=${DUALVIEW_PART_HLT_CACHE_DIR}}"
: "${DUALVIEW_PART_SMOKE_PN_RECONSTRUCTED_VIEW_DIR:=${DUALVIEW_PART_PN_RECONSTRUCTED_VIEW_DIR}}"
: "${DUALVIEW_PART_SMOKE_HLT_ANCHOR_CHECKPOINT:=${DUALVIEW_PART_HLT_ANCHOR_CHECKPOINT}}"
: "${DUALVIEW_PART_SMOKE_VARIANTS:=frozen_anchor_pn_residual frozen_anchor_shuffled_pn_control}"
: "${DUALVIEW_PART_SMOKE_TIME:=04:00:00}"
: "${DUALVIEW_PART_SMOKE_MEM:=64G}"
: "${DUALVIEW_PART_SMOKE_CPUS:=4}"
: "${DUALVIEW_PART_SMOKE_REPORT_TIME:=01:00:00}"
: "${DUALVIEW_PART_SMOKE_REPORT_MEM:=8G}"
: "${DUALVIEW_PART_SMOKE_REPORT_CPUS:=2}"

export DUALVIEW_PART_ROOT="${DUALVIEW_PART_SMOKE_ROOT}"
export DUALVIEW_PART_EXPERIMENT_DIR="${DUALVIEW_PART_SMOKE_SOURCE_ROOT}"
export DUALVIEW_PART_HLT_CACHE_DIR="${DUALVIEW_PART_SMOKE_HLT_CACHE_DIR}"
export DUALVIEW_PART_PN_RECONSTRUCTED_VIEW_DIR="${DUALVIEW_PART_SMOKE_PN_RECONSTRUCTED_VIEW_DIR}"
export DUALVIEW_PART_HLT_ANCHOR_CHECKPOINT="${DUALVIEW_PART_SMOKE_HLT_ANCHOR_CHECKPOINT}"
export DUALVIEW_PART_TAGGER_ROOT="${DUALVIEW_PART_SMOKE_ROOT}/taggers"
export DUALVIEW_PART_FINAL_REPORT_DIR="${DUALVIEW_PART_SMOKE_ROOT}/final_report"

export DUALVIEW_PART_STACK_TRAIN_SIZE="${DUALVIEW_PART_SMOKE_STACK_TRAIN_SIZE:-10000}"
export DUALVIEW_PART_STACK_VAL_SIZE="${DUALVIEW_PART_SMOKE_STACK_VAL_SIZE:-5000}"
export DUALVIEW_PART_FINAL_TEST_SIZE="${DUALVIEW_PART_SMOKE_FINAL_TEST_SIZE:-10000}"
export DUALVIEW_PART_EPOCHS="${DUALVIEW_PART_SMOKE_EPOCHS:-2}"
export DUALVIEW_PART_EARLY_STOP_PATIENCE="${DUALVIEW_PART_SMOKE_EARLY_STOP_PATIENCE:-1}"
export DUALVIEW_PART_BATCH_SIZE="${DUALVIEW_PART_SMOKE_BATCH_SIZE:-32}"
export DUALVIEW_PART_EVAL_BATCH_SIZE="${DUALVIEW_PART_SMOKE_EVAL_BATCH_SIZE:-64}"
export DUALVIEW_PART_NUM_WORKERS="${DUALVIEW_PART_SMOKE_NUM_WORKERS:-2}"
export DUALVIEW_PART_SELECTION_METRIC="${DUALVIEW_PART_SMOKE_SELECTION_METRIC:-fpr_at_signal_eff_0p50}"
export DUALVIEW_PART_CONFIRM_FINAL_TEST=1
export DUALVIEW_PART_INITIALIZATION_CHECK_BATCHES="${DUALVIEW_PART_SMOKE_INITIALIZATION_CHECK_BATCHES:-1}"
export DUALVIEW_PART_SKIP_INITIALIZATION_CHECK="${DUALVIEW_PART_SMOKE_SKIP_INITIALIZATION_CHECK:-0}"
export DUALVIEW_PART_MAX_CASE_ROWS_PER_TYPE="${DUALVIEW_PART_SMOKE_MAX_CASE_ROWS_PER_TYPE:-200}"
export DUALVIEW_PART_REPORT_VARIANTS="${DUALVIEW_PART_SMOKE_VARIANTS}"
export DUALVIEW_PART_REPORT_REAL_VARIANT="${DUALVIEW_PART_SMOKE_REAL_VARIANT:-frozen_anchor_pn_residual}"
export DUALVIEW_PART_REPORT_SHUFFLED_VARIANT="${DUALVIEW_PART_SMOKE_SHUFFLED_VARIANT:-frozen_anchor_shuffled_pn_control}"
export DUALVIEW_PART_REPORT_COMPARISON_SPLIT="${DUALVIEW_PART_SMOKE_REPORT_COMPARISON_SPLIT:-final_test}"
export DUALVIEW_PART_REPORT_REQUIRE_REAL_BEATS_SHUFFLED="${DUALVIEW_PART_SMOKE_REQUIRE_REAL_BEATS_SHUFFLED:-1}"

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

fresh_split_words variant_args "${DUALVIEW_PART_SMOKE_VARIANTS}"

submitter_lock_dir="${DUALVIEW_PART_SMOKE_ROOT}/.submission_lock"
fresh_refuse_existing_dir "${DUALVIEW_PART_SMOKE_ROOT}"
fresh_claim_new_dir "${submitter_lock_dir}"
if ! fresh_is_dry_run; then
  {
    echo "created_at=$(date -Is)"
    echo "project_dir=${PROJECT_DIR}"
    echo "source_commit=$(fresh_source_commit)"
    echo "task=QCD_vs_Hgg_HLT0p6_dualview_part_residual_smoke"
    echo "smoke_root=${DUALVIEW_PART_SMOKE_ROOT}"
    echo "source_root=${DUALVIEW_PART_SMOKE_SOURCE_ROOT}"
    echo "hlt_anchor_checkpoint=${DUALVIEW_PART_HLT_ANCHOR_CHECKPOINT}"
    echo "hlt_cache_dir=${DUALVIEW_PART_HLT_CACHE_DIR}"
    echo "pn_reconstructed_view_dir=${DUALVIEW_PART_PN_RECONSTRUCTED_VIEW_DIR}"
    echo "variants=$(fresh_join_by_space "${variant_args[@]}")"
    echo "stack_train=${DUALVIEW_PART_STACK_TRAIN_SIZE}"
    echo "stack_val=${DUALVIEW_PART_STACK_VAL_SIZE}"
    echo "final_test=${DUALVIEW_PART_FINAL_TEST_SIZE}"
    echo "epochs=${DUALVIEW_PART_EPOCHS}"
    echo "selection_metric=${DUALVIEW_PART_SELECTION_METRIC}"
    echo "initialization_check_batches=${DUALVIEW_PART_INITIALIZATION_CHECK_BATCHES}"
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
      --time="${DUALVIEW_PART_SMOKE_TIME}" \
      --cpus-per-task="${DUALVIEW_PART_SMOKE_CPUS}" \
      --mem="${DUALVIEW_PART_SMOKE_MEM}" \
      "${SCRIPT_DIR}/run_train_dualview_part_residual.sh" \
      "${variant}"
  )
  tagger_jid="$(submit_job "dualview_part_smoke_${variant}" "${tagger_args[@]}")"
  job_ids+=("${tagger_jid}")
  echo "submitted dualview_part_smoke_${variant}=${tagger_jid}"
done
report_dep="$(fresh_join_by_colon "${job_ids[@]}")"
report_jid="$(submit_job "dualview_part_smoke_report" \
  --time="${DUALVIEW_PART_SMOKE_REPORT_TIME}" \
  --cpus-per-task="${DUALVIEW_PART_SMOKE_REPORT_CPUS}" \
  --mem="${DUALVIEW_PART_SMOKE_REPORT_MEM}" \
  --dependency="afterok:${report_dep}" \
  "${SCRIPT_DIR}/run_write_dualview_part_report.sh")"
echo "submitted dualview_part_smoke_report=${report_jid}"

cat <<SUMMARY
dualview_part_residual_smoke_submission:
  warning: smoke metrics are for pipeline correctness only, not physics interpretation
  task: QCD_vs_Hgg_HLT0p6
  smoke_root: ${DUALVIEW_PART_SMOKE_ROOT}
  source_root: ${DUALVIEW_PART_SMOKE_SOURCE_ROOT}
  upstream_artifacts:
    hlt_anchor_checkpoint: ${DUALVIEW_PART_HLT_ANCHOR_CHECKPOINT}
    hlt_cache_dir: ${DUALVIEW_PART_HLT_CACHE_DIR}
    pn_reconstructed_view_dir: ${DUALVIEW_PART_PN_RECONSTRUCTED_VIEW_DIR}
  smoke_sizes:
    stack_train: ${DUALVIEW_PART_STACK_TRAIN_SIZE}
    stack_val: ${DUALVIEW_PART_STACK_VAL_SIZE}
    final_test: ${DUALVIEW_PART_FINAL_TEST_SIZE}
  smoke_training:
    epochs: ${DUALVIEW_PART_EPOCHS}
    selection_metric: ${DUALVIEW_PART_SELECTION_METRIC}
    initialization_check_batches: ${DUALVIEW_PART_INITIALIZATION_CHECK_BATCHES}
    skip_initialization_check: ${DUALVIEW_PART_SKIP_INITIALIZATION_CHECK}
    max_case_rows_per_type: ${DUALVIEW_PART_MAX_CASE_ROWS_PER_TYPE}
  smoke_report:
    comparison_split: ${DUALVIEW_PART_REPORT_COMPARISON_SPLIT}
    real_variant: ${DUALVIEW_PART_REPORT_REAL_VARIANT}
    shuffled_variant: ${DUALVIEW_PART_REPORT_SHUFFLED_VARIANT}
    require_real_beats_shuffled: ${DUALVIEW_PART_REPORT_REQUIRE_REAL_BEATS_SHUFFLED}
  variants: $(fresh_join_by_space "${variant_args[@]}")
  shuffled_pn_control: submitted
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
