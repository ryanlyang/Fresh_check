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

export SET_MATCHING_MODEL_TRAIN_SIZE="${HBB_QCD_MODEL_TRAIN_SIZE:-100000}"
export SET_MATCHING_MODEL_VAL_SIZE="${HBB_QCD_MODEL_VAL_SIZE:-30000}"
export SET_MATCHING_STACK_TRAIN_SIZE="${HBB_QCD_STACK_TRAIN_SIZE:-100000}"
export SET_MATCHING_STACK_VAL_SIZE="${HBB_QCD_STACK_VAL_SIZE:-30000}"
export SET_MATCHING_FINAL_TEST_SIZE="${HBB_QCD_FINAL_TEST_SIZE:-100000}"
export SET_MATCHING_CACHE_MAX_JETS_PER_SPLIT="${HBB_QCD_CACHE_MAX_JETS_PER_SPLIT:-${SET_MATCHING_FINAL_TEST_SIZE}}"

export SET_MATCHING_RECO_EPOCHS="${HBB_QCD_RECO_EPOCHS:-8}"
export SET_MATCHING_RECO_EARLY_STOP_PATIENCE="${HBB_QCD_RECO_EARLY_STOP_PATIENCE:-3}"
export SET_MATCHING_TAGGER_EPOCHS="${HBB_QCD_TAGGER_EPOCHS:-12}"
export SET_MATCHING_TAGGER_EARLY_STOP_PATIENCE="${HBB_QCD_TAGGER_EARLY_STOP_PATIENCE:-4}"
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
    echo "tagger_variants=$(fresh_join_by_space "${tagger_args[@]}")"
  } > "${submitter_lock_dir}/metadata.txt"
fi

reco_train_job_ids=()
cache_job_ids=()
tagger_job_ids=()
declare -A reco_train_job_id_by_arch=()

for architecture in "${reco_args[@]}"; do
  mapfile -t train_args < <(
    afterok_args \
      "${UPSTREAM_DEPENDENCY}" \
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
    --dependency="afterok:${cache_dep}" \
    "${SCRIPT_DIR}/run_train_five_view_tagger.sh" \
    "${variant}")"
  tagger_job_ids+=("${tagger_jid}")
  echo "submitted hbbqcd_setmatch_tagger_${variant}=${tagger_jid}"
done

audit_dep="$(fresh_join_by_colon "${tagger_job_ids[@]}")"
audit_jid="$(submit_job "hbbqcd_setmatch_audit" \
  --dependency="afterok:${audit_dep}" \
  "${SCRIPT_DIR}/run_audit_five_view_tagger.sh")"
final_report_jid="$(submit_job "hbbqcd_setmatch_final_report" \
  --dependency="afterok:${audit_jid}" \
  "${SCRIPT_DIR}/run_write_set_matching_multiview_final_report.sh")"

cat <<SUMMARY
hbb_qcd_binary_set_matching_submission:
  task: Hbb_vs_QCD
  label_filter: ${SET_MATCHING_LABEL_FILTER_NAMES}
  num_classes: ${SET_MATCHING_NUM_CLASSES}
  reco_train_job_ids: $(fresh_join_by_space "${reco_train_job_ids[@]}")
  cache_job_ids: $(fresh_join_by_space "${cache_job_ids[@]}")
  tagger_job_ids: $(fresh_join_by_space "${tagger_job_ids[@]}")
  audit_job_id: ${audit_jid}
  final_report_job_id: ${final_report_jid}
  expected_jobs:
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
  output_dirs:
    root: ${SET_MATCHING_ROOT}
    taggers: ${SET_MATCHING_TAGGER_ROOT}
    audit: ${SET_MATCHING_ABLATION_DIR}
    final_report: ${SET_MATCHING_FINAL_REPORT_DIR}
    logs: ${PROJECT_DIR}/fresh_check_logs
  key_metrics:
    tagger_run_reports: ${SET_MATCHING_TAGGER_ROOT}/<variant>/run_report.json
    audit_summary: ${SET_MATCHING_ABLATION_DIR}/summary.csv
    final_report: ${SET_MATCHING_FINAL_REPORT_DIR}/final_report.json
SUMMARY
