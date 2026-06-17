#!/usr/bin/env bash
# Submit the set-matching multi-view reconstructor, cache, tagger, and audit graph.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${UPSTREAM_DEPENDENCY:=}"
: "${TAGGER_UPSTREAM_DEPENDENCY:=}"

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
fresh_split_words tagger_variant_args "${SET_MATCHING_TAGGER_VARIANTS}"
fresh_split_words cache_split_args "${SET_MATCHING_CACHE_SPLITS}"

submitter_lock_dir="${SET_MATCHING_ROOT}/.submission_lock"
fresh_claim_new_dir "${submitter_lock_dir}"
if ! fresh_is_dry_run; then
  {
    echo "created_at=$(date -Is)"
    echo "project_dir=${PROJECT_DIR}"
    echo "source_commit=$(fresh_source_commit)"
    echo "reconstructors=$(fresh_join_by_space "${reco_args[@]}")"
    echo "tagger_variants=$(fresh_join_by_space "${tagger_variant_args[@]}")"
    echo "cache_splits=$(fresh_join_by_space "${cache_split_args[@]}")"
    echo "manifest=${SET_MATCHING_MANIFEST_PATH}"
    echo "hlt_cache_dir=${SET_MATCHING_HLT_CACHE_DIR}"
    echo "set_matching_root=${SET_MATCHING_ROOT}"
    echo "reconstructor_dir=${SET_MATCHING_RECONSTRUCTOR_DIR}"
    echo "reconstructed_view_dir=${SET_MATCHING_RECONSTRUCTED_VIEW_DIR}"
    echo "tagger_root=${SET_MATCHING_TAGGER_ROOT}"
    echo "ablation_dir=${SET_MATCHING_ABLATION_DIR}"
    echo "upstream_dependency=${UPSTREAM_DEPENDENCY:-none}"
    echo "tagger_upstream_dependency=${TAGGER_UPSTREAM_DEPENDENCY:-none}"
  } > "${submitter_lock_dir}/metadata.txt"
fi

if ! fresh_is_dry_run; then
  fresh_require_file "${SET_MATCHING_MANIFEST_PATH}"
  fresh_require_dir "${SET_MATCHING_HLT_CACHE_DIR}"
  for split in model_train model_val "${cache_split_args[@]}"; do
    fresh_require_file "${SET_MATCHING_HLT_CACHE_DIR}/${split}_fixed_hlt_metadata.json"
  done
fi

for architecture in "${reco_args[@]}"; do
  fresh_refuse_existing_dir "${SET_MATCHING_RECONSTRUCTOR_DIR}/${architecture}"
  fresh_refuse_existing_dir "${SET_MATCHING_RECONSTRUCTED_VIEW_DIR}/${architecture}"
done
for variant in "${tagger_variant_args[@]}"; do
  fresh_refuse_existing_dir "${SET_MATCHING_TAGGER_ROOT}/${variant}"
done
fresh_refuse_existing_dir "${SET_MATCHING_ABLATION_DIR}"
fresh_refuse_existing_dir "${SET_MATCHING_FINAL_REPORT_DIR}"

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
  train_jid="$(submit_job "setmatch_train_${architecture}" "${train_args[@]}")"
  reco_train_job_ids+=("${train_jid}")
  reco_train_job_id_by_arch["${architecture}"]="${train_jid}"
  echo "submitted setmatch_train_${architecture}=${train_jid}"
done

for architecture in "${reco_args[@]}"; do
  train_jid="${reco_train_job_id_by_arch[$architecture]}"
  cache_jid="$(submit_job "setmatch_cache_${architecture}" \
    --dependency="afterok:${train_jid}" \
    "${SCRIPT_DIR}/run_cache_set_matching_multiview.sh" \
    "${architecture}")"
  cache_job_ids+=("${cache_jid}")
  echo "submitted setmatch_cache_${architecture}=${cache_jid}"
done

cache_dep="$(fresh_join_by_colon "${cache_job_ids[@]}")"
if [[ -n "${TAGGER_UPSTREAM_DEPENDENCY}" ]]; then
  cache_dep="${cache_dep}:${TAGGER_UPSTREAM_DEPENDENCY}"
fi
for variant in "${tagger_variant_args[@]}"; do
  tagger_jid="$(submit_job "setmatch_tagger_${variant}" \
    --dependency="afterok:${cache_dep}" \
    "${SCRIPT_DIR}/run_train_five_view_tagger.sh" \
    "${variant}")"
  tagger_job_ids+=("${tagger_jid}")
  echo "submitted setmatch_tagger_${variant}=${tagger_jid}"
done

audit_dep="$(fresh_join_by_colon "${tagger_job_ids[@]}")"
audit_jid="$(submit_job "setmatch_audit" \
  --dependency="afterok:${audit_dep}" \
  "${SCRIPT_DIR}/run_audit_five_view_tagger.sh")"
final_report_jid="$(submit_job "setmatch_final_report" \
  --dependency="afterok:${audit_jid}" \
  "${SCRIPT_DIR}/run_write_set_matching_multiview_final_report.sh")"

cat <<SUMMARY
set_matching_multiview_submission:
  reco_train_job_ids: $(fresh_join_by_space "${reco_train_job_ids[@]}")
  cache_job_ids: $(fresh_join_by_space "${cache_job_ids[@]}")
  tagger_job_ids: $(fresh_join_by_space "${tagger_job_ids[@]}")
  audit_job_id: ${audit_jid}
  final_report_job_id: ${final_report_jid}
  dependency_summary:
    reco_train_afterok_extra: ${UPSTREAM_DEPENDENCY:-none}
    each_cache_after_its_reco_train: true
    taggers_afterok: ${cache_dep}
    audit_afterok: ${audit_dep}
    final_report_afterok: ${audit_jid}
  expected_jobs:
    reco_train: 4
    cache_reconstructed_views: 4
    tagger_train: ${#tagger_job_ids[@]}
    audit: 1
    final_report: 1
    total_submitted: ${submit_count}
  variants:
    reconstructors: $(fresh_join_by_space "${reco_args[@]}")
    taggers: $(fresh_join_by_space "${tagger_variant_args[@]}")
    cache_splits: $(fresh_join_by_space "${cache_split_args[@]}")
  split_sizes:
    model_train: ${SET_MATCHING_MODEL_TRAIN_SIZE}
    model_val: ${SET_MATCHING_MODEL_VAL_SIZE}
    stack_train: ${SET_MATCHING_STACK_TRAIN_SIZE}
    stack_val: ${SET_MATCHING_STACK_VAL_SIZE}
    final_test: ${SET_MATCHING_FINAL_TEST_SIZE}
  output_dirs:
    root: ${SET_MATCHING_ROOT}
    reconstructors: ${SET_MATCHING_RECONSTRUCTOR_DIR}
    reconstructed_views: ${SET_MATCHING_RECONSTRUCTED_VIEW_DIR}
    taggers: ${SET_MATCHING_TAGGER_ROOT}
    ablations: ${SET_MATCHING_ABLATION_DIR}
    final_report: ${SET_MATCHING_FINAL_REPORT_DIR}
    logs: ${PROJECT_DIR}/fresh_check_logs
SUMMARY
