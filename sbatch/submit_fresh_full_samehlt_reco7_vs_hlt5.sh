#!/usr/bin/env bash
# Submit the full clean same-HLT reco7-vs-HLT5 replication graph.

set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

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

fresh_refuse_existing_path "${MANIFEST_PATH}"
fresh_refuse_existing_dir "${HLT_CACHE_DIR}"
fresh_refuse_existing_dir "${HLT_BASELINE_DIR}"
fresh_refuse_existing_dir "${HLT5_ROOT}"
fresh_refuse_existing_dir "${OFFLINE_TEACHER_DIR}"
fresh_refuse_existing_dir "${RECO7_ROOT}"
fresh_refuse_existing_dir "${RECO7_FUSION_DIR}"
fresh_refuse_existing_dir "${HLT5_FUSION_DIR}"
fresh_refuse_existing_dir "${RECO7_AUDIT_DIR}"
fresh_refuse_existing_dir "${HLT5_AUDIT_DIR}"
fresh_refuse_existing_dir "${FINAL_REPORT_DIR}"

split_jid="$(submit_job "build_splits" "${SCRIPT_DIR}/run_build_fresh_splits.sh")"
cache_jid="$(submit_job "build_hlt_cache" --dependency="afterok:${split_jid}" "${SCRIPT_DIR}/run_build_fresh_hlt_cache.sh")"
offline_jid="$(submit_job "offline_teacher" --dependency="afterok:${split_jid}" "${SCRIPT_DIR}/run_train_fresh_offline_teacher.sh")"
baseline_jid="$(submit_job "hlt_baseline" --dependency="afterok:${cache_jid}" "${SCRIPT_DIR}/run_train_fresh_hlt_baseline.sh")"

fresh_split_words seed_args "${HLT5_SEEDS}"
hlt_seed_job_ids=()
for seed in "${seed_args[@]}"; do
  jid="$(submit_job "hlt_seed_${seed}" --dependency="afterok:${cache_jid}" "${SCRIPT_DIR}/run_train_fresh_hlt_seed.sh" "${seed}")"
  hlt_seed_job_ids+=("${jid}")
done
hlt5_dep="$(fresh_join_by_colon "${hlt_seed_job_ids[@]}")"
hlt5_fusion_jid="$(submit_job "hlt5_fusion" --dependency="afterok:${hlt5_dep}" "${SCRIPT_DIR}/run_fuse_fresh_hlt5_seed_control.sh")"

fresh_split_words variant_args "${RECO7_VARIANTS}"
reco_job_ids=()
for variant in "${variant_args[@]}"; do
  jid="$(submit_job "reco7_${variant}" --dependency="afterok:${baseline_jid}" "${SCRIPT_DIR}/run_train_fresh_reco7_variant.sh" "${variant}")"
  reco_job_ids+=("${jid}")
done
reco7_dep="$(fresh_join_by_colon "${baseline_jid}" "${reco_job_ids[@]}")"
reco7_fusion_jid="$(submit_job "reco7_fusion" --dependency="afterok:${reco7_dep}" "${SCRIPT_DIR}/run_fuse_fresh_samehlt7_plus_hlt.sh")"

audit_dep="$(fresh_join_by_colon "${reco7_fusion_jid}" "${hlt5_fusion_jid}")"
audit_jid="$(submit_job "leakage_audits" --dependency="afterok:${audit_dep}" "${SCRIPT_DIR}/run_audit_fresh_samehlt7_plus_hlt.sh")"

final_dep="$(fresh_join_by_colon "${audit_jid}" "${offline_jid}")"
final_jid="$(submit_job "final_report" --dependency="afterok:${final_dep}" "${SCRIPT_DIR}/run_write_fresh_final_report.sh")"

cat <<SUMMARY
full_samehlt_reco7_vs_hlt5_submission:
  split_job_id: ${split_jid}
  hlt_cache_job_id: ${cache_jid}
  offline_teacher_job_id: ${offline_jid}
  hlt_baseline_job_id: ${baseline_jid}
  hlt_seed_job_ids: $(fresh_join_by_space "${hlt_seed_job_ids[@]}")
  hlt5_fusion_job_id: ${hlt5_fusion_jid}
  reco7_variant_job_ids: $(fresh_join_by_space "${reco_job_ids[@]}")
  reco7_fusion_job_id: ${reco7_fusion_jid}
  audit_job_id: ${audit_jid}
  final_report_job_id: ${final_jid}
  dependency_summary:
    hlt5_fusion_afterok: ${hlt5_dep}
    reco7_fusion_afterok: ${reco7_dep}
    audits_afterok: ${audit_dep}
    final_report_afterok: ${final_dep}
  output_dirs:
    manifest: ${MANIFEST_PATH}
    hlt_cache: ${HLT_CACHE_DIR}
    hlt_baseline: ${HLT_BASELINE_DIR}
    hlt5_root: ${HLT5_ROOT}
    reco7_root: ${RECO7_ROOT}
    reco7_fusion: ${RECO7_FUSION_DIR}
    hlt5_fusion: ${HLT5_FUSION_DIR}
    audits_reco7: ${RECO7_AUDIT_DIR}
    audits_hlt5: ${HLT5_AUDIT_DIR}
    final_report: ${FINAL_REPORT_DIR}
    logs: ${PROJECT_DIR}/fresh_check_logs
SUMMARY
