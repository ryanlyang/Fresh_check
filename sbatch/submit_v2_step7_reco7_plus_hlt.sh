#!/usr/bin/env bash
# Submit V2 Step 7 seven-variant training, then dependent Step 10 fusion and Step 11 audits.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${UPSTREAM_DEPENDENCY:=}"
: "${HLT_BASELINE_REPORT:=${HLT_BASELINE_DIR}/model_val_report.json}"

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

dep_args() {
  local dependency="$1"
  shift
  if [[ -n "${dependency}" ]]; then
    printf '%s\n' --dependency="afterok:${dependency}"
  fi
  printf '%s\n' "$@"
}

if [[ ! -f "${HLT_BASELINE_DIR}/best_model_val.pt" ]] && ! fresh_is_dry_run; then
  echo "Required HLT baseline checkpoint is missing: ${HLT_BASELINE_DIR}/best_model_val.pt" >&2
  echo "Run the HLT baseline first or point HLT_BASELINE_DIR at an existing baseline." >&2
  exit 2
fi
if [[ ! -f "${HLT_BASELINE_REPORT}" ]] && ! fresh_is_dry_run; then
  echo "Required HLT baseline model_val report is missing: ${HLT_BASELINE_REPORT}" >&2
  echo "Run the HLT baseline first or point HLT_BASELINE_REPORT at an existing report." >&2
  exit 2
fi

fresh_split_words variant_args "${V2_STEP7_VARIANTS}"
variant_job_ids=()
for variant in "${variant_args[@]}"; do
  fresh_refuse_existing_dir "${V2_STEP7_RECO_ROOT}/${variant}"
  mapfile -t args < <(dep_args "${UPSTREAM_DEPENDENCY}" "${SCRIPT_DIR}/run_v2_step7_train_variant.sh" "${variant}")
  jid="$(submit_job "v2_step7_${variant}" "${args[@]}")"
  variant_job_ids+=("${jid}")
  echo "submitted v2_step7_${variant}=${jid}"
done

fusion_dependency="$(fresh_join_by_colon "${variant_job_ids[@]}")"
fresh_refuse_existing_dir "${V2_STEP7_FUSION_DIR}"
fusion_jid="$(submit_job "v2_step10_fusion_reco7_plus_hlt" \
  --dependency="afterok:${fusion_dependency}" \
  "${SCRIPT_DIR}/run_v2_step10_fuse_reco7_plus_hlt.sh")"

fresh_refuse_existing_dir "${V2_STEP7_AUDIT_DIR}"
audit_jid="$(submit_job "v2_step11_audit_reco7_plus_hlt" \
  --dependency="afterok:${fusion_jid}" \
  "${SCRIPT_DIR}/run_v2_step11_audit_reco7_plus_hlt.sh")"

cat <<SUMMARY
v2_step7_reco7_plus_hlt_submission:
  variant_job_ids: $(fresh_join_by_space "${variant_job_ids[@]}")
  fusion_job_id: ${fusion_jid}
  audit_job_id: ${audit_jid}
  dependency_summary:
    fusion_afterok: ${fusion_dependency}
    audit_afterok: ${fusion_jid}
  variants: $(fresh_join_by_space "${variant_args[@]}")
  output_dirs:
    reco7_root: ${V2_STEP7_RECO_ROOT}
    fusion: ${V2_STEP7_FUSION_DIR}
    audits: ${V2_STEP7_AUDIT_DIR}
    logs: ${PROJECT_DIR}/fresh_check_logs
  skipped:
    hlt5_seed_control: true
SUMMARY
