#!/usr/bin/env bash
# Submit V2 Step 6 training, then dependent Step 10 fusion, then Step 11 audits.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${UPSTREAM_DEPENDENCY:=}"
: "${HLT_BASELINE_REPORT:=${HLT_BASELINE_DIR}/model_val_report.json}"

if [[ "${V2_STEP6_VARIANT}" != "m2_base" ]]; then
  echo "This V2 Step 6 submitter trains m2_base only; got V2_STEP6_VARIANT=${V2_STEP6_VARIANT}" >&2
  exit 2
fi

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

fresh_refuse_existing_dir "${V2_STEP6_RECO_ROOT}/${V2_STEP6_VARIANT}"
fresh_refuse_existing_dir "${V2_STEP6_FUSION_DIR}"
fresh_refuse_existing_dir "${V2_STEP6_AUDIT_DIR}"

mapfile -t train_args < <(dep_args "${UPSTREAM_DEPENDENCY}" "${SCRIPT_DIR}/run_v2_step6_train_m2_base.sh")
train_jid="$(submit_job "v2_step6_train_m2_base" "${train_args[@]}")"

fusion_jid="$(submit_job "v2_step10_fusion_m2_base_plus_hlt" \
  --dependency="afterok:${train_jid}" \
  "${SCRIPT_DIR}/run_v2_step10_fuse_m2_base_plus_hlt.sh")"

audit_jid="$(submit_job "v2_step11_audit_m2_base_plus_hlt" \
  --dependency="afterok:${fusion_jid}" \
  "${SCRIPT_DIR}/run_v2_step11_audit_m2_base_plus_hlt.sh")"

cat <<SUMMARY
v2_step6_m2_base_end_to_end_submission:
  train_job_id: ${train_jid}
  fusion_job_id: ${fusion_jid}
  audit_job_id: ${audit_jid}
  dependency_summary:
    fusion_afterok: ${train_jid}
    audit_afterok: ${fusion_jid}
  output_dirs:
    stage_a: ${V2_STEP6_RECO_ROOT}/${V2_STEP6_VARIANT}/stage_a
    stage_b: ${V2_STEP6_RECO_ROOT}/${V2_STEP6_VARIANT}/stage2_dual_view
    fusion: ${V2_STEP6_FUSION_DIR}
    audits: ${V2_STEP6_AUDIT_DIR}
    logs: ${PROJECT_DIR}/fresh_check_logs
  skipped:
    hlt5_seed_control: true
SUMMARY
