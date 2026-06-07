#!/usr/bin/env bash
# Submit seven same-HLT reco7 variant jobs, then reco7+HLT fusion and audits.

set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${UPSTREAM_DEPENDENCY:=}"
: "${SUBMIT_BASELINE_IF_MISSING:=1}"
: "${HLT5_FUSION_JOB_ID:=}"

fresh_prepare_submitter

submit_count=0
submit_job() {
  local label="$1"
  shift
  submit_count=$((submit_count + 1))
  if fresh_is_dry_run; then
    echo "DRY_RUN sbatch ${label}: sbatch $*" >&2
    printf 'DRYRUN_%s_%03d\n' "${label}" "${submit_count}"
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

baseline_jid=""
if [[ ! -f "${HLT_BASELINE_DIR}/best_model_val.pt" ]] || fresh_is_dry_run; then
  if fresh_bool_enabled "${SUBMIT_BASELINE_IF_MISSING}"; then
    fresh_refuse_existing_dir "${HLT_BASELINE_DIR}"
    mapfile -t args < <(dep_args "${UPSTREAM_DEPENDENCY}" "${SCRIPT_DIR}/run_train_fresh_hlt_baseline.sh")
    baseline_jid="$(submit_job "hlt_baseline" "${args[@]}")"
    echo "submitted hlt_baseline=${baseline_jid}"
  else
    echo "HLT baseline checkpoint is missing and SUBMIT_BASELINE_IF_MISSING=0" >&2
    exit 2
  fi
else
  echo "using existing HLT baseline checkpoint: ${HLT_BASELINE_DIR}/best_model_val.pt"
fi

variant_dependency="${UPSTREAM_DEPENDENCY}"
if [[ -n "${baseline_jid}" ]]; then
  variant_dependency="$(fresh_join_by_colon "${variant_dependency}" "${baseline_jid}")"
fi
variant_dependency="${variant_dependency#:}"
variant_dependency="${variant_dependency%:}"

read -r -a variant_args <<< "${RECO7_VARIANTS}"
variant_job_ids=()
for variant in "${variant_args[@]}"; do
  fresh_refuse_existing_dir "${RECO7_ROOT}/${variant}"
  mapfile -t args < <(dep_args "${variant_dependency}" "${SCRIPT_DIR}/run_train_fresh_reco7_variant.sh" "${variant}")
  jid="$(submit_job "reco7_${variant}" "${args[@]}")"
  variant_job_ids+=("${jid}")
  echo "submitted reco7_${variant}=${jid}"
done

fusion_deps=("${variant_job_ids[@]}")
if [[ -n "${baseline_jid}" ]]; then
  fusion_deps+=("${baseline_jid}")
fi
fusion_dependency="$(fresh_join_by_colon "${fusion_deps[@]}")"
fresh_refuse_existing_dir "${RECO7_FUSION_DIR}"
fusion_jid="$(submit_job "reco7_fusion" --dependency="afterok:${fusion_dependency}" "${SCRIPT_DIR}/run_fuse_fresh_samehlt7_plus_hlt.sh")"

audit_deps=("${fusion_jid}")
if [[ -n "${HLT5_FUSION_JOB_ID}" ]]; then
  audit_deps+=("${HLT5_FUSION_JOB_ID}")
fi
audit_dependency="$(fresh_join_by_colon "${audit_deps[@]}")"
fresh_refuse_existing_dir "${RECO7_AUDIT_DIR}"
audit_export="ALL"
if [[ -z "${HLT5_FUSION_JOB_ID}" ]]; then
  audit_export="ALL,RUN_HLT5_AUDIT=0,REQUIRE_HLT5_AUDIT=0"
fi
audit_jid="$(submit_job "reco7_audit" --dependency="afterok:${audit_dependency}" --export="${audit_export}" "${SCRIPT_DIR}/run_audit_fresh_samehlt7_plus_hlt.sh")"

echo "variant_job_ids=${variant_job_ids[*]}"
echo "reco7_fusion_job_id=${fusion_jid}"
echo "audit_job_id=${audit_jid}"
echo "expected_reco7_fusion_output=${RECO7_FUSION_DIR}"
echo "expected_reco7_audit_output=${RECO7_AUDIT_DIR}"
