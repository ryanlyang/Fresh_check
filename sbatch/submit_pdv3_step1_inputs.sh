#!/usr/bin/env bash
# Submit only the PDV3 Step 1 split/cache/audit graph.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

fresh_prepare_submitter

: "${PDV3_UPSTREAM_DEPENDENCY:=${UPSTREAM_DEPENDENCY:-}}"

submit_job() {
  local label="$1"
  shift
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

join_nonempty_by_colon() {
  local values=()
  local item
  for item in "$@"; do
    if [[ -n "${item}" ]]; then
      values+=("${item}")
    fi
  done
  if [[ "${#values[@]}" -eq 0 ]]; then
    return 0
  fi
  fresh_join_by_colon "${values[@]}"
}

afterok_args() {
  local dependency="$1"
  shift
  if [[ -n "${dependency}" ]]; then
    printf '%s\n' --dependency="afterok:${dependency}"
  fi
  printf '%s\n' "$@"
}

skip_existing_artifact() {
  local label="$1"
  local path="$2"
  if fresh_bool_enabled "${SKIP_EXISTING}" && ! fresh_is_dry_run && [[ -e "${path}" ]]; then
    echo "skipped ${label}; found existing artifact: ${path}" >&2
    printf 'skipped_existing\n'
    return 0
  fi
  return 1
}

submitter_lock_dir="${PDV3_ROOT}/submission_logs/pdv3_step1_$(date +%Y%m%d_%H%M%S)"
fresh_claim_new_dir "${submitter_lock_dir}"
if ! fresh_is_dry_run; then
  {
    echo "created_at=$(date -Is)"
    echo "project_dir=${PROJECT_DIR}"
    echo "source_commit=$(fresh_source_commit)"
    echo "source_status_hash=$(fresh_source_status_hash)"
    echo "root=${PDV3_ROOT}"
    echo "data_dir=${PDV3_DATA_DIR}"
    echo "manifest=${PDV3_MANIFEST_PATH}"
    echo "hlt_cache=${PDV3_HLT_CACHE_DIR}"
    echo "offline_cache=${PDV3_OFFLINE_CACHE_DIR}"
    echo "audit=${PDV3_STEP1_AUDIT_DIR}"
    echo "upstream_dependency=${PDV3_UPSTREAM_DEPENDENCY}"
    echo "hlt_degradation_strength=${PDV3_HLT_DEGRADATION_STRENGTH}"
    echo "model_train_size=${PDV3_MODEL_TRAIN_SIZE}"
    echo "model_val_size=${PDV3_MODEL_VAL_SIZE}"
    echo "final_test_size=${PDV3_FINAL_TEST_SIZE}"
  } > "${submitter_lock_dir}/metadata.txt"
fi

if ! fresh_bool_enabled "${SKIP_EXISTING}"; then
  fresh_refuse_existing_path "${PDV3_MANIFEST_PATH}"
  fresh_refuse_existing_dir "${PDV3_HLT_CACHE_DIR}"
  fresh_refuse_existing_dir "${PDV3_OFFLINE_CACHE_DIR}"
  fresh_refuse_existing_dir "${PDV3_STEP1_AUDIT_DIR}"
fi

split_jid=""
if skip_existing_artifact "pdv3_splits" "${PDV3_MANIFEST_PATH}" >/dev/null; then
  split_jid=""
else
  mapfile -t split_args < <(afterok_args "${PDV3_UPSTREAM_DEPENDENCY}" "${SCRIPT_DIR}/run_pdv3_build_splits.sh")
  split_jid="$(submit_job "pdv3_splits" "${split_args[@]}")"
  echo "submitted pdv3_splits=${split_jid}"
fi

cache_dep="$(join_nonempty_by_colon "${PDV3_UPSTREAM_DEPENDENCY}" "${split_jid}")"
hlt_jid=""
if skip_existing_artifact "pdv3_hlt_cache" "${PDV3_HLT_CACHE_DIR}/final_test_fixed_hlt_metadata.json" >/dev/null; then
  hlt_jid=""
else
  mapfile -t hlt_args < <(afterok_args "${cache_dep}" "${SCRIPT_DIR}/run_pdv3_build_hlt_cache.sh")
  hlt_jid="$(submit_job "pdv3_hlt_cache" "${hlt_args[@]}")"
  echo "submitted pdv3_hlt_cache=${hlt_jid}"
fi

offline_jid=""
if skip_existing_artifact "pdv3_offline_cache" "${PDV3_OFFLINE_CACHE_DIR}/final_test_offline_metadata.json" >/dev/null; then
  offline_jid=""
else
  mapfile -t offline_args < <(afterok_args "${cache_dep}" "${SCRIPT_DIR}/run_pdv3_cache_offline_inputs.sh")
  offline_jid="$(submit_job "pdv3_offline_cache" "${offline_args[@]}")"
  echo "submitted pdv3_offline_cache=${offline_jid}"
fi

audit_dep="$(join_nonempty_by_colon "${PDV3_UPSTREAM_DEPENDENCY}" "${hlt_jid}" "${offline_jid}")"
audit_jid=""
if skip_existing_artifact "pdv3_step1_audit" "${PDV3_STEP1_AUDIT_DIR}/pdv3_step1_input_audit_report.json" >/dev/null; then
  audit_jid=""
else
  mapfile -t audit_args < <(afterok_args "${audit_dep}" "${SCRIPT_DIR}/run_pdv3_audit_inputs.sh")
  audit_jid="$(submit_job "pdv3_step1_audit" "${audit_args[@]}")"
  echo "submitted pdv3_step1_audit=${audit_jid}"
fi

cat <<SUMMARY
pdv3_step1_submission:
  root: ${PDV3_ROOT}
  data_dir: ${PDV3_DATA_DIR}
  hlt_degradation_strength: ${PDV3_HLT_DEGRADATION_STRENGTH}
  job_ids:
    split_manifest: ${split_jid:-skipped_existing}
    hlt_cache: ${hlt_jid:-skipped_existing}
    offline_cache: ${offline_jid:-skipped_existing}
    step1_audit: ${audit_jid:-skipped_existing}
  outputs:
    manifest: ${PDV3_MANIFEST_PATH}
    hlt_cache: ${PDV3_HLT_CACHE_DIR}
    offline_cache: ${PDV3_OFFLINE_CACHE_DIR}
    audit: ${PDV3_STEP1_AUDIT_DIR}/pdv3_step1_input_audit_report.json
SUMMARY
