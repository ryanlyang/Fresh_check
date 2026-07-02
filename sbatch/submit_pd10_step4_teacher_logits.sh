#!/usr/bin/env bash
# Submit PD10 HLT/offline teacher-logit cache jobs.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${UPSTREAM_DEPENDENCY:=}"

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

fresh_split_words teacher_args "${PD10_TEACHER_LOGIT_TARGETS}"
submitter_lock_dir="${PD10_ROOT}/.step4_teacher_logit_submission_lock"
fresh_claim_new_dir "${submitter_lock_dir}"
if ! fresh_is_dry_run; then
  {
    echo "created_at=$(date -Is)"
    echo "project_dir=${PROJECT_DIR}"
    echo "source_commit=$(fresh_source_commit)"
    echo "teacher_targets=$(fresh_join_by_space "${teacher_args[@]}")"
    echo "teacher_logits_dir=${PD10_TEACHER_LOGITS_DIR}"
    echo "teachers_dir=${PD10_TEACHERS_DIR}"
    echo "manifest=${PD10_MANIFEST_PATH}"
    echo "hlt_cache_dir=${PD10_HLT_CACHE_DIR}"
    echo "splits=${PD10_TEACHER_LOGIT_SPLITS}"
    echo "batch_size=${PD10_TEACHER_LOGIT_BATCH_SIZE}"
    echo "device=${PD10_TEACHER_LOGIT_DEVICE}"
    echo "no_skip_existing=${PD10_TEACHER_LOGIT_NO_SKIP_EXISTING}"
  } > "${submitter_lock_dir}/metadata.txt"
fi

teacher_logit_job_ids=()
for teacher in "${teacher_args[@]}"; do
  model_name="$(fresh_pd10_teacher_model_name "${teacher}")"
  if [[ -z "${UPSTREAM_DEPENDENCY}" ]]; then
    fresh_require_file "${PD10_TEACHERS_DIR}/${model_name}/best_model_val.pt"
  fi
  if fresh_bool_enabled "${PD10_TEACHER_LOGIT_NO_SKIP_EXISTING}"; then
    fresh_refuse_existing_dir "${PD10_TEACHER_LOGITS_DIR}/${model_name}"
  fi
  mapfile -t args < <(dep_args "${UPSTREAM_DEPENDENCY}" "${SCRIPT_DIR}/run_pd10_cache_teacher_logits.sh" "${teacher}")
  jid="$(submit_job "pd10_teacher_logits_${teacher}" "${args[@]}")"
  teacher_logit_job_ids+=("${jid}")
  echo "submitted pd10_teacher_logits_${teacher}=${jid}"
done

cat <<SUMMARY
pd10_step4_teacher_logits_submission:
  teacher_logit_job_ids: $(fresh_join_by_space "${teacher_logit_job_ids[@]}")
  total_submitted: ${submit_count}
  dependency_summary:
    teacher_logits_afterok: ${UPSTREAM_DEPENDENCY:-none}
  teacher_targets: $(fresh_join_by_space "${teacher_args[@]}")
  splits: ${PD10_TEACHER_LOGIT_SPLITS}
  output_dirs:
    root: ${PD10_ROOT}
    teachers: ${PD10_TEACHERS_DIR}
    teacher_logits: ${PD10_TEACHER_LOGITS_DIR}
    hlt_part_teacher_10class: ${PD10_TEACHER_LOGITS_DIR}/hlt_part_teacher_10class
    offline_part_teacher_10class: ${PD10_TEACHER_LOGITS_DIR}/offline_part_teacher_10class
    logs: ${PROJECT_DIR}/fresh_check_logs
SUMMARY
