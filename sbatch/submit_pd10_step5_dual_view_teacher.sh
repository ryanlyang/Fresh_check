#!/usr/bin/env bash
# Submit the PD10 dual-view logit-fusion teacher job.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${UPSTREAM_DEPENDENCY:=}"

fresh_prepare_submitter

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

dep_args() {
  local dependency="$1"
  shift
  if [[ -n "${dependency}" ]]; then
    printf '%s\n' --dependency="afterok:${dependency}"
  fi
  printf '%s\n' "$@"
}

submitter_lock_dir="${PD10_ROOT}/.step5_dual_view_submission_lock"
fresh_claim_new_dir "${submitter_lock_dir}"
if ! fresh_is_dry_run; then
  {
    echo "created_at=$(date -Is)"
    echo "project_dir=${PROJECT_DIR}"
    echo "source_commit=$(fresh_source_commit)"
    echo "teacher_logits_dir=${PD10_TEACHER_LOGITS_DIR}"
    echo "dual_view_teacher_dir=${PD10_DUAL_VIEW_TEACHER_DIR}"
    echo "dual_view_teacher_logits_dir=${PD10_DUAL_VIEW_TEACHER_LOGITS_DIR}"
    echo "upstream_dependency=${UPSTREAM_DEPENDENCY:-none}"
  } > "${submitter_lock_dir}/metadata.txt"
fi

if [[ -z "${UPSTREAM_DEPENDENCY}" ]]; then
  fresh_require_file "${PD10_TEACHER_LOGITS_DIR}/hlt_part_teacher_10class/teacher_logit_manifest.json"
  fresh_require_file "${PD10_TEACHER_LOGITS_DIR}/offline_part_teacher_10class/teacher_logit_manifest.json"
fi
fresh_refuse_existing_dir "${PD10_DUAL_VIEW_TEACHER_DIR}"
if fresh_bool_enabled "${PD10_DUAL_VIEW_NO_SKIP_EXISTING_PREDICTIONS}"; then
  fresh_refuse_existing_dir "${PD10_DUAL_VIEW_TEACHER_LOGITS_DIR}"
fi

mapfile -t args < <(dep_args "${UPSTREAM_DEPENDENCY}" "${SCRIPT_DIR}/run_pd10_train_dual_view_teacher.sh")
jid="$(submit_job "pd10_dual_view_logit_teacher" "${args[@]}")"
echo "submitted pd10_dual_view_logit_teacher=${jid}"

cat <<SUMMARY
pd10_step5_dual_view_teacher_submission:
  dual_view_job_id: ${jid}
  dependency_summary:
    dual_view_afterok: ${UPSTREAM_DEPENDENCY:-none}
  output_dirs:
    root: ${PD10_ROOT}
    teacher: ${PD10_DUAL_VIEW_TEACHER_DIR}
    teacher_logits: ${PD10_DUAL_VIEW_TEACHER_LOGITS_DIR}
    logs: ${PROJECT_DIR}/fresh_check_logs
SUMMARY
