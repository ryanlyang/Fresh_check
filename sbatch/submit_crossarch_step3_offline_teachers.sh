#!/usr/bin/env bash
# Submit the four cross-architecture offline teacher jobs.

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

fresh_split_words teacher_args "${CROSSARCH_OFFLINE_TEACHER_ARCHITECTURES}"
submitter_lock_dir="${CROSSARCH_ROOT}/.step3_offline_teacher_submission_lock"
fresh_claim_new_dir "${submitter_lock_dir}"
if ! fresh_is_dry_run; then
  {
    echo "created_at=$(date -Is)"
    echo "project_dir=${PROJECT_DIR}"
    echo "source_commit=$(fresh_source_commit)"
    echo "teacher_architectures=$(fresh_join_by_space "${teacher_args[@]}")"
    echo "manifest=${CROSSARCH_MANIFEST_PATH}"
    echo "offline_teacher_dir=${CROSSARCH_OFFLINE_TEACHER_DIR}"
    echo "model_train_size=${CROSSARCH_MODEL_TRAIN_SIZE}"
    echo "model_val_size=${CROSSARCH_MODEL_VAL_SIZE}"
  } > "${submitter_lock_dir}/metadata.txt"
fi

teacher_job_ids=()
for architecture in "${teacher_args[@]}"; do
  fresh_refuse_existing_dir "${CROSSARCH_OFFLINE_TEACHER_DIR}/${architecture}"
  mapfile -t args < <(dep_args "${UPSTREAM_DEPENDENCY}" "${SCRIPT_DIR}/run_crossarch_train_offline_teacher.sh" "${architecture}")
  jid="$(submit_job "crossarch_teacher_${architecture}" "${args[@]}")"
  teacher_job_ids+=("${jid}")
  echo "submitted crossarch_teacher_${architecture}=${jid}"
done

cat <<SUMMARY
crossarch_step3_offline_teachers_submission:
  teacher_job_ids: $(fresh_join_by_space "${teacher_job_ids[@]}")
  dependency_summary:
    teachers_afterok: ${UPSTREAM_DEPENDENCY:-none}
  architectures: $(fresh_join_by_space "${teacher_args[@]}")
  split_sizes:
    model_train: ${CROSSARCH_MODEL_TRAIN_SIZE}
    model_val: ${CROSSARCH_MODEL_VAL_SIZE}
  output_dirs:
    root: ${CROSSARCH_ROOT}
    offline_teachers: ${CROSSARCH_OFFLINE_TEACHER_DIR}
    logs: ${PROJECT_DIR}/fresh_check_logs
SUMMARY
