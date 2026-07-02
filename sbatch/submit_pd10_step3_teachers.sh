#!/usr/bin/env bash
# Submit the PD10 HLT and offline ParT teacher jobs.

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

fresh_split_words teacher_args "${PD10_TEACHER_TARGETS}"
submitter_lock_dir="${PD10_ROOT}/.step3_teacher_submission_lock"
fresh_claim_new_dir "${submitter_lock_dir}"
if ! fresh_is_dry_run; then
  {
    echo "created_at=$(date -Is)"
    echo "project_dir=${PROJECT_DIR}"
    echo "source_commit=$(fresh_source_commit)"
    echo "teacher_targets=$(fresh_join_by_space "${teacher_args[@]}")"
    echo "manifest=${PD10_MANIFEST_PATH}"
    echo "hlt_cache_dir=${PD10_HLT_CACHE_DIR}"
    echo "teachers_dir=${PD10_TEACHERS_DIR}"
    echo "model_train_size=${PD10_MODEL_TRAIN_SIZE}"
    echo "model_val_size=${PD10_MODEL_VAL_SIZE}"
    echo "final_test_size=${PD10_FINAL_TEST_SIZE}"
    echo "hlt_degradation_strength=${PD10_HLT_DEGRADATION_STRENGTH}"
  } > "${submitter_lock_dir}/metadata.txt"
fi

teacher_job_ids=()
for teacher in "${teacher_args[@]}"; do
  model_name="$(fresh_pd10_teacher_model_name "${teacher}")"
  fresh_refuse_existing_dir "${PD10_TEACHERS_DIR}/${model_name}"
  mapfile -t args < <(dep_args "${UPSTREAM_DEPENDENCY}" "${SCRIPT_DIR}/run_pd10_train_teacher.sh" "${teacher}")
  jid="$(submit_job "pd10_teacher_${teacher}" "${args[@]}")"
  teacher_job_ids+=("${jid}")
  echo "submitted pd10_teacher_${teacher}=${jid}"
done

cat <<SUMMARY
pd10_step3_teachers_submission:
  teacher_job_ids: $(fresh_join_by_space "${teacher_job_ids[@]}")
  total_submitted: ${submit_count}
  dependency_summary:
    teachers_afterok: ${UPSTREAM_DEPENDENCY:-none}
  teacher_targets: $(fresh_join_by_space "${teacher_args[@]}")
  split_sizes:
    model_train: ${PD10_MODEL_TRAIN_SIZE}
    model_val: ${PD10_MODEL_VAL_SIZE}
    final_test: ${PD10_FINAL_TEST_SIZE}
  hlt_degradation_strength: ${PD10_HLT_DEGRADATION_STRENGTH}
  output_dirs:
    root: ${PD10_ROOT}
    teachers: ${PD10_TEACHERS_DIR}
    hlt_part_teacher_10class: ${PD10_TEACHERS_DIR}/hlt_part_teacher_10class
    offline_part_teacher_10class: ${PD10_TEACHERS_DIR}/offline_part_teacher_10class
    logs: ${PROJECT_DIR}/fresh_check_logs
SUMMARY
