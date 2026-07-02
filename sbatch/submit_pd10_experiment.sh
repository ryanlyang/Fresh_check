#!/usr/bin/env bash
# Submit the full PD10 privileged-distillation experiment graph.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

fresh_prepare_submitter

if ! fresh_bool_enabled "${CONFIRM_FINAL_TEST}"; then
  echo "Refusing to submit PD10 final-test graph without CONFIRM_FINAL_TEST=1." >&2
  exit 2
fi

submit_count=0
skip_count=0
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
    skip_count=$((skip_count + 1))
    echo "skipped ${label}; found existing artifact: ${path}" >&2
    return 0
  fi
  return 1
}

fresh_split_words teacher_args "${PD10_TEACHER_TARGETS}"
fresh_split_words student_specs "${PD10_STUDENT_SPECS}"

submitter_lock_dir="${PD10_ROOT}/submission_logs/pd10_full_$(date +%Y%m%d_%H%M%S)"
fresh_claim_new_dir "${submitter_lock_dir}"
if ! fresh_is_dry_run; then
  {
    echo "created_at=$(date -Is)"
    echo "project_dir=${PROJECT_DIR}"
    echo "source_commit=$(fresh_source_commit)"
    echo "source_status_hash=$(fresh_source_status_hash)"
    echo "root=${PD10_ROOT}"
    echo "skip_existing=${SKIP_EXISTING}"
    echo "confirm_final_test=${CONFIRM_FINAL_TEST}"
    echo "manifest=${PD10_MANIFEST_PATH}"
    echo "hlt_cache=${PD10_HLT_CACHE_DIR}"
    echo "teachers_dir=${PD10_TEACHERS_DIR}"
    echo "teacher_logits_dir=${PD10_TEACHER_LOGITS_DIR}"
    echo "students_dir=${PD10_STUDENTS_DIR}"
    echo "final_report_dir=${PD10_FINAL_REPORT_DIR}"
    echo "teacher_targets=$(fresh_join_by_space "${teacher_args[@]}")"
    echo "student_specs=$(fresh_join_by_space "${student_specs[@]}")"
    echo "model_train_size=${PD10_MODEL_TRAIN_SIZE}"
    echo "model_val_size=${PD10_MODEL_VAL_SIZE}"
    echo "final_test_size=${PD10_FINAL_TEST_SIZE}"
    echo "hlt_degradation_strength=${PD10_HLT_DEGRADATION_STRENGTH}"
  } > "${submitter_lock_dir}/metadata.txt"
fi

if ! fresh_bool_enabled "${SKIP_EXISTING}"; then
  fresh_refuse_existing_path "${PD10_MANIFEST_PATH}"
  fresh_refuse_existing_dir "${PD10_HLT_CACHE_DIR}"
  fresh_refuse_existing_dir "${PD10_STEP2_AUDIT_DIR}"
  fresh_refuse_existing_dir "${PD10_TEACHERS_DIR}"
  fresh_refuse_existing_dir "${PD10_TEACHER_LOGITS_DIR}"
  fresh_refuse_existing_dir "${PD10_STUDENTS_DIR}"
  fresh_refuse_existing_dir "${PD10_FINAL_REPORT_DIR}"
fi

split_jid=""
if ! skip_existing_artifact "pd10_splits" "${PD10_MANIFEST_PATH}"; then
  split_jid="$(submit_job "pd10_splits" "${SCRIPT_DIR}/run_pd10_build_splits.sh")"
  echo "submitted pd10_splits=${split_jid}"
fi

cache_dep="$(join_nonempty_by_colon "${split_jid}")"
cache_jid=""
if ! skip_existing_artifact "pd10_hlt_cache" "${PD10_HLT_CACHE_DIR}/final_test_fixed_hlt_metadata.json"; then
  mapfile -t cache_args < <(afterok_args "${cache_dep}" "${SCRIPT_DIR}/run_pd10_build_hlt_cache.sh")
  cache_jid="$(submit_job "pd10_hlt_cache" "${cache_args[@]}")"
  echo "submitted pd10_hlt_cache=${cache_jid}"
fi

audit_dep="$(join_nonempty_by_colon "${cache_jid}")"
audit_jid=""
if ! skip_existing_artifact "pd10_step2_audit" "${PD10_STEP2_AUDIT_DIR}/pd10_step2_audit_report.json"; then
  mapfile -t audit_args < <(afterok_args "${audit_dep}" "${SCRIPT_DIR}/run_pd10_audit_splits_hlt_cache.sh")
  audit_jid="$(submit_job "pd10_step2_audit" "${audit_args[@]}")"
  echo "submitted pd10_step2_audit=${audit_jid}"
fi

declare -A teacher_job_by_target=()
teacher_job_ids=()
for teacher in "${teacher_args[@]}"; do
  model_name="$(fresh_pd10_teacher_model_name "${teacher}")"
  teacher_done="${PD10_TEACHERS_DIR}/${model_name}/run_report.json"
  teacher_jid=""
  if ! skip_existing_artifact "pd10_teacher_${teacher}" "${teacher_done}"; then
    mapfile -t teacher_submit_args < <(
      afterok_args "${audit_jid}" "${SCRIPT_DIR}/run_pd10_train_teacher.sh" "${teacher}"
    )
    teacher_jid="$(submit_job "pd10_teacher_${teacher}" "${teacher_submit_args[@]}")"
    teacher_job_ids+=("${teacher_jid}")
    echo "submitted pd10_teacher_${teacher}=${teacher_jid}"
  fi
  teacher_job_by_target["${teacher}"]="${teacher_jid}"
done

declare -A logit_job_by_target=()
teacher_logit_job_ids=()
for teacher in "${teacher_args[@]}"; do
  model_name="$(fresh_pd10_teacher_model_name "${teacher}")"
  logit_done="${PD10_TEACHER_LOGITS_DIR}/${model_name}/teacher_logit_manifest.json"
  logit_dep="$(join_nonempty_by_colon "${audit_jid}" "${teacher_job_by_target[${teacher}]}")"
  logit_jid=""
  if ! skip_existing_artifact "pd10_teacher_logits_${teacher}" "${logit_done}"; then
    mapfile -t logit_args < <(
      afterok_args "${logit_dep}" "${SCRIPT_DIR}/run_pd10_cache_teacher_logits.sh" "${teacher}"
    )
    logit_jid="$(submit_job "pd10_teacher_logits_${teacher}" "${logit_args[@]}")"
    teacher_logit_job_ids+=("${logit_jid}")
    echo "submitted pd10_teacher_logits_${teacher}=${logit_jid}"
  fi
  logit_job_by_target["${teacher}"]="${logit_jid}"
done

dual_dep="$(join_nonempty_by_colon "${logit_job_by_target[hlt]:-}" "${logit_job_by_target[offline]:-}")"
dual_jid=""
if ! skip_existing_artifact "pd10_dual_view_teacher" "${PD10_DUAL_VIEW_TEACHER_LOGITS_DIR}/teacher_logit_manifest.json"; then
  mapfile -t dual_args < <(afterok_args "${dual_dep}" "${SCRIPT_DIR}/run_pd10_train_dual_view_teacher.sh")
  dual_jid="$(submit_job "pd10_dual_view_teacher" "${dual_args[@]}")"
  echo "submitted pd10_dual_view_teacher=${dual_jid}"
fi

student_job_ids=()
student_variants=()
for spec in "${student_specs[@]}"; do
  old_ifs="${IFS}"
  IFS='|'
  read -r student_init teacher_target target_mode temperature kd_alpha top_k variant_name <<< "${spec}"
  IFS="${old_ifs}"
  if [[ -z "${student_init}" || -z "${teacher_target}" || -z "${variant_name}" ]]; then
    echo "Malformed PD10 student spec: ${spec}" >&2
    exit 2
  fi
  student_variants+=("${variant_name}")
  student_done="${PD10_STUDENTS_DIR}/${variant_name}/run_report.json"
  student_dep_parts=("${audit_jid}")
  if [[ "${student_init}" == "warm_start" ]]; then
    student_dep_parts+=("${teacher_job_by_target[hlt]:-}")
  fi
  case "${teacher_target}" in
    none) ;;
    hlt|offline) student_dep_parts+=("${logit_job_by_target[${teacher_target}]:-}") ;;
    dual_view) student_dep_parts+=("${dual_jid}") ;;
    *)
      echo "Unknown PD10 student teacher target in spec: ${teacher_target}" >&2
      exit 2
      ;;
  esac
  student_dep="$(join_nonempty_by_colon "${student_dep_parts[@]}")"
  if ! skip_existing_artifact "pd10_student_${variant_name}" "${student_done}"; then
    mapfile -t student_args < <(
      afterok_args "${student_dep}" "${SCRIPT_DIR}/run_pd10_train_student.sh" "${spec}"
    )
    student_jid="$(submit_job "pd10_student_${variant_name}" "${student_args[@]}")"
    student_job_ids+=("${student_jid}")
    echo "submitted pd10_student_${variant_name}=${student_jid}"
  fi
done

report_dep="$(join_nonempty_by_colon "${audit_jid}" "${teacher_job_ids[@]}" "${teacher_logit_job_ids[@]}" "${dual_jid}" "${student_job_ids[@]}")"
report_jid=""
if ! skip_existing_artifact "pd10_final_report" "${PD10_FINAL_REPORT_DIR}/pd10_report.json"; then
  mapfile -t report_args < <(afterok_args "${report_dep}" "${SCRIPT_DIR}/run_pd10_write_report.sh")
  report_jid="$(submit_job "pd10_final_report" "${report_args[@]}")"
  echo "submitted pd10_final_report=${report_jid}"
fi

cat <<SUMMARY
pd10_experiment_submission:
  root: ${PD10_ROOT}
  skip_existing: ${SKIP_EXISTING}
  confirm_final_test: ${CONFIRM_FINAL_TEST}
  job_ids:
    split_manifest: ${split_jid:-skipped_existing}
    hlt_cache: ${cache_jid:-skipped_existing}
    step2_audit: ${audit_jid:-skipped_existing}
    teachers: $(fresh_join_by_space "${teacher_job_ids[@]}")
    teacher_logits: $(fresh_join_by_space "${teacher_logit_job_ids[@]}")
    dual_view_teacher: ${dual_jid:-skipped_existing}
    students: $(fresh_join_by_space "${student_job_ids[@]}")
    final_report: ${report_jid:-skipped_existing}
  dependency_summary:
    hlt_cache_afterok: ${cache_dep:-none}
    step2_audit_afterok: ${audit_dep:-none}
    teachers_afterok: ${audit_jid:-none}
    teacher_logits_after_teachers_and_audit: true
    dual_view_afterok: ${dual_dep:-none}
    students_afterok: teacher-specific plus audit
    final_report_afterok: ${report_dep:-none}
  expected_jobs:
    teachers: ${#teacher_args[@]}
    students: ${#student_specs[@]}
    total_submitted: ${submit_count}
    total_skipped_existing: ${skip_count}
  student_variants: $(fresh_join_by_space "${student_variants[@]}")
  split_sizes:
    model_train: ${PD10_MODEL_TRAIN_SIZE}
    model_val: ${PD10_MODEL_VAL_SIZE}
    final_test: ${PD10_FINAL_TEST_SIZE}
  outputs:
    manifest: ${PD10_MANIFEST_PATH}
    hlt_cache: ${PD10_HLT_CACHE_DIR}
    audit: ${PD10_STEP2_AUDIT_DIR}/pd10_step2_audit_report.json
    teachers: ${PD10_TEACHERS_DIR}
    teacher_logits: ${PD10_TEACHER_LOGITS_DIR}
    dual_view_teacher: ${PD10_DUAL_VIEW_TEACHER_DIR}
    students: ${PD10_STUDENTS_DIR}
    final_report: ${PD10_FINAL_REPORT_DIR}/pd10_report.json
    logs: ${PROJECT_DIR}/fresh_check_logs
SUMMARY
