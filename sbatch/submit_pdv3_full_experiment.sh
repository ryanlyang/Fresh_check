#!/usr/bin/env bash
# Submit the complete PDV3 Step 7 graph for one root: inputs, teachers, students, report.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_ENV:=atlas_kd}"
export CONDA_ENV
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

fresh_prepare_submitter

: "${PDV3_UPSTREAM_DEPENDENCY:=${UPSTREAM_DEPENDENCY:-}}"
: "${PDV3_SUBMIT_STEP1:=1}"
: "${PDV3_SUBMIT_STEP2:=1}"
: "${PDV3_SUBMIT_STUDENTS:=1}"
: "${PDV3_SUBMIT_REPORT:=1}"

dependency_token_is_valid() {
  local token="$1"
  if [[ "${token}" =~ ^[0-9]+$ ]]; then
    return 0
  fi
  if fresh_is_dry_run && [[ "${token}" =~ ^DRYRUN_[A-Za-z0-9_]+$ ]]; then
    return 0
  fi
  return 1
}

validate_dependency_list() {
  local label="$1"
  local dependency="$2"
  if [[ -z "${dependency}" ]]; then
    return 0
  fi
  local old_ifs="${IFS}"
  local tokens=()
  IFS=':'
  read -r -a tokens <<< "${dependency}"
  IFS="${old_ifs}"
  local token
  for token in "${tokens[@]}"; do
    if [[ -z "${token}" ]] || ! dependency_token_is_valid "${token}"; then
      echo "Invalid Slurm dependency for ${label}: '${dependency}'." >&2
      return 2
    fi
  done
}

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
  if ! output="$(sbatch "$@")"; then
    echo "Failed to submit ${label}." >&2
    return 2
  fi
  echo "${output}" >&2
  local job_id
  job_id="$(echo "${output}" | awk '{print $NF}')"
  if ! dependency_token_is_valid "${job_id}"; then
    echo "Failed to submit ${label}; expected a Slurm job ID but got '${job_id:-empty}'." >&2
    return 2
  fi
  printf '%s\n' "${job_id}"
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
  validate_dependency_list "afterok" "${dependency}"
  if [[ -n "${dependency}" ]]; then
    printf '%s\n' --dependency="afterok:${dependency}"
  fi
  printf '%s\n' "$@"
}

dependency_or_empty() {
  local token="$1"
  if dependency_token_is_valid "${token}"; then
    printf '%s\n' "${token}"
  fi
}

field_from_submission_output() {
  local output="$1"
  local field="$2"
  echo "${output}" | awk -F': ' -v key="${field}" '$1 ~ ("^[[:space:]]*" key "$") {print $2; exit}'
}

job_tokens_from_output() {
  local output="$1"
  echo "${output}" \
    | awk '
        /^[[:space:]]+job_ids:/ {in_jobs=1; next}
        /^[[:space:]]+dependency_summary:/ {in_jobs=0}
        /^[[:space:]]+expected_jobs:/ {in_jobs=0}
        in_jobs {print}
      ' \
    | tr ' ' '\n' \
    | tr -d ',' \
    | awk '/^[0-9]+$/ || /^DRYRUN_[A-Za-z0-9_]+$/ {print}'
}

pdv3_student_output_complete() {
  local output_dir="$1"
  [[ -f "${output_dir}/run_report.json" ]] \
    && [[ -f "${output_dir}/best_model_val.pt" ]] \
    && [[ -f "${output_dir}/last.pt" ]] \
    && [[ -f "${output_dir}/model_val_report.json" ]] \
    && [[ -f "${output_dir}/final_test_report.json" ]] \
    && [[ -f "${output_dir}/config.json" ]] \
    && [[ -f "${output_dir}/training_curves.json" ]]
}

archive_incomplete_student_output() {
  local output_dir="$1"
  if fresh_is_dry_run || [[ ! -d "${output_dir}" ]]; then
    return 0
  fi
  if pdv3_student_output_complete "${output_dir}"; then
    return 0
  fi
  local archived="${output_dir}_incomplete_$(date +%Y%m%d_%H%M%S)"
  echo "found incomplete existing PDV3 student output; moving ${output_dir} to ${archived}" >&2
  mv "${output_dir}" "${archived}"
}

submitter_lock_dir="${PDV3_ROOT}/submission_logs/pdv3_full_$(date +%Y%m%d_%H%M%S)"
fresh_claim_new_dir "${submitter_lock_dir}"
if ! fresh_is_dry_run; then
  {
    echo "created_at=$(date -Is)"
    echo "project_dir=${PROJECT_DIR}"
    echo "source_commit=$(fresh_source_commit)"
    echo "source_status_hash=$(fresh_source_status_hash)"
    echo "root=${PDV3_ROOT}"
    echo "data_dir=${PDV3_DATA_DIR}"
    echo "upstream_dependency=${PDV3_UPSTREAM_DEPENDENCY}"
    echo "hlt_profile=${PDV3_HLT_PROFILE}"
    echo "hlt_degradation_strength=${PDV3_HLT_DEGRADATION_STRENGTH}"
    echo "model_train_size=${PDV3_MODEL_TRAIN_SIZE}"
    echo "model_val_size=${PDV3_MODEL_VAL_SIZE}"
    echo "final_test_size=${PDV3_FINAL_TEST_SIZE}"
    echo "student_variants=${PDV3_STUDENT_VARIANTS}"
  } > "${submitter_lock_dir}/metadata.txt"
fi

echo "pdv3_full_submission_start:"
echo "  root: ${PDV3_ROOT}"
echo "  upstream_dependency: ${PDV3_UPSTREAM_DEPENDENCY:-none}"
echo "  hlt_profile: ${PDV3_HLT_PROFILE}"
echo "  hlt_degradation_strength: ${PDV3_HLT_DEGRADATION_STRENGTH}"
echo "  sizes: ${PDV3_MODEL_TRAIN_SIZE}/${PDV3_MODEL_VAL_SIZE}/${PDV3_FINAL_TEST_SIZE}"

step1_output=""
step1_audit_value=""
step1_dependency=""
if fresh_bool_enabled "${PDV3_SUBMIT_STEP1}"; then
  step1_output="$(
    PDV3_UPSTREAM_DEPENDENCY="${PDV3_UPSTREAM_DEPENDENCY}" \
      bash "${SCRIPT_DIR}/submit_pdv3_step1_inputs.sh"
  )"
  echo "${step1_output}"
  step1_audit_value="$(field_from_submission_output "${step1_output}" "step1_audit")"
  step1_dependency="$(dependency_or_empty "${step1_audit_value}")"
else
  fresh_require_file "${PDV3_STEP1_AUDIT_DIR}/pdv3_step1_input_audit_report.json"
fi

step2_output=""
teacher_dependencies=""
if fresh_bool_enabled "${PDV3_SUBMIT_STEP2}"; then
  step2_output="$(
    PDV3_STEP1_DEPENDENCY="${step1_dependency}" \
      bash "${SCRIPT_DIR}/submit_pdv3_step2_teachers.sh"
  )"
  echo "${step2_output}"
  mapfile -t teacher_tokens < <(job_tokens_from_output "${step2_output}")
  teacher_dependencies="$(join_nonempty_by_colon "${teacher_tokens[@]:-}")"
else
  fresh_require_file "${PDV3_TEACHERS_DIR}/hlt_part_teacher_10class/best_model_val.pt"
  fresh_require_file "${PDV3_TEACHERS_DIR}/offline_part_teacher_10class/best_model_val.pt"
  fresh_require_file "${PDV3_TEACHER_LOGITS_DIR}/dual_view_logit_teacher_10class/teacher_logit_manifest.json"
  fresh_require_file "${PDV3_TEACHER_LOGITS_DIR}/particle_dual_view_teacher_10class/particle_dual_view_cache_manifest.json"
  fresh_require_file "${PDV3_TEACHER_REPRESENTATIONS_DIR}/particle_dual_view_teacher_10class/teacher_representation_manifest.json"
fi

student_dependencies="${teacher_dependencies}"
student_job_ids=()
student_skip_count=0
if fresh_bool_enabled "${PDV3_SUBMIT_STUDENTS}"; then
  fresh_split_words student_variants "${PDV3_STUDENT_VARIANTS}"
  for variant in "${student_variants[@]}"; do
    output_dir="${PDV3_STUDENTS_DIR}/${variant}"
    if fresh_bool_enabled "${SKIP_EXISTING}" && ! fresh_is_dry_run && pdv3_student_output_complete "${output_dir}"; then
      student_skip_count=$((student_skip_count + 1))
      echo "skipped pdv3_student_${variant}; found complete existing output: ${output_dir}" >&2
      continue
    fi
    if fresh_bool_enabled "${SKIP_EXISTING}"; then
      archive_incomplete_student_output "${output_dir}"
    fi
    mapfile -t student_args < <(afterok_args "${student_dependencies}" "${SCRIPT_DIR}/run_pdv3_train_student.sh" "${variant}")
    student_jid="$(submit_job "pdv3_student_${variant}" "${student_args[@]}")"
    echo "submitted pdv3_student_${variant}=${student_jid}"
    student_job_ids+=("${student_jid}")
  done
else
  fresh_require_dir "${PDV3_STUDENTS_DIR}"
fi

report_dependency="$(join_nonempty_by_colon "${student_job_ids[@]:-}")"
report_jid=""
if fresh_bool_enabled "${PDV3_SUBMIT_REPORT}"; then
  if fresh_bool_enabled "${SKIP_EXISTING}" && ! fresh_is_dry_run && [[ -f "${PDV3_FINAL_REPORT_DIR}/pdv3_report.json" ]]; then
    echo "skipped pdv3_report; found existing artifact: ${PDV3_FINAL_REPORT_DIR}/pdv3_report.json" >&2
  else
    mapfile -t report_args < <(afterok_args "${report_dependency}" "${SCRIPT_DIR}/run_pdv3_write_report.sh")
    report_jid="$(submit_job "pdv3_report" "${report_args[@]}")"
    echo "submitted pdv3_report=${report_jid}"
  fi
fi

cat <<SUMMARY
pdv3_full_submission:
  root: ${PDV3_ROOT}
  data_dir: ${PDV3_DATA_DIR}
  hlt_profile: ${PDV3_HLT_PROFILE}
  hlt_degradation_strength: ${PDV3_HLT_DEGRADATION_STRENGTH}
  sizes:
    model_train: ${PDV3_MODEL_TRAIN_SIZE}
    model_val: ${PDV3_MODEL_VAL_SIZE}
    final_test: ${PDV3_FINAL_TEST_SIZE}
  job_ids:
    step1_audit: ${step1_audit_value:-skipped_or_existing}
    teacher_dependencies: ${teacher_dependencies:-skipped_or_existing}
    students: $(fresh_join_by_space "${student_job_ids[@]:-}")
    final_report: ${report_jid:-skipped_or_existing}
  expected_jobs:
    student_variants: ${PDV3_STUDENT_VARIANTS}
    students_submitted: ${#student_job_ids[@]}
    students_skipped_existing: ${student_skip_count}
  outputs:
    root: ${PDV3_ROOT}
    students: ${PDV3_STUDENTS_DIR}
    final_report: ${PDV3_FINAL_REPORT_DIR}
    logs: ${PROJECT_DIR}/fresh_check_logs
SUMMARY
