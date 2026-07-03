#!/usr/bin/env bash
# Submit the PD10-V2 representation-KD plus particle dual-view teacher graph.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_ENV:=atlas_kd}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${UPSTREAM_DEPENDENCY:=}"
: "${PD10_V2_GLOBAL_UPSTREAM_DEPENDENCY:=${UPSTREAM_DEPENDENCY}}"
: "${PD10_V2_PARTICLE_TEACHER_UPSTREAM_DEPENDENCY:=${PD10_V2_GLOBAL_UPSTREAM_DEPENDENCY}}"
: "${PD10_V2_DUAL_REP_UPSTREAM_DEPENDENCY:=${PD10_V2_GLOBAL_UPSTREAM_DEPENDENCY}}"
: "${PD10_V2_REPORT_ANCHOR_UPSTREAM_DEPENDENCY:=${PD10_V2_GLOBAL_UPSTREAM_DEPENDENCY}}"

fresh_prepare_submitter

if ! fresh_bool_enabled "${CONFIRM_FINAL_TEST}"; then
  echo "Refusing to submit PD10-V2 final-test graph without CONFIRM_FINAL_TEST=1." >&2
  exit 2
fi

submit_count=0
skip_count=0
submitted_job_id=""

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
      echo "Use colon-separated numeric job IDs, for example 12345:12346. Do not pass placeholder text such as HLT_TEACHER_JOB." >&2
      return 2
    fi
  done
}

validate_submitted_job_id() {
  local label="$1"
  local job_id="$2"
  if ! dependency_token_is_valid "${job_id}"; then
    echo "Failed to submit ${label}; expected a Slurm job ID but got '${job_id:-empty}'." >&2
    return 2
  fi
}

submit_job() {
  local label="$1"
  shift
  submit_count=$((submit_count + 1))
  if fresh_is_dry_run; then
    printf 'DRY_RUN sbatch %s: ' "${label}" >&2
    fresh_print_shell_command sbatch "$@" >&2
    printf '\n' >&2
    local clean_label="${label//[^A-Za-z0-9_]/_}"
    submitted_job_id="DRYRUN_${clean_label}"
    return 0
  fi
  local output
  if ! output="$(sbatch "$@")"; then
    echo "Failed to submit ${label}." >&2
    return 2
  fi
  echo "${output}" >&2
  submitted_job_id="$(echo "${output}" | awk '{print $NF}')"
  validate_submitted_job_id "${label}" "${submitted_job_id}"
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

fresh_require_file_unless_deferred() {
  local dependency="$1"
  local path="$2"
  if [[ -n "${dependency}" ]]; then
    return 0
  fi
  fresh_require_file "${path}"
}

fresh_require_dir_unless_deferred() {
  local dependency="$1"
  local path="$2"
  if [[ -n "${dependency}" ]]; then
    return 0
  fi
  fresh_require_dir "${path}"
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

require_artifact_or_submitted_job() {
  local label="$1"
  local artifact="$2"
  local job_id="$3"
  if [[ -e "${artifact}" ]]; then
    return 0
  fi
  if [[ -n "${job_id}" ]]; then
    validate_submitted_job_id "${label}" "${job_id}"
    return 0
  fi
  echo "Cannot queue jobs that need ${label}; neither an existing artifact nor a submitted prerequisite job is available." >&2
  echo "Missing artifact: ${artifact}" >&2
  return 2
}

validate_dependency_list "UPSTREAM_DEPENDENCY" "${UPSTREAM_DEPENDENCY}"
validate_dependency_list "PD10_V2_GLOBAL_UPSTREAM_DEPENDENCY" "${PD10_V2_GLOBAL_UPSTREAM_DEPENDENCY}"
validate_dependency_list "PD10_V2_PARTICLE_TEACHER_UPSTREAM_DEPENDENCY" "${PD10_V2_PARTICLE_TEACHER_UPSTREAM_DEPENDENCY}"
validate_dependency_list "PD10_V2_DUAL_REP_UPSTREAM_DEPENDENCY" "${PD10_V2_DUAL_REP_UPSTREAM_DEPENDENCY}"
validate_dependency_list "PD10_V2_REPORT_ANCHOR_UPSTREAM_DEPENDENCY" "${PD10_V2_REPORT_ANCHOR_UPSTREAM_DEPENDENCY}"

job_export_arg() {
  local extra="${1:-}"
  local value="ALL"
  value+=",PROJECT_DIR=${PROJECT_DIR}"
  value+=",DATA_DIR=${DATA_DIR}"
  value+=",OUTPUT_ROOT=${OUTPUT_ROOT}"
  value+=",DIAGNOSTICS_ROOT=${DIAGNOSTICS_ROOT}"
  value+=",LOG_DIR=${LOG_DIR}"
  value+=",CONDA_ENV=${CONDA_ENV}"
  value+=",PYTHON_BIN=${PYTHON_BIN}"
  value+=",PD10_DATA_DIR=${PD10_DATA_DIR}"
  value+=",PD10_ROOT=${PD10_ROOT}"
  value+=",PD10_V2_ROOT=${PD10_V2_ROOT}"
  value+=",PD10_MANIFEST_PATH=${PD10_MANIFEST_PATH}"
  value+=",PD10_HLT_CACHE_DIR=${PD10_HLT_CACHE_DIR}"
  value+=",PD10_STEP2_AUDIT_DIR=${PD10_STEP2_AUDIT_DIR}"
  value+=",PD10_TEACHERS_DIR=${PD10_TEACHERS_DIR}"
  value+=",PD10_TEACHER_LOGITS_DIR=${PD10_TEACHER_LOGITS_DIR}"
  value+=",PD10_DUAL_VIEW_TEACHER_DIR=${PD10_DUAL_VIEW_TEACHER_DIR}"
  value+=",PD10_V2_TEACHERS_DIR=${PD10_V2_TEACHERS_DIR}"
  value+=",PD10_V2_TEACHER_LOGITS_DIR=${PD10_V2_TEACHER_LOGITS_DIR}"
  value+=",PD10_V2_TEACHER_REPRESENTATIONS_DIR=${PD10_V2_TEACHER_REPRESENTATIONS_DIR}"
  value+=",PD10_V2_STUDENTS_DIR=${PD10_V2_STUDENTS_DIR}"
  value+=",PD10_V2_FINAL_REPORT_DIR=${PD10_V2_FINAL_REPORT_DIR}"
  value+=",PD10_V2_PARTICLE_DUAL_VIEW_TEACHER_DIR=${PD10_V2_PARTICLE_DUAL_VIEW_TEACHER_DIR}"
  value+=",PD10_V2_PARTICLE_DUAL_VIEW_TEACHER_CHECKPOINT=${PD10_V2_PARTICLE_DUAL_VIEW_TEACHER_CHECKPOINT}"
  value+=",PD10_V2_REPRESENTATION_CACHE_NO_SKIP_EXISTING=${PD10_V2_REPRESENTATION_CACHE_NO_SKIP_EXISTING}"
  value+=",PD10_MODEL_TRAIN_SIZE=${PD10_MODEL_TRAIN_SIZE}"
  value+=",PD10_MODEL_VAL_SIZE=${PD10_MODEL_VAL_SIZE}"
  value+=",PD10_FINAL_TEST_SIZE=${PD10_FINAL_TEST_SIZE}"
  value+=",CONFIRM_FINAL_TEST=${CONFIRM_FINAL_TEST}"
  value+=",SKIP_EXISTING=${SKIP_EXISTING}"
  value+=",OVERWRITE=${OVERWRITE}"
  value+=",DEVICE=${DEVICE}"
  if [[ -n "${extra}" ]]; then
    value+=",${extra}"
  fi
  printf '%s\n' "--export=${value}"
}

prepare_anchor_student_links() {
  if fresh_is_dry_run; then
    return 0
  fi
  mkdir -p "${PD10_V2_STUDENTS_DIR}"
  local variant src dst
  for variant in \
    pd10_student_scratch_ce_only \
    pd10_student_scratch_hlt_full_logits_t2_a0p5 \
    pd10_student_scratch_offline_full_logits_t2_a0p5 \
    pd10_student_scratch_dual_view_full_logits_t2_a0p5 \
    pd10_student_warm_start_ce_only \
    pd10_student_warm_start_hlt_full_logits_t2_a0p5 \
    pd10_student_warm_start_offline_full_logits_t2_a0p5 \
    pd10_student_warm_start_dual_view_full_logits_t2_a0p5; do
    src="${PD10_STUDENTS_DIR}/${variant}"
    dst="${PD10_V2_STUDENTS_DIR}/${variant}"
    if [[ "${src}" == "${dst}" || -e "${dst}" || -L "${dst}" ]]; then
      continue
    fi
    ln -s "${src}" "${dst}"
  done
}

prepare_anchor_teacher_links() {
  if fresh_is_dry_run; then
    return 0
  fi
  mkdir -p "${PD10_V2_TEACHERS_DIR}"
  local model_name src dst
  for model_name in hlt_part_teacher_10class offline_part_teacher_10class dual_view_logit_teacher_10class; do
    src="${PD10_TEACHERS_DIR}/${model_name}"
    dst="${PD10_V2_TEACHERS_DIR}/${model_name}"
    if [[ "${src}" == "${dst}" || -e "${dst}" || -L "${dst}" ]]; then
      continue
    fi
    ln -s "${src}" "${dst}"
  done
}

prepare_anchor_teacher_logit_links() {
  if fresh_is_dry_run; then
    return 0
  fi
  mkdir -p "${PD10_V2_TEACHER_LOGITS_DIR}"
  local model_name src dst
  for model_name in hlt_part_teacher_10class offline_part_teacher_10class dual_view_logit_teacher_10class; do
    src="${PD10_TEACHER_LOGITS_DIR}/${model_name}"
    dst="${PD10_V2_TEACHER_LOGITS_DIR}/${model_name}"
    if [[ "${src}" == "${dst}" || -e "${dst}" || -L "${dst}" ]]; then
      continue
    fi
    ln -s "${src}" "${dst}"
  done
}

if fresh_bool_enabled "${PD10_V2_INCLUDE_PRIORITY_STUDENTS}"; then
  fresh_split_words v2_student_specs "${PD10_V2_STUDENT_SPECS}"
else
  fresh_split_words v2_student_specs "${PD10_V2_CORE_STUDENT_SPECS}"
fi

particle_teacher_base_dep="${PD10_V2_PARTICLE_TEACHER_UPSTREAM_DEPENDENCY}"
dual_rep_base_dep="${PD10_V2_DUAL_REP_UPSTREAM_DEPENDENCY}"
report_anchor_dep="${PD10_V2_REPORT_ANCHOR_UPSTREAM_DEPENDENCY}"

fresh_require_file_unless_deferred "${particle_teacher_base_dep}" "${PD10_MANIFEST_PATH}"
fresh_require_dir_unless_deferred "${particle_teacher_base_dep}" "${PD10_HLT_CACHE_DIR}"
fresh_require_file_unless_deferred "${particle_teacher_base_dep}" "${PD10_HLT_CACHE_DIR}/model_train_fixed_hlt_metadata.json"
fresh_require_file_unless_deferred "${particle_teacher_base_dep}" "${PD10_HLT_CACHE_DIR}/model_val_fixed_hlt_metadata.json"
fresh_require_file_unless_deferred "${particle_teacher_base_dep}" "${PD10_HLT_CACHE_DIR}/final_test_fixed_hlt_metadata.json"
fresh_require_file_unless_deferred "${particle_teacher_base_dep}" "${PD10_TEACHERS_DIR}/hlt_part_teacher_10class/best_model_val.pt"
fresh_require_file_unless_deferred "${particle_teacher_base_dep}" "${PD10_TEACHERS_DIR}/offline_part_teacher_10class/best_model_val.pt"

fresh_require_file_unless_deferred "${dual_rep_base_dep}" "${PD10_DUAL_VIEW_TEACHER_DIR}/best_model_val.pt"
fresh_require_file_unless_deferred "${dual_rep_base_dep}" "${PD10_TEACHER_LOGITS_DIR}/hlt_part_teacher_10class/teacher_logit_manifest.json"
fresh_require_file_unless_deferred "${dual_rep_base_dep}" "${PD10_TEACHER_LOGITS_DIR}/offline_part_teacher_10class/teacher_logit_manifest.json"
fresh_require_file_unless_deferred "${dual_rep_base_dep}" "${PD10_TEACHER_LOGITS_DIR}/dual_view_logit_teacher_10class/teacher_logit_manifest.json"

fresh_require_file_unless_deferred "${report_anchor_dep}" "${PD10_STEP2_AUDIT_DIR}/pd10_step2_audit_report.json"
fresh_require_file_unless_deferred "${report_anchor_dep}" "${PD10_TEACHERS_DIR}/hlt_part_teacher_10class/run_report.json"
fresh_require_file_unless_deferred "${report_anchor_dep}" "${PD10_TEACHERS_DIR}/offline_part_teacher_10class/run_report.json"
fresh_require_file_unless_deferred "${report_anchor_dep}" "${PD10_DUAL_VIEW_TEACHER_DIR}/run_report.json"
fresh_require_file_unless_deferred "${report_anchor_dep}" "${PD10_TEACHER_LOGITS_DIR}/hlt_part_teacher_10class/teacher_logit_manifest.json"
fresh_require_file_unless_deferred "${report_anchor_dep}" "${PD10_TEACHER_LOGITS_DIR}/offline_part_teacher_10class/teacher_logit_manifest.json"
fresh_require_file_unless_deferred "${report_anchor_dep}" "${PD10_TEACHER_LOGITS_DIR}/dual_view_logit_teacher_10class/teacher_logit_manifest.json"
fresh_require_file_unless_deferred "${report_anchor_dep}" "${PD10_STUDENTS_DIR}/pd10_student_scratch_ce_only/run_report.json"
fresh_require_file_unless_deferred "${report_anchor_dep}" "${PD10_STUDENTS_DIR}/pd10_student_scratch_hlt_full_logits_t2_a0p5/run_report.json"
fresh_require_file_unless_deferred "${report_anchor_dep}" "${PD10_STUDENTS_DIR}/pd10_student_scratch_offline_full_logits_t2_a0p5/run_report.json"
fresh_require_file_unless_deferred "${report_anchor_dep}" "${PD10_STUDENTS_DIR}/pd10_student_scratch_dual_view_full_logits_t2_a0p5/run_report.json"
fresh_require_file_unless_deferred "${report_anchor_dep}" "${PD10_STUDENTS_DIR}/pd10_student_warm_start_ce_only/run_report.json"
fresh_require_file_unless_deferred "${report_anchor_dep}" "${PD10_STUDENTS_DIR}/pd10_student_warm_start_hlt_full_logits_t2_a0p5/run_report.json"
fresh_require_file_unless_deferred "${report_anchor_dep}" "${PD10_STUDENTS_DIR}/pd10_student_warm_start_offline_full_logits_t2_a0p5/run_report.json"
fresh_require_file_unless_deferred "${report_anchor_dep}" "${PD10_STUDENTS_DIR}/pd10_student_warm_start_dual_view_full_logits_t2_a0p5/run_report.json"

submitter_lock_dir="${PD10_V2_ROOT}/submission_logs/pd10_v2_$(date +%Y%m%d_%H%M%S)"
fresh_claim_new_dir "${submitter_lock_dir}"
if ! fresh_is_dry_run; then
  {
    echo "created_at=$(date -Is)"
    echo "project_dir=${PROJECT_DIR}"
    echo "source_commit=$(fresh_source_commit)"
    echo "source_status_hash=$(fresh_source_status_hash)"
    echo "pd10_root=${PD10_ROOT}"
    echo "pd10_v2_root=${PD10_V2_ROOT}"
    echo "upstream_dependency=${UPSTREAM_DEPENDENCY}"
    echo "pd10_v2_global_upstream_dependency=${PD10_V2_GLOBAL_UPSTREAM_DEPENDENCY}"
    echo "pd10_v2_particle_teacher_upstream_dependency=${PD10_V2_PARTICLE_TEACHER_UPSTREAM_DEPENDENCY}"
    echo "pd10_v2_dual_rep_upstream_dependency=${PD10_V2_DUAL_REP_UPSTREAM_DEPENDENCY}"
    echo "pd10_v2_report_anchor_upstream_dependency=${PD10_V2_REPORT_ANCHOR_UPSTREAM_DEPENDENCY}"
    echo "skip_existing=${SKIP_EXISTING}"
    echo "confirm_final_test=${CONFIRM_FINAL_TEST}"
    echo "conda_env=${CONDA_ENV}"
    echo "model_train_size=${PD10_MODEL_TRAIN_SIZE}"
    echo "model_val_size=${PD10_MODEL_VAL_SIZE}"
    echo "final_test_size=${PD10_FINAL_TEST_SIZE}"
    echo "v2_student_specs=$(fresh_join_by_space "${v2_student_specs[@]}")"
  } > "${submitter_lock_dir}/metadata.txt"
fi

prepare_anchor_student_links
prepare_anchor_teacher_links
prepare_anchor_teacher_logit_links

particle_teacher_jid=""
if ! skip_existing_artifact "pd10_v2_particle_teacher" "${PD10_V2_PARTICLE_DUAL_VIEW_TEACHER_DIR}/run_report.json"; then
  mapfile -t args < <(
    afterok_args "${particle_teacher_base_dep}" \
      "$(job_export_arg)" \
      "${SCRIPT_DIR}/run_pd10_train_particle_dual_view_teacher.sh"
  )
  submit_job "pd10_v2_particle_teacher" "${args[@]}"
  particle_teacher_jid="${submitted_job_id}"
  echo "submitted pd10_v2_particle_teacher=${particle_teacher_jid}"
fi

particle_cache_dep="$(join_nonempty_by_colon "${particle_teacher_base_dep}" "${particle_teacher_jid}")"
particle_cache_jid=""
if ! skip_existing_artifact "pd10_v2_particle_cache" "${PD10_V2_TEACHER_REPRESENTATIONS_DIR}/particle_dual_view_teacher_10class/teacher_representation_manifest.json"; then
  mapfile -t args < <(
    afterok_args "${particle_cache_dep}" \
      "$(job_export_arg)" \
      "${SCRIPT_DIR}/run_pd10_cache_particle_dual_view_teacher.sh"
  )
  submit_job "pd10_v2_particle_cache" "${args[@]}"
  particle_cache_jid="${submitted_job_id}"
  echo "submitted pd10_v2_particle_cache=${particle_cache_jid}"
fi

needs_dual_rep=0
for spec in "${v2_student_specs[@]}"; do
  old_ifs="${IFS}"
  IFS='|'
  read -r _student_init teacher_target target_mode _temperature _kd_alpha _top_k _variant_name _rep_beta _rep_mode _rep_dim <<< "${spec}"
  IFS="${old_ifs}"
  if [[ "${teacher_target}" == "dual_view" ]]; then
    case "${target_mode}" in
      rep_only|full_logits_plus_rep|top3_plus_rep|confidence_weighted_plus_rep)
        needs_dual_rep=1
        ;;
    esac
  fi
done

dual_rep_dep="$(join_nonempty_by_colon "${dual_rep_base_dep}")"
dual_rep_jid=""
if [[ "${needs_dual_rep}" == "1" ]]; then
  if ! skip_existing_artifact "pd10_v2_dual_view_representations" "${PD10_V2_TEACHER_REPRESENTATIONS_DIR}/dual_view_logit_teacher_10class/teacher_representation_manifest.json"; then
    mapfile -t args < <(
      afterok_args "${dual_rep_dep}" \
        "$(job_export_arg)" \
        "${SCRIPT_DIR}/run_pd10_cache_dual_view_representations.sh"
    )
    submit_job "pd10_v2_dual_view_representations" "${args[@]}"
    dual_rep_jid="${submitted_job_id}"
    echo "submitted pd10_v2_dual_view_representations=${dual_rep_jid}"
  fi
fi

student_job_ids=()
student_variants=()
for spec in "${v2_student_specs[@]}"; do
  old_ifs="${IFS}"
  IFS='|'
  read -r student_init teacher_target target_mode _temperature _kd_alpha _top_k variant_name _rep_beta _rep_mode _rep_dim <<< "${spec}"
  IFS="${old_ifs}"
  if [[ -z "${student_init}" || -z "${teacher_target}" || -z "${target_mode}" || -z "${variant_name}" ]]; then
    echo "Malformed PD10-V2 student spec: ${spec}" >&2
    exit 2
  fi
  student_variants+=("${variant_name}")
  student_done="${PD10_V2_STUDENTS_DIR}/${variant_name}/run_report.json"
  student_dep_parts=()
  case "${teacher_target}" in
    particle_dual_view)
      require_artifact_or_submitted_job \
        "particle dual-view representation cache" \
        "${PD10_V2_TEACHER_REPRESENTATIONS_DIR}/particle_dual_view_teacher_10class/teacher_representation_manifest.json" \
        "${particle_cache_jid}"
      student_dep_parts+=("${particle_cache_jid}")
      ;;
    dual_view)
      require_artifact_or_submitted_job \
        "dual-view representation cache" \
        "${PD10_V2_TEACHER_REPRESENTATIONS_DIR}/dual_view_logit_teacher_10class/teacher_representation_manifest.json" \
        "${dual_rep_jid}"
      student_dep_parts+=("${dual_rep_jid}")
      ;;
    *)
      echo "Unsupported PD10-V2 student teacher target: ${teacher_target}" >&2
      exit 2
      ;;
  esac
  student_dep="$(join_nonempty_by_colon "${student_dep_parts[@]}")"
  if ! skip_existing_artifact "pd10_v2_student_${variant_name}" "${student_done}"; then
    mapfile -t args < <(
      afterok_args "${student_dep}" \
        "$(job_export_arg "PD10_STUDENTS_DIR=${PD10_V2_STUDENTS_DIR},PD10_TEACHER_LOGITS_DIR=${PD10_V2_TEACHER_LOGITS_DIR},PD10_TEACHER_REPRESENTATIONS_DIR=${PD10_V2_TEACHER_REPRESENTATIONS_DIR}")" \
        "${SCRIPT_DIR}/run_pd10_train_student.sh" "${spec}"
    )
    submit_job "pd10_v2_student_${variant_name}" "${args[@]}"
    student_jid="${submitted_job_id}"
    student_job_ids+=("${student_jid}")
    echo "submitted pd10_v2_student_${variant_name}=${student_jid}"
  fi
done

report_dep="$(join_nonempty_by_colon "${report_anchor_dep}" "${particle_teacher_jid}" "${particle_cache_jid}" "${dual_rep_jid}" "${student_job_ids[@]}")"
report_jid=""
if ! skip_existing_artifact "pd10_v2_report" "${PD10_V2_FINAL_REPORT_DIR}/pd10_report.json"; then
  report_extra="PD10_TEACHERS_DIR=${PD10_V2_TEACHERS_DIR},PD10_STUDENTS_DIR=${PD10_V2_STUDENTS_DIR},PD10_FINAL_REPORT_DIR=${PD10_V2_FINAL_REPORT_DIR},PD10_TEACHER_LOGITS_DIR=${PD10_V2_TEACHER_LOGITS_DIR},PD10_TEACHER_REPRESENTATIONS_DIR=${PD10_V2_TEACHER_REPRESENTATIONS_DIR},PD10_REPORT_ALLOW_MISSING_CORE_STUDENTS=${PD10_V2_REPORT_ALLOW_MISSING_CORE_STUDENTS},PD10_REPORT_ALLOW_MISSING_TEACHER_REPORTS=${PD10_V2_REPORT_ALLOW_MISSING_TEACHER_REPORTS},PD10_REPORT_ALLOW_MISSING_AUDIT=${PD10_V2_REPORT_ALLOW_MISSING_AUDIT}"
  mapfile -t args < <(
    afterok_args "${report_dep}" \
      "$(job_export_arg "${report_extra}")" \
      "${SCRIPT_DIR}/run_pd10_write_report.sh"
  )
  submit_job "pd10_v2_report" "${args[@]}"
  report_jid="${submitted_job_id}"
  echo "submitted pd10_v2_report=${report_jid}"
fi

cat <<SUMMARY
pd10_v2_repkd_particle_dualview_submission:
  pd10_root: ${PD10_ROOT}
  pd10_v2_root: ${PD10_V2_ROOT}
  conda_env: ${CONDA_ENV}
  skip_existing: ${SKIP_EXISTING}
  confirm_final_test: ${CONFIRM_FINAL_TEST}
  upstream_dependency: ${UPSTREAM_DEPENDENCY:-none}
  granular_upstream_dependencies:
    global: ${PD10_V2_GLOBAL_UPSTREAM_DEPENDENCY:-none}
    particle_teacher: ${PD10_V2_PARTICLE_TEACHER_UPSTREAM_DEPENDENCY:-none}
    dual_view_representations: ${PD10_V2_DUAL_REP_UPSTREAM_DEPENDENCY:-none}
    report_anchors: ${PD10_V2_REPORT_ANCHOR_UPSTREAM_DEPENDENCY:-none}
  job_ids:
    particle_dual_view_teacher: ${particle_teacher_jid:-skipped_existing}
    particle_dual_view_cache: ${particle_cache_jid:-skipped_existing}
    dual_view_representations: ${dual_rep_jid:-skipped_or_not_needed}
    students: $(fresh_join_by_space "${student_job_ids[@]}")
    final_report: ${report_jid:-skipped_existing}
  dependency_summary:
    particle_teacher_afterok: ${particle_teacher_base_dep:-none}
    particle_cache_afterok: ${particle_cache_dep:-none}
    dual_view_representations_afterok: ${dual_rep_dep:-none}
    students_afterok: teacher-specific representation/cache jobs
    final_report_afterok: ${report_dep:-none}
  expected_jobs:
    students: ${#v2_student_specs[@]}
    total_submitted: ${submit_count}
    total_skipped_existing: ${skip_count}
  student_variants: $(fresh_join_by_space "${student_variants[@]}")
  split_sizes:
    model_train: ${PD10_MODEL_TRAIN_SIZE}
    model_val: ${PD10_MODEL_VAL_SIZE}
    final_test: ${PD10_FINAL_TEST_SIZE}
  outputs:
    particle_teacher: ${PD10_V2_PARTICLE_DUAL_VIEW_TEACHER_DIR}
    teacher_logits: ${PD10_V2_TEACHER_LOGITS_DIR}
    teacher_representations: ${PD10_V2_TEACHER_REPRESENTATIONS_DIR}
    students: ${PD10_V2_STUDENTS_DIR}
    final_report: ${PD10_V2_FINAL_REPORT_DIR}/pd10_report.json
    logs: ${PROJECT_DIR}/fresh_check_logs
SUMMARY
