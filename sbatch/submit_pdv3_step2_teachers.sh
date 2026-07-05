#!/usr/bin/env bash
# Submit the PDV3 Step 2 teacher training and teacher-cache graph.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
# shellcheck source=pdv3_pd10_env.sh
source "${SCRIPT_DIR}/pdv3_pd10_env.sh"

fresh_prepare_submitter

submit_count=0
skip_count=0

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
      echo "Use colon-separated numeric job IDs, for example 12345:12346." >&2
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
  validate_submitted_job_id "${label}" "${job_id}"
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

require_step1_audit_or_dependency() {
  if [[ -n "${PDV3_STEP1_DEPENDENCY}" ]]; then
    validate_dependency_list "PDV3_STEP1_DEPENDENCY" "${PDV3_STEP1_DEPENDENCY}"
    return 0
  fi
  fresh_require_file "${PDV3_STEP1_AUDIT_DIR}/pdv3_step1_input_audit_report.json"
  fresh_assert_json_ok "${PDV3_STEP1_AUDIT_DIR}/pdv3_step1_input_audit_report.json"
}

require_step1_audit_or_dependency

submitter_lock_dir="${PDV3_ROOT}/submission_logs/pdv3_step2_teachers_$(date +%Y%m%d_%H%M%S)"
fresh_claim_new_dir "${submitter_lock_dir}"
if ! fresh_is_dry_run; then
  {
    echo "created_at=$(date -Is)"
    echo "project_dir=${PROJECT_DIR}"
    echo "source_commit=$(fresh_source_commit)"
    echo "source_status_hash=$(fresh_source_status_hash)"
    echo "root=${PDV3_ROOT}"
    echo "manifest=${PDV3_MANIFEST_PATH}"
    echo "hlt_cache=${PDV3_HLT_CACHE_DIR}"
    echo "offline_cache=${PDV3_OFFLINE_CACHE_DIR}"
    echo "step1_dependency=${PDV3_STEP1_DEPENDENCY}"
    echo "teacher_targets=${PDV3_TEACHER_TARGETS}"
    echo "teacher_logit_targets=${PDV3_TEACHER_LOGIT_TARGETS}"
    echo "teacher_logit_splits=${PDV3_TEACHER_LOGIT_SPLITS}"
    echo "dual_view_predict_splits=${PDV3_DUAL_VIEW_PREDICT_SPLITS}"
    echo "particle_dual_view_cache_splits=${PDV3_V2_PARTICLE_DUAL_VIEW_CACHE_SPLITS}"
    echo "hlt_degradation_strength=${PDV3_HLT_DEGRADATION_STRENGTH}"
    echo "model_train_size=${PDV3_MODEL_TRAIN_SIZE}"
    echo "model_val_size=${PDV3_MODEL_VAL_SIZE}"
    echo "final_test_size=${PDV3_FINAL_TEST_SIZE}"
  } > "${submitter_lock_dir}/metadata.txt"
fi

if ! fresh_bool_enabled "${SKIP_EXISTING}"; then
  fresh_refuse_existing_dir "${PDV3_TEACHERS_DIR}/hlt_part_teacher_10class"
  fresh_refuse_existing_dir "${PDV3_TEACHERS_DIR}/offline_part_teacher_10class"
  fresh_refuse_existing_dir "${PDV3_TEACHERS_DIR}/dual_view_logit_teacher_10class"
  fresh_refuse_existing_dir "${PDV3_TEACHERS_DIR}/particle_dual_view_teacher_10class"
  fresh_refuse_existing_dir "${PDV3_TEACHER_LOGITS_DIR}/hlt_part_teacher_10class"
  fresh_refuse_existing_dir "${PDV3_TEACHER_LOGITS_DIR}/offline_part_teacher_10class"
  fresh_refuse_existing_dir "${PDV3_TEACHER_LOGITS_DIR}/dual_view_logit_teacher_10class"
  fresh_refuse_existing_dir "${PDV3_TEACHER_LOGITS_DIR}/particle_dual_view_teacher_10class"
  fresh_refuse_existing_dir "${PDV3_TEACHER_REPRESENTATIONS_DIR}/particle_dual_view_teacher_10class"
fi

fresh_split_words teacher_targets "${PDV3_TEACHER_TARGETS}"
teacher_job_ids=()
hlt_teacher_jid=""
offline_teacher_jid=""
for teacher in "${teacher_targets[@]}"; do
  model_name="$(fresh_pd10_teacher_model_name "${teacher}")"
  teacher_done="${PDV3_TEACHERS_DIR}/${model_name}/run_report.json"
  teacher_jid=""
  if ! skip_existing_artifact "pdv3_teacher_${teacher}" "${teacher_done}"; then
    mapfile -t args < <(afterok_args "${PDV3_STEP1_DEPENDENCY}" "${SCRIPT_DIR}/run_pdv3_train_teacher.sh" "${teacher}")
    teacher_jid="$(submit_job "pdv3_teacher_${teacher}" "${args[@]}")"
    echo "submitted pdv3_teacher_${teacher}=${teacher_jid}"
  fi
  teacher_job_ids+=("${teacher_jid}")
  case "${teacher}" in
    hlt) hlt_teacher_jid="${teacher_jid}" ;;
    offline) offline_teacher_jid="${teacher_jid}" ;;
  esac
done

require_artifact_or_submitted_job \
  "PDV3 HLT ParT teacher" \
  "${PDV3_TEACHERS_DIR}/hlt_part_teacher_10class/best_model_val.pt" \
  "${hlt_teacher_jid}"
require_artifact_or_submitted_job \
  "PDV3 offline ParT teacher" \
  "${PDV3_TEACHERS_DIR}/offline_part_teacher_10class/best_model_val.pt" \
  "${offline_teacher_jid}"

fresh_split_words logit_targets "${PDV3_TEACHER_LOGIT_TARGETS}"
logit_job_ids=()
hlt_logit_jid=""
offline_logit_jid=""
for teacher in "${logit_targets[@]}"; do
  model_name="$(fresh_pd10_teacher_model_name "${teacher}")"
  cache_done="${PDV3_TEACHER_LOGITS_DIR}/${model_name}/teacher_logit_manifest.json"
  case "${teacher}" in
    hlt) teacher_dep="${hlt_teacher_jid}" ;;
    offline) teacher_dep="${offline_teacher_jid}" ;;
    *) teacher_dep="" ;;
  esac
  cache_dep="$(join_nonempty_by_colon "${PDV3_STEP1_DEPENDENCY}" "${teacher_dep}")"
  cache_jid=""
  if ! skip_existing_artifact "pdv3_teacher_logits_${teacher}" "${cache_done}"; then
    mapfile -t args < <(afterok_args "${cache_dep}" "${SCRIPT_DIR}/run_pdv3_cache_teacher_logits.sh" "${teacher}")
    cache_jid="$(submit_job "pdv3_teacher_logits_${teacher}" "${args[@]}")"
    echo "submitted pdv3_teacher_logits_${teacher}=${cache_jid}"
  fi
  logit_job_ids+=("${cache_jid}")
  case "${teacher}" in
    hlt) hlt_logit_jid="${cache_jid}" ;;
    offline) offline_logit_jid="${cache_jid}" ;;
  esac
done

require_artifact_or_submitted_job \
  "PDV3 HLT teacher logits" \
  "${PDV3_TEACHER_LOGITS_DIR}/hlt_part_teacher_10class/teacher_logit_manifest.json" \
  "${hlt_logit_jid}"
require_artifact_or_submitted_job \
  "PDV3 offline teacher logits" \
  "${PDV3_TEACHER_LOGITS_DIR}/offline_part_teacher_10class/teacher_logit_manifest.json" \
  "${offline_logit_jid}"

dual_view_dep="$(join_nonempty_by_colon "${PDV3_STEP1_DEPENDENCY}" "${hlt_logit_jid}" "${offline_logit_jid}")"
dual_view_jid=""
if ! skip_existing_artifact "pdv3_dual_view_logit_teacher" "${PDV3_TEACHERS_DIR}/dual_view_logit_teacher_10class/run_report.json"; then
  mapfile -t args < <(afterok_args "${dual_view_dep}" "${SCRIPT_DIR}/run_pdv3_train_dual_view_teacher.sh")
  dual_view_jid="$(submit_job "pdv3_dual_view_logit_teacher" "${args[@]}")"
  echo "submitted pdv3_dual_view_logit_teacher=${dual_view_jid}"
fi

particle_teacher_dep="$(join_nonempty_by_colon "${PDV3_STEP1_DEPENDENCY}" "${hlt_teacher_jid}" "${offline_teacher_jid}")"
particle_teacher_jid=""
if ! skip_existing_artifact "pdv3_particle_dual_view_teacher" "${PDV3_TEACHERS_DIR}/particle_dual_view_teacher_10class/run_report.json"; then
  mapfile -t args < <(afterok_args "${particle_teacher_dep}" "${SCRIPT_DIR}/run_pdv3_train_particle_dual_view_teacher.sh")
  particle_teacher_jid="$(submit_job "pdv3_particle_dual_view_teacher" "${args[@]}")"
  echo "submitted pdv3_particle_dual_view_teacher=${particle_teacher_jid}"
fi

particle_cache_dep="$(join_nonempty_by_colon "${PDV3_STEP1_DEPENDENCY}" "${particle_teacher_jid}")"
particle_cache_jid=""
particle_rep_manifest="${PDV3_TEACHER_REPRESENTATIONS_DIR}/particle_dual_view_teacher_10class/teacher_representation_manifest.json"
particle_logit_manifest="${PDV3_TEACHER_LOGITS_DIR}/particle_dual_view_teacher_10class/particle_dual_view_cache_manifest.json"
if ! skip_existing_artifact "pdv3_particle_dual_view_cache_representations" "${particle_rep_manifest}" >/dev/null \
  || ! skip_existing_artifact "pdv3_particle_dual_view_cache_logits" "${particle_logit_manifest}" >/dev/null; then
  mapfile -t args < <(afterok_args "${particle_cache_dep}" "${SCRIPT_DIR}/run_pdv3_cache_particle_dual_view_teacher.sh")
  particle_cache_jid="$(submit_job "pdv3_particle_dual_view_cache" "${args[@]}")"
  echo "submitted pdv3_particle_dual_view_cache=${particle_cache_jid}"
fi

cat <<SUMMARY
pdv3_step2_teachers_submission:
  root: ${PDV3_ROOT}
  hlt_degradation_strength: ${PDV3_HLT_DEGRADATION_STRENGTH}
  skip_existing: ${SKIP_EXISTING}
  step1_dependency: ${PDV3_STEP1_DEPENDENCY:-none}
  job_ids:
    teachers: $(fresh_join_by_space "${teacher_job_ids[@]}")
    teacher_logits: $(fresh_join_by_space "${logit_job_ids[@]}")
    dual_view_logit_teacher: ${dual_view_jid:-skipped_existing}
    particle_dual_view_teacher: ${particle_teacher_jid:-skipped_existing}
    particle_dual_view_cache: ${particle_cache_jid:-skipped_existing}
  dependency_summary:
    teachers_afterok: ${PDV3_STEP1_DEPENDENCY:-none}
    logits_afterok: teacher-specific plus step1
    dual_view_afterok: ${dual_view_dep:-none}
    particle_teacher_afterok: ${particle_teacher_dep:-none}
    particle_cache_afterok: ${particle_cache_dep:-none}
  expected_jobs:
    total_submitted: ${submit_count}
    total_skipped_existing: ${skip_count}
  outputs:
    teachers: ${PDV3_TEACHERS_DIR}
    teacher_logits: ${PDV3_TEACHER_LOGITS_DIR}
    teacher_representations: ${PDV3_TEACHER_REPRESENTATIONS_DIR}
    hlt_teacher: ${PDV3_TEACHERS_DIR}/hlt_part_teacher_10class
    offline_teacher: ${PDV3_TEACHERS_DIR}/offline_part_teacher_10class
    dual_view_teacher: ${PDV3_TEACHERS_DIR}/dual_view_logit_teacher_10class
    particle_dual_view_teacher: ${PDV3_TEACHERS_DIR}/particle_dual_view_teacher_10class
    particle_dual_view_logits: ${PDV3_TEACHER_LOGITS_DIR}/particle_dual_view_teacher_10class
    particle_dual_view_representations: ${PDV3_TEACHER_REPRESENTATIONS_DIR}/particle_dual_view_teacher_10class
    logs: ${PROJECT_DIR}/fresh_check_logs
SUMMARY
