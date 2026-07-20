#!/usr/bin/env bash
# Submit the dependency-safe first-stage local residual-field curriculum pilot.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
source "${PROJECT_DIR}/sbatch/common.sh"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"

: "${LOCAL_RESIDUAL_FIELD_CURRICULUM_STAGE:=stage1a}"
: "${LOCAL_RESIDUAL_FIELD_CURRICULUM_MODE:=first_stage_pilot}"
: "${LOCAL_RESIDUAL_FIELD_CURRICULUM_CONFIRM_FULL_FAMILY:=0}"
: "${LOCAL_RESIDUAL_FIELD_CURRICULUM_UPSTREAM_DEPENDENCY:=}"
: "${LOCAL_RESIDUAL_FIELD_ROOT:=${OUTPUT_ROOT}/local_particle_residual_field_curriculum/first_stage_pilot}"
: "${LOCAL_RESIDUAL_FIELD_MANIFEST_PATH:=${LOCAL_RESIDUAL_FIELD_ROOT}/inputs/split_manifest.json.gz}"
: "${LOCAL_RESIDUAL_FIELD_HLT_CACHE_DIR:=${LOCAL_RESIDUAL_FIELD_ROOT}/inputs/hlt_cache}"
: "${LOCAL_RESIDUAL_FIELD_OFFLINE_CACHE_DIR:=}"
: "${LOCAL_RESIDUAL_FIELD_TARGET_CACHE_DIR:=${LOCAL_RESIDUAL_FIELD_ROOT}/targets}"
: "${LOCAL_RESIDUAL_FIELD_RECON_ROOT:=${LOCAL_RESIDUAL_FIELD_ROOT}/reconstructors}"
: "${LOCAL_RESIDUAL_FIELD_TAGGER_ROOT:=${LOCAL_RESIDUAL_FIELD_ROOT}/taggers}"
: "${LOCAL_RESIDUAL_FIELD_ORACLE_SOURCE_ROOT:=${LOCAL_RESIDUAL_FIELD_ROOT}/oracle_training_sources}"
: "${LOCAL_RESIDUAL_FIELD_CURRICULUM_ROOT:=${LOCAL_RESIDUAL_FIELD_ROOT}/curriculum}"
: "${LOCAL_RESIDUAL_FIELD_ORACLE_DIAGNOSTICS_ROOT:=${LOCAL_RESIDUAL_FIELD_ROOT}/oracle_diagnostics}"
: "${LOCAL_RESIDUAL_FIELD_PREDICTION_DIR:=${LOCAL_RESIDUAL_FIELD_ROOT}/predictions}"
: "${LOCAL_RESIDUAL_FIELD_FUSION_DIR:=${LOCAL_RESIDUAL_FIELD_ROOT}/fusion}"
: "${LOCAL_RESIDUAL_FIELD_REPORT_DIR:=${LOCAL_RESIDUAL_FIELD_ROOT}/final_report}"
: "${LOCAL_RESIDUAL_FIELD_SELECTED_CONSUMER_JSON:=${LOCAL_RESIDUAL_FIELD_ROOT}/selected_consumer.json}"
: "${LOCAL_RESIDUAL_FIELD_SELECTED_STUDENT_JSON:=${LOCAL_RESIDUAL_FIELD_ROOT}/selected_curriculum_student.json}"
: "${LOCAL_RESIDUAL_FIELD_OFFLINE_TEACHER_LOGITS_DIR:=}"
: "${LOCAL_RESIDUAL_FIELD_HLT_PROFILE:=fixed_hlt_v2_realistic}"
: "${LOCAL_RESIDUAL_FIELD_HLT_DEGRADATION_STRENGTH:=2.5}"
: "${LOCAL_RESIDUAL_FIELD_REUSE_SPLIT_MANIFEST:=1}"
: "${LOCAL_RESIDUAL_FIELD_REUSE_HLT_CACHE:=1}"
: "${LOCAL_RESIDUAL_FIELD_REUSE_OFFLINE_CACHE:=0}"
: "${LOCAL_RESIDUAL_FIELD_REUSE_TARGET_CACHE:=1}"
: "${LOCAL_RESIDUAL_FIELD_REUSE_OFFLINE_TEACHER_LOGITS:=0}"
: "${LOCAL_RESIDUAL_FIELD_STAGE1A_ORACLE_IDS:=O0 Ofull Orobust_light}"
: "${LOCAL_RESIDUAL_FIELD_STAGE1A_ALPHA_IDS:=D_alpha_eval_Ofull D_alpha_eval_Orobust}"
: "${LOCAL_RESIDUAL_FIELD_STAGE1A_DEPLOYABLE_IDS:=P0}"
: "${LOCAL_RESIDUAL_FIELD_STAGE1B_DEPLOYABLE_IDS:=P2 P4 P7a P7b}"
: "${LOCAL_RESIDUAL_FIELD_STAGE1B_ABLATION_IDS:=Q0 Q3}"
: "${LOCAL_RESIDUAL_FIELD_STAGE1B_FUSION_IDS:=G0}"
: "${LOCAL_RESIDUAL_FIELD_MAX_GPU_TRAINING_JOBS_TOTAL:=12}"
: "${LOCAL_RESIDUAL_FIELD_MAX_GPU_TRAINING_JOBS_BEFORE_SELECTOR:=6}"
: "${LOCAL_RESIDUAL_FIELD_MAX_GPU_TRAINING_JOBS_AFTER_SELECTOR:=6}"
: "${LOCAL_RESIDUAL_FIELD_SBATCH_ACCOUNT:=}"
: "${LOCAL_RESIDUAL_FIELD_SBATCH_PARTITION:=tier3}"
: "${LOCAL_RESIDUAL_FIELD_GPU_GRES:=gpu:1}"
: "${LOCAL_RESIDUAL_FIELD_GPU_CPUS_PER_TASK:=8}"
: "${LOCAL_RESIDUAL_FIELD_GPU_MEM:=180G}"
: "${LOCAL_RESIDUAL_FIELD_CPU_CPUS_PER_TASK:=2}"
: "${LOCAL_RESIDUAL_FIELD_CPU_MEM:=8G}"
: "${LOCAL_RESIDUAL_FIELD_SUBMISSION_LOG_DIR:=${LOCAL_RESIDUAL_FIELD_ROOT}/submission_logs/curriculum_$(date +%Y%m%d_%H%M%S)}"
: "${CONFIRM_FINAL_TEST:=1}"

case "${LOCAL_RESIDUAL_FIELD_CURRICULUM_STAGE}" in
  stage1a|select_consumer|stage1b|full_first_stage) ;;
  *)
    echo "LOCAL_RESIDUAL_FIELD_CURRICULUM_STAGE must be stage1a, select_consumer, stage1b, or full_first_stage" >&2
    exit 2
    ;;
esac
if [[ "${LOCAL_RESIDUAL_FIELD_CURRICULUM_MODE}" != first_stage_pilot ]] \
  && ! fresh_bool_enabled "${LOCAL_RESIDUAL_FIELD_CURRICULUM_CONFIRM_FULL_FAMILY}"; then
  echo "Full-family expansion requires LOCAL_RESIDUAL_FIELD_CURRICULUM_CONFIRM_FULL_FAMILY=1" >&2
  exit 2
fi

export PYTHONNOUSERSITE=1
export LOCAL_RESIDUAL_FIELD_ROOT LOCAL_RESIDUAL_FIELD_MANIFEST_PATH
export LOCAL_RESIDUAL_FIELD_HLT_CACHE_DIR LOCAL_RESIDUAL_FIELD_OFFLINE_CACHE_DIR
export LOCAL_RESIDUAL_FIELD_TARGET_CACHE_DIR LOCAL_RESIDUAL_FIELD_RECON_ROOT LOCAL_RESIDUAL_FIELD_TAGGER_ROOT
export LOCAL_RESIDUAL_FIELD_CURRICULUM_ROOT LOCAL_RESIDUAL_FIELD_ORACLE_DIAGNOSTICS_ROOT
export LOCAL_RESIDUAL_FIELD_PREDICTION_DIR LOCAL_RESIDUAL_FIELD_FUSION_DIR LOCAL_RESIDUAL_FIELD_REPORT_DIR
export LOCAL_RESIDUAL_FIELD_SELECTED_CONSUMER_JSON LOCAL_RESIDUAL_FIELD_SELECTED_STUDENT_JSON
export LOCAL_RESIDUAL_FIELD_OFFLINE_TEACHER_LOGITS_DIR
export LOCAL_RESIDUAL_FIELD_HLT_PROFILE LOCAL_RESIDUAL_FIELD_HLT_DEGRADATION_STRENGTH
export LOCAL_RESIDUAL_FIELD_REUSE_SPLIT_MANIFEST LOCAL_RESIDUAL_FIELD_REUSE_HLT_CACHE
export LOCAL_RESIDUAL_FIELD_REUSE_OFFLINE_CACHE LOCAL_RESIDUAL_FIELD_REUSE_TARGET_CACHE
export LOCAL_RESIDUAL_FIELD_REUSE_OFFLINE_TEACHER_LOGITS
export CONFIRM_FINAL_TEST

dependency_token_is_valid() {
  [[ "$1" =~ ^[0-9]+$ ]] || { fresh_is_dry_run && [[ "$1" =~ ^DRYRUN_[A-Za-z0-9_]+$ ]]; }
}

dependency_chain_is_valid() {
  local chain="$1" token
  local -a tokens=()
  [[ -n "${chain}" ]] || return 0
  IFS=: read -r -a tokens <<< "${chain}"
  ((${#tokens[@]} > 0)) || return 1
  for token in "${tokens[@]}"; do
    dependency_token_is_valid "${token}" || return 1
  done
}

submit_job() {
  local label="$1" class="$2" dependency="$3" export_overrides="$4"
  shift 4
  local args=(--parsable --job-name="${label}")
  if [[ -n "${LOCAL_RESIDUAL_FIELD_SBATCH_ACCOUNT}" ]]; then args+=(--account="${LOCAL_RESIDUAL_FIELD_SBATCH_ACCOUNT}"); fi
  if [[ -n "${LOCAL_RESIDUAL_FIELD_SBATCH_PARTITION}" ]]; then args+=(--partition="${LOCAL_RESIDUAL_FIELD_SBATCH_PARTITION}"); fi
  if [[ -n "${dependency}" ]]; then args+=(--dependency="afterok:${dependency}"); fi
  if [[ -n "${export_overrides}" ]]; then args+=(--export="ALL,${export_overrides}"); fi
  if [[ "${class}" == gpu ]]; then
    args+=(--gres="${LOCAL_RESIDUAL_FIELD_GPU_GRES}" --cpus-per-task="${LOCAL_RESIDUAL_FIELD_GPU_CPUS_PER_TASK}" --mem="${LOCAL_RESIDUAL_FIELD_GPU_MEM}")
  else
    args+=(--cpus-per-task="${LOCAL_RESIDUAL_FIELD_CPU_CPUS_PER_TASK}" --mem="${LOCAL_RESIDUAL_FIELD_CPU_MEM}")
  fi
  if fresh_is_dry_run; then
    fresh_print_shell_command sbatch "${args[@]}" "$@" >&2
    printf 'DRYRUN_%s\n' "${label//[^A-Za-z0-9_]/_}"
    return
  fi
  local submitted job_id
  submitted="$(sbatch "${args[@]}" "$@")"
  job_id="${submitted%%;*}"
  dependency_token_is_valid "${job_id}" || { echo "invalid sbatch response for ${label}: ${submitted}" >&2; exit 2; }
  printf '%s\n' "${job_id}"
}

join_colon() {
  local output="" item
  for item in "$@"; do
    [[ -n "${item}" ]] || continue
    output="${output:+${output}:}${item}"
  done
  printf '%s\n' "${output}"
}

dependency_chain_is_valid "${LOCAL_RESIDUAL_FIELD_CURRICULUM_UPSTREAM_DEPENDENCY}" || {
  echo "LOCAL_RESIDUAL_FIELD_CURRICULUM_UPSTREAM_DEPENDENCY must be a colon-separated Slurm job-ID chain" >&2
  exit 2
}

teacher_complete() {
  [[ -f "${LOCAL_RESIDUAL_FIELD_TAGGER_ROOT}/$1/best_model_val.pt" \
    && -f "${LOCAL_RESIDUAL_FIELD_TAGGER_ROOT}/$1/teacher_config.json" \
    && -f "${LOCAL_RESIDUAL_FIELD_TAGGER_ROOT}/$1/run_report.json" ]]
}
curriculum_complete() {
  [[ -f "${LOCAL_RESIDUAL_FIELD_CURRICULUM_ROOT}/$1/best_model_val.pt" \
    && -f "${LOCAL_RESIDUAL_FIELD_CURRICULUM_ROOT}/$1/run_report.json" ]]
}

stage1a_enabled=0
stage1b_enabled=0
if [[ "${LOCAL_RESIDUAL_FIELD_CURRICULUM_STAGE}" == stage1a || "${LOCAL_RESIDUAL_FIELD_CURRICULUM_STAGE}" == full_first_stage ]]; then stage1a_enabled=1; fi
if [[ "${LOCAL_RESIDUAL_FIELD_CURRICULUM_STAGE}" == stage1b || "${LOCAL_RESIDUAL_FIELD_CURRICULUM_STAGE}" == full_first_stage ]]; then stage1b_enabled=1; fi

before_training_jobs=$((stage1a_enabled * 4))
after_training_jobs=$((stage1b_enabled * 6))
total_training_jobs=$((before_training_jobs + after_training_jobs))
((before_training_jobs <= LOCAL_RESIDUAL_FIELD_MAX_GPU_TRAINING_JOBS_BEFORE_SELECTOR)) || { echo "pre-selector GPU training job limit exceeded" >&2; exit 2; }
((after_training_jobs <= LOCAL_RESIDUAL_FIELD_MAX_GPU_TRAINING_JOBS_AFTER_SELECTOR)) || { echo "post-selector GPU training job limit exceeded" >&2; exit 2; }
((total_training_jobs <= LOCAL_RESIDUAL_FIELD_MAX_GPU_TRAINING_JOBS_TOTAL)) || { echo "total GPU training job limit exceeded" >&2; exit 2; }

if [[ "${LOCAL_RESIDUAL_FIELD_CURRICULUM_STAGE}" != select_consumer ]] \
  && [[ -z "${LOCAL_RESIDUAL_FIELD_CURRICULUM_UPSTREAM_DEPENDENCY}" ]] \
  && ! fresh_is_dry_run; then
  fresh_require_file "${LOCAL_RESIDUAL_FIELD_TAGGER_ROOT}/A0/best_model_val.pt"
  fresh_require_file "${LOCAL_RESIDUAL_FIELD_TAGGER_ROOT}/A0/run_report.json"
  fresh_require_file "${LOCAL_RESIDUAL_FIELD_RECON_ROOT}/C0/best_model_val.pt"
fi
if fresh_is_dry_run; then
  manifest=/dev/null
else
  fresh_claim_new_dir "${LOCAL_RESIDUAL_FIELD_SUBMISSION_LOG_DIR}"
  manifest="${LOCAL_RESIDUAL_FIELD_SUBMISSION_LOG_DIR}/submitted_jobs.tsv"
  printf 'stage\trole\trun_id\tjob_id\tdependency\n' > "${manifest}"
fi

record_job() { printf '%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$5" >> "${manifest}"; }

input_audit_jid=""
if [[ "${LOCAL_RESIDUAL_FIELD_CURRICULUM_STAGE}" != select_consumer ]]; then
  input_audit_jid="$(submit_job lprf_input_audit cpu "${LOCAL_RESIDUAL_FIELD_CURRICULUM_UPSTREAM_DEPENDENCY}" "" "${SCRIPT_DIR}/run_validate_local_residual_curriculum_reused_inputs.sh")"
  record_job inputs audit reused_inputs "${input_audit_jid}" "${LOCAL_RESIDUAL_FIELD_CURRICULUM_UPSTREAM_DEPENDENCY}"
fi

declare -A stage1a_jobs=()
if ((stage1a_enabled)); then
  for consumer in O0 Ofull Orobust_light; do
    if fresh_bool_enabled "${SKIP_EXISTING}" && teacher_complete "${consumer}"; then
      stage1a_jobs["${consumer}"]=""
      continue
    fi
    source_dir="${LOCAL_RESIDUAL_FIELD_ORACLE_SOURCE_ROOT}/${consumer}"
    train_export="LOCAL_RESIDUAL_FIELD_TAGGER_ROOT=${LOCAL_RESIDUAL_FIELD_ORACLE_SOURCE_ROOT}"
    train_jid="$(submit_job "lprf_train_${consumer}" gpu "${input_audit_jid}" "${train_export}" "${SCRIPT_DIR}/run_train_local_residual_field_tagger.sh" "${consumer}")"
    record_job stage1a train_oracle_source "${consumer}" "${train_jid}" "${input_audit_jid}"
    register_jid="$(submit_job "lprf_register_${consumer}" cpu "${train_jid}" "" "${SCRIPT_DIR}/run_register_local_residual_oracle_teacher.sh" "${consumer}" "${source_dir}")"
    record_job stage1a register_oracle "${consumer}" "${register_jid}" "${train_jid}"
    stage1a_jobs["${consumer}"]="${register_jid}"
  done
  if fresh_bool_enabled "${SKIP_EXISTING}" && curriculum_complete P0; then
    stage1a_jobs[P0]=""
  else
    stage1a_jobs[P0]="$(submit_job lprf_P0 gpu "${input_audit_jid}" "" "${SCRIPT_DIR}/run_train_local_residual_field_curriculum_student.sh" P0)"
    record_job stage1a train_deployable P0 "${stage1a_jobs[P0]}" "${input_audit_jid}"
  fi

  alpha_parent="$(join_colon "${stage1a_jobs[O0]}" "${stage1a_jobs[Ofull]}" "${stage1a_jobs[Orobust_light]}" "${stage1a_jobs[P0]}")"
  ofull_alpha_jid="$(submit_job lprf_D_alpha_eval_Ofull gpu "${alpha_parent}" "" "${SCRIPT_DIR}/run_evaluate_local_residual_oracle_alpha.sh" Ofull)"
  robust_alpha_jid="$(submit_job lprf_D_alpha_eval_Orobust gpu "${alpha_parent}" "" "${SCRIPT_DIR}/run_evaluate_local_residual_oracle_alpha.sh" Orobust_light)"
  record_job stage1a alpha_diagnostic D_alpha_eval_Ofull "${ofull_alpha_jid}" "${alpha_parent}"
  record_job stage1a alpha_diagnostic D_alpha_eval_Orobust "${robust_alpha_jid}" "${alpha_parent}"
fi

selector_jid=""
if [[ "${LOCAL_RESIDUAL_FIELD_CURRICULUM_STAGE}" == select_consumer ]]; then
  if ! fresh_is_dry_run; then
    fresh_require_file "${LOCAL_RESIDUAL_FIELD_ORACLE_DIAGNOSTICS_ROOT}/D_alpha_eval_Ofull/run_report.json"
    fresh_require_file "${LOCAL_RESIDUAL_FIELD_ORACLE_DIAGNOSTICS_ROOT}/D_alpha_eval_Orobust/run_report.json"
  fi
  selector_jid="$(submit_job lprf_select_consumer cpu "" "" "${SCRIPT_DIR}/run_select_local_residual_curriculum_consumer.sh")"
  record_job selector select_consumer selected_consumer.json "${selector_jid}" ""
elif [[ "${LOCAL_RESIDUAL_FIELD_CURRICULUM_STAGE}" == full_first_stage ]]; then
  selector_dep="$(join_colon "${ofull_alpha_jid}" "${robust_alpha_jid}")"
  selector_jid="$(submit_job lprf_select_consumer cpu "${selector_dep}" "" "${SCRIPT_DIR}/run_select_local_residual_curriculum_consumer.sh")"
  record_job selector select_consumer selected_consumer.json "${selector_jid}" "${selector_dep}"
fi

declare -A stage1b_jobs=()
if ((stage1b_enabled)); then
  stage1b_parent="$(join_colon "${input_audit_jid}" "${selector_jid}")"
  if [[ "${LOCAL_RESIDUAL_FIELD_CURRICULUM_STAGE}" == stage1b ]] && ! fresh_is_dry_run; then
    fresh_require_file "${LOCAL_RESIDUAL_FIELD_SELECTED_CONSUMER_JSON}"
    selected_consumer="$("${PYTHON_BIN}" -c 'import json,sys; value=json.load(open(sys.argv[1], encoding="utf-8"))["selected_consumer_id"]; assert value in {"Ofull","Orobust_light"}; print(value)' "${LOCAL_RESIDUAL_FIELD_SELECTED_CONSUMER_JSON}")"
    fresh_require_file "${LOCAL_RESIDUAL_FIELD_TAGGER_ROOT}/${selected_consumer}/best_model_val.pt"
    fresh_require_file "${LOCAL_RESIDUAL_FIELD_TAGGER_ROOT}/${selected_consumer}/teacher_config.json"
  fi
  for run_id in P2 P4 P7a P7b Q0 Q3; do
    if fresh_bool_enabled "${SKIP_EXISTING}" && curriculum_complete "${run_id}"; then
      stage1b_jobs["${run_id}"]=""
      continue
    fi
    stage1b_jobs["${run_id}"]="$(submit_job "lprf_${run_id}" gpu "${stage1b_parent}" "" "${SCRIPT_DIR}/run_train_local_residual_field_curriculum_student.sh" "${run_id}")"
    record_job stage1b train "${run_id}" "${stage1b_jobs[${run_id}]}" "${stage1b_parent}"
  done

  p_select_dep="$(join_colon "${stage1b_jobs[P2]}" "${stage1b_jobs[P4]}" "${stage1b_jobs[P7a]}" "${stage1b_jobs[P7b]}")"
  best_p_jid="$(submit_job lprf_select_best_P cpu "${p_select_dep}" "" "${SCRIPT_DIR}/run_select_local_residual_curriculum_student.sh")"
  record_job stage1b select_best_p selected_curriculum_student.json "${best_p_jid}" "${p_select_dep}"

  declare -A prediction_jobs=()
  prediction_jobs[A0]="$(submit_job lprf_predict_A0 gpu "${input_audit_jid}" "LOCAL_RESIDUAL_FIELD_PREDICT_MODEL_ROOT=${LOCAL_RESIDUAL_FIELD_TAGGER_ROOT},LOCAL_RESIDUAL_FIELD_PREDICT_SPLITS=stack_train stack_val final_test" "${SCRIPT_DIR}/run_predict_local_residual_field_tagger.sh" A0)"
  record_job stage1b predict A0 "${prediction_jobs[A0]}" "${input_audit_jid}"
  for run_id in P2 P4 P7a P7b; do
    prediction_jobs["${run_id}"]="$(submit_job "lprf_predict_${run_id}" gpu "${stage1b_jobs[${run_id}]}" "LOCAL_RESIDUAL_FIELD_PREDICT_MODEL_ROOT=${LOCAL_RESIDUAL_FIELD_CURRICULUM_ROOT},LOCAL_RESIDUAL_FIELD_PREDICT_SPLITS=stack_train stack_val" "${SCRIPT_DIR}/run_predict_local_residual_field_tagger.sh" "${run_id}")"
    record_job stage1b predict "${run_id}" "${prediction_jobs[${run_id}]}" "${stage1b_jobs[${run_id}]}"
  done
  selected_final_dep="$(join_colon "${best_p_jid}" "${prediction_jobs[P2]}" "${prediction_jobs[P4]}" "${prediction_jobs[P7a]}" "${prediction_jobs[P7b]}")"
  selected_final_jid="$(submit_job lprf_predict_selected_P_final gpu "${selected_final_dep}" "" "${SCRIPT_DIR}/run_predict_selected_local_residual_curriculum_student.sh")"
  record_job stage1b predict_selected_final selected_P "${selected_final_jid}" "${selected_final_dep}"
  fusion_dep="$(join_colon "${best_p_jid}" "${selected_final_jid}" "${prediction_jobs[A0]}" "${prediction_jobs[P2]}" "${prediction_jobs[P4]}" "${prediction_jobs[P7a]}" "${prediction_jobs[P7b]}")"
  fusion_export="LOCAL_RESIDUAL_FIELD_SELECTED_STUDENT_JSON=${LOCAL_RESIDUAL_FIELD_SELECTED_STUDENT_JSON},LOCAL_RESIDUAL_FIELD_REQUIRED_FUSION_GROUPS=G0,LOCAL_RESIDUAL_FIELD_FUSION_MODES=uniform_logit_mean"
  fusion_jid="$(submit_job lprf_G0 cpu "${fusion_dep}" "${fusion_export}" "${SCRIPT_DIR}/run_local_residual_field_fusion.sh")"
  record_job stage1b fusion G0 "${fusion_jid}" "${fusion_dep}"

  report_export="LOCAL_RESIDUAL_FIELD_REQUIRE_CURRICULUM=1,LOCAL_RESIDUAL_FIELD_REQUIRED_TAGGER_RUN_IDS=A0,LOCAL_RESIDUAL_FIELD_REQUIRED_RECON_RUN_IDS=C0,LOCAL_RESIDUAL_FIELD_REQUIRED_FUSION_GROUPS=G0,LOCAL_RESIDUAL_FIELD_REQUIRE_FUSION=1"
  report_jid="$(submit_job lprf_curriculum_report cpu "${fusion_jid}" "${report_export}" "${SCRIPT_DIR}/run_write_local_residual_field_report.sh")"
  record_job report final curriculum_report "${report_jid}" "${fusion_jid}"
fi

if ! fresh_is_dry_run; then
cat > "${LOCAL_RESIDUAL_FIELD_SUBMISSION_LOG_DIR}/policy.txt" <<EOF
mode=${LOCAL_RESIDUAL_FIELD_CURRICULUM_MODE}
stage=${LOCAL_RESIDUAL_FIELD_CURRICULUM_STAGE}
selected_consumer_json=${LOCAL_RESIDUAL_FIELD_SELECTED_CONSUMER_JSON}
max_gpu_training_jobs_total=${LOCAL_RESIDUAL_FIELD_MAX_GPU_TRAINING_JOBS_TOTAL}
max_gpu_training_jobs_before_selector=${LOCAL_RESIDUAL_FIELD_MAX_GPU_TRAINING_JOBS_BEFORE_SELECTOR}
max_gpu_training_jobs_after_selector=${LOCAL_RESIDUAL_FIELD_MAX_GPU_TRAINING_JOBS_AFTER_SELECTOR}
requeue_oom=once_with_larger_memory_or_smaller_batch
requeue_nonfinite_validation=disabled_until_inspected
requeue_provenance_or_cache_mismatch=disabled_fix_inputs
requeue_terminated_before_run_report=same_run_id_with_OVERWRITE_1_after_curve_inspection
EOF
fi

echo "LPRF curriculum submission complete: ${manifest}"
