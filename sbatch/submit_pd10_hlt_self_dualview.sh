#!/usr/bin/env bash
# Submit the deployable PD10 HLT self-dualview graph.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_ENV:=atlas_kd}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${SMOKE:=0}"
: "${UPSTREAM_DEPENDENCY:=}"
: "${PD10_HLT_SDV_UPSTREAM_DEPENDENCY:=${UPSTREAM_DEPENDENCY}}"
: "${PD10_HLT_SDV_SMOKE_MODEL_TRAIN_SIZE:=20000}"
: "${PD10_HLT_SDV_SMOKE_MODEL_VAL_SIZE:=5000}"
: "${PD10_HLT_SDV_SMOKE_FINAL_TEST_SIZE:=10000}"
: "${PD10_HLT_SDV_SMOKE_EPOCHS:=1}"
: "${PD10_HLT_SDV_SMOKE_HLT2_ONLY_EPOCHS:=1}"

if fresh_bool_enabled "${SMOKE}"; then
  default_sdv_root="${PD10_ROOT}/hlt_self_dualview"
  if [[ "${PD10_HLT_SDV_ROOT}" == "${default_sdv_root}" ]]; then
    PD10_HLT_SDV_ROOT="${PD10_ROOT}/hlt_self_dualview_smoke_20k_5k_10k"
  fi
  PD10_HLT_SDV_HLT2_CACHE_ROOT="${PD10_HLT_SDV_ROOT}/hlt2_cache"
  PD10_HLT_SDV_AUDIT_DIR="${PD10_HLT_SDV_ROOT}/audits"
  PD10_HLT_SDV_MODELS_DIR="${PD10_HLT_SDV_ROOT}/models"
  PD10_HLT_SDV_FINAL_REPORT_DIR="${PD10_HLT_SDV_ROOT}/final_report"
  PD10_MODEL_TRAIN_SIZE="${PD10_HLT_SDV_SMOKE_MODEL_TRAIN_SIZE}"
  PD10_MODEL_VAL_SIZE="${PD10_HLT_SDV_SMOKE_MODEL_VAL_SIZE}"
  PD10_FINAL_TEST_SIZE="${PD10_HLT_SDV_SMOKE_FINAL_TEST_SIZE}"
  PD10_HLT_SDV_EPOCHS="${PD10_HLT_SDV_SMOKE_EPOCHS}"
  PD10_HLT2_ONLY_EPOCHS="${PD10_HLT_SDV_SMOKE_HLT2_ONLY_EPOCHS}"
  PD10_HLT_SDV_EARLY_STOP_PATIENCE=1
  PD10_HLT2_ONLY_EARLY_STOP_PATIENCE=1
fi

export PROJECT_DIR DATA_DIR OUTPUT_ROOT DIAGNOSTICS_ROOT LOG_DIR CONDA_ENV PYTHON_BIN
export PD10_DATA_DIR PD10_ROOT PD10_MANIFEST_PATH PD10_HLT_CACHE_DIR PD10_TEACHERS_DIR
export PD10_TEACHER_LOGITS_DIR PD10_STUDENTS_DIR PD10_FINAL_REPORT_DIR
export PD10_DUAL_VIEW_TEACHER_DIR PD10_DUAL_VIEW_TEACHER_LOGITS_DIR
export PD10_STUDENT_WARM_START_BASELINE_CHECKPOINT
export PD10_MODEL_TRAIN_SIZE PD10_MODEL_VAL_SIZE PD10_FINAL_TEST_SIZE PD10_HLT_SPLITS
export PD10_HLT_SDV_ROOT PD10_HLT_SDV_HLT2_CACHE_ROOT PD10_HLT_SDV_AUDIT_DIR
export PD10_HLT_SDV_MODELS_DIR PD10_HLT_SDV_FINAL_REPORT_DIR PD10_HLT_SDV_STRENGTHS
export PD10_HLT_SDV_VARIANTS PD10_HLT_SDV_CONTROL_VARIANTS PD10_HLT_SDV_PRIMARY_STRENGTH
export PD10_HLT_SDV_HLT_TEACHER_CHECKPOINT PD10_HLT_SDV_HLT2_BRANCH_CHECKPOINT
export PD10_HLT_SDV_SUBMIT_ANCHORS PD10_HLT_SDV_RETRAIN_ANCHORS
export PD10_HLT_SDV_WARM_CE_ANCHOR_SPEC PD10_HLT_SDV_WARM_DUAL_KD_ANCHOR_SPEC
export PD10_HLT_SDV_HLT2_SEED PD10_HLT_SDV_HLT2_SHOW_PROGRESS
export PD10_HLT_SDV_SEED PD10_HLT_SDV_EPOCHS PD10_HLT_SDV_HEAD_WARMUP_EPOCHS
export PD10_HLT_SDV_BATCH_SIZE PD10_HLT_SDV_EVAL_BATCH_SIZE PD10_HLT_SDV_HEAD_WARMUP_LR
export PD10_HLT_SDV_BRANCH_LR PD10_HLT_SDV_HEAD_LR PD10_HLT_SDV_WEIGHT_DECAY
export PD10_HLT_SDV_DROPOUT PD10_HLT_SDV_FUSION_HIDDEN_DIM PD10_HLT_SDV_REPRESENTATION_DIM
export PD10_HLT_SDV_EARLY_STOP_PATIENCE PD10_HLT_SDV_GRAD_CLIP_NORM PD10_HLT_SDV_NUM_WORKERS
export PD10_HLT_SDV_DEVICE PD10_HLT_SDV_MODEL_SIZE PD10_HLT_SDV_NO_AMP PD10_HLT_SDV_COMPILE_MODEL
export PD10_HLT_SDV_NO_BRANCH_INIT PD10_HLT_SDV_SKIP_MODEL_VAL_PREDICTIONS PD10_HLT_SDV_SKIP_FINAL_TEST
export PD10_HLT_SDV_MAX_TRAIN_BATCHES PD10_HLT_SDV_MAX_VAL_BATCHES PD10_HLT_SDV_MAX_FINAL_TEST_BATCHES
export PD10_HLT2_ONLY_SEED PD10_HLT2_ONLY_EPOCHS PD10_HLT2_ONLY_BATCH_SIZE PD10_HLT2_ONLY_EVAL_BATCH_SIZE
export PD10_HLT2_ONLY_LR PD10_HLT2_ONLY_WEIGHT_DECAY PD10_HLT2_ONLY_EARLY_STOP_PATIENCE
export PD10_HLT2_ONLY_NUM_WORKERS PD10_HLT2_ONLY_DEVICE PD10_HLT2_ONLY_MODEL_SIZE
export PD10_HLT2_ONLY_NO_AMP PD10_HLT2_ONLY_COMPILE_MODEL PD10_HLT2_ONLY_NO_WARM_START
export PD10_HLT2_ONLY_SKIP_MODEL_VAL_PREDICTIONS PD10_HLT2_ONLY_SKIP_FINAL_TEST
export PD10_HLT2_ONLY_MAX_TRAIN_BATCHES PD10_HLT2_ONLY_MAX_VAL_BATCHES PD10_HLT2_ONLY_MAX_FINAL_TEST_BATCHES
export PD10_HLT_TTA_SEED PD10_HLT_TTA_BATCH_SIZE PD10_HLT_TTA_NUM_WORKERS PD10_HLT_TTA_DEVICE
export PD10_HLT_TTA_SKIP_FINAL_TEST PD10_HLT_TTA_MAX_VAL_BATCHES PD10_HLT_TTA_MAX_FINAL_TEST_BATCHES
export PD10_HLT_SDV_REPORT_SKIP_PREDICTION_METRICS PD10_HLT_SDV_REPORT_ALLOW_MISSING_SDV_VARIANTS
export PD10_HLT_SDV_REPORT_REQUIRE_ANCHORS CONFIRM_FINAL_TEST SKIP_EXISTING OVERWRITE DEVICE DRY_RUN PRINT_ONLY

fresh_prepare_submitter

if ! fresh_bool_enabled "${CONFIRM_FINAL_TEST}"; then
  echo "Refusing to submit HLT-SDV final-test graph without CONFIRM_FINAL_TEST=1." >&2
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

json_ok_true() {
  local path="$1"
  [[ -f "${path}" ]] || return 1
  "${PYTHON_BIN}" -c 'import json, sys; payload=json.load(open(sys.argv[1], "r", encoding="utf-8")); sys.exit(0 if payload.get("ok") is True else 1)' "${path}" >/dev/null 2>&1
}

all_artifacts_exist() {
  local path
  for path in "$@"; do
    [[ -e "${path}" ]] || return 1
  done
}

refuse_partial_existing_output_dir() {
  local label="$1"
  local path="$2"
  local overwrite_override="${3:-0}"
  if fresh_is_dry_run || fresh_bool_enabled "${OVERWRITE}" || fresh_bool_enabled "${overwrite_override}"; then
    return 0
  fi
  if [[ -e "${path}" ]]; then
    echo "Refusing to submit ${label}; found an existing output path that did not pass the complete-artifact skip check:" >&2
    echo "  ${path}" >&2
    echo "Use OVERWRITE=1, remove the partial output, or choose a fresh output root before requeueing." >&2
    return 2
  fi
}

skip_existing_json_ok() {
  local label="$1"
  local path="$2"
  if ! fresh_bool_enabled "${SKIP_EXISTING}" || fresh_is_dry_run || fresh_bool_enabled "${OVERWRITE}"; then
    return 1
  fi
  json_ok_true "${path}" || return 1
  skip_count=$((skip_count + 1))
  echo "skipped ${label}; found ok=True JSON artifact: ${path}" >&2
  return 0
}

skip_existing_hlt2_cache() {
  local label="$1"
  local cache_dir="$2"
  local audit_dir="${3:-}"
  : "${audit_dir}"
  if ! fresh_bool_enabled "${SKIP_EXISTING}" || fresh_is_dry_run || fresh_bool_enabled "${OVERWRITE}"; then
    return 1
  fi
  all_artifacts_exist \
    "${cache_dir}/model_train_fixed_hlt.npz" \
    "${cache_dir}/model_train_fixed_hlt_metadata.json" \
    "${cache_dir}/model_val_fixed_hlt.npz" \
    "${cache_dir}/model_val_fixed_hlt_metadata.json" \
    "${cache_dir}/final_test_fixed_hlt.npz" \
    "${cache_dir}/final_test_fixed_hlt_metadata.json" || return 1
  skip_count=$((skip_count + 1))
  echo "skipped ${label}; found complete HLT2 cache artifact set" >&2
  return 0
}

skip_existing_trained_model() {
  local label="$1"
  local output_dir="$2"
  local skip_final_test="${3:-0}"
  if ! fresh_bool_enabled "${SKIP_EXISTING}" || fresh_is_dry_run || fresh_bool_enabled "${OVERWRITE}"; then
    return 1
  fi
  all_artifacts_exist \
    "${output_dir}/best_model_val.pt" \
    "${output_dir}/last.pt" \
    "${output_dir}/model_val_report.json" \
    "${output_dir}/config.json" \
    "${output_dir}/training_curves.json" || return 1
  if ! fresh_bool_enabled "${skip_final_test}"; then
    [[ -f "${output_dir}/final_test_report.json" ]] || return 1
  fi
  json_ok_true "${output_dir}/run_report.json" || return 1
  skip_count=$((skip_count + 1))
  echo "skipped ${label}; found complete trained-model artifact set" >&2
  return 0
}

skip_existing_teacher_anchor() {
  local label="$1"
  local output_dir="$2"
  local skip_final_test="${3:-0}"
  if fresh_is_dry_run || fresh_bool_enabled "${OVERWRITE}" || fresh_bool_enabled "${PD10_HLT_SDV_RETRAIN_ANCHORS}"; then
    return 1
  fi
  all_artifacts_exist \
    "${output_dir}/best_model_val.pt" \
    "${output_dir}/run_report.json" \
    "${output_dir}/model_val_report.json" \
    "${output_dir}/source_metadata.json" \
    "${output_dir}/config.json" || return 1
  if ! fresh_bool_enabled "${skip_final_test}"; then
    [[ -f "${output_dir}/final_test_report.json" ]] || return 1
  fi
  json_ok_true "${output_dir}/run_report.json" || return 1
  skip_count=$((skip_count + 1))
  echo "skipped ${label}; found complete teacher-anchor artifact set" >&2
  return 0
}

skip_existing_teacher_logits_anchor() {
  local label="$1"
  local logits_dir="$2"
  if fresh_is_dry_run || fresh_bool_enabled "${OVERWRITE}" || fresh_bool_enabled "${PD10_HLT_SDV_RETRAIN_ANCHORS}"; then
    return 1
  fi
  [[ -f "${logits_dir}/teacher_logit_manifest.json" ]] || return 1
  json_ok_true "${logits_dir}/teacher_logit_manifest.json" || return 1
  local splits=()
  fresh_split_words splits "${PD10_TEACHER_LOGIT_SPLITS}"
  local split
  for split in "${splits[@]}"; do
    all_artifacts_exist \
      "${logits_dir}/${split}_predictions.npz" \
      "${logits_dir}/${split}_predictions_metadata.json" || return 1
  done
  skip_count=$((skip_count + 1))
  echo "skipped ${label}; found complete teacher-logit-anchor artifact set" >&2
  return 0
}

skip_existing_dual_view_anchor() {
  local label="$1"
  local output_dir="$2"
  local logits_dir="$3"
  if fresh_is_dry_run || fresh_bool_enabled "${OVERWRITE}" || fresh_bool_enabled "${PD10_HLT_SDV_RETRAIN_ANCHORS}"; then
    return 1
  fi
  all_artifacts_exist \
    "${output_dir}/best_model_val.pt" \
    "${output_dir}/last.pt" \
    "${output_dir}/run_report.json" \
    "${output_dir}/model_val_report.json" \
    "${output_dir}/source_metadata.json" \
    "${output_dir}/config.json" \
    "${logits_dir}/teacher_logit_manifest.json" || return 1
  json_ok_true "${output_dir}/run_report.json" || return 1
  json_ok_true "${logits_dir}/teacher_logit_manifest.json" || return 1
  local splits=()
  fresh_split_words splits "${PD10_DUAL_VIEW_PREDICT_SPLITS}"
  local split
  for split in "${splits[@]}"; do
    all_artifacts_exist \
      "${logits_dir}/${split}_predictions.npz" \
      "${logits_dir}/${split}_predictions_metadata.json" || return 1
  done
  skip_count=$((skip_count + 1))
  echo "skipped ${label}; found complete dual-view-anchor artifact set" >&2
  return 0
}

student_variant_from_spec() {
  local spec="$1"
  local old_ifs="${IFS}"
  local init teacher mode temp alpha top_k variant rest
  IFS='|'
  read -r init teacher mode temp alpha top_k variant rest <<< "${spec}"
  IFS="${old_ifs}"
  if [[ -z "${variant}" ]]; then
    echo "Malformed PD10 student anchor spec: ${spec}" >&2
    return 2
  fi
  printf '%s\n' "${variant}"
}

skip_existing_student_anchor() {
  local label="$1"
  local output_dir="$2"
  local variant="$3"
  local skip_final_test="${4:-0}"
  if fresh_is_dry_run || fresh_bool_enabled "${OVERWRITE}" || fresh_bool_enabled "${PD10_HLT_SDV_RETRAIN_ANCHORS}"; then
    return 1
  fi
  all_artifacts_exist \
    "${output_dir}/best_model_val.pt" \
    "${output_dir}/last.pt" \
    "${output_dir}/run_report.json" \
    "${output_dir}/model_val_report.json" \
    "${output_dir}/config.json" \
    "${output_dir}/training_curves.json" \
    "${output_dir}/student_predictions/${variant}/model_val_predictions.npz" \
    "${output_dir}/student_predictions/${variant}/model_val_predictions_metadata.json" || return 1
  if ! fresh_bool_enabled "${skip_final_test}"; then
    all_artifacts_exist \
      "${output_dir}/final_test_report.json" \
      "${output_dir}/student_predictions/${variant}/final_test_predictions.npz" \
      "${output_dir}/student_predictions/${variant}/final_test_predictions_metadata.json" || return 1
  fi
  json_ok_true "${output_dir}/run_report.json" || return 1
  skip_count=$((skip_count + 1))
  echo "skipped ${label}; found complete student-anchor artifact set" >&2
  return 0
}

skip_existing_tta_control() {
  local label="$1"
  local output_dir="$2"
  if ! fresh_bool_enabled "${SKIP_EXISTING}" || fresh_is_dry_run || fresh_bool_enabled "${OVERWRITE}"; then
    return 1
  fi
  all_artifacts_exist "${output_dir}/model_val_report.json" || return 1
  if ! fresh_bool_enabled "${PD10_HLT_TTA_SKIP_FINAL_TEST}"; then
    [[ -f "${output_dir}/final_test_report.json" ]] || return 1
  fi
  json_ok_true "${output_dir}/run_report.json" || return 1
  skip_count=$((skip_count + 1))
  echo "skipped ${label}; found complete TTA artifact set" >&2
  return 0
}

skip_existing_final_report() {
  local label="$1"
  local output_dir="$2"
  if ! fresh_bool_enabled "${SKIP_EXISTING}" || fresh_is_dry_run || fresh_bool_enabled "${OVERWRITE}"; then
    return 1
  fi
  all_artifacts_exist \
    "${output_dir}/summary.json" \
    "${output_dir}/hlt_self_dualview_report.json" \
    "${output_dir}/hlt_self_dualview_report.md" \
    "${output_dir}/metric_table.csv" \
    "${output_dir}/comparison_table.csv" \
    "${output_dir}/binary_projection_table.csv" || return 1
  json_ok_true "${output_dir}/run_report.json" || return 1
  skip_count=$((skip_count + 1))
  echo "skipped ${label}; found complete final-report artifact set" >&2
  return 0
}

job_export_arg() {
  printf '%s\n' "--export=ALL"
}

anchor_job_export_arg() {
  if fresh_bool_enabled "${PD10_HLT_SDV_RETRAIN_ANCHORS}"; then
    printf '%s\n' "--export=ALL,OVERWRITE=1"
    return 0
  fi
  job_export_arg
}

validate_dependency_list "UPSTREAM_DEPENDENCY" "${UPSTREAM_DEPENDENCY}"
validate_dependency_list "PD10_HLT_SDV_UPSTREAM_DEPENDENCY" "${PD10_HLT_SDV_UPSTREAM_DEPENDENCY}"

base_dep="${PD10_HLT_SDV_UPSTREAM_DEPENDENCY}"
fresh_require_file_unless_deferred "${base_dep}" "${PD10_MANIFEST_PATH}"
fresh_require_dir_unless_deferred "${base_dep}" "${PD10_HLT_CACHE_DIR}"
fresh_require_file_unless_deferred "${base_dep}" "${PD10_HLT_CACHE_DIR}/model_train_fixed_hlt_metadata.json"
fresh_require_file_unless_deferred "${base_dep}" "${PD10_HLT_CACHE_DIR}/model_val_fixed_hlt_metadata.json"
fresh_require_file_unless_deferred "${base_dep}" "${PD10_HLT_CACHE_DIR}/final_test_fixed_hlt_metadata.json"
if ! fresh_bool_enabled "${PD10_HLT_SDV_SUBMIT_ANCHORS}"; then
  fresh_require_file_unless_deferred "${base_dep}" "${PD10_HLT_SDV_HLT_TEACHER_CHECKPOINT}"
fi

fresh_split_words hlt2_strengths "${PD10_HLT_SDV_STRENGTHS}"
fresh_split_words sdv_variants "${PD10_HLT_SDV_VARIANTS}"
fresh_split_words control_variants "${PD10_HLT_SDV_CONTROL_VARIANTS}"

submitter_lock_dir="${PD10_HLT_SDV_ROOT}/submission_logs/pd10_hlt_sdv_$(date +%Y%m%d_%H%M%S)"
fresh_claim_new_dir "${submitter_lock_dir}"
if ! fresh_is_dry_run; then
  {
    echo "created_at=$(date -Is)"
    echo "project_dir=${PROJECT_DIR}"
    echo "source_commit=$(fresh_source_commit)"
    echo "source_status_hash=$(fresh_source_status_hash)"
    echo "pd10_root=${PD10_ROOT}"
    echo "hlt_sdv_root=${PD10_HLT_SDV_ROOT}"
    echo "smoke=${SMOKE}"
    echo "upstream_dependency=${UPSTREAM_DEPENDENCY}"
    echo "pd10_hlt_sdv_upstream_dependency=${PD10_HLT_SDV_UPSTREAM_DEPENDENCY}"
    echo "skip_existing=${SKIP_EXISTING}"
    echo "overwrite=${OVERWRITE}"
    echo "confirm_final_test=${CONFIRM_FINAL_TEST}"
    echo "submit_anchors=${PD10_HLT_SDV_SUBMIT_ANCHORS}"
    echo "retrain_anchors=${PD10_HLT_SDV_RETRAIN_ANCHORS}"
    echo "conda_env=${CONDA_ENV}"
    echo "model_train_size=${PD10_MODEL_TRAIN_SIZE}"
    echo "model_val_size=${PD10_MODEL_VAL_SIZE}"
    echo "final_test_size=${PD10_FINAL_TEST_SIZE}"
    echo "hlt2_strengths=$(fresh_join_by_space "${hlt2_strengths[@]}")"
    echo "sdv_variants=$(fresh_join_by_space "${sdv_variants[@]}")"
    echo "control_variants=$(fresh_join_by_space "${control_variants[@]}")"
  } > "${submitter_lock_dir}/metadata.txt"
fi

anchor_job_ids=()
hlt_teacher_anchor_job_id=""
offline_teacher_anchor_job_id=""
hlt_logit_anchor_job_id=""
offline_logit_anchor_job_id=""
dual_view_anchor_job_id=""
warm_ce_anchor_job_id=""
warm_dual_kd_anchor_job_id=""

if fresh_bool_enabled "${PD10_HLT_SDV_SUBMIT_ANCHORS}"; then
  for teacher in hlt offline; do
    model_name="$(fresh_pd10_teacher_model_name "${teacher}")"
    teacher_dir="${PD10_TEACHERS_DIR}/${model_name}"
    teacher_jid=""
    if ! skip_existing_teacher_anchor "pd10_hlt_sdv_anchor_teacher_${teacher}" "${teacher_dir}" "${PD10_TEACHER_SKIP_FINAL_TEST:-0}"; then
      refuse_partial_existing_output_dir "pd10_hlt_sdv_anchor_teacher_${teacher}" "${teacher_dir}" "${PD10_HLT_SDV_RETRAIN_ANCHORS}"
      mapfile -t args < <(
        afterok_args "${base_dep}" \
          "$(anchor_job_export_arg)" \
          "${SCRIPT_DIR}/run_pd10_train_teacher.sh" "${teacher}"
      )
      submit_job "pd10_hlt_sdv_anchor_teacher_${teacher}" "${args[@]}"
      teacher_jid="${submitted_job_id}"
      anchor_job_ids+=("${teacher_jid}")
      echo "submitted pd10_hlt_sdv_anchor_teacher_${teacher}=${teacher_jid}"
    fi
    case "${teacher}" in
      hlt) hlt_teacher_anchor_job_id="${teacher_jid}" ;;
      offline) offline_teacher_anchor_job_id="${teacher_jid}" ;;
    esac
  done

  for teacher in hlt offline; do
    model_name="$(fresh_pd10_teacher_model_name "${teacher}")"
    logits_dir="${PD10_TEACHER_LOGITS_DIR}/${model_name}"
    teacher_dep="${base_dep}"
    if [[ "${teacher}" == "hlt" ]]; then
      teacher_dep="$(join_nonempty_by_colon "${base_dep}" "${hlt_teacher_anchor_job_id}")"
    else
      teacher_dep="$(join_nonempty_by_colon "${base_dep}" "${offline_teacher_anchor_job_id}")"
    fi
    logit_jid=""
    if ! skip_existing_teacher_logits_anchor "pd10_hlt_sdv_anchor_teacher_logits_${teacher}" "${logits_dir}"; then
      refuse_partial_existing_output_dir "pd10_hlt_sdv_anchor_teacher_logits_${teacher}" "${logits_dir}" "${PD10_HLT_SDV_RETRAIN_ANCHORS}"
      mapfile -t args < <(
        afterok_args "${teacher_dep}" \
          "$(anchor_job_export_arg)" \
          "${SCRIPT_DIR}/run_pd10_cache_teacher_logits.sh" "${teacher}"
      )
      submit_job "pd10_hlt_sdv_anchor_teacher_logits_${teacher}" "${args[@]}"
      logit_jid="${submitted_job_id}"
      anchor_job_ids+=("${logit_jid}")
      echo "submitted pd10_hlt_sdv_anchor_teacher_logits_${teacher}=${logit_jid}"
    fi
    case "${teacher}" in
      hlt) hlt_logit_anchor_job_id="${logit_jid}" ;;
      offline) offline_logit_anchor_job_id="${logit_jid}" ;;
    esac
  done

  dual_view_dep="$(join_nonempty_by_colon "${base_dep}" "${hlt_logit_anchor_job_id}" "${offline_logit_anchor_job_id}")"
  if ! skip_existing_dual_view_anchor "pd10_hlt_sdv_anchor_dual_view_teacher" "${PD10_DUAL_VIEW_TEACHER_DIR}" "${PD10_DUAL_VIEW_TEACHER_LOGITS_DIR}"; then
    refuse_partial_existing_output_dir "pd10_hlt_sdv_anchor_dual_view_teacher" "${PD10_DUAL_VIEW_TEACHER_DIR}" "${PD10_HLT_SDV_RETRAIN_ANCHORS}"
    refuse_partial_existing_output_dir "pd10_hlt_sdv_anchor_dual_view_logits" "${PD10_DUAL_VIEW_TEACHER_LOGITS_DIR}" "${PD10_HLT_SDV_RETRAIN_ANCHORS}"
    mapfile -t args < <(
      afterok_args "${dual_view_dep}" \
        "$(anchor_job_export_arg)" \
        "${SCRIPT_DIR}/run_pd10_train_dual_view_teacher.sh"
    )
    submit_job "pd10_hlt_sdv_anchor_dual_view_teacher" "${args[@]}"
    dual_view_anchor_job_id="${submitted_job_id}"
    anchor_job_ids+=("${dual_view_anchor_job_id}")
    echo "submitted pd10_hlt_sdv_anchor_dual_view_teacher=${dual_view_anchor_job_id}"
  fi

  warm_ce_variant="$(student_variant_from_spec "${PD10_HLT_SDV_WARM_CE_ANCHOR_SPEC}")"
  warm_ce_dir="${PD10_STUDENTS_DIR}/${warm_ce_variant}"
  warm_ce_dep="$(join_nonempty_by_colon "${base_dep}" "${hlt_teacher_anchor_job_id}")"
  if ! skip_existing_student_anchor "pd10_hlt_sdv_anchor_student_${warm_ce_variant}" "${warm_ce_dir}" "${warm_ce_variant}" "${PD10_STUDENT_SKIP_FINAL_TEST}"; then
    refuse_partial_existing_output_dir "pd10_hlt_sdv_anchor_student_${warm_ce_variant}" "${warm_ce_dir}" "${PD10_HLT_SDV_RETRAIN_ANCHORS}"
    mapfile -t args < <(
      afterok_args "${warm_ce_dep}" \
        "$(anchor_job_export_arg)" \
        "${SCRIPT_DIR}/run_pd10_train_student.sh" "${PD10_HLT_SDV_WARM_CE_ANCHOR_SPEC}"
    )
    submit_job "pd10_hlt_sdv_anchor_student_${warm_ce_variant}" "${args[@]}"
    warm_ce_anchor_job_id="${submitted_job_id}"
    anchor_job_ids+=("${warm_ce_anchor_job_id}")
    echo "submitted pd10_hlt_sdv_anchor_student_${warm_ce_variant}=${warm_ce_anchor_job_id}"
  fi

  warm_dual_kd_variant="$(student_variant_from_spec "${PD10_HLT_SDV_WARM_DUAL_KD_ANCHOR_SPEC}")"
  warm_dual_kd_dir="${PD10_STUDENTS_DIR}/${warm_dual_kd_variant}"
  warm_dual_kd_dep="$(join_nonempty_by_colon "${base_dep}" "${hlt_teacher_anchor_job_id}" "${dual_view_anchor_job_id}")"
  if ! skip_existing_student_anchor "pd10_hlt_sdv_anchor_student_${warm_dual_kd_variant}" "${warm_dual_kd_dir}" "${warm_dual_kd_variant}" "${PD10_STUDENT_SKIP_FINAL_TEST}"; then
    refuse_partial_existing_output_dir "pd10_hlt_sdv_anchor_student_${warm_dual_kd_variant}" "${warm_dual_kd_dir}" "${PD10_HLT_SDV_RETRAIN_ANCHORS}"
    mapfile -t args < <(
      afterok_args "${warm_dual_kd_dep}" \
        "$(anchor_job_export_arg)" \
        "${SCRIPT_DIR}/run_pd10_train_student.sh" "${PD10_HLT_SDV_WARM_DUAL_KD_ANCHOR_SPEC}"
    )
    submit_job "pd10_hlt_sdv_anchor_student_${warm_dual_kd_variant}" "${args[@]}"
    warm_dual_kd_anchor_job_id="${submitted_job_id}"
    anchor_job_ids+=("${warm_dual_kd_anchor_job_id}")
    echo "submitted pd10_hlt_sdv_anchor_student_${warm_dual_kd_variant}=${warm_dual_kd_anchor_job_id}"
  fi
fi

cache_job_ids=()
audit_job_ids=()
for strength in "${hlt2_strengths[@]}"; do
  tag="$(fresh_pd10_hlt_sdv_strength_tag "${strength}")"
  cache_dir="$(fresh_pd10_hlt_sdv_hlt2_cache_dir "${strength}")"
  audit_dir="$(fresh_pd10_hlt_sdv_hlt2_audit_dir "${strength}")"
  cache_jid=""
  if ! skip_existing_hlt2_cache "pd10_hlt2_cache_${tag}" "${cache_dir}" "${audit_dir}"; then
    refuse_partial_existing_output_dir "pd10_hlt2_cache_${tag}" "${cache_dir}"
    mapfile -t args < <(
      afterok_args "${base_dep}" \
        "$(job_export_arg)" \
        "${SCRIPT_DIR}/run_pd10_build_hlt2_cache.sh" "${strength}"
    )
    submit_job "pd10_hlt2_cache_${tag}" "${args[@]}"
    cache_jid="${submitted_job_id}"
    cache_job_ids+=("${cache_jid}")
    echo "submitted pd10_hlt2_cache_${tag}=${cache_jid}"
  fi

  audit_dep="$(join_nonempty_by_colon "${base_dep}" "${cache_jid}")"
  if ! skip_existing_json_ok "pd10_hlt2_audit_${tag}" "${audit_dir}/hlt2_cache_audit_report.json"; then
    mapfile -t args < <(
      afterok_args "${audit_dep}" \
        "$(job_export_arg)" \
        "${SCRIPT_DIR}/run_pd10_audit_hlt2_cache.sh" "${strength}"
    )
    submit_job "pd10_hlt2_audit_${tag}" "${args[@]}"
    audit_jid="${submitted_job_id}"
    audit_job_ids+=("${audit_jid}")
    echo "submitted pd10_hlt2_audit_${tag}=${audit_jid}"
  fi
done

model_base_dep="$(join_nonempty_by_colon "${base_dep}" "${hlt_teacher_anchor_job_id}" "${cache_job_ids[@]}" "${audit_job_ids[@]}")"
model_job_ids=()
for variant in "${sdv_variants[@]}"; do
  model_dir="${PD10_HLT_SDV_MODELS_DIR}/${variant}"
  if ! skip_existing_trained_model "pd10_hlt_sdv_${variant}" "${model_dir}" "${PD10_HLT_SDV_SKIP_FINAL_TEST}"; then
    refuse_partial_existing_output_dir "pd10_hlt_sdv_${variant}" "${model_dir}"
    mapfile -t args < <(
      afterok_args "${model_base_dep}" \
        "$(job_export_arg)" \
        "${SCRIPT_DIR}/run_pd10_train_hlt_self_dualview.sh" "${variant}"
    )
    submit_job "pd10_hlt_sdv_${variant}" "${args[@]}"
    model_jid="${submitted_job_id}"
    model_job_ids+=("${model_jid}")
    echo "submitted pd10_hlt_sdv_${variant}=${model_jid}"
  fi
done

control_job_ids=()
for variant in "${control_variants[@]}"; do
  strength="$(fresh_pd10_hlt_sdv_strength_from_variant "${variant}")"
  case "${variant}" in
    hlt2_only_part_s*)
      control_dir="${PD10_HLT_SDV_MODELS_DIR}/${variant}"
      if skip_existing_trained_model "pd10_hlt_sdv_control_${variant}" "${control_dir}" "${PD10_HLT2_ONLY_SKIP_FINAL_TEST}"; then
        continue
      fi
      refuse_partial_existing_output_dir "pd10_hlt_sdv_control_${variant}" "${control_dir}"
      mapfile -t args < <(
        afterok_args "${model_base_dep}" \
          "$(job_export_arg)" \
          "${SCRIPT_DIR}/run_pd10_train_hlt2_only_control.sh" "${strength}" "${variant}"
      )
      ;;
    tta_hlt_part_hlt_plus_hlt2_s*)
      control_dir="${PD10_HLT_SDV_MODELS_DIR}/${variant}"
      if skip_existing_tta_control "pd10_hlt_sdv_control_${variant}" "${control_dir}"; then
        continue
      fi
      refuse_partial_existing_output_dir "pd10_hlt_sdv_control_${variant}" "${control_dir}"
      mapfile -t args < <(
        afterok_args "${model_base_dep}" \
          "$(job_export_arg)" \
          "${SCRIPT_DIR}/run_pd10_eval_hlt_tta_control.sh" "${strength}" "${variant}"
      )
      ;;
    *)
      echo "Unsupported HLT-SDV control variant: ${variant}" >&2
      exit 2
      ;;
  esac
  submit_job "pd10_hlt_sdv_control_${variant}" "${args[@]}"
  control_jid="${submitted_job_id}"
  control_job_ids+=("${control_jid}")
  echo "submitted pd10_hlt_sdv_control_${variant}=${control_jid}"
done

report_dep="$(join_nonempty_by_colon "${base_dep}" "${cache_job_ids[@]}" "${audit_job_ids[@]}" "${anchor_job_ids[@]}" "${model_job_ids[@]}" "${control_job_ids[@]}")"
report_jid=""
if ! skip_existing_final_report "pd10_hlt_sdv_report" "${PD10_HLT_SDV_FINAL_REPORT_DIR}"; then
  refuse_partial_existing_output_dir "pd10_hlt_sdv_report" "${PD10_HLT_SDV_FINAL_REPORT_DIR}"
  mapfile -t args < <(
    afterok_args "${report_dep}" \
      "$(job_export_arg)" \
      "${SCRIPT_DIR}/run_pd10_write_hlt_self_dualview_report.sh"
  )
  submit_job "pd10_hlt_sdv_report" "${args[@]}"
  report_jid="${submitted_job_id}"
  echo "submitted pd10_hlt_sdv_report=${report_jid}"
fi

cat <<SUMMARY
pd10_hlt_self_dualview_submission:
  pd10_root: ${PD10_ROOT}
  hlt_sdv_root: ${PD10_HLT_SDV_ROOT}
  conda_env: ${CONDA_ENV}
  smoke: ${SMOKE}
  submit_anchors: ${PD10_HLT_SDV_SUBMIT_ANCHORS}
  retrain_anchors: ${PD10_HLT_SDV_RETRAIN_ANCHORS}
  skip_existing: ${SKIP_EXISTING}
  overwrite: ${OVERWRITE}
  confirm_final_test: ${CONFIRM_FINAL_TEST}
  upstream_dependency: ${PD10_HLT_SDV_UPSTREAM_DEPENDENCY:-none}
  job_ids:
    hlt2_cache: $(fresh_join_by_space "${cache_job_ids[@]}")
    hlt2_audit: $(fresh_join_by_space "${audit_job_ids[@]}")
    anchors: $(fresh_join_by_space "${anchor_job_ids[@]}")
    models: $(fresh_join_by_space "${model_job_ids[@]}")
    controls: $(fresh_join_by_space "${control_job_ids[@]}")
    final_report: ${report_jid:-skipped_existing}
  dependency_summary:
    hlt2_cache_afterok: ${base_dep:-none}
    anchor_hlt_teacher_job: ${hlt_teacher_anchor_job_id:-skipped_existing_or_disabled}
    anchor_dual_view_job: ${dual_view_anchor_job_id:-skipped_existing_or_disabled}
    model_and_control_afterok: ${model_base_dep:-none}
    final_report_afterok: ${report_dep:-none}
  expected_jobs:
    hlt2_strengths: ${#hlt2_strengths[@]}
    sdv_models: ${#sdv_variants[@]}
    controls: ${#control_variants[@]}
    total_submitted: ${submit_count}
    total_skipped_existing: ${skip_count}
  split_sizes:
    model_train: ${PD10_MODEL_TRAIN_SIZE}
    model_val: ${PD10_MODEL_VAL_SIZE}
    final_test: ${PD10_FINAL_TEST_SIZE}
  outputs:
    hlt2_cache_root: ${PD10_HLT_SDV_HLT2_CACHE_ROOT}
    models: ${PD10_HLT_SDV_MODELS_DIR}
    final_report: ${PD10_HLT_SDV_FINAL_REPORT_DIR}/summary.json
    logs: ${PROJECT_DIR}/fresh_check_logs
SUMMARY
