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
export PD10_MODEL_TRAIN_SIZE PD10_MODEL_VAL_SIZE PD10_FINAL_TEST_SIZE PD10_HLT_SPLITS
export PD10_HLT_SDV_ROOT PD10_HLT_SDV_HLT2_CACHE_ROOT PD10_HLT_SDV_AUDIT_DIR
export PD10_HLT_SDV_MODELS_DIR PD10_HLT_SDV_FINAL_REPORT_DIR PD10_HLT_SDV_STRENGTHS
export PD10_HLT_SDV_VARIANTS PD10_HLT_SDV_CONTROL_VARIANTS PD10_HLT_SDV_PRIMARY_STRENGTH
export PD10_HLT_SDV_HLT_TEACHER_CHECKPOINT PD10_HLT_SDV_HLT2_SEED PD10_HLT_SDV_HLT2_SHOW_PROGRESS
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

job_export_arg() {
  printf '%s\n' "--export=ALL"
}

validate_dependency_list "UPSTREAM_DEPENDENCY" "${UPSTREAM_DEPENDENCY}"
validate_dependency_list "PD10_HLT_SDV_UPSTREAM_DEPENDENCY" "${PD10_HLT_SDV_UPSTREAM_DEPENDENCY}"

base_dep="${PD10_HLT_SDV_UPSTREAM_DEPENDENCY}"
fresh_require_file_unless_deferred "${base_dep}" "${PD10_MANIFEST_PATH}"
fresh_require_dir_unless_deferred "${base_dep}" "${PD10_HLT_CACHE_DIR}"
fresh_require_file_unless_deferred "${base_dep}" "${PD10_HLT_CACHE_DIR}/model_train_fixed_hlt_metadata.json"
fresh_require_file_unless_deferred "${base_dep}" "${PD10_HLT_CACHE_DIR}/model_val_fixed_hlt_metadata.json"
fresh_require_file_unless_deferred "${base_dep}" "${PD10_HLT_CACHE_DIR}/final_test_fixed_hlt_metadata.json"
fresh_require_file_unless_deferred "${base_dep}" "${PD10_HLT_SDV_HLT_TEACHER_CHECKPOINT}"

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
    echo "conda_env=${CONDA_ENV}"
    echo "model_train_size=${PD10_MODEL_TRAIN_SIZE}"
    echo "model_val_size=${PD10_MODEL_VAL_SIZE}"
    echo "final_test_size=${PD10_FINAL_TEST_SIZE}"
    echo "hlt2_strengths=$(fresh_join_by_space "${hlt2_strengths[@]}")"
    echo "sdv_variants=$(fresh_join_by_space "${sdv_variants[@]}")"
    echo "control_variants=$(fresh_join_by_space "${control_variants[@]}")"
  } > "${submitter_lock_dir}/metadata.txt"
fi

cache_job_ids=()
audit_job_ids=()
for strength in "${hlt2_strengths[@]}"; do
  tag="$(fresh_pd10_hlt_sdv_strength_tag "${strength}")"
  cache_dir="$(fresh_pd10_hlt_sdv_hlt2_cache_dir "${strength}")"
  audit_dir="$(fresh_pd10_hlt_sdv_hlt2_audit_dir "${strength}")"
  cache_jid=""
  if ! skip_existing_artifact "pd10_hlt2_cache_${tag}" "${cache_dir}/final_test_fixed_hlt_metadata.json"; then
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
  if ! skip_existing_artifact "pd10_hlt2_audit_${tag}" "${audit_dir}/hlt2_cache_audit_report.json"; then
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

model_base_dep="$(join_nonempty_by_colon "${base_dep}" "${cache_job_ids[@]}" "${audit_job_ids[@]}")"
model_job_ids=()
for variant in "${sdv_variants[@]}"; do
  model_done="${PD10_HLT_SDV_MODELS_DIR}/${variant}/run_report.json"
  if ! skip_existing_artifact "pd10_hlt_sdv_${variant}" "${model_done}"; then
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
  control_done="${PD10_HLT_SDV_MODELS_DIR}/${variant}/run_report.json"
  if skip_existing_artifact "pd10_hlt_sdv_control_${variant}" "${control_done}"; then
    continue
  fi
  strength="$(fresh_pd10_hlt_sdv_strength_from_variant "${variant}")"
  case "${variant}" in
    hlt2_only_part_s*)
      mapfile -t args < <(
        afterok_args "${model_base_dep}" \
          "$(job_export_arg)" \
          "${SCRIPT_DIR}/run_pd10_train_hlt2_only_control.sh" "${strength}" "${variant}"
      )
      ;;
    tta_hlt_part_hlt_plus_hlt2_s*)
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

report_dep="$(join_nonempty_by_colon "${base_dep}" "${cache_job_ids[@]}" "${audit_job_ids[@]}" "${model_job_ids[@]}" "${control_job_ids[@]}")"
report_jid=""
if ! skip_existing_artifact "pd10_hlt_sdv_report" "${PD10_HLT_SDV_FINAL_REPORT_DIR}/summary.json"; then
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
  skip_existing: ${SKIP_EXISTING}
  overwrite: ${OVERWRITE}
  confirm_final_test: ${CONFIRM_FINAL_TEST}
  upstream_dependency: ${PD10_HLT_SDV_UPSTREAM_DEPENDENCY:-none}
  job_ids:
    hlt2_cache: $(fresh_join_by_space "${cache_job_ids[@]}")
    hlt2_audit: $(fresh_join_by_space "${audit_job_ids[@]}")
    models: $(fresh_join_by_space "${model_job_ids[@]}")
    controls: $(fresh_join_by_space "${control_job_ids[@]}")
    final_report: ${report_jid:-skipped_existing}
  dependency_summary:
    hlt2_cache_afterok: ${base_dep:-none}
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
