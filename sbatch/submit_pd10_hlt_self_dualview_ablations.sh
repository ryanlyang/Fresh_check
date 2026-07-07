#!/usr/bin/env bash
# Submit cache-reusing high-data HLT-SDV branch-init and scratch ablations.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_ENV:=atlas_kd}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${UPSTREAM_DEPENDENCY:=}"
: "${PD10_HLT_SDV_ABLATION_UPSTREAM_DEPENDENCY:=${UPSTREAM_DEPENDENCY}}"
: "${PD10_HLT_SDV_ABLATION_STRENGTHS:=0.10 0.20 0.35 1.00}"
: "${PD10_HLT_SDV_ABLATION_RUN_HLT2_BRANCH_INIT:=1}"
: "${PD10_HLT_SDV_ABLATION_RUN_SCRATCH:=1}"
: "${PD10_HLT_SDV_ABLATION_REQUIRE_ANCHORS:=0}"
: "${PD10_HLT_SDV_ABLATION_ROOT:=${PD10_HLT_SDV_ROOT}/ablations}"
: "${PD10_HLT_SDV_ABLATION_HLT2_INIT_MODELS_DIR:=${PD10_HLT_SDV_ABLATION_ROOT}/hlt2_branch_init/models}"
: "${PD10_HLT_SDV_ABLATION_HLT2_INIT_REPORT_DIR:=${PD10_HLT_SDV_ABLATION_ROOT}/hlt2_branch_init/final_report}"
: "${PD10_HLT_SDV_ABLATION_SCRATCH_MODELS_DIR:=${PD10_HLT_SDV_ABLATION_ROOT}/scratch/models}"
: "${PD10_HLT_SDV_ABLATION_SCRATCH_REPORT_DIR:=${PD10_HLT_SDV_ABLATION_ROOT}/scratch/final_report}"
: "${PD10_HLT_SDV_SCRATCH_SEED:=9801}"

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
export PD10_HLT_SDV_REPORT_SKIP_PREDICTION_METRICS PD10_HLT_SDV_REPORT_ALLOW_MISSING_SDV_VARIANTS
export PD10_HLT_SDV_REPORT_REQUIRE_ANCHORS CONFIRM_FINAL_TEST SKIP_EXISTING OVERWRITE DEVICE DRY_RUN PRINT_ONLY

fresh_prepare_submitter

if ! fresh_bool_enabled "${CONFIRM_FINAL_TEST}"; then
  echo "Refusing to submit HLT-SDV ablations without CONFIRM_FINAL_TEST=1." >&2
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

export_arg_with() {
  local arg="--export=ALL"
  local assignment
  for assignment in "$@"; do
    arg="${arg},${assignment}"
  done
  printf '%s\n' "${arg}"
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
  if fresh_is_dry_run || fresh_bool_enabled "${OVERWRITE}"; then
    return 0
  fi
  if [[ -e "${path}" ]]; then
    echo "Refusing to submit ${label}; found an existing output path that did not pass the complete-artifact skip check:" >&2
    echo "  ${path}" >&2
    echo "Use OVERWRITE=1, remove the partial output, or choose a fresh ablation root before requeueing." >&2
    return 2
  fi
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

validate_dependency_list "UPSTREAM_DEPENDENCY" "${UPSTREAM_DEPENDENCY}"
validate_dependency_list "PD10_HLT_SDV_ABLATION_UPSTREAM_DEPENDENCY" "${PD10_HLT_SDV_ABLATION_UPSTREAM_DEPENDENCY}"

base_dep="${PD10_HLT_SDV_ABLATION_UPSTREAM_DEPENDENCY}"
fresh_require_file_unless_deferred "${base_dep}" "${PD10_MANIFEST_PATH}"
fresh_require_dir_unless_deferred "${base_dep}" "${PD10_HLT_CACHE_DIR}"
fresh_require_file_unless_deferred "${base_dep}" "${PD10_HLT_CACHE_DIR}/model_train_fixed_hlt_metadata.json"
fresh_require_file_unless_deferred "${base_dep}" "${PD10_HLT_CACHE_DIR}/model_val_fixed_hlt_metadata.json"
fresh_require_file_unless_deferred "${base_dep}" "${PD10_HLT_CACHE_DIR}/final_test_fixed_hlt_metadata.json"
fresh_require_file_unless_deferred "${base_dep}" "${PD10_HLT_SDV_HLT_TEACHER_CHECKPOINT}"

fresh_split_words requested_strengths "${PD10_HLT_SDV_ABLATION_STRENGTHS}"
active_strengths=()
sdv_variants=()
hlt2_only_variants=()
for strength in "${requested_strengths[@]}"; do
  normalized="$(fresh_pd10_hlt_sdv_strength_value "${strength}")"
  if [[ "${normalized}" == "0.00" ]]; then
    continue
  fi
  tag="$(fresh_pd10_hlt_sdv_strength_tag "${normalized}")"
  cache_dir="$(fresh_pd10_hlt_sdv_hlt2_cache_dir "${normalized}")"
  fresh_require_dir_unless_deferred "${base_dep}" "${cache_dir}"
  fresh_require_file_unless_deferred "${base_dep}" "${cache_dir}/model_train_fixed_hlt_metadata.json"
  fresh_require_file_unless_deferred "${base_dep}" "${cache_dir}/model_val_fixed_hlt_metadata.json"
  fresh_require_file_unless_deferred "${base_dep}" "${cache_dir}/final_test_fixed_hlt_metadata.json"
  active_strengths+=("${normalized}")
  sdv_variants+=("sdv_hlt_hlt2_${tag}")
  hlt2_only_variants+=("hlt2_only_part_${tag}")
done
if [[ "${#active_strengths[@]}" -eq 0 ]]; then
  echo "No non-identity strengths requested for HLT-SDV ablations." >&2
  exit 2
fi

submitter_lock_dir="${PD10_HLT_SDV_ABLATION_ROOT}/submission_logs/pd10_hlt_sdv_ablations_$(date +%Y%m%d_%H%M%S)"
fresh_claim_new_dir "${submitter_lock_dir}"
if ! fresh_is_dry_run; then
  {
    echo "created_at=$(date -Is)"
    echo "project_dir=${PROJECT_DIR}"
    echo "source_commit=$(fresh_source_commit)"
    echo "source_status_hash=$(fresh_source_status_hash)"
    echo "pd10_root=${PD10_ROOT}"
    echo "hlt_sdv_root=${PD10_HLT_SDV_ROOT}"
    echo "ablation_root=${PD10_HLT_SDV_ABLATION_ROOT}"
    echo "upstream_dependency=${PD10_HLT_SDV_ABLATION_UPSTREAM_DEPENDENCY}"
    echo "strengths=$(fresh_join_by_space "${active_strengths[@]}")"
    echo "run_hlt2_branch_init=${PD10_HLT_SDV_ABLATION_RUN_HLT2_BRANCH_INIT}"
    echo "run_scratch=${PD10_HLT_SDV_ABLATION_RUN_SCRATCH}"
    echo "skip_existing=${SKIP_EXISTING}"
    echo "overwrite=${OVERWRITE}"
    echo "confirm_final_test=${CONFIRM_FINAL_TEST}"
  } > "${submitter_lock_dir}/metadata.txt"
fi

hlt2_branch_jobs=()
hlt2_sdv_jobs=()
scratch_sdv_jobs=()
declare -A hlt2_only_job_by_tag=()

if fresh_bool_enabled "${PD10_HLT_SDV_ABLATION_RUN_HLT2_BRANCH_INIT}"; then
  for i in "${!active_strengths[@]}"; do
    strength="${active_strengths[$i]}"
    tag="$(fresh_pd10_hlt_sdv_strength_tag "${strength}")"
    hlt2_variant="${hlt2_only_variants[$i]}"
    hlt2_output_dir="${PD10_HLT_SDV_ABLATION_HLT2_INIT_MODELS_DIR}/${hlt2_variant}"
    hlt2_jid=""
    if ! skip_existing_trained_model "pd10_hlt_sdv_ablation_hlt2_only_${tag}" "${hlt2_output_dir}" "${PD10_HLT2_ONLY_SKIP_FINAL_TEST}"; then
      refuse_partial_existing_output_dir "pd10_hlt_sdv_ablation_hlt2_only_${tag}" "${hlt2_output_dir}"
      mapfile -t args < <(
        afterok_args "${base_dep}" \
          "$(export_arg_with "PD10_HLT_SDV_MODELS_DIR=${PD10_HLT_SDV_ABLATION_HLT2_INIT_MODELS_DIR}" "PD10_HLT2_ONLY_NO_WARM_START=0")" \
          "${SCRIPT_DIR}/run_pd10_train_hlt2_only_control.sh" "${strength}" "${hlt2_variant}"
      )
      submit_job "pd10_hlt_sdv_ablation_hlt2_only_${tag}" "${args[@]}"
      hlt2_jid="${submitted_job_id}"
      hlt2_branch_jobs+=("${hlt2_jid}")
      echo "submitted pd10_hlt_sdv_ablation_hlt2_only_${tag}=${hlt2_jid}"
    fi
    hlt2_only_job_by_tag["${tag}"]="${hlt2_jid}"
  done

  for i in "${!active_strengths[@]}"; do
    strength="${active_strengths[$i]}"
    tag="$(fresh_pd10_hlt_sdv_strength_tag "${strength}")"
    sdv_variant="${sdv_variants[$i]}"
    hlt2_variant="${hlt2_only_variants[$i]}"
    hlt2_checkpoint="${PD10_HLT_SDV_ABLATION_HLT2_INIT_MODELS_DIR}/${hlt2_variant}/best_model_val.pt"
    sdv_output_dir="${PD10_HLT_SDV_ABLATION_HLT2_INIT_MODELS_DIR}/${sdv_variant}"
    hlt2_dep="${hlt2_only_job_by_tag[$tag]:-}"
    if [[ -z "${hlt2_dep}" ]]; then
      fresh_require_file_unless_deferred "${base_dep}" "${hlt2_checkpoint}"
    fi
    sdv_dep="$(join_nonempty_by_colon "${base_dep}" "${hlt2_dep}")"
    if ! skip_existing_trained_model "pd10_hlt_sdv_ablation_hlt2_branch_init_${tag}" "${sdv_output_dir}" "${PD10_HLT_SDV_SKIP_FINAL_TEST}"; then
      refuse_partial_existing_output_dir "pd10_hlt_sdv_ablation_hlt2_branch_init_${tag}" "${sdv_output_dir}"
      mapfile -t args < <(
        afterok_args "${sdv_dep}" \
          "$(export_arg_with "PD10_HLT_SDV_MODELS_DIR=${PD10_HLT_SDV_ABLATION_HLT2_INIT_MODELS_DIR}" "PD10_HLT_SDV_HLT2_BRANCH_CHECKPOINT=${hlt2_checkpoint}" "PD10_HLT_SDV_NO_BRANCH_INIT=0")" \
          "${SCRIPT_DIR}/run_pd10_train_hlt_self_dualview.sh" "${sdv_variant}"
      )
      submit_job "pd10_hlt_sdv_ablation_hlt2_branch_init_${tag}" "${args[@]}"
      hlt2_sdv_jobs+=("${submitted_job_id}")
      echo "submitted pd10_hlt_sdv_ablation_hlt2_branch_init_${tag}=${submitted_job_id}"
    fi
  done

  hlt2_report_variants="$(fresh_join_by_space "${sdv_variants[@]}")"
  hlt2_report_controls="$(fresh_join_by_space "${hlt2_only_variants[@]}")"
  hlt2_report_dep="$(join_nonempty_by_colon "${base_dep}" "${hlt2_branch_jobs[@]}" "${hlt2_sdv_jobs[@]}")"
  if ! skip_existing_final_report "pd10_hlt_sdv_ablation_hlt2_branch_init_report" "${PD10_HLT_SDV_ABLATION_HLT2_INIT_REPORT_DIR}"; then
    refuse_partial_existing_output_dir "pd10_hlt_sdv_ablation_hlt2_branch_init_report" "${PD10_HLT_SDV_ABLATION_HLT2_INIT_REPORT_DIR}"
    mapfile -t args < <(
      afterok_args "${hlt2_report_dep}" \
        "$(export_arg_with "PD10_HLT_SDV_MODELS_DIR=${PD10_HLT_SDV_ABLATION_HLT2_INIT_MODELS_DIR}" "PD10_HLT_SDV_FINAL_REPORT_DIR=${PD10_HLT_SDV_ABLATION_HLT2_INIT_REPORT_DIR}" "PD10_HLT_SDV_VARIANTS=${hlt2_report_variants}" "PD10_HLT_SDV_CONTROL_VARIANTS=${hlt2_report_controls}" "PD10_HLT_SDV_REPORT_ALLOW_MISSING_SDV_VARIANTS=0" "PD10_HLT_SDV_REPORT_REQUIRE_ANCHORS=${PD10_HLT_SDV_ABLATION_REQUIRE_ANCHORS}")" \
        "${SCRIPT_DIR}/run_pd10_write_hlt_self_dualview_report.sh"
    )
    submit_job "pd10_hlt_sdv_ablation_hlt2_branch_init_report" "${args[@]}"
    echo "submitted pd10_hlt_sdv_ablation_hlt2_branch_init_report=${submitted_job_id}"
  fi
fi

if fresh_bool_enabled "${PD10_HLT_SDV_ABLATION_RUN_SCRATCH}"; then
  for i in "${!active_strengths[@]}"; do
    strength="${active_strengths[$i]}"
    tag="$(fresh_pd10_hlt_sdv_strength_tag "${strength}")"
    sdv_variant="${sdv_variants[$i]}"
    scratch_output_dir="${PD10_HLT_SDV_ABLATION_SCRATCH_MODELS_DIR}/${sdv_variant}"
    if ! skip_existing_trained_model "pd10_hlt_sdv_ablation_scratch_${tag}" "${scratch_output_dir}" "${PD10_HLT_SDV_SKIP_FINAL_TEST}"; then
      refuse_partial_existing_output_dir "pd10_hlt_sdv_ablation_scratch_${tag}" "${scratch_output_dir}"
      mapfile -t args < <(
        afterok_args "${base_dep}" \
          "$(export_arg_with "PD10_HLT_SDV_MODELS_DIR=${PD10_HLT_SDV_ABLATION_SCRATCH_MODELS_DIR}" "PD10_HLT_SDV_NO_BRANCH_INIT=1" "PD10_HLT_SDV_HEAD_WARMUP_EPOCHS=0" "PD10_HLT_SDV_HLT2_BRANCH_CHECKPOINT=" "PD10_HLT_SDV_SEED=${PD10_HLT_SDV_SCRATCH_SEED}")" \
          "${SCRIPT_DIR}/run_pd10_train_hlt_self_dualview.sh" "${sdv_variant}"
      )
      submit_job "pd10_hlt_sdv_ablation_scratch_${tag}" "${args[@]}"
      scratch_sdv_jobs+=("${submitted_job_id}")
      echo "submitted pd10_hlt_sdv_ablation_scratch_${tag}=${submitted_job_id}"
    fi
  done

  scratch_report_variants="$(fresh_join_by_space "${sdv_variants[@]}")"
  scratch_report_dep="$(join_nonempty_by_colon "${base_dep}" "${scratch_sdv_jobs[@]}")"
  if ! skip_existing_final_report "pd10_hlt_sdv_ablation_scratch_report" "${PD10_HLT_SDV_ABLATION_SCRATCH_REPORT_DIR}"; then
    refuse_partial_existing_output_dir "pd10_hlt_sdv_ablation_scratch_report" "${PD10_HLT_SDV_ABLATION_SCRATCH_REPORT_DIR}"
    mapfile -t args < <(
      afterok_args "${scratch_report_dep}" \
        "$(export_arg_with "PD10_HLT_SDV_MODELS_DIR=${PD10_HLT_SDV_ABLATION_SCRATCH_MODELS_DIR}" "PD10_HLT_SDV_FINAL_REPORT_DIR=${PD10_HLT_SDV_ABLATION_SCRATCH_REPORT_DIR}" "PD10_HLT_SDV_VARIANTS=${scratch_report_variants}" "PD10_HLT_SDV_CONTROL_VARIANTS=" "PD10_HLT_SDV_REPORT_ALLOW_MISSING_SDV_VARIANTS=0" "PD10_HLT_SDV_REPORT_REQUIRE_ANCHORS=${PD10_HLT_SDV_ABLATION_REQUIRE_ANCHORS}")" \
        "${SCRIPT_DIR}/run_pd10_write_hlt_self_dualview_report.sh"
    )
    submit_job "pd10_hlt_sdv_ablation_scratch_report" "${args[@]}"
    echo "submitted pd10_hlt_sdv_ablation_scratch_report=${submitted_job_id}"
  fi
fi

cat <<SUMMARY
pd10_hlt_self_dualview_ablation_submission:
  pd10_root: ${PD10_ROOT}
  hlt_sdv_root: ${PD10_HLT_SDV_ROOT}
  ablation_root: ${PD10_HLT_SDV_ABLATION_ROOT}
  conda_env: ${CONDA_ENV}
  strengths: $(fresh_join_by_space "${active_strengths[@]}")
  run_hlt2_branch_init: ${PD10_HLT_SDV_ABLATION_RUN_HLT2_BRANCH_INIT}
  run_scratch: ${PD10_HLT_SDV_ABLATION_RUN_SCRATCH}
  skip_existing: ${SKIP_EXISTING}
  overwrite: ${OVERWRITE}
  upstream_dependency: ${PD10_HLT_SDV_ABLATION_UPSTREAM_DEPENDENCY:-none}
  submitted_jobs: ${submit_count}
  skipped_existing: ${skip_count}
  outputs:
    hlt2_branch_init_models: ${PD10_HLT_SDV_ABLATION_HLT2_INIT_MODELS_DIR}
    hlt2_branch_init_report: ${PD10_HLT_SDV_ABLATION_HLT2_INIT_REPORT_DIR}/summary.json
    scratch_models: ${PD10_HLT_SDV_ABLATION_SCRATCH_MODELS_DIR}
    scratch_report: ${PD10_HLT_SDV_ABLATION_SCRATCH_REPORT_DIR}/summary.json
SUMMARY
