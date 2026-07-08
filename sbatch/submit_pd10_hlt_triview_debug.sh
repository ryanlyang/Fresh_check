#!/usr/bin/env bash
# Submit the quick deployable HLT tri-view test on debug, reusing existing HLT/HLT2 caches.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_ENV:=atlas_kd}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${UPSTREAM_DEPENDENCY:=}"
: "${PD10_HLT_TRIVIEW_UPSTREAM_DEPENDENCY:=${UPSTREAM_DEPENDENCY}}"
: "${PD10_HLT_TRIVIEW_ROOT:=${PD10_ROOT}/hlt_triview_debug_$(date +%Y%m%d_%H%M%S)}"
: "${PD10_HLT_TRIVIEW_SOURCE_MODELS_DIR:=${PD10_HLT_TRIVIEW_ROOT}/source_models}"
: "${PD10_HLT_TRIVIEW_MODELS_DIR:=${PD10_HLT_TRIVIEW_ROOT}/models}"
: "${PD10_HLT_TRIVIEW_MODEL_NAME:=tri_hlt_hlt2_s0p35_s1p00}"
: "${PD10_HLT_TRIVIEW_HLT2_S0P35_CACHE_DIR:=$(fresh_pd10_hlt_sdv_hlt2_cache_dir 0.35)}"
: "${PD10_HLT_TRIVIEW_HLT2_S1P00_CACHE_DIR:=$(fresh_pd10_hlt_sdv_hlt2_cache_dir 1.00)}"
: "${PD10_HLT_TRIVIEW_TRAIN_SIZE:=1000000}"
: "${PD10_HLT_TRIVIEW_VAL_SIZE:=250000}"
: "${PD10_HLT_TRIVIEW_FINAL_TEST_SIZE:=500000}"
: "${PD10_HLT_TRIVIEW_REQUIRE_SOURCE_WARM_START:=0}"
: "${PD10_HLT_TRIVIEW_SOURCE_WARM_START_CHECKPOINT:=}"
: "${PD10_HLT_TRIVIEW_SOURCE_SKIP_FINAL_TEST:=0}"
: "${PD10_HLT_TRIVIEW_SKIP_FINAL_TEST:=0}"

export PROJECT_DIR DATA_DIR OUTPUT_ROOT DIAGNOSTICS_ROOT LOG_DIR CONDA_ENV PYTHON_BIN
export PD10_ROOT PD10_HLT_CACHE_DIR PD10_HLT_SDV_ROOT PD10_HLT_SDV_HLT2_CACHE_ROOT
export PD10_HLT_TRIVIEW_ROOT PD10_HLT_TRIVIEW_SOURCE_MODELS_DIR PD10_HLT_TRIVIEW_MODELS_DIR
export PD10_HLT_TRIVIEW_MODEL_NAME PD10_HLT_TRIVIEW_HLT2_S0P35_CACHE_DIR PD10_HLT_TRIVIEW_HLT2_S1P00_CACHE_DIR
export PD10_HLT_TRIVIEW_TRAIN_SIZE PD10_HLT_TRIVIEW_VAL_SIZE PD10_HLT_TRIVIEW_FINAL_TEST_SIZE
export PD10_HLT_TRIVIEW_SOURCE_WARM_START_CHECKPOINT
export PD10_HLT_TRIVIEW_SOURCE_SEED PD10_HLT_TRIVIEW_SOURCE_EPOCHS PD10_HLT_TRIVIEW_SOURCE_BATCH_SIZE
export PD10_HLT_TRIVIEW_SOURCE_EVAL_BATCH_SIZE PD10_HLT_TRIVIEW_SOURCE_LR
export PD10_HLT_TRIVIEW_SOURCE_WEIGHT_DECAY PD10_HLT_TRIVIEW_SOURCE_EARLY_STOP_PATIENCE
export PD10_HLT_TRIVIEW_SOURCE_NUM_WORKERS PD10_HLT_TRIVIEW_SOURCE_DEVICE PD10_HLT_TRIVIEW_SOURCE_MODEL_SIZE
export PD10_HLT_TRIVIEW_SOURCE_NO_AMP PD10_HLT_TRIVIEW_SOURCE_COMPILE_MODEL
export PD10_HLT_TRIVIEW_SOURCE_SKIP_MODEL_VAL_PREDICTIONS PD10_HLT_TRIVIEW_SOURCE_SKIP_FINAL_TEST
export PD10_HLT_TRIVIEW_SOURCE_MAX_TRAIN_BATCHES PD10_HLT_TRIVIEW_SOURCE_MAX_VAL_BATCHES
export PD10_HLT_TRIVIEW_SOURCE_MAX_FINAL_TEST_BATCHES
export PD10_HLT_TRIVIEW_SEED PD10_HLT_TRIVIEW_EPOCHS PD10_HLT_TRIVIEW_HEAD_WARMUP_EPOCHS
export PD10_HLT_TRIVIEW_BATCH_SIZE PD10_HLT_TRIVIEW_EVAL_BATCH_SIZE
export PD10_HLT_TRIVIEW_HEAD_WARMUP_LR PD10_HLT_TRIVIEW_BRANCH_LR PD10_HLT_TRIVIEW_HEAD_LR
export PD10_HLT_TRIVIEW_WEIGHT_DECAY PD10_HLT_TRIVIEW_DROPOUT PD10_HLT_TRIVIEW_FUSION_HIDDEN_DIM
export PD10_HLT_TRIVIEW_REPRESENTATION_DIM PD10_HLT_TRIVIEW_EARLY_STOP_PATIENCE
export PD10_HLT_TRIVIEW_NUM_WORKERS PD10_HLT_TRIVIEW_DEVICE PD10_HLT_TRIVIEW_MODEL_SIZE
export PD10_HLT_TRIVIEW_NO_AMP PD10_HLT_TRIVIEW_COMPILE_MODEL
export PD10_HLT_TRIVIEW_SKIP_MODEL_VAL_PREDICTIONS PD10_HLT_TRIVIEW_SKIP_FINAL_TEST
export PD10_HLT_TRIVIEW_MAX_TRAIN_BATCHES PD10_HLT_TRIVIEW_MAX_VAL_BATCHES
export PD10_HLT_TRIVIEW_MAX_FINAL_TEST_BATCHES
export CONFIRM_FINAL_TEST SKIP_EXISTING OVERWRITE DEVICE DRY_RUN PRINT_ONLY

fresh_prepare_submitter

if ! fresh_bool_enabled "${CONFIRM_FINAL_TEST}"; then
  echo "Refusing to submit HLT tri-view debug run without CONFIRM_FINAL_TEST=1." >&2
  exit 2
fi

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

validate_submitted_job_id() {
  local label="$1"
  local job_id="$2"
  if ! dependency_token_is_valid "${job_id}"; then
    echo "Failed to submit ${label}; expected a Slurm job ID but got '${job_id:-empty}'." >&2
    return 2
  fi
}

submit_count=0
skip_count=0
submitted_job_id=""

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

refuse_partial_existing_output_dir() {
  local label="$1"
  local path="$2"
  if fresh_is_dry_run || fresh_bool_enabled "${OVERWRITE}"; then
    return 0
  fi
  if [[ -e "${path}" ]]; then
    echo "Refusing to submit ${label}; found an existing output path that did not pass the complete-artifact skip check:" >&2
    echo "  ${path}" >&2
    echo "Use OVERWRITE=1, remove the partial output, or choose a fresh tri-view root before requeueing." >&2
    return 2
  fi
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

validate_dependency_list "PD10_HLT_TRIVIEW_UPSTREAM_DEPENDENCY" "${PD10_HLT_TRIVIEW_UPSTREAM_DEPENDENCY}"

base_dep="${PD10_HLT_TRIVIEW_UPSTREAM_DEPENDENCY}"
fresh_require_dir_unless_deferred "${base_dep}" "${PD10_HLT_CACHE_DIR}"
fresh_require_dir_unless_deferred "${base_dep}" "${PD10_HLT_TRIVIEW_HLT2_S0P35_CACHE_DIR}"
fresh_require_dir_unless_deferred "${base_dep}" "${PD10_HLT_TRIVIEW_HLT2_S1P00_CACHE_DIR}"
for cache_dir in "${PD10_HLT_CACHE_DIR}" "${PD10_HLT_TRIVIEW_HLT2_S0P35_CACHE_DIR}" "${PD10_HLT_TRIVIEW_HLT2_S1P00_CACHE_DIR}"; do
  fresh_require_file_unless_deferred "${base_dep}" "${cache_dir}/model_train_fixed_hlt_metadata.json"
  fresh_require_file_unless_deferred "${base_dep}" "${cache_dir}/model_val_fixed_hlt_metadata.json"
  fresh_require_file_unless_deferred "${base_dep}" "${cache_dir}/final_test_fixed_hlt_metadata.json"
done
if fresh_bool_enabled "${PD10_HLT_TRIVIEW_REQUIRE_SOURCE_WARM_START}" && [[ -z "${PD10_HLT_TRIVIEW_SOURCE_WARM_START_CHECKPOINT}" ]]; then
  echo "PD10_HLT_TRIVIEW_REQUIRE_SOURCE_WARM_START=1 requires PD10_HLT_TRIVIEW_SOURCE_WARM_START_CHECKPOINT." >&2
  exit 2
fi
if fresh_bool_enabled "${PD10_HLT_TRIVIEW_REQUIRE_SOURCE_WARM_START}"; then
  fresh_require_file_unless_deferred "${base_dep}" "${PD10_HLT_TRIVIEW_SOURCE_WARM_START_CHECKPOINT}"
fi

submitter_lock_dir="${PD10_HLT_TRIVIEW_ROOT}/submission_logs/pd10_hlt_triview_debug_$(date +%Y%m%d_%H%M%S)"
fresh_claim_new_dir "${submitter_lock_dir}"
if ! fresh_is_dry_run; then
  {
    echo "created_at=$(date -Is)"
    echo "project_dir=${PROJECT_DIR}"
    echo "source_commit=$(fresh_source_commit)"
    echo "source_status_hash=$(fresh_source_status_hash)"
    echo "pd10_root=${PD10_ROOT}"
    echo "hlt_cache_dir=${PD10_HLT_CACHE_DIR}"
    echo "hlt2_s0p35_cache_dir=${PD10_HLT_TRIVIEW_HLT2_S0P35_CACHE_DIR}"
    echo "hlt2_s1p00_cache_dir=${PD10_HLT_TRIVIEW_HLT2_S1P00_CACHE_DIR}"
    echo "triview_root=${PD10_HLT_TRIVIEW_ROOT}"
    echo "train_size=${PD10_HLT_TRIVIEW_TRAIN_SIZE}"
    echo "val_size=${PD10_HLT_TRIVIEW_VAL_SIZE}"
    echo "final_test_size=${PD10_HLT_TRIVIEW_FINAL_TEST_SIZE}"
    echo "source_warm_start_checkpoint=${PD10_HLT_TRIVIEW_SOURCE_WARM_START_CHECKPOINT:-none}"
    echo "require_source_warm_start=${PD10_HLT_TRIVIEW_REQUIRE_SOURCE_WARM_START}"
    echo "upstream_dependency=${PD10_HLT_TRIVIEW_UPSTREAM_DEPENDENCY}"
    echo "skip_existing=${SKIP_EXISTING}"
    echo "overwrite=${OVERWRITE}"
  } > "${submitter_lock_dir}/metadata.txt"
fi

source_jobs=()
declare -A source_job_by_name=()

submit_source() {
  local name="$1"
  local cache_dir="$2"
  local source_view="$3"
  local output_dir="${PD10_HLT_TRIVIEW_SOURCE_MODELS_DIR}/${name}"
  local job_id=""
  if ! skip_existing_trained_model "pd10_hlt_triview_source_${name}" "${output_dir}" "${PD10_HLT_TRIVIEW_SOURCE_SKIP_FINAL_TEST}"; then
    refuse_partial_existing_output_dir "pd10_hlt_triview_source_${name}" "${output_dir}"
    mapfile -t args < <(
      afterok_args "${base_dep}" \
        "$(export_arg_with "PD10_HLT_TRIVIEW_SOURCE_MODELS_DIR=${PD10_HLT_TRIVIEW_SOURCE_MODELS_DIR}")" \
        "${SCRIPT_DIR}/run_pd10_train_hlt_triview_source.sh" "${name}" "${cache_dir}" "${source_view}"
    )
    submit_job "pd10_hlt_triview_source_${name}" "${args[@]}"
    job_id="${submitted_job_id}"
    source_jobs+=("${job_id}")
    echo "submitted pd10_hlt_triview_source_${name}=${job_id}"
  fi
  source_job_by_name["${name}"]="${job_id}"
}

submit_source "hlt_source" "${PD10_HLT_CACHE_DIR}" "fixed_hlt"
submit_source "hlt2_s0p35_source" "${PD10_HLT_TRIVIEW_HLT2_S0P35_CACHE_DIR}" "hlt2"
submit_source "hlt2_s1p00_source" "${PD10_HLT_TRIVIEW_HLT2_S1P00_CACHE_DIR}" "hlt2"

tri_output_dir="${PD10_HLT_TRIVIEW_MODELS_DIR}/${PD10_HLT_TRIVIEW_MODEL_NAME}"
tri_checkpoint_hlt="${PD10_HLT_TRIVIEW_SOURCE_MODELS_DIR}/hlt_source/best_model_val.pt"
tri_checkpoint_s035="${PD10_HLT_TRIVIEW_SOURCE_MODELS_DIR}/hlt2_s0p35_source/best_model_val.pt"
tri_checkpoint_s100="${PD10_HLT_TRIVIEW_SOURCE_MODELS_DIR}/hlt2_s1p00_source/best_model_val.pt"
if [[ "${#source_jobs[@]}" -eq 0 ]]; then
  fresh_require_file_unless_deferred "${base_dep}" "${tri_checkpoint_hlt}"
  fresh_require_file_unless_deferred "${base_dep}" "${tri_checkpoint_s035}"
  fresh_require_file_unless_deferred "${base_dep}" "${tri_checkpoint_s100}"
fi
tri_dep="$(join_nonempty_by_colon "${base_dep}" "${source_jobs[@]}")"
tri_job=""
if ! skip_existing_trained_model "pd10_hlt_triview_fusion" "${tri_output_dir}" "${PD10_HLT_TRIVIEW_SKIP_FINAL_TEST}"; then
  refuse_partial_existing_output_dir "pd10_hlt_triview_fusion" "${tri_output_dir}"
  mapfile -t args < <(
    afterok_args "${tri_dep}" \
      "$(export_arg_with "PD10_HLT_TRIVIEW_SOURCE_MODELS_DIR=${PD10_HLT_TRIVIEW_SOURCE_MODELS_DIR}" "PD10_HLT_TRIVIEW_MODELS_DIR=${PD10_HLT_TRIVIEW_MODELS_DIR}" "PD10_HLT_TRIVIEW_HLT_SOURCE_CHECKPOINT=${tri_checkpoint_hlt}" "PD10_HLT_TRIVIEW_HLT2_S0P35_SOURCE_CHECKPOINT=${tri_checkpoint_s035}" "PD10_HLT_TRIVIEW_HLT2_S1P00_SOURCE_CHECKPOINT=${tri_checkpoint_s100}")" \
      "${SCRIPT_DIR}/run_pd10_train_hlt_triview.sh"
  )
  submit_job "pd10_hlt_triview_fusion" "${args[@]}"
  tri_job="${submitted_job_id}"
  echo "submitted pd10_hlt_triview_fusion=${tri_job}"
fi

cat <<SUMMARY
pd10_hlt_triview_debug_submission:
  pd10_root: ${PD10_ROOT}
  hlt_cache_dir: ${PD10_HLT_CACHE_DIR}
  hlt2_s0p35_cache_dir: ${PD10_HLT_TRIVIEW_HLT2_S0P35_CACHE_DIR}
  hlt2_s1p00_cache_dir: ${PD10_HLT_TRIVIEW_HLT2_S1P00_CACHE_DIR}
  triview_root: ${PD10_HLT_TRIVIEW_ROOT}
  conda_env: ${CONDA_ENV}
  partition: debug
  sizes: ${PD10_HLT_TRIVIEW_TRAIN_SIZE}/${PD10_HLT_TRIVIEW_VAL_SIZE}/${PD10_HLT_TRIVIEW_FINAL_TEST_SIZE}
  source_warm_start_checkpoint: ${PD10_HLT_TRIVIEW_SOURCE_WARM_START_CHECKPOINT:-none}
  require_source_warm_start: ${PD10_HLT_TRIVIEW_REQUIRE_SOURCE_WARM_START}
  upstream_dependency: ${PD10_HLT_TRIVIEW_UPSTREAM_DEPENDENCY:-none}
  submitted_jobs: ${submit_count}
  skipped_existing: ${skip_count}
  source_jobs:
    hlt_source: ${source_job_by_name[hlt_source]:-skipped_existing}
    hlt2_s0p35_source: ${source_job_by_name[hlt2_s0p35_source]:-skipped_existing}
    hlt2_s1p00_source: ${source_job_by_name[hlt2_s1p00_source]:-skipped_existing}
  fusion_job: ${tri_job:-skipped_existing}
  outputs:
    source_models: ${PD10_HLT_TRIVIEW_SOURCE_MODELS_DIR}
    fusion_model: ${tri_output_dir}/run_report.json
SUMMARY
