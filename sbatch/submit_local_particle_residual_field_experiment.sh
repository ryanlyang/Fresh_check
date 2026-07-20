#!/usr/bin/env bash
# Submit one local particle residual-field campaign graph.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_ENV:=atlas_kd}"
export CONDA_ENV
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

fresh_prepare_submitter

: "${LOCAL_RESIDUAL_FIELD_CAMPAIGN_MODE:=pilot}"
case "${LOCAL_RESIDUAL_FIELD_CAMPAIGN_MODE}" in
  highdata)
    : "${LOCAL_RESIDUAL_FIELD_MODEL_TRAIN_SIZE:=5000000}"
    : "${LOCAL_RESIDUAL_FIELD_MODEL_VAL_SIZE:=1000000}"
    : "${LOCAL_RESIDUAL_FIELD_STACK_TRAIN_SIZE:=2000000}"
    : "${LOCAL_RESIDUAL_FIELD_STACK_VAL_SIZE:=1000000}"
    : "${LOCAL_RESIDUAL_FIELD_FINAL_TEST_SIZE:=1000000}"
    ;;
  pilot)
    : "${LOCAL_RESIDUAL_FIELD_MODEL_TRAIN_SIZE:=500000}"
    : "${LOCAL_RESIDUAL_FIELD_MODEL_VAL_SIZE:=150000}"
    : "${LOCAL_RESIDUAL_FIELD_STACK_TRAIN_SIZE:=300000}"
    : "${LOCAL_RESIDUAL_FIELD_STACK_VAL_SIZE:=150000}"
    : "${LOCAL_RESIDUAL_FIELD_FINAL_TEST_SIZE:=150000}"
    ;;
  *)
    echo "LOCAL_RESIDUAL_FIELD_CAMPAIGN_MODE must be pilot or highdata, got ${LOCAL_RESIDUAL_FIELD_CAMPAIGN_MODE}" >&2
    exit 2
    ;;
esac

if [[ -z "${LOCAL_RESIDUAL_FIELD_ROOT:-}" ]]; then
  LOCAL_RESIDUAL_FIELD_ROOT="${OUTPUT_ROOT}/local_particle_residual_field_hltv2_s2p5_${LOCAL_RESIDUAL_FIELD_CAMPAIGN_MODE}_$(date +%Y%m%d_%H%M%S)"
fi
: "${LOCAL_RESIDUAL_FIELD_DATA_DIR:=${PD10_DATA_DIR:-${DATA_DIR}}}"
: "${LOCAL_RESIDUAL_FIELD_INPUTS_DIR:=${LOCAL_RESIDUAL_FIELD_ROOT}/inputs}"
: "${LOCAL_RESIDUAL_FIELD_SPLIT_MANIFEST_DIR:=${LOCAL_RESIDUAL_FIELD_INPUTS_DIR}/split_manifest}"
: "${LOCAL_RESIDUAL_FIELD_MANIFEST_PATH:=${LOCAL_RESIDUAL_FIELD_SPLIT_MANIFEST_DIR}/split_manifest.json.gz}"
: "${LOCAL_RESIDUAL_FIELD_HLT_CACHE_DIR:=${LOCAL_RESIDUAL_FIELD_INPUTS_DIR}/hlt_cache}"
: "${LOCAL_RESIDUAL_FIELD_OFFLINE_CACHE_DIR:=${LOCAL_RESIDUAL_FIELD_INPUTS_DIR}/offline_cache}"
: "${LOCAL_RESIDUAL_FIELD_TARGET_CACHE_DIR:=${LOCAL_RESIDUAL_FIELD_ROOT}/targets}"
: "${LOCAL_RESIDUAL_FIELD_TEACHER_ROOT:=${LOCAL_RESIDUAL_FIELD_ROOT}/offline_teacher_kd}"
: "${LOCAL_RESIDUAL_FIELD_INTERNAL_TEACHER_LOGITS_DIR:=${LOCAL_RESIDUAL_FIELD_TEACHER_ROOT}/teacher_logits}"
: "${LOCAL_RESIDUAL_FIELD_RECON_ROOT:=${LOCAL_RESIDUAL_FIELD_ROOT}/reconstructors}"
: "${LOCAL_RESIDUAL_FIELD_TAGGER_ROOT:=${LOCAL_RESIDUAL_FIELD_ROOT}/taggers}"
: "${LOCAL_RESIDUAL_FIELD_PREDICTION_DIR:=${LOCAL_RESIDUAL_FIELD_ROOT}/predictions}"
: "${LOCAL_RESIDUAL_FIELD_FUSION_DIR:=${LOCAL_RESIDUAL_FIELD_ROOT}/fusion}"
: "${LOCAL_RESIDUAL_FIELD_REPORT_DIR:=${LOCAL_RESIDUAL_FIELD_ROOT}/final_report}"
: "${LOCAL_RESIDUAL_FIELD_HLT_PROFILE:=fixed_hlt_v2_realistic}"
: "${LOCAL_RESIDUAL_FIELD_HLT_DEGRADATION_STRENGTH:=2.5}"
: "${LOCAL_RESIDUAL_FIELD_TARGET_SPLITS:=model_train model_val stack_val}"
: "${LOCAL_RESIDUAL_FIELD_CACHE_SPLITS:=model_train model_val stack_train stack_val final_test}"
: "${LOCAL_RESIDUAL_FIELD_OFFLINE_SPLITS:=${LOCAL_RESIDUAL_FIELD_TARGET_SPLITS}}"
: "${LOCAL_RESIDUAL_FIELD_SUBMIT_SPLITS:=1}"
: "${LOCAL_RESIDUAL_FIELD_SUBMIT_HLT_CACHE:=1}"
: "${LOCAL_RESIDUAL_FIELD_SUBMIT_OFFLINE_CACHE:=1}"
: "${LOCAL_RESIDUAL_FIELD_SUBMIT_TARGETS:=1}"
: "${LOCAL_RESIDUAL_FIELD_SUBMIT_TEACHER_LOGITS:=1}"
: "${LOCAL_RESIDUAL_FIELD_SUBMIT_RECONSTRUCTORS:=1}"
: "${LOCAL_RESIDUAL_FIELD_SUBMIT_TAGGERS:=1}"
: "${LOCAL_RESIDUAL_FIELD_SUBMIT_PREDICTIONS:=1}"
: "${LOCAL_RESIDUAL_FIELD_SUBMIT_FUSION:=1}"
: "${LOCAL_RESIDUAL_FIELD_SUBMIT_REPORT:=1}"
: "${LOCAL_RESIDUAL_FIELD_BOOTSTRAP_DEPENDENCY_FILE:=}"
: "${LOCAL_RESIDUAL_FIELD_RECON_RUN_IDS:=C0 C1 C2 C3 C4 C5 C6}"
: "${LOCAL_RESIDUAL_FIELD_TAGGER_RUN_IDS:=A0 A1 A2 B0 B1 B2 B3 B4 D0 D1 D2 D3 D4 D5 D5_seed1 D5_seed2 D5_seed3 D6 E0 E1 E2 E3 E4 E5 E6 F0 F1 F2 F3 F4 F5}"
: "${LOCAL_RESIDUAL_FIELD_PREDICT_RUN_IDS:=A0 D5 D5_seed1 D5_seed2 D5_seed3 D6 E6 E5 E3}"
: "${LOCAL_RESIDUAL_FIELD_PREDICT_SPLITS:=stack_train stack_val final_test}"
: "${LOCAL_RESIDUAL_FIELD_FUSION_GROUPS:=G0:A0,D5 G1:D5,D5_seed1,D5_seed2,D5_seed3 G2:D5,D6 G3:E6,E5,E3}"
: "${LOCAL_RESIDUAL_FIELD_REQUIRED_FUSION_GROUPS:=G0 G1 G2 G3}"
: "${LOCAL_RESIDUAL_FIELD_BASELINE_CHECKPOINT:=}"
: "${LOCAL_RESIDUAL_FIELD_TEACHER_LOGITS_DIR:=}"
: "${LOCAL_RESIDUAL_FIELD_TEACHER_LOGIT_SPLITS:=model_train model_val stack_val}"
: "${LOCAL_RESIDUAL_FIELD_D6_RECON_RUN_ID:=C5}"
: "${LOCAL_RESIDUAL_FIELD_SBATCH_ACCOUNT:=}"
: "${LOCAL_RESIDUAL_FIELD_SBATCH_PARTITION:=}"
: "${LOCAL_RESIDUAL_FIELD_GPU_GRES:=}"
: "${LOCAL_RESIDUAL_FIELD_GPU_CPUS_PER_TASK:=}"
: "${LOCAL_RESIDUAL_FIELD_GPU_MEM:=}"
: "${LOCAL_RESIDUAL_FIELD_CPU_CPUS_PER_TASK:=}"
: "${LOCAL_RESIDUAL_FIELD_CPU_MEM:=}"

export LOCAL_RESIDUAL_FIELD_ROOT
export LOCAL_RESIDUAL_FIELD_MANIFEST_PATH
export LOCAL_RESIDUAL_FIELD_HLT_CACHE_DIR
export LOCAL_RESIDUAL_FIELD_OFFLINE_CACHE_DIR
export LOCAL_RESIDUAL_FIELD_TARGET_CACHE_DIR
export LOCAL_RESIDUAL_FIELD_TEACHER_ROOT
export LOCAL_RESIDUAL_FIELD_INTERNAL_TEACHER_LOGITS_DIR
export LOCAL_RESIDUAL_FIELD_RECON_ROOT
export LOCAL_RESIDUAL_FIELD_TAGGER_ROOT
export LOCAL_RESIDUAL_FIELD_PREDICTION_DIR
export LOCAL_RESIDUAL_FIELD_FUSION_DIR
export LOCAL_RESIDUAL_FIELD_REPORT_DIR
export LOCAL_RESIDUAL_FIELD_PREDICT_SPLITS
export LOCAL_RESIDUAL_FIELD_REQUIRED_TAGGER_RUN_IDS="${LOCAL_RESIDUAL_FIELD_TAGGER_RUN_IDS}"
export LOCAL_RESIDUAL_FIELD_REQUIRED_FUSION_GROUPS
if [[ -z "${LOCAL_RESIDUAL_FIELD_TEACHER_LOGITS_DIR}" ]]; then
  LOCAL_RESIDUAL_FIELD_TEACHER_LOGITS_DIR="${LOCAL_RESIDUAL_FIELD_INTERNAL_TEACHER_LOGITS_DIR}"
  LOCAL_RESIDUAL_FIELD_USE_INTERNAL_TEACHER_LOGITS=1
else
  LOCAL_RESIDUAL_FIELD_USE_INTERNAL_TEACHER_LOGITS=0
fi
export LOCAL_RESIDUAL_FIELD_TEACHER_LOGITS_DIR
export LOCAL_RESIDUAL_FIELD_USE_INTERNAL_TEACHER_LOGITS
export LOCAL_RESIDUAL_FIELD_TEACHER_LOGIT_SPLITS
export LOCAL_RESIDUAL_FIELD_D6_RECON_RUN_ID
export CONFIRM_FINAL_TEST

dependency_token_is_valid() {
  local token="$1"
  if [[ "${token}" =~ ^[0-9]+$ ]]; then return 0; fi
  if fresh_is_dry_run && [[ "${token}" =~ ^DRYRUN_[A-Za-z0-9_]+$ ]]; then return 0; fi
  return 1
}

submit_job() {
  local label="$1"
  shift
  local sbatch_args=()
  if [[ -n "${LOCAL_RESIDUAL_FIELD_SBATCH_ACCOUNT}" ]]; then
    sbatch_args+=(--account="${LOCAL_RESIDUAL_FIELD_SBATCH_ACCOUNT}")
  fi
  if [[ -n "${LOCAL_RESIDUAL_FIELD_SBATCH_PARTITION}" ]]; then
    sbatch_args+=(--partition="${LOCAL_RESIDUAL_FIELD_SBATCH_PARTITION}")
  fi
  local gpu_job=0
  local arg
  for arg in "$@"; do
    case "${arg}" in
      */run_train_local_residual_reconstructor.sh|*/run_train_local_residual_field_tagger.sh|*/run_predict_local_residual_field_tagger.sh)
        gpu_job=1
        ;;
      */run_pd10_train_teacher.sh|*/run_pd10_cache_teacher_logits.sh)
        gpu_job=1
        ;;
    esac
  done
  if [[ "${gpu_job}" -eq 1 ]]; then
    if [[ -n "${LOCAL_RESIDUAL_FIELD_GPU_GRES}" ]]; then sbatch_args+=(--gres="${LOCAL_RESIDUAL_FIELD_GPU_GRES}"); fi
    if [[ -n "${LOCAL_RESIDUAL_FIELD_GPU_CPUS_PER_TASK}" ]]; then sbatch_args+=(--cpus-per-task="${LOCAL_RESIDUAL_FIELD_GPU_CPUS_PER_TASK}"); fi
    if [[ -n "${LOCAL_RESIDUAL_FIELD_GPU_MEM}" ]]; then sbatch_args+=(--mem="${LOCAL_RESIDUAL_FIELD_GPU_MEM}"); fi
  else
    if [[ -n "${LOCAL_RESIDUAL_FIELD_CPU_CPUS_PER_TASK}" ]]; then sbatch_args+=(--cpus-per-task="${LOCAL_RESIDUAL_FIELD_CPU_CPUS_PER_TASK}"); fi
    if [[ -n "${LOCAL_RESIDUAL_FIELD_CPU_MEM}" ]]; then sbatch_args+=(--mem="${LOCAL_RESIDUAL_FIELD_CPU_MEM}"); fi
  fi
  if fresh_is_dry_run; then
    printf 'DRY_RUN sbatch %s: ' "${label}" >&2
    fresh_print_shell_command sbatch "${sbatch_args[@]}" "$@" >&2
    printf '\n' >&2
    local clean_label="${label//[^A-Za-z0-9_]/_}"
    printf 'DRYRUN_%s\n' "${clean_label}"
    return 0
  fi
  local output
  if ! output="$(sbatch "${sbatch_args[@]}" "$@")"; then
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
    if [[ -n "${item}" ]]; then values+=("${item}"); fi
  done
  if [[ "${#values[@]}" -eq 0 ]]; then return 0; fi
  fresh_join_by_colon "${values[@]}"
}

afterok_args() {
  local dependency="$1"
  shift
  if [[ -n "${dependency}" ]]; then printf '%s\n' --dependency="afterok:${dependency}"; fi
  printf '%s\n' "$@"
}

target_complete() {
  local expected_splits=()
  local split
  fresh_split_words expected_splits "${LOCAL_RESIDUAL_FIELD_TARGET_SPLITS}"
  for split in "${expected_splits[@]}"; do
    [[ -f "${LOCAL_RESIDUAL_FIELD_TARGET_CACHE_DIR}/${split}_local_particle_residual_fields.npz" ]] || return 1
    [[ -f "${LOCAL_RESIDUAL_FIELD_TARGET_CACHE_DIR}/${split}_local_particle_residual_fields_metadata.json" ]] || return 1
  done
  return 0
}

hlt_cache_complete() {
  local expected_splits=()
  local split
  fresh_split_words expected_splits "${LOCAL_RESIDUAL_FIELD_CACHE_SPLITS}"
  for split in "${expected_splits[@]}"; do
    [[ -f "${LOCAL_RESIDUAL_FIELD_HLT_CACHE_DIR}/${split}_fixed_hlt.npz" ]] || return 1
    [[ -f "${LOCAL_RESIDUAL_FIELD_HLT_CACHE_DIR}/${split}_fixed_hlt_metadata.json" ]] || return 1
  done
  return 0
}

offline_cache_complete() {
  local expected_splits=()
  local split
  fresh_split_words expected_splits "${LOCAL_RESIDUAL_FIELD_OFFLINE_SPLITS}"
  for split in "${expected_splits[@]}"; do
    [[ -f "${LOCAL_RESIDUAL_FIELD_OFFLINE_CACHE_DIR}/${split}_offline.npz" ]] || return 1
    [[ -f "${LOCAL_RESIDUAL_FIELD_OFFLINE_CACHE_DIR}/${split}_offline_metadata.json" ]] || return 1
  done
  return 0
}

reco_complete() {
  local run_id="$1"
  [[ -f "${LOCAL_RESIDUAL_FIELD_RECON_ROOT}/${run_id}/best_model_val.pt" && -f "${LOCAL_RESIDUAL_FIELD_RECON_ROOT}/${run_id}/run_report.json" ]]
}

tagger_complete() {
  local run_id="$1"
  [[ -f "${LOCAL_RESIDUAL_FIELD_TAGGER_ROOT}/${run_id}/best_model_val.pt" && -f "${LOCAL_RESIDUAL_FIELD_TAGGER_ROOT}/${run_id}/run_report.json" ]]
}

prediction_complete() {
  local run_id="$1"
  local expected_splits=()
  local split
  fresh_split_words expected_splits "${LOCAL_RESIDUAL_FIELD_PREDICT_SPLITS}"
  for split in "${expected_splits[@]}"; do
    [[ -f "${LOCAL_RESIDUAL_FIELD_PREDICTION_DIR}/${run_id}/${split}_predictions.npz" ]] || return 1
    [[ -f "${LOCAL_RESIDUAL_FIELD_PREDICTION_DIR}/${run_id}/${split}_predictions_metadata.json" ]] || return 1
  done
  return 0
}

external_baseline_checkpoint_is_set() {
  [[ -n "${LOCAL_RESIDUAL_FIELD_BASELINE_CHECKPOINT}" ]]
}

tagger_requires_baseline() {
  case "$1" in
    A1|D1|D2|D5|D5_seed*|D6|E*) return 0 ;;
    *) return 1 ;;
  esac
}

tagger_requires_kd() {
  case "$1" in
    D5|D5_seed*|D6|E*) return 0 ;;
    *) return 1 ;;
  esac
}

tagger_reconstructor_run_id() {
  case "$1" in
    D0|D1|D2|D5|D5_seed*|E*) printf '%s\n' C0 ;;
    D6) printf '%s\n' "${LOCAL_RESIDUAL_FIELD_D6_RECON_RUN_ID}" ;;
    *) return 1 ;;
  esac
}

run_id_in_list() {
  local needle="$1"
  shift
  local item
  for item in "$@"; do
    if [[ "${item}" == "${needle}" ]]; then return 0; fi
  done
  return 1
}

teacher_logits_split_complete() {
  local split="$1"
  local root="${LOCAL_RESIDUAL_FIELD_TEACHER_LOGITS_DIR}"
  [[ -n "${root}" ]] || return 1
  [[ -f "${root}/${split}_teacher_logits.npz" \
    || -f "${root}/${split}_logits.npz" \
    || -f "${root}/${split}.npz" \
    || -f "${root}/${split}_predictions.npz" \
    || -f "${root}/offline_part_teacher_10class/${split}_teacher_logits.npz" \
    || -f "${root}/offline_part_teacher_10class/${split}_logits.npz" \
    || -f "${root}/offline_part_teacher_10class/${split}.npz" \
    || -f "${root}/offline_part_teacher_10class/${split}_predictions.npz" ]]
}

teacher_logits_complete() {
  teacher_logits_split_complete model_train || return 1
  teacher_logits_split_complete model_val || return 1
  teacher_logits_split_complete stack_val || return 1
  return 0
}

internal_teacher_logits_will_be_built() {
  fresh_bool_enabled "${LOCAL_RESIDUAL_FIELD_USE_INTERNAL_TEACHER_LOGITS}" \
    && fresh_bool_enabled "${LOCAL_RESIDUAL_FIELD_SUBMIT_TEACHER_LOGITS}"
}

baseline_checkpoint_path_for_campaign() {
  if external_baseline_checkpoint_is_set; then
    printf '%s\n' "${LOCAL_RESIDUAL_FIELD_BASELINE_CHECKPOINT}"
  else
    printf '%s\n' "${LOCAL_RESIDUAL_FIELD_TAGGER_ROOT}/A0/best_model_val.pt"
  fi
}

preflight_campaign_requirements() {
  local tagger_ids=("$@")
  local needs_baseline=0
  local needs_kd=0
  local run_id
  local needed_reco=""
  local a0_available=0
  if external_baseline_checkpoint_is_set; then
    fresh_require_file "${LOCAL_RESIDUAL_FIELD_BASELINE_CHECKPOINT}"
  fi
  if tagger_complete A0; then a0_available=1; fi
  for run_id in "${tagger_ids[@]}"; do
    if [[ "${run_id}" == "A0" ]]; then a0_available=1; fi
    if tagger_requires_baseline "${run_id}" && ! { fresh_bool_enabled "${SKIP_EXISTING}" && tagger_complete "${run_id}"; }; then
      needs_baseline=1
      if ! external_baseline_checkpoint_is_set && [[ "${a0_available}" -ne 1 ]]; then
        echo "${run_id} requires a baseline, but A0 is not complete and does not appear earlier in LOCAL_RESIDUAL_FIELD_TAGGER_RUN_IDS." >&2
        echo "Put A0 before warm-start runs, or set LOCAL_RESIDUAL_FIELD_BASELINE_CHECKPOINT to an existing HLT ParT checkpoint." >&2
        exit 2
      fi
    fi
    if tagger_requires_kd "${run_id}" && ! { fresh_bool_enabled "${SKIP_EXISTING}" && tagger_complete "${run_id}"; }; then
      needs_kd=1
    fi
    if needed_reco="$(tagger_reconstructor_run_id "${run_id}")"; then
      if [[ -z "${needed_reco}" ]]; then
        echo "${run_id} resolved an empty reconstructor run ID." >&2
        exit 2
      fi
      if ! reco_complete "${needed_reco}" && ! run_id_in_list "${needed_reco}" "${configured_recon_ids[@]}"; then
        echo "${run_id} requires reconstructor ${needed_reco}, but it is neither complete nor listed in LOCAL_RESIDUAL_FIELD_RECON_RUN_IDS." >&2
        exit 2
      fi
    fi
  done
  if [[ "${needs_baseline}" -eq 1 ]] && ! external_baseline_checkpoint_is_set; then
    if ! tagger_complete A0 && ! run_id_in_list A0 "${tagger_ids[@]}"; then
      echo "Requested warm-start taggers require a baseline, but LOCAL_RESIDUAL_FIELD_BASELINE_CHECKPOINT is empty and A0 is neither complete nor queued." >&2
      echo "Queue A0 too, or set LOCAL_RESIDUAL_FIELD_BASELINE_CHECKPOINT to an existing HLT ParT checkpoint." >&2
      exit 2
    fi
  fi
  if [[ "${needs_kd}" -eq 1 ]] && ! teacher_logits_complete && ! internal_teacher_logits_will_be_built; then
    echo "Requested KD taggers require LOCAL_RESIDUAL_FIELD_TEACHER_LOGITS_DIR with model_train/model_val/stack_val logits." >&2
    echo "Expected one of <split>_teacher_logits.npz, <split>_logits.npz, <split>.npz, or <split>_predictions.npz under: ${LOCAL_RESIDUAL_FIELD_TEACHER_LOGITS_DIR:-<empty>}" >&2
    exit 2
  fi
}

report_required_recon_ids() {
  local output=()
  local tagger_ids=()
  local run_id
  local seen=0
  local needs_d6=0
  fresh_split_words output "${LOCAL_RESIDUAL_FIELD_RECON_RUN_IDS}"
  fresh_split_words tagger_ids "${LOCAL_RESIDUAL_FIELD_TAGGER_RUN_IDS}"
  for run_id in "${tagger_ids[@]}"; do
    if [[ "${run_id}" == "D6" ]]; then
      needs_d6=1
      break
    fi
  done
  if [[ "${needs_d6}" -eq 1 && -n "${LOCAL_RESIDUAL_FIELD_D6_RECON_RUN_ID}" ]]; then
    seen=0
    for run_id in "${output[@]}"; do
      if [[ "${run_id}" == "${LOCAL_RESIDUAL_FIELD_D6_RECON_RUN_ID}" ]]; then
        seen=1
        break
      fi
    done
    if [[ "${seen}" -eq 0 ]]; then
      output+=("${LOCAL_RESIDUAL_FIELD_D6_RECON_RUN_ID}")
    fi
  fi
  local old_ifs="${IFS}"
  IFS=' '
  printf '%s\n' "${output[*]}"
  IFS="${old_ifs}"
}

submitter_log_dir="${LOCAL_RESIDUAL_FIELD_ROOT}/submission_logs/local_residual_field_$(date +%Y%m%d_%H%M%S)"
fresh_claim_new_dir "${submitter_log_dir}"
if ! fresh_is_dry_run; then
  {
    echo "created_at=$(date -Is)"
    echo "project_dir=${PROJECT_DIR}"
    echo "source_commit=$(fresh_source_commit)"
    echo "source_status_hash=$(fresh_source_status_hash)"
    echo "root=${LOCAL_RESIDUAL_FIELD_ROOT}"
    echo "mode=${LOCAL_RESIDUAL_FIELD_CAMPAIGN_MODE}"
    echo "data_dir=${LOCAL_RESIDUAL_FIELD_DATA_DIR}"
    echo "hlt_profile=${LOCAL_RESIDUAL_FIELD_HLT_PROFILE}"
    echo "hlt_degradation_strength=${LOCAL_RESIDUAL_FIELD_HLT_DEGRADATION_STRENGTH}"
    echo "sizes=${LOCAL_RESIDUAL_FIELD_MODEL_TRAIN_SIZE}/${LOCAL_RESIDUAL_FIELD_MODEL_VAL_SIZE}/${LOCAL_RESIDUAL_FIELD_STACK_TRAIN_SIZE}/${LOCAL_RESIDUAL_FIELD_STACK_VAL_SIZE}/${LOCAL_RESIDUAL_FIELD_FINAL_TEST_SIZE}"
    echo "recon_run_ids=${LOCAL_RESIDUAL_FIELD_RECON_RUN_IDS}"
    echo "tagger_run_ids=${LOCAL_RESIDUAL_FIELD_TAGGER_RUN_IDS}"
    echo "predict_run_ids=${LOCAL_RESIDUAL_FIELD_PREDICT_RUN_IDS}"
    echo "fusion_groups=${LOCAL_RESIDUAL_FIELD_FUSION_GROUPS}"
  } > "${submitter_log_dir}/metadata.txt"
fi

echo "local_residual_field_submission_start:"
echo "  root: ${LOCAL_RESIDUAL_FIELD_ROOT}"
echo "  mode: ${LOCAL_RESIDUAL_FIELD_CAMPAIGN_MODE}"
echo "  hlt_profile: ${LOCAL_RESIDUAL_FIELD_HLT_PROFILE}"
echo "  hlt_degradation_strength: ${LOCAL_RESIDUAL_FIELD_HLT_DEGRADATION_STRENGTH}"
echo "  sizes: ${LOCAL_RESIDUAL_FIELD_MODEL_TRAIN_SIZE}/${LOCAL_RESIDUAL_FIELD_MODEL_VAL_SIZE}/${LOCAL_RESIDUAL_FIELD_STACK_TRAIN_SIZE}/${LOCAL_RESIDUAL_FIELD_STACK_VAL_SIZE}/${LOCAL_RESIDUAL_FIELD_FINAL_TEST_SIZE}"

declare -a configured_tagger_ids=()
declare -a configured_recon_ids=()
fresh_split_words configured_recon_ids "${LOCAL_RESIDUAL_FIELD_RECON_RUN_IDS}"
fresh_split_words configured_tagger_ids "${LOCAL_RESIDUAL_FIELD_TAGGER_RUN_IDS}"
LOCAL_RESIDUAL_FIELD_REQUIRED_RECON_RUN_IDS="$(report_required_recon_ids)"
export LOCAL_RESIDUAL_FIELD_REQUIRED_RECON_RUN_IDS
preflight_campaign_requirements "${configured_tagger_ids[@]}"

split_jid=""
hlt_jid=""
offline_jid=""
target_jid=""

export DATA_DIR="${LOCAL_RESIDUAL_FIELD_DATA_DIR}"
export MANIFEST_PATH="${LOCAL_RESIDUAL_FIELD_MANIFEST_PATH}"
export MODEL_TRAIN_SIZE="${LOCAL_RESIDUAL_FIELD_MODEL_TRAIN_SIZE}"
export MODEL_VAL_SIZE="${LOCAL_RESIDUAL_FIELD_MODEL_VAL_SIZE}"
export STACK_TRAIN_SIZE="${LOCAL_RESIDUAL_FIELD_STACK_TRAIN_SIZE}"
export STACK_VAL_SIZE="${LOCAL_RESIDUAL_FIELD_STACK_VAL_SIZE}"
export FINAL_TEST_SIZE="${LOCAL_RESIDUAL_FIELD_FINAL_TEST_SIZE}"
if fresh_bool_enabled "${LOCAL_RESIDUAL_FIELD_SUBMIT_SPLITS}"; then
  if fresh_bool_enabled "${SKIP_EXISTING}" && [[ -f "${LOCAL_RESIDUAL_FIELD_MANIFEST_PATH}" ]]; then
    echo "skip splits: manifest exists"
  else
    split_jid="$(submit_job local_residual_splits "${SCRIPT_DIR}/run_build_fresh_splits.sh")"
  fi
fi

input_dep="${split_jid}"
export HLT_CACHE_DIR="${LOCAL_RESIDUAL_FIELD_HLT_CACHE_DIR}"
export HLT_SPLITS="${LOCAL_RESIDUAL_FIELD_CACHE_SPLITS}"
export HLT_PROFILE="${LOCAL_RESIDUAL_FIELD_HLT_PROFILE}"
export HLT_DEGRADATION_STRENGTH="${LOCAL_RESIDUAL_FIELD_HLT_DEGRADATION_STRENGTH}"
if fresh_bool_enabled "${LOCAL_RESIDUAL_FIELD_SUBMIT_HLT_CACHE}"; then
  if fresh_bool_enabled "${SKIP_EXISTING}" && hlt_cache_complete; then
    echo "skip hlt cache: cache exists"
  else
    mapfile -t args < <(afterok_args "${input_dep}" "${SCRIPT_DIR}/run_build_fresh_hlt_cache.sh")
    hlt_jid="$(submit_job local_residual_hlt_cache "${args[@]}")"
  fi
fi

export ARCHITECTURE_VIEW_10CLASS_OFFLINE_MANIFEST_PATH="${LOCAL_RESIDUAL_FIELD_MANIFEST_PATH}"
export ARCHITECTURE_VIEW_10CLASS_OFFLINE_CACHE_DIR="${LOCAL_RESIDUAL_FIELD_OFFLINE_CACHE_DIR}"
export ARCHITECTURE_VIEW_10CLASS_OFFLINE_SPLITS="${LOCAL_RESIDUAL_FIELD_OFFLINE_SPLITS}"
export ARCHITECTURE_VIEW_10CLASS_OFFLINE_DATA_DIRS="${LOCAL_RESIDUAL_FIELD_DATA_DIR}"
export ARCHITECTURE_VIEW_10CLASS_OFFLINE_OVERWRITE="${OVERWRITE}"
if fresh_bool_enabled "${LOCAL_RESIDUAL_FIELD_SUBMIT_OFFLINE_CACHE}"; then
  if fresh_bool_enabled "${SKIP_EXISTING}" && offline_cache_complete; then
    echo "skip offline cache: cache exists"
  else
    mapfile -t args < <(afterok_args "${input_dep}" "${SCRIPT_DIR}/run_cache_architecture_view_offline_inputs.sh")
    offline_jid="$(submit_job local_residual_offline_cache "${args[@]}")"
  fi
fi

teacher_jid=""
teacher_logits_jid=""
if internal_teacher_logits_will_be_built; then
  export PD10_ROOT="${LOCAL_RESIDUAL_FIELD_TEACHER_ROOT}"
  export PD10_MANIFEST_PATH="${LOCAL_RESIDUAL_FIELD_MANIFEST_PATH}"
  export PD10_HLT_CACHE_DIR="${LOCAL_RESIDUAL_FIELD_HLT_CACHE_DIR}"
  export PD10_OFFLINE_CACHE_DIR="${LOCAL_RESIDUAL_FIELD_OFFLINE_CACHE_DIR}"
  export PD10_TEACHERS_DIR="${LOCAL_RESIDUAL_FIELD_TEACHER_ROOT}/teachers"
  export PD10_TEACHER_LOGITS_DIR="${LOCAL_RESIDUAL_FIELD_INTERNAL_TEACHER_LOGITS_DIR}"
  export PD10_DATA_DIR="${LOCAL_RESIDUAL_FIELD_DATA_DIR}"
  export PD10_MODEL_TRAIN_SIZE="${LOCAL_RESIDUAL_FIELD_MODEL_TRAIN_SIZE}"
  export PD10_MODEL_VAL_SIZE="${LOCAL_RESIDUAL_FIELD_MODEL_VAL_SIZE}"
  export PD10_FINAL_TEST_SIZE="${LOCAL_RESIDUAL_FIELD_FINAL_TEST_SIZE}"
  export PD10_TEACHER_SKIP_FINAL_TEST=1
  export PD10_TEACHER_LOGIT_SPLITS="${LOCAL_RESIDUAL_FIELD_TEACHER_LOGIT_SPLITS}"
  teacher_dep="${offline_jid}"
  if [[ -z "${teacher_dep}" ]]; then teacher_dep="${input_dep}"; fi
  if fresh_bool_enabled "${SKIP_EXISTING}" \
    && [[ -f "${PD10_TEACHERS_DIR}/offline_part_teacher_10class/best_model_val.pt" ]] \
    && [[ -f "${PD10_TEACHERS_DIR}/offline_part_teacher_10class/run_report.json" ]]; then
    echo "skip offline KD teacher: teacher exists"
  else
    mapfile -t args < <(afterok_args "${teacher_dep}" "${SCRIPT_DIR}/run_pd10_train_teacher.sh" offline)
    teacher_jid="$(submit_job local_residual_offline_kd_teacher "${args[@]}")"
  fi
  if fresh_bool_enabled "${SKIP_EXISTING}" && teacher_logits_complete; then
    echo "skip offline KD teacher logits: logits exist"
  else
    mapfile -t args < <(afterok_args "${teacher_jid:-${teacher_dep}}" "${SCRIPT_DIR}/run_pd10_cache_teacher_logits.sh" offline)
    teacher_logits_jid="$(submit_job local_residual_offline_kd_logits "${args[@]}")"
  fi
fi

cache_dep="$(join_nonempty_by_colon "${hlt_jid}" "${offline_jid}")"
if [[ -z "${cache_dep}" ]]; then cache_dep="${input_dep}"; fi
export LOCAL_RESIDUAL_FIELD_TARGET_SPLITS
if fresh_bool_enabled "${LOCAL_RESIDUAL_FIELD_SUBMIT_TARGETS}"; then
  if fresh_bool_enabled "${SKIP_EXISTING}" && target_complete; then
    echo "skip targets: target cache exists"
  else
    mapfile -t args < <(afterok_args "${cache_dep}" "${SCRIPT_DIR}/run_cache_local_particle_residual_fields.sh")
    target_jid="$(submit_job local_residual_targets "${args[@]}")"
  fi
fi

base_dep="$(join_nonempty_by_colon "${target_jid}")"
if [[ -z "${base_dep}" ]]; then base_dep="${cache_dep}"; fi

declare -A reco_jobs=()
if fresh_bool_enabled "${LOCAL_RESIDUAL_FIELD_SUBMIT_RECONSTRUCTORS}"; then
  fresh_split_words recon_ids "${LOCAL_RESIDUAL_FIELD_RECON_RUN_IDS}"
  for run_id in "${recon_ids[@]}"; do
    if fresh_bool_enabled "${SKIP_EXISTING}" && reco_complete "${run_id}"; then
      echo "skip reconstructor ${run_id}: complete"
      reco_jobs["${run_id}"]=""
      continue
    fi
    mapfile -t args < <(afterok_args "${base_dep}" "${SCRIPT_DIR}/run_train_local_residual_reconstructor.sh" "${run_id}")
    reco_jobs["${run_id}"]="$(submit_job "local_residual_reco_${run_id}" "${args[@]}")"
  done
fi

declare -A tagger_jobs=()
tagger_dependency_for() {
  local run_id="$1"
  local baseline_dep=""
  local kd_dep=""
  if tagger_requires_baseline "${run_id}" && ! external_baseline_checkpoint_is_set && ! tagger_complete A0; then
    baseline_dep="${tagger_jobs[A0]:-}"
  fi
  if tagger_requires_kd "${run_id}" && ! teacher_logits_complete; then
    kd_dep="${teacher_logits_jid}"
  fi
  case "${run_id}" in
    A1) join_nonempty_by_colon "${base_dep}" "${baseline_dep}" ;;
    D0|D1|D2|D5|D5_seed*|E* ) join_nonempty_by_colon "${base_dep}" "${baseline_dep}" "${kd_dep}" "${reco_jobs[C0]:-}" ;;
    D6) join_nonempty_by_colon "${base_dep}" "${baseline_dep}" "${kd_dep}" "${reco_jobs[${LOCAL_RESIDUAL_FIELD_D6_RECON_RUN_ID}]:-}" ;;
    *) printf '%s\n' "${base_dep}" ;;
  esac
}

if fresh_bool_enabled "${LOCAL_RESIDUAL_FIELD_SUBMIT_TAGGERS}"; then
  for run_id in "${configured_tagger_ids[@]}"; do
    if fresh_bool_enabled "${SKIP_EXISTING}" && tagger_complete "${run_id}"; then
      echo "skip tagger ${run_id}: complete"
      tagger_jobs["${run_id}"]=""
      continue
    fi
    dep="$(tagger_dependency_for "${run_id}")"
    mapfile -t args < <(afterok_args "${dep}" "${SCRIPT_DIR}/run_train_local_residual_field_tagger.sh" "${run_id}")
    saved_baseline_checkpoint="${LOCAL_RESIDUAL_FIELD_BASELINE_CHECKPOINT:-}"
    if tagger_requires_baseline "${run_id}"; then
      export LOCAL_RESIDUAL_FIELD_BASELINE_CHECKPOINT
      LOCAL_RESIDUAL_FIELD_BASELINE_CHECKPOINT="$(baseline_checkpoint_path_for_campaign)"
    fi
    tagger_jobs["${run_id}"]="$(submit_job "local_residual_tagger_${run_id}" "${args[@]}")"
    if [[ -n "${saved_baseline_checkpoint}" ]]; then
      LOCAL_RESIDUAL_FIELD_BASELINE_CHECKPOINT="${saved_baseline_checkpoint}"
      export LOCAL_RESIDUAL_FIELD_BASELINE_CHECKPOINT
    else
      LOCAL_RESIDUAL_FIELD_BASELINE_CHECKPOINT=""
      export LOCAL_RESIDUAL_FIELD_BASELINE_CHECKPOINT
    fi
  done
fi

declare -A predict_jobs=()
if fresh_bool_enabled "${LOCAL_RESIDUAL_FIELD_SUBMIT_PREDICTIONS}"; then
  fresh_split_words predict_ids "${LOCAL_RESIDUAL_FIELD_PREDICT_RUN_IDS}"
  for run_id in "${predict_ids[@]}"; do
    if fresh_bool_enabled "${SKIP_EXISTING}" && prediction_complete "${run_id}"; then
      echo "skip prediction ${run_id}: complete"
      predict_jobs["${run_id}"]=""
      continue
    fi
    dep="${tagger_jobs[${run_id}]:-}"
    if [[ -z "${dep}" ]]; then dep="${base_dep}"; fi
    mapfile -t args < <(afterok_args "${dep}" "${SCRIPT_DIR}/run_predict_local_residual_field_tagger.sh" "${run_id}")
    predict_jobs["${run_id}"]="$(submit_job "local_residual_predict_${run_id}" "${args[@]}")"
  done
fi

fusion_jid=""
if fresh_bool_enabled "${LOCAL_RESIDUAL_FIELD_SUBMIT_FUSION}"; then
  fusion_dep="$(join_nonempty_by_colon "${predict_jobs[@]}")"
  mapfile -t args < <(afterok_args "${fusion_dep}" "${SCRIPT_DIR}/run_local_residual_field_fusion.sh")
  fusion_jid="$(submit_job local_residual_fusion "${args[@]}")"
fi

report_jid=""
if fresh_bool_enabled "${LOCAL_RESIDUAL_FIELD_SUBMIT_REPORT}"; then
  mapfile -t args < <(afterok_args "${fusion_jid}" "${SCRIPT_DIR}/run_write_local_residual_field_report.sh")
  report_jid="$(submit_job local_residual_report "${args[@]}")"
fi

if [[ -n "${LOCAL_RESIDUAL_FIELD_BOOTSTRAP_DEPENDENCY_FILE}" ]]; then
  bootstrap_reco_jid="${reco_jobs[C0]:-}"
  bootstrap_a0_jid="${tagger_jobs[A0]:-}"
  if [[ -z "${bootstrap_reco_jid}" || -z "${bootstrap_a0_jid}" ]]; then
    echo "bootstrap dependency receipt requires newly submitted C0 and A0 jobs" >&2
    exit 2
  fi
  bootstrap_dependency="$(join_nonempty_by_colon "${bootstrap_reco_jid}" "${bootstrap_a0_jid}")"
  mkdir -p "$(dirname "${LOCAL_RESIDUAL_FIELD_BOOTSTRAP_DEPENDENCY_FILE}")"
  printf '%s\n' "${bootstrap_dependency}" > "${LOCAL_RESIDUAL_FIELD_BOOTSTRAP_DEPENDENCY_FILE}"
fi

write_submitted_jobs_json() {
  echo "{"
  echo "  \"root\": \"${LOCAL_RESIDUAL_FIELD_ROOT}\","
  echo "  \"splits\": \"${split_jid}\","
  echo "  \"hlt_cache\": \"${hlt_jid}\","
  echo "  \"offline_cache\": \"${offline_jid}\","
  echo "  \"targets\": \"${target_jid}\","
  echo "  \"offline_kd_teacher\": \"${teacher_jid}\","
  echo "  \"offline_kd_logits\": \"${teacher_logits_jid}\","
  echo "  \"fusion\": \"${fusion_jid}\","
  echo "  \"report\": \"${report_jid}\""
  echo "}"
}
if fresh_is_dry_run; then
  write_submitted_jobs_json
else
  write_submitted_jobs_json | tee "${submitter_log_dir}/submitted_jobs.json"
fi

echo "local_residual_field_submission_complete:"
echo "  root: ${LOCAL_RESIDUAL_FIELD_ROOT}"
echo "  fusion: ${fusion_jid:-none}"
echo "  report: ${report_jid:-none}"
echo "  note: G4/G5 particle-view training hooks are exposed in code; queue them after a concrete particle-view training runner lands."
