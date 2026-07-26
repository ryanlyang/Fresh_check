#!/usr/bin/env bash
# Resume-safe 3M-jet matched A0/P7b validation and confirmed final-test campaign.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
export PYTHONNOUSERSITE=1
source "${PROJECT_DIR}/sbatch/common.sh"

STAGE="${1:-full_validation}"
CAMPAIGN_ID="${2:-${LOCAL_RESIDUAL_FIELD_HIGH_DATA_CAMPAIGN_ID:-p7b_highdata3m_$(date +%Y%m%d_%H%M%S)}}"
case "${STAGE}" in
  prepare|train|validation_report|full_validation|final_test) ;;
  *) echo "stage must be prepare, train, validation_report, full_validation, or final_test" >&2; exit 2 ;;
esac

: "${LOCAL_RESIDUAL_FIELD_HIGH_DATA_ROOT:=${OUTPUT_ROOT}/local_particle_residual_field_high_data_seed_study/${CAMPAIGN_ID}}"
: "${LOCAL_RESIDUAL_FIELD_HIGH_DATA_REFERENCE_ROOT:=${OUTPUT_ROOT}/local_particle_residual_field_curriculum/rebuild_and_pilot_20260720_185817}"
: "${LOCAL_RESIDUAL_FIELD_HIGH_DATA_DATA_DIR:=/home/ryreu/atlas/PracticeTagging/data}"
: "${LOCAL_RESIDUAL_FIELD_HIGH_DATA_MANIFEST:=${LOCAL_RESIDUAL_FIELD_HIGH_DATA_ROOT}/study_manifest.json}"
: "${LOCAL_RESIDUAL_FIELD_HIGH_DATA_SBATCH_ACCOUNT:=reu-aisocial}"
: "${LOCAL_RESIDUAL_FIELD_HIGH_DATA_SBATCH_PARTITION:=tigris}"
: "${LOCAL_RESIDUAL_FIELD_HIGH_DATA_GPU_GRES:=gpu:gh200:1}"
: "${LOCAL_RESIDUAL_FIELD_HIGH_DATA_MIN_FREE_GIB:=150}"

export LOCAL_RESIDUAL_FIELD_HIGH_DATA_CAMPAIGN_ID="${CAMPAIGN_ID}"
export LOCAL_RESIDUAL_FIELD_HIGH_DATA_ROOT LOCAL_RESIDUAL_FIELD_HIGH_DATA_REFERENCE_ROOT
export LOCAL_RESIDUAL_FIELD_HIGH_DATA_DATA_DIR LOCAL_RESIDUAL_FIELD_HIGH_DATA_MANIFEST
export LOCAL_RESIDUAL_FIELD_ROOT="${LOCAL_RESIDUAL_FIELD_HIGH_DATA_ROOT}"
export LOCAL_RESIDUAL_FIELD_DATA_DIR="${LOCAL_RESIDUAL_FIELD_HIGH_DATA_DATA_DIR}"
export LOCAL_RESIDUAL_FIELD_MANIFEST_PATH="${LOCAL_RESIDUAL_FIELD_HIGH_DATA_ROOT}/inputs/split_manifest/split_manifest.json.gz"
export LOCAL_RESIDUAL_FIELD_HLT_CACHE_DIR="${LOCAL_RESIDUAL_FIELD_HIGH_DATA_ROOT}/inputs/hlt_cache"
export LOCAL_RESIDUAL_FIELD_OFFLINE_CACHE_DIR="${LOCAL_RESIDUAL_FIELD_HIGH_DATA_ROOT}/inputs/offline_cache"
export LOCAL_RESIDUAL_FIELD_TARGET_CACHE_DIR="${LOCAL_RESIDUAL_FIELD_HIGH_DATA_ROOT}/targets"
export LOCAL_RESIDUAL_FIELD_RECON_ROOT="${LOCAL_RESIDUAL_FIELD_HIGH_DATA_ROOT}/reconstructors"
export LOCAL_RESIDUAL_FIELD_TAGGER_ROOT="${LOCAL_RESIDUAL_FIELD_HIGH_DATA_ROOT}/taggers"
export LOCAL_RESIDUAL_FIELD_ORACLE_SOURCE_ROOT="${LOCAL_RESIDUAL_FIELD_HIGH_DATA_ROOT}/oracle_training_sources"

# Immutable data inventory. stack_train is intentionally empty because this is
# a matched model comparison, not a learned fusion campaign.
export DATA_DIR="${LOCAL_RESIDUAL_FIELD_HIGH_DATA_DATA_DIR}"
export MANIFEST_PATH="${LOCAL_RESIDUAL_FIELD_MANIFEST_PATH}"
export MODEL_TRAIN_SIZE=3000000
export MODEL_VAL_SIZE=250000
export STACK_TRAIN_SIZE=0
export STACK_VAL_SIZE=500000
export FINAL_TEST_SIZE=1000000
export HLT_CACHE_DIR="${LOCAL_RESIDUAL_FIELD_HLT_CACHE_DIR}"
export HLT_SPLITS="model_train model_val stack_val final_test"
export HLT_PROFILE=fixed_hlt_v2_realistic
export HLT_DEGRADATION_STRENGTH=2.5
export ARCHITECTURE_VIEW_10CLASS_OFFLINE_MANIFEST_PATH="${LOCAL_RESIDUAL_FIELD_MANIFEST_PATH}"
export ARCHITECTURE_VIEW_10CLASS_OFFLINE_CACHE_DIR="${LOCAL_RESIDUAL_FIELD_OFFLINE_CACHE_DIR}"
export ARCHITECTURE_VIEW_10CLASS_OFFLINE_SPLITS="model_train model_val stack_val"
export ARCHITECTURE_VIEW_10CLASS_OFFLINE_DATA_DIRS="${LOCAL_RESIDUAL_FIELD_HIGH_DATA_DATA_DIR}"
export ARCHITECTURE_VIEW_10CLASS_OFFLINE_OVERWRITE=0
export ARCHITECTURE_VIEW_10CLASS_OFFLINE_APPEND=0
export LOCAL_RESIDUAL_FIELD_TARGET_SPLITS="model_train model_val stack_val"
export LOCAL_RESIDUAL_FIELD_INCLUDE_FINAL_TEST_TARGETS=0
export LOCAL_RESIDUAL_FIELD_RECO_SEED=10421
export LOCAL_RESIDUAL_FIELD_TAGGER_SEED=20421
export LOCAL_RESIDUAL_FIELD_TAGGER_EPOCHS=45
export LOCAL_RESIDUAL_FIELD_TAGGER_EARLY_STOP_PATIENCE=6
export OVERWRITE=0
export SKIP_EXISTING=1
export DEVICE="${DEVICE:-cuda}"

fresh_prepare_submitter
mkdir -p "${LOCAL_RESIDUAL_FIELD_HIGH_DATA_ROOT}"
exec 9>>"${LOCAL_RESIDUAL_FIELD_HIGH_DATA_ROOT}/.submission.lock"
if ! flock -n 9; then
  echo "another submitter owns ${LOCAL_RESIDUAL_FIELD_HIGH_DATA_ROOT}" >&2
  exit 3
fi
SUBMISSION_MANIFEST="${LOCAL_RESIDUAL_FIELD_HIGH_DATA_ROOT}/submission_jobs.tsv"
touch "${SUBMISSION_MANIFEST}"

if [[ "${STAGE}" == "prepare" || "${STAGE}" == "full_validation" ]]; then
  if [[ ! -f "${LOCAL_RESIDUAL_FIELD_MANIFEST_PATH}" ]] && ! fresh_is_dry_run; then
    available_kib="$(df -Pk "${LOCAL_RESIDUAL_FIELD_HIGH_DATA_ROOT}" | awk 'NR==2 {print $4}')"
    required_kib="$((LOCAL_RESIDUAL_FIELD_HIGH_DATA_MIN_FREE_GIB * 1024 * 1024))"
    if [[ ! "${available_kib}" =~ ^[0-9]+$ || "${available_kib}" -lt "${required_kib}" ]]; then
      echo "3M source build requires at least ${LOCAL_RESIDUAL_FIELD_HIGH_DATA_MIN_FREE_GIB} GiB free" >&2
      exit 2
    fi
  fi
fi

echo "stage=${STAGE}"
echo "campaign_id=${CAMPAIGN_ID}"
echo "campaign_root=${LOCAL_RESIDUAL_FIELD_HIGH_DATA_ROOT}"
echo "reference_p7b_root=${LOCAL_RESIDUAL_FIELD_HIGH_DATA_REFERENCE_ROOT}"
echo "split_counts=3000000/250000/0/500000/1000000"
echo "matched_seeds=20421,20522,20623"
echo "validation_report=model_val_selection_plus_locked_stack_val"
echo "final_test=sealed_unless_stage_final_test_and_CONFIRM_FINAL_TEST_1"
echo "privileged_final_test_inputs=forbidden"
echo "account=${LOCAL_RESIDUAL_FIELD_HIGH_DATA_SBATCH_ACCOUNT}"
echo "partition=${LOCAL_RESIDUAL_FIELD_HIGH_DATA_SBATCH_PARTITION}"

afterok_arg() {
  local ids=() value
  for value in "$@"; do [[ -n "${value}" ]] && ids+=("${value}"); done
  if ((${#ids[@]})); then
    local joined
    joined="$(IFS=:; echo "${ids[*]}")"
    echo "--dependency=afterok:${joined}"
  fi
}

normalize_dependency() {
  local dependency="$1" ids
  dependency="${dependency#--dependency=afterok:}"
  [[ -n "${dependency}" ]] || { echo ""; return 0; }
  ids="$(printf '%s\n' "${dependency//:/$'\n'}" | sed '/^$/d' | sort | paste -sd: -)"
  echo "--dependency=afterok:${ids}"
}

active_job() {
  local label="$1" completion="$2" expected_dependency="$3"
  local record job_id recorded_dependency status state reason
  record="$(awk -F '\t' -v label="${label}" -v completion="${completion}" '
    $2 == label && $3 == completion { job_id = $4; dependency = $5 }
    END { if (job_id != "") printf "%s\t%s", job_id, dependency }
  ' "${SUBMISSION_MANIFEST}")"
  job_id="${record%%$'\t'*}"
  recorded_dependency=""
  [[ "${record}" == *$'\t'* ]] && recorded_dependency="${record#*$'\t'}"
  [[ -n "${job_id}" ]] || return 1
  status="$(squeue -h -j "${job_id}" -o '%T|%r' 2>/dev/null | head -n 1 || true)"
  [[ -n "${status}" ]] || return 1
  state="${status%%|*}"
  reason="${status#*|}"
  if [[ "$(normalize_dependency "${recorded_dependency}")" != "$(normalize_dependency "${expected_dependency}")" ]] \
    || [[ "${reason}" == *DependencyNeverSatisfied* ]]; then
    echo "stale_label=${label} job=${job_id} state=${state} reason=${reason} action=cancel_and_resubmit" >&2
    scancel "${job_id}" 2>/dev/null || true
    return 1
  fi
  echo "${job_id}"
}

submit_or_reuse() {
  local label="$1" completion="$2" dependency="$3" class="$4" export_overrides="$5" partial_path="$6" script="$7"
  shift 7
  if [[ -f "${completion}" ]]; then
    echo "reuse_label=${label} completion=${completion}" >&2
    echo ""
    return 0
  fi
  local existing job_id
  if existing="$(active_job "${label}" "${completion}" "${dependency}")"; then
    echo "reuse_label=${label} active_job=${existing}" >&2
    echo "${existing}"
    return 0
  fi
  if [[ -n "${partial_path}" && -e "${partial_path}" ]] && ! fresh_is_dry_run; then
    local quarantine="${partial_path}.partial_$(date -u +%Y%m%dT%H%M%SZ)_submit"
    echo "quarantine_label=${label} partial=${partial_path} destination=${quarantine}" >&2
    mv -- "${partial_path}" "${quarantine}"
  fi
  local command=(sbatch --parsable
    --job-name="${label}"
    --partition="${LOCAL_RESIDUAL_FIELD_HIGH_DATA_SBATCH_PARTITION}"
    --account="${LOCAL_RESIDUAL_FIELD_HIGH_DATA_SBATCH_ACCOUNT}")
  [[ -n "${dependency}" ]] && command+=("${dependency}")
  [[ -n "${export_overrides}" ]] && command+=(--export="ALL,${export_overrides}")
  case "${class}" in
    gpu) command+=(--gres="${LOCAL_RESIDUAL_FIELD_HIGH_DATA_GPU_GRES}" --cpus-per-task=16 --mem=500G) ;;
    cpu_big) command+=(--cpus-per-task=24 --mem=512G) ;;
    cpu) command+=(--cpus-per-task=4 --mem=32G) ;;
    *) echo "unknown job class ${class}" >&2; exit 2 ;;
  esac
  command+=("${script}" "$@")
  if fresh_is_dry_run; then
    job_id="DRYRUN_${label//[^A-Za-z0-9_]/_}"
    fresh_print_shell_command "${command[@]}" >&2
  else
    job_id="$("${command[@]}")"
    job_id="${job_id%%;*}"
    [[ "${job_id}" =~ ^[0-9]+$ ]] || {
      echo "invalid sbatch job ID for ${label}: ${job_id}" >&2
      exit 2
    }
    printf '%s\t%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      "${label}" "${completion}" "${job_id}" "$(normalize_dependency "${dependency}")" \
      >>"${SUBMISSION_MANIFEST}"
  fi
  echo "${job_id}"
}

prepare_jobs=()
training_jobs=()
prediction_jobs=()
preflight_job=""

submit_prepare() {
  local split hlt offline target c0 a0 oracle register dependency
  split="$(submit_or_reuse lprf_hd_splits "${LOCAL_RESIDUAL_FIELD_MANIFEST_PATH}" "" cpu_big "" \
    "" sbatch/run_build_fresh_splits.sh)"
  dependency="$(afterok_arg "${split}")"
  hlt="$(submit_or_reuse lprf_hd_hlt \
    "${LOCAL_RESIDUAL_FIELD_HLT_CACHE_DIR}/final_test_fixed_hlt_metadata.json" \
    "${dependency}" cpu_big "" "${LOCAL_RESIDUAL_FIELD_HLT_CACHE_DIR}" \
    sbatch/run_build_fresh_hlt_cache.sh)"
  offline="$(submit_or_reuse lprf_hd_offline \
    "${LOCAL_RESIDUAL_FIELD_OFFLINE_CACHE_DIR}/stack_val_offline_metadata.json" \
    "${dependency}" cpu_big "" "${LOCAL_RESIDUAL_FIELD_OFFLINE_CACHE_DIR}" \
    sbatch/run_cache_architecture_view_offline_inputs.sh)"
  dependency="$(afterok_arg "${hlt}" "${offline}")"
  target="$(submit_or_reuse lprf_hd_targets \
    "${LOCAL_RESIDUAL_FIELD_TARGET_CACHE_DIR}/stack_val_local_particle_residual_fields_metadata.json" \
    "${dependency}" cpu_big "" "${LOCAL_RESIDUAL_FIELD_TARGET_CACHE_DIR}" \
    sbatch/run_cache_local_particle_residual_fields.sh)"
  dependency="$(afterok_arg "${target}")"
  c0="$(submit_or_reuse lprf_hd_C0 \
    "${LOCAL_RESIDUAL_FIELD_RECON_ROOT}/C0/run_report.json" \
    "${dependency}" gpu "" "${LOCAL_RESIDUAL_FIELD_RECON_ROOT}/C0" \
    sbatch/run_train_local_residual_reconstructor.sh C0)"
  a0="$(submit_or_reuse lprf_hd_A0_20421 \
    "${LOCAL_RESIDUAL_FIELD_TAGGER_ROOT}/A0/run_report.json" \
    "${dependency}" gpu "" "${LOCAL_RESIDUAL_FIELD_TAGGER_ROOT}/A0" \
    sbatch/run_train_local_residual_field_tagger.sh A0)"
  oracle="$(submit_or_reuse lprf_hd_Orobust \
    "${LOCAL_RESIDUAL_FIELD_ORACLE_SOURCE_ROOT}/Orobust_light/run_report.json" \
    "${dependency}" gpu \
    "LOCAL_RESIDUAL_FIELD_TAGGER_ROOT=${LOCAL_RESIDUAL_FIELD_ORACLE_SOURCE_ROOT}" \
    "${LOCAL_RESIDUAL_FIELD_ORACLE_SOURCE_ROOT}/Orobust_light" \
    sbatch/run_train_local_residual_field_tagger.sh Orobust_light)"
  dependency="$(afterok_arg "${oracle}")"
  register="$(submit_or_reuse lprf_hd_register \
    "${LOCAL_RESIDUAL_FIELD_TAGGER_ROOT}/Orobust_light/registration_report.json" \
    "${dependency}" cpu "" "${LOCAL_RESIDUAL_FIELD_TAGGER_ROOT}/Orobust_light" \
    sbatch/run_register_local_residual_oracle_teacher.sh \
    Orobust_light "${LOCAL_RESIDUAL_FIELD_ORACLE_SOURCE_ROOT}/Orobust_light")"
  dependency="$(afterok_arg "${c0}" "${a0}" "${register}")"
  preflight_job="$(submit_or_reuse lprf_hd_preflight \
    "${LOCAL_RESIDUAL_FIELD_HIGH_DATA_MANIFEST}" "${dependency}" cpu "" "" \
    sbatch/run_prepare_local_residual_field_high_data_seed_study.sh)"
  prepare_jobs=("${split}" "${hlt}" "${offline}" "${target}" "${c0}" "${a0}" "${oracle}" "${register}" "${preflight_job}")
}

submit_training() {
  local dependency seed job completion
  dependency="$(afterok_arg "${preflight_job}")"
  if [[ -z "${dependency}" && ! -f "${LOCAL_RESIDUAL_FIELD_HIGH_DATA_MANIFEST}" && ! fresh_is_dry_run ]]; then
    echo "training requires a completed or queued high-data preflight manifest" >&2
    exit 2
  fi
  for seed in 20522 20623; do
    completion="${LOCAL_RESIDUAL_FIELD_HIGH_DATA_ROOT}/runs/seed_${seed}/A0/high_data_completion.json"
    job="$(submit_or_reuse "lprf_hd_A0_${seed}" "${completion}" "${dependency}" gpu "" \
      "" \
      sbatch/run_train_local_residual_field_high_data_a0.sh "${seed}")"
    [[ -n "${job}" ]] && training_jobs+=("${job}")
  done
  for seed in 20421 20522 20623; do
    completion="${LOCAL_RESIDUAL_FIELD_HIGH_DATA_ROOT}/runs/seed_${seed}/P7b/high_data_completion.json"
    job="$(submit_or_reuse "lprf_hd_P7b_${seed}" "${completion}" "${dependency}" gpu "" \
      "" \
      sbatch/run_train_local_residual_field_high_data_p7b.sh "${seed}")"
    [[ -n "${job}" ]] && training_jobs+=("${job}")
  done
}

require_training_completions() {
  local seed
  for seed in 20522 20623; do
    fresh_require_file "${LOCAL_RESIDUAL_FIELD_HIGH_DATA_ROOT}/runs/seed_${seed}/A0/high_data_completion.json"
  done
  for seed in 20421 20522 20623; do
    fresh_require_file "${LOCAL_RESIDUAL_FIELD_HIGH_DATA_ROOT}/runs/seed_${seed}/P7b/high_data_completion.json"
  done
}

submit_validation_report() {
  local dependency
  dependency="$(afterok_arg "${training_jobs[@]}")"
  if [[ -z "${dependency}" && ! fresh_is_dry_run ]]; then
    require_training_completions
  fi
  validation_report_job="$(submit_or_reuse lprf_hd_validation \
    "${LOCAL_RESIDUAL_FIELD_HIGH_DATA_ROOT}/validation_report/run_report.json" \
    "${dependency}" cpu "" "" sbatch/run_write_local_residual_field_high_data_report.sh validation)"
}

submit_final_test() {
  fresh_bool_enabled "${CONFIRM_FINAL_TEST}" || {
    echo "final_test requires explicit CONFIRM_FINAL_TEST=1" >&2
    exit 2
  }
  fresh_require_file "${LOCAL_RESIDUAL_FIELD_HIGH_DATA_MANIFEST}"
  fresh_require_file "${LOCAL_RESIDUAL_FIELD_HIGH_DATA_ROOT}/validation_report/run_report.json"
  require_training_completions
  local seed recipe job completion dependency
  for seed in 20421 20522 20623; do
    for recipe in A0 P7b; do
      completion="${LOCAL_RESIDUAL_FIELD_HIGH_DATA_ROOT}/final_test_predictions/seed_${seed}/${recipe}/final_test_predictions_metadata.json"
      job="$(submit_or_reuse "lprf_hd_final_${recipe}_${seed}" "${completion}" "" gpu \
        "CONFIRM_FINAL_TEST=1" "" sbatch/run_predict_local_residual_field_high_data_final_test.sh \
        "${recipe}" "${seed}")"
      [[ -n "${job}" ]] && prediction_jobs+=("${job}")
    done
  done
  dependency="$(afterok_arg "${prediction_jobs[@]}")"
  final_report_job="$(submit_or_reuse lprf_hd_final_report \
    "${LOCAL_RESIDUAL_FIELD_HIGH_DATA_ROOT}/final_report/run_report.json" \
    "${dependency}" cpu "CONFIRM_FINAL_TEST=1" "" \
    sbatch/run_write_local_residual_field_high_data_report.sh final_test)"
}

case "${STAGE}" in
  prepare) submit_prepare ;;
  train) submit_training ;;
  validation_report) submit_validation_report ;;
  full_validation) submit_prepare; submit_training; submit_validation_report ;;
  final_test) submit_final_test ;;
esac

echo "submission_summary_begin"
printf 'prepare_job=%s\n' "${prepare_jobs[@]:-}"
printf 'training_job=%s\n' "${training_jobs[@]:-}"
echo "preflight_job=${preflight_job:-completed_or_not_requested}"
echo "validation_report_job=${validation_report_job:-not_requested}"
printf 'final_prediction_job=%s\n' "${prediction_jobs[@]:-}"
echo "final_report_job=${final_report_job:-not_requested}"
echo "campaign_root=${LOCAL_RESIDUAL_FIELD_HIGH_DATA_ROOT}"
echo "submission_summary_end"
