#!/usr/bin/env bash
# Queue the HLT v2 realistic-degradation Step 4 model-val baseline sweep.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

if [[ -z "${HLT_V2_BASELINE_SWEEP_ROOT:-}" ]]; then
  HLT_V2_BASELINE_SWEEP_ROOT="${OUTPUT_ROOT}/hlt_v2_baseline_sweep_$(date +%Y%m%d_%H%M%S)"
fi
: "${HLT_V2_BASELINE_SWEEP_STRENGTHS:=0.0 0.75 1.0 1.25}"
: "${HLT_V2_BASELINE_MODEL_TRAIN_SIZE:=500000}"
: "${HLT_V2_BASELINE_MODEL_VAL_SIZE:=150000}"
: "${HLT_V2_BASELINE_STACK_TRAIN_SIZE:=10}"
: "${HLT_V2_BASELINE_STACK_VAL_SIZE:=10}"
: "${HLT_V2_BASELINE_FINAL_TEST_SIZE:=10}"
: "${HLT_V2_BASELINE_EPOCHS:=20}"
: "${HLT_V2_BASELINE_BATCH_SIZE:=${BATCH_SIZE}}"
: "${HLT_V2_BASELINE_LR:=${LR}}"
: "${HLT_V2_BASELINE_WEIGHT_DECAY:=${WEIGHT_DECAY}}"
: "${HLT_V2_BASELINE_EARLY_STOP_PATIENCE:=${EARLY_STOP_PATIENCE}}"
: "${HLT_V2_BASELINE_TEACHER_MODEL_SIZE:=${PD10_TEACHER_MODEL_SIZE}}"
: "${HLT_V2_BASELINE_HLT_PROFILE:=fixed_hlt_v2_realistic}"
: "${HLT_V2_BASELINE_HLT_SPLITS:=model_train model_val}"

fresh_prepare_submitter

strength_tag() {
  local value="$1"
  local tag="${value//-/m}"
  tag="${tag//./p}"
  printf '%s' "${tag}"
}

print_env_args() {
  local arg
  for arg in "$@"; do
    printf ' %q' "${arg}"
  done
}

SUBMITTED_JID=""
submit_sbatch() {
  local label="$1"
  shift
  local -a env_args=()
  while (($#)); do
    if [[ "$1" == "--" ]]; then
      shift
      break
    fi
    env_args+=("$1")
    shift
  done
  local -a cmd=(sbatch --parsable "$@")
  if fresh_is_dry_run; then
    SUBMITTED_JID="dryrun_${label}"
    printf 'DRY_RUN %s:' "${label}"
    printf ' env'
    print_env_args "${env_args[@]}"
    printf ' '
    fresh_print_shell_command "${cmd[@]}"
    printf '\n'
  else
    SUBMITTED_JID="$(env "${env_args[@]}" "${cmd[@]}")"
  fi
  echo "${label}=${SUBMITTED_JID}"
}

shared_root="${HLT_V2_BASELINE_SWEEP_ROOT}/shared_inputs"
manifest_dir="${shared_root}/split_manifest"
manifest_path="${manifest_dir}/split_manifest.json.gz"

common_pd10_env=(
  "PD10_DATA_DIR=${PD10_DATA_DIR}"
  "PD10_MODEL_TRAIN_SIZE=${HLT_V2_BASELINE_MODEL_TRAIN_SIZE}"
  "PD10_MODEL_VAL_SIZE=${HLT_V2_BASELINE_MODEL_VAL_SIZE}"
  "PD10_STACK_TRAIN_SIZE=${HLT_V2_BASELINE_STACK_TRAIN_SIZE}"
  "PD10_STACK_VAL_SIZE=${HLT_V2_BASELINE_STACK_VAL_SIZE}"
  "PD10_FINAL_TEST_SIZE=${HLT_V2_BASELINE_FINAL_TEST_SIZE}"
  "PD10_SPLIT_MANIFEST_DIR=${manifest_dir}"
  "PD10_MANIFEST_PATH=${manifest_path}"
  "EPOCHS=${HLT_V2_BASELINE_EPOCHS}"
  "BATCH_SIZE=${HLT_V2_BASELINE_BATCH_SIZE}"
  "LR=${HLT_V2_BASELINE_LR}"
  "WEIGHT_DECAY=${HLT_V2_BASELINE_WEIGHT_DECAY}"
  "EARLY_STOP_PATIENCE=${HLT_V2_BASELINE_EARLY_STOP_PATIENCE}"
  "PD10_TEACHER_MODEL_SIZE=${HLT_V2_BASELINE_TEACHER_MODEL_SIZE}"
  "PD10_TEACHER_SKIP_FINAL_TEST=1"
)

submit_sbatch "hltv2_splits" \
  "PD10_ROOT=${shared_root}" \
  "${common_pd10_env[@]}" \
  -- "${SCRIPT_DIR}/run_pd10_build_splits.sh"
split_jid="${SUBMITTED_JID}"

offline_root="${HLT_V2_BASELINE_SWEEP_ROOT}/offline_reference"
submit_sbatch "hltv2_offline" \
  "PD10_ROOT=${offline_root}" \
  "PD10_HLT_CACHE_DIR=${offline_root}/unused_hlt_cache" \
  "PD10_TEACHERS_DIR=${offline_root}/teachers" \
  "${common_pd10_env[@]}" \
  -- --dependency="afterok:${split_jid}" "${SCRIPT_DIR}/run_pd10_train_teacher.sh" offline
offline_jid="${SUBMITTED_JID}"

fresh_split_words strength_values "${HLT_V2_BASELINE_SWEEP_STRENGTHS}"
hlt_teacher_jids=()
for strength in "${strength_values[@]}"; do
  tag="$(strength_tag "${strength}")"
  run_root="${HLT_V2_BASELINE_SWEEP_ROOT}/hlt_v2_strength_${tag}"
  cache_dir="${run_root}/hlt_cache"
  teachers_dir="${run_root}/teachers"
  submit_sbatch "hltv2_cache_${tag}" \
    "PD10_ROOT=${run_root}" \
    "PD10_HLT_CACHE_DIR=${cache_dir}" \
    "PD10_HLT_PROFILE=${HLT_V2_BASELINE_HLT_PROFILE}" \
    "PD10_HLT_DEGRADATION_STRENGTH=${strength}" \
    "PD10_HLT_SPLITS=${HLT_V2_BASELINE_HLT_SPLITS}" \
    "${common_pd10_env[@]}" \
    -- --dependency="afterok:${split_jid}" "${SCRIPT_DIR}/run_pd10_build_hlt_cache.sh"
  cache_jid="${SUBMITTED_JID}"

  submit_sbatch "hltv2_train_${tag}" \
    "PD10_ROOT=${run_root}" \
    "PD10_HLT_CACHE_DIR=${cache_dir}" \
    "PD10_TEACHERS_DIR=${teachers_dir}" \
    "PD10_HLT_PROFILE=${HLT_V2_BASELINE_HLT_PROFILE}" \
    "PD10_HLT_DEGRADATION_STRENGTH=${strength}" \
    "${common_pd10_env[@]}" \
    -- --dependency="afterok:${cache_jid}" "${SCRIPT_DIR}/run_pd10_train_teacher.sh" hlt
  hlt_teacher_jids+=("${SUBMITTED_JID}")
done

report_deps=("${offline_jid}" "${hlt_teacher_jids[@]}")
report_dep="$(IFS=:; echo "${report_deps[*]}")"
submit_sbatch "hltv2_report" \
  "HLT_V2_BASELINE_SWEEP_ROOT=${HLT_V2_BASELINE_SWEEP_ROOT}" \
  "HLT_V2_BASELINE_SWEEP_REPORT_DIR=${HLT_V2_BASELINE_SWEEP_ROOT}/baseline_sweep_report" \
  "HLT_V2_BASELINE_SWEEP_STRENGTHS=${HLT_V2_BASELINE_SWEEP_STRENGTHS}" \
  -- --dependency="afterok:${report_dep}" "${SCRIPT_DIR}/run_hlt_v2_baseline_sweep_report.sh"
report_jid="${SUBMITTED_JID}"

cat <<EOF
hlt_v2_baseline_sweep_submit_complete:
  sweep_root: ${HLT_V2_BASELINE_SWEEP_ROOT}
  split_job: ${split_jid}
  offline_job: ${offline_jid}
  hlt_teacher_jobs: ${hlt_teacher_jids[*]}
  report_job: ${report_jid}
  report_dir: ${HLT_V2_BASELINE_SWEEP_ROOT}/baseline_sweep_report
EOF
