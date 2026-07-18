#!/usr/bin/env bash
# Submit one constrained coarse-to-fine pseudo-offline campaign graph.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_ENV:=atlas_kd}"
export CONDA_ENV
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
# shellcheck source=constrained_coarse_to_fine_claim_contract.sh
source "${SCRIPT_DIR}/constrained_coarse_to_fine_claim_contract.sh"

fresh_prepare_submitter
fresh_activate_env
echo "submitter_python=$(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"

: "${CONSTRAINED_C2F_CAMPAIGN_MODE:=pilot}"
case "${CONSTRAINED_C2F_CAMPAIGN_MODE}" in
  pilot)
    : "${CONSTRAINED_C2F_MODEL_TRAIN_SIZE:=500000}"
    : "${CONSTRAINED_C2F_MODEL_VAL_SIZE:=150000}"
    : "${CONSTRAINED_C2F_STACK_TRAIN_SIZE:=300000}"
    : "${CONSTRAINED_C2F_STACK_VAL_SIZE:=150000}"
    : "${CONSTRAINED_C2F_FINAL_TEST_SIZE:=150000}"
    ;;
  highdata)
    : "${CONSTRAINED_C2F_MODEL_TRAIN_SIZE:=5000000}"
    : "${CONSTRAINED_C2F_MODEL_VAL_SIZE:=1000000}"
    : "${CONSTRAINED_C2F_STACK_TRAIN_SIZE:=2000000}"
    : "${CONSTRAINED_C2F_STACK_VAL_SIZE:=1000000}"
    : "${CONSTRAINED_C2F_FINAL_TEST_SIZE:=1000000}"
    ;;
  *) echo "CONSTRAINED_C2F_CAMPAIGN_MODE must be pilot or highdata" >&2; exit 2 ;;
esac

if [[ "${CONSTRAINED_C2F_CAMPAIGN_MODE}" == "highdata" ]]; then
  : "${CONSTRAINED_C2F_RECO_NUM_WORKERS:=0}"
  : "${CONSTRAINED_C2F_TAGGER_NUM_WORKERS:=0}"
else
  : "${CONSTRAINED_C2F_RECO_NUM_WORKERS:=4}"
  : "${CONSTRAINED_C2F_TAGGER_NUM_WORKERS:=4}"
fi

: "${CONSTRAINED_C2F_STAGE_MODE:=full}"
: "${CONSTRAINED_C2F_RUNTIME_PROFILE:=fp32_reference}"
: "${CONSTRAINED_C2F_RUNTIME_PROFILE_ARTIFACT:=}"
case "${CONSTRAINED_C2F_RUNTIME_PROFILE}" in
  fp32_reference|fp16_diagnostic|bf16_calibration|bf16_exploratory_pilot_v1|accelerated_candidate_v1|accelerated_approved_v1) ;;
  *) echo "Unsupported CONSTRAINED_C2F_RUNTIME_PROFILE: ${CONSTRAINED_C2F_RUNTIME_PROFILE}" >&2; exit 2 ;;
esac
: "${CONSTRAINED_C2F_DATA_DIR:=${PD10_DATA_DIR:-${DATA_DIR}}}"
if [[ -z "${CONSTRAINED_C2F_ROOT:-}" ]]; then
  CONSTRAINED_C2F_ROOT="${OUTPUT_ROOT}/constrained_coarse_to_fine_pseudooffline_hltv2_s2p5_${CONSTRAINED_C2F_CAMPAIGN_MODE}_${CONSTRAINED_C2F_RUNTIME_PROFILE}_$(date +%Y%m%d_%H%M%S)"
fi
: "${CONSTRAINED_C2F_INPUTS_DIR:=${CONSTRAINED_C2F_ROOT}/inputs}"
: "${CONSTRAINED_C2F_MANIFEST_PATH:=${CONSTRAINED_C2F_INPUTS_DIR}/split_manifest/split_manifest.json.gz}"
: "${CONSTRAINED_C2F_HLT_CACHE_DIR:=${CONSTRAINED_C2F_INPUTS_DIR}/hlt_cache}"
: "${CONSTRAINED_C2F_OFFLINE_CACHE_DIR:=${CONSTRAINED_C2F_INPUTS_DIR}/offline_cache}"
: "${CONSTRAINED_C2F_TARGET_CACHE_DIR:=${CONSTRAINED_C2F_ROOT}/targets}"
: "${CONSTRAINED_C2F_RECON_ROOT:=${CONSTRAINED_C2F_ROOT}/reconstructors}"
: "${CONSTRAINED_C2F_TAGGER_ROOT:=${CONSTRAINED_C2F_ROOT}/taggers}"
: "${CONSTRAINED_C2F_PREDICTION_DIR:=${CONSTRAINED_C2F_ROOT}/predictions}"
: "${CONSTRAINED_C2F_FUSION_DIR:=${CONSTRAINED_C2F_ROOT}/fusion}"
: "${CONSTRAINED_C2F_REPORT_DIR:=${CONSTRAINED_C2F_ROOT}/final_report}"
: "${CONSTRAINED_C2F_APPROVE_HIGHDATA:=0}"
: "${CONSTRAINED_C2F_PILOT_REPORT_PATH:=}"
: "${CONSTRAINED_C2F_HLT_PROFILE:=fixed_hlt_v2_realistic}"
: "${CONSTRAINED_C2F_HLT_DEGRADATION_STRENGTH:=2.5}"
: "${CONSTRAINED_C2F_CACHE_SPLITS:=model_train model_val stack_train stack_val final_test}"
: "${CONSTRAINED_C2F_OFFLINE_SPLITS:=model_train model_val stack_train stack_val}"
: "${CONSTRAINED_C2F_TARGET_SPLITS:=model_train model_val stack_val}"
: "${CONSTRAINED_C2F_PREDICT_SPLITS:=model_val stack_train stack_val}"
: "${CONSTRAINED_C2F_RECON_RUN_IDS:=${C2F_FROZEN_RECON_RUN_IDS}}"
: "${CONSTRAINED_C2F_TAGGER_RUN_IDS:=${C2F_FROZEN_TAGGER_RUN_IDS}}"
: "${CONSTRAINED_C2F_PREDICT_RUN_IDS:=${CONSTRAINED_C2F_TAGGER_RUN_IDS}}"
: "${CONSTRAINED_C2F_FUSION_GROUPS:=${C2F_FROZEN_FUSION_GROUPS}}"
: "${CONSTRAINED_C2F_REQUIRED_FUSION_GROUPS:=${C2F_FROZEN_REQUIRED_FUSION_GROUPS}}"
: "${CONSTRAINED_C2F_REPORT_RECON_RUN_IDS:=${CONSTRAINED_C2F_RECON_RUN_IDS}}"
: "${CONSTRAINED_C2F_REPORT_TAGGER_RUN_IDS:=${CONSTRAINED_C2F_TAGGER_RUN_IDS}}"
: "${CONSTRAINED_C2F_HLT_WARM_START_CHECKPOINT:=${CONSTRAINED_C2F_TAGGER_ROOT}/A0/best_model_val.pt}"
: "${CONSTRAINED_C2F_APPROVE_FINAL_TEST:=0}"
: "${CONSTRAINED_C2F_SELECTION_REPORT_PATH:=${CONSTRAINED_C2F_ROOT}/final_report/final_report.json}"
: "${CONSTRAINED_C2F_RECO_PRECISION_MODE:=}"
: "${CONSTRAINED_C2F_RECO_PREFETCH_FACTOR:=}"
: "${CONSTRAINED_C2F_RECO_LR_SCHEDULE:=constant}"
: "${CONSTRAINED_C2F_RECO_WARMUP_FRACTION:=0.10}"
: "${CONSTRAINED_C2F_RECO_MIN_LR_RATIO:=0.05}"
: "${CONSTRAINED_C2F_RECO_MIN_EPOCHS:=0}"
: "${CONSTRAINED_C2F_RECO_FIXED_HORIZON:=0}"
: "${CONSTRAINED_C2F_HUNGARIAN_WORKERS:=1}"
: "${CONSTRAINED_C2F_HUNGARIAN_EXECUTOR:=serial}"

CONSTRAINED_C2F_RUNTIME_PROFILE_EXPECTED_STATUS=""
if [[ "${CONSTRAINED_C2F_CAMPAIGN_MODE}" == "highdata" && "${CONSTRAINED_C2F_STAGE_MODE}" != "final_claims" ]]; then
  [[ "${CONSTRAINED_C2F_RUNTIME_PROFILE}" == "accelerated_approved_v1" ]] || {
    echo "High-data and final-claim submission require CONSTRAINED_C2F_RUNTIME_PROFILE=accelerated_approved_v1." >&2
    exit 2
  }
  CONSTRAINED_C2F_RUNTIME_PROFILE_EXPECTED_STATUS="accelerated_approved_v1"
elif [[ "${CONSTRAINED_C2F_STAGE_MODE}" != "final_claims" && "${CONSTRAINED_C2F_RUNTIME_PROFILE}" == "accelerated_candidate_v1" ]]; then
  CONSTRAINED_C2F_RUNTIME_PROFILE_EXPECTED_STATUS="accelerated_candidate_v1"
elif [[ "${CONSTRAINED_C2F_STAGE_MODE}" != "final_claims" && "${CONSTRAINED_C2F_RUNTIME_PROFILE}" == "accelerated_approved_v1" ]]; then
  CONSTRAINED_C2F_RUNTIME_PROFILE_EXPECTED_STATUS="accelerated_approved_v1"
fi
if [[ -n "${CONSTRAINED_C2F_RUNTIME_PROFILE_EXPECTED_STATUS}" ]]; then
  [[ -n "${CONSTRAINED_C2F_RUNTIME_PROFILE_ARTIFACT}" ]] || {
    echo "${CONSTRAINED_C2F_RUNTIME_PROFILE_EXPECTED_STATUS} requires CONSTRAINED_C2F_RUNTIME_PROFILE_ARTIFACT." >&2
    exit 2
  }
  "${PYTHON_BIN}" scripts/validate_constrained_coarse_to_fine_runtime_profile.py \
    --profile "${CONSTRAINED_C2F_RUNTIME_PROFILE_ARTIFACT}" \
    --expected-status "${CONSTRAINED_C2F_RUNTIME_PROFILE_EXPECTED_STATUS}" >/dev/null
fi

if [[ "${CONSTRAINED_C2F_CAMPAIGN_MODE}" == "highdata" && "${CONSTRAINED_C2F_STAGE_MODE}" != "final_claims" ]]; then
  fresh_bool_enabled "${CONSTRAINED_C2F_APPROVE_HIGHDATA}" || {
    echo "High-data submission requires CONSTRAINED_C2F_APPROVE_HIGHDATA=1 after pilot review." >&2
    exit 2
  }
  [[ -n "${CONSTRAINED_C2F_PILOT_REPORT_PATH}" ]] || {
    echo "High-data submission requires CONSTRAINED_C2F_PILOT_REPORT_PATH." >&2
    exit 2
  }
  fresh_require_file "${CONSTRAINED_C2F_PILOT_REPORT_PATH}"
  "${PYTHON_BIN}" - "${CONSTRAINED_C2F_PILOT_REPORT_PATH}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    report = json.load(handle)
if not bool(report.get("ok")):
    raise SystemExit("pilot final report is not ok; refusing high-data submission")
PY
fi

if [[ "${CONSTRAINED_C2F_CAMPAIGN_MODE}" == "highdata" ]]; then
  c2f_require_frozen_value CONSTRAINED_C2F_RECON_RUN_IDS "${CONSTRAINED_C2F_RECON_RUN_IDS}" "${C2F_FROZEN_RECON_RUN_IDS}"
  c2f_require_frozen_value CONSTRAINED_C2F_TAGGER_RUN_IDS "${CONSTRAINED_C2F_TAGGER_RUN_IDS}" "${C2F_FROZEN_TAGGER_RUN_IDS}"
  c2f_require_frozen_value CONSTRAINED_C2F_PREDICT_RUN_IDS "${CONSTRAINED_C2F_PREDICT_RUN_IDS}" "${C2F_FROZEN_TAGGER_RUN_IDS}"
  c2f_require_frozen_value CONSTRAINED_C2F_FUSION_GROUPS "${CONSTRAINED_C2F_FUSION_GROUPS}" "${C2F_FROZEN_FUSION_GROUPS}"
  c2f_require_frozen_value CONSTRAINED_C2F_REQUIRED_FUSION_GROUPS "${CONSTRAINED_C2F_REQUIRED_FUSION_GROUPS}" "${C2F_FROZEN_REQUIRED_FUSION_GROUPS}"
fi

: "${CONSTRAINED_C2F_SUBMIT_SPLITS:=1}"
: "${CONSTRAINED_C2F_SUBMIT_HLT_CACHE:=1}"
: "${CONSTRAINED_C2F_SUBMIT_OFFLINE_CACHE:=1}"
: "${CONSTRAINED_C2F_SUBMIT_TARGETS:=1}"
: "${CONSTRAINED_C2F_SUBMIT_RECONSTRUCTORS:=1}"
: "${CONSTRAINED_C2F_SUBMIT_TAGGERS:=1}"
: "${CONSTRAINED_C2F_SUBMIT_PREDICTIONS:=1}"
: "${CONSTRAINED_C2F_SUBMIT_FUSION:=1}"
: "${CONSTRAINED_C2F_SUBMIT_REPORT:=1}"

: "${CONSTRAINED_C2F_SBATCH_ACCOUNT:=}"
: "${CONSTRAINED_C2F_SBATCH_PARTITION:=}"
: "${CONSTRAINED_C2F_GPU_GRES:=}"
: "${CONSTRAINED_C2F_GPU_CPUS_PER_TASK:=}"
: "${CONSTRAINED_C2F_GPU_MEM:=}"
: "${CONSTRAINED_C2F_CPU_CPUS_PER_TASK:=}"
: "${CONSTRAINED_C2F_CPU_MEM:=}"

case "${CONSTRAINED_C2F_STAGE_MODE}" in
  full) ;;
  targets_only)
    CONSTRAINED_C2F_SUBMIT_RECONSTRUCTORS=0; CONSTRAINED_C2F_SUBMIT_TAGGERS=0
    CONSTRAINED_C2F_SUBMIT_PREDICTIONS=0; CONSTRAINED_C2F_SUBMIT_FUSION=0; CONSTRAINED_C2F_SUBMIT_REPORT=0
    ;;
  reconstructors_only)
    CONSTRAINED_C2F_SUBMIT_SPLITS=0; CONSTRAINED_C2F_SUBMIT_HLT_CACHE=0; CONSTRAINED_C2F_SUBMIT_OFFLINE_CACHE=0
    CONSTRAINED_C2F_SUBMIT_TARGETS=0; CONSTRAINED_C2F_SUBMIT_RECONSTRUCTORS=1; CONSTRAINED_C2F_SUBMIT_TAGGERS=0
    CONSTRAINED_C2F_SUBMIT_PREDICTIONS=0; CONSTRAINED_C2F_SUBMIT_FUSION=0; CONSTRAINED_C2F_SUBMIT_REPORT=0
    ;;
  taggers_only)
    CONSTRAINED_C2F_SUBMIT_SPLITS=0; CONSTRAINED_C2F_SUBMIT_HLT_CACHE=0; CONSTRAINED_C2F_SUBMIT_OFFLINE_CACHE=0
    CONSTRAINED_C2F_SUBMIT_TARGETS=0; CONSTRAINED_C2F_SUBMIT_RECONSTRUCTORS=0; CONSTRAINED_C2F_SUBMIT_TAGGERS=1
    CONSTRAINED_C2F_SUBMIT_FUSION=0; CONSTRAINED_C2F_SUBMIT_REPORT=0
    ;;
  final_claims)
    CONSTRAINED_C2F_FUSION_DIR="${CONSTRAINED_C2F_ROOT}/fusion_final_claim"
    CONSTRAINED_C2F_REPORT_DIR="${CONSTRAINED_C2F_ROOT}/final_claim_report"
    CONSTRAINED_C2F_SELECTION_REPORT_SHA256="$(c2f_validate_final_claim_contract \
      "${CONSTRAINED_C2F_SELECTION_REPORT_PATH}" \
      "${CONSTRAINED_C2F_PREDICTION_DIR}" \
      "${CONSTRAINED_C2F_FUSION_DIR}" \
      "${CONSTRAINED_C2F_REPORT_DIR}")"
    CONSTRAINED_C2F_PREDICT_SPLITS="final_test"
    CONSTRAINED_C2F_OFFLINE_SPLITS="final_test"
    CONSTRAINED_C2F_SUBMIT_SPLITS=0; CONSTRAINED_C2F_SUBMIT_HLT_CACHE=0; CONSTRAINED_C2F_SUBMIT_OFFLINE_CACHE=1
    CONSTRAINED_C2F_SUBMIT_TARGETS=0; CONSTRAINED_C2F_SUBMIT_RECONSTRUCTORS=0; CONSTRAINED_C2F_SUBMIT_TAGGERS=0
    CONSTRAINED_C2F_SUBMIT_PREDICTIONS=1; CONSTRAINED_C2F_SUBMIT_FUSION=1; CONSTRAINED_C2F_SUBMIT_REPORT=1
    ;;
  depth_d5)
    CONSTRAINED_C2F_RECON_RUN_IDS="C5-B1 C5-B2 C5-B3"
    CONSTRAINED_C2F_TAGGER_RUN_IDS="D5 D5-B1 D5-B2 D5-B3"
    CONSTRAINED_C2F_PREDICT_RUN_IDS="${CONSTRAINED_C2F_TAGGER_RUN_IDS}"
    CONSTRAINED_C2F_SUBMIT_SPLITS=0; CONSTRAINED_C2F_SUBMIT_HLT_CACHE=0; CONSTRAINED_C2F_SUBMIT_OFFLINE_CACHE=0
    CONSTRAINED_C2F_SUBMIT_TARGETS=0; CONSTRAINED_C2F_SUBMIT_RECONSTRUCTORS=0; CONSTRAINED_C2F_SUBMIT_TAGGERS=1
    CONSTRAINED_C2F_SUBMIT_FUSION=0; CONSTRAINED_C2F_SUBMIT_REPORT=0
    ;;
  d8_only)
    CONSTRAINED_C2F_RECON_RUN_IDS="C5-B1 C5-B2 C5-B3"
    CONSTRAINED_C2F_TAGGER_RUN_IDS="D8"
    CONSTRAINED_C2F_PREDICT_RUN_IDS="D8"
    CONSTRAINED_C2F_SUBMIT_SPLITS=0; CONSTRAINED_C2F_SUBMIT_HLT_CACHE=0; CONSTRAINED_C2F_SUBMIT_OFFLINE_CACHE=0
    CONSTRAINED_C2F_SUBMIT_TARGETS=0; CONSTRAINED_C2F_SUBMIT_RECONSTRUCTORS=0; CONSTRAINED_C2F_SUBMIT_TAGGERS=1
    CONSTRAINED_C2F_SUBMIT_FUSION=0; CONSTRAINED_C2F_SUBMIT_REPORT=0
    ;;
  fusion_only)
    CONSTRAINED_C2F_SUBMIT_SPLITS=0; CONSTRAINED_C2F_SUBMIT_HLT_CACHE=0; CONSTRAINED_C2F_SUBMIT_OFFLINE_CACHE=0
    CONSTRAINED_C2F_SUBMIT_TARGETS=0; CONSTRAINED_C2F_SUBMIT_RECONSTRUCTORS=0; CONSTRAINED_C2F_SUBMIT_TAGGERS=0
    CONSTRAINED_C2F_SUBMIT_PREDICTIONS=0; CONSTRAINED_C2F_SUBMIT_FUSION=1; CONSTRAINED_C2F_SUBMIT_REPORT=0
    ;;
  report_only)
    CONSTRAINED_C2F_SUBMIT_SPLITS=0; CONSTRAINED_C2F_SUBMIT_HLT_CACHE=0; CONSTRAINED_C2F_SUBMIT_OFFLINE_CACHE=0
    CONSTRAINED_C2F_SUBMIT_TARGETS=0; CONSTRAINED_C2F_SUBMIT_RECONSTRUCTORS=0; CONSTRAINED_C2F_SUBMIT_TAGGERS=0
    CONSTRAINED_C2F_SUBMIT_PREDICTIONS=0; CONSTRAINED_C2F_SUBMIT_FUSION=0; CONSTRAINED_C2F_SUBMIT_REPORT=1
    ;;
  *) echo "Unknown CONSTRAINED_C2F_STAGE_MODE: ${CONSTRAINED_C2F_STAGE_MODE}" >&2; exit 2 ;;
esac

if [[ "${CONSTRAINED_C2F_STAGE_MODE}" == "final_claims" ]]; then
  [[ "${CONSTRAINED_C2F_RUNTIME_PROFILE}" == "accelerated_approved_v1" ]] || {
    echo "Final-claim submission requires CONSTRAINED_C2F_RUNTIME_PROFILE=accelerated_approved_v1." >&2
    exit 2
  }
  [[ -n "${CONSTRAINED_C2F_RUNTIME_PROFILE_ARTIFACT}" ]] || {
    echo "Final-claim submission requires CONSTRAINED_C2F_RUNTIME_PROFILE_ARTIFACT." >&2
    exit 2
  }
  CONSTRAINED_C2F_RUNTIME_PROFILE_EXPECTED_STATUS="accelerated_approved_v1"
  "${PYTHON_BIN}" scripts/validate_constrained_coarse_to_fine_runtime_profile.py \
    --profile "${CONSTRAINED_C2F_RUNTIME_PROFILE_ARTIFACT}" \
    --expected-status "${CONSTRAINED_C2F_RUNTIME_PROFILE_EXPECTED_STATUS}" >/dev/null
fi

if [[ "${CONSTRAINED_C2F_STAGE_MODE}" != "final_claims" ]] \
  && fresh_bool_enabled "${CONSTRAINED_C2F_SUBMIT_PREDICTIONS}" \
  && ! fresh_bool_enabled "${CONSTRAINED_C2F_SUBMIT_TAGGERS}" \
  && { fresh_bool_enabled "${CONSTRAINED_C2F_SUBMIT_HLT_CACHE}" \
    || fresh_bool_enabled "${CONSTRAINED_C2F_SUBMIT_OFFLINE_CACHE}" \
    || fresh_bool_enabled "${CONSTRAINED_C2F_SUBMIT_TARGETS}" \
    || fresh_bool_enabled "${CONSTRAINED_C2F_SUBMIT_RECONSTRUCTORS}"; }; then
  echo "Cannot rebuild caches/targets/reconstructors and reuse stale taggers for new predictions." >&2
  exit 2
fi

export CONSTRAINED_C2F_ROOT CONSTRAINED_C2F_MANIFEST_PATH CONSTRAINED_C2F_HLT_CACHE_DIR
export CONSTRAINED_C2F_OFFLINE_CACHE_DIR CONSTRAINED_C2F_TARGET_CACHE_DIR CONSTRAINED_C2F_RECON_ROOT
export CONSTRAINED_C2F_TAGGER_ROOT CONSTRAINED_C2F_PREDICTION_DIR CONSTRAINED_C2F_FUSION_DIR
export CONSTRAINED_C2F_REPORT_DIR CONSTRAINED_C2F_TARGET_SPLITS CONSTRAINED_C2F_PREDICT_SPLITS
export CONSTRAINED_C2F_HLT_WARM_START_CHECKPOINT CONSTRAINED_C2F_FUSION_GROUPS
export CONSTRAINED_C2F_REQUIRED_FUSION_GROUPS CONSTRAINED_C2F_REPORT_RECON_RUN_IDS
export CONSTRAINED_C2F_REPORT_TAGGER_RUN_IDS CONFIRM_FINAL_TEST
export CONSTRAINED_C2F_RECO_NUM_WORKERS CONSTRAINED_C2F_TAGGER_NUM_WORKERS
export CONSTRAINED_C2F_SELECTION_REPORT_PATH CONSTRAINED_C2F_SELECTION_REPORT_SHA256
export CONSTRAINED_C2F_RUNTIME_PROFILE CONSTRAINED_C2F_RECO_PRECISION_MODE
export CONSTRAINED_C2F_RECO_PREFETCH_FACTOR CONSTRAINED_C2F_RECO_LR_SCHEDULE
export CONSTRAINED_C2F_RECO_WARMUP_FRACTION CONSTRAINED_C2F_RECO_MIN_LR_RATIO
export CONSTRAINED_C2F_RECO_MIN_EPOCHS CONSTRAINED_C2F_RECO_FIXED_HORIZON
export CONSTRAINED_C2F_HUNGARIAN_WORKERS CONSTRAINED_C2F_HUNGARIAN_EXECUTOR
export CONSTRAINED_C2F_RUNTIME_PROFILE_ARTIFACT CONSTRAINED_C2F_RUNTIME_PROFILE_EXPECTED_STATUS

dependency_token_is_valid() {
  [[ "$1" =~ ^[0-9]+$ ]] || { fresh_is_dry_run && [[ "$1" =~ ^DRYRUN_[A-Za-z0-9_]+$ ]]; }
}

submit_job() {
  local label="$1"; shift
  local args=()
  [[ -n "${CONSTRAINED_C2F_SBATCH_ACCOUNT}" ]] && args+=(--account="${CONSTRAINED_C2F_SBATCH_ACCOUNT}")
  [[ -n "${CONSTRAINED_C2F_SBATCH_PARTITION}" ]] && args+=(--partition="${CONSTRAINED_C2F_SBATCH_PARTITION}")
  local gpu=0 arg
  for arg in "$@"; do
    case "${arg}" in
      */run_train_constrained_coarse_to_fine_reconstructor.sh|*/run_train_constrained_coarse_to_fine_tagger.sh|*/run_cache_constrained_coarse_to_fine_predictions.sh) gpu=1 ;;
    esac
  done
  if [[ "${gpu}" -eq 1 ]]; then
    [[ -n "${CONSTRAINED_C2F_GPU_GRES}" ]] && args+=(--gres="${CONSTRAINED_C2F_GPU_GRES}")
    [[ -n "${CONSTRAINED_C2F_GPU_CPUS_PER_TASK}" ]] && args+=(--cpus-per-task="${CONSTRAINED_C2F_GPU_CPUS_PER_TASK}")
    [[ -n "${CONSTRAINED_C2F_GPU_MEM}" ]] && args+=(--mem="${CONSTRAINED_C2F_GPU_MEM}")
  else
    [[ -n "${CONSTRAINED_C2F_CPU_CPUS_PER_TASK}" ]] && args+=(--cpus-per-task="${CONSTRAINED_C2F_CPU_CPUS_PER_TASK}")
    [[ -n "${CONSTRAINED_C2F_CPU_MEM}" ]] && args+=(--mem="${CONSTRAINED_C2F_CPU_MEM}")
  fi
  if fresh_is_dry_run; then
    fresh_print_shell_command sbatch "${args[@]}" "$@" >&2; printf '\n' >&2
    printf 'DRYRUN_%s\n' "${label//[^A-Za-z0-9_]/_}"; return 0
  fi
  local output job_id
  output="$(sbatch "${args[@]}" "$@")" || { echo "Failed to submit ${label}" >&2; return 2; }
  echo "${output}" >&2
  job_id="$(awk '{print $NF}' <<<"${output}")"
  dependency_token_is_valid "${job_id}" || { echo "Invalid Slurm job id for ${label}: ${job_id}" >&2; return 2; }
  printf '%s\n' "${job_id}"
}

join_dependencies() {
  local rows=() row
  for row in "$@"; do [[ -n "${row}" ]] && rows+=("${row}"); done
  [[ "${#rows[@]}" -gt 0 ]] && fresh_join_by_colon "${rows[@]}"
  return 0
}

submit_afterok() {
  local label="$1" dependency="$2"; shift 2
  if [[ -n "${dependency}" ]]; then submit_job "${label}" --dependency="afterok:${dependency}" "$@"; else submit_job "${label}" "$@"; fi
}

hlt_cache_complete() {
  [[ -f "${CONSTRAINED_C2F_MANIFEST_PATH}" ]] || return 1
  local rows=(); fresh_split_words rows "${CONSTRAINED_C2F_CACHE_SPLITS}"
  "${PYTHON_BIN}" scripts/validate_constrained_coarse_to_fine_artifact.py \
    --kind hlt-cache --path "${CONSTRAINED_C2F_HLT_CACHE_DIR}" \
    --manifest "${CONSTRAINED_C2F_MANIFEST_PATH}" --splits "${rows[@]}" >/dev/null 2>&1
}
offline_cache_complete() {
  [[ -f "${CONSTRAINED_C2F_MANIFEST_PATH}" ]] || return 1
  local rows=(); fresh_split_words rows "${CONSTRAINED_C2F_OFFLINE_SPLITS}"
  "${PYTHON_BIN}" scripts/validate_constrained_coarse_to_fine_artifact.py \
    --kind offline-cache --path "${CONSTRAINED_C2F_OFFLINE_CACHE_DIR}" \
    --manifest "${CONSTRAINED_C2F_MANIFEST_PATH}" --splits "${rows[@]}" >/dev/null 2>&1
}
target_complete() {
  [[ -f "${CONSTRAINED_C2F_MANIFEST_PATH}" ]] || return 1
  local rows=(); fresh_split_words rows "${CONSTRAINED_C2F_TARGET_SPLITS}"
  "${PYTHON_BIN}" scripts/validate_constrained_coarse_to_fine_artifact.py \
    --kind target-cache --path "${CONSTRAINED_C2F_TARGET_CACHE_DIR}" \
    --manifest "${CONSTRAINED_C2F_MANIFEST_PATH}" --splits "${rows[@]}" \
    --hlt-cache-dir "${CONSTRAINED_C2F_HLT_CACHE_DIR}" \
    --offline-cache-dir "${CONSTRAINED_C2F_OFFLINE_CACHE_DIR}" >/dev/null 2>&1
}
runtime_profile_hash_for_run() {
  local run_id="$1"
  [[ -n "${CONSTRAINED_C2F_RUNTIME_PROFILE_ARTIFACT}" ]] || return 0
  [[ -n "${CONSTRAINED_C2F_RUNTIME_PROFILE_EXPECTED_STATUS}" ]] || return 0
  "${PYTHON_BIN}" - "${CONSTRAINED_C2F_RUNTIME_PROFILE_ARTIFACT}" \
    "${CONSTRAINED_C2F_RUNTIME_PROFILE_EXPECTED_STATUS}" "${run_id}" <<'PY'
import sys
from teacher_logit_reco.constrained_coarse_to_fine.runtime_profiles import resolve_execution, validate_runtime_profile

validated = validate_runtime_profile(sys.argv[1], expected_status=sys.argv[2])
print(resolve_execution(validated["profile"], sys.argv[3])["runtime_profile_hash"])
PY
}
recon_complete() {
  [[ -f "${CONSTRAINED_C2F_MANIFEST_PATH}" ]] || return 1
  local expected_hash="" args=()
  expected_hash="$(runtime_profile_hash_for_run "$1")"
  [[ -n "${expected_hash}" ]] && args+=(--expected-runtime-profile-hash "${expected_hash}")
  "${PYTHON_BIN}" scripts/validate_constrained_coarse_to_fine_artifact.py \
    --kind reconstructor --path "${CONSTRAINED_C2F_RECON_ROOT}/$1" --run-id "$1" \
    --manifest "${CONSTRAINED_C2F_MANIFEST_PATH}" \
    --hlt-cache-dir "${CONSTRAINED_C2F_HLT_CACHE_DIR}" \
    --offline-cache-dir "${CONSTRAINED_C2F_OFFLINE_CACHE_DIR}" \
    --target-cache-dir "${CONSTRAINED_C2F_TARGET_CACHE_DIR}" "${args[@]}" >/dev/null 2>&1
}
tagger_complete() {
  [[ -f "${CONSTRAINED_C2F_MANIFEST_PATH}" ]] || return 1
  local hashes=() value args=() hash_output=""
  hash_output="$(reconstructor_hashes_for_tagger "$1")" || return 1
  while IFS= read -r value; do [[ -n "${value}" ]] && hashes+=("${value}"); done <<< "${hash_output}"
  for value in "${hashes[@]}"; do args+=(--expected-reconstructor-checkpoint-sha256 "${value}"); done
  "${PYTHON_BIN}" scripts/validate_constrained_coarse_to_fine_artifact.py \
    --kind tagger --path "${CONSTRAINED_C2F_TAGGER_ROOT}/$1" --run-id "$1" \
    --manifest "${CONSTRAINED_C2F_MANIFEST_PATH}" \
    --hlt-cache-dir "${CONSTRAINED_C2F_HLT_CACHE_DIR}" \
    --offline-cache-dir "${CONSTRAINED_C2F_OFFLINE_CACHE_DIR}" \
    --target-cache-dir "${CONSTRAINED_C2F_TARGET_CACHE_DIR}" "${args[@]}" >/dev/null 2>&1
}
prediction_complete() {
  [[ -f "${CONSTRAINED_C2F_MANIFEST_PATH}" ]] || return 1
  local rows=() hashes=() value args=() hash_output=""; fresh_split_words rows "${CONSTRAINED_C2F_PREDICT_SPLITS}"
  hash_output="$(reconstructor_hashes_for_tagger "$1")" || return 1
  while IFS= read -r value; do [[ -n "${value}" ]] && hashes+=("${value}"); done <<< "${hash_output}"
  for value in "${hashes[@]}"; do args+=(--expected-reconstructor-checkpoint-sha256 "${value}"); done
  "${PYTHON_BIN}" scripts/validate_constrained_coarse_to_fine_artifact.py \
    --kind prediction --path "${CONSTRAINED_C2F_PREDICTION_DIR}/$1" --run-id "$1" \
    --manifest "${CONSTRAINED_C2F_MANIFEST_PATH}" --splits "${rows[@]}" \
    --hlt-cache-dir "${CONSTRAINED_C2F_HLT_CACHE_DIR}" \
    --offline-cache-dir "${CONSTRAINED_C2F_OFFLINE_CACHE_DIR}" \
    --target-cache-dir "${CONSTRAINED_C2F_TARGET_CACHE_DIR}" \
    --tagger-root "${CONSTRAINED_C2F_TAGGER_ROOT}" "${args[@]}" >/dev/null 2>&1
}

tagger_reconstructors() {
  local requested="$1"
  if [[ "${requested}" =~ ^(D[0-8]|D5-B[12])-seed[12]$ ]]; then requested="${BASH_REMATCH[1]}"; fi
  case "${requested}" in
    A0|A1|A2|A4|E6) return 0 ;;
    D0|D1|D2|D3|D4|D7|E0|E1|E2|E3) printf '%s\n' C5-B3 ;;
    D5|D5-B3) printf '%s\n' C5-B3 ;;
    D5-B1) printf '%s\n' C5-B1 ;;
    D5-B2) printf '%s\n' C5-B2 ;;
    D6) printf '%s\n' C6 ;;
    D8) printf '%s\n' C5-B1 C5-B2 C5-B3 ;;
    E5) printf '%s\n' C5-no-slot ;;
    E4) printf '%s\n' Cdirect-unconstrained ;;
    *) echo "Unsupported tagger run ID in submitter: $1" >&2; return 2 ;;
  esac
}

reconstructor_hashes_for_tagger() {
  local tagger_id="$1" source_id checkpoint
  while IFS= read -r source_id; do
    [[ -z "${source_id}" ]] && continue
    recon_complete "${source_id}" || return 1
    checkpoint="${CONSTRAINED_C2F_RECON_ROOT}/${source_id}/best_model_val.pt"
    "${PYTHON_BIN}" - "${checkpoint}" <<'PY'
import hashlib
import sys

digest = hashlib.sha256()
with open(sys.argv[1], "rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
print(digest.hexdigest())
PY
  done < <(tagger_reconstructors "${tagger_id}")
}

preflight_reused_inputs() {
  if ! fresh_bool_enabled "${CONSTRAINED_C2F_SUBMIT_SPLITS}"; then fresh_require_file "${CONSTRAINED_C2F_MANIFEST_PATH}"; fi
  if ! fresh_bool_enabled "${CONSTRAINED_C2F_SUBMIT_HLT_CACHE}"; then hlt_cache_complete || { echo "Required HLT cache is incomplete: ${CONSTRAINED_C2F_HLT_CACHE_DIR}" >&2; exit 2; }; fi
  if ! fresh_bool_enabled "${CONSTRAINED_C2F_SUBMIT_OFFLINE_CACHE}"; then offline_cache_complete || { echo "Required offline cache is incomplete: ${CONSTRAINED_C2F_OFFLINE_CACHE_DIR}" >&2; exit 2; }; fi
  if ! fresh_bool_enabled "${CONSTRAINED_C2F_SUBMIT_TARGETS}" && {
    fresh_bool_enabled "${CONSTRAINED_C2F_SUBMIT_RECONSTRUCTORS}" \
      || fresh_bool_enabled "${CONSTRAINED_C2F_SUBMIT_TAGGERS}" \
      || fresh_bool_enabled "${CONSTRAINED_C2F_SUBMIT_PREDICTIONS}" \
      || fresh_bool_enabled "${CONSTRAINED_C2F_SUBMIT_REPORT}"
  }; then
    target_complete || { echo "Required hierarchy target cache is incomplete: ${CONSTRAINED_C2F_TARGET_CACHE_DIR}" >&2; exit 2; }
  fi
  if fresh_bool_enabled "${CONSTRAINED_C2F_SUBMIT_TAGGERS}"; then
    local rows=() run_id needs_hlt=0 has_a0=0
    fresh_split_words rows "${CONSTRAINED_C2F_TAGGER_RUN_IDS}"
    for run_id in "${rows[@]}"; do
      [[ "${run_id}" == "A0" ]] && has_a0=1
      [[ ! "${run_id}" =~ ^(A0|A1|A2|D0)$ ]] && needs_hlt=1
    done
    if [[ "${needs_hlt}" -eq 1 ]]; then
      if [[ "${CONSTRAINED_C2F_HLT_WARM_START_CHECKPOINT}" != "${CONSTRAINED_C2F_TAGGER_ROOT}/A0/best_model_val.pt" ]]; then
        fresh_require_file "${CONSTRAINED_C2F_HLT_WARM_START_CHECKPOINT}"
        "${PYTHON_BIN}" scripts/validate_constrained_coarse_to_fine_warm_start.py \
          --checkpoint "${CONSTRAINED_C2F_HLT_WARM_START_CHECKPOINT}" >/dev/null
      elif [[ "${has_a0}" -eq 0 ]]; then
        tagger_complete A0 || {
          echo "Schedule-matched taggers require a complete active-campaign A0 or A0 in CONSTRAINED_C2F_TAGGER_RUN_IDS." >&2
          exit 2
        }
      fi
    fi
  fi
}

preflight_posthoc() {
  local required_fusion_report="${CONSTRAINED_C2F_FUSION_DIR}/fusion_report.json"
  if fresh_bool_enabled "${CONFIRM_FINAL_TEST}"; then
    required_fusion_report="${CONSTRAINED_C2F_FUSION_DIR}/fusion_final_claim_report.json"
  fi
  if fresh_bool_enabled "${CONSTRAINED_C2F_SUBMIT_REPORT}" && ! fresh_bool_enabled "${CONSTRAINED_C2F_SUBMIT_FUSION}"; then
    fresh_require_file "${required_fusion_report}"
  fi
  if [[ "${CONSTRAINED_C2F_STAGE_MODE}" == "report_only" ]]; then
    local rows=() run_id
    fresh_split_words rows "${CONSTRAINED_C2F_REPORT_RECON_RUN_IDS}"
    for run_id in "${rows[@]}"; do
      fresh_require_file "${CONSTRAINED_C2F_RECON_ROOT}/${run_id}/run_report.json"
    done
    fresh_split_words rows "${CONSTRAINED_C2F_REPORT_TAGGER_RUN_IDS}"
    for run_id in "${rows[@]}"; do
      fresh_require_file "${CONSTRAINED_C2F_TAGGER_ROOT}/${run_id}/run_report.json"
      prediction_complete "${run_id}" || {
        echo "Report-only rerun requires complete predictions for ${run_id}." >&2
        exit 2
      }
    done
  fi
}

preflight_reused_inputs
preflight_posthoc

submitter_log_dir="${CONSTRAINED_C2F_ROOT}/submission_logs/c2f_${CONSTRAINED_C2F_STAGE_MODE}_$(date +%Y%m%d_%H%M%S)"
fresh_claim_new_dir "${submitter_log_dir}"
if ! fresh_is_dry_run; then
  {
    echo "created_at=$(date -Is)"
    echo "source_commit=$(fresh_source_commit)"
    echo "source_status_hash=$(fresh_source_status_hash)"
    echo "root=${CONSTRAINED_C2F_ROOT}"
    echo "campaign_mode=${CONSTRAINED_C2F_CAMPAIGN_MODE}"
    echo "stage_mode=${CONSTRAINED_C2F_STAGE_MODE}"
    echo "runtime_profile=${CONSTRAINED_C2F_RUNTIME_PROFILE}"
    echo "precision_mode=${CONSTRAINED_C2F_RECO_PRECISION_MODE:-profile_default}"
    echo "lr_schedule=${CONSTRAINED_C2F_RECO_LR_SCHEDULE}"
    echo "prefetch_factor=${CONSTRAINED_C2F_RECO_PREFETCH_FACTOR:-none}"
    echo "hungarian_executor=${CONSTRAINED_C2F_HUNGARIAN_EXECUTOR}"
    echo "hungarian_workers=${CONSTRAINED_C2F_HUNGARIAN_WORKERS}"
    echo "hlt_profile=${CONSTRAINED_C2F_HLT_PROFILE}"
    echo "hlt_strength=${CONSTRAINED_C2F_HLT_DEGRADATION_STRENGTH}"
    echo "sizes=${CONSTRAINED_C2F_MODEL_TRAIN_SIZE}/${CONSTRAINED_C2F_MODEL_VAL_SIZE}/${CONSTRAINED_C2F_STACK_TRAIN_SIZE}/${CONSTRAINED_C2F_STACK_VAL_SIZE}/${CONSTRAINED_C2F_FINAL_TEST_SIZE}"
    echo "reconstructors=${CONSTRAINED_C2F_RECON_RUN_IDS}"
    echo "taggers=${CONSTRAINED_C2F_TAGGER_RUN_IDS}"
  } > "${submitter_log_dir}/metadata.txt"
fi

echo "constrained_c2f_submission_start:"
echo "  root: ${CONSTRAINED_C2F_ROOT}"
echo "  campaign_mode: ${CONSTRAINED_C2F_CAMPAIGN_MODE}"
echo "  stage_mode: ${CONSTRAINED_C2F_STAGE_MODE}"
echo "  runtime_profile: ${CONSTRAINED_C2F_RUNTIME_PROFILE}"
echo "  hlt: ${CONSTRAINED_C2F_HLT_PROFILE} strength=${CONSTRAINED_C2F_HLT_DEGRADATION_STRENGTH}"

split_jid=""; hlt_jid=""; offline_jid=""; target_jid=""
export DATA_DIR="${CONSTRAINED_C2F_DATA_DIR}" MANIFEST_PATH="${CONSTRAINED_C2F_MANIFEST_PATH}"
export MODEL_TRAIN_SIZE="${CONSTRAINED_C2F_MODEL_TRAIN_SIZE}" MODEL_VAL_SIZE="${CONSTRAINED_C2F_MODEL_VAL_SIZE}"
export STACK_TRAIN_SIZE="${CONSTRAINED_C2F_STACK_TRAIN_SIZE}" STACK_VAL_SIZE="${CONSTRAINED_C2F_STACK_VAL_SIZE}"
export FINAL_TEST_SIZE="${CONSTRAINED_C2F_FINAL_TEST_SIZE}"
if fresh_bool_enabled "${CONSTRAINED_C2F_SUBMIT_SPLITS}"; then
  if fresh_bool_enabled "${SKIP_EXISTING}" && [[ -f "${CONSTRAINED_C2F_MANIFEST_PATH}" ]]; then echo "skip splits: complete"; else
    split_jid="$(submit_job c2f_splits "${SCRIPT_DIR}/run_build_fresh_splits.sh")"
  fi
fi

export HLT_CACHE_DIR="${CONSTRAINED_C2F_HLT_CACHE_DIR}" HLT_SPLITS="${CONSTRAINED_C2F_CACHE_SPLITS}"
export HLT_PROFILE="${CONSTRAINED_C2F_HLT_PROFILE}" HLT_DEGRADATION_STRENGTH="${CONSTRAINED_C2F_HLT_DEGRADATION_STRENGTH}"
if fresh_bool_enabled "${CONSTRAINED_C2F_SUBMIT_HLT_CACHE}"; then
  if fresh_bool_enabled "${SKIP_EXISTING}" && hlt_cache_complete; then echo "skip HLT cache: complete"; else
    hlt_jid="$(submit_afterok c2f_hlt_cache "${split_jid}" "${SCRIPT_DIR}/run_build_fresh_hlt_cache.sh")"
  fi
fi

export ARCHITECTURE_VIEW_10CLASS_OFFLINE_MANIFEST_PATH="${CONSTRAINED_C2F_MANIFEST_PATH}"
export ARCHITECTURE_VIEW_10CLASS_OFFLINE_CACHE_DIR="${CONSTRAINED_C2F_OFFLINE_CACHE_DIR}"
export ARCHITECTURE_VIEW_10CLASS_OFFLINE_SPLITS="${CONSTRAINED_C2F_OFFLINE_SPLITS}"
export ARCHITECTURE_VIEW_10CLASS_OFFLINE_DATA_DIRS="${CONSTRAINED_C2F_DATA_DIR}"
export ARCHITECTURE_VIEW_10CLASS_OFFLINE_OVERWRITE="${OVERWRITE}"
export ARCHITECTURE_VIEW_10CLASS_OFFLINE_APPEND=0
if [[ "${CONSTRAINED_C2F_STAGE_MODE}" == "final_claims" ]]; then
  export ARCHITECTURE_VIEW_10CLASS_OFFLINE_APPEND=1
fi
if fresh_bool_enabled "${CONSTRAINED_C2F_SUBMIT_OFFLINE_CACHE}"; then
  if fresh_bool_enabled "${SKIP_EXISTING}" && offline_cache_complete; then echo "skip offline cache: complete"; else
    offline_cache_dep="$(join_dependencies "${split_jid}" "${hlt_jid}")"
    offline_jid="$(submit_afterok c2f_offline_cache "${offline_cache_dep}" "${SCRIPT_DIR}/run_cache_architecture_view_offline_inputs.sh")"
  fi
fi

cache_dep="$(join_dependencies "${hlt_jid}" "${offline_jid}")"
if ! fresh_bool_enabled "${CONSTRAINED_C2F_SUBMIT_TARGETS}" \
  && [[ -n "${cache_dep}" ]] \
  && { fresh_bool_enabled "${CONSTRAINED_C2F_SUBMIT_RECONSTRUCTORS}" || fresh_bool_enabled "${CONSTRAINED_C2F_SUBMIT_TAGGERS}"; }; then
  echo "Cannot reuse hierarchy targets while active HLT/offline caches are being rebuilt." >&2
  exit 2
fi
if fresh_bool_enabled "${CONSTRAINED_C2F_SUBMIT_TARGETS}"; then
  if fresh_bool_enabled "${SKIP_EXISTING}" && [[ -z "${cache_dep}" ]] && target_complete; then echo "skip targets: complete"; else
    target_jid="$(submit_afterok c2f_targets "${cache_dep}" "${SCRIPT_DIR}/run_cache_constrained_coarse_to_fine_targets.sh")"
  fi
fi

declare -A recon_jids=()
declare -a recon_ids=()
fresh_split_words recon_ids "${CONSTRAINED_C2F_RECON_RUN_IDS}"
if fresh_bool_enabled "${CONSTRAINED_C2F_SUBMIT_RECONSTRUCTORS}"; then
  for run_id in "${recon_ids[@]}"; do
    if fresh_bool_enabled "${SKIP_EXISTING}" \
      && [[ -z "${hlt_jid}" && -z "${offline_jid}" && -z "${target_jid}" ]] \
      && recon_complete "${run_id}"; then echo "skip reconstructor ${run_id}: complete"; else
      recon_jids["${run_id}"]="$(submit_afterok "c2f_reco_${run_id}" "${target_jid}" "${SCRIPT_DIR}/run_train_constrained_coarse_to_fine_reconstructor.sh" "${run_id}")"
    fi
  done
fi

declare -A tagger_jids=()
declare -a tagger_ids=()
fresh_split_words tagger_ids "${CONSTRAINED_C2F_TAGGER_RUN_IDS}"
if fresh_bool_enabled "${CONSTRAINED_C2F_SUBMIT_TAGGERS}"; then
  for run_id in "${tagger_ids[@]}"; do
    if fresh_bool_enabled "${SKIP_EXISTING}" \
      && [[ -z "${hlt_jid}" && -z "${offline_jid}" && -z "${target_jid}" && "${#recon_jids[@]}" -eq 0 ]] \
      && tagger_complete "${run_id}"; then echo "skip tagger ${run_id}: complete"; continue; fi
    if [[ "${run_id}" == "D5-B3" ]]; then
      if [[ -z "${tagger_jids[D5]:-}" ]]; then tagger_complete D5 || { echo "D5-B3 alias requires complete or queued D5" >&2; exit 2; }; fi
      tagger_jids["${run_id}"]="$(submit_afterok c2f_tagger_D5-B3_alias "${tagger_jids[D5]:-}" "${SCRIPT_DIR}/run_alias_constrained_coarse_to_fine_tagger.sh" D5 D5-B3)"
      continue
    fi
    dep_rows=()
    [[ -n "${target_jid}" ]] && dep_rows+=("${target_jid}")
    while IFS= read -r source_id; do
      [[ -z "${source_id}" ]] && continue
      if [[ -n "${recon_jids[${source_id}]:-}" ]]; then dep_rows+=("${recon_jids[${source_id}]}"); else
        if [[ -n "${target_jid}" ]]; then
          echo "${run_id} requires ${source_id} to be retrained after the active target-cache rebuild." >&2
          exit 2
        fi
        recon_complete "${source_id}" || { echo "${run_id} requires incomplete reconstructor ${source_id}" >&2; exit 2; }
      fi
    done < <(tagger_reconstructors "${run_id}")
    if [[ ! "${run_id}" =~ ^(A0|A1|A2|D0)$ && -n "${tagger_jids[A0]:-}" ]]; then
      dep_rows+=("${tagger_jids[A0]}")
    elif [[ ! "${run_id}" =~ ^(A0|A1|A2|D0)$ && "${CONSTRAINED_C2F_HLT_WARM_START_CHECKPOINT}" == "${CONSTRAINED_C2F_TAGGER_ROOT}/A0/best_model_val.pt" ]]; then
      tagger_complete A0 || { echo "${run_id} requires an incomplete A0 warm start" >&2; exit 2; }
    fi
    tagger_dep="$(join_dependencies "${dep_rows[@]}")"
    tagger_jids["${run_id}"]="$(submit_afterok "c2f_tagger_${run_id}" "${tagger_dep}" "${SCRIPT_DIR}/run_train_constrained_coarse_to_fine_tagger.sh" "${run_id}")"
  done
fi

declare -A prediction_jids=()
declare -a predict_ids=()
fresh_split_words predict_ids "${CONSTRAINED_C2F_PREDICT_RUN_IDS}"
if fresh_bool_enabled "${CONSTRAINED_C2F_SUBMIT_PREDICTIONS}"; then
  for run_id in "${predict_ids[@]}"; do
    if fresh_bool_enabled "${SKIP_EXISTING}" \
      && [[ -z "${hlt_jid}" && -z "${offline_jid}" && -z "${target_jid}" \
        && "${#recon_jids[@]}" -eq 0 && "${#tagger_jids[@]}" -eq 0 ]] \
      && prediction_complete "${run_id}"; then echo "skip predictions ${run_id}: complete"; continue; fi
    if [[ "${run_id}" == "D5-B3" ]]; then
      if [[ -z "${prediction_jids[D5]:-}" ]]; then
        if [[ -n "${hlt_jid}" || -n "${target_jid}" || "${#recon_jids[@]}" -gt 0 || "${#tagger_jids[@]}" -gt 0 ]]; then
          echo "D5-B3 prediction alias cannot reuse stale D5 predictions during an active rebuild." >&2
          exit 2
        fi
        prediction_complete D5 || { echo "D5-B3 prediction alias requires complete or queued D5 predictions" >&2; exit 2; }
      fi
      prediction_jids["${run_id}"]="$(submit_afterok c2f_predict_D5-B3_alias "${prediction_jids[D5]:-}" "${SCRIPT_DIR}/run_alias_constrained_coarse_to_fine_predictions.sh" D5 D5-B3)"
      continue
    fi
    prediction_dep_rows=()
    [[ -n "${tagger_jids[${run_id}]:-}" ]] && prediction_dep_rows+=("${tagger_jids[${run_id}]}")
    [[ -n "${hlt_jid}" ]] && prediction_dep_rows+=("${hlt_jid}")
    [[ -n "${target_jid}" ]] && prediction_dep_rows+=("${target_jid}")
    if [[ "${run_id}" == "A2" && -n "${offline_jid}" ]]; then prediction_dep_rows+=("${offline_jid}"); fi
    while IFS= read -r source_id; do
      [[ -z "${source_id}" ]] && continue
      [[ -n "${recon_jids[${source_id}]:-}" ]] && prediction_dep_rows+=("${recon_jids[${source_id}]}")
    done < <(tagger_reconstructors "${run_id}")
    if [[ -z "${tagger_jids[${run_id}]:-}" ]]; then
      if [[ "${CONSTRAINED_C2F_STAGE_MODE}" != "final_claims" \
        && ( -n "${hlt_jid}" || -n "${target_jid}" || "${#prediction_dep_rows[@]}" -gt 0 ) ]]; then
        echo "Predictions for ${run_id} cannot reuse a stale tagger while its active inputs are rebuilding." >&2
        exit 2
      fi
      tagger_complete "${run_id}" || { echo "Predictions require incomplete tagger ${run_id}" >&2; exit 2; }
    fi
    prediction_dep="$(join_dependencies "${prediction_dep_rows[@]}")"
    prediction_jids["${run_id}"]="$(submit_afterok "c2f_predict_${run_id}" "${prediction_dep}" "${SCRIPT_DIR}/run_cache_constrained_coarse_to_fine_predictions.sh" "${run_id}")"
  done
fi

fusion_jid=""
if fresh_bool_enabled "${CONSTRAINED_C2F_SUBMIT_FUSION}"; then
  fusion_deps=(); for run_id in "${predict_ids[@]}"; do
    if [[ -n "${prediction_jids[${run_id}]:-}" ]]; then fusion_deps+=("${prediction_jids[${run_id}]}"); else prediction_complete "${run_id}" || { echo "Fusion requires incomplete predictions ${run_id}" >&2; exit 2; }; fi
  done
  fusion_jid="$(submit_afterok c2f_fusion "$(join_dependencies "${fusion_deps[@]}")" "${SCRIPT_DIR}/run_constrained_coarse_to_fine_fusion.sh")"
fi

report_jid=""
if fresh_bool_enabled "${CONSTRAINED_C2F_SUBMIT_REPORT}"; then
  report_dep="${fusion_jid}"
  if [[ -z "${report_dep}" ]]; then
    if fresh_bool_enabled "${CONFIRM_FINAL_TEST}"; then
      fresh_require_file "${CONSTRAINED_C2F_FUSION_DIR}/fusion_final_claim_report.json"
    else
      fresh_require_file "${CONSTRAINED_C2F_FUSION_DIR}/fusion_report.json"
    fi
  fi
  report_jid="$(submit_afterok c2f_report "${report_dep}" "${SCRIPT_DIR}/run_write_constrained_coarse_to_fine_report.sh")"
fi

if ! fresh_is_dry_run; then
  {
    echo -e "stage\trun_id\tjob_id"
    [[ -n "${split_jid}" ]] && echo -e "input\tsplits\t${split_jid}"
    [[ -n "${hlt_jid}" ]] && echo -e "input\thlt_cache\t${hlt_jid}"
    [[ -n "${offline_jid}" ]] && echo -e "input\toffline_cache\t${offline_jid}"
    [[ -n "${target_jid}" ]] && echo -e "target\thierarchy\t${target_jid}"
    for run_id in "${!recon_jids[@]}"; do echo -e "reconstructor\t${run_id}\t${recon_jids[${run_id}]}"; done
    for run_id in "${!tagger_jids[@]}"; do echo -e "tagger\t${run_id}\t${tagger_jids[${run_id}]}"; done
    for run_id in "${!prediction_jids[@]}"; do echo -e "prediction\t${run_id}\t${prediction_jids[${run_id}]}"; done
    [[ -n "${fusion_jid}" ]] && echo -e "posthoc\tfusion\t${fusion_jid}"
    [[ -n "${report_jid}" ]] && echo -e "posthoc\treport\t${report_jid}"
  } > "${submitter_log_dir}/job_ids.tsv"
fi

cat <<SUMMARY
constrained_c2f_submission_complete:
  root: ${CONSTRAINED_C2F_ROOT}
  stage_mode: ${CONSTRAINED_C2F_STAGE_MODE}
  submission_log: ${submitter_log_dir}
  fusion_queued: ${CONSTRAINED_C2F_SUBMIT_FUSION}
  report_queued: ${CONSTRAINED_C2F_SUBMIT_REPORT}
SUMMARY
