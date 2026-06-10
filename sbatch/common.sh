#!/usr/bin/env bash
# Common helpers for the fresh JetClass same-HLT research-compute runners.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${DATA_DIR:=/home/ryreu/atlas/PracticeTagging/data/jetclass_part0}"
: "${OUTPUT_ROOT:=${PROJECT_DIR}/checkpoints}"
: "${LOG_DIR:=${PROJECT_DIR}/fresh_check_logs}"
: "${CONDA_ENV:=weaver}"
: "${PYTHON_BIN:=python}"
: "${DRY_RUN:=0}"
: "${PRINT_ONLY:=0}"
: "${OVERWRITE:=0}"
: "${DEVICE:=auto}"
: "${NUM_WORKERS:=4}"
: "${BATCH_SIZE:=128}"
: "${EPOCHS:=20}"
: "${LR:=0.001}"
: "${WEIGHT_DECAY:=0.0001}"
: "${EARLY_STOP_PATIENCE:=5}"
: "${GRAD_CLIP_NORM:=1.0}"
: "${MANIFEST_PATH:=${OUTPUT_ROOT}/jetclass_fresh_splits/split_manifest.json.gz}"
: "${HLT_CACHE_DIR:=${OUTPUT_ROOT}/jetclass_fresh_hlt_cache}"
: "${HLT_BASELINE_SEED:=101}"
: "${HLT_BASELINE_DIR:=${OUTPUT_ROOT}/jetclass_fresh_hlt_baselines/single_hlt_seed${HLT_BASELINE_SEED}}"
: "${HLT_CHECKPOINT:=${HLT_BASELINE_DIR}/best_model_val.pt}"
: "${HLT5_ROOT:=${OUTPUT_ROOT}/jetclass_fresh_hlt_baselines/hlt5_seed_control}"
: "${OFFLINE_TEACHER_DIR:=${OUTPUT_ROOT}/jetclass_fresh_offline_teacher/offline_teacher_seed707}"
: "${RECO7_ROOT:=${OUTPUT_ROOT}/jetclass_fresh_reco7}"
: "${RECO7_FUSION_DIR:=${OUTPUT_ROOT}/jetclass_fresh_fusion/reco7_plus_hlt}"
: "${HLT5_FUSION_DIR:=${OUTPUT_ROOT}/jetclass_fresh_fusion/hlt5_seed_control}"
: "${RECO7_AUDIT_DIR:=${OUTPUT_ROOT}/jetclass_fresh_audits/reco7_plus_hlt}"
: "${HLT5_AUDIT_DIR:=${OUTPUT_ROOT}/jetclass_fresh_audits/hlt5_seed_control}"
: "${FINAL_REPORT_DIR:=${OUTPUT_ROOT}/jetclass_fresh_final_report}"
: "${V2_STEP6_ROOT:=${OUTPUT_ROOT}/jetclass_v2_original_mechanism_step6}"
: "${V2_STEP6_RECO_ROOT:=${V2_STEP6_ROOT}/reco}"
: "${V2_STEP6_FUSION_DIR:=${V2_STEP6_ROOT}/fusion/m2_base_plus_hlt}"
: "${V2_STEP6_AUDIT_DIR:=${V2_STEP6_ROOT}/audits/m2_base_plus_hlt}"
: "${V2_STEP6_VARIANT:=m2_base}"
: "${V2_STEP7_ROOT:=${OUTPUT_ROOT}/jetclass_v2_original_mechanism_step7}"
: "${V2_STEP7_RECO_ROOT:=${V2_STEP7_ROOT}/reco7}"
: "${V2_STEP7_FUSION_DIR:=${V2_STEP7_ROOT}/fusion/reco7_plus_hlt}"
: "${V2_STEP7_AUDIT_DIR:=${V2_STEP7_ROOT}/audits/reco7_plus_hlt}"
: "${RECO7_VARIANTS:=m2_base m2_consstrong m2_budgetlite m2_genlow m2_genhigh m2_topk60ish m2_antioverlap}"
: "${V2_STEP7_VARIANTS:=${RECO7_VARIANTS}}"
: "${FUSION_MODEL_LOADING_ROOT:=${OUTPUT_ROOT}/jetclass_fresh_independent_fusion_handoff}"
: "${FUSION_MODEL_LOADING_SMALL_DIR:=${FUSION_MODEL_LOADING_ROOT}/small_50k_20k_100k}"
: "${FUSION_MODEL_LOADING_LARGE_DIR:=${FUSION_MODEL_LOADING_ROOT}/large_250k_50k_500k}"
: "${FUSION_MODEL_LOADING_VARIANTS:=${RECO7_VARIANTS}}"
: "${FUSION_MODEL_LOADING_FEATURE_MODES:=logits probs logits_probs}"
: "${FUSION_MODEL_LOADING_C_GRID:=}"
: "${FUSION_MODEL_LOADING_MAX_ITER:=2000}"
: "${FUSION_MODEL_LOADING_SKIP_CONTROLS:=0}"
: "${FUSION_MODEL_LOADING_CONTROL_SEED:=12345}"
: "${HETERO_HLT4_ROOT:=${OUTPUT_ROOT}/jetclass_hetero_hlt4_150k_50k_300k}"
: "${HETERO_HLT4_MODEL_ROOT:=${HETERO_HLT4_ROOT}/models}"
: "${HETERO_HLT4_FUSION_DIR:=${HETERO_HLT4_ROOT}/fusion_run}"
: "${HETERO_HLT4_ARCHITECTURES:=part pn pfn pcnn}"
: "${HETERO_HLT4_TRAIN_SIZE:=150000}"
: "${HETERO_HLT4_VAL_SIZE:=50000}"
: "${HETERO_HLT4_STACK_TRAIN_SIZE:=150000}"
: "${HETERO_HLT4_STACK_VAL_SIZE:=50000}"
: "${HETERO_HLT4_FINAL_TEST_SIZE:=300000}"
: "${HETERO_HLT4_FEATURE_MODES:=logits probs logits_probs}"
: "${HETERO_HLT4_C_GRID:=}"
: "${HETERO_HLT4_MAX_ITER:=2000}"
: "${HETERO_HLT4_SKIP_CONTROLS:=0}"
: "${HETERO_HLT4_CONTROL_SEED:=12345}"
: "${TEACHER_LOGIT_GT_ROOT:=${OUTPUT_ROOT}/teacher_logit_reco_gt}"
: "${TEACHER_LOGIT_GT_RECO_ROOT:=${TEACHER_LOGIT_GT_ROOT}/reco}"
: "${TEACHER_LOGIT_GT_PREDICTION_RUN_ROOT:=${TEACHER_LOGIT_GT_ROOT}/prediction_runs}"
: "${TEACHER_LOGIT_GT_PREDICTION_DIR:=${TEACHER_LOGIT_GT_ROOT}/predictions}"
: "${TEACHER_LOGIT_GT_FUSION_DIR:=${TEACHER_LOGIT_GT_ROOT}/fusion}"
: "${TEACHER_LOGIT_GT_TEACHERS:=part}"
: "${TEACHER_LOGIT_GT_PART_TEACHER_CHECKPOINT:=${OFFLINE_TEACHER_DIR}/best_model_val.pt}"
: "${TEACHER_LOGIT_GT_PN_TEACHER_CHECKPOINT:=${OUTPUT_ROOT}/teacher_logit_reco_offline_teachers/pn/best_model_val.pt}"
: "${TEACHER_LOGIT_GT_PFN_TEACHER_CHECKPOINT:=${OUTPUT_ROOT}/teacher_logit_reco_offline_teachers/pfn/best_model_val.pt}"
: "${TEACHER_LOGIT_GT_PCNN_TEACHER_CHECKPOINT:=${OUTPUT_ROOT}/teacher_logit_reco_offline_teachers/pcnn/best_model_val.pt}"
: "${TEACHER_LOGIT_GT_BATCH_SIZE:=64}"
: "${TEACHER_LOGIT_GT_EPOCHS:=20}"
: "${TEACHER_LOGIT_GT_LR:=0.0003}"
: "${TEACHER_LOGIT_GT_WEIGHT_DECAY:=0.0001}"
: "${TEACHER_LOGIT_GT_EARLY_STOP_PATIENCE:=5}"
: "${TEACHER_LOGIT_GT_HIDDEN_DIM:=128}"
: "${TEACHER_LOGIT_GT_NUM_LAYERS:=4}"
: "${TEACHER_LOGIT_GT_NUM_HEADS:=4}"
: "${TEACHER_LOGIT_GT_NUM_EXTRA_CANDIDATES:=32}"
: "${TEACHER_LOGIT_GT_DROPOUT:=0.05}"
: "${TEACHER_LOGIT_GT_MAX_TRAIN_JETS:=}"
: "${TEACHER_LOGIT_GT_MAX_VAL_JETS:=}"
: "${TEACHER_LOGIT_GT_MAX_TRAIN_BATCHES:=}"
: "${TEACHER_LOGIT_GT_MAX_VAL_BATCHES:=}"
: "${TEACHER_LOGIT_GT_PREDICT_BATCH_SIZE:=128}"
: "${TEACHER_LOGIT_GT_PREDICT_NUM_WORKERS:=4}"
: "${TEACHER_LOGIT_GT_PREDICT_DEVICE:=${DEVICE}}"
: "${TEACHER_LOGIT_GT_MAX_JETS_PER_SPLIT:=}"
: "${TEACHER_LOGIT_GT_FEATURE_MODES:=logits probs logits_probs}"
: "${TEACHER_LOGIT_GT_C_GRID:=}"
: "${TEACHER_LOGIT_GT_MAX_ITER:=2000}"
: "${TEACHER_LOGIT_GT_SKIP_CONTROLS:=0}"
: "${TEACHER_LOGIT_GT_CONTROL_SEED:=12345}"
: "${TEACHER_LOGIT_PN_ROOT:=${OUTPUT_ROOT}/teacher_logit_reco_pn}"
: "${TEACHER_LOGIT_PN_RECO_ROOT:=${TEACHER_LOGIT_PN_ROOT}/reco}"
: "${TEACHER_LOGIT_PN_PREDICTION_RUN_ROOT:=${TEACHER_LOGIT_PN_ROOT}/prediction_runs}"
: "${TEACHER_LOGIT_PN_PREDICTION_DIR:=${TEACHER_LOGIT_PN_ROOT}/predictions}"
: "${TEACHER_LOGIT_PN_FUSION_DIR:=${TEACHER_LOGIT_PN_ROOT}/fusion}"
: "${TEACHER_LOGIT_PN_TEACHERS:=part}"
: "${TEACHER_LOGIT_PN_PART_TEACHER_CHECKPOINT:=${OFFLINE_TEACHER_DIR}/best_model_val.pt}"
: "${TEACHER_LOGIT_PN_PN_TEACHER_CHECKPOINT:=${OUTPUT_ROOT}/teacher_logit_reco_offline_teachers/pn/best_model_val.pt}"
: "${TEACHER_LOGIT_PN_PFN_TEACHER_CHECKPOINT:=${OUTPUT_ROOT}/teacher_logit_reco_offline_teachers/pfn/best_model_val.pt}"
: "${TEACHER_LOGIT_PN_PCNN_TEACHER_CHECKPOINT:=${OUTPUT_ROOT}/teacher_logit_reco_offline_teachers/pcnn/best_model_val.pt}"
: "${TEACHER_LOGIT_PN_BATCH_SIZE:=64}"
: "${TEACHER_LOGIT_PN_EPOCHS:=20}"
: "${TEACHER_LOGIT_PN_LR:=0.0003}"
: "${TEACHER_LOGIT_PN_WEIGHT_DECAY:=0.0001}"
: "${TEACHER_LOGIT_PN_EARLY_STOP_PATIENCE:=5}"
: "${TEACHER_LOGIT_PN_EDGECONV_DIMS:=64 128 128}"
: "${TEACHER_LOGIT_PN_K:=16}"
: "${TEACHER_LOGIT_PN_NUM_EXTRA_CANDIDATES:=32}"
: "${TEACHER_LOGIT_PN_DROPOUT:=0.05}"
: "${TEACHER_LOGIT_PN_MAX_TRAIN_JETS:=50000}"
: "${TEACHER_LOGIT_PN_MAX_VAL_JETS:=10000}"
: "${TEACHER_LOGIT_PN_MAX_TRAIN_BATCHES:=}"
: "${TEACHER_LOGIT_PN_MAX_VAL_BATCHES:=}"
: "${TEACHER_LOGIT_PN_PREDICT_BATCH_SIZE:=128}"
: "${TEACHER_LOGIT_PN_PREDICT_NUM_WORKERS:=4}"
: "${TEACHER_LOGIT_PN_PREDICT_DEVICE:=${DEVICE}}"
: "${TEACHER_LOGIT_PN_MAX_JETS_PER_SPLIT:=50000}"
: "${TEACHER_LOGIT_PN_FEATURE_MODES:=logits probs logits_probs}"
: "${TEACHER_LOGIT_PN_C_GRID:=}"
: "${TEACHER_LOGIT_PN_MAX_ITER:=2000}"
: "${TEACHER_LOGIT_PN_SKIP_CONTROLS:=0}"
: "${TEACHER_LOGIT_PN_CONTROL_SEED:=12345}"
: "${HLT5_SEEDS:=101 202 303 404 505}"
: "${SPLIT_SEEDS:=model_train=153 model_val=254 stack_train=356 stack_val=457 final_test=558}"
: "${FIXED_HLT_SEEDS:=model_train=1053 model_val=1054 stack_train=1055 stack_val=1056 final_test=1057}"
: "${FIXED_HLT_PARAMS:=jetclass_fixed_hlt.FixedHLTParams defaults}"

fresh_bool_enabled() {
  local value="${1:-0}"
  [[ "${value}" == "1" || "${value}" == "true" || "${value}" == "TRUE" || "${value}" == "yes" || "${value}" == "YES" ]]
}

fresh_is_dry_run() {
  fresh_bool_enabled "${DRY_RUN}" || fresh_bool_enabled "${PRINT_ONLY}"
}

fresh_source_commit() {
  git -C "${PROJECT_DIR}" rev-parse HEAD 2>/dev/null || printf 'unknown'
}

fresh_source_status_hash() {
  if command -v sha256sum >/dev/null 2>&1; then
    git -C "${PROJECT_DIR}" status --short 2>/dev/null | sha256sum | awk '{print $1}'
  else
    git -C "${PROJECT_DIR}" status --short 2>/dev/null | shasum -a 256 | awk '{print $1}'
  fi
}

fresh_print_context() {
  echo "job_name=${SLURM_JOB_NAME:-local}"
  echo "job_id=${SLURM_JOB_ID:-local}"
  echo "hostname=$(hostname)"
  echo "date=$(date -Is)"
  echo "pwd=$(pwd)"
  echo "args=$*"
  echo "PROJECT_DIR=${PROJECT_DIR}"
  echo "DATA_DIR=${DATA_DIR}"
  echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
  echo "LOG_DIR=${LOG_DIR}"
  echo "CONDA_ENV=${CONDA_ENV}"
  echo "PYTHON_BIN=${PYTHON_BIN}"
  echo "DRY_RUN=${DRY_RUN}"
  echo "PRINT_ONLY=${PRINT_ONLY}"
  echo "OVERWRITE=${OVERWRITE}"
  echo "DEVICE=${DEVICE}"
  echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"
  echo "SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE:-unset}"
  echo "SLURM_GPUS=${SLURM_GPUS:-unset}"
  echo "source_commit=$(fresh_source_commit)"
  echo "source_status_hash=$(fresh_source_status_hash)"
}

fresh_activate_env() {
  if fresh_bool_enabled "${SKIP_CONDA:-0}"; then
    echo "SKIP_CONDA=1; not activating conda"
    return 0
  fi
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV}"
    return 0
  fi
  local conda_sh="${CONDA_BASE:-${HOME}/miniconda3}/etc/profile.d/conda.sh"
  if [[ -f "${conda_sh}" ]]; then
    # shellcheck disable=SC1090
    source "${conda_sh}"
    conda activate "${CONDA_ENV}"
    return 0
  fi
  echo "Could not find conda. Set SKIP_CONDA=1 or CONDA_BASE=/path/to/miniconda." >&2
  return 2
}

fresh_setup() {
  if [[ ! -d "${PROJECT_DIR}" ]]; then
    echo "PROJECT_DIR does not exist: ${PROJECT_DIR}" >&2
    return 2
  fi
  mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}"
  cd "${PROJECT_DIR}"
  fresh_print_context "$@"
  fresh_activate_env
  "${PYTHON_BIN}" --version
}

fresh_prepare_submitter() {
  if [[ ! -d "${PROJECT_DIR}" ]]; then
    echo "PROJECT_DIR does not exist: ${PROJECT_DIR}" >&2
    return 2
  fi
  mkdir -p "${PROJECT_DIR}/fresh_check_logs"
  cd "${PROJECT_DIR}"
  echo "submitter_project_dir=${PROJECT_DIR}"
  echo "submitter_log_dir=${PROJECT_DIR}/fresh_check_logs"
  echo "submitter_dry_run=${DRY_RUN}"
}

fresh_require_data_dir() {
  if fresh_is_dry_run; then
    return 0
  fi
  if [[ ! -d "${DATA_DIR}" ]]; then
    echo "DATA_DIR does not exist on this machine: ${DATA_DIR}" >&2
    return 2
  fi
}

fresh_require_file() {
  local path="$1"
  if fresh_is_dry_run; then
    return 0
  fi
  if [[ ! -f "${path}" ]]; then
    echo "Required file is missing: ${path}" >&2
    return 2
  fi
}

fresh_require_dir() {
  local path="$1"
  if fresh_is_dry_run; then
    return 0
  fi
  if [[ ! -d "${path}" ]]; then
    echo "Required directory is missing: ${path}" >&2
    return 2
  fi
}

fresh_refuse_existing_path() {
  local path="$1"
  if fresh_is_dry_run; then
    return 0
  fi
  if [[ -e "${path}" ]]; then
    if ! fresh_bool_enabled "${OVERWRITE}"; then
      echo "Refusing to use existing path without OVERWRITE=1: ${path}" >&2
      return 2
    fi
  fi
}

fresh_refuse_existing_dir() {
  local path="$1"
  if fresh_is_dry_run; then
    return 0
  fi
  if [[ -d "${path}" ]]; then
    if ! fresh_bool_enabled "${OVERWRITE}"; then
      echo "Refusing to use existing directory without OVERWRITE=1: ${path}" >&2
      return 2
    fi
  fi
}

fresh_claim_new_dir() {
  local path="$1"
  if fresh_is_dry_run; then
    return 0
  fi
  if fresh_bool_enabled "${OVERWRITE}"; then
    mkdir -p "${path}"
    return 0
  fi
  mkdir -p "$(dirname "${path}")"
  if ! mkdir "${path}" 2>/dev/null; then
    echo "Refusing to use existing directory without OVERWRITE=1: ${path}" >&2
    return 2
  fi
}

fresh_run() {
  echo "PYTHON_COMMAND:"
  printf '  %q' "$@"
  echo
  if fresh_is_dry_run; then
    echo "DRY_RUN/PRINT_ONLY enabled; command not executed."
    return 0
  fi
  "$@"
}

fresh_append_optional_arg() {
  local -n _cmd_ref=$1
  local flag="$2"
  local value="${3:-}"
  if [[ -n "${value}" ]]; then
    _cmd_ref+=("${flag}" "${value}")
  fi
}

fresh_append_flag_if_enabled() {
  local -n _cmd_ref=$1
  local flag="$2"
  local value="${3:-0}"
  if fresh_bool_enabled "${value}"; then
    _cmd_ref+=("${flag}")
  fi
}

fresh_split_words() {
  local -n _out_ref=$1
  local value="${2:-}"
  local old_ifs="${IFS}"
  IFS=' '
  read -r -a _out_ref <<< "${value}"
  IFS="${old_ifs}"
}

fresh_print_shell_command() {
  printf '%q' "$1"
  shift || true
  while (($#)); do
    printf ' %q' "$1"
    shift
  done
}

fresh_write_run_config() {
  local output_dir="$1"
  local job_kind="$2"
  shift 2
  mkdir -p "${output_dir}"
  local command_text="$*"
  RUN_CONFIG_OUTPUT_DIR="${output_dir}" \
  RUN_CONFIG_JOB_KIND="${job_kind}" \
  RUN_CONFIG_COMMAND="${command_text}" \
  RUN_CONFIG_SOURCE_COMMIT="$(fresh_source_commit)" \
  RUN_CONFIG_SOURCE_STATUS_HASH="$(fresh_source_status_hash)" \
  "${PYTHON_BIN}" - <<'PY'
import json
import os
import platform
from pathlib import Path

keys = [
    "PROJECT_DIR",
    "DATA_DIR",
    "OUTPUT_ROOT",
    "MANIFEST_PATH",
    "HLT_CACHE_DIR",
    "HLT_BASELINE_SEED",
    "HLT_BASELINE_DIR",
    "HLT_CHECKPOINT",
    "HLT5_ROOT",
    "OFFLINE_TEACHER_DIR",
    "RECO7_ROOT",
    "RECO7_FUSION_DIR",
    "HLT5_FUSION_DIR",
    "RECO7_AUDIT_DIR",
    "HLT5_AUDIT_DIR",
    "FINAL_REPORT_DIR",
    "V2_STEP6_ROOT",
    "V2_STEP6_RECO_ROOT",
    "V2_STEP6_FUSION_DIR",
    "V2_STEP6_AUDIT_DIR",
    "V2_STEP6_VARIANT",
    "V2_STEP7_ROOT",
    "V2_STEP7_RECO_ROOT",
    "V2_STEP7_FUSION_DIR",
    "V2_STEP7_AUDIT_DIR",
    "V2_STEP7_VARIANTS",
    "RECO7_VARIANTS",
    "FUSION_MODEL_LOADING_ROOT",
    "FUSION_MODEL_LOADING_SMALL_DIR",
    "FUSION_MODEL_LOADING_LARGE_DIR",
    "FUSION_MODEL_LOADING_VARIANTS",
    "FUSION_MODEL_LOADING_FEATURE_MODES",
    "FUSION_MODEL_LOADING_C_GRID",
    "FUSION_MODEL_LOADING_MAX_ITER",
    "FUSION_MODEL_LOADING_SKIP_CONTROLS",
    "FUSION_MODEL_LOADING_CONTROL_SEED",
    "HETERO_HLT4_ROOT",
    "HETERO_HLT4_MODEL_ROOT",
    "HETERO_HLT4_FUSION_DIR",
    "HETERO_HLT4_ARCHITECTURES",
    "HETERO_HLT4_TRAIN_SIZE",
    "HETERO_HLT4_VAL_SIZE",
    "HETERO_HLT4_STACK_TRAIN_SIZE",
    "HETERO_HLT4_STACK_VAL_SIZE",
    "HETERO_HLT4_FINAL_TEST_SIZE",
    "HETERO_HLT4_FEATURE_MODES",
    "HETERO_HLT4_C_GRID",
    "HETERO_HLT4_MAX_ITER",
    "HETERO_HLT4_SKIP_CONTROLS",
    "HETERO_HLT4_CONTROL_SEED",
    "TEACHER_LOGIT_GT_ROOT",
    "TEACHER_LOGIT_GT_RECO_ROOT",
    "TEACHER_LOGIT_GT_PREDICTION_RUN_ROOT",
    "TEACHER_LOGIT_GT_PREDICTION_DIR",
    "TEACHER_LOGIT_GT_FUSION_DIR",
    "TEACHER_LOGIT_GT_TEACHERS",
    "TEACHER_LOGIT_GT_PART_TEACHER_CHECKPOINT",
    "TEACHER_LOGIT_GT_PN_TEACHER_CHECKPOINT",
    "TEACHER_LOGIT_GT_PFN_TEACHER_CHECKPOINT",
    "TEACHER_LOGIT_GT_PCNN_TEACHER_CHECKPOINT",
    "TEACHER_LOGIT_GT_BATCH_SIZE",
    "TEACHER_LOGIT_GT_EPOCHS",
    "TEACHER_LOGIT_GT_LR",
    "TEACHER_LOGIT_GT_WEIGHT_DECAY",
    "TEACHER_LOGIT_GT_EARLY_STOP_PATIENCE",
    "TEACHER_LOGIT_GT_HIDDEN_DIM",
    "TEACHER_LOGIT_GT_NUM_LAYERS",
    "TEACHER_LOGIT_GT_NUM_HEADS",
    "TEACHER_LOGIT_GT_NUM_EXTRA_CANDIDATES",
    "TEACHER_LOGIT_GT_DROPOUT",
    "TEACHER_LOGIT_GT_MAX_TRAIN_JETS",
    "TEACHER_LOGIT_GT_MAX_VAL_JETS",
    "TEACHER_LOGIT_GT_MAX_TRAIN_BATCHES",
    "TEACHER_LOGIT_GT_MAX_VAL_BATCHES",
    "TEACHER_LOGIT_GT_PREDICT_BATCH_SIZE",
    "TEACHER_LOGIT_GT_PREDICT_NUM_WORKERS",
    "TEACHER_LOGIT_GT_PREDICT_DEVICE",
    "TEACHER_LOGIT_GT_MAX_JETS_PER_SPLIT",
    "TEACHER_LOGIT_GT_FEATURE_MODES",
    "TEACHER_LOGIT_GT_C_GRID",
    "TEACHER_LOGIT_GT_MAX_ITER",
    "TEACHER_LOGIT_GT_SKIP_CONTROLS",
    "TEACHER_LOGIT_GT_CONTROL_SEED",
    "TEACHER_LOGIT_PN_ROOT",
    "TEACHER_LOGIT_PN_RECO_ROOT",
    "TEACHER_LOGIT_PN_PREDICTION_RUN_ROOT",
    "TEACHER_LOGIT_PN_PREDICTION_DIR",
    "TEACHER_LOGIT_PN_FUSION_DIR",
    "TEACHER_LOGIT_PN_TEACHERS",
    "TEACHER_LOGIT_PN_PART_TEACHER_CHECKPOINT",
    "TEACHER_LOGIT_PN_PN_TEACHER_CHECKPOINT",
    "TEACHER_LOGIT_PN_PFN_TEACHER_CHECKPOINT",
    "TEACHER_LOGIT_PN_PCNN_TEACHER_CHECKPOINT",
    "TEACHER_LOGIT_PN_BATCH_SIZE",
    "TEACHER_LOGIT_PN_EPOCHS",
    "TEACHER_LOGIT_PN_LR",
    "TEACHER_LOGIT_PN_WEIGHT_DECAY",
    "TEACHER_LOGIT_PN_EARLY_STOP_PATIENCE",
    "TEACHER_LOGIT_PN_EDGECONV_DIMS",
    "TEACHER_LOGIT_PN_K",
    "TEACHER_LOGIT_PN_NUM_EXTRA_CANDIDATES",
    "TEACHER_LOGIT_PN_DROPOUT",
    "TEACHER_LOGIT_PN_MAX_TRAIN_JETS",
    "TEACHER_LOGIT_PN_MAX_VAL_JETS",
    "TEACHER_LOGIT_PN_MAX_TRAIN_BATCHES",
    "TEACHER_LOGIT_PN_MAX_VAL_BATCHES",
    "TEACHER_LOGIT_PN_PREDICT_BATCH_SIZE",
    "TEACHER_LOGIT_PN_PREDICT_NUM_WORKERS",
    "TEACHER_LOGIT_PN_PREDICT_DEVICE",
    "TEACHER_LOGIT_PN_MAX_JETS_PER_SPLIT",
    "TEACHER_LOGIT_PN_FEATURE_MODES",
    "TEACHER_LOGIT_PN_C_GRID",
    "TEACHER_LOGIT_PN_MAX_ITER",
    "TEACHER_LOGIT_PN_SKIP_CONTROLS",
    "TEACHER_LOGIT_PN_CONTROL_SEED",
    "FUSION_STACK_TRAIN_SIZE",
    "FUSION_STACK_VAL_SIZE",
    "FUSION_FINAL_TEST_SIZE",
    "HLT5_SEEDS",
    "SPLIT_SEEDS",
    "FIXED_HLT_SEEDS",
    "FIXED_HLT_PARAMS",
    "HLT_BASELINE_REPORT",
    "TRAIN_SEED",
    "VARIANT",
    "OFFLINE_TEACHER_SEED",
    "MODEL_TRAIN_SIZE",
    "MODEL_VAL_SIZE",
    "STACK_TRAIN_SIZE",
    "STACK_VAL_SIZE",
    "FINAL_TEST_SIZE",
    "CONDA_ENV",
    "DEVICE",
    "MODEL_SIZE",
    "NO_AMP",
    "BATCH_SIZE",
    "EPOCHS",
    "LR",
    "STAGE_A_LR",
    "STAGE2_LR",
    "WEIGHT_DECAY",
    "EARLY_STOP_PATIENCE",
    "GRAD_CLIP_NORM",
    "FUSION_BATCH_SIZE",
    "FUSION_NUM_WORKERS",
    "FUSION_DEVICE",
    "FUSION_MAX_JETS_PER_SPLIT",
    "FUSION_FEATURE_MODE",
    "FUSION_MAX_ITER",
    "CONFIRM_FINAL_TEST",
]
payload = {
    "job_kind": os.environ["RUN_CONFIG_JOB_KIND"],
    "python_command": os.environ["RUN_CONFIG_COMMAND"],
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
    "hostname": platform.node(),
    "source_commit": os.environ.get("RUN_CONFIG_SOURCE_COMMIT"),
    "source_status_hash": os.environ.get("RUN_CONFIG_SOURCE_STATUS_HASH"),
    "environment": {key: os.environ.get(key) for key in keys if key in os.environ},
}
path = Path(os.environ["RUN_CONFIG_OUTPUT_DIR"]) / "slurm_run_config.json"
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"wrote {path}")
PY
}

fresh_assert_json_ok() {
  local path="$1"
  if fresh_is_dry_run; then
    return 0
  fi
  fresh_require_file "${path}"
  "${PYTHON_BIN}" - "${path}" <<'PY'
import json
import sys
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as handle:
    payload = json.load(handle)
if payload.get("ok") is not True:
    raise SystemExit(f"JSON report is not ok=True: {path}")
PY
}

fresh_write_audit_summary() {
  local report_path="$1"
  local summary_path="$2"
  if fresh_is_dry_run; then
    return 0
  fi
  fresh_require_file "${report_path}"
  "${PYTHON_BIN}" - "${report_path}" "${summary_path}" <<'PY'
import json
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
with report_path.open("r", encoding="utf-8") as handle:
    report = json.load(handle)
lines = [
    f"audit_report={report_path}",
    f"ok={report.get('ok')}",
    f"experiment_step={report.get('experiment_step')}",
    "",
    "audit_items:",
]
for name, item in sorted((report.get("audits") or {}).items()):
    ok = item.get("ok") if isinstance(item, dict) else None
    lines.append(f"  {name}: ok={ok}")
    problems = item.get("problems") if isinstance(item, dict) else None
    if problems:
        for problem in problems[:10]:
            lines.append(f"    - {problem}")
summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {summary_path}")
PY
}

fresh_join_by_colon() {
  local IFS=":"
  echo "$*"
}

fresh_join_by_space() {
  local IFS=" "
  echo "$*"
}

fresh_join_by_comma() {
  local IFS=","
  echo "$*"
}

fresh_teacher_logit_gt_teacher_checkpoint() {
  local architecture="$1"
  case "${architecture}" in
    part) echo "${TEACHER_LOGIT_GT_PART_TEACHER_CHECKPOINT}" ;;
    pn) echo "${TEACHER_LOGIT_GT_PN_TEACHER_CHECKPOINT}" ;;
    pfn) echo "${TEACHER_LOGIT_GT_PFN_TEACHER_CHECKPOINT}" ;;
    pcnn) echo "${TEACHER_LOGIT_GT_PCNN_TEACHER_CHECKPOINT}" ;;
    *)
      echo "Unknown teacher-logit GT teacher architecture: ${architecture}" >&2
      return 2
      ;;
  esac
}

fresh_teacher_logit_gt_model_name() {
  local architecture="$1"
  case "${architecture}" in
    part|pn|pfn|pcnn) echo "gt_reco_to_${architecture}_teacher" ;;
    *)
      echo "Unknown teacher-logit GT teacher architecture: ${architecture}" >&2
      return 2
      ;;
  esac
}

fresh_teacher_logit_pn_teacher_checkpoint() {
  local architecture="$1"
  case "${architecture}" in
    part) echo "${TEACHER_LOGIT_PN_PART_TEACHER_CHECKPOINT}" ;;
    pn) echo "${TEACHER_LOGIT_PN_PN_TEACHER_CHECKPOINT}" ;;
    pfn) echo "${TEACHER_LOGIT_PN_PFN_TEACHER_CHECKPOINT}" ;;
    pcnn) echo "${TEACHER_LOGIT_PN_PCNN_TEACHER_CHECKPOINT}" ;;
    *)
      echo "Unknown teacher-logit PN teacher architecture: ${architecture}" >&2
      return 2
      ;;
  esac
}

fresh_teacher_logit_pn_model_name() {
  local architecture="$1"
  case "${architecture}" in
    part|pn|pfn|pcnn) echo "pn_reco_to_${architecture}_teacher" ;;
    *)
      echo "Unknown teacher-logit PN teacher architecture: ${architecture}" >&2
      return 2
      ;;
  esac
}
