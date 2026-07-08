#!/usr/bin/env bash
# Submit the HLT multiview source/fusion graph on an existing PDV3 HLT-v2 root.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_ENV:=atlas_kd}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${UPSTREAM_DEPENDENCY:=}"
: "${HLT_MV_UPSTREAM_DEPENDENCY:=${UPSTREAM_DEPENDENCY}}"
: "${HLT_MV_PDV3_EXPERIMENT_NAME:=privileged_distill_v3_av10_adapter_fixed_hlt_v2_realistic_s1p0_highdata_20260705_190747}"
: "${HLT_MV_PDV3_ROOT:=${OUTPUT_ROOT}/${HLT_MV_PDV3_EXPERIMENT_NAME}}"
: "${HLT_MV_ROOT:=${HLT_MV_PDV3_ROOT}/hlt_multiview_source_fusion}"
: "${HLT_MV_ROOT_DIRNAME:=$(basename "${HLT_MV_ROOT}")}"
: "${HLT_MV_HLT_CACHE_DIR:=${HLT_MV_PDV3_ROOT}/inputs/hlt_cache}"
: "${HLT_MV_HLT2_CACHE_ROOT:=${HLT_MV_PDV3_ROOT}/hlt_self_dualview/hlt2_cache}"
: "${HLT_MV_SOURCE_MODELS_DIR:=${HLT_MV_ROOT}/source_models}"
: "${HLT_MV_RANDOM_HLT_CONTROLS_DIR:=${HLT_MV_ROOT}/hlt_random_seed_controls}"
: "${HLT_MV_LOGIT_FUSIONS_DIR:=${HLT_MV_ROOT}/logit_fusions}"
: "${HLT_MV_PRETRAINED_DUALVIEW_DIR:=${HLT_MV_ROOT}/particle_dualview_pretrained}"
: "${HLT_MV_SCRATCH_DUALVIEW_DIR:=${HLT_MV_ROOT}/particle_dualview_scratch}"
: "${HLT_MV_CONTROLS_DIR:=${HLT_MV_ROOT}/controls}"
: "${HLT_MV_TRIVIEW_DIR:=${HLT_MV_ROOT}/triview}"
: "${HLT_MV_FINAL_REPORT_DIR:=${HLT_MV_ROOT}/final_report}"
: "${HLT_MV_CANONICAL_HLT_SOURCE_NAME:=hlt_part_seed8801}"
: "${HLT_MV_STRENGTHS:=0.10 0.20 0.35 1.00}"
: "${HLT_MV_HLT2_SOURCE_SEEDS:=0.10=8811 0.20=8821 0.35=8831 0.50=8851 1.00=8841 1.50=8861 2.00=8871}"
: "${HLT_MV_SOURCE_NAMES:=hlt_part_seed8801 hlt2_part_s0p10_seed8811 hlt2_part_s0p20_seed8821 hlt2_part_s0p35_seed8831 hlt2_part_s1p00_seed8841}"
: "${HLT_MV_RANDOM_HLT_SOURCE_NAMES:=hlt_part_seed9101 hlt_part_seed9102 hlt_part_seed9103 hlt_part_seed9104}"
: "${HLT_MV_PRETRAINED_DUALVIEW_NAMES:=sdv_hlt_hlt2_s0p10 sdv_hlt_hlt2_s0p20 sdv_hlt_hlt2_s0p35 sdv_hlt_hlt2_s1p00}"
: "${HLT_MV_SCRATCH_DUALVIEW_NAMES:=sdv_hlt_hlt2_s0p10_scratch sdv_hlt_hlt2_s0p20_scratch sdv_hlt_hlt2_s0p35_scratch sdv_hlt_hlt2_s1p00_scratch}"
: "${HLT_MV_TTA_STRENGTHS:=0.10 0.20 0.35 1.00}"
: "${HLT_MV_TRIVIEW_MODEL_NAME:=tri_hlt_hlt2_s0p35_s1p00}"
: "${HLT_MV_ALLOW_PENDING_HLT2_CACHES:=0}"
: "${HLT_MV_SBATCH_PARTITION:=}"
: "${HLT_MV_GPU_GRES:=}"
: "${HLT_MV_GPU_CPUS_PER_TASK:=}"
: "${HLT_MV_GPU_MEM:=}"
: "${HLT_MV_GPU_TIME:=}"
: "${HLT_MV_CPU_CPUS_PER_TASK:=}"
: "${HLT_MV_CPU_MEM:=}"
: "${HLT_MV_CPU_TIME:=}"
: "${HLT_MV_FINAL_REPORT_ALLOW_MISSING:=0}"
: "${HLT_MV_FINAL_REPORT_REQUIRE_TRIVIEW:=0}"

export PROJECT_DIR DATA_DIR OUTPUT_ROOT DIAGNOSTICS_ROOT LOG_DIR CONDA_ENV PYTHON_BIN
export HLT_MV_PDV3_EXPERIMENT_NAME HLT_MV_PDV3_ROOT HLT_MV_ROOT HLT_MV_ROOT_DIRNAME
export HLT_MV_HLT_CACHE_DIR HLT_MV_HLT2_CACHE_ROOT HLT_MV_SOURCE_MODELS_DIR
export HLT_MV_RANDOM_HLT_CONTROLS_DIR HLT_MV_LOGIT_FUSIONS_DIR
export HLT_MV_PRETRAINED_DUALVIEW_DIR HLT_MV_SCRATCH_DUALVIEW_DIR
export HLT_MV_CONTROLS_DIR HLT_MV_TRIVIEW_DIR HLT_MV_FINAL_REPORT_DIR
export HLT_MV_CANONICAL_HLT_SOURCE_NAME HLT_MV_STRENGTHS HLT_MV_HLT2_SOURCE_SEEDS
export HLT_MV_SOURCE_NAMES HLT_MV_RANDOM_HLT_SOURCE_NAMES
export HLT_MV_PRETRAINED_DUALVIEW_NAMES HLT_MV_SCRATCH_DUALVIEW_NAMES
export HLT_MV_TTA_STRENGTHS HLT_MV_TRIVIEW_MODEL_NAME
export HLT_MV_ALLOW_PENDING_HLT2_CACHES
export HLT_MV_SBATCH_PARTITION HLT_MV_GPU_GRES HLT_MV_GPU_CPUS_PER_TASK HLT_MV_GPU_MEM HLT_MV_GPU_TIME
export HLT_MV_CPU_CPUS_PER_TASK HLT_MV_CPU_MEM HLT_MV_CPU_TIME
export HLT_MV_SOURCE_EPOCHS HLT_MV_SOURCE_BATCH_SIZE HLT_MV_SOURCE_EVAL_BATCH_SIZE
export HLT_MV_SOURCE_LR HLT_MV_SOURCE_WEIGHT_DECAY HLT_MV_SOURCE_EARLY_STOP_PATIENCE
export HLT_MV_SOURCE_GRAD_CLIP_NORM HLT_MV_SOURCE_NUM_WORKERS HLT_MV_SOURCE_DEVICE
export HLT_MV_SOURCE_MODEL_SIZE HLT_MV_SOURCE_AMP HLT_MV_SOURCE_COMPILE_MODEL
export HLT_MV_SOURCE_SKIP_MODEL_VAL_PREDICTIONS HLT_MV_SOURCE_SKIP_FINAL_TEST
export HLT_MV_SOURCE_MAX_TRAIN_BATCHES HLT_MV_SOURCE_MAX_VAL_BATCHES
export HLT_MV_SOURCE_MAX_FINAL_TEST_BATCHES HLT_MV_SOURCE_TRAIN_SIZE
export HLT_MV_SOURCE_VAL_SIZE HLT_MV_SOURCE_FINAL_TEST_SIZE
export HLT_MV_PRETRAINED_DUALVIEW_EPOCHS HLT_MV_PRETRAINED_DUALVIEW_HEAD_WARMUP_EPOCHS
export HLT_MV_PRETRAINED_DUALVIEW_BATCH_SIZE HLT_MV_PRETRAINED_DUALVIEW_EVAL_BATCH_SIZE
export HLT_MV_PRETRAINED_DUALVIEW_HEAD_WARMUP_LR HLT_MV_PRETRAINED_DUALVIEW_BRANCH_LR
export HLT_MV_PRETRAINED_DUALVIEW_HEAD_LR HLT_MV_PRETRAINED_DUALVIEW_WEIGHT_DECAY
export HLT_MV_PRETRAINED_DUALVIEW_DROPOUT HLT_MV_PRETRAINED_DUALVIEW_FUSION_HIDDEN_DIM
export HLT_MV_PRETRAINED_DUALVIEW_REPRESENTATION_DIM HLT_MV_PRETRAINED_DUALVIEW_EARLY_STOP_PATIENCE
export HLT_MV_PRETRAINED_DUALVIEW_GRAD_CLIP_NORM HLT_MV_PRETRAINED_DUALVIEW_NUM_WORKERS
export HLT_MV_PRETRAINED_DUALVIEW_DEVICE HLT_MV_PRETRAINED_DUALVIEW_SEED
export HLT_MV_PRETRAINED_DUALVIEW_MODEL_SIZE HLT_MV_PRETRAINED_DUALVIEW_AMP
export HLT_MV_PRETRAINED_DUALVIEW_COMPILE_MODEL HLT_MV_PRETRAINED_DUALVIEW_SKIP_MODEL_VAL_PREDICTIONS
export HLT_MV_PRETRAINED_DUALVIEW_SKIP_FINAL_TEST HLT_MV_PRETRAINED_DUALVIEW_MAX_TRAIN_BATCHES
export HLT_MV_PRETRAINED_DUALVIEW_MAX_VAL_BATCHES HLT_MV_PRETRAINED_DUALVIEW_MAX_FINAL_TEST_BATCHES
export HLT_MV_PRETRAINED_DUALVIEW_TRAIN_SIZE HLT_MV_PRETRAINED_DUALVIEW_VAL_SIZE
export HLT_MV_PRETRAINED_DUALVIEW_FINAL_TEST_SIZE
export HLT_MV_SCRATCH_DUALVIEW_EPOCHS HLT_MV_SCRATCH_DUALVIEW_HEAD_WARMUP_EPOCHS
export HLT_MV_SCRATCH_DUALVIEW_BATCH_SIZE HLT_MV_SCRATCH_DUALVIEW_EVAL_BATCH_SIZE
export HLT_MV_SCRATCH_DUALVIEW_HEAD_WARMUP_LR HLT_MV_SCRATCH_DUALVIEW_BRANCH_LR
export HLT_MV_SCRATCH_DUALVIEW_HEAD_LR HLT_MV_SCRATCH_DUALVIEW_WEIGHT_DECAY
export HLT_MV_SCRATCH_DUALVIEW_DROPOUT HLT_MV_SCRATCH_DUALVIEW_FUSION_HIDDEN_DIM
export HLT_MV_SCRATCH_DUALVIEW_REPRESENTATION_DIM HLT_MV_SCRATCH_DUALVIEW_EARLY_STOP_PATIENCE
export HLT_MV_SCRATCH_DUALVIEW_GRAD_CLIP_NORM HLT_MV_SCRATCH_DUALVIEW_NUM_WORKERS
export HLT_MV_SCRATCH_DUALVIEW_DEVICE HLT_MV_SCRATCH_DUALVIEW_SEED
export HLT_MV_SCRATCH_DUALVIEW_MODEL_SIZE HLT_MV_SCRATCH_DUALVIEW_AMP
export HLT_MV_SCRATCH_DUALVIEW_COMPILE_MODEL HLT_MV_SCRATCH_DUALVIEW_SKIP_MODEL_VAL_PREDICTIONS
export HLT_MV_SCRATCH_DUALVIEW_SKIP_FINAL_TEST HLT_MV_SCRATCH_DUALVIEW_MAX_TRAIN_BATCHES
export HLT_MV_SCRATCH_DUALVIEW_MAX_VAL_BATCHES HLT_MV_SCRATCH_DUALVIEW_MAX_FINAL_TEST_BATCHES
export HLT_MV_SCRATCH_DUALVIEW_TRAIN_SIZE HLT_MV_SCRATCH_DUALVIEW_VAL_SIZE
export HLT_MV_SCRATCH_DUALVIEW_FINAL_TEST_SIZE
export HLT_MV_SAME_VIEW_SEED HLT_MV_SAME_VIEW_EPOCHS HLT_MV_SAME_VIEW_HEAD_WARMUP_EPOCHS
export HLT_MV_SAME_VIEW_BATCH_SIZE HLT_MV_SAME_VIEW_EVAL_BATCH_SIZE
export HLT_MV_SAME_VIEW_HEAD_WARMUP_LR HLT_MV_SAME_VIEW_BRANCH_LR HLT_MV_SAME_VIEW_HEAD_LR
export HLT_MV_SAME_VIEW_WEIGHT_DECAY HLT_MV_SAME_VIEW_DROPOUT HLT_MV_SAME_VIEW_FUSION_HIDDEN_DIM
export HLT_MV_SAME_VIEW_REPRESENTATION_DIM HLT_MV_SAME_VIEW_EARLY_STOP_PATIENCE
export HLT_MV_SAME_VIEW_NUM_WORKERS HLT_MV_SAME_VIEW_DEVICE HLT_MV_SAME_VIEW_MODEL_SIZE
export HLT_MV_SAME_VIEW_NO_AMP HLT_MV_SAME_VIEW_COMPILE_MODEL
export HLT_MV_SAME_VIEW_SKIP_MODEL_VAL_PREDICTIONS HLT_MV_SAME_VIEW_SKIP_FINAL_TEST
export HLT_MV_SAME_VIEW_MAX_TRAIN_BATCHES HLT_MV_SAME_VIEW_MAX_VAL_BATCHES
export HLT_MV_SAME_VIEW_MAX_FINAL_TEST_BATCHES HLT_MV_SAME_VIEW_TRAIN_SIZE
export HLT_MV_SAME_VIEW_VAL_SIZE HLT_MV_SAME_VIEW_FINAL_TEST_SIZE
export HLT_MV_TTA_SEED HLT_MV_TTA_BATCH_SIZE HLT_MV_TTA_NUM_WORKERS HLT_MV_TTA_DEVICE
export HLT_MV_TTA_SKIP_FINAL_TEST HLT_MV_TTA_MAX_VAL_BATCHES HLT_MV_TTA_MAX_FINAL_TEST_BATCHES
export HLT_MV_TTA_VAL_SIZE HLT_MV_TTA_FINAL_TEST_SIZE
export HLT_MV_TRIVIEW_SEED HLT_MV_TRIVIEW_EPOCHS HLT_MV_TRIVIEW_HEAD_WARMUP_EPOCHS
export HLT_MV_TRIVIEW_BATCH_SIZE HLT_MV_TRIVIEW_EVAL_BATCH_SIZE
export HLT_MV_TRIVIEW_HEAD_WARMUP_LR HLT_MV_TRIVIEW_BRANCH_LR HLT_MV_TRIVIEW_HEAD_LR
export HLT_MV_TRIVIEW_WEIGHT_DECAY HLT_MV_TRIVIEW_DROPOUT HLT_MV_TRIVIEW_FUSION_HIDDEN_DIM
export HLT_MV_TRIVIEW_REPRESENTATION_DIM HLT_MV_TRIVIEW_EARLY_STOP_PATIENCE
export HLT_MV_TRIVIEW_GRAD_CLIP_NORM HLT_MV_TRIVIEW_NUM_WORKERS HLT_MV_TRIVIEW_DEVICE
export HLT_MV_TRIVIEW_MODEL_SIZE HLT_MV_TRIVIEW_AMP HLT_MV_TRIVIEW_COMPILE_MODEL
export HLT_MV_TRIVIEW_SKIP_MODEL_VAL_PREDICTIONS HLT_MV_TRIVIEW_SKIP_FINAL_TEST
export HLT_MV_TRIVIEW_MAX_TRAIN_BATCHES HLT_MV_TRIVIEW_MAX_VAL_BATCHES
export HLT_MV_TRIVIEW_MAX_FINAL_TEST_BATCHES HLT_MV_TRIVIEW_TRAIN_SIZE
export HLT_MV_TRIVIEW_VAL_SIZE HLT_MV_TRIVIEW_FINAL_TEST_SIZE
export HLT_MV_LOGIT_FUSION_SKIP_WEIGHTED_AVERAGE HLT_MV_LOGIT_FUSION_MAX_WEIGHT_STEPS
export HLT_MV_LOGIT_FUSION_SPLITS HLT_MV_FINAL_REPORT_ALLOW_MISSING HLT_MV_FINAL_REPORT_REQUIRE_TRIVIEW
export CONFIRM_FINAL_TEST SKIP_EXISTING OVERWRITE DEVICE DRY_RUN PRINT_ONLY

fresh_prepare_submitter

if ! fresh_bool_enabled "${CONFIRM_FINAL_TEST}"; then
  echo "Refusing to submit HLT-MV final-test graph without CONFIRM_FINAL_TEST=1." >&2
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

job_export_arg() {
  printf '%s\n' "--export=ALL"
}

sbatch_resource_args() {
  local profile="$1"
  if [[ -n "${HLT_MV_SBATCH_PARTITION}" ]]; then
    printf '%s\n' "--partition=${HLT_MV_SBATCH_PARTITION}"
  fi
  if [[ "${profile}" == "gpu" ]]; then
    if [[ -n "${HLT_MV_GPU_GRES}" ]]; then
      printf '%s\n' "--gres=${HLT_MV_GPU_GRES}"
    fi
    if [[ -n "${HLT_MV_GPU_CPUS_PER_TASK}" ]]; then
      printf '%s\n' "--cpus-per-task=${HLT_MV_GPU_CPUS_PER_TASK}"
    fi
    if [[ -n "${HLT_MV_GPU_MEM}" ]]; then
      printf '%s\n' "--mem=${HLT_MV_GPU_MEM}"
    fi
    if [[ -n "${HLT_MV_GPU_TIME}" ]]; then
      printf '%s\n' "--time=${HLT_MV_GPU_TIME}"
    fi
  else
    if [[ -n "${HLT_MV_CPU_CPUS_PER_TASK}" ]]; then
      printf '%s\n' "--cpus-per-task=${HLT_MV_CPU_CPUS_PER_TASK}"
    fi
    if [[ -n "${HLT_MV_CPU_MEM}" ]]; then
      printf '%s\n' "--mem=${HLT_MV_CPU_MEM}"
    fi
    if [[ -n "${HLT_MV_CPU_TIME}" ]]; then
      printf '%s\n' "--time=${HLT_MV_CPU_TIME}"
    fi
  fi
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
    echo "Refusing to submit ${label}; found existing incomplete output path:" >&2
    echo "  ${path}" >&2
    echo "Use SKIP_EXISTING=1 for complete artifacts, OVERWRITE=1 to repair, or remove the partial output." >&2
    return 2
  fi
}

skip_existing_model_with_predictions() {
  local label="$1"
  local output_dir="$2"
  local model_name="$3"
  local sidecar="${4:-}"
  if ! fresh_bool_enabled "${SKIP_EXISTING}" || fresh_is_dry_run || fresh_bool_enabled "${OVERWRITE}"; then
    return 1
  fi
  all_artifacts_exist \
    "${output_dir}/best_model_val.pt" \
    "${output_dir}/last.pt" \
    "${output_dir}/run_report.json" \
    "${output_dir}/model_val_report.json" \
    "${output_dir}/config.json" \
    "${output_dir}/training_curves.json" \
    "${output_dir}/final_test_report.json" \
    "${output_dir}/predictions/${model_name}/model_val_predictions.npz" \
    "${output_dir}/predictions/${model_name}/model_val_predictions_metadata.json" \
    "${output_dir}/predictions/${model_name}/final_test_predictions.npz" \
    "${output_dir}/predictions/${model_name}/final_test_predictions_metadata.json" || return 1
  if [[ -n "${sidecar}" ]]; then
    [[ -f "${output_dir}/${sidecar}" ]] || return 1
  fi
  json_ok_true "${output_dir}/run_report.json" || return 1
  skip_count=$((skip_count + 1))
  echo "skipped ${label}; found complete model artifact set" >&2
  return 0
}

skip_existing_tta_control() {
  local label="$1"
  local output_dir="$2"
  local model_name="$3"
  if ! fresh_bool_enabled "${SKIP_EXISTING}" || fresh_is_dry_run || fresh_bool_enabled "${OVERWRITE}"; then
    return 1
  fi
  all_artifacts_exist \
    "${output_dir}/run_report.json" \
    "${output_dir}/model_val_report.json" \
    "${output_dir}/final_test_report.json" \
    "${output_dir}/predictions/${model_name}/model_val_predictions.npz" \
    "${output_dir}/predictions/${model_name}/model_val_predictions_metadata.json" \
    "${output_dir}/predictions/${model_name}/final_test_predictions.npz" \
    "${output_dir}/predictions/${model_name}/final_test_predictions_metadata.json" || return 1
  json_ok_true "${output_dir}/run_report.json" || return 1
  skip_count=$((skip_count + 1))
  echo "skipped ${label}; found complete TTA control artifact set" >&2
  return 0
}

skip_existing_fusion() {
  local label="$1"
  local output_dir="$2"
  if ! fresh_bool_enabled "${SKIP_EXISTING}" || fresh_is_dry_run || fresh_bool_enabled "${OVERWRITE}"; then
    return 1
  fi
  all_artifacts_exist \
    "${output_dir}/fusion_report.json" \
    "${output_dir}/summary.json" \
    "${output_dir}/run_report.json" \
    "${output_dir}/metric_table.csv" || return 1
  json_ok_true "${output_dir}/run_report.json" || return 1
  skip_count=$((skip_count + 1))
  echo "skipped ${label}; found complete logit-fusion artifact set" >&2
  return 0
}

skip_existing_final_report() {
  if ! fresh_bool_enabled "${SKIP_EXISTING}" || fresh_is_dry_run || fresh_bool_enabled "${OVERWRITE}"; then
    return 1
  fi
  all_artifacts_exist \
    "${HLT_MV_FINAL_REPORT_DIR}/summary.json" \
    "${HLT_MV_FINAL_REPORT_DIR}/hlt_multiview_source_fusion_report.json" \
    "${HLT_MV_FINAL_REPORT_DIR}/hlt_multiview_source_fusion_report.md" \
    "${HLT_MV_FINAL_REPORT_DIR}/metric_table.csv" \
    "${HLT_MV_FINAL_REPORT_DIR}/run_report.json" || return 1
  json_ok_true "${HLT_MV_FINAL_REPORT_DIR}/run_report.json" || return 1
  skip_count=$((skip_count + 1))
  echo "skipped hlt_mv_final_report; found complete final-report artifact set" >&2
  return 0
}

name_in_words() {
  local needle="$1"
  local words="$2"
  local names=()
  local name
  fresh_split_words names "${words}"
  for name in "${names[@]}"; do
    if [[ "${name}" == "${needle}" ]]; then
      return 0
    fi
  done
  return 1
}

hlt_mv_hlt2_tag_from_pretrained_variant() {
  local variant="$1"
  if [[ "${variant}" =~ ^sdv_hlt_hlt2_(s[0-9]+p[0-9]+)$ ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
    return 0
  fi
  echo "Unknown HLT-MV pretrained dual-view variant: ${variant}" >&2
  return 2
}

hlt_mv_hlt2_tag_from_scratch_variant() {
  local variant="$1"
  if [[ "${variant}" =~ ^sdv_hlt_hlt2_(s[0-9]+p[0-9]+)_scratch$ ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
    return 0
  fi
  echo "Unknown HLT-MV scratch dual-view variant: ${variant}" >&2
  return 2
}

hlt_mv_hlt2_source_name_for_tag() {
  local tag="$1"
  local source_name
  for source_name in "${source_names[@]}"; do
    if [[ "${source_name}" =~ ^hlt2_part_${tag}_seed[0-9]+$ ]]; then
      printf '%s\n' "${source_name}"
      return 0
    fi
  done
  echo "No HLT-MV source name configured for HLT2 tag ${tag}." >&2
  return 2
}

hlt_mv_submitted_job_id_for_source() {
  local source_name="$1"
  local index
  for index in "${!submitted_source_names[@]}"; do
    if [[ "${submitted_source_names[$index]}" == "${source_name}" ]]; then
      printf '%s\n' "${submitted_source_job_ids[$index]}"
      return 0
    fi
  done
  return 0
}

hlt_mv_triview_tags() {
  if [[ "${HLT_MV_TRIVIEW_MODEL_NAME}" =~ ^tri_hlt_hlt2_(s[0-9]+p[0-9]+)_(s[0-9]+p[0-9]+)$ ]]; then
    printf '%s\n%s\n' "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"
    return 0
  fi
  echo "HLT-MV tri-view model must look like tri_hlt_hlt2_sXpYY_sXpYY; got ${HLT_MV_TRIVIEW_MODEL_NAME}" >&2
  return 2
}

source_output_dir() {
  local source_name="$1"
  if name_in_words "${source_name}" "${HLT_MV_RANDOM_HLT_SOURCE_NAMES}"; then
    printf '%s\n' "${HLT_MV_RANDOM_HLT_CONTROLS_DIR}/${source_name}"
  else
    printf '%s\n' "${HLT_MV_SOURCE_MODELS_DIR}/${source_name}"
  fi
}

submit_source_model() {
  local source_name="$1"
  local dependency="$2"
  local output_dir
  output_dir="$(source_output_dir "${source_name}")"
  if skip_existing_model_with_predictions "hlt_mv_source_${source_name}" "${output_dir}" "${source_name}"; then
    submitted_job_id=""
    return 0
  fi
  refuse_partial_existing_output_dir "hlt_mv_source_${source_name}" "${output_dir}"
  local resource_args=()
  mapfile -t resource_args < <(sbatch_resource_args gpu)
  mapfile -t args < <(
    afterok_args "${dependency}" \
      "$(job_export_arg)" \
      "${resource_args[@]}" \
      "${SCRIPT_DIR}/run_hlt_mv_train_source_model.sh" "${source_name}"
  )
  submit_job "hlt_mv_source_${source_name}" "${args[@]}"
  echo "submitted hlt_mv_source_${source_name}=${submitted_job_id}"
}

submit_model_job() {
  local label="$1"
  local dependency="$2"
  local output_dir="$3"
  local model_name="$4"
  local sidecar="$5"
  shift 5
  if skip_existing_model_with_predictions "${label}" "${output_dir}" "${model_name}" "${sidecar}"; then
    submitted_job_id=""
    return 0
  fi
  refuse_partial_existing_output_dir "${label}" "${output_dir}"
  local resource_args=()
  mapfile -t resource_args < <(sbatch_resource_args gpu)
  mapfile -t args < <(
    afterok_args "${dependency}" \
      "$(job_export_arg)" \
      "${resource_args[@]}" \
      "$@"
  )
  submit_job "${label}" "${args[@]}"
  echo "submitted ${label}=${submitted_job_id}"
}

submit_tta_job() {
  local strength="$1"
  local dependency="$2"
  local tag
  tag="$(fresh_pd10_hlt_sdv_strength_tag "${strength}")"
  local variant="tta_hlt_part_hlt_plus_hlt2_${tag}"
  local output_dir="${HLT_MV_CONTROLS_DIR}/${variant}"
  if skip_existing_tta_control "hlt_mv_tta_${tag}" "${output_dir}" "${variant}"; then
    submitted_job_id=""
    return 0
  fi
  refuse_partial_existing_output_dir "hlt_mv_tta_${tag}" "${output_dir}"
  local resource_args=()
  mapfile -t resource_args < <(sbatch_resource_args gpu)
  mapfile -t args < <(
    afterok_args "${dependency}" \
      "$(job_export_arg)" \
      "${resource_args[@]}" \
      "${SCRIPT_DIR}/run_hlt_mv_eval_tta_control.sh" "${strength}" "${variant}"
  )
  submit_job "hlt_mv_tta_${tag}" "${args[@]}"
  echo "submitted hlt_mv_tta_${tag}=${submitted_job_id}"
}

submit_fusion_job() {
  local fusion_name="$1"
  local dependency="$2"
  local output_dir="${HLT_MV_LOGIT_FUSIONS_DIR}/${fusion_name}"
  if skip_existing_fusion "hlt_mv_fusion_${fusion_name}" "${output_dir}"; then
    submitted_job_id=""
    return 0
  fi
  refuse_partial_existing_output_dir "hlt_mv_fusion_${fusion_name}" "${output_dir}"
  local resource_args=()
  mapfile -t resource_args < <(sbatch_resource_args cpu)
  mapfile -t args < <(
    afterok_args "${dependency}" \
      "$(job_export_arg)" \
      "${resource_args[@]}" \
      "${SCRIPT_DIR}/run_hlt_mv_logit_fusion.sh" "${fusion_name}"
  )
  submit_job "hlt_mv_fusion_${fusion_name}" "${args[@]}"
  echo "submitted hlt_mv_fusion_${fusion_name}=${submitted_job_id}"
}

validate_dependency_list "HLT_MV_UPSTREAM_DEPENDENCY" "${HLT_MV_UPSTREAM_DEPENDENCY}"

fresh_split_words source_names "${HLT_MV_SOURCE_NAMES}"
fresh_split_words random_source_names "${HLT_MV_RANDOM_HLT_SOURCE_NAMES}"
fresh_split_words pretrained_names "${HLT_MV_PRETRAINED_DUALVIEW_NAMES}"
fresh_split_words scratch_names "${HLT_MV_SCRATCH_DUALVIEW_NAMES}"
fresh_split_words tta_strengths "${HLT_MV_TTA_STRENGTHS}"

base_dep="${HLT_MV_UPSTREAM_DEPENDENCY}"
fresh_require_dir "${HLT_MV_HLT_CACHE_DIR}"
for split in model_train model_val final_test; do
  fresh_require_file "${HLT_MV_HLT_CACHE_DIR}/${split}_fixed_hlt_metadata.json"
done
for strength in "${tta_strengths[@]}"; do
  tag="$(fresh_pd10_hlt_sdv_strength_tag "${strength}")"
  cache_dir="${HLT_MV_HLT2_CACHE_ROOT}/hlt_second_degrade_mild_v1_${tag}"
  if [[ -d "${cache_dir}" ]]; then
    for split in model_train model_val final_test; do
      fresh_require_file "${cache_dir}/${split}_fixed_hlt_metadata.json"
    done
  elif fresh_bool_enabled "${HLT_MV_ALLOW_PENDING_HLT2_CACHES}"; then
    echo "allowing pending HLT2 cache for ${tag}: ${cache_dir}" >&2
  else
    fresh_require_dir "${cache_dir}"
  fi
done

submission_dir="${HLT_MV_ROOT}/submission_logs/hlt_mv_$(date +%Y%m%d_%H%M%S)"
fresh_claim_new_dir "${submission_dir}"
if ! fresh_is_dry_run; then
  {
    echo "created_at=$(date -Is)"
    echo "project_dir=${PROJECT_DIR}"
    echo "source_commit=$(fresh_source_commit)"
    echo "source_status_hash=$(fresh_source_status_hash)"
    echo "pdv3_root=${HLT_MV_PDV3_ROOT}"
    echo "hlt_mv_root=${HLT_MV_ROOT}"
    echo "hlt_mv_root_dirname=${HLT_MV_ROOT_DIRNAME}"
    echo "upstream_dependency=${HLT_MV_UPSTREAM_DEPENDENCY}"
    echo "skip_existing=${SKIP_EXISTING}"
    echo "overwrite=${OVERWRITE}"
    echo "confirm_final_test=${CONFIRM_FINAL_TEST}"
    echo "strengths=${HLT_MV_STRENGTHS}"
    echo "hlt2_source_seeds=${HLT_MV_HLT2_SOURCE_SEEDS}"
    echo "canonical_hlt_source_name=${HLT_MV_CANONICAL_HLT_SOURCE_NAME}"
    echo "source_names=${HLT_MV_SOURCE_NAMES}"
    echo "random_hlt_source_names=${HLT_MV_RANDOM_HLT_SOURCE_NAMES}"
    echo "pretrained_dualview_names=${HLT_MV_PRETRAINED_DUALVIEW_NAMES}"
    echo "scratch_dualview_names=${HLT_MV_SCRATCH_DUALVIEW_NAMES}"
    echo "tta_strengths=${HLT_MV_TTA_STRENGTHS}"
    echo "triview_model_name=${HLT_MV_TRIVIEW_MODEL_NAME}"
    echo "allow_pending_hlt2_caches=${HLT_MV_ALLOW_PENDING_HLT2_CACHES}"
    echo "final_report_require_triview=${HLT_MV_FINAL_REPORT_REQUIRE_TRIVIEW}"
  } > "${submission_dir}/metadata.txt"
fi

source_job_ids=()
submitted_source_names=()
submitted_source_job_ids=()
for source_name in "${source_names[@]}"; do
  submit_source_model "${source_name}" "${base_dep}"
  if [[ -n "${submitted_job_id}" ]]; then
    source_job_ids+=("${submitted_job_id}")
    submitted_source_names+=("${source_name}")
    submitted_source_job_ids+=("${submitted_job_id}")
  fi
done
canonical_hlt_source_job_id="$(hlt_mv_submitted_job_id_for_source "${HLT_MV_CANONICAL_HLT_SOURCE_NAME}")"
source_dep="$(join_nonempty_by_colon "${base_dep}" "${source_job_ids[@]}")"

random_job_ids=()
for source_name in "${random_source_names[@]}"; do
  submit_source_model "${source_name}" "${base_dep}"
  if [[ -n "${submitted_job_id}" ]]; then
    random_job_ids+=("${submitted_job_id}")
  fi
done
random_dep="$(join_nonempty_by_colon "${base_dep}" "${random_job_ids[@]}")"

submit_fusion_job source_5view "${source_dep}"
source_fusion_job_id="${submitted_job_id}"
submit_fusion_job hlt_random_4seed "${random_dep}"
random_fusion_job_id="${submitted_job_id}"

pretrained_job_ids=()
for variant in "${pretrained_names[@]}"; do
  output_dir="${HLT_MV_PRETRAINED_DUALVIEW_DIR}/${variant}"
  hlt2_tag="$(hlt_mv_hlt2_tag_from_pretrained_variant "${variant}")"
  hlt2_source_name="$(hlt_mv_hlt2_source_name_for_tag "${hlt2_tag}")"
  hlt2_source_job_id="$(hlt_mv_submitted_job_id_for_source "${hlt2_source_name}")"
  variant_dep="$(join_nonempty_by_colon "${base_dep}" "${canonical_hlt_source_job_id}" "${hlt2_source_job_id}")"
  submit_model_job \
    "hlt_mv_pretrained_${variant}" \
    "${variant_dep}" \
    "${output_dir}" \
    "${variant}" \
    "hlt_mv_pretrained_dualview_report.json" \
    "${SCRIPT_DIR}/run_hlt_mv_train_pretrained_dualview.sh" "${variant}"
  if [[ -n "${submitted_job_id}" ]]; then
    pretrained_job_ids+=("${submitted_job_id}")
  fi
done
pretrained_dep="$(join_nonempty_by_colon "${base_dep}" "${pretrained_job_ids[@]}")"
submit_fusion_job pretrained_dualview_4model "${pretrained_dep}"
pretrained_fusion_job_id="${submitted_job_id}"

scratch_job_ids=()
for variant in "${scratch_names[@]}"; do
  hlt_mv_hlt2_tag_from_scratch_variant "${variant}" >/dev/null
  output_dir="${HLT_MV_SCRATCH_DUALVIEW_DIR}/${variant}"
  submit_model_job \
    "hlt_mv_scratch_${variant}" \
    "${base_dep}" \
    "${output_dir}" \
    "${variant}" \
    "hlt_mv_scratch_dualview_report.json" \
    "${SCRIPT_DIR}/run_hlt_mv_train_scratch_dualview.sh" "${variant}"
  if [[ -n "${submitted_job_id}" ]]; then
    scratch_job_ids+=("${submitted_job_id}")
  fi
done
scratch_dep="$(join_nonempty_by_colon "${base_dep}" "${scratch_job_ids[@]}")"
submit_fusion_job scratch_dualview_4model "${scratch_dep}"
scratch_fusion_job_id="${submitted_job_id}"

control_job_ids=()
same_view_dep="$(join_nonempty_by_colon "${base_dep}" "${canonical_hlt_source_job_id}")"
submit_model_job \
  "hlt_mv_same_view_control" \
  "${same_view_dep}" \
  "${HLT_MV_CONTROLS_DIR}/sdv_hlt_hlt_same_view" \
  "sdv_hlt_hlt_same_view" \
  "" \
  "${SCRIPT_DIR}/run_hlt_mv_train_same_view_control.sh"
if [[ -n "${submitted_job_id}" ]]; then
  control_job_ids+=("${submitted_job_id}")
fi
for strength in "${tta_strengths[@]}"; do
  submit_tta_job "${strength}" "${same_view_dep}"
  if [[ -n "${submitted_job_id}" ]]; then
    control_job_ids+=("${submitted_job_id}")
  fi
done
control_dep="$(join_nonempty_by_colon "${base_dep}" "${control_job_ids[@]}")"

triview_tags=()
mapfile -t triview_tags < <(hlt_mv_triview_tags)
first_triview_source="$(hlt_mv_hlt2_source_name_for_tag "${triview_tags[0]}")"
second_triview_source="$(hlt_mv_hlt2_source_name_for_tag "${triview_tags[1]}")"
triview_dep="$(join_nonempty_by_colon \
  "${base_dep}" \
  "${canonical_hlt_source_job_id}" \
  "$(hlt_mv_submitted_job_id_for_source "${first_triview_source}")" \
  "$(hlt_mv_submitted_job_id_for_source "${second_triview_source}")")"
submit_model_job \
  "hlt_mv_triview" \
  "${triview_dep}" \
  "${HLT_MV_TRIVIEW_DIR}/${HLT_MV_TRIVIEW_MODEL_NAME}" \
  "${HLT_MV_TRIVIEW_MODEL_NAME}" \
  "hlt_mv_triview_report.json" \
  "${SCRIPT_DIR}/run_hlt_mv_train_triview.sh"
triview_job_id="${submitted_job_id}"

report_triview_dep=""
if fresh_bool_enabled "${HLT_MV_FINAL_REPORT_REQUIRE_TRIVIEW}"; then
  report_triview_dep="${triview_job_id}"
fi

final_dep="$(join_nonempty_by_colon \
  "${base_dep}" \
  "${source_fusion_job_id}" \
  "${random_fusion_job_id}" \
  "${pretrained_fusion_job_id}" \
  "${scratch_fusion_job_id}" \
  "${control_dep}" \
  "${report_triview_dep}")"

if ! skip_existing_final_report; then
  refuse_partial_existing_output_dir "hlt_mv_final_report" "${HLT_MV_FINAL_REPORT_DIR}"
  final_report_resource_args=()
  mapfile -t final_report_resource_args < <(sbatch_resource_args cpu)
  mapfile -t args < <(
    afterok_args "${final_dep}" \
      "$(job_export_arg)" \
      "${final_report_resource_args[@]}" \
      "${SCRIPT_DIR}/run_hlt_mv_final_report.sh"
  )
  submit_job "hlt_mv_final_report" "${args[@]}"
  echo "submitted hlt_mv_final_report=${submitted_job_id}"
fi

echo "hlt_mv_submit_complete:"
echo "  submitted_jobs: ${submit_count}"
echo "  skipped_jobs: ${skip_count}"
echo "  hlt_mv_root: ${HLT_MV_ROOT}"
echo "  final_report_dir: ${HLT_MV_FINAL_REPORT_DIR}"
