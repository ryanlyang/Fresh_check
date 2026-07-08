#!/usr/bin/env bash
# Build the missing heavy HLT2 caches and submit the HLT-MV source/fusion graph.

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
: "${HLT_MV_ROOT:=${HLT_MV_PDV3_ROOT}/hlt_multiview_source_fusion_heavy_$(date +%Y%m%d_%H%M%S)}"
: "${HLT_MV_ROOT_DIRNAME:=$(basename "${HLT_MV_ROOT}")}"
: "${HLT_MV_HLT_CACHE_DIR:=${HLT_MV_PDV3_ROOT}/inputs/hlt_cache}"
: "${HLT_MV_REUSE_HLT2_CACHE_ROOT:=${HLT_MV_PDV3_ROOT}/hlt_self_dualview_20260707_054638/hlt2_cache}"
: "${HLT_MV_HLT2_CACHE_ROOT:=${HLT_MV_ROOT}/hlt2_cache}"
: "${HLT_MV_HEAVY_BUILD_STRENGTHS:=0.50 1.50 2.00}"
: "${HLT_MV_HEAVY_REUSE_STRENGTHS:=1.00}"
: "${HLT_MV_STRENGTHS:=0.50 1.00 1.50 2.00}"
: "${HLT_MV_HLT2_SOURCE_SEEDS:=0.50=8851 1.00=8841 1.50=8861 2.00=8871}"
: "${HLT_MV_CANONICAL_HLT_SOURCE_NAME:=hlt_part_seed8801}"
: "${HLT_MV_SOURCE_NAMES:=hlt_part_seed8801 hlt2_part_s0p50_seed8851 hlt2_part_s1p00_seed8841 hlt2_part_s1p50_seed8861 hlt2_part_s2p00_seed8871}"
: "${HLT_MV_RANDOM_HLT_SOURCE_NAMES:=hlt_part_seed9101 hlt_part_seed9102 hlt_part_seed9103 hlt_part_seed9104}"
: "${HLT_MV_PRETRAINED_DUALVIEW_NAMES:=sdv_hlt_hlt2_s0p50 sdv_hlt_hlt2_s1p00 sdv_hlt_hlt2_s1p50 sdv_hlt_hlt2_s2p00}"
: "${HLT_MV_SCRATCH_DUALVIEW_NAMES:=sdv_hlt_hlt2_s0p50_scratch sdv_hlt_hlt2_s1p00_scratch sdv_hlt_hlt2_s1p50_scratch sdv_hlt_hlt2_s2p00_scratch}"
: "${HLT_MV_TTA_STRENGTHS:=0.50 1.00 1.50 2.00}"
: "${HLT_MV_TRIVIEW_MODEL_NAME:=tri_hlt_hlt2_s1p00_s2p00}"
: "${HLT_MV_HEAVY_CACHE_AUDIT_DIR:=${HLT_MV_ROOT}/audits/hlt2_cache_builds}"
: "${HLT_MV_HEAVY_CACHE_SEED:=710053}"
: "${HLT_MV_HEAVY_SUBMIT_GRAPH:=1}"
: "${HLT_MV_SBATCH_PARTITION:=}"
: "${HLT_MV_GPU_GRES:=}"
: "${HLT_MV_GPU_CPUS_PER_TASK:=}"
: "${HLT_MV_GPU_MEM:=}"
: "${HLT_MV_GPU_TIME:=}"
: "${HLT_MV_CPU_CPUS_PER_TASK:=}"
: "${HLT_MV_CPU_MEM:=}"
: "${HLT_MV_CPU_TIME:=}"

export PROJECT_DIR DATA_DIR OUTPUT_ROOT DIAGNOSTICS_ROOT LOG_DIR CONDA_BASE CONDA_ENV PYTHON_BIN
export HLT_MV_PDV3_EXPERIMENT_NAME HLT_MV_PDV3_ROOT HLT_MV_ROOT HLT_MV_ROOT_DIRNAME
export HLT_MV_HLT_CACHE_DIR HLT_MV_HLT2_CACHE_ROOT HLT_MV_REUSE_HLT2_CACHE_ROOT
export HLT_MV_STRENGTHS HLT_MV_HLT2_SOURCE_SEEDS HLT_MV_CANONICAL_HLT_SOURCE_NAME
export HLT_MV_SOURCE_NAMES HLT_MV_RANDOM_HLT_SOURCE_NAMES HLT_MV_PRETRAINED_DUALVIEW_NAMES
export HLT_MV_SCRATCH_DUALVIEW_NAMES HLT_MV_TTA_STRENGTHS HLT_MV_TRIVIEW_MODEL_NAME
export HLT_MV_SBATCH_PARTITION HLT_MV_GPU_GRES HLT_MV_GPU_CPUS_PER_TASK HLT_MV_GPU_MEM HLT_MV_GPU_TIME
export HLT_MV_CPU_CPUS_PER_TASK HLT_MV_CPU_MEM HLT_MV_CPU_TIME
export CONFIRM_FINAL_TEST SKIP_EXISTING OVERWRITE DEVICE DRY_RUN PRINT_ONLY PYTHONNOUSERSITE

fresh_prepare_submitter

if ! fresh_bool_enabled "${CONFIRM_FINAL_TEST}"; then
  echo "Refusing to submit HLT-MV heavy final-test graph without CONFIRM_FINAL_TEST=1." >&2
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

hlt2_cache_dir_for_strength() {
  local strength="$1"
  local tag
  tag="$(fresh_pd10_hlt_sdv_strength_tag "${strength}")"
  printf '%s/hlt_second_degrade_mild_v1_%s\n' "${HLT_MV_HLT2_CACHE_ROOT}" "${tag}"
}

source_hlt2_cache_dir_for_strength() {
  local strength="$1"
  local tag
  tag="$(fresh_pd10_hlt_sdv_strength_tag "${strength}")"
  printf '%s/hlt_second_degrade_mild_v1_%s\n' "${HLT_MV_REUSE_HLT2_CACHE_ROOT}" "${tag}"
}

cache_complete() {
  local cache_dir="$1"
  local split
  [[ -d "${cache_dir}" ]] || return 1
  for split in model_train model_val final_test; do
    [[ -f "${cache_dir}/${split}_fixed_hlt.npz" ]] || return 1
    [[ -f "${cache_dir}/${split}_fixed_hlt_metadata.json" ]] || return 1
  done
}

require_complete_cache() {
  local label="$1"
  local cache_dir="$2"
  if ! cache_complete "${cache_dir}"; then
    echo "Required complete ${label} cache is missing or incomplete:" >&2
    echo "  ${cache_dir}" >&2
    return 2
  fi
}

reuse_hlt2_cache() {
  local strength="$1"
  local source_dir
  local target_dir
  source_dir="$(source_hlt2_cache_dir_for_strength "${strength}")"
  target_dir="$(hlt2_cache_dir_for_strength "${strength}")"
  require_complete_cache "reusable HLT2 strength ${strength}" "${source_dir}"
  if cache_complete "${target_dir}"; then
    echo "reusing existing heavy HLT2 cache target for strength ${strength}: ${target_dir}" >&2
    return 0
  fi
  if [[ -e "${target_dir}" || -L "${target_dir}" ]]; then
    echo "Refusing to reuse HLT2 cache; target exists but is incomplete:" >&2
    echo "  ${target_dir}" >&2
    echo "Remove it or set HLT_MV_HLT2_CACHE_ROOT to a clean location." >&2
    return 2
  fi
  if fresh_is_dry_run; then
    echo "DRY_RUN ln -s ${source_dir} ${target_dir}" >&2
    return 0
  fi
  mkdir -p "$(dirname "${target_dir}")"
  ln -s "${source_dir}" "${target_dir}"
  echo "linked reusable HLT2 cache strength ${strength}: ${target_dir} -> ${source_dir}" >&2
}

submit_cache_job() {
  local strength="$1"
  local cache_dir
  local tag
  cache_dir="$(hlt2_cache_dir_for_strength "${strength}")"
  tag="$(fresh_pd10_hlt_sdv_strength_tag "${strength}")"
  if cache_complete "${cache_dir}" && ! fresh_bool_enabled "${OVERWRITE}"; then
    echo "skipped heavy HLT2 cache ${tag}; found complete cache" >&2
    submitted_cache_job_id=""
    return 0
  fi
  if [[ -e "${cache_dir}" || -L "${cache_dir}" ]]; then
    if ! fresh_bool_enabled "${OVERWRITE}"; then
      echo "Refusing to submit heavy HLT2 cache ${tag}; found incomplete existing path:" >&2
      echo "  ${cache_dir}" >&2
      echo "Use OVERWRITE=1 to repair or remove the partial output." >&2
      return 2
    fi
  fi
  local cache_resource_args=()
  mapfile -t cache_resource_args < <(sbatch_resource_args cpu)
  if fresh_is_dry_run; then
    printf 'DRY_RUN sbatch hlt_mv_heavy_cache_%s: ' "${tag}" >&2
    fresh_print_shell_command sbatch --parsable --export=ALL "${cache_resource_args[@]}" "${SCRIPT_DIR}/run_pd10_build_hlt2_cache.sh" "${strength}" >&2
    printf '\n' >&2
    submitted_cache_job_id="DRYRUN_hlt_mv_heavy_cache_${tag}"
    return 0
  fi
  submitted_cache_job_id="$(
    PD10_ROOT="${HLT_MV_PDV3_ROOT}" \
    PD10_MANIFEST_PATH="${HLT_MV_PDV3_ROOT}/inputs/split_manifest/split_manifest.json.gz" \
    PD10_HLT_CACHE_DIR="${HLT_MV_HLT_CACHE_DIR}" \
    PD10_HLT_SDV_HLT2_CACHE_ROOT="${HLT_MV_HLT2_CACHE_ROOT}" \
    PD10_HLT_SDV_AUDIT_DIR="${HLT_MV_HEAVY_CACHE_AUDIT_DIR}" \
    PD10_HLT_SDV_HLT2_SEED="${HLT_MV_HEAVY_CACHE_SEED}" \
    PD10_MODEL_TRAIN_SIZE=5000000 \
    PD10_MODEL_VAL_SIZE=1000000 \
    PD10_FINAL_TEST_SIZE=1000000 \
    PD10_HLT_SPLITS="model_train model_val final_test" \
    sbatch --parsable --export=ALL "${cache_resource_args[@]}" "${SCRIPT_DIR}/run_pd10_build_hlt2_cache.sh" "${strength}"
  )"
  if ! dependency_token_is_valid "${submitted_cache_job_id}"; then
    echo "Failed to submit heavy HLT2 cache ${tag}; got '${submitted_cache_job_id:-empty}'." >&2
    return 2
  fi
  echo "submitted hlt_mv_heavy_cache_${tag}=${submitted_cache_job_id}" >&2
}

validate_dependency_list "HLT_MV_UPSTREAM_DEPENDENCY" "${HLT_MV_UPSTREAM_DEPENDENCY}"
fresh_require_dir "${HLT_MV_HLT_CACHE_DIR}"
for split in model_train model_val final_test; do
  fresh_require_file "${HLT_MV_HLT_CACHE_DIR}/${split}_fixed_hlt_metadata.json"
done

reuse_strengths=()
build_strengths=()
cache_job_ids=()
fresh_split_words reuse_strengths "${HLT_MV_HEAVY_REUSE_STRENGTHS}"
fresh_split_words build_strengths "${HLT_MV_HEAVY_BUILD_STRENGTHS}"

for strength in "${reuse_strengths[@]}"; do
  reuse_hlt2_cache "${strength}"
done

submitted_cache_job_id=""
for strength in "${build_strengths[@]}"; do
  submit_cache_job "${strength}"
  if [[ -n "${submitted_cache_job_id}" ]]; then
    cache_job_ids+=("${submitted_cache_job_id}")
  fi
done

cache_dep="$(join_nonempty_by_colon "${cache_job_ids[@]}")"
graph_dep="$(join_nonempty_by_colon "${HLT_MV_UPSTREAM_DEPENDENCY}" "${cache_dep}")"

echo "hlt_mv_heavy_cache_submit_complete:"
echo "  hlt_mv_root: ${HLT_MV_ROOT}"
echo "  hlt2_cache_root: ${HLT_MV_HLT2_CACHE_ROOT}"
echo "  reuse_hlt2_cache_root: ${HLT_MV_REUSE_HLT2_CACHE_ROOT}"
echo "  cache_dependency: ${cache_dep}"
echo "  graph_dependency: ${graph_dep}"

if fresh_bool_enabled "${HLT_MV_HEAVY_SUBMIT_GRAPH}"; then
  HLT_MV_UPSTREAM_DEPENDENCY="${graph_dep}" \
  HLT_MV_ALLOW_PENDING_HLT2_CACHES="$([[ -n "${cache_dep}" ]] && echo 1 || echo 0)" \
  bash "${SCRIPT_DIR}/submit_hlt_mv_source_fusion.sh"
else
  echo "HLT_MV_HEAVY_SUBMIT_GRAPH=0; cache jobs submitted but graph submission skipped."
fi
