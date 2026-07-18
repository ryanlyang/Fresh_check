#!/usr/bin/env bash
# Queue the fixed Step-10 one-GPU/DDP4 runtime acceptance matrix on Tigris.

set -euo pipefail
IFS=$'\n\t'
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
: "${CONDA_ENV:=atlas_kd_tigris}"
export PROJECT_DIR CONDA_BASE CONDA_ENV
source "${PROJECT_DIR}/sbatch/common.sh"
fresh_prepare_submitter
fresh_activate_env

: "${ABPH_ROOT:?Set ABPH_ROOT to a prepared pilot campaign root}"
: "${ABPH_RUNTIME_ACCEPTANCE_ROOT:=${ABPH_ROOT}/audits/runtime_acceptance}"
: "${ABPH_SBATCH_ACCOUNT:=reu-aisocial}"
: "${ABPH_SBATCH_PARTITION:=tigris}"
: "${ABPH_RUNTIME_BATCH_CONTRACT_ROOT:=${ABPH_ROOT}/runtime_batch_contracts}"
: "${ABPH_SINGLE_PATH_ACCEPTANCE_PATH:=${ABPH_ROOT}/audits/runtime_reference/single_path_acceptance.json}"
export ABPH_ROOT ABPH_RUNTIME_ACCEPTANCE_ROOT ABPH_RUNTIME_BATCH_CONTRACT_ROOT
export ABPH_SINGLE_PATH_ACCEPTANCE_PATH PYTHONNOUSERSITE=1

for required in \
  "${ABPH_ROOT}/audits/actual_target_feasibility.json" \
  "${ABPH_ROOT}/runs/A0_hlt_part/best_model_val.pt"; do
  fresh_require_file "${required}"
done
contract_paths=(
  "${ABPH_RUNTIME_BATCH_CONTRACT_ROOT}/B1_semantic_query_root/runtime_batch_contract.json"
  "${ABPH_RUNTIME_BATCH_CONTRACT_ROOT}/D1_kt32_mh4_particles/runtime_batch_contract.json"
)
if [[ -z "${ABPH_RUNTIME_BATCH_DEPENDENCY:-}" ]]; then
  for required in "${contract_paths[@]}"; do fresh_require_file "${required}"; done
fi
if ! fresh_is_dry_run && [[ -e "${ABPH_RUNTIME_ACCEPTANCE_ROOT}" ]]; then
  echo "Refusing to replace existing runtime acceptance evidence: ${ABPH_RUNTIME_ACCEPTANCE_ROOT}" >&2
  exit 2
fi
mkdir -p "${ABPH_RUNTIME_ACCEPTANCE_ROOT}"

single_args=(--parsable --account="${ABPH_SBATCH_ACCOUNT}" --partition="${ABPH_SBATCH_PARTITION}"
  --nodes=1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=16 --mem=220G --gres=gpu:gh200:1)
ddp4_args=(--parsable --account="${ABPH_SBATCH_ACCOUNT}" --partition="${ABPH_SBATCH_PARTITION}"
  --nodes=4 --ntasks=4 --ntasks-per-node=1 --cpus-per-task=16 --mem=220G --gres=gpu:gh200:1)
worker="${PROJECT_DIR}/sbatch/run_adaptive_binary_runtime_acceptance.sh"
job_ids=()
declare -A case_job_ids=()
submission_manifest="${ABPH_RUNTIME_ACCEPTANCE_ROOT}/submission.tsv"
printf 'profile\tmode\tvariant\tjob_id\n' > "${submission_manifest}"

submit_case() {
  local profile="$1" mode="$2" variant="${3:-}" submitted job_id
  if [[ "${profile}" == "single" ]]; then
    export ABPH_RECONSTRUCTOR_PARALLELISM=single ABPH_JOB_LAUNCHER=direct
    export ABPH_DISTRIBUTED_NODES=1 ABPH_DISTRIBUTED_NTASKS=1 ABPH_DISTRIBUTED_NTASKS_PER_NODE=1 ABPH_DISTRIBUTED_WORLD_SIZE=1
    args=("${single_args[@]}")
  else
    export ABPH_RECONSTRUCTOR_PARALLELISM=ddp4 ABPH_JOB_LAUNCHER=srun
    export ABPH_DISTRIBUTED_NODES=4 ABPH_DISTRIBUTED_NTASKS=4 ABPH_DISTRIBUTED_NTASKS_PER_NODE=1 ABPH_DISTRIBUTED_WORLD_SIZE=4
    args=("${ddp4_args[@]}")
  fi
  if fresh_is_dry_run; then
    fresh_print_shell_command sbatch "${args[@]}" "${worker}" "${mode}" "${variant}"
    return
  fi
  submitted="$(sbatch "${args[@]}" "${worker}" "${mode}" "${variant}")"
  job_id="${submitted%%;*}"
  [[ "${job_id}" =~ ^[0-9]+$ ]] || { echo "Invalid sbatch response: ${submitted}" >&2; exit 2; }
  printf '%s\t%s\t%s\t%s\n' "${profile}" "${mode}" "${variant}" "${job_id}" >> "${submission_manifest}"
  job_ids+=("${job_id}")
  case_job_ids["${profile}:${mode}:${variant}"]="${job_id}"
  echo "${submitted}"
}

for profile in single ddp4; do
  submit_case "${profile}" smoke
  submit_case "${profile}" benchmark B1_semantic_query_root
  submit_case "${profile}" benchmark D1_kt32_mh4_particles
done
submit_case single benchmark_uninstrumented D1_kt32_mh4_particles

if fresh_is_dry_run; then
  exit 0
fi
single_path_dependency="afterok:${case_job_ids[single:benchmark:D1_kt32_mh4_particles]}:${case_job_ids[single:benchmark_uninstrumented:D1_kt32_mh4_particles]}"
single_path_submitted="$(sbatch --parsable --account="${ABPH_SBATCH_ACCOUNT}" \
  --partition="${ABPH_SBATCH_PARTITION}" --dependency="${single_path_dependency}" \
  "${PROJECT_DIR}/sbatch/run_compile_adaptive_binary_single_path_acceptance.sh")"
single_path_job_id="${single_path_submitted%%;*}"
printf 'single\tcompile_single_path\tD1_kt32_mh4_particles\t%s\n' "${single_path_job_id}" >> "${submission_manifest}"
echo "${single_path_submitted}"

report_dependencies=("${job_ids[@]}" "${single_path_job_id}")
if [[ -n "${ABPH_RUNTIME_BATCH_DEPENDENCY:-}" ]]; then
  IFS=: read -r -a upstream_ids <<< "${ABPH_RUNTIME_BATCH_DEPENDENCY#afterok:}"
  report_dependencies+=("${upstream_ids[@]}")
fi
dependency="afterok:$(IFS=:; echo "${report_dependencies[*]}")"
report_submitted="$(sbatch --parsable --account="${ABPH_SBATCH_ACCOUNT}" \
  --partition="${ABPH_SBATCH_PARTITION}" --dependency="${dependency}" \
  "${PROJECT_DIR}/sbatch/run_write_adaptive_binary_runtime_acceptance.sh")"
report_job_id="${report_submitted%%;*}"
printf 'report\tcompile\tall\t%s\n' "${report_job_id}" >> "${submission_manifest}"
echo "${report_submitted}"
echo "adaptive_binary_runtime_acceptance_submission_complete:"
echo "  root: ${ABPH_RUNTIME_ACCEPTANCE_ROOT}"
echo "  evidence_jobs: ${#job_ids[@]}"
echo "  single_path_job: ${single_path_job_id}"
echo "  report_job: ${report_job_id}"
