#!/usr/bin/env bash
# Queue the fixed Step-10 one-GPU/DDP4 runtime acceptance matrix on Tigris.

set -euo pipefail
IFS=$'\n\t'
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
source "${PROJECT_DIR}/sbatch/common.sh"
fresh_prepare_submitter
fresh_activate_env

: "${ABPH_ROOT:?Set ABPH_ROOT to a prepared pilot campaign root}"
: "${ABPH_RUNTIME_ACCEPTANCE_ROOT:=${ABPH_ROOT}/audits/runtime_acceptance}"
: "${ABPH_SBATCH_ACCOUNT:=reu-aisocial}"
: "${ABPH_SBATCH_PARTITION:=tigris}"
export ABPH_ROOT ABPH_RUNTIME_ACCEPTANCE_ROOT PYTHONNOUSERSITE=1

for required in \
  "${ABPH_ROOT}/audits/actual_target_feasibility.json" \
  "${ABPH_SINGLE_PATH_ACCEPTANCE_PATH:-${ABPH_ROOT}/audits/runtime_reference/single_path_acceptance.json}" \
  "${ABPH_ROOT}/runs/A0_hlt_part/best_model_val.pt" \
  "${ABPH_ROOT}/runs/C5_kt_32/best_model_val.pt" \
  "${ABPH_ROOT}/runtime_batch_contracts/B1_semantic_query_root/runtime_batch_contract.json" \
  "${ABPH_ROOT}/runtime_batch_contracts/D1_kt32_mh4_particles/runtime_batch_contract.json"; do
  fresh_require_file "${required}"
done
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
  echo "${submitted}"
}

for profile in single ddp4; do
  submit_case "${profile}" smoke
  submit_case "${profile}" benchmark B1_semantic_query_root
  submit_case "${profile}" benchmark D1_kt32_mh4_particles
done

if fresh_is_dry_run; then
  exit 0
fi
dependency="afterok:$(IFS=:; echo "${job_ids[*]}")"
report_submitted="$(sbatch --parsable --account="${ABPH_SBATCH_ACCOUNT}" \
  --partition="${ABPH_SBATCH_PARTITION}" --dependency="${dependency}" \
  "${PROJECT_DIR}/sbatch/run_write_adaptive_binary_runtime_acceptance.sh")"
report_job_id="${report_submitted%%;*}"
printf 'report\tcompile\tall\t%s\n' "${report_job_id}" >> "${submission_manifest}"
echo "${report_submitted}"
echo "adaptive_binary_runtime_acceptance_submission_complete:"
echo "  root: ${ABPH_RUNTIME_ACCEPTANCE_ROOT}"
echo "  evidence_jobs: ${#job_ids[@]}"
echo "  report_job: ${report_job_id}"
