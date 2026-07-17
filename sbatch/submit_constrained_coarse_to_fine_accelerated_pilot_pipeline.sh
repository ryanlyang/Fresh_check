#!/usr/bin/env bash
# Submit the dependency-driven C2F accelerated-pilot pipeline without operator polling.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

fresh_prepare_submitter
fresh_activate_env

: "${CONSTRAINED_C2F_PARENT_MANIFEST_PATH:?Set CONSTRAINED_C2F_PARENT_MANIFEST_PATH to the completed parent manifest}"
: "${CONSTRAINED_C2F_PIPELINE_STAMP:=$(date +%Y%m%d_%H%M%S)}"
: "${CONSTRAINED_C2F_CALIBRATION_ROOT:=${OUTPUT_ROOT}/constrained_c2f_runtime_calibration_${CONSTRAINED_C2F_PIPELINE_STAMP}}"
: "${CONSTRAINED_C2F_BENCHMARK_ROOT:=${CONSTRAINED_C2F_CALIBRATION_ROOT}/runtime_benchmarks}"
: "${CONSTRAINED_C2F_ACCELERATED_CANDIDATE_PATH:=${CONSTRAINED_C2F_BENCHMARK_ROOT}/accelerated_candidate_v1.json}"
: "${CONSTRAINED_C2F_ACCELERATED_PILOT_ROOT:=${OUTPUT_ROOT}/constrained_coarse_to_fine_pseudooffline_hltv2_s2p5_pilot_accel_v1_${CONSTRAINED_C2F_PIPELINE_STAMP}}"
: "${CONSTRAINED_C2F_RUNTIME_PIPELINE_SUBMISSION:=${OUTPUT_ROOT}/constrained_c2f_runtime_pipeline_${CONSTRAINED_C2F_PIPELINE_STAMP}.tsv}"
: "${CONSTRAINED_C2F_GPU_MEMORY_GB:?Set CONSTRAINED_C2F_GPU_MEMORY_GB to the physical GPU memory in GiB}"
: "${CONSTRAINED_C2F_SBATCH_ACCOUNT:=}"
: "${CONSTRAINED_C2F_SBATCH_PARTITION:=}"
: "${CONSTRAINED_C2F_CALIBRATION_CPUS:=16}"
: "${CONSTRAINED_C2F_CALIBRATION_MEM:=300G}"

fresh_require_file "${CONSTRAINED_C2F_PARENT_MANIFEST_PATH}"
if ! fresh_is_dry_run && [[ -e "${CONSTRAINED_C2F_CALIBRATION_ROOT}" ]]; then
  echo "Refusing to reuse an existing calibration root: ${CONSTRAINED_C2F_CALIBRATION_ROOT}" >&2
  exit 2
fi
if ! fresh_is_dry_run && [[ -e "${CONSTRAINED_C2F_ACCELERATED_PILOT_ROOT}" ]]; then
  echo "Refusing to reuse an existing accelerated pilot root: ${CONSTRAINED_C2F_ACCELERATED_PILOT_ROOT}" >&2
  exit 2
fi

export CONSTRAINED_C2F_PIPELINE_STAMP
export CONSTRAINED_C2F_CALIBRATION_ROOT CONSTRAINED_C2F_BENCHMARK_ROOT
export CONSTRAINED_C2F_ACCELERATED_CANDIDATE_PATH CONSTRAINED_C2F_ACCELERATED_PILOT_ROOT
export CONSTRAINED_C2F_RUNTIME_PIPELINE_SUBMISSION CONSTRAINED_C2F_GPU_MEMORY_GB
export CONSTRAINED_C2F_PARENT_MANIFEST_PATH CONSTRAINED_C2F_SBATCH_ACCOUNT CONSTRAINED_C2F_SBATCH_PARTITION
export CONSTRAINED_C2F_CALIBRATION_CPUS CONSTRAINED_C2F_CALIBRATION_MEM

if fresh_is_dry_run; then
  echo "Would submit calibration, then a dependency controller that submits benchmarks, the candidate, and the accelerated pilot."
  exit 0
fi

calibration_output="$(bash "${SCRIPT_DIR}/submit_constrained_coarse_to_fine_runtime_calibration.sh")"
printf '%s\n' "${calibration_output}"
calibration_job_id="$(printf '%s\n' "${calibration_output}" | awk '/Submitted batch job/ {print $NF; exit}')"
[[ "${calibration_job_id}" =~ ^[0-9]+$ ]] || { echo "Could not parse calibration job id" >&2; exit 2; }

printf 'stage\tjob_id\ncalibration\t%s\n' "${calibration_job_id}" > "${CONSTRAINED_C2F_RUNTIME_PIPELINE_SUBMISSION}"

controller_args=(--cpus-per-task=4 --mem=32G --dependency="afterok:${calibration_job_id}")
[[ -n "${CONSTRAINED_C2F_SBATCH_ACCOUNT}" ]] && controller_args+=(--account="${CONSTRAINED_C2F_SBATCH_ACCOUNT}")
[[ -n "${CONSTRAINED_C2F_SBATCH_PARTITION}" ]] && controller_args+=(--partition="${CONSTRAINED_C2F_SBATCH_PARTITION}")
controller_output="$(sbatch "${controller_args[@]}" "${SCRIPT_DIR}/run_submit_constrained_coarse_to_fine_runtime_benchmarks_and_candidate.sh")"
printf '%s\n' "${controller_output}"
controller_job_id="${controller_output##* }"
[[ "${controller_job_id}" =~ ^[0-9]+$ ]] || { echo "Could not parse benchmark-controller job id" >&2; exit 2; }
printf 'benchmark_candidate_controller\t%s\n' "${controller_job_id}" >> "${CONSTRAINED_C2F_RUNTIME_PIPELINE_SUBMISSION}"

cat <<SUMMARY
constrained_c2f_accelerated_pilot_pipeline_submitted:
  calibration_root: ${CONSTRAINED_C2F_CALIBRATION_ROOT}
  pilot_root: ${CONSTRAINED_C2F_ACCELERATED_PILOT_ROOT}
  calibration_job: ${calibration_job_id}
  controller_job: ${controller_job_id}
  submission: ${CONSTRAINED_C2F_RUNTIME_PIPELINE_SUBMISSION}
SUMMARY
