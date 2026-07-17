#!/usr/bin/env bash
# After calibration succeeds, fan out runtime benchmarks and submit candidate/pilot handoffs.

#SBATCH --job-name=c2f_rt_pipe
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${CONSTRAINED_C2F_CALIBRATION_ROOT:?CONSTRAINED_C2F_CALIBRATION_ROOT is required}"
: "${CONSTRAINED_C2F_BENCHMARK_ROOT:=${CONSTRAINED_C2F_CALIBRATION_ROOT}/runtime_benchmarks}"
: "${CONSTRAINED_C2F_ACCELERATED_CANDIDATE_PATH:=${CONSTRAINED_C2F_BENCHMARK_ROOT}/accelerated_candidate_v1.json}"
: "${CONSTRAINED_C2F_ACCELERATED_PILOT_ROOT:?CONSTRAINED_C2F_ACCELERATED_PILOT_ROOT is required}"
: "${CONSTRAINED_C2F_RUNTIME_PIPELINE_SUBMISSION:?CONSTRAINED_C2F_RUNTIME_PIPELINE_SUBMISSION is required}"
: "${CONSTRAINED_C2F_GPU_MEMORY_GB:?CONSTRAINED_C2F_GPU_MEMORY_GB is required}"
: "${CONSTRAINED_C2F_SBATCH_ACCOUNT:=}"
: "${CONSTRAINED_C2F_SBATCH_PARTITION:=}"

fresh_setup "$@"
fresh_require_file "${CONSTRAINED_C2F_CALIBRATION_ROOT}/calibration_manifest.json.gz"
fresh_require_file "${CONSTRAINED_C2F_CALIBRATION_ROOT}/targets/hierarchy_target_cache_manifest.json"

export CONSTRAINED_C2F_BENCHMARK_ROOT CONSTRAINED_C2F_ACCELERATED_CANDIDATE_PATH
bash "${SCRIPT_DIR}/submit_constrained_coarse_to_fine_runtime_benchmarks.sh"

benchmark_submission="${CONSTRAINED_C2F_BENCHMARK_ROOT}/benchmark_submission.tsv"
fresh_require_file "${benchmark_submission}"
benchmark_report_job_id="$(awk -F $'\t' '$1 == "report" && $2 == "job_id" {print $3; exit}' "${benchmark_submission}")"
[[ "${benchmark_report_job_id}" =~ ^[0-9]+$ ]] || { echo "Could not parse runtime benchmark report job id" >&2; exit 2; }
printf 'benchmark_report\t%s\n' "${benchmark_report_job_id}" >> "${CONSTRAINED_C2F_RUNTIME_PIPELINE_SUBMISSION}"

candidate_args=(--cpus-per-task=4 --mem=32G --dependency="afterok:${benchmark_report_job_id}")
[[ -n "${CONSTRAINED_C2F_SBATCH_ACCOUNT}" ]] && candidate_args+=(--account="${CONSTRAINED_C2F_SBATCH_ACCOUNT}")
[[ -n "${CONSTRAINED_C2F_SBATCH_PARTITION}" ]] && candidate_args+=(--partition="${CONSTRAINED_C2F_SBATCH_PARTITION}")
candidate_output="$(sbatch "${candidate_args[@]}" "${SCRIPT_DIR}/run_write_constrained_coarse_to_fine_accelerated_candidate.sh")"
printf '%s\n' "${candidate_output}"
candidate_job_id="${candidate_output##* }"
[[ "${candidate_job_id}" =~ ^[0-9]+$ ]] || { echo "Could not parse candidate job id" >&2; exit 2; }
printf 'candidate\t%s\n' "${candidate_job_id}" >> "${CONSTRAINED_C2F_RUNTIME_PIPELINE_SUBMISSION}"

pilot_args=(--cpus-per-task=4 --mem=32G --dependency="afterok:${candidate_job_id}")
[[ -n "${CONSTRAINED_C2F_SBATCH_ACCOUNT}" ]] && pilot_args+=(--account="${CONSTRAINED_C2F_SBATCH_ACCOUNT}")
[[ -n "${CONSTRAINED_C2F_SBATCH_PARTITION}" ]] && pilot_args+=(--partition="${CONSTRAINED_C2F_SBATCH_PARTITION}")
pilot_output="$(sbatch "${pilot_args[@]}" "${SCRIPT_DIR}/run_submit_constrained_coarse_to_fine_accelerated_pilot.sh")"
printf '%s\n' "${pilot_output}"
pilot_job_id="${pilot_output##* }"
[[ "${pilot_job_id}" =~ ^[0-9]+$ ]] || { echo "Could not parse pilot-controller job id" >&2; exit 2; }
printf 'pilot_controller\t%s\n' "${pilot_job_id}" >> "${CONSTRAINED_C2F_RUNTIME_PIPELINE_SUBMISSION}"
