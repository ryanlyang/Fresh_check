#!/usr/bin/env bash
# Apply the Step 8 C2F benchmark gates and write one immutable candidate profile.

#SBATCH --job-name=c2f_rt_candidate
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

: "${CONSTRAINED_C2F_BENCHMARK_ROOT:?CONSTRAINED_C2F_BENCHMARK_ROOT is required}"
: "${CONSTRAINED_C2F_BENCHMARK_REPORT:=${CONSTRAINED_C2F_BENCHMARK_ROOT}/runtime_benchmark_report.json}"
: "${CONSTRAINED_C2F_ACCELERATED_CANDIDATE_PATH:=${CONSTRAINED_C2F_BENCHMARK_ROOT}/accelerated_candidate_v1.json}"
: "${CONSTRAINED_C2F_GPU_MEMORY_GB:?Set CONSTRAINED_C2F_GPU_MEMORY_GB to the physical GPU memory in GiB}"
: "${CONSTRAINED_C2F_GPU_RESERVED_FRACTION_CAP:=0.80}"

fresh_setup "$@"
fresh_require_file "${CONSTRAINED_C2F_BENCHMARK_REPORT}"
if ! fresh_is_dry_run && [[ -e "${CONSTRAINED_C2F_ACCELERATED_CANDIDATE_PATH}" ]]; then
  echo "Refusing to overwrite immutable candidate artifact: ${CONSTRAINED_C2F_ACCELERATED_CANDIDATE_PATH}" >&2
  exit 2
fi

cmd=(
  "${PYTHON_BIN}" scripts/write_constrained_coarse_to_fine_accelerated_candidate.py
  --benchmark-report "${CONSTRAINED_C2F_BENCHMARK_REPORT}"
  --output "${CONSTRAINED_C2F_ACCELERATED_CANDIDATE_PATH}"
  --gpu-memory-gb "${CONSTRAINED_C2F_GPU_MEMORY_GB}"
  --gpu-reserved-fraction-cap "${CONSTRAINED_C2F_GPU_RESERVED_FRACTION_CAP}"
)
fresh_write_run_config "${CONSTRAINED_C2F_BENCHMARK_ROOT}" c2f_runtime_candidate "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${CONSTRAINED_C2F_ACCELERATED_CANDIDATE_PATH}"
fi
