#!/usr/bin/env bash
# Write the fail-closed Step 7 C2F runtime benchmark report.

#SBATCH --job-name=c2f_rt_report
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

: "${CONSTRAINED_C2F_CALIBRATION_ROOT:?CONSTRAINED_C2F_CALIBRATION_ROOT is required}"
: "${CONSTRAINED_C2F_BENCHMARK_ROOT:=${CONSTRAINED_C2F_CALIBRATION_ROOT}/runtime_benchmarks}"
: "${CONSTRAINED_C2F_BENCHMARK_PLAN:=${CONSTRAINED_C2F_BENCHMARK_ROOT}/benchmark_plan.json}"

fresh_setup "$@"
fresh_require_file "${CONSTRAINED_C2F_BENCHMARK_PLAN}"
fresh_require_dir "${CONSTRAINED_C2F_BENCHMARK_ROOT}/reconstructors"

cmd=(
  "${PYTHON_BIN}" scripts/write_constrained_coarse_to_fine_runtime_benchmark_report.py
  --plan "${CONSTRAINED_C2F_BENCHMARK_PLAN}"
  --benchmark-root "${CONSTRAINED_C2F_BENCHMARK_ROOT}"
  --output "${CONSTRAINED_C2F_BENCHMARK_ROOT}/runtime_benchmark_report.json"
)
fresh_write_run_config "${CONSTRAINED_C2F_BENCHMARK_ROOT}" c2f_runtime_benchmark_report "${cmd[@]}"
fresh_run "${cmd[@]}"

if ! fresh_is_dry_run; then
  fresh_require_file "${CONSTRAINED_C2F_BENCHMARK_ROOT}/runtime_benchmark_report.json"
fi
