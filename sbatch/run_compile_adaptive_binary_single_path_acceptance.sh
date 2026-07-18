#!/usr/bin/env bash
# Compile measured single-rank profiler overhead and parity evidence.

#SBATCH --job-name=abph_single_path_gate
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tigris
#SBATCH --account=reu-aisocial
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

set -euo pipefail
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
source "${PROJECT_DIR}/sbatch/common.sh"
: "${ABPH_ROOT:?Set ABPH_ROOT}"
: "${ABPH_RUNTIME_ACCEPTANCE_ROOT:=${ABPH_ROOT}/audits/runtime_acceptance}"
: "${ABPH_SINGLE_PATH_ACCEPTANCE_PATH:=${ABPH_ROOT}/audits/runtime_reference/single_path_acceptance.json}"
export PYTHONNOUSERSITE=1
fresh_setup

fresh_run "${PYTHON_BIN}" -u scripts/compile_adaptive_binary_bootstrap_single_path_acceptance.py \
  --uninstrumented-run "${ABPH_RUNTIME_ACCEPTANCE_ROOT}/single_path/uninstrumented/D1_kt32_mh4_particles" \
  --instrumented-run "${ABPH_RUNTIME_ACCEPTANCE_ROOT}/single/benchmarks/D1_kt32_mh4_particles" \
  --output "${ABPH_SINGLE_PATH_ACCEPTANCE_PATH}"
