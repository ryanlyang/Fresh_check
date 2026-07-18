#!/usr/bin/env bash
# Compile Step-10 evidence; exit nonzero unless the DDP4 runtime gate passes.

#SBATCH --job-name=abph_runtime_gate
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tigris
#SBATCH --account=reu-aisocial
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

set -euo pipefail
IFS=$'\n\t'
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
source "${PROJECT_DIR}/sbatch/common.sh"
: "${ABPH_ROOT:?Set ABPH_ROOT}"
: "${ABPH_RUNTIME_ACCEPTANCE_ROOT:=${ABPH_ROOT}/audits/runtime_acceptance}"
export PYTHONNOUSERSITE=1
fresh_setup

cmd=("${PYTHON_BIN}" -u scripts/write_adaptive_binary_runtime_acceptance.py
  --single-root-run "${ABPH_RUNTIME_ACCEPTANCE_ROOT}/single/benchmarks/B1_semantic_query_root"
  --single-deep-run "${ABPH_RUNTIME_ACCEPTANCE_ROOT}/single/benchmarks/D1_kt32_mh4_particles"
  --ddp4-root-run "${ABPH_RUNTIME_ACCEPTANCE_ROOT}/ddp4/benchmarks/B1_semantic_query_root"
  --ddp4-deep-run "${ABPH_RUNTIME_ACCEPTANCE_ROOT}/ddp4/benchmarks/D1_kt32_mh4_particles"
  --single-smoke "${ABPH_RUNTIME_ACCEPTANCE_ROOT}/single/transport_smoke/smoke_report.json"
  --ddp4-smoke "${ABPH_RUNTIME_ACCEPTANCE_ROOT}/ddp4/transport_smoke/smoke_report.json"
  --ddp4-root-batch-contract "${ABPH_ROOT}/runtime_batch_contracts/B1_semantic_query_root/runtime_batch_contract.json"
  --ddp4-deep-batch-contract "${ABPH_ROOT}/runtime_batch_contracts/D1_kt32_mh4_particles/runtime_batch_contract.json"
  --single-path-acceptance "${ABPH_SINGLE_PATH_ACCEPTANCE_PATH:-${ABPH_ROOT}/audits/runtime_reference/single_path_acceptance.json}"
  --expected-validation-jets "${ABPH_EXPECTED_MODEL_VAL_JETS:-150000}"
  --output "${ABPH_RUNTIME_ACCEPTANCE_ROOT}/runtime_acceptance.json")

if [[ -n "${ABPH_ROOT_EXTENSION_REPORT:-}" || -n "${ABPH_DEEP_EXTENSION_REPORT:-}" ]]; then
  : "${ABPH_ROOT_EXTENSION_REPORT:?Both extension reports are required}"
  : "${ABPH_DEEP_EXTENSION_REPORT:?Both extension reports are required}"
  cmd+=(--root-extension-report "${ABPH_ROOT_EXTENSION_REPORT}")
  cmd+=(--deep-extension-report "${ABPH_DEEP_EXTENSION_REPORT}")
fi
if [[ -n "${ABPH_OPTIMIZED_PILOT_REPORT:-}" ]]; then
  cmd+=(--optimized-pilot-report "${ABPH_OPTIMIZED_PILOT_REPORT}")
fi
fresh_run "${cmd[@]}"
