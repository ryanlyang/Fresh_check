#!/usr/bin/env bash
# Approval-gated cleanup before the fresh storage-constrained pilot starts.

#SBATCH --job-name=abph_bootstrap_prune
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tigris
#SBATCH --account=reu-aisocial
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

set -euo pipefail
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
CONDA_BASE="${ABPH_CONDA_BASE:-/home/ryreu/miniforge3-aarch64}"
CONDA_ENV="${ABPH_CONDA_ENV:-atlas_kd_tigris}"
export CONDA_BASE CONDA_ENV
source "${PROJECT_DIR}/sbatch/common.sh"
: "${ABPH_PREPARED_ROOT:?Set the prepared campaign root}"
: "${ABPH_BOOTSTRAP_EVIDENCE_ROOT:?Set the retained bootstrap evidence root}"
: "${ABPH_BOOTSTRAP_RUNTIME_ACCEPTANCE:?Set the runtime acceptance path}"
[[ "${ABPH_APPROVE_PREPARED_ROOT_PRUNE:-0}" == "1" ]] || {
  echo "Prepared-root pruning was not explicitly approved" >&2
  exit 2
}
[[ "${ABPH_CONFIRM_PREPARED_ROOT_IDLE:-0}" == "1" ]] || {
  echo "Prepared root was not explicitly confirmed idle" >&2
  exit 2
}
export PYTHONNOUSERSITE=1
fresh_setup

fresh_run "${PYTHON_BIN}" -u scripts/prune_adaptive_binary_prepared_root.py \
  --prepared-root "${ABPH_PREPARED_ROOT}" \
  --bootstrap-evidence-root "${ABPH_BOOTSTRAP_EVIDENCE_ROOT}" \
  --runtime-acceptance "${ABPH_BOOTSTRAP_RUNTIME_ACCEPTANCE}" \
  --maximum-retained-bytes 5000000000 \
  --approve-prune \
  --output "${ABPH_BOOTSTRAP_EVIDENCE_ROOT}/prepared_root_prune_receipt.json"
