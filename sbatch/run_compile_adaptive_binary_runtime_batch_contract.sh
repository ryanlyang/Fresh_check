#!/usr/bin/env bash
# Compile provenance-bound candidate measurements into one immutable contract.

#SBATCH --job-name=abph_batch_contract
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tigris
#SBATCH --account=reu-aisocial
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
source "${PROJECT_DIR}/sbatch/common.sh"
VARIANT="${1:?Usage: compiler <variant>}"
: "${ABPH_ROOT:?Set ABPH_ROOT}"
export PYTHONNOUSERSITE=1
fresh_setup
fresh_run "${PYTHON_BIN}" -u scripts/compile_adaptive_binary_runtime_batch_contract.py \
  --campaign-root "${ABPH_ROOT}" \
  --variant "${VARIANT}" \
  --measurement-dir "${ABPH_ROOT}/runtime_batch_measurements/${VARIANT}/ddp4" \
  --requested-world-size 4
