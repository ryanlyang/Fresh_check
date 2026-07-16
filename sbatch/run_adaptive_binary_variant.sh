#!/usr/bin/env bash
# Run one baseline/reconstructor/renderer/tagger variant via the production executor.

#SBATCH --job-name=abph_variant
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=3-00:00:00
#SBATCH --mem=300G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16

set -euo pipefail
IFS=$'\n\t'
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
source "${PROJECT_DIR}/sbatch/common.sh"
VARIANT="${1:?Usage: run_adaptive_binary_variant.sh <registry-variant>}"
SEED_INDEX="${2:-1}"
: "${ABPH_ROOT:=${OUTPUT_ROOT}/adaptive_binary_pseudooffline}"
: "${ABPH_VARIANT_EXECUTOR:=${PROJECT_DIR}/scripts/train_adaptive_binary_pseudooffline_variant.py}"
export PYTHONNOUSERSITE=1
fresh_setup
fresh_require_file "${ABPH_VARIANT_EXECUTOR}"
if [[ "${VARIANT}" =~ ^[BCD] ]]; then
  "${PYTHON_BIN}" scripts/validate_adaptive_binary_orchestration.py preflight \
    --path "${ABPH_ROOT}/audits/actual_target_feasibility.json"
fi
cmd=("${PYTHON_BIN}" -u "${ABPH_VARIANT_EXECUTOR}" --variant "${VARIANT}" --seed-index "${SEED_INDEX}" --campaign-root "${ABPH_ROOT}" --device "${DEVICE}")
fresh_run "${cmd[@]}"
