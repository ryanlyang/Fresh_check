#!/usr/bin/env bash
# Evaluate non-selection pseudo/hierarchy/fusion ablations.

#SBATCH --job-name=abph_diagnostics
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=2-00:00:00
#SBATCH --mem=220G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16

set -euo pipefail
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
source "${PROJECT_DIR}/sbatch/common.sh"
: "${ABPH_ROOT:=${OUTPUT_ROOT}/adaptive_binary_pseudooffline}"
: "${ABPH_DIAGNOSTIC_EXECUTOR:=${PROJECT_DIR}/scripts/diagnose_adaptive_binary_pseudooffline.py}"
export PYTHONNOUSERSITE=1
fresh_setup
fresh_require_file "${ABPH_DIAGNOSTIC_EXECUTOR}"
cmd=("${PYTHON_BIN}" -u "${ABPH_DIAGNOSTIC_EXECUTOR}" --campaign-root "${ABPH_ROOT}" --variants "$@" --split model_val --device "${DEVICE}")
fresh_run "${cmd[@]}"
