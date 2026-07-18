#!/usr/bin/env bash
# Compile the immutable E7/F0 tagger DDP promotion artifact.
#SBATCH --job-name=abph_tag_gate
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tigris
#SBATCH --time=00:20:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2

set -euo pipefail
IFS=$'\n\t'
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
source "${PROJECT_DIR}/sbatch/common.sh"
: "${ABPH_ROOT:?ABPH_ROOT is required}"
: "${ABPH_TAGGER_DDP_EVIDENCE_ROOT:=${ABPH_ROOT}/tagger_ddp_acceptance/ddp4}"
: "${ABPH_TAGGER_DDP_ACCEPTANCE_PATH:=${ABPH_ROOT}/tagger_ddp_acceptance/tagger_ddp_acceptance.json}"
export PYTHONNOUSERSITE=1
fresh_setup
fresh_run "${PYTHON_BIN}" -u scripts/write_adaptive_binary_tagger_ddp_acceptance.py \
  --single-root "${ABPH_ROOT}" \
  --ddp4-root "${ABPH_TAGGER_DDP_EVIDENCE_ROOT}" \
  --output "${ABPH_TAGGER_DDP_ACCEPTANCE_PATH}"
