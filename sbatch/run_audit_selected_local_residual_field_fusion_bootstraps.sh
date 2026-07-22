#!/usr/bin/env bash
# Verify final paired bootstrap coverage and hashes without reopening data.
#SBATCH --job-name=lprf_fuse_boot
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tigris
#SBATCH --account=reu-aisocial
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
set -euo pipefail
IFS=$'\n\t'
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
export PYTHONNOUSERSITE=1
source "${PROJECT_DIR}/sbatch/common.sh"
: "${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT:?campaign root required}"
fresh_setup "$@"
fresh_run "${PYTHON_BIN}" -u scripts/audit_selected_local_residual_field_fusion_bootstraps.py --selected-fusion "${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT}/selection/selected_fusion.json"
