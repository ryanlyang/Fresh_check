#!/usr/bin/env bash
# Tigris defaults for the dependency-safe P7b fusion campaign submitter.
set -euo pipefail
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${OUTPUT_ROOT:=${PROJECT_DIR}/checkpoints}"
: "${DIAGNOSTICS_ROOT:=${PROJECT_DIR}/fresh_check_diagnostics}"
: "${LOG_DIR:=${PROJECT_DIR}/fresh_check_logs}"
: "${CONDA_BASE:=/home/ryreu/miniconda3}"
: "${CONDA_ENV:=weaver}"
: "${LPRF_FUSION_SBATCH_ACCOUNT:=reu-aisocial}"
: "${LPRF_FUSION_SBATCH_PARTITION:=tigris}"
export PROJECT_DIR OUTPUT_ROOT DIAGNOSTICS_ROOT LOG_DIR CONDA_BASE CONDA_ENV
export LPRF_FUSION_SBATCH_ACCOUNT LPRF_FUSION_SBATCH_PARTITION
export PYTHONNOUSERSITE=1
exec bash "${PROJECT_DIR}/sbatch/submit_lprf_p7b_fusion_campaign.sh" "$@"
