#!/usr/bin/env bash
# Tigris defaults for the matched five-seed A0/P7b study.

set -euo pipefail
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${OUTPUT_ROOT:=${PROJECT_DIR}/checkpoints}"
: "${DIAGNOSTICS_ROOT:=${PROJECT_DIR}/fresh_check_diagnostics}"
: "${LOG_DIR:=${PROJECT_DIR}/fresh_check_logs}"
: "${CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
: "${CONDA_ENV:=atlas_kd_tigris}"
: "${DEVICE:=cuda}"
: "${LOCAL_RESIDUAL_FIELD_CURRICULUM_ROOT:=${OUTPUT_ROOT}/local_particle_residual_field_curriculum/rebuild_and_pilot_20260720_185817}"
: "${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT:=${OUTPUT_ROOT}/local_particle_residual_field_fusion/p7b_seed_control_p7b_fusion_20260722_220045}"
: "${LOCAL_RESIDUAL_FIELD_SEED_STUDY_SBATCH_ACCOUNT:=reu-aisocial}"
: "${LOCAL_RESIDUAL_FIELD_SEED_STUDY_SBATCH_PARTITION:=tigris}"
export PROJECT_DIR OUTPUT_ROOT DIAGNOSTICS_ROOT LOG_DIR CONDA_BASE CONDA_ENV DEVICE
export LOCAL_RESIDUAL_FIELD_CURRICULUM_ROOT LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT
export LOCAL_RESIDUAL_FIELD_SEED_STUDY_SBATCH_ACCOUNT LOCAL_RESIDUAL_FIELD_SEED_STUDY_SBATCH_PARTITION
export PYTHONNOUSERSITE=1
exec bash "${PROJECT_DIR}/sbatch/submit_lprf_p7b_seed_study.sh" "$@"
