#!/usr/bin/env bash
# Tigris defaults for the locked 3M A0/P7b matched-seed study.

set -euo pipefail
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${OUTPUT_ROOT:=${PROJECT_DIR}/checkpoints}"
: "${DIAGNOSTICS_ROOT:=${PROJECT_DIR}/fresh_check_diagnostics}"
: "${LOG_DIR:=${PROJECT_DIR}/fresh_check_logs}"
: "${CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
: "${CONDA_ENV:=atlas_kd_tigris}"
: "${DEVICE:=cuda}"
: "${LOCAL_RESIDUAL_FIELD_HIGH_DATA_REFERENCE_ROOT:=${OUTPUT_ROOT}/local_particle_residual_field_curriculum/rebuild_and_pilot_20260720_185817}"
: "${LOCAL_RESIDUAL_FIELD_HIGH_DATA_DATA_DIR:=/home/ryreu/atlas/PracticeTagging/data}"
: "${LOCAL_RESIDUAL_FIELD_HIGH_DATA_SBATCH_ACCOUNT:=reu-aisocial}"
: "${LOCAL_RESIDUAL_FIELD_HIGH_DATA_SBATCH_PARTITION:=tigris}"
: "${LOCAL_RESIDUAL_FIELD_HIGH_DATA_GPU_GRES:=gpu:gh200:1}"
export PROJECT_DIR OUTPUT_ROOT DIAGNOSTICS_ROOT LOG_DIR CONDA_BASE CONDA_ENV DEVICE
export LOCAL_RESIDUAL_FIELD_HIGH_DATA_REFERENCE_ROOT LOCAL_RESIDUAL_FIELD_HIGH_DATA_DATA_DIR
export LOCAL_RESIDUAL_FIELD_HIGH_DATA_SBATCH_ACCOUNT LOCAL_RESIDUAL_FIELD_HIGH_DATA_SBATCH_PARTITION
export LOCAL_RESIDUAL_FIELD_HIGH_DATA_GPU_GRES
export PYTHONNOUSERSITE=1
exec bash "${PROJECT_DIR}/sbatch/submit_lprf_p7b_high_data_seed_study.sh" "$@"
