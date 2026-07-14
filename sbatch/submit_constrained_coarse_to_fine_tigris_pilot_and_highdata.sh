#!/usr/bin/env bash
# Submit both campaigns on RIT Tigris GH200 nodes.
set -euo pipefail
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${PD10_DATA_DIR:=/home/ryreu/atlas/PracticeTagging/data}"
: "${OUTPUT_ROOT:=${PROJECT_DIR}/checkpoints}"
: "${CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
: "${CONDA_ENV:=atlas_kd_tigris}"
: "${DEVICE:=cuda}"
: "${CONFIRM_FINAL_TEST:=1}"

export PROJECT_DIR PD10_DATA_DIR OUTPUT_ROOT CONDA_BASE CONDA_ENV DEVICE CONFIRM_FINAL_TEST
export CONSTRAINED_C2F_SBATCH_PARTITION="${CONSTRAINED_C2F_TIGRIS_PARTITION:-tigris}"
export CONSTRAINED_C2F_GPU_GRES="${CONSTRAINED_C2F_TIGRIS_GPU_GRES:-gpu:gh200:1}"
export CONSTRAINED_C2F_GPU_CPUS_PER_TASK="${CONSTRAINED_C2F_TIGRIS_GPU_CPUS_PER_TASK:-16}"
export CONSTRAINED_C2F_GPU_MEM="${CONSTRAINED_C2F_TIGRIS_GPU_MEM:-220G}"
export CONSTRAINED_C2F_CPU_CPUS_PER_TASK="${CONSTRAINED_C2F_TIGRIS_CPU_CPUS_PER_TASK:-16}"
export CONSTRAINED_C2F_CPU_MEM="${CONSTRAINED_C2F_TIGRIS_CPU_MEM:-220G}"

bash "${PROJECT_DIR}/sbatch/submit_constrained_coarse_to_fine_pilot_and_highdata.sh"
