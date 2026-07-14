#!/usr/bin/env bash
# Submit both campaigns on sporcsubmit/tier3.
set -euo pipefail
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${PD10_DATA_DIR:=/home/ryreu/atlas/PracticeTagging/data}"
: "${OUTPUT_ROOT:=${PROJECT_DIR}/checkpoints}"
: "${CONDA_ENV:=atlas_kd}"
: "${DEVICE:=cuda}"
: "${CONFIRM_FINAL_TEST:=0}"

if [[ "${CONDA_BASE:-}" == *miniforge3-aarch64* ]]; then unset CONDA_BASE; fi
export PROJECT_DIR PD10_DATA_DIR OUTPUT_ROOT CONDA_ENV DEVICE CONFIRM_FINAL_TEST
export CONSTRAINED_C2F_SBATCH_PARTITION="${CONSTRAINED_C2F_SPORC_PARTITION:-tier3}"
export CONSTRAINED_C2F_GPU_GRES="${CONSTRAINED_C2F_SPORC_GPU_GRES:-gpu:1}"
export CONSTRAINED_C2F_GPU_CPUS_PER_TASK="${CONSTRAINED_C2F_SPORC_GPU_CPUS_PER_TASK:-12}"
export CONSTRAINED_C2F_GPU_MEM="${CONSTRAINED_C2F_SPORC_GPU_MEM:-260G}"
export CONSTRAINED_C2F_CPU_CPUS_PER_TASK="${CONSTRAINED_C2F_SPORC_CPU_CPUS_PER_TASK:-16}"
export CONSTRAINED_C2F_CPU_MEM="${CONSTRAINED_C2F_SPORC_CPU_MEM:-300G}"

bash "${PROJECT_DIR}/sbatch/submit_constrained_coarse_to_fine_pilot_and_highdata.sh"
