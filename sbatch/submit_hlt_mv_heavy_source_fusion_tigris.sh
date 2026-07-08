#!/usr/bin/env bash
# TIGRIS/GH200 defaults for the heavy HLT-MV source/fusion graph.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"

: "${HLT_MV_SBATCH_PARTITION:=tigris}"
: "${HLT_MV_GPU_GRES:=gpu:gh200:1}"
: "${HLT_MV_GPU_CPUS_PER_TASK:=16}"
: "${HLT_MV_GPU_MEM:=300G}"
: "${HLT_MV_CPU_CPUS_PER_TASK:=16}"
: "${HLT_MV_CPU_MEM:=220G}"
: "${DEVICE:=cuda}"
: "${CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
: "${CONDA_ENV:=atlas_kd_tigris}"
: "${PYTHONNOUSERSITE:=1}"

export HLT_MV_SBATCH_PARTITION HLT_MV_GPU_GRES
export HLT_MV_GPU_CPUS_PER_TASK HLT_MV_GPU_MEM HLT_MV_GPU_TIME
export HLT_MV_CPU_CPUS_PER_TASK HLT_MV_CPU_MEM HLT_MV_CPU_TIME
export DEVICE CONDA_BASE CONDA_ENV PYTHONNOUSERSITE

exec bash "${SCRIPT_DIR}/submit_hlt_mv_heavy_source_fusion.sh"
