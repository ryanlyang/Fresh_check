#!/usr/bin/env bash
# Queue local residual-field pilot and high-data campaigns on sporcsubmit/tier3.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_ENV:=atlas_kd}"
: "${LOCAL_RESIDUAL_FIELD_SBATCH_PARTITION:=tier3}"
: "${LOCAL_RESIDUAL_FIELD_GPU_GRES:=gpu:1}"
: "${LOCAL_RESIDUAL_FIELD_GPU_CPUS_PER_TASK:=8}"
: "${LOCAL_RESIDUAL_FIELD_GPU_MEM:=220G}"
: "${LOCAL_RESIDUAL_FIELD_CPU_CPUS_PER_TASK:=8}"
: "${LOCAL_RESIDUAL_FIELD_CPU_MEM:=220G}"
: "${CONFIRM_FINAL_TEST:=1}"

export PROJECT_DIR CONDA_ENV
export LOCAL_RESIDUAL_FIELD_SBATCH_PARTITION LOCAL_RESIDUAL_FIELD_GPU_GRES
export LOCAL_RESIDUAL_FIELD_GPU_CPUS_PER_TASK LOCAL_RESIDUAL_FIELD_GPU_MEM
export LOCAL_RESIDUAL_FIELD_CPU_CPUS_PER_TASK LOCAL_RESIDUAL_FIELD_CPU_MEM
export CONFIRM_FINAL_TEST

LOCAL_RESIDUAL_FIELD_CAMPAIGN_MODE=pilot bash "${PROJECT_DIR}/sbatch/submit_local_particle_residual_field_experiment.sh"
LOCAL_RESIDUAL_FIELD_CAMPAIGN_MODE=highdata bash "${PROJECT_DIR}/sbatch/submit_local_particle_residual_field_experiment.sh"
