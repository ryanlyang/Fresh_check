#!/usr/bin/env bash
# One-command production submission/recovery entry point.

set -euo pipefail
IFS=$'\n\t'
export PYTHONNOUSERSITE=1
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
: "${CONDA_ENV:=atlas_kd_tigris}"
source "${PROJECT_DIR}/sbatch/common.sh"
fresh_prepare_submitter
fresh_activate_env
cd "${PROJECT_DIR}"
fresh_run "${PYTHON_BIN}" -u scripts/submit_particle_view_full_pilot.py "$@"
