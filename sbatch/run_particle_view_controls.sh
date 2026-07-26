#!/usr/bin/env bash
# Evaluate one non-gating particle-view structural control.
#SBATCH --job-name=pview_control
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tigris
#SBATCH --account=reu-aisocial
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

set -euo pipefail
IFS=$'\n\t'
export PYTHONNOUSERSITE=1
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
: "${CONDA_ENV:=atlas_kd_tigris}"
source "${PROJECT_DIR}/sbatch/common.sh"
fresh_setup
fresh_run "${PYTHON_BIN}" -u scripts/evaluate_particle_view_controls.py "$@"
