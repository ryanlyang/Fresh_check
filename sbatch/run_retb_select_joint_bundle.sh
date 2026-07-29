#!/usr/bin/env bash
#SBATCH --job-name=retb_joint_select
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/home/ryreu/atlas/Fresh_check}"
source "${PROJECT_DIR}/sbatch/retb_common.sh"
retb_setup
retb_run_task "joint_predictor_selector"
