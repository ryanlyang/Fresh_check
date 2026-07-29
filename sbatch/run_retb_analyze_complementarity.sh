#!/usr/bin/env bash
#SBATCH --job-name=retb_complement
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/home/ryreu/atlas/Fresh_check}"
source "${PROJECT_DIR}/sbatch/retb_common.sh"
retb_setup
retb_run_task "offline_complementarity"
