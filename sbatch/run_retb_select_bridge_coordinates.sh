#!/usr/bin/env bash
#SBATCH --job-name=retb_bridge_select
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/home/ryreu/atlas/Fresh_check}"
source "${PROJECT_DIR}/sbatch/retb_common.sh"
retb_setup
retb_run_task "target_coordinate_selector"
retb_materialize_downstream "target_coordinate_selector"
