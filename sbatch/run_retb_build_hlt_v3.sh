#!/usr/bin/env bash
#SBATCH --job-name=retb_hlt_v3
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=24:00:00
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/home/ryreu/atlas/Fresh_check}"
# shellcheck source=retb_common.sh
source "${PROJECT_DIR}/sbatch/retb_common.sh"
retb_setup
retb_run_task "hlt_v3_cache"
