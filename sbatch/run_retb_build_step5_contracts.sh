#!/usr/bin/env bash
#SBATCH --job-name=retb_step5
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/home/ryreu/atlas/Fresh_check}"
source "${PROJECT_DIR}/sbatch/retb_common.sh"
retb_setup
python scripts/build_retb_step5_contracts.py --campaign-root "${CAMPAIGN_ROOT}"
