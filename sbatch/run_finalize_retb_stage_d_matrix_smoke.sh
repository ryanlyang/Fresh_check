#!/usr/bin/env bash
#SBATCH --job-name=retb_stage_d_matrix_report
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/home/ryreu/atlas/Fresh_check}"
# shellcheck source=retb_common.sh
source "${PROJECT_DIR}/sbatch/retb_common.sh"
retb_setup
python scripts/finalize_retb_stage_d_matrix_smoke.py \
  --campaign-root "${CAMPAIGN_ROOT}"
