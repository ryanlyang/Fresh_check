#!/usr/bin/env bash
#SBATCH --job-name=retb_stage_a_obase
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --time=7-00:00:00
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/home/ryreu/atlas/Fresh_check}"
source "${PROJECT_DIR}/sbatch/retb_common.sh"
retb_setup
python scripts/prepare_retb_stage_b_prerequisites.py \
  --campaign-root "${CAMPAIGN_ROOT}" \
  --train-control O_BASE \
  --device "${RETB_DEVICE:-auto}"
