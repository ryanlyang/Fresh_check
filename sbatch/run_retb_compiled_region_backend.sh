#!/usr/bin/env bash
#SBATCH --job-name=retb_region_backend
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=02:00:00
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/home/ryreu/atlas/Fresh_check}"
# shellcheck source=retb_common.sh
source "${PROJECT_DIR}/sbatch/retb_common.sh"
retb_setup
python -c \
  'from torch.utils.cpp_extension import verify_ninja_availability; verify_ninja_availability()'
python scripts/build_relational_part_tree_backend.py \
  --contract relational_ca_tree_v1 \
  --build-dir "${CAMPAIGN_ROOT}/backend/build" \
  --output-dir "${CAMPAIGN_ROOT}/backend"
