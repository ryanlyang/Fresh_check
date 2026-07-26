#!/usr/bin/env bash
#SBATCH --job-name=rpt_tree_backend
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=02:00:00

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=relational_part_common.sh
source "${SCRIPT_DIR}/relational_part_common.sh"
rpt_setup

python scripts/build_relational_part_tree_backend.py \
  --contract relational_ca_tree_v1 \
  --build-dir "${CAMPAIGN_ROOT}/backend/build" \
  --output-dir "${CAMPAIGN_ROOT}/backend"
