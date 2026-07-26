#!/usr/bin/env bash
#SBATCH --job-name=rpt_semantic
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --time=12:00:00

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=relational_part_common.sh
source "${SCRIPT_DIR}/relational_part_common.sh"
rpt_setup

python scripts/run_relational_part_semantic_perturbation.py \
  --campaign-root "${CAMPAIGN_ROOT}" \
  --device "${RPT_DEVICE}"
