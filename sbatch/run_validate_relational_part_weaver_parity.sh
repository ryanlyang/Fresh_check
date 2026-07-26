#!/usr/bin/env bash
#SBATCH --job-name=rpt_weaver_parity
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=01:00:00

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=relational_part_common.sh
source "${SCRIPT_DIR}/relational_part_common.sh"
rpt_setup

python scripts/validate_relational_part_weaver_parity.py \
  --device cpu \
  --output "${CAMPAIGN_ROOT}/backend/weaver_parity.json"
