#!/usr/bin/env bash
#SBATCH --job-name=rpt_splits
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=08:00:00

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=relational_part_common.sh
source "${SCRIPT_DIR}/relational_part_common.sh"
rpt_setup

sizes=(1000000 125000 0 125000 500000)
if [[ "${RPT_MINIATURE:-0}" == "1" ]]; then
  sizes=(20 10 0 10 20)
fi
mkdir -p "${CAMPAIGN_ROOT}/bootstrap"
python scripts/build_jetclass_splits.py \
  --data-dir "${DATA_DIR}" \
  --out "${CAMPAIGN_ROOT}/bootstrap/split_manifest.json.gz" \
  --tree-name tree \
  --max-constits 128 \
  --model-train "${sizes[0]}" \
  --model-val "${sizes[1]}" \
  --stack-train "${sizes[2]}" \
  --stack-val "${sizes[3]}" \
  --final-test "${sizes[4]}"
