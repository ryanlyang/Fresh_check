#!/usr/bin/env bash
#SBATCH --job-name=retb_splits
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=12:00:00
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/home/ryreu/atlas/Fresh_check}"
# shellcheck source=retb_common.sh
source "${PROJECT_DIR}/sbatch/retb_common.sh"
retb_require_campaign_root
retb_activate
readarray -t sizes < <(
  python -c \
    'import json,sys; sizes=json.load(open(sys.argv[1]))["split_sizes"]; print(sizes["model_train"]); print(sizes["model_val"]); print(sizes["stack_train"]); print(sizes["stack_val"]); print(sizes["final_test"])' \
    "${CAMPAIGN_ROOT}/job_ledgers/production_graph.json"
)
if [[ "${#sizes[@]}" -ne 5 ]]; then
  echo "RETB production graph split-size vector differs" >&2
  exit 2
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
