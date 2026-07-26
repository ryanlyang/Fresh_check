#!/usr/bin/env bash
#SBATCH --job-name=rpt_select
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

results="${CAMPAIGN_ROOT}/selection/screening_results.json"
python scripts/collect_relational_part_results.py \
  --campaign-root "${CAMPAIGN_ROOT}" \
  --registry "${CAMPAIGN_ROOT}/registry/screening_registry.json" \
  --mode screening \
  --output "${results}"
split_sha="$(rpt_field "${CAMPAIGN_ROOT}/campaign_spec.json" split_manifest_hash)"
mapfile -t hashes < <(rpt_hlt_hash_args)
hash_args=()
for value in "${hashes[@]}"; do
  hash_args+=(--hlt-cache-hash "${value}")
done
python scripts/select_relational_part_screening.py \
  --screening-registry "${CAMPAIGN_ROOT}/registry/screening_registry.json" \
  --results "${results}" \
  --campaign-spec "${CAMPAIGN_ROOT}/campaign_spec.json" \
  --split-manifest-sha256 "${split_sha}" \
  "${hash_args[@]}" \
  --output "${CAMPAIGN_ROOT}/selection/screening_summary.json"
