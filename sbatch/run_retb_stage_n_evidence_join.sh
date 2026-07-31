#!/usr/bin/env bash
#SBATCH --job-name=retb_stage_n_join
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
set -euo pipefail
IFS=$'\n\t'
PROJECT_DIR="${PROJECT_DIR:-/home/ryreu/atlas/Fresh_check}"
source "${PROJECT_DIR}/sbatch/retb_common.sh"
retb_setup
output="${CAMPAIGN_ROOT}/selection/stage_n_evidence_join.json"
python scripts/join_retb_stage_n_evidence.py \
  --campaign-root "${CAMPAIGN_ROOT}" \
  --output "${output}"
python scripts/attest_retb_direct_node_completion.py \
  --campaign-root "${CAMPAIGN_ROOT}" \
  --node-id stage_n_evidence_join \
  --output-artifact "${output}" >/dev/null
retb_materialize_downstream "stage_n_evidence_join"
