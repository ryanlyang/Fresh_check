#!/usr/bin/env bash
#SBATCH --job-name=retb_ledger
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/home/ryreu/atlas/Fresh_check}"
source "${PROJECT_DIR}/sbatch/retb_common.sh"
retb_setup
python scripts/write_retb_job_ledger.py \
  --production-graph "${CAMPAIGN_ROOT}/job_ledgers/production_graph.json" \
  --previous-ledger "${CAMPAIGN_ROOT}/job_ledgers/initial_submission_ledger.json" \
  --submission-mode completed \
  --output "${CAMPAIGN_ROOT}/job_ledgers/completed_job_ledger.json"
if [[ "${RETB_SUBMISSION_SCOPE:-complete}" == "full_streamed" ]]; then
  python scripts/cleanup_retb_full_streamed_terminal_payloads.py \
    --campaign-root "${CAMPAIGN_ROOT}"
fi
