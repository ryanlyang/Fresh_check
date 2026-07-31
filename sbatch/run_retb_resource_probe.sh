#!/usr/bin/env bash
#SBATCH --job-name=retb_probe
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=01:00:00
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/home/ryreu/atlas/Fresh_check}"
# shellcheck source=retb_common.sh
source "${PROJECT_DIR}/sbatch/retb_common.sh"
retb_setup
: "${RETB_RESOURCE_KIND:?RETB_RESOURCE_KIND is required}"
: "${RETB_COMPILED_REGION_PARITY:=${CAMPAIGN_ROOT}/backend/backend_manifest.json}"
requested="${RETB_CPU_REQUIRED_MEMORY_BYTES:-68719476736}"
if [[ "${RETB_RESOURCE_KIND}" == "gpu" ]]; then
  requested="${RETB_GPU_REQUIRED_DEVICE_BYTES:-68719476736}"
fi
python scripts/probe_retb_resources.py \
  --campaign-root "${CAMPAIGN_ROOT}" \
  --resource-kind "${RETB_RESOURCE_KIND}" \
  --compiled-region-parity "${RETB_COMPILED_REGION_PARITY}" \
  --requested-memory-bytes "${requested}" \
  --output "${CAMPAIGN_ROOT}/job_ledgers/resource_probes/${RETB_RESOURCE_KIND}.json"
