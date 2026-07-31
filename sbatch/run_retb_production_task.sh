#!/usr/bin/env bash
#SBATCH --job-name=retb_task
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=7-00:00:00
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/home/ryreu/atlas/Fresh_check}"
# shellcheck source=retb_common.sh
source "${PROJECT_DIR}/sbatch/retb_common.sh"
retb_setup
: "${RETB_NODE_ID:?RETB_NODE_ID is required}"
retb_run_task "${RETB_NODE_ID}"
if [[ "${RETB_DEFER_MANIFEST_MATERIALIZATION:-0}" != "1" ]]; then
  retb_materialize_downstream "${RETB_NODE_ID}"
fi
