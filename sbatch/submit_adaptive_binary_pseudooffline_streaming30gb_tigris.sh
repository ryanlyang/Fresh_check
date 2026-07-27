#!/usr/bin/env bash
# Queue the complete storage-wave ABPH pilot under the hard 30 GB contract.
set -euo pipefail

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${ABPH_DATA_DIR:=/home/ryreu/atlas/PracticeTagging/data/jetclass_part1}"
: "${ABPH_SBATCH_ACCOUNT:=reu-aisocial}"
: "${ABPH_CAMPAIGN_MODE:=pilot}"
: "${ABPH_STAGE_MODE:=full}"
: "${ABPH_RECONSTRUCTOR_PARALLELISM:=ddp8}"
: "${ABPH_RECONSTRUCTOR_SCHEDULE_POLICY:=accelerated_screening_v2_7day}"
: "${ABPH_STORAGE_PROFILE:=streaming_30gb_v1}"
: "${PYTHONNOUSERSITE:=1}"

[[ "${ABPH_CAMPAIGN_MODE}" == "pilot" ]] || {
  echo "The 30 GB production wrapper queues the pilot first; high-data remains approval-gated." >&2
  exit 2
}
: "${ABPH_STORAGE_PROJECTION_PATH:?Set ABPH_STORAGE_PROJECTION_PATH to the measured Wave-0 projection}"
: "${ABPH_RUNTIME_ACCEPTANCE_PATH:?Set ABPH_RUNTIME_ACCEPTANCE_PATH to the accepted DDP8 runtime artifact}"

export PROJECT_DIR ABPH_DATA_DIR ABPH_SBATCH_ACCOUNT ABPH_CAMPAIGN_MODE
export ABPH_STAGE_MODE ABPH_RECONSTRUCTOR_PARALLELISM ABPH_STORAGE_PROFILE
export ABPH_RECONSTRUCTOR_SCHEDULE_POLICY
export ABPH_STORAGE_PROJECTION_PATH ABPH_RUNTIME_ACCEPTANCE_PATH PYTHONNOUSERSITE
exec bash "${PROJECT_DIR}/sbatch/submit_adaptive_binary_pseudooffline.sh"
