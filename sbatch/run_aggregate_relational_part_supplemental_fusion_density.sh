#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

: "${CAMPAIGN_ROOT:?CAMPAIGN_ROOT is required}"
: "${PINNED_SOURCE_ROOT:?PINNED_SOURCE_ROOT is required}"
: "${PROJECT_DIR:=${PINNED_SOURCE_ROOT}}"
: "${FUSION_AGGREGATOR:=${CAMPAIGN_ROOT}/aggregate_relational_part_supplemental_fusion.py}"
SCRIPT_DIR="${PROJECT_DIR}/sbatch"
# shellcheck source=relational_part_common.sh
source "${SCRIPT_DIR}/relational_part_common.sh"
rpt_setup

python "${FUSION_AGGREGATOR}" \
  --source-root "${PINNED_SOURCE_ROOT}" \
  --campaign-root "${CAMPAIGN_ROOT}"
