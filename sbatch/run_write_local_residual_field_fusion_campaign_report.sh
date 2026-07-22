#!/usr/bin/env bash
# Compose and atomically publish the complete immutable final report.
#SBATCH --job-name=lprf_fuse_report
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tigris
#SBATCH --account=reu-aisocial
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
set -euo pipefail
IFS=$'\n\t'
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
export PYTHONNOUSERSITE=1
source "${PROJECT_DIR}/sbatch/common.sh"
: "${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT:?campaign root required}"
fresh_setup "$@"
fresh_run "${PYTHON_BIN}" -u scripts/write_local_residual_field_fusion_campaign_report.py --selected-fusion "${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT}/selection/selected_fusion.json"
if ! fresh_is_dry_run; then fresh_assert_json_ok "${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT}/final_report/run_report.json"; fi
