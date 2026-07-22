#!/usr/bin/env bash
# Refit the F_method winner on F_seed with its recipe frozen.
#SBATCH --job-name=lprf_fuse_replay
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tigris
#SBATCH --account=reu-aisocial
#SBATCH --time=1-00:00:00
#SBATCH --mem=180G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
set -euo pipefail
IFS=$'\n\t'
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
export PYTHONNOUSERSITE=1
source "${PROJECT_DIR}/sbatch/common.sh"
: "${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT:?campaign root required}"
fresh_setup "$@"
fresh_run "${PYTHON_BIN}" -u scripts/replay_selected_local_residual_field_fusion_recipe.py --selected-fusion "${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT}/selection/selected_fusion.json"
