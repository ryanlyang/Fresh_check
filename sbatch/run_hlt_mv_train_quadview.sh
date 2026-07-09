#!/usr/bin/env bash
# Train the default HLT-MV quad-view particle fusion model.

#SBATCH --job-name=hlt_mv_quad
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=3-00:00:00
#SBATCH --mem=300G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${HLT_MV_MULTIVIEW_MODEL_NAME:=quad_hlt_hlt2_s0p20_s0p35_s1p00}"
: "${HLT_MV_MULTIVIEW_BATCH_SIZE:=40}"
: "${HLT_MV_MULTIVIEW_EVAL_BATCH_SIZE:=56}"
: "${HLT_MV_MULTIVIEW_SEED:=9401}"

export HLT_MV_MULTIVIEW_MODEL_NAME
export HLT_MV_MULTIVIEW_BATCH_SIZE
export HLT_MV_MULTIVIEW_EVAL_BATCH_SIZE
export HLT_MV_MULTIVIEW_SEED

exec bash "${PROJECT_DIR}/sbatch/run_hlt_mv_train_multiview.sh" "$@"
