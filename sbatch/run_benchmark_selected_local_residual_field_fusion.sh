#!/usr/bin/env bash
# Benchmark deployable members and selected fusion overhead.
#SBATCH --job-name=lprf_fuse_runtime
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tigris
#SBATCH --account=reu-aisocial
#SBATCH --time=12:00:00
#SBATCH --mem=180G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
set -euo pipefail
IFS=$'\n\t'
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
export PYTHONNOUSERSITE=1
source "${PROJECT_DIR}/sbatch/common.sh"
: "${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT:?campaign root required}"
: "${LOCAL_RESIDUAL_FIELD_FUSION_RUNTIME_BATCH_SIZE:=128}"
: "${LOCAL_RESIDUAL_FIELD_FUSION_RUNTIME_WARMUP_BATCHES:=10}"
: "${LOCAL_RESIDUAL_FIELD_FUSION_RUNTIME_MEASURED_BATCHES:=50}"
fresh_setup "$@"
fresh_run "${PYTHON_BIN}" -u scripts/benchmark_selected_local_residual_field_fusion.py --selected-fusion "${LOCAL_RESIDUAL_FIELD_FUSION_CAMPAIGN_ROOT}/selection/selected_fusion.json" --batch-size "${LOCAL_RESIDUAL_FIELD_FUSION_RUNTIME_BATCH_SIZE}" --warmup-batches "${LOCAL_RESIDUAL_FIELD_FUSION_RUNTIME_WARMUP_BATCHES}" --measured-batches "${LOCAL_RESIDUAL_FIELD_FUSION_RUNTIME_MEASURED_BATCHES}"
