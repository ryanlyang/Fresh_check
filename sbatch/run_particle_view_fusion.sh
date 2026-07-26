#!/usr/bin/env bash
# Fit/evaluate one sealed stack-validation fusion recipe.
#SBATCH --job-name=pview_fusion
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tigris
#SBATCH --account=reu-aisocial
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12

set -euo pipefail
IFS=$'\n\t'
export PYTHONNOUSERSITE=1
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
: "${CONDA_ENV:=atlas_kd_tigris}"
source "${PROJECT_DIR}/sbatch/common.sh"
fresh_setup
if [[ "${1:-}" == pv* ]]; then
  : "${PARTICLE_VIEW_GRAPH:?Set PARTICLE_VIEW_GRAPH to the immutable production graph}"
  fresh_run "${PYTHON_BIN}" -u scripts/execute_particle_view_graph_node.py \
    --graph "${PARTICLE_VIEW_GRAPH}" --node-id "$1"
  exit 0
fi
fresh_run "${PYTHON_BIN}" -u scripts/run_particle_view_fusion.py "$@"
