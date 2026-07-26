#!/usr/bin/env bash
# Production Step-10 source/preflight node.
#SBATCH --job-name=pview_prepare
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tigris
#SBATCH --account=reu-aisocial
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=12

set -euo pipefail
IFS=$'\n\t'
export PYTHONNOUSERSITE=1
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
: "${CONDA_ENV:=atlas_kd_tigris}"
: "${PARTICLE_VIEW_GRAPH:?Set PARTICLE_VIEW_GRAPH}"
source "${PROJECT_DIR}/sbatch/common.sh"
fresh_setup
fresh_run "${PYTHON_BIN}" -u scripts/execute_particle_view_graph_node.py \
  --graph "${PARTICLE_VIEW_GRAPH}" --node-id "${1:?missing logical node}"
