#!/usr/bin/env bash
# Step-8 seed aggregation, winner selection, and fairness closure.
#SBATCH --job-name=pview_confirm
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tigris
#SBATCH --account=reu-aisocial
#SBATCH --nodes=1
#SBATCH --time=3-00:00:00
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

ACTION="${1:?Usage: run_particle_view_confirmation.sh <select|stage-g|confirm> [arguments...]}"
shift
if [[ "${ACTION}" == pv* ]]; then
  : "${PARTICLE_VIEW_GRAPH:?Set PARTICLE_VIEW_GRAPH to the immutable production graph}"
  fresh_run "${PYTHON_BIN}" -u scripts/execute_particle_view_graph_node.py \
    --graph "${PARTICLE_VIEW_GRAPH}" --node-id "${ACTION}"
  exit 0
fi
case "${ACTION}" in
  select)
    fresh_run "${PYTHON_BIN}" -u scripts/select_particle_view_configuration.py "$@"
    ;;
  stage-g)
    fresh_run "${PYTHON_BIN}" -u scripts/build_particle_view_stage_g_controls.py "$@"
    ;;
  confirm)
    fresh_run "${PYTHON_BIN}" -u scripts/confirm_particle_view_deployable.py "$@"
    ;;
  *)
    echo "Unknown Step-8 confirmation action: ${ACTION}" >&2
    exit 2
    ;;
esac
