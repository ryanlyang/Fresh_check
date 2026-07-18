#!/usr/bin/env bash
# Compile provenance-bound candidate measurements into one immutable contract.

#SBATCH --job-name=abph_batch_contract
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tigris
#SBATCH --account=reu-aisocial
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
source "${PROJECT_DIR}/sbatch/common.sh"
VARIANT="${1:?Usage: compiler <variant> [world-size]}"
WORLD_SIZE="${2:-4}"
: "${ABPH_ROOT:?Set ABPH_ROOT}"
export PYTHONNOUSERSITE=1
fresh_setup
cmd=("${PYTHON_BIN}" -u scripts/compile_adaptive_binary_runtime_batch_contract.py \
  --campaign-root "${ABPH_ROOT}" \
  --variant "${VARIANT}" \
  --measurement-dir "${ABPH_ROOT}/runtime_batch_measurements/${VARIANT}/ddp${WORLD_SIZE}" \
  --requested-world-size "${WORLD_SIZE}")
if [[ "${ABPH_STORAGE_PROFILE:-cache_heavy_v1}" == "streaming_30gb_v1" ]]; then
  cmd=(bash "${PROJECT_DIR}/sbatch/run_with_adaptive_binary_ram_workspace.sh" "${cmd[@]}")
fi
fresh_run "${cmd[@]}"
