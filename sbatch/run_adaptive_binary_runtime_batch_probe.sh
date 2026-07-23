#!/usr/bin/env bash
# Probe one actual ABPH variant/batch candidate in the requested Slurm topology.

#SBATCH --job-name=abph_batch_probe
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tigris
#SBATCH --account=reu-aisocial
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --time=02:00:00

set -euo pipefail
IFS=$'\n\t'
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
source "${PROJECT_DIR}/sbatch/common.sh"
source "${PROJECT_DIR}/sbatch/adaptive_binary_ddp_launch.sh"
VARIANT="${1:?Usage: worker <variant> <stage-family> <local-batch-size>}"
STAGE_FAMILY="${2:?Missing stage family}"
LOCAL_BATCH_SIZE="${3:?Missing local batch size}"
: "${ABPH_ROOT:?Set ABPH_ROOT}"
: "${ABPH_RUNTIME_BATCH_MEASUREMENT_ROOT:=${ABPH_ROOT}/runtime_batch_measurements}"
export PYTHONNOUSERSITE=1
fresh_setup

world_size="${ABPH_DISTRIBUTED_WORLD_SIZE:-${SLURM_NTASKS:-1}}"
nodes="${ABPH_DISTRIBUTED_NODES:-${SLURM_JOB_NUM_NODES:-1}}"
tasks_per_node="${ABPH_DISTRIBUTED_NTASKS_PER_NODE:-1}"
output="${ABPH_RUNTIME_BATCH_MEASUREMENT_ROOT}/${VARIANT}/ddp${world_size}/${STAGE_FAMILY}_b${LOCAL_BATCH_SIZE}.json"
probe_command=(
  "${PYTHON_BIN}" -u scripts/probe_adaptive_binary_runtime_batch.py
  --campaign-root "${ABPH_ROOT}"
  --variant "${VARIANT}"
  --stage-family "${STAGE_FAMILY}"
  --local-batch-size "${LOCAL_BATCH_SIZE}"
  --expected-world-size "${world_size}"
  --output "${output}"
  --device "${DEVICE}"
)
if [[ "${ABPH_STORAGE_PROFILE:-cache_heavy_v1}" == "streaming_30gb_v1" ]]; then
  probe_command=(bash "${PROJECT_DIR}/sbatch/run_with_adaptive_binary_ram_workspace.sh" "${probe_command[@]}")
fi

if ((world_size > 1)); then
  mapfile -t hosts < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
  export MASTER_ADDR="${hosts[0]:?Unable to resolve DDP master host}"
  abph_fresh_run_srun_with_port_retry --nodes="${nodes}" --ntasks="${world_size}" \
    --ntasks-per-node="${tasks_per_node}" --kill-on-bad-exit=1 \
    --cpu-bind=cores --export=ALL "${probe_command[@]}"
else
  fresh_run "${probe_command[@]}"
fi
