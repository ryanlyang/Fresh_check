#!/usr/bin/env bash
# Probe one actual ABPH variant/batch candidate inside a four-rank Tigris allocation.

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
VARIANT="${1:?Usage: worker <variant> <stage-family> <local-batch-size>}"
STAGE_FAMILY="${2:?Missing stage family}"
LOCAL_BATCH_SIZE="${3:?Missing local batch size}"
: "${ABPH_ROOT:?Set ABPH_ROOT}"
export PYTHONNOUSERSITE=1
fresh_setup

mapfile -t hosts < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
export MASTER_ADDR="${hosts[0]:?Unable to resolve DDP master host}"
numeric_job_id="${SLURM_JOB_ID%%_*}"
export MASTER_PORT="$((20000 + numeric_job_id % 20000))"
output="${ABPH_ROOT}/runtime_batch_measurements/${VARIANT}/ddp4/${STAGE_FAMILY}_b${LOCAL_BATCH_SIZE}.json"

fresh_run srun --nodes=4 --ntasks=4 --ntasks-per-node=1 \
  --kill-on-bad-exit=1 --cpu-bind=cores --export=ALL \
  "${PYTHON_BIN}" -u scripts/probe_adaptive_binary_runtime_batch.py \
  --campaign-root "${ABPH_ROOT}" \
  --variant "${VARIANT}" \
  --stage-family "${STAGE_FAMILY}" \
  --local-batch-size "${LOCAL_BATCH_SIZE}" \
  --expected-world-size 4 \
  --output "${output}" \
  --device "${DEVICE}"
