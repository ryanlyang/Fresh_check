#!/usr/bin/env bash
# Run one baseline/reconstructor/renderer/tagger variant via the production executor.

#SBATCH --job-name=abph_variant
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tier3
#SBATCH --time=3-00:00:00
#SBATCH --mem=300G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16

set -euo pipefail
IFS=$'\n\t'
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
source "${PROJECT_DIR}/sbatch/common.sh"
VARIANT="${1:?Usage: run_adaptive_binary_variant.sh <registry-variant>}"
SEED_INDEX="${2:-1}"
: "${ABPH_ROOT:=${OUTPUT_ROOT}/adaptive_binary_pseudooffline}"
: "${ABPH_VARIANT_EXECUTOR:=${PROJECT_DIR}/scripts/train_adaptive_binary_pseudooffline_variant.py}"
: "${ABPH_JOB_LAUNCHER:=direct}"
: "${ABPH_DISTRIBUTED_NODES:=1}"
: "${ABPH_DISTRIBUTED_NTASKS:=1}"
: "${ABPH_DISTRIBUTED_NTASKS_PER_NODE:=1}"
: "${ABPH_DISTRIBUTED_GPUS_PER_NODE:=1}"
: "${ABPH_DISTRIBUTED_WORLD_SIZE:=1}"
export PYTHONNOUSERSITE=1
fresh_setup
fresh_require_file "${ABPH_VARIANT_EXECUTOR}"
if [[ "${VARIANT}" =~ ^[BCD] ]]; then
  "${PYTHON_BIN}" scripts/validate_adaptive_binary_orchestration.py preflight \
    --path "${ABPH_ROOT}/audits/actual_target_feasibility.json"
fi
cmd=("${PYTHON_BIN}" -u "${ABPH_VARIANT_EXECUTOR}" --variant "${VARIANT}" --seed-index "${SEED_INDEX}" --campaign-root "${ABPH_ROOT}" --device "${DEVICE}")

if [[ "${VARIANT}" =~ ^[BCD] ]]; then
  case "${ABPH_RECONSTRUCTOR_PARALLELISM:-single}" in
    single)
      [[ "${ABPH_JOB_LAUNCHER}" == "direct" && "${ABPH_DISTRIBUTED_WORLD_SIZE}" == "1" ]] || {
        echo "single reconstructor mode requires launcher=direct and world_size=1" >&2
        exit 2
      }
      ;;
    ddp4)
      [[ "${ABPH_JOB_LAUNCHER}" == "srun" ]] || { echo "ddp4 requires srun" >&2; exit 2; }
      [[ "${ABPH_DISTRIBUTED_NODES}" == "4" && "${ABPH_DISTRIBUTED_NTASKS}" == "4" ]] || {
        echo "ddp4 requires four nodes and four tasks" >&2
        exit 2
      }
      [[ "${ABPH_DISTRIBUTED_NTASKS_PER_NODE}" == "1" && "${ABPH_DISTRIBUTED_GPUS_PER_NODE}" == "1" ]] || {
        echo "ddp4 requires one task and one GPU per node" >&2
        exit 2
      }
      [[ "${ABPH_DISTRIBUTED_WORLD_SIZE}" == "4" ]] || { echo "ddp4 world size must be four" >&2; exit 2; }
      ;;
    *)
      echo "unsupported ABPH_RECONSTRUCTOR_PARALLELISM=${ABPH_RECONSTRUCTOR_PARALLELISM}" >&2
      exit 2
      ;;
  esac
else
  [[ "${ABPH_JOB_LAUNCHER}" == "direct" && "${ABPH_DISTRIBUTED_WORLD_SIZE}" == "1" ]] || {
    echo "only B/C/D reconstructor jobs may use distributed launch" >&2
    exit 2
  }
fi

if [[ "${ABPH_JOB_LAUNCHER}" == "srun" ]]; then
  [[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_JOB_NODELIST:-}" ]] || {
    echo "srun launch requires an active Slurm allocation" >&2
    exit 2
  }
  [[ "${SLURM_JOB_NUM_NODES:-0}" == "${ABPH_DISTRIBUTED_NODES}" ]] || {
    echo "allocated node count ${SLURM_JOB_NUM_NODES:-unset} differs from ${ABPH_DISTRIBUTED_NODES}" >&2
    exit 2
  }
  [[ "${SLURM_NTASKS:-0}" == "${ABPH_DISTRIBUTED_NTASKS}" ]] || {
    echo "allocated task count ${SLURM_NTASKS:-unset} differs from ${ABPH_DISTRIBUTED_NTASKS}" >&2
    exit 2
  }
  mapfile -t allocated_hosts < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
  master_addr="${allocated_hosts[0]:-}"
  [[ -n "${master_addr}" ]] || { echo "could not resolve DDP master host" >&2; exit 2; }
  numeric_job_id="${SLURM_JOB_ID%%_*}"
  [[ "${numeric_job_id}" =~ ^[0-9]+$ ]] || { echo "invalid Slurm job id ${SLURM_JOB_ID}" >&2; exit 2; }
  export MASTER_ADDR="${master_addr}"
  export MASTER_PORT="$((20000 + numeric_job_id % 20000))"
  fresh_run srun \
    --nodes="${ABPH_DISTRIBUTED_NODES}" \
    --ntasks="${ABPH_DISTRIBUTED_NTASKS}" \
    --ntasks-per-node="${ABPH_DISTRIBUTED_NTASKS_PER_NODE}" \
    --kill-on-bad-exit=1 \
    --cpu-bind=cores \
    --export=ALL \
    "${cmd[@]}"
else
  fresh_run "${cmd[@]}"
fi
