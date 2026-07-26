#!/usr/bin/env bash
# Measure dense profiler overhead with both controls on one Tigris allocation.

#SBATCH --job-name=abph_single_path_pair
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tigris
#SBATCH --account=reu-aisocial
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=220G
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=16

set -euo pipefail
IFS=$'\n\t'
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${ABPH_ROOT:?Set ABPH_ROOT to the prepared pilot campaign root}"
: "${ABPH_RUNTIME_ACCEPTANCE_ROOT:=${ABPH_ROOT}/audits/runtime_acceptance}"

[[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_JOB_NODELIST:-}" ]] || {
  echo "Matched single-path acceptance must run inside one Slurm allocation" >&2
  exit 2
}

export ABPH_RECONSTRUCTOR_PARALLELISM=single
export ABPH_JOB_LAUNCHER=direct
export ABPH_DISTRIBUTED_NODES=1
export ABPH_DISTRIBUTED_NTASKS=1
export ABPH_DISTRIBUTED_NTASKS_PER_NODE=1
export ABPH_DISTRIBUTED_WORLD_SIZE=1
export ABPH_MATCHED_SINGLE_PATH_PAIR_ID="${SLURM_JOB_ID}:D1_kt32_mh4_particles"
export PYTHONNOUSERSITE=1

worker="${PROJECT_DIR}/sbatch/run_adaptive_binary_runtime_acceptance.sh"

# The controls are deliberately sequential. Their wall-time artifacts must report
# this same job, host, node list, and pair id before the compiler will accept them.
bash "${worker}" benchmark_uninstrumented D1_kt32_mh4_particles
bash "${worker}" benchmark D1_kt32_mh4_particles

