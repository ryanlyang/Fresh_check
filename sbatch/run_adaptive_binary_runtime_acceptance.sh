#!/usr/bin/env bash
# Run one Step-10 single/DDP4 smoke or representative runtime benchmark.

#SBATCH --job-name=abph_runtime_accept
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
#SBATCH --partition=tigris
#SBATCH --account=reu-aisocial
#SBATCH --time=3-00:00:00
#SBATCH --mem=220G
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=16

set -euo pipefail
IFS=$'\n\t'
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
source "${PROJECT_DIR}/sbatch/common.sh"
MODE="${1:?Usage: run_adaptive_binary_runtime_acceptance.sh <smoke|benchmark> [variant]}"
VARIANT="${2:-}"
: "${ABPH_ROOT:?Set ABPH_ROOT to the prepared pilot campaign root}"
: "${ABPH_RUNTIME_ACCEPTANCE_ROOT:=${ABPH_ROOT}/audits/runtime_acceptance}"
: "${ABPH_RECONSTRUCTOR_PARALLELISM:?Set single or ddp4}"
: "${ABPH_JOB_LAUNCHER:?Set direct or srun}"
: "${ABPH_DISTRIBUTED_NODES:?Set requested node count}"
: "${ABPH_DISTRIBUTED_NTASKS:?Set requested task count}"
: "${ABPH_DISTRIBUTED_NTASKS_PER_NODE:=1}"
: "${ABPH_DISTRIBUTED_WORLD_SIZE:?Set expected world size}"
export PYTHONNOUSERSITE=1
fresh_setup

case "${ABPH_RECONSTRUCTOR_PARALLELISM}" in
  single)
    [[ "${ABPH_JOB_LAUNCHER}" == "direct" && "${ABPH_DISTRIBUTED_WORLD_SIZE}" == "1" ]] || {
      echo "single acceptance jobs require direct/world-size-one" >&2
      exit 2
    }
    ;;
  ddp4)
    [[ "${ABPH_JOB_LAUNCHER}" == "srun" ]] || { echo "ddp4 acceptance requires srun" >&2; exit 2; }
    [[ "${ABPH_DISTRIBUTED_NODES}" == "4" && "${ABPH_DISTRIBUTED_NTASKS}" == "4" ]] || {
      echo "ddp4 acceptance requires four nodes/tasks" >&2
      exit 2
    }
    ;;
  *) echo "Unknown acceptance parallelism ${ABPH_RECONSTRUCTOR_PARALLELISM}" >&2; exit 2 ;;
esac

profile_root="${ABPH_RUNTIME_ACCEPTANCE_ROOT}/${ABPH_RECONSTRUCTOR_PARALLELISM}"
case "${MODE}" in
  smoke)
    output_dir="${profile_root}/transport_smoke"
    cmd=("${PYTHON_BIN}" -u scripts/run_adaptive_binary_ddp_acceptance_smoke.py
      --output-dir "${output_dir}"
      --expected-world-size "${ABPH_DISTRIBUTED_WORLD_SIZE}"
      --device "${DEVICE}")
    ;;
  benchmark)
    [[ "${VARIANT}" == "B1_semantic_query_root" || "${VARIANT}" == "D1_kt32_mh4_particles" ]] || {
      echo "Unsupported representative benchmark variant ${VARIANT}" >&2
      exit 2
    }
    "${PYTHON_BIN}" scripts/validate_adaptive_binary_orchestration.py preflight \
      --path "${ABPH_ROOT}/audits/actual_target_feasibility.json"
    output_dir="${profile_root}/benchmarks/${VARIANT}"
    if [[ "${VARIANT}" == "D1_kt32_mh4_particles" ]]; then
      export ABPH_RENDERER_UPDATES=1
    fi
    cmd=("${PYTHON_BIN}" -u scripts/train_adaptive_binary_pseudooffline_variant.py
      --variant "${VARIANT}"
      --campaign-root "${ABPH_ROOT}"
      --output-dir "${output_dir}"
      --device "${DEVICE}"
      --batch-size "${ABPH_RUNTIME_ACCEPTANCE_LOCAL_BATCH_SIZE:-64}"
      --maximum-updates 20
      --runtime-reference-benchmark)
    ;;
  *) echo "Unknown runtime acceptance mode ${MODE}" >&2; exit 2 ;;
esac

if [[ "${ABPH_JOB_LAUNCHER}" == "srun" ]]; then
  [[ "${SLURM_JOB_NUM_NODES:-0}" == "${ABPH_DISTRIBUTED_NODES}" ]] || {
    echo "allocated node count differs from the acceptance contract" >&2
    exit 2
  }
  [[ "${SLURM_NTASKS:-0}" == "${ABPH_DISTRIBUTED_NTASKS}" ]] || {
    echo "allocated task count differs from the acceptance contract" >&2
    exit 2
  }
  mapfile -t allocated_hosts < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
  export MASTER_ADDR="${allocated_hosts[0]:?No DDP master host found}"
  numeric_job_id="${SLURM_JOB_ID%%_*}"
  [[ "${numeric_job_id}" =~ ^[0-9]+$ ]] || { echo "Invalid Slurm job id" >&2; exit 2; }
  export MASTER_PORT="$((20000 + numeric_job_id % 20000))"
  fresh_run srun --nodes=4 --ntasks=4 --ntasks-per-node=1 \
    --kill-on-bad-exit=1 --cpu-bind=cores --export=ALL "${cmd[@]}"
else
  fresh_run "${cmd[@]}"
fi
