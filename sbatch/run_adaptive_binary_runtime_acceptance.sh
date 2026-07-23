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
# This worker is Tigris-specific. Pin its operational defaults before
# common.sh supplies the repository-wide Weaver/legacy-data fallbacks. The
# ABPH-prefixed overrides remain available for deliberate alternate setups.
CONDA_BASE="${ABPH_CONDA_BASE:-/home/ryreu/miniforge3-aarch64}"
CONDA_ENV="${ABPH_CONDA_ENV:-atlas_kd_tigris}"
DATA_DIR="${ABPH_DATA_DIR:-/home/ryreu/atlas/PracticeTagging/data/jetclass_part1}"
export CONDA_BASE CONDA_ENV DATA_DIR
source "${PROJECT_DIR}/sbatch/common.sh"
source "${PROJECT_DIR}/sbatch/adaptive_binary_ddp_launch.sh"
MODE="${1:?Usage: run_adaptive_binary_runtime_acceptance.sh <smoke|benchmark|benchmark_uninstrumented> [variant]}"
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
cleanup_benchmark_checkpoints() {
  [[ "${MODE}" == "benchmark" || "${MODE}" == "benchmark_uninstrumented" ]] || return 0
  [[ -n "${output_dir:-}" ]] || return 0
  local acceptance_root output_root
  acceptance_root="$(realpath -m "${ABPH_RUNTIME_ACCEPTANCE_ROOT}")"
  output_root="$(realpath -m "${output_dir}")"
  if [[ "${output_root}" != "${acceptance_root}"/* ]]; then
    echo "Refusing runtime checkpoint cleanup outside acceptance root: ${output_root}" >&2
    return 0
  fi
  if [[ -d "${output_root}" ]]; then
    find "${output_root}" -type f \
      \( -name '*.pt' -o -name '*.pt.tmp.*' \) -print -delete
  fi
}
trap cleanup_benchmark_checkpoints EXIT

case "${MODE}" in
  smoke)
    output_dir="${profile_root}/transport_smoke"
    cmd=("${PYTHON_BIN}" -u scripts/run_adaptive_binary_ddp_acceptance_smoke.py
      --output-dir "${output_dir}"
      --expected-world-size "${ABPH_DISTRIBUTED_WORLD_SIZE}"
      --device "${DEVICE}")
    ;;
  benchmark|benchmark_uninstrumented)
    [[ "${VARIANT}" == "B1_semantic_query_root" || "${VARIANT}" == "D1_kt32_mh4_particles" ]] || {
      echo "Unsupported representative benchmark variant ${VARIANT}" >&2
      exit 2
    }
    "${PYTHON_BIN}" scripts/validate_adaptive_binary_orchestration.py preflight \
      --path "${ABPH_ROOT}/audits/actual_target_feasibility.json"
    if [[ "${MODE}" == "benchmark_uninstrumented" ]]; then
      [[ "${ABPH_RECONSTRUCTOR_PARALLELISM}" == "single" ]] || {
        echo "The uninstrumented control is a single-rank benchmark" >&2
        exit 2
      }
      [[ "${VARIANT}" == "D1_kt32_mh4_particles" ]] || {
        echo "The uninstrumented control is locked to the deep representative" >&2
        exit 2
      }
      export ABPH_RUNTIME_PROFILE_ENABLED=0
      output_dir="${ABPH_RUNTIME_ACCEPTANCE_ROOT}/single_path/uninstrumented/${VARIANT}"
    else
      export ABPH_RUNTIME_PROFILE_ENABLED=1
      output_dir="${profile_root}/benchmarks/${VARIANT}"
    fi
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

start_ns="$(date +%s%N)"
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
  abph_fresh_run_srun_with_port_retry --nodes=4 --ntasks=4 --ntasks-per-node=1 \
    --kill-on-bad-exit=1 --cpu-bind=cores --export=ALL "${cmd[@]}"
else
  fresh_run "${cmd[@]}"
fi
if [[ "${MODE}" == "benchmark" || "${MODE}" == "benchmark_uninstrumented" ]]; then
  end_ns="$(date +%s%N)"
  elapsed_seconds="$(${PYTHON_BIN} -c 'import sys; print((int(sys.argv[2])-int(sys.argv[1]))/1e9)' "${start_ns}" "${end_ns}")"
  fresh_run "${PYTHON_BIN}" scripts/write_adaptive_binary_runtime_walltime.py \
    --elapsed-seconds "${elapsed_seconds}" \
    --profile-enabled "${ABPH_RUNTIME_PROFILE_ENABLED}" \
    --run-report "${output_dir}/run_report.json" \
    --delete-selected-checkpoint \
    --output "${output_dir}/wall_time.json"
fi
