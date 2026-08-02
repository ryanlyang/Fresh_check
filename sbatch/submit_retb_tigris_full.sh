#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=retb_common.sh
source "${SCRIPT_DIR}/retb_common.sh"

: "${RETB_MINIATURE:=0}"
: "${RETB_SUBMISSION_SCOPE:=complete}"
: "${RETB_STORAGE_MEASUREMENTS:=${OUTPUT_ROOT}/relation_expert_token_bridge/bootstrap/storage_measurements.json}"
: "${RETB_OPERATIONAL_AUTHORIZATION:=${OUTPUT_ROOT}/relation_expert_token_bridge/bootstrap/full_submission_authorization.json}"
: "${RETB_CPU_CACHE_CONCURRENCY:=64}"
: "${RETB_GPU_EXPERT_CONCURRENCY:=64}"
: "${RETB_GPU_PREDICTOR_CONCURRENCY:=64}"
: "${RETB_GPU_SCALE_CONCURRENCY:=64}"
: "${RETB_GPU_FINAL_CONCURRENCY:=64}"
export RETB_CPU_CACHE_CONCURRENCY
export RETB_GPU_EXPERT_CONCURRENCY
export RETB_GPU_PREDICTOR_CONCURRENCY
export RETB_GPU_SCALE_CONCURRENCY
export RETB_GPU_FINAL_CONCURRENCY

mode="submit"
RETB_SUBMISSION_SCOPE="complete"
case "${1:-}" in
  --dry-run) mode="dry-run" ;;
  --smoke-simulate)
    mode="smoke-simulate"
    RETB_MINIATURE=1
    ;;
  --smoke-submit)
    mode="submit"
    RETB_MINIATURE=1
    ;;
  --offline-submit)
    mode="submit"
    RETB_MINIATURE=0
    RETB_SUBMISSION_SCOPE="offline_abc"
    ;;
  --offline-streamed-submit)
    mode="submit"
    RETB_MINIATURE=0
    RETB_SUBMISSION_SCOPE="offline_abc_streamed"
    ;;
  --streamed-submit)
    mode="submit"
    RETB_MINIATURE=0
    RETB_SUBMISSION_SCOPE="full_streamed"
    ;;
  --streamed-smoke-submit)
    mode="submit"
    RETB_MINIATURE=1
    RETB_SUBMISSION_SCOPE="streamed_smoke"
    ;;
  --resume)
    echo "Use scripts/plan_retb_resume.py with authenticated completed-node outputs, then resubmit only its ready nodes." >&2
    echo "Automatic state guessing is intentionally disabled." >&2
    exit 2
    ;;
  "")
    ;;
  *)
    echo "Usage: $0 [--dry-run|--smoke-simulate|--smoke-submit|--offline-submit|--offline-streamed-submit|--streamed-submit|--streamed-smoke-submit|--resume CAMPAIGN_ROOT]" >&2
    exit 2
    ;;
esac

# A submitted campaign never executes from the mutable checkout.  Re-enter the
# committed launcher from a detached worktree before any campaign artifact or
# Slurm job is created.  The user's main checkout may then change freely.
if [[ "${mode}" == "submit" && "${RETB_FROZEN_REENTRY:-0}" != "1" ]]; then
  submission_project_dir="$(git -C "${PROJECT_DIR}" rev-parse --show-toplevel)"
  frozen_source_commit="$(git -C "${submission_project_dir}" rev-parse HEAD)"
  : "${RETB_SOURCE_WORKTREE_ROOT:=${submission_project_dir%/*}/retb_source_worktrees}"
  worktree_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  frozen_project_dir="${RETB_SOURCE_WORKTREE_ROOT}/retb_${worktree_stamp}_${frozen_source_commit:0:10}_$$"
  case "${frozen_project_dir}/" in
    "${submission_project_dir}/"*)
      echo "RETB source worktrees must be outside the mutable checkout: ${frozen_project_dir}" >&2
      exit 2
      ;;
  esac
  if [[ -e "${frozen_project_dir}" ]]; then
    echo "Refusing existing frozen source worktree: ${frozen_project_dir}" >&2
    exit 2
  fi
  mkdir -p "${RETB_SOURCE_WORKTREE_ROOT}"
  git -C "${submission_project_dir}" worktree add --detach \
    "${frozen_project_dir}" "${frozen_source_commit}" >/dev/null
  exec env \
    PROJECT_DIR="${frozen_project_dir}" \
    DATA_DIR="${DATA_DIR}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    CONDA_BASE="${CONDA_BASE}" \
    CONDA_ENV="${CONDA_ENV}" \
    SBATCH_ACCOUNT="${SBATCH_ACCOUNT}" \
    SBATCH_PARTITION="${SBATCH_PARTITION}" \
    GPU_GRES="${GPU_GRES}" \
    GPU_CPUS_PER_TASK="${GPU_CPUS_PER_TASK}" \
    GPU_MEM="${GPU_MEM}" \
    CPU_CPUS_PER_TASK="${CPU_CPUS_PER_TASK}" \
    CPU_MEM="${CPU_MEM}" \
    RETB_SMOKE_GPU_CPUS_PER_TASK="${RETB_SMOKE_GPU_CPUS_PER_TASK}" \
    RETB_SMOKE_GPU_MEM="${RETB_SMOKE_GPU_MEM}" \
    RETB_SMOKE_CPU_CPUS_PER_TASK="${RETB_SMOKE_CPU_CPUS_PER_TASK}" \
    RETB_SMOKE_CPU_MEM="${RETB_SMOKE_CPU_MEM}" \
    RETB_DEVICE="${RETB_DEVICE}" \
    RETB_MINIATURE="${RETB_MINIATURE}" \
    RETB_SUBMISSION_SCOPE="${RETB_SUBMISSION_SCOPE}" \
    RETB_STORAGE_MEASUREMENTS="${RETB_STORAGE_MEASUREMENTS}" \
    RETB_OPERATIONAL_AUTHORIZATION="${RETB_OPERATIONAL_AUTHORIZATION}" \
    RETB_SUBMISSION_PROJECT_DIR="${submission_project_dir}" \
    RETB_FROZEN_SOURCE_COMMIT="${frozen_source_commit}" \
    RETB_FROZEN_REENTRY=1 \
    RETB_SOURCE_WORKTREE_ROOT="${RETB_SOURCE_WORKTREE_ROOT}" \
    bash "${frozen_project_dir}/sbatch/submit_retb_tigris_full.sh" "$@"
fi

retb_activate
graph_arguments=(
  --output-root "${OUTPUT_ROOT}"
  --submission-scope "${RETB_SUBMISSION_SCOPE}"
  --cpu-cache-concurrency "${RETB_CPU_CACHE_CONCURRENCY}"
  --gpu-expert-concurrency "${RETB_GPU_EXPERT_CONCURRENCY}"
  --gpu-predictor-concurrency "${RETB_GPU_PREDICTOR_CONCURRENCY}"
  --gpu-scale-concurrency "${RETB_GPU_SCALE_CONCURRENCY}"
  --gpu-final-concurrency "${RETB_GPU_FINAL_CONCURRENCY}"
)
if [[ "${RETB_MINIATURE}" == "1" ]]; then
  # Miniature campaigns are bound to the deterministic miniature storage
  # contract in both the graph and Step-1 campaign builder.  A production
  # measurement may legitimately exist at the default path, but allowing it
  # into only the graph would make campaign bootstrap fail its lineage check.
  :
elif [[ -f "${RETB_STORAGE_MEASUREMENTS}" ]]; then
  graph_arguments+=(--storage-measurements "${RETB_STORAGE_MEASUREMENTS}")
elif [[ "${mode}" == "dry-run" ]]; then
  graph_arguments+=(--storage-measurements-sha256 "$(printf '0%.0s' {1..64})")
fi
if [[ "${RETB_MINIATURE}" == "1" ]]; then
  graph_arguments+=(--miniature)
fi
if [[ "${mode}" == "dry-run" ]]; then
  python scripts/submit_retb_graph.py "${graph_arguments[@]}" --dry-run
  exit 0
fi
if [[ "${mode}" == "smoke-simulate" ]]; then
  python scripts/submit_retb_graph.py \
    "${graph_arguments[@]}" --smoke-simulate
  exit 0
fi

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "JetClass data root does not exist: ${DATA_DIR}" >&2
  exit 2
fi
if [[ "${RETB_MINIATURE}" != "1" && ! -f "${RETB_STORAGE_MEASUREMENTS}" ]]; then
  echo "Authenticated RETB storage measurements are absent: ${RETB_STORAGE_MEASUREMENTS}" >&2
  if [[ "${RETB_SUBMISSION_SCOPE}" == "offline_abc_streamed" ]]; then
    echo "Create them with scripts/measure_retb_streamed_abc_storage.py using a completed RETB evidence campaign." >&2
  fi
  exit 2
fi
if [[ "${RETB_SUBMISSION_SCOPE}" == "full_streamed" ]]; then
  python scripts/validate_retb_full_streamed_storage.py \
    --measurements "${RETB_STORAGE_MEASUREMENTS}" >/dev/null
fi
if [[ "${RETB_MINIATURE}" != "1" && "${RETB_SUBMISSION_SCOPE}" =~ ^(complete|full_streamed)$ ]]; then
  if [[ ! -f "${RETB_OPERATIONAL_AUTHORIZATION}" ]]; then
    echo "Authenticated RETB operational authorization is absent: ${RETB_OPERATIONAL_AUTHORIZATION}" >&2
    echo "Complete local validation, a real miniature Tigris smoke, and the authenticated production dry run first." >&2
    exit 2
  fi
  python scripts/validate_retb_operational_readiness.py \
    verify-authorization \
    --authorization "${RETB_OPERATIONAL_AUTHORIZATION}" >/dev/null
elif [[ "${RETB_SUBMISSION_SCOPE}" == "offline_abc" ]]; then
  printf 'Submitting authenticated real-data Stages A-C only; Stages D-N are excluded.\n'
elif [[ "${RETB_SUBMISSION_SCOPE}" == "offline_abc_streamed" ]]; then
  printf 'Submitting authenticated streamed real-data Stages A-C; future inputs are deferred and frozen-token banks are task-local.\n'
fi
required_per_class=345000
if [[ "${RETB_MINIATURE}" == "1" ]]; then
  required_per_class=9
fi
python scripts/preflight_relational_part_data.py \
  --data-dir "${DATA_DIR}" \
  --tree-name tree \
  --required-per-class "${required_per_class}"
python -c \
  'from torch.utils.cpp_extension import verify_ninja_availability; verify_ninja_availability(); print("PyTorch C++ extension toolchain: ninja OK")'

readarray -t source_fields < <(
  python -c \
    'from teacher_logit_reco.relation_expert_token_bridge.provenance import source_snapshot; from pathlib import Path; s=source_snapshot(Path(".")); print(s["source_commit"]); print(s["source_status_sha256"]); print("1" if s["source_dirty"] else "0")'
)
source_commit="${source_fields[0]}"
source_status_sha="${source_fields[1]}"
source_dirty="${source_fields[2]}"
if [[ "${source_commit}" != "${RETB_FROZEN_SOURCE_COMMIT}" || "${source_dirty}" != "0" ]]; then
  echo "Frozen RETB source checkout is not the clean bound commit" >&2
  exit 2
fi
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
campaign_id="retb_relation_expert_bridge_${timestamp}_${source_commit:0:10}_${source_status_sha:0:10}"
campaign_root="${OUTPUT_ROOT}/relation_expert_token_bridge/${campaign_id}"
if [[ -e "${campaign_root}" ]]; then
  echo "Refusing existing campaign root: ${campaign_root}" >&2
  exit 2
fi
mkdir -p \
  "${campaign_root}/bootstrap" \
  "${campaign_root}/job_ledgers/slurm" \
  "${campaign_root}/job_ledgers/tasks"
export CAMPAIGN_ID="${campaign_id}"
export CAMPAIGN_ROOT="${campaign_root}"
export RETB_MINIATURE
export RETB_SUBMISSION_SCOPE
export RETB_STORAGE_MEASUREMENTS
export RETB_OPERATIONAL_AUTHORIZATION
export RETB_FROZEN_REENTRY
export RETB_FROZEN_SOURCE_COMMIT
export RETB_SUBMISSION_PROJECT_DIR
export RETB_SOURCE_WORKTREE_ROOT

python scripts/submit_retb_graph.py \
  "${graph_arguments[@]}" \
  --campaign-id "${campaign_id}" \
  --campaign-root "${campaign_root}" \
  --write-artifacts

graph="${campaign_root}/job_ledgers/production_graph.json"
log_pattern="${campaign_root}/job_ledgers/slurm/%x_%A_%a.out"
error_pattern="${campaign_root}/job_ledgers/slurm/%x_%A_%a.err"

submit_node() {
  local node_id="$1"
  local dependency="$2"
  local resource="$3"
  local worker="$4"
  local is_array="$5"
  local dispatch_mode="$6"
  local dependency_arguments=()
  local cpu_count="${CPU_CPUS_PER_TASK}"
  local memory="${CPU_MEM}"
  local gpu_cpu_count="${GPU_CPUS_PER_TASK}"
  local gpu_memory="${GPU_MEM}"
  if [[ "${RETB_SUBMISSION_SCOPE}" == "streamed_smoke" ]]; then
    cpu_count="${RETB_SMOKE_CPU_CPUS_PER_TASK}"
    memory="${RETB_SMOKE_CPU_MEM}"
    gpu_cpu_count="${RETB_SMOKE_GPU_CPUS_PER_TASK}"
    gpu_memory="${RETB_SMOKE_GPU_MEM}"
  fi
  if [[ -n "${dependency}" ]]; then
    dependency_arguments=(--dependency="afterok:${dependency}")
  fi
  local resource_arguments=(
    --account="${SBATCH_ACCOUNT}"
    --partition="${SBATCH_PARTITION}"
    --cpus-per-task="${cpu_count}"
    --mem="${memory}"
  )
  if [[ "${resource}" == "gpu" && "${is_array}" == "0" ]]; then
    resource_arguments+=(
      --gres="${GPU_GRES}"
      --cpus-per-task="${gpu_cpu_count}"
      --mem="${gpu_memory}"
    )
  fi
  local executable="${SCRIPT_DIR}/${worker}"
  case "${dispatch_mode}" in
    direct_worker)
      ;;
    task_manifest_worker)
      if [[ "${is_array}" == "1" ]]; then
        executable="${SCRIPT_DIR}/run_retb_array_launcher.sh"
      else
        executable="${SCRIPT_DIR}/run_retb_production_task.sh"
      fi
      ;;
    *)
      echo "Unsupported RETB dispatch mode for ${node_id}: ${dispatch_mode}" >&2
      return 2
      ;;
  esac
  sbatch --parsable \
    "${resource_arguments[@]}" \
    "${dependency_arguments[@]}" \
    --job-name="${campaign_id}_${node_id}" \
    --output="${log_pattern}" \
    --error="${error_pattern}" \
    --export="ALL,PROJECT_DIR=${PROJECT_DIR},CAMPAIGN_ROOT=${campaign_root},CAMPAIGN_ID=${campaign_id},RETB_NODE_ID=${node_id},RETB_NODE_RESOURCE=${resource},RETB_RESOURCE_KIND=${resource},RETB_MINIATURE=${RETB_MINIATURE},RETB_SUBMISSION_SCOPE=${RETB_SUBMISSION_SCOPE},RETB_STORAGE_MEASUREMENTS=${RETB_STORAGE_MEASUREMENTS},RETB_FROZEN_REENTRY=1,RETB_FROZEN_SOURCE_COMMIT=${RETB_FROZEN_SOURCE_COMMIT},RETB_SUBMISSION_PROJECT_DIR=${RETB_SUBMISSION_PROJECT_DIR},RETB_SOURCE_WORKTREE_ROOT=${RETB_SOURCE_WORKTREE_ROOT}" \
    "${executable}"
}

if [[ "${RETB_SUBMISSION_SCOPE}" == "streamed_smoke" ]]; then
  declare -a smoke_bindings=()
  split_job="$(submit_node split_build "" cpu run_retb_build_splits.sh 0 direct_worker)"
  smoke_bindings+=("split_build=${split_job}")
  bootstrap_job="$(submit_node campaign_bootstrap "${split_job}" cpu run_retb_build_campaign.sh 0 direct_worker)"
  smoke_bindings+=("campaign_bootstrap=${bootstrap_job}")
  previous_job="${bootstrap_job}"
  while IFS='|' read -r phase_id stage resource kind; do
    resource_arguments=(
      --account="${SBATCH_ACCOUNT}" --partition="${SBATCH_PARTITION}"
      --cpus-per-task="${RETB_SMOKE_CPU_CPUS_PER_TASK}"
      --mem="${RETB_SMOKE_CPU_MEM}"
    )
    if [[ "${resource}" == "gpu" ]]; then
      resource_arguments+=(
        --gres="${GPU_GRES}"
        --cpus-per-task="${RETB_SMOKE_GPU_CPUS_PER_TASK}"
        --mem="${RETB_SMOKE_GPU_MEM}"
      )
    fi
    phase_job="$(sbatch --parsable "${resource_arguments[@]}" \
      --dependency="afterok:${previous_job}" \
      --job-name="${campaign_id}_streamed_smoke_${phase_id}" \
      --output="${log_pattern}" --error="${error_pattern}" \
      --export="ALL,PROJECT_DIR=${PROJECT_DIR},CAMPAIGN_ROOT=${campaign_root},CAMPAIGN_ID=${campaign_id},RETB_SMOKE_PHASE_ID=${phase_id},RETB_NODE_RESOURCE=${resource},RETB_SUBMISSION_SCOPE=streamed_smoke,RETB_FROZEN_REENTRY=1,RETB_FROZEN_SOURCE_COMMIT=${RETB_FROZEN_SOURCE_COMMIT},RETB_SUBMISSION_PROJECT_DIR=${RETB_SUBMISSION_PROJECT_DIR},RETB_SOURCE_WORKTREE_ROOT=${RETB_SOURCE_WORKTREE_ROOT}" \
      "${SCRIPT_DIR}/run_retb_streamed_smoke_phase.sh")"
    smoke_bindings+=("${phase_id}=${phase_job}")
    previous_job="${phase_job}"
    printf 'submitted compact smoke Stage %s %-24s %s (%s)\n' "${stage}" "${phase_id}" "${phase_job}" "${kind}"
  done < <(python scripts/print_retb_streamed_smoke_phases.py)
  smoke_ledger_arguments=()
  for binding in "${smoke_bindings[@]}"; do
    smoke_ledger_arguments+=(--job "${binding}")
  done
  python scripts/write_retb_streamed_smoke_ledger.py \
    --production-graph "${graph}" \
    "${smoke_ledger_arguments[@]}" \
    --output "${campaign_root}/job_ledgers/streamed_smoke_submission_ledger.json"
  printf 'campaign root: %s\ncompact streamed smoke allocations: %s\n' "${campaign_root}" "${#smoke_bindings[@]}"
  printf 'monitor: squeue -u "$USER" -o "%%i %%j %%T %%R" | grep %q\n' "${campaign_id}"
  exit 0
fi

declare -A jobs
while IFS='|' read -r node_id stage dependencies resource worker is_array alias dispatch_mode; do
  if [[ -n "${alias}" ]]; then
    if [[ -z "${jobs[${alias}]:-}" ]]; then
      echo "Virtual alias ${node_id} precedes ${alias}" >&2
      exit 2
    fi
    jobs["${node_id}"]="${jobs[${alias}]}"
    printf 'aliased  Stage %s %-36s %s (same selector job as %s)\n' \
      "${stage}" "${node_id}" "${jobs[${node_id}]}" "${alias}"
    continue
  fi
  dependency_ids=""
  if [[ -n "${dependencies}" ]]; then
    IFS=':' read -r -a parent_names <<< "${dependencies}"
    parent_ids=()
    for parent in "${parent_names[@]}"; do
      if [[ -z "${jobs[${parent}]:-}" ]]; then
        echo "Dependency ${parent} is unbound for ${node_id}" >&2
        exit 2
      fi
      parent_ids+=("${jobs[${parent}]}")
    done
    dependency_ids="$(IFS=:; printf '%s' "${parent_ids[*]}")"
  fi
  jobs["${node_id}"]="$(submit_node \
    "${node_id}" "${dependency_ids}" "${resource}" "${worker}" "${is_array}" \
    "${dispatch_mode}")"
  printf 'submitted Stage %s %-36s %s\n' \
    "${stage}" "${node_id}" "${jobs[${node_id}]}"
done < <(python scripts/print_retb_submission_plan.py \
  --production-graph "${graph}" \
  --submission-scope "${RETB_SUBMISSION_SCOPE}")

ledger_arguments=()
binding_strings=()
for name in "${!jobs[@]}"; do
  ledger_arguments+=(--job "${name}=${jobs[${name}]}")
  binding_strings+=("${name}=${jobs[${name}]}")
done
submission_mode="production_submitted"
if [[ "${RETB_MINIATURE}" == "1" ]]; then
  submission_mode="smoke_submitted"
elif [[ "${RETB_SUBMISSION_SCOPE}" == "offline_abc" ]]; then
  submission_mode="offline_production_submitted"
elif [[ "${RETB_SUBMISSION_SCOPE}" == "offline_abc_streamed" ]]; then
  submission_mode="offline_streamed_production_submitted"
fi
python scripts/write_retb_job_ledger.py \
  --production-graph "${graph}" \
  --submission-mode "${submission_mode}" \
  "${ledger_arguments[@]}" \
  --output "${campaign_root}/job_ledgers/initial_submission_ledger.json"

printf 'campaign root: %s\n' "${campaign_root}"
printf 'source commit: %s\nsource dirty-status hash: %s\n' \
  "${source_commit}" "${source_status_sha}"
printf 'frozen source checkout: %s\nmutable submission checkout: %s\n' \
  "${PROJECT_DIR}" "${RETB_SUBMISSION_PROJECT_DIR}"
printf 'degradation profile: D_NOMINAL\n'
printf 'submission scope: %s\n' "${RETB_SUBMISSION_SCOPE}"
if [[ -f "${RETB_STORAGE_MEASUREMENTS}" ]]; then
  readarray -t storage_fields < <(
    python -c \
      'import json,sys; p=json.load(open(sys.argv[1])); m=p["measurements"]; print(m["projected_peak_concurrent_bytes"]); print(m["available_storage_bytes"]); print(p["content_hash"])' \
      "${RETB_STORAGE_MEASUREMENTS}"
  )
  printf 'storage projection: peak=%s available=%s measurement-sha=%s\n' \
    "${storage_fields[0]}" "${storage_fields[1]}" "${storage_fields[2]}"
else
  printf 'storage projection: miniature authenticated defaults\n'
fi
printf 'bounded concurrency: cpu-cache=%s expert=%s predictor=%s scale=%s final=%s\n' \
  "${RETB_CPU_CACHE_CONCURRENCY}" "${RETB_GPU_EXPERT_CONCURRENCY}" \
  "${RETB_GPU_PREDICTOR_CONCURRENCY}" "${RETB_GPU_SCALE_CONCURRENCY}" \
  "${RETB_GPU_FINAL_CONCURRENCY}"
printf 'ledger: %s/job_ledgers/initial_submission_ledger.json\n' "${campaign_root}"
if [[ "${RETB_SUBMISSION_SCOPE}" == "offline_abc" ]]; then
  printf 'offline scope attestation: %s/job_ledgers/offline_submission_scope.json\n' "${campaign_root}"
  printf 'offline outputs: %s/selection/locked_offline_shapes.json and Stage-C controls\n' "${campaign_root}"
elif [[ "${RETB_SUBMISSION_SCOPE}" == "offline_abc_streamed" ]]; then
  printf 'streamed scope attestation: %s/job_ledgers/streamed_offline_submission_scope.json\n' "${campaign_root}"
  printf 'streamed execution profile: %s/registry/retb_streamed_abc_execution_profile.json\n' "${campaign_root}"
  printf 'offline outputs: %s/selection/locked_offline_shapes.json and Stage-C controls\n' "${campaign_root}"
else
  printf 'HLT-v3 cache hashes: %s/inputs/hlt_v3/**/hlt_v3_metadata.json\n' "${campaign_root}"
  printf 'selection locks: %s/selection/locked_scale_finalists.json and %s/selection/final_test_execution_lock.json\n' \
    "${campaign_root}" "${campaign_root}"
fi
printf 'monitor: python scripts/monitor_retb_campaign.py --campaign-root %q\n' "${campaign_root}"
printf 'accounting: sacct -X --starttime today --name retb_ --format=JobID,JobName,State,Elapsed,ExitCode\n'
printf 'cancel stale only: python scripts/monitor_retb_campaign.py --campaign-root %q --cancel-stale --stale-job-id JOBID\n' "${campaign_root}"
printf 'download: rsync -av tigris:%s/reports/ ./retb_reports/\n' "${campaign_root}"
