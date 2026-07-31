#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=retb_common.sh
source "${SCRIPT_DIR}/retb_common.sh"

: "${RETB_MINIATURE:=0}"
: "${RETB_STORAGE_MEASUREMENTS:=${OUTPUT_ROOT}/relation_expert_token_bridge/bootstrap/storage_measurements.json}"
: "${RETB_OPERATIONAL_AUTHORIZATION:=${OUTPUT_ROOT}/relation_expert_token_bridge/bootstrap/full_submission_authorization.json}"
: "${RETB_CPU_CACHE_CONCURRENCY:=12}"
: "${RETB_GPU_EXPERT_CONCURRENCY:=4}"
: "${RETB_GPU_PREDICTOR_CONCURRENCY:=4}"
: "${RETB_GPU_SCALE_CONCURRENCY:=3}"
: "${RETB_GPU_FINAL_CONCURRENCY:=3}"

mode="submit"
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
  --resume)
    echo "Use scripts/plan_retb_resume.py with authenticated completed-node outputs, then resubmit only its ready nodes." >&2
    echo "Automatic state guessing is intentionally disabled." >&2
    exit 2
    ;;
  "")
    ;;
  *)
    echo "Usage: $0 [--dry-run|--smoke-simulate|--smoke-submit|--resume CAMPAIGN_ROOT]" >&2
    exit 2
    ;;
esac

retb_activate
graph_arguments=(
  --output-root "${OUTPUT_ROOT}"
  --cpu-cache-concurrency "${RETB_CPU_CACHE_CONCURRENCY}"
  --gpu-expert-concurrency "${RETB_GPU_EXPERT_CONCURRENCY}"
  --gpu-predictor-concurrency "${RETB_GPU_PREDICTOR_CONCURRENCY}"
  --gpu-scale-concurrency "${RETB_GPU_SCALE_CONCURRENCY}"
  --gpu-final-concurrency "${RETB_GPU_FINAL_CONCURRENCY}"
)
if [[ -f "${RETB_STORAGE_MEASUREMENTS}" ]]; then
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
  exit 2
fi
if [[ "${RETB_MINIATURE}" != "1" ]]; then
  if [[ ! -f "${RETB_OPERATIONAL_AUTHORIZATION}" ]]; then
    echo "Authenticated RETB operational authorization is absent: ${RETB_OPERATIONAL_AUTHORIZATION}" >&2
    echo "Complete local validation, a real miniature Tigris smoke, and the authenticated production dry run first." >&2
    exit 2
  fi
  python scripts/validate_retb_operational_readiness.py \
    verify-authorization \
    --authorization "${RETB_OPERATIONAL_AUTHORIZATION}" >/dev/null
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
    'from teacher_logit_reco.relation_expert_token_bridge.provenance import source_snapshot; from pathlib import Path; s=source_snapshot(Path(".")); print(s["source_commit"]); print(s["source_status_sha256"])'
)
source_commit="${source_fields[0]}"
source_status_sha="${source_fields[1]}"
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
export RETB_STORAGE_MEASUREMENTS
export RETB_OPERATIONAL_AUTHORIZATION

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
  if [[ -n "${dependency}" ]]; then
    dependency_arguments=(--dependency="afterok:${dependency}")
  fi
  local resource_arguments=(
    --account="${SBATCH_ACCOUNT}"
    --partition="${SBATCH_PARTITION}"
    --cpus-per-task="${CPU_CPUS_PER_TASK}"
    --mem="${CPU_MEM}"
  )
  if [[ "${resource}" == "gpu" && "${is_array}" == "0" ]]; then
    resource_arguments+=(
      --gres="${GPU_GRES}"
      --cpus-per-task="${GPU_CPUS_PER_TASK}"
      --mem="${GPU_MEM}"
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
    --export="ALL,CAMPAIGN_ROOT=${campaign_root},CAMPAIGN_ID=${campaign_id},RETB_NODE_ID=${node_id},RETB_NODE_RESOURCE=${resource},RETB_RESOURCE_KIND=${resource},RETB_MINIATURE=${RETB_MINIATURE},RETB_STORAGE_MEASUREMENTS=${RETB_STORAGE_MEASUREMENTS}" \
    "${executable}"
}

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
done < <(python scripts/print_retb_submission_plan.py --production-graph "${graph}")

ledger_arguments=()
binding_strings=()
for name in "${!jobs[@]}"; do
  ledger_arguments+=(--job "${name}=${jobs[${name}]}")
  binding_strings+=("${name}=${jobs[${name}]}")
done
submission_mode="production_submitted"
if [[ "${RETB_MINIATURE}" == "1" ]]; then
  submission_mode="smoke_submitted"
fi
python scripts/write_retb_job_ledger.py \
  --production-graph "${graph}" \
  --submission-mode "${submission_mode}" \
  "${ledger_arguments[@]}" \
  --output "${campaign_root}/job_ledgers/initial_submission_ledger.json"

printf 'campaign root: %s\n' "${campaign_root}"
printf 'source commit: %s\nsource dirty-status hash: %s\n' \
  "${source_commit}" "${source_status_sha}"
printf 'degradation profile: D_NOMINAL\n'
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
printf 'HLT-v3 cache hashes: %s/inputs/hlt_v3/**/hlt_v3_metadata.json\n' "${campaign_root}"
printf 'selection locks: %s/selection/locked_scale_finalists.json and %s/selection/final_test_execution_lock.json\n' \
  "${campaign_root}" "${campaign_root}"
printf 'monitor: python scripts/monitor_retb_campaign.py --campaign-root %q\n' "${campaign_root}"
printf 'accounting: sacct -X --starttime today --name retb_ --format=JobID,JobName,State,Elapsed,ExitCode\n'
printf 'cancel stale only: python scripts/monitor_retb_campaign.py --campaign-root %q --cancel-stale --stale-job-id JOBID\n' "${campaign_root}"
printf 'download: rsync -av tigris:%s/reports/ ./retb_reports/\n' "${campaign_root}"
