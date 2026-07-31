#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${DATA_DIR:=/home/ryreu/atlas/PracticeTagging/data}"
: "${OUTPUT_ROOT:=/home/ryreu/atlas/Fresh_check/checkpoints}"
: "${CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
: "${CONDA_ENV:=atlas_kd_tigris}"
: "${SBATCH_ACCOUNT:=reu-aisocial}"
: "${SBATCH_PARTITION:=tigris}"
: "${GPU_GRES:=gpu:gh200:1}"
: "${GPU_CPUS_PER_TASK:=16}"
: "${GPU_MEM:=220G}"
: "${CPU_CPUS_PER_TASK:=16}"
: "${CPU_MEM:=192G}"
: "${RETB_DEVICE:=auto}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1

retb_activate() {
  local conda_hook="${CONDA_BASE}/etc/profile.d/conda.sh"
  if [[ ! -f "${conda_hook}" ]]; then
    echo "Conda activation hook is absent: ${conda_hook}" >&2
    exit 2
  fi
  # shellcheck disable=SC1090
  source "${conda_hook}"
  conda activate "${CONDA_ENV}"
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  cd "${PROJECT_DIR}"
  python -c 'import sys; assert sys.version_info[:2] == (3, 10); print(sys.executable)'
}

retb_require_campaign_root() {
  : "${CAMPAIGN_ROOT:?CAMPAIGN_ROOT is required}"
  if [[ ! -d "${CAMPAIGN_ROOT}" ]]; then
    echo "Campaign root does not exist: ${CAMPAIGN_ROOT}" >&2
    exit 2
  fi
}

retb_setup() {
  retb_require_campaign_root
  retb_activate
  python -c \
    'from pathlib import Path; from teacher_logit_reco.relation_expert_token_bridge.workflow import load_and_validate_campaign_source; load_and_validate_campaign_source(Path("'"${CAMPAIGN_ROOT}"'"), repo_root=Path("'"${PROJECT_DIR}"'"))'
}

retb_task_manifest_path() {
  local node_id="$1"
  python -c \
    'import json,sys; from teacher_logit_reco.relation_expert_token_bridge.production import task_manifest_path_for_graph; graph=json.load(open(sys.argv[1])); print(task_manifest_path_for_graph(graph,node_id=sys.argv[2],campaign_root=sys.argv[3]))' \
    "${CAMPAIGN_ROOT}/job_ledgers/production_graph.json" \
    "${node_id}" \
    "${CAMPAIGN_ROOT}"
}

retb_run_task() {
  local node_id="$1"
  local manifest="${RETB_TASK_MANIFEST:-$(retb_task_manifest_path "${node_id}")}"
  if [[ ! -f "${manifest}" ]]; then
    echo "Authenticated task manifest is absent: ${manifest}" >&2
    exit 2
  fi
  local arguments=(
    --campaign-root "${CAMPAIGN_ROOT}"
    --task-manifest "${manifest}"
  )
  if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    arguments+=(--task-index "${SLURM_ARRAY_TASK_ID}")
  else
    arguments+=(--task-index "${RETB_TASK_INDEX:-0}")
  fi
  if [[ "${RETB_DRY_RUN:-0}" == "1" ]]; then
    arguments+=(--dry-run)
  fi
  python scripts/run_retb_task.py "${arguments[@]}"
}

retb_materialize_downstream() {
  local producer_node_id="$1"
  if [[ "${RETB_DRY_RUN:-0}" == "1" ]]; then
    return 0
  fi
  python scripts/produce_retb_downstream_manifest_plans.py \
    --campaign-root "${CAMPAIGN_ROOT}" \
    --producer-node-id "${producer_node_id}" >/dev/null
  python scripts/materialize_retb_downstream_manifests.py \
    --campaign-root "${CAMPAIGN_ROOT}" \
    --producer-node-id "${producer_node_id}" >/dev/null
}

retb_record_dynamic_job() {
  local logical_name="$1"
  local job_id="$2"
  local dependency="$3"
  local manifest_sha="${4:-}"
  local ledger="${CAMPAIGN_ROOT}/job_ledgers/dynamic_jobs.json"
  local arguments=(
    --campaign-root "${CAMPAIGN_ROOT}"
    --production-graph "${CAMPAIGN_ROOT}/job_ledgers/production_graph.json"
    --ledger "${ledger}"
    --logical-name "${logical_name}"
    --job-id "${job_id}"
    --dependency "${dependency}"
  )
  if [[ -n "${manifest_sha}" ]]; then
    arguments+=(--task-manifest-sha256 "${manifest_sha}")
  fi
  python scripts/register_retb_dynamic_job.py "${arguments[@]}" >/dev/null
}

retb_submit_dynamic_once() {
  local logical_name="$1"
  local dependency="$2"
  local task_manifest="$3"
  shift 3
  local ledger="${CAMPAIGN_ROOT}/job_ledgers/dynamic_jobs.json"
  local lock="${CAMPAIGN_ROOT}/job_ledgers/dynamic_jobs.lock"
  local existing=""
  local job_id=""
  local manifest_sha=""
  (
    flock 9
    existing="$(python scripts/register_retb_dynamic_job.py \
      --campaign-root "${CAMPAIGN_ROOT}" \
      --production-graph "${CAMPAIGN_ROOT}/job_ledgers/production_graph.json" \
      --ledger "${ledger}" \
      --logical-name "${logical_name}" \
      --query)"
    if [[ -n "${existing}" ]]; then
      printf '%s\n' "${existing}"
      exit 0
    fi
    manifest_sha="$(python -c \
      'import json,sys; print(json.load(open(sys.argv[1]))["content_hash"])' \
      "${task_manifest}")"
    job_id="$(sbatch --parsable "$@")"
    retb_record_dynamic_job \
      "${logical_name}" "${job_id}" "${dependency}" "${manifest_sha}"
    printf '%s\n' "${job_id}"
  ) 9>"${lock}"
}
