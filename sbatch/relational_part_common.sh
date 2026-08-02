#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${DATA_DIR:=/home/ryreu/atlas/PracticeTagging/data}"
: "${OUTPUT_ROOT:=/home/ryreu/atlas/Fresh_check/checkpoints}"
: "${CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
: "${CONDA_ENV:=atlas_kd_tigris}"
: "${PYTHONNOUSERSITE:=1}"
: "${SBATCH_ACCOUNT:=reu-aisocial}"
: "${SBATCH_PARTITION:=tigris}"
: "${GPU_GRES:=gpu:gh200:1}"
: "${GPU_CPUS_PER_TASK:=16}"
: "${GPU_MEM:=220G}"
: "${CPU_CPUS_PER_TASK:=16}"
: "${CPU_MEM:=192G}"
: "${RPT_TREE_SHARD_SIZE:=10000}"
: "${RPT_DEVICE:=auto}"
export PYTHONNOUSERSITE
export PYTHONDONTWRITEBYTECODE=1

rpt_require_campaign_root() {
  if [[ -z "${CAMPAIGN_ROOT:-}" ]]; then
    echo "CAMPAIGN_ROOT is required" >&2
    exit 2
  fi
  if [[ ! -d "${CAMPAIGN_ROOT}" ]]; then
    echo "Campaign root does not exist: ${CAMPAIGN_ROOT}" >&2
    exit 2
  fi
}

rpt_setup() {
  rpt_require_campaign_root
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
  if [[ -n "${RPT_SOURCE_RECOVERY_AUTHORIZATION:-}" ]]; then
    python -c \
      'from pathlib import Path; from teacher_logit_reco.relational_part import load_hashed_json, validate_campaign_source; root=Path("'"${CAMPAIGN_ROOT}"'"); validate_campaign_source(load_hashed_json(root / "campaign_spec.json"), repo_root=Path.cwd()); print("authenticated RPT source recovery")'
    python -c "import sys; print(sys.executable)"
    return 0
  fi
  local production_graph="${CAMPAIGN_ROOT}/job_ledgers/production_graph.json"
  if [[ ! -f "${production_graph}" ]]; then
    echo "Production graph is absent: ${production_graph}" >&2
    exit 2
  fi
  local expected_project=""
  local expected_commit=""
  expected_project="$(
    python -c \
      'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["execution_source"]["root"])' \
      "${production_graph}"
  )"
  expected_commit="$(
    python -c \
      'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["execution_source"]["pinned_commit"])' \
      "${production_graph}"
  )"
  local actual_project=""
  local actual_commit=""
  actual_project="$(pwd -P)"
  actual_commit="$(git rev-parse HEAD)"
  if [[ "${actual_project}" != "${expected_project}" ]]; then
    echo "Worker source root differs from the pinned production graph." >&2
    echo "Expected: ${expected_project}" >&2
    echo "Observed: ${actual_project}" >&2
    exit 2
  fi
  if [[ "${actual_commit}" != "${expected_commit}" ]]; then
    echo "Worker source commit differs from the pinned production graph." >&2
    exit 2
  fi
  if [[ -n "$(git status --porcelain=v1 --untracked-files=all)" ]]; then
    echo "Pinned campaign source worktree is dirty." >&2
    exit 2
  fi
  python -c "import sys; print(sys.executable)"
}

rpt_field() {
  python scripts/query_relational_part_artifact.py "$1" "$2"
}

rpt_record_dynamic_job() {
  local logical_name="$1"
  local job_id="$2"
  local dependency="$3"
  local ledger="${CAMPAIGN_ROOT}/job_ledgers/dynamic_jobs.tsv"
  (
    flock 9
    printf '%s\t%s\t%s\n' "${logical_name}" "${job_id}" "${dependency}" >&9
  ) 9>>"${ledger}"
}

rpt_submit_dynamic_once() {
  local logical_name="$1"
  local dependency="$2"
  shift 2
  local ledger="${CAMPAIGN_ROOT}/job_ledgers/dynamic_jobs.json"
  local lock="${CAMPAIGN_ROOT}/job_ledgers/dynamic_jobs.lock"
  local job_name="${CAMPAIGN_ID:-rpt}_${logical_name}"
  local existing=""
  local recovered=""
  local job_id=""
  (
    flock 9
    existing="$(python scripts/register_relational_part_dynamic_job.py \
      --ledger "${ledger}" \
      --campaign-spec "${CAMPAIGN_ROOT}/campaign_spec.json" \
      --logical-name "${logical_name}" \
      --query)"
    if [[ -n "${existing}" ]]; then
      printf '%s\n' "${existing}"
      exit 0
    fi
    recovered="$(squeue --noheader --name="${job_name}" --format='%A' \
      | head -n 1 | tr -d '[:space:]')"
    if [[ -z "${recovered}" ]]; then
      recovered="$(sacct --noheader --name="${job_name}" \
        --starttime=1970-01-01 --format=JobIDRaw \
        | awk 'NF && $1 !~ /[._]/ {print $1; exit}')"
    fi
    if [[ -n "${recovered}" && "${recovered}" =~ ^[0-9]+$ ]]; then
      job_id="${recovered}"
    else
      job_id="$(sbatch --parsable --job-name="${job_name}" "$@")"
    fi
    python scripts/register_relational_part_dynamic_job.py \
      --ledger "${ledger}" \
      --campaign-spec "${CAMPAIGN_ROOT}/campaign_spec.json" \
      --logical-name "${logical_name}" \
      --job-id "${job_id}" \
      --dependency "${dependency}" >/dev/null
    rpt_record_dynamic_job "${logical_name}" "${job_id}" "${dependency}"
    printf '%s\n' "${job_id}"
  ) 9>"${lock}"
}

rpt_hlt_hash_args() {
  local binding="${CAMPAIGN_ROOT}/inputs/hlt_cache_audit.json"
  local split
  for split in model_train model_val stack_val final_test; do
    printf '%s=%s\n' \
      "${split}" \
      "$(rpt_field "${binding}" "split_reports.${split}.hlt_content_hash")"
  done
}

rpt_tree_root() {
  printf '%s\n' "${CAMPAIGN_ROOT}/inputs/relation_tree_cache"
}
