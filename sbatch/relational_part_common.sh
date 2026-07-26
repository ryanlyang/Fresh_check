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
  cd "${PROJECT_DIR}"
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
