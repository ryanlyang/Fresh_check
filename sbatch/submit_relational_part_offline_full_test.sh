#!/usr/bin/env bash
# Independent inference-only supplemental evaluation; never queues training.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
: "${RPT_FULL_TEST_PARENT:=/home/ryreu/atlas/Fresh_check/checkpoints/relational_particle_transformer/rpt_offline_transfer_20260809T195641Z_0737194d3c}"
: "${RPT_FULL_TEST_DATA:=/home/ryreu/atlas/jetclass_test20m/test_20M}"
: "${RPT_FULL_TEST_JOBS:=20}"
: "${RPT_FULL_TEST_CONCURRENCY:=8}"
: "${SBATCH_ACCOUNT:=reu-aisocial}"
: "${SBATCH_PARTITION:=tigris}"
: "${GPU_GRES:=gpu:gh200:1}"
: "${CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
: "${CONDA_ENV:=atlas_kd_tigris}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

usage() {
  echo "Usage: $0 [--dry-run | --resume OUTPUT_DIRECTORY]"
}
mode="${1:-submit}"
if [[ "${mode}" == --resume ]]; then
  [[ $# == 2 ]] || { usage; exit 2; }
  export RPT_FULL_TEST_OUTPUT="$(realpath -- "${2}")"
  test -f "${RPT_FULL_TEST_OUTPUT}/submission_config.json"
  mapfile -t saved_config < <(python -c 'import json,sys; c=json.load(open(sys.argv[1])); print(c["parent"]); print(c["source"]); print(c["data"]); print(c["jobs"])' "${RPT_FULL_TEST_OUTPUT}/submission_config.json")
  [[ "${#saved_config[@]}" == 4 ]] || exit 2
  RPT_FULL_TEST_PARENT="${saved_config[0]}"
  RPT_FULL_TEST_SOURCE="${saved_config[1]}"
  RPT_FULL_TEST_DATA="${saved_config[2]}"
  RPT_FULL_TEST_JOBS="${saved_config[3]}"
elif [[ "${mode}" != submit && "${mode}" != --dry-run ]]; then
  usage; exit 2
fi
[[ "${RPT_FULL_TEST_JOBS}" =~ ^[0-9]+$ && "${RPT_FULL_TEST_CONCURRENCY}" =~ ^[0-9]+$ ]] || exit 2
((RPT_FULL_TEST_JOBS >= 1 && RPT_FULL_TEST_JOBS <= 200 && RPT_FULL_TEST_CONCURRENCY >= 1)) || exit 2
test -f "${RPT_FULL_TEST_PARENT}/campaign_spec.json"
test -d "${RPT_FULL_TEST_DATA}"
if [[ "${mode}" == --dry-run ]]; then
  printf 'Parent: %s\nTest directory: %s\nGPU tasks: %s; concurrency: %s\n' \
    "${RPT_FULL_TEST_PARENT}" "${RPT_FULL_TEST_DATA}" "${RPT_FULL_TEST_JOBS}" "${RPT_FULL_TEST_CONCURRENCY}"
  echo 'Inference only; all 12 checkpoints per ROOT file; no performance gate.'
  exit 0
fi

if [[ "${mode}" != --resume ]]; then
  source_commit="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["source"]["commit"])' "${RPT_FULL_TEST_PARENT}/campaign_spec.json")"
  parent_id="$(basename -- "${RPT_FULL_TEST_PARENT}")"
  : "${RPT_FULL_TEST_SOURCE:=/home/ryreu/atlas/.rpt_worktrees/${parent_id}}"
  if [[ ! -d "${RPT_FULL_TEST_SOURCE}" ]]; then
    mkdir -p -- "$(dirname -- "${RPT_FULL_TEST_SOURCE}")"
    git -C "${REPO_ROOT}" worktree add --detach "${RPT_FULL_TEST_SOURCE}" "${source_commit}"
  fi
  : "${RPT_FULL_TEST_OUTPUT:=$(dirname -- "${RPT_FULL_TEST_PARENT}")/rpt_offline_test20m_$(date -u +%Y%m%dT%H%M%SZ)}"
  if [[ -e "${RPT_FULL_TEST_OUTPUT}" ]]; then
    echo "Output already exists. Use --resume to reuse its frozen driver and predictions." >&2
    exit 2
  fi
  mkdir -p -- "${RPT_FULL_TEST_OUTPUT}/logs"
  cp -- "${REPO_ROOT}/scripts/evaluate_relational_part_offline_full_test.py" "${RPT_FULL_TEST_OUTPUT}/driver.py"
  cp -- "${SCRIPT_DIR}/run_rpt_offline_full_test.sh" "${RPT_FULL_TEST_OUTPUT}/worker.sh"
  python -c 'import json,sys; p,s,d,n=sys.argv[1:]; print(json.dumps(dict(parent=p,source=s,data=d,jobs=int(n))))' \
    "${RPT_FULL_TEST_PARENT}" "${RPT_FULL_TEST_SOURCE}" "${RPT_FULL_TEST_DATA}" "${RPT_FULL_TEST_JOBS}" \
    > "${RPT_FULL_TEST_OUTPUT}/submission_config.json"
fi
export RPT_FULL_TEST_OUTPUT RPT_FULL_TEST_SOURCE RPT_FULL_TEST_PARENT RPT_FULL_TEST_DATA RPT_FULL_TEST_JOBS CONDA_BASE CONDA_ENV
worker="${RPT_FULL_TEST_OUTPUT}/worker.sh"

# Serialize submissions to this output; refuse to silently duplicate a live graph.
exec 9>"${RPT_FULL_TEST_OUTPUT}/submission.lock"
flock -n 9 || { echo 'Another submitter owns this output.' >&2; exit 2; }
if [[ -f "${RPT_FULL_TEST_OUTPUT}/submitted_job_ids.txt" ]]; then
  if ! live_listing="$(squeue --noheader --user "$(id -un)" --format='%i')"; then
    echo 'Cannot query Slurm safely; refusing a potentially duplicate submission.' >&2
    exit 2
  fi
  while read -r phase job; do
    while read -r active; do
      if [[ "${active%%_*}" == "${job}" ]]; then
        echo "Previous ${phase} job ${job} is still queued/running. Do not duplicate it; inspect or cancel only this evaluation's stale jobs first." >&2
        exit 2
      fi
    done <<< "${live_listing}"
  done < "${RPT_FULL_TEST_OUTPUT}/submitted_job_ids.txt"
fi

submit() {
  local phase="$1" dependency="$2"
  shift 2
  local dep=()
  [[ -z "${dependency}" ]] || dep=(--dependency="afterok:${dependency}" --kill-on-invalid-dep=yes)
  local job
  job="$(sbatch --parsable --account="${SBATCH_ACCOUNT}" --partition="${SBATCH_PARTITION}" \
    --job-name="rpt20m_${phase}" --output="${RPT_FULL_TEST_OUTPUT}/logs/%x_%A_%a.out" \
    --error="${RPT_FULL_TEST_OUTPUT}/logs/%x_%A_%a.err" \
    --export="ALL,RPT_FULL_TEST_PHASE=${phase}" "${dep[@]}" "$@" "${worker}")"
  job="${job%%;*}"
  [[ "${job}" =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch result: ${job}" >&2; return 1; }
  printf '%s %s\n' "${phase}" "${job}" >> "${RPT_FULL_TEST_OUTPUT}/submitted_job_ids.txt"
  echo "${job}"
}

dependency=""
if [[ ! -f "${RPT_FULL_TEST_OUTPUT}/evaluation_plan.json" ]]; then
  dependency="$(submit prepare '' --cpus-per-task=4 --mem=32G --time=04:00:00)"
fi
inference="$(submit run "${dependency}" --gres="${GPU_GRES}" --cpus-per-task=8 --mem=64G \
  --time=2-00:00:00 --array="0-$((RPT_FULL_TEST_JOBS-1))%${RPT_FULL_TEST_CONCURRENCY}")"
metrics="$(submit aggregate "${inference}" --cpus-per-task=4 --mem=64G --time=12:00:00 --array=0-11%3)"
final="$(submit report "${metrics}" --cpus-per-task=1 --mem=8G --time=01:00:00)"
printf '\nEvaluation: %s\nGPU array: %s\nMetric array: %s\nReport job: %s\n' "${RPT_FULL_TEST_OUTPUT}" "${inference}" "${metrics}" "${final}"
echo 'Retry after a failure: cancel only stale jobs listed in this output, then invoke this submitter with --resume OUTPUT_DIRECTORY.'
