#!/usr/bin/env bash
# Independent, inference-only SPORC graph. Never changes or cancels Tigris jobs.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<'USAGE'
Usage:
  bash sbatch/submit_relational_part_sporc_inference.sh --source-campaign ABS_ROOT [--output NEW_ROOT] [--dry-run]
  bash sbatch/submit_relational_part_sporc_inference.sh --source-job 207556 [--search-root DIRECTORY] [--dry-run]
  bash sbatch/submit_relational_part_sporc_inference.sh --resume NEW_ROOT [--dry-run]

An explicit source campaign may also be the sole positional argument.
Options: --jobs N (1..200), --concurrency N (1..jobs).
Source-job discovery requires a unique submitted_job_ids.txt with the exact
two-field record "infer JOB_ID"; no newest/latest campaign is ever selected.
Dry-run validates inputs and prints the graph, without creating or submitting it.
USAGE
}

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
source_campaign=""
source_job=""
search_root="${RPT_SPORC_SEARCH_ROOT:-${REPO_ROOT}/checkpoints/relational_particle_transformer}"
output="${RPT_SPORC_OUTPUT:-}"
resume=""
dry_run=0
while (($#)); do
  case "$1" in
    --source-campaign|--source-job|--search-root|--output|--resume|--jobs|--concurrency)
      (($# >= 2)) || fail "Missing argument for $1"
      case "$1" in
        --source-campaign) source_campaign="$2" ;;
        --source-job) source_job="$2" ;;
        --search-root) search_root="$2" ;;
        --output) output="$2" ;;
        --resume) resume="$2" ;;
        --jobs) RPT_SPORC_JOBS="$2" ;;
        --concurrency) RPT_SPORC_CONCURRENCY="$2" ;;
      esac
      shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    --help|-h) usage; exit 0 ;;
    --*) usage >&2; fail "Unknown option: $1" ;;
    *) [[ -z "${source_campaign}" ]] || fail 'Specify exactly one source campaign'
       source_campaign="$1"; shift ;;
  esac
done

if [[ -n "${resume}" ]]; then
  [[ -z "${source_campaign}${source_job}${output}" ]] || fail '--resume cannot be combined with source/output options'
  output="$(realpath -e -- "${resume}")"
  [[ -f "${output}/sporc_campaign.json" && -f "${output}/submission_config.json" ]] || fail 'Resume needs both frozen campaign and submission manifests'
elif [[ -n "${source_job}" ]]; then
  [[ -z "${source_campaign}" ]] || fail 'Choose --source-job or --source-campaign, not both'
  [[ "${source_job}" =~ ^[1-9][0-9]*$ ]] || fail '--source-job must be a positive Slurm job ID'
  [[ -d "${search_root}" ]] || fail "Search directory does not exist: ${search_root}"
elif [[ -z "${source_campaign}" ]]; then
  usage >&2
  fail 'An explicit source campaign or source job is required'
fi

# Never activate an inherited ARM/Tigris runtime on SPORC.
: "${CONDA_BASE:=/home/ryreu/miniconda3}"
: "${CONDA_ENV:=atlas_kd_sporc}"
[[ "$(uname -m)" == x86_64 ]] || fail 'This launcher requires SPORC x86_64, not ARM/Tigris'
case "${CONDA_BASE}" in *aarch64*|*arm64*) fail 'An ARM Conda prefix is not permitted' ;; esac
[[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]] || fail "Missing Conda activation hook under ${CONDA_BASE}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
# shellcheck disable=SC1090
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export MAX_JOBS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
python -c 'import platform,sys; assert platform.machine() == "x86_64", platform.machine(); assert sys.version_info >= (3,10), sys.version; print("SPORC Python:", sys.executable, sys.version.split()[0])'

if [[ -n "${source_job}" ]]; then
  source_campaign="$(python - "${search_root}" "${source_job}" <<'PY'
from pathlib import Path
import sys
root, job = Path(sys.argv[1]).resolve(strict=True), sys.argv[2]
matches = sorted({p.parent.resolve() for p in root.rglob("submitted_job_ids.txt")
                  if any(line.split() == ["infer", job] for line in p.read_text().splitlines())})
if len(matches) != 1:
    raise SystemExit(f"Expected one source campaign for infer {job}, found {len(matches)}: {matches}")
print(matches[0])
PY
  )"
fi

if [[ -n "${resume}" ]]; then
  saved_text="$(python - "${output}" <<'PY'
from pathlib import Path
import json, sys
root = Path(sys.argv[1]).resolve(strict=True)
manifest = json.loads((root / "sporc_campaign.json").read_text())
saved = json.loads((root / "submission_config.json").read_text())
if manifest["jobs"] != saved["jobs"] or Path(manifest["output"]).resolve() != root:
    raise SystemExit("Frozen campaign and submission configuration disagree")
for key in ("jobs", "concurrency", "aggregate_concurrency", "account", "partition", "gres", "conda_base", "conda_env"):
    print(saved[key])
print(manifest["source_campaign"])
PY
  )"
  mapfile -t saved_config <<< "${saved_text}"
  [[ ${#saved_config[@]} == 9 ]] || fail 'Malformed frozen submission configuration'
  : "${RPT_SPORC_JOBS:=${saved_config[0]}}"
  : "${RPT_SPORC_CONCURRENCY:=${saved_config[1]}}"
  : "${RPT_SPORC_AGGREGATE_CONCURRENCY:=${saved_config[2]}}"
  : "${SBATCH_ACCOUNT:=${saved_config[3]}}"
  : "${SBATCH_PARTITION:=${saved_config[4]}}"
  : "${GPU_GRES:=${saved_config[5]}}"
  [[ "${CONDA_BASE}" == "${saved_config[6]}" && "${CONDA_ENV}" == "${saved_config[7]}" ]] || fail 'Resume must use the original x86 Conda environment'
  source_campaign="${saved_config[8]}"
else
  source_campaign="$(realpath -e -- "${source_campaign}")"
  [[ -f "${source_campaign}/selection/locked_models.json" ]] || fail 'Source lacks selection/locked_models.json'
  [[ -f "${source_campaign}/test/evaluation_plan.json" ]] || fail 'Source lacks test/evaluation_plan.json'
  : "${output:=$(dirname -- "${source_campaign}")/rpt_offline_sporc_$(date -u +%Y%m%dT%H%M%SZ)}"
  output="$(realpath -m -- "${output}")"
  [[ ! -e "${output}" ]] || fail 'Fresh output already exists; use --resume to reuse only its frozen campaign'
  [[ "${output}/" != "${source_campaign}/"* ]] || fail 'The separate SPORC output must not be inside the original campaign'
fi

: "${RPT_SPORC_JOBS:=40}"
: "${RPT_SPORC_CONCURRENCY:=2}"
: "${RPT_SPORC_AGGREGATE_CONCURRENCY:=2}"
: "${SBATCH_ACCOUNT:=reu-aisocial}"
: "${SBATCH_PARTITION:=debug}"
: "${GPU_GRES:=gpu:a100:1}"
for value in "${RPT_SPORC_JOBS}" "${RPT_SPORC_CONCURRENCY}" "${RPT_SPORC_AGGREGATE_CONCURRENCY}"; do
  [[ "${value}" =~ ^[1-9][0-9]*$ && ${#value} -le 3 ]] || fail 'Task counts/concurrency must be positive decimal integers'
done
((RPT_SPORC_JOBS <= 200 && RPT_SPORC_CONCURRENCY <= RPT_SPORC_JOBS && RPT_SPORC_AGGREGATE_CONCURRENCY <= 12)) || fail 'Require jobs <= 200, concurrency <= jobs, and aggregate concurrency <= 12'
[[ "${output}" == /* && "${output}" != *','* ]] || fail 'Output must be absolute and cannot contain commas (Slurm export delimiter)'
export RPT_SPORC_ROOT="${output}" RPT_SPORC_JOBS RPT_SPORC_CONCURRENCY RPT_SPORC_AGGREGATE_CONCURRENCY
export SBATCH_ACCOUNT SBATCH_PARTITION GPU_GRES CONDA_BASE CONDA_ENV

# Scheduler choices are immutable on resume, just like task/file ownership.
if [[ -n "${resume}" ]]; then
  python - "${output}/submission_config.json" <<'PY'
import json, os, sys
saved = json.load(open(sys.argv[1]))
mapping = dict(jobs="RPT_SPORC_JOBS", concurrency="RPT_SPORC_CONCURRENCY",
               aggregate_concurrency="RPT_SPORC_AGGREGATE_CONCURRENCY", account="SBATCH_ACCOUNT",
               partition="SBATCH_PARTITION", gres="GPU_GRES", conda_base="CONDA_BASE", conda_env="CONDA_ENV")
for key, env in mapping.items():
    if str(saved[key]) != os.environ[env]:
        raise SystemExit(f"Cannot change frozen {key}: {saved[key]!r} -> {os.environ[env]!r}")
PY
  export PYTHONPATH="${output}/source"
  python -u "${output}/source/scripts/run_relational_part_sporc_inference.py" check --output "${output}"
fi

worker="${output}/source/sbatch/run_relational_part_sporc_inference.sh"
job_prefix="rpt_sporc_$(python -c 'import hashlib,sys; print(hashlib.sha256(sys.argv[1].encode()).hexdigest()[:12])' "${output}")"
bootstrap=(python -u "${REPO_ROOT}/scripts/run_relational_part_sporc_inference.py" bootstrap
  --source-campaign "${source_campaign}" --output "${output}" --jobs "${RPT_SPORC_JOBS}")

if ((dry_run)); then
  printf 'DRY RUN: source=%s\noutput=%s\n' "${source_campaign}" "${output}"
  if [[ -z "${resume}" ]]; then printf '%q ' "${bootstrap[@]}"; printf '\n'; fi
else
  command -v flock >/dev/null || fail 'flock is required for duplicate-submission protection'
  command -v sbatch >/dev/null || fail 'sbatch is unavailable'
  command -v squeue >/dev/null || fail 'squeue is required for duplicate-submission protection'
  if [[ -z "${resume}" ]]; then
    export PYTHONPATH="${REPO_ROOT}"
    "${bootstrap[@]}"
    [[ -f "${worker}" ]] || fail 'Bootstrap did not freeze the SPORC worker'
    python - "${output}/submission_config.json" <<'PY'
import json, os, sys
mapping = dict(jobs="RPT_SPORC_JOBS", concurrency="RPT_SPORC_CONCURRENCY",
               aggregate_concurrency="RPT_SPORC_AGGREGATE_CONCURRENCY", account="SBATCH_ACCOUNT",
               partition="SBATCH_PARTITION", gres="GPU_GRES", conda_base="CONDA_BASE", conda_env="CONDA_ENV")
config = {key: int(os.environ[env]) if key in ("jobs", "concurrency", "aggregate_concurrency")
          else os.environ[env] for key, env in mapping.items()}
with open(sys.argv[1], "x") as stream:
    json.dump(config, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY
  fi
  export PYTHONPATH="${output}/source"
  python -u "${output}/source/scripts/run_relational_part_sporc_inference.py" check --output "${output}"
  [[ -f "${worker}" ]] || fail 'Frozen worker is missing'
  mkdir -p -- "${output}/logs"
  exec 9>"${output}/submission.lock"
  flock -n 9 || fail 'Another submitter owns this output'
  prior_jobs=()
  if [[ -f "${output}/submitted_job_ids.txt" ]]; then
    while read -r phase job rest; do
      [[ "${job}" =~ ^[0-9]+$ && -z "${rest:-}" ]] || fail 'Malformed submission ledger; inspect before retrying'
      prior_jobs+=("${job}")
    done < "${output}/submitted_job_ids.txt"
  fi
  if ! live_listing="$(squeue --noheader --user "$(id -un)" --format='%i|%j')"; then
    fail 'Cannot query Slurm safely; refusing a potentially duplicate graph'
  fi
  while IFS='|' read -r active name; do
    [[ -n "${active}" ]] || continue
    [[ "${name}" != "${job_prefix}_"* ]] || fail "This campaign still has queued/running job ${active}; no duplicate submitted"
    for job in "${prior_jobs[@]}"; do
      [[ "${active%%_*}" != "${job}" ]] || fail "Previous job ${job} remains queued/running"
    done
  done <<< "${live_listing}"
fi

submit() {
  local phase="$1" dependency="$2" job
  shift 2
  local command=(sbatch --parsable --account="${SBATCH_ACCOUNT}" --partition="${SBATCH_PARTITION}"
    --job-name="${job_prefix}_${phase}" --chdir="${output}/source"
    --output="${output}/logs/%x_%A_%a.out" --error="${output}/logs/%x_%A_%a.err"
    --export="ALL,RPT_SPORC_ROOT=${output},RPT_SPORC_PHASE=${phase}")
  [[ -z "${dependency}" ]] || command+=(--dependency="afterok:${dependency}" --kill-on-invalid-dep=yes)
  command+=("$@" "${worker}")
  if ((dry_run)); then
    printf '%q ' "${command[@]}" >&2; printf '\n' >&2
    printf 'DRY_%s\n' "${phase}"
    return
  fi
  # Explicitly set the process environment too: an inherited phase must never
  # take precedence when Slurm expands --export=ALL.
  if ! job="$(RPT_SPORC_PHASE="${phase}" "${command[@]}")"; then
    printf 'Submission failed at %s. Prior jobs remain untouched; inspect %s/submitted_job_ids.txt before --resume.\n' "${phase}" "${output}" >&2
    return 1
  fi
  job="${job%%;*}"
  [[ "${job}" =~ ^[0-9]+$ ]] || { printf 'Invalid sbatch response at %s: %s. Inspect squeue before retrying.\n' "${phase}" "${job}" >&2; return 1; }
  printf '%s %s\n' "${phase}" "${job}" >> "${output}/submitted_job_ids.txt"
  printf '%s\n' "${job}"
}

build="$(submit build '' --cpus-per-task=4 --mem=16G --time=01:00:00)"
validate="$(submit validate "${build}" --gres="${GPU_GRES}" --cpus-per-task=4 --mem=64G --time=02:00:00)"
infer="$(submit infer "${validate}" --gres="${GPU_GRES}" --cpus-per-task=8 --mem=64G --time=23:30:00 --array="0-$((RPT_SPORC_JOBS-1))%${RPT_SPORC_CONCURRENCY}")"
aggregate="$(submit aggregate "${infer}" --cpus-per-task=4 --mem=64G --time=12:00:00 --array="0-11%${RPT_SPORC_AGGREGATE_CONCURRENCY}")"
report="$(submit report "${aggregate}" --cpus-per-task=2 --mem=16G --time=01:00:00)"
printf '\nCampaign: %s\nBuild: %s; validate: %s; infer: %s; aggregate: %s; report: %s\n' "${output}" "${build}" "${validate}" "${infer}" "${aggregate}" "${report}"
printf 'Inference only. Original Tigris outputs and queued jobs are untouched.\n'
