#!/usr/bin/env bash
# One isolated diagnostic job. Never submit/resume production or cancel jobs.
set -euo pipefail

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
usage() {
  cat <<'USAGE'
Usage:
  bash sbatch/submit_relational_part_sporc_parity_diagnostic.sh --campaign ABS_SPORC_CAMPAIGN [--dry-run]
  bash sbatch/submit_relational_part_sporc_parity_diagnostic.sh ABS_SPORC_CAMPAIGN [--dry-run]

Submit exactly one debug/A100 diagnostic job (4 CPU, 32G, 1 hour).
The tool and worker are frozen in a unique CAMPAIGN/diagnostics/parity.XXXXXX
directory. Results go in its new results/ child, never production predictions
or the portability authorization receipt. No dependencies, cancellation, or
production retry is performed. --dry-run creates nothing and submits nothing.
USAGE
}

setup_environment() {
  : "${CONDA_BASE:=/home/ryreu/miniconda3}"
  : "${CONDA_ENV:=atlas_kd_sporc}"
  [[ "$(uname -m)" == x86_64 ]] || fail 'The diagnostic requires the SPORC x86_64 runtime'
  case "${CONDA_BASE}" in *aarch64*|*arm64*) fail 'An ARM Conda prefix is forbidden' ;; esac
  [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]] || fail "Missing Conda activation hook under ${CONDA_BASE}"
  export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
  # shellcheck disable=SC1090
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  export MAX_JOBS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
  export CONDA_BASE CONDA_ENV
  python -c 'import platform,sys; assert platform.machine() == "x86_64", platform.machine(); assert sys.version_info >= (3,10), sys.version; print("Diagnostic Python:", sys.executable, sys.version.split()[0])'
}

# Slurm executes a copy from its spool: use only explicit absolute paths here.
if [[ "${1:-}" == --worker ]]; then
  (($# == 1)) || fail '--worker accepts no additional arguments'
  : "${RPT_SPORC_DIAGNOSTIC_ROOT:?Missing frozen diagnostic root}"
  : "${RPT_SPORC_DIAGNOSTIC_CAMPAIGN:?Missing existing SPORC campaign}"
  [[ "${RPT_SPORC_DIAGNOSTIC_ROOT}" == /* && "${RPT_SPORC_DIAGNOSTIC_CAMPAIGN}" == /* ]] || fail 'Worker paths must be absolute'
  [[ -f "${RPT_SPORC_DIAGNOSTIC_CAMPAIGN}/sporc_campaign.json" ]] || fail 'Existing campaign manifest is absent'
  [[ -f "${RPT_SPORC_DIAGNOSTIC_ROOT}/diagnose.py" ]] || fail 'Frozen diagnostic tool is absent'
  [[ ! -e "${RPT_SPORC_DIAGNOSTIC_ROOT}/results" ]] || fail 'Diagnostic results already exist; do not overwrite or retry in place'
  setup_environment
  cd -- "${RPT_SPORC_DIAGNOSTIC_ROOT}"
  sha256sum --check source_sha256.txt
  export PYTHONPATH="${RPT_SPORC_DIAGNOSTIC_CAMPAIGN}/source"
  cd -- "${RPT_SPORC_DIAGNOSTIC_CAMPAIGN}/source"
  exec python -s -u "${RPT_SPORC_DIAGNOSTIC_ROOT}/diagnose.py" \
    --campaign "${RPT_SPORC_DIAGNOSTIC_CAMPAIGN}" \
    --output "${RPT_SPORC_DIAGNOSTIC_ROOT}/results"
fi

campaign=""
dry_run=0
while (($#)); do
  case "$1" in
    --campaign)
      (($# >= 2)) || fail 'Missing --campaign argument'
      [[ -z "${campaign}" ]] || fail 'Specify the campaign only once'
      campaign="$2"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    --help|-h) usage; exit 0 ;;
    --*) fail "Unknown option: $1" ;;
    *) [[ -z "${campaign}" ]] || fail 'Specify exactly one existing SPORC campaign'
       campaign="$1"; shift ;;
  esac
done
[[ -n "${campaign}" ]] || { usage >&2; fail 'An explicit existing SPORC campaign is required'; }
campaign="$(realpath -e -- "${campaign}")"
[[ -f "${campaign}/sporc_campaign.json" && -d "${campaign}/source" ]] || fail 'Expected an existing SPORC campaign with its frozen source'
[[ "${campaign}" == /* && "${campaign}" != *','* ]] || fail 'Campaign must be absolute and cannot contain a Slurm export comma'
[[ ! -L "${campaign}/diagnostics" ]] || fail 'Refusing a symlinked diagnostics directory'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
tool="${REPO_ROOT}/scripts/diagnose_relational_part_sporc_parity.py"
[[ -f "${tool}" ]] || fail "Diagnostic tool is missing: ${tool}"
: "${CONDA_BASE:=/home/ryreu/miniconda3}"
: "${CONDA_ENV:=atlas_kd_sporc}"

if ((dry_run)); then
  printf 'DRY RUN: existing campaign=%s\n' "${campaign}"
  printf 'Freeze tool and worker under a new %s/diagnostics/parity.XXXXXX directory.\n' "${campaign}"
  printf 'Runtime: %s/envs/%s; isolated results: NEW_DIAGNOSTIC_ROOT/results\n' "${CONDA_BASE}" "${CONDA_ENV}"
  printf '%s\n' 'sbatch --account=reu-aisocial --partition=debug --nodes=1 --ntasks=1 --gres=gpu:a100:1 --cpus-per-task=4 --mem=32G --time=01:00:00 NEW_DIAGNOSTIC_ROOT/worker.sh --worker'
  printf '%s\n' 'No dependencies, production submission/retry, receipt authorization, or cancellation.'
  exit 0
fi

command -v sbatch >/dev/null || fail 'sbatch is unavailable'
command -v sha256sum >/dev/null || fail 'sha256sum is required to authenticate the frozen diagnostic'
setup_environment
mkdir -p -- "${campaign}/diagnostics"
diagnostic_root="$(mktemp -d --tmpdir="${campaign}/diagnostics" parity.XXXXXX)"
cp -- "${tool}" "${diagnostic_root}/diagnose.py"
cp -- "${BASH_SOURCE[0]}" "${diagnostic_root}/worker.sh"
(
  cd -- "${diagnostic_root}"
  sha256sum diagnose.py worker.sh > source_sha256.txt
)
printf '%s\n' "${campaign}" > "${diagnostic_root}/source_campaign.txt"
export RPT_SPORC_DIAGNOSTIC_ROOT="${diagnostic_root}"
export RPT_SPORC_DIAGNOSTIC_CAMPAIGN="${campaign}"
# An inherited production sbatch environment must not turn this into an array
# or attach it to an existing dependency graph or another cluster.
unset SBATCH_DEPENDENCY SBATCH_ARRAY_INX SBATCH_CLUSTERS
if ! job="$(sbatch --parsable --account=reu-aisocial --partition=debug \
    --nodes=1 --ntasks=1 --gres=gpu:a100:1 --cpus-per-task=4 --mem=32G --time=01:00:00 \
    --job-name=rpt_sporc_parity --chdir="${campaign}/source" \
    --output="${diagnostic_root}/diagnostic_%j.out" --error="${diagnostic_root}/diagnostic_%j.err" \
    --export="ALL,RPT_SPORC_DIAGNOSTIC_ROOT=${diagnostic_root},RPT_SPORC_DIAGNOSTIC_CAMPAIGN=${campaign}" \
    "${diagnostic_root}/worker.sh" --worker)"; then
  printf 'Diagnostic submission failed. Frozen files remain at %s; no production action was taken.\n' "${diagnostic_root}" >&2
  exit 1
fi
printf '%s\n' "${job}" > "${diagnostic_root}/sbatch_response.txt"
job="${job%%;*}"
[[ "${job}" =~ ^[0-9]+$ ]] || fail "Unexpected sbatch response; inspect ${diagnostic_root}/sbatch_response.txt and squeue before any resubmission"
printf 'diagnose %s\n' "${job}" > "${diagnostic_root}/submitted_job_ids.txt"
printf 'Diagnostic job: %s\nFrozen diagnostic: %s\nResults: %s/results\n' "${job}" "${diagnostic_root}" "${diagnostic_root}"
printf 'Queue: squeue -j %s\n' "${job}"
printf 'Stdout: %s/diagnostic_%s.out\nStderr: %s/diagnostic_%s.err\n' "${diagnostic_root}" "${job}" "${diagnostic_root}" "${job}"
