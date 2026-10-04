#!/usr/bin/env bash
# Two small independent diagnostics, not a production retry or authorization.
set -euo pipefail
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }

if [[ "${1:-}" == --worker ]]; then
  : "${RPT_REPLAY_ROOT:?Missing frozen diagnostic directory}"
  cd -- "${RPT_REPLAY_ROOT}"
  sha256sum --check source_sha256.txt
  source ./config.sh
  [[ "$(uname -m)" == "${expected_arch}" ]] || fail 'Wrong compute architecture'
  export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
  export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
  source "${conda_base}/etc/profile.d/conda.sh"
  conda activate "${conda_env}"
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  export PYTHONPATH="${campaign}/source"
  args=("${phase}" --campaign "${campaign}" --diagnostic "${diagnostic}" --output "${RPT_REPLAY_ROOT}/results")
  [[ -z "${capture}" ]] || args+=(--capture "${capture}")
  exec python -s -u ./replay_relational_part_cross_platform_inputs.py "${args[@]}"
fi

phase='' campaign='' diagnostic='' capture='' partition='' dry_run=0
while (($#)); do
  case "$1" in
    --phase|--campaign|--diagnostic|--capture|--partition)
      (($# >= 2)) || fail "Missing argument for $1"
      case "$1" in
        --phase) phase="$2" ;; --campaign) campaign="$2" ;;
        --diagnostic) diagnostic="$2" ;; --capture) capture="$2" ;;
        --partition) partition="$2" ;;
      esac
      shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    --help|-h)
      printf '%s\n' \
        'Usage: bash sbatch/submit_relational_part_input_replay.sh --phase capture|replay --campaign ABS --diagnostic ABS_RESULTS [--capture ABS_CAPTURE_RESULTS] [--partition NAME] [--dry-run]' \
        'capture: Tigris CPU, 2 CPU/16G/20 minutes, no GPU.' \
        'replay: SPORC tier3, one A100, 2 CPU/16G/15 minutes; requires completed capture.' \
        'No production submission, cancellation, changed tolerances, or inference authorization.'
      exit 0 ;;
    *) fail "Unknown argument: $1" ;;
  esac
done
[[ -n "${campaign}" && -n "${diagnostic}" ]] || fail '--campaign and --diagnostic are required'
campaign="$(realpath -e -- "${campaign}")"
diagnostic="$(realpath -e -- "${diagnostic}")"
[[ -f "${campaign}/sporc_campaign.json" && -d "${campaign}/source" ]] || fail 'Missing frozen SPORC campaign'
[[ -f "${diagnostic}/diagnostic_report.json" && -f "${diagnostic}/first_batch_inputs_and_pairs.npz" && -f "${diagnostic}/diagnostic_logits.npz" ]] || fail 'Incomplete earlier diagnostic'
case "${phase}" in
  capture)
    [[ -z "${capture}" ]] || fail 'capture phase does not accept --capture'
    expected_arch=aarch64 conda_base=/home/ryreu/miniforge3-aarch64 conda_env=atlas_kd_tigris
    partition="${partition:-tigris}" time_limit=00:20:00
    resources=() ;;
  replay)
    [[ -n "${capture}" ]] || fail 'replay requires --capture with completed Tigris results'
    capture="$(realpath -e -- "${capture}")"
    [[ -f "${capture}/replay_report.json" && -f "${capture}/replay_arrays.npz" ]] || fail 'Tigris capture is not complete'
    expected_arch=x86_64 conda_base=/home/ryreu/miniconda3 conda_env=atlas_kd_sporc
    partition="${partition:-tier3}" time_limit=00:15:00
    resources=(--gres=gpu:a100:1) ;;
  *) fail '--phase must be capture or replay' ;;
esac
[[ "${partition}" =~ ^[a-zA-Z0-9_-]+$ ]] || fail 'Specify a single partition name'
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
files=(replay_relational_part_cross_platform_inputs.py diagnose_relational_part_sporc_parity.py)
for name in "${files[@]}"; do [[ -f "${repo}/scripts/${name}" ]] || fail "Missing ${name}"; done
if ((dry_run)); then
  printf 'DRY RUN: %s on %s; runtime %s/envs/%s\n' "${phase}" "${partition}" "${conda_base}" "${conda_env}"
  printf '%q ' sbatch --account=reu-aisocial --partition="${partition}" --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=16G --time="${time_limit}" "${resources[@]}"
  printf '\nNo jobs or files created. No production actions.\n'
  exit 0
fi
[[ "$(uname -m)" == "${expected_arch}" ]] || fail "Run ${phase} from the ${expected_arch} cluster login"
command -v sbatch >/dev/null || fail 'sbatch is unavailable'
[[ ! -L "${campaign}/diagnostics" ]] || fail 'Refusing symlinked diagnostic directory'
mkdir -p -- "${campaign}/diagnostics"
replay_root="$(mktemp -d --tmpdir="${campaign}/diagnostics" "matched_${phase}.XXXXXX")"
for name in "${files[@]}"; do cp -- "${repo}/scripts/${name}" "${replay_root}/${name}"; done
cp -- "${BASH_SOURCE[0]}" "${replay_root}/worker.sh"
for name in phase campaign diagnostic capture expected_arch conda_base conda_env; do
  printf '%s=%q\n' "${name}" "${!name}"
done > "${replay_root}/config.sh"
(
  cd -- "${replay_root}"
  sha256sum "${files[@]}" worker.sh config.sh > source_sha256.txt
)
export RPT_REPLAY_ROOT="${replay_root}"
# Strip inherited scheduler options, especially GPU/array/dependency requests.
while IFS= read -r name; do unset "${name}"; done < <(compgen -A variable SBATCH_)
job="$(sbatch --parsable --account=reu-aisocial --partition="${partition}" \
  --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=16G --time="${time_limit}" "${resources[@]}" \
  --job-name="rpt_input_${phase}" --chdir="${replay_root}" \
  --output="${replay_root}/replay_%j.out" --error="${replay_root}/replay_%j.err" \
  --export=ALL "${replay_root}/worker.sh" --worker)"
printf '%s\n' "${job}" > "${replay_root}/sbatch_response.txt"
job="${job%%;*}"
[[ "${job}" =~ ^[0-9]+$ ]] || fail "Unexpected sbatch response; inspect ${replay_root}/sbatch_response.txt before retrying"
printf 'Job: %s\nResults: %s/results\nStdout: %s/replay_%s.out\nStderr: %s/replay_%s.err\n' \
  "${job}" "${replay_root}" "${replay_root}" "${job}" "${replay_root}" "${job}"
if [[ "${phase}" == capture ]]; then
  printf '\nAfter completion, run on sporcsubmit:\n'
  printf '%q ' bash sbatch/submit_relational_part_input_replay.sh --phase replay --campaign "${campaign}" --diagnostic "${diagnostic}" --capture "${replay_root}/results" --partition tier3
  printf '\n'
fi
