#!/usr/bin/env bash
# Complete ONLY the four missing factorial cells (12 new trainings).
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
: "${RPT_ABLATION_PARENT:=/home/ryreu/atlas/Fresh_check/checkpoints/relational_particle_transformer/rpt_offline_transfer_20260809T195641Z_0737194d3c}"
: "${RPT_ABLATION_PREVIOUS:=/home/ryreu/atlas/Fresh_check/checkpoints/relational_particle_transformer/rpt_offline_test20m_20260925T235646Z}"
: "${RPT_ABLATION_TRAIN_DATA:=/home/ryreu/atlas/jetclass_part1}"
: "${RPT_ABLATION_TEST_DATA:=/home/ryreu/atlas/jetclass_test20m/test_20M}"
: "${RPT_ABLATION_TRAIN_CONCURRENCY:=4}"
: "${RPT_ABLATION_TEST_CONCURRENCY:=8}"
: "${RPT_ABLATION_TREE_CONCURRENCY:=8}"
: "${SBATCH_ACCOUNT:=reu-aisocial}"
: "${SBATCH_PARTITION:=tigris}"
: "${GPU_GRES:=gpu:gh200:1}"
: "${CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
: "${CONDA_ENV:=atlas_kd_tigris}"
mode="${1:-submit}"
case "${mode}" in
  --dry-run|submit) [[ $# -le 1 ]] || exit 2 ;;
  --resume) [[ $# == 2 ]] || { echo 'Usage: --resume CAMPAIGN_ROOT' >&2; exit 2; } ;;
  *) echo 'Usage: [--dry-run | --resume CAMPAIGN_ROOT]' >&2; exit 2 ;;
esac
for value in "${RPT_ABLATION_TRAIN_CONCURRENCY}" "${RPT_ABLATION_TEST_CONCURRENCY}" "${RPT_ABLATION_TREE_CONCURRENCY}"; do
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || { echo 'Concurrency must be positive.' >&2; exit 2; }
done
if [[ "${mode}" == --dry-run ]]; then
  printf 'Original models: %s\nOriginal full test: %s\nTraining ROOTs: %s\nTest ROOTs: %s\n' \
    "${RPT_ABLATION_PARENT}" "${RPT_ABLATION_PREVIOUS}" "${RPT_ABLATION_TRAIN_DATA}" "${RPT_ABLATION_TEST_DATA}"
  echo 'prepare -> cache[3] -> storage -> trees[126] -> bind -> GPU validation -> train[12] -> lock -> infer[20] -> aggregate[12] -> report'
  printf 'Concurrency train=%s, inference=%s, CPU trees=%s\n' "${RPT_ABLATION_TRAIN_CONCURRENCY}" "${RPT_ABLATION_TEST_CONCURRENCY}" "${RPT_ABLATION_TREE_CONCURRENCY}"
  echo 'No performance gate; no old model retraining; no dataset deletion. Dry-run does not authenticate remote artifacts.'
  exit 0
fi
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
if [[ "${mode}" == --resume ]]; then
  RPT_ABLATION_ROOT="$(realpath -- "$2")"
  test -f "${RPT_ABLATION_ROOT}/campaign_spec.json"
else
  : "${RPT_ABLATION_ROOT:=$(dirname -- "${RPT_ABLATION_PARENT}")/rpt_offline_factorial_$(date -u +%Y%m%dT%H%M%SZ)}"
  python -s "${REPO_ROOT}/scripts/run_relational_part_offline_ablations.py" bootstrap \
    --parent "${RPT_ABLATION_PARENT}" --previous "${RPT_ABLATION_PREVIOUS}" \
    --train-data "${RPT_ABLATION_TRAIN_DATA}" --test-data "${RPT_ABLATION_TEST_DATA}" --output "${RPT_ABLATION_ROOT}"
  RPT_ABLATION_ROOT="$(realpath -- "${RPT_ABLATION_ROOT}")"
fi
export RPT_ABLATION_ROOT CONDA_BASE CONDA_ENV
exec 9>"${RPT_ABLATION_ROOT}/submission.lock"
flock -n 9 || { echo 'Another submitter owns this campaign.' >&2; exit 2; }
if [[ -f "${RPT_ABLATION_ROOT}/submitted_job_ids.txt" ]]; then
  live="$(squeue --noheader --user "$(id -un)" --format='%i')" || { echo 'Cannot check live jobs; refusing duplicate submission.' >&2; exit 2; }
  while read -r phase job; do
    while read -r active; do
      if [[ "${active%%_*}" == "${job}" ]]; then
        echo "Campaign ${phase} job ${job} is still live. Inspect/cancel only stale jobs belonging to this campaign before --resume." >&2
        exit 2
      fi
    done <<< "${live}"
  done < "${RPT_ABLATION_ROOT}/submitted_job_ids.txt"
fi
worker="${RPT_ABLATION_ROOT}/source/sbatch/run_rpt_offline_ablations.sh"
submit() {
  local phase="$1" dependency="$2" job
  shift 2
  local dep=()
  [[ -z "${dependency}" ]] || dep=(--dependency="afterok:${dependency}" --kill-on-invalid-dep=yes)
  job="$(sbatch --parsable --account="${SBATCH_ACCOUNT}" --partition="${SBATCH_PARTITION}" \
    --job-name="rptabl_${phase}" --output="${RPT_ABLATION_ROOT}/logs/%x_%A_%a.out" \
    --error="${RPT_ABLATION_ROOT}/logs/%x_%A_%a.err" --export="ALL,RPT_ABLATION_PHASE=${phase}" \
    "${dep[@]}" "$@" "${worker}")"
  job="${job%%;*}"
  [[ "${job}" =~ ^[0-9]+$ ]] || { echo "Invalid sbatch response: ${job}" >&2; return 1; }
  printf '%s %s\n' "${phase}" "${job}" >> "${RPT_ABLATION_ROOT}/submitted_job_ids.txt"
  echo "${job}"
}
prep="$(submit prepare '' --cpus-per-task=4 --mem=32G --time=01:00:00)"
cache="$(submit cache "${prep}" --cpus-per-task=4 --mem=64G --time=12:00:00 --array=0-2%1)"
space="$(submit storage "${cache}" --cpus-per-task=1 --mem=8G --time=00:30:00)"
trees="$(submit trees "${space}" --cpus-per-task=4 --mem=16G --time=02:00:00 --array="0-125%${RPT_ABLATION_TREE_CONCURRENCY}")"
bind="$(submit bind "${trees}" --cpus-per-task=4 --mem=16G --time=02:00:00)"
validation="$(submit validate "${bind}" --gres="${GPU_GRES}" --cpus-per-task=4 --mem=32G --time=01:00:00)"
training="$(submit train "${validation}" --gres="${GPU_GRES}" --cpus-per-task=8 --mem=96G --time=3-00:00:00 --array="0-11%${RPT_ABLATION_TRAIN_CONCURRENCY}")"
lock="$(submit lock "${training}" --cpus-per-task=4 --mem=16G --time=01:00:00)"
inference="$(submit infer "${lock}" --gres="${GPU_GRES}" --cpus-per-task=8 --mem=64G --time=2-00:00:00 --array="0-19%${RPT_ABLATION_TEST_CONCURRENCY}")"
metrics="$(submit aggregate "${inference}" --cpus-per-task=4 --mem=64G --time=12:00:00 --array=0-11%3)"
report="$(submit report "${metrics}" --cpus-per-task=2 --mem=16G --time=01:00:00)"
printf '\nCampaign: %s\nTraining array: %s\nTest inference array: %s\nReport: %s\n' "${RPT_ABLATION_ROOT}" "${training}" "${inference}" "${report}"
echo "Resume after failures using --resume CAMPAIGN_ROOT, after inspecting/removing only this campaign's stale live jobs."
