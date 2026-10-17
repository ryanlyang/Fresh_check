#!/usr/bin/env bash
# Short diagnostic continuation of saved probes, never production submission.
set -euo pipefail
recovery_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
campaign= concurrency=2 dry=0
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
while (( $# )); do
  case "$1" in
    --help|-h)
      echo 'Usage: bash sbatch/submit_part_ev_screen_amp_recovery.sh --campaign DIR [--concurrency 1..7] [--dry-run]'
      echo 'Seven 1-GPU, 1-hour diagnostics -> afterany 10-minute CPU summary. No training or changes to original artifacts.'
      exit 0 ;;
    --dry-run) dry=1; shift ;;
    --campaign|--concurrency)
      (( $# >= 2 )) || fail "Missing $1 value"
      case "$1" in --campaign) campaign=$2 ;; --concurrency) concurrency=$2 ;; esac
      shift 2 ;;
    *) fail "Unknown option: $1" ;;
  esac
done
[[ "$campaign" == /* && "$concurrency" =~ ^[1-7]$ ]] || fail 'Use an absolute campaign and concurrency 1..7'
campaign="$(realpath -e -- "$campaign")"
[[ -f "$campaign/launch.env" && -f "$campaign/campaign.json" ]] || fail 'Missing prepared campaign'
source "$campaign/launch.env"
if (( dry )); then
  printf 'Campaign: %s\n' "$campaign"
  printf 'Recovery: --array=0-6%%%s --gres=gpu:%s:1 --cpus-per-task=%s --mem=%s --account=%s --partition=%s --time=01:00:00\n' \
    "$concurrency" "$SCREEN_GPU" "$SCREEN_CPUS" "$SCREEN_MEMORY" "$SCREEN_ACCOUNT" "$SCREEN_PARTITION"
  echo 'Summary: afterany recovery-array, 1 CPU, 2G, 00:10:00. No files or jobs created; no production authorization.'
  exit 0
fi
[[ "$(uname -m)" == x86_64 ]] || fail 'Run this on sporcsubmit, not Tigris'
for command in git sha256sum realpath flock sbatch squeue scontrol mktemp; do
  command -v "$command" >/dev/null || fail "Missing command: $command"
done
for option_name in ${!SBATCH_@}; do unset "$option_name"; done
files=(scripts/check_part_ev_screen_amp_recovery.py sbatch/run_part_ev_screen_amp_recovery.sh sbatch/submit_part_ev_screen_amp_recovery.sh)
for name in "${files[@]}"; do
  git -C "$recovery_repo" cat-file -e "HEAD:$name" || fail 'Commit/push/pull the recovery implementation first'
done
[[ -z "$(git -C "$recovery_repo" status --porcelain -- "${files[@]}")" ]] || fail 'Commit recovery source edits before submitting'
[[ -x "$SCREEN_PYTHON" ]] || fail 'Missing frozen Python environment'
scontrol show partition "$SCREEN_PARTITION" >/dev/null || fail 'Frozen partition unavailable'
# Use the same lock as the production launcher. No overlapping benchmark,
# production or recovery pipeline for this campaign is submitted here.
exec 9>"$campaign/.submission.lock"
flock -n 9 || fail 'Another campaign submitter is active'
live="$(squeue -h -u "$(id -un)" -o '%i' | sed 's/_.*//' | sort -u)"
for ledger in "$campaign/submitted_job_ids.tsv" "$campaign/diagnostics/amp_recovery_jobs.tsv"; do
  [[ -f "$ledger" ]] || continue
  while read -r phase job rest; do
    if grep -qx "$job" <<< "$live"; then fail "Campaign job $job ($phase) is still active"; fi
  done < "$ledger"
done
(cd -- "$campaign/source"; sha256sum --quiet -c "$campaign/source.sha256")
[[ -d "$campaign/diagnostics" ]] || mkdir -- "$campaign/diagnostics"
output="$(mktemp -d "$campaign/diagnostics/amp_recovery.XXXXXX")"
mkdir -- "$output/logs"
for name in "${files[@]}"; do
  git -C "$recovery_repo" show "HEAD:$name" > "$output/${name##*/}"
done
cp -- "$campaign/launch.env" "$output/launch.env"
cp -- "$campaign/launch.json" "$output/launch.json"
git -C "$recovery_repo" rev-parse HEAD > "$output/diagnostic_commit.txt"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$campaign/source"
recovery_bin="$(dirname -- "$SCREEN_PYTHON")"
export PATH="$recovery_bin:$PATH" LD_LIBRARY_PATH="$recovery_bin/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$SCREEN_PYTHON" "$output/check_part_ev_screen_amp_recovery.py" --phase prepare --campaign "$campaign" --output "$output"
(cd -- "$output"; sha256sum ./*.py ./*.sh ./launch.env ./launch.json ./recovery.json) > "$output/diagnostic.sha256"
common=(--parsable --account="$SCREEN_ACCOUNT" --partition="$SCREEN_PARTITION" --nodes=1 --ntasks=1
  --output="$output/logs/%x_%A_%a.out" --error="$output/logs/%x_%A_%a.err" --export=ALL)
submit() {
  local phase=$1 result
  shift
  result="$(sbatch "${common[@]}" "$@" "$output/run_part_ev_screen_amp_recovery.sh" "$output" "$phase")"
  result="${result%%;*}"
  [[ "$result" =~ ^[0-9]+$ ]] || fail "Unexpected sbatch output: $result; inspect scheduler before retrying"
  printf '%s\t%s\t%s\n' "$phase" "$result" "$output" >> "$campaign/diagnostics/amp_recovery_jobs.tsv"
  printf '%s\t%s\n' "$phase" "$result" >> "$output/submitted_job_ids.tsv"
  printf '%s' "$result"
}
printf 'Recovery directory: %s\n' "$output"
run_id="$(submit run --job-name=evscr_amp_recovery --array="0-6%$concurrency" --gres="gpu:$SCREEN_GPU:1" \
  --cpus-per-task="$SCREEN_CPUS" --mem="$SCREEN_MEMORY" --time=01:00:00 --signal=B:USR1@120)"
printf 'Recovery array: %s\n' "$run_id"
summary_id="$(submit summary --job-name=evscr_amp_summary --cpus-per-task=1 --mem=2G --time=00:10:00 --dependency="afterany:$run_id")"
printf 'Summary job: %s\nSummary log: %s/logs/evscr_amp_summary_%s_*.out\n' "$summary_id" "$output" "$summary_id"
echo 'Original benchmarks/checkpoints and training gate are unchanged. No training submitted.'
