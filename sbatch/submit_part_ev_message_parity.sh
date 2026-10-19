#!/usr/bin/env bash
# One short, read-only GPU investigation; never authorize or queue training.
set -euo pipefail
parity_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
campaign= task_index=0 dry=0
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
while (( $# )); do
  case "$1" in
    --help|-h)
      echo 'Usage: bash sbatch/submit_part_ev_message_parity.sh --campaign DIR [--task-index 0..4] [--dry-run]'
      echo 'One 1-hour GPU diagnostic on the campaign partition. Default task 0. No training or tolerance changes.'
      exit 0 ;;
    --dry-run) dry=1; shift ;;
    --campaign|--task-index)
      (( $# >= 2 )) || fail "Missing $1 value"
      case "$1" in --campaign) campaign=$2 ;; --task-index) task_index=$2 ;; esac
      shift 2 ;;
    *) fail "Unknown option: $1" ;;
  esac
done
[[ "$campaign" == /* && "$task_index" =~ ^[0-4]$ ]] || fail 'Use an absolute campaign and task index 0..4'
campaign="$(realpath -e -- "$campaign")"
[[ -f "$campaign/campaign.json" ]] || fail 'Missing prepared message-study campaign'
(cd -- "$campaign"; sha256sum --quiet -c launch.sha256)
source "$campaign/launch.env"
if (( dry )); then
  printf 'Campaign: %s\nTask: %s\n' "$campaign" "$task_index"
  printf 'Diagnostic: --gres=gpu:%s:1 --cpus-per-task=%s --mem=%s --account=%s --partition=%s --time=01:00:00\n' \
    "$MESSAGE_GPU" "$MESSAGE_CPUS" "$MESSAGE_MEMORY" "$MESSAGE_ACCOUNT" "$MESSAGE_PARTITION"
  echo 'No files/jobs created. Original benchmarks may continue; this diagnostic never authorizes training.'
  exit 0
fi
[[ "$(uname -m)" == x86_64 ]] || fail 'Run on sporcsubmit, not Tigris'
for command in git sha256sum realpath flock sbatch squeue scontrol mktemp; do
  command -v "$command" >/dev/null || fail "Missing command: $command"
done
for option_name in ${!SBATCH_@}; do unset "$option_name"; done
files=(scripts/diagnose_part_ev_message_parity.py sbatch/run_part_ev_message_parity.sh sbatch/submit_part_ev_message_parity.sh)
for name in "${files[@]}"; do
  git -C "$parity_repo" cat-file -e "HEAD:$name" || fail 'Commit/push/pull the diagnostic implementation first'
done
[[ -z "$(git -C "$parity_repo" status --porcelain -- "${files[@]}")" ]] || fail 'Commit diagnostic source edits before submission'
[[ -x "$MESSAGE_PYTHON" ]] || fail 'Missing frozen Python environment'
scontrol show partition "$MESSAGE_PARTITION" >/dev/null || fail 'Frozen partition unavailable'
(cd -- "$campaign/source"; sha256sum --quiet -c "$campaign/source.sha256")
[[ -d "$campaign/diagnostics" ]] || mkdir -- "$campaign/diagnostics"
[[ "$(realpath -e -- "$campaign/diagnostics")" == "$campaign/diagnostics" ]] || fail 'Unexpected diagnostics directory location'
exec 9>"$campaign/diagnostics/.message_parity_submission.lock"
flock -n 9 || fail 'Another parity submitter is active'
ledger="$campaign/diagnostics/message_parity_jobs.tsv"
if [[ -s "$ledger" ]]; then
  live="$(squeue -h -u "$(id -un)" -o '%i' | sed 's/_.*//' | sort -u)"
  while read -r index job rest; do
    if [[ "$index" == "$task_index" ]] && grep -qx "$job" <<< "$live"; then
      fail "Diagnostic job $job for task $index is still active"
    fi
  done < "$ledger"
fi
output="$(mktemp -d "$campaign/diagnostics/message_parity.XXXXXX")"
mkdir -- "$output/logs"
for name in "${files[@]}"; do
  git -C "$parity_repo" show "HEAD:$name" > "$output/${name##*/}"
done
cp -- "$campaign/launch.env" "$output/launch.env"
cp -- "$campaign/launch.json" "$output/launch.json"
git -C "$parity_repo" rev-parse HEAD > "$output/diagnostic_commit.txt"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
unset PYTHONPATH
parity_bin="$(dirname -- "$MESSAGE_PYTHON")"
export PATH="$parity_bin:$PATH" LD_LIBRARY_PATH="$parity_bin/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$MESSAGE_PYTHON" "$output/diagnose_part_ev_message_parity.py" \
  --phase prepare --campaign "$campaign" --output "$output" --task-index "$task_index"
(cd -- "$output"; sha256sum ./*.py ./*.sh ./launch.env ./launch.json ./diagnostic_plan.json) > "$output/diagnostic.sha256"
job="$(sbatch --parsable --job-name=evmsg_parity --account="$MESSAGE_ACCOUNT" --partition="$MESSAGE_PARTITION" \
  --nodes=1 --ntasks=1 --gres="gpu:$MESSAGE_GPU:1" --cpus-per-task="$MESSAGE_CPUS" --mem="$MESSAGE_MEMORY" \
  --time=01:00:00 --signal=B:USR1@120 --output="$output/logs/%x_%j.out" --error="$output/logs/%x_%j.err" \
  --export=ALL "$output/run_part_ev_message_parity.sh" "$output")"
job="${job%%;*}"
[[ "$job" =~ ^[0-9]+$ ]] || fail "Unexpected sbatch output: $job; inspect scheduler before retrying"
printf '%s\t%s\t%s\n' "$task_index" "$job" "$output" >> "$ledger"
printf '%s\n' "$job" > "$output/submitted_job_id.txt"
printf 'Diagnostic job: %s\nDiagnostic directory: %s\n' "$job" "$output"
printf 'Stdout: %s/logs/evmsg_parity_%s.out\nStderr: %s/logs/evmsg_parity_%s.err\n' "$output" "$job" "$output" "$job"
printf 'Report: %s/parity_report.md\nDetailed report: %s/parity_report.json\n' "$output" "$output"
echo 'Original campaign/source/checkpoints/tolerances unchanged. No training submitted or authorized.'
