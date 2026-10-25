#!/usr/bin/env bash
# Append a resource-only recovery; never edit the campaign's frozen source/spec.
set -euo pipefail
selected_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
output= recovery= train_hours=168 evaluate_hours= concurrency=2 resume=0 dry=0
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
usage() {
  cat <<'USAGE'
Usage: bash sbatch/submit_part_selected_tigris_recovery.sh --output CAMPAIGN [options]
Reuses all four successful GH200 benchmarks. No preparation/benchmark reruns.
  --train-hours N       Default 168 (7 days); can only increase original limits.
  --evaluate-hours N    Default original evaluation limit; checked with 30% margin.
  --concurrency 1..4    Default 2. Same one-GPU-per-model recipe.
  --recovery DIR       Default CAMPAIGN/recoveries/walltime_168h_12h (actual limits).
  --resume             Retry the same recovery only after all campaign jobs end.
  --dry-run            Validate evidence, partition limit, and budgets; no writes/jobs.
Queues gate(2h) -> train[4] -> lock(2h) -> evaluate[4] -> report(2h).
Existing source, spec, benchmarks, data, controls and environment stay unchanged.
Only walltimes may change. Commit/push/pull this recovery tooling before submission.
USAGE
}
while (( $# )); do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --resume) resume=1; shift ;; --dry-run) dry=1; shift ;;
    --output|--recovery|--train-hours|--evaluate-hours|--concurrency)
      (( $# >= 2 )) || fail "Missing $1 value"
      case "$1" in
        --output) output=$2 ;; --recovery) recovery=$2 ;; --train-hours) train_hours=$2 ;;
        --evaluate-hours) evaluate_hours=$2 ;; --concurrency) concurrency=$2 ;;
      esac
      shift 2 ;;
    *) fail "Unknown option: $1" ;;
  esac
done
[[ "$output" == /* && "$concurrency" =~ ^[1-4]$ ]] || fail 'Need absolute --output and concurrency 1..4'
[[ "$(uname -m)" == aarch64 ]] || fail 'Run on Tigris, not SPORC'
for command in git tar sha256sum find sort xargs realpath flock sbatch squeue scontrol; do
  command -v "$command" >/dev/null || fail "Missing command: $command"
done
output="$(realpath -e -- "$output")"
[[ -f "$output/experiment_spec.json" && -f "$output/launch.env" ]] || fail 'Need a prepared original campaign'
source "$output/launch.env"
evaluate_hours=${evaluate_hours:-$SELECTED_EVALUATE_HOURS}
[[ "$train_hours" =~ ^[1-9][0-9]*$ && "$evaluate_hours" =~ ^[1-9][0-9]*$ ]] || fail 'Invalid whole-hour limits'
recovery=${recovery:-$output/recoveries/walltime_${train_hours}h_${evaluate_hours}h}
recovery="$(realpath -m -- "$recovery")"
[[ "$(dirname -- "$recovery")" == "$output/recoveries" ]] || fail 'Recovery must be an immediate child of CAMPAIGN/recoveries'
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PATH="$(dirname -- "$SELECTED_PYTHON"):$PATH"
export LD_LIBRARY_PATH="$(dirname -- "$SELECTED_PYTHON")/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
for option_name in ${!SBATCH_@}; do unset "$option_name"; done
(cd -- "$output/source"; sha256sum --quiet -c "$output/source.sha256")
partition_info="$(scontrol show partition "$SELECTED_PARTITION" -o)"
max_time="$(tr ' ' '\n' <<< "$partition_info" | sed -n 's/^MaxTime=//p')"
[[ -n "$max_time" ]] || fail 'Could not read partition MaxTime'
files=(scripts/recover_part_selected_tigris_walltime.py sbatch/submit_part_selected_tigris_recovery.sh sbatch/run_part_selected_tigris_recovery.sh)
driver="$selected_repo/scripts/recover_part_selected_tigris_walltime.py"
if [[ -d "$recovery/source" ]]; then
  driver="$recovery/source/scripts/recover_part_selected_tigris_walltime.py"
  (cd -- "$recovery/source"; sha256sum --quiet -c ../source.sha256)
elif (( resume )); then
  fail 'Recovery snapshot missing; inspect the output before retrying'
fi
configure=(--phase configure --output "$output" --recovery "$recovery" --train-hours "$train_hours"
  --evaluate-hours "$evaluate_hours" --partition-max-time "$max_time")
# Standard-library-only validation: no GPU work, package installs or campaign writes.
"$SELECTED_PYTHON" "$driver" "${configure[@]}" --dry-run
if (( dry )); then
  printf 'Dry run: gate 2h -> train[0-3%%%s] %sh -> lock 2h -> evaluate[0-3%%%s] %sh -> report 2h\n' \
    "$concurrency" "$train_hours" "$concurrency" "$evaluate_hours"
  echo 'Full source/environment/numerical admission is rechecked by the CPU gate. No files or jobs changed.'
  exit 0
fi
exec 9>"$output/.submission.lock"
flock -n 9 || fail 'Another campaign submitter is active'
ledger="$output/submitted_job_ids.tsv"
live="$(squeue -h -u "$(id -un)" -o '%i|%j')"
for selected_ledger in "$ledger" "$recovery/submitted_job_ids.tsv"; do
  [[ -f "$selected_ledger" ]] || continue
  while read -r phase job; do
    if awk -F '|' '{split($1, id, "_"); print id[1]}' <<< "$live" | grep -qx "$job"; then
      fail "Campaign job $job ($phase) is active; no duplicate jobs queued"
    fi
    if grep -q "|selgh_autostart_${job}$" <<< "$live"; then
      fail "A continuation for gate $job is still active; no duplicate jobs queued"
    fi
  done < "$selected_ledger"
done
if [[ -e "$recovery" ]]; then
  (( resume )) || fail 'Recovery already exists; inspect its jobs then use --resume'
else
  for name in "${files[@]}"; do
    git -C "$selected_repo" cat-file -e "HEAD:$name" || fail 'Commit/push/pull recovery tooling first'
  done
  [[ -z "$(git -C "$selected_repo" status --porcelain -- "${files[@]}")" ]] || fail 'Commit recovery tooling before submitting'
  [[ -d "$output/recoveries" ]] || mkdir -- "$output/recoveries"
  mkdir -- "$recovery"
  mkdir -- "$recovery/source" "$recovery/logs"
  git -C "$selected_repo" archive HEAD -- "${files[@]}" | tar -xf - -C "$recovery/source"
  git -C "$selected_repo" rev-parse HEAD > "$recovery/source_commit.txt"
  (cd -- "$recovery/source"; find . -type f -print0 | sort -z | xargs -0 sha256sum) > "$recovery/source.sha256"
fi
driver="$recovery/source/scripts/recover_part_selected_tigris_walltime.py"
"$SELECTED_PYTHON" "$driver" "${configure[@]}"
submit() {
  local phase=$1 dependency=$2 hours=$3 result
  local request=(--parsable --job-name="selgh_recover_${phase}" --account="$SELECTED_ACCOUNT" --partition="$SELECTED_PARTITION"
    --nodes=1 --ntasks=1 --cpus-per-task="$SELECTED_CPUS" --mem="$SELECTED_MEMORY" --time="${hours}:00:00"
    --output="$recovery/logs/%x_%A_%a.out" --error="$recovery/logs/%x_%A_%a.err" --export=ALL --signal=B:USR1@180)
  case "$phase" in train|evaluate)
    request+=(--array="0-3%$concurrency")
    if [[ "$SELECTED_GPU" == any ]]; then request+=(--gpus-per-node=1)
    else request+=(--gres="gpu:$SELECTED_GPU:1"); fi ;;
  esac
  [[ -z "$dependency" ]] || request+=(--dependency="afterok:$dependency" --kill-on-invalid-dep=yes)
  result="$(PART_SELECTED_TIGRIS_OUTPUT="$output" PART_SELECTED_TIGRIS_PHASE="$phase" SELECTED_WALLTIME_RECOVERY="$recovery" \
    sbatch "${request[@]}" "$recovery/source/sbatch/run_part_selected_tigris_recovery.sh")"
  result="${result%%;*}"
  [[ "$result" =~ ^[0-9]+$ ]] || fail "Unexpected Slurm response: $result; inspect scheduler before retrying"
  printf '%s job: %s\n' "$phase" "$result" >&2
  printf '%s\t%s\n' "$phase" "$result" >> "$recovery/submitted_job_ids.tsv"
  printf 'walltime_%s\t%s\n' "$phase" "$result" >> "$ledger"
  printf '%s' "$result"
}
printf 'Original campaign: %s\nRecovery: %s\n' "$output" "$recovery"
gate="$(submit gate '' 2)"
train="$(submit train "$gate" "$train_hours")"
lock="$(submit lock "$train" 2)"
evaluate="$(submit evaluate "$lock" "$evaluate_hours")"
report="$(submit report "$evaluate" 2)"
printf 'Queued recovery gate -> four trainings -> lock -> four evaluations -> report.\nLogs: %s/logs\n' "$recovery"
