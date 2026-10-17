#!/usr/bin/env bash
# Reviewed AMP recovery -> original frozen training/evaluation/report graph.
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
campaign= recovery= recovery_job= concurrency=2 mode=train reviewed=0 resume=0 dry=0 train_hours=
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
while (( $# )); do
  case "$1" in
    --help|-h)
      echo 'Usage: bash sbatch/submit_part_ev_screen_after_recovery.sh --campaign DIR --recovery-job ARRAY_ID --reviewed [--concurrency 2]'
      echo 'Alternative: --recovery DIR. Modes: train (default), evaluate, report, review.'
      echo 'Options: --resume (terminal jobs only), --train-hours N (may increase estimate), --dry-run.'
      echo 'Fresh seed initialization; no probe weights or test data. Original source/results stay unchanged.'
      exit 0 ;;
    --reviewed) reviewed=1; shift ;; --resume) resume=1; shift ;; --dry-run) dry=1; shift ;;
    --campaign|--recovery|--recovery-job|--concurrency|--mode|--train-hours)
      (( $# >= 2 )) || fail "Missing $1 value"
      case "$1" in
        --campaign) campaign=$2 ;; --recovery) recovery=$2 ;; --recovery-job) recovery_job=$2 ;;
        --concurrency) concurrency=$2 ;; --mode) mode=$2 ;; --train-hours) train_hours=$2 ;;
      esac
      shift 2 ;;
    *) fail "Unknown option: $1" ;;
  esac
done
[[ "$campaign" == /* && "$concurrency" =~ ^[1-7]$ ]] || fail 'Use absolute campaign path and concurrency 1..7'
[[ -z "$recovery_job" || "$recovery_job" =~ ^[0-9]+$ ]] || fail 'Invalid recovery array ID'
[[ -z "$recovery" || "$recovery" == /* ]] || fail 'Recovery path must be absolute'
[[ -z "$recovery" || -z "$recovery_job" ]] || fail 'Choose either --recovery or --recovery-job'
[[ -z "$train_hours" || "$train_hours" =~ ^[1-9][0-9]*$ ]] || fail 'Train hours must be a positive integer'
case "$mode" in train|evaluate|report|review) ;; *) fail 'Invalid mode' ;; esac
campaign="$(realpath -e -- "$campaign")"
approval="$campaign/production_approval"
source "$campaign/launch.env"
if (( dry )); then
  printf 'Campaign: %s\nMode: %s\n' "$campaign" "$mode"
  printf 'GPU arrays: --array=0-6%%%s --gres=gpu:%s:1 --cpus-per-task=%s --mem=%s --partition=%s --account=%s\n' \
    "$concurrency" "$SCREEN_GPU" "$SCREEN_CPUS" "$SCREEN_MEMORY" "$SCREEN_PARTITION" "$SCREEN_ACCOUNT"
  echo 'Walltimes: original measured estimates +30%. train[7] -> full-validation[7] -> report (2h).'
  echo 'Requires explicit review and verified recovery evidence. No files/jobs created or evidence approved.'
  exit 0
fi
[[ "$(uname -m)" == x86_64 ]] || fail 'Run on sporcsubmit, not Tigris'
for command in git sha256sum realpath flock sbatch squeue scontrol cmp; do
  command -v "$command" >/dev/null || fail "Missing command: $command"
done
for option_name in ${!SBATCH_@}; do unset "$option_name"; done
[[ "$mode" != train || "$reviewed" == 1 ]] || fail 'Training requires --reviewed after inspecting recovery results'
[[ -x "$SCREEN_PYTHON" ]] || fail 'Missing frozen Python environment'
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
unset PYTHONPATH
screen_bin="$(dirname -- "$SCREEN_PYTHON")"
export PATH="$screen_bin:$PATH" LD_LIBRARY_PATH="$screen_bin/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec 9>"$campaign/.submission.lock"
flock -n 9 || fail 'Another campaign submitter is active'
if [[ -n "$recovery_job" ]]; then
  [[ -f "$campaign/diagnostics/amp_recovery_jobs.tsv" ]] || fail 'Recovery ledger not found'
  mapfile -t matches < <(awk -F '\t' -v job="$recovery_job" '$1 == "run" && $2 == job {print $3}' "$campaign/diagnostics/amp_recovery_jobs.tsv")
  [[ "${#matches[@]}" == 1 ]] || fail 'Recovery array ID must identify exactly one diagnostic directory'
  recovery="${matches[0]}"
fi
if [[ -n "$recovery" ]]; then
  recovery="$(realpath -e -- "$recovery")"
fi
if [[ -f "$approval/approval.json" ]]; then
  approved_recovery="$("$SCREEN_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["recovery"])' "$approval/approval.json")"
  [[ -z "$recovery" || "$recovery" == "$approved_recovery" ]] || fail 'Cannot replace existing production approval with another diagnostic'
  recovery="$approved_recovery"
fi
[[ -n "$recovery" ]] || fail 'Specify the reviewed --recovery-job or --recovery directory'
ledger="$campaign/submitted_job_ids.tsv"
if [[ "$mode" != review ]]; then
  live="$(squeue -h -u "$(id -un)" -o '%i' | sed 's/_.*//' | sort -u)"
  for job_ledger in "$ledger" "$campaign/diagnostics/amp_recovery_jobs.tsv"; do
    [[ -f "$job_ledger" ]] || continue
    while read -r phase job rest; do
      if grep -qx "$job" <<< "$live"; then fail "Campaign job $job ($phase) is still active. No duplicate submission."; fi
    done < "$job_ledger"
  done
  if [[ -f "$ledger" ]] && grep -q "^${mode}[[:space:]]" "$ledger" && (( ! resume )); then
    fail 'Phase already submitted; inspect terminal jobs, then explicitly --resume'
  fi
fi
(cd -- "$campaign/source"; sha256sum --quiet -c "$campaign/source.sha256")
driver="$approval/run_part_ev_screen_after_recovery.py"
if [[ ! -f "$approval/approval.json" ]]; then
  [[ "$mode" == train && "$reviewed" == 1 && "$resume" == 0 ]] || fail 'Initial approval requires fresh reviewed training'
  [[ ! -e "$approval" ]] || fail 'Incomplete approval directory; inspect it before retrying'
  files=(scripts/run_part_ev_screen_after_recovery.py scripts/check_part_ev_screen_amp_recovery.py
    sbatch/run_part_ev_screen_after_recovery.sh sbatch/submit_part_ev_screen_after_recovery.sh)
  for name in "${files[@]}"; do
    git -C "$repo" cat-file -e "HEAD:$name" || fail 'Commit/push/pull the production approval implementation first'
  done
  [[ -z "$(git -C "$repo" status --porcelain -- "${files[@]}")" ]] || fail 'Commit relevant approval code before submission'
  mkdir -- "$approval"
  for name in "${files[@]}"; do
    git -C "$repo" show "HEAD:$name" > "$approval/${name##*/}"
  done
  cp -- "$campaign/launch.env" "$approval/launch.env"
  cp -- "$campaign/launch.json" "$approval/launch.json"
  git -C "$repo" rev-parse HEAD > "$approval/approval_commit.txt"
  "$SCREEN_PYTHON" "$driver" --phase approve --approval "$approval" --campaign "$campaign" --recovery "$recovery" --reviewed
  (cd -- "$approval"; sha256sum ./*.py ./*.sh ./launch.env ./launch.json ./approval.json) > "$approval/approval.sha256"
fi
(cd -- "$approval"; sha256sum --quiet -c approval.sha256)
cmp -s -- "$campaign/launch.env" "$approval/launch.env" || fail 'Campaign launch resources changed after approval'
recommendation="$("$SCREEN_PYTHON" "$driver" --phase review --approval "$approval")"
printf '%s\n' "$recommendation"
[[ "$mode" != review ]] || exit 0
scontrol show partition "$SCREEN_PARTITION" >/dev/null || fail 'Frozen partition unavailable'
needed="$("$SCREEN_PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["recommended_train_hours"])' <<< "$recommendation")"
eval_hours="$("$SCREEN_PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["recommended_evaluate_hours"])' <<< "$recommendation")"
train_hours="${train_hours:-$needed}"
(( train_hours >= needed )) || fail "Use at least $needed training hours; do not shorten the measured estimate"
[[ -d "$campaign/logs" ]] || mkdir -- "$campaign/logs"
submit() {
  local phase=$1 dependency=$2 hours=$3 result
  local request=(--parsable --job-name="evscr_$phase" --account="$SCREEN_ACCOUNT" --partition="$SCREEN_PARTITION"
    --nodes=1 --ntasks=1 --cpus-per-task="$SCREEN_CPUS" --mem="$SCREEN_MEMORY" --time="${hours}:00:00"
    --output="$campaign/logs/%x_%A_%a.out" --error="$campaign/logs/%x_%A_%a.err" --export=ALL --signal=B:USR1@180)
  [[ "$phase" == report ]] || request+=(--array="0-6%$concurrency" --gres="gpu:$SCREEN_GPU:1")
  [[ -z "$dependency" ]] || request+=(--dependency="afterok:$dependency" --kill-on-invalid-dep=yes)
  result="$(sbatch "${request[@]}" "$approval/run_part_ev_screen_after_recovery.sh" "$approval" "$phase" "$resume")"
  result="${result%%;*}"
  [[ "$result" =~ ^[0-9]+$ ]] || fail "Unexpected sbatch result: $result; inspect scheduler before retrying"
  printf '%s\t%s\n' "$phase" "$result" >> "$ledger"
  printf '%s\t%s\n' "$phase" "$result" >> "$approval/submitted_job_ids.tsv"
  printf '%s job: %s\n' "$phase" "$result" >&2
  printf '%s' "$result"
}
dependency=
if [[ "$mode" == train ]]; then
  dependency="$(submit train '' "$train_hours")"
fi
if [[ "$mode" != report ]]; then
  dependency="$(submit evaluate "$dependency" "$eval_hours")"
fi
submit report "$dependency" 2
printf '\nCampaign: %s\nApproval: %s\nJob ledger: %s\n' "$campaign" "$approval/approval.json" "$ledger"
