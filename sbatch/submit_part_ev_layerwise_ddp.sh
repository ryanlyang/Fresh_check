#!/usr/bin/env bash
# New output namespace only. Never cancels jobs or rewrites an existing campaign.
set -euo pipefail
ddp_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
mode= output= scaling= dry=0 resume=0 reviewed=0 train_hours= evaluate_hours=
ddp_python=/home/ryreu/miniconda3/envs/part_ev_screen_sporc/bin/python
account=reu-aisocial partition=tier3
explicit_resources=()
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
while (( $# )); do
  case "$1" in
    --help|-h)
      echo 'Usage: bash sbatch/submit_part_ev_layerwise_ddp.sh --mode MODE [options]'
      echo 'prepare --scaling DIR [--output NEW_DIR]: freeze source, queue two-A100 04:30:00 restart/validation check'
      echo 'review --output DIR: show passing preflight and full walltime recommendation'
      echo 'train --output DIR --reviewed: one two-A100 seed-101 training -> one-GPU full-5M validation -> report'
      echo 'evaluate|report --output DIR [--resume]: retry terminal downstream work'
      echo 'Options: --python PATH --account NAME --partition NAME --train-hours N --evaluate-hours N --dry-run'
      echo 'No old jobs changed. No other architectures/seeds trained. Resume keeps world size two.'
      exit 0 ;;
    --dry-run) dry=1; shift ;; --resume) resume=1; shift ;; --reviewed) reviewed=1; shift ;;
    --mode|--output|--scaling|--python|--account|--partition|--train-hours|--evaluate-hours)
      (( $#>=2 )) || fail "Missing $1 value"
      case "$1" in --scaling|--python|--account|--partition) explicit_resources+=("$1" "$2") ;; esac
      case "$1" in
        --mode) mode=$2 ;; --output) output=$2 ;; --scaling) scaling=$2 ;; --python) ddp_python=$2 ;;
        --account) account=$2 ;; --partition) partition=$2 ;; --train-hours) train_hours=$2 ;; --evaluate-hours) evaluate_hours=$2 ;;
      esac
      shift 2 ;;
    *) fail "Unknown option $1" ;;
  esac
done
case "$mode" in prepare|review|train|evaluate|report) ;; *) fail 'Choose --mode; see --help' ;; esac
[[ "$account" =~ ^[a-zA-Z0-9_-]+$ && "$partition" =~ ^[a-zA-Z0-9_-]+$ ]] || fail 'Invalid resource name'
[[ -z "$train_hours" || "$train_hours" =~ ^[1-9][0-9]*$ ]] || fail 'Invalid training hours'
[[ -z "$evaluate_hours" || "$evaluate_hours" =~ ^[1-9][0-9]*$ ]] || fail 'Invalid evaluation hours'
if [[ "$mode" == prepare && -z "$output" ]]; then
  output="$ddp_repo/checkpoints/part_ev_layerwise_ddp/layerwise_2g_$(date -u +%Y%m%dT%H%M%SZ)"
fi
[[ "$output" == /* && "$ddp_python" == /* ]] || fail 'Use absolute output/Python paths'
[[ "$mode" != prepare || "$scaling" == /* ]] || fail 'Prepare needs absolute --scaling DIR'
if (( dry )); then
  printf 'Mode: %s\nNew campaign: %s\nRead-only scaling: %s\n' "$mode" "$output" "$scaling"
  echo 'Preflight: --nodes=1 --ntasks=1 --gres=gpu:a100:2 --cpus-per-task=8 --mem=64G --time=04:30:00'
  echo 'Train: exactly one seed-101 layerwise all-eight State-EV job, two A100s, measured walltime +30%.'
  echo 'Afterok -> one-A100 full-5M validation -> CPU report. No old jobs cancelled or changed.'
  exit 0
fi
[[ "$(uname -m)" == x86_64 ]] || fail 'Run on sporcsubmit'
for tool in git tar sha256sum find sort xargs realpath sbatch squeue scontrol flock; do
  command -v "$tool" >/dev/null || fail "Missing $tool"
done
for name in ${!SBATCH_@}; do unset "$name"; done
if [[ "$mode" == prepare ]]; then
  (( resume==0 )) || fail 'A changed/failed preflight needs a fresh campaign; --resume is for production retries'
  [[ ! -e "$output" && ! -L "$output" ]] || fail 'Prepare requires NEW output'
  scaling="$(realpath -e -- "$scaling")"
  [[ -x "$ddp_python" && -f "$scaling/gpus_2.json" ]] || fail 'Need the existing environment and completed scaling evidence'
  candidate="$(realpath -m -- "$output")"
  [[ "$candidate" != "$scaling" && "$candidate" != "$scaling/"* && "$scaling" != "$candidate/"* ]] || fail 'Cannot prepare inside old scaling evidence'
  [[ -z "$(git -C "$ddp_repo" status --porcelain -- part_ev_fullscale part_ev_screen part_ev_message_study part_ev_state_speed part_ev_state_fullscale part_ev_state_capacity part_ev_state_compare part_layerwise_screen part_ev_gpu_scaling part_ev_layerwise_ddp scripts/*part_ev* sbatch/*part_ev* particle_transformer/data/JetClass/JetClass_full.yaml)" ]] || fail 'Commit relevant changes before preparing'
  for file in part_ev_layerwise_ddp/protocol.py part_ev_layerwise_ddp/engine.py scripts/run_part_ev_layerwise_ddp.py sbatch/run_part_ev_layerwise_ddp.sh sbatch/submit_part_ev_layerwise_ddp.sh; do
    git -C "$ddp_repo" cat-file -e "HEAD:$file" || fail 'Commit, push and pull this implementation first'
  done
  # Resolve the original production directory before creating any new files.
  original="$("$ddp_python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["parent"])' "$scaling/plan.json")"
  original="$(realpath -e -- "$original")"
  [[ "$candidate" != "$original" && "$candidate" != "$original/"* && "$original" != "$candidate/"* ]] || fail 'Cannot prepare inside the old production campaign'
  ddp_parent="$(dirname -- "$output")"
  [[ -d "$ddp_parent" ]] || mkdir -p -- "$ddp_parent"
  mkdir -- "$output"
  output="$(realpath -e -- "$output")"
  mkdir -- "$output/source" "$output/logs"
  git -C "$ddp_repo" rev-parse HEAD > "$output/source_commit.txt"
  git -C "$ddp_repo" archive HEAD | tar -xf - -C "$output/source"
  (cd -- "$output/source"; find . -type f -print0 | sort -z | xargs -0 sha256sum) > "$output/source.sha256"
  printf 'EVDDP_PYTHON=%q\nEVDDP_SCALING=%q\nEVDDP_ACCOUNT=%q\nEVDDP_PARTITION=%q\n' \
    "$ddp_python" "$scaling" "$account" "$partition" > "$output/config.sh"
  (cd -- "$output"; sha256sum config.sh) > "$output/config.sha256"
fi
output="$(realpath -e -- "$output")"
(cd -- "$output"; sha256sum --quiet -c config.sha256)
source "$output/config.sh"
if [[ "$mode" != prepare ]]; then
  for ((i=0;i<${#explicit_resources[@]};i+=2)); do
    case "${explicit_resources[i]}" in
      --scaling) frozen="$EVDDP_SCALING" ;; --python) frozen="$EVDDP_PYTHON" ;;
      --account) frozen="$EVDDP_ACCOUNT" ;; --partition) frozen="$EVDDP_PARTITION" ;;
    esac
    [[ "${explicit_resources[i+1]}" == "$frozen" ]] || fail 'Resource differs from frozen campaign; prepare a new directory'
  done
fi
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$output/source"
export PATH="$(dirname -- "$EVDDP_PYTHON"):$PATH" LD_LIBRARY_PATH="$(dirname -- "$EVDDP_PYTHON")/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
(cd -- "$output/source"; sha256sum --quiet -c "$output/source.sha256")
driver="$output/source/scripts/run_part_ev_layerwise_ddp.py"
if [[ "$mode" == review ]]; then
  exec "$EVDDP_PYTHON" "$driver" --phase review --output "$output"
fi
exec 9>"$output/.submission.lock"
flock -n 9 || fail 'Another submitter holds this campaign lock'
ledger="$output/submitted_job_ids.tsv"
if [[ -s "$ledger" ]]; then
  live="$(squeue -h -u "$(id -un)" -o '%i' | sed 's/_.*//' | sort -u)"
  while read -r phase job; do
    if grep -qx "$job" <<< "$live"; then fail "Job $job ($phase) is still active. No duplicates."; fi
  done < "$ledger"
  if grep -q "^${mode}[[:space:]]" "$ledger" && (( ! resume )); then fail 'Already submitted; inspect failure and pass --resume'; fi
fi
partition_info="$(scontrol show partition "$EVDDP_PARTITION" -o)"
check_time() {
  local seconds=$1 limit day=0 hh mm ss
  limit="$(sed -n 's/.*MaxTime=\([^ ]*\).*/\1/p' <<< "$partition_info")"
  [[ "$limit" != UNLIMITED && "$limit" != INFINITE ]] || return 0
  if [[ "$limit" == *-* ]]; then day="${limit%%-*}"; limit="${limit#*-}"; fi
  IFS=: read -r hh mm ss <<< "$limit"
  [[ "$day:$hh:$mm:$ss" =~ ^[0-9]+:[0-9]+:[0-9]+:[0-9]+$ ]] || fail 'Cannot parse partition time limit'
  (( seconds <= 10#$day*86400 + 10#$hh*3600 + 10#$mm*60 + 10#$ss )) || fail 'Requested walltime exceeds partition limit'
}
submit() {
  local phase=$1 dependency=$2 walltime=$3 seconds=$4 gpus=$5 job
  check_time "$seconds"
  local request=(--parsable --job-name="evddp_$phase" --partition="$EVDDP_PARTITION" --account="$EVDDP_ACCOUNT"
    --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G --time="$walltime"
    --output="$output/logs/%x_%j.out" --error="$output/logs/%x_%j.err" --export=ALL --signal=B:TERM@180)
  (( gpus==0 )) || request+=(--gres="gpu:a100:$gpus")
  [[ -z "$dependency" ]] || request+=(--dependency="afterok:$dependency" --kill-on-invalid-dep=yes)
  job="$(EVDDP_OUTPUT="$output" EVDDP_PHASE="$phase" sbatch "${request[@]}" "$output/source/sbatch/run_part_ev_layerwise_ddp.sh")"
  job="${job%%;*}"
  [[ "$job" =~ ^[0-9]+$ ]] || fail 'Unexpected sbatch response; inspect scheduler'
  printf '%s\t%s\n' "$phase" "$job" >> "$ledger"
  printf '%s job: %s\n' "$phase" "$job" >&2
  printf '%s' "$job"
}
case "$mode" in
  prepare)
    check_time 16200
    "$EVDDP_PYTHON" "$driver" --phase prepare --output "$output" --scaling "$EVDDP_SCALING"
    submit preflight '' 04:30:00 16200 2 ;;
  train|evaluate)
    review_args=(--phase review --output "$output")
    if [[ "$mode" == train ]]; then
      (( reviewed==1 )) || fail 'Review the preflight, then supply --reviewed'
      review_args+=(--approve --reviewed)
    fi
    estimate="$("$EVDDP_PYTHON" "$driver" "${review_args[@]}")"
    needed="$("$EVDDP_PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["train_hours"])' <<< "$estimate")"
    eval_needed="$("$EVDDP_PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["evaluate_hours"])' <<< "$estimate")"
    evaluate_hours="${evaluate_hours:-$eval_needed}"
    (( evaluate_hours>=eval_needed )) || fail "Need at least $eval_needed evaluation hours"
    check_time "$((evaluate_hours*3600))"
    dependency=
    if [[ "$mode" == train ]]; then
      train_hours="${train_hours:-$needed}"
      (( train_hours>=needed )) || fail "Need at least $needed training hours including overhead/margin"
      check_time "$((train_hours*3600))"
      dependency="$(submit train '' "${train_hours}:00:00" "$((train_hours*3600))" 2)"
    fi
    evaluation="$(submit evaluate "$dependency" "${evaluate_hours}:00:00" "$((evaluate_hours*3600))" 1)"
    submit report "$evaluation" 04:30:00 16200 0 ;;
  report) submit report '' 04:30:00 16200 0 ;;
esac
printf '\nCampaign: %s\nLedger: %s\n' "$output" "$ledger"
