#!/usr/bin/env bash
# Scheduling adapter only: never cancels jobs, edits old evidence or deletes files.
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
mode= campaign= chain= segments=4 dry=0 reviewed=0
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
while (( $# )); do
  case "$1" in
    --help|-h)
      echo 'review --campaign DIR: read-only approval and restart status'
      echo 'submit --campaign DIR --reviewed [--segments 4] [--chain NEW_DIR]'
      echo 'evaluate|report --chain DIR: retry downstream work after full completion'
      echo '--dry-run: print request without writing or submitting'
      echo 'Default: four sequential 72h two-A100 slots, then completion gate -> evaluation -> report.'
      echo 'Same model/seed/1M steps/50 checks. Original jobs must be terminal first.'
      exit 0 ;;
    --dry-run) dry=1; shift ;; --reviewed) reviewed=1; shift ;;
    --mode|--campaign|--chain|--segments)
      (( $#>=2 )) || fail "Missing $1 value"
      case "$1" in --mode) mode=$2 ;; --campaign) campaign=$2 ;; --chain) chain=$2 ;; --segments) segments=$2 ;; esac
      shift 2 ;;
    *) fail "Unknown option $1" ;;
  esac
done
case "$mode" in review|submit) [[ "$campaign" == /* ]] || fail 'Use absolute --campaign' ;;
  evaluate|report) [[ "$chain" == /* ]] || fail 'Use absolute --chain' ;;
  *) fail 'Choose --mode; see --help' ;;
esac
[[ "$segments" =~ ^[1-8]$ ]] || fail 'Choose 1..8 sequential segments'
if (( dry )); then
  printf 'Mode: %s\nCampaign: %s\nChain: %s\n' "$mode" "$campaign" "$chain"
  printf '%s sequential slots: --nodes=1 --ntasks=1 --gres=gpu:a100:2 --cpus-per-task=8 --mem=64G --time=72:00:00\n' "$segments"
  echo 'Stop at allocation end minus 1h; verified handoff -> afterok successor.'
  echo 'Full-completion gate -> one-A100 full-5M validation -> CPU report. No jobs/files changed.'
  exit 0
fi
[[ "$(uname -m)" == x86_64 ]] || fail 'Run on sporcsubmit'
for tool in git tar sha256sum find sort xargs realpath flock; do
  command -v "$tool" >/dev/null || fail "Missing $tool"
done
if [[ "$mode" == evaluate || "$mode" == report ]]; then
  chain="$(realpath -e -- "$chain")"
  (cd -- "$chain"; sha256sum --quiet -c config.sha256)
  source "$chain/config.sh"
  campaign="$EVSEG_CAMPAIGN"
fi
campaign="$(realpath -e -- "$campaign")"
(cd -- "$campaign"; sha256sum --quiet -c config.sha256)
source "$campaign/config.sh"
[[ -x "$EVDDP_PYTHON" ]] || fail 'Frozen campaign Python is missing'
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PATH="$(dirname -- "$EVDDP_PYTHON"):$PATH" LD_LIBRARY_PATH="$(dirname -- "$EVDDP_PYTHON")/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ "$mode" == review ]]; then
  export PYTHONPATH="$repo"
  exec "$EVDDP_PYTHON" "$repo/scripts/run_part_ev_ddp_segments.py" --phase review --campaign "$campaign"
fi
for tool in sbatch squeue scontrol; do command -v "$tool" >/dev/null || fail "Missing $tool"; done
for name in ${!SBATCH_@}; do unset "$name"; done
exec 9>"$campaign/.submission.lock"
flock -n 9 || fail 'Another submitter holds the original campaign lock'
# Include the original nine-day chain and EVERY previous segment chain.
live="$(squeue -h -u "$(id -un)" -o '%i' | sed 's/_.*//' | sort -u)"
ledgers=("$campaign/submitted_job_ids.tsv")
shopt -s nullglob
ledgers+=("$campaign"/segment_chains/*/submitted_job_ids.tsv)
for ledger_file in "${ledgers[@]}"; do
  [[ -f "$ledger_file" ]] || continue
  while read -r phase job; do
    [[ "$job" =~ ^[0-9]+$ ]] || fail "Malformed job ledger: $ledger_file"
    if grep -qx "$job" <<< "$live"; then
      fail "Job $job ($phase) is still active. Inspect/cancel the old chain yourself before replacement."
    fi
  done < "$ledger_file"
done
partition_info="$(scontrol show partition "$EVDDP_PARTITION" -o)"
check_time() {
  local seconds=$1 limit day=0 hh mm ss
  limit="$(sed -n 's/.*MaxTime=\([^ ]*\).*/\1/p' <<< "$partition_info")"
  [[ "$limit" != UNLIMITED && "$limit" != INFINITE ]] || return 0
  if [[ "$limit" == *-* ]]; then day="${limit%%-*}"; limit="${limit#*-}"; fi
  IFS=: read -r hh mm ss <<< "$limit"
  [[ "$day:$hh:$mm:$ss" =~ ^[0-9]+:[0-9]+:[0-9]+:[0-9]+$ ]] || fail 'Cannot parse partition limit'
  (( seconds <= 10#$day*86400 + 10#$hh*3600 + 10#$mm*60 + 10#$ss )) || fail 'Requested time exceeds partition limit'
}
if [[ "$mode" == submit ]]; then
  (( reviewed )) || fail 'Run review first, then supply --reviewed'
  check_time 259200
  [[ -z "$(git -C "$repo" status --porcelain -- part_ev_ddp_segments part_ev_fullscale part_ev_screen part_ev_message_study part_ev_state_speed part_ev_state_fullscale part_ev_state_capacity part_ev_state_compare part_layerwise_screen part_ev_gpu_scaling part_ev_layerwise_ddp scripts/*part_ev* sbatch/*part_ev* particle_transformer/data/JetClass/JetClass_full.yaml)" ]] || fail 'Commit, push and pull relevant changes first'
  for file in part_ev_ddp_segments/protocol.py part_ev_ddp_segments/engine.py scripts/run_part_ev_ddp_segments.py sbatch/run_part_ev_ddp_segments.sh; do
    git -C "$repo" cat-file -e "HEAD:$file" || fail 'Implementation must be committed'
  done
  chain="${chain:-$campaign/segment_chains/segments_$(date -u +%Y%m%dT%H%M%SZ)}"
  [[ "$chain" == /* && ! -e "$chain" && ! -L "$chain" ]] || fail 'Use a NEW absolute chain directory'
  [[ "$(realpath -m -- "$(dirname -- "$chain")")" == "$campaign/segment_chains" ]] || fail 'Chain must be directly inside campaign/segment_chains'
  [[ -d "$campaign/segment_chains" ]] || mkdir -- "$campaign/segment_chains"
  mkdir -- "$chain"
  chain="$(realpath -e -- "$chain")"
  mkdir -- "$chain/source" "$chain/logs"
  git -C "$repo" rev-parse HEAD > "$chain/source_commit.txt"
  git -C "$repo" archive HEAD | tar -xf - -C "$chain/source"
  (cd -- "$chain/source"; find . -type f -print0 | sort -z | xargs -0 sha256sum) > "$chain/source.sha256"
  printf 'EVSEG_PYTHON=%q\nEVSEG_CAMPAIGN=%q\nEVSEG_ACCOUNT=%q\nEVSEG_PARTITION=%q\n' \
    "$EVDDP_PYTHON" "$campaign" "$EVDDP_ACCOUNT" "$EVDDP_PARTITION" > "$chain/config.sh"
  (cd -- "$chain"; sha256sum config.sh) > "$chain/config.sha256"
fi
(cd -- "$chain"; sha256sum --quiet -c config.sha256)
source "$chain/config.sh"
[[ "$EVSEG_CAMPAIGN" == "$campaign" && "$EVSEG_PYTHON" == "$EVDDP_PYTHON" && "$EVSEG_ACCOUNT" == "$EVDDP_ACCOUNT" && "$EVSEG_PARTITION" == "$EVDDP_PARTITION" ]] || fail 'Original frozen resources changed'
(cd -- "$chain/source"; sha256sum --quiet -c "$chain/source.sha256")
export PYTHONPATH="$chain/source"
driver="$chain/source/scripts/run_part_ev_ddp_segments.py"
if [[ "$mode" == submit ]]; then
  "$EVSEG_PYTHON" "$driver" --phase prepare --campaign "$campaign" --chain "$chain" --segments "$segments"
else
  "$EVSEG_PYTHON" "$driver" --phase gate --chain "$chain"
fi
eval_hours="$("$EVSEG_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["evaluate_hours"])' "$chain/plan.json")"
[[ "$eval_hours" =~ ^[1-9][0-9]*$ ]] || fail 'Invalid frozen evaluation walltime'
check_time "$((eval_hours*3600))"
check_time 16200
ledger="$chain/submitted_job_ids.tsv"
submit() {
  local phase=$1 dependency=$2 walltime=$3 gpus=$4 index=${5:-} job
  local request=(--parsable --job-name="evseg_$phase" --partition="$EVSEG_PARTITION" --account="$EVSEG_ACCOUNT"
    --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G --time="$walltime"
    --output="$chain/logs/%x_%j.out" --error="$chain/logs/%x_%j.err" --export=ALL --signal=B:TERM@180)
  (( gpus==0 )) || request+=(--gres="gpu:a100:$gpus")
  [[ -z "$dependency" ]] || request+=(--dependency="afterok:$dependency" --kill-on-invalid-dep=yes)
  job="$(EVSEG_CHAIN="$chain" EVSEG_PHASE="$phase" EVSEG_INDEX="$index" sbatch "${request[@]}" "$chain/source/sbatch/run_part_ev_ddp_segments.sh")" || fail "Submission failed; inspect $ledger for jobs already queued. Nothing cancelled."
  job="${job%%;*}"
  [[ "$job" =~ ^[0-9]+$ ]] || fail 'Unexpected sbatch response; inspect scheduler before retrying'
  printf '%s\t%s\n' "${phase}${index:+_$index}" "$job" >> "$ledger"
  printf '%s%s job: %s\n' "$phase" "${index:+_$index}" "$job" >&2
  printf '%s' "$job"
}
dependency=
if [[ "$mode" == submit ]]; then
  for (( index=0; index<segments; index++ )); do
    dependency="$(submit segment "$dependency" 72:00:00 2 "$index")"
  done
  dependency="$(submit gate "$dependency" 00:15:00 0)"
fi
if [[ "$mode" != report ]]; then
  dependency="$(submit evaluate "$dependency" "${eval_hours}:00:00" 1)"
fi
submit report "$dependency" 04:30:00 0
printf '\nCampaign: %s\nChain: %s\nLedger: %s\n' "$campaign" "$chain" "$ledger"
