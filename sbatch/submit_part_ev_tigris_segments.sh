#!/usr/bin/env bash
# Queue all successors now. No cancellation, checkpoint migration or self-submission.
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
mode= benchmark= benchmark_job= campaign= chain= segments=4 dry=0 reviewed=0
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
while (( $# )); do
  case "$1" in
    --help|-h)
      echo 'submit --benchmark-job ID --reviewed [--benchmark DIR] [--segments 4]'
      echo 'continue --campaign DIR --reviewed [--segments 4]: new chain after old jobs are terminal'
      echo 'review --campaign DIR; evaluate|report --chain DIR: retry downstream phases'
      echo '--dry-run: print only. All training slots are 72h, one GH200; no SPORC changes.'
      exit 0 ;;
    --dry-run) dry=1; shift ;; --reviewed) reviewed=1; shift ;;
    --mode|--benchmark|--benchmark-job|--campaign|--chain|--segments)
      (( $#>=2 )) || fail "Missing $1 value"
      case "$1" in --mode) mode=$2 ;; --benchmark) benchmark=$2 ;; --benchmark-job) benchmark_job=$2 ;;
        --campaign) campaign=$2 ;; --chain) chain=$2 ;; --segments) segments=$2 ;; esac
      shift 2 ;;
    *) fail "Unknown option $1" ;;
  esac
done
case "$mode" in
  submit)
    [[ "$benchmark_job" =~ ^[1-9][0-9]*$ && -z "$campaign" && -z "$chain" ]] || fail 'Use --benchmark-job for a fresh campaign'
    if [[ -z "$benchmark" ]]; then
      matches=()
      for ledger in "$repo"/checkpoints/part_ev_state_tigris/*/submitted_job_ids.tsv; do
        [[ -f "$ledger" ]] || continue
        if awk -v job="$benchmark_job" '$1=="benchmark" && $2==job {ok=1} END {exit !ok}' "$ledger"; then
          matches+=("$(dirname -- "$ledger")")
        fi
      done
      (( ${#matches[@]}==1 )) || fail 'Expected exactly one local benchmark ledger for this ID; supply --benchmark DIR explicitly'
      benchmark=${matches[0]}
    fi
    [[ "$benchmark" == /* ]] || fail 'Use an absolute benchmark path'
    campaign="$repo/checkpoints/part_ev_tigris_segments/benchmark_$benchmark_job" ;;
  continue|review) [[ "$campaign" == /* && -z "$chain" ]] || fail 'Use absolute --campaign' ;;
  evaluate|report) [[ "$chain" == /* ]] || fail 'Use absolute --chain' ;;
  *) fail 'Choose --mode; see --help' ;;
esac
[[ "$segments" =~ ^[1-8]$ ]] || fail 'Choose 1..8 segments'
if (( dry )); then
  printf 'Mode: %s\nCampaign: %s\nBenchmark job: %s\n' "$mode" "$campaign" "$benchmark_job"
  printf 'Benchmark -> CPU admission -> %s x 72h one-GH200 slots -> completion gate -> 5h full-validation -> report.\n' "$segments"
  echo 'Stop at allocation end minus one hour. No files/jobs changed.'
  exit 0
fi
[[ "$(uname -m)" == aarch64 ]] || fail 'Run on Tigris'
for tool in git tar sha256sum find sort xargs realpath flock; do command -v "$tool" >/dev/null || fail "Missing $tool"; done
if [[ "$mode" == evaluate || "$mode" == report ]]; then
  chain="$(realpath -e -- "$chain")"
  (cd -- "$chain"; sha256sum --quiet -c config.sha256)
  source "$chain/config.sh"
  campaign="$EVTG_CAMPAIGN"
fi
if [[ "$mode" == submit ]]; then
  benchmark="$(realpath -e -- "$benchmark")"
  (cd -- "$benchmark"; sha256sum --quiet -c launch.sha256)
  source "$benchmark/launch.env"
  EVTG_PYTHON=$EVGH_PYTHON EVTG_ACCOUNT=$EVGH_ACCOUNT EVTG_PARTITION=$EVGH_PARTITION
else
  campaign="$(realpath -e -- "$campaign")"
  (cd -- "$campaign"; sha256sum --quiet -c launch.sha256)
  source "$campaign/launch.env"
  benchmark=$EVTG_BENCHMARK benchmark_job=$EVTG_BENCHMARK_JOB
fi
[[ -x "$EVTG_PYTHON" ]] || fail 'Frozen ARM Python is missing'
[[ "$EVTG_PARTITION" == tigris && "$EVTG_ACCOUNT" =~ ^[a-zA-Z0-9_-]+$ ]] || fail 'Expected frozen Tigris resources'
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PATH="$(dirname -- "$EVTG_PYTHON"):$PATH" LD_LIBRARY_PATH="$(dirname -- "$EVTG_PYTHON")/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ "$mode" == review ]]; then
  export PYTHONPATH="$repo"
  exec "$EVTG_PYTHON" "$repo/scripts/run_part_ev_tigris_segments.py" --phase review --campaign "$campaign"
fi
for tool in sbatch squeue scontrol sacct; do command -v "$tool" >/dev/null || fail "Missing $tool"; done
for name in ${!SBATCH_@}; do unset "$name"; done
if [[ "$mode" == submit ]]; then
  parent="$(dirname -- "$campaign")"
  [[ -d "$parent" ]] || mkdir -p -- "$parent"
  exec 9>"$parent/.submission_$benchmark_job.lock"
else
  exec 9>"$campaign/.submission.lock"
fi
flock -n 9 || fail 'Another submitter is active'
if [[ "$mode" == submit ]]; then
  [[ ! -e "$campaign" && ! -L "$campaign" ]] || fail 'Campaign exists; inspect it, then use continue if needed'
fi
live="$(squeue -h -u "$(id -un)" -o '%i' | sed 's/_.*//' | sort -u)"
shopt -s nullglob
for ledger in "$campaign"/segment_chains/*/submitted_job_ids.tsv; do
  while read -r phase job; do
    [[ "$job" =~ ^[0-9]+$ ]] || fail "Malformed ledger $ledger"
    ! grep -qx "$job" <<< "$live" || fail "Job $job ($phase) is still active; no duplicate chain submitted"
  done < "$ledger"
done
dependency=
if [[ "$mode" == submit || "$mode" == continue ]]; then
  (( reviewed )) || fail 'Use --reviewed to authorize automatic training only after benchmark admission'
  [[ "$(awk '$1=="benchmark" {print $2}' "$benchmark/submitted_job_ids.tsv")" == "$benchmark_job" ]] || fail 'Benchmark job does not match its ledger'
  if grep -qx "$benchmark_job" <<< "$live"; then
    dependency=$benchmark_job
  else
    status="$(sacct -j "$benchmark_job" -X -n -P -o JobID,State,ExitCode)"
    awk -F'|' -v job="$benchmark_job" '$1==job && $2=="COMPLETED" && $3=="0:0" {ok=1} END {exit !ok}' <<< "$status" || fail 'Benchmark is neither active nor successfully completed; inspect its state first'
    # Already retired from Slurm: the CPU admission still requires all benchmark evidence.
  fi
  partition_info="$(scontrol show partition "$EVTG_PARTITION" -o)"
  limit="$(sed -n 's/.*MaxTime=\([^ ]*\).*/\1/p' <<< "$partition_info")"
  if [[ "$limit" != UNLIMITED && "$limit" != INFINITE ]]; then
    day=0
    if [[ "$limit" == *-* ]]; then day=${limit%%-*}; limit=${limit#*-}; fi
    IFS=: read -r hh mm ss <<< "$limit"
    [[ "$day:$hh:$mm:$ss" =~ ^[0-9]+:[0-9]+:[0-9]+:[0-9]+$ ]] || fail 'Cannot parse partition limit'
    (( 10#$day*86400+10#$hh*3600+10#$mm*60+10#$ss>=259200 )) || fail 'Partition does not permit 72-hour slots'
  fi
  [[ -z "$(git -C "$repo" status --porcelain -- part_ev_tigris_segments part_ev_state_tigris part_ev_fullscale part_ev_screen part_ev_message_study part_ev_state_speed part_ev_state_fullscale part_ev_state_capacity part_ev_state_compare part_layerwise_screen part_ev_gpu_scaling part_ev_layerwise_ddp scripts/*part_ev* sbatch/*part_ev* particle_transformer/data/JetClass/JetClass_full.yaml)" ]] || fail 'Commit/push/pull relevant changes first'
  for file in part_ev_tigris_segments/protocol.py part_ev_tigris_segments/engine.py scripts/run_part_ev_tigris_segments.py sbatch/run_part_ev_tigris_segments.sh; do
    git -C "$repo" cat-file -e "HEAD:$file" || fail 'Implementation must be committed'
  done
  if [[ "$mode" == submit ]]; then
    mkdir -- "$campaign"
    exec 8>"$campaign/.submission.lock"
    flock -n 8 || fail 'Another continuation submitter is active'
    printf 'EVTG_PYTHON=%q\nEVTG_ACCOUNT=%q\nEVTG_PARTITION=%q\nEVTG_BENCHMARK=%q\nEVTG_BENCHMARK_JOB=%q\n' \
      "$EVTG_PYTHON" "$EVTG_ACCOUNT" "$EVTG_PARTITION" "$benchmark" "$benchmark_job" > "$campaign/launch.env"
    "$EVTG_PYTHON" - "$campaign/launch.json" "$EVTG_PYTHON" "$benchmark" "$benchmark_job" "$EVTG_ACCOUNT" "$EVTG_PARTITION" <<'PY'
import json,sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps(dict(zip(('python','benchmark','benchmark_job','account','partition'),sys.argv[2:])),indent=2)+'\n')
PY
    (cd -- "$campaign"; sha256sum launch.env launch.json) > "$campaign/launch.sha256"
  fi
  [[ -d "$campaign/segment_chains" ]] || mkdir -- "$campaign/segment_chains"
  chain="$(mktemp -d "$campaign/segment_chains/segments.XXXXXXXX")"
  mkdir -- "$chain/source" "$chain/logs"
  git -C "$repo" rev-parse HEAD > "$chain/source_commit.txt"
  git -C "$repo" archive HEAD | tar -xf - -C "$chain/source"
  (cd -- "$chain/source"; find . -type f -print0 | sort -z | xargs -0 sha256sum) > "$chain/source.sha256"
  printf 'EVTG_CAMPAIGN=%q\n' "$campaign" > "$chain/config.sh"
  (cd -- "$chain"; sha256sum config.sh) > "$chain/config.sha256"
fi
(cd -- "$chain"; sha256sum --quiet -c config.sha256)
(cd -- "$chain/source"; sha256sum --quiet -c "$chain/source.sha256")
export PYTHONPATH="$chain/source"
driver="$chain/source/scripts/run_part_ev_tigris_segments.py"
if [[ "$mode" == submit || "$mode" == continue ]]; then
  phase=prepare-chain
  [[ "$mode" != submit ]] || phase=prepare
  "$EVTG_PYTHON" "$driver" --phase "$phase" --campaign "$campaign" --benchmark "$benchmark" \
    --benchmark-job "$benchmark_job" --chain "$chain" --segments "$segments"
else
  "$EVTG_PYTHON" "$driver" --phase gate --chain "$chain"
fi
ledger="$chain/submitted_job_ids.tsv"
submit() {
  local phase=$1 dep=$2 walltime=$3 gpu=$4 index=${5:-} job
  local args=(--parsable --job-name="evtg_$phase" --partition="$EVTG_PARTITION" --account="$EVTG_ACCOUNT"
    --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=128G --time="$walltime" --export=ALL
    --output="$chain/logs/%x_%j.out" --error="$chain/logs/%x_%j.err" --signal=B:TERM@180)
  (( gpu==0 )) || args+=(--gpus-per-node=1)
  [[ -z "$dep" ]] || args+=(--dependency="afterok:$dep" --kill-on-invalid-dep=yes)
  job="$(EVTG_CHAIN="$chain" EVTG_PHASE="$phase" EVTG_INDEX="$index" sbatch "${args[@]}" "$chain/source/sbatch/run_part_ev_tigris_segments.sh")" || fail "Submission failed; inspect $ledger for jobs already queued. Nothing cancelled."
  job=${job%%;*}
  [[ "$job" =~ ^[0-9]+$ ]] || fail 'Unexpected sbatch response; inspect scheduler'
  printf '%s\t%s\n' "${phase}${index:+_$index}" "$job" >> "$ledger"
  printf '%s%s job: %s\n' "$phase" "${index:+_$index}" "$job" >&2
  printf '%s' "$job"
}
if [[ "$mode" == submit || "$mode" == continue ]]; then
  dependency="$(submit admit "$dependency" 00:30:00 0)"
  for ((index=0;index<segments;index++)); do
    dependency="$(submit segment "$dependency" 72:00:00 1 "$index")"
  done
  dependency="$(submit gate "$dependency" 00:15:00 0)"
fi
if [[ "$mode" != report ]]; then dependency="$(submit evaluate "$dependency" 05:00:00 1)"; fi
submit report "$dependency" 04:30:00 0
printf '\nCampaign: %s\nChain: %s\nNo SPORC jobs or benchmark artifacts changed.\n' "$campaign" "$chain"
