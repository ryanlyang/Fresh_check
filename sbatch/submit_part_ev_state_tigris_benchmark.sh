#!/usr/bin/env bash
# Independent diagnostic only. Never changes, cancels, or resumes SPORC work.
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
mode=benchmark output= dry=0
evgh_python=/home/ryreu/miniforge3-aarch64/envs/part_ev_fullscale/bin/python
donor=/home/ryreu/atlas/Fresh_check/checkpoints/part_ev_layerwise_ddp/layerwise_2g_20261029T162243Z
account=reu-aisocial partition=tigris
explicit=()
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
while (( $# )); do
  case "$1" in
    --help|-h)
      echo 'benchmark [--output NEW_DIR]: 2h CPU preparation -> 2h one-GH200 benchmark, then STOP.'
      echo 'review --output DIR: read-only timing, memory and full-schedule projection.'
      echo 'Options: --mode benchmark|review --donor DIR --python PATH --account NAME --partition NAME --dry-run'
      echo 'Same all-eight layerwise State-EV, seed101, batch512; no full training, cancellation or environment install.'
      exit 0 ;;
    --dry-run) dry=1; shift ;;
    --mode|--output|--donor|--python|--account|--partition)
      (( $#>=2 )) || fail "Missing $1 value"
      case "$1" in --donor|--python|--account|--partition) explicit+=("$1" "$2") ;; esac
      case "$1" in --mode) mode=$2 ;; --output) output=$2 ;; --donor) donor=$2 ;;
        --python) evgh_python=$2 ;; --account) account=$2 ;; --partition) partition=$2 ;; esac
      shift 2 ;;
    *) fail "Unknown option $1" ;;
  esac
done
case "$mode" in benchmark|review) ;; *) fail 'Only benchmark/review supported; full training is not submitted' ;; esac
for name in "$account" "$partition"; do [[ "$name" =~ ^[a-zA-Z0-9_-]+$ ]] || fail 'Invalid resource name'; done
if [[ -z "$output" && "$mode" == benchmark ]]; then
  output="$repo/checkpoints/part_ev_state_tigris/gh200_$(date -u +%Y%m%dT%H%M%SZ)"
fi
[[ "$output" == /* && "$donor" == /* && "$evgh_python" == /* ]] || fail 'Use absolute paths'
if (( dry )); then
  printf 'Mode: %s\nOutput: %s\nRead-only SPORC donor: %s\n' "$mode" "$output" "$donor"
  echo 'CPU prepare 02:00:00 -> --gpus-per-node=1 --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=128G --time=02:00:00'
  printf 'Partition: %s; account: %s. Runtime must be ARM/GH200. No training or cancellation.\n' "$partition" "$account"
  exit 0
fi
[[ "$(uname -m)" == aarch64 ]] || fail 'Run on tigris, not sporcsubmit'
for tool in git tar sha256sum find sort xargs realpath flock; do command -v "$tool" >/dev/null || fail "Missing $tool"; done
if [[ "$mode" == benchmark ]]; then
  for tool in sbatch scontrol; do command -v "$tool" >/dev/null || fail "Missing $tool"; done
  [[ ! -e "$output" && ! -L "$output" ]] || fail 'Use a NEW diagnostic directory; existing evidence is never overwritten'
  [[ -x "$evgh_python" && -f "$donor/campaign.json" ]] || fail 'Missing existing ARM Python or SPORC donor; no installations performed'
  donor="$(realpath -e -- "$donor")"
  candidate="$(realpath -m -- "$output")"
  [[ "$candidate" != "$donor" && "$candidate" != "$donor/"* && "$donor" != "$candidate/"* ]] || fail 'Output must be separate from SPORC campaign'
  [[ -z "$(git -C "$repo" status --porcelain -- part_ev_state_tigris part_ev_fullscale part_ev_screen part_ev_message_study part_ev_state_speed part_ev_state_fullscale part_ev_state_capacity part_ev_state_compare part_layerwise_screen part_ev_gpu_scaling part_ev_layerwise_ddp scripts/*part_ev* sbatch/*part_ev* particle_transformer/data/JetClass/JetClass_full.yaml)" ]] || fail 'Commit/push/pull relevant implementation first'
  for name in part_ev_state_tigris/protocol.py part_ev_state_tigris/benchmark.py scripts/benchmark_part_ev_state_tigris.py sbatch/run_part_ev_state_tigris_benchmark.sh; do
    git -C "$repo" cat-file -e "HEAD:$name" || fail 'Commit/push/pull implementation first'
  done
  info="$(scontrol show partition "$partition" -o)"
  limit="$(sed -n 's/.*MaxTime=\([^ ]*\).*/\1/p' <<< "$info")"
  if [[ "$limit" != UNLIMITED && "$limit" != INFINITE ]]; then
    day=0
    if [[ "$limit" == *-* ]]; then day=${limit%%-*}; limit=${limit#*-}; fi
    IFS=: read -r hh mm ss <<< "$limit"
    [[ "$day:$hh:$mm:$ss" =~ ^[0-9]+:[0-9]+:[0-9]+:[0-9]+$ ]] || fail 'Cannot parse partition time limit'
    (( 10#$day*86400+10#$hh*3600+10#$mm*60+10#$ss >= 7200 )) || fail 'Partition must allow the two-hour diagnostic'
  fi
  parent="$(dirname -- "$output")"
  [[ -d "$parent" ]] || mkdir -p -- "$parent"
  mkdir -- "$output" || fail 'Concurrent creation; refusing overwrite'
  output="$(realpath -e -- "$output")"
  mkdir -- "$output/source" "$output/logs"
  git -C "$repo" rev-parse HEAD > "$output/source_commit.txt"
  git -C "$repo" archive HEAD | tar -xf - -C "$output/source"
  (cd -- "$output/source"; find . -type f -print0 | sort -z | xargs -0 sha256sum) > "$output/source.sha256"
  printf 'EVGH_PYTHON=%q\nEVGH_DONOR=%q\nEVGH_ACCOUNT=%q\nEVGH_PARTITION=%q\n' \
    "$evgh_python" "$donor" "$account" "$partition" > "$output/launch.env"
  "$evgh_python" - "$output/launch.json" "$evgh_python" "$donor" "$account" "$partition" <<'PY'
import json,sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps(dict(zip(('python','donor','account','partition'),sys.argv[2:])),indent=2)+'\n')
PY
  (cd -- "$output"; sha256sum launch.env launch.json) > "$output/launch.sha256"
fi
output="$(realpath -e -- "$output")"
(cd -- "$output"; sha256sum --quiet -c launch.sha256)
source "$output/launch.env"
for ((i=0;i<${#explicit[@]};i+=2)); do
  case "${explicit[i]}" in --donor) frozen=$EVGH_DONOR ;; --python) frozen=$EVGH_PYTHON ;;
    --account) frozen=$EVGH_ACCOUNT ;; --partition) frozen=$EVGH_PARTITION ;; esac
  [[ "${explicit[i+1]}" == "$frozen" ]] || fail 'Use frozen resources for this diagnostic'
done
(cd -- "$output/source"; sha256sum --quiet -c "$output/source.sha256")
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$output/source"
export PATH="$(dirname -- "$EVGH_PYTHON"):$PATH" LD_LIBRARY_PATH="$(dirname -- "$EVGH_PYTHON")/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ "$mode" == review ]]; then
  exec "$EVGH_PYTHON" "$output/source/scripts/benchmark_part_ev_state_tigris.py" --phase review --output "$output"
fi
for name in ${!SBATCH_@}; do unset "$name"; done
exec 9>"$output/.submission.lock"
flock -n 9 || fail 'Another submitter holds the diagnostic lock'
submit() {
  local phase=$1 dependency=$2 job
  local request=(--parsable --job-name="evgh_$phase" --partition="$EVGH_PARTITION" --account="$EVGH_ACCOUNT"
    --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=128G --time=02:00:00
    --output="$output/logs/%x_%j.out" --error="$output/logs/%x_%j.err" --export=ALL --signal=B:USR1@180)
  [[ "$phase" != benchmark ]] || request+=(--gpus-per-node=1)
  [[ -z "$dependency" ]] || request+=(--dependency="afterok:$dependency" --kill-on-invalid-dep=yes)
  job="$(EVGH_OUTPUT="$output" EVGH_PHASE="$phase" sbatch "${request[@]}" "$output/source/sbatch/run_part_ev_state_tigris_benchmark.sh")" || fail "Submission failed. Inspect $output/submitted_job_ids.tsv for jobs already submitted."
  job=${job%%;*}
  [[ "$job" =~ ^[0-9]+$ ]] || fail 'Unexpected sbatch response; inspect scheduler'
  printf '%s\t%s\n' "$phase" "$job" >> "$output/submitted_job_ids.tsv"
  printf '%s job: %s\n' "$phase" "$job" >&2
  printf '%s' "$job"
}
prepare_job="$(submit prepare '')"
submit benchmark "$prepare_job"
printf '\nDiagnostic: %s\nNo SPORC jobs changed. No full training submitted.\n' "$output"
