#!/usr/bin/env bash
# Four selected-relation runs with automatic prepare/benchmark gates and read-only controls.
set -euo pipefail
selected_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
mode=benchmark output= resume=0 dry=0 concurrency=2 train_hours=96 evaluate_hours=12
selected_python=/home/ryreu/miniforge3-aarch64/envs/part_selected_tigris/bin/python
donor=/home/ryreu/atlas/Fresh_check/checkpoints/part_ev_screen/context_gate_ampfix_20261016T215608Z
layerwise_donor=/home/ryreu/atlas/Fresh_check/checkpoints/part_layerwise_screen/layerwise_20261019T190751Z
account=reu-aisocial partition=tigris gpu=any cpus=8 memory=128G
explicit=()
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
usage() {
  cat <<'USAGE'
Usage: bash sbatch/submit_part_selected_tigris.sh [options]
Default benchmark: prepare(5h) -> benchmark[4](5h) -> gate(2h), then STOP.
Mode run adds train[4] -> lock -> evaluate[4] -> report after the gate.
Four fresh seed-101 Ranger models; four completed controls reused read-only.
  --mode benchmark|run|train|evaluate|report|status|review   Default benchmark.
  Status (validation) and review (benchmarks) are read-only.
  --output DIR --donor DIR --layerwise-donor DIR --python PATH
  --partition NAME --account NAME --gpu TYPE --cpus N --mem SIZE
  Default: tigris, reu-aisocial, one GPU (any), 8 CPUs, 128G host RAM.
  Runtime must be GH200 with at least 80 GiB GPU memory.
  --concurrency 1..4                       Default 2.
  --train-hours N --evaluate-hours N       Defaults 96 / 12; checked against benchmarks.
  --resume                                Only after all prior campaign jobs terminate.
  --dry-run                               No files or jobs changed.
Mode run queues the full dependency graph; failed checks block training.
Controls remain on SPORC; comparisons to them are cross-hardware exploratory.
Use --mode train --resume only when prepare/benchmarks already passed.
No test data, control retraining, environment installation, cancellation or deletion.
Commit/push/pull implementation before queueing.
USAGE
}
while (( $# )); do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --resume) resume=1; shift ;; --dry-run) dry=1; shift ;;
    --mode|--output|--donor|--layerwise-donor|--python|--partition|--account|--gpu|--cpus|--mem|--concurrency|--train-hours|--evaluate-hours)
      (( $# >= 2 )) || fail "Missing $1 value"
      case "$1" in --donor|--layerwise-donor|--python|--partition|--account|--gpu|--cpus|--mem|--train-hours|--evaluate-hours) explicit+=("$1" "$2") ;; esac
      case "$1" in
        --layerwise-donor) layerwise_donor=$2 ;;
        --mode) mode=$2 ;; --output) output=$2 ;; --donor) donor=$2 ;; --python) selected_python=$2 ;;
        --partition) partition=$2 ;; --account) account=$2 ;; --gpu) gpu=$2 ;; --cpus) cpus=$2 ;;
        --mem) memory=$2 ;; --concurrency) concurrency=$2 ;; --train-hours) train_hours=$2 ;;
        --evaluate-hours) evaluate_hours=$2 ;;
      esac
      shift 2 ;;
    *) fail "Unknown option: $1" ;;
  esac
done
case "$mode" in benchmark|run|train|evaluate|report|status|review) ;; *) fail 'Unknown mode; see --help' ;; esac
[[ "$concurrency" =~ ^[1-4]$ && "$cpus" =~ ^[1-9][0-9]*$ ]] || fail 'Invalid concurrency/CPU count'
[[ "$memory" =~ ^[1-9][0-9]*[MGT]$ ]] || fail 'Invalid memory'
[[ "$train_hours" =~ ^[1-9][0-9]*$ && "$evaluate_hours" =~ ^[1-9][0-9]*$ ]] || fail 'Invalid walltime'
for value in "$partition" "$account" "$gpu"; do [[ "$value" =~ ^[a-zA-Z0-9_-]+$ ]] || fail 'Invalid resource name'; done
if [[ -z "$output" && ( "$mode" == run || "$mode" == benchmark ) && "$resume" == 0 ]]; then
  output="$selected_repo/checkpoints/part_selected_tigris/selected_gh200_$(date -u +%Y%m%dT%H%M%SZ)"
fi
[[ "$output" == /* && "$donor" == /* && "$layerwise_donor" == /* && "$selected_python" == /* ]] || fail 'Use absolute paths; --resume needs --output'
if (( dry )); then
  printf 'Mode: %s\nCampaign: %s\nDonor (read-only): %s\n' "$mode" "$output" "$donor"
  gpu_request=--gpus-per-node=1
  [[ "$gpu" == any ]] || gpu_request="--gres=gpu:$gpu:1"
  printf 'New models: --array=0-3%%%s %s --cpus-per-task=%s --mem=%s --partition=%s --account=%s\n' "$concurrency" "$gpu_request" "$cpus" "$memory" "$partition" "$account"
  if [[ "$mode" == benchmark ]]; then
    echo 'prepare 5h -> benchmark[4] 5h -> gate 2h -> STOP (no training submitted)'
  else
    printf 'prepare 5h -> benchmark[4] 5h -> gate 2h -> train[4] %sh -> lock 2h -> evaluate[4] %sh -> report 2h\n' "$train_hours" "$evaluate_hours"
  fi
  printf 'Layerwise donor (read-only): %s\n' "$layerwise_donor"
  echo 'No donor training or inference. No files or jobs changed.'
  exit 0
fi
[[ "$(uname -m)" == aarch64 ]] || fail 'Run on Tigris using an ARM CUDA environment'
for command in git tar sha256sum find sort xargs realpath flock sbatch squeue scontrol; do
  command -v "$command" >/dev/null || fail "Missing command: $command"
done
for option_name in ${!SBATCH_@}; do unset "$option_name"; done
if [[ ! -f "$output/launch.env" ]]; then
  [[ ( "$mode" == run || "$mode" == benchmark ) && "$resume" == 0 ]] || fail 'Frozen campaign missing'
  [[ ! -e "$output" && ! -L "$output" ]] || fail 'Output exists; refusing overwrite'
  [[ -x "$selected_python" && -f "$donor/campaign.json" && -f "$layerwise_donor/experiment_spec.json" ]] || fail 'Missing Tigris Python or read-only donor campaign'
  donor="$(realpath -e -- "$donor")"
  layerwise_donor="$(realpath -e -- "$layerwise_donor")"
  candidate="$(realpath -m -- "$output")"
  case "$candidate/" in "$donor/"*) fail 'Output must not be inside donor' ;; esac
  case "$donor/" in "$candidate/"*) fail 'Output must not contain donor' ;; esac
  case "$candidate/" in "$layerwise_donor/"*) fail 'Output must not be inside layerwise donor' ;; esac
  case "$layerwise_donor/" in "$candidate/"*) fail 'Output must not contain layerwise donor' ;; esac
  for name in part_selected_screen/model.py part_selected_screen/reporting.py part_selected_tigris/protocol.py part_selected_tigris/engine.py part_selected_tigris/inputs.py part_selected_tigris/reporting.py scripts/run_part_selected_tigris.py sbatch/run_part_selected_tigris.sh; do
    git -C "$selected_repo" cat-file -e "HEAD:$name" || fail 'Commit/push/pull implementation before queueing'
  done
  [[ -z "$(git -C "$selected_repo" status --porcelain -- part_selected_tigris part_selected_screen part_layerwise_screen teacher_logit_reco/relational_part part_screen_common part_ev_screen part_ev_fullscale scripts/run_part_selected_tigris.py scripts/setup_part_selected_tigris_env.sh scripts/run_part_selected_screen.py scripts/run_part_ev_screen.py scripts/build_relational_part_tree_backend.py sbatch/*part_selected_screen* sbatch/*part_selected_tigris* particle_transformer/data/JetClass/JetClass_full.yaml)" ]] || fail 'Commit relevant source changes before queueing'
  selected_parent="$(dirname -- "$output")"
  [[ -d "$selected_parent" ]] || mkdir -p -- "$selected_parent"
  mkdir -- "$output" || fail 'Concurrent submission; refusing overwrite'
  output="$(realpath -e -- "$output")"
  mkdir -- "$output/source" "$output/logs"
  git -C "$selected_repo" rev-parse HEAD > "$output/source_commit.txt"
  git -C "$selected_repo" archive HEAD | tar -xf - -C "$output/source"
  (cd -- "$output/source"; find . -type f -print0 | sort -z | xargs -0 sha256sum) > "$output/source.sha256"
  printf 'SELECTED_PYTHON=%q\nSELECTED_DONOR=%q\nSELECTED_LAYERWISE_DONOR=%q\nSELECTED_ACCOUNT=%q\nSELECTED_PARTITION=%q\nSELECTED_GPU=%q\nSELECTED_CPUS=%q\nSELECTED_MEMORY=%q\nSELECTED_TRAIN_HOURS=%q\nSELECTED_EVALUATE_HOURS=%q\n' \
    "$selected_python" "$donor" "$layerwise_donor" "$account" "$partition" "$gpu" "$cpus" "$memory" "$train_hours" "$evaluate_hours" > "$output/launch.env"
  "$selected_python" - "$output/launch.json" "$donor" "$layerwise_donor" "$selected_python" "$account" "$partition" "$gpu" "$cpus" "$memory" "$train_hours" "$evaluate_hours" <<'PY'
import json, sys
from pathlib import Path
keys = ('donor', 'layerwise_donor', 'python', 'account', 'partition', 'gpu', 'cpus', 'memory', 'train_hours', 'evaluate_hours')
row = dict(zip(keys, sys.argv[2:]))
row['controls'] = ['PART_RANGER', 'EV_RANGER', 'PART_LAYERWISE_RANGER', 'EV_LAYERWISE_RANGER']
row['submission_policy'] = 'gh200_benchmark_gate_before_four_fresh_runs'
row['hardware_matched_to_sporc_controls'] = False
Path(sys.argv[1]).write_text(json.dumps(row, indent=2) + '\n')
PY
elif [[ ( "$mode" == run || "$mode" == benchmark ) && "$resume" == 0 ]]; then
  fail 'Existing campaign: inspect state then explicitly --resume; no duplicate submission'
fi
output="$(realpath -e -- "$output")"
source "$output/launch.env"
for ((i=0; i<${#explicit[@]}; i+=2)); do
  case "${explicit[i]}" in
    --layerwise-donor) frozen=$SELECTED_LAYERWISE_DONOR ;; --python) frozen=$SELECTED_PYTHON ;; --donor) frozen=$SELECTED_DONOR ;; --partition) frozen=$SELECTED_PARTITION ;;
    --account) frozen=$SELECTED_ACCOUNT ;; --gpu) frozen=$SELECTED_GPU ;; --cpus) frozen=$SELECTED_CPUS ;; --mem) frozen=$SELECTED_MEMORY ;;
    --train-hours) frozen=$SELECTED_TRAIN_HOURS ;; --evaluate-hours) frozen=$SELECTED_EVALUATE_HOURS ;;
  esac
  [[ "${explicit[i+1]}" == "$frozen" ]] || fail "${explicit[i]} differs from frozen campaign"
done
train_hours=$SELECTED_TRAIN_HOURS
evaluate_hours=$SELECTED_EVALUATE_HOURS
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$output/source"
export PATH="$(dirname -- "$SELECTED_PYTHON"):$PATH"
export LD_LIBRARY_PATH="$(dirname -- "$SELECTED_PYTHON")/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
(cd -- "$output/source"; sha256sum --quiet -c "$output/source.sha256")
if [[ "$mode" == status || "$mode" == review ]]; then
  if [[ ! -f "$output/experiment_spec.json" ]]; then
    echo 'Preparation has not completed yet (possibly still queued).'
    [[ ! -f "$output/submitted_job_ids.tsv" ]] || cat "$output/submitted_job_ids.tsv"
    exit 0
  fi
  exec "$SELECTED_PYTHON" "$output/source/scripts/run_part_selected_tigris.py" --phase "$mode" --output "$output"
fi
exec 9>"$output/.submission.lock"
flock -n 9 || fail 'Concurrent submitter'
ledger="$output/submitted_job_ids.tsv"
if [[ -s "$ledger" ]]; then
  live="$(squeue -h -u "$(id -un)" -o '%i' | sed 's/_.*//' | sort -u)"
  while read -r phase job; do
    if grep -qx "$job" <<< "$live"; then fail "Campaign job $job ($phase) is active; no duplicates queued"; fi
  done < "$ledger"
  (( resume )) || fail 'Use --resume only after prior jobs terminate'
fi
scontrol show partition "$SELECTED_PARTITION" >/dev/null || fail 'Requested partition unavailable'
submit() {
  local phase=$1 dependency=$2 hours=$3 result
  local request=(--parsable --job-name="selgh_${phase}" --account="$SELECTED_ACCOUNT" --partition="$SELECTED_PARTITION"
    --nodes=1 --ntasks=1 --cpus-per-task="$SELECTED_CPUS" --mem="$SELECTED_MEMORY" --time="${hours}:00:00"
    --output="$output/logs/%x_%A_%a.out" --error="$output/logs/%x_%A_%a.err" --export=ALL --signal=B:USR1@180)
  case "$phase" in benchmark|train|evaluate)
    request+=(--array="0-3%$concurrency")
    if [[ "$SELECTED_GPU" == any ]]; then request+=(--gpus-per-node=1)
    else request+=(--gres="gpu:$SELECTED_GPU:1"); fi ;;
  esac
  [[ -z "$dependency" ]] || request+=(--dependency="afterok:$dependency" --kill-on-invalid-dep=yes)
  result="$(PART_SELECTED_TIGRIS_OUTPUT="$output" PART_SELECTED_TIGRIS_PHASE="$phase" sbatch "${request[@]}" "$output/source/sbatch/run_part_selected_tigris.sh")"
  result="${result%%;*}"
  [[ "$result" =~ ^[0-9]+$ ]] || fail "Unexpected Slurm response: $result; inspect scheduler before retrying"
  printf '%s\t%s\n' "$phase" "$result" >> "$ledger"
  printf '%s job: %s\n' "$phase" "$result" >&2
  printf '%s' "$result"
}
printf 'Campaign: %s\n' "$output"
if [[ "$mode" == report ]]; then
  report_id="$(submit report '' 2)"
else
  dependency=
  if [[ "$mode" == run || "$mode" == benchmark ]]; then
    prep_id="$(submit prepare '' 5)"
    bench_id="$(submit benchmark "$prep_id" 5)"
    gate_id="$(submit gate "$bench_id" 2)"
    if [[ "$mode" == benchmark ]]; then
      printf 'Benchmarks queued; no training submitted. After success: --mode train --output %q --resume\n' "$output"
      exit 0
    fi
    dependency="$(submit train "$gate_id" "$train_hours")"
  elif [[ "$mode" == train ]]; then
    gate_id="$(submit gate '' 2)"
    dependency="$(submit train "$gate_id" "$train_hours")"
  fi
  lock_id="$(submit lock "$dependency" 2)"
  eval_id="$(submit evaluate "$lock_id" "$evaluate_hours")"
  report_id="$(submit report "$eval_id" 2)"
fi
printf 'Job ledger: %s\nAll checks are dependency-gated; no manual preflight submission needed.\n' "$ledger"
