#!/usr/bin/env bash
# A dedicated screen; no changes to full-scale or historical campaign launchers.
set -euo pipefail
screen_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
mode= output= resume=0 dry=0 reviewed=0 concurrency=2 time_hours=
screen_python=/home/ryreu/miniconda3/envs/part_ev_screen_sporc/bin/python
screen_data=/home/ryreu/atlas/datasets/JetClass
account=reu-aisocial partition=tier3 gpu=a100 cpus=8 memory=64G
explicit_resources=()
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
usage() {
  cat <<'USAGE'
Usage: bash sbatch/submit_part_ev_screen_sporc.sh --mode MODE [options]
  prepare    Freeze committed source; queue 2h prepare -> seven 2h GPU benchmarks.
  benchmark  Retry the benchmark array after a terminal/incomplete attempt.
  review     Print benchmark results and a 30%-margin production time estimate.
  train      Queue seven trainings -> seven full-5M-validation evaluations -> report.
  evaluate   Retry full validation + report after terminal downstream jobs.
  report     Retry report only.
Options:
  --output DIR       New prepare directory or existing campaign for other phases.
  --python PATH      Dedicated x86 CUDA environment Python (prepare only).
  --data-root DIR    Official JetClass parent with Pythia/train_100M and val_5M.
  --partition NAME   Default tier3. Verify it exists and your account can use it.
  --account NAME     Default reu-aisocial.
  --gpu TYPE         Default a100; frozen across all benchmark/train/evaluation jobs.
  --cpus N --mem SIZE  Default 8 / 64G; frozen at prepare.
  --concurrency N    Default 2, at most 7 simultaneous tasks.
  --reviewed        Required to train after examining all seven benchmarks.
  --train-hours N   Optional walltime >= the 30%-margin measured recommendation.
  --resume          Retry terminal submissions; refuses active duplicate pipelines.
  --dry-run         Show requests; create no files or jobs.
No test-set evaluation, job cancellation, deletion, or environment installation.
USAGE
}
while (( $# )); do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --resume) resume=1; shift ;; --dry-run) dry=1; shift ;; --reviewed) reviewed=1; shift ;;
    --mode|--output|--python|--data-root|--partition|--account|--gpu|--cpus|--mem|--concurrency|--train-hours)
      (( $# >= 2 )) || fail "Missing $1 value"
      case "$1" in --python|--data-root|--partition|--account|--gpu|--cpus|--mem) explicit_resources+=("$1" "$2") ;; esac
      case "$1" in
        --mode) mode=$2 ;; --output) output=$2 ;; --python) screen_python=$2 ;;
        --data-root) screen_data=$2 ;; --partition) partition=$2 ;; --account) account=$2 ;;
        --gpu) gpu=$2 ;; --cpus) cpus=$2 ;; --mem) memory=$2 ;;
        --concurrency) concurrency=$2 ;; --train-hours) time_hours=$2 ;;
      esac
      shift 2 ;;
    *) fail "Unknown option: $1" ;;
  esac
done
case "$mode" in prepare|benchmark|review|train|evaluate|report) ;; *) fail 'Choose --mode; see --help' ;; esac
[[ "$concurrency" =~ ^[1-7]$ && "$cpus" =~ ^[1-9][0-9]*$ ]] || fail 'Invalid concurrency/CPUs'
[[ "$memory" =~ ^[1-9][0-9]*[MGT]$ ]] || fail 'Invalid memory'
[[ -z "$time_hours" || "$time_hours" =~ ^[1-9][0-9]*$ ]] || fail 'Train hours must be a positive integer'
for value in "$account" "$partition" "$gpu"; do [[ "$value" =~ ^[a-zA-Z0-9_-]+$ ]] || fail 'Invalid resource name'; done
if [[ "$mode" == prepare && -z "$output" ]]; then
  output="$screen_repo/checkpoints/part_ev_screen/context_gate_$(date -u +%Y%m%dT%H%M%SZ)"
fi
[[ "$output" == /* && "$screen_python" == /* && "$screen_data" == /* ]] || fail 'Use absolute campaign, Python and data paths'
if (( dry )); then
  printf 'Mode: %s\nCampaign: %s\n' "$mode" "$output"
  printf 'GPU arrays: --array=0-6%%%s --gres=gpu:%s:1 --cpus-per-task=%s --mem=%s --partition=%s --account=%s\n' "$concurrency" "$gpu" "$cpus" "$memory" "$partition" "$account"
  echo 'Prepare 02:00:00 -> benchmark[7] 02:00:00. Production requires reviewed passing benchmarks.'
  echo 'Train[7] (measured walltime +30%) -> evaluate[7] (measured walltime +30%) -> report 02:00:00.'
  exit 0
fi
[[ "$(uname -m)" == x86_64 ]] || fail 'Run on sporcsubmit, not Tigris'
for command in git tar sha256sum find sort xargs realpath flock sbatch squeue scontrol; do
  command -v "$command" >/dev/null || fail "Missing command: $command"
done
for option_name in ${!SBATCH_@}; do unset "$option_name"; done
if [[ "$mode" == prepare && ! -f "$output/launch.env" ]]; then
  (( resume == 0 )) || fail '--resume requires an existing prepared source snapshot'
  [[ ! -e "$output" ]] || fail 'New campaign path must not already exist'
  [[ -x "$screen_python" && -d "$screen_data/Pythia/train_100M" && -d "$screen_data/Pythia/val_5M" ]] || fail 'Missing environment or train/validation directories'
  for name in part_ev_screen/model.py scripts/run_part_ev_screen.py sbatch/run_part_ev_screen_sporc.sh; do
    git -C "$screen_repo" cat-file -e "HEAD:$name" || fail 'Commit and push the screen implementation first'
  done
  [[ -z "$(git -C "$screen_repo" status --porcelain -- part_ev_screen part_ev_fullscale scripts/run_part_ev_screen.py sbatch/*part_ev_screen* particle_transformer/data/JetClass/JetClass_full.yaml)" ]] || fail 'Commit relevant source changes before preparing'
  screen_parent="$(dirname -- "$output")"
  [[ -d "$screen_parent" ]] || mkdir -p -- "$screen_parent"
  mkdir -- "$output" || fail 'Campaign was created concurrently; refusing overwrite'
  output="$(realpath -e -- "$output")"
  mkdir -- "$output/source" "$output/logs"
  git -C "$screen_repo" rev-parse HEAD > "$output/source_commit.txt"
  git -C "$screen_repo" archive HEAD | tar -xf - -C "$output/source"
  (cd -- "$output/source"; find . -type f -print0 | sort -z | xargs -0 sha256sum) > "$output/source.sha256"
  printf 'SCREEN_PYTHON=%q\nSCREEN_ACCOUNT=%q\nSCREEN_PARTITION=%q\nSCREEN_GPU=%q\nSCREEN_CPUS=%q\nSCREEN_MEMORY=%q\nSCREEN_DATA_ROOT=%q\n' \
    "$screen_python" "$account" "$partition" "$gpu" "$cpus" "$memory" "$screen_data" > "$output/launch.env"
  "$screen_python" - "$output/launch.json" "$screen_data" "$screen_python" "$account" "$partition" "$gpu" "$cpus" "$memory" <<'PY'
import json, sys
from pathlib import Path
keys = ('data_root', 'python', 'account', 'partition', 'gpu', 'cpus', 'memory')
Path(sys.argv[1]).write_text(json.dumps(dict(zip(keys, sys.argv[2:])), indent=2) + '\n')
PY
fi
output="$(realpath -e -- "$output")"
[[ -f "$output/launch.env" ]] || fail 'Prepare this campaign first'
source "$output/launch.env"
for ((i=0; i<${#explicit_resources[@]}; i+=2)); do
  requested="${explicit_resources[i+1]}"
  case "${explicit_resources[i]}" in
    --python) frozen="$SCREEN_PYTHON" ;; --account) frozen="$SCREEN_ACCOUNT" ;;
    --partition) frozen="$SCREEN_PARTITION" ;; --gpu) frozen="$SCREEN_GPU" ;;
    --cpus) frozen="$SCREEN_CPUS" ;; --mem) frozen="$SCREEN_MEMORY" ;;
    --data-root) frozen="$SCREEN_DATA_ROOT" ;;
  esac
  [[ "$requested" == "$frozen" ]] || fail "${explicit_resources[i]} differs from the frozen campaign ($requested != $frozen); use a new output directory"
done
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$output/source"
export PATH="$(dirname -- "$SCREEN_PYTHON"):$PATH"
export LD_LIBRARY_PATH="$(dirname -- "$SCREEN_PYTHON")/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
driver="$output/source/scripts/run_part_ev_screen.py"
if [[ "$mode" == review ]]; then
  exec "$SCREEN_PYTHON" "$driver" --phase review --output "$output"
fi
# Serialize submitters, including partially submitted dependency chains.
exec 9>"$output/.submission.lock"
flock -n 9 || fail 'Another submitter holds this campaign lock'
ledger="$output/submitted_job_ids.tsv"
if [[ -s "$ledger" ]]; then
  live="$(squeue -h -u "$(id -un)" -o '%i' | sed 's/_.*//' | sort -u)"
  while read -r phase job; do
    if grep -qx "$job" <<< "$live"; then fail "Campaign job $job ($phase) is still active. No duplicate submission."; fi
  done < "$ledger"
  if grep -q "^${mode}[[:space:]]" "$ledger" && (( ! resume )); then fail 'Phase already submitted; inspect then use --resume for terminal jobs'; fi
fi
# Verify the current site's partition exists; never silently reroute to debug.
scontrol show partition "$SCREEN_PARTITION" >/dev/null || fail 'Requested partition unavailable; inspect sinfo/my-accounts'
submit() {
  local phase=$1 dependency=$2 hours=$3 result
  local request=(--parsable --job-name="evscr_$phase" --account="$SCREEN_ACCOUNT" --partition="$SCREEN_PARTITION"
    --nodes=1 --ntasks=1 --cpus-per-task="$SCREEN_CPUS" --mem="$SCREEN_MEMORY"
    --time="${hours}:00:00" --output="$output/logs/%x_%A_%a.out" --error="$output/logs/%x_%A_%a.err"
    --export=ALL --signal=B:USR1@180)
  case "$phase" in
    benchmark|train|evaluate) request+=(--array="0-6%$concurrency" --gres="gpu:$SCREEN_GPU:1") ;;
  esac
  [[ -z "$dependency" ]] || request+=(--dependency="afterok:$dependency" --kill-on-invalid-dep=yes)
  result="$(PART_EV_SCREEN_OUTPUT="$output" PART_EV_SCREEN_PHASE="$phase" sbatch "${request[@]}" "$output/source/sbatch/run_part_ev_screen_sporc.sh")"
  result="${result%%;*}"
  [[ "$result" =~ ^[0-9]+$ ]] || fail "Unexpected sbatch output: $result (inspect scheduler before retrying)"
  printf '%s\t%s\n' "$phase" "$result" >> "$ledger"
  printf '%s job: %s\n' "$phase" "$result" >&2
  printf '%s' "$result"
}
case "$mode" in
  prepare)
    prepare_id="$(submit prepare '' 2)"
    submit benchmark "$prepare_id" 2 ;;
  benchmark) submit benchmark '' 2 ;;
  train|evaluate)
    [[ "$mode" != train || "$reviewed" == 1 ]] || fail 'Inspect --mode review, then supply --reviewed'
    recommendation="$("$SCREEN_PYTHON" "$driver" --phase review --output "$output")"
    needed="$("$SCREEN_PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["recommended_train_hours"])' <<< "$recommendation")"
    eval_hours="$("$SCREEN_PYTHON" -c 'import json,sys,math; r=json.load(sys.stdin); print(max(2, math.ceil(1.3 * max(x["validation"]["seconds"] * 5000000/x["validation"]["events"] for x in r["models"])/3600)))' <<< "$recommendation")"
    dependency=
    if [[ "$mode" == train ]]; then
      time_hours="${time_hours:-$needed}"
      (( time_hours >= needed )) || fail "Use at least $needed training hours based on the measured gate"
      dependency="$(submit train '' "$time_hours")"
    fi
    eval_id="$(submit evaluate "$dependency" "$eval_hours")"
    submit report "$eval_id" 2 ;;
  report) submit report '' 2 ;;
esac
printf '\nCampaign: %s\nJob ledger: %s\n' "$output" "$ledger"
