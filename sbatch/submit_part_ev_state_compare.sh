#!/usr/bin/env bash
# Fresh full-schedule campaign; historical scripts and campaigns remain unchanged.
set -euo pipefail
screen_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
mode= output= resume=0 dry=0 reviewed=0 concurrency=3 time_hours= evaluate_hours=
screen_python=/home/ryreu/miniconda3/envs/part_ev_screen_sporc/bin/python
screen_data=/home/ryreu/atlas/datasets/JetClass
account=reu-aisocial partition=tier3 gpu=a100 cpus=8 memory=64G
controls=/home/ryreu/atlas/Fresh_check/checkpoints/part_ev_state_fullscale/state_full_20261024T185049Z
explicit_resources=()
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
usage() {
  cat <<'USAGE'
Usage: bash sbatch/submit_part_ev_state_compare.sh --mode MODE [options]
  prepare    Freeze committed source; queue 5h prepare -> three architecture 5h GPU benchmarks.
  benchmark  Retry the benchmark array after a terminal/incomplete attempt.
  review     Print benchmark results and a 30%-margin production time estimate.
  train      Queue three seed-101 trainings -> lock five checkpoints -> five full-5M evaluations -> report.
  evaluate   Retry full validation + report after terminal downstream jobs.
  report     Retry report only.
Options:
  --controls DIR     Original seven-task full campaign; seed-101 ParT and EV are reused.
  --output DIR       New prepare directory or existing campaign for other phases.
  --python PATH      Dedicated x86 CUDA environment Python (prepare only).
  --data-root DIR    Official JetClass parent with Pythia/train_100M and val_5M.
  --partition NAME   Default tier3. Verify it exists and your account can use it.
  --account NAME     Default reu-aisocial.
  --gpu TYPE         Default a100; frozen across all benchmark/train/evaluation jobs.
  --cpus N --mem SIZE  Default 8 / 64G; frozen at prepare.
  --concurrency N    Default 3, at most 3 simultaneous production tasks (benchmarks at most 2).
  --reviewed        Required to train after examining all three benchmarks.
  --train-hours N   Optional walltime >= the 30%-margin measured recommendation.
  --evaluate-hours N  Full-validation walltime override (minimum measured recommendation, at least 5h).
  --resume          Retry terminal submissions; refuses active duplicate pipelines.
  --dry-run         Show requests; create no files or jobs.
This launcher never cancels jobs or resubmits the old campaign.
No independent test-set evaluation, deletion, or environment installation.
USAGE
}
while (( $# )); do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --resume) resume=1; shift ;; --dry-run) dry=1; shift ;; --reviewed) reviewed=1; shift ;;
    --mode|--output|--controls|--python|--data-root|--partition|--account|--gpu|--cpus|--mem|--concurrency|--train-hours|--evaluate-hours)
      (( $# >= 2 )) || fail "Missing $1 value"
      case "$1" in --controls|--python|--data-root|--partition|--account|--gpu|--cpus|--mem) explicit_resources+=("$1" "$2") ;; esac
      case "$1" in
        --controls) controls=$2 ;; --mode) mode=$2 ;; --output) output=$2 ;; --python) screen_python=$2 ;;
        --data-root) screen_data=$2 ;; --partition) partition=$2 ;; --account) account=$2 ;;
        --gpu) gpu=$2 ;; --cpus) cpus=$2 ;; --mem) memory=$2 ;;
        --concurrency) concurrency=$2 ;; --train-hours) time_hours=$2 ;; --evaluate-hours) evaluate_hours=$2 ;;
      esac
      shift 2 ;;
    *) fail "Unknown option: $1" ;;
  esac
done
case "$mode" in prepare|benchmark|review|train|evaluate|report) ;; *) fail 'Choose --mode; see --help' ;; esac
[[ "$concurrency" =~ ^[1-3]$ && "$cpus" =~ ^[1-9][0-9]*$ ]] || fail 'Invalid concurrency/CPUs'
[[ "$memory" =~ ^[1-9][0-9]*[MGT]$ ]] || fail 'Invalid memory'
[[ -z "$time_hours" || "$time_hours" =~ ^[1-9][0-9]*$ ]] || fail 'Train hours must be a positive integer'
[[ -z "$evaluate_hours" || "$evaluate_hours" =~ ^[1-9][0-9]*$ ]] || fail 'Evaluation hours must be a positive integer'
for value in "$account" "$partition" "$gpu"; do [[ "$value" =~ ^[a-zA-Z0-9_-]+$ ]] || fail 'Invalid resource name'; done
if [[ "$mode" == prepare && -z "$output" ]]; then
  output="$screen_repo/checkpoints/part_ev_state_compare/state_compare_$(date -u +%Y%m%dT%H%M%SZ)"
fi
[[ "$output" == /* && "$screen_python" == /* && "$screen_data" == /* ]] || fail 'Use absolute campaign, Python and data paths'
if (( dry )); then
  printf 'Mode: %s\nCampaign: %s\n' "$mode" "$output"
  printf 'GPU arrays: --array=0-2%%%s --gres=gpu:%s:1 --cpus-per-task=%s --mem=%s --partition=%s --account=%s\n' "$concurrency" "$gpu" "$cpus" "$memory" "$partition" "$account"
  echo 'Prepare 05:00:00 -> benchmark array 0-2 (seed 101) 05:00:00. Production requires reviewed passing benchmarks.'
  echo 'Train[3] + kept individual control tasks -> lock -> evaluate[5] -> report. CPU jobs <=5h.'
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
  for name in scripts/part_ev_state_compare_dependencies.py part_ev_state_compare/model.py part_ev_state_compare/engine.py part_ev_state_compare/protocol.py part_ev_state_compare/reporting.py scripts/run_part_ev_state_compare.py sbatch/run_part_ev_state_compare.sh sbatch/submit_part_ev_state_compare.sh; do
    git -C "$screen_repo" cat-file -e "HEAD:$name" || fail 'Commit and push the full-schedule implementation first'
  done
  [[ -z "$(git -C "$screen_repo" status --porcelain -- part_ev_state_compare part_ev_state_capacity part_layerwise_screen part_ev_state_fullscale part_ev_state_speed part_ev_message_study part_ev_screen part_ev_fullscale scripts/part_ev_state_compare_dependencies.py scripts/run_part_ev_state_fullscale.py scripts/run_part_ev_state_compare.py scripts/run_part_ev_message_study.py scripts/run_part_ev_screen.py scripts/run_part_ev_screen_after_recovery.py scripts/check_part_ev_screen_amp_recovery.py sbatch/*part_ev_state_compare* sbatch/*part_ev_state_fullscale* particle_transformer/data/JetClass/JetClass_full.yaml)" ]] || fail 'Commit relevant source changes before preparing'
  controls="$(realpath -e -- "$controls")"
  screen_parent="$(dirname -- "$output")"
  [[ -d "$screen_parent" ]] || mkdir -p -- "$screen_parent"
  mkdir -- "$output" || fail 'Campaign was created concurrently; refusing overwrite'
  output="$(realpath -e -- "$output")"
  mkdir -- "$output/source" "$output/logs"
  git -C "$screen_repo" rev-parse HEAD > "$output/source_commit.txt"
  git -C "$screen_repo" archive HEAD | tar -xf - -C "$output/source"
  (cd -- "$output/source"; find . -type f -print0 | sort -z | xargs -0 sha256sum) > "$output/source.sha256"
  printf 'STATE_COMPARE_PYTHON=%q\nSTATE_COMPARE_ACCOUNT=%q\nSTATE_COMPARE_PARTITION=%q\nSTATE_COMPARE_GPU=%q\nSTATE_COMPARE_CPUS=%q\nSTATE_COMPARE_MEMORY=%q\nSTATE_COMPARE_DATA_ROOT=%q\nSTATE_COMPARE_CONTROLS=%q\n' \
    "$screen_python" "$account" "$partition" "$gpu" "$cpus" "$memory" "$screen_data" "$controls" > "$output/launch.env"
  "$screen_python" - "$output/launch.json" "$screen_data" "$screen_python" "$account" "$partition" "$gpu" "$cpus" "$memory" "$controls" <<'PY'
import json, sys
from pathlib import Path
keys = ('data_root', 'python', 'account', 'partition', 'gpu', 'cpus', 'memory', 'controls')
Path(sys.argv[1]).write_text(json.dumps(dict(zip(keys, sys.argv[2:])), indent=2) + '\n')
PY
  (cd -- "$output"; sha256sum launch.env launch.json) > "$output/launch.sha256"
fi
output="$(realpath -e -- "$output")"
[[ -f "$output/launch.env" ]] || fail 'Prepare this campaign first'
(cd -- "$output"; sha256sum --quiet -c launch.sha256)
source "$output/launch.env"
for ((i=0; i<${#explicit_resources[@]}; i+=2)); do
  requested="${explicit_resources[i+1]}"
  case "${explicit_resources[i]}" in
    --controls) frozen="$STATE_COMPARE_CONTROLS" ;; --python) frozen="$STATE_COMPARE_PYTHON" ;; --account) frozen="$STATE_COMPARE_ACCOUNT" ;;
    --partition) frozen="$STATE_COMPARE_PARTITION" ;; --gpu) frozen="$STATE_COMPARE_GPU" ;;
    --cpus) frozen="$STATE_COMPARE_CPUS" ;; --mem) frozen="$STATE_COMPARE_MEMORY" ;;
    --data-root) frozen="$STATE_COMPARE_DATA_ROOT" ;;
  esac
  [[ "$requested" == "$frozen" ]] || fail "${explicit_resources[i]} differs from the frozen campaign ($requested != $frozen); use a new output directory"
done
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$output/source"
export PATH="$(dirname -- "$STATE_COMPARE_PYTHON"):$PATH"
export LD_LIBRARY_PATH="$(dirname -- "$STATE_COMPARE_PYTHON")/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
driver="$output/source/scripts/run_part_ev_state_compare.py"
(cd -- "$output/source"; sha256sum --quiet -c "$output/source.sha256")
if [[ "$mode" == review ]]; then
  exec "$STATE_COMPARE_PYTHON" "$driver" --phase review --output "$output"
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
partition_info="$(scontrol show partition "$STATE_COMPARE_PARTITION" -o)" || fail 'Requested partition unavailable; inspect sinfo/my-accounts'
check_walltime() {
  local hours=$1 limit day=0 hh mm ss
  limit="$(sed -n 's/.*MaxTime=\([^ ]*\).*/\1/p' <<< "$partition_info")"
  [[ -n "$limit" ]] || fail 'Cannot read partition MaxTime; inspect scontrol show partition'
  [[ "$limit" != UNLIMITED && "$limit" != INFINITE ]] || return 0
  if [[ "$limit" == *-* ]]; then day="${limit%%-*}"; limit="${limit#*-}"; fi
  IFS=: read -r hh mm ss <<< "$limit"
  [[ "$day" =~ ^[0-9]+$ && "$hh" =~ ^[0-9]+$ && "$mm" =~ ^[0-9]+$ && "$ss" =~ ^[0-9]+$ ]] || fail 'Unrecognized partition time limit'
  (( hours * 3600 <= 10#$day * 86400 + 10#$hh * 3600 + 10#$mm * 60 + 10#$ss )) ||
    fail "Requested ${hours}h exceeds partition MaxTime. No shorter training budget substituted."
}
submit() {
  local phase=$1 dependency=$2 hours=$3 result
  check_walltime "$hours"
  local request=(--parsable --job-name="evcmp_$phase" --account="$STATE_COMPARE_ACCOUNT" --partition="$STATE_COMPARE_PARTITION"
    --nodes=1 --ntasks=1 --cpus-per-task="$STATE_COMPARE_CPUS" --mem="$STATE_COMPARE_MEMORY"
    --time="${hours}:00:00" --output="$output/logs/%x_%A_%a.out" --error="$output/logs/%x_%A_%a.err"
    --export=ALL --signal=B:USR1@180)
  case "$phase" in
    benchmark) request+=(--array="0-2%$((concurrency < 2 ? concurrency : 2))" --gres="gpu:$STATE_COMPARE_GPU:1") ;;
    train) request+=(--array="0-2%$concurrency" --gres="gpu:$STATE_COMPARE_GPU:1") ;;
    evaluate) request+=(--array="0-4%$concurrency" --gres="gpu:$STATE_COMPARE_GPU:1") ;;
  esac
  [[ -z "$dependency" ]] || request+=(--dependency="afterok:$dependency" --kill-on-invalid-dep=yes)
  result="$(PART_EV_STATE_COMPARE_OUTPUT="$output" PART_EV_STATE_COMPARE_PHASE="$phase" sbatch "${request[@]}" "$output/source/sbatch/run_part_ev_state_compare.sh")"
  result="${result%%;*}"
  [[ "$result" =~ ^[0-9]+$ ]] || fail "Unexpected sbatch output: $result (inspect scheduler before retrying)"
  printf '%s\t%s\n' "$phase" "$result" >> "$ledger"
  printf '%s job: %s\n' "$phase" "$result" >&2
  printf '%s' "$result"
}
case "$mode" in
  prepare)
    prepare_id="$(submit prepare '' 5)"
    submit benchmark "$prepare_id" 5 ;;
  benchmark) submit benchmark '' 5 ;;
  train|evaluate)
    [[ "$mode" != train || "$reviewed" == 1 ]] || fail 'Inspect --mode review, then supply --reviewed'
    review_args=(--phase review --output "$output")
    [[ "$mode" != train ]] || review_args+=(--approve --reviewed)
    recommendation="$("$STATE_COMPARE_PYTHON" "$driver" "${review_args[@]}")"
    needed="$("$STATE_COMPARE_PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["recommended_train_hours"])' <<< "$recommendation")"
    eval_hours="$("$STATE_COMPARE_PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["recommended_evaluate_hours"])' <<< "$recommendation")"
    evaluate_hours="${evaluate_hours:-$eval_hours}"
    (( evaluate_hours >= eval_hours )) || fail "Use at least $eval_hours evaluation hours"
    check_walltime "$evaluate_hours"
    kept="$("$STATE_COMPARE_PYTHON" "$output/source/scripts/part_ev_state_compare_dependencies.py" --output "$output")"
    dependency="$kept"
    if [[ "$mode" == train ]]; then
      time_hours="${time_hours:-$needed}"
      (( time_hours >= needed )) || fail "Use at least $needed training hours based on the measured gate"
      check_walltime "$time_hours"
      train_id="$(submit train '' "$time_hours")"
      dependency="${train_id}${kept:+:$kept}"
    fi
    lock_id="$(submit lock "$dependency" 5)"
    eval_id="$(submit evaluate "$lock_id" "$evaluate_hours")"
    submit report "$eval_id" 5 ;;
  report) submit report '' 5 ;;
esac
printf '\nCampaign: %s\nJob ledger: %s\n' "$output" "$ledger"
