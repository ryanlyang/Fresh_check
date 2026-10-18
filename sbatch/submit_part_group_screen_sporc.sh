#!/usr/bin/env bash
# Independent grouped screen. Donor controls are read-only artifact references.
set -euo pipefail
group_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
mode= output= resume=0 dry=0 reviewed=0 concurrency=2 train_hours= include_ev=0
group_python=/home/ryreu/miniconda3/envs/part_ev_screen_sporc/bin/python
donor=/home/ryreu/atlas/Fresh_check/checkpoints/part_ev_screen/context_gate_ampfix_20261016T215608Z
account=reu-aisocial partition=tier3 gpu=a100 cpus=8 memory=64G
explicit=()
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
usage() {
  cat <<'USAGE'
Usage: bash sbatch/submit_part_group_screen_sporc.sh --mode MODE [options]
  prepare          Freeze NEW source -> 2h prepare -> 2h grouped GPU benchmarks.
  benchmark        Retry terminal/incomplete benchmark submissions.
  review           Read-only benchmark review and walltime recommendation.
  train            Reviewed new-model train -> lock -> full-validation -> report.
  evaluate         Retry new-model lock -> full-validation -> report.
  refresh-controls Queue a 2h CPU donor-artifact verification/status refresh.
  report           Retry 2h CPU joined report; exit incomplete if controls pending.
  status           Read-only progress (no Slurm submission).
Options:
  --output DIR --donor DIR --python PATH
  --include-ev-control   Also reuse EV_RANGER (prepare only; no extra training).
  --partition NAME --account NAME --gpu TYPE --cpus N --mem SIZE
  --concurrency N       Default 2, maximum 4 new model tasks.
  --reviewed            Required for production admission.
  --train-hours N       At least the measured 30%-margin recommendation.
  --resume              Retry terminal stages; refuses active duplicate jobs.
  --dry-run             Show requests only; no files, jobs or environment changes.
The initial registry has four new models. No baseline training or inference,
test-set evaluation, environment installation, deletion or cancellation occurs.
USAGE
}
while (( $# )); do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --resume) resume=1; shift ;; --dry-run) dry=1; shift ;; --reviewed) reviewed=1; shift ;;
    --include-ev-control) include_ev=1; explicit+=("$1" 1); shift ;;
    --mode|--output|--donor|--python|--partition|--account|--gpu|--cpus|--mem|--concurrency|--train-hours)
      (( $# >= 2 )) || fail "Missing $1 value"
      case "$1" in --donor|--python|--partition|--account|--gpu|--cpus|--mem) explicit+=("$1" "$2") ;; esac
      case "$1" in
        --mode) mode=$2 ;; --output) output=$2 ;; --donor) donor=$2 ;; --python) group_python=$2 ;;
        --partition) partition=$2 ;; --account) account=$2 ;; --gpu) gpu=$2 ;;
        --cpus) cpus=$2 ;; --mem) memory=$2 ;; --concurrency) concurrency=$2 ;; --train-hours) train_hours=$2 ;;
      esac
      shift 2 ;;
    *) fail "Unknown option: $1" ;;
  esac
done
case "$mode" in prepare|benchmark|review|train|evaluate|refresh-controls|report|status) ;; *) fail 'Choose --mode; see --help' ;; esac
[[ "$concurrency" =~ ^[1-4]$ && "$cpus" =~ ^[1-9][0-9]*$ ]] || fail 'Invalid concurrency/CPUs'
[[ "$memory" =~ ^[1-9][0-9]*[MGT]$ ]] || fail 'Invalid memory'
[[ -z "$train_hours" || "$train_hours" =~ ^[1-9][0-9]*$ ]] || fail 'Invalid train hours'
for value in "$partition" "$account" "$gpu"; do [[ "$value" =~ ^[a-zA-Z0-9_-]+$ ]] || fail 'Invalid resource name'; done
if [[ "$mode" == prepare && -z "$output" ]]; then
  output="$group_repo/checkpoints/part_group_screen/groups_$(date -u +%Y%m%dT%H%M%SZ)"
fi
[[ "$output" == /* && "$donor" == /* && "$group_python" == /* ]] || fail 'Use absolute paths'
if (( dry )); then
  printf 'Mode: %s\nCampaign: %s\nDonor (read-only): %s\n' "$mode" "$output" "$donor"
  printf 'New model arrays: --array=0-3%%%s --gres=gpu:%s:1 --cpus-per-task=%s --mem=%s --partition=%s --account=%s\n' "$concurrency" "$gpu" "$cpus" "$memory" "$partition" "$account"
  echo 'Prepare 02:00:00 -> benchmark[4] 02:00:00; then explicit review.'
  echo 'New train[4] -> lock -> new evaluation[4] -> report. No donor tasks.'
  echo 'Production durations use measured throughput +30%; CPU lock/report 02:00:00.'
  exit 0
fi
[[ "$(uname -m)" == x86_64 ]] || fail 'Run on sporcsubmit with donor-matched x86 environment'
for command in git tar sha256sum find sort xargs realpath flock sbatch squeue scontrol; do
  command -v "$command" >/dev/null || fail "Missing command: $command"
done
for option_name in ${!SBATCH_@}; do unset "$option_name"; done
if [[ "$mode" == prepare && ! -f "$output/launch.env" ]]; then
  (( resume == 0 )) || fail '--resume requires a frozen campaign'
  [[ ! -e "$output" && ! -L "$output" ]] || fail 'Campaign already exists; refusing overwrite'
  [[ -x "$group_python" && -f "$donor/campaign.json" ]] || fail 'Missing Python or donor campaign'
  donor="$(realpath -e -- "$donor")"
  candidate="$(realpath -m -- "$output")"
  case "$candidate/" in "$donor/"*) fail 'New campaign must not be inside donor' ;; esac
  case "$donor/" in "$candidate/"*) fail 'New campaign must not contain donor' ;; esac
  for name in part_group_screen/model.py part_screen_common/controls.py scripts/run_part_group_screen.py sbatch/run_part_group_screen_sporc.sh; do
    git -C "$group_repo" cat-file -e "HEAD:$name" || fail 'Commit/push implementation before preparing'
  done
  [[ -z "$(git -C "$group_repo" status --porcelain -- part_group_screen part_screen_common part_ev_screen part_ev_fullscale scripts/run_part_group_screen.py scripts/run_part_ev_screen.py scripts/export_part_screen_controls.py sbatch/*part_group_screen* particle_transformer/data/JetClass/JetClass_full.yaml)" ]] || fail 'Commit relevant source changes before preparing'
  group_parent="$(dirname -- "$output")"
  [[ -d "$group_parent" ]] || mkdir -p -- "$group_parent"
  mkdir -- "$output" || fail 'Concurrent prepare; refusing overwrite'
  output="$(realpath -e -- "$output")"
  mkdir -- "$output/source" "$output/logs"
  git -C "$group_repo" rev-parse HEAD > "$output/source_commit.txt"
  git -C "$group_repo" archive HEAD | tar -xf - -C "$output/source"
  (cd -- "$output/source"; find . -type f -print0 | sort -z | xargs -0 sha256sum) > "$output/source.sha256"
  printf 'GROUP_PYTHON=%q\nGROUP_DONOR=%q\nGROUP_INCLUDE_EV=%q\nGROUP_ACCOUNT=%q\nGROUP_PARTITION=%q\nGROUP_GPU=%q\nGROUP_CPUS=%q\nGROUP_MEMORY=%q\n' \
    "$group_python" "$donor" "$include_ev" "$account" "$partition" "$gpu" "$cpus" "$memory" > "$output/launch.env"
  "$group_python" - "$output/launch.json" "$donor" "$group_python" "$account" "$partition" "$gpu" "$cpus" "$memory" "$include_ev" <<'PY'
import json, sys
from pathlib import Path
keys = ('donor', 'python', 'account', 'partition', 'gpu', 'cpus', 'memory')
row = dict(zip(keys, sys.argv[2:9]))
row['controls'] = ['PART_RANGER'] + (['EV_RANGER'] if sys.argv[9] == '1' else [])
Path(sys.argv[1]).write_text(json.dumps(row, indent=2) + '\n')
PY
fi
output="$(realpath -e -- "$output")"
[[ -f "$output/launch.env" ]] || fail 'Prepare the campaign first'
source "$output/launch.env"
for ((i=0; i<${#explicit[@]}; i+=2)); do
  case "${explicit[i]}" in
    --python) frozen=$GROUP_PYTHON ;; --donor) frozen=$GROUP_DONOR ;;
    --include-ev-control) frozen=$GROUP_INCLUDE_EV ;; --account) frozen=$GROUP_ACCOUNT ;;
    --partition) frozen=$GROUP_PARTITION ;; --gpu) frozen=$GROUP_GPU ;;
    --cpus) frozen=$GROUP_CPUS ;; --mem) frozen=$GROUP_MEMORY ;;
  esac
  [[ "${explicit[i+1]}" == "$frozen" ]] || fail "${explicit[i]} differs from frozen campaign"
done
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$output/source"
export PATH="$(dirname -- "$GROUP_PYTHON"):$PATH"
export LD_LIBRARY_PATH="$(dirname -- "$GROUP_PYTHON")/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
(cd -- "$output/source"; sha256sum --quiet -c "$output/source.sha256")
driver="$output/source/scripts/run_part_group_screen.py"
if [[ "$mode" == review || "$mode" == status ]]; then
  exec "$GROUP_PYTHON" "$driver" --phase "$mode" --output "$output"
fi
exec 9>"$output/.submission.lock"
flock -n 9 || fail 'Concurrent submitter'
ledger="$output/submitted_job_ids.tsv"
if [[ -s "$ledger" ]]; then
  live="$(squeue -h -u "$(id -un)" -o '%i' | sed 's/_.*//' | sort -u)"
  while read -r phase job; do
    if grep -qx "$job" <<< "$live"; then fail "Campaign job $job ($phase) is still active"; fi
  done < "$ledger"
  if grep -q "^${mode}[[:space:]]" "$ledger" && (( ! resume )); then fail 'Phase already submitted; inspect then --resume terminal jobs'; fi
fi
scontrol show partition "$GROUP_PARTITION" >/dev/null || fail 'Requested partition unavailable'
# Registry, not reused-control count, determines array size.
task_count="$("$GROUP_PYTHON" -c 'from part_group_screen.config import TASKS; print(len(TASKS))')"
[[ "$task_count" =~ ^[1-9][0-9]*$ ]] || fail 'Invalid frozen task count'
(( concurrency <= task_count )) || fail 'Concurrency exceeds new-model task count'
submit() {
  local phase=$1 dependency=$2 hours=$3 result
  local request=(--parsable --job-name="grp_$phase" --account="$GROUP_ACCOUNT" --partition="$GROUP_PARTITION"
    --nodes=1 --ntasks=1 --cpus-per-task="$GROUP_CPUS" --mem="$GROUP_MEMORY"
    --time="${hours}:00:00" --output="$output/logs/%x_%A_%a.out" --error="$output/logs/%x_%A_%a.err"
    --export=ALL --signal=B:USR1@180)
  case "$phase" in benchmark|train|evaluate) request+=(--array="0-$((task_count-1))%$concurrency" --gres="gpu:$GROUP_GPU:1") ;; esac
  [[ -z "$dependency" ]] || request+=(--dependency="afterok:$dependency" --kill-on-invalid-dep=yes)
  result="$(PART_GROUP_OUTPUT="$output" PART_GROUP_PHASE="$phase" sbatch "${request[@]}" "$output/source/sbatch/run_part_group_screen_sporc.sh")"
  result="${result%%;*}"
  [[ "$result" =~ ^[0-9]+$ ]] || fail "Unexpected Slurm output: $result; inspect scheduler before retrying"
  printf '%s\t%s\n' "$phase" "$result" >> "$ledger"
  printf '%s job: %s\n' "$phase" "$result" >&2
  printf '%s' "$result"
}
case "$mode" in
  prepare) prepare_id="$(submit prepare '' 2)"; submit benchmark "$prepare_id" 2 ;;
  benchmark|refresh-controls|report) submit "$mode" '' 2 ;;
  train|evaluate)
    [[ "$mode" != train || "$reviewed" == 1 ]] || fail 'Review probes, then supply --reviewed'
    if [[ "$mode" == train ]]; then
      recommendation="$("$GROUP_PYTHON" "$driver" --phase approve --output "$output" --reviewed)"
    else
      recommendation="$("$GROUP_PYTHON" "$driver" --phase review --output "$output")"
    fi
    needed="$("$GROUP_PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["recommended_train_hours"])' <<< "$recommendation")"
    eval_hours="$("$GROUP_PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["recommended_evaluate_hours"])' <<< "$recommendation")"
    dependency=
    if [[ "$mode" == train ]]; then
      train_hours="${train_hours:-$needed}"
      (( train_hours >= needed )) || fail "Use at least $needed hours from measured admission"
      dependency="$(submit train '' "$train_hours")"
    fi
    lock_id="$(submit lock "$dependency" 2)"
    eval_id="$(submit evaluate "$lock_id" "$eval_hours")"
    submit report "$eval_id" 2 ;;
esac
printf '\nCampaign: %s\nJob ledger: %s\n' "$output" "$ledger"
