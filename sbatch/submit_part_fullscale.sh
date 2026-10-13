#!/usr/bin/env bash
# Freeze, prepare, benchmark, and submit the six original-recipe ParT comparisons.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mode= output= dependency= test_data= resume=0 dry_run=0
data_root=/home/ryreu/atlas/datasets/JetClass
python_path=/home/ryreu/miniforge3-aarch64/envs/part_ev_fullscale/bin/python
account=reu-aisocial partition=tigris gpus=1 concurrency=2 cpus=8 memory=128G
time_limit=3-00:00:00 benchmark_steps=200 benchmark_warmup=20
usage() {
  cat <<'USAGE'
Usage: bash sbatch/submit_part_fullscale.sh --mode MODE [options]

Modes (none submits another training phase implicitly):
  prepare    Freeze committed source; queue a CPU dataset/environment preflight.
  benchmark  Compare both architectures on --gpus 1, 2, or 4 (one GPU per node).
  train      Queue six trainings: ParT/ParT-EV, three matched seeds each.
  test       Explicitly queue six 1-GPU test evaluations, then their report.

Options:
  --output DIR         New campaign directory for prepare; existing for other modes.
  --data-root DIR      JetClass parent containing Pythia/train_100M and Pythia/val_5M.
  --test-data DIR      Required for test; staged directory of 200 test ROOT files.
  --gpus 1|2|4         Training/benchmark ranks; global batch remains 512.
  --concurrency N      Simultaneous runs in the six-task train/test array (default 2).
  --cpus N            CPUs per rank/node (default 8).
  --mem SIZE          Memory per node (default 128G; reduce after measured benchmark).
  --time LIMIT        Train/test wall time (default 3-00:00:00); benchmark is 1 hour.
  --python PATH       Absolute Tigris environment Python; never installs packages.
  --account NAME      Slurm account (default reu-aisocial).
  --partition NAME    Slurm partition (default tigris).
  --dependency IDS    afterok job ID, or colon-separated IDs (same Slurm cluster).
  --resume            Resubmit a terminal phase; active jobs always block duplicates.
  --steps N           Benchmark measured steps (default 200).
  --warmup N          Benchmark warmup steps (default 20).
  --dry-run           Print resource requests only; no files/jobs are created.

Run prepare only after the train/validation upload has finished. Benchmark before
choosing the production GPU count. Source is pinned once; unrelated repository
commits cannot change a campaign. All six trainings must use the same GPU count.
Test data need not be present for prepare, benchmark, or train. To queue testing
ahead of training completion, explicitly use --mode test --dependency TRAIN_JOB
and --test-data DIR. Training checkpoints and models are never deleted here.

After prepare completes, benchmark the choices separately with the same output:
  ... --mode benchmark --output DIR --gpus 1
  ... --mode benchmark --output DIR --gpus 2
  ... --mode benchmark --output DIR --gpus 4
These are measured comparisons, not permission to change the global batch/LR.
Train jobs receive a checkpoint warning 180 seconds before their time limit.
If a run reaches its time limit, inspect it and resubmit with --mode train
--output DIR --gpus N --resume; completed runs are retained/skipped by the driver.
USAGE
}
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
while (( $# )); do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --resume) resume=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    --mode|--output|--data-root|--test-data|--gpus|--concurrency|--cpus|--mem|--time|--python|--account|--partition|--dependency|--steps|--warmup)
      (( $# >= 2 )) || fail "Missing value for $1"
      case "$1" in
        --mode) mode=$2 ;; --output) output=$2 ;; --data-root) data_root=$2 ;;
        --test-data) test_data=$2 ;; --gpus) gpus=$2 ;; --concurrency) concurrency=$2 ;;
        --cpus) cpus=$2 ;; --mem) memory=$2 ;; --time) time_limit=$2 ;;
        --python) python_path=$2 ;; --account) account=$2 ;; --partition) partition=$2 ;;
        --dependency) dependency=$2 ;; --steps) benchmark_steps=$2 ;; --warmup) benchmark_warmup=$2 ;;
      esac
      shift 2 ;;
    *) fail "Unknown option: $1" ;;
  esac
done
case "$mode" in prepare|benchmark|train|test) ;; *) fail 'Specify --mode prepare|benchmark|train|test' ;; esac
case "$gpus" in 1|2|4) ;; *) fail '--gpus must be 1, 2, or 4' ;; esac
for value in "$concurrency" "$cpus" "$benchmark_steps" "$benchmark_warmup"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || fail 'Concurrency, CPUs, steps, and warmup must be positive integers'
done
(( concurrency <= 6 )) || fail 'There are only six train/test tasks'
[[ -z "$dependency" || "$dependency" =~ ^[0-9]+(:[0-9]+)*$ ]] || fail 'Dependency must contain only numeric job IDs separated by colons'
[[ "$python_path" == /* ]] || fail '--python must be an absolute path'
[[ "$account" =~ ^[a-zA-Z0-9_-]+$ && "$partition" =~ ^[a-zA-Z0-9_-]+$ ]] || fail 'Invalid account or partition'
[[ "$memory" =~ ^[1-9][0-9]*[KMGTP]?$ ]] || fail 'Invalid --mem value'
[[ "$time_limit" =~ ^([0-9]+-)?[0-9]+:[0-9]{2}:[0-9]{2}$ ]] || fail 'Use a Slurm time such as 3-00:00:00'
if [[ "$mode" == prepare && -z "$output" ]]; then
  output="$repo_root/checkpoints/part_fullscale/part_ev_$(date -u +%Y%m%dT%H%M%SZ)"
fi
[[ -n "$output" ]] || fail '--output is required for an existing campaign'
[[ "$mode" != test || -n "$test_data" ]] || fail 'Test submission requires explicit --test-data'
[[ "$mode" != test || "$gpus" == 1 ]] || fail 'Each test task uses one GPU; omit --gpus or use --gpus 1'
if (( dry_run )); then
  printf 'Dry run: mode=%s output=%s account=%s partition=%s\n' "$mode" "$output" "$account" "$partition"
  if [[ "$mode" == prepare ]]; then
    printf 'CPU preflight: --nodes=1 --cpus-per-task=4 --mem=16G --time=02:00:00\n'
  else
    printf 'GPU allocation per run: --nodes=%s --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=%s --mem=%s\n' "$gpus" "$cpus" "$memory"
    [[ "$mode" == benchmark ]] || printf 'Array: --array=0-5%%%s --time=%s\n' "$concurrency" "$time_limit"
    [[ "$mode" != test ]] || printf 'Test ROOT directory: %s; report depends on all six test tasks.\n' "$test_data"
  fi
  [[ -z "$dependency" ]] || printf 'Dependency: afterok:%s\n' "$dependency"
  printf 'Nothing created, submitted, cancelled, or deleted.\n'
  exit 0
fi
for command in git tar sha256sum find sort xargs realpath flock sbatch squeue; do
  command -v "$command" >/dev/null || fail "Missing command: $command"
done
[[ -x "$python_path" ]] || fail "Python is not executable: $python_path"
# Inherited SBATCH_* options must not silently alter this graph or allocation.
for option_name in ${!SBATCH_@}; do unset "$option_name"; done
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
if [[ "$mode" == prepare && ! -e "$output" ]]; then
  (( resume == 0 )) || fail '--resume requires an existing campaign'
  relevant=(part_ev_fullscale scripts/run_part_fullscale.py scripts/setup_part_fullscale_env.sh sbatch/run_part_fullscale.sh sbatch/submit_part_fullscale.sh
            requirements-part-fullscale.txt particle_transformer)
  [[ -z "$(git -C "$repo_root" status --porcelain -- "${relevant[@]}")" ]] || fail 'Commit/push the relevant setup code first; no dirty source will be frozen'
  for file in scripts/run_part_fullscale.py sbatch/run_part_fullscale.sh requirements-part-fullscale.txt; do
    git -C "$repo_root" cat-file -e "HEAD:$file" || fail "Required committed source missing: $file"
  done
  data_root="$(realpath -e -- "$data_root")"
  campaign_parent="$(dirname -- "$output")"
  [[ -d "$campaign_parent" ]] || mkdir -p -- "$campaign_parent"
  mkdir -- "$output" || fail 'Campaign already exists; refusing overwrite'
  output="$(realpath -e -- "$output")"
  mkdir -- "$output/source" "$output/logs"
  git -C "$repo_root" archive HEAD | tar -xf - -C "$output/source"
  git -C "$repo_root" rev-parse HEAD > "$output/source_commit.txt"
  (
    cd -- "$output/source"
    find . -type f -print0 | sort -z | xargs -0 sha256sum
  ) > "$output/source.sha256"
  {
    printf 'PART_FULLSCALE_PYTHON=%q\n' "$python_path"
    printf 'PART_FULLSCALE_DATA_ROOT=%q\n' "$data_root"
  } > "$output/launch.env"
elif [[ "$mode" == prepare && "$resume" == 0 ]]; then
  fail 'Campaign already exists; use --resume only after inspecting its previous prepare job'
else
  output="$(realpath -e -- "$output")"
fi
[[ -f "$output/launch.env" && -f "$output/source.sha256" ]] || fail 'Missing frozen campaign source/launch configuration'
# shellcheck disable=SC1090
source "$output/launch.env"
[[ -x "$PART_FULLSCALE_PYTHON" ]] || fail 'Pinned Python environment no longer exists'
[[ "$mode" == prepare || -f "$output/campaign.json" || -n "$dependency" ]] || fail 'Prepare must finish first, or supply its --dependency job ID'
if [[ "$mode" == test ]]; then
  test_data="$(realpath -e -- "$test_data")"
  [[ -d "$test_data" ]] || fail 'Test data path is not a directory'
  [[ -n "$(find "$test_data" -maxdepth 1 -type f -name '*.root' -print -quit)" ]] || fail 'No staged ROOT files in --test-data; refusing test submission'
fi
exec 9>"$output/submission.lock"
flock -n 9 || fail 'Another submitter owns this campaign'
phase_key="$mode"
[[ "$mode" != benchmark ]] || phase_key="benchmark${gpus}gpu"
if [[ -f "$output/submitted_job_ids.tsv" ]]; then
  live="$(squeue --noheader --user "$(id -un)" --format='%i')" || fail 'Cannot inspect live jobs; refusing duplicate submission'
  previous=0
  while IFS=$'\t' read -r old_phase old_id _; do
    [[ "$old_phase" == "$phase_key" || ( "$mode" == test && "$old_phase" == report ) ]] || continue
    previous=1
    while read -r active; do
      [[ "${active%%_*}" != "$old_id" ]] || fail "Previous $old_phase job $old_id is still live; nothing submitted"
    done <<< "$live"
  done < "$output/submitted_job_ids.tsv"
  [[ "$previous" == 0 || "$resume" == 1 ]] || fail 'This phase has a prior submission; inspect it, then explicitly use --resume'
fi
export PART_FULLSCALE_OUTPUT="$output" PART_FULLSCALE_TEST_DATA="$test_data"
export PART_FULLSCALE_STEPS="$benchmark_steps" PART_FULLSCALE_WARMUP="$benchmark_warmup"
worker="$output/source/sbatch/run_part_fullscale.sh"
submit() {
  local phase=$1 depends=$2 key=$3 response job
  shift 3
  local deps=()
  [[ -z "$depends" ]] || deps=("--dependency=afterok:$depends" --kill-on-invalid-dep=yes)
  export PART_FULLSCALE_PHASE="$phase"
  response="$(sbatch --parsable "--account=$account" "--partition=$partition" \
    "--job-name=parte_${phase}" "--output=$output/logs/%x_%A_%a.out" \
    "--error=$output/logs/%x_%A_%a.err" --export=ALL "${deps[@]}" "$@" "$worker")"
  job="${response%%;*}"
  [[ "$job" =~ ^[0-9]+$ ]] || fail "Invalid sbatch response: $response. Inspect the queue before retrying"
  printf '%s\t%s\t%s\n' "$key" "$job" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$output/submitted_job_ids.tsv"
  printf '%s\n' "$job"
}
gpu_resources=("--nodes=$gpus" --ntasks-per-node=1 --gpus-per-node=1 "--cpus-per-task=$cpus" "--mem=$memory")
case "$mode" in
  prepare) job="$(submit prepare "$dependency" "$phase_key" --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --time=02:00:00)" ;;
  benchmark) job="$(submit benchmark "$dependency" "$phase_key" "${gpu_resources[@]}" --time=01:00:00)" ;;
  train) job="$(submit train "$dependency" "$phase_key" "${gpu_resources[@]}" "--array=0-5%$concurrency" "--time=$time_limit" --signal=USR1@180)" ;;
  test)
    job="$(submit test "$dependency" "$phase_key" "${gpu_resources[@]}" "--array=0-5%$concurrency" "--time=$time_limit")"
    report="$(submit report "$job" report --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --time=01:00:00)"
    printf 'Report job: %s\n' "$report" ;;
esac
printf 'Campaign: %s\n%s job: %s\nQueue: squeue -j %s\n' "$output" "$mode" "$job" "$job"
