#!/usr/bin/env bash
# Diagnostic only. No old campaign writes, cancellations, or production submissions.
set -euo pipefail
scale_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
campaign= output= mode=submit dry=0
scale_python=/home/ryreu/miniconda3/envs/part_ev_screen_sporc/bin/python
account=reu-aisocial partition=tier3
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
while (( $# )); do
  case "$1" in
    --help|-h)
      echo 'Usage: bash sbatch/submit_part_ev_gpu_scaling.sh --campaign DIR [--output NEW_DIR] [--dry-run]'
      echo '       bash sbatch/submit_part_ev_gpu_scaling.sh --mode review --output DIR'
      echo 'Options: --python PATH --account NAME --partition NAME (default tier3)'
      echo 'Two sequential diagnostic jobs: 1 A100, then 2 A100s on ONE node. Each 04:30:00.'
      echo 'Both use global batch 512, 8 CPUs, 64G host RAM and two data workers total.'
      echo 'Fresh frozen source required. No training submission or automatic approval.'
      exit 0 ;;
    --dry-run) dry=1; shift ;;
    --campaign|--output|--mode|--python|--account|--partition)
      (( $# >= 2 )) || fail "Missing value: $1"
      case "$1" in
        --campaign) campaign=$2 ;; --output) output=$2 ;; --mode) mode=$2 ;;
        --python) scale_python=$2 ;; --account) account=$2 ;; --partition) partition=$2 ;;
      esac
      shift 2 ;;
    *) fail "Unknown option $1" ;;
  esac
done
[[ "$mode" == submit || "$mode" == review ]] || fail 'Mode must be submit or review'
[[ "$account" =~ ^[a-zA-Z0-9_-]+$ && "$partition" =~ ^[a-zA-Z0-9_-]+$ ]] || fail 'Invalid account/partition'
if [[ -z "$output" && "$mode" == submit ]]; then
  output="$scale_repo/checkpoints/part_ev_gpu_scaling/scaling_$(date -u +%Y%m%dT%H%M%SZ)"
fi
[[ "$output" == /* && "$scale_python" == /* ]] || fail 'Use absolute output and Python paths'
[[ "$mode" == review || "$campaign" == /* ]] || fail 'Use an absolute --campaign path'
if (( dry )); then
  printf 'Mode: %s\nDiagnostic: %s\nRead-only campaign: %s\n' "$mode" "$output" "$campaign"
  echo '1 GPU: --nodes=1 --ntasks=1 --gres=gpu:a100:1 --cpus-per-task=8 --mem=64G --time=04:30:00'
  echo '2 GPUs: --nodes=1 --ntasks=1 --gres=gpu:a100:2 --cpus-per-task=8 --mem=64G --time=04:30:00'
  echo 'Second job uses afterany:first, including if the one-GPU job runs out of memory.'
  echo 'No files created or jobs submitted.'
  exit 0
fi
[[ "$(uname -m)" == x86_64 ]] || fail 'Run on sporcsubmit'
for tool in git tar sha256sum find sort xargs realpath sbatch scontrol; do
  command -v "$tool" >/dev/null || fail "Missing $tool"
done
if [[ "$mode" == submit ]]; then
  [[ -x "$scale_python" ]] || fail 'Missing Python environment'
  campaign="$(realpath -e -- "$campaign")"
  [[ -f "$campaign/campaign.json" && ! -e "$output" && ! -L "$output" ]] || fail 'Need prepared parent and NEW diagnostic output'
  [[ -z "$(git -C "$scale_repo" status --porcelain -- part_ev_fullscale part_ev_screen part_ev_message_study part_ev_state_speed part_ev_state_fullscale part_ev_state_capacity part_ev_state_compare part_layerwise_screen part_ev_gpu_scaling scripts/*part_ev* scripts/check_part_ev_screen_amp_recovery.py sbatch/*part_ev* particle_transformer/data/JetClass/JetClass_full.yaml)" ]] || fail 'Commit relevant changes before submitting'
  for file in part_ev_gpu_scaling/protocol.py part_ev_gpu_scaling/benchmark.py scripts/benchmark_part_ev_gpu_scaling.py sbatch/run_part_ev_gpu_scaling.sh; do
    git -C "$scale_repo" cat-file -e "HEAD:$file" || fail 'Commit/push/pull the diagnostic first'
  done
  # A typo must not create a source snapshot inside an existing campaign.
  candidate="$(realpath -m -- "$output")"
  [[ "$candidate" != "$campaign" && "$candidate" != "$campaign/"* && "$campaign" != "$candidate/"* ]] || fail 'Diagnostic must be outside the original campaign'
  partition_info="$(scontrol show partition "$partition" -o)"
  limit="$(sed -n 's/.*MaxTime=\([^ ]*\).*/\1/p' <<< "$partition_info")"
  if [[ "$limit" != UNLIMITED && "$limit" != INFINITE ]]; then
    days=0
    if [[ "$limit" == *-* ]]; then days="${limit%%-*}"; limit="${limit#*-}"; fi
    IFS=: read -r hours minutes seconds <<< "$limit"
    [[ "$days:$hours:$minutes:$seconds" =~ ^[0-9]+:[0-9]+:[0-9]+:[0-9]+$ ]] || fail 'Cannot read partition MaxTime'
    (( 10#$days*86400 + 10#$hours*3600 + 10#$minutes*60 + 10#$seconds >= 16200 )) || fail 'Partition time limit is below 04:30:00'
  fi
  scale_parent="$(dirname -- "$output")"
  [[ -d "$scale_parent" ]] || mkdir -p -- "$scale_parent"
  mkdir -- "$output"
  output="$(realpath -e -- "$output")"
  mkdir -- "$output/source" "$output/logs"
  git -C "$scale_repo" rev-parse HEAD > "$output/source_commit.txt"
  git -C "$scale_repo" archive HEAD | tar -xf - -C "$output/source"
  (cd -- "$output/source"; find . -type f -print0 | sort -z | xargs -0 sha256sum) > "$output/source.sha256"
  printf 'EVSCALE_PYTHON=%q\nEVSCALE_CAMPAIGN=%q\nEVSCALE_ACCOUNT=%q\nEVSCALE_PARTITION=%q\n' \
    "$scale_python" "$campaign" "$account" "$partition" > "$output/config.sh"
  (cd -- "$output"; sha256sum config.sh) > "$output/config.sha256"
fi
output="$(realpath -e -- "$output")"
(cd -- "$output"; sha256sum --quiet -c config.sha256)
source "$output/config.sh"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$output/source"
export PATH="$(dirname -- "$EVSCALE_PYTHON"):$PATH"
export LD_LIBRARY_PATH="$(dirname -- "$EVSCALE_PYTHON")/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
(cd -- "$output/source"; sha256sum --quiet -c "$output/source.sha256")
driver="$output/source/scripts/benchmark_part_ev_gpu_scaling.py"
if [[ "$mode" == review ]]; then
  exec "$EVSCALE_PYTHON" "$driver" --phase review --output "$output"
fi
"$EVSCALE_PYTHON" "$driver" --phase prepare --campaign "$EVSCALE_CAMPAIGN" --output "$output"
for option_name in ${!SBATCH_@}; do unset "$option_name"; done
dependency=
for gpus in 1 2; do
  request=(--parsable --job-name="evscale_${gpus}g" --nodes=1 --ntasks=1
    --gres="gpu:a100:$gpus" --cpus-per-task=8 --mem=64G --time=04:30:00
    --partition="$EVSCALE_PARTITION" --account="$EVSCALE_ACCOUNT"
    --output="$output/logs/%x_%j.out" --error="$output/logs/%x_%j.err" --export=ALL)
  [[ -z "$dependency" ]] || request+=(--dependency="afterany:$dependency" --kill-on-invalid-dep=yes)
  job="$(EVSCALE_OUTPUT="$output" EVSCALE_GPUS="$gpus" sbatch "${request[@]}" "$output/source/sbatch/run_part_ev_gpu_scaling.sh")"
  job="${job%%;*}"
  [[ "$job" =~ ^[0-9]+$ ]] || fail 'Unexpected sbatch response: inspect queue before doing anything else'
  printf '%s\t%s\n' "$gpus" "$job" >> "$output/submitted_job_ids.tsv"
  printf '%s-GPU diagnostic job: %s\n' "$gpus" "$job"
  dependency="$job"
done
printf 'Diagnostic: %s\n' "$output"
printf 'Review: bash sbatch/submit_part_ev_gpu_scaling.sh --mode review --output %q\n' "$output"
