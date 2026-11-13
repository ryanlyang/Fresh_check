#!/usr/bin/env bash
# Independent benchmark only. No cancellation, production, or environment writes.
set -euo pipefail
memory_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
mode= output= dry=0
reference=/home/ryreu/atlas/Fresh_check/checkpoints/part_ev_layerwise_ddp/layerwise_2g_20261029T162243Z
memory_python=/home/ryreu/miniconda3/envs/part_ev_screen_sporc/bin/python
account=reu-aisocial partition=tier3
explicit=()
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
while (( $# )); do
  case "$1" in
    --help|-h)
      echo 'Usage: bash sbatch/submit_part_ev_pair_memory.sh --mode prepare|review --output ABSOLUTE_DIR'
      echo 'prepare: freeze NEW output; queue CPU prepare -> two-A100 benchmark -> CPU summary.'
      echo 'Options: --reference DIR --python PATH --account NAME --partition NAME --dry-run'
      echo 'Walltimes: prepare 05:00:00; benchmark 05:30:00; summary 00:15:00. No production mode.'
      exit 0 ;;
    --dry-run) dry=1; shift ;;
    --mode|--output|--reference|--python|--account|--partition)
      (( $#>=2 )) || fail "Missing $1 value"
      case "$1" in --reference|--python|--account|--partition) explicit+=("$1" "$2") ;; esac
      case "$1" in
        --mode) mode=$2 ;; --output) output=$2 ;; --reference) reference=$2 ;;
        --python) memory_python=$2 ;; --account) account=$2 ;; --partition) partition=$2 ;;
      esac
      shift 2 ;;
    *) fail "Unknown option: $1" ;;
  esac
done
[[ "$mode" == prepare || "$mode" == review ]] || fail 'Choose prepare or review; no training mode exists'
[[ "$account" =~ ^[a-zA-Z0-9_-]+$ && "$partition" =~ ^[a-zA-Z0-9_-]+$ ]] || fail 'Invalid resource name'
if [[ -z "$output" && "$mode" == prepare ]]; then
  output="$memory_repo/checkpoints/part_ev_pair_memory/pair_memory_$(date -u +%Y%m%dT%H%M%SZ)"
fi
[[ "$output" == /* && "$reference" == /* && "$memory_python" == /* ]] || fail 'Use absolute paths'
if (( dry )); then
  printf 'Mode: %s\nOutput: %s\nReference (read only): %s\nPartition: %s\n' "$mode" "$output" "$reference" "$partition"
  echo 'prepare: 0 GPUs, 8 CPUs, 64G RAM, 05:00:00'
  echo 'benchmark: --nodes=1 --ntasks=1 --gres=gpu:a100:2 --cpus-per-task=8 --mem=64G --time=05:30:00'
  echo 'summary: 0 GPUs, 2 CPUs, 4G RAM, 00:15:00; afterany benchmark; no training authorization.'
  echo 'No files changed or jobs submitted. Live partition/account checks occur on real submission.'
  exit 0
fi
[[ "$(uname -m)" == x86_64 ]] || fail 'Use sporcsubmit with the existing x86_64 environment'
for tool in git tar sha256sum realpath; do command -v "$tool" >/dev/null || fail "Missing $tool"; done
for name in ${!SBATCH_@}; do unset "$name"; done
if [[ "$mode" == prepare ]]; then
  for tool in sbatch scontrol flock; do command -v "$tool" >/dev/null || fail "Missing $tool"; done
  [[ ! -e "$output" && ! -L "$output" ]] || fail 'Use a NEW output; old evidence is never overwritten'
  reference="$(realpath -e -- "$reference")"
  candidate="$(realpath -m -- "$output")"
  [[ "$candidate" != "$reference" && "$candidate" != "$reference/"* && "$reference" != "$candidate/"* ]] || fail 'Output must be separate from reference'
  [[ -f "$reference/campaign.json" && -x "$memory_python" ]] || fail 'Missing reference/environment'
  # Never silently route tier3 to a different partition. Check live availability.
  part_info="$(scontrol show partition "$partition" -o)"
  [[ "$part_info" == *"PartitionName=$partition "* && "$part_info" == *'State=UP'* ]] || fail 'Requested partition is not UP; inspect before choosing another'
  limit="$(sed -n 's/.*MaxTime=\([^ ]*\).*/\1/p' <<< "$part_info")"
  if [[ "$limit" != UNLIMITED && "$limit" != INFINITE ]]; then
    day=0
    if [[ "$limit" == *-* ]]; then day="${limit%%-*}"; limit="${limit#*-}"; fi
    IFS=: read -r hh mm ss <<< "$limit"
    [[ "$day:$hh:$mm:$ss" =~ ^[0-9]+:[0-9]+:[0-9]+:[0-9]+$ ]] || fail 'Cannot parse partition MaxTime'
    (( 19800 <= 10#$day*86400+10#$hh*3600+10#$mm*60+10#$ss )) || fail 'Partition limit below benchmark walltime'
  fi
  paths=(part_ev_pair_memory part_ev_fullscale part_ev_screen part_ev_message_study part_ev_state_speed
    part_ev_state_fullscale part_ev_state_capacity part_ev_state_compare part_layerwise_screen
    part_ev_gpu_scaling part_ev_layerwise_ddp particle_transformer/data/JetClass/JetClass_full.yaml
    scripts/run_part_ev_screen.py scripts/check_part_ev_screen_amp_recovery.py
    scripts/run_part_ev_pair_memory.py sbatch/submit_part_ev_pair_memory.sh sbatch/run_part_ev_pair_memory.sh)
  [[ -z "$(git -C "$memory_repo" status --porcelain -- "${paths[@]}")" ]] || fail 'Commit/push/pull relevant source before preparation'
  memory_commit="$(git -C "$memory_repo" rev-parse HEAD)"
  for file in part_ev_pair_memory/model.py part_ev_pair_memory/benchmark.py part_ev_pair_memory/protocol.py part_ev_pair_memory/oracle.py scripts/run_part_ev_pair_memory.py sbatch/run_part_ev_pair_memory.sh; do
    git -C "$memory_repo" cat-file -e "$memory_commit:$file" || fail 'Implementation must be committed in HEAD'
  done
  parent="$(dirname -- "$output")"
  [[ -d "$parent" ]] || mkdir -p -- "$parent"
  mkdir -- "$output"
  output="$(realpath -e -- "$output")"
  mkdir -- "$output/source" "$output/logs"
  printf '%s\n' "$memory_commit" > "$output/source_commit.txt"
  git -C "$memory_repo" archive "$memory_commit" | tar -xf - -C "$output/source"
  "$memory_python" -B - "$output" "$reference" "$memory_python" "$account" "$partition" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
root=Path(sys.argv[1])
manifest={p.relative_to(root/'source').as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
          for p in sorted((root/'source').rglob('*')) if p.is_file()}
launch=dict(reference=sys.argv[2],python=sys.argv[3],account=sys.argv[4],partition=sys.argv[5],
            nccl_environment={k:v for k,v in os.environ.items() if k.startswith('NCCL_')})
(root/'source_manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
(root/'launch.json').write_text(json.dumps(launch,sort_keys=True,indent=2)+'\n')
PY
  printf 'PAIRMEM_PYTHON=%q\nPAIRMEM_ACCOUNT=%q\nPAIRMEM_PARTITION=%q\nPAIRMEM_REFERENCE=%q\n' \
    "$memory_python" "$account" "$partition" "$reference" > "$output/config.sh"
  (cd -- "$output"; sha256sum config.sh launch.json source_manifest.json) > "$output/config.sha256"
fi
output="$(realpath -e -- "$output")"
(cd -- "$output"; sha256sum --quiet -c config.sha256)
source "$output/config.sh"
if [[ "$mode" == review ]]; then
  for ((i=0;i<${#explicit[@]};i+=2)); do
    case "${explicit[i]}" in
      --reference) frozen="$PAIRMEM_REFERENCE" ;; --python) frozen="$PAIRMEM_PYTHON" ;;
      --account) frozen="$PAIRMEM_ACCOUNT" ;; --partition) frozen="$PAIRMEM_PARTITION" ;;
    esac
    [[ "${explicit[i+1]}" == "$frozen" ]] || fail 'Override differs from frozen campaign'
  done
fi
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$output/source"
export PATH="$(dirname -- "$PAIRMEM_PYTHON"):$PATH"
export LD_LIBRARY_PATH="$(dirname -- "$PAIRMEM_PYTHON")/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd -- "$output/source"
if [[ "$mode" == review ]]; then
  exec "$PAIRMEM_PYTHON" -B scripts/run_part_ev_pair_memory.py --phase review --output "$output"
fi
exec 9>"$output/.submission.lock"
flock -n 9 || fail 'Another submission holds this campaign lock'
[[ ! -s "$output/submitted_job_ids.tsv" ]] || fail 'Already submitted; use a fresh campaign'
submit() {
  local phase=$1 dep=$2 hours=$3 gpu=$4 cpus=$5 ram=$6 job
  local request=(--parsable --job-name="evmem_$phase" --partition="$PAIRMEM_PARTITION" --account="$PAIRMEM_ACCOUNT"
    --nodes=1 --ntasks=1 --cpus-per-task="$cpus" --mem="$ram" --time="$hours" --export=ALL
    --output="$output/logs/%x_%j.out" --error="$output/logs/%x_%j.err")
  (( gpu==0 )) || request+=(--gres="gpu:a100:$gpu" --signal=B:TERM@180)
  [[ -z "$dep" ]] || request+=(--dependency="$dep" --kill-on-invalid-dep=yes)
  job="$(PAIRMEM_OUTPUT="$output" PAIRMEM_PHASE="$phase" sbatch "${request[@]}" "$output/source/sbatch/run_part_ev_pair_memory.sh")"
  job="${job%%;*}"
  [[ "$job" =~ ^[0-9]+$ ]] || fail 'Unexpected sbatch response; inspect scheduler before retrying'
  printf '%s\t%s\n' "$phase" "$job" >> "$output/submitted_job_ids.tsv"
  printf '%s job: %s\n' "$phase" "$job" >&2
  printf '%s' "$job"
}
prepare_job="$(submit prepare '' 05:00:00 0 8 64G)"
benchmark_job="$(submit benchmark "afterok:$prepare_job" 05:30:00 2 8 64G)"
submit summary "afterany:$benchmark_job" 00:15:00 0 2 4G
printf '\nCampaign: %s\nBenchmarks only. Existing campaigns/jobs unchanged.\n' "$output"
