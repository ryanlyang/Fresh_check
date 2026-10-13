#!/usr/bin/env bash
# Frozen alongside the standalone diagnostic, independent of production launchers.
set -euo pipefail
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
: "${PART_FULLSCALE_TIMING_OUTPUT:?Missing absolute diagnostic output}"
[[ "$PART_FULLSCALE_TIMING_OUTPUT" == /* ]] || fail 'Diagnostic output must be absolute'
output="$(realpath -e -- "$PART_FULLSCALE_TIMING_OUTPUT")"
[[ -f "$output/diagnostic.sha256" ]] || fail 'Diagnostic checksum manifest missing'
cd -- "$output"
sha256sum --quiet -c diagnostic.sha256 || fail 'Diagnostic checksum verification failed'
# Authenticate configuration before reading it; the original configuration is
# verified separately against the checksums captured when this job was submitted.
# shellcheck disable=SC1091
source "$output/launch.env"
: "${PART_FULLSCALE_TIMING_CAMPAIGN:?Missing campaign}"
campaign="$(realpath -e -- "$PART_FULLSCALE_TIMING_CAMPAIGN")"
[[ "$output" == "$campaign"/diagnostics/timing.* && "$(dirname -- "$output")" == "$campaign/diagnostics" ]] || fail 'Diagnostic output must be directly inside campaign/diagnostics'
cd -- "$campaign"
sha256sum --quiet -c "$output/campaign.sha256" || fail 'Original campaign configuration checksum verification failed'
# shellcheck disable=SC1090
source "$campaign/launch.env"
[[ "${PART_FULLSCALE_PYTHON:-}" == /* && -x "$PART_FULLSCALE_PYTHON" ]] || fail 'Pinned campaign Python must be an existing absolute executable'
[[ "$(uname -m)" == aarch64 ]] || fail 'This worker requires Tigris aarch64'
[[ "${SLURM_NNODES:-1}" == 1 && "${SLURM_NTASKS:-1}" == 1 ]] || fail 'Timing requires exactly one node and one task'
source_root="$campaign/source"
cd -- "$source_root"
sha256sum --quiet -c "$campaign/source.sha256" || fail 'Original campaign source checksum verification failed'
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$source_root"
env_bin="$(dirname -- "$PART_FULLSCALE_PYTHON")"
export PATH="$env_bin:$PATH" LD_LIBRARY_PATH="$env_bin/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 LOCAL_WORLD_SIZE=1
unset MASTER_ADDR MASTER_PORT GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE
for option_name in ${!SRUN_@}; do unset "$option_name"; done
# A single Slurm step receives USR1@300; no torchrun/NCCL or other rank is started.
exec srun --kill-on-bad-exit=1 --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  "$PART_FULLSCALE_PYTHON" -u "$output/source/scripts/diagnose_part_fullscale_timing.py" \
  --campaign "$campaign" --output "$output" \
  --train-steps "$PART_FULLSCALE_TIMING_TRAIN_STEPS" \
  --validation-steps "$PART_FULLSCALE_TIMING_VALIDATION_STEPS" \
  --budget-minutes "$PART_FULLSCALE_TIMING_BUDGET_MINUTES"
