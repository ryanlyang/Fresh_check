#!/usr/bin/env bash
# Execute only from the independently frozen validation evaluation snapshot.
set -euo pipefail
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
: "${PART_VALIDATION_OUTPUT:?Missing validation output}"
: "${PART_VALIDATION_PHASE:?Missing validation phase}"
[[ "$PART_VALIDATION_OUTPUT" == /* ]] || fail 'Validation output must be absolute'
output="$(realpath -e -- "$PART_VALIDATION_OUTPUT")"
phase="$PART_VALIDATION_PHASE"
[[ -f "$output/validation.sha256" ]] || fail 'Missing validation snapshot manifest'
cd -- "$output"
sha256sum --quiet -c validation.sha256 || fail 'Validation snapshot checksum verification failed'
source "$output/validation_launch.env"
[[ "$PART_VALIDATION_OUTPUT" == "$output" ]] || fail 'Frozen output path differs'
campaign="$(realpath -e -- "$PART_VALIDATION_CAMPAIGN")"
[[ "$output" != "$campaign" && "$(dirname -- "$output")" == "$(dirname -- "$campaign")" ]] || fail 'Validation output must be a separate sibling of the campaign'
cd -- "$campaign"
sha256sum --quiet -c "$output/original.sha256" || fail 'Original campaign metadata checksum verification failed'
unset PART_FULLSCALE_PYTHON PART_FULLSCALE_COMPARISON PART_FULLSCALE_TRAINING_RECIPE
source "$output/original_metadata/launch.env"
[[ "${PART_FULLSCALE_COMPARISON:-standard}" == standard && "${PART_FULLSCALE_TRAINING_RECIPE:-original}" == original ]] || fail 'Validation evaluation requires original standard/Ranger models'
[[ "$PART_FULLSCALE_PYTHON" == "$PART_VALIDATION_PYTHON" && -x "$PART_VALIDATION_PYTHON" ]] || fail 'Pinned original Python is unavailable or changed'
[[ "$(uname -m)" == aarch64 ]] || fail 'Validation worker requires Tigris aarch64'
[[ "${SLURM_NNODES:-1}" == 1 && "${SLURM_NTASKS:-1}" == 1 ]] || fail 'Validation requires exactly one node and one task'
source_root="$campaign/source"
cd -- "$source_root"
sha256sum --quiet -c "$campaign/source.sha256" || fail 'Original source checksum verification failed'
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$source_root:$output/source/scripts"
env_bin="$(dirname -- "$PART_VALIDATION_PYTHON")"
export PATH="$env_bin:$PATH" LD_LIBRARY_PATH="$env_bin/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" OPENBLAS_NUM_THREADS=1
export RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 LOCAL_WORLD_SIZE=1
unset MASTER_ADDR MASTER_PORT GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE
for option_name in ${!SRUN_@}; do unset "$option_name"; done
driver="$output/source/scripts/evaluate_part_fullscale_validation.py"
args=("$phase" --output "$output")
case "$phase" in
  prepare) args+=(--campaign "$campaign") ;;
  evaluate)
    [[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-5]$ ]] || fail 'Evaluation task index must be 0..5'
    args+=(--task-index "$SLURM_ARRAY_TASK_ID") ;;
  report) ;;
  *) fail 'Unknown validation-only phase' ;;
esac
cd -- "$output"
if [[ "$phase" == evaluate ]]; then
  exec srun --kill-on-bad-exit=1 --nodes=1 --ntasks=1 --ntasks-per-node=1 \
    "$PART_VALIDATION_PYTHON" -u "$driver" "${args[@]}"
fi
exec "$PART_VALIDATION_PYTHON" -u "$driver" "${args[@]}"
