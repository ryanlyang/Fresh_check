#!/usr/bin/env bash
# Queue canonical full-step probes and immutable contracts for an ABPH topology.

set -euo pipefail
IFS=$'\n\t'
: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
: "${CONDA_ENV:=atlas_kd_tigris}"
export PROJECT_DIR CONDA_BASE CONDA_ENV
source "${PROJECT_DIR}/sbatch/common.sh"
fresh_prepare_submitter
fresh_activate_env
: "${ABPH_ROOT:?Set ABPH_ROOT to a prepared campaign root}"
: "${ABPH_SBATCH_ACCOUNT:=reu-aisocial}"
: "${ABPH_SBATCH_PARTITION:=tigris}"
: "${ABPH_RUNTIME_BATCH_MEASUREMENT_ROOT:=${ABPH_ROOT}/runtime_batch_measurements}"
: "${ABPH_RUNTIME_BATCH_CONTRACT_ROOT:=${ABPH_ROOT}/runtime_batch_contracts}"
: "${ABPH_RUNTIME_BATCH_PROBE_MANIFEST:=${ABPH_ROOT}/submission_logs/abph_runtime_batch_probes.tsv}"
: "${ABPH_TARGET_MODE_REPORT:=${ABPH_ROOT}/audits/target_mode_selection.json}"
: "${ABPH_RUNTIME_BATCH_WORLD_SIZE:=4}"
: "${ABPH_RUNTIME_BATCH_UPSTREAM_DEPENDENCY:=}"
[[ "${ABPH_RUNTIME_BATCH_WORLD_SIZE}" == "4" || "${ABPH_RUNTIME_BATCH_WORLD_SIZE}" == "8" ]] || {
  echo "ABPH_RUNTIME_BATCH_WORLD_SIZE must be 4 or 8" >&2
  exit 2
}
if [[ -n "${ABPH_RUNTIME_BATCH_UPSTREAM_DEPENDENCY}" ]] &&
   [[ ! "${ABPH_RUNTIME_BATCH_UPSTREAM_DEPENDENCY}" =~ ^afterok:[0-9]+(:[0-9]+)*$ ]]; then
  echo "ABPH_RUNTIME_BATCH_UPSTREAM_DEPENDENCY must be an afterok dependency." >&2
  exit 2
fi
export ABPH_DISTRIBUTED_NODES="${ABPH_RUNTIME_BATCH_WORLD_SIZE}"
export ABPH_DISTRIBUTED_NTASKS="${ABPH_RUNTIME_BATCH_WORLD_SIZE}"
export ABPH_DISTRIBUTED_NTASKS_PER_NODE=1
export ABPH_DISTRIBUTED_WORLD_SIZE="${ABPH_RUNTIME_BATCH_WORLD_SIZE}"
export ABPH_ROOT ABPH_RUNTIME_BATCH_MEASUREMENT_ROOT
export ABPH_RUNTIME_BATCH_CONTRACT_ROOT PYTHONNOUSERSITE=1
export ABPH_TARGET_MODE_REPORT

fresh_require_file "${ABPH_ROOT}/audits/actual_target_feasibility.json"
fresh_require_file "${ABPH_ROOT}/runs/A0_hlt_part/best_model_val.pt"
if [[ "${ABPH_STORAGE_PROFILE:-}" == "streaming_30gb_v1" ]]; then
  fresh_require_file "${ABPH_TARGET_MODE_REPORT}"
fi

if (($#)); then
  variants=("$@")
else
  mapfile -t variants < <("${PYTHON_BIN}" - <<'PY'
from teacher_logit_reco.adaptive_binary_pseudooffline.orchestration import (
    ABPH_RECONSTRUCTOR_VARIANTS,
    ABPH_RENDERER_VARIANTS,
)
print(*ABPH_RECONSTRUCTOR_VARIANTS, *ABPH_RENDERER_VARIANTS, sep="\n")
PY
  )
fi

probe_worker="${PROJECT_DIR}/sbatch/run_adaptive_binary_runtime_batch_probe.sh"
compile_worker="${PROJECT_DIR}/sbatch/run_compile_adaptive_binary_runtime_batch_contract.sh"
manifest="${ABPH_RUNTIME_BATCH_PROBE_MANIFEST}"
mkdir -p "$(dirname "${manifest}")"
printf 'variant\tstage_family\tlocal_batch_size\tjob_id\n' > "${manifest}"

for variant in "${variants[@]}"; do
  probe_ids=()
  if [[ "${ABPH_RUNTIME_BATCH_WORLD_SIZE}" == "8" ]]; then
    specs=(root_hierarchy:128 root_hierarchy:64 \
           renderer_distribution:64 renderer_distribution:32)
  else
    specs=(root_hierarchy:256 root_hierarchy:128 root_hierarchy:64 \
           renderer_distribution:128 renderer_distribution:64 renderer_distribution:32)
  fi
  for spec in "${specs[@]}"; do
    family="${spec%%:*}"
    batch="${spec##*:}"
    dependency_args=()
    if [[ -n "${ABPH_RUNTIME_BATCH_UPSTREAM_DEPENDENCY}" ]]; then
      dependency_args=(
        "--dependency=${ABPH_RUNTIME_BATCH_UPSTREAM_DEPENDENCY}"
      )
    fi
    if fresh_is_dry_run; then
      submitted="dryrun_${variant}_${family}_b${batch}"
      printf 'DRY_RUN sbatch probe: variant=%q family=%q batch=%q dependency=%q\n' \
        "${variant}" "${family}" "${batch}" \
        "${ABPH_RUNTIME_BATCH_UPSTREAM_DEPENDENCY}"
    else
      submitted="$(sbatch --parsable --account="${ABPH_SBATCH_ACCOUNT}" \
        --partition="${ABPH_SBATCH_PARTITION}" \
        --nodes="${ABPH_RUNTIME_BATCH_WORLD_SIZE}" \
        --ntasks="${ABPH_RUNTIME_BATCH_WORLD_SIZE}" \
        --ntasks-per-node=1 --cpus-per-task=16 --mem=220G \
        --gres=gpu:gh200:1 "${dependency_args[@]}" \
        "${probe_worker}" "${variant}" "${family}" "${batch}")"
    fi
    job_id="${submitted%%;*}"
    probe_ids+=("${job_id}")
    printf '%s\t%s\t%s\t%s\n' "${variant}" "${family}" "${batch}" "${job_id}" >> "${manifest}"
    echo "${submitted}"
  done
  dependency="afterok:$(IFS=:; echo "${probe_ids[*]}")"
  if fresh_is_dry_run; then
    submitted="dryrun_${variant}_compile"
    printf 'DRY_RUN sbatch compiler: variant=%q dependency=%q\n' \
      "${variant}" "${dependency}"
  else
    submitted="$(sbatch --parsable --account="${ABPH_SBATCH_ACCOUNT}" \
      --partition="${ABPH_SBATCH_PARTITION}" --dependency="${dependency}" \
      "${compile_worker}" "${variant}" "${ABPH_RUNTIME_BATCH_WORLD_SIZE}")"
  fi
  printf '%s\tcompile\tall\t%s\n' "${variant}" "${submitted%%;*}" >> "${manifest}"
  echo "${submitted}"
done

echo "adaptive_binary_runtime_batch_probe_submission_complete:"
echo "  root: ${ABPH_ROOT}"
echo "  variants: ${#variants[@]}"
echo "  world_size: ${ABPH_RUNTIME_BATCH_WORLD_SIZE}"
echo "  manifest: ${manifest}"
