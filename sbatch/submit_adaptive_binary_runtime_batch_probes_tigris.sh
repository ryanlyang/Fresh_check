#!/usr/bin/env bash
# Queue canonical DDP4 full-step probes and immutable contracts for ABPH variants.

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
export ABPH_ROOT PYTHONNOUSERSITE=1

fresh_require_file "${ABPH_ROOT}/audits/actual_target_feasibility.json"
fresh_require_file "${ABPH_ROOT}/runs/A0_hlt_part/best_model_val.pt"

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
manifest="${ABPH_ROOT}/submission_logs/abph_runtime_batch_probes.tsv"
mkdir -p "$(dirname "${manifest}")"
printf 'variant\tstage_family\tlocal_batch_size\tjob_id\n' > "${manifest}"

for variant in "${variants[@]}"; do
  probe_ids=()
  for spec in root_hierarchy:256 root_hierarchy:128 root_hierarchy:64 \
              renderer_distribution:128 renderer_distribution:64 renderer_distribution:32; do
    family="${spec%%:*}"
    batch="${spec##*:}"
    if fresh_is_dry_run; then
      submitted="dryrun_${variant}_${family}_b${batch}"
      printf 'DRY_RUN sbatch probe: variant=%q family=%q batch=%q\n' \
        "${variant}" "${family}" "${batch}"
    else
      submitted="$(sbatch --parsable --account="${ABPH_SBATCH_ACCOUNT}" \
        --partition="${ABPH_SBATCH_PARTITION}" --nodes=4 --ntasks=4 \
        --ntasks-per-node=1 --cpus-per-task=16 --mem=220G \
        --gres=gpu:gh200:1 "${probe_worker}" "${variant}" "${family}" "${batch}")"
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
      "${compile_worker}" "${variant}")"
  fi
  printf '%s\tcompile\tall\t%s\n' "${variant}" "${submitted%%;*}" >> "${manifest}"
  echo "${submitted}"
done

echo "adaptive_binary_runtime_batch_probe_submission_complete:"
echo "  root: ${ABPH_ROOT}"
echo "  variants: ${#variants[@]}"
echo "  manifest: ${manifest}"
