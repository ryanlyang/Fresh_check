#!/usr/bin/env bash
# Queue the supplemental C3 -> kT8 renderer -> HLT+kT8 tagger screen.

set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
: "${CONDA_ENV:=atlas_kd_tigris}"
export PROJECT_DIR CONDA_BASE CONDA_ENV
source "${PROJECT_DIR}/sbatch/common.sh"
fresh_prepare_submitter
fresh_activate_env

: "${ABPH_ROOT:?Set ABPH_ROOT to the campaign containing the completed C3 run}"
: "${ABPH_DATA_DIR:=/home/ryreu/atlas/PracticeTagging/data/jetclass_part1}"
: "${ABPH_SBATCH_ACCOUNT:=reu-aisocial}"
: "${ABPH_SBATCH_PARTITION:=tigris}"
: "${ABPH_STORAGE_PROFILE:=streaming_30gb_v1}"
: "${ABPH_STORAGE_CONTRACT_PATH:=${ABPH_ROOT}/storage/storage_contract.json}"
: "${ABPH_TARGET_MODE_REPORT:=${ABPH_ROOT}/audits/target_mode_selection.json}"
: "${ABPH_KT8_MANIFEST:=${ABPH_ROOT}/submission_logs/abph_kt8_screen_submission.tsv}"
: "${ABPH_KT8_CONFIRM_RESUBMIT:=0}"

source_variant=C3_kt_8
renderer_variant=D7_kt8_mh4_particles_screen
tagger_variant=E12_kt8_mh4_dualcross_screen
source_checkpoint="${ABPH_ROOT}/runs/${source_variant}/best_model_val.pt"
source_report="${ABPH_ROOT}/runs/${source_variant}/run_report.json"
renderer_checkpoint="${ABPH_ROOT}/runs/${renderer_variant}/best_model_val.pt"
tagger_report="${ABPH_ROOT}/runs/${tagger_variant}/run_report.json"
contract_path="${ABPH_ROOT}/runtime_batch_contracts/${renderer_variant}/runtime_batch_contract.json"

export DATA_DIR="${ABPH_DATA_DIR}"
export ABPH_ROOT ABPH_STORAGE_PROFILE ABPH_STORAGE_CONTRACT_PATH
export ABPH_TARGET_MODE_REPORT PYTHONNOUSERSITE=1 DEVICE=cuda
export ABPH_RUNTIME_BATCH_WORLD_SIZE=8
export ABPH_DISTRIBUTED_NODES=8
export ABPH_DISTRIBUTED_NTASKS=8
export ABPH_DISTRIBUTED_NTASKS_PER_NODE=1
export ABPH_DISTRIBUTED_GPUS_PER_NODE=1
export ABPH_DISTRIBUTED_WORLD_SIZE=8
export ABPH_RECONSTRUCTOR_PARALLELISM=ddp8
export ABPH_RECONSTRUCTOR_SCHEDULE_POLICY=accelerated_screening_v2_7day
export ABPH_JOB_LAUNCHER=srun

fresh_require_file "${source_checkpoint}"
fresh_require_file "${source_report}"
fresh_require_file "${ABPH_ROOT}/runs/A0_hlt_part/best_model_val.pt"
fresh_require_file "${ABPH_ROOT}/audits/actual_target_feasibility.json"
fresh_require_file "${ABPH_TARGET_MODE_REPORT}"
fresh_require_file "${ABPH_STORAGE_CONTRACT_PATH}"

"${PYTHON_BIN}" - "${source_checkpoint}" "${source_report}" <<'PY'
import json
import sys

from teacher_logit_reco.adaptive_binary_pseudooffline.checkpoints import (
    load_torch_checkpoint,
    selected_checkpoint_provenance,
)

checkpoint_path, report_path = sys.argv[1:]
checkpoint = load_torch_checkpoint(checkpoint_path, device="cpu")
provenance = selected_checkpoint_provenance(checkpoint_path, checkpoint)
with open(report_path, encoding="utf-8") as handle:
    report = json.load(handle)

if report.get("ok") is not True or report.get("variant_name") != "C3_kt_8":
    raise SystemExit("C3 run report is not a successful C3_kt_8 result")
if checkpoint.get("checkpoint_role") != "best_model_val":
    raise SystemExit("C3 source is not the final model-val-selected checkpoint")
if checkpoint.get("final_test_loaded") is not False:
    raise SystemExit("C3 checkpoint does not attest final-test isolation")
if provenance.get("file_sha256") in (None, ""):
    raise SystemExit("C3 checkpoint provenance lacks its file hash")
print(f"validated_C3_checkpoint_sha256={provenance['file_sha256']}")
PY

if [[ -f "${tagger_report}" ]]; then
  echo "kT8 screen is already complete: ${tagger_report}"
  exit 0
fi

if [[ -f "${ABPH_KT8_MANIFEST}" ]] &&
   ! fresh_bool_enabled "${ABPH_KT8_CONFIRM_RESUBMIT}"; then
  echo "Refusing duplicate kT8 submission; manifest exists: ${ABPH_KT8_MANIFEST}" >&2
  echo "Set ABPH_KT8_CONFIRM_RESUBMIT=1 only after auditing its recorded jobs." >&2
  exit 2
fi

if [[ -f "${ABPH_KT8_MANIFEST}" ]]; then
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  archive_root="${ABPH_ROOT}/archives/kt8_screen_resubmit_${stamp}"
  mapfile -t previous_jobs < <(
    awk -F $'\t' 'NR > 1 && $3 ~ /^[0-9]+$/ {print $3}' \
      "${ABPH_KT8_MANIFEST}"
  )
  if ((${#previous_jobs[@]})); then
    scancel "${previous_jobs[@]}" 2>/dev/null || true
    previous_job_csv="$(IFS=,; echo "${previous_jobs[*]}")"
    for _attempt in {1..30}; do
      [[ -z "$(squeue -h -j "${previous_job_csv}" 2>/dev/null)" ]] && break
      sleep 2
    done
    [[ -z "$(squeue -h -j "${previous_job_csv}" 2>/dev/null)" ]] || {
      echo "Timed out waiting for superseded kT8 jobs to leave the queue" >&2
      exit 2
    }
  fi
  mkdir -p "${archive_root}"
  for artifact_kind in runtime_batch_measurements runtime_batch_contracts; do
    previous="${ABPH_ROOT}/${artifact_kind}/${renderer_variant}"
    if [[ -e "${previous}" ]]; then
      [[ ! -L "${previous}" ]] || {
        echo "Refusing to archive symlinked kT8 evidence: ${previous}" >&2
        exit 2
      }
      mkdir -p "${archive_root}/${artifact_kind}"
      mv -- "${previous}" "${archive_root}/${artifact_kind}/"
    fi
  done
  mv -- "${ABPH_KT8_MANIFEST}" "${archive_root}/"
  echo "Archived superseded kT8 evidence under ${archive_root}"
fi

mkdir -p "$(dirname "${ABPH_KT8_MANIFEST}")"
manifest_tmp="${ABPH_KT8_MANIFEST}.tmp.$$"
printf 'role\tvariant\tjob_id\tdependency\n' > "${manifest_tmp}"

if fresh_is_dry_run; then
  echo "DRY_RUN: certify ${renderer_variant} with four DDP8 batch probes"
  echo "DRY_RUN: train ${renderer_variant} from ${source_variant}"
  echo "DRY_RUN: train ${tagger_variant} from A0 plus ${renderer_variant}"
  rm -f "${manifest_tmp}"
  exit 0
fi

renderer_dependency=()
if [[ ! -f "${contract_path}" && ! -f "${renderer_checkpoint}" ]]; then
  probe_ids=()
  for spec in root_hierarchy:128 root_hierarchy:64 \
              renderer_distribution:64 renderer_distribution:32; do
    family="${spec%%:*}"
    batch="${spec##*:}"
    submitted="$(sbatch --parsable \
      --account="${ABPH_SBATCH_ACCOUNT}" \
      --partition="${ABPH_SBATCH_PARTITION}" \
      --nodes=8 --ntasks=8 --ntasks-per-node=1 \
      --cpus-per-task=16 --mem=220G --gres=gpu:gh200:1 \
      "${PROJECT_DIR}/sbatch/run_adaptive_binary_runtime_batch_probe.sh" \
      "${renderer_variant}" "${family}" "${batch}")"
    job_id="${submitted%%;*}"
    probe_ids+=("${job_id}")
    printf 'batch_probe:%s:%s\t%s\t%s\t\n' \
      "${family}" "${batch}" "${renderer_variant}" "${job_id}" >> "${manifest_tmp}"
  done

  probe_dependency="afterok:$(IFS=:; echo "${probe_ids[*]}")"
  submitted="$(sbatch --parsable \
    --account="${ABPH_SBATCH_ACCOUNT}" \
    --partition="${ABPH_SBATCH_PARTITION}" \
    --dependency="${probe_dependency}" \
    "${PROJECT_DIR}/sbatch/run_compile_adaptive_binary_runtime_batch_contract.sh" \
    "${renderer_variant}" 8)"
  contract_job="${submitted%%;*}"
  printf 'batch_contract\t%s\t%s\t%s\n' \
    "${renderer_variant}" "${contract_job}" "${probe_dependency}" >> "${manifest_tmp}"
  renderer_dependency=("--dependency=afterok:${contract_job}")
fi

renderer_job=""
if [[ ! -f "${renderer_checkpoint}" ]]; then
  submitted="$(sbatch --parsable \
    --job-name=abph_kt8_renderer \
    --account="${ABPH_SBATCH_ACCOUNT}" \
    --partition="${ABPH_SBATCH_PARTITION}" \
    --nodes=8 --ntasks=8 --ntasks-per-node=1 \
    --cpus-per-task=16 --mem=220G --gres=gpu:gh200:1 \
    --time=7-00:00:00 --output=/dev/null --error=/dev/null \
    "${renderer_dependency[@]}" \
    "${PROJECT_DIR}/sbatch/run_adaptive_binary_variant.sh" \
    "${renderer_variant}")"
  renderer_job="${submitted%%;*}"
  printf 'renderer\t%s\t%s\t%s\n' \
    "${renderer_variant}" "${renderer_job}" \
    "${renderer_dependency[*]:-none}" >> "${manifest_tmp}"
fi

# Keep this early-signal tagger to one GPU instead of recreating the broad
# footprint of the original full graph.
export ABPH_TAGGER_PARALLELISM=single
export ABPH_TAGGER_DISTRIBUTED_WORLD_SIZE=1
export ABPH_DISTRIBUTED_NODES=1
export ABPH_DISTRIBUTED_NTASKS=1
export ABPH_DISTRIBUTED_NTASKS_PER_NODE=1
export ABPH_DISTRIBUTED_GPUS_PER_NODE=1
export ABPH_DISTRIBUTED_WORLD_SIZE=1
export ABPH_JOB_LAUNCHER=direct

tagger_dependency=()
if [[ -n "${renderer_job}" ]]; then
  tagger_dependency=("--dependency=afterok:${renderer_job}")
fi
submitted="$(sbatch --parsable \
  --job-name=abph_kt8_tagger \
  --account="${ABPH_SBATCH_ACCOUNT}" \
  --partition="${ABPH_SBATCH_PARTITION}" \
  --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --cpus-per-task=16 --mem=300G --gres=gpu:gh200:1 \
  --time=7-00:00:00 --output=/dev/null --error=/dev/null \
  "${tagger_dependency[@]}" \
  "${PROJECT_DIR}/sbatch/run_adaptive_binary_variant.sh" \
  "${tagger_variant}")"
tagger_job="${submitted%%;*}"
printf 'tagger\t%s\t%s\t%s\n' \
  "${tagger_variant}" "${tagger_job}" \
  "${tagger_dependency[*]:-none}" >> "${manifest_tmp}"

mv -f "${manifest_tmp}" "${ABPH_KT8_MANIFEST}"

echo "adaptive_binary_kt8_screen_submission_complete:"
echo "  root: ${ABPH_ROOT}"
echo "  source: ${source_variant}"
echo "  renderer_job: ${renderer_job:-reused}"
echo "  tagger_job: ${tagger_job}"
echo "  manifest: ${ABPH_KT8_MANIFEST}"
