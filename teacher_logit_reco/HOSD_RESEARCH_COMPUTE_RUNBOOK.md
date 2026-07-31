# HOSD research-compute runbook

This runbook is part of the executable contract. Full submission is
fail-closed until the same-source real miniature, both intentional recovery
tests, resource measurement, production preflight, and authorization have
completed.

## 1. Environment and immutable inputs

Run from the repository root on Tigris. Keep the source snapshot unchanged
from miniature creation through full authorization.

```bash
source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate atlas_kd_tigris
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

export PROJECT_ROOT=/home/ryreu/atlas/Fresh_check
export HOSD_BOOTSTRAP="${PROJECT_ROOT}/checkpoints/hlt_offline_structure_distillation/bootstrap"
export MINI_ROOT="${PROJECT_ROOT}/checkpoints/hlt_offline_structure_distillation/miniature"
export PROD_ROOT="${PROJECT_ROOT}/checkpoints/hlt_offline_structure_distillation/production"
export JETCLASS_DATA_ROOT=/home/ryreu/atlas/PracticeTagging/data
export MINI_PARENT_MANIFEST="${HOSD_BOOTSTRAP}/miniature_source_split_manifest.json.gz"
export PROD_PARENT_MANIFEST="${HOSD_BOOTSTRAP}/production_source_split_manifest.json.gz"
export RETB_STORAGE_MEASUREMENTS=/absolute/path/to/authenticated_retb_storage_measurements.json
export MINI_RUNTIME_CONFIG="${HOSD_BOOTSTRAP}/miniature_runtime_config.json"
export PROD_RUNTIME_CONFIG="${HOSD_BOOTSTRAP}/production_runtime_config.json"

cd "${PROJECT_ROOT}"
python -s -c 'import torch,uproot,awkward,ninja,weaver; print(torch.__version__, torch.cuda.is_available())'
```

Build each exact source population with the canonical split builder. The
miniature and production profiles deliberately have different validation
counts and are not interchangeable.

```bash
mkdir -p "${HOSD_BOOTSTRAP}"
python -s scripts/build_jetclass_splits.py \
  --data-dir "${JETCLASS_DATA_ROOT}" \
  --out "${MINI_PARENT_MANIFEST}" \
  --tree-name tree --max-constits 128 \
  --model-train 20 --model-val 40 --stack-train 0 \
  --stack-val 10 --final-test 20

python -s scripts/build_jetclass_splits.py \
  --data-dir "${JETCLASS_DATA_ROOT}" \
  --out "${PROD_PARENT_MANIFEST}" \
  --tree-name tree --max-constits 128 \
  --model-train 500000 --model-val 100000 --stack-train 0 \
  --stack-val 50000 --final-test 300000
```

Create the complete runtime template once, edit every `__REQUIRED_*__`
value, and leave all scientific row/graph/seed coordinates absent. Repeated
replica and split keys are already frozen in the template.

The Stage-J runtime bindings are source inputs, not manually prepared HOSD
outputs. For `scale_input_prepare`, bind:

- one authenticated RETB offline-cache directory whose
  `offline_input_manifest.json` has logical role `scale_train`;
- exactly four authenticated RETB HLT-v3 cache directories, replicas 0--3,
  each with role `scale_train`, policy `R_MULTI`, and profile `D_NOMINAL`.

All five must bind the campaign's exact `scale_train_manifest` hash. The DAG
then materializes the label-blind HOSD scale views, builds all five scale tree
caches, refits separate offline and shared-HLT relation/REGION statistics,
re-trains and locks both scale teachers, and compiles their adapter configs.
It also builds the four scale native-relation aggregates exactly once before
graph training. Do not place those future `scale_up/` paths in the runtime
config.

The other required Stage-J bindings are existing source evidence: the
compiled tree resource/backend manifest/backend binary; normalization,
relation-registry, raw-schema, and HLT-profile contracts; both teacher model
contracts; the five-way offline manifest and validation partition; the
screening/relation registries and global determinism artifact; the existing
`val_stop`/`val_design` tree root; scale/validation label files; the four
fixed validation HLT caches used by graph evaluation; and the efficiency and
stack inference inputs. The generated template names every one explicitly.
For the miniature, use the corresponding miniature identity populations;
never point a miniature placeholder at production-scale identities.

```bash
mkdir -p "${HOSD_BOOTSTRAP}"
python -s scripts/write_hosd_runtime_config_template.py \
  --output "${MINI_RUNTIME_CONFIG}.template"
cp "${MINI_RUNTIME_CONFIG}.template" "${MINI_RUNTIME_CONFIG}"
# Edit MINI_RUNTIME_CONFIG. The storage node's probe input must be a real,
# label-blind miniature offline input NPZ; it measures every current physical
# target itself. Do not edit generated row or seed options.
```

## 2. Build and submit the real miniature

Add every available inherited parent with repeated
`--inherited-parent PARENT_ID=PATH`. Missing registered parents are rebuilt by
the DAG.

```bash
python -s scripts/build_hosd_campaign.py \
  --parent-manifest "${MINI_PARENT_MANIFEST}" \
  --output-dir "${MINI_ROOT}" \
  --campaign-id hosd_real_miniature \
  --miniature

# Resolve any missing shared parents before compiling runtime paths. These
# wrappers wait for their child Slurm jobs and validate the outputs. If the
# group is already satisfied, the command publishes only its no-op receipt.
python -s scripts/build_hosd_shared_hlt_parents.py \
  --campaign-root "${MINI_ROOT}" --submit
python -s scripts/build_hosd_tree_parents.py \
  --campaign-root "${MINI_ROOT}" --submit
python -s scripts/fit_hosd_relation_normalizers.py \
  --campaign-root "${MINI_ROOT}" --submit
python -s scripts/lock_hosd_inherited_parents.py \
  --campaign-root "${MINI_ROOT}"

# Materialize exactly three offline views and the RETB-contract replica set:
# four model-train replicas and replica zero for each fixed validation role.
# Supply --hlt-cache-root when reusable HLT caches live outside the default
# shared-parent campaign.
python -s scripts/materialize_hosd_runtime_inputs.py \
  --campaign-root "${MINI_ROOT}"

# The parent controllers idempotently publish the shared RETB graph and
# Stage-A task manifests before submission. They submit the complete
# prerequisite arrays and route every Slurm log under the campaign's ignored
# job_ledgers/slurm/parent_controllers directory.

# Now replace every template value with the exact materialized input,
# tree-cache, normalizer, model-contract, label-manifest, and cache path.
# The target-array keys are model_train/val_stop/val_design for offline and
# model_train:0..3,val_stop:0,val_design:0 for HLT.
python -s scripts/prepare_hosd_execution.py \
  --campaign-root "${MINI_ROOT}" \
  --runtime-config "${MINI_RUNTIME_CONFIG}" \
  --profile miniature_test

export CAMPAIGN_ROOT="${MINI_ROOT}"
bash sbatch/submit_hosd_tigris_full.sh \
  --smoke-submit --attempt 1
```

## 3. Required target-shard and training-row recovery

Run the target interruption promptly after attempt 1. The helper waits for a
real running coordinate and records the exact cancellation.

```bash
python -s scripts/interrupt_hosd_miniature_coordinate.py \
  --campaign-root "${MINI_ROOT}" \
  --node canonical_target_build \
  --coordinate 0

python -s scripts/snapshot_hosd_scheduler.py \
  --campaign-root "${MINI_ROOT}" \
  --output "${MINI_ROOT}/job_ledgers/target_interrupt_monitor.json" \
  --cancel-recovery-closure

bash sbatch/submit_hosd_tigris_full.sh \
  --resume-submit \
  --monitor "${MINI_ROOT}/job_ledgers/target_interrupt_monitor.json" \
  --attempt 2

python -s scripts/interrupt_hosd_miniature_coordinate.py \
  --campaign-root "${MINI_ROOT}" \
  --node baseline_train \
  --coordinate 0

python -s scripts/snapshot_hosd_scheduler.py \
  --campaign-root "${MINI_ROOT}" \
  --output "${MINI_ROOT}/job_ledgers/training_interrupt_monitor.json" \
  --cancel-recovery-closure

bash sbatch/submit_hosd_tigris_full.sh \
  --resume-submit \
  --monitor "${MINI_ROOT}/job_ledgers/training_interrupt_monitor.json" \
  --attempt 3
```

Wait until attempt 3 and all inherited jobs are terminal, then capture and
verify the evidence. The verifier issues all nine receipts and writes
`miniature_acceptance.json`; it does not inspect performance to decide
whether continuation was allowed.

```bash
python -s scripts/capture_hosd_scheduler_evidence.py \
  --campaign-root "${MINI_ROOT}" \
  --target-monitor "${MINI_ROOT}/job_ledgers/target_interrupt_monitor.json" \
  --training-monitor "${MINI_ROOT}/job_ledgers/training_interrupt_monitor.json" \
  --output "${MINI_ROOT}/job_ledgers/miniature_scheduler_evidence.json"

python -s scripts/verify_hosd_miniature.py \
  --campaign-root "${MINI_ROOT}" \
  --scheduler-evidence "${MINI_ROOT}/job_ledgers/miniature_scheduler_evidence.json"
```

## 4. Derive authenticated production resources

No resource number is entered by hand. Slurm allocation/usage rows,
concurrency, checkpoint bytes, and export bytes are measured from the
miniature. The compiler accepts only that hashed evidence.

```bash
python -s scripts/measure_hosd_miniature_resources.py \
  --campaign-root "${MINI_ROOT}" \
  --scheduler-evidence "${MINI_ROOT}/job_ledgers/miniature_scheduler_evidence.json" \
  --output "${MINI_ROOT}/job_ledgers/miniature_resource_evidence.json"

python -s scripts/compile_hosd_resource_measurements.py \
  --campaign-root "${MINI_ROOT}" \
  --miniature-execution-plan "${MINI_ROOT}/job_ledgers/production_execution_plan.json" \
  --measurement-evidence "${MINI_ROOT}/job_ledgers/miniature_resource_evidence.json" \
  --output "${MINI_ROOT}/job_ledgers/production_resource_measurements.json"
```

## 5. Build, preflight, authorize, and submit production

Use the measured runtime storage artifact, not the miniature placeholder
bundled at campaign creation.

```bash
python -s scripts/build_hosd_campaign.py \
  --parent-manifest "${PROD_PARENT_MANIFEST}" \
  --output-dir "${PROD_ROOT}" \
  --campaign-id hosd_500k_scale3m \
  --storage-measurements "${RETB_STORAGE_MEASUREMENTS}"

# As for the miniature, resolve/reuse the production campaign's exact parent
# artifacts and materialize its own 500k validation-role views. Never reuse
# the miniature runtime config: its files bind the miniature identities.
python -s scripts/build_hosd_shared_hlt_parents.py \
  --campaign-root "${PROD_ROOT}" --submit
python -s scripts/build_hosd_tree_parents.py \
  --campaign-root "${PROD_ROOT}" --submit
python -s scripts/fit_hosd_relation_normalizers.py \
  --campaign-root "${PROD_ROOT}" --submit
python -s scripts/lock_hosd_inherited_parents.py \
  --campaign-root "${PROD_ROOT}"
python -s scripts/materialize_hosd_runtime_inputs.py \
  --campaign-root "${PROD_ROOT}"

python -s scripts/write_hosd_runtime_config_template.py \
  --output "${PROD_RUNTIME_CONFIG}.template"
cp "${PROD_RUNTIME_CONFIG}.template" "${PROD_RUNTIME_CONFIG}"
# Edit PROD_RUNTIME_CONFIG with the authenticated source bindings described
# in Section 1 and the measured scalar values. Stage-J derived inputs, trees,
# normalizers, teacher-adapter configs, targets, and graph loaders are
# generated inside the DAG and are intentionally absent here.

python -s scripts/prepare_hosd_execution.py \
  --campaign-root "${PROD_ROOT}" \
  --runtime-config "${PROD_RUNTIME_CONFIG}" \
  --profile production_500k_scale3m \
  --resource-measurements "${MINI_ROOT}/job_ledgers/production_resource_measurements.json"

srun --partition=tigris --gres=gpu:gh200:1 --cpus-per-task=4 --mem=32G \
  python -s scripts/preflight_hosd_resources.py \
  --campaign-root "${PROD_ROOT}" \
  --profile production_500k_scale3m \
  --require-cuda \
  --storage-measurements "${MINI_ROOT}/job_ledgers/runtime_storage_measurements.json" \
  --resource-measurements "${MINI_ROOT}/job_ledgers/production_resource_measurements.json" \
  --output "${PROD_ROOT}/job_ledgers/resource_preflight.json"

python -s scripts/authorize_hosd_campaign.py \
  --campaign-root "${PROD_ROOT}" \
  --mode full \
  --miniature-acceptance "${MINI_ROOT}/job_ledgers/miniature_acceptance.json" \
  --resource-preflight "${PROD_ROOT}/job_ledgers/resource_preflight.json"

export CAMPAIGN_ROOT="${PROD_ROOT}"
bash sbatch/submit_hosd_tigris_full.sh \
  --full-submit \
  --authorization "${PROD_ROOT}/job_ledgers/full_authorization.json" \
  --attempt 1
```

Scientific underperformance never fails a row, prunes a later row, or cancels
the DAG. Only source, lineage, data-integrity, runtime, or scheduler failures
block dependent work. Recover those failures with
`snapshot_hosd_scheduler.py` followed by `--resume-submit`; never create a
replacement scientific matrix.
