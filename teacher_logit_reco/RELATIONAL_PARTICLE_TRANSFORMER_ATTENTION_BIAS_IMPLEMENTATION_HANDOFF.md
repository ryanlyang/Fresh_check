# Relational Particle Transformer Implementation Handoff

This is the onboarding document for the agent that will implement:

```text
teacher_logit_reco/RELATIONAL_PARTICLE_TRANSFORMER_ATTENTION_BIAS_PLAN.md
```

The plan is the scientific and deterministic source of truth. This handoff
does not replace it. Its purpose is to explain the repository, identify the
existing partial implementation, point to the trusted HLT Particle Transformer
and data code, and document how this project is run on Tigris.

## 1. Mission

Implement the relational Particle Transformer campaign without restarting the
existing work.

The campaign is deliberately independent of the residual-field,
prediction-anchored bridge, particle-view, oracle, and knowledge-distillation
campaigns. It trains Particle Transformer variants from random initialization
using only fixed-HLT inputs and ordinary class labels.

The scientific question is:

> Can explicit, physically motivated particle-pair relations improve a matched
> HLT-only Particle Transformer?

The primary comparison is always against the exact matched from-scratch
`RPT_BASE`, not against an old checkpoint from another campaign.

## 2. What the implementing agent must do first

Before editing:

1. Read this handoff completely.
2. Read
   `teacher_logit_reco/RELATIONAL_PARTICLE_TRANSFORMER_ATTENTION_BIAS_PLAN.md`
   completely, especially Sections 4-22 and the eight implementation steps in
   Section 24.
3. Run `git status --short --untracked-files=all`.
4. Inspect every existing file under
   `teacher_logit_reco/relational_part/`, plus
   `scripts/build_relational_part_campaign.py` and
   `tests/test_relational_particle_transformer_step1.py`.
5. Run the existing Step-1 test before changing Step 1.
6. Implement in the plan's Step 2 through Step 8 order, adding focused tests
   with each step.

Do not rewrite Step 1 merely to impose a different coding style. Extend its
contracts and hashes carefully.

## 3. Current repository snapshot

At handoff creation:

```text
local repository:
  C:\Users\22rya\ComputerScience\CERN\Fresh_check

branch:
  master

HEAD:
  647786aef56bbcc19bd6efbb9790702fcbc16119
```

This information is only a snapshot. Recheck it rather than assuming it is
still current.

The worktree is intentionally dirty. In particular, the relational plan,
Step-1 package, CLI, and test are currently untracked. There is also a large
unrelated untracked particle-view implementation. Preserve all of it.

Relational files already present:

```text
teacher_logit_reco/RELATIONAL_PARTICLE_TRANSFORMER_ATTENTION_BIAS_PLAN.md
teacher_logit_reco/relational_part/__init__.py
teacher_logit_reco/relational_part/ca_tree.py
teacher_logit_reco/relational_part/campaign.py
teacher_logit_reco/relational_part/contracts.py
teacher_logit_reco/relational_part/normalization.py
teacher_logit_reco/relational_part/provenance.py
teacher_logit_reco/relational_part/registry.py
teacher_logit_reco/relational_part/splits.py
teacher_logit_reco/relational_part/storage.py
scripts/build_relational_part_campaign.py
tests/test_relational_particle_transformer_step1.py
```

These files are a real Step-1 implementation, not placeholders. They already
define:

- the production and miniature split contracts;
- exact HLT expectations and cache authentication;
- content-hashed campaign artifacts;
- the relation-family registry;
- the fixed 21-row screening registry;
- confirmation-only architecture templates;
- configuration roles and selectability;
- semantic-control registry entries;
- the normalization contract;
- compact angular-tree resource and backend contracts;
- source snapshot and artifact-layout contracts;
- evidence-bound storage measurements and projection;
- immutable Step-1 bundle publication.

Local verification at handoff creation:

```text
.\.venv\Scripts\python.exe -m pytest -q \
  tests/test_relational_particle_transformer_step1.py

14 passed
```

The system `python` on this Windows machine did not have NumPy. Use the
repository `.venv` for local tests unless the environment has changed.

Step 1 is implemented and tested locally, but it has not thereby proved that a
production Tigris campaign was built. It also remains untracked in the current
snapshot. Inspect and preserve it, then continue with Step 2.

Important version-control consequence: untracked local files do not appear on
Tigris after `git pull`. Before any cluster run, the relational implementation
must exist in the remote branch that Tigris pulls. Do not assume this happened
merely because local tests passed.

## 4. Locked scientific contract

The exact details and formulas live in the plan. The following is only an
orientation summary.

### Data

```text
model_train  = 1,000,000  -> train
model_val    =   125,000  -> val_stop/checkpoint selection
stack_train  =         0  -> unused and must not be cached
stack_val    =   125,000  -> val_select/architecture comparison
final_test   =   500,000  -> sealed final evaluation
```

All nonempty splits are exactly balanced across the ten JetClass classes and
are disjoint.

The HLT contract is:

```text
profile                  = fixed_hlt_v1
degradation strength     = 0.6
maximum constituents     = 128
model_train HLT seed     = 1053
model_val HLT seed       = 1054
stack_val HLT seed       = 1056
final_test HLT seed      = 1057
```

Do not substitute the prediction-anchored campaign's
`fixed_hlt_v2_realistic` profile or strength `2.5`.

### Model

The matched reference is:

```text
input dimension          = 17
particle embed dims      = [128, 512, 128]
pair embed dims          = [64, 64, 64]
attention heads          = 8
particle blocks          = 8
class-attention blocks   = 2
standard pair features   = Weaver's exact four features
activation               = GELU
initialization           = from scratch
```

Every scientific row trains from scratch. There is no pretrained HLT
checkpoint, warm start, oracle, teacher, KD target, or residual field.

### Relations

Canonical order:

```text
base4, PT, TRACK, PID, CHARGE, DENSITY, REGION
```

Registered family shapes:

```text
PT       10 raw continuous pair features -> 8 encoded
TRACK    7 node features, Siamese endpoint encoder,
         17 explicit pair channels, 81 pair inputs -> 12 encoded
PID      6 endpoint categories, 36 directional pair states -> 8 encoded
CHARGE   9 directional states, 12 encoder inputs -> 6 encoded
DENSITY  22 node descriptors, 66 pair inputs -> 12 encoded
REGION   41 angular-tree pair channels -> 12 encoded
```

The global numerical constant is `epsilon = 1e-6`.

All relations must be:

- HLT-only;
- query/context directional where declared;
- padding safe;
- finite in mixed precision;
- permutation equivariant;
- zero after learned encoding for invalid pairs;
- generated per batch rather than persisted as dense `N x N` caches.

Only the compact `REGION` angular-tree sidecar is persisted.

### Training and selection

The plan locks unweighted ten-class CE, AdamW, warm-up plus cosine decay,
40 maximum epochs, BF16 on GH200, effective batch size 128, and deterministic
per-seed ordering.

Checkpoint selection is not the old trainer's ordinary "better than current"
logic. It must:

1. find the global maximum `model_val` accuracy over completed epochs;
2. retain every epoch within `0.0001` of that maximum;
3. select the lowest cross entropy;
4. select the earliest epoch on an exact serialized float64 tie.

Early-stopping patience resets only when the newly completed epoch becomes the
globally preferred checkpoint under that rule. `stack_val` evaluates the
already selected checkpoint and never selects an epoch.

Screening uses seed `101`. Confirmation uses matched seeds `101`, `202`, and
`303`.

Poor performance is a scientific result, not an execution failure. A losing
variant must complete and remain reportable. Only invalid inputs, stale
provenance, missing artifacts, or nonfinite execution should fail a job.

### Final-test seal

Before model training, a sealed preparation job may construct and hash the
fixed-HLT and compact tree resources for `final_test`. It may not:

- load a checkpoint;
- run inference;
- calculate label-dependent metrics;
- fit normalization;
- write model-selection summaries.

No scientific worker may consume `final_test` until immutable
`locked_finalists.json` exists. Final-test results are for reporting and cannot
trigger another selection.

## 5. Canonical JetClass data implementation

### Raw data and split logic

Read:

```text
jetclass_fresh/jetclass_data.py
scripts/build_jetclass_splits.py
```

`jetclass_fresh/jetclass_data.py` owns:

- the ten-class order;
- filename-prefix-to-label mapping;
- stable `JetIdentity`;
- five-split manifests;
- balanced, disjoint deterministic sampling;
- recursive ROOT discovery;
- chunked offline ROOT loading;
- manifest hashing and audits.

Class order:

```text
QCD, Hbb, Hcc, Hgg, H4q, Hqql, Zqq, Wqq, Tbqq, Tbl
```

Raw cached token layout:

```text
0   pt
1   eta
2   phi
3   energy
4   charge
5   isChargedHadron
6   isNeutralHadron
7   isPhoton
8   isElectron
9   isMuon
10  d0val
11  d0err
12  dzval
13  dzerr
```

The ROOT loader reads `part_px`, `part_py`, `part_pz`, `part_energy`, charge,
five PID branches, and four track branches, then constructs the 14-channel raw
representation. The ROOT tree name is `tree`.

The production data root on Tigris is:

```text
/home/ryreu/atlas/PracticeTagging/data
```

Use the parent data root, not only `jetclass_part0`. File discovery is
recursive and needs enough events from the available `jetclass_part*`
subdirectories for all ten balanced classes.

The generic split builder already supports the required zero-size
`stack_train`:

```bash
python -u scripts/build_jetclass_splits.py \
  --data-dir /home/ryreu/atlas/PracticeTagging/data \
  --out <campaign-root>/inputs/split_manifest.json.gz \
  --model-train 1000000 \
  --model-val 125000 \
  --stack-train 0 \
  --stack-val 125000 \
  --final-test 500000 \
  --max-constits 128 \
  --pretty
```

The dedicated relational worker still needs to enforce the plan's cheap raw
schema audit before expensive HLT or tree work.

### HLT generation and cache

Read:

```text
jetclass_fixed_hlt.py
jetclass_fresh/hlt_cache.py
scripts/build_fixed_hlt_cache.py
tests/test_hlt_cache.py
```

Do not reimplement the corruption. Use:

```text
scaled_fixed_hlt_params(0.6)
build_fixed_hlt_view(...)
generate_and_cache_hlt_split(...)
load_cached_hlt_view(..., verify_hash=True)
audit_hlt_cache(...)
```

Each cached split has:

```text
<split>_fixed_hlt.npz
<split>_fixed_hlt_metadata.json
```

The metadata binds generator parameters, seed, source manifest, ordered jet
identities, source content, HLT content, and diagnostics.

For this campaign, cache only:

```text
model_train model_val stack_val final_test
```

There must be no `stack_train` HLT cache. Offline arrays may be read
transiently to generate HLT, but this campaign must not persist an offline
cache.

Useful worker precedents:

```text
sbatch/run_pd10_build_splits.sh
sbatch/run_pd10_build_hlt_cache.sh
sbatch/run_build_fresh_splits.sh
sbatch/run_build_fresh_hlt_cache.sh
```

The PD10 workers are the better production-scale pattern, but their scientific
sizes/profile are not this campaign's contract. Create dedicated relational
workers rather than inheriting their environment variables blindly.

### Canonical Particle Transformer inputs

Read:

```text
jetclass_fresh/part_inputs.py
tests/test_part_inputs.py
particle_transformer/data/JetClass/JetClass_full.yaml
```

The conversion from raw HLT tokens produces:

```text
points           [B, 2, N]
features         [B, 17, N]
lorentz_vectors  [B, 4, N]
mask             [B, 1, N]
```

`part_inputs.py` recomputes the jet axis only from the supplied HLT view. It
also inserts a deterministic epsilon-energy placeholder for a rare all-empty
HLT jet so Weaver attention does not see an entirely masked row.

The relational batch path needs both:

- these canonical 17 ParT features and four-vectors;
- the aligned raw 14-channel HLT tokens needed for PID, charge, track, density,
  and tree relations.

Preserve their exact row and constituent alignment.

## 6. Canonical HLT Particle Transformer implementation

Read:

```text
jetclass_fresh/hlt_baseline.py
scripts/train_hlt_baseline.py
sbatch/run_train_fresh_hlt_baseline.sh
particle_transformer/networks/example_ParticleTransformer.py
```

`jetclass_fresh/hlt_baseline.py` is the local canonical wrapper and config.
Its current forward path is:

```python
self.mod(features, v=lorentz_vectors, mask=mask)
```

That causes Weaver to construct its ordinary four pair features internally.
Step 2 must add a campaign-owned explicit `uu` path and prove it is
indistinguishable from the reference call.

The repository does not vendor Weaver's core
`weaver.nn.model.ParticleTransformer` implementation. It is installed in the
Tigris `atlas_kd_tigris` environment. Therefore:

- do not guess the installed `pairwise_lv_fts` signature or `uu` layout;
- inspect the installed version on Tigris;
- bind relevant version/source information into provenance;
- run the real Weaver parity test on Tigris even if local CI skips it.

Useful Tigris inspection:

```bash
cd /home/ryreu/atlas/Fresh_check
export PYTHONNOUSERSITE=1
source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate atlas_kd_tigris

python - <<'PY'
import importlib
import inspect

module = importlib.import_module("weaver.nn.model.ParticleTransformer")
print("module:", module.__file__)
print("ParticleTransformer.forward:", inspect.signature(module.ParticleTransformer.forward))
print("pairwise_lv_fts:", inspect.signature(module.pairwise_lv_fts))
PY
```

Step-2 parity must cover:

- logits;
- gradients;
- state dictionary structure;
- padding behavior;
- all-invalid/forced-nonempty and one-particle cases;
- explicit standard-four pair generation versus Weaver's internal path.

Useful guides to Weaver internals:

```text
teacher_logit_reco/adaptive_binary_pseudooffline/tagger.py
  WeaverStagewiseParticleTransformer

teacher_logit_reco/local_graph_part/edge_features.py
teacher_logit_reco/subtoken_part/pairwise.py
```

The stagewise adapter shows the installed ParT module surfaces:

```text
embed
pair_embed
blocks
cls_token
cls_blocks
norm
fc
```

It is especially useful when implementing the later layerwise-bias and
edge-value paths. The older pairwise modules are reusable numerical and
masking references, not the scientific definition of the new relations.

The old `train_hlt_baseline` loop is useful scaffolding for datasets,
dataloaders, finite checks, and checkpoint payloads. It does not implement the
new plan's exact BF16, warm-up/cosine, global checkpoint window, patience,
resume, `val_select`, or provenance contracts. Do not reuse it unchanged as
the campaign trainer.

The pretrained files under `particle_transformer/models/` are not inputs to
this campaign.

## 7. Implementation map

The plan defines eight equal steps.

### Step 1: existing

Existing package contracts, splits, registries, artifact layout, normalization
schema, tree resource schema, and measured storage projection are described in
Section 3 of this handoff.

Before extending it, keep the Step-1 test green and add any missing
cross-contract tests rather than changing hashes casually.

### Step 2: explicit standard-four pair path

Create the shared pair base and exact ParT wrapper:

```text
teacher_logit_reco/relational_part/pair_base.py
teacher_logit_reco/relational_part/model.py
```

The first goal is exact `RPT_BASE` parity, not new features.

### Step 3: normalization plus PT/PID/CHARGE

Create:

```text
teacher_logit_reco/relational_part/relation_pt.py
teacher_logit_reco/relational_part/relation_pid_charge.py
teacher_logit_reco/relational_part/pair_builder.py
```

Extend the existing normalization contract with a real deterministic fitter.
Follow the plan's applicability masks, salted sampling, fixed-scale channels,
categorical handling, clipping, and immutable scaler artifact exactly.

### Step 4: TRACK and DENSITY

Create:

```text
teacher_logit_reco/relational_part/relation_track.py
teacher_logit_reco/relational_part/relation_density.py
```

The raw-versus-transformed displacement distinction is important:
`part_inputs.py` applies `tanh` to token-side d0/dz, but TRACK must use the raw
HLT displacement and uncertainty columns from the 14-channel cache.

Neutral or invalid particles must never become tracks because a placeholder
value happened to be zero.

### Step 5: REGION, combinations, and capacity controls

Create:

```text
teacher_logit_reco/relational_part/relation_region.py
teacher_logit_reco/relational_part/csrc/relational_ca_tree_v1.cpp
```

Extend `ca_tree.py` from its current contract-only role into backend build,
loading, ABI verification, sidecar, shard, and finalization support.

The REGION backend is a repository-owned CPU PyTorch `CppExtension`, not
FastJet production clustering. The exact contract includes:

- C++17 and OpenMP;
- `-O3`;
- `-fno-fast-math`;
- `-fno-associative-math`;
- `-ffp-contract=off`;
- float64 tree arithmetic and topology;
- float32 persisted continuous node values;
- deterministic canonical leaf and merge tie rules;
- a backend manifest, source/binary hashes, ABI check, and self-test;
- deterministic contiguous shards of at most 10,000 jets;
- atomic publication and hash-valid resume;
- a mandatory stratified 20,000-jet throughput probe;
- a 1,000-jet Python-reference parity subset.

Also implement the registered combinations, `RPT_BASE_WIDE_MAX`, and
`RPT_FULL_ZERO_REL`. Wide capacity matching is against active incremental
parameters, not total parameters.

### Step 6: training, evaluation, and attention extensions

Create:

```text
teacher_logit_reco/relational_part/train.py
teacher_logit_reco/relational_part/evaluation.py
scripts/train_relational_part.py
scripts/evaluate_relational_part.py
```

Implement exact deterministic training, resume, global checkpoint selection,
`val_select` evaluation, diagnostics, parameter/FLOP/memory profiling, and
bounded checkpoint retention.

Then implement:

```text
RPT_BASE_LAYERWISE
RPT_SELECTED_LAYERWISE
RPT_BASE_EDGEVALUE
RPT_SELECTED_EDGEVALUE
```

Use `WeaverStagewiseParticleTransformer` only as an implementation guide.
The relational campaign owns its model metadata and provenance.

### Step 7: selection, confirmation, semantic controls, final lock and report

Create:

```text
teacher_logit_reco/relational_part/selection.py
teacher_logit_reco/relational_part/semantic_controls.py
teacher_logit_reco/relational_part/reporting.py
```

Implement:

- deterministic role-filtered screening;
- confirmation registry resolution;
- matched three-seed aggregation;
- selected union;
- selected layerwise and edge-value rows;
- unary endpoint control;
- relation shuffle;
- wrong-event derangement;
- directional swap;
- immutable `locked_finalists.json`;
- sealed final-test evaluation;
- paired bootstrap statistics;
- JSON and Markdown reporting.

Only `scientific_finalist` rows can become the nominal relational winner.
Controls remain mandatory comparisons and final-test rows.

### Step 8: Tigris campaign

Create every CLI and worker listed in plan Section 21, plus:

```text
scripts/submit_relational_part_graph.py
sbatch/submit_relational_part_tigris_full.sh
```

The top-level submitter must support a complete non-mutating dry run, create a
new unique campaign root, publish an immutable graph/ledger, and queue the
whole workflow.

## 8. Artifact and provenance rules

Campaign root:

```text
/home/ryreu/atlas/Fresh_check/checkpoints/
  relational_particle_transformer/<campaign_id>/
```

The exact layout is in plan Section 19. Important top-level artifacts include:

```text
campaign_spec.json
backend/backend_manifest.json
backend/throughput_probe.json
inputs/split_manifest.json.gz
inputs/preconstruction_raw_input_audit.json
inputs/hlt_cache/
inputs/relation_normalization.json
inputs/relation_tree_cache/
inputs/postconstruction_input_audit.json
registry/relation_family_registry.json
registry/screening_registry.json
runs/<run_id>/seed_<seed>/
selection/screening_summary.json
selection/confirmation_registry.json
selection/confirmation_summary.json
selection/locked_finalists.json
final_test/
reports/relational_part_report.json
reports/relational_part_report.md
job_ledgers/
```

Each scientific JSON artifact must have its contract/schema, canonical content
hash, parent hashes, and source snapshot. Loaders must reject stale,
cross-campaign, symlinked, malformed, or mismatched inputs.

Checkpoint registration must bind:

- model state;
- exact run registry row and role;
- split and HLT hashes;
- raw schema and normalization hashes;
- tree backend and split resources when REGION is enabled;
- seed and selected epoch;
- parameter/FLOP profile;
- explicit HLT-only inference;
- explicit absence of offline/teacher dependencies.

Do not use timestamps as scientific identity. They are audit metadata.

Storage policy:

- never persist dense pair matrices or attention matrices;
- keep compact REGION sidecars only;
- retain the best checkpoint;
- keep `last.pt` only while a job is resumable;
- keep final predictions only for locked finalists;
- use measured storage evidence and preserve the configured free-space reserve.

## 9. Tests and validation

The detailed required test list is plan Section 22. Add tests step by step,
preferably:

```text
tests/test_relational_particle_transformer_step1.py
tests/test_relational_particle_transformer_step2.py
...
tests/test_relational_particle_transformer_step8.py
```

Existing reusable test references:

```text
tests/test_hlt_baseline.py
tests/test_part_inputs.py
tests/test_hlt_cache.py
tests/test_jetclass_data.py
tests/test_local_graph_part_step5_classifier.py
tests/test_subtoken_part_step10_pairwise.py
tests/test_adaptive_binary_pseudooffline_step10_tagger.py
tests/test_prediction_anchored_bridge_step10.py
tests/test_prediction_anchored_bridge_bootstrap_submission.py
tests/test_sbatch_scripts.py
```

Use the local virtual environment:

```powershell
cd C:\Users\22rya\ComputerScience\CERN\Fresh_check
.\.venv\Scripts\python.exe -m pytest -q tests\test_relational_particle_transformer_step1.py
```

At appropriate milestones also run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_relational_particle_transformer_step*.py
git diff --check
```

Tests needing the real Weaver stack may skip locally, but they must be exercised
in `atlas_kd_tigris` before production submission. The Tigris environment has
historically not included `pytest`; check with:

```bash
python -m pytest --version
```

Do not confuse a missing cluster test runner with a model failure. If pytest is
absent, retain local tests and provide a small standalone parity/smoke CLI for
the installed Weaver environment rather than silently skipping production
parity.

The highest-risk tests are:

- exact Weaver external-pair logits and gradient parity;
- padding, all-empty, and one-particle safety;
- applicability-aware train-only normalization;
- TRACK validity and raw displacement use;
- average-tied rank permutation equivariance;
- REGION canonical ties, ABI, parity, resume, and atomic finalization;
- active incremental capacity matching;
- layerwise/edge-value zero-projection parity;
- global checkpoint-window and patience behavior;
- poor-performance campaigns continuing successfully;
- final-test seal and cross-campaign rejection;
- shell environment and dependency ordering.

## 10. Tigris environment and paths

Use:

```text
SSH host       = ryreu@tigris.rc.rit.edu
PROJECT_DIR    = /home/ryreu/atlas/Fresh_check
DATA_DIR       = /home/ryreu/atlas/PracticeTagging/data
OUTPUT_ROOT    = /home/ryreu/atlas/Fresh_check/checkpoints
LOG_DIR        = /home/ryreu/atlas/Fresh_check/fresh_check_logs
DIAGNOSTICS    = /home/ryreu/atlas/Fresh_check/fresh_check_diagnostics
CONDA_BASE     = /home/ryreu/miniforge3-aarch64
CONDA_ENV      = atlas_kd_tigris
account        = reu-aisocial
partition      = tigris
GPU GRES       = gpu:gh200:1
```

The account string is exactly `reu-aisocial`. Do not use the truncated
`reu-aisoc`.

Always:

```bash
export PYTHONNOUSERSITE=1
```

This avoids the known user-site `uproot` shadowing problem.

Correct interactive activation:

```bash
source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate atlas_kd_tigris
```

Do not run `/home/ryreu/miniconda3/bin/conda` directly. It has previously
resolved to an unusable wrapper. `weaver` is a Python package dependency, not
the name of the Tigris conda environment.

Plan defaults:

```text
GPU worker  = 16 CPUs, 220G RAM, gpu:gh200:1
CPU worker  = 16 CPUs, 192G RAM
```

The storage projection and throughput probe may justify adjusted worker
resources or bounded concurrency, but changing scientific formulas,
precision, or backend flags is not an operational optimization.

Check available space at submission time:

```bash
df -h "$HOME"
```

Do not rely on an old free-space number.

## 11. Slurm conventions

Reuse `sbatch/common.sh`. Important helpers include:

```text
fresh_setup
fresh_prepare_submitter
fresh_run
fresh_write_run_config
fresh_require_file
fresh_require_dir
fresh_claim_new_dir
fresh_refuse_existing_path
fresh_split_words
```

Every worker should:

```bash
set -euo pipefail
IFS=$'\n\t'

: "${PROJECT_DIR:=/home/ryreu/atlas/Fresh_check}"
: "${CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
: "${CONDA_ENV:=atlas_kd_tigris}"
export CONDA_BASE CONDA_ENV
export PYTHONNOUSERSITE=1

source "${PROJECT_DIR}/sbatch/common.sh"
fresh_setup "$@"
```

Set the Tigris conda values before sourcing/running common setup because
`sbatch/common.sh` retains legacy defaults such as `CONDA_ENV=weaver` and a
`jetclass_part0` data path.

Use:

```text
#SBATCH --output=fresh_check_logs/%x_%j.out
#SBATCH --error=fresh_check_logs/%x_%j.err
```

Call `fresh_write_run_config` for each material output directory. The common
diagnostics trap mirrors small text/JSON/CSV/Markdown files to
`fresh_check_diagnostics`; it intentionally does not mirror large `.pt`,
`.npz`, or ROOT artifacts.

Every variable needed in a later Slurm allocation must be:

- initialized inside that worker; or
- explicitly exported in the submit command.

Shell-local assignments from an earlier job do not propagate to a dependent
job. Use `--export=ALL,...` carefully. Avoid commas in exported path values
because Slurm parses the export list by comma.

Parse `sbatch --parsable` output robustly:

```bash
output="$(sbatch --parsable ...)"
job_id="${output%%;*}"
[[ "${job_id}" =~ ^[0-9]+$ ]]
```

Use a bounded screening array over the immutable 21-row registry, for example
`--array=0-20%<concurrency>`. Tree construction is a separate resumable shard
array followed by one atomic finalizer per split.

Recommended orchestration references:

```text
sbatch/submit_prediction_anchored_bridge_full_bootstrap.sh
scripts/submit_prediction_anchored_bridge_graph.py
sbatch/prediction_anchored_bridge_common.sh
sbatch/submit_canonical_state_experiment.sh
```

Reuse their graph, immutable-ledger, unique-root, job-ID parsing, explicit
export, and recovery patterns. Do not reuse their scientific stages.

## 12. Required production DAG

The dependency order is:

```text
split manifest
  -> cheap raw-input/schema audit
      -> fixed-HLT cache
      -> compiled REGION backend
          -> deterministic throughput/parity/storage probe
              -> resumable angular-tree shard arrays
                  -> per-split atomic tree finalizers

fixed-HLT cache + raw audit
  -> train-only relation normalization

HLT cache + normalization + backend + every tree finalizer
  -> full postconstruction audit
      -> 21-row seed-101 screening
          -> screening aggregation/selection
              -> confirmation registry and seed jobs
                  -> three-seed aggregation + semantic controls
                      -> locked_finalists.json
                          -> sealed final-test evaluation
                              -> final JSON/Markdown report
```

Use `afterok` for actual correctness/provenance dependencies. A valid model
that loses must exit zero and write its metrics. Scientific-performance
warnings must not create `DependencyNeverSatisfied`.

The top-level script eventually runs as:

```bash
cd /home/ryreu/atlas/Fresh_check
export PYTHONNOUSERSITE=1
bash sbatch/submit_relational_part_tigris_full.sh
```

Before real submission it must support a dry run that prints:

- campaign root;
- source commit and dirty-status hash;
- every resolved configuration;
- the entire dependency DAG;
- array sizes and concurrency;
- resource requests;
- expected artifacts;
- the job-ledger path;
- final-test seal dependencies.

## 13. Monitoring, recovery, and downloading results

Monitor:

```bash
squeue --me -o "%.18i %.9P %.30j %.8u %.14a %.2t %.10M %.6D %R"
```

Inspect history:

```bash
sacct -u "$USER" -S now-14days -X -n -P \
  -o JobID,JobName%80,State,ExitCode,Elapsed,End
```

The graph submitter should write a JSON ledger containing graph/spec hashes,
job IDs, submission commands, dependencies, and reused/completed nodes.

Recovery must distinguish:

- a currently live job, which can be bound as an existing job;
- a successfully completed artifact, which can be authenticated and marked
  completed without adding an obsolete Slurm dependency;
- a failed job, which must be rerun from the first missing/invalid artifact;
- a poor-performing but valid completed model, which must not be rerun merely
  because it lost.

Copy lightweight logs and diagnostics to the user's WSL/Windows download tree:

```bash
ssh ryreu@tigris.rc.rit.edu \
  'cd /home/ryreu/atlas/Fresh_check && tar -czf - fresh_check_logs fresh_check_diagnostics' \
  | tar -xzf - \
      --no-same-owner \
      --no-same-permissions \
      --touch \
      -C /mnt/c/Users/22rya/ComputerScience/CERN/a_download_checkpoints/tigris_pull
```

This does not download checkpoints, HLT caches, tree shards, or large
prediction NPZs. Use `scp` for a specific campaign report, selection artifact,
or requested metrics subtree.

## 14. Known failure modes to avoid

1. **Wrong environment**

   `weaver` is not the conda environment. Use `atlas_kd_tigris` under
   `miniforge3-aarch64`.

2. **User-site uproot shadowing**

   Export `PYTHONNOUSERSITE=1` in submitters and every worker.

3. **Wrong HLT profile**

   This campaign uses `fixed_hlt_v1` strength `0.6`, not HLT v2.

4. **Insufficient data root**

   Use `/home/ryreu/atlas/PracticeTagging/data`, which recursively exposes the
   available JetClass parts.

5. **Persisting forbidden data**

   Offline arrays are transient HLT-construction inputs only. No offline cache,
   teacher logits, or oracle artifact belongs in this campaign.

6. **Guessing Weaver internals**

   Inspect the installed source/signatures and prove explicit `uu` parity.

7. **Treating old helper code as the new contract**

   Reuse numerical utilities, not old feature definitions, registries, or
   hashes.

8. **Dense pairwise storage**

   Pair tensors are transient. Only compact `O(N)` REGION sidecars persist.

9. **Unbound Slurm variables**

   A variable initialized by one job is not visible in another. Initialize or
   export it in every allocation.

10. **Performance gates**

    A model losing to baseline is not a failed job. Continue to confirmation
    and reporting according to the plan.

11. **Premature final-test access**

    Preparation is allowed; inference and metrics require
    `locked_finalists.json`.

12. **Stale dependency recovery**

    Do not submit dependencies against failed or aged-out job IDs without
    authenticating artifacts and rebuilding the recovery graph.

13. **Overwriting campaign roots**

    Use unique roots and fail closed on conflicting immutable artifacts.

14. **Ignoring the dirty worktree**

    The existing relational implementation and unrelated particle-view work
    belong to the user. Do not reset, clean, or overwrite them.

## 15. Practical completion checklist

An implementation is not production-ready merely because modules import.
Before telling the user to submit the full campaign, verify:

- all eight plan steps have implementation surfaces and focused tests;
- all existing Step-1 tests remain green;
- the real installed Weaver explicit-pair parity test passes;
- every fixed screening row resolves and builds;
- the compiled backend builds in `atlas_kd_tigris`;
- backend self-test and ABI checks pass;
- the 20k probe and Python parity subset complete;
- a miniature end-to-end campaign completes even when all relations lose;
- the dry-run DAG contains every stage in the correct order;
- poor performance exits zero and remains reportable;
- storage projection uses measured evidence and current free space;
- final-test CLIs fail before the immutable lock;
- shell tests enforce the full account, conda root/env,
  `PYTHONNOUSERSITE=1`, and GH200 GRES;
- the implementation is committed/pushed before expecting Tigris to pull it;
- `git diff --check` passes.

The first implementation action after this handoff should be Step 2: inspect
the installed Weaver pair interface, build the explicit standard-four pair
path, and prove that `RPT_BASE` remains exactly the matched HLT ParT.
