# Codex Handoff: Local Residual Field / Curriculum Distillation

Last updated: 2026-07-20

This file is meant to let a new Codex thread recover the project state without
requiring the full chat history.

## Current Focus

The active research direction is no longer the canonical jet-state token
campaign by itself. The current best path is:

```text
Local Particle Residual Fields
  -> oracle fields show a large tagging ceiling
  -> predicted fields have not reached that ceiling
  -> train predicted fields through tagger-aware curriculum distillation
```

The current source-of-truth plan is:

```text
teacher_logit_reco/LOCAL_RESIDUAL_FIELD_CURRICULUM_DISTILLATION_PLAN.md
```

The older but still important base plan is:

```text
teacher_logit_reco/LOCAL_PARTICLE_RESIDUAL_FIELD_PLAN.md
```

## Big Picture

The user has been trying to recover offline-particle tagging performance from
HLT-only runtime inputs.

Important observed pattern:

```text
HLT ParT baseline:                 about 77.6% high-data accuracy
Oracle local residual fields:      about 81.8% high-data accuracy
Offline-particle ParT reference:   about 84.1% model-val accuracy
Best deployable predicted fields:  only modestly above HLT after fusion
```

Interpretation:

```text
true residual fields contain useful information
but current predictors do not produce fields that the tagger can exploit enough
```

The new curriculum idea is to avoid asking the predictor to jump straight from
HLT-only to full oracle behavior. Instead, it trains through weaker responses
from fixed oracle consumers.

## Current Curriculum Plan

The latest plan intentionally uses fixed-consumer alpha responses:

```text
Ofull(HLT, alpha * true_fields)
Orobust_light(HLT, alpha * true_fields)
```

It does not treat separately trained alpha-scaled teachers as the primary
curriculum, because those teachers can rescale their first layers and recover
full information.

### Two-Stage Pilot

The pilot is explicitly two-stage.

Stage 1a:

```text
O0
Ofull
Orobust_light
D_alpha_eval_Ofull
D_alpha_eval_Orobust
P0
```

Then run a selector that writes:

```text
selected_consumer.json
```

Required fields:

```text
selected_consumer_id
selected_alpha_endpoint
selection_source
selection_reason
model_val_alpha_curve
stack_val_alpha_curve
```

Stage 1b may queue only after `selected_consumer.json` exists and validates:

```text
P2
P4
P7a
P7b
Q0
Q3
G0
```

Stage 1b jobs must record:

```text
selected_consumer_id
selected_alpha_endpoint
selected_consumer_source_report
selected_consumer_hash
paired_consumer_mode
```

Default mode requires all Stage 1b rows to agree on the selected consumer and
alpha endpoint. Paired-consumer mode is allowed only with:

```text
LOCAL_RESIDUAL_FIELD_CURRICULUM_CONFIRM_PAIRED_CONSUMERS=1
```

### Run Semantics

```text
A0 = true clean HLT ParT baseline
O0 = zero/blank-field oracle-consumer diagnostic, not the HLT floor
Ofull = full oracle consumer trained on true fields
Orobust_light = full oracle consumer trained with light field noise/dropout
D_alpha_eval_Ofull = fixed Ofull alpha sweep
D_alpha_eval_Orobust = fixed Orobust_light alpha sweep
P0 = current Huber-only predicted-field baseline
P2 = weak fixed selected-consumer alpha target
P4 = selected-consumer alpha curriculum
P7a = P4 + field Huber + gates, student initialized from A0
P7b = P4 + field Huber + gates, student initialized from selected consumer
Q0 = selected P7a recipe minus oracle-path KD
Q3 = selected consumer + selected alpha endpoint from epoch 0, no curriculum
G0 = A0 + best deployable P logit fusion
```

P7b must include an adaptation/reset schedule:

```text
load selected consumer into S_phi
reset or shrink residual-field input projection, default scale 0.1
initialize residual-field gate bias low, around sigmoid probability 0.1
residual-path warmup
upper gentle unfreeze
full gentle unfreeze only if validation is stable
```

## Remaining MD Review Notes

The curriculum MD is close to implementation-ready, but two small cleanup items
were identified in the latest review:

1. One hard-stop rule still references only `Ofull`.

   Search for this stale sentence:

   ```text
   If fixed Ofull alpha=0.25 and alpha=0.75 do not improve over alpha=0.0,
   stop the first-stage pilot after P0.
   ```

   It should instead say:

   ```text
   If neither D_alpha_eval_Ofull nor D_alpha_eval_Orobust shows alpha=0.25
   or alpha=0.75 improving over alpha=0.0, stop after P0.
   ```

2. Selector discipline should be model-val primary, stack-val confirmatory.

   It is acceptable to inspect stack-val before final-test, but the safer
   selection policy is:

   ```text
   choose selected_consumer_id primarily from model_val
   use stack_val as stability/confirmation
   fit fusion on stack_train
   choose fusion on stack_val
   run final_test once
   ```

## Current Working Tree State

At the time this handoff was written, the working tree was dirty. Do not assume
the current implementation is committed or fully reviewed.

Modified files:

```text
sbatch/run_adaptive_binary_runtime_acceptance.sh
sbatch/run_train_local_residual_field_tagger.sh
scripts/train_local_residual_field_tagger.py
teacher_logit_reco/LOCAL_RESIDUAL_FIELD_CURRICULUM_DISTILLATION_PLAN.md
teacher_logit_reco/local_particle_residual_field/__init__.py
teacher_logit_reco/local_particle_residual_field/fusion.py
teacher_logit_reco/local_particle_residual_field/tagger.py
teacher_logit_reco/local_particle_residual_field/tagger_train.py
tests/test_adaptive_binary_pseudooffline_step12_slurm.py
tests/test_local_particle_residual_field_step5_tagger.py
tests/test_local_particle_residual_field_step8_sbatch.py
```

Untracked files:

```text
sbatch/run_cache_local_residual_oracle_teacher_logits.sh
scripts/register_local_residual_oracle_teacher.py
teacher_logit_reco/local_particle_residual_field/curriculum.py
teacher_logit_reco/local_particle_residual_field/oracle.py
tests/test_local_residual_field_curriculum_step2.py
tests/test_local_residual_field_curriculum_step3.py
tests/test_local_residual_field_curriculum_step4.py
tests/test_local_residual_field_curriculum_step5.py
tests/test_local_residual_field_curriculum_step6.py
```

Before claiming the curriculum implementation is done, inspect these files and
run targeted tests.

## Current Partial Curriculum Implementation

Some curriculum code exists already:

```text
teacher_logit_reco/local_particle_residual_field/curriculum.py
teacher_logit_reco/local_particle_residual_field/oracle.py
scripts/register_local_residual_oracle_teacher.py
sbatch/run_cache_local_residual_oracle_teacher_logits.sh
```

Important objects visible from source inspection:

```text
LOCAL_RESIDUAL_FIELD_CURRICULUM_JOINT_CONTRACT
LOCAL_RESIDUAL_FIELD_CURRICULUM_SCHEDULER_CONTRACT
LOCAL_RESIDUAL_FIELD_SELECTED_CONSUMER_CONTRACT
SelectedConsumerRecord
load_selected_consumer_record
LocalResidualFieldCurriculumSchedulerConfig
LocalResidualFieldCurriculumScheduler
LocalResidualFieldCurriculumJointConfig
LocalResidualFieldCurriculumJointModel
FrozenOracleConsumerConfig
FrozenLocalResidualFieldOracleConsumer
```

Known intended safety behavior:

```text
oracle teacher is training-only
deployable checkpoint payload must omit oracle consumer
Stage 1b scheduler must require selected_consumer.json unless paired mode is confirmed
frozen oracle parameters must not receive gradients
predicted fields must receive gradients through the frozen oracle forward
```

Do not assume the Slurm submitter side is complete. The plan still calls for:

```text
submit_lprf_curriculum_pilot.sh
submit_lprf_curriculum_tigris_pilot.sh
submit_lprf_curriculum_highdata.sh
submit_lprf_curriculum_tigris_highdata.sh
```

## Cluster / Environment Notes

The user is running primarily on RIT Tigris now, not sporcsubmit.

Known Tigris paths:

```text
PROJECT_DIR=/home/ryreu/atlas/Fresh_check
PD10_DATA_DIR=/home/ryreu/atlas/PracticeTagging/data
OUTPUT_ROOT=/home/ryreu/atlas/Fresh_check/checkpoints
CONDA_BASE=/home/ryreu/miniforge3-aarch64
CONDA_ENV=atlas_kd_tigris
DEVICE=cuda
```

Tigris account detail:

```text
correct account string appears to be: reu-aisocial
bad truncated string: reu-aisoc
rc-onboard worked in tests but the user said they are not supposed to use it
```

The submitters should support an account env var rather than hardcoding the
wrong account.

## Critical Python / Uproot Issue On Tigris

The user hit repeated ROOT read failures like:

```text
OverflowError: Python integer ... out of bounds for uint8
Failed to read JetClass ROOT file ... HToBB_010.root
```

Diagnosis:

The activated conda env was still importing user-site `uproot 5.1.2` from:

```text
/home/ryreu/.local/lib/python3.10/site-packages/uproot
```

When forcing no user site, conda `uproot 5.7.5` worked:

```text
PYTHONNOUSERSITE=1 python - <<'PY'
import pathlib, uproot
p = pathlib.Path("/home/ryreu/atlas/PracticeTagging/data/jetclass_part1/HToBB_010.root")
print("uproot:", uproot.__version__, uproot.__file__)
print("file size:", p.stat().st_size)
with uproot.open(p) as f:
    print("keys:", f.keys())
    print("entries:", f["tree"].num_entries)
PY
```

Expected successful output included:

```text
uproot: 5.7.5 .../envs/atlas_kd_tigris/.../site-packages/uproot
keys: ['tree;1']
entries: 100000
```

Therefore Tigris jobs should export:

```text
PYTHONNOUSERSITE=1
```

or otherwise ensure user-site packages do not override conda packages.

## Dataset Notes

The user deleted old checkpoint folders to save space and may need to rebuild
splits/caches.

The user uploaded `JetClass_Pythia_train_100M_part1.tar` and created/used:

```text
/home/ryreu/atlas/PracticeTagging/data/jetclass_part1
```

Earlier `jetclass_part0` became suspect after repeated ROOT read errors, but
the actual root cause was likely the user-site uproot override. Still verify
the exact data directory before queueing anything expensive.

Split sizes used by the local residual field campaign:

Pilot:

```text
model_train: 500k
model_val: 150k
stack_train: 300k
stack_val: 150k
final_test: 150k
```

High data:

```text
model_train: 5M
model_val: 1M
stack_train: 2M
stack_val: 1M
final_test: 1M
```

HLT profile:

```text
fixed_hlt_v2_realistic
HLT degradation strength: 2.5
max constituents: 128
10 JetClass classes
```

## Space / OOM History

Checkpoint folders became large, especially because of caches and `.pt` files.
The user has deleted checkpoint folders to recover space.

LPRF high-data target caching OOMed at least once. A later target job completed
after using more memory, but many downstream reconstructor/tagger jobs then
OOMed quickly. Inspect the current logs before requeueing; do not blindly
reuse stale failed output directories.

General policy:

```text
failed due to OOM: requeue once with larger memory or smaller batch
failed due to nonfinite validation: inspect before requeue
missing provenance/cache mismatch: fix inputs, do not requeue blindly
terminated before run_report: inspect curves, then requeue same run with OVERWRITE=1
```

## Earlier Campaigns And What They Taught Us

### AV10 Feature MLP Adapter

The AV10 feature MLP adapter was a strong earlier direction. It suggested that
small learned corrections to HLT features/embeddings can outperform plain HLT
ParT.

The shuffled-feature adapter result was unexpectedly strong, which shifted the
interpretation: some gains may come from capacity, regularization, or letting
the model reinterpret HLT tokens, not only from precise per-particle correction.

### PDV3 AV10 Adapter / Privileged Distillation

PDV3 combined AV10 adapter ideas with privileged distillation. It was patched
to support HLT v2 and to avoid final-test teacher-cache leakage. It may still
be running separately, but it is not the current active handoff focus.

Important prior fixes included:

```text
final-test path teacher-free by default
RepKD uses intended ParT jet-level representation
nonfinite guards
split/cache/baseline consistency checks
combined adapter variants
```

### Canonical Multi-Scale Jet State

Plan:

```text
teacher_logit_reco/CANONICAL_MULTI_SCALE_JET_STATE_PLAN.md
```

It created deterministic multi-scale jet-state tokens and residuals in a
particle-count-independent representation. It was implemented and reviewed
heavily.

Observed issue:

```text
Phi/state-only signals were much weaker than hoped
the tagger could mostly ignore the canonical state view
oracle diagnostics were interesting but deployable gains looked limited
```

The takeaway was not to abandon all residual ideas, but to move residual
information closer to the particle-level ParT input where the tagger has more
native capacity to use it.

### Local Particle Residual Field

Plan:

```text
teacher_logit_reco/LOCAL_PARTICLE_RESIDUAL_FIELD_PLAN.md
```

Core target:

For each HLT particle, predict local offline-derived residual fields describing
nearby missing/changed offline structure, without hallucinating individual
offline particles.

Field groups include:

```text
local pT / energy flow
local centroid / axis shift
local multiplicity
composition residuals
merge/split/reliability fields
```

The oracle fields worked well, but predicted fields did not close much of the
gap. This is why the new curriculum plan exists.

## What To Do Next

Recommended next actions for a fresh Codex session:

1. Read:

   ```text
   teacher_logit_reco/LOCAL_RESIDUAL_FIELD_CURRICULUM_DISTILLATION_PLAN.md
   teacher_logit_reco/CODEX_HANDOFF_LOCAL_RESIDUAL_FIELD.md
   ```

2. Patch the two small remaining MD review notes:

   ```text
   hard-stop rule references only Ofull
   selector should be model-val primary and stack-val confirmatory
   ```

3. Inspect current dirty implementation files before editing:

   ```text
   teacher_logit_reco/local_particle_residual_field/curriculum.py
   teacher_logit_reco/local_particle_residual_field/oracle.py
   teacher_logit_reco/local_particle_residual_field/tagger.py
   teacher_logit_reco/local_particle_residual_field/tagger_train.py
   scripts/train_local_residual_field_tagger.py
   sbatch/run_train_local_residual_field_tagger.sh
   ```

4. Run targeted tests if available:

   ```text
   python -m pytest tests/test_local_residual_field_curriculum_step2.py
   python -m pytest tests/test_local_residual_field_curriculum_step3.py
   python -m pytest tests/test_local_residual_field_curriculum_step4.py
   python -m pytest tests/test_local_residual_field_curriculum_step5.py
   python -m pytest tests/test_local_residual_field_curriculum_step6.py
   python -m pytest tests/test_local_particle_residual_field_step5_tagger.py
   python -m pytest tests/test_local_particle_residual_field_step8_sbatch.py
   ```

5. Implement/review Slurm support for the two-stage pilot:

   ```text
   stage1a
   select_consumer
   stage1b
   full_first_stage with afterok dependencies
   ```

6. Ensure Tigris submitters export:

   ```text
   PYTHONNOUSERSITE=1
   LOCAL_RESIDUAL_FIELD_SLURM_ACCOUNT=reu-aisocial
   ```

7. Before queueing, verify data and cache paths on Tigris.

8. Queue pilot before high data. Do not queue high data until the pilot report
   passes the gate.

## Things To Avoid

Do not:

```text
use O0 as the HLT baseline
mix oracle diagnostics into deployable leaderboards
use final-test oracle diagnostics for selection
queue Stage 1b with a guessed consumer
let user-site uproot override conda uproot on Tigris
requeue nonfinite-validation failures without inspection
delete current cache/target/checkpoint folders without confirming active jobs
```

## Useful Local Paths

Local Windows workspace:

```text
C:\Users\22rya\ComputerScience\CERN\Fresh_check
```

Downloaded logs and diagnostics:

```text
C:\Users\22rya\ComputerScience\CERN\a_download_checkpoints
C:\Users\22rya\ComputerScience\CERN\a_download_checkpoints\fresh_check_logs
C:\Users\22rya\ComputerScience\CERN\a_download_checkpoints\fresh_check_diagnostics
```

Tigris pull example used before:

```text
ssh ryreu@tigris.rc.rit.edu 'cd /home/ryreu/atlas/Fresh_check && tar -czf - fresh_check_logs fresh_check_diagnostics' | tar -xzf - -C /mnt/c/Users/22rya/ComputerScience/CERN/a_download_checkpoints/tigris_pull
```

## Final Current Judgment

The strongest current idea is not "predict residual fields more accurately" in
the abstract. It is:

```text
predict residual fields that move a deployable HLT-only tagger toward the
behavior of a fixed oracle field consumer, using a staged curriculum and
strict deployment/reporting gates
```

The plan is coherent and worth implementing carefully, but the implementation
should be treated as partial until the dirty working tree is reviewed, tests
pass, and the two-stage submitter/report path is in place.
