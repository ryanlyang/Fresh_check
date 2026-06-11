# Cross-Architecture Teacher-Logit Reconstructor Ensemble Plan

This document defines the fresh 500k/150k/500k cross-architecture experiment.
It is intentionally separate from the older same-HLT reco7, V2 original
mechanism, heterogeneous HLT4, and individual teacher-logit reconstructor
runners.

The research question is:

```text
Can architecture diversity in both reconstructors and frozen offline teachers
recover HLT-only tagging signal that one strong HLT Particle Transformer cannot?
```

## Current Status

- Step 1 complete: `crossarch_experiment.py` defines the fresh experiment
  namespace, split sizes, source naming, 16 reco-teacher grid, 4 HLT baselines,
  required/optional fusion groups, fuser list, and output layout helpers.
- Step 2 complete: fresh cross-architecture split/cache/audit runners write
  into `checkpoints/teacher_logit_reco_crossarch_500k`, and
  `audit_crossarch_step2_splits_hlt_cache.py` verifies exact counts,
  class balance, disjoint jet identities, fixed-HLT seeds, fixed-HLT params,
  manifest hashes, and cache content hashes.
- Step 3 complete: `train_or_register_crossarch_offline_teacher.py` and
  `run_crossarch_train_offline_teacher.sh` train or register the four frozen
  offline teacher architectures under the fresh cross-arch namespace, with
  source metadata and checkpoint hashes recorded.
- Step 4 complete: fresh direct-HLT baseline training and prediction runners
  train `hlt_part`, `hlt_pn`, `hlt_pfn`, and `hlt_pcnn` on the cross-arch
  fixed-HLT cache and write fusion-compatible prediction blocks for
  `stack_train`, `stack_val`, and `final_test`.

The experiment trains:

```text
16 teacher-logit reconstructor models
= 4 reconstructor families x 4 frozen offline teacher tagger families
```

and a direct HLT baseline control:

```text
4 direct HLT taggers
= part, pn, pfn, pcnn trained directly on fixed-HLT jets
```

Then it evaluates multiple fusion groups and multiple fuser families from a
fresh, auditable prediction-block namespace.

## Core Inference Contract

For every teacher-logit reconstructor model, inference is:

```text
fixed HLT jet
-> trained reconstructor
-> reconstructed soft view
-> same frozen offline teacher tagger used during reconstructor training
-> logits / probabilities
```

Example:

```text
pn_reco_to_pfn_teacher
= fixed HLT -> ParticleNet-style reconstructor -> frozen PFN offline teacher
```

The frozen teacher remains in the inference path.  It is called an offline
teacher because it was trained on offline/full-quality views.  During
reconstructor inference it receives only the HLT-derived reconstructed soft
view.

This matters for deployment interpretation:

```text
direct HLT tagger:
  fixed HLT -> HLT tagger -> logits

teacher-logit reco model:
  fixed HLT -> reconstructor -> frozen offline teacher -> logits
```

Both are HLT-only at inference with respect to input data, but the
teacher-logit reco model contains two neural modules.

## Splits

Use the same fixed five-way split machinery and leakage rules as the previous
JetClass protocol.

Target sizes:

```text
model_train = 500,000
model_val   = 150,000
stack_train = 500,000
stack_val   = 150,000
final_test  = 500,000
```

Every model in the experiment must use the exact same split manifest and seeds:

```text
all 16 reconstructors:
  same model_train rows
  same model_val rows
  same stack_train rows
  same stack_val rows
  same final_test rows

all 4 direct HLT baselines:
  same model_train rows
  same model_val rows
  same stack_train rows
  same stack_val rows
  same final_test rows
```

The split names should retain the existing meanings:

```text
model_train:
  trains offline teachers, direct HLT baselines, and reconstructors

model_val:
  selects checkpoints and early stopping for all trained models

stack_train:
  trains fusion/stacking models only

stack_val:
  selects fusion hyperparameters, bins, gates, and regularization

final_test:
  final locked reporting only
```

For early exploration, using the same *sizes* for model and stack splits is OK.
Using the same *rows* would not be OK.  The manifest must contain disjoint
partitions.

## Model Families

### Reconstructor Families

Use the four teacher-logit reconstructor families implemented in this package:

```text
gt    = Global Transformer / ParT-ish reconstructor
pn    = ParticleNet-style reconstructor
pfn   = Particle Flow / PFN-style reconstructor
pcnn  = Particle-CNN-style reconstructor
```

For naming consistency:

```text
gt_reco_to_part_teacher
pn_reco_to_part_teacher
pfn_reco_to_part_teacher
pcnn_reco_to_part_teacher
...
```

### Frozen Offline Teacher Families

Use four offline-trained teacher tagger families:

```text
part
pn
pfn
pcnn
```

Expected teacher checkpoints:

```text
teacher_logit_crossarch_500k/offline_teachers/part/best_model_val.pt
teacher_logit_crossarch_500k/offline_teachers/pn/best_model_val.pt
teacher_logit_crossarch_500k/offline_teachers/pfn/best_model_val.pt
teacher_logit_crossarch_500k/offline_teachers/pcnn/best_model_val.pt
```

If some offline teachers already exist from a previous trusted run, the fresh
runner may accept explicit checkpoint overrides.  It must record the source
checkpoint path and hash in every report.

### Sixteen Reconstructor-Teacher Models

The full grid is:

```text
gt   -> part
gt   -> pn
gt   -> pfn
gt   -> pcnn

pn   -> part
pn   -> pn
pn   -> pfn
pn   -> pcnn

pfn  -> part
pfn  -> pn
pfn  -> pfn
pfn  -> pcnn

pcnn -> part
pcnn -> pn
pcnn -> pfn
pcnn -> pcnn
```

The diagonal pairs are not invalid, but they are scientifically interesting as
a control:

```text
gt -> part
pn -> pn
pfn -> pfn
pcnn -> pcnn
```

They test whether a reconstructor and teacher with similar inductive biases are
redundant or still useful.

### Direct HLT Baselines

Train four direct HLT taggers on the same fixed-HLT view:

```text
hlt_part
hlt_pn
hlt_pfn
hlt_pcnn
```

These are not reconstructors.  They establish what architecture diversity alone
can do without the offline-teacher reconstruction mechanism.

## Output Namespace

Use a fresh root:

```text
checkpoints/teacher_logit_reco_crossarch_500k
```

Recommended layout:

```text
teacher_logit_reco_crossarch_500k/
  split_manifest/
  hlt_cache/

  offline_teachers/
    part/
    pn/
    pfn/
    pcnn/

  hlt_baselines/
    part/
    pn/
    pfn/
    pcnn/

  reco_models/
    gt/part/
    gt/pn/
    ...
    pcnn/pcnn/

  predictions/
    hlt_part/
    hlt_pn/
    hlt_pfn/
    hlt_pcnn/
    gt_reco_to_part_teacher/
    ...
    pcnn_reco_to_pcnn_teacher/

  prediction_runs/
    hlt/
    reco/

  fusion/
    mean_logits/
    logistic/
    uncertainty/
    bin_gated/
    reports/

  audits/

  final_report/
```

Do not write into older roots such as:

```text
teacher_logit_reco_gt
teacher_logit_reco_pn
teacher_logit_reco_pfn
teacher_logit_reco_pcnn
jetclass_fresh_fusion
jetclass_hetero_hlt4_*
```

This experiment should be reproducible and inspectable without relying on old
output directories.

## Prediction Blocks

Every trained source writes fusion-compatible prediction blocks for:

```text
stack_train
stack_val
final_test
```

Each prediction block must include:

```text
model_name
model_kind
reconstructor_architecture, if applicable
teacher_architecture, if applicable
training_step
prediction_step
source checkpoint path
source checkpoint hash if available
split name
jet identity hash
label hash
logits
probabilities
metrics
allowed_inputs
```

Direct HLT models should use:

```text
allowed_inputs = cached_fixed_hlt_only
```

Teacher-logit reco models should use:

```text
allowed_inputs = cached_fixed_hlt_only_then_reconstructed_soft_view_to_frozen_teacher
```

## Fusion Groups

The initial required groups are:

### `all16`

All reconstructor-teacher prediction sources:

```text
all 16 reco models
```

Question:

```text
What is the maximum gain from all cross-architecture reco/teacher diversity?
```

### `cross12`

All off-diagonal reco-teacher sources:

```text
all16 minus:
  gt   -> part
  pn   -> pn
  pfn  -> pfn
  pcnn -> pcnn
```

Question:

```text
Does excluding same-bias reco/teacher pairs help or hurt?
```

### `part_teacher4`

All reconstructors targeting the ParT offline teacher:

```text
gt   -> part
pn   -> part
pfn  -> part
pcnn -> part
```

Question:

```text
If the teacher is held fixed at the strongest teacher, does reconstructor
architecture diversity help?
```

### `mixed4`

One intentionally cyclic mixed family:

```text
gt   -> pn
pn   -> pfn
pfn  -> pcnn
pcnn -> part
```

Question:

```text
Can a compact, nonredundant cross-bias set compete with the full ensemble?
```

### `hlt4`

Direct HLT baselines:

```text
hlt_part
hlt_pn
hlt_pfn
hlt_pcnn
```

Question:

```text
How much does ordinary architecture diversity on HLT-only inputs buy us?
```

### Recommended Optional Groups

These are not required for the first implementation, but the fusion code should
make them easy:

```text
all16_plus_hlt4
cross12_plus_hlt4
part_teacher4_plus_hlt_part
best_reco_per_teacher4
best_teacher_per_reco4
```

These help answer whether direct HLT logits remain complementary once the
reconstructor ensemble is present.

## Fuser Ladder

The fusion script should train several fuser families from the same prediction
blocks.  The point is not just to chase accuracy; it is to understand where the
diversity lives.

### F0: Mean Logit / Mean Probability Ensemble

No learned parameters:

```text
mean logits
mean probabilities
temperature-normalized mean logits, optional
```

Purpose:

```text
lowest-risk sanity baseline
```

If mean fusion is terrible, the logits may be badly calibrated or misaligned.

### F1: Regularized Multinomial Logistic Regression

Train a logistic stacker on:

```text
concatenated logits
concatenated probabilities
concatenated logits + probabilities
```

Select regularization on `stack_val`.

Purpose:

```text
main baseline for learned linear fusion
```

This is the anchor.  Every smarter fuser must beat this while passing controls.

### F2: Logistic Regression With Uncertainty/Diversity Features

Augment the source logits/probs with features such as:

```text
per-model entropy
per-model max probability
per-model top1-top2 margin
per-class mean logit across models
per-class std logit across models
model disagreement count
number of distinct predicted classes
teacher-family agreement flags
reconstructor-family agreement flags
raw HLT confidence features, when HLT models are in the group
```

Purpose:

```text
let a linear fuser learn when model disagreement itself is informative
```

This is often the best first "smarter than logistic" move because it remains
auditable and relatively hard to overfit compared with a neural gate.

### F3: Bin-Gated Logistic Regression

Train separate regularized logistic stackers in interpretable bins.  The first
implementation should support:

```text
entropy bins:
  based on anchor model entropy, usually hlt_part or mean ensemble entropy

confidence/margin bins:
  based on top1 probability or top1-top2 margin

predicted-class bins:
  one gate per predicted class from an anchor model

multiplicity bins:
  based on valid HLT constituent count

disagreement bins:
  based on number of distinct predicted classes or mean pairwise disagreement
```

Recommended first bins:

```text
entropy:
  low / medium / high using stack_train quantiles

margin:
  low / medium / high using stack_train quantiles

multiplicity:
  low / medium / high using stack_train quantiles

disagreement:
  0-1 distinct alternatives / moderate / high
```

Select bin definitions and logistic `C` on `stack_val`, not `final_test`.

Purpose:

```text
different reconstructors may help in different jet regimes; a global linear
stacker can average that away
```

### F4: Soft Learned Gating / Mixture Of Experts

Optional later fuser:

```text
small MLP gate over uncertainty/diversity/HLT summary features
weighted sum of source logits
```

This should not be the first trusted result.  It can overfit and is harder to
audit.  Implement only after F0-F3 and controls are stable.

## Recommended Initial Fusers

Implement these first:

```text
mean_logits
mean_probs
logistic_logits
logistic_probs
logistic_logits_probs
uncertainty_logistic_logits_probs
entropy_bin_gated_logistic
margin_bin_gated_logistic
multiplicity_bin_gated_logistic
disagreement_bin_gated_logistic
predicted_class_bin_gated_logistic
```

Each fusion report should identify:

```text
group name
fuser family
feature mode
selected hyperparameters
stack_train metrics
stack_val metrics
final_test metrics
per-class metrics where available
control metrics
```

## Controls And Audits

Every fuser family should have controls:

### Label Permutation Control

Train with labels permuted on `stack_train`.

Expected:

```text
stack_val/final_test collapse near chance
```

### Row-Shuffled Feature Control

Shuffle prediction rows independently from labels.

Expected:

```text
stack_val/final_test collapse near chance
```

### Source Alignment Audit

Before fitting any fuser, verify:

```text
same split names
same number of jets
same labels
same jet identity hash
same row ordering
no duplicated jet identities across split partitions
```

### Final-Test Guardrail

Any script touching `final_test` must require:

```text
--confirm-final-test
```

No hyperparameter choice should be made using `final_test`.

### Group Overlap Audit

Report the exact model names in every group and verify:

```text
all16 has 16 unique sources
cross12 has 12 unique sources
part_teacher4 has 4 unique sources
mixed4 has 4 unique sources
hlt4 has 4 unique sources
```

### Calibration Diagnostics

Save:

```text
accuracy
macro OVR AUC
cross entropy
ECE if available
mean confidence
entropy summaries
confusion matrix
per-class accuracy
```

## Expected Outcomes

Most likely:

```text
HLT4 gives a modest gain over HLT ParT.
Part-teacher reconstructors are strong but may be redundant.
Full all16 may beat any 4-source group, but not necessarily by much.
Bin-gated fusers may reveal where non-ParT reconstructors help.
```

Success does not require every reconstructor to beat raw HLT alone.  A
reconstructor is useful if its errors are complementary and fusion uses it.

Good signs:

```text
all16 > hlt4 on stack_val and final_test
cross12 close to or better than all16
part_teacher4 > best single part-teacher source
mixed4 surprisingly close to all16
bin-gated logistic > global logistic
controls collapse
```

Bad signs:

```text
all16 barely beats hlt_part
all fusers collapse to hlt_part-like behavior
row-shuffled controls do not collapse
bin-gated fusers improve stack_val but fail final_test
same-family diagonal dominates everything
```

## Implementation Principles

1. Fresh namespace only.

   Do not reuse old prediction/fusion directories.

2. Prediction-block first.

   Train every model, write prediction blocks, then fuse from blocks.  Fusion
   should never reload training data or model internals.

3. One manifest.

   All jobs consume the same split manifest and HLT cache.

4. Every output records provenance.

   Source commit, status hash, checkpoint paths, checkpoint metadata, split
   hash, HLT cache hash, and config should be saved.

5. Controls are first-class outputs.

   A higher score without passing controls is not a result.

6. Keep fusers interpretable before neural.

   Start with mean/logistic/uncertainty/bin-gated.  Add learned gates later.

## Implementation Steps

### Step 1: Write Fresh Experiment Config And Naming Layer

Add a new crossarch config module that defines:

```text
experiment root
split sizes
reconstructor architectures
teacher architectures
direct HLT architectures
16 model names
4 HLT baseline names
fusion group definitions
default fuser list
```

Expected file:

```text
teacher_logit_reco/crossarch_experiment.py
```

Tests:

```text
tests/test_teacher_logit_reco_crossarch_experiment.py
```

Done when group definitions produce exactly:

```text
all16 = 16
cross12 = 12
part_teacher4 = 4
mixed4 = 4
hlt4 = 4
```

### Step 2: Build Or Verify Fresh Splits And HLT Cache

Status: complete.

Implemented artifacts:

```text
sbatch/run_crossarch_build_splits.sh
sbatch/run_crossarch_build_hlt_cache.sh
sbatch/run_crossarch_audit_splits_hlt_cache.sh
scripts/audit_crossarch_step2_splits_hlt_cache.py
```

Fresh outputs:

```text
checkpoints/teacher_logit_reco_crossarch_500k/split_manifest/split_manifest.json.gz
checkpoints/teacher_logit_reco_crossarch_500k/hlt_cache/
checkpoints/teacher_logit_reco_crossarch_500k/audits/step2_splits_hlt_cache/
```

Add scripts/runners to build a fresh split manifest and fixed-HLT cache with:

```text
model_train = 500k
model_val   = 150k
stack_train = 500k
stack_val   = 150k
final_test  = 500k
```

If existing split/cache builders can be parameterized safely, reuse them.  The
output root must still be fresh.

Done when:

```text
split manifest exists
HLT cache exists for all five splits
audit report verifies disjoint splits and expected counts
```

### Step 3: Train Or Register Four Offline Teachers

Status: complete.

Implemented artifacts:

```text
teacher_logit_reco/crossarch_offline_teachers.py
scripts/train_or_register_crossarch_offline_teacher.py
sbatch/run_crossarch_train_offline_teacher.sh
sbatch/submit_crossarch_step3_offline_teachers.sh
```

Fresh outputs:

```text
checkpoints/teacher_logit_reco_crossarch_500k/offline_teachers/part/best_model_val.pt
checkpoints/teacher_logit_reco_crossarch_500k/offline_teachers/pn/best_model_val.pt
checkpoints/teacher_logit_reco_crossarch_500k/offline_teachers/pfn/best_model_val.pt
checkpoints/teacher_logit_reco_crossarch_500k/offline_teachers/pcnn/best_model_val.pt
```

Each teacher directory must contain:

```text
best_model_val.pt
run_report.json
model_val_report.json
source_metadata.json
config.json
```

The runner trains on offline `model_train` and selects on offline `model_val`.
If `CROSSARCH_<ARCH>_TEACHER_SOURCE_CHECKPOINT` is set, it registers that
trusted checkpoint instead of retraining and records the source/registered
checkpoint hashes.

Train or register:

```text
part offline teacher
pn offline teacher
pfn offline teacher
pcnn offline teacher
```

They must train only on offline `model_train` and select on offline
`model_val`.

Done when every teacher has:

```text
best_model_val.pt
run_report.json
model_val_report.json
source metadata
```

### Step 4: Train Four Direct HLT Baselines

Status: complete.

Implemented artifacts:

```text
teacher_logit_reco/crossarch_hlt_baselines.py
scripts/train_crossarch_hlt_baseline.py
scripts/predict_crossarch_hlt_baseline.py
sbatch/run_crossarch_train_hlt_baseline.sh
sbatch/run_crossarch_predict_hlt_baseline.sh
sbatch/submit_crossarch_step4_hlt_baselines.sh
```

Fresh model outputs:

```text
checkpoints/teacher_logit_reco_crossarch_500k/hlt_baselines/part/best_model_val.pt
checkpoints/teacher_logit_reco_crossarch_500k/hlt_baselines/pn/best_model_val.pt
checkpoints/teacher_logit_reco_crossarch_500k/hlt_baselines/pfn/best_model_val.pt
checkpoints/teacher_logit_reco_crossarch_500k/hlt_baselines/pcnn/best_model_val.pt
```

Fresh prediction-block outputs:

```text
checkpoints/teacher_logit_reco_crossarch_500k/predictions/hlt_part/
checkpoints/teacher_logit_reco_crossarch_500k/predictions/hlt_pn/
checkpoints/teacher_logit_reco_crossarch_500k/predictions/hlt_pfn/
checkpoints/teacher_logit_reco_crossarch_500k/predictions/hlt_pcnn/
```

Each prediction source contains:

```text
stack_train_predictions.npz
stack_train_predictions_metadata.json
stack_val_predictions.npz
stack_val_predictions_metadata.json
final_test_predictions.npz
final_test_predictions_metadata.json
```

The training runner consumes only cached fixed-HLT `model_train` and
`model_val`.  The prediction runner consumes only cached fixed-HLT
`stack_train`, `stack_val`, and `final_test`; final-test access is guarded by
`--confirm-final-test`.

Train:

```text
hlt_part
hlt_pn
hlt_pfn
hlt_pcnn
```

using fixed-HLT `model_train` and `model_val`.

Then write prediction blocks for:

```text
stack_train
stack_val
final_test
```

Done when `hlt4` prediction blocks are fusion-compatible.

### Step 5: Train The Sixteen Teacher-Logit Reconstructors

Status: complete.

Implemented artifacts:

```text
teacher_logit_reco/crossarch_reconstructors.py
sbatch/run_crossarch_train_reconstructor.sh
sbatch/submit_crossarch_step5_reconstructors.sh
```

Fresh model outputs:

```text
checkpoints/teacher_logit_reco_crossarch_500k/reco_models/gt/part/
checkpoints/teacher_logit_reco_crossarch_500k/reco_models/gt/pn/
...
checkpoints/teacher_logit_reco_crossarch_500k/reco_models/pcnn/pcnn/
```

Each model directory is expected to contain:

```text
best_model_val.pt
last.pt
training_curves.json
model_val_report.json
run_report.json
slurm_run_config.json
```

For every `(reco_arch, teacher_arch)` pair:

```text
fixed HLT -> reco_arch reconstructor -> frozen teacher_arch teacher
offline view -> frozen teacher_arch teacher
```

Train on `model_train`, select on `model_val`.

Recommended Slurm strategy:

```text
16 independent GPU jobs
```

Done when every model has:

```text
best_model_val.pt
training_curves.json
run_report.json
```

### Step 6: Generate Prediction Blocks For All Sources

Status: complete.

Implemented artifacts:

```text
teacher_logit_reco/crossarch_predictions.py
sbatch/run_crossarch_predict_reconstructor.sh
sbatch/submit_crossarch_step6_predictions.sh
```

Fresh prediction-block outputs:

```text
checkpoints/teacher_logit_reco_crossarch_500k/predictions/hlt_part/
checkpoints/teacher_logit_reco_crossarch_500k/predictions/hlt_pn/
checkpoints/teacher_logit_reco_crossarch_500k/predictions/hlt_pfn/
checkpoints/teacher_logit_reco_crossarch_500k/predictions/hlt_pcnn/
checkpoints/teacher_logit_reco_crossarch_500k/predictions/gt_reco_to_part_teacher/
...
checkpoints/teacher_logit_reco_crossarch_500k/predictions/pcnn_reco_to_pcnn_teacher/
```

The Step 6 submitter queues the 16 reconstructor prediction jobs and, by
default, the 4 direct-HLT prediction jobs.  If Step 4 already wrote the HLT
prediction blocks, set:

```text
CROSSARCH_STEP6_SUBMIT_HLT_PREDICTIONS=0
```

or:

```text
CROSSARCH_STEP6_SKIP_EXISTING_PREDICTIONS=1
```

For every trained reconstructor source and every HLT baseline source, write:

```text
stack_train predictions
stack_val predictions
final_test predictions
```

Recommended Slurm strategy:

```text
20 independent prediction jobs
```

Done when the prediction namespace contains:

```text
16 reco source directories
4 HLT baseline source directories
```

### Step 7: Implement Fresh Fusion Feature Builder

Status: complete.

Implemented artifacts:

```text
teacher_logit_reco/crossarch_fusion.py
scripts/run_crossarch_fusion.py
```

Fresh feature-builder outputs:

```text
checkpoints/teacher_logit_reco_crossarch_500k/fusion/feature_builder/feature_config.json
checkpoints/teacher_logit_reco_crossarch_500k/fusion/feature_builder/feature_build_report.json
checkpoints/teacher_logit_reco_crossarch_500k/fusion/feature_builder/features/<group>/<split>_feature_metadata.json
```

By default, Step 7 writes metadata, alignment summaries, feature names, matrix
shapes, uncertainty/diversity feature summaries, and train-quantile bin specs.
Full feature matrices can also be persisted with:

```text
--write-feature-matrices
```

No fuser is trained in this step.

Build a new fusion feature module that loads prediction blocks and creates:

```text
logits features
probability features
logits+probability features
uncertainty/diversity features
bin assignment features
```

Expected files:

```text
teacher_logit_reco/crossarch_fusion.py
scripts/run_crossarch_fusion.py
```

Done when it can load groups and produce aligned feature matrices without
fitting.

### Step 8: Implement F0-F3 Fusers

Implement:

```text
mean_logits
mean_probs
logistic_logits
logistic_probs
logistic_logits_probs
uncertainty_logistic_logits_probs
entropy_bin_gated_logistic
margin_bin_gated_logistic
multiplicity_bin_gated_logistic
disagreement_bin_gated_logistic
predicted_class_bin_gated_logistic
```

All fusers must train on `stack_train`, select on `stack_val`, and only then
score `final_test`.

Done when every fusion group has a report for every fuser.

Status: implemented.

Artifacts:

```text
teacher_logit_reco/crossarch_fusion.py
scripts/run_crossarch_fusion.py --fit-fusers
tests/test_crossarch_fusion.py
```

Notes:

```text
mean_logits and mean_probs are deterministic zero-fit fusers.
logistic_* fusers train on stack_train and select C on stack_val.
bin-gated fusers train per-bin stackers from stack_train bins, select on stack_val,
and fall back to the global stacker when a bin is too small.
multiplicity_bin_gated_logistic is reported as skipped until prediction blocks
carry row-wise HLT constituent multiplicity.
```

### Step 9: Add Controls And Audits

Add:

```text
label permutation controls
row-shuffled feature controls
source alignment audits
group size audits
split leakage audits
final_test guardrail audit
```

Done when each fusion report includes controls and an `ok` field.

Status: implemented.

Artifacts:

```text
teacher_logit_reco/crossarch_fusion.py
scripts/run_crossarch_fusion.py --fit-fusers
tests/test_crossarch_fusion.py
```

Report additions:

```text
ok
audit_summary
controls_summary
suspicious_flags
groups.<group>.ok
groups.<group>.audits.source_alignment
groups.<group>.audits.group_size
groups.<group>.audits.split_leakage
groups.<group>.controls.mode_reports
```

Controls:

```text
label permutation controls train only on stack_train with permuted labels
row-shuffled controls train only on independently shuffled stack_train feature columns
stack_val remains the regularization-selection split
final_test remains locked until after selection
```

### Step 10: Slurm Submitter For Full Experiment

Write fresh Slurm runners and one submitter:

```text
build splits/cache
train/register offline teachers
train HLT baselines
train 16 reconstructors
predict all 20 sources
run fusion groups/fusers/audits
write final report
```

The dependency graph should be:

```text
splits/cache
  -> offline teachers
  -> HLT baselines
  -> 16 reconstructors
  -> all prediction jobs
  -> fusion/audits
  -> final report
```

Done when `DRY_RUN=1` prints all expected jobs and paths.

Status: implemented.

Artifacts:

```text
sbatch/run_crossarch_fusion.sh
sbatch/run_crossarch_write_final_report.sh
sbatch/submit_crossarch_full_experiment.sh
scripts/write_crossarch_final_report.py
tests/test_sbatch_scripts.py
```

Submitter graph:

```text
run_crossarch_build_splits.sh
  -> run_crossarch_build_hlt_cache.sh
  -> run_crossarch_audit_splits_hlt_cache.sh
  -> 4x run_crossarch_train_offline_teacher.sh
  -> 4x run_crossarch_train_hlt_baseline.sh
  -> 4x run_crossarch_predict_hlt_baseline.sh
  -> 16x run_crossarch_train_reconstructor.sh
  -> 16x run_crossarch_predict_reconstructor.sh
  -> run_crossarch_fusion.sh
  -> run_crossarch_write_final_report.sh
```

The fusion runner uses Step 8/9 fusers, controls, and audits through
`scripts/run_crossarch_fusion.py --fit-fusers`.  Optional plus-HLT fusion
groups are enabled by default for the full run.

### Step 11: Final Report Writer

Write one report that summarizes:

```text
single-source metrics
HLT4 fusion
all16 fusion
cross12 fusion
part_teacher4 fusion
mixed4 fusion
optional plus-HLT groups
best fuser by stack_val
final_test results for selected fusers
control status
class-level effects
source correlation/diversity diagnostics
```

Done when the final report can be copied off the research compute and inspected
without loading model checkpoints.
