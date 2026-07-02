# Privileged Distillation 10-Class Plan

## Short Name

PD10, or Privileged Distillation 10-Class.

## Goal

Train an HLT-only 10-class JetClass tagger that beats a strong basic HLT ParT
at high training counts by using matched offline information only during
training.

The deployed model must consume only the pseudo-HLT view:

```text
inference:
  HLT particles -> student ParT -> 10-class logits
```

During training, the student may learn from teachers that had access to:

```text
HLT only
offline only
HLT + offline
```

The main question is:

```text
Can teacher soft targets built from offline privilege move an HLT-only ParT
closer to Offline ParT performance when the HLT student already has millions
of labeled jets?
```

## Why This Plan Exists

The strongest high-data evidence from privileged distillation does not say:

```text
reconstruct the privileged input exactly.
```

It more often says:

```text
use privileged information to build a better, smoother, lower-variance target
for the deployable-input student.
```

For this project, that means we should first test the simplest strong version:

```text
paired HLT/offline views
  -> train strong teachers
  -> cache teacher logits
  -> train HLT-only student with hard labels plus teacher soft targets
```

This is a better first high-data test than full offline-particle
reconstruction because the HLT view is intrinsically lossy. Dropped, merged, or
misassigned HLT constituents may not be exactly recoverable. A teacher target
can still transfer useful decision structure without forcing the student to
hallucinate every offline particle.

## Core Hypothesis

The HLT input cannot contain all offline information. But a trained HLT ParT
may still leave usable performance on the table because hard 10-class labels are
a sparse training signal.

Offline and dual-view teachers can expose:

```text
which classes are genuinely confused
which boundaries should move
which HLT mistakes are systematic
which class probabilities should be softened rather than forced one-hot
```

The highest-probability win is expected from a teacher that sees both the
deployable view and the privileged view:

```text
teacher(x, z) = teacher(HLT, offline)
student(x)    = student(HLT)
```

This mirrors privileged-feature distillation in ranking and recommendation:
the best teacher is often not the teacher that sees only privileged features,
but the teacher that sees both ordinary and privileged features. Such a teacher
is stronger than an HLT-only teacher while still anchored to what the HLT-only
student can plausibly imitate.

## Data Contract

Use the standard paired JetClass setup:

```text
offline:
  original high-quality JetClass/offline particle view

HLT:
  deterministic pseudo-HLT degraded view generated from the same jet
```

The paired identity must be preserved:

```text
HLT jet i and offline jet i describe the same physical JetClass jet.
```

The first serious test uses:

```text
model_train: 5,000,000 jets
model_val:   1,000,000 jets
final_test:  1,000,000 jets
```

This is intentionally high enough to test whether gains survive beyond a
small-data regime, but not so high that the first serious run becomes a
cluster-marathon.

No stack split is required for the first PD10 experiment because there is no
learned final fusion layer. If later work ensembles multiple distilled
students, add separate `stack_train` and `stack_val` splits then.

## Task

Full 10-class JetClass classification:

```text
QCD
Hbb
Hcc
Hgg
H4q
Hqql
Zqq
Wqq
Tbqq
Tbl
```

Every model in this plan must use the same:

```text
split manifest
label mapping
HLT cache
offline source files
JetClass class order
```

## Model Families

### 1. HLT ParT Baseline

This is the model to beat:

```text
HLT particles -> ParT -> 10-class logits
```

It trains only on HLT particles and hard labels.

It also provides the HLT teacher for self-distillation controls.

Required outputs:

```text
best_model_val.pt
run_report.json
model_val metrics
final_test metrics
prediction/logit caches for model_train/model_val/final_test
```

### 2. Offline ParT Reference Teacher

This teacher sees the privileged view:

```text
offline particles -> ParT -> 10-class logits
```

It is not deployable. It measures how much performance exists in the offline
view and supplies the offline-only distillation target.

Required outputs:

```text
best_model_val.pt
run_report.json
model_val metrics
final_test metrics
teacher logits for model_train/model_val/final_test
```

### 3. Dual-View Logit-Fusion Teacher

The first dual-view teacher should be deliberately auditable and cheap.

It consumes cached logits from the HLT and offline teachers:

```text
z_hlt = HLT teacher logits on HLT view
z_off = offline teacher logits on offline view
```

It then trains a small fusion model:

```text
[z_hlt, z_off, z_off - z_hlt, p_hlt, p_off, summary_features]
  -> dual-view teacher logits
```

Where:

```text
p_hlt = softmax(z_hlt)
p_off = softmax(z_off)
```

The summary features are:

```text
HLT entropy
offline entropy
HLT top1-top2 margin
offline top1-top2 margin
HLT max probability
offline max probability
HLT/offline KL divergence
whether HLT and offline top1 classes agree
```

The exact fusion architecture is:

```text
input_dim = 10 + 10 + 10 + 10 + 10 + 8 = 58

base_logits = 0.5 * (z_hlt + z_off)

delta_logits =
  LayerNorm(58)
  Linear(58, 128)
  GELU
  Dropout(0.05)
  Linear(128, 128)
  GELU
  Dropout(0.05)
  Linear(128, 10)

teacher_logits = base_logits + delta_logits
```

The final `Linear(128, 10)` is zero-initialized so the teacher begins as the
simple average of HLT and offline teacher logits.

Training objective:

```text
CE(teacher_logits, labels)
```

Selection:

```text
select by model_val mean cross entropy
tie-break by model_val accuracy
```

This teacher is not meant to be the final deployed model. Its job is to create
a privileged but HLT-anchored target for the HLT-only student.

Why logit-fusion first instead of particle-level dual-view cross-attention:

```text
1. It directly tests the privileged-feature distillation idea.
2. It is cheap enough for the 5M/1M/1M first run.
3. It avoids brittle dependence on Weaver ParT internals.
4. It gives a clear comparison against HLT-only and offline-only teachers.
5. If it works, a heavier particle-level dual-view teacher becomes a V2.
```

### 4. HLT-Only Student ParT

The deployed student always has the same inference contract:

```text
HLT particles -> ParT -> 10-class logits
```

The student is trained with:

```text
hard-label CE
+ optional teacher KD
```

Two initialization modes are required:

```text
scratch:
  random initialization, same architecture as the HLT ParT baseline

warm_start:
  initialize from the selected HLT ParT baseline checkpoint
```

Warm-start is the highest-likelihood deployment candidate. Scratch is the
science control that tells us whether distillation helps the full training
trajectory rather than only fine-tuning an already strong HLT model.

## Student Loss

For a batch with labels `y`, student logits `s`, and teacher logits `t`:

```text
L = (1 - alpha) * CE(s, y)
  + alpha * T^2 * KL(
      softmax(t / T),
      softmax(s / T)
    )
```

PyTorch implementation convention:

```text
teacher_prob = softmax(teacher_logits / T, dim=-1)
student_log_prob = log_softmax(student_logits / T, dim=-1)
kd_loss = kl_div(student_log_prob, teacher_prob, reduction="batchmean") * T * T
```

Default first-run settings:

```text
T = 2.0
alpha = 0.5
hard CE always enabled
```

KD weight warmup:

```text
scratch students:
  alpha ramps linearly from 0 to target alpha over the first 3 epochs

warm-start students:
  alpha ramps linearly from 0 to target alpha over the first 1 epoch
```

This imports the warmup lesson from high-scale privileged distillation: avoid
letting a teacher dominate before the student has stable class structure.

## Teacher Target Modes

### Full-Logit KD

Use all 10 teacher probabilities.

This is the default and should run for every teacher type.

### Top-K KD

For the first ablation, use:

```text
k = 3
```

For each jet:

```text
keep the teacher's top-3 classes
renormalize their probabilities to sum to 1
compute KD only over those classes
```

The hard-label CE still supervises all 10 classes, so top-k KD does not remove
class coverage. It only prevents tiny low-probability teacher logits from
injecting noise.

### Confidence-Weighted KD

Use a per-jet KD weight:

```text
confidence = max_c softmax(teacher_logits)_c
uniform_floor = 1 / 10
w = clamp((confidence - uniform_floor) / (1 - uniform_floor), 0, 1)
```

Then:

```text
L = CE_weight * CE + alpha * mean_i(w_i * KD_i)
```

This tests whether the teacher is most useful on jets where it has a decisive
privileged view.

Do not use final-test labels or final-test teacher correctness to define these
weights.

## First High-Data Experiment Matrix

### Mandatory Core Matrix

Run these eight student conditions:

```text
scratch CE only
scratch HLT-teacher KD
scratch offline-teacher KD
scratch dual-view-teacher KD

warm-start CE only
warm-start HLT-teacher KD
warm-start offline-teacher KD
warm-start dual-view-teacher KD
```

The most important comparisons are:

```text
warm-start dual-view KD
  vs warm-start CE only
  vs HLT ParT baseline

scratch dual-view KD
  vs scratch CE only
```

### Priority Ablations

Run these if the mandatory matrix is healthy:

```text
warm-start dual-view KD, T=4, alpha=0.5
warm-start dual-view KD, T=2, alpha=0.3
warm-start dual-view KD, top-3 KD
warm-start dual-view KD, confidence-weighted KD
```

Do not sweep every setting for every teacher in the first pass. If dual-view KD
wins, sweep around that winning setup.

## Training Schedules

The exact optimizer should reuse the repo's standard ParT training defaults
where possible. PD10 only fixes the differences needed for fair comparison.

### Teacher Training

HLT and offline ParT teachers:

```text
train split: model_train
selection split: model_val
selection metric: accuracy
tie-breaker: mean cross entropy
epochs: same as current high-data HLT ParT baseline recipe
```

Dual-view logit-fusion teacher:

```text
train split: model_train
selection split: model_val
selection metric: mean cross entropy
tie-breaker: accuracy
epochs: 20
optimizer: AdamW
lr: 1e-3
weight_decay: 1e-4
batch size: as large as CPU/GPU memory allows for cached logits
early stopping patience: 4 epochs
```

The dual-view teacher is cheap because it trains on cached logits, not raw
particles.

### Scratch Student Training

```text
init: random
train split: model_train
selection split: model_val
epochs: 20
selection metric: accuracy
tie-breaker: mean cross entropy
KD alpha warmup: 3 epochs
optimizer/LR: same as baseline HLT ParT scratch training
```

Scratch CE-only must use the same schedule as scratch KD students.

### Warm-Start Student Training

```text
init: selected HLT ParT baseline checkpoint
train split: model_train
selection split: model_val
epochs: 8
selection metric: accuracy
tie-breaker: mean cross entropy
KD alpha warmup: 1 epoch
optimizer: AdamW
lr: 1e-5 to 3e-5 for ParT parameters
weight_decay: same as baseline
```

Warm-start CE-only is required. Without it, a distilled warm-start win could be
explained by extra training rather than by privileged targets.

## Metrics

### Primary 10-Class Metrics

Report for every teacher and student:

```text
accuracy
macro per-class accuracy
mean cross entropy
per-class recall
per-class precision
confusion matrix
per-class one-vs-rest AUC, if available
macro one-vs-rest AUC, if available
expected calibration error, if available
```

### Binary Projection Metrics

Even though training is 10-class, report important binary projections:

```text
QCD vs Hgg
QCD vs Hbb
QCD vs Tbqq
QCD vs Wqq
QCD vs Zqq
```

For pair `(A, B)`:

```text
score = logit_B - logit_A
```

Report:

```text
AUC
FPR at 30% signal efficiency
FPR at 50% signal efficiency
background rejection at 50% signal efficiency
validation-threshold final-test FPR
```

Validation-threshold final-test FPR means:

```text
choose threshold on model_val at the target signal efficiency
apply that fixed threshold once to final_test
```

### Gap Closure

Report HLT-to-offline gap closure.

For metrics where higher is better:

```text
gap_closure =
  (student_metric - HLT_baseline_metric)
  / (offline_reference_metric - HLT_baseline_metric)
```

For FPR where lower is better:

```text
gap_closure =
  (HLT_baseline_FPR - student_FPR)
  / (HLT_baseline_FPR - offline_reference_FPR)
```

This tells us whether a 0.2% accuracy gain is tiny or actually a meaningful
fraction of the offline gap.

## Cache Format

Teacher logit caches should be written for:

```text
model_train
model_val
final_test
```

At minimum:

```text
labels: [N]
logits: [N, 10]
probabilities: [N, 10]
preds: [N]
indices or jet ids: [N]
split name
label names
teacher name
source checkpoint path
source checkpoint hash
split manifest hash
HLT cache hash, when applicable
offline source/hash, when applicable
allowed inputs
```

Allowed input metadata should be explicit:

```text
HLT teacher:
  allowed_inputs = HLT_only

offline teacher:
  allowed_inputs = offline_only_train_time_privileged

dual-view teacher:
  allowed_inputs = HLT_plus_offline_train_time_privileged

student:
  allowed_inputs = HLT_only
```

The student checkpoint must not require offline caches or teacher checkpoints
for inference.

## Leakage Rules

No final-test information may influence:

```text
teacher checkpoint selection
dual-view fusion teacher selection
student checkpoint selection
KD hyperparameter selection
top-k/confidence ablation selection
```

Final-test teacher logits may be computed only for:

```text
teacher reference metrics
gap-closure analysis
final report diagnostics
```

They must not be used to train or select students.

## Expected Outcomes

### Strong Positive

```text
warm-start dual-view KD beats:
  HLT ParT baseline
  warm-start CE-only
  warm-start HLT KD
  warm-start offline KD
```

Interpretation:

```text
HLT+offline privileged teacher targets transfer useful decision structure that
is not explained by extra training or generic self-distillation.
```

### Useful Positive

```text
offline KD beats HLT KD
dual-view KD ties offline KD
```

Interpretation:

```text
offline teacher signal is useful and learnable, but the simple dual-view
logit-fusion teacher may not yet improve over offline-only targets.
```

Next move:

```text
build a heavier particle-level dual-view teacher or tune fusion/calibration.
```

### Self-Distillation Positive

```text
HLT KD beats CE-only
offline/dual-view KD do not beat HLT KD
```

Interpretation:

```text
KD helps as regularization, but privileged information is not yet transferring.
```

Next move:

```text
test teacher confidence filtering, top-k KD, or student architectures with
more HLT-side capacity.
```

### Warm-Start Only Positive

```text
warm-start dual-view KD wins
scratch dual-view KD does not
```

Interpretation:

```text
privileged targets are most useful as high-data fine-tuning signals for an
already strong HLT ParT.
```

This is still deployable and valuable.

### Scratch Only Positive

```text
scratch dual-view KD wins
warm-start dual-view KD ties warm-start CE-only
```

Interpretation:

```text
KD improves the training trajectory or representation learning, but a fully
trained HLT ParT may already sit near the warm-start local optimum.
```

Next move:

```text
run data-scaling curves and consider training baseline longer or with the same
schedule.
```

### Negative

```text
no KD variant beats matched CE-only controls
```

Interpretation:

```text
either HLT ParT already extracts the learnable HLT information, the teachers
are not student-compatible, or the student needs a different architecture to
use the target.
```

Next move:

```text
combine privileged KD with architecture-view or local-compression residual ParT.
```

## Why This Might Improve Over HLT ParT

HLT ParT trained with hard labels learns from one-hot class targets:

```text
this jet is Hgg
```

A privileged teacher can provide richer structure:

```text
this jet is Hgg, but it has Wqq-like and QCD-like ambiguity
this QCD jet is suspiciously Hgg-like offline but not enough to cross boundary
this Hbb/Hcc confusion should remain soft
this top-like class pair should separate differently
```

At high data counts, the gain is not expected to come from label scarcity alone.
The plausible mechanisms are:

```text
variance reduction:
  teacher probabilities are smoother than hard labels

student-compatible privilege:
  dual-view teacher uses offline information while remaining anchored to HLT

better class geometry:
  10-class soft targets encode inter-class similarity

calibration:
  KD often improves probability quality and cross-entropy

optimization:
  warm-start KD fine-tuning can move a strong HLT model toward a better basin
```

The method cannot create information that is absent from HLT at inference. It
can only improve how the HLT-only function is learned.

That is exactly the right high-data question:

```text
Does privileged training improve the practical HLT-only model, even when the
student has millions of HLT-labeled jets?
```

## V2 Extensions If PD10 Works

Do not start here. Add these only after the core matrix is measured.

### Particle-Level Dual-View Teacher

Replace the logit-fusion teacher with:

```text
HLT ParT branch
offline ParT branch
cross-attention or pooled-context fusion
10-class teacher logits
```

This tests whether raw paired particle views contain teacher information beyond
what independent HLT/offline logits capture.

### Distilled Residual Student

Keep the HLT ParT baseline fixed or warm-started and train:

```text
student_logits = hlt_part_logits + gate(HLT) * residual_logits(HLT)
```

Use dual-view teacher KD on the final logits.

This asks whether privileged targets are best expressed as corrections to the
baseline rather than a full student fine-tune.

### Architecture-View Student With KD

Use the architecture-view residual ParT as the student:

```text
HLT particles
  -> PN/PFN/PCNN latent view
  -> residual injection into ParT
  -> logits
```

Train with:

```text
CE + dual-view KD
```

This is the natural bridge between PD10 and the AV10 ensemble plan.

### Local-Compression Student With KD

Use local-compression feature adapter ParT as the student:

```text
HLT particles
  -> local modality compression
  -> residual delta_F
  -> exact HLT ParT
```

Train with:

```text
CE + dual-view KD
```

This tests whether privileged teacher targets need a better HLT-side
architecture to become useful.

## Implementation Steps

### Step 1: Add PD10 Config And Naming Layer

Create a new package or module namespace:

```text
teacher_logit_reco/privileged_distill_10class/
```

Define:

```text
label_names = full 10-class JetClass list
split sizes = 5M / 1M / 1M
teacher names = hlt_part, offline_part, dual_view_logit
student init modes = scratch, warm_start
student KD modes = none, hlt, offline, dual_view
target modes = full_logits, top3, confidence_weighted
default T = 2.0
default alpha = 0.5
```

Tests:

```text
label order validation
variant-name generation
split-size config validation
student/teacher compatibility checks
```

### Step 2: Build Or Register 5M/1M/1M Paired Splits

Add or parameterize split/cache runners for:

```text
model_train = 5,000,000
model_val   = 1,000,000
final_test  = 1,000,000
```

Audit:

```text
split sizes
class balance
disjoint jet identities
paired HLT/offline identity alignment
HLT degradation metadata
cache/source hashes
```

### Step 3: Train Or Register HLT And Offline ParT Teachers

Train or register:

```text
hlt_part_teacher_10class
offline_part_teacher_10class
```

Both use:

```text
model_train for training
model_val for checkpoint selection
final_test only after selection
```

Store:

```text
best_model_val.pt
run_report.json
model_val_report.json
final_test_report.json
source_metadata.json
```

### Step 4: Cache HLT And Offline Teacher Logits

Implement:

```text
scripts/cache_pd10_teacher_logits.py
```

Write logits for:

```text
model_train
model_val
final_test
```

For both:

```text
hlt_part_teacher
offline_part_teacher
```

Tests:

```text
shape [N, 10]
labels and jet ids aligned
metadata hashes present
HLT teacher never loads offline
offline teacher never uses HLT
```

### Step 5: Train Dual-View Logit-Fusion Teacher

Implement:

```text
teacher_logit_reco/privileged_distill_10class/dual_view_teacher.py
scripts/train_pd10_dual_view_logit_teacher.py
```

Inputs:

```text
cached HLT teacher logits
cached offline teacher logits
labels
jet ids
```

Output:

```text
dual-view teacher checkpoint
dual-view teacher logits for model_train/model_val/final_test
run_report.json
```

Tests:

```text
feature builder correctness
zero-init delta means initial output equals 0.5*(HLT+offline) logits
training improves or preserves model_val CE on a toy case
row alignment refused on mismatch
```

### Step 6: Implement Distillation Dataset

Add a student dataset that returns:

```text
HLT particles
mask
label
teacher_logits, optional
jet id
```

The dataset must never return offline particles to the student model.

Tests:

```text
teacher logits align with HLT rows
missing teacher logits fail loudly
inference/export path does not require teacher logits
```

### Step 7: Implement Student KD Training Runner

Implement:

```text
scripts/train_pd10_student.py
```

Required args:

```text
--student-init scratch|warm_start
--teacher-target none|hlt|offline|dual_view
--target-mode full_logits|top3|confidence_weighted
--temperature
--kd-alpha
--hlt-cache-dir
--teacher-logit-cache
--baseline-checkpoint, for warm_start
--output-dir
--confirm-final-test
```

Training report must include:

```text
init mode
teacher target
target mode
temperature
kd alpha
KD warmup epochs
baseline checkpoint hash, if warm-started
best epoch
selection metric
model_val metrics
final_test metrics
```

### Step 8: Add PD10 Report Writer

Implement:

```text
scripts/write_pd10_report.py
teacher_logit_reco/privileged_distill_10class/reports.py
```

Report sections:

```text
teacher metrics
student core matrix
warm-start comparisons
scratch comparisons
teacher-target comparison
top-k/confidence ablations
binary projection table
gap-closure table
calibration table
leakage/audit summary
```

The report should answer:

```text
Did any student beat HLT ParT?
Did dual-view KD beat HLT self-KD?
Did dual-view KD beat offline-only KD?
Did warm-start KD beat warm-start CE-only?
Did scratch KD beat scratch CE-only?
How much of the offline gap closed?
Which class pairs improved?
```

### Step 9: Add Slurm Runners

Add:

```text
sbatch/run_pd10_cache_teacher_logits.sh
sbatch/run_pd10_train_dual_view_teacher.sh
sbatch/run_pd10_train_student.sh
sbatch/run_pd10_write_report.sh
sbatch/submit_pd10_experiment.sh
```

The submitter dependency graph:

```text
splits/cache
  -> HLT/offline teachers
  -> cache HLT/offline logits
  -> train/cache dual-view teacher
  -> student matrix
  -> final report
```

The submitter should support:

```text
DRY_RUN=1
SKIP_EXISTING=1
CONFIRM_FINAL_TEST=1
```

### Step 10: Smoke Test

Before the 5M run, run:

```text
model_train = 20k
model_val   = 5k
final_test  = 10k
epochs      = 2
```

Smoke goals:

```text
all scripts run
teacher logits align
student KD loss finite
warm-start loads correctly
final report writes
no offline tensors appear in student inference checkpoint
```

Do not interpret smoke-test performance.

### Step 11: First Serious 5M/1M/1M Run

Run the mandatory core matrix:

```text
scratch CE only
scratch HLT KD
scratch offline KD
scratch dual-view KD
warm-start CE only
warm-start HLT KD
warm-start offline KD
warm-start dual-view KD
```

Then run priority dual-view ablations if the core jobs are healthy:

```text
warm-start dual-view T=4
warm-start dual-view alpha=0.3
warm-start dual-view top3
warm-start dual-view confidence-weighted
```

### Step 12: Decide Next Branch

If dual-view KD wins:

```text
promote to scaling curves and V2 particle-level dual-view teacher
```

If offline KD wins but dual-view does not:

```text
improve dual-view teacher or teacher calibration
```

If HLT KD wins but privileged KD does not:

```text
focus on distillation mechanics and confidence/top-k filtering
```

If no KD wins:

```text
combine privileged KD with AV10 or local-compression residual students
```

