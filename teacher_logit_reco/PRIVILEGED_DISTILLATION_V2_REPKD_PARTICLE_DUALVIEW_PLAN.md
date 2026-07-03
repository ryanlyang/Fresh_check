# Privileged Distillation V2: Representation KD And Particle Dual-View Teacher

## Short Name

PD10-V2, or PD10 representation-KD plus particle-level dual-view teacher.

## Goal

Extend the current PD10 5M/1M/1M privileged distillation v0.4 setup without
changing its deployment contract:

```text
inference:
  HLT particles -> HLT-only student ParT -> 10-class logits
```

The V2 question is:

```text
If v0.4 logit distillation is not enough, can a stronger train-time teacher
and a richer hidden-representation target move the same HLT-only student closer
to Offline ParT?
```

V2 should reuse the existing v0.4 split manifest, HLT cache, HLT teacher,
offline teacher, metrics, prediction-cache contract, and final-test reporting.
It adds two pieces:

```text
1. A particle-level dual-view teacher that sees paired HLT and offline
   constituents directly.

2. Representation distillation, where the HLT-only student matches a compact
   teacher hidden representation in addition to, or instead of, teacher logits.
```

## What Stays Fixed From PD10 v0.4

Use the exact current 10-class setup:

```text
model_train: 5,000,000 jets
model_val:   1,000,000 jets
final_test:  1,000,000 jets
```

Keep these v0.4 artifacts as anchors:

```text
HLT ParT baseline / HLT teacher
offline ParT teacher
dual-view logit-fusion teacher
warm-start CE-only student
warm-start dual-view logit-KD student
scratch controls already in PD10
```

The new runs should be compared against:

```text
warm-start CE-only
warm-start HLT KD
warm-start offline KD
warm-start logit-fusion dual-view KD
HLT ParT baseline
offline ParT reference
```

Do not replace the current PD10 v0.4 run. Add V2 outputs under the same
experiment root or under a clearly named sibling root:

```text
checkpoints/privileged_distill_10class_5m_run1/v2_repkd_particle_dualview/
```

or, if the original root should remain frozen:

```text
checkpoints/privileged_distill_10class_5m_v2_repkd_particle_dualview_run1/
```

## Why These Are The Next Best Additions

The high-data literature points to two likely failure modes for basic logit KD:

```text
teacher signal too compressed:
  10 logits may not expose enough of the privileged geometry.

teacher too shallow:
  a logit-fusion teacher can only combine what the independent HLT/offline
  teachers already expressed in their final logits.
```

Representation KD addresses the first issue. A hidden teacher vector can carry
more class-boundary and ambiguity structure than a 10-way softmax.

The particle-level dual-view teacher addresses the second issue. It learns from
the actual matched HLT/offline constituents, so it can see degradation patterns
that are invisible after both views are collapsed to logits.

The student still receives only HLT particles at inference, so any gain is a
true train-time-privilege gain.

## Model A: Particle-Level Dual-View Teacher

### Input Contract

The teacher may consume:

```text
HLT particles for jet i
offline particles for the same jet i
label y_i
```

The teacher may not be used as a deployable model. It is only used to create:

```text
particle_dual_view teacher logits
particle_dual_view teacher representations
teacher reference metrics
```

### Architecture

Use a two-branch Particle Transformer teacher:

```text
HLT particles     -> HLT ParT embedding branch     -> h_hlt
offline particles -> offline ParT embedding branch -> h_off

fusion_input = concat(
  h_hlt,
  h_off,
  abs(h_off - h_hlt),
  h_hlt * h_off
)

fusion_repr =
  LayerNorm(4D)
  Linear(4D, 512)
  GELU
  Dropout(0.05)
  Linear(512, 256)
  GELU
  LayerNorm(256)

teacher_logits =
  Linear(256, 10)
```

Where `D` is the ParT branch CLS embedding width.

The representation target for student distillation is:

```text
teacher_repr = L2_normalize(fusion_repr)
```

This is deliberately not full offline-particle reconstruction. It is a compact
teacher state that should be easier for an HLT-only student to approximate than
the entire offline constituent set.

### Initialization

Initialize branches from the v0.4 teachers:

```text
HLT branch:
  load encoder weights from hlt_part_teacher_10class

offline branch:
  load encoder weights from offline_part_teacher_10class

fusion MLP and classifier:
  random init
```

This is the highest-likelihood version because both branches start from
already-useful 10-class particle representations.

### Training Schedule

Train only on `model_train`, select only on `model_val`.

Use a short stabilized schedule:

```text
stage 1:
  freeze both ParT branches
  train fusion MLP + classifier for 1 epoch
  lr head = 1.0e-3

stage 2:
  unfreeze all parameters
  train 8 to 12 epochs
  lr branches = 3.0e-5
  lr fusion/head = 3.0e-4
  weight_decay = 1.0e-4
  batch_size = 128 if memory allows
  AMP enabled
  early_stop_patience = 3
```

Select checkpoint by:

```text
primary: model_val cross entropy
tie-break: model_val accuracy
```

### Teacher Caches

Cache the selected teacher on:

```text
model_train
model_val
final_test
```

Write two aligned cache types:

```text
logit cache:
  logits [N, 10]
  labels [N]
  jet_ids [N]

representation cache:
  representations [N, 256]
  labels [N]
  jet_ids [N]
```

Both caches must declare:

```text
teacher_target = particle_dual_view
allowed_inputs = HLT_plus_offline_train_time_privileged
student_deployment_inputs = HLT_only
train_time_only = true
split
content hash
jet identity hash
source HLT cache hash
source offline manifest/hash
teacher checkpoint hash
```

Final-test caches are allowed only for reference metrics and final reporting.
They must not be used for student checkpoint selection.

## Model B: Representation Distillation Student

### Student Contract

The deployed student remains:

```text
HLT particles -> student ParT -> 10-class logits
```

During training only, the student also exposes a pooled ParT representation:

```text
student_repr_raw = student CLS / pooled embedding
student_repr = L2_normalize(
  LayerNorm(D)
  Linear(D, 256)
)
```

The projection head is training-time auxiliary state. Export/inference should
ignore it or tolerate it without requiring any teacher cache.

### Losses

For labels `y`, student logits `s`, teacher logits `t`, student representation
`r_s`, and teacher representation `r_t`:

```text
CE = cross_entropy(s, y)

KD = T^2 * KL(
  softmax(t / T),
  softmax(s / T)
)

RepKD = mean(1 - cosine_similarity(r_s, r_t))
```

The combined loss is:

```text
L = (1 - alpha) * CE
  + alpha * KD
  + beta * RepKD
```

For representation-only ablations:

```text
L = CE + beta * RepKD
```

Default values:

```text
T = 2.0
alpha = 0.5
beta = 0.10
```

Warmup:

```text
warm-start students:
  alpha and beta ramp from 0 to target over 1 epoch

scratch students:
  alpha and beta ramp from 0 to target over 3 epochs
```

Selection remains deployment-metric first:

```text
primary: model_val cross entropy from HLT-only student logits
tie-break: model_val accuracy
```

Do not select by representation loss alone. A student that imitates the teacher
representation but tags worse is not a win.

## First V2 Run Matrix

Do not explode the matrix. Start with warm-start runs because that is the
deployment-relevant high-data path.

### Required V2 Core

Train the particle-level dual-view teacher:

```text
particle_dual_view_teacher_ptinit
```

Then run these HLT-only warm-start students:

```text
warm_particle_dual_logit_kd
  teacher = particle_dual_view
  loss = CE + logit KD
  T = 2.0
  alpha = 0.5

warm_particle_dual_rep_kd
  teacher = particle_dual_view
  loss = CE + representation KD
  beta = 0.10

warm_particle_dual_logit_rep_kd
  teacher = particle_dual_view
  loss = CE + logit KD + representation KD
  T = 2.0
  alpha = 0.5
  beta = 0.10

warm_logit_fusion_dual_logit_rep_kd
  teacher = existing dual_view_logit teacher
  loss = CE + logit KD + representation KD
  T = 2.0
  alpha = 0.5
  beta = 0.10
```

The logit-fusion representation target is the hidden layer immediately before
the final `Linear(..., 10)` in the existing dual-view logit teacher. This is a
cheap control: it tests representation KD without requiring the particle-level
teacher to be the source of the effect.

### Priority V2 Ablations

Run these only if the required V2 jobs are healthy:

```text
warm_particle_dual_logit_rep_kd_beta03
  beta = 0.30

warm_particle_dual_logit_rep_top3
  logit KD mode = top-3
  beta = 0.10

warm_particle_dual_logit_rep_confidence
  logit KD mode = confidence-weighted
  beta = 0.10

scratch_particle_dual_logit_rep_kd
  same loss as warm_particle_dual_logit_rep_kd
  init = scratch
```

The scratch run is a science control. It should not block the main
deployment-relevant warm-start comparison.

## Success Criteria

The main win condition is:

```text
warm_particle_dual_logit_rep_kd beats warm-start CE-only
and beats warm-start logit-fusion dual-view KD
on final_test cross entropy, accuracy, macro AUC, and key binary FPR metrics.
```

The strongest outcome is:

```text
particle dual-view teacher > logit-fusion teacher
and
particle dual-view logit+rep KD > particle dual-view logit-only KD
```

That would imply both V2 ideas matter:

```text
raw paired particles improve the teacher
hidden representation targets improve transfer
```

Useful partial outcomes:

```text
particle logit-only KD wins:
  the better teacher matters, but representation KD may need tuning.

logit-fusion logit+rep KD wins:
  representation KD is valuable even with the cheap teacher.

rep-only KD wins or ties logit KD:
  hidden geometry may be a cleaner target than softened class probabilities.

none of the V2 students beat warm-start CE:
  the HLT ParT may already be near the learnable HLT limit, or the teacher
  representations are still not student-compatible.
```

## Reporting Additions

The existing PD10 report should gain a V2 section with:

```text
particle dual-view teacher metrics
V2 student metrics
comparison against v0.4 warm-start CE
comparison against v0.4 warm-start dual-view logit KD
HLT-to-offline gap closure
representation loss curves
teacher-student logit KL on model_val
teacher-student representation cosine on model_val
```

Final-test reporting must use only selected checkpoints. No final-test metric
may influence teacher or student selection.

## Leakage Rules

The particle-level teacher is allowed to load offline particles.

The representation cache is allowed to contain teacher features derived from
offline particles.

The student training dataset may return teacher logits and teacher
representations only during training/validation.

The student model, exported checkpoint, and final-test prediction path must
work with:

```text
HLT particles only
no offline particles
no teacher logits
no teacher representations
no teacher checkpoint
```

Every new cache and checkpoint should explicitly record:

```text
allowed_inputs
student_deployment_inputs
returns_offline_particles
inference_export_requires_teacher_features
```

## Implementation Steps

### Step 1: Add V2 Config And Cache Contracts

Extend `teacher_logit_reco/privileged_distill_10class/` with V2 constants:

```text
teacher target = particle_dual_view
student target modes = rep_only, full_logits_plus_rep, top3_plus_rep,
                       confidence_weighted_plus_rep
default beta = 0.10
representation_dim = 256
```

Add representation cache helpers:

```text
teacher_logit_reco/privileged_distill_10class/representations.py
```

Contracts:

```text
pd10_teacher_representation_cache_v1
pd10_student_representation_kd_training_v1
```

### Step 2: Implement Particle-Level Dual-View Teacher

Add:

```text
teacher_logit_reco/privileged_distill_10class/particle_dual_view_teacher.py
scripts/train_pd10_particle_dual_view_teacher.py
scripts/cache_pd10_particle_dual_view_teacher.py
```

Implement:

```text
paired HLT/offline dataset aligned by jet identity
two-branch ParT teacher
branch initialization from HLT/offline teacher checkpoints
stage-1 frozen-branch head training
stage-2 low-LR full fine-tuning
logit and representation cache writing
```

### Step 3: Add Representation-Aware Student Training

Extend the student data path so a KD batch may include:

```text
teacher_logits, optional
teacher_representations, optional
```

Extend the student model wrapper to expose:

```text
logits
student_repr_raw
student_repr_projected
```

The normal prediction/export path must still return logits from HLT inputs
without needing teacher representations.

### Step 4: Add V2 Student Variants And CLI

Extend:

```text
scripts/train_pd10_student.py
teacher_logit_reco/privileged_distill_10class/students.py
```

Add CLI args:

```text
--teacher-representation-cache
--representation-beta
--representation-dim
--representation-mode none|cosine
```

Add V2 variant naming:

```text
warm_particle_dual_logit_kd
warm_particle_dual_rep_kd
warm_particle_dual_logit_rep_kd
warm_logit_fusion_dual_logit_rep_kd
```

### Step 5: Extend Reports And Tests

Update PD10 reports to include V2 teachers/students when present.

Add tests for:

```text
representation cache alignment by jet identity
student batch remains HLT-only
rep-KD loss is finite and backpropagates
rep-only students do not require teacher logits
logit+rep students require both requested caches
final-test prediction works without teacher caches
V2 report compares against v0.4 anchors
```

### Step 6: Add Slurm Runners

Add:

```text
sbatch/run_pd10_train_particle_dual_view_teacher.sh
sbatch/run_pd10_cache_particle_dual_view_teacher.sh
sbatch/submit_pd10_v2_repkd_particle_dualview.sh
```

The dependency graph should be:

```text
existing PD10 split/cache/audit
  -> existing HLT/offline teachers
  -> particle dual-view teacher
  -> particle dual-view logit/representation caches
  -> V2 students
  -> V2 report
```

The submitter should accept:

```text
PD10_ROOT
PD10_V2_ROOT
CONDA_ENV=atlas_kd
CONFIRM_FINAL_TEST=1
PD10_MODEL_TRAIN_SIZE / MODEL_VAL_SIZE / FINAL_TEST_SIZE
```

Smoke-test first with:

```text
20k train / 5k val / 10k final_test
epochs:
  particle teacher = 2
  students = 2
```

Then queue the 5M V2 run if the smoke test confirms:

```text
paired HLT/offline alignment
teacher logits finite
teacher representations finite
student rep loss finite
HLT-only final-test prediction succeeds
report includes V2 comparisons
```

