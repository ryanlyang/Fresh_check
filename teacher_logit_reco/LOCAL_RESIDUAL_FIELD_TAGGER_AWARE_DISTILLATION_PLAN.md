# Local Residual Field Tagger-Aware Distillation Plan

## Core Motivation

The local particle residual field campaign gave a very useful scientific split:

```text
HLT ParT baseline:                  about 77.6% high-data model-val
HLT + true oracle residual fields:  about 81.8% high-data model-val
Offline-particle ParT reference:    about 84.1% high-data model-val
Best learned deployable LPRF so far: about 77.9% final-test after seed fusion
```

That means the residual-field representation is not a dead end. The true fields
recover a large fraction of the offline-vs-HLT gap. The problem is that the
current learned predictor does not produce fields that are useful enough for the
tagger.

The next strategy is to stop training the residual predictor only as a numerical
regressor. Instead, train it through a strong frozen field-consuming tagger, so
the predictor learns residual fields that preserve the oracle tagger's tagging
decisions.

The deployable path remains:

```text
HLT particles -> predicted local residual fields -> HLT + predicted fields tagger
```

No offline particles, no true residual fields, and no teacher logits are allowed
at runtime or final-test prediction time.

## Central Idea

Train an oracle residual-field tagger:

```text
T_oracle(HLT particles, true residual fields) -> class logits
```

Then freeze it and train a deployable residual-field predictor:

```text
P(HLT particles) -> predicted residual fields
```

The predictor is optimized by passing its predicted fields through the frozen
oracle tagger:

```text
T_oracle(HLT particles, P(HLT particles)) -> class logits
```

and matching:

```text
T_oracle(HLT particles, true residual fields)
labels
optional offline-particle teacher logits
```

This is a tagger-aware or perceptual residual-field loss. It rewards fields for
being useful to the downstream ParT decision, not just for having low average
MAE.

## Why A Curriculum Is Needed

The full oracle jump is large:

```text
HLT baseline -> full oracle residual fields = roughly +4 percentage points
```

Asking a deployable HLT-only predictor to learn that full correction in one shot
may be too hard. The predictor can get stuck producing average-looking fields
that minimize Huber loss but do not move the classifier.

The smarter setup is a ladder of easier oracle teachers:

```text
HLT + 0.25 * true fields
HLT + 0.50 * true fields
HLT + 0.75 * true fields
HLT + 1.00 * true fields
```

The deployable predictor first learns to imitate a weak oracle teacher that only
nudges HLT performance upward. Then the teacher difficulty increases.

This creates smaller learning steps:

```text
77.6 -> 78.x -> 79.x -> 80.x -> 81.x
```

instead of one brittle leap.

## Data Contract

Use the same local residual field setup unless explicitly overridden:

```text
HLT profile: fixed_hlt_v2_realistic
HLT degradation strength: 2.5
labels: all 10 JetClass labels
max_constits: 128
```

Pilot first:

```text
model_train: 500k
model_val: 150k
stack_train: 300k
stack_val: 150k
final_test: 150k
```

High-data only if pilot is promising:

```text
model_train: 5M
model_val: 1M
stack_train: 2M
stack_val: 1M
final_test: 1M
```

Every run must be hash-bound to:

```text
split manifest
HLT cache
offline cache
residual-field target cache
teacher-logit cache when used
label ordering
selected field schema
```

The final deployable evaluation must load only:

```text
manifest
HLT cache
deployable model checkpoint
```

It must not require true residual-field targets or offline particles.

## Existing Inputs To Reuse

This plan should reuse the existing local residual-field cache format:

```text
targets/<split>_local_residual_fields.npz
```

with the current field groups:

```text
pT / energy flow
centroid / axis shift
multiplicity
composition
merge / split / reliability
```

The plan should also reuse:

```text
local_particle_residual_field/tagger.py
local_particle_residual_field/tagger_train.py
local_particle_residual_field/train.py
local_particle_residual_field/report.py
```

where possible. This should be a new training mode and campaign family, not a
new unrelated model zoo.

## Model Objects

### Residual Field Predictor `P`

Input:

```text
HLT particles, HLT mask
```

Output:

```text
predicted residual field tensor: [batch, particles, fields]
optional confidence / uncertainty tensor: [batch, particles, fields or groups]
```

The best initial predictor should be the strongest existing LPRF reconstructor
family, preferably the C6-style consistency-aware version if it is stable:

```text
particle-local encoder
jet-context pooling
small attention/context block
field-group heads
optional uncertainty head
finite clamp and normalization
```

The predictor should always sanitize output before passing fields into ParT:

```text
nan_to_num
clip to configured range
mask invalid particles
scale by residual_field_scale
```

### Oracle Field Consumer `T_alpha`

Input:

```text
HLT particles
scaled true residual fields: alpha * true_fields
optional residual confidence/reliability fields
```

Output:

```text
10-class logits
```

This is a ParT tagger trained with CE on labels. It consumes residual fields as
additional per-particle features.

Each `T_alpha` is privileged at training/diagnostic time because it consumes
true residual fields. It is not directly deployable unless true fields are
replaced by predicted fields.

### Robust Oracle Consumer `T_robust`

The best final teacher should not be trained only on perfect true fields. It
should be trained to tolerate imperfect predicted fields by seeing controlled
corruption:

```text
sample alpha from a schedule or distribution
apply field-group dropout
apply Gaussian residual-field noise
apply confidence/reliability masks
occasionally replace true fields with predictor warmup fields
```

This teacher is still privileged because its main positive examples use true
fields, but it creates a smoother landscape for the deployable predictor.

### Deployable Student `S`

The deployable student has two possible forms:

1. Frozen-consumer form:

```text
S(x_hlt) = T_oracle(x_hlt, P(x_hlt))
```

Here `T_oracle` is frozen after oracle training. Only `P` is trained in the
tagger-aware predictor stage.

2. Fine-tuned-consumer form:

```text
S(x_hlt) = T_student(x_hlt, P(x_hlt))
```

Here `T_student` is initialized from the oracle consumer or HLT baseline, then
fine-tuned on predicted fields with strict HLT-only inputs.

The best expected endpoint is the fine-tuned-consumer form, but the
frozen-consumer form is the cleanest diagnostic.

## Losses

### Field Reconstruction Loss

Keep the existing normalized field Huber loss:

```text
L_field = Huber(P(x_hlt), true_fields)
```

Use valid HLT particle masks and field-group weights. Do not let padded tokens
contribute.

This loss keeps the predictor physically anchored, but it should not dominate
late training.

### Oracle Logit Distillation Loss

For a chosen alpha:

```text
teacher_logits = T_alpha(x_hlt, alpha * true_fields)
student_logits = T_alpha(x_hlt, alpha * P(x_hlt))
```

Then:

```text
L_oracle_kd = KL(student_logits / tau, teacher_logits / tau) * tau^2
```

Gradients flow through the frozen `T_alpha` into `P`. The teacher forward should
be no-grad. The student forward through `T_alpha` must keep gradients into
`P`, while `T_alpha` parameters stay frozen.

### Label CE Through Frozen Consumer

Use:

```text
L_ce = CE(T_alpha(x_hlt, alpha * P(x_hlt)), labels)
```

This prevents the predictor from merely imitating teacher mistakes and keeps the
fields classification-useful.

### Offline Teacher KD

Optionally add:

```text
L_offline_kd = KL(student_logits, offline_particle_teacher_logits)
```

This should be a late-stage loss only. It can help if the oracle consumer is
underpowered relative to the original offline ParT.

### Confidence Calibration Loss

If the predictor emits uncertainty:

```text
predicted_fields = gate * raw_fields
```

and:

```text
L_uncertainty = calibration loss against true field error or oracle-logit impact
```

The goal is to let the model say:

```text
I do not know the residual here, so use HLT more heavily.
```

This is important because a bad residual field can be worse than no residual.

### Total Loss

The default late-stage loss should be:

```text
L =
  w_field     * L_field
+ w_oracle_kd * L_oracle_kd
+ w_ce        * L_ce
+ w_offline   * L_offline_kd
+ w_uncert    * L_uncertainty
```

Recommended starting weights:

```text
w_field:     1.0 during warmup, then 0.1 to 0.3
w_oracle_kd: 1.0
w_ce:        0.5
w_offline:   0.25 late only
w_uncert:    0.05 if enabled
tau:         2.0
```

## Alpha Curriculum

The training loop needs a configurable alpha scheduler:

```text
phase 1: alpha = 0.25
phase 2: alpha = 0.50
phase 3: alpha = 0.75
phase 4: alpha = 1.00
```

or a smooth ramp:

```text
alpha(epoch) = min_alpha + (max_alpha - min_alpha) * ramp_fraction
```

The same alpha must be applied to both true and predicted fields when using an
`alpha` teacher:

```text
teacher path: alpha * true_fields
student path: alpha * predicted_fields
```

This keeps the difficulty controlled. It teaches the predictor to make small
useful corrections first.

For `T_robust`, alpha can be sampled per batch:

```text
alpha ~ Uniform(0.25, 1.0)
```

with a higher probability of the current curriculum target.

## Teacher Ladder

### Tier O: Oracle Consumer Teachers

Train these first:

```text
O0_hlt_only_consumer
O25_alpha_0p25_true_fields
O50_alpha_0p50_true_fields
O75_alpha_0p75_true_fields
O100_alpha_1p00_true_fields
Orobust_alpha_noise_dropout
```

`O0` should be close to the HLT ParT baseline. `O100` should approach the
current oracle result. `Orobust` is expected to be the best teacher for
deployable predictor training, even if its perfect-field accuracy is slightly
lower than `O100`.

Save for each oracle teacher:

```text
best_model_val.pt
run_report.json
model_val logits with true fields
stack_val logits with true fields
optional model_train logits with true fields
diagnostic alpha response curve
```

Do not use oracle teachers for final-test model selection.

### Tier M: Alpha-Mix Diagnostic

For an existing predictor `P`, evaluate:

```text
mixed_fields(alpha) = alpha * true_fields + (1 - alpha) * P(x_hlt)
```

through a frozen oracle consumer:

```text
T_oracle(x_hlt, mixed_fields(alpha))
```

for:

```text
alpha = 0.0, 0.25, 0.5, 0.75, 1.0
```

Interpretation:

```text
smooth improvement: predicted fields are close to useful
only alpha near 1 works: oracle signal is brittle
alpha 0 already helps: predictor fields are useful
alpha 0 hurts: predictor fields are actively misleading
```

This diagnostic is one of the most important parts of the pilot.

## Pilot Variant Ladder

The pilot should stay compact enough to iterate, but broad enough to answer the
scientific question.

### T0: Current Huber-Only Baseline

```text
P trained with field Huber only
tagger trained normally on predicted fields
```

This anchors the new method against the current LPRF approach.

### T1: Fixed Weak Oracle KD

```text
teacher: O25
train P through frozen O25
loss: oracle KD + CE
no field Huber
```

Purpose:

```text
Can tagger-aware loss alone learn a small useful correction?
```

### T2: Fixed Medium Oracle KD

```text
teacher: O50
train P through frozen O50
loss: oracle KD + CE
no field Huber
```

Purpose:

```text
How hard does the target become when the teacher is stronger?
```

### T3: Alpha Curriculum KD

```text
teacher sequence: O25 -> O50 -> O75
loss: oracle KD + CE
no field Huber
```

Purpose:

```text
Does gradual teacher difficulty outperform fixed alpha?
```

### T4: Alpha Curriculum + Field Huber

```text
teacher sequence: O25 -> O50 -> O75 -> O100
loss: oracle KD + CE + decayed field Huber
```

Purpose:

```text
Does physical anchoring prevent degenerate tagger-aware fields?
```

This is the first serious candidate.

### T5: Robust Teacher + Field Huber

```text
teacher: Orobust
loss: robust oracle KD + CE + decayed field Huber
```

Purpose:

```text
Does a noise/dropout-trained oracle consumer make predicted fields easier to use?
```

This is likely the strongest frozen-consumer candidate.

### T6: Robust Teacher + Field Huber + Confidence Gates

```text
teacher: Orobust
P outputs fields and reliability gates
loss: robust oracle KD + CE + field Huber + uncertainty calibration
```

Purpose:

```text
Can the student avoid harming itself on uncertain local corrections?
```

This is likely the best predictor-only candidate.

### T7: Warmup Then Upper-ParT Fine-Tune

```text
initialize P from T6
initialize consumer from Orobust or HLT baseline
freeze lower ParT blocks first
train P + classifier/head + upper ParT blocks
loss: CE + oracle KD + offline KD + small field Huber
```

Purpose:

```text
Let the tagger adapt to the imperfect predicted-field distribution.
```

This is the best expected single-model endpoint.

### T8: Seed Ensemble Of T7

```text
train T7 with 3 seeds
fit scalar-weighted logit fusion on stack_train
evaluate on stack_val and final_test
```

Purpose:

```text
Best headline deployable result if T7 has seed diversity.
```

## High-Data Expansion Rule

Run the pilot first.

Expand to high-data only if at least one deployable variant shows one of:

```text
single model: >= +0.3 percentage points over A0 on model_val or stack_val
fusion:       >= +0.5 percentage points over A0 on stack_val
alpha-mix diagnostic: smooth improvement from alpha=0 to alpha=1
```

Do not expand based only on oracle performance. We already know the oracle is
strong. The expansion condition must be about learned deployable predictions.

## Expected Best Configuration

The best expected single model is:

```text
T7 = robust oracle curriculum + field Huber + confidence gates + upper-ParT fine-tune
```

Why:

```text
Orobust makes the oracle consumer tolerant of imperfect fields.
The alpha curriculum makes the objective learnable.
Field Huber keeps the fields physically meaningful.
Confidence gates let the model ignore uncertain corrections.
Upper-ParT fine-tuning lets the classifier adapt to predicted-field artifacts.
Offline KD gives a final weak pull toward the true offline particle model.
```

The best expected overall result is:

```text
T8 = scalar-weighted seed ensemble of T7
```

The most important diagnostic is:

```text
alpha-mix curve for T6/T7
```

If T7 improves but the alpha-mix curve is brittle, the method may be exploiting
capacity rather than true residual prediction. If T7 improves and the alpha-mix
curve is smooth, the idea is scientifically much stronger.

## Implementation Plan

### Step 1: Registry And Config

Add a new campaign namespace:

```text
local_residual_field_tagger_aware
```

or add a clearly separated tagger-aware mode under:

```text
local_particle_residual_field
```

Implement config objects for:

```text
oracle consumer training
alpha teacher schedule
tagger-aware predictor training
confidence gates
frozen-consumer vs fine-tuned-consumer mode
```

Run IDs must be explicit:

```text
O0, O25, O50, O75, O100, Orobust
T0, T1, T2, T3, T4, T5, T6, T7, T7_seed1, T7_seed2, T7_seed3, T8
M_alpha_mix
```

### Step 2: Oracle Consumer Training

Extend the LPRF tagger trainer so it can train oracle consumers with:

```text
field_source = oracle
oracle_alpha = fixed float or sampled distribution
oracle_field_noise
oracle_field_dropout
oracle_field_group_dropout
```

The saved run report must record:

```text
oracle_alpha_mode
field_noise
field_dropout
target_cache_hash
selected_field_names
not_deployable_runtime_reason
```

Oracle consumers may evaluate model_val and stack_val. Final-test oracle
evaluation should be disabled by default or clearly marked diagnostic-only.

### Step 3: Oracle Logit Cache

Cache teacher logits for each oracle consumer on:

```text
model_train
model_val
stack_val
```

Do not cache final-test oracle logits unless a special diagnostic flag is used.

The cache metadata must include:

```text
teacher_checkpoint_hash
oracle_alpha
field_schema_hash
target_cache_hash
manifest_hash
hlt_cache_hash
```

### Step 4: Frozen-Consumer Predictor Training

Implement a training path where:

```text
P is trainable
T_alpha or T_robust is frozen
student_logits = T_frozen(HLT, alpha * P(HLT))
teacher_logits = T_frozen(HLT, alpha * true_fields)
```

The optimizer must include only trainable predictor parameters unless the
variant explicitly requests fine-tuning.

The report must expose:

```text
field_loss
oracle_kd_loss
ce_loss
offline_kd_loss
confidence_loss
alpha
field MAE
teacher/student agreement
student accuracy through frozen consumer
```

### Step 5: Alpha Curriculum Scheduler

Implement:

```text
fixed alpha
piecewise alpha schedule
linear ramp
sampled alpha for robust teacher
```

The per-epoch training curves must record:

```text
alpha_mean
alpha_min
alpha_max
active_teacher_id
```

### Step 6: Confidence-Gated Fields

Add optional predictor heads:

```text
field_delta
field_gate
field_uncertainty
```

Use bounded gates:

```text
gate = sigmoid(raw_gate)
fields_for_tagger = gate * predicted_fields
```

Diagnostics:

```text
mean gate by field group
gate/error correlation
gate/oracle-impact correlation
fraction of clipped fields
```

### Step 7: Fine-Tuned Consumer Stage

Implement a second-stage trainer:

```text
load P from T6
load consumer from Orobust or HLT baseline
freeze lower blocks for warmup
unfreeze classifier/head and upper ParT blocks
train with CE + oracle KD + offline KD + small field Huber
```

The final exported model must be deployable with:

```text
HLT particles only
```

and internally produce:

```text
predicted fields -> tagger logits
```

### Step 8: Alpha-Mix Diagnostics

Write a diagnostic evaluator:

```text
scripts/evaluate_local_residual_field_alpha_mix.py
```

For each trained predictor:

```text
alpha_mix in [0.0, 0.25, 0.5, 0.75, 1.0]
mixed_fields = alpha_mix * true_fields + (1 - alpha_mix) * predicted_fields
logits = T_oracle(HLT, mixed_fields)
```

Output:

```text
alpha_mix_curve.csv
per_class_alpha_mix.csv
teacher_agreement_by_alpha.csv
```

### Step 9: Prediction, Fusion, And Report

Add report outputs:

```text
oracle_teacher_ladder.csv
tagger_aware_metrics.csv
alpha_mix_curve.csv
field_usefulness.csv
deployable_gap.csv
fusion_metrics.csv
provenance_audit.json
summary.md
```

The report must fail closed if:

```text
required oracle teacher missing
required deployable model missing
teacher cache hash mismatch
target cache hash mismatch
final-test deployable pass used target/offline cache
nonfinite validation coverage below threshold
```

Fusion should include:

```text
T7 seed ensemble
best tagger-aware model + HLT A0
best tagger-aware model + current LPRF D5 ensemble if available
```

### Step 10: Slurm Submitters

Write submitters for:

```text
pilot full campaign
pilot oracle-only
pilot deployable-only from existing oracle teachers
report rerun
high-data full campaign
```

The default should queue pilot only. High-data should require an explicit env
flag:

```text
LOCAL_RESIDUAL_FIELD_TA_CONFIRM_HIGHDATA=1
```

Tigris settings should remain explicit:

```text
partition=tigris
account=reu-aisocial
PYTHONNOUSERSITE=1
CONDA_ENV=atlas_kd_tigris
```

### Step 11: Tests

Add tests for:

```text
oracle alpha scaling changes only residual fields
frozen consumer has no trainable gradients
gradients flow from frozen consumer logits into predictor
teacher logits are no-grad
alpha scheduler records correct per-epoch metadata
final-test deployable loader refuses target/offline cache
confidence gates are finite and masked
alpha-mix diagnostic uses true fields only on non-final diagnostic splits
report fails on missing oracle teacher or hash mismatch
```

## Interpretation Rules

### If O25/O50/O75 Improve Smoothly

Good. The oracle signal has a learnable ladder. Continue to tagger-aware
predictor training.

### If Only O100 Improves

The residual representation is brittle. Train `Orobust` with noise/dropout
before trusting deployable predictor results.

### If T1/T2 Improve But T4/T5 Do Not

Weak teachers are easier to imitate than full teachers. Keep the curriculum
shallower or stop at the teacher strength that improves deployable accuracy.

### If Field MAE Improves But Accuracy Does Not

The predictor is numerically good but tagger-useless. Increase oracle KD and
decrease Huber weight.

### If Accuracy Improves But Field MAE Gets Worse

That can be acceptable if controls fail. It means the model found
classification-useful pseudo-fields. Check field distributions and alpha-mix
curves before declaring victory.

### If Controls Match The Gain

The gain is probably capacity/fine-tuning, not residual information. The report
must say that plainly.

## Success Criteria

Pilot success:

```text
T7 single model beats A0 by >= 0.3 pp on stack_val
or
T8 fusion beats A0 by >= 0.5 pp on stack_val
```

Strong success:

```text
T7/T8 beats current LPRF D5 seed fusion
alpha-mix curve is smooth
teacher/student agreement improves over Huber-only predictor
controls do not explain the gain
```

High-data success:

```text
final-test deployable result beats HLT A0
report is ok=true
no final-test privileged cache dependency
provenance audit is clean
```

## Bottom Line

The oracle result says the residual-field language is valuable. The current
learned predictor is the bottleneck. This plan turns the oracle into a training
curriculum and trains the predictor through the tagger, so the residual fields
are optimized for the thing we actually care about:

```text
better HLT-only tagging.
```

