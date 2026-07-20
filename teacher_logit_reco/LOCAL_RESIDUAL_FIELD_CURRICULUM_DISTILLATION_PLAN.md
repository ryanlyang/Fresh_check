# Local Residual Field Tagger-Aware Curriculum Distillation Plan

## Purpose

The local particle residual field campaign showed a very important split:

```text
HLT ParT baseline:                 about 77.6% high-data accuracy
Oracle local residual fields:      about 81.8% high-data accuracy
Offline-particle ParT reference:   about 84.1% model-val accuracy
Best deployable predicted fields:  only modestly above HLT after fusion
```

That is a compelling but uncomfortable result.

The residual-field representation has real information. When the tagger sees
the true offline-derived local fields, it recovers a large fraction of the
offline-vs-HLT gap. But the current learned reconstructor does not predict
fields that are useful enough for tagging.

This plan keeps the same local residual field idea, but changes the training
objective. Instead of asking the predictor to jump directly from HLT-only input
to full oracle behavior, we train against a curriculum of weaker oracle
teachers:

```text
HLT + weak true residual signal      -> small gain over HLT
HLT + medium true residual signal    -> medium gain over HLT
HLT + strong true residual signal    -> large gain over HLT
HLT + full true residual signal      -> oracle ceiling
```

The deployable model then learns the residual-field language in smaller steps.

The goal is not merely to lower residual-field MAE. The goal is:

```text
predict residual fields that cause an augmented HLT-only ParT to make better
tagging decisions
```

## Core Hypothesis

The first LPRF result suggests three things:

1. True local residual fields are tagger-useful.
2. Average reconstruction quality is not the same as tagger-useful
   reconstruction quality.
3. Directly matching the full oracle field target may be too hard or too
   brittle.

The new hypothesis is:

```text
The residual predictor can learn more useful corrections if it is trained
through oracle taggers of gradually increasing difficulty.
```

In other words, the oracle should not only be a final ceiling. It should become
a curriculum teacher.

## Deployment Contract

The deployable runtime path is always:

```text
HLT particles -> residual predictor -> predicted fields
HLT particles + predicted fields -> student tagger logits
```

At runtime, the model must not use:

```text
offline particles
true residual fields
oracle fields
teacher logits
offline teacher models
final-test labels
```

Oracle teachers are train-time objects only. They may be used on
`model_train`, `model_val`, and `stack_train/stack_val` for diagnostics and
distillation. Final-test use must be limited to deployable HLT-only prediction
unless an output is explicitly marked as a non-deployable oracle diagnostic.

## Data Regime

Use the same data regime as the current LPRF campaign:

```text
HLT profile: fixed_hlt_v2_realistic
HLT degradation strength: 2.5
labels: 10 JetClass classes
max constituents: 128
```

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

Run the pilot first. Do not expand to high data until the deployable
curriculum models show a meaningful model-val and stack-val gain over the HLT
baseline.

Suggested high-data gate:

```text
best deployable curriculum model >= A0 + 0.3 percentage points on stack_val
and
no evidence of nonfinite validation collapse
and
all final-test decisions remain HLT-only
```

## Existing Components Reused

This plan builds on the existing local residual field setup:

```text
target cache:
  local per-HLT-particle residual fields computed from HLT and offline views

residual fields:
  pT / energy flow
  centroid / axis shift
  multiplicity
  composition
  merge/split/reliability

augmented tagger:
  original HLT particle features + residual field channels -> ParT

existing baselines:
  A0 HLT ParT
  A1 schedule-matched HLT fine-tune
  A2 capacity-matched HLT control
  B oracle true-field taggers
  C residual reconstructors
  D/E predicted-field taggers
  F controls
  G fusion
```

The new work should avoid rewriting these pieces unless needed. It should add
the missing training bridge between the oracle tagger and the deployable
predictor.

## Oracle Field Ladder

### Alpha-Scaled Oracle Fields

Define a scaled oracle field:

```text
field_alpha = alpha * true_residual_field
```

where:

```text
alpha in {0.0, 0.10, 0.25, 0.50, 0.75, 1.0}
```

`alpha = 0.0` is equivalent to blank residual fields.

`alpha = 1.0` is the full oracle field.

Intermediate alphas create weaker oracle teachers:

```text
O_alpha_0p10: HLT + 10% true residual fields
O_alpha_0p25: HLT + 25% true residual fields
O_alpha_0p50: HLT + 50% true residual fields
O_alpha_0p75: HLT + 75% true residual fields
O_alpha_1p00: HLT + full true residual fields
```

The exact accuracy targets should not be hard-coded. The implementation should
run an alpha sweep and record:

```text
alpha
model_val accuracy
stack_val accuracy
accuracy gap vs A0
accuracy gap vs full oracle
cross entropy
per-class accuracy
```

The expected pattern is a smooth ladder, for example:

```text
alpha = 0.10 -> about 78% to 78.5%
alpha = 0.25 -> about 78.5% to 79.5%
alpha = 0.50 -> about 79.5% to 80.5%
alpha = 0.75 -> about 80.5% to 81.5%
alpha = 1.00 -> about full oracle
```

If the curve is flat until very high alpha, the field representation is brittle
and will be much harder to distill.

### Field-Subset Oracle Teachers

Create additional oracle teachers using selected field groups:

```text
O_pt:        pT / energy-flow fields only
O_mult:      multiplicity fields only
O_pt_mult:   pT / energy-flow + multiplicity
O_geom:      centroid / axis-shift fields only
O_comp:      composition fields only
O_rel:       reliability / uncertainty fields only
O_all:       all fields
```

Purpose:

```text
Find which true residual signals provide useful but learnable intermediate
teachers.
```

The best teacher for distillation may not be the strongest full oracle. A
slightly weaker teacher with easier-to-predict fields may pull the deployable
student up more reliably.

### Noisy And Dropout Oracle Teachers

Create robust oracle teachers by corrupting true fields during teacher
training:

```text
field_dropout
per-field Gaussian noise in normalized units
per-group dropout
radius dropout
confidence/reliability channel dropout
```

These teachers should still evaluate on clean true fields and optionally on
noisy true fields. The point is to make teacher behavior less brittle when the
student later supplies imperfect predicted fields.

Recommended teachers:

```text
O_noise_light:  light field noise + light field dropout
O_noise_med:    medium field noise + group dropout
O_curriculum:   alpha increases during teacher training
```

## Frozen Oracle Tagger Distillation

The central new mechanism is a frozen oracle teacher used as a differentiable
loss network.

Let:

```text
R_theta(HLT) -> predicted residual fields and confidence
S_phi(HLT, predicted fields) -> deployable student logits
T_alpha(HLT, fields) -> frozen oracle teacher logits
```

During training:

```text
pred_fields = R_theta(HLT)

student_logits = S_phi(HLT, pred_fields)

oracle_true_logits = T_alpha(HLT, alpha * true_fields)       # no gradient
oracle_pred_logits = T_alpha(HLT, pred_fields)              # gradient to R_theta only
```

The teacher parameters are frozen. Gradients flow through the teacher forward
pass into `pred_fields`, then into `R_theta`.

This makes the predictor learn fields that sit in the oracle tagger's useful
input space.

### Why This Is Different From Plain Field MAE

Plain reconstruction says:

```text
make every predicted field numerically close to the true residual field
```

Tagger-aware distillation says:

```text
make the predicted fields cause the oracle tagger to behave like it had useful
true residual information
```

The second objective is closer to what we actually care about.

## Loss Function

The main deployable model should optimize:

```text
L_total =
    w_ce        * CE(student_logits, labels)
  + w_student_kd * KD(student_logits, oracle_true_logits or offline_teacher_logits)
  + w_oracle_path * KD(oracle_pred_logits, oracle_true_logits)
  + w_field     * masked_field_loss(pred_fields, true_fields)
  + w_gate      * confidence_or_reliability_loss
  + w_reg       * residual_magnitude / smoothness regularization
```

Recommended default:

```text
w_ce = 1.0
w_student_kd = 0.3
w_oracle_path = 1.0
w_field = 0.2
w_gate = 0.05
w_reg = 0.01
temperature = 2.0 or 3.0
```

The first pilot should sweep a small number of weights rather than trying to
solve all tuning at once.

### Field Loss

Use masked Huber loss in normalized field units:

```text
Huber(pred_fields, true_fields)
```

Mask invalid particle slots. Keep the existing finite/clipping guard:

```text
nan_to_num
clip residual field inputs
record clip fraction
skip nonfinite batches
```

### Oracle-Path KD

Use:

```text
KD(T_alpha(HLT, pred_fields), T_alpha(HLT, alpha * true_fields))
```

This is the main curriculum loss.

It should be computed only on train/eval splits where true residual fields are
allowed.

### Student KD

Use:

```text
KD(S_phi(HLT, pred_fields), T_alpha(HLT, alpha * true_fields))
```

or:

```text
KD(S_phi(HLT, pred_fields), offline_teacher_logits)
```

The oracle-teacher logits are usually safer for early stages because they are
conditioned on the same residual-field representation. Offline teacher logits
can be introduced after the predictor is stable.

### Confidence Gates

The predictor should output:

```text
pred_fields_raw
field_log_variance or uncertainty
field_gate in [0, 1]
```

The effective deployable field is:

```text
effective_fields = field_gate * clipped(pred_fields_raw)
```

The tagger should receive:

```text
HLT features
effective predicted fields
field_gate
field_uncertainty
```

Rationale:

```text
A bad correction can be worse than no correction. The model needs a learned
way to say "do not trust this field here."
```

## Curriculum Schedules

### Static Alpha Teachers

Train separate students against fixed teachers:

```text
alpha = 0.25
alpha = 0.50
alpha = 0.75
alpha = 1.00
```

These runs reveal whether weaker teachers are easier to learn.

### Increasing Alpha Schedule

Train one student with alpha increasing over epochs:

```text
epochs 0-2:    alpha = 0.25
epochs 3-5:    alpha = 0.50
epochs 6-8:    alpha = 0.75
epochs 9+:     alpha = 1.00
```

For pilot, keep this short. For high data, use a smoother schedule:

```text
alpha(t) = min_alpha + (max_alpha - min_alpha) * sigmoid_ramp(t)
```

Recommended:

```text
min_alpha = 0.25
max_alpha = 1.00
warmup_epochs = 3
ramp_epochs = 10
```

### Teacher Mixing

Optionally mix teacher logits:

```text
teacher_logits_mix =
    (1 - beta) * logits_HLT_teacher
  + beta * logits_oracle_alpha
```

where beta increases during training.

This is a softer bridge from HLT behavior to oracle behavior. It is useful if
direct oracle-path KD destabilizes the predictor.

## Recommended Pilot Campaign

### Tier A: Baselines And References

`A0`: HLT ParT baseline on the same split.

`A1`: schedule-matched HLT fine-tune control.

`A2`: capacity-matched HLT control.

`A4`: offline-particle ParT reference, at least model-val. Final-test only if
explicitly confirmed.

### Tier O: Oracle Teacher Ladder

`O0`: blank residual fields, equivalent to alpha 0.

`O1`: alpha-scaled oracle, `alpha=0.10`.

`O2`: alpha-scaled oracle, `alpha=0.25`.

`O3`: alpha-scaled oracle, `alpha=0.50`.

`O4`: alpha-scaled oracle, `alpha=0.75`.

`O5`: full oracle, `alpha=1.00`.

`O6`: pT/density plus multiplicity oracle.

`O7`: robust full oracle with field dropout/noise.

The report should rank these by:

```text
accuracy gain vs A0
teacher softness / entropy
per-class gains
stability under noisy field inputs
```

### Tier P: Deployable Predictor-Student Runs

`P0`: current best Huber-only reconstructor + augmented tagger baseline.

`P1`: fixed `alpha=0.25` oracle-path KD.

`P2`: fixed `alpha=0.50` oracle-path KD.

`P3`: fixed `alpha=0.75` oracle-path KD.

`P4`: curriculum `0.25 -> 0.50 -> 0.75`.

`P5`: curriculum `0.25 -> 0.50 -> 0.75 -> 1.00`.

`P6`: `P5` plus masked field Huber.

`P7`: `P6` plus confidence gates.

`P8`: `P7` plus offline-teacher logit KD after the alpha ramp.

Expected best single model:

```text
P7 or P8
```

Why:

```text
P7 learns the oracle field language gradually and can gate uncertain fields.
P8 adds the strongest privileged class-level signal after the field predictor
is already stable.
```

### Tier Q: Ablations

`Q0`: full `P7` recipe but no oracle-path KD.

`Q1`: full `P7` recipe but no field Huber.

`Q2`: full `P7` recipe but no confidence gates.

`Q3`: direct full-oracle `alpha=1.00` from the start.

`Q4`: student KD only, no frozen oracle teacher input-gradient path.

`Q5`: frozen predictor after Huber pretraining, train only tagger.

Purpose:

```text
Prove whether the gain comes from curriculum teacher guidance, confidence
gating, field matching, or plain extra optimization.
```

### Tier G: Fusion

`G0`: logit fusion of `A0` and best `P` model.

`G1`: seed ensemble of the best `P` recipe.

`G2`: complementary alpha ensemble:

```text
best fixed alpha student
+ best curriculum student
+ best confidence-gated student
```

`G3`: field-family ensemble:

```text
best all-field curriculum model
+ best pT/multiplicity-focused curriculum model
+ best robust/noisy-teacher model
```

Fusion should use stack-train only for fitting weights and stack-val for model
selection. Final-test remains untouched until the chosen fusion is frozen.

## Diagnostics That Decide Whether This Works

### 1. Alpha Teacher Curve

For each alpha teacher:

```text
accuracy(alpha)
CE(alpha)
per-class accuracy(alpha)
teacher entropy(alpha)
teacher confidence(alpha)
```

This tells us whether the oracle can be decomposed into learnable steps.

### 2. Predicted Field Usefulness Curve

Evaluate:

```text
T_alpha(HLT, pred_fields)
```

against:

```text
T_alpha(HLT, alpha * true_fields)
```

Record:

```text
oracle-path KD
oracle-path accuracy
agreement with true-field oracle
agreement with labels
```

This directly measures whether the predictor is producing fields in the
teacher's useful input space.

### 3. Alpha-Mix Diagnostic

For each trained predictor:

```text
field_mix = lambda * true_fields + (1 - lambda) * pred_fields
```

Evaluate:

```text
lambda in {0, 0.25, 0.50, 0.75, 1.0}
```

If performance jumps only near `lambda=1.0`, the learned fields are still too
far from the useful manifold. If performance improves smoothly, the predictor
is close and may benefit from more capacity/training.

### 4. Error Where It Matters

Average MAE is not enough. Report field errors conditioned on:

```text
high missing_pt_frac
high missing_n_frac
merged-token regions
large local centroid shifts
classes where HLT confuses labels
jets where oracle fixes HLT mistakes
```

The predictor may look decent globally while failing exactly on the rare
regions that drive the oracle gain.

### 5. Gate Calibration

For confidence-gated runs:

```text
gate value vs field error
gate value vs oracle improvement
gate value vs student correctness
gate value by class
gate value by radius
```

The gate should be high when predicted fields are useful and low when they are
likely harmful.

## Implementation Plan

### Step 1: Scaled And Corrupted Oracle Field Sources

Add field-source modes:

```text
oracle_scaled
oracle_field_subset
oracle_noisy
oracle_dropout
```

Required arguments:

```text
--oracle-field-alpha
--oracle-field-noise-std
--oracle-field-dropout
--oracle-field-group-dropout
--field-subset
```

The same normalization, masking, clipping, and finite guards used in the LPRF
tagger path must apply here.

### Step 2: Oracle Teacher Training And Logit Caching

Train oracle teacher taggers for Tier O.

Each teacher writes:

```text
run_report.json
training_curves.json
best_model_val.pt
teacher_config.json
oracle_teacher_logits/<teacher_id>/<split>_predictions.npz
oracle_teacher_logits/<teacher_id>/<split>_metadata.json
```

Cache logits for:

```text
model_train
model_val
stack_train
stack_val
```

Do not cache final-test oracle logits for primary selection.

### Step 3: Frozen Oracle Teacher Wrapper

Implement a wrapper that loads a frozen oracle teacher checkpoint and supports:

```text
teacher_logits_true = T_alpha(HLT, true_or_scaled_fields).detach()
teacher_logits_pred = T_alpha(HLT, predicted_fields)
```

Teacher parameters must have:

```text
requires_grad = False
```

but gradients must flow through `predicted_fields`.

The wrapper must expose diagnostics:

```text
teacher_id
teacher_alpha
teacher_field_subset
teacher_checkpoint_hash
teacher_train_split
teacher_model_val_accuracy
```

### Step 4: Predictor-Student Joint Model

Create a training path with:

```text
ResidualFieldPredictor R_theta
DeployableAugmentedParT S_phi
FrozenOracleTeacher T_alpha
```

The forward pass returns:

```text
pred_fields_raw
pred_fields_effective
field_gate
field_uncertainty
student_logits
oracle_pred_logits
```

The deployed checkpoint must save only:

```text
R_theta
S_phi
normalization metadata
field names
provenance hashes
```

It must not require the oracle teacher at inference.

### Step 5: Curriculum Scheduler

Implement scheduler support:

```text
fixed_alpha
piecewise_alpha
sigmoid_alpha
teacher_sequence
loss_weight_schedule
```

The run report must record the alpha and loss weights used in every epoch.

### Step 6: Confidence Gates

Add predictor heads:

```text
field_delta_head
field_log_var_head
field_gate_head
```

Support gate modes:

```text
none
learned_sigmoid
supervised_reliability
uncertainty_inverse
```

Default:

```text
learned_sigmoid + light reliability supervision
```

### Step 7: Training Runner

Add:

```text
scripts/train_local_residual_field_curriculum_student.py
```

It should support:

```text
oracle teacher checkpoint
oracle teacher logits cache
offline teacher logits cache
field target cache
alpha schedule
loss weights
confidence gates
student warm start
predictor warm start
final-test deployable evaluation
```

Training must skip nonfinite batches and fail closed if validation coverage is
too low.

### Step 8: Prediction And Fusion

Reuse existing LPRF prediction and fusion where possible.

Predictions must record:

```text
student checkpoint hash
predictor checkpoint hash
teacher used during training, if any
runtime_inputs = HLT_only
```

Fusion should fail closed if any selected member prediction is missing.

### Step 9: Reporting

Extend the LPRF report with:

```text
oracle_teacher_curve.csv
curriculum_student_metrics.csv
alpha_mix_diagnostics.csv
teacher_student_agreement.csv
field_error_where_oracle_helps.csv
gate_calibration.csv
```

The report should clearly separate:

```text
deployable results
oracle diagnostics
offline-particle references
```

Primary tables must not mix oracle final-test diagnostics into deployable
leaderboards.

### Step 10: Slurm Submitters

Add pilot submitters first:

```text
submit_lprf_curriculum_pilot.sh
submit_lprf_curriculum_tigris_pilot.sh
```

High-data submitters should exist but default to off until the pilot gate is
met:

```text
submit_lprf_curriculum_highdata.sh
submit_lprf_curriculum_tigris_highdata.sh
```

The submitter should support reusing existing LPRF caches:

```text
split manifest
HLT cache
offline cache
target residual field cache
offline teacher logits
```

If any cache is reused, enforce content hashes and profile/strength metadata.

### Step 11: Tests

Add tests for:

```text
scaled oracle fields preserve masks
alpha=0 matches blank fields
alpha=1 matches true oracle fields
field subset teachers select physical columns
noisy/dropout teachers are deterministic under seed
frozen teacher parameters receive no gradients
predicted fields receive gradients through frozen teacher
oracle-path KD decreases on a toy batch
confidence gates are finite and clipped
curriculum schedule records alpha per epoch
final-test path is HLT-only
report separates deployable and oracle diagnostics
```

## Pilot Decision Criteria

The pilot should answer four questions:

1. Does the alpha teacher curve rise smoothly from HLT to oracle?
2. Can a deployable predictor trained through weak teachers beat the HLT
   baseline?
3. Do curriculum students beat direct `alpha=1.0` students?
4. Do confidence gates improve or at least stabilize deployable performance?

Proceed to high data only if:

```text
best deployable curriculum student > A0 by at least 0.3 pp on stack_val
or
fusion of deployable curriculum students > A0 by at least 0.5 pp on stack_val
```

Also require:

```text
nonfinite batches near zero
validation coverage above 99%
provenance checks pass
oracle diagnostics clearly marked non-deployable
```

## Expected Best Version

The best-performing single deployable setup is expected to be:

```text
P8:
  HLT warm start
  residual predictor warm-started from Huber pretraining
  alpha curriculum 0.25 -> 0.50 -> 0.75 -> 1.00
  frozen oracle-path KD
  masked field Huber
  confidence-gated residual fields
  offline-teacher KD only after the alpha ramp stabilizes
```

The best overall deployable setup is expected to be:

```text
G1 or G2:
  seed/logit fusion of the best curriculum students
```

The scientific expectation is not that the first deployable student reaches
the full 81.8% oracle ceiling. A successful pilot would be:

```text
A0 HLT baseline:       about 77.6%
current LPRF fusion:   about 77.9%
curriculum student:    about 78.2% to 78.6%
curriculum fusion:     about 78.5% to 79.0%
oracle residual field: about 81.8%
offline ParT:          about 84.1% model-val
```

That would prove the oracle residual signal can be distilled in useful pieces.

## Interpretation

If the curriculum works, the result says:

```text
The local residual field target is a useful privileged representation, but it
must be distilled through tagger-aware objectives rather than pure regression.
```

If it fails, the result is still informative:

```text
The oracle fields may be too brittle or too dependent on exact offline
information to predict from HLT alone.
```

Either way, this plan turns the current oracle gap from a frustration into a
controlled learning problem.
