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
to full oracle behavior, we train against a curriculum of weaker fixed-consumer
oracle responses and genuinely weakened subset/robust oracle consumers:

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

## Pilot-Only Execution Contract

The full campaign is intentionally broad, but the first pilot must be small and
decision-oriented. The first-stage pilot is explicitly two-stage so deployable
students do not guess which oracle consumer to follow before the alpha
diagnostics have run.

Stage 1a queues:

```text
oracle teachers:
  O0, Ofull, Orobust_light

alpha diagnostic:
  D_alpha_eval_Ofull using fixed Ofull, not separately retrained alpha teachers
  D_alpha_eval_Orobust using fixed Orobust_light, not separately retrained alpha teachers

deployable baseline:
  P0
```

Stage 1a then runs a selector/report job that writes:

```text
selected_consumer.json
```

with:

```text
selected_consumer_id: Ofull or Orobust_light
selected_alpha_endpoint: usually 0.75, or 0.25/0.50 if the curve says so
selection_source: D_alpha_eval_Ofull and D_alpha_eval_Orobust
selection_reason
selection_primary_split: model_val
selection_confirmation_split: stack_val
model_val_alpha_curve
stack_val_alpha_curve
```

Stage 1b queues only after `selected_consumer.json` exists and passes
validation:

```text
deployable predictor-students:
  P2, P4, P7a, P7b


ablations:
  Q0, Q3

fusion:
  G0 only
```

This is the first-stage prune. Do not queue Stage 1b until Stage 1a has chosen
a consumer. Do not queue the full alpha-teacher family, extra subset/noise
teachers, `P1/P3/P5/P6/P8`, `Q1/Q2/Q4/Q5`, or `G1/G2/G3` until the pilot gates
below are passed.

Expected first-stage job count:

```text
Stage 1a new GPU training jobs if A0/A4 are reused: 4
Stage 1a new GPU training jobs if A0/A4 must be rerun: 6
Stage 1a GPU alpha-response evaluation jobs: 2
Stage 1a selector/report jobs: 1
Stage 1b GPU training jobs after selected_consumer.json: 6
Stage 1b fusion/report jobs: 2
optional alpha-mix diagnostic jobs: 2
```

Infrastructure cache jobs are separate. `D_alpha_eval_Ofull` and
`D_alpha_eval_Orobust` are GPU evaluation jobs, not training jobs. If the
split, HLT cache, offline cache, target residual-field cache, or offline
teacher logits are missing, rebuild them first and do not count that as
evidence for the model ladder.

Pilot budget guardrails:

```text
max first-stage GPU training jobs: 12
max Stage 1a GPU training jobs before selector: 6
max Stage 1b GPU training jobs before pilot report: 6
max oracle teacher training jobs: 3 unless an existing full-oracle B-tier run is reused
max walltime per pilot model job: cluster-specific, but fail/requeue rather than silently truncating reports
max report attempts before inspection: 1
```

First-stage pass conditions:

```text
at least one fixed-consumer alpha response curve is non-degenerate
selector chooses a consumer with a useful/smooth alpha response, or records a stop decision
best deployable P run beats A0 by >= 0.3 pp on stack_val
or G0 beats A0 by >= 0.5 pp on stack_val
validation valid_fraction >= 0.99 for selected checkpoints
nonfinite train/eval batches are near zero and explicitly reported
final-test deployable rows have runtime_inputs = HLT_only
```

First-stage stop conditions:

```text
both D_alpha_eval_Ofull and D_alpha_eval_Orobust alpha=0.25/0.75 fail to beat alpha=0.0 by at least 0.2 pp on model_val
both fixed-consumer alpha responses are strongly non-monotonic and no intermediate alpha beats A0
P2/P4/P7a/P7b all fail to beat P0
any selected run has validation coverage below 0.99
any report mixes oracle diagnostics into the deployable leaderboard
```

If the fixed-consumer alpha response curves are non-monotonic, do not blindly continue to
stronger alpha targets. Compare `D_alpha_eval_Ofull` and
`D_alpha_eval_Orobust`, select the smoother and more monotonic consumer, then
select the best stable alpha endpoint seen so far and rerun only the
corresponding deployable student. For example:

```text
if alpha=0.25 > alpha=0.75, train from alpha=0.25 rather than ramping to alpha=0.75
if alpha=0.75 is unstable but alpha=0.25 is useful, use alpha=0.25 as the curriculum endpoint
if both fixed-consumer alpha responses are flat, stop and revise the residual-field target or predictor
```

Stage 1b consistency rule:

```text
P2, P4, P7a, P7b, Q0, Q3, and G0 must all record the same selected_consumer_id
unless the campaign is explicitly launched in paired-consumer mode.
Q0 must be the selected P7a recipe minus oracle-path KD.
Q3 must be the selected consumer with direct selected_alpha_endpoint from epoch 0.
```

Optional paired-consumer mode:

```text
LOCAL_RESIDUAL_FIELD_CURRICULUM_CONFIRM_PAIRED_CONSUMERS=1
```

This queues paired deployable variants such as `P4_Ofull`,
`P4_Orobust`, `P7a_Ofull`, `P7a_Orobust`, `P7b_Ofull`, and
`P7b_Orobust`. It is more robust but is not the default first-stage pilot
because it roughly doubles Stage 1b.

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

## Canonical Run-ID And Config Schema

Human-readable IDs are useful for discussion:

```text
O0, Ofull, Orobust_light
D_alpha_eval_Ofull, D_alpha_eval_Orobust
P0, P2, P4, P7a, P7b
Q0, Q3
G0
```

but every run must also write a machine-parsable config name:

```text
oracle teacher:
  O_kind-{blank|full|robust|subset}_subset-{subset_tag}_noise-{noise_tag}_drop-{drop_tag}

fixed-consumer diagnostic:
  D_alpha-response_consumer-{consumer_id}_alphas-{alpha_tags}

deployable student:
  P_recipe-{recipe_tag}_consumer-{consumer_id}_schedule-{schedule_tag}_gate-{gate_tag}_studentinit-{init_tag}

ablation:
  Q_ablate-{ablation_tag}_consumer-{consumer_id}_schedule-{schedule_tag}

fusion:
  G_mode-{fusion_mode}_members-{member_tag}
```

Examples:

```text
O_kind-full_subset-all_noise-0p00_drop-0p00
O_kind-robust_subset-all_noise-0p05_drop-0p10
D_alpha-response_consumer-Ofull_alphas-0p00-0p10-0p25-0p50-0p75-1p00
D_alpha-response_consumer-Orobust-light_alphas-0p00-0p10-0p25-0p50-0p75-1p00
P_recipe-curriculum_consumer-Ofull_schedule-alpha0p25-to-alpha0p75_gate-learned_studentinit-A0
P_recipe-curriculum_consumer-Orobust_light_schedule-alpha0p25-to-alpha0p75_gate-learned_studentinit-oracle
Q_ablate-no-oracle-path_consumer-selected_schedule-alpha0p25-to-alpha0p75
Q_ablate-direct-alpha_consumer-selected_schedule-alpha0p75-direct
```

The compact IDs map to the first-stage pilot as:

```text
O0 = blank/zero-field consumer
Ofull = fixed full oracle consumer trained on true fields, alpha 1.00
Orobust_light = full oracle consumer trained with light field noise/dropout
D_alpha_eval_Ofull = fixed Ofull(HLT, alpha * true_fields) diagnostic sweep
D_alpha_eval_Orobust = fixed Orobust_light(HLT, alpha * true_fields) diagnostic sweep
P0 = current Huber-only predicted-field baseline
P2 = fixed weak target using selected consumer response at alpha 0.25
P4 = curriculum alpha 0.25 -> selected endpoint through selected consumer response
P7a = P4 + field Huber + confidence gates, student initialized from A0
P7b = P4 + field Huber + confidence gates, student initialized from selected consumer
Q0 = selected P7a recipe without oracle-path KD
Q3 = selected consumer and selected alpha endpoint from the start, no curriculum
G0 = A0 + best deployable P logit fusion
```

The selected consumer is not implicit. Stage 1b jobs must read
`selected_consumer.json` and write these fields into every report:

```text
selected_consumer_id
selected_alpha_endpoint
selected_consumer_source_report
selected_consumer_hash
paired_consumer_mode
```

The report must fail if default Stage 1b rows disagree on
`selected_consumer_id` or `selected_alpha_endpoint`.

Baseline semantics must be impossible to confuse:

```text
A0 = clean HLT ParT baseline.
O0 = zero/blank-field oracle-consumer diagnostic.
```

`O0` must not be used as the primary HLT floor. Even with zero fields, `O0`
may use the augmented residual-field input path, different input projection
width, different initialization, or different oracle-consumer training history.
The main "improvement over HLT" denominator is always `A0` unless a table is
explicitly labeled as an oracle-consumer diagnostic.

Every `run_report.json` must store both:

```text
run_id
canonical_config_id
```

and the final report must group by `run_id` while validating the full
`canonical_config_id`.

## Oracle Field Ladder

### Fixed-Consumer Alpha Response

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

Important correction:

```text
Do not assume that separately trained alpha-scaled teachers are weaker.
```

If a teacher is trained from scratch on `0.25 * true_fields`, it can learn a
larger first-layer weight and recover much of the same information as a teacher
trained on `1.0 * true_fields`. Scaling the input alone does not necessarily
remove information.

Therefore the primary alpha ladder is a diagnostic and training signal through
a fixed consumer:

```text
Ofull(HLT, alpha * true_fields)
Orobust_light(HLT, alpha * true_fields)
```

The first pilot should train or register:

```text
O0:            blank/zero-field consumer
Ofull:         full oracle consumer trained on true fields
Orobust_light: full oracle consumer trained with light field noise/dropout
```

Then run:

```text
D_alpha_eval_Ofull:
  evaluate fixed Ofull(HLT, alpha * true_fields)
  alpha = 0.0, 0.10, 0.25, 0.50, 0.75, 1.0

D_alpha_eval_Orobust:
  evaluate fixed Orobust_light(HLT, alpha * true_fields)
  alpha = 0.0, 0.10, 0.25, 0.50, 0.75, 1.0
```

This is the actual difficulty ladder. It asks:

```text
How does one fixed field-consuming model respond as the amount of true
residual-field information is reduced, and is the robust consumer smoother
under weakened/imperfect fields than the full oracle consumer?
```

Both alpha evaluations are GPU evaluation jobs, not training jobs. They should
load fixed consumer checkpoints and sweep field scaling on `model_val` and
`stack_val`. The selector should use `model_val` as the primary decision split.
`stack_val` is a confirmation/stability split only: it can break very close
model-val ties or flag a brittle choice, but it must not rescue a consumer or
alpha endpoint whose model-val curve is weak.

`D_alpha_eval_Orobust` is not optional in the first pilot. `Ofull` may be
nonlinear or brittle away from alpha 1.0 because it learned to consume perfect
true fields. `Orobust_light` is trained with field noise/dropout and is expected
to give a smoother curriculum target when predicted fields are imperfect.

The exact accuracy targets should not be hard-coded. The implementation should
run both fixed-consumer alpha sweeps and record:

```text
alpha
consumer_id
model_val accuracy
stack_val accuracy (confirmation only)
accuracy gap vs A0
accuracy gap vs the same consumer at alpha 1.0
cross entropy
per-class accuracy
teacher entropy
monotonicity score
smoothing score
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

Separately trained alpha teachers are expansion-only and should be clearly
marked as:

```text
trained_alpha_teacher_diagnostic
```

not as the primary curriculum evidence.

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

These are true weak teachers because fields are physically omitted, not merely
rescaled.

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
Orobust_light: light field noise + light field dropout
Orobust_med:   medium field noise + group dropout
Orobust_ramp:  robust-consumer expansion with field alpha/noise ramping
```

`Orobust_ramp` is not first-stage evidence for the alpha curriculum. It is an
expansion-only way to make a field consumer less brittle, and its report must
describe the training-time corruption schedule separately from
`D_alpha_eval_Ofull` and `D_alpha_eval_Orobust`.

## Frozen Oracle Tagger Distillation

The central new mechanism is a frozen oracle teacher used as a differentiable
loss network.

Let:

```text
R_theta(HLT) -> predicted residual fields and confidence
S_phi(HLT, predicted fields) -> deployable student logits
T_consumer(HLT, fields) -> frozen oracle/robust consumer logits
```

During training:

```text
pred_fields = R_theta(HLT)

student_logits = S_phi(HLT, pred_fields)

oracle_true_logits = T_consumer(HLT, alpha * true_fields)       # no gradient
oracle_pred_logits = T_consumer(HLT, alpha * pred_fields)       # gradient to R_theta only
```

The teacher parameters are frozen. Gradients flow through the teacher forward
pass into `pred_fields`, then into `R_theta`.

This makes the predictor learn fields that sit in the oracle tagger's useful
input space.

### Student Initialization Choices

The deployable student `S_phi` must be explicit about initialization. The pilot
should test both high-probability choices:

```text
P7a:
  S_phi initialized from A0 / HLT ParT baseline.
  Copy compatible HLT feature/input weights.
  Initialize residual-field input channels conservatively near zero.

P7b:
  S_phi initialized from the selected oracle consumer.
  Keep the residual-field input pathway already learned by the oracle consumer.
  Fine-tune on predicted fields so it becomes deployable and less brittle.
```

Expected tradeoff:

```text
P7a is safer because it starts from a strong HLT-only solution.
P7b may use residual fields better, but it may over-trust imperfect predicted fields.
```

`P7b` must include an explicit adaptation/reset policy. Initializing from an
oracle consumer is not enough by itself, because that model learned to consume
true residual fields while deployment gives predicted residual fields. The
default `P7b` schedule should be:

```text
phase 0: checkpoint surgery
  load selected_consumer_id into S_phi
  reset or shrink residual-field input projection weights by 0.1
  initialize learned residual-field gate bias low, e.g. sigmoid bias near 0.1
  preserve the original HLT feature projection weights

phase 1: residual-path adaptation, 1-3 epochs
  freeze most of the ParT body
  train residual-field projection, gate heads, predictor, classifier/head
  use low alpha endpoint, e.g. alpha 0.25

phase 2: upper gentle unfreeze
  unfreeze classifier/head and upper ParT blocks
  continue alpha curriculum toward the selected endpoint
  keep a small field Huber loss and gate regularization alive

phase 3: full gentle unfreeze only if validation is stable
  unfreeze all ParT blocks at a small LR
  stop immediately if validation valid_fraction drops or nonfinite batches rise
```

The run report must record:

```text
student_init_source
residual_projection_reset_mode
residual_projection_scale
initial_gate_bias
freeze_schedule
per-phase optimizer groups and learning rates
```

This is an important pilot comparison. Do not collapse `P7a` and `P7b` into one
run ID.

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

Pilot loss-weight table:

```text
run  w_ce  w_student_kd  w_oracle_path  w_field  w_gate  w_reg  teacher
P0   1.0   0.0           0.0            1.0      0.0     0.01   none
P2   1.0   0.0           1.0            0.0      0.0     0.01   selected consumer alpha=0.25
P4   1.0   0.0           1.0            0.1      0.0     0.01   selected consumer alpha=0.25 -> selected endpoint
P7a  1.0   0.0           1.0            0.2      0.05    0.01   P4 + A0 student init
P7b  1.0   0.0           1.0            0.2      0.05    0.01   P4 + selected-consumer student init
Q0   1.0   0.0           0.0            0.2      0.05    0.01   selected P7a without oracle-path
Q3   1.0   0.0           1.0            0.2      0.05    0.01   selected consumer direct endpoint
```

Do not introduce additional free loss weights in the first-stage pilot. If a
run fails, inspect diagnostics before adding a new weight sweep.

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
KD(T_consumer(HLT, alpha * pred_fields), T_consumer(HLT, alpha * true_fields))
```

This is the main curriculum loss.

It should be computed only on train/eval splits where true residual fields are
allowed.

### Student KD

Use:

```text
KD(S_phi(HLT, pred_fields), T_consumer(HLT, alpha * true_fields))
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

Default supervised reliability target:

```text
field_error_i,g = mean_abs_normalized_error over field group g for particle i
gate_target_i,g = exp(-field_error_i,g / gate_error_scale)
```

with:

```text
gate_error_scale = 1.0 by default
gate_target clipped to [0, 1]
invalid particle slots masked out
target detached from predictor gradients
```

The loss is:

```text
L_gate = MSE(field_gate_i,g, gate_target_i,g)
```

Optional second-stage reliability target:

```text
oracle_impact_i,g = KL(
  T_consumer(HLT, pred_fields),
  T_consumer(HLT, pred_fields with group g replaced by true group g)
)
```

This is more expensive and should not be part of the first-stage pilot unless
the simple field-error gate is clearly insufficient.

## Oracle-Path Memory Policy

The frozen oracle-consumer path is more memory-heavy than ordinary KD because
the teacher parameters are frozen but its activations must still preserve a
gradient path back to `pred_fields`. The first-stage pilot should use
conservative defaults:

```text
student_train_batch_size: 16 to 32 on GH200/Tigris
eval_batch_size: 64 when memory allows
gradient_accumulation_steps: increase instead of increasing per-step batch size
mixed_precision: bf16 preferred on GH200; fp16 only if nonfinite checks stay clean
gradient_checkpointing: enable for student and frozen consumer attention blocks where supported
oracle_forward_microbatch_size: <= student_train_batch_size
```

If the frozen-consumer path OOMs, do not silently change the scientific recipe.
Use a named fallback:

```text
oracle_logit_only_fallback=true
```

where cached `T_consumer(HLT, alpha * true_fields)` logits are used for
student KD, but the differentiable loss
`KD(T_consumer(HLT, alpha * pred_fields), T_consumer(HLT, alpha * true_fields))`
is disabled. This fallback is useful operationally, but it is not equivalent to
`P2/P4/P7a/P7b`; reports must label it separately and exclude it from claims
about oracle-path input-gradient distillation.

## Curriculum Schedules

### Fixed-Consumer Static Alpha Targets

Train separate students against fixed alpha responses of the same consumer:

```text
alpha = 0.25
alpha = 0.50
alpha = 0.75
alpha = 1.00
```

These runs reveal whether weaker fixed-consumer targets are easier to learn.
They should call:

```text
T_consumer(HLT, alpha * true_fields)
T_consumer(HLT, alpha * pred_fields)
```

Do not train a separate new teacher for each alpha in the first-stage pilot.
Those separately trained alpha teachers are diagnostic expansion runs only.

### Increasing Alpha Schedule Through A Fixed Consumer

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
max_alpha = 0.75 for first-stage pilot
warmup_epochs = 3
ramp_epochs = 10
```

Only ramp to `alpha=1.00` after `D_alpha_eval_Ofull` and
`D_alpha_eval_Orobust` show that full-alpha supervision is stable and genuinely
stronger than intermediate alpha. The default pilot endpoint is `alpha=0.75`,
not full oracle.

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

This section lists the full family, but the first-stage pilot must follow the
`Pilot-Only Execution Contract` above. Treat the other variants as expansion
runs after the pilot proves that tagger-aware curriculum distillation is worth
more GPU time.

### Tier A: Baselines And References

`A0`: HLT ParT baseline on the same split.

`A1`: schedule-matched HLT fine-tune control.

`A2`: capacity-matched HLT control.

`A4`: offline-particle ParT reference.

Required split coverage:

```text
model_val: required for offline-gap comparisons
stack_val: optional, useful if already cached
final_test: optional and only if explicitly precomputed/confirmed
```

Reports must not compare an `A4` model-val-only number against deployable
final-test numbers. If `A4` has no final-test evaluation, final-test tables
should show `A4` as `not_available_model_val_only`.

### Tier O: Oracle Teacher Ladder

`O0`: blank/zero-field oracle-consumer diagnostic.

`O0` is not the clean HLT baseline. It may use the augmented residual-field
input path with zero fields, so it can differ from `A0` by architecture,
capacity, or training history. Reports must label it as `zero_field_consumer`
and must use `A0` as the main HLT floor.

`Ofull`: full oracle consumer trained on true residual fields. Reuse existing
`B1/B4` if the provenance and model contract match.

`Orobust_light`: full oracle consumer trained with light field noise/dropout.

`D_alpha_eval_Ofull`: GPU evaluation diagnostic only, using fixed `Ofull`:

```text
Ofull(HLT, alpha * true_fields)
alpha = 0.0, 0.10, 0.25, 0.50, 0.75, 1.0
```

`D_alpha_eval_Orobust`: GPU evaluation diagnostic only, using fixed
`Orobust_light`:

```text
Orobust_light(HLT, alpha * true_fields)
alpha = 0.0, 0.10, 0.25, 0.50, 0.75, 1.0
```

The deployable curriculum should use whichever consumer has the smoother,
more monotonic, and more useful alpha response. This choice is made only by
the Stage 1a selector and then passed to Stage 1b as `selected_consumer_id`.

`Osubset_pt_mult`: pT/density plus multiplicity oracle.

`Osubset_best`: best field-subset oracle after pilot diagnostics.

`Orobust_med`: medium-noise/dropout oracle consumer.

First-stage pilot subset:

```text
O0, Ofull, Orobust_light, D_alpha_eval_Ofull, D_alpha_eval_Orobust
```

Expansion-only until pilot passes:

```text
Osubset_pt_mult, Osubset_best, Orobust_med, separately trained alpha teachers
```

Reuse rule:

```text
If an existing B-tier oracle run has the same split manifest hash, HLT cache
hash, target cache hash, field schema hash, labels, and alpha/subset/noise
contract, register it instead of retraining an oracle teacher.
```

The report should rank these by:

```text
accuracy gain vs A0
teacher softness / entropy
per-class gains
stability under noisy field inputs
```

### Tier P: Deployable Predictor-Student Runs

`P0`: current best Huber-only reconstructor + augmented tagger baseline.

`P1`: fixed `alpha=0.10` oracle-path KD through the selected consumer.

`P2`: fixed `alpha=0.25` oracle-path KD through the selected consumer.

`P3`: fixed `alpha=0.50` oracle-path KD through the selected consumer.

`P4`: curriculum `0.25 -> 0.50 -> selected_alpha_endpoint` through the
selected consumer.

`P5`: curriculum `0.25 -> 0.50 -> 0.75 -> 1.00` through the selected consumer.

`P6`: `P5` plus masked field Huber.

`P7a`: `P4/P6` plus confidence gates, with `S_phi` initialized from A0.

`P7b`: `P4/P6` plus confidence gates, with `S_phi` initialized from the
selected consumer.

`P8`: best `P7a/P7b` recipe plus offline-teacher logit KD after the alpha ramp.

First-stage pilot subset:

```text
Stage 1a: P0
Stage 1b: P2, P4, P7a, P7b after selected_consumer.json
```

Expansion-only until pilot passes:

```text
P1, P3, P5, P6, P8
```

Expected best single model:

```text
P7a/P7b or P8
```

Why:

```text
P7a/P7b learn the oracle field language gradually and can gate uncertain fields.
P8 adds the strongest privileged class-level signal after the field predictor
is already stable.
```

### Tier Q: Ablations

`Q0`: selected `P7a` recipe but no oracle-path KD.

`Q1`: selected `P7a` recipe but no field Huber.

`Q2`: selected `P7a` recipe but no confidence gates.

`Q3`: direct `selected_alpha_endpoint` from the start through the selected
consumer, no curriculum.

`Q4`: student KD only, no frozen oracle teacher input-gradient path.

`Q5`: frozen predictor after Huber pretraining, train only tagger.

First-stage pilot subset:

```text
Stage 1b: Q0, Q3 after selected_consumer.json
```

Expansion-only until pilot passes:

```text
Q1, Q2, Q4, Q5
```

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

First-stage pilot subset:

```text
G0 only
```

Expansion-only until pilot passes:

```text
G1, G2, G3
```

## Diagnostics That Decide Whether This Works

### 1. Fixed-Consumer Alpha Response Curve

For each alpha value through the same fixed consumer:

```text
accuracy(alpha)
CE(alpha)
per-class accuracy(alpha)
teacher entropy(alpha)
teacher confidence(alpha)
```

This tells us whether the oracle response can be decomposed into learnable
steps without retraining a new teacher at each alpha.

Hard stop rule:

```text
If neither D_alpha_eval_Ofull nor D_alpha_eval_Orobust shows alpha=0.25 or
alpha=0.75 improving over alpha=0.0 on model_val, stop the first-stage pilot
after P0.
If alpha=0.75 is worse than alpha=0.25 by more than 0.2 pp, use alpha=0.25 as
the strongest deployable target and do not ramp deeper.
If alpha=0.25/0.75 are useful but non-monotonic within 0.2 pp, continue with
P2/P4/P7a/P7b but mark the response ladder as non-monotonic in the report.
```

### 2. Predicted Field Usefulness Curve

Evaluate:

```text
T_consumer(HLT, alpha * pred_fields)
```

against:

```text
T_consumer(HLT, alpha * true_fields)
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

Train or register oracle teacher taggers for Tier O.

Before launching a new oracle teacher, check whether an existing LPRF B-tier or
oracle run can be reused. Reuse is allowed only if all of these match:

```text
split manifest hash
HLT cache hash
offline cache hash
target residual-field cache hash
field schema hash
label ordering
alpha
field subset
noise/dropout contract
model architecture contract
```

If a matching existing `B1/B3/B4` oracle run exists, register it as `Ofull` or
`Orobust_light` instead of retraining. If no match exists, the first-stage
pilot may train only:

```text
O0
Ofull
Orobust_light
```

Subset, medium-noise, full-alpha-ramp, and separately trained alpha teachers
are expansion-only unless explicitly enabled.

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
teacher_logits_true = T_consumer(HLT, alpha * true_fields).detach()
teacher_logits_pred = T_consumer(HLT, alpha * predicted_fields)
```

Teacher parameters must have:

```text
requires_grad = False
```

but gradients must flow through `predicted_fields`.

The wrapper must expose diagnostics:

```text
consumer_id
alpha
teacher_field_subset
teacher_checkpoint_hash
teacher_train_split
teacher_model_val_accuracy
oracle_logit_only_fallback
```

### Step 4: Predictor-Student Joint Model

Create a training path with:

```text
ResidualFieldPredictor R_theta
DeployableAugmentedParT S_phi
FrozenOracleConsumer T_consumer
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

For `P7b`, the implementation must support oracle-consumer initialization with
an adaptation/reset schedule:

```text
--student-init-source {A0,Ofull,Orobust_light}
--residual-projection-reset {none,scale,reset}
--residual-projection-scale 0.1
--initial-gate-bias-prob 0.1
--freeze-schedule residual_path_warmup_then_upper_unfreeze
```

The trainer must be able to freeze/unfreeze parameter groups by phase:

```text
phase 1: predictor + residual projection + gates + classifier/head
phase 2: phase 1 + upper ParT blocks
phase 3: full gentle unfreeze
```

Reports must expose the phase schedule, group learning rates, trainable
parameter counts, and the residual projection reset policy.

### Step 5: Curriculum Scheduler

Implement scheduler support:

```text
fixed_alpha
piecewise_alpha
sigmoid_alpha
teacher_sequence
loss_weight_schedule
selected_consumer_id
selected_alpha_endpoint
```

The run report must record the alpha and loss weights used in every epoch.
For Stage 1b, the scheduler must read `selected_consumer.json` unless
`LOCAL_RESIDUAL_FIELD_CURRICULUM_CONFIRM_PAIRED_CONSUMERS=1` is set.

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
selected_consumer.json
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

The runner must fail closed if `Q0` or `Q3` is launched with a `consumer_id`
that differs from the selected `P7a` consumer, unless paired-consumer mode is
explicitly enabled.

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
consumer_selection.csv
```

The report should clearly separate:

```text
deployable results
oracle diagnostics
offline-particle references
```

This separation must be automatic. Every row must include:

```text
runtime_inputs
uses_true_fields
uses_offline_particles
uses_teacher_logits_at_runtime
deployable
split
selection_allowed
```

The report must build separate tables:

```text
deployable_leaderboard.csv
oracle_diagnostics.csv
offline_reference.csv
curriculum_training_diagnostics.csv
consumer_selection.csv
```

Rules:

```text
deployable_leaderboard.csv requires deployable=true
deployable_leaderboard.csv requires runtime_inputs=HLT_only
deployable_leaderboard.csv rejects uses_true_fields=true
deployable_leaderboard.csv rejects uses_offline_particles=true
deployable_leaderboard.csv rejects uses_teacher_logits_at_runtime=true
oracle/reference rows can appear only in diagnostic/reference tables
```

Consumer-selection enforcement:

```text
consumer_selection.csv must summarize D_alpha_eval_Ofull and D_alpha_eval_Orobust
selected_consumer.json must be present before Stage 1b rows are accepted
P2/P4/P7a/P7b/Q0/Q3 must match selected_consumer_id
P4/P7a/P7b must use the selected schedule
Q0 must match selected P7a except oracle-path KD is disabled
Q3 must match selected_consumer_id and selected_alpha_endpoint, with no curriculum ramp
```

If paired-consumer mode is enabled, the report should build separate paired
tables for `Ofull` and `Orobust_light` and only then compare them. In default
two-stage mode, mixed-consumer Stage 1b rows are a report failure.

Baseline labels must be explicit in every summary and CSV:

```text
A0 = clean HLT ParT baseline
O0 = zero-field oracle-consumer diagnostic
```

`O0` may be useful to debug the augmented residual-field input path, but it is
not allowed to define the main HLT floor or any headline improvement
denominator. Headline deployable gaps should be computed against `A0`.

Final-test selection policy:

```text
select individual checkpoints on model_val only
fit fusion weights on stack_train only
choose fusion settings on stack_val only
run final_test once after the deployable model/fusion choice is frozen
do not use oracle alpha diagnostics for final-test selection
do not use final-test oracle diagnostics for any primary selection
```

Primary tables must not mix oracle final-test diagnostics into deployable
leaderboards. If a final-test oracle diagnostic is ever produced, it must be
named and tabled as non-deployable.

### Step 10: Slurm Submitters

Add pilot submitters first:

```text
submit_lprf_curriculum_pilot.sh
submit_lprf_curriculum_tigris_pilot.sh
```

Pilot submitters must support explicit stages:

```text
LOCAL_RESIDUAL_FIELD_CURRICULUM_STAGE=stage1a
LOCAL_RESIDUAL_FIELD_CURRICULUM_STAGE=select_consumer
LOCAL_RESIDUAL_FIELD_CURRICULUM_STAGE=stage1b
LOCAL_RESIDUAL_FIELD_CURRICULUM_STAGE=full_first_stage
```

`full_first_stage` is allowed, but internally it must still submit the
selector as an `afterok` dependency of `D_alpha_eval_Ofull` and
`D_alpha_eval_Orobust`, then submit Stage 1b as an `afterok` dependency of the
selector. It must not submit `P2/P4/P7a/P7b/Q0/Q3/G0` with a guessed consumer.

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

Submitter guardrails:

```text
default mode: first_stage_pilot
Stage 1a required oracle IDs: O0 Ofull Orobust_light
Stage 1a required alpha diagnostic IDs: D_alpha_eval_Ofull D_alpha_eval_Orobust
Stage 1a required deployable IDs: P0
Stage 1a required selector artifact: selected_consumer.json
Stage 1b required deployable IDs: P2 P4 P7a P7b
Stage 1b required ablation IDs: Q0 Q3
Stage 1b required fusion IDs: G0
max_gpu_training_jobs_total: 12
max_gpu_training_jobs_before_selector: 6
max_gpu_training_jobs_after_selector: 6
```

Dependency graph:

```text
O0, Ofull, Orobust_light, P0
  -> D_alpha_eval_Ofull, D_alpha_eval_Orobust
  -> select_lprf_curriculum_consumer
  -> P2, P4, P7a, P7b, Q0, Q3
  -> G0
  -> final report
```

The selector should choose:

```text
selected_consumer_id
selected_alpha_endpoint
```

using `model_val` diagnostics as the primary signal and `stack_val` only as a
confirmation/stability check. It should prefer the smoother, monotonic consumer
when model-val accuracy is close, and it should choose the strongest stable
alpha endpoint rather than blindly using `1.00`. `stack_val` may break close
model-val ties or flag brittle choices, but it must not be allowed to override
a weak model-val alpha response because `stack_val` is also used later for
fusion choice and pilot pass/fail.

Stage 1b submitter validation:

```text
P2/P4/P7a/P7b/Q0/Q3 must consume the same selected_consumer.json
Q0 must match selected P7a except w_oracle_path=0
Q3 must match selected consumer and selected_alpha_endpoint, with no ramp
G0 must fuse A0 with the best deployable Stage 1b P model
```

Full-family expansion requires:

```text
LOCAL_RESIDUAL_FIELD_CURRICULUM_CONFIRM_FULL_FAMILY=1
```

High-data requires both:

```text
LOCAL_RESIDUAL_FIELD_CURRICULUM_CONFIRM_HIGHDATA=1
LOCAL_RESIDUAL_FIELD_CURRICULUM_PILOT_REPORT_OK=1
```

The submitter should refuse to queue high-data if the pilot report does not
show the configured uplift threshold, unless an explicit override is set:

```text
LOCAL_RESIDUAL_FIELD_CURRICULUM_OVERRIDE_PILOT_GATE=1
```

Requeue policy:

```text
failed due to OOM: requeue once with larger memory or smaller batch
failed due to nonfinite validation: do not requeue until inspected
failed due to missing provenance/cache mismatch: do not requeue; fix inputs
terminated before run_report: requeue same run ID with OVERWRITE=1 after checking curves
```

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

1. Does the fixed-consumer alpha response curve rise smoothly from HLT-like
   behavior to the full oracle response?
2. Can a deployable predictor trained through weak fixed-consumer targets or
   genuinely weakened subset/robust consumers beat the HLT baseline?
3. Do curriculum students beat direct `alpha=0.75` students?
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

The best-performing single deployable setup hypothesis is:

```text
P8:
  HLT warm start
  residual predictor warm-started from Huber pretraining
  fixed-consumer alpha curriculum 0.25 -> 0.50 -> 0.75, and only then 1.00 if stable
  frozen oracle-consumer path KD
  masked field Huber
  confidence-gated residual fields
  offline-teacher KD only after the alpha ramp stabilizes
```

This is not an assumption for queue decisions. It becomes the preferred
expansion target only if the first-stage pilot shows:

```text
best(P7a, P7b) > A0 by >= 0.3 pp on stack_val
best(P7a, P7b) >= P4
Q0 < best(P7a, P7b), showing oracle-path KD matters
Q3 <= best(P7a, P7b), showing curriculum helps relative to direct selected-endpoint training
validation coverage >= 0.99
```

If those conditions do not hold, do not queue `P8` by default. Instead expand
around the best proven fixed consumer, alpha endpoint, and student
initialization from `O0`, `Ofull`, `Orobust_light`,
`D_alpha_eval_Ofull`, `D_alpha_eval_Orobust`, and
`P0/P2/P4/P7a/P7b`.

The best overall deployable setup hypothesis is:

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
