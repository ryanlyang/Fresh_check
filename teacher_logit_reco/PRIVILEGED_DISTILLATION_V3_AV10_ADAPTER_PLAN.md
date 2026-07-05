# Privileged Distillation V3: AV10 Adapter Students

## Short Version

This plan combines the two most promising directions we have found so far:

```text
1. AV10 feature/input adapters:
   improve the deployable HLT ParT architecture itself.

2. Privileged distillation:
   train the deployable HLT-only student with offline/dual-view teacher signals.
```

The deployment contract remains strict:

```text
HLT particles -> HLT-only student -> 10-class logits
```

No offline particles are used at inference. Offline information is used only
during training through teacher logits and teacher representations.

The highest-upside candidate is:

```text
HLT particles
  -> canonical ParT inputs
  -> feature MLP embedding residual adapter
  -> warm-started HLT ParT
  -> 10-class logits

trained with:
  CE hard-label loss
  + V2 particle-dual-view teacher logit KD
  + V2 particle-dual-view teacher representation KD
```

This is the cleanest combined bet because the adapter and the teacher attack
different bottlenecks:

```text
adapter: improves the HLT student's particle embedding pathway
teacher: improves the training signal and hidden class geometry
```

## Core Hypothesis

The HLT ParT baseline may be limited by two different things:

```text
1. Architecture/representation bottleneck:
   the first ParT particle embedding may not be the best lift of degraded HLT
   particle features into transformer token space.

2. Supervision bottleneck:
   hard 10-class labels may not provide enough class-boundary and calibration
   structure, even at high data counts.
```

`av10_feature_mlp_adapter` addresses the first bottleneck:

```text
canonical per-particle feature row F_i
  -> small MLP + gate
  -> delta_h_i

ParT embedding h_i
  -> h_i + delta_h_i
```

Privileged distillation addresses the second:

```text
teacher sees HLT + offline during training
student sees HLT only
student learns from teacher logits and teacher representation
```

The V3 bet is that these gains should stack:

```text
better HLT-only student architecture
+ better privileged train-time supervision
= better deployable 10-class HLT tagger
```

## New HLT Degradation Regime

This plan should be run as a new HLT0.2 experiment family:

```text
HLT degradation strength: 0.2
```

This is intentionally different from the earlier HLT0.6 runs.

At HLT0.6, the degraded view can lose enough information that a privileged
teacher may teach distinctions the HLT-only student cannot infer. That can
make the experiment harder to interpret.

At HLT0.2:

```text
HLT is still meaningfully degraded,
but closer to the offline view.
```

That should make the teacher signal more learnable and the result more
defensible. If an HLT-only student beats the HLT ParT baseline at HLT0.2, the
claim is cleaner:

```text
we improved a realistic mildly degraded HLT tagger,
not just rescued an aggressively damaged input.
```

Do not compare HLT0.2 final numbers directly against HLT0.6 final numbers.
Compare them only qualitatively. Within this plan, all baselines, teachers,
students, caches, and reports must use the same HLT0.2 split and HLT0.2 cache.

## Data And Split Discipline

Use the same serious 10-class high-data split sizes as the PD10 setup:

```text
model_train: 5,000,000 jets
model_val:   1,000,000 jets
final_test:  1,000,000 jets
```

Task:

```text
10-class JetClass classification
```

Labels:

```text
QCD, Hbb, Hcc, Hgg, H4q, Hqql, Zqq, Wqq, Tbqq, Tbl
```

Split rules:

```text
model_train:
  train HLT baseline, offline teacher, dual-view teacher, and students.

model_val:
  select checkpoints, tune KD schedules if needed, and choose final models.

final_test:
  read exactly once for final selected reports.
```

The student variants in this plan should all use the exact same:

```text
split manifest
HLT0.2 cache
offline paired cache
label order
train/val/final-test rows
baseline checkpoint contract
```

This matters because the expected gains may be modest. Any split drift can
create fake wins or hide real ones.

## Required Baselines

Before the combined V3 variants, the run must establish these baselines on the
same HLT0.2 data:

### Baseline 1: Plain Warm-Started HLT ParT + CE

```text
variant: pdv3_hlt_part_ce

student:
  exact HLT ParT

training:
  warm-start from HLT0.2 baseline checkpoint if available,
  otherwise train the baseline normally from scratch.

loss:
  CE only
```

This is the anchor. Every V3 claim is measured against it.

### Baseline 2: Plain Warm-Started HLT ParT + V1 Dual-View Logit KD

```text
variant: pdv3_hlt_part_v1_dual_logit_kd

student:
  exact HLT ParT

teacher:
  V1 dual-view logit-fusion teacher

loss:
  CE + logit KD
```

This answers:

```text
How much does V1 privileged logit KD help without any adapter?
```

### Baseline 3: Plain Warm-Started HLT ParT + V2 Logit + Representation KD

```text
variant: pdv3_hlt_part_v2_logit_rep_kd

student:
  exact HLT ParT

teacher:
  V2 particle-dual-view teacher

loss:
  CE + logit KD + representation KD
```

This answers:

```text
How much does the best privileged teacher help the plain ParT student?
```

These two non-adapter KD baselines are essential. If the adapter+KD model wins,
we must know whether it beats:

```text
same student architecture without KD
same privileged KD without adapter
```

## Student Families

V3 tests three deployable HLT-only student families.

## Family A: Plain HLT ParT Students

These are control students. They do not have feature adapters.

```text
HLT particles
  -> exact HLT ParT
  -> 10-class logits
```

They establish how much privileged distillation alone can move the baseline.

Required variants:

```text
pdv3_hlt_part_ce
pdv3_hlt_part_v1_dual_logit_kd
pdv3_hlt_part_v2_logit_rep_kd
```

## Family B: Feature MLP Embedding-Residual Adapter Students

This is the main V3 architecture.

```text
HLT particles
  -> canonical ParT inputs
  -> ParT embedding h_i
  -> feature MLP predicts delta_h_i from canonical feature row F_i
  -> h_i + delta_h_i
  -> normal ParT attention/classifier
```

The adapter is HLT-only and deployable. It does not use offline information at
inference.

The adapter starts from exact baseline behavior:

```text
final adapter projection initialized to zero
initial delta_h_i = 0
initial logits ~= warm-start HLT ParT logits
```

Required variants:

```text
pdv3_feature_mlp_ce
pdv3_feature_mlp_v1_dual_logit_kd
pdv3_feature_mlp_v2_logit_rep_kd
pdv3_feature_mlp_v2_logit_rep_kd_frozen_start
```

The expected best candidate is:

```text
pdv3_feature_mlp_v2_logit_rep_kd
```

The frozen-start version tests whether it helps to first teach the adapter a
privileged residual while the ParT body is stable:

```text
epochs 1-2:
  freeze ParT
  train adapter/head with CE + KD

later epochs:
  unfreeze ParT
  fine-tune adapter + ParT with lower ParT LR
```

## Family C: LC MLP Delta Input-Feature Adapter Students

This is the input-space repair counterpart to the feature MLP embedding
adapter.

```text
HLT particles
  -> canonical feature rows F_i
  -> LC MLP predicts bounded delta_F_i
  -> adapted features F_i + delta_F_i
  -> exact HLT ParT
  -> 10-class logits
```

It answers a different question:

```text
Should the model improve ParT by changing hidden embeddings,
or by gently repairing canonical input features before ParT embeds them?
```

The LC MLP delta should:

```text
use zero-init final projection
use tanh-bounded feature deltas
use feature-wise delta scales
optionally freeze or strongly limit PID and geometry deltas
preserve original points, Lorentz vectors, and masks
```

Required variants:

```text
pdv3_lc_mlp_delta_ce
pdv3_lc_mlp_delta_v2_logit_rep_kd
pdv3_lc_mlp_delta_v2_logit_rep_kd_frozen_start
```

This is the highest-upside surprise candidate:

```text
pdv3_lc_mlp_delta_v2_logit_rep_kd_frozen_start
```

Why it might surprise:

```text
Privileged KD may be especially useful when the student can make small,
bounded input-space repairs before ParT embeds the particles.
```

Why it is riskier:

```text
If feature deltas become too large or physically inconsistent with unchanged
Lorentz vectors, the model may exploit artifacts instead of learning a robust
representation.
```

Therefore this family needs strong delta diagnostics.

## Family D: Combined LC Delta + Feature MLP Embedding Adapter Students

This is the new highest-capacity deployable student family. It combines the two
adapter mechanisms instead of treating them as competing alternatives.

```text
HLT particles
  -> canonical feature rows F_i
  -> LC MLP_F predicts bounded delta_F_i
  -> adapted feature rows F'_i = F_i + delta_F_i
  -> exact HLT ParT embedding h'_i
  -> separate feature MLP_H predicts gated delta_h'_i
  -> adapted embedding h'_i + delta_h'_i
  -> normal ParT attention/classifier
  -> 10-class logits
```

The two MLPs must be separate modules:

```text
MLP_F: feature row F_i -> bounded input-feature correction delta_F_i
MLP_H: adapted feature/context row -> embedding residual delta_h_i
```

They should not share weights. They operate at different abstraction levels and
answer different questions:

```text
delta_F asks:
  Can the student gently repair the canonical particle feature representation
  before ParT embeds it?

delta_h asks:
  Can the student add useful local/contextual information in ParT embedding
  space after the input representation is formed?
```

The reason this family is interesting is that the two corrections may be
complementary. The LC delta can make small low-level feature repairs, while the
feature MLP can still add residual information that is easier to express after
ParT has embedded the particle. Privileged V2 KD may then coordinate both
adapters toward the teacher's offline-aware decision surface.

Required variants:

```text
pdv3_lc_plus_feature_mlp_ce_joint
pdv3_lc_plus_feature_mlp_ce_staged
pdv3_lc_plus_feature_mlp_v2_logit_rep_kd_joint
pdv3_lc_plus_feature_mlp_v2_logit_rep_kd_staged
```

The two must-run variants are:

```text
pdv3_lc_plus_feature_mlp_ce_staged
pdv3_lc_plus_feature_mlp_v2_logit_rep_kd_staged
```

The CE staged variant is the key capacity/control run:

```text
If it improves over both single-adapter CE variants, the combined architecture
itself is useful even without privileged distillation.
```

The V2 KD staged variant is the main candidate:

```text
If it improves over feature MLP V2 and LC delta V2, then input repair,
embedding residual context, and privileged V2 supervision are complementary.
```

The joint variants are diagnostic:

```text
joint CE:
  trains both adapters from epoch 1 with CE only

joint V2 KD:
  trains both adapters from epoch 1 with CE + V2 logit KD + RepKD
```

They test whether the staged schedule is actually necessary. If joint training
matches staged training, the simpler training recipe may be enough. If staged
wins, it supports the hypothesis that the adapters need separated roles.

Recommended staged schedule:

```text
Phase 0: warm start
  load HLT0.2 ParT baseline
  initialize delta_F final projection to zero
  initialize delta_h final projection/gate to near-zero output

Phase 1: input repair warmup
  train MLP_F only
  freeze MLP_H
  freeze ParT
  objective: CE, or CE + V2 KD for KD variant
  keep delta_F L2/bounds active

Phase 2: embedding residual warmup
  freeze or strongly slow MLP_F
  train MLP_H only
  keep ParT frozen
  objective: same as variant

Phase 3: gentle joint fine-tune
  unfreeze MLP_F and MLP_H
  unfreeze ParT at low LR
  keep adapter LR > ParT LR
  keep delta_F regularization active
  monitor both delta_F and delta_h norms
```

Implementation should make the schedule explicit in the student spec, not
hidden in ad hoc trainer flags. The run report should record:

```text
combined_adapter = true
input_delta_adapter_active = true
embedding_delta_adapter_active = true
training_schedule = joint | staged
phase boundaries
per-phase trainable module groups
```

This family needs all diagnostics from both single-adapter families. It should
also report cross-adapter interaction:

```text
delta_F_l2_mean
delta_h_norm_mean
delta_F_to_feature_norm_ratio
delta_h_to_embedding_norm_ratio
correlation between per-jet |delta_F| and |delta_h|
fraction of jets where both adapters are active
per-phase adapter gradient norms
```

Failure modes to watch:

```text
Both adapters learn the same correction:
  staged should reduce this risk.

delta_F grows large while delta_h stays small:
  model may be overusing input repair.

delta_h grows large while delta_F stays near zero:
  combined model is effectively just feature MLP again.

joint training beats staged but only by overfitting model_val:
  require final_test confirmation and delta diagnostics.
```

## Teacher Families

## V1 Teacher: Dual-View Logit-Fusion Teacher

The V1 teacher fuses cached HLT and offline teacher logits:

```text
HLT teacher logits      z_hlt
offline teacher logits  z_off

[z_hlt, z_off, z_off - z_hlt, p_hlt, p_off, summary features]
  -> residual fusion MLP
  -> teacher logits
```

It starts from:

```text
0.5 * (z_hlt + z_off)
```

and learns residual corrections.

Why it is useful here:

```text
It is cheap, auditable, and anchored to both what HLT can see and what offline
knows better.
```

Use it for:

```text
pdv3_hlt_part_v1_dual_logit_kd
pdv3_feature_mlp_v1_dual_logit_kd
```

## V2 Teacher: Particle-Dual-View Teacher

The V2 teacher sees paired HLT and offline particles:

```text
HLT particles     -> HLT branch     -> h_hlt
offline particles -> offline branch -> h_off
```

Then it fuses:

```text
concat(h_hlt, h_off, abs(h_off - h_hlt), h_hlt * h_off)
  -> fusion MLP
  -> teacher_repr
  -> teacher logits
```

This teacher is train-time only.

Use it for:

```text
pdv3_hlt_part_v2_logit_rep_kd
pdv3_feature_mlp_v2_logit_rep_kd
pdv3_feature_mlp_v2_logit_rep_kd_frozen_start
pdv3_lc_mlp_delta_v2_logit_rep_kd
pdv3_lc_mlp_delta_v2_logit_rep_kd_frozen_start
pdv3_lc_plus_feature_mlp_v2_logit_rep_kd_joint
pdv3_lc_plus_feature_mlp_v2_logit_rep_kd_staged
```

The V2 teacher should be trained and cached on HLT0.2. Do not reuse HLT0.6
teachers or logits.

## Losses

## Cross Entropy

All students keep hard-label supervision:

```text
CE = CrossEntropyLoss(student_logits, y)
```

Hard labels are never removed. The teacher is an auxiliary signal, not a
replacement for truth.

## Logit KD

For student logits `s`, teacher logits `t`, temperature `T`, and KD weight
`alpha`:

```text
KD = T^2 * KL(
  softmax(t / T),
  softmax(s / T)
)

L = (1 - alpha) * CE + alpha * KD
```

Default:

```text
T = 2.0
alpha = 0.5
```

Primary ablations can later test:

```text
T = 4.0
alpha = 0.3
top-3 KD
confidence-weighted KD
```

But the first combined run should avoid too many degrees of freedom.

## Representation KD

Representation KD should be jet-level, not particle-by-particle.

Teacher:

```text
teacher_repr = normalized fused V2 teacher representation
```

Student:

```text
student_repr = normalized projection of student pooled/penultimate ParT representation
```

Loss:

```text
RepKD = mean(1 - cosine_similarity(student_repr, teacher_repr))
```

Combined V2 loss:

```text
L = (1 - alpha) * CE
  + alpha * LogitKD
  + beta * RepKD
```

Default:

```text
T = 2.0
alpha = 0.5
beta = 0.10
```

Why jet-level representation KD:

```text
It transfers privileged class-boundary geometry without requiring noisy
particle-level HLT/offline slot matching.
```

Do not start by asking the student to reconstruct offline particles. That is a
different experiment and can distract from the clean V3 question.

## KD Warmup

Use KD warmup so the teacher does not dominate unstable early training.

Suggested defaults:

```text
warm-start students:
  alpha ramps to target over 1 epoch
  beta ramps to target over 1-2 epochs

scratch teacher/student controls, if any:
  alpha ramps over 3 epochs
  beta ramps over 3 epochs
```

Because all serious students should warm-start from the same HLT0.2 ParT
baseline, the warm-start schedule is the important one.

## Warm Start And Freeze Schedule

Every serious student should warm-start from the same strong HLT0.2 HLT ParT
checkpoint.

For adapter students:

```text
adapter LR: larger
ParT LR:    smaller
```

Suggested defaults:

```text
adapter_lr = 3e-4
part_lr    = 1e-5
weight_decay = 1e-4
```

For normal adapter runs:

```text
freeze_part_epochs = 1 or 2
```

For explicit frozen-start variants:

```text
freeze_part_epochs = 2
then unfreeze ParT and continue fine-tuning
```

This lets the adapter first learn a useful residual direction relative to a
stable baseline representation.

For plain ParT KD baselines:

```text
freeze_part_epochs = 0
```

There is no adapter to learn during a freeze, so freezing the whole plain ParT
would mostly waste epochs.

## Required Variant Matrix

Run the following core matrix first:

```text
pdv3_hlt_part_ce
pdv3_hlt_part_v1_dual_logit_kd
pdv3_hlt_part_v2_logit_rep_kd

pdv3_feature_mlp_ce
pdv3_feature_mlp_v1_dual_logit_kd
pdv3_feature_mlp_v2_logit_rep_kd
pdv3_feature_mlp_v2_logit_rep_kd_frozen_start

pdv3_lc_mlp_delta_ce
pdv3_lc_mlp_delta_v2_logit_rep_kd
pdv3_lc_mlp_delta_v2_logit_rep_kd_frozen_start

pdv3_lc_plus_feature_mlp_ce_joint
pdv3_lc_plus_feature_mlp_ce_staged
pdv3_lc_plus_feature_mlp_v2_logit_rep_kd_joint
pdv3_lc_plus_feature_mlp_v2_logit_rep_kd_staged
```

This is enough to answer the central questions:

```text
Does privileged KD help plain ParT?
Does the feature MLP adapter help without KD?
Does feature MLP + KD stack?
Does input-feature repair + KD beat embedding residual KD?
Do input-feature repair and embedding residuals complement each other?
Does staged two-adapter training beat joint two-adapter training?
Does frozen-start matter?
```

## Expected Ranking Before Running

Prior expectation:

```text
1. pdv3_lc_plus_feature_mlp_v2_logit_rep_kd_staged
2. pdv3_feature_mlp_v2_logit_rep_kd
3. pdv3_lc_plus_feature_mlp_v2_logit_rep_kd_joint
4. pdv3_feature_mlp_v2_logit_rep_kd_frozen_start
5. pdv3_lc_mlp_delta_v2_logit_rep_kd_frozen_start
6. pdv3_lc_plus_feature_mlp_ce_staged
7. pdv3_lc_mlp_delta_v2_logit_rep_kd
8. pdv3_hlt_part_v2_logit_rep_kd
9. pdv3_feature_mlp_v1_dual_logit_kd
10. pdv3_hlt_part_v1_dual_logit_kd
11. pdv3_lc_plus_feature_mlp_ce_joint
12. pdv3_feature_mlp_ce
13. pdv3_lc_mlp_delta_ce
14. pdv3_hlt_part_ce
```

Why the combined staged V2 candidate is ranked first:

```text
It gives the student two separate deployable correction channels:
small bounded input-space repair and embedding-space residual context.
The staged schedule reduces adapter role collapse.
The V2 teacher provides the richest train-time signal for coordinating both.
```

Why the feature MLP V2 candidate still matters:

```text
The AV10 feature MLP adapter already appears to improve the deployable model.
The V2 teacher should provide the richest train-time supervision.
It remains the cleanest single-adapter comparison against the combined family.
```

Why LC MLP delta could win:

```text
The privileged teacher may guide the model toward small input-space repairs
that ParT can exploit more naturally than hidden delta_h residuals.
```

Why plain ParT V2 is still critical:

```text
If plain ParT + V2 KD matches the adapter models, then architecture changes are
less important than privileged supervision.
```

## Metrics

Primary 10-class metrics:

```text
accuracy
macro_per_class_accuracy
cross_entropy
macro AUC if available
per-class accuracy
confusion matrix
```

Selection:

```text
select best checkpoint on model_val accuracy
```

Final:

```text
evaluate selected checkpoint once on final_test
```

For comparisons, report:

```text
absolute accuracy gain
relative error reduction
macro per-class accuracy gain
per-class wins/losses
```

Relative error reduction is important because a 0.3 percent absolute accuracy
gain can be meaningful when the baseline is already strong:

```text
error_reduction = (baseline_error - model_error) / baseline_error
```

## Diagnostics

## Feature MLP Adapter Diagnostics

Report:

```text
delta_h_norm_mean
delta_h_norm_p90
delta_h_norm_max
adapter_output_norm_mean
adapter_output_norm_p90
gate_mean
gate_p10
gate_p90
embedding_norm_mean
delta_to_embedding_norm_ratio
adapter gradient norm
ParT gradient norm
active adapter parameter count
```

Key interpretation:

```text
Small but nonzero delta_h can be good.
Huge delta_h may mean the adapter is overpowering the warm-started ParT.
```

## LC MLP Delta Diagnostics

Report:

```text
delta_F_l2_mean
delta_F_l2_p90
delta_F_l2_max
delta_F_abs_mean
delta_F_abs_p90
delta_F_abs_max
per-feature delta_F RMS
per-feature delta_F p90
delta_F_to_feature_norm_ratio
PID/geometry active masks
feature-wise delta scales
```

Key interpretation:

```text
The adapter should make small, bounded corrections.
If it changes geometry/PID-like channels too aggressively, the result may be
harder to trust.
```

## Combined Adapter Diagnostics

For combined LC delta plus feature MLP variants, report every diagnostic from
both single-adapter families and add interaction diagnostics:

```text
delta_F_l2_mean
delta_h_norm_mean
delta_F_to_feature_norm_ratio
delta_h_to_embedding_norm_ratio
per-jet correlation between |delta_F| and |delta_h|
per-class mean |delta_F|
per-class mean |delta_h|
fraction of jets with nontrivial delta_F only
fraction of jets with nontrivial delta_h only
fraction of jets with both adapters active
MLP_F gradient norm by phase
MLP_H gradient norm by phase
ParT gradient norm by phase
```

Key interpretation:

```text
The best combined model should not simply make both adapters large.
The healthiest pattern is small input repairs, modest embedding residuals, and
different classes/jets using the two correction channels differently.
```

## Distillation Diagnostics

Report:

```text
CE loss
KD logit loss
RepKD loss
effective alpha by epoch
effective beta by epoch
teacher entropy
student entropy
teacher/student KL
teacher top-1 agreement with labels
teacher/student top-1 agreement
representation cosine similarity
```

Also report whether KD helps most on:

```text
teacher-confident jets
teacher/student disagreement jets
class-confusable jets
high-entropy jets
```

## Trust Checks

Before any serious final claim, verify:

```text
same split manifest hash for every baseline/teacher/student
same label names and label order
same HLT degradation strength = 0.2
same model_train/model_val/final_test sizes
same final_test held-out discipline
same baseline checkpoint hash for warm-started students
no offline tensors used in student forward at inference
```

For student checkpoints, save:

```text
student variant
teacher variant
teacher checkpoint hash
baseline warm-start hash
split manifest hash
HLT cache metadata hash
offline cache metadata hash
KD temperature
alpha
beta
warmup schedule
freeze schedule
```

## What Would Count As A Strong Result

The strongest result would be:

```text
pdv3_lc_plus_feature_mlp_v2_logit_rep_kd_staged
  beats pdv3_hlt_part_ce
  beats pdv3_hlt_part_v2_logit_rep_kd
  beats pdv3_feature_mlp_ce
  beats pdv3_lc_mlp_delta_v2_logit_rep_kd
  beats pdv3_feature_mlp_v2_logit_rep_kd
  holds on final_test
  improves macro per-class accuracy
  does not get gains from one weird class only
```

That would show:

```text
the adapter helps beyond privileged KD alone
privileged KD helps beyond adapter alone
input-space repair and embedding-space residual context are complementary
the staged combined model is real
```

An equally interesting result would be:

```text
pdv3_lc_mlp_delta_v2_logit_rep_kd_frozen_start
  beats the feature MLP embedding adapter
```

That would suggest:

```text
privileged teachers are most useful when the student can perform small,
bounded HLT feature repairs before ParT embedding.
```

## What Would Be A Negative Result

If:

```text
pdv3_hlt_part_v2_logit_rep_kd
  ~= pdv3_feature_mlp_v2_logit_rep_kd
```

then the adapter may not add much once privileged KD is used.

If:

```text
pdv3_feature_mlp_ce
  beats pdv3_feature_mlp_v2_logit_rep_kd
```

then the teacher may be over-regularizing the adapter student, or the offline
signal may not be student-compatible at HLT0.2.

If:

```text
pdv3_lc_mlp_delta_v2_logit_rep_kd
  wins but has huge feature deltas
```

then the performance may be real but the mechanism is less clean. Add stricter
delta bounds and rerun.

## Implementation Steps

## Step 1: HLT0.2 Paired Data And Cache Contract

Create or reuse a single HLT0.2 10-class split/cache family with:

```text
model_train: 5M
model_val:   1M
final_test:  1M
```

Persist:

```text
split_manifest.json.gz
HLT fixed-cache files and metadata
offline paired-cache files and metadata
label map
HLT degradation config
cache hashes
```

Hard-fail if any later stage sees a different split hash, label order, or HLT
degradation strength.

## Step 2: Train And Cache Teachers

Train or load:

```text
HLT ParT teacher/baseline on HLT0.2
offline ParT teacher on offline particles
V1 dual-view logit-fusion teacher
V2 particle-dual-view teacher
```

Cache:

```text
teacher logits for model_train/model_val/final_test
teacher representations for model_train/model_val/final_test
teacher metadata and checkpoint hashes
```

Do not use final_test for selection. Final-test teacher outputs can be cached
for final evaluation only, but no final-test metrics should guide training or
variant choice.

## Step 3: Implement Student Variant Registry

Add the V3 student variants:

```text
pdv3_hlt_part_ce
pdv3_hlt_part_v1_dual_logit_kd
pdv3_hlt_part_v2_logit_rep_kd
pdv3_feature_mlp_ce
pdv3_feature_mlp_v1_dual_logit_kd
pdv3_feature_mlp_v2_logit_rep_kd
pdv3_feature_mlp_v2_logit_rep_kd_frozen_start
pdv3_lc_mlp_delta_ce
pdv3_lc_mlp_delta_v2_logit_rep_kd
pdv3_lc_mlp_delta_v2_logit_rep_kd_frozen_start
pdv3_lc_plus_feature_mlp_ce_joint
pdv3_lc_plus_feature_mlp_ce_staged
pdv3_lc_plus_feature_mlp_v2_logit_rep_kd_joint
pdv3_lc_plus_feature_mlp_v2_logit_rep_kd_staged
```

Reuse the existing AV10 feature MLP adapter and LC MLP delta implementations
where possible. Do not fork the model logic unless the existing code cannot
express the needed student.

## Step 4: Implement KD Training Loop

Extend the student trainer to support:

```text
CE only
CE + V1 logit KD
CE + V2 logit KD
CE + V2 representation KD
CE + V2 logit KD + representation KD
```

Requirements:

```text
warm-start from HLT0.2 baseline checkpoint
separate adapter and ParT optimizer groups
KD alpha/beta warmup
optional freeze-start schedule
model_val checkpoint selection
strict teacher/student/cache alignment checks
```

## Step 5: Reporting And Comparisons

Write a final V3 report that compares:

```text
plain ParT CE
plain ParT V1 KD
plain ParT V2 KD
feature MLP CE
feature MLP V1 KD
feature MLP V2 KD
feature MLP V2 KD frozen-start
LC MLP delta CE
LC MLP delta V2 KD
LC MLP delta V2 KD frozen-start
combined LC delta + feature MLP CE joint
combined LC delta + feature MLP CE staged
combined LC delta + feature MLP V2 KD joint
combined LC delta + feature MLP V2 KD staged
```

The report should include:

```text
model_val metrics
final_test metrics
absolute gains over baseline
relative error reduction
per-class metrics
adapter diagnostics
KD diagnostics
teacher/student agreement diagnostics
runtime and parameter accounting
```

## Step 6: Implement Combined LC Delta + Feature MLP Family

Add the combined-adapter student family as a first-class V3 variant family, not
as an ad hoc special case in the training script.

The model path should be:

```text
raw HLT particles
  -> canonical ParT feature builder
  -> MLP_F input-feature delta adapter
  -> adapted canonical features F'
  -> exact ParT embedding path
  -> MLP_H feature-to-embedding residual adapter
  -> adapted embedding h' + delta_h
  -> normal ParT attention and classifier
```

`MLP_F` and `MLP_H` must be separate modules:

```text
MLP_F:
  input: canonical per-particle features F_i
  output: bounded delta_F_i
  applies before ParT embedding

MLP_H:
  input: adapted per-particle feature/context signal
  output: gated delta_h_i
  applies after ParT embedding
```

Do not share weights between the two MLPs. The whole point is to let them learn
different correction roles: `MLP_F` edits the input representation that ParT
will embed, while `MLP_H` supplies a residual latent context after the embedding
exists.

Required implementation details:

```text
reuse the existing LC MLP delta feature adapter for MLP_F
reuse the existing feature MLP adapter path for MLP_H
ensure MLP_H sees the adapted feature rows F', not stale pre-delta rows
keep ParT points, Lorentz vectors, and masks aligned with the canonical cache
zero-init both final adapter projections
apply bounded feature deltas with the existing feature-wise scales
track active and dormant parameter counts separately
save both adapter configs in the run report
```

Add these student specs:

```text
pdv3_lc_plus_feature_mlp_ce_joint
pdv3_lc_plus_feature_mlp_ce_staged
pdv3_lc_plus_feature_mlp_v2_logit_rep_kd_joint
pdv3_lc_plus_feature_mlp_v2_logit_rep_kd_staged
```

Joint training is the simple control:

```text
epoch 0 onward:
  train MLP_F
  train MLP_H
  train/fine-tune ParT using the normal adapter/ParT LR split
```

Staged training is the serious candidate:

```text
Phase 0: baseline-preserving initialization
  load the same HLT0.2 ParT baseline
  zero-init MLP_F final projection
  zero-init MLP_H final projection
  verify logits match the baseline before training

Phase 1: input repair warmup
  train MLP_F only
  freeze MLP_H
  freeze ParT
  use CE for CE variants
  use CE + warmed KD for V2 variants

Phase 2: embedding residual warmup
  freeze or very-slow-train MLP_F
  train MLP_H
  keep ParT frozen
  continue the same CE/KD objective

Phase 3: gentle joint fine-tune
  unfreeze MLP_F
  unfreeze MLP_H
  unfreeze ParT at low LR
  keep adapter LR larger than ParT LR
  select on model_val accuracy
```

Suggested first schedule:

```text
epochs 0-1: Phase 1
epochs 2-3: Phase 2
epochs 4+:  Phase 3
```

If the total pilot budget is small, compress this to:

```text
epoch 0: Phase 1
epoch 1: Phase 2
epoch 2+: Phase 3
```

The run report must store:

```text
combined_adapter = true
input_delta_adapter_active = true
embedding_delta_adapter_active = true
training_schedule = joint | staged
phase boundaries
phase-specific trainable module groups
phase-specific optimizer group learning rates
baseline-logit equality check before training
```

The training loop must expose module freezing by group:

```text
part
input_delta_adapter
embedding_delta_adapter
classifier/head
```

For staged variants, log per-epoch trainability and gradient norms for each
group. This is necessary to distinguish "both adapters helped" from "one
adapter carried the run while the other stayed dormant."

The report should compare:

```text
combined staged V2 vs feature MLP V2
combined staged V2 vs LC MLP delta V2
combined staged CE vs feature MLP CE
combined staged CE vs LC MLP delta CE
combined staged V2 vs combined staged CE
combined staged V2 vs combined joint V2
```

Interpretation:

```text
If combined staged V2 wins, the two deployable adapters and privileged
distillation are complementary.

If combined staged CE wins but V2 does not, the teacher is probably
over-constraining the larger student.

If joint beats staged, the roles do not need explicit separation.

If neither combined model beats the best single-adapter model, the extra
capacity is redundant or harder to optimize.
```

## Step 7: Queue Pilot Then Serious Run

First run a small pilot to validate mechanics:

```text
pilot model_train subset
pilot model_val subset
no final-test claim unless explicitly marked as pilot
```

Then run the full HLT0.2 high-data matrix:

```text
5M model_train
1M model_val
1M final_test
```

Queue order:

```text
1. build HLT0.2 split/cache
2. train HLT baseline and offline teacher
3. cache teacher logits/reprs
4. train V1/V2 teachers
5. train V3 students, including combined-adapter variants
6. write final report
```

Implemented queue entrypoints:

```text
sbatch/submit_pdv3_full_experiment.sh
  Queues one complete PDV3 campaign root:
    Step 1 inputs/cache/audit
    Step 2 HLT/offline teachers and V1/V2 teacher caches
    Step 4 all configured students
    Step 5 final report

sbatch/submit_pdv3_pilot_and_highdata.sh
  Queues both the pilot and the high-data campaign.
  By default the high-data campaign waits for the pilot final report via afterok.
```

Default Step 7 student matrix:

```text
pdv3_hlt_part_ce
pdv3_hlt_part_v1_dual_logit_kd
pdv3_hlt_part_v2_logit_rep_kd
pdv3_feature_mlp_ce
pdv3_feature_mlp_v1_dual_logit_kd
pdv3_feature_mlp_v2_logit_rep_kd
pdv3_feature_mlp_v2_logit_rep_kd_frozen_start
pdv3_lc_mlp_delta_ce
pdv3_lc_mlp_delta_v2_logit_rep_kd
pdv3_lc_mlp_delta_v2_logit_rep_kd_frozen_start
pdv3_lc_plus_feature_mlp_ce_joint
pdv3_lc_plus_feature_mlp_ce_staged
pdv3_lc_plus_feature_mlp_v2_logit_rep_kd_joint
pdv3_lc_plus_feature_mlp_v2_logit_rep_kd_staged
```

Queue both campaigns from the research-compute checkout:

```bash
cd /home/ryreu/atlas/Fresh_check
git pull --ff-only
export DEVICE=cuda
export CONDA_ENV=atlas_kd

bash sbatch/submit_pdv3_pilot_and_highdata.sh
```

To queue only the high-data run:

```bash
cd /home/ryreu/atlas/Fresh_check
git pull --ff-only
export DEVICE=cuda
export CONDA_ENV=atlas_kd
export PDV3_SUBMIT_PILOT=0
export PDV3_SUBMIT_HIGHDATA=1
export PDV3_HIGHDATA_AFTER_PILOT=0

bash sbatch/submit_pdv3_pilot_and_highdata.sh
```

## Final Scientific Claim This Plan Tests

The clean claim is:

```text
An HLT-only ParT student can be improved by combining:

1. a deployable input-feature repair adapter,
2. a deployable feature-to-embedding residual adapter, and
3. privileged dual-view distillation from offline information during training.
```

If the best V3 student beats:

```text
plain HLT ParT CE
plain HLT ParT privileged KD
feature MLP CE
feature MLP privileged KD
LC MLP delta privileged KD
```

then we have a genuinely stronger story than either AV10 or privileged
distillation alone.
