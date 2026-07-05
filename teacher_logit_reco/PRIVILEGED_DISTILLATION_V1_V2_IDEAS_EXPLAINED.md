# Privileged Distillation V1/V2 Ideas Explained

## Short Version

The PD10 privileged-distillation program is trying to make the deployable HLT
tagger better without requiring offline inputs at inference.

The deployment contract is always:

```text
HLT particles -> HLT-only student -> 10-class logits
```

During training, however, we can use matched offline information from the same
physical jet. The core idea is:

```text
offline or dual-view teacher sees more information during training
HLT-only student learns from that teacher
student is then used with HLT particles only
```

V1 tests the cleanest logit-distillation version of this idea. V2 asks whether
we need a stronger teacher and richer hidden-representation targets.

## Why Distillation Is A Natural Fit Here

The HLT view is degraded. Some constituents are dropped, smeared, merged, or
reassigned. That means an HLT-only model cannot literally recover all offline
information at inference.

But the student may still be improved by better training targets. Hard labels
say only:

```text
this jet is Hgg
```

A teacher distribution can say:

```text
this jet is mostly Hgg
but has QCD-like ambiguity
and should not be treated like a clean Wqq jet
```

That extra class-geometry signal can matter even with millions of labeled jets,
because high-data hard-label training can still leave calibration, class
boundary, and representation-shaping gains on the table.

## Shared Setup

The serious run uses the paired pseudo-HLT/offline JetClass setup:

```text
model_train: 5,000,000 jets
model_val:   1,000,000 jets
final_test:  1,000,000 jets
```

The task is full 10-class JetClass:

```text
QCD, Hbb, Hcc, Hgg, H4q, Hqql, Zqq, Wqq, Tbqq, Tbl
```

Every student is selected on `model_val`. `final_test` is only for the final
measurement after selection.

## V1: Logit Distillation From Three Teacher Types

V1 asks:

```text
Can better teacher logits improve a plain HLT-only ParT student?
```

The student architecture is the ordinary HLT Particle Transformer. The only
change is the training target.

### V1 Teacher 1: HLT Teacher

The HLT teacher sees only the deployable view:

```text
HLT particles -> HLT ParT teacher -> teacher logits
```

This is the self-distillation control.

If HLT KD helps, then KD itself is useful as smoothing or regularization. If
offline or dual-view KD helps more than HLT KD, then the privileged information
is contributing something beyond generic self-distillation.

### V1 Teacher 2: Offline Teacher

The offline teacher sees the privileged particle view:

```text
offline particles -> offline ParT teacher -> teacher logits
```

This teacher is not deployable. It answers two questions:

```text
How much better is the offline view?
Can an HLT-only student imitate some of that offline decision structure?
```

Offline KD may help because the teacher's soft labels encode cleaner class
boundaries. It may fail if the offline teacher relies on information that is
too absent from the HLT view.

### V1 Teacher 3: Dual-View Logit-Fusion Teacher

The V1 dual-view teacher sees both cached HLT-teacher logits and cached
offline-teacher logits:

```text
HLT teacher logits      z_hlt
offline teacher logits  z_off

[z_hlt, z_off, z_off - z_hlt, p_hlt, p_off, summary_features]
  -> small fusion MLP
  -> dual-view teacher logits
```

The summary features include entropy, margins, max probabilities, KL
divergence, and whether the two teachers agree on top class.

The exact fusion input is 58-dimensional:

```text
10 HLT logits
10 offline logits
10 offline-minus-HLT logits
10 HLT probabilities
10 offline probabilities
8 summary features
```

The model starts from the average of HLT and offline logits:

```text
base_logits = 0.5 * (z_hlt + z_off)
teacher_logits = base_logits + residual_mlp(features)
```

The final residual layer is zero-initialized, so the teacher starts as a safe
logit average and learns only useful corrections.

Why this might help:

```text
HLT teacher anchors the target to what the student can plausibly learn.
Offline teacher injects privileged class-boundary information.
Fusion teacher can learn when to trust HLT, when to trust offline, and when
their disagreement itself is informative.
```

This is why V1's highest-likelihood student is the warm-start dual-view KD
student.

## V1 Student Loss

For student logits `s`, teacher logits `t`, labels `y`, temperature `T`, and
KD weight `alpha`:

```text
CE = cross_entropy(s, y)

KD = T^2 * KL(
  softmax(t / T),
  softmax(s / T)
)

L = (1 - alpha) * CE + alpha * KD
```

The default is:

```text
T = 2.0
alpha = 0.5
```

Hard labels are always kept. This is important. The teacher is a training
signal, not a replacement for truth labels.

### Warmup

KD is warmed up:

```text
scratch students:    alpha ramps over 3 epochs
warm-start students: alpha ramps over 1 epoch
```

This avoids letting the teacher dominate before the student has stable class
structure.

## V1 Student Matrix

Core V1 students:

```text
pd10_student_scratch_ce_only
pd10_student_scratch_hlt_full_logits_t2_a0p5
pd10_student_scratch_offline_full_logits_t2_a0p5
pd10_student_scratch_dual_view_full_logits_t2_a0p5

pd10_student_warm_start_ce_only
pd10_student_warm_start_hlt_full_logits_t2_a0p5
pd10_student_warm_start_offline_full_logits_t2_a0p5
pd10_student_warm_start_dual_view_full_logits_t2_a0p5
```

Priority V1 dual-view ablations:

```text
pd10_student_warm_start_dual_view_full_logits_t4_a0p5
pd10_student_warm_start_dual_view_full_logits_t2_a0p3
pd10_student_warm_start_dual_view_top3_t2_a0p5
pd10_student_warm_start_dual_view_confidence_weighted_t2_a0p5
```

These answer:

```text
Does a higher temperature make the teacher signal smoother and more useful?
Does a lower alpha prevent over-regularization?
Does top-3 KD remove noisy low-probability classes?
Does confidence weighting help by trusting the teacher mainly when it is sure?
```

## Why Warm Start Is Important

At high data counts, the HLT baseline is already strong. The highest-likelihood
deployment path is not necessarily to train a distilled student from random
initialization. It is:

```text
train strong HLT ParT baseline
warm-start from that checkpoint
fine-tune with CE + privileged KD
```

This asks a sharper question:

```text
Can privileged supervision move an already strong HLT model to a better point?
```

The scratch runs are still useful science controls, but the warm-start runs are
the real deployment candidates.

## V2: Stronger Teacher Plus Representation KD

V2 asks:

```text
What if V1 logits are too compressed?
What if the logit-fusion teacher is too shallow?
```

It adds two ideas:

```text
1. A particle-level dual-view teacher.
2. Representation distillation from teacher hidden states.
```

## V2 Idea 1: Particle-Level Dual-View Teacher

The V1 dual-view teacher fuses final logits. That is cheap and auditable, but
it can only use information that the independent HLT/offline teachers already
expressed in their 10 logits.

The V2 particle dual-view teacher sees both particle clouds directly:

```text
HLT particles     -> HLT ParT branch     -> h_hlt
offline particles -> offline ParT branch -> h_off
```

Then it fuses learned embeddings:

```text
fusion_input = concat(
  h_hlt,
  h_off,
  abs(h_off - h_hlt),
  h_hlt * h_off
)

fusion_input -> fusion MLP -> teacher_repr -> classifier -> teacher_logits
```

The current architecture is:

```text
LayerNorm(4D)
Linear(4D, 512)
GELU
Dropout(0.05)
Linear(512, 256)
GELU
LayerNorm(256)
Linear(256, 10)
```

The teacher representation is:

```text
teacher_repr = L2_normalize(fusion_repr)
```

Why this might help:

```text
It can learn from actual paired HLT/offline constituents.
It can see degradation patterns before both views are collapsed to logits.
It can learn a compact privileged representation of "what the offline view
adds to the HLT view."
```

This teacher is still train-time only. It requires offline particles and is not
deployable.

## V2 Idea 2: Representation KD

Logits are only 10 numbers. They may not contain enough of the teacher's
privileged reasoning.

Representation KD gives the student a hidden target:

```text
teacher_repr: compact normalized teacher hidden vector
student_repr: projected normalized student hidden vector
```

The representation loss is:

```text
RepKD = mean(1 - cosine_similarity(student_repr, teacher_repr))
```

The combined V2 loss is:

```text
L = (1 - alpha) * CE
  + alpha * logit_KD
  + beta * RepKD
```

Defaults:

```text
T = 2.0
alpha = 0.5
beta = 0.10
```

Why this might help:

```text
It shapes the student's internal HLT representation, not just its final logits.
It can transfer more information than a 10-class softmax.
It may help the student organize HLT jets closer to the privileged teacher's
class-boundary geometry.
```

The student still predicts from HLT particles only. Teacher representations are
only training targets.

## V2 Student Matrix

Required V2 core:

```text
warm_particle_dual_logit_kd
  teacher = particle_dual_view
  loss = CE + logit KD

warm_particle_dual_rep_kd
  teacher = particle_dual_view
  loss = CE + representation KD

warm_particle_dual_logit_rep_kd
  teacher = particle_dual_view
  loss = CE + logit KD + representation KD

warm_logit_fusion_dual_logit_rep_kd
  teacher = V1 dual-view logit-fusion teacher
  loss = CE + logit KD + representation KD
```

Priority V2 ablations:

```text
warm_particle_dual_logit_rep_kd_beta0p3
warm_particle_dual_logit_rep_top3
warm_particle_dual_logit_rep_confidence
scratch_particle_dual_logit_rep_kd
```

The main V2 bet is:

```text
warm_particle_dual_logit_rep_kd
```

It combines:

```text
strongest train-time teacher
hard labels
soft logit targets
hidden representation targets
warm-start from HLT ParT
```

## What Each Comparison Tells Us

### HLT KD vs CE Only

If HLT KD beats CE only:

```text
KD itself is useful as smoothing, calibration, or regularization.
```

### Offline KD vs HLT KD

If offline KD beats HLT KD:

```text
privileged offline decision structure transfers to the HLT student.
```

If it does not:

```text
the offline teacher may be too unanchored to what HLT can imitate.
```

### Dual-View KD vs Offline KD

If dual-view KD beats offline KD:

```text
the best teacher is not "offline only"; it is teacher(HLT, offline).
```

That is the most important privileged-learning signal, because the teacher is
both stronger than HLT and more student-compatible than offline-only.

### Particle Dual-View KD vs Logit-Fusion Dual-View KD

If particle dual-view wins:

```text
there is useful paired particle-level information that final-logit fusion
could not capture.
```

### Logit KD vs Representation KD

If logit-only wins:

```text
the main useful transfer is final decision geometry.
```

If rep-only wins or ties:

```text
hidden teacher geometry may be a cleaner target than soft class probabilities.
```

If logit+rep wins:

```text
the student benefits from both final teacher beliefs and hidden teacher
representation structure.
```

### Warm Start vs Scratch

If warm-start wins:

```text
privileged KD is most useful as high-data fine-tuning of a strong HLT baseline.
```

If scratch wins:

```text
privileged KD improves the full training trajectory, not just fine-tuning.
```

## Why V1 Might Work

V1 might work because teacher soft targets provide smoother and more informative
class structure than hard labels:

```text
Hbb vs Hcc ambiguity
Wqq vs Zqq ambiguity
QCD vs boosted-object ambiguity
top-like class structure
```

The dual-view logit-fusion teacher is especially plausible because it can learn
when the offline teacher is adding useful information and when the HLT teacher
is already sufficient.

V1 is also relatively low-risk. It changes the training target, not the student
inference architecture.

## Why V2 Might Work Better

V2 might work better because V1 has two bottlenecks:

```text
10 logits are a small information channel.
logit fusion cannot recover particle-level paired-view structure.
```

The particle dual-view teacher can learn from raw HLT/offline paired
constituents, and representation KD gives the student a richer target than a
10-way softmax.

The best V2 story is:

```text
offline view teaches the teacher what information HLT lost
HLT branch teaches the teacher what the student still sees
fusion representation captures useful "offline correction" structure
student learns an HLT-only approximation to that structure
```

## What The Methods Cannot Do

None of these methods can make the HLT-only student use offline information at
inference. If a distinction is completely absent from the HLT view, the student
cannot recover it perfectly.

The realistic win condition is more modest and more useful:

```text
learn the best possible HLT-only decision function using better train-time
supervision.
```

## Expected Ranking Before Seeing Results

My prior for the highest-performing students is:

```text
1. warm_particle_dual_logit_rep_kd
2. warm_particle_dual_logit_rep_kd_beta0p3
3. pd10_student_warm_start_dual_view_full_logits_t2_a0p5
4. pd10_student_warm_start_dual_view_confidence_weighted_t2_a0p5
5. pd10_student_warm_start_offline_full_logits_t2_a0p5
6. pd10_student_warm_start_hlt_full_logits_t2_a0p5
7. pd10_student_warm_start_ce_only
```

That ranking is not guaranteed. At 5M labels, the HLT baseline may already be
close to the deployable information limit. But if there is still room, the
combination of warm start, dual-view privilege, and representation KD is the
most likely path to finding it.

## How This Connects To The AV10 Feature-MLP Adapter

The current V1/V2 student is still basically plain HLT ParT. The AV10
feature-MLP adapter suggests a different bottleneck:

```text
the HLT particle embedding itself may be improvable.
```

That suggests a natural next step after V1/V2:

```text
HLT-only student =
  HLT ParT + feature-MLP embedding residual adapter

training =
  CE + particle-dual-view logit KD + particle-dual-view representation KD
```

This would combine:

```text
better deployable HLT architecture
better train-time privileged teacher
better final-logit target
better hidden-representation target
```

If V1/V2 gains are modest but AV10 adapter gains are real, this combined V3 is
probably the highest-upside next experiment.

