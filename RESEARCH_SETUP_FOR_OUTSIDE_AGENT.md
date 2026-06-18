# Research Setup For Outside Agent

## One-Sentence Problem Statement

We are working with paired JetClass jets where each original/offline jet has a
matched HLT-like degraded version, and the research goal is to achieve the best
possible jet classification using only the HLT-like view at inference while
using the richer offline view only as a training-time signal.

## Core Setup

The base data source is JetClass.  Each event/jet belongs to one of the usual
JetClass classes:

```text
QCD, Hbb, Hcc, Hgg, H4q, Hqql, Zqq, Wqq, Tbqq, Tbl
```

For each jet, we maintain two aligned views:

```text
offline view:
  The richer original JetClass constituent representation.  This is treated as
  the high-quality detector/offline reconstruction view.

HLT view:
  A deterministic cached degradation of the same jet, meant to approximate a
  lower-fidelity trigger/HLT-like reconstruction.  This view has fewer and/or
  altered constituents due to thresholding, merging, efficiency effects,
  reassignment, and smearing.
```

The two views are matched one-to-one by jet identity.  For a given row index in
a split:

```text
HLT jet i <-> offline jet i <-> label i
```

This alignment is central.  The offline view is not an unpaired target domain;
it is paired privileged information for the exact same jet.

## Inference Constraint

At deployment/inference time, the model must consume only the HLT-like view.

Allowed at inference:

```text
cached HLT tokens
cached HLT mask
model weights/checkpoints
fixed preprocessing constants learned from training data
```

Not allowed at inference:

```text
offline constituents
offline masks
offline teacher logits computed from the offline view
offline-only metadata that identifies the true high-quality particle set
```

The entire research question is about whether offline information can improve
an HLT-only inference system through training, architecture, reconstruction, or
fusion.  Any method that requires the offline view at inference is solving the
wrong problem.

## Why Offline Is Still Useful

Although offline is unavailable at inference, it is available during training.
We use it as privileged training information in several possible ways:

```text
1. As a reconstruction target:
   HLT -> reconstructor -> synthetic offline-like view
   Compare synthetic view to matched offline view during training.

2. As a teacher signal:
   offline view -> frozen offline teacher -> offline logits
   HLT -> reconstructor -> synthetic view -> same frozen teacher -> synthetic logits
   Train the reconstructor so synthetic logits match offline logits.

3. As a label-supervised source view:
   Train strong offline taggers to understand the ceiling/reference behavior.

4. As an evaluation reference:
   Compare HLT-only systems against offline-only upper references.
```

The key distinction is:

```text
offline during training/evaluation: allowed
offline during inference prediction path: forbidden
```

## HLT Degradation

The HLT view is generated once per split and cached.  All models in a given
experiment must use the exact same cached HLT arrays.  We do not regenerate HLT
inside individual model jobs.

The current fixed HLT profile is defined in `jetclass_fixed_hlt.py` and includes
effects such as:

```text
constituent pT thresholding
constituent merging/unmerging loss
efficiency turn-on
barrel/endcap efficiency differences
kinematic smearing
constituent reassignment
```

The cached HLT files record hashes and metadata so we can verify that every
model saw the same HLT corruption.

## Feature Representation

The repository stores particle constituents as fixed-size padded token arrays:

```text
tokens: [num_jets, max_constituents, raw_feature_dim]
mask:   [num_jets, max_constituents]
label:  [num_jets]
```

In this implementation, the cached raw token dimension is currently 14.  It
contains kinematics and particle/track-like auxiliary fields, including:

```text
pt, eta, phi, energy,
charge,
isChargedHadron, isNeutralHadron, isPhoton, isElectron, isMuon,
d0, d0err, dz, dzerr
```

Downstream taggers transform these raw fields into architecture-specific input
features.  For example, Particle Transformer style inputs derive features such
as log pT, log energy, log pT fraction, deltaR, deta/dphi, and related
constituent features.

Important: the exact raw feature count is not the conceptual point.  The
conceptual point is paired particle sets:

```text
HLT particle set with lower fidelity
offline particle set with higher fidelity
same jet, same label, same split row
```

## Split Discipline

We use strict split roles to avoid leakage.  The common full-size split design
is:

```text
model_train = 500000 jets
model_val   = 150000 jets
stack_train = 250000 jets
stack_val   = 50000 jets
final_test  = 500000 jets
```

Roles:

```text
model_train:
  Train base taggers, offline teachers, reconstructors, and dual-view taggers.

model_val:
  Early stopping and checkpoint selection for base models/reconstructors.

stack_train:
  Train fusion/stacking models after base predictions are frozen.

stack_val:
  Select fusion hyperparameters or compare fusion methods.

final_test:
  Locked final evaluation only.
```

For debug jobs, we often reduce these counts drastically, for example:

```text
train = 20000
val   = 5000
test  = 20000
```

The smaller debug counts are for pipeline/proof-of-concept runs only.  They do
not replace the strict split logic.

## Leakage Rules

These rules are non-negotiable:

```text
1. A final-test label must not affect training, early stopping, fusion weight
   selection, hyperparameter selection, threshold selection, or debugging
   decisions for a run claimed as final.

2. Offline constituents may supervise reconstructors on model_train/model_val.
   They must not be read by prediction code for stack_train, stack_val, or
   final_test.

3. Offline teacher logits are allowed as a training target for a reconstructor
   only if they are computed from model_train/model_val offline views.
   They must not be included as inference-time features.

4. HLT cache must be shared across all models in a comparison.  Different model
   variants must not receive different HLT corruptions.

5. Fusion/stacking must train on held-out prediction splits, not on the same
   rows used to train the base models.
```

If a proposed method violates these rules, it is not comparable to our target
setting.

## Baselines And References

The most important references are:

```text
HLT-only tagger:
  A model trained and evaluated directly on the HLT view.  This is the baseline
  we need to beat.

Offline-only tagger:
  A model trained/evaluated on the offline view.  This is an upper reference,
  not deployable under the HLT-only inference rule.

HLT architecture ensemble:
  Multiple HLT-only models such as Particle Transformer, ParticleNet, PFN, and
  P-CNN trained on the same HLT view and fused.

Reconstructor-based model:
  HLT -> reconstructor -> synthetic/corrected view -> tagger.

Dual-view model:
  HLT -> reconstructor -> corrected view
  [original HLT view, corrected view] -> tagger.

Multi-view model:
  HLT -> several architecture-diverse reconstructors -> several reconstructed
  views
  [HLT, reconstructed view A, reconstructed view B, ...] -> tagger.
```

## Model Families We Care About

We have been exploring four broad architecture families:

```text
Particle Transformer / ParT-style:
  Strong global attention over particles.  Usually the strongest HLT-only
  baseline.

ParticleNet / EdgeConv-style:
  Local neighborhood/subjet-style inductive bias.

PFN / DeepSets-style:
  Permutation-invariant set summary with simpler global structure.

P-CNN-style:
  Local ordered/sequence-like convolutional inductive bias.
```

The motivation is that an ensemble of different architectures may expose
different failure modes and complementary class evidence.  A pure ensemble of
many copies of the same architecture may be too redundant.

## Reconstructor Strategies Tried Or Considered

### 1. Teacher-Logit Reconstruction

This strategy trains an HLT-only reconstructor so that a frozen offline teacher
reacts to the reconstructed view similarly to how it reacts to the true offline
view.

Training signal:

```text
offline view -> frozen offline teacher -> offline logits
HLT -> reconstructor -> synthetic view -> frozen offline teacher -> reco logits

loss = KL(offline teacher distribution || reco teacher distribution)
     + CE(reco logits, true label)
     + correction budget terms
     + weak jet-summary consistency
```

Inference:

```text
HLT -> reconstructor -> synthetic view -> teacher/tagger prediction
```

This uses offline as privileged supervision but still obeys HLT-only inference.

### 2. Operation-Aware V2/m2-Hybrid Reconstruction

This strategy uses an explicit mechanism:

```text
edit existing HLT constituents
split possible merged HLT constituents
generate missing constituents
assign soft weights/budgets
build a parent-aligned corrected view for a dual-view tagger
```

The important distinction is that this reconstructor has interpretable
operation branches rather than just emitting a generic set.

### 3. Set-Matching Reconstruction

This strategy trains reconstructors to directly match the offline particle set:

```text
HLT -> reconstructor -> predicted particle set
offline view -> target particle set
loss = Hungarian/set matching + count/existence + jet-summary + budget terms
```

This moves away from matching teacher logits and asks for a physically closer
offline-like particle set.

### 4. Multi-View Reconstruction/Tagging

This strategy trains several diverse reconstructors and lets a final model see
their different hypotheses:

```text
HLT view
ParT-style reconstructed view
ParticleNet-style reconstructed view
PFN-style reconstructed view
P-CNN-style reconstructed view
       -> multi-view transformer/tagger -> prediction
```

The point is not iterative refinement of one view.  The point is to expose
architecture-diverse hypotheses about what the offline jet might have been.

## Current Research Tension

HLT-only Particle Transformer models are already strong, especially at large
training counts.  Simple knowledge distillation and same-architecture
ensembling tend to give limited gains when enough HLT data is available.

So the real challenge is not:

```text
Can we train a slightly better HLT Particle Transformer?
```

It is closer to:

```text
Can offline paired supervision and architecture diversity induce additional
HLT-only features or representations that a direct HLT tagger does not easily
learn?
```

This is why we are interested in ideas from:

```text
learning using privileged information (LUPI)
teacher/student training with privileged modalities
missing-modality reconstruction
sensor degradation and restoration
set-to-set reconstruction
multi-view learning
mixture-of-experts / conditional fusion
uncertainty-aware ensembling
domain adaptation from high-quality to low-quality observations
```

## What We Want An Outside Agent To Research

The outside research question is:

```text
Given paired high-fidelity and low-fidelity particle-set observations, labels,
and an inference constraint that only the low-fidelity observation is available,
what machine learning strategies have previously worked to recover or exploit
the high-fidelity information at training time?
```

Useful prior art may come from outside particle physics too.  Relevant analogies
include:

```text
training with privileged sensors but deploying with cheap sensors
teacher models with privileged inputs
cross-modal distillation where one modality is missing at inference
image/video restoration used only as an intermediate representation
multi-hypothesis reconstruction followed by discriminative classification
latent-variable models for corrupted observations
set prediction with Hungarian matching
learned denoising/imputation where the downstream task is classification
```

We especially care about methods that can plausibly beat a very strong
low-fidelity baseline, not just methods that work when the low-fidelity model is
weak.

## What Would Count As A Useful Suggestion

A useful suggestion should specify:

```text
1. The training objective.
2. The inference path, explicitly confirming it uses only HLT.
3. How paired offline/HLT supervision is used.
4. How leakage is avoided.
5. Why the method might beat a strong direct HLT Particle Transformer.
6. What ablation would prove the gain is not just extra parameters or leakage.
```

For example, a vague suggestion like:

```text
Use knowledge distillation.
```

is not enough.  We have already considered basic distillation-like approaches.
A better suggestion would explain what is different:

```text
Use privileged offline teacher uncertainty to train a conditional mixture of
HLT-only experts, where gating is trained on HLT-only features and the offline
teacher is used only to shape expert specialization on model_train/model_val.
```

That kind of proposal can be tested under our constraints.

## Summary Of The Hard Constraint

The entire setup can be reduced to this:

```text
Training:
  See HLT, offline, and labels for matched jets.

Inference:
  See only HLT.

Goal:
  Classify as close as possible to offline-level performance, while being
  demonstrably better than strong direct HLT-only taggers and HLT-only
  architecture ensembles.
```

Any proposed method should be evaluated through that lens.
