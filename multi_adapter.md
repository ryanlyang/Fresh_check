# Contextual AV10 Adapter and Offline Teacher Residual Representation Plan

## Short Version

The AV10 high-data results changed the interpretation of the adapter work.

The important empirical signal is no longer simply:

```text
feature MLP adapter understands particle features better than baseline ParT
```

The stronger and more honest signal is:

```text
a small residual adapter near the ParT embedding pathway improves HLT ParT,
but the gain may come from contextual adaptation, optimization geometry,
regularization, or representation reparameterization rather than from
per-particle feature semantics alone.
```

The clearest clue is that `av10_shuffled_feature_adapter` performed very well.
That control corrupts the input to the adapter but still leaves the main HLT
ParT pathway intact. If it performs similarly to the semantic feature adapter,
then the adapter is probably not winning only because it understands the exact
meaning of each particle feature column.

This plan turns that observation into a controlled experiment family:

```text
1. Test whether the adapter should see full-jet context.
2. Test whether the adapter can operate purely on ParT embeddings.
3. Test whether the gain is just fine-tuning of the warm-started ParT.
4. Test whether offline teacher information is best used as a residual
   jet-level representation target, not as particle reconstruction.
```

The central model family is:

```text
HLT particles
  -> canonical ParT features and masks
  -> ParT particle embedding h_i
  -> small contextual residual adapter predicts delta_h_i
  -> h_i + gated(delta_h_i)
  -> normal ParT blocks/classifier
  -> 10-class logits
```

The best privileged version adds:

```text
offline teacher ParT
  -> offline jet-level pre-classifier representation z_off

HLT student ParT
  -> HLT jet-level representation z_hlt
  -> predicts residual correction delta_z
  -> z_hlt + delta_z is trained toward z_off
```

The student remains deployable:

```text
inference input: HLT particles only
inference output: 10-class logits
```

Offline particles and offline teacher representations are used only during
training.

## Existing Evidence

The AV10 high-data ablation suite showed a cluster of adapter-like variants
beating the plain HLT ParT baseline.

The rough high-data pattern was:

```text
plain HLT ParT baseline:        about 0.7508 final-test accuracy
feature/context adapter family: about 0.7553 to 0.7557
larger vanilla ParT:            worse than baseline
```

Important interpretation:

```text
The lift over plain HLT ParT looks real.
The spread among adapter variants is small.
The shuffled adapter doing well weakens the "semantic feature repair" story.
The larger-ParT control doing poorly weakens the "just more parameters" story.
```

So the next experiment should not simply add more side architectures. It should
isolate what part of the adapter mechanism matters.

## Required Comparison Space

All runs in this plan should use the same data split family as the AV10
10-class architecture-view ablation work.

For the HLT0.6 comparison campaign:

```text
task: 10-class JetClass tagging
input view: fixed HLT particles
HLT degradation: same HLT0.6 cache/profile used by AV10 high-data
train split: same model_train split
validation split: same model_val split
stack/fusion split: same stack_val split if fusion is run
final test: same final_test split
selection metric: model_val accuracy
final-test access: only after model selection
```

This matters because the key question is:

```text
Do contextual adapters beat the exact AV10 feature MLP adapter and the exact
plain HLT ParT baseline under the same split/cache conditions?
```

Do not create a new split for this experiment unless the goal is explicitly a
new realism campaign, such as HLT v2.

## Model Objects And Notation

For one jet:

```text
F_i      canonical per-particle feature row for particle i
M_i      particle-valid mask
h_i      ParT particle embedding for particle i
H        sequence of particle embeddings, shape [N_particles, d_model]
z_hlt    HLT student jet-level pre-classifier representation
z_off    offline teacher jet-level pre-classifier representation
logits   10-class output logits
```

The standard HLT ParT baseline is:

```text
F_i, mask, pairwise inputs
  -> ParT embedding h_i
  -> ParT attention blocks
  -> pooled/class jet representation z_hlt
  -> classifier
  -> logits
```

The adapterized version inserts a small residual correction:

```text
h_i_adapted = h_i + alpha_h * delta_h_i
```

where:

```text
delta_h_i = adapter(...)
alpha_h starts at 0 or near 0
adapter final projection starts at 0
```

The model should recover the warm-started baseline at initialization.

## Design Principles

### 1. Preserve The Real HLT ParT Pathway

The main model remains a real HLT ParT.

The adapter should not replace ParT. It should provide a bounded residual
correction around ParT's embedding pathway.

### 2. Context Must Preserve Particle Identity

The context adapter should see the whole jet, but the output must still be
per-particle:

```text
input:  F_1 ... F_N or h_1 ... h_N
output: delta_h_1 ... delta_h_N
```

Particle `i` receives `delta_h_i`. The adapter can look at the whole jet, but
it must know which particle it is correcting.

### 3. Zero-Init And Gated Residuals

Every adapter should start as a no-op:

```text
delta_h_i = 0 at initialization
h_i_adapted = h_i
```

This makes warm-start comparisons meaningful and prevents the adapter from
destroying the baseline early in training.

### 4. Keep Adapters Small

The goal is not to build a second ParT. The adapter should be small enough that
the result cannot be dismissed as a hidden large model.

Report:

```text
trainable ParT parameters
trainable adapter parameters
total trainable parameters
adapter/ParT parameter ratio
```

### 5. Run Strong Controls

The shuffled adapter result means controls are not optional.

Every serious run family should include:

```text
fine-tune-only control
ParT-only adapter
semantic feature-context adapter
shuffled/noise context controls
larger-ParT control if available
```

## Experiment Family A: Clean Contextual Adapter Ablations

These are the first runs to implement. They answer what mechanism is actually
helping.

### A0. Plain HLT ParT Recheck

Purpose:

```text
Anchor the exact split/cache/checkpoint/training setup.
```

Architecture:

```text
HLT particles -> HLT ParT -> logits
```

No adapter.

This should match the existing `av10_hlt_baseline_recheck` number within
expected seed noise.

### A1. Fine-Tune-Only Baseline

Purpose:

```text
Test whether the AV10 gain is just warm-started ParT fine-tuning.
```

Architecture:

```text
HLT ParT baseline checkpoint
  -> same optimizer schedule as adapter variants
  -> no adapter modules
  -> logits
```

This is different from a passive baseline recheck. It should use the same
training budget and unfreezing schedule as the adapter runs, but with no
adapter. If this matches the adapter gains, then the adapter may not be the
source of improvement.

Required reporting:

```text
model_val accuracy
stack_val accuracy
final_test accuracy
number of epochs trained
learning rates
trainable parameter count
```

### A2. Feature MLP Adapter Repeat

Purpose:

```text
Repeat the existing av10_feature_mlp_adapter under the exact new campaign.
```

Architecture:

```text
F_i -> MLP -> gate -> delta_h_i
h_i + delta_h_i -> ParT blocks -> logits
```

This is the old local per-particle adapter:

```text
delta_h_i depends only on F_i
```

It does not see full-jet context except through ParT after injection.

### A3. ParT-Only MLP Adapter

Purpose:

```text
Test whether raw feature conditioning is needed at all.
```

Architecture:

```text
h_i -> MLP -> gate -> delta_h_i
h_i + delta_h_i -> ParT blocks -> logits
```

The adapter sees ParT's own initial embedding, not the raw feature row.

Interpretation:

```text
If A3 matches A2:
  the gain is likely embedding-space adaptation, not feature semantics.

If A2 beats A3:
  canonical features provide useful side information beyond h_i.
```

### A4. Feature DeepSets Context Adapter

Purpose:

```text
Test whether the adapter improves when it sees the whole jet.
```

Architecture:

```text
u_i = phi(F_i)
c_global = masked_mean_i(u_i)

delta_h_i = MLP([F_i, c_global, optional rank_i])
h_i_adapted = h_i + gate_i * delta_h_i
```

The adapter knows which particle it is correcting because `F_i` is still part
of the input. The global context vector tells it what kind of jet that particle
lives in.

Suggested extra inputs:

```text
log-pt rank or normalized particle index
particle-valid mask
optional pooled statistics: mean, max, sum of phi(F_i)
```

Keep this small. A good first version:

```text
phi width: 64 or 96
global context width: 64 or 96
delta MLP hidden width: 128
dropout: 0.05
```

### A5. Feature Self-Attention Context Adapter

Purpose:

```text
Give the adapter per-particle context through a tiny attention block.
```

Architecture:

```text
F_i -> feature projection
projected sequence -> 1 lightweight self-attention block
contextual output u_i -> MLP/gate -> delta_h_i
h_i + delta_h_i -> ParT blocks -> logits
```

This is the clean version of the intuition hinted at by the shuffled adapter:

```text
maybe the adapter should see other particles before deciding delta_h_i
```

Unlike the shuffled adapter, this keeps context principled and permutation
aware. Particle `i` gets an output derived from its own token attending to the
rest of the jet.

Suggested first settings:

```text
adapter dim: 64 or 96
heads: 4
layers: 1
MLP ratio: 2
dropout: 0.05
zero-init final delta projection
```

### A6. ParT-Embedding DeepSets Adapter

Purpose:

```text
Test jet context in ParT embedding space instead of raw feature space.
```

Architecture:

```text
u_i = phi(h_i)
c_global = masked_mean_i(u_i)

delta_h_i = MLP([h_i, c_global])
h_i_adapted = h_i + gate_i * delta_h_i
```

This may be stronger than feature DeepSets because `h_i` is already a learned
ParT representation of the particle.

Interpretation:

```text
If A6 beats A4:
  ParT's own embedding is the better substrate for adaptation.

If A4 beats A6:
  raw/canonical features contain useful correction information not preserved
  in the baseline embedding.
```

### A7. ParT-Embedding Self-Attention Adapter

Purpose:

```text
Test the strongest pure embedding-context adapter.
```

Architecture:

```text
h_1 ... h_N
  -> 1 lightweight self-attention adapter block
  -> contextualized v_i
  -> MLP/gate -> delta_h_i
  -> h_i + delta_h_i
  -> normal ParT blocks
```

This is likely one of the most important variants. It asks whether ParT wants a
small residual contextual refinement before the full transformer stack.

### A8. Within-Jet Shuffled Context Control

Purpose:

```text
Test whether contextual adapter gains survive when particle-feature semantics
are broken but batch leakage is avoided.
```

Architecture:

```text
same as A4 or A5
but adapter inputs are shuffled within each jet
```

Important distinction:

```text
Do not roll across the whole batch.
Do not let a particle receive features from a different jet.
```

This is cleaner than the earlier `av10_shuffled_feature_adapter`, because the
old cross-batch roll can introduce confusing batch-order effects.

Interpretation:

```text
If within-jet shuffle still works:
  semantics are less important than context/perturbation.

If within-jet shuffle collapses:
  the old shuffled win may have been an artifact of cross-batch rolling or
  structured leakage-like behavior.
```

### A9. Pure Noise Adapter Control

Purpose:

```text
Test whether learned random perturbation capacity alone helps.
```

Architecture:

```text
fixed random noise r_i
  -> same adapter/gate capacity
  -> delta_h_i
```

The noise should be deterministic per sample/particle if possible, not freshly
resampled at every evaluation call.

Interpretation:

```text
If pure noise helps similarly:
  the adapter may be acting like a generic regularized perturbation.

If pure noise fails but contextual adapters work:
  real particle/jet information matters.
```

## Experiment Family B: Offline Teacher Residual Representation Distillation

This is the best version of the offline teacher embedding idea.

The goal is not to reconstruct offline particles. The goal is to learn the
missing part of the offline teacher's jet-level belief state.

### Why Jet-Level Representation Is The Right Target

Per-particle teacher embeddings are shaped like:

```text
N_particles x d_model
```

That is messy for HLT/offline because:

```text
HLT and offline can have different particle counts
particle order may differ
particle correspondence is not guaranteed
```

Logits are shaped like:

```text
10
```

They are easy to distill but heavily compressed.

The best compromise is the pre-classifier jet-level representation:

```text
z = one vector per jet
```

This might be:

```text
pooled ParT representation
class-token representation
attention-pooling output
final hidden vector before classifier head
```

The exact name depends on the implementation, but the contract should be:

```text
z is one fixed-width vector per jet
z is computed before the final classifier
z is the representation used by the classifier to produce logits
```

### Predict A Residual, Not The Whole Representation

The student should not predict `z_off` from scratch.

The smart version is:

```text
z_hlt = HLT student jet representation
delta_z = residual projector(z_hlt or pooled adapter state)
z_corr = z_hlt + delta_z
```

Then train:

```text
z_corr should move toward z_off
```

This matches the whole residual-adapter philosophy:

```text
HLT already knows a lot.
Learn the missing correction.
```

### Residual Predictor Architecture

The jet-level residual predictor should be small relative to ParT, but it
should not be a toy linear layer.

Because `z_hlt` is already one vector per jet, a full transformer is not the
natural first choice. A transformer is useful when the model still has a
sequence:

```text
h_1 ... h_N
```

Once ParT has pooled to:

```text
z_hlt
```

the best first predictor is a gated residual MLP:

```text
z_hlt
  -> LayerNorm
  -> Linear d -> hidden
  -> SwiGLU or GELU MLP
  -> Dropout
  -> zero-init Linear hidden -> d
  -> learned scalar/vector gate
  -> delta_z

z_corr = z_hlt + delta_z
```

Recommended first settings:

```text
hidden width: 2d or 4d
activation: SwiGLU preferred, GELU acceptable
dropout: 0.05
final projection: zero initialized
gate: initialized so delta_z is zero or very small at step 0
delta penalty: small norm penalty on delta_z
```

The predictor is residual because the HLT representation is already useful.
The model should learn:

```text
what is missing from z_hlt relative to z_off
```

not:

```text
how to rebuild z_off from nothing
```

If this simple predictor works, a later ablation can try a class-gated or
mixture-of-experts residual predictor:

```text
z_hlt -> soft class/context gates
expert_k(z_hlt) -> delta_z_k
delta_z = sum_k gate_k * delta_z_k
```

But that should be a second-round experiment, not the first implementation.

### Best Distillation Loss

Use a combined objective:

```text
loss =
  CE(student_logits, labels)
  + lambda_logit * KD(student_logits, teacher_logits)
  + lambda_rep * RepKD(z_corr, z_off)
  + lambda_delta * residual_norm_penalty
```

Recommended starting losses:

```text
CE: normal 10-class cross entropy
logit KD: KL divergence with temperature 2 or 3
RepKD: cosine loss on normalized representations
optional RepKD MSE: small MSE after learned projection/norm
```

Preferred representation loss:

```text
rep_cosine_loss = 1 - cosine(normalize(z_corr), normalize(z_off_projected))
```

If dimensions differ:

```text
z_off_projected = frozen-or-trained linear projection(z_off)
```

or:

```text
z_corr_projected = linear projection(z_corr)
```

The projection must be trained only on `model_train` and selected on
`model_val`.

### Teacher-Free Final Test

Final-test evaluation must not require offline teacher caches.

Allowed during final-test primary evaluation:

```text
HLT particles
student checkpoint
student logits
labels for metrics
```

Not allowed during final-test primary evaluation:

```text
offline teacher logits
offline teacher representations
offline particles
```

Optional teacher diagnostics can be run later, but they must be marked as
non-selection diagnostics and kept separate from the primary final-test table.

### Freeze And Unfreeze Ladder

The biggest training choice is whether the HLT ParT is allowed to move while
the residual predictor learns.

The main concern is:

```text
if HLT ParT moves from step 0, then z_hlt changes while delta_z is trying to
learn a stable correction toward z_off
```

The opposite failure mode is:

```text
if HLT ParT stays frozen forever, the residual predictor may improve the
representation but cap final tagging performance
```

Therefore the plan should include a small schedule ladder rather than one
arbitrary choice.

Run these variants:

```text
R0: HLT ParT CE baseline
R1: HLT ParT + logit KD only
R2: frozen delta_z RepKD only, ParT frozen
R3: frozen warmup -> upper-unfreeze, CE + logit KD + RepKD
R4: frozen warmup -> full gentle-unfreeze, CE + logit KD + RepKD
R5: joint-from-start, CE + logit KD + RepKD
```

#### R0. HLT ParT CE Baseline

Purpose:

```text
anchor the HLT-only student without teacher help
```

Architecture:

```text
HLT particles -> HLT ParT -> logits
```

Loss:

```text
CE(labels)
```

No teacher caches are needed for training or evaluation.

#### R1. HLT ParT + Logit KD Only

Purpose:

```text
test whether teacher logits alone help without representation matching
```

Architecture:

```text
HLT particles -> HLT ParT -> logits
```

Loss:

```text
CE(labels) + lambda_logit * KD(student_logits, teacher_logits)
```

This is the cleanest distillation baseline.

#### R2. Frozen Delta-Z RepKD Only

Purpose:

```text
test whether a residual jet-level correction has signal when the HLT encoder
is fixed
```

Architecture:

```text
frozen HLT ParT -> z_hlt
residual predictor -> delta_z
z_corr = z_hlt + delta_z
classifier/logits
```

Trainable:

```text
delta_z predictor
representation projection if needed
classifier/head if explicitly enabled
```

Frozen:

```text
HLT ParT body
```

Loss:

```text
CE(labels) + lambda_logit * KD + lambda_rep * RepKD(z_corr, z_off)
```

This variant is scientifically clean. If it improves, the teacher
representation contains useful correction information that can be extracted
without moving the HLT encoder.

#### R3. Frozen Warmup Then Upper-Unfreeze

Purpose:

```text
best balance between stable residual learning and final performance
```

Schedule:

```text
Stage 1:
  freeze HLT ParT
  train delta_z predictor and intended head/projector

Stage 2:
  unfreeze only the upper K ParT blocks and classifier/head
  keep lower ParT blocks frozen
  keep delta_z predictor trainable
```

Loss:

```text
CE + logit KD + RepKD
```

This is the default recommended RepKD schedule. It lets the predictor first
learn a correction for a stable `z_hlt`, then lets the top of ParT co-adapt
without rewriting the whole baseline.

#### R4. Frozen Warmup Then Full Gentle-Unfreeze

Purpose:

```text
highest performance bet among the residual representation schedules
```

Schedule:

```text
Stage 1:
  freeze HLT ParT
  train delta_z predictor

Stage 2:
  unfreeze all of HLT ParT
  keep ParT LR much smaller than adapter/predictor LR
  keep delta_z predictor trainable
```

Loss:

```text
CE + logit KD + RepKD
```

Use a gentle ParT LR, for example:

```text
predictor LR: 3e-4
adapter LR:   3e-4
ParT LR:      1e-5 to 3e-5
```

Do not freeze the residual predictor after Stage 1 by default. If ParT moves,
`z_hlt` moves too, and the predictor should be allowed to track that shift.

#### R5. Joint-From-Start Control

Purpose:

```text
test whether staged training is actually necessary
```

Schedule:

```text
from epoch 0:
  HLT ParT trainable
  delta_z predictor trainable
```

Loss:

```text
CE + logit KD + RepKD
```

If R5 matches R3/R4, the staged schedule may not be necessary. If R5 is worse,
the residual predictor likely needs a stable HLT representation during warmup.

## Combined Best Model

The best full model to try after the clean ablations is:

```text
HLT particles
  -> canonical features F_i
  -> ParT embedding h_i
  -> contextual adapter predicts delta_h_i
  -> h_i + delta_h_i
  -> ParT blocks
  -> jet representation z_hlt
  -> residual projector predicts delta_z
  -> z_corr = z_hlt + delta_z
  -> classifier logits
```

Training supervision:

```text
hard labels
offline teacher logits
offline teacher jet-level representation z_off
```

Inference:

```text
HLT particles only
```

Recommended first combined variants:

```text
B0: HLT ParT CE baseline
B1: best contextual adapter, CE only
B2: HLT ParT + logit KD only
B3: HLT ParT + logit KD + residual RepKD
B4: av10_feature_mlp_adapter + residual RepKD schedule
B5: av10_feature_mlp_adapter + logit KD + residual RepKD
B6: best contextual adapter + logit KD
B7: best contextual adapter + logit KD + residual RepKD
B8: best contextual adapter + residual RepKD only
```

If compute is limited, run:

```text
B0, B1, B3, B5, B7
```

### Feature MLP Adapter Plus Offline Residual RepKD

The `av10_feature_mlp_adapter` should be explicitly combined with offline
teacher residual representation distillation because it is the current simple,
well-understood adapter baseline.

The clean training order is:

```text
Stage 1:
  load warm-started HLT ParT
  attach feature MLP delta_h adapter
  attach jet-level delta_z residual predictor
  freeze HLT ParT body
  train feature adapter + delta_z predictor + intended head/projector

Stage 2:
  unfreeze upper ParT blocks
  keep feature adapter trainable
  keep delta_z predictor trainable
  train CE + logit KD + RepKD

Stage 3 optional:
  unfreeze full ParT gently
  reduce KD/RepKD weights
  continue CE-dominant fine-tuning
```

The feature MLP adapter should come before the residual representation
predictor in the data flow:

```text
F_i -> feature MLP adapter -> delta_h_i
h_i + delta_h_i -> ParT blocks -> z_hlt
z_hlt -> residual predictor -> delta_z
z_corr = z_hlt + delta_z
```

Do not freeze the feature MLP adapter after Stage 1 by default. The same logic
applies as with `delta_z`: once ParT starts moving, the adapter should be able
to co-adapt.

Add one control where Stage 2 freezes the feature MLP adapter and trains only
upper ParT + delta_z. That control tests whether the adapter is a fixed
preconditioner or whether ongoing co-adaptation matters.

## Training Schedule

Use the same schedule across variants as much as possible.

### Stage 0: Load Warm Start

Load the same HLT ParT baseline checkpoint for all HLT student variants.

Record:

```text
baseline checkpoint path
baseline checkpoint hash
split manifest hash
HLT cache/profile identity
label mapping
selection metric used by baseline
```

### Stage 1: Adapter Warmup

For adapter variants:

```text
freeze most or all of ParT for 1 epoch
train adapter and classifier/head if intended
adapter LR: normal
ParT LR: 0
```

If the head is frozen, record that explicitly. If the head is trainable, record
that explicitly. Do not leave this ambiguous.

### Stage 2: Joint Fine-Tuning

```text
unfreeze ParT
adapter LR: e.g. 3e-4
ParT LR: e.g. 3e-5 or lower
weight decay: same as AV10
gradient clipping: same as AV10
early stopping: model_val accuracy
```

For distillation variants:

```text
start KD/RepKD weights low
optionally ramp them over first few epochs
```

Suggested initial weights:

```text
lambda_logit: 0.1 to 0.3
lambda_rep: 0.05 to 0.2
lambda_delta: small, e.g. 1e-4 to 1e-3
```

Do not let RepKD dominate CE unless model-val proves it helps.

### Stage 3: Selection

Select checkpoints by:

```text
primary: model_val accuracy
secondary: model_val loss
tie-breaker: smaller adapter norm / simpler model
```

Do not select on final_test.

## Diagnostics To Require

Every run should report:

```text
model_val accuracy/loss
stack_val accuracy/loss
final_test accuracy/loss
per-class accuracy
confusion matrix
macro accuracy
trainable parameter counts
adapter parameter counts
```

Adapter variants should also report:

```text
delta_h_norm_mean
delta_h_norm_p90
delta_h_norm_max
embedding_norm_mean
delta_to_embedding_norm_ratio
gate_mean
gate_p10
gate_p90
adapter_output_norm_mean
adapter_output_norm_p90
```

Context variants should report:

```text
context_norm_mean
context_norm_p90
attention entropy if self-attention adapter
mask handling checks
batch-size sensitivity check for shuffled/noise controls
```

Distillation variants should report on model_val:

```text
teacher/student top-1 agreement
teacher entropy
student entropy
KD loss
RepKD loss
representation cosine
representation MSE if used
accuracy by teacher-confidence bucket
accuracy on teacher/student disagreement cases
```

Final-test primary report should remain teacher-free.

## Interpretation Rules

### If Fine-Tune-Only Matches The Adapter

Then the old AV10 gain was probably mostly fine-tuning schedule, not adapter
architecture.

Next step:

```text
do not add more adapter complexity until fine-tune-only is beaten
```

### If ParT-Only Adapter Matches Feature Adapter

Then raw feature conditioning is not necessary.

Next step:

```text
focus on embedding-space adapters and adapter placement
```

### If Contextual Adapters Beat Local Feature MLP

Then the shuffled result may have been hinting that nonlocal jet context matters.

Next step:

```text
promote best contextual adapter into PDV3-style distillation
```

### If Shuffled/Noise Controls Also Win

Then the mechanism is likely regularization, capacity, or optimization geometry.

Next step:

```text
run multi-seed controls
check batch-size sensitivity
avoid physical claims about feature semantics
```

### If Residual RepKD Helps On Top Of Best Adapter

Then the offline teacher representation is giving useful training signal beyond
hard labels.

Next step:

```text
make best contextual adapter + logit KD + residual RepKD the new mainline
```

### If RepKD Hurts

Then the offline representation target is either too hard, misaligned, or
overweighted.

Next step:

```text
reduce lambda_rep
use logit KD only
try representation projection/norm changes
```

## Implementation Steps

### Step 1: Add Contextual Adapter Variant Specs

Add variant names and config for:

```text
av10_finetune_only_control
av10_part_only_mlp_adapter
av10_feature_deepsets_context_adapter
av10_feature_self_attention_context_adapter
av10_part_embedding_deepsets_adapter
av10_part_embedding_self_attention_adapter
av10_within_jet_shuffled_context_adapter
av10_noise_context_adapter
```

Keep them in the same architecture-view/AV10 code path where possible so data
loading, metrics, and reports stay comparable.

### Step 2: Implement Adapter Modules

Implement:

```text
FeatureDeepSetsContextAdapter
FeatureSelfAttentionContextAdapter
PartEmbeddingDeepSetsAdapter
PartEmbeddingSelfAttentionAdapter
NoiseContextAdapter
WithinJetShuffleContextAdapter
```

All modules must:

```text
respect particle masks
return per-particle delta_h
zero-init final projection
support diagnostics
support active parameter accounting
```

### Step 3: Add Fine-Tune-Only Control

Implement a no-adapter training variant that uses the same warm start, optimizer
schedule, epoch budget, and selection path as adapter variants.

This is essential for interpreting all adapter gains.

### Step 4: Add Clean Reports

Extend the AV10 ablation report to compare:

```text
plain HLT baseline
fine-tune-only
old feature MLP adapter
new contextual adapters
shuffled/noise controls
larger ParT if available
```

The report should explicitly answer:

```text
did fine-tune-only explain the gain?
did ParT-only adapter explain the gain?
did full-jet context improve over local MLP?
did shuffled/noise controls collapse?
```

### Step 5: Implement Offline Teacher Residual RepKD Hooks

Capture and cache:

```text
offline teacher logits
offline teacher jet-level pre-classifier representation z_off
HLT baseline/student jet-level representation z_hlt
```

Use jet-level vectors, not per-particle matrices.

Implement the residual predictor as:

```text
LayerNorm(z_hlt)
  -> SwiGLU/GELU residual MLP
  -> zero-init projection
  -> learned gate
  -> delta_z

z_corr = z_hlt + delta_z
```

Add residual representation loss:

```text
z_corr = z_hlt + delta_z
RepKD = 1 - cosine(normalize(z_corr), normalize(z_off_projected))
```

Keep final-test primary evaluation teacher-free.

### Step 6: Add Residual Representation Freeze/Unfreeze Ladder

Add:

```text
R0_hlt_part_ce_baseline
R1_hlt_part_logit_kd
R2_frozen_delta_z_repkd
R3_delta_z_upper_unfreeze_repkd
R4_delta_z_full_gentle_unfreeze_repkd
R5_delta_z_joint_from_start_repkd
```

The implementation must expose:

```text
phase name
which ParT blocks are trainable
whether classifier/head is trainable
whether delta_z predictor is trainable
optimizer LR by group
KD/RepKD weights by phase
```

### Step 7: Add Combined Context Adapter + Distillation Variants

Add:

```text
context_adapter_ce
hlt_part_logit_kd
hlt_part_logit_kd_repkd
feature_mlp_adapter_repkd
feature_mlp_adapter_logit_kd_repkd
feature_mlp_adapter_logit_kd_repkd_freeze_adapter_after_warmup
context_adapter_logit_kd
context_adapter_logit_kd_repkd
context_adapter_repkd_only
```

If the context adapter winner is unknown, use the best model-val performer from
Experiment Family A.

For the feature MLP adapter combination, train in this order:

```text
1. attach feature MLP delta_h adapter
2. attach jet-level delta_z predictor
3. freeze ParT body for warmup
4. train feature adapter + delta_z predictor
5. unfreeze upper ParT blocks
6. keep both adapters trainable by default
7. optionally run the freeze-adapter-after-warmup control
```

### Step 8: Add Slurm Submitters

Submitters should support:

```text
pilot run
high-data run
reuse existing AV10 split/cache
all contextual ablations
optional distillation variants
strict checkpoint/cache identity checks
final report with dependencies
```

The submitter must record:

```text
run root
variant list
manifest path
HLT cache path/profile
baseline checkpoint path/hash
teacher checkpoint path/hash if used
source commit
```

## Recommended First Queue

If compute is limited, run this first:

```text
A0 plain HLT ParT recheck
A1 fine-tune-only baseline
A2 feature MLP adapter repeat
A3 ParT-only MLP adapter
A4 feature DeepSets context adapter
A5 feature self-attention context adapter
A7 ParT-embedding self-attention adapter
A8 within-jet shuffled context control
A9 pure noise adapter control
```

Then, only after reading that report:

```text
R0 HLT ParT CE baseline
R1 HLT ParT + logit KD only
R2 frozen delta_z RepKD only, ParT frozen
R3 frozen warmup -> upper-unfreeze, CE + logit KD + RepKD
R4 frozen warmup -> full gentle-unfreeze, CE + logit KD + RepKD
R5 joint-from-start, CE + logit KD + RepKD
```

Then run the most important adapter-plus-teacher combinations:

```text
B1 best contextual adapter CE
B5 av10_feature_mlp_adapter + logit KD + residual RepKD
B5-control av10_feature_mlp_adapter + logit KD + residual RepKD
           with feature adapter frozen after warmup
B7 best contextual adapter + logit KD + residual RepKD
```

## Expected Outcomes

My current expectation is:

```text
ParT-embedding self-attention adapter and feature self-attention adapter are
the highest-upside clean variants.
```

The most important diagnostic outcome is not only the top accuracy. It is the
mechanism:

```text
If context adapters win and controls fail:
  full-jet context is genuinely useful.

If ParT-only wins:
  adapter placement matters more than raw feature conditioning.

If fine-tune-only wins:
  the old result was mostly schedule/fine-tuning.

If noise/shuffle wins:
  this is a PEFT/regularization phenomenon and should be studied as such.

If residual RepKD stacks on the best adapter:
  offline teacher jet-level belief-state correction is the best next mainline.
```

## Bottom Line

The next best step is not to add more architecture views.

The next best step is to test whether the adapter should:

```text
1. see the whole jet,
2. operate directly on ParT embeddings,
3. learn a residual correction toward an offline teacher jet-level
   representation.
```

The cleanest final hypothesis is:

```text
HLT ParT is already strong, but a small contextual residual adapter can move
its particle/jet representation into a better local function class. Offline
teacher distillation should be used to shape that residual representation,
not to reconstruct particles.
```
