# Set-Matching Multi-View Reconstruction Plan

This document defines a new branch that deliberately turns away from the
teacher-logit reconstructor objective.  The teacher-logit branch asked:

```text
Can an HLT-only reconstructed view make a frozen offline teacher produce
offline-like logits?
```

This branch asks a more direct reconstruction question:

```text
Can diverse HLT-only reconstructors recover offline-like particle sets well
enough that a multi-view tagger can classify better than direct HLT taggers?
```

The high-level flow is:

```text
fixed HLT view -> ParT-style set reconstructor  -> reconstructed view A
fixed HLT view -> ParticleNet set reconstructor -> reconstructed view B
fixed HLT view -> PFN set reconstructor         -> reconstructed view C
fixed HLT view -> PCNN set reconstructor        -> reconstructed view D

[HLT view, A, B, C, D] -> five-view tagger -> class prediction
```

The offline view is used only as a training target for the reconstructors.  It
is never available to the final tagger at inference.

## Why This Branch Exists

The previous teacher-logit reconstructors gave useful but uneven behavior.
Some reco/teacher pairs produced strong class-specific gains, while many hurt
global accuracy.  That suggests the models may be learning useful correction
hypotheses, but the teacher-logit objective is not necessarily forcing them to
recover a physically faithful offline-like particle set.

This branch changes the premise:

```text
Do not ask the reconstructor to imitate the teacher's decision surface first.
Ask it to recover the offline particle set first.
```

The hope is that architecture diversity matters more under this objective:

- A ParT-style reconstructor can model long-range particle interactions.
- A ParticleNet-style reconstructor can model local neighborhood/subjet
  corrections.
- A PFN-style reconstructor can model global set-level shifts.
- A PCNN-style reconstructor can model ordered/local sequence patterns.

The final tagger then receives all four reconstruction hypotheses plus the
original HLT evidence.  This is not iterative refinement of one model's output;
it is a multi-hypothesis view of what the offline jet might have looked like.

## First Implementation Scope

The first implementation should focus on the reconstructors and their loss.
The final five-view tagger is intentionally left for the next section of this
plan.

Initial scope:

```text
1. Keep the four reconstructor encoder families:
   - global transformer / ParT-ish
   - ParticleNet / EdgeConv-ish
   - PFN / DeepSets-ish
   - PCNN / local convolution-ish

2. Replace the teacher-logit loss with an offline set-matching loss.

3. Keep a common output contract across all four architectures.

4. Train each reconstructor independently:
   fixed HLT view -> reconstructed offline-like particle set

5. Evaluate each reconstructed view before building the five-view tagger:
   - set-matching loss on heldout splits
   - multiplicity/count quality
   - jet-summary quality
   - simple downstream single-reco tagger quality
```

## Reconstructor Output Contract

Every reconstructor should output a fixed-size set of candidate offline
particles:

```text
predicted_features: [batch, max_slots, feature_dim]
existence_logits:   [batch, max_slots]
candidate_mask:     [batch, max_slots]
optional_metadata:  architecture-specific diagnostics
```

`max_slots` should be at least the current offline maximum used by the tagger,
probably 128 initially.  If offline multiplicity is often near the cap, we can
test 160 later, but the first pass should avoid changing too many knobs.

The model is not required to preserve one output slot per HLT input particle.
The four set reconstructors should be allowed to generate a full predicted set.
Architectures can still condition on HLT particles internally, but the loss
should treat the output as an unordered set.

This is important.  The objective should not encode:

```text
HLT particle i must become offline particle i
```

Instead it should encode:

```text
The predicted set should match the offline set up to permutation.
```

## Target Representation

The target is the offline particle set from the same jet:

```text
offline_features: [batch, max_offline_particles, feature_dim]
offline_mask:     [batch, max_offline_particles]
```

The actual JetClass full feature dimension may be larger than the handful of
physics variables we want to match strongly.  The loss should therefore split
features into groups:

```text
core matching features:
  log_pt
  eta
  phi represented as sin_phi and cos_phi, or angular delta_phi
  log_energy

weak auxiliary features:
  remaining continuous particle features

categorical/binary auxiliary features:
  particle ID / charge / one-hot-like fields, if present and reliable
```

The first version should not weight all features equally.  Matching all 19-ish
fields equally is likely brittle, because auxiliary fields can be sparse,
discrete, redundant, or scaled very differently from kinematic quantities.

The plan is:

```text
Use core physics features for Hungarian assignment.
Use core + weak auxiliary terms for matched regression.
Use discrete auxiliary losses only where the feature semantics are known.
```

## Main Loss: Hungarian Set Matching

The main reconstruction loss should be a hard-assignment set loss using the
Hungarian algorithm.

For each jet in a batch:

```text
predicted active slots: P = {p_i}_{i=1..M}
offline target particles: O = {o_j}_{j=1..N}
```

Build a pairwise cost matrix:

```text
C[i, j] = cost(predicted slot i, offline particle j)
```

Then run Hungarian assignment:

```text
matched_pairs = linear_sum_assignment(C)
```

The assignment itself is non-differentiable, but this is fine.  It is used only
to choose which predicted slots supervise which offline particles.  Gradients
flow through the selected prediction/target losses after matching.

### Why Hungarian Over Sinkhorn

Sinkhorn/OT is attractive because it is differentiable and soft, but in this
problem softness can be a liability.  Soft transport can reward blurred average
particles and smeared mass distributions.  We want crisp constituent recovery:

```text
this predicted slot corresponds to this offline particle
```

Hungarian matching gives a clear one-to-one assignment and a cleaner failure
mode.  It may be less elegant mathematically, but it is easier to audit and
less likely to hide bad reconstructions behind smooth fractional transport.

Sinkhorn can remain a later ablation, but it should not be the first primary
loss.

## Hungarian Cost Matrix

The assignment cost should be based mostly on normalized kinematics:

```text
C[i, j] =
    w_pt   * Huber(pred_log_pt_i - off_log_pt_j)
  + w_eta  * Huber(pred_eta_i    - off_eta_j)
  + w_phi  * phi_distance(pred_phi_i, off_phi_j)
  + w_e    * Huber(pred_log_e_i  - off_log_e_j)
  + w_aux  * weak_aux_distance(i, j)
```

Recommended first weights:

```text
w_pt  = 2.0
w_eta = 1.0
w_phi = 1.0
w_e   = 1.0
w_aux = 0.1
```

These are not sacred; they are starting values.  The key principle is that
matching should be driven by physical geometry and scale, not by every feature
equally.

### Phi Handling

Do not use raw `phi` subtraction without wraparound handling.

Preferred options:

```text
delta_phi = atan2(sin(phi_pred - phi_true), cos(phi_pred - phi_true))
phi_distance = Huber(delta_phi)
```

or represent phi as:

```text
sin_phi, cos_phi
```

and use:

```text
Huber(pred_sin_phi - true_sin_phi)
+ Huber(pred_cos_phi - true_cos_phi)
```

The second option is often more stable inside neural networks, but the first
option is more direct if the data already stores phi.

### Feature Normalization

All continuous matching features should be normalized using training-split
statistics:

```text
z = (x - mean_train) / std_train
```

This prevents `log_pt` or `log_energy` from dominating simply because of scale.
The same normalization constants must be saved into the run report/checkpoint
so prediction and evaluation are reproducible.

## Matched Particle Regression Loss

After Hungarian assignment, compute a regression loss over matched pairs:

```text
L_matched_core =
  mean over matched (Huber(pred_core_features - offline_core_features))
```

Use Huber/SmoothL1 rather than MSE.  Offline recovery will contain hard tails,
and MSE can let rare badly matched particles dominate training.

Suggested Huber beta:

```text
beta = 1.0 in normalized feature units
```

The matched core loss should be the dominant reconstruction term.

## Auxiliary Feature Loss

Auxiliary features should be included, but weakly at first:

```text
L_matched_aux =
  mean over matched (Huber(pred_aux_features - offline_aux_features))
```

with:

```text
aux_weight = 0.05 to 0.20
```

If some of the 19 features are categorical or one-hot-like, they should not be
treated as continuous values forever.  But the first pass can use weak Huber on
all non-core features while we audit whether those columns are meaningful.

Later, after feature semantics are locked down:

```text
continuous aux -> Huber
binary aux     -> BCEWithLogits
categorical aux -> cross entropy
```

## Existence And Multiplicity Loss

The reconstructor must learn how many offline particles to output.  Each output
slot therefore has an existence logit:

```text
existence_prob_i = sigmoid(existence_logit_i)
```

Targets after Hungarian matching:

```text
matched predicted slots   -> existence target 1
unmatched predicted slots -> existence target 0
```

The slot-existence loss:

```text
L_exist = BCEWithLogits(existence_logits, matched_slot_targets)
```

This term is essential.  Without it, the model can activate too many slots,
create duplicate particles, or hide behind arbitrary unused predictions.

There should also be a soft count loss:

```text
predicted_count = sum_i sigmoid(existence_logit_i)
offline_count   = number of offline particles

L_count = Huber(predicted_count - offline_count)
```

This helps the model learn multiplicity globally, not just slot-by-slot.

Recommended first weights:

```text
existence_weight = 1.0
count_weight     = 0.1
```

If the model overproduces particles, increase `count_weight` or unmatched
existence weighting.  If it underproduces, increase matched-slot positive
weight or lower the unmatched penalty.

## Handling Unmatched Slots

Unmatched predicted slots should primarily be supervised through existence:

```text
target existence = 0
```

Do not strongly regress unmatched slot features toward zero.  That can create
weird feature-space attractors.  It is enough that inactive slots have low
existence probability and are ignored by downstream view building.

Optional weak regularizers for inactive slots:

```text
inactive_feature_l2_weight = very small, e.g. 1e-4
inactive_pt_penalty        = small penalty on high-pt inactive slots
```

These should be disabled by default unless inactive slots become numerically
wild.

## Missing Target Penalty

If `max_slots >= max_offline_particles`, Hungarian can match every target
particle to some predicted slot.  If `max_slots < offline_count`, some offline
particles cannot be represented.  The first implementation should avoid this
case by choosing a sufficiently large `max_slots`.

If we later allow fewer slots than targets, we need an explicit missing-target
penalty.  For now:

```text
require max_slots >= target cap
```

and record in the run report:

```text
fraction of jets where offline_count == max_slots
```

If that fraction is high, increase `max_slots`.

## Jet Summary Loss

Set matching alone can recover local particles while still producing globally
wrong jets.  Add a weak jet-level summary loss:

```text
pred_jet_summary = summary(predicted active particles)
true_jet_summary = summary(offline particles)

L_jet = Huber(pred_jet_summary - true_jet_summary)
```

Useful summaries:

```text
sum_pt
sum_energy
pt-weighted eta
pt-weighted sin_phi/cos_phi
particle count
maybe simple mass-like or spread features if already available
```

This should be weak:

```text
jet_summary_weight = 0.05 to 0.10
```

It is a stabilizer, not the main objective.

## Correction Budget

Even with set matching, we should keep a budget term.  The model should not
learn absurd offline-like sets by making every predicted particle enormous or
wildly far from the HLT support.

The exact budget depends on output mechanics.  For a free slot-based
reconstructor, use broad but meaningful regularization:

```text
L_budget =
    mean predicted active pt outside reasonable range
  + mean large |eta| penalty
  + mean high active-slot count penalty beyond cap
  + optional distance-to-nearest-HLT support penalty
```

The important budget is not "stay close to particle i."  The new branch is
set-based, so the budget should be:

```text
stay physically plausible and HLT-supported
```

not:

```text
preserve parent indexing
```

Recommended starting value:

```text
budget_weight = 0.01 to 0.05
```

Too much budget will recreate the conservative reconstructor failure mode.  Too
little budget may generate unrealistic particles.

## Optional Chamfer Stabilizer

Chamfer distance can be used as a weak stabilizer:

```text
L_chamfer =
  mean_pred min_true distance(pred, true)
  + mean_true min_pred distance(true, pred)
```

But Chamfer should not be the main loss because it can tolerate duplicate
predictions and does not enforce one-to-one matching.

Recommended first choice:

```text
Hungarian main loss: enabled
Chamfer stabilizer: disabled or weight <= 0.02
Sinkhorn: disabled
```

If Hungarian training is unstable, enable weak Chamfer.  If Hungarian is too
slow, test approximate matching or batch-level optimizations before switching
to Sinkhorn.

## Proposed Total Loss

The first reconstructor loss should be:

```text
L_total =
    matched_core_weight      * L_matched_core
  + matched_aux_weight       * L_matched_aux
  + existence_weight         * L_exist
  + count_weight             * L_count
  + jet_summary_weight       * L_jet
  + correction_budget_weight * L_budget
  + chamfer_weight           * L_chamfer
```

Recommended first values:

```text
matched_core_weight      = 1.0
matched_aux_weight       = 0.10
existence_weight         = 1.0
count_weight             = 0.10
jet_summary_weight       = 0.05
correction_budget_weight = 0.02
chamfer_weight           = 0.00
```

Good ablations:

```text
loss_base:
  Hungarian core + existence + count

loss_with_summary:
  base + jet summary

loss_with_aux:
  summary + weak aux features

loss_with_chamfer:
  aux + weak Chamfer stabilizer
```

The first serious experiment should not tune all weights at once.  It should
start with `loss_with_summary` or `loss_with_aux`, then add Chamfer only if
training diagnostics say the set matching is unstable.

## Diagnostics Required For Every Reconstructor

Every training run should write diagnostics that tell us whether the
reconstructor is actually recovering offline sets:

```text
matched_core_loss by split
matched_aux_loss by split
existence accuracy / BCE by split
predicted count mean/std
offline count mean/std
count error mean/std
jet summary errors
fraction of high-existence slots
fraction of jets hitting max_slots
per-feature matched errors
```

Also write qualitative/statistical comparison summaries:

```text
offline vs reconstructed particle count histogram
offline vs reconstructed log_pt histogram
offline vs reconstructed eta histogram
offline vs reconstructed energy histogram
delta_phi / angular spread summaries
```

The five-view tagger should not be trained blindly.  We need to know whether
each view is physically plausible and where each architecture is making
different mistakes.

## Leakage Rules

The split discipline remains the same:

```text
model_train:
  train reconstructors

model_val:
  select reconstructor checkpoints / early stopping

stack_train:
  later train five-view tagger or fusers, depending on final design

stack_val:
  tune/select five-view tagger/fuser

final_test:
  locked final report only
```

Offline particles are allowed only as reconstruction targets while training
the reconstructors.  At inference and at final tagger evaluation, the deployed
inputs are:

```text
fixed HLT view only
```

The reconstructed views must be produced from fixed HLT alone.

## Option B: Five-View Tagger

The final tagger should not merge the four reconstructed views into one view
before classification.  Instead, it should classify from all five views:

```text
view 0: original fixed HLT view
view 1: aggressive ParT/GT set-matching reconstructed view
view 2: aggressive ParticleNet set-matching reconstructed view
view 3: aggressive PFN set-matching reconstructed view
view 4: aggressive PCNN set-matching reconstructed view

[view 0, view 1, view 2, view 3, view 4] -> five-view tagger -> class logits
```

This is the most direct test of the new hypothesis:

```text
Diverse reconstructors produce complementary offline-recovery hypotheses.
A multi-view particle transformer can learn which hypotheses matter for each
jet, class, and local particle configuration.
```

The tagger is still HLT-only at inference.  The offline view is used only to
train the four set-matching reconstructors.  At deployment:

```text
fixed HLT -> four frozen reconstructors -> five HLT-derived views -> tagger
```

No offline particles, offline labels beyond ordinary training labels, or
teacher logits are used at inference.

## Why Option B Is Preferred First

There are two plausible downstream choices:

```text
Option A:
  five views -> merge reconstructor -> one final reconstructed view -> tagger

Option B:
  five views -> multi-view tagger -> prediction
```

Option B should be implemented first because it avoids a premature bottleneck.
The four reconstructed views may contain incompatible but useful hypotheses.
Forcing them into one merged set can average away useful disagreement.  A
tagger can instead learn:

```text
When PN and PCNN agree on a local split, trust that evidence.
When PFN changes global radiation summaries, use that for QCD-like decisions.
When all reconstructed views disagree, fall back toward HLT.
When ParT/GT sees long-range structure, use it for multi-prong classes.
```

This is still a clean reconstruction story because the tagger consumes
particle-level reconstructed views, not just model logits.

## Token Definition

The tagger input is one token per particle per view.

If each view has up to `N` particles or candidate slots, and there are five
views, the flattened token set has up to:

```text
5 * N tokens
```

For example, with `N = 128`:

```text
HLT tokens:       128
GT reco tokens:   128
PN reco tokens:   128
PFN reco tokens:  128
PCNN reco tokens: 128
total:            640
```

This is not one "matched particle" with five embeddings.  We deliberately do
not manually match particles across views.  Each particle hypothesis remains
its own token:

```text
HLT particle 17 is one token.
PN reconstructed slot 42 is another token.
PCNN reconstructed slot 8 is another token.
```

If two tokens represent the same underlying offline particle, the transformer
can learn that through attention.

## Token Feature Construction

Each token should be built from:

```text
token_embedding =
    particle_feature_projection(full_particle_features)
  + view_embedding(view_id)
  + source_embedding(source_type)
  + confidence_projection(slot_confidence)
  + optional_rank_embedding(pt_rank_or_slot_rank)
```

### Particle Features

Use the full available particle feature vector as the base input, probably the
same 19-ish feature vector used elsewhere in the JetClass pipeline.  The
projection MLP should learn how to use the full vector, but the preprocessing
should keep core physics features well behaved:

```text
log_pt
eta
phi or sin_phi/cos_phi
log_energy
remaining auxiliary features
```

Recommended first representation:

```text
full_feature_vector -> normalization -> shared MLP -> token_dim
```

If phi is present as a raw angle, either:

```text
replace phi with sin_phi and cos_phi
```

or keep raw phi but make all pairwise geometry calculations wraparound-aware.

### View Embedding

Every token receives a learned view embedding:

```text
view_id = 0 -> HLT
view_id = 1 -> GT/ParT-style reco
view_id = 2 -> ParticleNet reco
view_id = 3 -> PFN reco
view_id = 4 -> PCNN reco
```

This tells the tagger where a token came from.  Without view embeddings, the
model sees an undifferentiated pile of particles and cannot easily learn that a
PN hypothesis and a PCNN hypothesis have different meaning.

View embeddings let the model learn rules like:

```text
PN-reco tokens are often useful for local splitting.
PFN-reco tokens are often useful for global set shifts.
HLT tokens are the original detector evidence and should be treated as anchor
evidence.
```

### Source Embedding

Use a small source-type embedding:

```text
source_type = original_hlt
source_type = reconstructed
```

This is partially redundant with `view_id`, but useful because it gives the
model a clean binary distinction:

```text
original detector input vs generated reconstruction hypothesis
```

### Confidence Feature

Reconstructed views have slot existence probabilities:

```text
confidence = sigmoid(existence_logit)
```

HLT particles should use:

```text
confidence = 1.0
```

The confidence should be passed as a continuous feature, either by:

```text
confidence_projection([confidence])
```

or by concatenating confidence before the particle-feature projection.

The model should know whether a reconstructed slot is highly trusted or barely
active.  This is especially important if we keep all predicted slots rather
than hard-thresholding them.

### Rank Or Slot Feature

An optional weak rank feature can be useful:

```text
pt_rank within view
normalized slot index
```

Do not overuse positional encodings.  Particle sets are unordered.  A `pt_rank`
feature can be reasonable because it is physics-derived; a raw slot index may
accidentally teach architecture-specific ordering artifacts.  If used, slot
index should be weak and ablated.

## Masking And Token Selection

The five-view tagger must handle variable particle counts.

For the HLT view:

```text
mask = real HLT particle mask
confidence = 1 for real particles
```

For reconstructed views:

```text
mask = candidate slot mask, optionally filtered by confidence
confidence = existence probability
```

There are two reasonable first modes:

```text
mode all_slots:
  keep all candidate slots
  pass confidence to the model
  use attention mask only for structurally invalid slots

mode topk_or_threshold:
  keep top K slots by confidence or slots above a threshold
  pass confidence to the model
```

Recommended first version:

```text
topk_or_threshold
```

with:

```text
max_tokens_per_view = 96 or 128
minimum confidence threshold = 0.03 to 0.05
always keep at least a small top-K per view
```

Reason: 640 tokens is feasible but expensive.  If the reconstructors emit many
low-confidence slots, all-slot attention wastes compute and may confuse early
training.  The confidence threshold should be recorded and ablated later.

## Best First Architecture: Two-Stage Multi-View Transformer

The recommended first serious architecture is a two-stage transformer:

```text
Stage 1: within-view encoding
Stage 2: cross-view reasoning
```

This is stronger and cleaner than immediately flattening all tokens into one
large transformer.

## Stage 1: Within-View Encoder

For each view independently:

```text
tokens_v = particle tokens from view v
encoded_tokens_v = shared_view_encoder(tokens_v, mask_v)
view_summary_v = attention/CLS pooled summary of encoded_tokens_v
```

The view encoder should be shared across views in the first version:

```text
same transformer weights for HLT, GT-reco, PN-reco, PFN-reco, PCNN-reco
different learned view embeddings
```

Why shared weights:

- The physical meaning of particle features is the same across views.
- Shared weights reduce parameter count.
- View embeddings still let the model treat views differently.
- It is a cleaner first comparison.

Later ablation:

```text
separate per-view encoders
```

but this should not be the first implementation.

### Stage 1 Inputs

For each view:

```text
view_tokens: [batch, max_tokens_per_view, token_dim]
view_mask:   [batch, max_tokens_per_view]
```

Add one view-level CLS token per view before the Stage 1 transformer:

```text
[view_CLS_v, particle_1, particle_2, ...]
```

The output CLS is `view_summary_v`.

### Stage 1 Geometry Bias

Stage 1 should ideally use geometry-aware attention, similar in spirit to the
Particle Transformer:

```text
pairwise features within one view:
  delta_eta
  delta_phi
  delta_R
  log_pt ratio
  maybe invariant-mass-like pair summaries
```

If reusing the Particle Transformer reference code is straightforward, this is
where to reuse its pairwise attention mechanism.  If not, implement a minimal
pairwise-bias MLP:

```text
pair_features -> bias per attention head
attention_logits += pair_bias
```

For a first smoke implementation, Stage 1 can run without pairwise bias.  For
the serious experiment, geometry bias is strongly preferred.

## Stage 2: Cross-View Transformer

After Stage 1:

```text
encoded_tokens = concat(
  HLT encoded particle tokens,
  GT encoded particle tokens,
  PN encoded particle tokens,
  PFN encoded particle tokens,
  PCNN encoded particle tokens
)

view_summaries = [summary_HLT, summary_GT, summary_PN, summary_PFN, summary_PCNN]
global_CLS = learned classification token
```

Stage 2 input:

```text
[global_CLS, view_summaries, encoded_tokens]
```

Then:

```text
cross_encoded = cross_view_transformer(stage2_input)
class_logits = classifier(cross_encoded[global_CLS])
```

Stage 2 is where the model learns cross-view agreement and disagreement.

## Stage 2 Cross-View Geometry Bias

Stage 2 should be geometry-aware too, but now pairwise relations can connect
particles from different views.

For particle-token pairs:

```text
delta_eta
delta_phi
delta_R
log_pt difference or ratio
same_view flag
view_pair embedding
confidence_i
confidence_j
confidence_i * confidence_j
```

For view summary tokens and global CLS tokens, use a default learned relation
type rather than particle geometry.

A practical implementation can split Stage 2 attention bias into:

```text
particle-particle geometric bias
summary/CLS relation-type bias
view-pair learned bias
```

The most important part is particle-particle geometry plus view-pair identity:

```text
PN token near PCNN token means something different from PN token near HLT token.
```

## Why No Manual Cross-View Matching

The model should not require a hand-built matching between particles across
views.  Manual matching would force us to decide:

```text
which PN particle corresponds to which PCNN particle?
which reconstructed particles correspond to which HLT particles?
```

That can be wrong, especially when one view splits a particle and another view
does not.

Attention is a better fit:

```text
softly compare all plausible particle hypotheses
learn class-dependent correspondences
learn when disagreement matters
```

This is one of the core reasons Option B is attractive.

## Classification Head

The first classification head should be simple:

```text
global_CLS -> LayerNorm -> MLP -> class_logits
```

Recommended:

```text
MLP hidden dim = 2 * token_dim
activation = GELU
dropout = 0.05 to 0.10
```

Training loss:

```text
L_class = CrossEntropy(class_logits, true_label)
```

No teacher logits in the first version.  If the architecture works, teacher
distillation can be an optional later ablation, not the core premise.

## Optional Auxiliary Heads

Optional auxiliary heads can help interpretability and stabilize training, but
they should be introduced carefully.

Useful auxiliary diagnostics:

```text
view_summary_v -> class logits for each individual view
```

Auxiliary loss:

```text
L_aux_view_cls = average CE(view_logits_v, true_label)
```

Start with:

```text
aux_view_cls_weight = 0.0
```

Then test:

```text
aux_view_cls_weight = 0.05
```

This can encourage each Stage 1 view summary to remain class-useful, but too
much auxiliary pressure may reduce cross-view specialization.

## Training Protocol For The Five-View Tagger

The reconstructors should be frozen while training the five-view tagger.

Recommended split usage:

```text
model_train:
  train the four set-matching reconstructors

model_val:
  select reconstructor checkpoints

stack_train:
  train the five-view tagger

stack_val:
  select/tune the five-view tagger

final_test:
  locked final evaluation
```

This is conservative and leakage-clean.  It avoids training the tagger on the
same jets used to fit the reconstructors.

If compute is tight, the tagger can be trained on `model_train` in an ablation,
but the primary result should use `stack_train` to keep the cleanest story.

## Caching Strategy

Five-view training can be expensive because every batch needs four frozen
reconstructor forward passes.  There are two options:

```text
on_the_fly:
  load frozen reconstructors
  generate four reconstructed views during tagger training

cached_views:
  precompute reconstructed views for stack_train, stack_val, final_test
  train tagger from cached HLT + cached reco views
```

Recommended first serious implementation:

```text
cached_views
```

Reasons:

- Faster tagger training.
- Easier debugging.
- Reconstructor outputs become auditable artifacts.
- Multiple tagger architectures/ablations can reuse the same views.
- Avoids repeated numerical differences from stochastic settings.

Cache format should include:

```text
tokens/features:        [num_jets, num_views, max_slots, feature_dim]
masks:                  [num_jets, num_views, max_slots]
confidences:            [num_jets, num_views, max_slots]
labels:                 [num_jets]
jet identities:         file/entry/label
view names:             hlt, gt_reco, pn_reco, pfn_reco, pcnn_reco
source checkpoints:     four reconstructor checkpoints
normalization metadata: feature means/stds, slot thresholds, caps
content hash:           reproducibility hash
```

The HLT view can be included as view 0 in the cache or loaded from the existing
HLT cache and combined at runtime.  Including it in the five-view cache makes
the tagger runner simpler and makes alignment audits easier.

## Main Ablations

The first implementation should include ablations that tell us where gains
come from:

```text
HLT-only transformer tagger:
  view 0 only

single reconstructed view taggers:
  HLT + GT reco
  HLT + PN reco
  HLT + PFN reco
  HLT + PCNN reco

five-view tagger without cross-view geometry bias:
  all five views, view embeddings, plain attention

five-view tagger with cross-view geometry bias:
  all five views, view embeddings, geometry-aware attention

five-view tagger without confidence features:
  tests whether slot confidence matters

five-view tagger with shuffled view labels:
  negative control for view identity
```

The most important comparison:

```text
HLT ParT / HLT4 fusion
vs
best single-reco view tagger
vs
five-view tagger
```

If five-view does not beat the best single-reco view, the cross-view machinery
is not earning its complexity.

## Expected Failure Modes

### Reconstructors Produce Plausible But Class-Irrelevant Sets

The set-matching losses may improve physical similarity without improving
classification.  This is why the single-view downstream tagger ablations are
necessary.

### One View Dominates

The five-view tagger may learn to ignore four views and use only HLT or one
reconstructor.  This is not necessarily bad, but it means the multi-view story
is weak.  Inspect attention summaries, view ablations, and drop-one-view
tests.

### Too Many Tokens

Five views at 128 tokens each can be expensive:

```text
640 tokens -> 409,600 pair interactions per jet per layer
```

Mitigations:

```text
top-K confidence filtering per reconstructed view
smaller Stage 1 token dimension
fewer Stage 2 layers
summary-token-only ablation
```

### Reconstruction Artifacts

The tagger may learn artifacts of the reconstructors instead of robust physics.
Controls:

```text
view-label shuffle
slot-confidence shuffle
drop reconstructed views
train on one split and verify stable gains on final_test only once
```

## Recommended First Hyperparameters

Start modestly:

```text
max_tokens_per_view = 128
token_dim = 128
stage1_layers = 2
stage1_heads = 4
stage2_layers = 4
stage2_heads = 4
mlp_ratio = 4
dropout = 0.05
attention_dropout = 0.05
batch_size = as large as GPU memory allows
epochs = 20 to 30
early_stop_patience = 5
optimizer = AdamW
lr = 0.0003 to 0.001
weight_decay = 0.0001
grad_clip_norm = 1.0
```

If memory is tight:

```text
token_dim = 96
stage2_layers = 3
max_tokens_per_view = 96
```

Do not start with a giant model.  We first need to prove that the input
structure is useful.

## Implementation Plan

### Step 1: Define The New Package Boundary

Create a new package or submodule namespace separate from teacher-logit code:

```text
set_matching_multiview/
```

or, if keeping the existing top-level package:

```text
teacher_logit_reco/set_matching/
```

The key is that the new branch should not silently reuse teacher-logit losses.
It can reuse architecture components, data loaders, and HLT cache utilities,
but the loss and experiment runners should be visibly separate.

Deliverables:

```text
set_matching config dataclasses
shared constants for view names
split/path layout helper
basic unit tests for layout and config
```

### Step 2: Implement Offline/HLT Paired Set Dataset

Build a dataset that returns both:

```text
fixed HLT particle set
offline target particle set
label
jet identity
```

It should respect the existing five-way split and fixed HLT cache rules.

Deliverables:

```text
SetMatchingJetDataset
collate function for variable masks
feature normalization statistic builder
split identity audit
tests with small synthetic fixtures
```

### Step 3: Implement Set-Matching Losses

Implement the Hungarian loss suite:

```text
Hungarian assignment on core normalized features
matched core Huber loss
weak auxiliary loss
existence BCE
count loss
jet summary loss
budget loss
optional Chamfer stabilizer
```

Use scipy's `linear_sum_assignment` if available.  If not, add a clear fallback
or fail with a helpful message.

Deliverables:

```text
set_matching_losses.py
loss report dict with individual components
tests for permutation invariance
tests for existence targets
tests for phi wraparound
```

### Step 4: Adapt The Four Aggressive Reconstructors

Reuse the aggressive reconstructor families as encoders/heads, but train them
with the new set-matching objective:

```text
aggressive GT / ParT-style
aggressive PN
aggressive PFN
aggressive PCNN
```

The output contract should be:

```text
predicted_features
existence_logits
candidate_mask
diagnostics
```

Deliverables:

```text
builder for set-matching reconstructor by architecture
checkpoint save/load
unit tests for output shapes and finite values
```

### Step 5: Write Reconstructor Training Script

Add one training script that handles all four architectures:

```text
scripts/train_set_matching_reconstructor.py
```

Arguments:

```text
--architecture gt|pn|pfn|pcnn
--hlt-cache-dir
--data-dir
--manifest-path
--output-dir
--max-train-jets
--max-val-jets
--loss weights
--max-slots
--confirm split settings
```

Deliverables:

```text
training loop
validation loop
early stopping on model_val set-matching score
run_report.json
best_model_val.pt
diagnostic CSV/JSON
```

### Step 6: Write Reconstructed-View Prediction/Cache Script

After four reconstructors are trained, cache their outputs for:

```text
stack_train
stack_val
final_test
```

Script:

```text
scripts/cache_set_matching_reco_views.py
```

Deliverables:

```text
five-view cache shards or NPZ files
metadata with source checkpoints
jet identity hashes
alignment audit
content hash
heldout set-matching metrics
```

### Step 7: Implement Five-View Dataset And Collator

Build a loader for:

```text
view_features:   [batch, 5, max_tokens_per_view, feature_dim]
view_masks:      [batch, 5, max_tokens_per_view]
view_confidence: [batch, 5, max_tokens_per_view]
labels:          [batch]
jet_ids
```

It should support:

```text
all slots
top-K per view
confidence threshold
drop selected views for ablations
view-label shuffle control
```

Deliverables:

```text
FiveViewJetDataset
collate function
tests for shape/alignment/filtering
```

### Step 8: Implement Five-View Tagger Model

Implement:

```text
FiveViewParticleTransformerTagger
```

with:

```text
shared particle feature projection
view embeddings
source embeddings
confidence projection
Stage 1 shared within-view encoder
view CLS tokens
Stage 2 cross-view encoder
global CLS classifier
```

Start with plain attention.  Add geometry-aware attention once the plain model
passes smoke tests.

Deliverables:

```text
model module
forward pass shape tests
mask tests
view embedding tests
save/load checkpoint tests
```

### Step 9: Add Geometry-Aware Attention Bias

Add pairwise bias support for:

```text
within-view geometry
cross-view geometry
view-pair learned relation type
summary-token relation type
```

This can be implemented in the five-view model directly or via a reusable
attention block.

Deliverables:

```text
pairwise feature builder
attention bias module
tests for delta_phi wraparound
tests for same-view/different-view relation ids
```

### Step 10: Write Five-View Tagger Training Script

Script:

```text
scripts/train_five_view_tagger.py
```

It trains on cached five-view data:

```text
stack_train -> train
stack_val   -> select checkpoint
final_test  -> evaluate only with explicit confirmation
```

Deliverables:

```text
training loop
validation loop
final_test guardrail
run_report.json
best_model_val.pt
per-class metrics
view ablation metrics
```

### Step 11: Write Baseline And Ablation Runners

Required comparisons:

```text
HLT-only tagger
HLT + each single reconstructed view
five-view tagger without geometry bias
five-view tagger with geometry bias
five-view tagger without confidence features
view-label shuffle control
```

Deliverables:

```text
scripts/evaluate_five_view_ablation.py
summary CSV/JSON
```

### Step 12: Write Slurm Runners

Add sbatch runners for:

```text
train four set-matching reconstructors
cache five-view reconstructed views
train five-view tagger
run ablations/audits
write final report
```

The runners must use isolated output roots, for example:

```text
checkpoints/set_matching_multiview_500k/
```

Do not write into the existing teacher-logit crossarch roots.

Deliverables:

```text
run_train_set_matching_reconstructor.sh
run_cache_set_matching_multiview.sh
run_train_five_view_tagger.sh
run_audit_five_view_tagger.sh
submit_set_matching_multiview_experiment.sh
```

### Step 13: Smoke Test

Before full training, run a tiny smoke experiment:

```text
model_train = 10k
model_val = 2k
stack_train = 5k
stack_val = 2k
final_test = 10k
epochs = 1 or 2
```

The smoke test is for pipeline correctness only.

Deliverables:

```text
all jobs complete
all reports write
all audits pass
no final_test use without confirmation
```

### Step 14: Full First Experiment

Run the first serious branch:

```text
500k model_train
150k model_val
500k stack_train
150k stack_val
500k final_test
```

Primary result:

```text
five-view geometry-aware tagger vs HLT ParT and HLT4 fusion
```

Secondary results:

```text
single-reco view ablations
drop-one-view ablations
per-class deltas
set-matching quality vs classification gain
```
