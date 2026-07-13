# Constrained Coarse-To-Fine Pseudo-Offline Reconstruction Plan

## Core Idea

The goal is to improve HLT-only 10-class tagging by giving the tagger a
deployable, constrained pseudo-offline particle view.

The central problem with direct reconstruction is that offline particles are
not uniquely recoverable from HLT particles. HLT can merge, drop, smear, and
misclassify constituents. A model that directly hallucinates particles can
easily become hard to justify:

```text
HLT particles -> invented offline particles
```

This plan makes the hallucination principled by changing the task into
hierarchical accounting:

```text
HLT particles
-> global offline-like additive totals
-> coarse spatial allocations of those totals
-> finer spatial allocations
-> pseudo-particle slots that exactly account for final-cell totals
```

The model does not freely invent energy, composition, or multiplicity. It
predicts fractions and bounded local parameters. A deterministic accounting
layer turns those predictions into values that satisfy parent-child
consistency by construction.

The final tagger sees two views:

```text
original HLT particles
constrained pseudo-offline particles
```

The HLT view remains the trusted source. The pseudo-offline view is an
auxiliary, uncertainty-marked view that may recover statistically predictable
offline structure.

## Scientific Story

This is not an oracle reconstruction method. At inference time the model uses
only HLT particles.

The method learns offline-like structure during training because offline
particles are available as privileged supervision. At deployment it predicts
the most likely missing offline structure from the HLT jet alone.

The story is:

```text
HLT degradation removes detail.
Some lost detail is statistically predictable from the remaining jet.
Exact particle recovery is ill-posed.
Coarse-to-fine additive accounting is more stable and physically interpretable.
The final tagger learns when to trust or ignore the pseudo-offline view.
```

This is a reconstructor or hallucinator, but one with hard accounting
constraints.

## Data Regime

Use the same 10-class JetClass setup as the current high-data HLT campaigns
unless explicitly overridden.

Primary recommended setting:

```text
HLT profile: fixed_hlt_v2_realistic
HLT degradation strength: 2.5
model_train: 5M
model_val: 1M
stack_train: 2M
stack_val: 1M
final_test: 1M
```

Pilot setting:

```text
model_train: 500k
model_val: 150k
stack_train: 300k
stack_val: 150k
final_test: 150k
```

Final-test rules:

```text
No final-test offline targets may be used for model selection.
No final-test offline targets may be loaded by deployable prediction paths.
Final-test pseudo-offline particles must be generated from HLT particles only.
```

## Representation Philosophy

The hierarchy predicts additive nonnegative accounting channels wherever
possible.

Bad target style:

```text
predict jet axis directly
predict charged fraction directly
predict arbitrary particle list directly
```

Preferred target style:

```text
predict additive pT totals
predict additive composition-pT totals
predict additive expected counts
predict additive moment sums
derive axis/fractions/widths from those sums
render pseudo-particles only after the accounting field is predicted
```

This lets the network reason freely internally while the output transform
enforces consistency.

## Additive Accounting Channels

The full accounting vector should be rich enough to help tagging, not merely
easy to reconstruct.

### Core Positive Channels

Use nonnegative channels that can be allocated exactly through softmax
fractions:

```text
total_pT
total_energy
expected_constituent_count
charged_pT
neutral_hadron_pT
photon_pT
electron_pT
muon_pT
charged_count
neutral_hadron_count
photon_count
electron_count
muon_count
```

The composition channels should be amounts, not only percentages. Fractions
can be derived later:

```text
charged_fraction = charged_pT / total_pT
photon_fraction  = photon_pT / total_pT
```

### Moment Channels

Use additive pT-weighted moments in jet-local coordinates. Let:

```text
deta = eta_particle - eta_reference
dphi = wrapped(phi_particle - phi_reference)
r    = sqrt(deta^2 + dphi^2)
```

Recommended moment channels:

```text
sum_pT_abs_deta_pos = sum pT * max(deta, 0)
sum_pT_abs_deta_neg = sum pT * max(-deta, 0)
sum_pT_abs_dphi_pos = sum pT * max(dphi, 0)
sum_pT_abs_dphi_neg = sum pT * max(-dphi, 0)
sum_pT_deta2        = sum pT * deta^2
sum_pT_dphi2        = sum pT * dphi^2
sum_pT_r            = sum pT * r
sum_pT_r2           = sum pT * r^2
```

This avoids directly enforcing axis angle averages. Axis and width can be
derived:

```text
S_deta = sum_pT_abs_deta_pos - sum_pT_abs_deta_neg
axis_deta = S_deta / total_pT

width_eta = sum_pT_deta2 / total_pT - axis_deta^2
```

### Optional High-Value Auxiliary Channels

These are useful losses but should not be hard parent-child conservation
channels in the first implementation:

```text
leading_pT_fraction
subleading_pT_fraction
max_cell_pT_fraction
soft_track_width
soft_photon_width
entropy_of_pT_distribution
```

They can be predicted at global/grid levels as auxiliary diagnostics and may
help the latent representation.

## Consistency By Construction

The model should not output child totals directly. It should output allocation
logits.

For a parent cell with nonnegative accounting vector:

```text
A_parent[c]
```

and `M` child cells, the child decoder emits:

```text
allocation_logits[c, i] for child i
```

Then:

```text
fraction[c, i] = softmax_i(allocation_logits[c, i])
A_child[i, c]  = A_parent[c] * fraction[c, i]
```

Therefore:

```text
sum_i A_child[i, c] = A_parent[c]
```

exactly, up to numerical precision.

### Coupled Totals

Some quantities must be internally consistent:

```text
total_pT should equal sum of PID-category pT channels
total_count should equal sum of PID-category count channels
```

The strongest version should make category totals primitive:

```text
charged_pT
neutral_hadron_pT
photon_pT
electron_pT
muon_pT
```

and derive:

```text
total_pT = sum category_pT
```

This avoids conflicts where total pT and composition pT are allocated
separately and disagree.

### Signed Quantities

Avoid raw signed conservation channels when possible. Use positive/negative
decompositions:

```text
sum_pT_deta = sum_pT_abs_deta_pos - sum_pT_abs_deta_neg
```

Both positive and negative pieces can be allocated with softmax.

### Cell Bounds

For final cell centroids and pseudo-particle coordinates, outputs should be
bounded to the cell:

```text
local_u = sigmoid(raw_u)
local_v = sigmoid(raw_v)
eta = eta_min + local_u * (eta_max - eta_min)
phi = phi_min + local_v * (phi_max - phi_min)
```

This prevents the pseudo-view from placing particles outside the region whose
accounting it is supposed to render.

## Hierarchy

Use a fixed deterministic hierarchy in jet-local coordinates. The hierarchy is
not meant to be a perfect detector segmentation. It is a scaffold for stable
coarse-to-fine prediction.

### Level 0: Global

One global token predicts full-jet offline accounting totals.

```text
HLT particles -> global accounting vector G
```

### Level 1: Coarse Grid

Use 8 cells:

```text
2 eta half-planes
x 2 phi half-planes
x 2 radial shells
= 8 cells
```

The radial shell boundary should be configurable. A good default:

```text
r_inner_outer_boundary = median offline/Hlt pT-weighted radius over model_train
```

or a fixed normalized jet radius if that is easier operationally.

Level 1 predicts allocations of the global accounting totals into these 8
cells:

```text
G -> C1[8]
```

### Level 2: Medium Grid

Refine each Level 1 cell into children. Use a dyadic split that preserves
cell partitioning:

```text
each C1 cell -> 4 eta/phi children
```

This produces 32 medium cells:

```text
C1[8] -> C2[32]
```

### Level 3: Fine Grid

Refine each medium cell into 4 smaller eta/phi children:

```text
C2[32] -> C3[128]
```

This is the default final cell grid for the first full model. It gives enough
locality for pseudo-particle rendering without exploding slot count too badly.

### Alternative Depths

The campaign should include:

```text
global only
global -> 8 cells
global -> 8 -> 32 cells
global -> 8 -> 32 -> 128 cells
```

This will tell us whether the hierarchy is helping or if most of the value is
global/medium-scale.

## Model Architecture

Optimize for end tagging performance, not minimal compute.

### Shared HLT Encoder

Use a strong ParT-style encoder over original HLT particles.

Inputs:

```text
HLT PF features
HLT points
HLT lorentz vectors
HLT mask
```

Outputs:

```text
particle embeddings H_i
global jet embedding z_hlt
optional pairwise/geometric attention features
```

Implementation preference:

```text
reuse existing ParT encoder blocks where possible
expose pre-classifier particle and pooled representations
support warm-start from HLT ParT
```

### Global Predictor

Inputs:

```text
HLT particle embeddings H_i
global jet embedding z_hlt
pooled HLT summary features
```

Architecture:

```text
attention pooling over H_i
MLP head with LayerNorm/GELU/dropout
positive outputs through softplus/log-space parameterization
uncertainty outputs for each global target
```

Outputs:

```text
global accounting vector G_hat
global uncertainty sigma_G
global auxiliary predictions
```

### Grid Allocation Decoder

Each grid level uses learned cell query tokens with deterministic geometry
embeddings.

Inputs at level `l`:

```text
HLT particle embeddings H_i
global accounting token
all ancestor cell tokens
parent accounting vectors
cell geometry embeddings
```

Architecture:

```text
cell query self-attention
cross-attention from cell queries to HLT particle embeddings
cross-attention to parent/ancestor tokens
MLP allocation heads per accounting channel group
```

Output:

```text
allocation logits for every child of every parent
child accounting vectors computed by constrained allocation layer
cell hidden tokens
cell uncertainty estimates
```

The decoder should be powerful:

```text
d_model: 256 or 384
layers per grid level: 2-4
attention heads: 8
dropout: 0.05-0.10
```

### Particle Slot Decoder

Each final cell has `K` learned slot queries. Use a large enough `K` to avoid
forcing the model to under-render dense cells.

Recommended defaults:

```text
K = 8 slots per final cell for pilot
K = 12 or 16 slots per final cell for high-data/full model
```

Inputs:

```text
HLT particle embeddings H_i
final cell token
ancestor cell tokens
final cell accounting vector
cell geometry embedding
optional stochastic latent code for multi-view sampling
```

Architecture:

```text
slot query self-attention within each cell
cross-attention to HLT particles
cross-attention to ancestor/cell tokens
MLP heads for slot existence/count weights, pT allocation, coordinates, PID, uncertainty
```

Outputs per slot:

```text
existence/count weight
pT/category-pT allocation
eta/phi local coordinate
PID logits
charge/PID group logits
uncertainty/reliability
dust/null allocation
```

### Dust Slot

Each final cell should include a dust/null slot. This lets the decoder account
for unresolved diffuse energy without pretending it is a confident particle.

```text
cell pT = sum real slot pT + dust pT
```

The tagger can see dust as a special pseudo-token or as a cell-level
uncertainty feature.

## Particle-Level Consistency

For each final cell:

```text
cell accounting vector A_cell
slot outputs
```

Slot pT should be assigned by normalized allocation:

```text
w_slot = softmax(slot_pT_logits including dust)
pT_slot = A_cell.total_pT * w_slot
```

For composition:

```text
pT_slot * P(pid=charged)        should sum to charged_pT_cell
pT_slot * P(pid=photon)         should sum to photon_pT_cell
pT_slot * P(pid=neutral_hadron) should sum to neutral_hadron_pT_cell
```

The strongest version can enforce this more directly by allocating category
pT through separate slot softmaxes:

```text
charged_pT_slot[k] = charged_pT_cell * softmax(charged_slot_logits)[k]
photon_pT_slot[k]  = photon_pT_cell  * softmax(photon_slot_logits)[k]
...
total_pT_slot[k]   = sum_category category_pT_slot[k]
```

Then PID probabilities are derived or supervised to agree with the allocated
category mixture. This is more constrained and is the recommended full model.

For expected counts:

```text
count_weight_raw = sigmoid(raw_count_weight)
count_weight = N_cell * count_weight_raw / sum(count_weight_raw)
```

Then expected slot count sums exactly to the cell expected count.

## Losses

### Global Loss

Supervise global offline accounting targets:

```text
log-space Huber for positive pT/energy/count channels
relative error for composition totals
Gaussian NLL if uncertainty heads are enabled
auxiliary losses for derived axis, width, mass, and composition fractions
```

Selection diagnostics:

```text
global_total_pT_mae
global_total_pT_relative_mae
global_count_mae
global_composition_mae
global_axis_delta_mae
global_width_mae
```

### Grid Loss

For each grid level:

```text
offline particles are deterministically assigned to grid cells
offline cell accounting vectors are computed
model predicts child accounting vectors by allocation
loss compares predicted child vectors to offline child vectors
```

Use:

```text
weighted Huber/log-Huber for accounting channels
relative loss for pT-heavy channels
count NLL or Huber for expected counts
uncertainty-weighted NLL when enabled
```

High-pT cells should receive larger weight, but do not let the model ignore
low-pT composition structure entirely.

### Particle Slot Loss

Particle loss is cell-local and set-based.

For each final cell:

```text
predicted slots in cell
offline particles assigned to same cell
```

Use Sinkhorn or Hungarian matching. Sinkhorn is preferred for speed and
differentiability; Hungarian is useful as an evaluation diagnostic.

Matching cost:

```text
pT/log-pT distance
eta/phi distance inside cell
PID cross entropy
charge/category mismatch
optional mass/energy feature distance
```

Loss terms:

```text
matched coordinate loss
matched pT/category-pT loss
matched PID loss
existence/count loss
dust penalty for excessive unresolved pT
unmatched slot no-particle loss
cell-level accounting reconstruction loss
```

The cell-level accounting loss should remain even though consistency is mostly
by construction, because it verifies the rendered slots agree with the cell
accounting target and catches degenerate slot/PID behavior.

### Tagging Loss

For tagger training:

```text
10-class cross entropy
optional label smoothing
optional logit KD from offline teacher or strong HLT ensemble
optional auxiliary hierarchy reconstruction loss during fine-tuning
```

The final objective for end-to-end fine-tuning:

```text
L = L_CE
  + lambda_KD * L_KD
  + lambda_global * L_global
  + lambda_grid * L_grid
  + lambda_slot * L_slot
```

During final tagger fine-tuning, reconstruction losses should be downweighted
so tagging performance is the primary objective while the hierarchy remains
stable.

## Multi-View Pseudo-Particle Sampling

The offline particle set is not uniquely determined. The model should be able
to produce multiple plausible pseudo-offline views.

Recommended approach:

```text
deterministic accounting hierarchy
stochastic or dropout-conditioned particle slot decoder
M pseudo-particle views per jet
each view individually satisfies accounting constraints
```

Default:

```text
M = 1 for pilot deterministic runs
M = 4 for full multi-view runs
```

The tagger can receive all pseudo views as separate streams or as a
view-indexed particle set with learned view embeddings.

This is useful because:

```text
coarse accounting may be reliable
exact particle rendering may be ambiguous
multiple plausible renderings expose uncertainty to the tagger
```

### Structural Multi-Depth Pseudo Views

Stochastic samples from one reconstructor are not the only useful source of
view diversity. Train a controlled family whose terminal accounting resolution
is different while every other important design choice is held fixed:

```text
C5-B1: global -> 8 cells -> particle slots
C5-B2: global -> 8 cells -> 32 cells -> particle slots
C5-B3: global -> 8 cells -> 32 cells -> 128 cells -> particle slots
```

All three use the same full slot recipe:

```text
K = 16 slots per terminal cell
dust/unresolved slot enabled
Sinkhorn matching
composition/count/moment accounting
uncertainty heads and heteroscedastic NLL
identical HLT encoder capacity
identical optimization budget and data split
```

Only hierarchy depth and the terminal cell scale may differ. This makes the
views complementary for a principled reason:

```text
B1 rendering: coarse and robust, but spatially low resolution
B2 rendering: intermediate substructure and moderate ambiguity
B3 rendering: fine local structure with the greatest reconstruction risk
```

The canonical best-C reconstructor used by `D5` remains independently selected
from Tier C. If its complete configuration hash is identical to `C5-B3`, the
campaign must reuse the same checkpoint rather than retrain or duplicate the
same pseudo-view under two names.

## Tagger Fusion

The main tagger should not be logit fusion. Logit fusion is a control.

The best expected architecture is an uncertainty-aware dual-stream ParT:

```text
Stream A: original HLT particles
Stream B: pseudo-offline particles
```

Each stream has its own embedding and self-attention. Then apply cross-view
fusion before the classifier.

### Recommended Fusion Block

Use several layers of bidirectional cross-attention:

```text
HLT tokens attend to pseudo tokens
pseudo tokens attend to HLT tokens
global tokens attend to both
uncertainty/reliability features gate pseudo-token influence
```

The fused representation goes to a shared classifier head.

Pseudo-particle token features:

```text
pT/log pT
eta/phi relative to HLT jet axis
derived 4-vector features
PID probabilities or sampled PID
cell level index
slot index
dust/null flag
uncertainty/reliability
parent cell accounting features
```

HLT token features remain unchanged except optional extra cross-view positional
features.

### Why Early Fusion Should Win

Logit fusion only lets two models vote after they have committed to a class.
Early fusion lets the tagger decide locally:

```text
trust HLT here
trust pseudo-view here
ignore pseudo-view where uncertainty is high
use pseudo-view only for class-specific local patterns
```

This is especially important because pseudo-particles may be useful in some
regions and misleading in others.

### Multi-Depth Early Fusion

The structural multi-depth model must keep reconstructed views distinguishable.
Do not concatenate their pseudo-particles and pretend they form one physical
jet. Use separate pseudo streams:

```text
trusted original HLT stream
best-C pseudo stream, when structurally unique
C5-B1 coarse pseudo stream
C5-B2 intermediate pseudo stream
C5-B3 fine pseudo stream
```

Each pseudo token receives:

```text
learned hierarchy-depth/view embedding
terminal cell scale and cell identity
particle-slot features
parent accounting features
per-channel and aggregate uncertainty
dust/null indicator
```

Encode each pseudo stream with view-specific input projections and self-attention
before fusion. The HLT stream then queries every pseudo stream independently.
Use uncertainty-conditioned gates to combine the resulting cross-attention
residuals at both particle-token and pooled jet levels:

```text
r_i^(v) = CrossAttention(HLT_i, pseudo_view_v)
g_i^(v) = gate(HLT_i, r_i^(v), uncertainty_v, view_id_v)
HLT_i' = HLT_i + sum_v g_i^(v) * r_i^(v)
```

The gates should be normalized across pseudo views, with a separate trusted-HLT
skip path that cannot be gated away. Apply pseudo-view dropout during training
so the classifier cannot become dependent on one reconstruction depth. A mild
early gate-entropy regularizer may prevent immediate collapse to one stream,
but anneal it to zero so final model selection is driven by tagging performance.

This fusion allows the model to use structures stable across resolutions while
discounting fine details that are unsupported by coarser views or assigned high
uncertainty.

## Training Schedule

### Phase 0: Targets and Baselines

Build deterministic hierarchy targets from offline particles:

```text
global accounting targets
grid accounting targets for each level
final-cell particle assignment targets
derived diagnostics
```

Train or reuse:

```text
HLT ParT baseline
Offline ParT reference
optional offline teacher logits
```

### Phase 1: Reconstructor Pretraining

Train hierarchy reconstructor from HLT particles to offline hierarchy targets.

Use:

```text
global loss
grid losses
slot matching loss
uncertainty NLL
```

No tagging loss yet.

Selection:

```text
model_val weighted hierarchy score
secondary metrics: global pT, count, composition, grid pT, slot OT
```

### Phase 2: Frozen-Reconstructor Tagger

Freeze the reconstructor. Generate pseudo-particles inside the forward pass.
Train the dual-view tagger:

```text
HLT particles + frozen pseudo-offline particles -> class logits
```

This tests whether the pseudo-view is useful as a deployable auxiliary view.

### Phase 3: Gentle End-To-End Fine-Tuning

Unfreeze selected reconstructor layers with small LR:

```text
tagger LR: normal
fusion LR: normal
reconstructor decoder LR: 0.1x
HLT encoder LR: 0.03x to 0.1x if shared/warm-started
```

Keep reconstruction losses active but downweighted.

This is expected to produce the best final tagger.

### Phase 4: Multi-View Fine-Tuning

Enable multiple pseudo-particle views per jet:

```text
M = 4
shared hierarchy accounting
stochastic slot renderings
multi-view cross-attention tagger
```

This is expensive and remains one of the strongest candidates. It tests
stochastic ambiguity within one hierarchy, while Phase 5 tests complementary
information across hierarchy resolutions.

### Phase 5: Multi-Depth Structural Fusion

Train the depth-matched reconstructors independently first. Then train their
depth-specific D5 taggers under an identical schedule. Finally build the
multi-depth early-fusion tagger:

```text
1. load the unique best-C, C5-B1, C5-B2, and C5-B3 reconstructors
2. initialize the trusted HLT stream from A0
3. freeze reconstructors and train view encoders, gates, fusion blocks, and head
4. unfreeze pseudo-stream encoders while keeping accounting predictors frozen
5. gently unfreeze terminal slot decoders at 0.05x to 0.1x tagger LR
6. optionally unfreeze upper hierarchy decoders at 0.02x to 0.05x tagger LR
7. keep every active view's reconstruction and hard-consistency objectives
```

Do not force diversity with an arbitrary disagreement loss. The hierarchy
bottlenecks already create meaningful structural diversity. Use view dropout
and independent initialization seeds, and let validation tagging performance
determine whether each view is useful.

## Campaign Runs

The campaign should be broad because the key question is not only whether this
works, but where the useful signal enters.

### Tier A: Baselines

`A0`: HLT ParT baseline.

`A1`: larger HLT ParT capacity control.

`A2`: Offline ParT reference.

`A3`: best previous AV10 feature-MLP adapter reference, if available on same
split/cache.

`A4`: HLT ParT plus extra attention block, no pseudo-view.

### Tier B: Hierarchy Reconstructor Depth

`B0`: global accounting only.

`B1`: global plus 8-cell grid.

`B2`: global plus 8-cell plus 32-cell grid.

`B3`: full global plus 8-cell plus 32-cell plus 128-cell grid, no particle
slots.

`B4`: full hierarchy with moment channels removed.

`B5`: full hierarchy with composition channels removed.

`B6`: full hierarchy with count channels removed.

`B7`: no hard allocation consistency, same architecture, direct child totals.
This is a key control.

### Tier C: Particle Slot Decoders

`C0`: full hierarchy plus deterministic particle slots, `K=8`.

`C1`: full hierarchy plus deterministic particle slots, `K=16`.

`C2`: full hierarchy plus dust slot disabled.

`C3`: full hierarchy plus Sinkhorn slot matching.

`C4`: full hierarchy plus Hungarian matching diagnostic/training where
feasible.

`C5`: full hierarchy plus uncertainty heads and NLL.

`C6`: full hierarchy plus multi-view slot rendering, `M=4`.

Depth-matched `C5` family used by the D5 structural sweep:

`C5-B1`: B1 hierarchy terminating at 8 cells, followed by the full `C5`
`K=16` slot decoder.

`C5-B2`: B2 hierarchy terminating at 32 cells, followed by the same full `C5`
slot decoder.

`C5-B3`: B3 hierarchy terminating at 128 cells, followed by the same full `C5`
slot decoder.

### Tier D: Tagger Fusion

Use the best Tier C reconstructor unless otherwise noted.

`D0`: pseudo-view-only ParT tagger.

`D1`: HLT ParT and pseudo ParT late logit fusion.

`D2`: HLT ParT and pseudo ParT representation-level fusion.

`D3`: frozen reconstructor plus dual-stream cross-attention tagger.

`D4`: frozen reconstructor plus uncertainty-gated dual-stream tagger.

`D5`: end-to-end gentle fine-tune of `D4`.

`D5-B1`: D5 recipe using `C5-B1`, with global-to-8-cell accounting followed by
particle decoding.

`D5-B2`: D5 recipe using `C5-B2`, with global-to-8-to-32-cell accounting
followed by particle decoding.

`D5-B3`: D5 recipe using `C5-B3`, with the full 128-cell hierarchy followed by
particle decoding.

The D5 depth variants must share the same HLT warm start, tagger capacity, slot
recipe, learning-rate schedule, epoch/update budget, and loss weights. If
canonical `D5` selects a reconstructor identical to `C5-B3`, `D5-B3` is an alias
to that run and its checkpoint/predictions are reused.

`D6`: multi-view pseudo-particle dual-stream tagger using `C6`.

`D7`: grid-token fusion only, no pseudo-particle slots. This tests whether
particle rendering is actually necessary.

`D8`: multi-depth early-fusion tagger over all unique pseudo views from
canonical `D5`, `D5-B1`, `D5-B2`, and `D5-B3`. It uses separate view encoders,
learned depth embeddings, pseudo-view dropout, and uncertainty-gated
bidirectional cross-attention while retaining an ungated trusted HLT path.

### Tier E: Controls

`E0`: pseudo-particles with shuffled cell assignments.

`E1`: pseudo-particles with random slot coordinates but correct accounting.

`E2`: pseudo-particles with correct coordinates but shuffled PID/composition.

`E3`: pseudo-particles with no uncertainty features.

`E4`: direct unconstrained particle reconstructor with comparable capacity.

`E5`: hierarchy reconstructor trained without offline particle slot loss.

`E6`: tagger with same parameter count as dual-view model but no pseudo-view.

### Tier F: Fusion and Ensembles

`F0`: logit mean of A0 and best D run.

`F1`: scalar-weighted logit fusion of A0 and best D run.

`F2`: representation fusion ensemble of D3, D4, D5, D6, and D8.

`F3`: particle-view plus logit fusion:

```text
train best dual-stream particle-view fusion
also logit-fuse with independent HLT ParT baseline
```

If `D8` wins model validation, it becomes the particle-view member of `F3`.

`F4`: seed ensemble of best dual-stream run.

`F5`: mixed reconstructor ensemble:

```text
D8 multi-depth structural fusion
C6 stochastic multi-rendering model
independent stochastic seeds of the strongest member
validation-fitted logit fusion over independently trained final taggers
```

`D8` is the early structural fusion model. `F5` is deliberately broader: it
combines D8's hierarchy-depth diversity with C6's within-reconstructor
stochastic diversity and seed diversity.

## Expected Best Model

The original strongest single-view candidate remains:

```text
C6 reconstructor:
  full additive hierarchy
  composition/count/moment accounting
  uncertainty heads
  multi-view particle slot decoder

D6 tagger:
  original HLT ParT stream
  M=4 pseudo-offline streams
  uncertainty-gated bidirectional cross-attention
  warm-started from HLT ParT
  reconstructor pretrained, then gently fine-tuned end-to-end
```

After adding structural multi-depth fusion, the strongest single architecture
is expected to be `D8`, with `D6` as its main competitor. `D8` can exploit
coarse/fine agreement and disagreement directly, whereas `D6` represents
multiple plausible renderings from one hierarchy.

The best total performance version is probably `F3` using `D8` plus an
independent A0, or `F5` combining structural, stochastic, and seed diversity.

## Diagnostics

The report must separate reconstruction quality from tagging quality.

### Reconstruction Diagnostics

Global:

```text
pT relative MAE
energy relative MAE
expected count MAE
composition pT MAE
derived axis MAE
derived width MAE
```

Grid:

```text
per-level accounting MAE
per-level pT allocation KL
per-level composition allocation error
high-pT cell recall
cell count calibration
parent-child consistency max error
```

Particle slots:

```text
cell-local OT distance
matched pT error
matched eta/phi error
PID accuracy/cross entropy
dust fraction
expected count calibration
slot usage entropy
```

### Tagging Diagnostics

```text
accuracy
cross entropy
macro per-class accuracy
per-class accuracy
confusion matrix
one-vs-rest AUC where available
gap vs A0 HLT ParT
gap vs A1 larger HLT ParT
gap vs A2 Offline ParT reference
```

### Fusion Diagnostics

```text
pseudo-view gate mean/p10/p90
uncertainty vs gate correlation
per-class gate usage
HLT-only vs pseudo-only disagreement
accuracy on high-disagreement jets
accuracy binned by reconstructor confidence
per-depth B1/B2/B3 gate usage
pairwise pseudo-view prediction disagreement
pairwise pseudo-token representation similarity
drop-one-depth model_val accuracy and CE
coarse/fine agreement-binned accuracy
fraction of jets dominated by one pseudo view
```

For `D8`, run evaluation-time view ablations without retraining:

```text
HLT only
HLT + B1
HLT + B2
HLT + B3
HLT + each pair of depths
HLT + all unique structural views
```

These are diagnostics, not separately selected final-test models. They reveal
whether early fusion extracts complementary information or simply collapses to
the deepest reconstructor.

## Leakage and Trust Rules

Hard rules:

```text
final_test offline hierarchy targets are not loaded by deployable predictions
final_test pseudo-particles are generated from HLT only
model selection uses model_val only
stack_train may be used only for fusion fitting
final_test is confirmed explicitly and run once for final claims
```

Every run report must record:

```text
split manifest hash
HLT cache hash
offline target cache hash for train/val only
hierarchy target version
reconstructor checkpoint hash
tagger checkpoint hash
source commit/status hash
```

## Implementation Steps

### Step 1: Hierarchy Target Builder

Implement deterministic offline hierarchy targets:

```text
global accounting vector
8-cell, 32-cell, 128-cell accounting vectors
offline particle assignment to final cells
derived global/grid diagnostics
```

Add cache metadata, content hashes, manifest checks, and tests for exact
parent-child target sums.

### Step 2: Constrained Accounting Layers

Implement reusable modules:

```text
positive accounting parameterization
softmax allocation layer
category-pT total derivation
positive/negative moment reconstruction
cell-bound coordinate transform
slot pT/count normalization
```

Tests must assert exact consistency:

```text
children sum to parent
slot pT sums to cell pT
category pT sums to cell category totals
expected slot counts sum to cell count
```

### Step 3: HLT Encoder and Global/Grid Reconstructor

Build the shared HLT encoder and grid decoder stack:

```text
ParT-style HLT encoder
global predictor
cell query decoder
multi-level allocation outputs
uncertainty heads
```

Implement B-tier variants.

### Step 4: Particle Slot Decoder and Slot Loss

Build final-cell particle slot decoder:

```text
slot query decoder
dust slot
PID/category heads
coordinate heads
Sinkhorn matching loss
Hungarian evaluation diagnostic
```

Implement C-tier variants.

The depth-controlled `C5-B1`, `C5-B2`, and `C5-B3` implementations must use one
shared slot-decoder specification object. Tests should compare their resolved
configs and assert that hierarchy depth/terminal cell count are the only
intended architectural differences.

### Step 5: Reconstructor Training

Create training script and reports:

```text
global/grid/slot losses
uncertainty NLL
nonfinite guards
gradient clipping
model_val selection
reconstruction diagnostics CSV
```

### Step 6: Pseudo-Particle Cache and Deployable Prediction

Implement pseudo-particle generation:

```text
HLT cache -> reconstructor -> pseudo-particle view
no offline targets needed for deployable splits
multi-view generation support
uncertainty/reliability fields
```

### Step 7: Dual-Stream Fusion Tagger

Build tagger architectures:

```text
pseudo-only tagger
late logit fusion control
representation fusion
dual-stream cross-attention fusion
uncertainty-gated multi-view fusion
multi-depth view embeddings and per-view encoders
per-token and pooled uncertainty gates
pseudo-view dropout with an ungated HLT skip path
```

Implement D/E-tier runs, including the full `D5-B1/B2/B3` sweep and `D8`.

### Step 8: End-To-End Fine-Tuning

Add staged training:

```text
freeze reconstructor
train tagger/fusion
unfreeze decoder only
gentle end-to-end fine-tune
optional KD
optional multi-view training
multi-depth frozen-reconstructor fusion warmup
gentle per-view terminal-decoder unfreezing
duplicate-config checkpoint reuse
```

### Step 9: Fusion, Reporting, and Audits

Implement:

```text
prediction caching
stack_train fusion fitting
logit fusion
representation/particle-view fusion summaries
multi-depth drop-one-view and gate-collapse diagnostics
final report tables
required run/fusion group enforcement
provenance audits
```

The report must preserve separate rows for `D5`, `D5-B1`, `D5-B2`, `D5-B3`,
and `D8`. When canonical `D5` aliases a depth variant, record the shared
configuration hash and checkpoint hash explicitly rather than presenting it as
an independent training replicate.

### Step 10: Slurm Campaign Submitters

Create queue scripts for:

```text
pilot campaign
high-data campaign
cache-only target build
reconstructor-only sweep
tagger-only sweep from existing reconstructors
depth-matched D5 sweep
multi-depth D8 fusion from existing reconstructors
report-only rerun
```

Submitter must fail early when required caches/checkpoints are missing.

## Queue Strategy

Do not queue everything blindly first. Recommended sequence:

```text
1. pilot targets
2. B0-B3 reconstructors
3. C0/C3/C5 and C5-B1/B2/B3 particle-slot reconstructors
4. D0-D5 plus D5-B1/B2/B3 taggers
5. train D8 from all unique depth-matched checkpoints
6. inspect pilot reconstruction, tagging, and drop-one-depth diagnostics
7. run full C6/D6/F-tier if pilot shows signal
8. high-data full campaign
```

If compute is abundant, queue all tiers after target/cache validation passes.
The important thing is that the submitter records all dependencies and refuses
partial unsafe campaigns.

## Success Criteria

Minimum useful signal:

```text
D3/D4 or a depth-matched D5 beats A0 HLT ParT on model_val and stack_val
final_test gap remains positive
controls E0-E6 do not explain away the gain as capacity alone
D8 improves over its strongest individual depth member, or clearly improves CE
fusion F3/F5 improves over best single D run
```

Strong result:

```text
best single dual-view run beats HLT ParT by >1 percent relative CE or meaningful accuracy gap
best fusion beats larger HLT ParT capacity control
multi-view pseudo-particles improve over deterministic pseudo-particles
multi-depth early fusion improves over both B1-only and B3-only pseudo views
reconstruction confidence correlates with tagger usage
```

Best possible result:

```text
the constrained pseudo-offline view consistently improves HLT tagging
the improvement survives high-data
the hierarchy controls show that hard accounting matters
the final story is interpretable and reviewer-defensible
```
