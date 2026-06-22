# DETR Free-Slot Reconstructor Plan

This document defines a new reconstructor branch based on DETR-style learned
slots.  The point is not to import object detection as a computer-vision trick.
The point is to use the most valuable DETR idea:

```text
Predict a set with free learned queries, then match predictions to targets.
```

For this project, that means:

```text
HLT jet -> encoder -> learned reconstruction slots -> offline-like particle set
```

The reconstructor is still HLT-only at inference.  The offline view is used only
as a paired training target.

## Core Motivation

The previous reconstructor families were still partly parent-aligned:

```text
HLT particle i -> corrected particle i
plus optional extra candidates
```

That is clean and auditable, but it may be too constrained for the actual
HLT-to-offline gap.  If the degradation includes merging, splitting, lost soft
particles, misleading HLT particles, or broad local reshaping, then forcing each
output to remain tied to an HLT parent can block the exact behavior we need.

The DETR/free-slot branch removes that constraint:

```text
K learned slots look at the whole HLT jet.
Each slot decides:
  do I exist?
  if yes, what offline-like particle do I represent?
```

The model can then express:

- one HLT constituent implies multiple offline-like particles,
- an HLT constituent should disappear,
- a soft offline-like constituent should be generated,
- a global/local HLT pattern implies a particle that no single HLT parent owns.

This is the cleanest reconstruction story we have so far:

```text
Do not imitate a frozen teacher first.
Do not force parent alignment.
Use paired offline data to learn an unordered offline-like particle set.
```

## High-Level Experiment

Train four DETR-style reconstructors with different HLT encoders:

```text
fixed HLT view -> ParT/Transformer encoder + DETR slots -> reconstructed view A
fixed HLT view -> ParticleNet encoder      + DETR slots -> reconstructed view B
fixed HLT view -> PFN/DeepSets encoder     + DETR slots -> reconstructed view C
fixed HLT view -> PCNN/local CNN encoder   + DETR slots -> reconstructed view D
```

Then reuse the multi-view tagger setup:

```text
[HLT view, A, B, C, D] -> five-view tagger -> prediction
```

The four reconstructors should share the same:

- output contract,
- feature representation,
- Hungarian matching loss,
- diagnostics,
- cache format,
- downstream tagger interface.

Only the HLT encoder changes.

## Relationship To Current Set-Matching Work

This branch should live next to the current set-matching branch, not replace it
in place.

The current set-matching branch asks:

```text
Can parent-edit/aggressive reconstruction plus set loss produce useful views?
```

The DETR branch asks:

```text
Can a fully free predicted particle set produce better offline-like views?
```

Both branches can feed the same five-view tagger machinery once reconstructed
views are cached.

## Output Contract

Every DETR reconstructor returns a fixed number of slots:

```text
slot_tokens:       [B, K, F]
existence_logits: [B, K]
slot_mask:        [B, K]      # normally all true before top-k/threshold export
aux:              diagnostics
```

Where:

- `B` is batch size.
- `K` is the number of learned reconstruction slots.
- `F` is the full particle feature dimension used by the taggers.

Recommended first default:

```text
num_slots = 160
export_max_tokens = 128
```

Rationale:

- Offline/HLT taggers currently operate around 128 constituents.
- A few extra slots let the model over-propose during reconstruction.
- The cache/export step can select the top 128 slots by existence probability.

If memory or Hungarian time is a problem, use:

```text
num_slots = 128
```

for the first smoke and binary runs.

## Feature Representation

The model should not regress all raw fields with equal importance.  The loss
needs a physically meaningful core plus weaker auxiliary terms.

Use a canonical matching representation internally:

```text
core features:
  log_pt
  eta
  phi
  log_energy

auxiliary continuous features:
  track/impact-parameter style fields
  any continuous extra particle fields

auxiliary binary/categorical features:
  charge and PID-like indicators, where semantics are clear
```

The model may output the full raw tagger feature vector, but the loss should
interpret it through this canonical feature adapter.

Important rules:

- `pt` and `energy` should be positive by construction or by safe transform.
- `phi` loss must use wrapped angular distance, not ordinary subtraction.
- `eta` should be bounded or softly penalized outside a physical range.
- Auxiliary fields should have weak weights until their semantics are verified.

The first version should include:

```text
raw_to_matching_features(tokens)
matching_loss_terms(pred_tokens, target_tokens)
sanitize_or_bound_predicted_tokens(...)
```

## Architecture

### Shared Structure

All four variants use the same outer structure:

```text
HLT tokens -> architecture-specific encoder -> memory tokens
learned slots + memory tokens -> DETR decoder -> slot embeddings
slot embeddings -> prediction heads
```

The common interface should be:

```python
class DetrSlotReconstructor(nn.Module):
    def forward(tokens, mask) -> DetrSlotOutput:
        memory, memory_mask, global_context = encoder(tokens, mask)
        slot_embeddings = decoder(learned_queries, memory, memory_mask)
        return heads(slot_embeddings, global_context)
```

### Encoder Interface

Every encoder adapter returns:

```text
memory_tokens: [B, M, D]
memory_mask:   [B, M]
global_context [B, D] or None
```

The DETR decoder should not care whether the memory came from a Transformer,
ParticleNet, PFN, or PCNN.

### ParT / Global Transformer Encoder

Use the same general idea as the existing global transformer reconstructor:

- embed HLT particles,
- add optional pair/geometric bias later,
- run Transformer encoder blocks,
- return particle memory tokens plus pooled context.

This is the most direct DETR-style variant.

### ParticleNet Encoder

Use EdgeConv-style local graph blocks:

- compute local neighborhoods in eta/phi/pt-derived coordinate space,
- produce per-particle local-context embeddings,
- return those embeddings as decoder memory,
- return pooled graph context.

This encoder should be good at local splitting/merging hypotheses.

### PFN Encoder

Use DeepSets/PFN structure:

- per-particle `phi` network,
- sum/mean pooled global context,
- optional broadcasted global context back to particle memory,
- return lightweight particle memory plus pooled context.

This is less expressive locally, but gives a deliberately different inductive
bias.  It may produce useful global calibration-like views.

### PCNN Encoder

Use local 1D convolution over a deterministic particle ordering:

- sort or retain current HLT order consistently,
- embed tokens,
- use residual/dilated conv blocks,
- return ordered memory tokens plus pooled context.

This branch tests whether stable local ordering captures substructure cues that
the graph/set models handle differently.

### DETR Decoder

Use learned query embeddings:

```text
queries: [K, D]
```

For each batch:

```text
queries -> repeated to [B, K, D]
queries cross-attend to encoder memory
queries self-attend with each other
```

Recommended first decoder:

```text
decoder_layers = 4
embed_dim      = 128 or 192
num_heads      = 4
mlp_ratio      = 4
dropout        = 0.05
```

Use pre-norm Transformer decoder layers for stability.

### Prediction Heads

Each slot predicts:

```text
existence_logit: scalar
core kinematic fields
auxiliary feature fields
```

For raw-token output, use bounded transforms:

```text
log_pt/log_energy -> exponentiated or converted through adapter
eta               -> max_abs_eta * tanh(raw_eta)
phi               -> wrapped angle, or output sin/cos then convert
binary/PID fields -> logits or sigmoid outputs
```

The cached view should be in the same token format expected by existing taggers.

## Training Target

For each jet:

```text
input:  fixed HLT token set
target: paired offline token set
```

The target is unordered:

```text
offline particle order has no semantic force.
```

The loss should treat target particles as a set, not a sequence and not aligned
to HLT parents.

## Hungarian Matching

For each jet in the batch:

```text
predicted slots: P = {p_i}_{i=1..K}
offline targets: O = {o_j}_{j=1..N}
```

Build a cost matrix:

```text
C[i, j] =
    w_pt  * Huber(pred_log_pt_i - target_log_pt_j)
  + w_eta * Huber(pred_eta_i    - target_eta_j)
  + w_phi * wrapped_phi_distance(pred_phi_i, target_phi_j)
  + w_e   * Huber(pred_log_e_i  - target_log_e_j)
  + w_aux * weak_aux_distance(i, j)
```

Then use Hungarian assignment:

```text
matched_pred_indices, matched_target_indices = linear_sum_assignment(C)
```

Since `K >= N`, every target should receive one predicted slot.  Unmatched
predicted slots become no-object slots.

### Why Hard Hungarian First

Sinkhorn/OT is a useful possible ablation, but it should not be the primary
first implementation.

Hungarian is preferable for the first DETR branch because:

- it gives crisp one-to-one assignments,
- it is easier to debug,
- it avoids rewarding blurred fractional particles,
- it mirrors the original DETR training story,
- it makes no-object supervision simple.

Sinkhorn can be a later experiment if Hungarian is too brittle.

## Loss Function

The recommended first objective:

```text
loss =
    matched_core_loss
  + weak_aux_loss
  + existence_loss
  + count_loss
  + weak_jet_summary_loss
  + optional_support_loss
  + optional_duplicate_penalty
```

### Matched Core Loss

For matched pairs only:

```text
matched_core_loss =
    Huber(log_pt)
  + Huber(eta)
  + wrapped_phi_loss
  + Huber(log_energy)
```

This is the main particle reconstruction signal.

Suggested first weights:

```text
log_pt:     1.0
eta:        1.0
phi:        1.0
log_energy: 0.5
```

### Weak Auxiliary Loss

For matched pairs only, apply weaker supervision on the remaining features:

```text
continuous aux: Huber or L1, weight 0.05 to 0.20
binary aux:     BCEWithLogits, weight 0.05 to 0.20
```

This prevents the model from ignoring the full vector, but keeps the assignment
from being dominated by sparse or brittle fields.

### Existence Loss

Matched slots should exist.  Unmatched slots should be no-object:

```text
target_existence = 1 for matched slots
target_existence = 0 for unmatched slots
```

Use BCE or focal BCE:

```text
existence_loss = BCEWithLogits(existence_logits, target_existence)
```

Because no-object slots may dominate, use either:

```text
positive_weight > 1
```

or a DETR-style no-object downweight:

```text
negative_weight = 0.1 to 0.3
```

Start simple:

```text
positive_weight = 1.0
negative_weight = 0.2
```

### Count Loss

The predicted number of particles should match target multiplicity:

```text
pred_count = sum(sigmoid(existence_logits))
target_count = number of offline target particles
count_loss = SmoothL1(pred_count - target_count)
```

Keep this weak:

```text
count_weight = 0.05 to 0.20
```

Existence loss should do most of the work.

### Weak Jet Summary Loss

Add a weak aggregate constraint:

```text
sum_pt
sum_energy
pt-weighted eta
pt-weighted phi, using vector sum
optional approximate mass-like summary
```

This should be weak enough that it does not force averaged particles:

```text
jet_summary_weight = 0.02 to 0.10
```

### Optional HLT Support Loss

This is a soft guardrail, not a parent alignment term.

For high-confidence predicted slots:

```text
distance_to_nearest_hlt = min_deltaR(pred_slot, hlt_particles)
support_loss = penalty(distance_to_nearest_hlt > threshold)
```

Use a loose threshold:

```text
max_nearest_hlt_deltaR = 0.8
support_weight = 0.0 initially, then 0.01 if hallucinations appear
```

Do not turn this into hard nearest-parent matching.  That would defeat the
point of the DETR branch.

### Optional Duplicate Penalty

If many active slots collapse onto the same target-like particle, add a weak
repulsion among high-confidence predicted slots:

```text
duplicate_penalty = mean exp(-deltaR_ij / temperature) * active_i * active_j
```

This should be off by default until diagnostics show duplicate collapse.

## Diagnostics

Every reconstructor run should write diagnostics outside the large checkpoint
folder as well as normal reports:

```text
loss components:
  matched_core
  aux
  existence
  count
  jet_summary
  support
  duplicate

set quality:
  target_count_mean
  predicted_count_mean
  count_mae
  existence_precision
  existence_recall
  matched_deltaR_mean
  matched_deltaR_p90
  matched_logpt_mae
  matched_eta_mae
  matched_phi_mae

jet summaries:
  sum_pt_relative_error
  sum_energy_relative_error

cache/export:
  active_slots_mean
  exported_tokens_mean
  top_existence_mean
  nonfinite_count
```

The most important early failure checks:

- existence logits all collapse to no-object,
- all slots become active,
- repeated duplicate particles,
- phi loss explodes because angle wrapping is wrong,
- matched loss improves but downstream tagger collapses,
- predicted count is far from target count.

## Cache Format

After training each reconstructor, cache reconstructed views:

```text
<split>_reconstructed_view.npz
```

The exported view should contain:

```text
tokens:           [N, export_max_tokens, F]
mask:             [N, export_max_tokens]
labels:           [N]
existence_scores: [N, export_max_tokens]
jet_ids / metadata
```

Selection should be:

```text
top export_max_tokens slots by existence score
```

For slots below threshold, keep mask false.  Do not reorder by predicted
kinematics first.  The tagger can learn from the score/order if useful.

Recommended export behavior:

```text
selection_mode = topk_or_threshold
confidence_threshold = 0.05
export_max_tokens = 128
```

For `five_view_no_confidence`, still keep all selected slots but do not expose
confidence as an input channel.

## Downstream Taggers

Use the existing five-view tagger branch as the first downstream consumer:

```text
views:
  hlt
  detr_gt
  detr_pn
  detr_pfn
  detr_pcnn
```

Run the same tagger variants:

```text
hlt_only
hlt_plus_gt
hlt_plus_pn
hlt_plus_pfn
hlt_plus_pcnn
five_view_plain
five_view_geometry
five_view_no_confidence
view_label_shuffle_control
```

For binary tasks, report:

```text
accuracy
AUC
FPR at 30% signal efficiency
FPR at 50% signal efficiency
```

The primary selection metric for binary should remain:

```text
fpr_at_signal_eff_0p50
```

For 10-class tasks, report:

```text
accuracy
macro per-class accuracy
confusion matrix
```

## First Target Task

The best first serious task is QCD vs Tbqq, because it is closer in spirit to
top tagging and likely harder/more relevant than Hbb vs QCD.

Use true binary splits:

```text
model_train: 500k binary jets
model_val:   150k binary jets
stack_train: 500k binary jets
stack_val:   150k binary jets
final_test:  500k binary jets
```

Here "500k binary jets" means after selecting QCD/Tbqq, balanced across the two
classes.

If part0 does not contain enough QCD/Tbqq jets, the manifest builder should fail
early with available counts.  Then use additional JetClass parts or a smaller
split.

## Implementation Plan

### Step 1: DETR Plan And Namespacing

Create a separate DETR branch under:

```text
teacher_logit_reco/set_matching/detr_slots/
```

or similar.  Do not modify the existing parent-aligned set-matching branch in
place.

Add the plan file, module namespace, and basic imports.

### Step 2: Canonical Feature Adapter

Implement feature utilities:

```text
raw_to_core_features
raw_to_aux_features
decode_slot_outputs_to_raw_tokens
wrapped_phi_difference
safe_log_pt_energy
```

Add tests for:

- shape stability,
- finite outputs,
- phi wrapping,
- positive pt/energy handling,
- binary/two-class compatibility.

### Step 3: Slot Output Dataclass

Add a common output object:

```text
DetrSlotOutput(
  tokens,
  existence_logits,
  slot_mask,
  aux
)
```

Add validation helpers so bad shapes fail immediately.

### Step 4: Shared DETR Decoder And Heads

Implement:

```text
LearnedSlotQueries
DetrSlotDecoder
DetrPredictionHeads
```

This step should only require fake memory tokens in tests.

Tests:

- forward shape,
- gradients are finite,
- all architectures can target the same output contract later.

### Step 5: Encoder Adapter Interface

Define:

```text
BaseHLTEncoderAdapter
EncoderOutput(memory_tokens, memory_mask, global_context)
```

Add a tiny dummy encoder for tests.

### Step 6: Global Transformer Encoder Adapter

Implement the ParT-ish/global transformer encoder adapter.

Use existing embedding and transformer patterns where possible, but keep it
cleanly separated from the old reconstructor head.

### Step 7: ParticleNet Encoder Adapter

Implement the EdgeConv/ParticleNet-style encoder adapter.

This can reuse existing ParticleNet-style blocks if they are sufficiently
modular.  Otherwise implement a small local graph encoder with the same defaults
used by current set-matching PN code.

### Step 8: PFN Encoder Adapter

Implement PFN/DeepSets encoder:

```text
per-particle phi network
masked sum/mean pooling
context broadcast
memory projection
```

### Step 9: PCNN Encoder Adapter

Implement PCNN/local convolution encoder:

```text
token embedding
masked ordered conv blocks
dilated residual stack
memory projection
```

### Step 10: Hungarian Loss

Implement batch Hungarian matching and loss components:

```text
pairwise cost
linear_sum_assignment
matched losses
unmatched no-object loss
count loss
jet summary loss
diagnostics
```

Use SciPy's `linear_sum_assignment` on the research environment.  Add a clear
error message if SciPy is missing.  A simple greedy fallback can be allowed only
for smoke/unit tests, not production runs.

Tests:

- perfect prediction has lower loss than shuffled/noisy prediction,
- unmatched slots receive no-object targets,
- empty/padded targets are handled,
- count loss moves in the expected direction,
- all diagnostics are finite.

### Step 11: Training Script

Add:

```text
scripts/train_detr_slot_reconstructor.py
```

Arguments:

```text
--architecture gt|pn|pfn|pcnn
--hlt-cache-dir
--data-dir
--manifest-path
--output-dir
--train-split model_train
--val-split model_val
--max-train-jets
--max-val-jets
--num-slots
--export-max-tokens
--loss weights...
```

Write:

```text
best_model_val.pt
last.pt
run_report.json
training_curves.json
diagnostics/*.csv
```

Select the best checkpoint by:

```text
val_total_loss
```

at first.  Later, optionally select by downstream binary metric if recon loss
does not correlate with tagging.

### Step 12: Cache Reconstructed Views

Add:

```text
scripts/cache_detr_slot_reco_views.py
```

For each split, export top slots into the existing reconstructed-view NPZ
format expected by five-view taggers.

Include existence scores and diagnostics.

### Step 13: Single-Reco Sanity Taggers

Before full five-view training, optionally train:

```text
HLT + one DETR reco view
```

for each architecture.  This helps tell whether a view is individually useful
or only useful in combination.

This can reuse existing `hlt_plus_<arch>` variants once cache names are wired.

### Step 14: Five-View Tagger Integration

Wire DETR cache directories into the existing five-view tagger:

```text
hlt
detr_gt
detr_pn
detr_pfn
detr_pcnn
```

Do not change the tagger architecture unless the cached view contract forces it.

### Step 15: Audits And Final Report

Add DETR-specific audit output:

```text
reconstructor summary table
cache/export summary
tagger summary
offline reference comparison
HLT-only comparison
best binary operating point table
```

The final report should make it easy to answer:

```text
Did free-slot reconstruction improve over HLT-only?
Which encoder helped?
Did five-view beat every single view?
Did reconstruction metrics correlate with tagging metrics?
```

### Step 16: Slurm Runners

Add runners:

```text
sbatch/run_train_detr_slot_reconstructor.sh
sbatch/run_cache_detr_slot_reco_views.sh
sbatch/submit_detr_slot_qcd_tbqq_binary_experiment.sh
sbatch/submit_detr_slot_smoke_test.sh
```

The full binary graph should queue:

```text
1 fresh binary manifest job
1 binary HLT cache job
1 offline reference job
4 DETR reconstructor train jobs
4 DETR reconstructed-view cache jobs
9 five-view tagger/ablation jobs
1 audit job
1 final report job
```

Use diagnostics mirroring by default.

### Step 17: Smoke Test

Run a small smoke:

```text
10k train
2k val
5k stack train
2k stack val
10k final test
2 epochs
num_slots = 64
export_max_tokens = 64
```

The smoke is only for shape, cache, and dependency correctness.

### Step 18: First Real Run

Run QCD vs Tbqq by default, or the same real-run protocol for another binary
pair such as QCD vs Hgg:

```text
model_train: 500k
model_val:   150k
stack_train: 500k
stack_val:   150k
final_test:  500k
epochs:      20 to 30 for reconstructors
tagger:      45 epochs
num_slots:   128 or 160
```

The QCD-vs-Hgg real run uses:

```text
sbatch/submit_detr_slot_qcd_hgg_binary_experiment.sh

hlt_degradation_strength: 0.6
model_train: 500k
model_val:   150k
stack_train: 500k
stack_val:   150k
final_test:  500k
epochs:      30 for reconstructors
tagger:      45 epochs
num_slots:   160
```

Compare against:

- offline-only ParT on offline,
- HLT-only ParT on HLT,
- current parent-aligned set-matching five-view taggers,
- single DETR reconstructed-view taggers,
- five-view DETR tagger.

## Expected Failure Modes

### Slot Collapse

Many slots predict the same particle.  Watch duplicate diagnostics.  Add weak
duplicate penalty only if needed.

### No-Object Collapse

The model predicts no particles.  Increase positive existence weight, reduce
negative no-object weight, or warm up with matched losses only.

### All-Slots Active

The model activates every slot.  Increase no-object weight or count loss.

### Good Reconstruction, Bad Tagging

This means the loss may be reconstructing irrelevant details or damaging
classification-relevant HLT cues.  Keep HLT as an anchor view in the final
tagger and inspect single-view results.

### Bad Reconstruction, Good Tagging

This is possible and interesting.  The views may be useful transformations even
if they are not faithful offline particle sets.  Keep both reconstruction and
tagging diagnostics.

### Hungarian Too Slow

Reduce `num_slots`, use smaller batch size, or cache CPU cost matrices more
efficiently.  The first implementation should favor correctness and diagnostics
over speed.

## What Would Count As A Win

For binary QCD vs Tbqq, a meaningful win is:

```text
five-view DETR improves FPR at 50% signal efficiency over HLT-only
and the improvement survives final_test.
```

Accuracy alone is secondary.  The physics-style low-FPR metrics matter more.

For 10-class JetClass, a meaningful win is:

```text
five-view DETR improves final_test accuracy over HLT-only ParT
without only trading one class for another.
```

Per-class diagnostics must be reported.

## Design Principle

The important research claim should be:

```text
Offline-paired training can shape an HLT-only architecture into a better
extractor of the information that remains in HLT, by giving it a structured
inverse-degradation objective and enough free set capacity to express splitting,
merging, deletion, and generation.
```

The DETR/free-slot branch is the cleanest test of that claim so far.
