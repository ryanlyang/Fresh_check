# Local Particle Residual Field Plan

## Core Idea

The previous canonical-state campaign tested a side-channel:

```text
HLT particles + canonical jet-state context -> ParT tagger
```

That helped a little, but it is too easy for ParT to treat the side-channel as
optional. The stronger idea is to attach the recovered information directly to
the particle tokens that ParT already uses.

This plan predicts a local offline-correction field for every HLT particle:

```text
HLT particle i
+ local HLT neighborhood
+ whole-jet context
-> local residual fields around particle i
```

The final deployable tagger sees:

```text
[original HLT particle features, predicted local residual fields]
```

The method does not hallucinate a full offline particle set. It asks a more
stable question:

```text
Around this HLT particle, what offline energy-flow, composition, multiplicity,
and reliability information is missing or distorted?
```

The hope is that these fields become useful extra per-particle features rather
than ignorable global context.

## Why This Is The Right Pivot

The canonical-state tokens were physically motivated, but they lived outside
the particle pathway. A strong ParT can often learn to ignore weak auxiliary
views. The older reconstructor-style approaches sometimes improved tagging,
but they were not fully satisfying because exact particle reconstruction is
ill-posed:

```text
HLT and offline have different particle counts.
HLT particles can be merges of several offline particles.
Some offline particles have no clean HLT counterpart.
Some HLT particles are noisy or over-smeared remnants.
```

Local residual fields avoid those traps.

Instead of saying:

```text
this HLT particle should become this offline particle
```

we say:

```text
the offline flow near this HLT token differs from HLT in these local ways
```

That target is per-particle, local, differentiable, and naturally usable by
ParT attention.

## Data Regime

Use the same 10-class JetClass HLT setup as the canonical-state campaign unless
explicitly overridden:

```text
HLT profile: fixed_hlt_v2_realistic
HLT degradation strength: 2.5
labels: all 10 JetClass labels
model_train: 5M
model_val: 1M
stack_train: 2M
stack_val: 1M
final_test: 1M
```

For fast iteration, also support a pilot mode:

```text
model_train: 500k
model_val: 150k
stack_train: 300k
stack_val: 150k
final_test: 150k
```

All runs must be hash-bound to the split manifest, HLT cache, offline cache, and
label ordering.

## Target Definition

For each HLT particle token `i`, define several local neighborhoods around its
eta/phi position:

```text
R0 = 0.02
R1 = 0.05
R2 = 0.10
```

For each radius `R`, compute HLT and offline local summaries using soft
assignment weights:

```text
w_ij(R) = exp(-0.5 * (deltaR(i, j) / R)^2)
```

Use the HLT particle position as the query point. The offline particles are not
matched one-to-one. They are summarized into local fields around the HLT query.

The core target is:

```text
local_residual_i,R = local_summary_offline_i,R - local_summary_hlt_i,R
```

The target is masked for invalid HLT particle slots.

### Field Group 1: Local pT / Energy Flow

For each radius:

```text
delta_log_pt_sum_R = log(sum_pt_offline_R + eps) - log(sum_pt_hlt_R + eps)
delta_pt_frac_R    = (sum_pt_offline_R - sum_pt_hlt_R) / (jet_pt_hlt + eps)
missing_pt_frac_R  = max(sum_pt_offline_R - sum_pt_hlt_R, 0) / (jet_pt_hlt + eps)
extra_pt_frac_R    = max(sum_pt_hlt_R - sum_pt_offline_R, 0) / (jet_pt_hlt + eps)
```

`delta_log_pt_sum_R` is not necessarily "this token's own pT should shift by
this amount." It means:

```text
the total offline energy flow around this HLT token is higher/lower than the
HLT local energy flow by this log ratio
```

This is the main correction field.

### Field Group 2: Local Centroid / Axis Shift

For each radius:

```text
delta_eta_centroid_R
delta_phi_centroid_R
delta_r_centroid_R
```

These describe whether the offline local flow is centered differently from the
HLT local flow.

Use angle-safe phi wrapping.

### Field Group 3: Local Multiplicity

For each radius:

```text
delta_log_n_R      = log(n_offline_eff_R + 1) - log(n_hlt_eff_R + 1)
missing_n_frac_R  = max(n_offline_eff_R - n_hlt_eff_R, 0) / (n_offline_eff_R + 1)
```

Use effective multiplicity from soft weights rather than hard integer counts:

```text
n_eff_R = sum_j w_ij(R)
```

This field helps ParT understand whether an HLT token is probably representing
a merged or under-resolved region.

### Field Group 4: Composition Residuals

For each radius, compute pT-weighted composition fractions from canonical input
features. The exact mapping should use the repo's canonical feature names, but
the target groups should include:

```text
charged fraction residual
neutral hadron fraction residual
photon/electromagnetic fraction residual
muon/lepton fraction residual if available
other/unknown fraction residual
```

If the dataset feature schema does not cleanly expose every group, the
implementation must record the exact feature-to-composition mapping in the
run report.

### Field Group 5: Merge / Split / Reliability

Create classification or bounded-regression targets:

```text
is_clean_token
is_merged_token
has_missing_local_activity
has_large_local_shift
local_reliability_score
target_uncertainty_scale
```

Suggested definitions:

```text
is_merged_token = n_offline_eff_Rsmall > n_hlt_eff_Rsmall + threshold
has_missing_local_activity = missing_pt_frac_Rmedium > threshold
has_large_local_shift = sqrt(delta_eta^2 + delta_phi^2) > threshold
```

These fields are not just diagnostics. They are inputs to the final tagger.
They tell ParT whether to trust each HLT token literally or treat it as a
compressed local region.

## Oracle Residual Fields

The oracle target is the exact residual field computed from offline particles.

Oracle runs are not deployable. They answer:

```text
If perfect local correction fields were available, would ParT use them?
```

This is the most important early test. If oracle local residual fields do not
improve tagging, the target design is wrong.

## Predicted Residual Fields

The deployable reconstructor predicts the residual fields from HLT-only inputs:

```text
predicted_fields = R_theta(HLT particles, optional HLT canonical/global context)
```

At inference, no offline particles are used.

The final tagger input becomes:

```text
augmented_particle_i = concat(
    hlt_particle_features_i,
    predicted_local_residual_fields_i
)
```

The tagger can be trained from scratch or warm-started from an HLT ParT
checkpoint with an expanded input projection.

## Best Reconstructor Architecture

The best-performance reconstructor should be a cross-attentive local residual
transformer. Runtime is secondary. Scientific clarity and top performance are
the priorities.

### Inputs

For every HLT particle:

```text
canonical HLT particle features
particle mask
relative pt
pt rank embedding
eta/phi geometry
optional HLT canonical-state tokens
global jet summary token
```

The global jet summary should include deterministic HLT summaries:

```text
jet pt proxy
constituent count
pt-weighted eta/phi moments
radial moments
composition fractions
```

### Backbone

Use three interacting streams:

```text
particle stream: HLT particle tokens
local context stream: neighborhood-aware particle tokens
global/state stream: whole-jet and optional canonical-state tokens
```

Architecture:

```text
1. Particle feature embedding MLP.
2. Several self-attention layers over HLT particles.
3. Geometry-biased attention using deltaR, pt ratio, and radial-sector relation.
4. Cross-attention from particle tokens to global/state tokens.
5. Multi-radius output heads attached to each HLT particle token.
```

### Geometry Bias

Attention logits should receive a learned geometry bias:

```text
b_ij = f(deltaR_ij, log_pt_ratio_ij, same_radial_bin, same_phi_sector)
```

The model should learn whether nearby particles matter more, but it should be
given an inductive bias toward local HLT neighborhoods.

### Output Heads

Use separate heads for:

```text
pt/density residuals
centroid residuals
multiplicity residuals
composition residuals
merge/split/reliability flags
uncertainty/log-variance
```

Use zero-init or near-zero-init final projections for residual heads so the
initial augmented particle view is close to the HLT baseline.

## Reconstructor Loss

Use a weighted multi-task loss:

```text
L_reco =
    w_pt        * Huber(pt_residuals)
  + w_centroid  * Huber(centroid_residuals)
  + w_mult      * Huber(multiplicity_residuals)
  + w_comp      * Huber(composition_residuals)
  + w_flags     * BCE/CE(merge/reliability targets)
  + w_unc       * uncertainty_weighted_regression
  + w_global    * jet_consistency_loss
  + w_smooth    * local_smoothness_loss
```

### Jet Consistency Loss

The sum of predicted local corrections should be compatible with a global
offline-HLT difference:

```text
sum_i predicted_delta_pt_i ~= jet_pt_offline - jet_pt_hlt
```

Do not force exact equality because local neighborhoods overlap. Use a soft
consistency target with learned or fixed radius weights.

### Local Smoothness Loss

Nearby HLT particles should not receive wildly contradictory corrections unless
the target says the region is ambiguous:

```text
L_smooth = sum_ij K(deltaR_ij) * ||delta_i - delta_j||^2
```

Gate this by predicted uncertainty so genuinely sharp structures are allowed.

## Tagger Training Modes

### Frozen Reconstructor

Train reconstructor first, freeze it, then train ParT on predicted fields.

This cleanly answers:

```text
Are the predicted fields useful as fixed deployable features?
```

### Warm-Start Expanded ParT

Load the HLT ParT baseline. Expand the particle input projection to accept
extra fields. Initialize the new field weights to zero or tiny values.

Then train:

```text
phase 1: train new projection/adapter/head, keep most ParT frozen
phase 2: unfreeze upper ParT layers
phase 3: gentle full fine-tune
```

This is stable and comparable to prior AV10-style runs.

### From-Scratch Augmented ParT

Train ParT from scratch on augmented particles.

This tests whether the model learns to rely on residual fields more naturally
when it does not begin as a pure HLT particle model.

### Joint Fine-Tuning

After reconstructor pretraining, train reconstructor and ParT together:

```text
L_total = CE + lambda_reco * L_reco + lambda_kd * L_logit_KD
```

Keep a small reconstruction loss alive so the residual fields do not collapse
into arbitrary hidden features.

This is expected to be the strongest single-model path.

## Campaign Runs

### Tier A: Baselines And References

`A0`: HLT ParT baseline.

`A1`: HLT ParT schedule-matched fine-tune-only control.

`A2`: larger or parameter-matched HLT ParT control.

`A3`: optional external AV10 feature MLP adapter reference on the same
split/cache.

`A4`: optional external Offline ParT reference.

Purpose:

```text
A0 gives the HLT floor.
A1 tests whether schedule/fine-tuning alone explains gains.
A2 tests whether gains are just extra ParT capacity.
A3 compares against the best previous adapter direction when queued through
the AV10 runner.
A4 gives the offline ceiling when queued through the offline reference runner.
```

### Tier B: Oracle Local Residual Fields

`B0`: oracle local residual fields at one radius.

`B1`: oracle multi-radius local residual fields.

`B2`: oracle pT/density fields only.

`B3`: oracle pT/density plus multiplicity.

`B4`: oracle all fields:

```text
pt/density
centroid
multiplicity
composition
merge/split/reliability
```

Purpose:

```text
Find whether the target has real tagging value.
```

If `B1/B4` do not beat `A0/A1` clearly, do not spend huge GPU time on
predicting these fields.

### Tier C: Reconstructor Variants

`C0`: best cross-attentive local residual transformer:

```text
geometry bias
global/state context
multi-radius heads
uncertainty heads optional
jet consistency loss optional
```

`C1`: no canonical/global state context.

`C2`: no local geometry bias.

`C3`: DeepSets/global-context predictor.

`C4`: local particle-neighborhood transformer, no canonical tokens.

`C5`: uncertainty-aware version of `C0`.

`C6`: `C0` plus stronger jet-consistency/global-flow loss.

Expected behavior:

```text
C0 should be the best default.
C5 or C6 may be second-best and may produce more tagger-useful fields even if
their raw MAE is not best.
```

Metrics:

```text
per-field MAE/MSE
per-radius MAE/MSE
delta_log_pt MAE
composition residual MAE
merge/split AUC
reliability calibration
global consistency error
Phi_hlt baseline error vs Phi_pred error
```

The most important diagnostic is:

```text
error(predicted_fields, oracle_fields)
compared to
error(zero/no-correction, oracle_fields)
```

### Tier D: Predicted Fields Into ParT

Use `C0` as the default reconstructor unless noted.

`D0`: frozen `C0` reconstructor, train augmented ParT from scratch.

`D1`: frozen `C0` reconstructor, warm-start HLT ParT with expanded input
projection.

`D2`: pretrain `C0`, then joint fine-tune reconstructor plus ParT with:

```text
CE + reconstruction loss
```

`D3`: joint from scratch, CE only.

`D4`: joint from scratch with:

```text
CE + reconstruction loss
```

`D5`: pretrain `C0`, joint fine-tune with:

```text
CE + reconstruction loss + offline-teacher logit KD
```

`D6`: same as `D5`, but use the second-best Tier C reconstructor, expected to
be `C5` or `C6`.

Purpose:

```text
Find the best way to turn predicted residual fields into tagging gains.
```

Expected winner:

```text
D5
```

Rationale:

```text
The reconstructor starts physically meaningful.
Joint fine-tuning lets fields become tagging-useful.
Reconstruction loss prevents arbitrary collapse.
Offline-teacher KD adds a privileged tagging signal without using offline data
at inference.
```

### Tier E: Field Importance

Use the best Tier D training recipe, expected to be `D5`.

`E0`: pT/density only.

`E1`: geometry/centroid only.

`E2`: multiplicity only.

`E3`: composition only.

`E4`: reliability/uncertainty only.

`E5`: pT/density plus multiplicity.

`E6`: all fields.

Purpose:

```text
Identify what ParT actually uses.
```

Expected strongest interpretable subset:

```text
E5 = pT/density + multiplicity
```

### Tier F: Controls

`F0`: random residual fields with the same shape.

`F1`: residual fields shuffled across jets.

`F2`: residual fields shuffled among particles within each jet.

`F3`: residual fields with radius channels permuted.

`F4`: learned extra field generator with no reconstruction target.

`F5`: parameter-matched augmented ParT with blank/zero residual fields.

Purpose:

```text
Separate true residual information from extra capacity, regularization, or
optimization effects.
```

The method is only convincing if:

```text
real predicted residual fields > random/shuffled/blank controls
```

### Tier G: Fusion And Complementarity

`G0`: logit fusion of `A0` and the best single residual-field model.

`G1`: residual-field seed ensemble:

```text
same best recipe
different seeds
```

`G2`: complementary reconstructor ensemble:

```text
best C0-based model
+ best C5/C6-based model
```

`G3`: complementary field-subset ensemble:

```text
best all-field model
+ best pT/density-only model
+ best composition/reliability-heavy model
```

`G4`: future particle-view fusion of complementary residual-field models.

`G5`: future hybrid particle-view plus logit fusion.

`G5` should:

```text
1. Train several complementary residual-field models.
2. Concatenate their residual fields at particle level.
3. Learn per-particle gates over the residual views.
4. Train one ParT on the fused particle-view input.
5. Logit-fuse that model with the best single residual-field model or seed
   ensemble.
```

Expected best future overall:

```text
G5
```

But `G5` is not the cleanest scientific story. The clean story comes from
oracle value, predicted-field value, and controls.

## Reporting Requirements

Every run must write:

```text
run_report.json
training_curves.json
epoch_metrics.csv
source_metadata.json
```

The final report must include:

```text
accuracy
cross entropy
macro AUC if available
per-class accuracy
confusion matrix
final-test metrics for allowed primary runs
model-val selection metrics
stack-val metrics
```

For reconstructors:

```text
overall MAE/MSE
per-field MAE/MSE
per-radius MAE/MSE
zero-baseline MAE/MSE
relative improvement over zero/no-correction baseline
global consistency error
merge/split AUC
uncertainty calibration
```

For taggers with residual fields:

```text
field usage diagnostics
input projection norm by field group
gate statistics if gated
residual-field ablation at eval time
shuffle sensitivity
per-class gain/loss vs A0
```

For fusions:

```text
member metrics
fusion weights
per-class fusion gains
agreement/disagreement accuracy
oracle/control comparison
```

## Provenance And Safety Requirements

All reports must enforce:

```text
same split manifest hash
same HLT cache hash
same offline cache hash for oracle/training targets
same label names/order
same HLT profile and strength
same final-test confirmation policy
```

Final-test evaluation must be deployment-clean:

```text
No offline particles.
No oracle fields.
No teacher logits.
No offline-derived residual targets.
```

Oracle runs are model-val/stack-val diagnostics unless explicitly marked as
non-deployable final-test diagnostics.

## Implementation Plan

### Step 1: Target Cache Builder

Implement a cache builder that reads:

```text
split_manifest.json.gz
HLT cache
offline cache
```

and writes:

```text
local_residual_fields/<split>_targets.npz
local_residual_fields/<split>_metadata.json
```

The cache should contain:

```text
target_fields: [N, max_particles, field_dim]
target_mask: [N, max_particles]
field_names
field_groups
radius_names
zero_baseline_metrics
source hashes
```

### Step 2: Dataset And Collate

Create datasets that return:

```text
HLT particle inputs
labels
target residual fields
target mask
optional oracle fields
optional offline teacher logits
```

Keep masking strict. Invalid particles must not contribute to reconstruction
losses or tagger fields.

### Step 3: Reconstructor Model

Implement:

```text
LocalResidualFieldTransformer
```

with:

```text
particle encoder
geometry-biased attention
global/context tokens
multi-radius heads
uncertainty heads
field-group heads
```

Also implement simpler ablation models:

```text
DeepSets reconstructor
local particle-only transformer
no-geometry-bias transformer
uncertainty-aware transformer
jet-consistency transformer
```

### Step 4: Reconstructor Training

Implement Tier C training:

```text
train_local_residual_reconstructor.py
```

It should support:

```text
field subsets
loss weights
uncertainty loss
jet consistency loss
early stopping by model_val MAE or weighted reconstruction score
stack_val evaluation
```

### Step 5: Augmented ParT Tagger

Implement an augmented particle tagger that accepts:

```text
original 19 HLT features + residual field vector
```

Support:

```text
oracle fields
predicted fields from frozen reconstructor
joint reconstructor + tagger training
warm-start expanded projection
from-scratch ParT
offline-teacher logit KD
```

### Step 6: Oracle And Controls

Implement Tier B and Tier F:

```text
oracle residual fields
random fields
cross-jet shuffled fields
within-jet shuffled fields
radius-permuted fields
blank fields
learned no-target fields
```

Controls must have the same tensor shapes and similar trainable parameter
counts where relevant.

### Step 7: Fusion Models

Implement:

```text
logit fusion
seed ensemble
complementary reconstructor ensemble
field-subset ensemble
```

Fusion must fail closed if required member predictions are missing.

The code should expose and test the particle-view gated fusion module, but
`G4/G5` are not part of the required submitter until a concrete particle-view
training runner exists. They should be treated as follow-up add-ons, not as
completed primary campaign jobs.

### Step 8: Slurm Submitters

Write submitters for:

```text
pilot campaign
high-data campaign
cache-only target generation
reconstructor-only Tier C
full implemented A0-G3 primary campaign
report generation
```

Support both:

```text
sporcsubmit/tier3
tigris
```

but keep environment variables explicit so cluster-specific settings do not
leak between submissions.

### Step 9: Final Report

Write:

```text
write_local_residual_field_report.py
```

The report should produce:

```text
summary.md
tagger_metrics.csv
reconstructor_metrics.csv
oracle_gap.csv
control_gap.csv
field_importance.csv
fusion_metrics.csv
provenance_audit.json
```

The report must exit nonzero if required primary runs are missing, failed, or
provenance-inconsistent.

### Step 10: Tests

Add focused tests for:

```text
target cache shape and masks
strength/provenance hash checks
soft-assignment local target calculation
phi wrapping in centroid targets
field subset selection
oracle target path
shuffled/random controls
frozen reconstructor path
joint training loss composition
fusion fail-closed behavior
report provenance enforcement
```

## Decision Logic

The campaign should be interpreted in this order:

1. `B1/B4` oracle value:

```text
Do perfect local residual fields help?
```

2. `C0-C6` prediction quality:

```text
Can we predict the useful fields?
```

3. `D0-D6` deployable tagging:

```text
Do predicted fields improve ParT?
```

4. `E0-E6` field importance:

```text
Which fields matter?
```

5. `F0-F5` controls:

```text
Is this real residual information or just extra capacity?
```

6. `G0-G3` fusion:

```text
Can complementary residual views produce the best headline result?
```

## Expected Outcome

Best clean single model:

```text
D5
```

Best alternative if uncertainty matters:

```text
D6 using C5 or C6
```

Best interpretable field subset:

```text
E5 = pT/density + multiplicity
```

Best future overall:

```text
G5
```

Most important sanity checks:

```text
B1/B4 oracle value
F4/F5 capacity controls
A4 offline ceiling
```

If oracle fields help a lot and predicted fields help modestly, the bottleneck
is reconstructor quality.

If oracle fields do not help, the target definition is wrong.

If predicted fields help but controls help just as much, the gain is probably
capacity or optimization.

If predicted fields beat controls and fusion adds more, this becomes a serious
candidate branch for beating HLT ParT.
