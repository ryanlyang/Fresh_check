# Architecture-View 10-Class Ensemble Plan

## Short Name

AV10 Ensemble, or Architecture-View 10-Class Ensemble.

## Goal

Use the promising architecture-view residual idea in the harder and more general
JetClass 10-class setting, then compare it against strong ensemble baselines.

The binary QCD-vs-Hgg pilot gave the first clear sign that architecture-specific
latent context can help a strong HLT ParT:

```text
HLT ParT baseline final FPR@50: 0.020624
PFN latent -> ParT final FPR@50: 0.019956
all views -> ParT final FPR@50: 0.020012
```

That result is too interesting to leave in only a binary setting. The next
question is broader:

```text
Can architecture-specific latent views improve a 10-class HLT ParT,
and can an ensemble of those view-conditioned ParTs beat normal model ensembles?
```

This plan intentionally focuses on the tagger/ensemble version first. Auxiliary
offline-correction targets can be added later as a V2 once the 10-class
architecture-view ensemble is measured cleanly.

## Setup

Task:

```text
10-class JetClass tagging on fixed HLT-degraded inputs
```

Classes:

```text
QCD
Hbb
Hcc
Hgg
H4q
Hqql
Zqq
Wqq
Tbqq
Tbl
```

Primary 10-class metrics:

```text
accuracy
macro per-class accuracy
macro one-vs-rest AUC, if available
per-class AUC, if available
confusion matrix
```

Important binary projections:

```text
QCD vs Hgg:
  score = logit_Hgg - logit_QCD

QCD vs Hbb:
  score = logit_Hbb - logit_QCD

QCD vs Tbqq:
  score = logit_Tbqq - logit_QCD
```

The QCD-vs-Hgg projection is especially useful because it lets us compare
against the current binary story while still training in the richer 10-class
setting.

Split discipline:

```text
model_train:
  train individual taggers

model_val:
  checkpoint selection for individual taggers

stack_train:
  fit cheap ensemble/fusion models

stack_val:
  select ensemble/fusion strategy and hyperparameters

final_test:
  final held-out reporting only
```

The final report should clearly separate:

```text
individual model performance
simple ensemble performance
learned fusion performance
binary projection performance
```

## Core Hypothesis

Standalone PN, PFN, and PCNN may underperform ParT as final classifiers, but
their internal reasoning can still be useful. Instead of asking them to vote
directly, give each architecture a chance to create a latent per-particle view,
then let ParT remain the final reasoner.

The architecture-view model is:

```text
HLT particles
  -> architecture-specific branch creates per-particle latent view
  -> latent view is projected into ParT embedding space
  -> delta_h is added to ParT's particle embedding
  -> original ParT blocks/head produce 10-class logits
```

For one jet and particle `i`:

```text
h_i_base = ParTEmbed(F_i)
view_i   = ArchitectureView(F_1, ..., F_N)_i
delta_i  = ProjectToParTSpace(view_i)
h_i      = h_i_base + gate_i * delta_i
```

Then the normal ParT attention stack runs on `h_i`.

The claim we want to test is not merely:

```text
PN/PFN/PCNN are useful weak models.
```

The stronger claim is:

```text
PN/PFN/PCNN create useful latent context that a strong ParT can exploit.
```

## Model Families

### 1. Standard 10-Class Baselines

Train or reuse strong HLT models:

```text
hlt_part_baseline_10class
hlt_pn_baseline_10class
hlt_pfn_baseline_10class
hlt_pcnn_baseline_10class
```

These are normal standalone classifiers. They provide the fair advisor-facing
baseline ensemble:

```text
standalone_arch_ensemble =
  ensemble(ParT, PN, PFN, PCNN)
```

### 2. ParT Seed Ensemble Control

Train or reuse multiple 10-class HLT ParT seeds:

```text
hlt_part_seed0
hlt_part_seed1
hlt_part_seed2
hlt_part_seed3
```

This is the most important capacity control. The architecture-view ensemble
uses multiple ParT final reasoners, so it must beat or at least compare cleanly
against:

```text
4x ParT seed ensemble
```

Without this control, a critic can say:

```text
The gain is just multiple ParTs, not architecture-view context.
```

### 3. Architecture-View 10-Class Models

Train four architecture-view ParT models:

```text
av10_part_context_to_part
av10_pn_context_to_part
av10_pfn_context_to_part
av10_pcnn_context_to_part
av10_all_views_to_part
```

The core requested ensemble uses:

```text
PN latent   -> ParT
PFN latent  -> ParT
PCNN latent -> ParT
all latents -> ParT
```

I would also include `ParT context -> ParT` if the implementation is simple,
because it is a useful "same architecture view" control. But the first ensemble
can be the four models above:

```text
architecture_view_ensemble =
  ensemble(
    PN latent -> ParT,
    PFN latent -> ParT,
    PCNN latent -> ParT,
    all latents -> ParT
  )
```

### 4. Controls

Controls matter because architecture-view models can win for several reasons:

```text
more parameters
more ParT fine-tuning
extra residual path
real architecture-specific information
```

Required controls:

```text
av10_baseline_recheck:
  exact HLT ParT checkpoint loaded through the AV10 path with zero injection

av10_context_mlp_control:
  context adapter with no PN/PFN/PCNN architecture views

av10_random_view_control:
  same view capacity, randomized feature semantics

4x_part_seed_ensemble:
  same number of ParT final reasoners, no architecture views

standalone_arch_ensemble:
  PN/PFN/PCNN/ParT as normal final classifiers
```

The key interpretation rule:

```text
If AV10 beats standalone_arch_ensemble but not 4x_part_seed_ensemble,
then architecture views may be useful but not beyond more ParT capacity.

If AV10 beats both standalone_arch_ensemble and 4x_part_seed_ensemble,
then architecture-view latent context is doing something genuinely interesting.
```

## Fusion Strategies

Fusion is cheap after prediction caching, so the plan should test many fusion
methods. All fusers must be trained on `stack_train`, selected on `stack_val`,
and evaluated once on `final_test`.

### 1. Uniform Logit Mean

```text
fused_logits = mean(logits_m)
```

This should be the default reference ensemble. It is simple and hard to
overfit.

### 2. Uniform Probability Mean

```text
fused_prob = mean(softmax(logits_m))
```

This sometimes works better when models have different calibration.

### 3. Temperature-Scaled Logit Mean

Fit one temperature per model on `stack_train`:

```text
logits_m_scaled = logits_m / T_m
fused_logits = mean(logits_m_scaled)
```

Select temperatures by stack NLL or stack accuracy.

### 4. Validation-Weighted Logit Mean

Use nonnegative scalar weights:

```text
fused_logits = sum_m w_m * logits_m
sum_m w_m = 1
```

Fit weights on `stack_train`, select on `stack_val`.

Start with simple constrained optimization or grid/random search. The number of
models is small, so this is cheap.

### 5. Classwise Weighted Logit Mean

Use one weight per model and class:

```text
fused_logits_c = sum_m w_{m,c} * logits_{m,c}
```

This can capture cases where one view helps Hgg while another helps top or b/c
classes. It is higher risk than scalar weights, so regularize toward uniform.

### 6. Ridge/Logistic Stacking

Train a small linear stacker:

```text
input  = concat(logits_1, logits_2, ..., logits_M)
output = 10-class logits
```

Use strong L2 regularization and early selection on `stack_val`.

Variants:

```text
linear logits -> logits
linear probabilities -> logits
logits + entropy/confidence features -> logits
```

This is likely the strongest cheap fuser, but also easiest to overfit. It must
never touch final-test labels during fitting or selection.

### 7. Binary Projection Fusion

For advisor-facing binary projections, also test fusing the binary scores:

```text
score_QCD_vs_Hgg_m = logit_Hgg_m - logit_QCD_m
fused_score = sum_m w_m * score_m
```

This is useful because the best 10-class fuser may not be the best QCD-vs-Hgg
projection fuser.

Report both:

```text
10-class selected fuser
binary-projection selected fuser
```

But keep the distinction explicit to avoid cherry-picking.

## Prediction Cache Format

Each trained model should write prediction caches for:

```text
model_val
stack_train
stack_val
final_test
```

At minimum:

```text
labels: [N]
logits: [N, 10]
preds: [N]
indices or jet ids
model metadata
checkpoint hash
split manifest hash
HLT cache hash
label names
```

For binary projections, the report can compute:

```text
score(A vs B) = logit_B - logit_A
```

Do not store only probabilities. Store logits too.

## Evaluation

### 10-Class Metrics

Report for each individual model and ensemble:

```text
accuracy
macro per-class accuracy
mean cross entropy
confusion matrix
per-class recall
per-class precision
per-class one-vs-rest AUC if available
macro AUC if available
```

### Binary Projection Metrics

For selected pairs:

```text
QCD vs Hgg
QCD vs Hbb
QCD vs Tbqq
QCD vs Wqq
QCD vs Zqq
```

Report:

```text
AUC
FPR at 30% signal efficiency
FPR at 50% signal efficiency
background rejection at 50% signal efficiency
validation-threshold final-test FPR
```

For validation-threshold final-test:

```text
choose threshold on stack_val or model_val at target signal efficiency
apply threshold once to final_test
```

The final report should clearly label oracle final-test FPR versus validation
threshold final-test FPR.

## Expected Readouts

### Strong Positive Result

```text
AV10 ensemble beats:
  single HLT ParT
  standalone PN/PFN/PCNN/ParT ensemble
  4x HLT ParT seed ensemble
```

Interpretation:

```text
architecture-specific latent context provides useful information beyond
ordinary score ensembling and beyond extra ParT capacity.
```

### Mixed Positive Result

```text
AV10 ensemble beats standalone arch ensemble
but ties 4x ParT seed ensemble
```

Interpretation:

```text
architecture views are useful, but the gain may be comparable to more ParT
capacity or seed diversity.
```

### Single-View Positive Result

```text
PFN latent -> ParT beats all-view and other single views
```

Interpretation:

```text
PFN-style global particle summaries are the useful latent source.
Focus V2 on PFN latent context instead of forcing all views.
```

### Control Failure

```text
random-view control or context-MLP control matches AV10
```

Interpretation:

```text
the win may be generic residual capacity or fine-tuning, not architecture
semantics.
```

This is still useful, but it weakens the architecture-view claim.

## Implementation Strategy

The cleanest implementation is to extend the existing
`teacher_logit_reco/architecture_view_part` package rather than create a
separate system.

Required upgrades:

```text
num_classes = 10 support
label_names = full JetClass class list
selection_metric = 10-class accuracy or macro per-class accuracy
binary projection reporting from 10-class logits
prediction cache writing for stack_train/stack_val/final_test
ensemble/fusion runner
final report comparing individual models, controls, and fusions
```

The binary QCD-vs-Hgg path should remain untouched.

## Implementation Steps

### Step 1: Add 10-Class Architecture-View Config And Training Support

Extend architecture-view training to support:

```text
num_classes = 10
full label names
10-class CE loss
10-class checkpoint metadata validation
selection by accuracy or macro per-class accuracy
optional binary projection metric logging
```

Add variants:

```text
av10_baseline_recheck
av10_part_context_to_part
av10_pn_context_to_part
av10_pfn_context_to_part
av10_pcnn_context_to_part
av10_all_views_to_part
av10_random_view_control
av10_context_mlp_control
```

Tests:

```text
10-class config validation
10-class logits shape [B, 10]
zero-injection baseline recovery
binary projection score extraction
strict label metadata checks
```

### Step 2: Add Prediction Cache Runner

Implement:

```text
scripts/cache_architecture_view_10class_predictions.py
```

For each model, write:

```text
predictions/model_val_logits.npz
predictions/stack_train_logits.npz
predictions/stack_val_logits.npz
predictions/final_test_logits.npz
prediction_manifest.json
```

Each cache must include:

```text
labels
logits
preds
indices or jet ids if available
split metadata
checkpoint hash
model variant
label names
```

Tests:

```text
cache shape validation
metadata compatibility checks
refuse mismatched labels/order
```

### Step 3: Add Ensemble And Fusion Runner

Implement:

```text
scripts/run_architecture_view_10class_fusion.py
teacher_logit_reco/architecture_view_part/ensemble.py
```

Fusion modes:

```text
uniform_logit_mean
uniform_probability_mean
temperature_scaled_logit_mean
scalar_weighted_logit_mean
classwise_weighted_logit_mean
ridge_logit_stacker
binary_projection_weighted
```

Training discipline:

```text
fit on stack_train
select on stack_val
evaluate on final_test
```

Tests:

```text
uniform fusion correctness
temperature scaling improves or preserves stack NLL on toy data
weighted fusion no final-test leakage
binary projection FPR@50 computation
```

### Step 4: Add Sbatch Submitters

Implement:

```text
sbatch/run_train_architecture_view_10class_part.sh
sbatch/run_cache_architecture_view_10class_predictions.sh
sbatch/run_architecture_view_10class_fusion.sh
sbatch/submit_architecture_view_10class_experiment.sh
```

The submitter should queue:

```text
individual baseline and architecture-view models
prediction cache jobs after each model
fusion job after all caches
final report job
```

Default pilot variants:

```text
av10_baseline_recheck
av10_pn_context_to_part
av10_pfn_context_to_part
av10_pcnn_context_to_part
av10_all_views_to_part
av10_random_view_control
av10_context_mlp_control
```

Optional controls:

```text
4x ParT seed ensemble
standalone PN/PFN/PCNN/ParT ensemble
```

### Step 5: Add Final Report

Implement:

```text
scripts/write_architecture_view_10class_report.py
```

Report sections:

```text
individual model table
architecture-view ensemble table
standalone architecture ensemble table
4x ParT seed ensemble table
binary projection table
per-class breakdown
fusion strategy comparison
controls and failure checks
```

The report should answer:

```text
Did AV10 beat single ParT?
Did AV10 beat standalone model ensemble?
Did AV10 beat 4x ParT seed ensemble?
Which class pairs improved?
Which view branch helped most?
Did random/context controls erase the claim?
```

### Step 6: Run Pilot, Then Promote

Pilot:

```text
500k model_train
150k model_val
500k stack_train if available, otherwise 150k
150k stack_val
500k final_test
45 epochs
```

Promote to high data only if:

```text
AV10 individual models beat baseline ParT, or
AV10 ensemble beats 4x ParT or standalone architecture ensemble, or
PFN-context shows a stable repeat of the binary result
```

High-data:

```text
3M model_train
1M model_val
1M stack_train
1M stack_val
1M final_test
```

## V2: Auxiliary Latent Shaping

Do not start here. Add this only after the tag-only 10-class ensemble is
measured.

V2 idea:

```text
architecture-view latent
  -> auxiliary heads predict HLT-vs-offline high-level residuals
  -> same latent is injected into ParT
```

Candidate general 10-class auxiliary targets:

```text
jet pT residual
jet mass residual
jet axis residual
charged/neutral/photon/lepton energy fraction residuals
charged/neutral/photon/lepton multiplicity residuals
track/displacement summary residuals
offline teacher 10-class logit residual
```

Required controls:

```text
tag-only AV10
physics-aux only
teacher-aux only
physics + teacher
shuffled auxiliary targets
latent-zeroed-at-eval
latent-shuffled-at-eval
```

The V2 claim is:

```text
offline/teacher auxiliary supervision shapes a latent that ParT actually uses
for 10-class tagging.
```

## Current Recommendation

Implement the tag-only AV10 ensemble first.

The first run should prioritize:

```text
PFN latent -> ParT
PN latent -> ParT
PCNN latent -> ParT
all latents -> ParT
uniform and learned logit fusion
context/random controls
```

If PFN remains the strongest single latent source, V2 should focus around
PFN-context and auxiliary high-level correction targets instead of trying to
make every architecture branch equally important.
