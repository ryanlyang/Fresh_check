# Target-Conditioned Pairwise Denoising ParT Plan

## Short Version

This plan turns the reconstructor idea into a more structured, ParT-native
denoising problem.

Instead of asking a generic transformer to reconstruct an entire jet, we train a
small target-conditioned denoiser to answer one question for each HLT particle:

```text
Given this HLT particle i and the rest of the HLT jet,
what small correction would make particle i more like its offline version?
```

The denoiser uses ParT-style pairwise attention, but its objective is not jet
classification. Its objective is a per-particle correction field:

```text
delta_log_pt_i
delta_eta_i
delta_phi_i
delta_log_energy_i
uncertainty_i
```

The final deployable tagger then receives both:

```text
original HLT particle features
predicted denoising deltas / uncertainties
```

and uses a normal HLT-only ParT to classify the jet.

The central bet is:

```text
Class labels alone do not efficiently force ParT to learn detector-correction
geometry, but paired HLT/offline supervision can.
```

So this is not merely "another transformer before ParT." It is an explicitly
supervised measurement-correction adapter whose outputs are exposed to the final
tagger.

## Why This Is Different From Earlier Reconstructors

Earlier reconstructor-style experiments were broad:

```text
HLT jet -> encoder/reconstructor -> reconstructed-ish jet/latent -> tagger
```

That can work at low data, but it is easy for gains to wash out at high data,
because the final ParT can eventually learn many of the same broad
classification-relevant patterns from labels.

This plan is narrower and more mechanistic:

```text
particle i is the query
all HLT particles are context
pairwise i-j geometry biases attention
the output is only a correction for particle i
```

The denoiser is trained with privileged offline targets:

```text
HLT particle_i -> offline particle_i residual
```

Then the tagger can use the denoiser outputs as explicit side information:

```text
this particle might need +delta_log_pt
this particle might need -delta_eta
this particle's correction is uncertain
this particle looks locally distorted
```

That is a different source of information than a tag-only ParT receives.

## Core Hypothesis

ParT already performs particle-to-particle attention:

```text
new_h_i = sum_j attention(i, j) * value_j
```

and ParT already uses pairwise physics features as an attention bias. Therefore,
"particle i looks at all other particles" is not new by itself.

The new ingredient is the training pressure:

```text
ParT tagger:
  Which particle-particle relations help classify the jet?

Target-conditioned denoiser:
  Which particle-particle relations help correct particle i's measured state?
```

Those are not the same objective.

The denoising task may force the model to learn a local correction field that is
useful to classification but not efficiently discoverable from labels alone,
especially when HLT degradation is mild and subtle.

This is why the strategy has a plausible high-data path:

```text
It adds privileged per-particle correction supervision,
not just more tagger capacity.
```

## Deployment Contract

At inference, the final system must be HLT-only:

```text
HLT particles
  -> target-conditioned denoiser
  -> predicted deltas + uncertainties
  -> HLT ParT tagger
  -> class logits
```

Offline particles are used only during training to define denoising targets.
The final-test tagger evaluation must not load offline targets or teacher
features except in clearly separated diagnostic-only passes.

## Recommended Data Regime

Use the new realistic HLT profile:

```text
hlt_profile: fixed_hlt_v2_realistic
hlt_degradation_strength: 1.0
```

This is the intended mild-HLT target regime:

```text
offline ParT accuracy: expected around low 80s
HLT ParT accuracy:     desired roughly 2-3 points lower
```

The model is not primarily designed to rescue an aggressively destroyed HLT
view. It is designed to recover useful correction structure when HLT is close
enough to offline that residual prediction is learnable.

Primary task:

```text
10-class JetClass classification
```

Primary split sizes:

```text
pilot:
  model_train: 500k
  model_val:   150k
  final_test:  150k

high-data:
  model_train: 5M
  model_val:   1M
  final_test:  1M
```

The same split manifest and HLT cache contract must be shared by:

```text
HLT ParT baseline
denoising pretraining
final denoiser-augmented tagger
controls
reports
```

## Particle Alignment Contract

This is the most important implementation detail.

The simplest denoising target assumes HLT particle `i` corresponds to offline
particle `i`:

```text
target_delta_i = offline_i - hlt_i
```

That is clean for smearing-only degradation:

```text
same particle count
same particle identity
only pt/eta/phi/energy are smeared
```

But realistic HLT can also include:

```text
dropped particles
merged particles
local reassignment
mask changes
sorting changes
```

Therefore the denoising dataset must expose an explicit alignment contract.

### Required Alignment Metadata

When building HLT v2 caches, store per-HLT-particle provenance if possible:

```text
source_offline_index[jet, hlt_particle]
source_weight[jet, hlt_particle, maybe k]
source_status:
  direct
  smeared
  merged
  dropped
  synthetic/padded
```

For the first implementation, support these target modes:

```text
aligned_direct:
  Train residuals only for HLT particles with one known offline source.

aligned_direct_plus_merge_centroid:
  For merged HLT particles, target the weighted offline-source centroid.

matched_hungarian:
  If provenance is missing, match HLT to offline particles by eta/phi/pt cost.

smearing_only_debug:
  Build a special HLT v2 profile with no drops/merges, only smearing, to
  validate the architecture.
```

The serious run should start with:

```text
aligned_direct
```

because it gives the cleanest supervision. Merge/drop recovery can be added
after the denoiser proves it helps on direct residuals.

## Architecture

The model has two deployable components:

```text
1. Target-conditioned denoiser
2. Final ParT tagger with denoiser features
```

### Component 1: Target-Conditioned Denoiser

For a jet with `N` HLT particles:

```text
F_i = canonical HLT feature row for particle i
M_i = valid-particle mask
```

Build a context representation:

```text
C_j = ContextEmbed(F_j)
```

Build a target query for each particle:

```text
Q_i = TargetEmbed(F_i)
```

Then perform cross-attention:

```text
score_i,j =
  Q_i dot K_j / sqrt(d)
  + base_pair_bias_i,j
  + denoise_pair_bias_i,j
  + optional_local_kernel_i,j

A_i,j = softmax_j(score_i,j)

Z_i = sum_j A_i,j * V_j
```

Finally predict bounded residuals and uncertainties:

```text
Head(Z_i, F_i) ->
  delta_log_pt_i
  delta_eta_i
  delta_phi_i
  delta_log_energy_i
  log_sigma_log_pt_i
  log_sigma_eta_i
  log_sigma_phi_i
  log_sigma_log_energy_i
  reliability_i
```

The deltas are bounded:

```text
delta_log_pt     = max_delta_log_pt     * tanh(raw_delta_log_pt)
delta_eta        = max_delta_eta        * tanh(raw_delta_eta)
delta_phi        = max_delta_phi        * tanh(raw_delta_phi)
delta_log_energy = max_delta_log_energy * tanh(raw_delta_log_energy)
```

Default first-run bounds:

```text
max_delta_log_pt:     0.30
max_delta_eta:        0.08
max_delta_phi:        0.08
max_delta_log_energy: 0.30
```

These are intentionally conservative. The denoiser should learn small,
plausible corrections, not invent new particles.

### Why Cross-Attention Rather Than Plain Self-Attention

Plain self-attention updates all particles for a generic representation goal.
The denoising cross-attention is explicitly query-targeted:

```text
query set:
  the particles being corrected

key/value set:
  the full jet context
```

In implementation, all particles can still be processed in parallel. The
distinction is conceptual and architectural:

```text
each output row Z_i exists to correct particle i
```

not to build a generic classification token.

### Denoising Pair Bias

The denoising pair bias should not merely say "near particles matter."
It should learn which context particles are useful evidence for correcting
target particle `i`.

Use a pair feature builder with:

```text
delta_eta_i,j
sin(delta_phi_i,j)
cos(delta_phi_i,j)
delta_R_i,j
log(delta_R_i,j)
delta_log_pt_i,j
log_pair_mass_i,j
log_relative_kt_i,j
z_i,j
same_or_compatible_pid
charge_product
charged_neutral_relation
target_pt_rank
context_pt_rank
local_density_around_i
local_density_around_j
```

Then:

```text
denoise_pair_bias_i,j = PairMLP(pair_features_i,j)
```

For stability:

```text
PairMLP final projection zero-initialized
global denoise_pair_gate initialized near zero
bias clipped or tanh-bounded before addition
```

The optional local kernel can provide a weak prior:

```text
local_kernel_i,j = alpha * exp(-delta_R_i,j^2 / tau^2)
```

with `alpha` initialized at zero or very small. This prevents hard-coding
nearest-neighbor behavior while still giving the model an easy local path.

### Component 2: Final Denoiser-Augmented ParT Tagger

The final tagger should keep the original HLT information, not replace it.

For each particle, build augmented features:

```text
F_aug_i = concat(
  F_hlt_i,
  predicted_delta_i,
  predicted_uncertainty_i,
  reliability_i,
  optional_corrected_feature_subset_i
)
```

There are two viable integration styles.

#### Preferred Style: Adapter Into ParT Embedding

Keep the canonical HLT ParT input unchanged:

```text
points_hlt
features_hlt
lorentz_vectors_hlt
mask_hlt
```

Use the denoiser outputs to produce an embedding residual:

```text
delta_h_i = MLP(F_aug_i)
h_i' = h_i + gate * delta_h_i
```

This is closest to the successful `av10_feature_mlp_adapter` family. It lets
ParT see correction information without corrupting the physical four-vector
inputs used by the pairwise machinery.

#### Secondary Style: Feature-Delta Input Adapter

Use the predicted deltas to construct corrected feature rows:

```text
F_corr_i = apply_delta(F_hlt_i, predicted_delta_i)
```

Then feed:

```text
features_corr
points_hlt
lorentz_vectors_hlt
mask_hlt
```

This can work, but it risks internal inconsistency if features are corrected
while Lorentz vectors and pairwise inputs remain HLT. Use this as an ablation,
not the primary run.

#### Best Full Version

The highest-upside model combines both but keeps them gated:

```text
HLT canonical features
  -> denoiser predicts delta_F_hat + uncertainty

delta_F_hat
  -> small bounded feature adapter gate_F
  -> optional F_corr

concat(F_hlt, delta_F_hat, uncertainty, reliability)
  -> embedding adapter gate_h
  -> delta_h

ParT embedding h
  -> h + gate_h * delta_h
  -> normal ParT attention/classifier
```

Start with embedding integration only. Add feature correction only after the
denoiser shows useful validation gains.

## Losses

### Denoiser Pretraining Loss

For aligned valid particles:

```text
L_delta =
  SmoothL1(delta_log_pt_pred,     delta_log_pt_true)
  + SmoothL1(delta_eta_pred,      delta_eta_true)
  + SmoothL1(delta_phi_pred,      wrapped_delta_phi_true)
  + SmoothL1(delta_log_energy_pred, delta_log_energy_true)
```

Use uncertainty weighting:

```text
L_nll =
  0.5 * exp(-log_var) * residual^2
  + 0.5 * log_var
```

The first implementation can use:

```text
L_reco = 0.5 * SmoothL1 + 0.5 * Gaussian NLL
```

Add regularizers:

```text
delta magnitude penalty
pair-bias magnitude penalty
uncertainty calibration penalty
masked-particle zero-output penalty
```

Optional auxiliary tasks:

```text
predict whether particle is direct/merged/dropped
predict local density change
predict correction confidence bucket
```

### Tagger Loss

For the final tagger:

```text
L_tag = CrossEntropy(logits, label)
```

Then use staged combined training:

```text
L_total =
  L_tag
  + lambda_reco * L_reco
  + lambda_delta * ||delta||^2
  + lambda_gate * gate_regularization
```

Recommended schedule:

```text
denoiser pretrain:
  lambda_reco = 1.0
  no tag loss

tagger warm start:
  lambda_reco = 0.1
  L_tag active

full fine-tune:
  lambda_reco decays to 0.01 or 0.0
```

The final tagger should not be forced to prefer perfect reconstruction over
better classification. Reconstruction is a useful prior, not the final goal.

## Training Schedule

### Stage 0: HLT And Offline Baselines

Train or load:

```text
offline ParT baseline
HLT v2 ParT baseline
```

Record:

```text
model_val accuracy/loss
final_test accuracy/loss
per-class accuracy
confusion matrix
HLT-vs-offline gap
```

### Stage 1: Smearing-Only Debug Denoiser

Before the full HLT v2 run, train on a smearing-only variant:

```text
same particle count
same particle identity
no drops
no merges
```

This validates:

```text
delta targets are correct
delta_phi wrapping is correct
attention masks are correct
denoiser improves residual RMSE
uncertainties correlate with residual error
```

Do not spend serious cluster time until this passes.

### Stage 2: Realistic HLT V2 Denoiser Pretraining

Train denoiser on `model_train` with `model_val` selection.

Selection metrics:

```text
primary:
  normalized residual RMSE on model_val

secondary:
  negative log likelihood
  delta_phi RMSE
  uncertainty calibration
  direct-particle residual RMSE
  merged-particle residual RMSE if enabled
```

Save:

```text
best_denoiser_model_val.pt
denoiser_run_report.json
denoiser_predictions_model_val.npz
denoiser_predictions_final_test.npz only during final report, not selection
```

### Stage 3: Frozen-Denoiser Tagger

Freeze the denoiser and train the final ParT adapter/tagger:

```text
HLT particles
  -> frozen denoiser outputs
  -> embedding adapter
  -> warm-started HLT ParT
```

This tests whether the denoising features themselves help.

### Stage 4: Joint Fine-Tune

Unfreeze parts of the denoiser with small LR:

```text
denoiser LR:       0.1x adapter LR
adapter LR:        normal
ParT LR:           0.05x to 0.2x adapter LR
pair-bias LR:      0.1x adapter LR
```

Keep a small reconstruction anchor for the first half of training:

```text
lambda_reco = 0.05 -> 0.01
```

Then optionally decay it to zero:

```text
let classification decide how to use the denoiser
```

### Stage 5: Final Evaluation

Use model-val selection only. Final-test is evaluated once after selection.

Report:

```text
HLT ParT baseline
denoiser features frozen tagger
denoiser features joint fine-tune
tag-only same-capacity adapter
smearing-only debug result
offline ParT reference
```

## Ablations And Controls

The controls are as important as the main model.

### A0: HLT ParT Baseline

Normal HLT v2 ParT.

Purpose:

```text
baseline to beat
```

### A1: Tag-Only Feature MLP Adapter

The successful AV10-style feature MLP adapter, no reconstruction pretraining.

Purpose:

```text
tests whether denoising supervision adds anything over the known adapter gain
```

### A2: Target Denoiser Features, Frozen

Pretrain denoiser, freeze it, train tagger on outputs.

Purpose:

```text
tests whether correction fields carry useful tag information
```

### A3: Target Denoiser Features, Joint Fine-Tune

Pretrain denoiser, then fine-tune denoiser + tagger jointly.

Purpose:

```text
highest-upside main candidate
```

### A4: Tag-Only Same Architecture

Same target-conditioned attention module, but trained only with tag loss.

Purpose:

```text
tests whether reconstruction supervision matters
```

### A5: Shuffled Residual Targets

Pretrain denoiser with residual targets shuffled across jets or particles.

Purpose:

```text
tests whether gains come from real HLT->offline correction structure
```

### A6: No Pair Bias Denoiser

Use cross-attention without pairwise denoising bias.

Purpose:

```text
tests whether pairwise physics bias is responsible
```

### A7: Local Kernel Only

Use hand-designed local `delta_R` kernel, no learned pair MLP.

Purpose:

```text
tests whether simple locality is enough
```

### A8: Pair Bias Only, No Deltas To Tagger

Train denoiser but pass only reliability/uncertainty or a latent summary.

Purpose:

```text
tests whether explicit corrected quantities are needed
```

### A9: Corrected Features Only

Feed corrected features instead of original HLT features.

Purpose:

```text
tests whether original HLT should always be preserved
```

Expected result:

```text
original HLT + deltas + uncertainty should beat corrected-only
```

because the tagger can decide when to trust the correction.

## Diagnostics

### Denoiser Diagnostics

Per split:

```text
delta_log_pt_rmse
delta_eta_rmse
delta_phi_rmse
delta_log_energy_rmse
normalized_rmse
nll
uncertainty_error_correlation
calibration by predicted uncertainty bucket
fraction valid direct targets
fraction merged targets
fraction dropped/no-target particles
```

Per class:

```text
residual RMSE by true class
mean predicted uncertainty by true class
mean correction magnitude by true class
```

Pairwise attention diagnostics:

```text
attention mass within delta_R < 0.05
attention mass within delta_R < 0.10
attention mass within delta_R < 0.20
attention mass to top-pt particles
attention entropy
pair-bias abs mean/p90/max
pair-bias gate value
```

### Tagger Diagnostics

Per split:

```text
accuracy
loss
macro AUC if available
per-class accuracy
confusion matrix
```

Adapter diagnostics:

```text
gate_h
delta_h norm mean/p90/max
delta_F_hat norm mean/p90/max
uncertainty mean/p90/max
classification accuracy by uncertainty bucket
```

Mechanism checks:

```text
tagger performance with denoiser outputs zeroed
tagger performance with deltas zeroed but uncertainties kept
tagger performance with uncertainties zeroed but deltas kept
tagger performance with shuffled denoiser outputs
```

These are the checks that tell us whether the model is using the denoiser for
real information or just gaining capacity/regularization.

## Expected Failure Modes

### 1. Denoiser Learns Pretty Corrections That Do Not Help Tagging

This is possible. Particle-level RMSE improvement does not guarantee better
classification.

Mitigation:

```text
keep original HLT features
pass uncertainty
allow tagger to ignore deltas
decay reconstruction loss during joint fine-tune
```

### 2. Denoiser Overfits Synthetic HLT Details

If HLT v2 is not realistic, the denoiser may learn artifacts.

Mitigation:

```text
run HLT v2 mild profile
validate across strength 0.8/1.0/1.2 if possible
check offline gap and constituent diagnostics
```

### 3. Alignment Noise Destroys Residual Targets

Dropped/merged particles can make one-to-one residuals invalid.

Mitigation:

```text
start with direct aligned particles only
store provenance in HLT cache
use merge-centroid targets later
use target masks and target-quality weights
```

### 4. Final Tagger Just Uses Extra Capacity

Mitigation:

```text
tag-only same architecture control
shuffled residual target control
no-pair-bias control
parameter-matched adapter control
```

## Why This Has A Real Shot At Holding At High Data

The reason to believe this can survive high-data scaling is not that the model
has more parameters. It is that it receives a different, denser training signal.

A 10-class label gives one supervision target per jet:

```text
jet -> class
```

Denoising gives many supervision targets per jet:

```text
particle_1 -> residual_1
particle_2 -> residual_2
...
particle_N -> residual_N
```

and those targets are physically tied to HLT/offline measurement differences.

If the denoiser only improves low-data performance, then it was mostly a data
efficiency tool. If it improves high-data performance, the interpretation is
stronger:

```text
paired HLT/offline residual supervision exposes useful correction structure
that class-label training alone does not fully exploit.
```

That is the scientific claim this plan is designed to test.

## Implementation Steps

### Step 1: Alignment-Aware Denoising Dataset

Add a dataset/cache layer that returns paired HLT/offline objects:

```text
hlt canonical inputs
offline canonical inputs
particle alignment/provenance
per-particle residual targets
target mask
target quality weights
```

Requirements:

```text
strict split manifest hash
strict HLT profile/strength metadata
no final-test target loading during training/selection
delta_phi wrapping tests
strength=0 identity tests
smearing-only debug profile support
```

### Step 2: Target-Conditioned Pairwise Denoiser Module

Implement:

```text
TargetConditionedPairwiseDenoiser
PairwiseDenoisingFeatureBuilder
DenoisingPairBiasEncoder
DenoisingResidualHead
```

Forward output:

```text
delta predictions
log-variance/uncertainty predictions
reliability predictions
attention/pair-bias diagnostics
```

Add tests for:

```text
shape/mask correctness
zero-init baseline behavior
finite outputs
no output for masked particles
pair-bias changes attention
```

### Step 3: Denoiser Pretraining Script

Add a script and sbatch runner:

```text
scripts/train_target_conditioned_denoising_part.py
sbatch/run_train_target_conditioned_denoising_part.sh
```

It should train only on `model_train`, select on `model_val`, and write:

```text
best_denoiser_model_val.pt
run_report.json
training_curves.json
model_val_diagnostics.json
```

### Step 4: Denoiser-Augmented ParT Tagger

Add a tagger wrapper:

```text
TargetDenoisingAugmentedParT
```

Variants:

```text
hlt_part_baseline
feature_mlp_adapter_tag_only
denoiser_features_frozen
denoiser_features_joint
denoiser_tag_only_same_arch
denoiser_shuffled_targets
denoiser_no_pair_bias
denoiser_local_kernel_only
```

The primary integration should be:

```text
original HLT ParT inputs unchanged
denoiser outputs -> embedding residual adapter
```

### Step 5: Reports And Mechanism Ablations

Write a report builder that compares:

```text
baselines
denoising RMSE/NLL
tagger accuracy/loss
per-class accuracy
confusion matrices
adapter/attention diagnostics
mechanism ablations
```

The report must explicitly separate:

```text
model_val selection metrics
final_test one-shot metrics
diagnostic-only teacher/offline analysis
```

### Step 6: Pilot And High-Data Submitters

Add submitters for:

```text
500k pilot
5M high-data
```

Run order:

```text
HLT/offline baselines
denoiser pretraining
tagger variants
report
```

The submitter should support:

```text
smearing-only debug run
realistic HLT v2 run
skip-existing with complete artifact checks
strict metadata checks
```

## First Run Recommendation

Do not start with every ablation at high data.

Recommended first pilot:

```text
A0 HLT ParT baseline
A1 feature MLP tag-only adapter
A2 denoiser features frozen
A3 denoiser features joint fine-tune
A4 tag-only same architecture
A6 no-pair-bias denoiser
```

If A2/A3 beats A1/A4, the denoising supervision is doing real work.

Then high-data:

```text
A0
A1
A3
A4
A5 shuffled residual target
A6 no-pair-bias
```

The result we most want to see:

```text
denoiser_features_joint
  > feature_mlp_adapter_tag_only
  > HLT ParT baseline

and

denoiser_features_joint
  > denoiser_tag_only_same_arch
  > denoiser_shuffled_targets
```

That would be the clean signal that target-conditioned pairwise denoising adds
useful information beyond ordinary ParT capacity.
