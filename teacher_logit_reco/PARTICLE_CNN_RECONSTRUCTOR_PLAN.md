# Teacher-Logit P-CNN-Style Reconstructor Plan

This document defines the Particle-CNN style reconstructor family for the
teacher-logit reconstruction line.  It should live beside, not replace, the
Global Transformer, ParticleNet, and PFN-style reconstructors.

The goal is still not literal offline particle recovery.  The goal is to learn
an HLT-only soft view that makes a frozen offline teacher respond similarly to
how it responds to the true offline view.

## Current Status

- Step 1 complete: `particle_cnn` is registered in the shared reconstructor
  builder, has a minimal validated config/module shell, supports config-based
  construction, and checkpoint loading routes through the shared loader.
- Step 2 complete: P-CNN per-particle input features, rank features,
  Conv1d-channel masking, and mask-safe sum, mean, and max pooling helpers are
  implemented with deterministic synthetic coverage.
- Step 3 complete: `ParticleCnnBlock`, `ParticleCnnEncoder`, and
  `ParticleCnnContextBuilder` produce masked rank-convolution particle
  embeddings, whole-jet context, rank diagnostics, and pooling reports.
- Step 4 complete: `ParticleCnnReconstructor` now emits `SoftReconstructedView`
  objects with bounded parent corrections, global extra candidates, physical
  token sanitation, and explicit cache-order diagnostics.
- Step 5 complete: `train_particle_cnn.py` and
  `scripts/train_teacher_logit_particle_cnn_reco.py` provide the P-CNN training
  API/CLI, checkpoint payloads, run reports, leakage metadata, and PCNN-specific
  architecture flags.
- Step 6 complete: `predict_particle_cnn.py` and
  `scripts/predict_teacher_logit_particle_cnn_reco.py` load trained P-CNN
  checkpoints, consume cached fixed-HLT views only, require explicit
  `final_test` confirmation, and write fusion-compatible prediction blocks.
- Step 7 complete: `pcnn_first_experiment.py` and
  `scripts/run_teacher_logit_particle_cnn_first_experiment.py` provide the
  first P-CNN -> frozen-teacher experiment harness, comparison-summary report,
  and no-final-stacker guardrail.
- Step 8 complete: Slurm train/predict/fusion runners and the
  `submit_teacher_logit_pcnn_reco_experiment.sh` submitter queue selected
  P-CNN teacher-logit reconstructor jobs with afterok dependencies and
  independent-fusion output.

## Core Idea

Use the same outer experiment:

```text
fixed HLT view -> P-CNN-style reconstructor -> reconstructed soft view -> frozen teacher -> reco logits
offline view                                        -> frozen teacher -> offline logits
```

Use the same loss family:

```text
loss =
  teacher_KL(offline_teacher_logits, reco_teacher_logits)
+ CE(reco_teacher_logits, true_label)
+ correction_budget
+ weak_jet_summary_loss
```

The architecture changes inside the reconstructor.  The Global Transformer
reconstructor models global all-to-all attention.  The PN reconstructor models
local graph neighborhoods.  The PFN reconstructor models pooled unordered set
structure.  The P-CNN reconstructor should instead model local patterns along a
canonical constituent-rank axis:

```text
rank-ordered HLT particles
-> per-particle feature projection
-> masked 1D residual convolution blocks over rank
-> global pooled conv context
-> rank-local parent corrections and global extra candidates
```

That gives us a fourth genuinely different bias for the ensemble.

## Why P-CNN Is Worth Adding

P-CNN is intentionally order-sensitive.  That makes it different from PFN and
ParticleNet in a useful way.  In particle tagging pipelines, constituent arrays
are usually stored in a stable rank-like order, often by descending `pt`.  A
1D convolution over that ordered list can learn patterns such as:

- Leading-particle correction behavior.
- Subleading particle texture.
- Rank-local energy-flow changes.
- Differences between dense high-rank tails and sparse low-multiplicity jets.
- Smooth correction motifs across nearby rank positions.

It may be weaker at:

- Arbitrary permutation handling.
- Explicit geometric neighborhoods.
- Long-range pairwise structure.

That weakness is acceptable if the P-CNN teacher-logit residuals are not
redundant with GT, PN, and PFN residuals.

## Canonical Ordering Assumption

This architecture only makes sense if the particle dimension has stable
meaning.  The first implementation should use the fixed-HLT cache order as the
canonical order and document that assumption in every report.

Recommended first policy:

```text
do not sort inside forward by default
trust the cache's canonical particle order
add normalized rank features
add diagnostics that report whether valid-token pt is mostly descending
```

Reason: sorting inside the model can silently reorder token-level auxiliary
channels and make debugging harder.  If the cache order later proves unstable,
add an explicit preprocessing option, not a hidden model behavior.

The P-CNN model should not be permutation invariant.  Tests should instead
verify that:

- Padded tokens do not affect outputs.
- Rank features are finite and stable.
- Convolutions cannot leak from padded regions.
- A deliberate rank permutation can change outputs, because order sensitivity
  is the intended bias.

## Output Contract

The P-CNN reconstructor must return the same `SoftReconstructedView` contract
as the existing teacher-logit reconstructors:

```text
SoftReconstructedView(
  tokens = corrected parent tokens + extra candidate tokens,
  mask = parent mask + extra mask,
  weights = parent weights + extra weights,
  labels,
  jet_ids,
  split,
  metadata,
  aux
)
```

Parent-aligned outputs:

```text
pt'     = pt * exp(delta_logpt)
eta'    = eta + delta_eta
phi'    = phi + delta_phi
energy' = energy * exp(delta_loge)
weight  = sigmoid(parent_weight_logit + bias)
```

Extra candidate outputs:

```text
global conv context + learned slot embedding -> slot MLP -> token + soft weight
```

The extra candidate decoder should remain P-CNN-compatible.  Do not add
transformer cross-attention or graph kNN operations here.

## Proposed Architecture

### 1. Input Feature Builder

Start from sanitized fixed-HLT tokens:

```text
pt, eta, phi, energy, charge, PID flags, track variables
```

Build a stable per-particle feature vector:

```text
log(pt)
log(energy)
eta / eta_scale
sin(phi)
cos(phi)
log(pt / jet_pt)
log(energy / jet_energy)
charge
PID channels
tanh(track impact parameters)
clipped track uncertainties
valid-particle mask channel
normalized rank in [0, 1]
log(1 + rank)
leading-rank indicator channels, optional
```

The rank features are important because a convolution kernel alone only knows
relative position, not whether it is acting near the leading particle or the
soft tail.

### 2. Masked Feature Projection

Project particle features into a channel representation:

```text
particle_features [B, N, F]
-> linear/MLP projection
-> conv_input [B, C, N]
```

Use LayerNorm or channel-wise normalization before the Conv1d stack.  Avoid
BatchNorm over padded particle positions in the first version.

After every projection or convolution block:

```text
conv_features = conv_features * mask[:, None, :]
```

This is non-negotiable.  Padding leakage would make the model look better than
it is and would confuse any comparison with the other reconstructors.

### 3. Residual Conv1d Encoder

Use a small stack of masked residual Conv1d blocks over particle rank:

```text
Conv1d kernel 3 or 5
GELU
dropout
pointwise Conv1d
residual connection
mask after block
```

Recommended first defaults:

```text
hidden_channels = 128
num_blocks = 6
kernel_sizes = 5 5 3 3 3 3
dilations = 1 2 4 1 2 4
dropout = 0.05
```

Dilations let the model see across rank ranges without becoming a transformer.
The stack should stay modest so it remains a distinct, interpretable baseline.

### 4. Global Conv Context

Build a global context from the masked conv features:

```text
sum_pool  = sum(conv_i)
mean_pool = mean(conv_i over valid particles)
max_pool  = max(conv_i over valid particles)
count     = log(1 + valid_count)
summary   = weak HLT jet summary features

global_context = MLP([sum_pool, mean_pool, max_pool, count, summary])
```

Recommended first defaults:

```text
context_dim = 256
context_mlp_dims = 256 256
```

This context supports extra candidates and gives parent corrections access to
whole-jet information without adding attention.

### 5. Parent Correction Decoder

Each valid HLT constituent receives a correction conditioned on local rank-conv
features, the original token, and the global context:

```text
decoder_input_i = concat(conv_feature_i, global_context, original_token_i, rank_features_i)
decoder MLP -> correction fields
```

Outputs:

```text
delta_logpt
delta_eta
delta_phi
delta_loge
parent_weight_logit
optional PID/track residual channels, if already supported by SoftReconstructedView
```

Start conservative:

```text
max_abs_delta_eta = 0.35
max_abs_delta_phi = 0.35
max_abs_delta_logpt = 1.0
max_abs_delta_loge = 1.0
parent_weight_bias = 2.0
```

### 6. Extra Candidate Decoder

Use learned slot embeddings, conditioned on the global conv context:

```text
slot_embedding_k + global_context -> slot MLP -> candidate token_k, candidate_weight_k
```

Recommended first defaults:

```text
num_extra_candidates = 32
slot_dim = context_dim
candidate_weight_bias = -3.0
```

The extra candidate decoder is global by design.  If we later want local
rank-conditioned extras, add that as a variant after the simple version works.

### 7. Teacher Compatibility

The P-CNN reconstructor should be compatible with the same frozen teacher
family:

```text
P-CNN reco -> ParT teacher
P-CNN reco -> PN teacher
P-CNN reco -> PFN teacher
P-CNN reco -> PCNN teacher
```

First experiment should stay simple:

```text
P-CNN reco -> ParT teacher
```

Once that works, extend via the same `teacher_architecture` argument and
checkpoint plumbing used by the GT, PN, and PFN reconstructor runners.

## Loss Details

Keep the loss formula aligned with the other reconstructors:

```text
total =
  lambda_teacher_kl * teacher_KL
+ lambda_teacher_ce * CE(reco_teacher_logits, label)
+ lambda_budget * correction_budget
+ lambda_summary * weak_jet_summary_loss
```

Recommended first weights:

```text
lambda_teacher_kl = 1.0
lambda_teacher_ce = 0.25
lambda_budget = 0.05
lambda_summary = 0.05
teacher_temperature = 2.0
```

### Teacher KL

Compare the frozen teacher's probability distribution on the true offline view
to its probability distribution on the reconstructed soft view:

```text
KL(
  softmax(offline_teacher_logits / T),
  softmax(reco_teacher_logits / T)
)
```

This remains the main objective.  It tells the reconstructor which
teacher-relevant class evidence the HLT view failed to express.

### Teacher CE

Add a smaller supervised term:

```text
CE(reco_teacher_logits, true_label)
```

This keeps the reconstructed view aligned with actual labels, not only with the
offline teacher's softened uncertainty.

### Correction Budget

Penalize:

```text
weighted parent token deltas
extra candidate total weight
very large candidate pt/energy
large changes to jet-level pt/energy
```

This prevents the P-CNN from exploiting the teacher with unrealistic soft
particles.

### Weak Jet Summary Loss

Compare only coarse reconstructed summary statistics to offline summary
statistics:

```text
total pt
total energy
constituent count proxy
charged/neutral/PID composition proxies
eta/phi weighted moments
```

This loss should remain weak.  It is a stabilizer, not the main objective.

## Expected Behavior

Do not expect P-CNN reco to dominate standalone.  The target is:

```text
P-CNN reco adds non-redundant fusion signal.
```

Useful signs:

- P-CNN reconstructed-view teacher logits beat raw HLT on `stack_val`.
- P-CNN errors differ from GT, PN, and PFN reco errors.
- P-CNN improves fusion even if standalone accuracy is lower.
- P-CNN helps classes where global attention, graph neighborhoods, and PFN
  pooling are flat.
- Row-shuffle and label-permutation controls collapse normally.

Bad signs:

- It exactly tracks raw HLT logits.
- It only learns a rank-position prior and ignores particle features.
- It becomes extremely sensitive to padded tail length.
- It duplicates PFN or GT prediction errors.
- It learns very high extra-candidate weights everywhere.

## Files To Add

Recommended module names:

```text
teacher_logit_reco/particle_cnn_reconstructor.py
teacher_logit_reco/train_particle_cnn.py
teacher_logit_reco/predict_particle_cnn.py
teacher_logit_reco/pcnn_first_experiment.py

scripts/train_teacher_logit_particle_cnn_reco.py
scripts/predict_teacher_logit_particle_cnn_reco.py
scripts/run_teacher_logit_particle_cnn_first_experiment.py
```

Recommended tests:

```text
tests/test_teacher_logit_reco_particle_cnn_reconstructor.py
tests/test_teacher_logit_reco_train_particle_cnn.py
tests/test_teacher_logit_reco_predict_particle_cnn.py
tests/test_teacher_logit_reco_pcnn_first_experiment.py
```

Recommended Slurm files:

```text
sbatch/run_train_teacher_logit_pcnn_reco.sh
sbatch/run_predict_teacher_logit_pcnn_reco.sh
sbatch/run_fuse_teacher_logit_pcnn_reco.sh
sbatch/submit_teacher_logit_pcnn_reco_experiment.sh
```

## Implementation Steps

### Step 1: Shared P-CNN Utilities And Builder Registration

Add the P-CNN architecture name to the shared reconstructor builder layer.

Implement:

```text
teacher_logit_reco/reconstructor_builders.py
teacher_logit_reco/particle_cnn_reconstructor.py
```

Expected public behavior:

```text
build_reconstructor_from_config({"reconstructor_architecture": "particle_cnn", ...})
load_teacher_logit_reconstructor_checkpoint(...)
```

Tests:

```text
tests/test_teacher_logit_reco_reconstructor_builders.py
tests/test_teacher_logit_reco_particle_cnn_reconstructor.py
```

Done when:

- Existing GT, PN, and PFN checkpoint loading still works.
- A minimal P-CNN config can be instantiated from the shared builder.
- Unknown architecture names still fail loudly.

### Step 2: P-CNN Input Features, Rank Features, And Masked Conv Helpers

Implement feature construction and convolution-safe masking helpers.

File:

```text
teacher_logit_reco/particle_cnn_reconstructor.py
```

Core helpers:

```text
build_particle_cnn_features(tokens, mask)
build_rank_features(mask)
apply_particle_mask_channels(x, mask)
masked_sum_pool(x, mask)
masked_mean_pool(x, mask)
masked_max_pool(x, mask)
```

Tests should verify:

- Rank features are finite for padded and empty synthetic batches.
- Masked particles do not affect pooled outputs.
- Conv channel tensors zero out padded positions after masking.
- Feature tensors are finite for zero/invalid padded particles.

### Step 3: Residual P-CNN Encoder

Implement:

```text
ParticleCnnBlock
ParticleCnnEncoder
ParticleCnnContextBuilder
```

The encoder should expose:

```text
rank_embeddings or rank_features
particle_embeddings
jet_context
pooling_report or aux diagnostics
```

Tests should verify:

- Output shapes.
- Mask safety after each residual block.
- Dilation/kernel configuration validation.
- Rank-order sensitivity on deliberately permuted synthetic inputs.

### Step 4: P-CNN Soft-View Reconstructor

Implement:

```text
ParticleCnnReconstructorConfig
ParticleCnnReconstructor
build_particle_cnn_reconstructor
```

The forward pass should return `SoftReconstructedView`.

Tests should verify:

- Corrected parent plus extra candidate shapes.
- Padded input particles stay masked.
- Extra candidates have valid masks/weights.
- Physical outputs are finite.
- Metadata says `reconstructor_architecture = particle_cnn`.
- Aux diagnostics include the cache-order assumption and descending-pt audit
  summary.

### Step 5: Training API And CLI

Mirror the PFN and PN training paths.

Files:

```text
teacher_logit_reco/train_particle_cnn.py
scripts/train_teacher_logit_particle_cnn_reco.py
```

Reuse:

```text
paired HLT/offline loader
teacher loading
teacher-logit loss
early stopping
checkpoint format
run_report.json
training_curves.json
model_val_report.json
```

P-CNN-specific CLI args:

```text
--hidden-channels 128
--num-blocks 6
--kernel-sizes 5 5 3 3 3 3
--dilations 1 2 4 1 2 4
--context-dim 256
--context-dims 256 256
--decoder-dims 256 128
--num-extra-candidates 32
--dropout 0.05
```

Done when:

- A tiny CPU smoke run can train one epoch.
- The saved checkpoint loads through the shared builder.
- `run_report.json` records architecture, teacher, losses, subset sizes, and
  ordering assumptions.

### Step 6: Prediction API And CLI

Mirror the PFN and PN prediction paths.

Files:

```text
teacher_logit_reco/predict_particle_cnn.py
scripts/predict_teacher_logit_particle_cnn_reco.py
```

It should:

- Load a P-CNN reconstructor checkpoint.
- Load a frozen teacher checkpoint.
- Consume cached fixed-HLT views only.
- Write `PredictionBlock` files compatible with independent fusion.
- Require `--confirm-final-test` for `final_test`.

Done when:

- Prediction blocks round-trip through `load_prediction_block`.
- Jet identity and label alignment are preserved.
- The model name is stable, e.g. `pcnn_reco_to_part_teacher`.

### Step 7: First P-CNN Experiment Harness

Add a small end-to-end harness for the first P-CNN reconstructor experiment.

File:

```text
scripts/run_teacher_logit_particle_cnn_first_experiment.py
```

Default experiment:

```text
P-CNN reco -> ParT teacher
model_train/model_val only for training
stack_val/final_test only for frozen prediction/evaluation
```

The harness should optionally compare against:

```text
raw HLT baseline
GT reco prediction blocks
PN reco prediction blocks
PFN reco prediction blocks
```

Done when:

- It produces comparison-ready metrics without fitting a final stacker unless
  explicitly requested.
- It refuses to touch `final_test` without confirmation.

### Step 8: Slurm Runners

Add Slurm scripts only after local tests and smoke paths pass.

Recommended scripts:

```text
sbatch/run_train_teacher_logit_pcnn_reco.sh
sbatch/run_predict_teacher_logit_pcnn_reco.sh
sbatch/run_fuse_teacher_logit_pcnn_reco.sh
sbatch/submit_teacher_logit_pcnn_reco_experiment.sh
```

The submitter should queue:

```text
train P-CNN reco for each selected teacher architecture
-> predict stack_train/stack_val/final_test for each trained reco
-> independent fusion over saved prediction blocks
```

Recommended first defaults:

```text
TEACHER_LOGIT_PCNN_TEACHERS="part"
train walltime: 12 hours, GPU
predict walltime: 5 hours, GPU
fusion walltime: 5 hours, CPU
```

Later, when PN/PFN/PCNN offline teachers are ready:

```text
TEACHER_LOGIT_PCNN_TEACHERS="part pn pfn pcnn"
```

## What Not To Do Yet

- Do not replace the existing GT, PN, or PFN reconstructor paths.
- Do not add graph operations; that belongs to PN.
- Do not add attention; that belongs to GT.
- Do not make the model permutation invariant; that belongs to PFN.
- Do not silently sort particles inside forward unless a future audit proves
  the cache order is unusable.
- Do not make offline particle matching the main objective.
- Do not tune on `final_test`.
- Do not use offline-teacher logits as deployable HLT fusion features.

## Success Criteria

The P-CNN reconstructor is worth keeping if at least one is true:

- P-CNN reconstructed-view teacher logits beat raw HLT on `stack_val`.
- P-CNN reconstructed-view logits improve fusion with GT, PN, and PFN
  reconstructors.
- P-CNN errors are less correlated with at least one existing reconstructor
  family than those families are with each other.
- P-CNN helps at least one physics class where GT/PN/PFN do not.
- Negative controls collapse normally.
