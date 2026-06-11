# Teacher-Logit PFN-Style Reconstructor Plan

This document defines the Particle Flow Network style reconstructor family for
the teacher-logit reconstruction line.  It should live beside, not replace, the
completed Global Transformer and ParticleNet-style reconstructors.

## Current Status

- Step 1 complete: `particle_flow` is registered in the shared reconstructor
  builder, has a minimal validated config/module shell, supports config-based
  construction, and checkpoint loading routes through the shared loader.
- Step 2 complete: PFN per-particle input features and mask-safe sum, mean, and
  max pooling helpers are implemented with deterministic synthetic coverage.
- Step 3 complete: `ParticleFlowEncoder` and `ParticleFlowContextBuilder`
  produce per-particle phi embeddings, permutation-invariant pooled context,
  and pooling diagnostics.
- Step 4 complete: `ParticleFlowReconstructor` now returns the shared
  `SoftReconstructedView` contract with bounded parent corrections,
  PFN-global extra candidates, physical output guards, metadata, and aux
  diagnostics.
- Step 5 complete: PFN training API and CLI mirror the established teacher-logit
  reconstructor loop, save compatible checkpoints/reports, and expose
  PFN-specific architecture/loss arguments.
- Step 6 complete: PFN prediction API and CLI load trained PFN reconstructors,
  consume cached fixed-HLT views only, and write fusion-compatible prediction
  blocks with final-test confirmation.
- Step 7 complete: PFN first-experiment harness trains PFN -> frozen teacher,
  collects HLT-only prediction blocks, summarizes comparison metrics, and keeps
  final-test access behind explicit confirmation.
- Step 8 complete: PFN Slurm train, prediction, fusion, and submitter scripts
  mirror the GT/PN teacher-logit runner chain with PFN-specific defaults and
  run-config/audit metadata.

The goal is still not literal offline particle recovery.  The goal is to learn
an HLT-only soft view that makes a frozen offline teacher respond similarly to
how it responds to the true offline view.

## Core Idea

Use the same outer experiment:

```text
fixed HLT view -> PFN-style reconstructor -> reconstructed soft view -> frozen teacher -> reco logits
offline view                                       -> frozen teacher -> offline logits
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
reconstructor models all-to-all attention.  The PN reconstructor models local
graph neighborhoods.  The PFN reconstructor should instead be a pooled-set
model:

```text
per-particle shared phi network
-> permutation-invariant pooled jet context
-> global-conditioned particle corrections and extra candidates
```

That gives us a third genuinely different bias for the ensemble.

## Why PFN Is Worth Adding

PFN is intentionally less expressive than the transformer and the graph model.
That is not a flaw for this experiment.  It is useful because it may make
different errors.

The PFN-style reconstructor should be good at:

- Learning global class-relevant corrections from aggregate jet structure.
- Producing stable corrections without relying on local neighbor topology.
- Serving as a low-variance, permutation-invariant ensemble member.
- Capturing coarse shifts in multiplicity, total energy flow, and particle-type
  composition.

It may be weaker at:

- Local splitting and merging.
- Fine angular substructure.
- Detailed pairwise particle relationships.

That weakness is acceptable if its teacher-logit residuals are not redundant
with the ParT and PN reconstructors.

## Output Contract

The PFN reconstructor must return the same `SoftReconstructedView` contract as
the existing teacher-logit reconstructors:

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
global pooled context + learned slot embedding -> slot MLP -> token + soft weight
```

The extra candidate decoder should remain PFN-like.  Do not add transformer
cross-attention or kNN graph operations here.

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
```

This should share utility code with the other teacher-logit reconstructors where
possible, but PFN-specific normalization can live in the PFN module if that is
cleaner.

### 2. Per-Particle Phi Network

Use a shared MLP over particles:

```text
particle_features [B, N, F]
-> phi MLP
-> particle_embeddings [B, N, D]
```

Recommended first defaults:

```text
phi_dims = 128 128 128
activation = GELU or ReLU
dropout = 0.05
layer_norm = true
```

The same phi network is applied to every constituent.  Masked particles must not
contribute to the pooled context.

### 3. Pooled Jet Context

Construct a permutation-invariant jet embedding from masked particle embeddings.
Use more than one pooling statistic so the model is not limited to pure sum:

```text
sum_pool  = sum(phi_i)
mean_pool = mean(phi_i over valid particles)
max_pool  = max(phi_i over valid particles)
count     = log(1 + valid_count)
summary   = weak HLT jet summary features

jet_context = MLP([sum_pool, mean_pool, max_pool, count, summary])
```

Recommended first defaults:

```text
context_dim = 256
context_mlp_dims = 256 256
```

This is the main PFN inductive bias: all particle communication happens through
the pooled set context.

### 4. Parent Correction Decoder

Each valid HLT constituent receives a correction conditioned on both its local
embedding and the global context:

```text
decoder_input_i = concat(phi_i, jet_context, original_token_i)
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

The correction budget loss should keep this from hallucinating a huge offline
jet from weak HLT evidence.

### 5. Extra Candidate Decoder

Use learned slot embeddings, conditioned only on the global PFN context:

```text
slot_embedding_k + jet_context -> slot MLP -> candidate token_k, candidate_weight_k
```

Recommended first defaults:

```text
num_extra_candidates = 32
slot_dim = context_dim
candidate_weight_bias = -3.0
```

This decoder is intentionally global.  It asks: "given the whole HLT jet, what
soft extra particles are plausible?"  It does not ask: "which local neighbor
should I split?"  That distinction is what separates PFN reco from PN reco.

### 6. Teacher Compatibility

The PFN reconstructor should be compatible with the same frozen teacher family:

```text
PFN reco -> ParT teacher
PFN reco -> PN teacher
PFN reco -> PFN teacher
PFN reco -> PCNN teacher
```

First experiment should stay simple:

```text
PFN reco -> ParT teacher
```

Once that works, extend via the same `teacher_architecture` argument and
checkpoint plumbing used by the Global Transformer and PN reconstructor
runners.

## Loss Details

Keep the loss formula aligned with the previous reconstructors:

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

This is the main target.  It tells the reconstructor which teacher-relevant
class evidence the HLT view failed to express.

### Teacher CE

Add a smaller supervised term:

```text
CE(reco_teacher_logits, true_label)
```

This prevents a degenerate solution where the reconstructed view matches
teacher uncertainty but fails the actual class decision.

### Correction Budget

Penalize:

```text
weighted parent token deltas
extra candidate total weight
very large candidate pt/energy
large changes to jet-level pt/energy
```

This keeps the model honest.  It can correct and softly add particles, but it
cannot invent unlimited offline information.

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

Do not expect PFN reco to be the strongest standalone model.  The target is:

```text
PFN reco adds non-redundant fusion signal.
```

Useful signs:

- PFN reco teacher logits beat raw HLT on `stack_val`.
- PFN reco errors differ from GT and PN reco errors.
- PFN reco improves fusion even if standalone accuracy is lower.
- PFN reco helps classes where GT/PN are flat.
- Row-shuffle and label-permutation controls collapse normally.

Bad signs:

- It exactly tracks raw HLT logits.
- It only improves calibration but not ranking/accuracy.
- It duplicates GT or PN prediction errors.
- It learns very high extra-candidate weights everywhere.

## Files To Add

Recommended module names:

```text
teacher_logit_reco/particle_flow_reconstructor.py
teacher_logit_reco/train_particle_flow.py
teacher_logit_reco/predict_particle_flow.py
teacher_logit_reco/pfn_first_experiment.py

scripts/train_teacher_logit_particle_flow_reco.py
scripts/predict_teacher_logit_particle_flow_reco.py
scripts/run_teacher_logit_particle_flow_first_experiment.py
```

Recommended tests:

```text
tests/test_teacher_logit_reco_particle_flow_reconstructor.py
tests/test_teacher_logit_reco_train_particle_flow.py
tests/test_teacher_logit_reco_predict_particle_flow.py
tests/test_teacher_logit_reco_pfn_first_experiment.py
```

Recommended Slurm files:

```text
sbatch/run_train_teacher_logit_pfn_reco.sh
sbatch/run_predict_teacher_logit_pfn_reco.sh
sbatch/run_fuse_teacher_logit_pfn_reco.sh
sbatch/submit_teacher_logit_pfn_reco_experiment.sh
```

## Implementation Steps

### Step 1: Shared PFN Utilities And Builder Registration

Add the PFN architecture name to the shared reconstructor builder layer.

Implement:

```text
teacher_logit_reco/reconstructor_builders.py
```

Expected public behavior:

```text
build_reconstructor_from_config({"reconstructor_architecture": "particle_flow", ...})
load_teacher_logit_reconstructor_checkpoint(...)
```

Tests:

```text
tests/test_teacher_logit_reco_reconstructor_builders.py
```

Done when:

- Existing GT and PN checkpoint loading still works.
- A minimal PFN config can be instantiated from the shared builder.
- Unknown architecture names still fail loudly.

### Step 2: PFN Input Features And Masked Pooling

Implement feature construction and pooling helpers.

File:

```text
teacher_logit_reco/particle_flow_reconstructor.py
```

Core helpers:

```text
build_particle_flow_features(tokens, mask)
masked_sum_pool(x, mask)
masked_mean_pool(x, mask)
masked_max_pool(x, mask)
```

Tests:

```text
tests/test_teacher_logit_reco_particle_flow_reconstructor.py
```

Done when:

- Masked particles do not affect pooled outputs.
- Feature tensors are finite for zero/invalid padded particles.
- Class-blocked or variable-length synthetic batches behave deterministically.

### Step 3: PFN Encoder And Global Context

Implement:

```text
ParticleFlowEncoder
ParticleFlowContextBuilder
```

The encoder should expose:

```text
particle_embeddings
jet_context
pooling_report or aux diagnostics
```

Tests should verify:

- Output shapes.
- Mask invariance.
- Permutation equivariance for particle embeddings.
- Permutation invariance for pooled jet context.

### Step 4: PFN Soft-View Reconstructor

Implement:

```text
ParticleFlowReconstructorConfig
ParticleFlowReconstructor
build_particle_flow_reconstructor
```

The forward pass should return `SoftReconstructedView`.

Tests should verify:

- Corrected parent plus extra candidate shapes.
- Padded input particles stay masked.
- Extra candidates have valid masks/weights.
- Physical outputs are finite.
- Metadata says `reconstructor_architecture = particle_flow`.

### Step 5: Training API And CLI

Mirror the PN training path.

Files:

```text
teacher_logit_reco/train_particle_flow.py
scripts/train_teacher_logit_particle_flow_reco.py
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

PFN-specific CLI args:

```text
--phi-dims 128 128 128
--context-dim 256
--context-dims 256 256
--decoder-dims 256 128
--num-extra-candidates 32
--dropout 0.05
```

Done when:

- A tiny CPU smoke run can train one epoch.
- The saved checkpoint loads through the shared builder.
- `run_report.json` records architecture, teacher, losses, and subset sizes.

### Step 6: Prediction API And CLI

Mirror the PN prediction path.

Files:

```text
teacher_logit_reco/predict_particle_flow.py
scripts/predict_teacher_logit_particle_flow_reco.py
```

It should:

- Load a PFN reconstructor checkpoint.
- Load a frozen teacher checkpoint.
- Consume cached fixed-HLT views only.
- Write `PredictionBlock` files compatible with independent fusion.
- Require `--confirm-final-test` for `final_test`.

Done when:

- Prediction blocks round-trip through `load_prediction_block`.
- Jet identity and label alignment are preserved.
- The model name is stable, e.g. `pfn_reco_to_part_teacher`.

### Step 7: First PFN Experiment Harness

Add a small end-to-end harness for the first PFN reconstructor experiment.

File:

```text
scripts/run_teacher_logit_particle_flow_first_experiment.py
```

Default experiment:

```text
PFN reco -> ParT teacher
model_train/model_val only for training
stack_val/final_test only for frozen prediction/evaluation
```

The harness should optionally compare against:

```text
raw HLT baseline
GT reco prediction blocks
PN reco prediction blocks
```

Done when:

- It produces comparison-ready metrics without fitting a final stacker unless
  explicitly requested.
- It refuses to touch `final_test` without confirmation.

### Step 8: Slurm Runners

Add Slurm scripts only after local tests and smoke paths pass.

Recommended scripts:

```text
sbatch/run_train_teacher_logit_pfn_reco.sh
sbatch/run_predict_teacher_logit_pfn_reco.sh
sbatch/run_fuse_teacher_logit_pfn_reco.sh
sbatch/submit_teacher_logit_pfn_reco_experiment.sh
```

The submitter should queue:

```text
train PFN reco for each selected teacher architecture
-> predict stack_train/stack_val/final_test for each trained reco
-> independent fusion over saved prediction blocks
```

Recommended first defaults:

```text
TEACHER_LOGIT_PFN_TEACHERS="part"
train walltime: 12 hours, GPU
predict walltime: 5 hours, GPU
fusion walltime: 5 hours, CPU
```

Later, when PN/PFN/PCNN offline teachers are ready:

```text
TEACHER_LOGIT_PFN_TEACHERS="part pn pfn pcnn"
```

## What Not To Do Yet

- Do not replace the existing GT or PN reconstructor paths.
- Do not add graph operations; that belongs to PN.
- Do not add attention; that belongs to GT.
- Do not make offline particle matching the main objective.
- Do not tune on `final_test`.
- Do not use offline-teacher logits as deployable HLT fusion features.

## Success Criteria

The PFN reconstructor is worth keeping if at least one is true:

- PFN reconstructed-view teacher logits beat raw HLT on `stack_val`.
- PFN reconstructed-view logits improve fusion with GT and PN reconstructors.
- PFN errors are less correlated with GT/PN errors than GT and PN are with each
  other.
- PFN helps at least one physics class where GT/PN do not.
- Negative controls collapse normally.
