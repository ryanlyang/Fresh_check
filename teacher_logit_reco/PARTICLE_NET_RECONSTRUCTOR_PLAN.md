# Teacher-Logit ParticleNet-Style Reconstructor Plan

This document defines the next reconstructor family for the teacher-logit
reconstruction line.  It is intentionally separate from the completed Global
Transformer reconstructor plan in `IMPLEMENTATION_PLAN.md`.

The goal is not to rebuild the offline particle set literally.  The goal is to
learn an HLT-only transformation that makes a frozen offline teacher interpret
the reconstructed soft view more like it interprets the true offline view.

## Core Idea

Use the same outer experiment:

```text
fixed HLT view -> PN-style reconstructor -> reconstructed soft view -> frozen teacher -> reco logits
offline view                                      -> frozen teacher -> offline logits
```

Use the same loss family:

```text
loss =
  teacher_KL(offline_teacher_logits, reco_teacher_logits)
+ CE(reco_teacher_logits, true_label)
+ correction_budget
+ weak_jet_summary_loss
```

The architectural change is inside the reconstructor.  The Global Transformer
reconstructor models global all-to-all relationships.  The ParticleNet-style
reconstructor should instead model local constituent neighborhoods with dynamic
graph / EdgeConv blocks.

## Why This Architecture Is Different

The Global Transformer can learn broad jet-level correlations, but it may be
redundant with transformer taggers.  A PN-style reconstructor has a different
inductive bias:

- It emphasizes local particle neighborhoods.
- It can learn local corrections from nearby constituents.
- It can specialize in substructure details that look like local splitting,
  merging, smearing, or missing soft radiation.
- It should produce errors that are less correlated with the transformer
  reconstructor.

This makes it a useful ensemble component even if its standalone reconstructed
teacher logits are not always better.

## Output Contract

The PN reconstructor must return the same `SoftReconstructedView` contract as
the Global Transformer reconstructor:

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
global graph context + learned slot embedding -> MLP -> token + soft weight
```

The extra candidates should not use transformer cross-attention.  The point of
this family is to stay graph/ParticleNet-like.

## Proposed Architecture

### 1. Input Features

Start from sanitized HLT tokens:

```text
pt, eta, phi, energy, charge, PID flags, track variables
```

Build stable graph input features:

```text
log(pt)
log(energy)
eta / eta_scale
sin(phi)
cos(phi)
charge
PID one-hot-ish channels
tanh(track impact parameters)
clipped track uncertainties
```

For kNN coordinates, use a small physical coordinate set:

```text
eta
phi
log(pt)
```

The first version should use kNN in this coordinate space.  Later variants can
try learned-feature kNN after the first EdgeConv block.

### 2. Dynamic EdgeConv Encoder

Use EdgeConv blocks:

```text
for each particle i:
  find k nearest valid particles j
  edge_ij = MLP([x_i, x_j - x_i])
  x'_i = max_j edge_ij
```

Recommended first defaults:

```text
k: 12 or 16
hidden_dims: [64, 128, 128]
num_edgeconv_blocks: 3
dropout: 0.05
aggregation: max
```

Masking rules:

- Invalid/padded particles must not be selected as neighbors.
- Invalid/padded particles must not contribute to pooled features.
- Empty jets should receive the same safe fallback behavior used by the Global
  Transformer path.

### 3. Parent Correction Head

Each encoded parent constituent gets an MLP head:

```text
encoded_parent_i -> MLP -> [
  raw_delta_logpt,
  raw_delta_eta,
  raw_delta_phi,
  raw_delta_loge,
  parent_weight_logit
]
```

Use the same bounded sanitizer as the GT reconstructor:

```text
delta_logpt = max_delta_logpt * tanh(raw_delta_logpt)
delta_eta   = max_delta_eta   * tanh(raw_delta_eta)
delta_phi   = max_delta_phi   * tanh(raw_delta_phi)
delta_loge  = max_delta_loge  * tanh(raw_delta_loge)
```

Then enforce:

```text
pt >= min_pt
eta in [-eta_limit, eta_limit]
phi wrapped to [-pi, pi)
energy >= physical_energy_floor(pt, eta)
```

### 4. Global Graph Pool

Pool encoded parent features into a graph-level context:

```text
global_context = concat(
  masked_mean(encoded_particles),
  masked_max(encoded_particles),
  simple jet summaries
)
```

The first implementation can use mean + max.  Adding simple jet summaries is
optional, but useful if done cleanly.

### 5. Extra Candidate Decoder

Use PN-native slot decoding:

```text
slot_embedding_k + global_context -> MLP -> extra candidate k
```

Outputs:

```text
extra_pt_fraction
extra_delta_eta_from_jet_axis
extra_delta_phi_from_jet_axis
extra_energy_scale
extra_charge
extra_pid_logits
extra_track_channels
extra_weight_logit
```

Constraints:

- Total extra pT budget should be capped relative to HLT jet pT.
- Extra `eta/phi` should be near the HLT jet axis.
- Extra weights should start small by default.
- Extra candidates are soft; they are not hard-created particles unless their
  learned weights become useful.

## Losses

Reuse the existing teacher-logit losses from `losses.py`:

```text
teacher_kl_loss
teacher_cross_entropy_loss
correction_budget_loss
weak_jet_summary_loss
compute_teacher_logit_reco_loss
```

The PN reconstructor should expose `aux` fields compatible with the existing
budget loss:

```text
aux["sanitized_hlt_tokens"]
aux["sanitized_hlt_mask"]
aux["parent_tokens"]
aux["parent_delta"]
aux["parent_weights"]
aux["extra_tokens"]
aux["extra_weights"]
aux["extra_mask"]
aux["jet_axes"]
aux["diagnostics"]
```

If these fields match the GT reconstructor, the Step 4 loss code can be reused
without modification.

## Implementation Steps

Current status:

- Step 1 complete: shared reconstructor builder/checkpoint loading exists,
  legacy Global Transformer checkpoints still load, and new Global Transformer
  checkpoints record `reconstructor_architecture: global_transformer`.
- Step 2 complete: PN input features, physical kNN coordinates, masked kNN
  indices, and neighbor feature gathering are implemented and exported.
- Step 3 complete: `EdgeConvBlock` and `ParticleNetEncoder` are implemented
  with fixed-coordinate masked kNN recomputed at each block.
- Step 4 complete: `ParticleNetReconstructorConfig`,
  `ParticleNetReconstructor`, and `build_particle_net_reconstructor` produce
  the shared soft-view/aux contract using PN graph encoding.
- Step 5 complete: PN-specific training config/module/CLI reuse the existing
  paired-view loader, teacher-logit loss, epoch runner, checkpoint artifacts,
  and leakage rules.
- Step 6 complete: PN-specific prediction config/module/CLI load trained PN
  checkpoints, consume cached fixed-HLT views only, and write fusion-compatible
  prediction blocks with PN architecture metadata.
- Step 7 complete: first PN -> ParT experiment harness trains a modest PN
  run, collects saved prediction blocks, and writes comparison-ready metrics
  without using final_test unless explicitly confirmed.
- Step 8 complete: Slurm train/predict/fuse runners and the PN submitter
  queue the teacher-logit PN experiment end to end.

### Step 1: Shared Reconstructor Builder Interface

Add a small architecture builder layer so training and prediction can construct
either reconstructor:

```text
teacher_logit_reco/reconstructor_builders.py
```

Proposed interface:

```python
build_teacher_logit_reconstructor(architecture, config)
infer_reconstructor_architecture_from_payload(payload)
load_teacher_logit_reconstructor_checkpoint(path, device, strict=True)
```

Supported initial architectures:

```text
global_transformer
particle_net
```

Deliverables:

- GT path still loads old Step 5 checkpoints.
- New checkpoints record `reconstructor_architecture`.
- Tests cover architecture inference and checkpoint loading.

### Step 2: Implement PN Graph Utilities

Create:

```text
teacher_logit_reco/particle_net_reconstructor.py
```

First implement standalone utilities:

```python
particle_net_input_features(tokens, mask)
particle_net_knn_coordinates(tokens, mask)
masked_knn_indices(coords, mask, k)
gather_neighbor_features(features, indices)
```

Deliverables:

- Unit tests for kNN masking.
- Invalid particles are never selected as neighbors.
- Shapes are stable for small jets, empty jets, and `k > n_valid`.

### Step 3: Implement EdgeConv Blocks

Implement:

```python
EdgeConvBlock
ParticleNetEncoder
```

Behavior:

```text
features + coords + mask -> encoded parent features
```

Start with dynamic kNN recomputed each block from either:

```text
Option A: fixed physical coords only
Option B: physical coords for first block, learned features afterward
```

Recommended first version: fixed physical coords for all blocks.  It is simpler
and easier to audit.

Deliverables:

- Forward pass finite.
- Encoded features are zeroed for invalid particles.
- Gradients flow through feature MLPs.

### Step 4: Implement PN Soft-View Reconstructor

Implement:

```python
ParticleNetReconstructorConfig
ParticleNetReconstructor
build_particle_net_reconstructor
```

The forward pass must return `SoftReconstructedView`.

Components:

- HLT sanitizer.
- PN encoder.
- Parent correction head.
- Global graph pooling.
- Extra candidate slot MLP.
- Same bounded output semantics as GT.

Deliverables:

- Unit tests for output shape.
- Unit tests for finite physical outputs.
- Unit tests that parent corrections are bounded.
- Unit tests that `aux` fields are compatible with `correction_budget_loss`.

### Step 5: Reuse The Loss And Training Step

Generalize the Step 5 training module so it can train:

```text
global_transformer
particle_net
```

Preferred CLI extension:

```text
scripts/train_teacher_logit_global_transformer_reco.py
  -> either rename later or keep as GT-only

scripts/train_teacher_logit_reconstructor.py
  --reconstructor-architecture global_transformer|particle_net
```

Safer first path:

```text
scripts/train_teacher_logit_particle_net_reco.py
```

Recommended approach:

1. Keep the existing GT script stable.
2. Add a PN-specific script first.
3. Consolidate only after both paths work.

Deliverables:

- PN training produces:

```text
config.json
training_curves.json
best_model_val.pt
last.pt
model_val_report.json
run_report.json
```

- Checkpoint records:

```text
reconstructor_architecture: particle_net
model_config
loss_config
teacher_metadata
source hashes
```

### Step 6: Reuse Prediction Blocks

Add PN prediction support using the same output format:

```text
predictions/pn_reco_to_part_teacher/stack_train_predictions.npz
predictions/pn_reco_to_part_teacher/stack_val_predictions.npz
predictions/pn_reco_to_part_teacher/final_test_predictions.npz
```

Model naming:

```text
pn_reco_to_part_teacher
pn_reco_to_pn_teacher
pn_reco_to_pfn_teacher
pn_reco_to_pcnn_teacher
```

Prediction path:

```text
cached fixed-HLT only -> PN reconstructor -> soft view -> frozen teacher -> logits
```

No offline constituents may be loaded during prediction.

Deliverables:

- Fusion-compatible `.npz` and metadata files.
- Metadata explicitly records:

```text
reconstructor_architecture: particle_net
allowed_inputs: cached_fixed_hlt_only_then_reconstructed_soft_view_to_frozen_teacher
```

### Step 7: First PN Experiment

Start with the known available teacher:

```text
PN reconstructor -> ParT offline teacher
```

Use a modest first run:

```text
model_train: optionally capped for smoke
model_val: optionally capped for smoke
stack_train/stack_val/final_test: use Step 6 prediction caps if needed
```

Compare against:

- Raw HLT ParT baseline.
- GT reconstructor -> ParT teacher.
- Heterogeneous HLT-only ensemble.

Key questions:

```text
Does PN reco improve teacher logits alone?
Does PN reco add fusion diversity?
Are PN reco errors less correlated with GT reco errors?
Does PN reco help particular classes more than GT reco?
```

### Step 8: Slurm Runners

Status: complete.

Add Slurm scripts only after the local smoke path works.

Recommended scripts:

```text
sbatch/run_train_teacher_logit_pn_reco.sh
sbatch/run_predict_teacher_logit_pn_reco.sh
sbatch/run_fuse_teacher_logit_pn_reco.sh
sbatch/submit_teacher_logit_pn_reco_experiment.sh
```

The submitter should queue:

```text
train PN reco for each selected teacher architecture
-> predict stack_train/stack_val/final_test for each trained reco
-> independent fusion over saved prediction blocks
```

Recommended first defaults:

```text
TEACHER_LOGIT_PN_TEACHERS="part"
train walltime: 12 hours, GPU
predict walltime: 5 hours, GPU
fusion walltime: 5 hours, CPU
```

Later, when PN/PFN/PCNN offline teachers exist:

```text
TEACHER_LOGIT_PN_TEACHERS="part pn pfn pcnn"
```

Implemented notes:

- `sbatch/common.sh` now has `TEACHER_LOGIT_PN_*` output, teacher-checkpoint, architecture, prediction, and fusion defaults.
- `sbatch/run_train_teacher_logit_pn_reco.sh` trains one PN reconstructor against one selected offline teacher.
- `sbatch/run_predict_teacher_logit_pn_reco.sh` writes stack/final-test prediction blocks from a trained PN reconstructor.
- `sbatch/run_fuse_teacher_logit_pn_reco.sh` runs the independent fusion script on those saved PN prediction blocks.
- `sbatch/submit_teacher_logit_pn_reco_experiment.sh` queues train -> predict for each selected teacher architecture, then fuses after all predictions finish.
- `tests/test_sbatch_scripts.py` now pins the PN Step 8 runner names, walltimes, GPU requests, output roots, and dependency wiring.

## What Not To Do Yet

- Do not replace the GT reconstructor path.
- Do not change the loss formula unless PN clearly needs it.
- Do not introduce offline particle matching as the main objective.
- Do not use transformer cross-attention in the PN extra candidate decoder.
- Do not evaluate or tune on `final_test`.
- Do not overwrite existing GT outputs.

## Success Criteria

The PN reconstructor is worth keeping if at least one is true:

- PN reconstructed-view teacher logits beat raw HLT on `stack_val`.
- PN reconstructed-view logits improve fusion even if not best alone.
- PN and GT reconstructors make meaningfully different errors.
- PN helps classes where GT does not.
- Diversity audits show PN contributes non-redundant signal.
