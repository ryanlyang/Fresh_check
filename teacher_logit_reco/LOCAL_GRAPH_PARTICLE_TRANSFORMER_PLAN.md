# Local Graph Particle Transformer Plan

## Purpose

This plan describes a serious attempt to beat the HLT Particle Transformer baseline in the specific regime we care about now:

- Task: QCD vs Hgg
- Inference view: HLT only
- HLT degradation strength: 0.6
- Primary metric: false-positive rate at 50% signal efficiency, `FPR@50`
- Baseline to beat: HLT ParT trained on the same HLT cache and split

The goal is not reconstruction. The goal is to build a better HLT-side classifier by adding the right inductive bias before or inside ParT.

The central hypothesis:

> HLT ParT is extremely strong at global particle-particle reasoning, but it does not explicitly force local detector/substructure reasoning before global attention. HLT degradation is local, and QCD vs Hgg is substructure-heavy. A local graph module before or inside ParT may extract useful HLT information that vanilla ParT leaves underused.

## Inspiration

### GraphGPS

GraphGPS combines local graph message passing with global transformer attention. The lesson is not that transformers are bad. The lesson is that global attention and local structure solve different parts of the problem. Local message passing captures neighborhood/subgraph patterns; global attention captures long-range context.

For jets:

```text
local graph reasoning:
  nearby constituents, local prongs, HLT merging/dropout artifacts

global ParT reasoning:
  full-jet class structure, all-particle pairwise relations, global topology
```

### Point Transformer

Point Transformer improves over generic set transformers by using local neighborhoods and relative geometry. Jets are point-cloud-like objects in eta-phi plus feature space. HLT degradation also acts locally in eta-phi. So a local point-attention module is a natural particle-physics analogue.

### ParticleNet Plus ParT

ParticleNet has strong local geometric bias through EdgeConv. ParT has strong global pairwise attention. The proposed architecture tries to combine those strengths:

```text
ParticleNet-like local geometry
  +
ParT-like global pairwise attention
```

This is not a random ensemble. It is a structured local-global model.

## Why Plain ParT May Leave Room

ParT treats each particle as a token and uses pairwise interaction features in attention. That is already excellent. But several things are still not explicit:

1. Local HLT reliability

   A particle may be trustworthy or suspicious depending on nearby HLT particles. A locally dense region, a missing nearby track, or a strange neighbor pattern can signal HLT corruption.

2. Local prong structure

   Hgg vs QCD is about distinguishing signal-like prong organization from QCD radiation. A local graph module can form local summaries before the global transformer makes the final decision.

3. Neighborhood consistency

   ParT global attention can learn this, but it is not forced to compute stable local neighborhoods early. A kNN graph in eta-phi gives the architecture a clear detector geometry prior.

4. Optimization floor

   Even if ParT is Bayes-consistent in principle, a better inductive bias can reduce the fixed-architecture/optimizer floor. This is the only defensible way to claim persistent gains at high data counts.

## High-Level Architecture

The best first architecture should not replace ParT. It should augment ParT.

```text
HLT raw tokens
  -> ParT-compatible particle embedding
  -> local graph adapter
  -> global ParT-style pairwise attention
  -> classifier
```

The local graph adapter should be residual:

```text
h0 = particle_embedding(raw_hlt)
h1 = h0 + gamma * LocalGraphAdapter(h0, raw_hlt, local_edges)
logits = ParTBackbone(h1, pairwise_features)
```

The scalar `gamma` should be initialized to zero or a tiny value. This makes the model start as close as possible to the baseline ParT. The local module only becomes active if it helps.

This residual-zero design is important. It reduces the risk that the new model loses because it is harder to optimize or because it perturbs a strong baseline too much.

## Model Variants

### Variant A: EdgeConv Residual Adapter

This is the simplest local graph module.

For each particle `i`, find k nearest neighbors `j` in eta-phi. Compute messages:

```text
m_ij = MLP([h_i, h_j - h_i, edge_ij])
h_i_local = pool_j m_ij
h_i_out = h_i + gamma * projection(h_i_local)
```

Recommended pooling:

- start with max plus mean concatenation
- max captures sharp local features
- mean captures density and smooth neighborhood context

Why this is useful:

- simple
- stable
- close to ParticleNet's successful bias
- good first test of whether local geometry helps at all

### Variant B: Point-Attention Residual Adapter

This is the more expressive version.

For each local edge `i -> j`:

```text
q_i = Wq h_i
k_j = Wk h_j + phi_k(edge_ij)
v_j = Wv h_j + phi_v(edge_ij)
alpha_ij = softmax_j(q_i dot k_j / sqrt(d))
h_i_local = sum_j alpha_ij * v_j
h_i_out = h_i + gamma * projection(h_i_local)
```

This is closer to Point Transformer. It lets each particle decide which neighbors matter, using geometry-aware attention.

Why this may beat EdgeConv:

- local reliability is context-dependent
- not all nearby particles should matter equally
- local attention can learn prong-specific or detector-artifact-specific neighbor patterns

### Variant C: Interleaved GraphGPS-ParT Blocks

This is the most principled long-term architecture:

```text
for block in layers:
    h = h + LocalGraphBlock(h, local_edges)
    h = h + ParTGlobalPairwiseAttention(h, pairwise_features)
    h = h + FFN(h)
```

This is closest to GraphGPS. It repeatedly alternates local geometry and global attention.

This should be implemented only after Variant A/B show promise. It is more invasive and harder to warm-start from a baseline ParT checkpoint.

## Recommended First Serious Setup

The first serious implementation should be:

```text
LocalPointAttentionAdapter + ParTGlobalBackbone
```

with an EdgeConv adapter as a control.

The design should keep these invariants:

- HLT-only inference
- same train/val/test split as the current QCD/Hgg runs
- same HLT cache, degradation 0.6
- same baseline HLT ParT in the same run/report
- primary metric `FPR@50`
- final-test confirmation required

## Warm-Start Strategy

The highest-probability path is to warm-start from an HLT ParT baseline.

Training schedule:

1. Train or load baseline HLT ParT.
2. Insert local graph adapter.
3. Initialize adapter residual scale `gamma = 0`.
4. Freeze the ParT backbone briefly and train only:
   - local graph adapter
   - residual scale
   - maybe classifier head
5. Unfreeze the full model with a lower learning rate.
6. Fine-tune end-to-end.

Why this matters:

- A new model trained from scratch may simply fail to catch up to ParT.
- A residual adapter starts at the baseline and learns improvements.
- This tests the clean hypothesis: can local graph structure improve an already strong ParT?

If warm-starting the existing Weaver ParT internals is too hard, still implement the residual adapter with a ParT-compatible wrapper. But the warm-start path should remain a first-class goal.

## Local Graph Construction

### Coordinates

Use HLT particle eta and phi.

Distance:

```text
delta_eta = eta_i - eta_j
delta_phi = wrapped(phi_i - phi_j)
delta_R = sqrt(delta_eta^2 + delta_phi^2)
```

### kNN

Recommended starting values:

- `k = 12`
- `k = 16`
- optionally `k = 24` as a larger-neighborhood ablation

Use only valid HLT particles. Padded particles must not appear as neighbors.

The graph can be rebuilt on the fly per batch. For 128 particles, exact pairwise distance is cheap enough:

```text
B x N x N distance matrix
mask invalid particles
topk nearest neighbors
```

### Edge Features

Use edge features that overlap with ParT physics but are focused on local context.

Minimum edge features:

```text
delta_eta
delta_phi
delta_R
log(delta_R + eps)
log_pt_i
log_pt_j
log_pt_j - log_pt_i
```

Physics pair features:

```text
log_pair_mass
relative_kT
z
```

Local detector/reliability hints:

```text
same_charge_or_both_neutral
pid_compatibility_summary
track_displacement_difference
local_pt_fraction
neighbor_rank_by_deltaR
```

Be careful not to create unphysical leakage. Everything must come from HLT tokens at inference.

## Pairwise Global Attention

The model should preserve ParT-style pairwise bias. This is non-negotiable.

The baseline ParT is strong partly because it adds particle-particle physics features to attention. A local graph module without global pairwise attention would not be a fair attempt to beat ParT.

The global stage should therefore use one of:

1. existing Weaver ParticleTransformer if we can inject modified particle embeddings cleanly
2. local repo's ParT-compatible attention implementation with pairwise bias
3. a faithful reimplementation only if needed

Avoid building a weaker "transformer-like" global stage and calling it ParT. The comparison must be against the real strength of HLT ParT.

Implementation update after Step 5 review:

- The custom pairwise-biased Transformer wrapper is only a prototype sandbox.
- It must not be called `hlt_part_baseline`.
- Its no-local mode is only an adapter-disabled control, not the baseline.
- It must not be reported as serious-comparison-ready.
- The serious Step 5 path is now:

```text
raw HLT tokens
  -> canonical ParT PF inputs
  -> local graph adapter in hidden feature space
  -> residual correction to canonical PF features
  -> real Weaver ParticleTransformer backbone
  -> classifier logits
```

This is not full internal embedding injection into Weaver, but it keeps the real ParT backbone and canonical ParT input contract. It is therefore the first serious implementation path. Later warm-start work should still try deeper internal integration if Weaver exposes a clean insertion point.

## Reliability-Aware Extension

A natural extension is to let the local graph module output reliability gates:

```text
reliability_i = sigmoid(MLP([h_i, h_i_local, local_stats_i]))
h_i_out = h_i + gamma * reliability_i * local_update_i
```

This is motivated by HLT degradation. Some particles are reliable, some are suspicious, and reliability often depends on the local neighborhood.

Optional diagnostics:

- mean reliability by class
- reliability vs pT
- reliability vs local density
- reliability vs HLT drop/merge diagnostics, only as analysis, not training input

This can make the model more interpretable and help us understand whether it is learning a real HLT correction strategy.

## Auxiliary Training Signals

The core model should first be HLT-label-only. But after that, auxiliary signals are worth testing.

Allowed training-only signals:

1. Offline teacher logits for distillation
2. Offline-vs-HLT residual/reliability targets
3. Local consistency/self-supervised masked-particle objectives

For the first serious architecture test, keep auxiliary losses off. We need to know whether the architecture itself beats HLT ParT.

Then test:

```text
local_graph_part
local_graph_part + residual auxiliary
local_graph_part + distillation
local_graph_part + residual + distillation
```

Our previous QCD/Hgg result suggests residual auxiliary training is more promising than distillation-only.

## Metrics

Primary:

```text
FPR@50 signal efficiency
```

Secondary:

```text
FPR@30
background rejection @ 50
AUC
accuracy
calibration
per-class / per-region diagnostics if multiclass later
```

For QCD vs Hgg, accuracy is not the main win condition.

The report must always compare:

```text
HLT ParT baseline
local EdgeConv adapter
local point-attention adapter
warm-started local point-attention adapter
optional residual/distillation variants
offline ParT reference
```

## Diagnostics

To understand whether the model is really using local structure, save:

### Local Graph Diagnostics

- average kNN deltaR
- p10/p50/p90 neighbor deltaR
- fraction of particles with fewer than k valid neighbors
- local density summary
- local pt fraction summary

### Adapter Diagnostics

- learned residual scale `gamma`
- adapter output norm vs particle embedding norm
- reliability gate mean/std if enabled
- local attention entropy if using point attention
- top neighbor distance distribution for high-attention edges

### Performance Slices

Evaluate FPR/accuracy by:

- jet constituent count
- mean HLT constituent count
- HLT drop fraction
- HLT merge count
- jet pT if available
- high-density vs low-density jets

This is especially important because the model may improve only in the HLT-degraded corner.

## Implementation Boundaries

Create a new module rather than tangling this with the subtoken code:

```text
teacher_logit_reco/local_graph_part/
```

Suggested files:

```text
config.py
features.py
knn.py
edge_features.py
local_blocks.py
model.py
train.py
reports.py
compat.py
```

Scripts:

```text
scripts/train_local_graph_part_tagger.py
scripts/run_local_graph_part_comparison.py
scripts/write_local_graph_part_report.py
```

Slurm:

```text
sbatch/run_local_graph_part_comparison.sh
sbatch/submit_local_graph_qcd_hgg_binary_experiment.sh
```

The code should reuse:

- existing HLT cache loaders
- existing binary split discipline
- existing metric code
- existing diagnostics mirroring
- existing FPR@30/FPR@50 reporting

## Proposed Experiment Matrix

First real run:

```text
Task: QCD vs Hgg
HLT degradation: 0.6
Train: 500k
Val: 150k
Stack val: 150k
Final test: 500k
Epochs: 45
Primary metric: FPR@50
```

Variants:

```text
hlt_part_baseline
local_edgeconv_adapter
local_point_attention_adapter
local_point_attention_adapter_warmstart
```

Second run if promising:

```text
local_point_attention_adapter
local_point_attention_adapter_residual_aux
local_point_attention_adapter_distill
local_point_attention_adapter_distill_residual
```

Third run if still promising:

```text
interleaved_graphgps_part
interleaved_graphgps_part_residual_aux
```

## Risk Register

### Risk 1: The local module helps accuracy but hurts FPR@50

This happened somewhat with previous subtoken variants. Selection must be based on FPR@50, not accuracy.

### Risk 2: The new model is just bigger

Parameter count and runtime must be reported. Include parameter-matched controls if the gain is large enough to matter.

### Risk 3: The global stage is weaker than real ParT

Do not compare a custom weaker "ParT-like" stage to real HLT ParT and overinterpret the result. Preserve the real pairwise global bias.

### Risk 4: Warm-start incompatibility

If existing Weaver ParT does not expose internals cleanly, build the non-warm-start version first, but keep the adapter initialization conservative.

### Risk 5: Local graph overfits HLT artifacts

Use final test strictly. Also inspect performance by HLT drop/merge bins.

## Implementation Steps

### Step 1: Freeze The Target Protocol

Write a small protocol note for this architecture:

- QCD vs Hgg only
- HLT degradation 0.6
- exact split names
- exact metrics
- exact baseline
- final test confirmation required

### Step 2: Build kNN And Edge Feature Utilities

Implement:

- valid-particle-aware eta-phi kNN
- wrapped delta phi
- edge feature builder
- tests for masks, padding, phi wraparound, and finite outputs

### Step 3: Implement EdgeConv Local Adapter

Implement a residual local EdgeConv block:

- input particle embeddings
- raw HLT tokens
- kNN edge index
- edge features
- pooled local message
- residual projection
- learnable `gamma` initialized to zero

### Step 4: Implement Point-Attention Local Adapter

Implement local attention over kNN neighborhoods:

- geometry-aware keys/values
- attention mask for invalid neighbors
- attention entropy diagnostics
- residual `gamma`

### Step 5: Integrate With A ParT-Compatible Backbone

Create a serious classifier wrapper:

```text
raw HLT tokens
  -> build canonical ParT PF features/vectors/mask
  -> embed PF feature rows into local hidden space
  -> local EdgeConv or Point-Attention adapter
  -> project local hidden residual back to PF feature residual
  -> real Weaver ParticleTransformer
  -> classifier logits
```

The adapter residual must initialize to exact identity, so the model starts as
the reference ParT input stream. The code may keep a custom pairwise-biased
Transformer prototype for fast smoke tests, but it is not the serious model and
must be labeled prototype-only in summaries/reports. Any no-local adapter mode
must be named as an adapter-disabled control, never as `baseline` or
`hlt_part_baseline`; the real baseline is trained/evaluated through the HLT ParT
baseline runner.

### Step 6: Add HLT ParT Baseline In Same Runner

The comparison runner must train/evaluate:

- baseline HLT ParT
- local adapter variants

on identical cache/splits.

Implementation update:

- `teacher_logit_reco/local_graph_part/train.py` now contains the Step 6 shared runner.
- `scripts/train_local_graph_part_tagger.py` is the CLI entry point for one variant per job.
- The real baseline variant is `hlt_part_baseline`, implemented through `HLTPartBaselineRawTokenClassifier`, which converts raw HLT tokens into canonical ParT inputs and feeds the reference Weaver `ParticleTransformerHLTClassifier`.
- The local variants use `LocalGraphAugmentedParticleTransformerClassifier`, so baseline and adapters share the same raw-token dataset, metric code, checkpoint-selection rule, and final-test guard.
- Checkpoint selection is frozen to `fpr_at_signal_eff_0p50`; accuracy is only diagnostic.
- The optional `local_graph_adapter_disabled_control` remains a control, not the HLT ParT baseline.

### Step 7: Add Warm-Start Path

Try to load baseline HLT ParT weights into the local-adapter model. If direct loading is not possible, document why and implement the closest safe alternative.

Training phases:

1. adapter-only
2. full fine-tune

Implementation update:

- `LocalGraphTaggerTrainConfig` now supports `warm_start_checkpoint`, `require_warm_start`, and `freeze_part_epochs`.
- `warm_start_local_graph_part_model(...)` loads baseline HLT ParT weights into the local adapter model's `part_model.*` subtree.
- It supports both Step 6 wrapper checkpoints with keys like `part_model.mod.*` and older direct HLT baseline checkpoints with keys like `mod.*`.
- The warm-start report is written to `diagnostics/warm_start_report.json` and includes loaded key counts, unmatched source-key samples, and shape mismatches.
- If `freeze_part_epochs > 0`, the trainer freezes the ParT backbone for initial adapter-only epochs, then restores full fine-tuning and records the phase in `training_curves.json`.
- Warm-start and freeze are rejected for the `hlt_part_baseline` variant; they are intended for local adapter variants.

### Step 8: Add Diagnostics And Reports

Add:

- metric table
- gate/adapter diagnostics
- local graph diagnostics
- parameter counts
- runtime summary
- HLT degradation slice metrics

Implementation update:

- `teacher_logit_reco/local_graph_part/reports.py` now builds the Step 8 final report from child `run_report.json` files.
- The report writes `metric_table.csv`, `adapter_diagnostics.csv`, `parameter_counts.csv`, `runtime_summary.csv`, `hlt_degradation_summary.csv`, `baseline_comparison.csv`, `local_graph_part_report.json`, `local_graph_part_report.md`, and a mirrored `run_report.json`.
- Binary QCD/Hgg reports default to `fpr_at_signal_eff_0p50` with lower-is-better semantics, matching the frozen protocol.
- Child report paths are resolved relative to the experiment directory first, so Slurm/root-report paths are stable even if the current working directory contains similarly named folders.
- Adapter diagnostics include local graph values such as local residual `gamma`, mean valid neighbors, mean neighbor deltaR, and point-attention entropy when present.
- HLT degradation reporting currently summarizes cache/run metadata such as `hlt_params`, HLT content hashes, label counts, and the frozen protocol degradation strength. Per-jet drop/merge slices can be added once the HLT cache persists per-jet degradation diagnostics.
- `scripts/write_local_graph_part_report.py` is the CLI entry point for writing these tables.

### Step 9: Add Slurm Runner

Add a submitter for:

```text
QCD vs Hgg
HLT degradation 0.6
500k / 150k / 500k / 150k / 500k split protocol
45 epochs
FPR@50 selection
```

### Step 10: First Serious Run

Run:

```text
hlt_part_baseline
local_edgeconv_adapter
local_point_attention_adapter
local_point_attention_adapter_warmstart
```

### Step 11: Analyze Before Adding Auxiliary Losses

Only after the architecture-only run:

- compare FPR@50
- inspect local graph diagnostics
- inspect bins by HLT degradation
- decide whether to add residual/distillation

### Step 12: Add Privileged Auxiliary Training If Warranted

If local graph architecture is close or better:

```text
local_point_attention_adapter_residual_aux
local_point_attention_adapter_distill
local_point_attention_adapter_distill_residual
```

This tests whether the previous residual-training gain stacks with local graph structure.

## Success Criteria

Minimum meaningful success:

```text
FPR@50 lower than HLT ParT on final_test
```

Stronger success:

```text
FPR@50 improves by >= 2 percent relative
and AUC/accuracy do not regress badly
```

Excellent success:

```text
FPR@50 improves by >= 5 percent relative
and improvement is concentrated in high-HLT-degradation bins
```

The scientific story is strongest if the model improves exactly where local HLT degradation is worst.
