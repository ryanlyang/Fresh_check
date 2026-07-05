# `av10_feature_mlp_adapter` Explained

This note explains the `av10_feature_mlp_adapter` variant in the Architecture-View 10-class ablation suite. It assumes the reader already knows JetClass, HLT/offline particle representations, and Particle Transformer-style tagging, but has not seen this specific adapter setup.

## Short Version

`av10_feature_mlp_adapter` is a 10-class HLT ParT tagger with a small per-particle MLP inserted into the ParT embedding pathway.

For each HLT particle, the model builds the same canonical PF feature row that the HLT ParT already consumes. A small MLP reads that feature row and predicts a residual vector in ParT embedding space:

```text
canonical particle features F_i
    -> feature MLP + sigmoid gate
    -> delta_h_i

ParT embedding h_i
    -> h_i + delta_h_i
    -> normal ParT global attention and classifier
```

The raw particle inputs, Lorentz vectors, pairwise geometry inputs, masks, and downstream ParT attention stack remain the real HLT ParT pathway. The adapter only changes the internal particle embedding after ParT has embedded each particle.

## Why This Variant Exists

The motivating observation from the AV10 runs was that some “context” variants improved over the plain HLT ParT baseline, and the best single run looked less like a fancy architecture-view branch and more like a simple learned per-particle feature adapter.

That raised the key ablation question:

```text
Is the gain really coming from PN/PFN/PCNN-style alternate architecture views,
or is ParT benefiting from a small learned feature-to-embedding residual adapter?
```

`av10_feature_mlp_adapter` tests the second possibility directly. It removes PN/PFN/PCNN view branches and gives ParT only a compact MLP-derived residual computed from the canonical particle features.

If it wins, the result says something important: the canonical HLT ParT embedding layer may not be the best way to lift raw PF features into particle embeddings, even when the rest of ParT is strong. A small nonlinear per-particle residual can reshape the embedding space before global attention.

## Relationship To Other AV10 Variants

`av10_feature_mlp_adapter` is part of the 10-class AV10 ablation suite:

```text
variant name: av10_feature_mlp_adapter
suite:        av10_ablation
adapter type: feature_mlp_context
selection:    model_val accuracy
labels:       10 JetClass classes
view:         HLT, usually HLT degradation strength 0.6
```

It differs from the nearby variants like this:

| Variant | What it changes | Main question |
| --- | --- | --- |
| `av10_hlt_baseline_recheck` | Nothing. Exact HLT ParT recheck. | What is the baseline on this split/cache? |
| `av10_feature_mlp_adapter` | Adds `delta_h` from canonical per-particle features. | Does a small feature MLP improve ParT embeddings? |
| `av10_feature_mlp_adapter_wide` | Same idea, bigger MLP. | Does scaling this adapter help? |
| `av10_frozen_part_feature_adapter` | Same adapter, but ParT is frozen. | Can the adapter help without ParT fine-tuning? |
| `av10_shuffled_feature_adapter` | Same capacity, broken feature semantics. | Is the gain semantic or just capacity/regularization? |
| `av10_pfn_context_repeat` | PFN-style view branch creates `delta_h`. | Does PFN-like reasoning add useful context? |
| `av10_pcnn_context_repeat` | PCNN-style view branch creates `delta_h`. | Does PCNN-like reasoning add useful context? |
| `av10_lc_mlp_delta_features` | Edits input PF feature rows before ParT embedding. | Is input-space feature repair better than embedding-space repair? |
| `av10_larger_part` | Larger ParT capacity control. | Is the gain just “more parameters”? |
| `av10_extra_part_block` | Extra self-attention-style block in embedding space. | Is another ParT-like block better than the MLP adapter? |

The important distinction is:

```text
av10_feature_mlp_adapter changes embeddings.
av10_lc_mlp_delta_features changes input feature rows.
PN/PFN/PCNN context variants use alternate architecture branches.
```

## Data Flow

At a high level, one forward pass looks like this:

```text
raw HLT tokens + mask
    -> canonical ParT input builder
       - points
       - PF feature tensor
       - Lorentz vectors
       - ParT mask
       - per-particle canonical feature rows

canonical feature row F_i
    -> context_control_gate(F_i)
    -> sigmoid gate g_i

canonical feature row F_i
    -> context_control(F_i)
    -> adapter output a_i

delta_h_i = g_i * a_i

real HLT ParT embedding:
    h_i = ParTEmbed(points_i, features_i, vectors_i)

embedding injection:
    h'_i = h_i + delta_h_i

normal HLT ParT:
    h'_i -> pairwise/global attention -> classifier logits
```

The adapter is particle-local. Particle `i` gets its residual from particle `i`'s canonical feature row. It does not do kNN message passing, PFN pooling, PCNN convolutions, or global attention itself. The global reasoning still happens in the ParT backbone after the residual is injected.

## Canonical Inputs

The model first converts HLT raw tokens into canonical ParT-compatible inputs using the same local-compression canonical input builder used elsewhere in this project. The canonical object contains:

- `points`: the ParT point/geometry tensor.
- `features`: the canonical PF feature tensor passed into ParT.
- `lorentz_vectors`: the four-vector-style tensor passed into ParT.
- `mask`: the ParT-compatible particle mask.
- `particle_mask`: a batch-first valid-particle mask.
- `feature_rows()`: a batch-first `[batch, particles, feature_dim]` view of the canonical PF feature rows.

For `av10_feature_mlp_adapter`, these canonical inputs are not repaired or replaced. They are used exactly as the HLT ParT would use them, except that `feature_rows()` also feeds the adapter MLP.

## Adapter Architecture

The active adapter modules are:

```text
context_control
context_control_gate
```

The standard feature MLP path is:

```text
context_control:
    LayerNorm(canonical_feature_dim)
    Linear(canonical_feature_dim, fusion_hidden_dim)
    GELU
    Dropout
    Linear(fusion_hidden_dim, part_embed_dim)

context_control_gate:
    LayerNorm(canonical_feature_dim)
    Linear(canonical_feature_dim, fusion_hidden_dim)
    GELU
    Linear(fusion_hidden_dim, 1)
    Sigmoid applied in forward
```

For each particle:

```text
a_i = context_control(F_i)
g_i = sigmoid(context_control_gate(F_i))
delta_h_i = g_i * a_i
```

Then invalid/padded particles are zeroed:

```text
delta_h_i = 0 for masked particles
```

`part_embed_dim` must match the internal particle embedding dimension of the HLT ParT backbone. The injected residual has shape:

```text
[batch, particles, part_embed_dim]
```

## Baseline-Preserving Initialization

The adapter is initialized to recover the baseline ParT at step zero.

The final projection of `context_control` is zero-initialized:

```text
context_control final Linear weight = 0
context_control final Linear bias   = 0
```

The final projection of the gate MLP is also zero-initialized, with a negative bias:

```text
context_control_gate final Linear weight = 0
context_control_gate final Linear bias   = gate_bias_init
```

The default gate bias is negative, so the initial gate is small. More importantly, because `context_control` initially outputs exactly zero, the initial residual is:

```text
delta_h_i = g_i * 0 = 0
```

Therefore, before training, this variant should produce the same logits as the baseline HLT ParT up to numerical details. That matters because the experiment is not asking whether a randomly perturbed ParT can eventually recover. It is asking whether a baseline-preserving residual adapter can improve the baseline.

## Embedding Injection

The implementation injects `delta_h` by registering a forward hook on the resolved ParT embedding module, normally `part_model.mod.embed`.

Conceptually:

```text
embed_output = ParT embedding output
embed_output = embed_output + aligned_delta_h
```

The hook handles common embedding layouts:

```text
[batch, particles, dim]
[particles, batch, dim]
[batch, dim, particles]
```

It also handles the case where ParT internally trims the particle dimension. If the embedding output has fewer particles than the original padded input, the residual is trimmed to match the actual embedding output length.

This means the adapter does not require rewriting the ParT implementation. It wraps the real ParT, intercepts the first particle embedding, adds the residual, and lets the rest of ParT run normally.

## What Actually Changes In The Model

The raw HLT event representation is unchanged:

- The selected HLT particles are unchanged.
- The particle mask is unchanged.
- The canonical PF feature tensor passed to ParT is unchanged.
- The Lorentz vectors passed to ParT are unchanged.
- Pairwise geometry inputs are unchanged.

The changed object is the internal particle embedding:

```text
h_i -> h_i + delta_h_i
```

This is why it is best to think of `av10_feature_mlp_adapter` as an embedding-space residual adapter, not a reconstructor and not an input feature repair model.

## Training Objective

This variant trains as a standard 10-class classifier.

The main loss is cross entropy over the 10 JetClass labels:

```text
loss = CrossEntropyLoss(logits, labels)
```

The AV10 ablation trainer can compute delta diagnostics, but the current regularized delta-F loss is intended for the input-feature delta variant (`av10_lc_mlp_delta_features`). For `av10_feature_mlp_adapter`, the important training signal is the classification cross entropy after embedding injection.

Selection is by model-validation accuracy for the 10-class suite:

```text
primary selection metric = accuracy on model_val
```

The final report can then evaluate model-val, stack-val, and final-test metrics from the selected checkpoint.

## Optimization Schedule

The optimizer separates the small adapter from the ParT backbone:

```text
adapter params -> adapter_lr
ParT params    -> part_lr
```

The usual serious defaults use a higher adapter learning rate and a lower ParT fine-tuning rate. The intended behavior is:

1. Start from the exact HLT ParT checkpoint.
2. Train the adapter while preserving the baseline at initialization.
3. Often freeze ParT for the first few epochs so the adapter learns a useful residual direction.
4. Unfreeze ParT and allow the full system to co-adapt gently.

This schedule matters for interpretation. If the adapter helps while ParT is frozen, it is strong evidence that the adapter itself adds useful representation. If it only helps after ParT is unfrozen, it may be helping as a fine-tuning scaffold or optimization perturbation.

## Parameter Accounting

The ablation code distinguishes active and dormant modules. For `av10_feature_mlp_adapter`, the active adapter modules are:

```text
context_control
context_control_gate
```

PN/PFN/PCNN branches, LC input-delta modules, extra ParT blocks, and other unrelated adapters are not supposed to count as active parameters for this variant. They may exist in the wrapper class for code reuse, but inactive modules are frozen and excluded from optimizer groups and active-parameter accounting.

This is essential because the point of the ablation is not merely “a wrapper class with more allocated parameters.” It is specifically “a feature MLP residual into ParT embeddings.”

## Diagnostics To Inspect

Important diagnostics for this variant include:

- `delta_h_norm_mean`: average residual size in embedding space.
- `delta_h_norm_p90`: high-end residual size.
- `delta_h_norm_max`: largest observed residual size.
- `gate_mean`: average sigmoid gate value.
- `gate_p10` / `gate_p90`: gate distribution.
- `adapter_output_norm_mean`: norm of the raw MLP output before gate multiplication.
- `embedding_norm_mean`: norm of the baseline ParT embedding.
- `delta_to_embedding_norm_ratio`: how large the residual is relative to the original embedding.
- `injection_applied`: whether the ParT embedding hook actually injected a residual.
- active adapter parameter counts.
- part/backbone parameter counts.
- adapter and ParT gradient norms.

The healthy pattern is not necessarily “large delta.” A good adapter may be small but consistently useful, especially if it nudges particles near difficult class boundaries.

## Why It Might Improve ParT

ParT already embeds PF features before attention, so at first glance this adapter can look redundant. The reason it may help is that the default embedding layer is a generic feature lift trained jointly with the whole transformer. A small residual MLP can give the model an extra local nonlinear correction before global attention.

Possible mechanisms:

- It can reshape feature interactions before attention sees them.
- It can learn class-useful particle-level nonlinearities that the baseline embedding underweights.
- It can act as a soft per-particle calibration layer for HLT-degraded features.
- It can make the downstream attention easier to optimize by presenting a better initial token representation.
- It can add a low-cost specialization layer without replacing the strong ParT backbone.

The key idea is not that the MLP is smarter than ParT globally. It is that a cheap per-particle residual may prepare better particle tokens for ParT's global reasoning.

## How To Interpret Results

If `av10_feature_mlp_adapter` beats the HLT ParT baseline:

- It supports the idea that embedding-space residual adaptation is useful.
- It does not prove that PN/PFN/PCNN views are necessary.
- It suggests the baseline ParT embedding layer may be a bottleneck or at least improvable.

If it beats PN/PFN/PCNN context variants:

- The alternate architecture branches may not be the main source of improvement.
- The useful ingredient may be “learned residual context into ParT” rather than architecture-specific reasoning.

If it loses to `av10_lc_mlp_delta_features`:

- Input-space feature repair may be more useful than embedding-space residuals.
- The final ParT may prefer physically interpretable adjusted features over hidden embedding shifts.

If it loses to `av10_larger_part`:

- The gain may be mostly capacity, not the specific adapter mechanism.

If `av10_shuffled_feature_adapter` performs similarly:

- Be suspicious. That would suggest capacity, regularization, or optimization dynamics rather than meaningful feature semantics.

If `av10_frozen_part_feature_adapter` improves:

- That is especially strong evidence that the adapter is adding usable information or representation without relying on full ParT fine-tuning.

## Limitations

This adapter is intentionally simple, and that simplicity is both a strength and a limitation.

It does not create new particles, recover dropped HLT constituents, or reconstruct offline information directly. It does not reason over particle neighborhoods before ParT. It does not pass explicit PN/PFN/PCNN latent views into ParT. It only gives each particle an extra learned embedding residual derived from its own canonical feature row.

So if it works, the most conservative interpretation is:

```text
The canonical HLT ParT pathway benefits from a learned per-particle embedding residual.
```

That is already a meaningful result. It says there may be a simple way to improve the particle-token representation that the transformer sees, without replacing the transformer itself.

## Main Code Pointers

The relevant implementation lives in:

- `teacher_logit_reco/architecture_view_part/config.py`
  - Defines `av10_feature_mlp_adapter`.
  - Maps it to the context-MLP effective behavior.
  - Marks it as an AV10 ablation variant.

- `teacher_logit_reco/architecture_view_part/model.py`
  - Builds `context_control` and `context_control_gate`.
  - Computes `delta_h`.
  - Injects `delta_h` into the exact HLT ParT embedding module.
  - Reports variant behavior and active parameter accounting.

- `teacher_logit_reco/architecture_view_part/train.py`
  - Loads data/cache and the baseline checkpoint.
  - Builds optimizer groups for adapter and ParT.
  - Applies freeze/unfreeze schedule.
  - Trains with 10-class cross entropy.
  - Selects checkpoints by validation accuracy.
  - Writes diagnostics and final metrics.

- `teacher_logit_reco/architecture_view_part/ablation_report.py`
  - Summarizes this variant alongside baseline, capacity controls, shuffled controls, and architecture-view repeats.

## Mental Model

The cleanest mental model is:

```text
HLT ParT baseline:
    particle features -> ParT embed -> ParT attention -> classifier

av10_feature_mlp_adapter:
    particle features -> ParT embed + MLP(feature residual) -> ParT attention -> classifier
```

Nothing about the downstream ParT changes conceptually. The model still asks ParT to do the final particle-token reasoning. The adapter simply gives ParT slightly better particle embeddings to reason over.

