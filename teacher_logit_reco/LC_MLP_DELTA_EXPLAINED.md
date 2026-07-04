# `lc_mlp_delta`: Learned Feature Repair Before HLT ParT

This note explains the `lc_mlp_delta` variant in the local-compression Particle Transformer work. It is meant as a handoff document for an outside agent: what the variant does, what it does not do, where the code lives, and why the result is scientifically interesting.

## Short Version

`lc_mlp_delta` is a residual input-feature repair adapter in front of the exact HLT Particle Transformer backbone.

Conceptually:

```text
fixed HLT particle tokens
  -> canonical ParT feature builder
  -> small MLP predicts bounded delta_F for each particle's ParT feature row
  -> adapted ParT features = original ParT features + delta_F
  -> exact HLT ParT backbone/classifier
  -> QCD vs Hgg logits
```

The important point: `lc_mlp_delta` does **not** replace ParT, and it does **not** use the full local-compression/modality-subtoken stack. It gives the already-strong ParT a learnable, near-identity correction to its input feature representation.

The code constants define it as:

- `LOCAL_COMPRESSION_VARIANT_MLP_DELTA = "lc_mlp_delta"` in `teacher_logit_reco/local_compression_part/config.py`.
- The variant description is `"Feature-delta MLP control with no modality tokens."`

## Experiment Context

The local-compression suite is frozen around the QCD vs Hgg HLT0.6 binary setting:

- Labels: `QCD=0`, `Hgg=1`.
- Input view: fixed HLT-degraded particles only.
- HLT degradation strength: `0.6`.
- Primary metric: `fpr_at_signal_eff_0p50`, minimized.
- Baseline: exact HLT ParT checkpoint, selected by FPR@50 on `model_val`.
- Splits:
  - `model_train`: train adapter and optional ParT fine-tune.
  - `model_val`: checkpoint selection.
  - `stack_val`: unbiased diagnostic/stacking validation.
  - `final_test`: held-out final comparison.

The protocol-level constants and guardrails are in:

- `teacher_logit_reco/local_compression_part/config.py`
  - variant constants around lines 41-64,
  - primary metric around lines 37-39,
  - HLT0.6/QCD-Hgg protocol validation in `LocalCompressionProtocol.validate`,
  - variant descriptions around lines 537-554.

The training config enforces the important split/metric constraints in:

- `teacher_logit_reco/local_compression_part/train.py`
  - `LocalCompressionTaggerTrainConfig.__post_init__`,
  - requires `model_train`/`model_val`,
  - requires `final_test`,
  - requires `fpr_at_signal_eff_0p50`,
  - requires QCD/Hgg label mapping.

## What Inputs Enter the Model?

The model starts from raw cached HLT tokens:

```text
tokens: [batch, particles, raw_dim]
mask:   [batch, particles]
```

The raw token names are currently:

```text
pt, eta, phi, energy,
charge,
isChargedHadron, isNeutralHadron, isPhoton, isElectron, isMuon,
d0, d0err, dz, dzerr
```

See `LOCAL_COMPRESSION_RAW_FEATURE_NAMES` in:

- `teacher_logit_reco/local_compression_part/config.py`

These raw tokens are converted into the exact canonical ParT input tensors by:

- `teacher_logit_reco/local_compression_part/features.py`
  - `prepare_local_compression_tokens_and_mask`
  - `build_local_compression_canonical_inputs`

That code reuses the shared ParT feature builder:

```python
build_part_inputs_torch(...)
```

from `jetclass_fresh.dual_view`.

The canonical ParT inputs are:

```text
points          # ParT point/geometry tensor
features        # PF feature tensor used by ParT
lorentz_vectors # ParT Lorentz-vector tensor
mask            # ParT particle mask
```

The adapter edits only the `features` tensor. It keeps `points`, `lorentz_vectors`, and `mask` unchanged.

The relevant helper is:

- `LocalCompressionCanonicalInputs.with_features(...)` in `teacher_logit_reco/local_compression_part/features.py`.

That method returns a new canonical input object with replaced/adapted feature rows while preserving the original points, Lorentz vectors, selected tokens, and masks.

## What Makes `lc_mlp_delta` Different From Full Local Compression?

The full local-compression variants can use:

```text
semantic modality subtokens
  -> local modality transformer/compressor
  -> learned pooling
  -> particle context block
  -> context-aware gates
  -> delta_F adapter
```

`lc_mlp_delta` disables the modality/local/context/gating parts as a scientific control.

The variant behavior is defined in:

- `teacher_logit_reco/local_compression_part/config.py`
  - `LocalCompressionVariantConfig.__post_init__`
  - for `lc_mlp_delta`, expected flags are:

```text
use_modalities = False
use_local_compressor = False
use_particle_context = False
use_context_gates = False
```

The model wrapper enforces the runtime behavior in:

- `teacher_logit_reco/local_compression_part/model.py`
  - `variant_uses_modalities`
  - `variant_uses_local_compressor`
  - `variant_uses_particle_context`
  - `variant_uses_context_gates`
  - `forward_outputs`

For `lc_mlp_delta`:

1. It still builds canonical ParT inputs.
2. It still builds modality objects for shape/diagnostic consistency, but the actual model path does not use local modality compression.
3. It creates a synthetic single-token compressor output from the canonical feature anchor.
4. It uses identity particle context.
5. It uses identity/no-op gates.
6. It predicts `delta_F` through the MLP adapter.
7. It feeds adapted features into the exact HLT ParT backbone.

The relevant code path is:

- `teacher_logit_reco/local_compression_part/model.py`
  - `_single_token_compressor_output`
  - `_identity_context_output`
  - `_identity_gate_output`
  - `forward_outputs`

This is why `lc_mlp_delta` should be understood as:

```text
canonical feature anchor + MLP delta adapter
```

not as:

```text
modality subtoken local-compression transformer
```

## The Delta-F Adapter

The actual residual feature repair module is:

- `LocalCompressionDeltaFAdapter` in `teacher_logit_reco/local_compression_part/adapter.py`.

Its job is to predict a bounded per-particle correction:

```text
delta_F_rows: [batch, particles, feature_dim]
```

where `feature_dim` is the canonical ParT PF feature dimension.

For every active particle:

```text
adapted_feature_rows = feature_rows + delta_F_rows
```

For invalid/padded particles:

```text
delta_F_rows = 0
```

The adapter input is a concatenation of:

```text
canonical feature_rows
local_particle_token
context_tokens
gate_weighted_token
gates
```

For the `lc_mlp_delta` variant, those token/gate objects are mostly identity/synthetic objects derived from the feature anchor, because modalities/context/gates are disabled. The key signal is the canonical feature row itself.

The adapter MLP is built in `adapter.py`:

```python
LayerNorm(input_dim)
Linear(input_dim, hidden_dim)
GELU
Dropout
Linear(hidden_dim, hidden_dim)
GELU
Dropout
Linear(hidden_dim, feature_dim)
```

The final linear layer is zero-initialized when `zero_init_delta_projection=True`. That means the model initially recovers the baseline ParT behavior exactly or nearly exactly:

```text
delta_F = 0
adapted features = original features
logits = baseline ParT logits
```

This is a major part of the design: the adapter starts as a near-identity residual correction rather than a replacement encoder.

## Delta Bounds and Feature-Specific Scales

The raw MLP output is not added directly. It is bounded:

```text
bounded_delta = tanh(raw_delta)
delta_F = bounded_delta * feature_delta_scale * feature_active_mask
```

See:

- `LocalCompressionDeltaFAdapter.forward` in `teacher_logit_reco/local_compression_part/adapter.py`.

Default feature scales are configured in:

- `default_feature_delta_scales()` in `teacher_logit_reco/local_compression_part/config.py`.

Examples:

```text
part_pt_log      0.05
part_e_log       0.05
part_logptrel    0.05
part_logerel     0.05
part_deltaR      0.025
PID flags        0.02
track d0/dz      0.20
track errors     0.20
part_deta/dphi   0.025
```

So `lc_mlp_delta` is deliberately conservative. It is allowed to nudge the canonical PF features, not freely rewrite the event.

Two additional guardrails exist:

- `freeze_pid_deltas`: optionally force PID-related feature deltas to zero.
- `freeze_geometry_deltas`: optionally force geometry-related feature deltas to zero.

These are wired through:

- `LocalCompressionPartConfig`
- `LocalCompressionTaggerTrainConfig`
- `LocalCompressionDeltaFAdapter`

## What Stays Unchanged?

For `lc_mlp_delta`, the following are not replaced:

```text
HLT raw tokens
particle ordering
particle mask
ParT points
ParT Lorentz vectors
exact HLT ParT backbone class
2-logit classification head shape
QCD/Hgg label mapping
checkpoint selection metric
```

Only the canonical PF feature rows are adapted:

```text
features -> features + delta_F
```

The final classifier remains the real HLT ParT wrapper:

- `ParticleTransformerHLTClassifier`

constructed through:

- `build_hlt_classifier(...)`

inside:

- `teacher_logit_reco/local_compression_part/model.py`

The model explicitly rejects a non-`ParticleTransformerHLTClassifier` backbone.

## Forward Pass in Code

The main forward pass lives in:

- `teacher_logit_reco/local_compression_part/model.py`
  - `LocalCompressionFeatureAdapterParT.forward_outputs`

The flow is:

```python
raw_tokens, raw_mask = _coerce_tokens_and_mask(...)

canonical = self.build_canonical_inputs(raw_tokens, raw_mask, ...)

modalities = build_local_compression_modalities(canonical, ...)
subtoken_output = self.subtoken_encoder(canonical, modalities)

if self.variant_uses_local_compressor:
    compressor_output = self.compressor(subtoken_output)
    pool_output = self.pooler(compressor_output)
else:
    compressor_output = self._single_token_compressor_output(canonical, ...)
    pool_output = self.pooler(compressor_output)

if self.variant_uses_particle_context:
    context_output = self.context_block(pool_output)
else:
    context_output = self._identity_context_output(pool_output)

if self.variant_uses_context_gates:
    gate_output = self.gate(compressor_output, context_output, canonical)
else:
    gate_output = self._identity_gate_output(...)

if self.variant_forces_zero_delta:
    delta_output = self._zero_delta_output(...)
else:
    delta_output = self.adapter(canonical, pool_output, context_output, gate_output)

adapted = canonical.with_features(delta_output.adapted_feature_rows)

logits = self.part_model(
    adapted.points,
    adapted.features,
    adapted.lorentz_vectors,
    adapted.mask,
)
```

For `lc_mlp_delta`, the `variant_uses_*` flags are all false except the general adapter path. So the important learned operation is the adapter MLP predicting `delta_F`.

## Training Objective

Training uses two-logit cross entropy:

```text
CrossEntropyLoss(logits, labels)
```

The implementation also adds a delta regularizer:

```text
loss = CE + delta_l2_weight * mean_per_particle(||delta_F||_2^2)
```

See:

- `run_local_compression_tagger_epoch` in `teacher_logit_reco/local_compression_part/train.py`.

The default delta L2 weight is configured in:

- `LocalCompressionFeatureConfig.delta_l2_weight`

The model is warm-started from the baseline HLT ParT checkpoint before training:

- `load_baseline_checkpoint_into_part_model(...)` in `teacher_logit_reco/local_compression_part/train.py`
- checkpoint metadata validation in `teacher_logit_reco/local_compression_part/checkpoint.py`

Training uses separate learning rates:

```text
adapter_lr: applied to non-ParT adapter parameters
part_lr:    applied to ParT backbone parameters when unfrozen
```

The code groups optimizer parameters in:

- `_optimizer_for_model(...)` in `teacher_logit_reco/local_compression_part/train.py`.

The current serious default freezes the ParT body for the first epoch:

```text
freeze_part_epochs = 1
```

That gives the adapter an initial chance to learn as a front-end repair layer before the ParT backbone is fine-tuned.

## What Does the Baseline Recheck Do?

The baseline recheck variant is:

```text
hlt_part_baseline_recheck
```

It is similar infrastructure but forces:

```text
delta_F = 0
```

and evaluates the loaded baseline checkpoint. It is meant to prove that the local-compression wrapper can recover the original HLT ParT baseline when the adapter does nothing.

The forced-zero-delta path is:

- `variant_forces_zero_delta`
- `_zero_delta_output(...)`

in `teacher_logit_reco/local_compression_part/model.py`.

This makes the comparison fair:

```text
baseline recheck: same wrapper, same data, same exact ParT, delta_F=0
lc_mlp_delta:     same wrapper, same data, same exact ParT, learned bounded delta_F
```

## How It Is Run

The single-variant CLI is:

- `scripts/train_local_compression_part_tagger.py`

The multi-variant runner is:

- `scripts/train_local_compression_part_variants.py`

The relevant argument is:

```bash
--variant lc_mlp_delta
```

or, for the variant suite:

```bash
--variants hlt_part_baseline_recheck lc_mlp_delta ...
```

The Slurm submitters include `lc_mlp_delta` in the default local-compression variant list:

- `sbatch/submit_local_compression_part_qcd_hgg_hlt0p6_experiment.sh`
- `sbatch/submit_local_compression_part_qcd_hgg_hlt0p6_pilot_and_highdata.sh`

## Diagnostics To Inspect

The training code records delta diagnostics when diagnostics are collected:

- `delta_F_l2_mean`
- `delta_F_abs_mean`
- `delta_F_abs_p90`
- `delta_F_abs_max`
- per-feature absolute mean/p90/max
- per-feature squared mean
- per-feature RMS
- final delta projection weight/bias norms

See:

- `_delta_diagnostics_from_output(...)` in `teacher_logit_reco/local_compression_part/train.py`.

These diagnostics are crucial for interpreting whether `lc_mlp_delta` is learning a physically meaningful feature repair or just exploiting extra capacity.

Useful questions:

```text
Which canonical PF features move the most?
Are deltas stable across random seeds?
Are deltas larger for low-quality or heavily degraded particles?
Do track/displacement channels move more than identity channels?
Do PID-like one-hot channels get softened?
Are geometry deltas small enough to avoid inconsistent physics inputs?
```

## Why This Variant Matters

`lc_mlp_delta` was originally a control. It was meant to answer:

```text
Does local compression help beyond a simple feature-delta MLP?
```

But if `lc_mlp_delta` improves over the HLT ParT baseline, the result is scientifically meaningful by itself.

It suggests that the bottleneck may not be ParT's ability to model particle-particle interactions. Instead, the bottleneck may be the input feature interface:

```text
HLT features are not quite in the coordinate system ParT wants.
```

In that interpretation, `lc_mlp_delta` acts as a learned, task-aware feature repair/calibration layer:

```text
HLT particle measurements
  -> small bounded residual repair
  -> better-conditioned ParT feature rows
  -> ParT attention and classifier
```

This is not the same as adding a larger classifier head after ParT. The repair happens before every ParT layer, so a small correction can change the representation that all downstream attention blocks consume.

## What It Does Not Prove By Itself

A gain from `lc_mlp_delta` does not automatically prove:

```text
local compression helped
the learned deltas are physically meaningful
the adapter is repairing HLT degradation rather than adding capacity
the same idea will improve Offline ParT
```

Those require controls.

Recommended controls:

```text
adapter-only with frozen ParT
fine-tuned ParT without adapter
same-parameter larger ParT
post-embedding adapter instead of raw-feature adapter
random/shuffled delta controls
label-shuffled adapter training
PID/geometry frozen-delta ablations
multiple seeds
feature-delta diagnostics by feature and pT/particle-type bins
Offline teacher distillation if paired or teacher outputs are available
```

## Best One-Sentence Explanation

`lc_mlp_delta` learns a small bounded residual correction to each particle's canonical ParT feature vector, then feeds those repaired features into the exact warm-started HLT ParT, letting the transformer reason over slightly better input tokens without replacing the ParT backbone.

