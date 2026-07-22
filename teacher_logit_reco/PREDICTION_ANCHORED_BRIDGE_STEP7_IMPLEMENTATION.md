# Prediction-Anchored Bridge Step 7 implementation

Step 7 is implemented in
`local_particle_residual_field/hierarchical_global_reconstructor.py`. It extends
the Step 6 particle interface; it does not replace or duplicate the locked
local graph implementation.

The canonical `D10_A3_hlg_primary` path is:

```text
raw HLT + physical f0 + stop-gradient h0
  -> locked 160-wide base fusion
  -> one HLT-only capped graph
  -> three persistent Gaussian local streams
  -> deterministic HLT seeds and learned scale-wise assignment
  -> exact 389-value region pool (4/4/2 regions)
  -> two pre-norm, four-head, 160-wide region-transformer layers
     with the existing 19 HLT-only pair features
  -> one exact residual region-to-particle readback
  -> three zero-initialized 15-channel bounded correction heads
  -> physical45 correction + exact f0 reliability-five pass-through
```

The region mass in `log1p(mass)` and the empty-token threshold use the existing
scale-wise `cluster_weights`. The existing per-region `assignment_weights` are
attention distributions normalized over particles and therefore always sum to
one for a live token; using them as “mass” would make the locked empty-region
contract meaningless. Both tensors are retained for pair features and audits,
and this choice is part of the hashed HLG configuration.

Implemented field variants are A2, A3, A4, A5, A5S, A6, A7, A7F, A7X, A8,
A9, AS, AL, AFIX, ASAME, and AGLOBAL. In particular:

- A4 performs exactly one extra repool/transformer/readback pass and reuses the
  original seeds and assignment tensor;
- A5/A5S use immutable q0.001/q0.999 physical45 bounds fitted only on
  `stack_train_distill`; their zero heads emit the channel centers exactly;
- A5S physically omits f0/h0 projection and pooling branches. `f0[P]` and `h0`
  cannot even be passed to its predictor API: it accepts only the five
  reliability pass-through channels with `h0=None`, and copies those channels
  outside the raw-HLT trainable path;
- A9 has exactly twelve radius-by-semantic gates. They initialize at 0.95,
  remain frozen through field warm-up, unfreeze at Phase 2, act after the trust
  bound, and alone receive coefficient 0.005;
- AFIX uses deterministic nearest valid HLT seed assignment within each scale;
- ASAME inserts the exact `-1e4` cross-scale attention mask;
- AGLOBAL retains one mass-averaged global token only after the ten region
  tokens have completed global reasoning.

The two direct controls are full HLG classifiers rather than field models.
`A0_CAP500_direct_hlt` accepts only raw HLT tokens. `A0_CAP500_r0rep_direct`
accepts frozen HLT-only R0 `f0/h0`. Both train with CE only on the declared
500k `stack_train_consumer union stack_train_distill` manifest, contain no
field head, KD path, bridge target, or T10, and are sized once against an
immutable measured canonical `R0 + A3 + selected T10` resource reference. The
builder refuses to continue unless each control is within +/-5% parameters and
+/-10% batch-1 inference FLOPs. This reference is canonical A3, not whichever
reconstructor is eventually selected.

`measure_step7_registry_states` first reproduces the Step 6 measurements, then
records exact in-RAM serialized weight sizes and executed resource profiles for
all 16 hierarchy rows and both direct rows. Step 8 loss/semantic/control rows
remain `UNMEASURED`. Representative serialization is used only to count bytes;
no representative checkpoint, optimizer state, generated field, or dense
diagnostic tensor is published.

Operator commands:

```bash
python scripts/measure_prediction_anchored_bridge_step7.py \
  --mode plan \
  --scaler bridge_scalers_physical45.json \
  --absolute-scaler absolute_output_scaler.json \
  --deployed-resource-reference canonical_a3_bundle_resources.json \
  --particle-width 128 --dry-run

python scripts/measure_prediction_anchored_bridge_step7.py \
  --mode measure \
  --scaler bridge_scalers_physical45.json \
  --absolute-scaler absolute_output_scaler.json \
  --deployed-resource-reference canonical_a3_bundle_resources.json \
  --registry campaign_registry_step6.json \
  --parent-manifest split_manifest.json \
  --output-dir step7_measurement
```

Persistent measurement is fail-closed on source-manifest width/hash,
absolute-scaler provenance, the deployed-reference hash, registry state, A0M
capacity tolerance, and both direct-control capacity tolerances. Explicit
particle width is accepted only for a non-persistent dry run.

The Step 7 test suite covers the hand-computed 389-value pool, masks and empty
jets, pair/readback shapes and equation, all variant factors, absolute and gate
semantics, finite gradients for every hierarchy row, direct 500k/CE contracts,
strict train/reload, measured registry state, and the complete three-seed CPU
matrix. Miniature results are marked non-scientific and cannot enter a selector.
