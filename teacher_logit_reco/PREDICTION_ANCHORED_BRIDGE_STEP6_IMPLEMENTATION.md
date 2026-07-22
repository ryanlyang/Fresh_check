# Prediction-Anchored Bridge Step 6 implementation

Step 6 implements only the non-hierarchical correction architectures declared
for this increment. Full region/global HLG models and the direct HLG classifiers
remain `UNMEASURED` until Step 7.

The implementation lives in
`local_particle_residual_field/hierarchical_reconstructor.py` and provides:

- the exact 32-neighbor, non-self, directed `DeltaR <= 0.30` HLT graph;
- the six locked source-minus-target relative edge features;
- independent 0.02/0.05/0.10 Gaussian or hard-radius weights;
- two layer-specific `M_l/U_l` pairs shared across radii within each layer;
- three persistent 160-wide radius streams and three separately routed
  `480 -> 160 -> 128 -> 64 -> 15` zero-initialized heads;
- a common particle reasoning interface whose region fields are explicitly
  absent and whose pre-global readback is the base particle state;
- `D10_A0M_capacity_particle`, a six-layer single-stream particle-message
  control with two 160→288→160 capacity blocks and no region/global path;
- executed-forward parameter/FLOP accounting and an analytical reference made
  directly from the locked Section 12.7 A3 components;
- C0-plus-Step-6 serialized-state measurement and strict tiny train/reload
  validation.

At production width 128, the locked A0M profile is approximately 4.2% above
the analytical A3 trainable-parameter reference and 0.24% above its correction
FLOPs. This satisfies the predeclared ±5% and ±10% tolerances. Step 7 must
compare its eventual concrete A3 implementation against the same reference;
the A0M dimensions may not be resized afterward.

Operator commands:

```bash
python scripts/measure_prediction_anchored_bridge_step6.py \
  --mode plan --scaler bridge_scalers_physical45.json \
  --particle-width 128 --dry-run

python scripts/measure_prediction_anchored_bridge_step6.py \
  --mode measure --scaler bridge_scalers_physical45.json \
  --registry campaign_registry_step5.json --parent-manifest split_manifest.json \
  --output-dir step6_measurement
```

The measurement command writes only the re-hashed registry and compact
measurement JSON. It serializes representative weights in memory to obtain
their exact byte counts; it does not publish a model checkpoint, optimizer,
generated bridge field, or dense diagnostic tensor.

Production measurement derives `max_constits` from the hashed source manifest.
`--particle-width` exists only for `--dry-run` CPU checks and cannot create a
persistent registry measurement.
