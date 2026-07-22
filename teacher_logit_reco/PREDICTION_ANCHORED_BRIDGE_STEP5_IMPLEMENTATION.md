# Prediction-Anchored Bridge Step 5 implementation

Step 5 implements the simple C0 particle residual predictor and the complete
eleven-run loss sweep. The implementation is split across these surfaces:

- `local_particle_residual_field/bridge_reconstructor.py` owns the
  `f0+h0+raw-HLT` C0 model, immutable numerical-space declarations,
  zero-initialized radius heads, componentwise physical trust bounds, exact
  five-channel pass-through, and the frozen-but-input-differentiable T10
  wrapper.
- `local_particle_residual_field/bridge_losses.py` owns the corrected L0-L10
  recipes, twelve-group standardized Huber loss, exact directed Gaussian
  smoothness graph, KD/CE/anchor/weak-truth terms, and L0 reachability metrics.
- `local_particle_residual_field/bridge_reconstructor_train.py` owns the fixed
  Phase 1 budget, `model_val_stop` Phase 2 selection, the L0 exception,
  same-allocation RAM resume, fail-closed teacher lineage, serialized-size
  measurement, paired aggregation, and median-only publication.
- `scripts/train_prediction_anchored_bridge_reconstructor.py` writes the launch
  plan, measures all C0 registry rows, and publishes a three-seed result.

The plan command makes the dependency boundary explicit. `D10_L0_bridge_only`
is a B3 job and launches in parallel with consumer training without a teacher.
The remaining ten configurations are B6 jobs and cannot enter training unless
the confirmed `selected_bridge_consumer.json` and its immutable live-teacher
configuration are supplied. Nonzero-KD runs additionally require the primary
physical-45 target cache. The training boundary verifies that selection,
binding, cache, and live graph all carry the same checkpoint SHA-256.

Typical validation commands are:

```bash
python scripts/train_prediction_anchored_bridge_reconstructor.py \
  --mode plan --scaler bridge_scalers_physical45.json --dry-run

python scripts/train_prediction_anchored_bridge_reconstructor.py \
  --mode measure --scaler bridge_scalers_physical45.json \
  --registry campaign_registry.json --output-dir step5_measurement

python scripts/train_prediction_anchored_bridge_reconstructor.py \
  --mode publish --run-id D10_L8_full_c0 \
  --replica-dir job_local_replica_exports --output-dir D10_L8_full_c0
```

Production model/data loading remains a Python runner callback, as it is for
the Step 3 consumer trainer. Construct `PredictionAnchoredC0Correction`, the
exact `FrozenLiveBridgeConsumer`, streamed RAM batch factories, and call
`train_c0_replica`. This keeps experiment-specific loaders out of the contract
layer while still enforcing the selected-teacher and split-access invariants at
the training boundary. Rotating snapshots remain in the current allocation;
after success, only all metrics and the ordered-median weights are published.
