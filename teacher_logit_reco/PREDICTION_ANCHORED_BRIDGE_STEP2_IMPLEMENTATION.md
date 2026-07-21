# Prediction-Anchored Bridge Step 2 implementation

Step 2 is implemented by the following surfaces:

- `local_particle_residual_field/bridge_ram.py` owns the single-node allocation
  ledger, one-open compressed-NPZ staging, immutable raw shards, rank ranges,
  derived LRU, streamed truth, frozen-R0 adapter, and R0 registration.
- `local_particle_residual_field/bridge.py` owns physical-45/all-50 virtual
  recipes, response points, matched controls, and immutable physical-space
  scalers.
- `local_particle_residual_field/bridge_r0.py` trains the ordinary HLT-only R0
  from streamed targets and publishes one weights-only checkpoint.
- `scripts/audit_prediction_anchored_bridge_inputs.py` performs the production
  low-storage smoke and publishes only two recipes, two scaler contracts, and
  one metrics contract.
- `scripts/register_prediction_anchored_r0.py` registers an existing R0;
  `scripts/write_prediction_anchored_bridge_recipe.py` writes an individual
  recipe. All three commands support `--dry-run`.

The production lifecycle is deliberately allocation-scoped:

1. Request exactly one node and choose a verified `tmpfs`/`ramfs` root.
2. Rank zero opens, hashes, verifies, and decompresses each HLT/offline NPZ once.
3. Every rank reads its deterministic range from non-evictable 8,192-event raw
   shards. Only derived fields may be evicted and regenerated.
4. Close rank-local providers, publish only allowlisted small artifacts, and
   clean the owned allocation RAM tree.

An audit invocation has this shape:

```bash
python scripts/audit_prediction_anchored_bridge_inputs.py \
  --hlt-npz HLT.npz --hlt-metadata HLT.json \
  --offline-npz OFFLINE.npz --offline-metadata OFFLINE.json \
  --r0-checkpoint r0_weights.pt \
  --split-manifest-sha256 SHA256 \
  --ram-root /dev/shm --allocation-id "$SLURM_JOB_ID" \
  --output-dir bridge_step2_audit
```

The command fails on multiple nodes, disk-backed production staging, source or
event-order changes, duplicate identities, mismatched labels/units, non-finite
data, RAM oversubscription, or any unexpected persistent output. Generated
`f_true`, `f0`, `h0`, bridge, and control tensors never enter the persistent
output directory.
