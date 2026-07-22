# Prediction-Anchored Bridge Step 8 implementation

Step 8 is implemented in
`local_particle_residual_field/bridge_semantic_evidence.py`. It is the semantic
and evidence layer around the canonical Step 7 A3 graph; it does not define a
second primary HLG architecture.

## A3 loss-interaction block

`Step8RunRecipe` locks the eight declared A3 supervision/schedule rows. The
primary alias `D10_XA3_full_primary` resolves to the already registered
`D10_A3_hlg_primary` row, so it does not create a second checkpoint or job.
The seven additional canonical rows retain their own recipe hashes. KD-only,
CE-only, KD+CE, and no-warm-up skip Phase 1. Every warm-up-enabled recipe runs
its exact configured `field_warmup_steps`; validation cannot shorten or extend
that phase, and a non-finite Phase 1 loss/gradient stops the replica.

`train_step8_replica` is the allocation-local optimizer loop. It accepts the
campaign runner's ordered RAM batches, freezes every live-consumer parameter,
keeps the field-input gradient active, refuses target logits for a zero-KD
recipe, and writes no batch, bridge, control, or optimizer tensor. N0–N3 are
prepared in RAM before entering the same loop, so their model and optimizer
budgets can be checked against their exact positive counterpart.

## All-50 semantics and teacher lineage

`PredictionAnchoredAll50HLG` wraps the exact canonical A3 model:

- B1 adds only a separate zero-initialized `160 -> 64 -> 5` continuous head;
- B2 has no five-channel head and preserves the exact `f0[S]` pass-through;
- both use the all-50 teacher/cache and are permanently non-selectable;
- the physical 45-channel graph/config is identical to canonical A3;
- B1 applies the all-50 correction scales/trust bounds and a 13-group
  standardized Huber objective (the twelve physical groups plus one
  reliability-five group);
- B1's equal-full-field KD invariant is executable, while B2 records its
  intentional irreducible reliability mismatch.

`build_teacher_binding` now optionally embeds the all-50 scaler hash and only
the five extra correction statistics. Step 8 requires those statistics to be
fit on `stack_train_distill`. The all-50 cache repeats that scaler hash, so the
binding, detached targets, live graph, and B1/B2 head cannot silently use
different scaling.

`validate_step8_teacher_lineage` enforces the acyclic
binding -> cache -> live-run chain. Primary, all-50, alternate, and N3 cache
namespaces are not interchangeable. N3 must have `field_condition=f0` and
`rho_endpoint=0`; all privileged KD runs must have
`field_condition=bridge_0.100`. The conditional TALT recipe remains
non-selectable and follows the registry's `SKIPPED_INVALID_PARENT` state.

## Trained negative controls

The four controls are separate hashed recipes:

- N0: KD-only, target logits permuted by the matched wrong-event map;
- N1: bridge-only, bridge corrections permuted by that map and rescaled to
  preserve each of the twelve groups' allocation-level marginal L2 scale;
- N2: primary objective, using the same wrong-event map for logits and bridge;
- N3: KD-only against the dedicated selected-teacher-on-`f0` cache.

The transformations are allocation-local. Their artifacts contain only the
recipe/map/cache hashes and explicitly forbid persistent dense fields.
`validate_four_control_matching` requires the same canonical A3 config, paired
seeds `{101,202,303}`, and optimizer-step budget as each positive run.

## Adversarial-channel and physical evidence

The small-field audit uses exactly seeds `{9101,9102,9103,9104}`. Its standard
normal is keyed by audit seed, event identity, valid particle index, and
physical channel, making it invariant to event reordering. It implements the
declared `0.05*sigma_delta` perturbation and `0.10*trust_scale` clip, copies the
five reliability coordinates exactly, keeps padding zero, retains negative
accuracy loss as an improvement, and enforces mean/worst losses of 0.002/0.003
only for selectable rows.

Bridge alignment reports overall and twelve groupwise finite cosine values.
The distribution reference contains exactly a `1001 x 45` float64 quantile
table from `stack_train_distill`; validation recomputes the same levels and
reports per-channel and groupwise approximate 1-D Wasserstein distances. It
cannot be evaluated on `final_test`. Neither alignment nor distribution has a
hidden numerical selection cutoff. The combined adversarial report applies
only the declared perturbation, 1% trust-saturation, and reliability
pass-through gates. Gain/recovery uses sign-correct loss improvement and writes
`recovery_fraction=null` whenever teacher bridge gain is non-positive.

## Final measured preflight

`measure_step8_registry_states` requires the eight actual upstream Step 3
measurements. It can inherit a fully measured Step 7 registry or instantiate
and measure the remaining Step 4–7 rows from a Step 3 registry. It then
serializes representative weights in RAM for all 14 Step 8 canonical rows,
including the conditional TALT state, and verifies:

- generated registry counts remain 54 / 46 / 45;
- the canonical A3 alias resolves to the same registry row;
- every runnable row is `MEASURED`;
- target-cache fixed bytes match the conditional alternate state;
- the complete measured projection fits the selected locked 5 or 6 GiB mode.

The preflight rejects partial upstream measurements, an alternate-cache/state
mismatch, any runnable `UNMEASURED` row, and any budget overrun. It never
persists representative states. `require_post_teacher_release` consults only a
confirmed scientifically valid teacher binding; C0 success is not an argument
and cannot block HLG.

The CPU miniature compares canonical HLG, the particle-only capacity control,
raw-HLT direct HLG, and `R0`-representation direct HLG for all three paired
seeds. It is explicitly a tensor-path/aggregation rehearsal and cannot produce
scientific results.

## Operator command

Plan and inspect every recipe/head without writing:

```bash
python scripts/measure_prediction_anchored_bridge_step8.py \
  --mode plan \
  --physical45-scaler bridge_scalers_physical45.json \
  --all50-scaler bridge_scalers_all50.json \
  --absolute-scaler absolute_output_scaler.json \
  --deployed-resource-reference canonical_a3_bundle_resources.json \
  --particle-width 128 --dry-run
```

Execute the final measured preflight and publish only the registry/report:

```bash
python scripts/measure_prediction_anchored_bridge_fixed_storage.py \
  --child-split-manifest prediction_anchored_child_splits.json \
  --r0-weights r0/median_weights.pt \
  --target-namespace physical45_selected_bridge_teacher=teacher_logits/physical45_selected_bridge_teacher \
  --target-namespace all50_selected_bridge_teacher=teacher_logits/all50_selected_bridge_teacher \
  --target-namespace physical45_selected_teacher_on_f0_control=teacher_logits/physical45_selected_teacher_on_f0_control \
  --metadata-path recipes_bindings_reports \
  --final-deployable-bundle representative_final_bundle.pt \
  --output-dir fixed_storage_measurement

python scripts/measure_prediction_anchored_bridge_step8.py \
  --mode measure \
  --physical45-scaler bridge_scalers_physical45.json \
  --all50-scaler bridge_scalers_all50.json \
  --absolute-scaler absolute_output_scaler.json \
  --deployed-resource-reference canonical_a3_bundle_resources.json \
  --registry campaign_registry_step7.json \
  --fixed-storage measured_fixed_storage.json \
  --parent-manifest split_manifest.json \
  --budget-mode 5gib \
  --output-dir step8_measurement
```

The fixed-storage artifact must contain measured byte counts for the child
manifest, R0, each materialized target-logit namespace, bounded
recipes/bindings/reports, and the final deployable bundle. It must contain no
dense field reservation.
