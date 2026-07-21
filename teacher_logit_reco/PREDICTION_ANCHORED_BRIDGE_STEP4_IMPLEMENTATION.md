# Prediction-Anchored Bridge Step 4 implementation

Step 4 is implemented by two dependency-light modules:

- `local_particle_residual_field/bridge_evaluation.py` computes ten-class
  discrimination/calibration metrics, response and control summaries, slices,
  deterministic event-paired bootstrap intervals, all eight Section 18.1
  gates, ordered-median selection, sealed confirmation, and immutable teacher
  bindings.
- `local_particle_residual_field/bridge_logits.py` binds target and live sides
  to one exact checkpoint and creates/validates detached target-logit caches in
  four non-interchangeable namespaces: primary physical-45, all-50,
  alternate-teacher, and the N3 selected-teacher-on-`f0` control.

The production order is intentionally acyclic:

1. Evaluate each `model_val_stop`-selected replica once on
   `model_val_select` with the same checkpoint at every response/control point.
2. Aggregate seeds 101/202/303 and write a pre-confirmation selection. The
   selector never ranks an individual seed and retains the fixed ordered median.
3. Use the split-access contract to unlock and claim `stack_val_consumer` once.
   A failed confirmation writes `stopped_campaign.json`; there is no runner-up
   or refit.
4. Write the exact primary/all-50/alternate `teacher_binding_v1` before any
   logit cache. Bindings contain no cache hash.
5. Within the allocation that has loaded the bound checkpoint, call
   `cache_bound_teacher_logits`. The cache manifest records the binding,
   checkpoint, recipe, channel, event order, class order, temperature, and
   exact `stack_train_distill` child identity. Only logits, labels, and hashed
   event identities persist.
6. Load the same binding with `build_live_teacher_config`. The frozen live
   consumer keeps gradients enabled with respect to its physical-field input.
7. Before packaging, call `verify_teacher_identity_chain` so selection,
   binding, target cache, live graph, and bundle configuration all carry the
   identical teacher checkpoint SHA-256.

Operator commands:

```bash
python scripts/select_prediction_anchored_bridge_consumer.py select --help
python scripts/select_prediction_anchored_bridge_consumer.py confirm --help
python scripts/select_prediction_anchored_bridge_consumer.py bind --help
python scripts/validate_prediction_anchored_teacher_logits.py --help
```

`cache_bound_teacher_logits` deliberately remains a Python runner API rather
than accepting an arbitrary precomputed logits file from a command line. It
hashes the checkpoint bytes and performs the forward callback itself, which is
what makes the target/live checkpoint-identity claim meaningful.
