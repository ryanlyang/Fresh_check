# Prediction-Anchored Bridge Step 9 implementation

Step 9 is implemented in
`local_particle_residual_field/bridge_campaign_policy.py`. It is the immutable
campaign-decision and deployment layer above the Step 1--8 model/evidence
code. It does not retrain or redefine any reconstructor.

## B0--B6 campaign policy

`build_campaign_reservations` consumes the final measured registry and the
passed 5/6 GiB preflight. Every runnable row receives its measured retained
byte reservation; a conditional skip receives zero. Frozen parent hashes must
be unique and their paths must not overlap, preventing double charging or a
second published parent copy. An over-quota projection fails before B0.

The versioned campaign state contains all 54 canonical rows throughout its
lifetime. `build_stage_evidence` requires the declared boolean gates and
explicit immutable parent hashes for each stage:

- B0: data/split/mask/unit, RAM, storage, and R0-prerequisite audits;
- B1: registered R0 plus valid live `f0/h0`;
- B2: physical-45/all-50 recipes and the live batch audit;
- B3: independent L0 launch and completed paired upstream/consumer block;
- B4: aggregate-only consumer selection and one-shot consumer confirmation;
- B5: the selected consumer, bindings, and separate primary/all-50/N3 caches;
- B6: the valid-teacher release and completed paired reconstruction breadth.

Stages advance strictly in order. A failed stage stops the campaign, marks its
descendants `SKIPPED_FAILED_PARENT`, and cannot be resumed or replaced by a
fallback. `D10_TALT_A3` stays visibly `SKIPPED_INVALID_PARENT` when its parent
is invalid and cannot be promoted by an outcome update. B6 release remains
based on the Step 8 teacher gate, not C0 success.

## Aggregate deployment selection and sealed confirmation

`DeployableReplicaEvidence` records each paired seed's absolute metrics,
gain/recovery, checkpoint epoch and hashes, exact deployed parameter count,
storage use, audit flags, trust saturation, and four-seed perturbation result.
Its constructor recomputes gain/recovery and requires `null` recovery for a
non-positive teacher gain.

`aggregate_deployable_configuration` requires exactly seeds 101/202/303 and
applies every Section 18.2 gate. It chooses the retained seed by the locked
ascending tuple `(accuracy, deployable_gain, -cross_entropy, seed_id)` and
does not retain the lucky best replica.

`select_deployable_preconfirmation` first filters to valid registry rows. It
forms one order-independent 0.0005 accuracy tie pool, then uses mean macro
accuracy, mean CE, gain standard deviation, exact deployed parameter count,
and lexical ID. Latency is diagnostic only. The hashed pre-confirmation embeds
the selected aggregate and fixes the seed, epoch, checkpoint, scaler, teacher,
and recipe before `stack_val_deploy` is unlocked.

The split helper issues the existing sealed one-shot access receipt.
Confirmation requires non-negative gain, finite metrics, and valid provenance.
Failure returns a terminal stopped artifact with `runner_up_promoted=false`.
Success writes a locked deployable manifest. Final-test remains sealed until a
clean reload audit passes, and any bridge/oracle/offline/privileged evaluation
flag is rejected.

## Reports and retained-state publication

`build_step9_reports` creates three physically separate row collections:

1. baselines/deployable, including `Tpred_continue(f0)`, selected `T10(f0)`,
   and selected `T10(f_hat)`;
2. watermarked non-deployable diagnostics, with separate physical-45 and
   full-50 oracle ceiling rows;
3. all 54 ablation rows, including missing, failed, and conditional-skipped
   states.

Completed rows require all three seed metrics, aggregate statistics, and the
retained median ID. Every A0 row additionally requires its training-manifest
hash, unique jet count, and optimizer-step budget. Every deployable row must
have passed HLT-only reload. Recovery remains null for a non-positive teacher
gain. Persistent and RAM byte telemetry are mandatory. Oracle ceiling rows
cannot enter the deployable section.

`build_median_publication_manifest` retains all three metrics but only the
ordered-median checkpoint reference per configuration. It explicitly records
that optimizer state, non-median weights, and duplicate frozen parents are not
published. Existing Step 3/5 publication functions perform the corresponding
weights-only writes.

## Locked HLT-only bundle

`PredictionAnchoredDeployableBundle` is the single inference entry point:

```text
HLT batch -> embedded frozen R0 -> selected correction -> embedded frozen T10
```

Its only external argument is a mapping of trigger-available tokens, masks,
and ordinary ParT inputs. The internally generated field is passed through the
tagger's direct `residual_fields` input. No oracle loader, cached target logit,
bridge tensor, offline matching input, or privileged path is reachable.

The bundle manifest binds the confirmed run/median/epoch and the exact R0,
correction, and selected-T10 hashes. Component hashes must be distinct; the
correction hash must equal the locked median and the consumer hash must equal
the selected teacher. Preprocessing, residual normalization, target schema,
class order, and architecture metadata are embedded, while external
oracle/offline/target-cache dependency keys are refused.

The manifest also binds the measured final-bundle byte reservation. Export
serializes to memory first and refuses an over-reservation payload before any
filesystem publication.

Export writes one weights-only checkpoint with one state namespace per
component. `clean_hlt_only_reload_audit` copies that file into a temporary
environment, requires every declared privileged source path to be absent,
reloads through the same HLT-only entry point, and checks tolerance-equivalent
logits on a fixed batch. Only its passed artifact can unlock final-test.

## Operator command

Inspect the Step 9 command surface without writing:

```bash
python scripts/evaluate_prediction_anchored_bridge_campaign.py --help
```

Aggregate paired replica evidence and preview the immutable selection:

```bash
python scripts/evaluate_prediction_anchored_bridge_campaign.py --dry-run select \
  --registry campaign_registry_step8.json \
  --evidence deployable_replica_evidence.json \
  --output selected_deployable_preconfirmation.json
```

After claiming the one-shot `stack_val_deploy` receipt, finalize confirmation:

```bash
python scripts/evaluate_prediction_anchored_bridge_campaign.py confirm \
  --preconfirmation selected_deployable_preconfirmation.json \
  --access-receipt stack_val_deploy_access.json \
  --metrics stack_val_deploy_metrics.json \
  --output locked_deployable.json
```

The `reports` and `validate-final-test` subcommands use the same immutable
artifact contracts. `--dry-run` validates and hashes results without
publishing them.

## Verification

`tests/test_prediction_anchored_bridge_step9.py` covers measured quota and
duplicate-parent failures, success/failure B0--B6 campaigns, conditional
skips, all validity gates, aggregate/tie/median rules, confirmation with no
runner-up, null recovery, distinct oracle ceilings, visible failed/skipped
rows, telemetry, median-only publication, forbidden final-test flags, and
tolerance-identical clean HLT-only reload with privileged paths absent.
