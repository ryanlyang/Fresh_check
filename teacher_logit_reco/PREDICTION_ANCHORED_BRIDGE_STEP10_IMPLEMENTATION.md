# Prediction-Anchored Bridge Step 10 implementation

Step 10 is implemented as the production boundary above the Step 1--9
scientific code. It renders the measured registry into an immutable Tigris
dependency graph, validates every allocation before model construction,
submits only after an explicit opt-in, and rehearses the same graph locally
without invoking Slurm.

## Production graph and scheduler policy

`local_particle_residual_field/bridge_production.py` provides the versioned
resource, node, graph, allocation-launch, job-ledger, scheduler-simulation,
and CPU-rehearsal contracts. Graph construction requires:

- the final measured registry with zero runnable `UNMEASURED` rows;
- a Step 9 reservation artifact bound to that exact registry;
- a passed measured 5 or 6 GiB quota mode;
- exact generated inventories of 54 maximum, 46 reconstruction-breadth, and
  45 post-teacher configurations.

Every runnable configuration appears in exactly one allocation. The graph
puts all eight upstream rows in the shared Tpred-lineage allocation, launches
L0 independently, and packs at most four reconstruction configurations by
teacher namespace/shared source. Each training allocation contains the three
paired seeds 101/202/303 and declares metrics for all three but retained
weights for the ordered median only.

The node reservations sum exactly to the measured campaign projection. All
nodes request one Tigris node and explicit host memory. Their allocation
contract fixes rank 0 as the sole source-opening/staging/ledger leader, one
persistent NPZ open per source, one allocation-wide RAM byte ledger,
non-evictable raw shards, derived-only eviction, and whole-pack restart after
failure/preemption. No node contains a persistent dense-field output path.

The dependency boundary is fail closed:

```text
B0 -> B1 -> B2 -> {L0, paired consumers}
                         |
                  select -> one-shot consumer confirmation
                         |
                  bindings -> namespace caches -> B6 release
                         |
                  packed paired3 matrix -> aggregate selection
                         |
                  one-shot deploy confirmation -> report/export/reload
                         |
                  protected HLT-only final-test
```

The consumer-confirmation job creates `selected_bridge_consumer.json`; it does
not require that future artifact at launch. B5 and every primary/N3 teacher
consumer require the confirmed locked artifact. No guessed consumer mode
exists. A failed consumer confirmation prevents bindings, caches, and B6 from
starting through `afterok`. A failed deployment confirmation prevents export
and final-test. The final-test node is omitted by default and remains HLT-only
when separately approved.

## Command-line and Slurm entry points

The production command surface now includes:

- `scripts/run_prediction_anchored_bridge_campaign.py` for graph publication,
  graph validation, CPU rehearsal, and job-ledger finalization;
- `scripts/submit_prediction_anchored_bridge_graph.py` for reviewed dry
  rendering or explicit `sbatch` execution;
- `scripts/run_prediction_anchored_bridge_allocation.py` for pre-accelerator
  node/memory/leader/selection validation;
- `scripts/inspect_prediction_anchored_bridge_graph.py` for immutable run-ID
  and teacher-namespace lookup inside allocations;
- `scripts/write_prediction_anchored_bridge_execution_spec.py` for one hashed
  site-local binding of split manifests, aligned caches, the HLT baseline, and
  numerical hyperparameters;
- `scripts/train_prediction_anchored_r0.py` and
  `scripts/prepare_prediction_anchored_bridge_inputs.py` for real streamed R0
  training and exact `stack_train_distill` recipe/scaler preparation;
- `scripts/execute_prediction_anchored_bridge_consumers.py` for all eight
  paired3 consumer rows plus full response/control selection evidence;
- `scripts/confirm_prediction_anchored_bridge_consumer.py` for the sealed
  one-shot `stack_val_consumer` evaluation;
- the canonical singular
  `scripts/train_prediction_anchored_bridge_consumer.py` entry point;
- `scripts/bind_prediction_anchored_bridge_teachers.py` for deriving the exact
  primary, all-50, and eligible alternate bindings from confirmed campaign
  artifacts without operator-supplied hashes;
- `scripts/cache_prediction_anchored_bridge_logits.py` for repository-owned
  bound-teacher forwards on `stack_train_distill`, publishing only detached
  logits/labels/event identities and no dense fields;
- dry-run support in consumer selection/binding and target-cache validation.

The five declared Slurm entry points are present, plus a shared bootstrap:

- `sbatch/run_prepare_prediction_anchored_bridge_ram.sh`;
- `sbatch/run_train_prediction_anchored_bridge_consumer.sh`;
- `sbatch/run_cache_prediction_anchored_bridge_logits.sh`;
- `sbatch/run_train_prediction_anchored_bridge_reconstructor.sh`;
- `sbatch/submit_prediction_anchored_bridge_pilot.sh`;
- `sbatch/prediction_anchored_bridge_common.sh`.

All Tigris workers use the full account `reu-aisocial`, request one node and
host memory, and export `PYTHONNOUSERSITE=1`. Allocation RAM is restricted to
`/dev/shm/prediction_anchored_bridge/$SLURM_JOB_ID`; cleanup refuses any other
prefix. Per-attempt launch manifests live under the node and Slurm job ID, so
a replacement allocation starts clean rather than resuming partial replicas.

B1 through B5 teacher-logit generation is repository-owned. It uses
`PAB_EXECUTION_SPEC`, writes no dense field artifact, retains only ordered
median consumer weights persistently, and creates selector-grade metrics and
the sealed confirmation receipt itself. Teacher weights are copied
byte-for-byte from the exact model-val-selected replica so their SHA-256 is
unchanged across selection, binding, target caching, and the future live KD
side. Reconstruction is now repository-owned as well: the packed executor
maps every one of the 46 registered reconstruction rows to its C0, local, HLG,
all-50, semantic-control, or direct graph; trains paired seeds; and publishes
only ordered-median weights. Deployable selection, sealed `stack_val_deploy`
confirmation, bundle export/reload, and the optional HLT-only final-test path
are repository-owned by `scripts/deploy_prediction_anchored_bridge.py`. Actual
submission refuses to call `sbatch` unless every still-required path exists
and is not a symlink.

## Operator workflow

Publish a graph from the final measured registry and reservation artifact:

```bash
python scripts/run_prediction_anchored_bridge_campaign.py \
  --campaign-action production-plan \
  --registry campaign_registry_step8.json \
  --reservations campaign_reservations.json \
  --artifact-root /home/ryreu/atlas/Fresh_check/fresh_check_outputs/prediction_anchored_bridge \
  --production-output prediction_anchored_tigris_graph.json
```

Run the local graph/failure/preemption rehearsal without writing or
submitting:

```bash
python scripts/run_prediction_anchored_bridge_campaign.py \
  --campaign-action rehearse-cpu \
  --registry campaign_registry_step8.json \
  --reservations campaign_reservations.json \
  --artifact-root /tmp/prediction_anchored_rehearsal \
  --dry-run
```

Preview the exact Slurm commands (the default is non-submitting):

```bash
python scripts/submit_prediction_anchored_bridge_graph.py \
  --graph prediction_anchored_tigris_graph.json
```

First write the immutable site-local execution binding (the cache/checkpoint
paths must be the paths visible on Tigris):

```bash
python scripts/write_prediction_anchored_bridge_execution_spec.py \
  --parent-manifest /absolute/path/split_manifest.json.gz \
  --child-manifest /absolute/path/prediction_anchored_child_splits.json \
  --hlt-cache-dir /absolute/path/hlt_cache \
  --offline-cache-dir /absolute/path/offline_cache \
  --baseline-checkpoint /absolute/path/hlt_baseline.pt \
  --output /absolute/path/prediction_anchored_execution_spec.json
```

After setting that binding and the remaining safe executor paths, actual
submission requires an additional explicit switch:

```bash
export PREDICTION_ANCHORED_GRAPH=/absolute/path/prediction_anchored_tigris_graph.json
export PAB_REGISTRY=/absolute/path/campaign_registry_step8.json
export PAB_RESERVATIONS=/absolute/path/campaign_reservations.json
export PAB_EXECUTION_SPEC=/absolute/path/prediction_anchored_execution_spec.json
export PREDICTION_ANCHORED_EXECUTE=1
bash sbatch/submit_prediction_anchored_bridge_pilot.sh
```

This submits B0--B6 but not final-test. Final-test is a separate invocation
requiring both `PREDICTION_ANCHORED_INCLUDE_FINAL_TEST=1` and
`PREDICTION_ANCHORED_APPROVE_FINAL_TEST=1`; its validation rejects every
bridge/oracle/offline/privileged flag.

## Verification

`tests/test_prediction_anchored_bridge_step10.py` covers:

- rendered 54/46/45 inventories, conditional TALT, and exact run coverage;
- zero-`UNMEASURED` enforcement and measured 5/6 GiB modes;
- one-node/one-leader/one-open/shared-ledger packing and host-memory checks;
- selected-consumer and both sealed confirmation dependency boundaries;
- scheduler success, consumer/deploy confirmation failure, dependency
  cancellation, and whole-pack preemption restart;
- paired3/median-only publication metadata and no dense-field paths;
- immutable dry job ledgers and `afterok` job-ID recording;
- non-submitting command rendering and local CPU rehearsal;
- refusal of actual submission before required scientific executors exist;
- CLI help, Tigris account/usersite/memory/node shell contracts, and the
  protected final-test policy.

The complete Step 1--10 bridge suite passes together, so the production layer
did not weaken the earlier numerical, provenance, selection, reporting, or
deployment contracts.
