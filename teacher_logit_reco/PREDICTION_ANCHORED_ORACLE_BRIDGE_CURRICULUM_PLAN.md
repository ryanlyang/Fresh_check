# Prediction-Anchored Oracle-Bridge Curriculum

Status: implementation plan for the first `rho = 0.10` pilot

This document specifies a new experiment family. It does not alter the meaning of the existing local residual-field curriculum in
`LOCAL_RESIDUAL_FIELD_CURRICULUM_DISTILLATION_PLAN.md`.

## 1. Purpose

The central question is whether a deployable HLT-only residual reconstructor can recover useful behavior from a deliberately small, physically interpretable improvement to an ordinary predicted residual field.

The first experiment must answer three questions independently:

1. Does moving an existing residual prediction ten percent toward its offline target make the same downstream tagger measurably better?
2. Can that improved tagger be distilled while its field input is replaced by an HLT-only prediction?
3. Which loss terms and reconstructor components are responsible for any recovered improvement?

The pilot therefore separates field prediction, bridge construction, privileged-consumer training, frozen-consumer distillation, and final HLT-only evaluation. It includes matched controls and a broad but gated ablation campaign.

## 2. Locked bridge definition and channel semantics

For an event with HLT inputs `x`, let

- `f_true` be the 50-channel offline residual-field target;
- `R0(x) = (f0, h0)` be the ordinary HLT-only residual prediction and its final per-particle hidden representation;
- `P = {0, ..., 44}` be the 45 continuous physical residual channels, arranged as three 15-channel radius blocks;
- `S = {45, ..., 49}` be the three derived flags, reliability score, and uncertainty scale;
- `Delta_true[P] = f_true[P] - stop_gradient(f0[P])`.

The primary bridge changes only the physical channels:

```text
b_rho[P] = f0[P] + rho * (f_true[P] - f0[P])
b_rho[S] = f0[S]
```

Therefore the first privileged bridge is:

```text
b_0.10[P] = 0.90 * f0[P] + 0.10 * f_true[P]
b_0.10[S] = f0[S]
```

This prevents fractional oracle flags and oracle-derived reliability summaries from becoming an easy privileged shortcut. It is a prediction-anchored interpolation, not `rho * f_true` and not a scaling of the whole field.

Primary endpoint semantics are fixed:

```text
rho = 0.00  -> ordinary predicted field f0
rho = 1.00  -> true physical 45 channels plus predicted five reliability channels
```

An explicit `all50_bridge` ablation applies the same interpolation to every channel and reaches the complete `f_true` tensor at `rho = 1`. The bridge channel mask is part of every recipe hash; `physical45` and `all50` artifacts can never be confused.

The initial campaign implements only `rho = 0.10` as a trainable privileged endpoint. Intermediate response points may be evaluated, but later ladder rungs must not be launched automatically.

## 3. Core hypotheses

### H1: reachable privileged gain

A tagger trained on `b_0.10` should outperform the same tagger evaluated on `f0`. Because the bridge starts from an existing prediction, this tests a small correction rather than asking a reconstructor to synthesize a full offline field at once.

### H2: same-consumer recovery

For the selected frozen bridge consumer `T10`, a learned HLT-only field `f_hat` should reduce

```text
T10(x, f_hat) - T10(x, b_0.10)
```

without changing `T10` itself.

### H3: deployability

The deployed graph must consist only of the HLT-only reconstructor and the frozen bridge consumer:

```text
x -> R10(x) -> f_hat -> T10(x, f_hat) -> logits
```

No offline field, bridge cache, oracle teacher, or cached teacher logits may be required at inference.

### H4: hierarchy is useful

Target-radius local processing followed by HLT-anchored region attention should recover more of the bridge gain than a same-capacity particle-only correction branch.

### H5: logit supervision and field supervision are complementary

Knowledge distillation tells the reconstructor which field errors matter to the frozen consumer. Bridge-field regression provides a stable local target. The ablations must establish whether either or both are actually needed.

## 4. Non-goals for the first pilot

The first pilot does not:

- train `rho = 0.20` through `0.50` consumers;
- replace the frozen `T10` with a separately trained student tagger;
- alter the target schema or invent unconstrained latent privileged tokens;
- select a model on final-test;
- compare fusion models trained on a split already consumed by bridge distillation;
- claim that a favorable oracle diagnostic is a deployable gain;
- allow Stage 1 distillation to proceed with a guessed bridge consumer.

## 5. Deployment and reporting contract

The following distinction is mandatory.

### Oracle/privileged diagnostics

These rows may use `f_true`, `b_0.10`, bridge response curves, shuffled bridge controls, or cached privileged logits. They must be labeled `oracle_diagnostic` or `privileged_diagnostic`.

### Deployable rows

These rows use only HLT-available inputs at inference. The field supplied to `T10` must come from `R0` or `R10`. A deployable checkpoint must reload and run after all bridge and target-logit cache paths are removed.

Final-test evaluation is allowed only for a locked deployable checkpoint and must be HLT-only. The report must never place an oracle bridge result in the deployable leaderboard.

`A0` remains a true HLT-only baseline family. A zero-field or predicted-field invocation of a bridge consumer is a consumer diagnostic, not a replacement for `A0`. The pilot trains five explicit fairness controls:

| Run ID | Training data and budget | Fairness question |
|---|---|---|
| `A0_C250` | Exact 250k `stack_train_consumer`; ordinary baseline horizon | Is Tpred/T10 better than HLT ParT at the same consumer jet count? |
| `A0_C250_LONG` | Warm-start from `A0_C250`; continue for exactly the bridge fine-tuning optimizer budget on the same 250k | Is improvement merely additional optimization on the same jets? |
| `A0_S500` | Full 500k `stack_train` union with the stable baseline schedule | Does the final pipeline beat HLT ParT when it receives the complete post-R0 labeled-jet allowance? |
| `A0_CAP500_direct_hlt` | Full 500k union; raw-HLT-only model matched to the canonical-A3 `R0+D10_A3_hlg_primary+selected_T10` bundle's total parameter count and approximate FLOPs, with a matched declared training budget | Is improvement merely additional raw-HLT model capacity? |
| `A0_CAP500_r0rep_direct` | Full 500k union; frozen HLT-only `R0` supplies `f0/h0` to the same HLG backbone, which emits logits directly with CE and no field/KD/`T10` path | Does residual-supervised `R0` representation learning explain the gain without bridge distillation? |

The first three use the standard HLT ParT architecture. The final two are complementary capacity/representation controls. Reports show all five; `A0_C250_LONG` is the primary consumer-level comparator and `A0_S500`, `A0_CAP500_direct_hlt`, and `A0_CAP500_r0rep_direct` are the primary final-pipeline fairness comparators. No 1M A0 run is part of this pilot.

## 6. Split contract

The low-data pilot is locked to:

```text
model_train: 500k
model_val:   150k
stack_train: 500k
stack_val:   150k
final_test:  150k
```

Two deterministic, class-stratified validation partitions are created from each 150k parent:

```text
model_val_stop:       75k
model_val_select:     75k
stack_val_consumer:   75k
stack_val_deploy:     75k
```

Each pair must have exact parent coverage and zero overlap. The existing non-overlapping curriculum splits are reused as follows:

| Split | Permitted use in this pilot |
|---|---|
| `model_train` | Train `R0` only |
| `stack_train_consumer` | Train `Tpred`, `Tpred_continue`, `T10_clean`, `T10_robust`, and diagnostic `T10_all50_clean` |
| `stack_train_distill` | Train `R10` variants against the permanently frozen selected `T10` |
| `model_val_stop` | Per-seed early stopping and checkpoint choice only |
| `model_val_select` | Consumer-recipe and reconstructor-configuration ranking only |
| `stack_val_consumer` | One-shot confirmation of the already locked consumer recipe and median replica |
| `stack_val_deploy` | One-shot confirmation of the already locked deployable configuration and median replica |
| `final_test` | One locked HLT-only deployable evaluation |

This prevents `R0` from being trained on the examples later used to teach the bridge consumer or distill the reconstructor. The 500k `stack_train` manifest is divided once into deterministic, class-stratified, non-overlapping 250k `stack_train_consumer` and 250k `stack_train_distill` child manifests. The preflight requires these exact counts, full parent coverage, and zero overlap. No `stack_val` child may choose an epoch, rank configurations, or break a tie. A failure on either one-shot confirmation child produces a failed/inconclusive result; it never triggers selection of the runner-up. `stack_val_deploy` remains unreadable until the deployable-selection manifest is locked.

The first pilot does not use cross-fitted teachers. One selected `T10` is trained only on `stack_train_consumer`, then frozen permanently. That exact checkpoint is used for both sides of every distillation loss on `stack_train_distill`:

```text
z_target = selected_T10(x, b_0.10)
z_live   = selected_T10(x, f_hat)
```

The selected checkpoint SHA-256 must be identical in `selected_bridge_consumer.json`, target-logit provenance, the live training graph, and the deployable bundle. If `f_hat == b_0.10`, the two forwards must agree numerically and KD must be zero within tolerance. `T10` must not be refit after target logits are produced. Cross-fitting is reserved for a later design with fold-specific live consumers and an explicit deployment-reconciliation strategy.

## 7. Pilot execution graph

The job graph is deliberately gated:

```text
B0  validate data, child splits, masks, units, RAM, storage, and R0 prerequisites
 |
B1  train or register R0 using RAM/streamed targets; validate live f0 and h0
 |
B2  write virtual physical45/all50 bridge recipes; audit batchwise construction
 |
B3  launch D10_L0; train three paired seeds of the three matched A0 controls and every consumer recipe
 |
B4  aggregate model-val-select results, lock one recipe/median T10, then confirm it once on stack_val_consumer
 |
B5  bind every teacher, then cache primary bridge, all50/alternate, and N3-on-f0 logits in separate namespaces
 |
B6  launch three paired seeds of every remaining valid loss, interaction, control, and architecture configuration
 |
aggregate model_val_select -> lock one config/median checkpoint -> one stack_val_deploy confirmation -> reload audit -> final-test HLT-only
```

Every primary Stage `B5` and later job must read `selected_bridge_consumer.json`. Non-primary jobs must additionally read their exact all-50 or alternate `teacher_binding_v1` and must reject the primary artifact as a substitute. All jobs fail closed if a binding is missing, invalid, points at an incomplete checkpoint, or disagrees with recipe/logit provenance. There is no implicit “best-looking checkpoint” fallback. B6 requires a provenance-valid privileged consumer but does not stop on a scientific quality warning or because a smaller C0 reconstructor fails: available GPUs should be used to test the full HLG hypothesis even when a simple model or a predeclared performance rule fails.

## 8. Artifact and provenance contract

Every artifact has a versioned schema and content hashes for its parents.

### RAM/streaming field source

The default low-storage mode does not persist dense `f_true`, `f0`, `h0`, `Delta_true`, bridge endpoints, or negative controls as pilot-specific artifacts. The repository's current HLT/offline sources are monolithic compressed NPZ files and cannot be memory-mapped or truthfully described as already sharded. Production packed jobs are locked to exactly one node. Allocation rank 0 opens, verifies, hashes, and decompresses each required persistent source exactly once into verified node RAM; raw staged source shards are non-evictable for the allocation lifetime. The provider exposes deterministic contiguous ranges from that staged copy, and GPU ranks receive disjoint assignments. Related seeds/configurations are packed into the same multi-GPU allocation when they share immutable sources and a teacher. Multi-node packed training is unsupported in this pilot and must fail preflight rather than accidentally create independent rank-zero staging domains.

A rank-local provider adapts the repository's verified RAM-workspace and bounded-LRU pattern. It must:

- accept only verified `tmpfs`/`ramfs`, preferring `/dev/shm` and never assuming `SLURM_TMPDIR` is memory-backed;
- maintain one allocation-wide atomic byte ledger guarded by a lock in the allocation RAM root; per-rank reservations are transactions against that ledger, and combined raw plus derived reservations must retain at least 20% filesystem headroom;
- unpack source arrays into non-evictable job-scoped RAM-backed `.npy` shards with a default logical shard size of 8,192 jets; fail preflight if the complete raw source set needed by the allocation does not fit with the required headroom;
- open and hash each persistent compressed NPZ only once per allocation, verify staged-array hashes before release to ranks, and record rank/shard ownership;
- apply bounded-LRU eviction and deterministic regeneration only to derived `f_true`, `f0`, bridge, control, and target-logit working shards; regeneration must read the resident raw RAM shards and may not reopen a persistent NPZ;
- permute shard order and then event order within a resident shard for training; never require unconstrained random reads across persistent compressed sources;
- never cache `h0` persistently; `R10` normally recomputes `f0, h0` once per batch;
- expose cache hits, misses, regeneration, evictions, peak resident bytes, and a `persistent_field_tensors_written=false` audit flag;
- clean rank-local derived allocations and the allocation-wide staged workspace on normal exit, and rely on the job-scoped RAM filesystem after abnormal termination.

A small persistent debug materialization is allowed only for tests/audits, must be explicitly requested, and must be excluded from production training inputs.

### Virtual bridge recipe

Schema: `prediction_anchored_residual_bridge_recipe_v2`.

The persistent artifact is a small JSON recipe, not a dense bridge tensor. It contains:

- exact decimal `rho`, formula version, and `physical45` or `all50` channel mask;
- R0 checkpoint, shared HLT/offline source, split-manifest, target-schema, preprocessing, and event-order hashes;
- target-kernel radii and field layout;
- control type and deterministic seed when applicable;
- verification summaries from a deterministic audit sample;
- RAM execution mode and storage-policy version.

Batch construction must reject non-finite fields, mismatched masks, duplicate event IDs, inconsistent particle order, source hash changes, or incompatible units. Response points and controls are regenerated from the same parent recipe and are never separate dense caches.

### Selected bridge consumer

`selected_bridge_consumer.json` must contain at least:

```json
{
  "schema": "selected_bridge_consumer_v2",
  "selected_consumer_recipe": "T10_clean_or_robust",
  "selected_consumer_id": "recipe_seed_replica_id",
  "paired_seed_ids": [101, 202, 303],
  "selected_median_seed_id": 202,
  "selected_rho_endpoint": 0.10,
  "selection_source": "model_val_stop_then_model_val_select__stack_val_consumer_confirmation_only",
  "selection_reason": "machine-readable summary",
  "recipe_aggregate_metrics": {},
  "replica_metrics": [],
  "pre_confirmation_selection_sha256": "...",
  "stack_val_consumer_confirmation": {},
  "checkpoint_sha256": "...",
  "f0_checkpoint_sha256": "...",
  "bridge_recipe_sha256": "...",
  "bridge_channel_policy": "physical45",
  "model_val_select_bridge_response": [],
  "negative_control_results": {},
  "deployable": false
}
```

This file selects a privileged teacher/consumer. It is not itself deployable. The selection portion is written and hashed before `stack_val_consumer` is opened; confirmation may append only the predeclared pass/fail record. A failed confirmation cannot change the recipe or seed ID.

Non-primary teachers never reuse or overwrite this artifact. `T10_all50_clean` and the provenance-valid alternate recipe each receive an immutable `teacher_binding_v1` artifact containing their recipe, aggregate seed results, selected median replica, exact checkpoint hash, channel policy, validation-manifest hashes, declared target-cache namespace/schema, eligibility boolean, failed quality rules, and any explicit warning override. A binding must not contain the hash of a cache artifact that does not yet exist.

### Target-logit cache

Required metadata binds the cache to:

- selected `T10` checkpoint hash;
- exact `rho = 0.10` bridge-recipe hash and `physical45` channel policy;
- split and event-order hashes;
- logit class order and temperature convention;
- `stack_train_distill` child-manifest identity;
- the same selected checkpoint hash used by the live consumer.

The selected physical-45 teacher, all-50 teacher, and alternate teacher each have a distinct cache manifest and cache namespace. A cache consumer must reject a primary-selection file when its declared teacher-binding hash is non-primary, and vice versa. Exact on-the-fly target logits are permitted for a diagnostic only when the same immutable binding is checked on both target and live sides.

The provenance direction is acyclic and immutable:

```text
teacher checkpoint + recipe + validation manifests
    -> teacher binding hash
        -> target-logit cache manifest containing that binding hash
            -> training run manifest containing both hashes
```

No binding references a final cache hash. In addition to the primary physical45, all50, and alternate namespaces, `D10_N3_nonprivileged_teacher_kd` has a fourth explicit namespace, `physical45_selected_teacher_on_f0_control`, caching `selected_T10(x,f0)` on the exact distillation manifest. N3 must use that cache and may not fall back to ambiguous on-the-fly targets.

The target logits are detached data. The live frozen-consumer forward pass used for reconstruction must remain differentiable with respect to the predicted field.

### Persistent storage budget

The pilot trades recomputation and RAM for persistent quota. It does not duplicate frozen parents inside each ablation checkpoint.

Persist only:

- weights-only `R0`, consumer, and trainable `R10` states;
- small recipe/provenance/selection manifests;
- selected target logits, which are small for ten classes;
- aggregate metrics, bounded logs, and one final deployable bundle.

Storage accounting has two explicit phases. The Step 1 registry assigns every not-yet-implemented architecture a conservative formula-based `provisional_bytes` value and `measurement_status=UNMEASURED`; this validates formulas and campaign upper bounds without pretending the model can already be instantiated. As Steps 3 and 5–8 implement configurations, each step instantiates its new architectures and serializes a representative weights-only state into verified RAM, recording `measured_bytes` plus 5% serialization headroom and changing the status to `MEASURED`. Target-logit arrays are projected from the actual event/class count and dtype. Logs/metrics have explicit per-configuration caps, and the final bundle has its own measured reservation. Production submission is forbidden until every runnable registry entry is `MEASURED` and the final measured 5/6 GiB preflight passes. Any artifact exceeding its measured reservation must fail publication or obtain a new preflight-approved budget; it may not silently overrun quota.

Keep optimizer/scheduler states, rotating resume checkpoints, batch diagnostics, generated fields, and non-retained seed replicas in verified job-local RAM. The three paired seeds for one configuration are scheduled so all replica metrics are known before publication. Persist metrics for all seeds but only the predeclared median-performing replica's weights; the other two replicas are metrics-only records. The selected deployable bundle copies each required parent once. This is a publication policy, not deletion of an existing user artifact: non-retained replicas never leave job-local RAM.

Cross-allocation resume is deliberately unsupported in the first pilot. Preemption, node failure, or wall-time expiry invalidates the incomplete allocation and restarts all three replicas of that configuration from their immutable parents. Submission must request a wall time that covers the measured three-replica smoke projection plus 25% headroom. No persistent optimizer/recovery checkpoint is permitted unless a later plan revision adds an explicit temporary-artifact quota and lifecycle.

Budgets, excluding already-shared HLT/offline source caches, are:

```text
preflight-derived expectation:    2.5-4.5 GiB
normal hard pilot budget:        5 GiB
paired-three-seed hard ceiling:  6 GiB
```

A preflight sums the generated run registry, measured checkpoint sizes, retained-replica policy, every teacher/logit binding, reports, and the bundle, and refuses submission above the selected budget. The current repository default is 128 particle slots; at that width a single dense 50-channel float16 field tensor for the locked 1.3M non-final-test jets would occupy about 15.5 GiB before masks/metadata. The projection must read the actual source manifest width rather than assume 128 or 256. Low-storage mode must not create such a cache. If an existing provenance-valid target cache already exists, it may be read without copying, but it is not required.

## 9. Ordinary residual predictor `R0`

`R0` is the existing local particle residual-field predictor trained in the ordinary supervised way:

```text
HLT particles + HLT jet features -> predicted residual fields f0, hidden particles h0
```

The first implementation should reuse the current local residual-field model and target schema rather than retraining an unrelated baseline. Its checkpoint, preprocessing, target normalization, matching policy, and validation metrics must be registered explicitly.

Requirements:

- inputs are HLT-available only;
- the training target is `f_true`;
- no bridge, privileged consumer, or offline feature enters the input graph;
- RAM/shard generation runs in deterministic evaluation mode;
- masked particles remain exactly masked after inverse normalization;
- regenerated and direct live `R0` outputs agree within configured numerical tolerance;
- the final per-particle hidden tensor `h0` is exposed without modifying `R0` weights;
- the checkpoint is frozen when used as the anchor for the first bridge pilot.

`R0` remains frozen inside `R10` for the initial campaign. Its output is evaluated once per batch under no parameter gradients, and both `stop_gradient(f0)` and `stop_gradient(h0)` are supplied to the correction branch. A raw-HLT skip is mandatory so `h0` cannot become an information bottleneck. A later experiment may unfreeze its last block, but that is not part of this pilot.

## 10. Bridge provider and controls

The bridge provider consumes aligned batchwise `f0` and generated `f_true` tensors and emits `b_0.10` without writing it persistently. It must implement the physical-45 formula directly and test the endpoints even when only `0.10` is used for training. The `all50` policy is a separate hashed recipe.

Required response points for evaluation are:

```text
rho in {0.000, 0.025, 0.050, 0.075, 0.100}
```

Required negative controls preserve as much superficial scale information as possible while breaking the useful correspondence:

1. `event_shuffled_delta`: assign a deterministic matched wrong event's valid delta.
2. `particle_shuffled_delta`: permute valid particles within each jet.
3. `sign_reversed_delta`: use `f0 - 0.10 * Delta_true`.
4. `same_norm_random_delta`: draw deterministic random directions and rescale each field group to the true correction norm.
5. `radius_group_permuted_delta`: permute residual groups among target radii when shapes permit.

All wrong-event controls use `matched_wrong_event_map_v1`. Fit multiplicity tertiles, jet-`pT` quartiles, and jet-mass quartiles from HLT-only features on the control's authorized training parent. Within each logical 8,192-event source block and class, group events by those coarse bins, sort by a seeded event-identity hash, and cyclically shift by one. A singleton bin is merged deterministically with the nearest bin in Manhattan bin-index distance within the same class/block until at least two events are available; ties prefer multiplicity, then `pT`, then mass, then lower bin index. A class/block with fewer than two events makes the control invalid rather than permitting self-matches or cross-class matches. Evaluation reuses the frozen training-bin edges.

Controls are generated batchwise from deterministic index/seed recipes. Event shuffling stores only this compact block/bin permutation plus within-block indices, so it never induces arbitrary cross-block reads or LRU thrashing; the other controls require no dense persistent payload. The same wrong-event map is used for corresponding shuffled logits and fields. Controls are diagnostics only. A negative-control checkpoint can never be selected for deployment.

Before consumer selection, promote `D10_L0_bridge_only` to an explicit reachability diagnostic. It predicts `0.10 * Delta_true[P]` out of sample and reports explained variance, normalized MSE, correlation, and cosine alignment for every radius and field group. This diagnostic describes HLT predictability; it does not by itself prevent HLG/KD jobs after a valid bridge teacher exists.

## 11. Bridge consumers

All bridge consumers use the same tagger architecture, input contract, optimizer family, stopping rule, and training budget unless the run explicitly tests one of those factors.

The scientific campaign uses the locked paired seed IDs `{101, 202, 303}`. `A0_C250`, `Tpred`, and every downstream branch have one replica per seed ID. Configuration comparisons pair equal seed IDs; recipe/configuration selection operates on three-seed aggregates and never on the best individual replica.

For each seed family, `A0_C250` and `Tpred` start from the same registered HLT ParT reference checkpoint and identical copied HLT weights. `Tpred` widens only the particle-input projection for the 50 residual fields: existing columns and every other parameter are copied bit-for-bit, and new field-input columns are initialized exactly to zero. Because Weaver applies input LayerNorm before that projection, the widened consumer must normalize the original HLT block and the new field block independently while retaining one checkpoint-compatible affine vector; otherwise merely appending 50 columns changes the HLT logits and a zero-gain field block cannot learn. New field normalization gains retain their trainable unit defaults, while only the new projection columns are zero. Before training, evaluation-mode logits from the widened model with any finite `f0` must equal the reference HLT logits within `atol=1e-6, rtol=1e-5`, and the first backward pass must produce a finite nonzero gradient in the new field-input projection columns.

### `Tpred`

Train a tagger on HLT inputs plus the ordinary prediction `f0`. This establishes the predicted-field consumer and supplies the initialization for bridge consumers.

### `Tpred_continue`

Continue `Tpred` for the same number of optimizer steps allocated to bridge fine-tuning, still using `f0`. This is the matched-extra-compute control. It distinguishes benefit from the bridge from benefit due merely to more training.

For a seed family, `Tpred_continue`, `T10_clean`, `T10_robust`, and `T10_all50_clean` branch from the exact same terminal `Tpred` weights and optimizer/scheduler state. Maximum steps, batch indices/order, label sampling, dropout/stochastic-depth random stream, evaluation cadence, and checkpoint opportunities are identical. The robust field-condition sampler has a separate recorded RNG stream because its declared input mixture differs. These branches should share one allocation or another job-local RAM lineage so the continuation optimizer state is never reconstructed approximately from a weights-only publication.

### `T10_clean`

Warm-start from `Tpred`, then fine-tune on the exact batchwise-generated `b_0.10` bridge. The default loss is supervised event classification. It must not use `f_true` except through that bridge provider.

### `T10_robust`

Warm-start from the same `Tpred` checkpoint. Each training example draws one input condition:

- 60%: exact `b_0.10`;
- 20%: exact `f0`;
- 15%: `b_rho` with `rho ~ Uniform(0.00, 0.10)`;
- 5%: `b_0.10` with light, deterministic-by-seed field corruption.

Default light corruption:

- Gaussian noise with per-field standard deviation `0.10 * std(b_0.10 - f0)` measured on the training partition;
- field-group dropout probability `0.05`;
- target-radius-group dropout probability `0.025`.

Noise and dropout are applied only to valid physical-45 residual entries in the primary robust consumer; the five reliability channels remain at `f0`. These defaults must be configurable and recorded in the run manifest.

### Same-consumer evaluation

Each consumer is evaluated without changing its weights on:

- `f0`;
- every bridge response point;
- `[f_true[P], f0[S]]` as the primary physical-45 oracle ceiling;
- complete `f_true` as a separate full-50 oracle diagnostic;
- all negative controls;
- zero field, labeled explicitly as a consumer diagnostic.

The primary bridge-gain comparison is always within the same consumer. Comparing `T10(b_0.10)` only to `Tpred(f0)` confounds the field with the consumer and is insufficient.

### Bridge-consumer selection

The selector evaluates the three-seed `T10_clean` and `T10_robust` recipe aggregates. Each seed's epoch is selected only on `model_val_stop`; the resulting checkpoint is evaluated once on `model_val_select`. Every aggregate rule below is still computed and reported:

1. positive same-consumer `b_0.10` versus `f0` gain;
2. positive `T10(b_0.10)` versus `Tpred_continue(f0)` matched-compute gain;
3. response-curve compliance with the numerical rule in Section 18;
4. resistance to negative controls;
5. limited `f0` degradation relative to `Tpred_continue`, so the frozen consumer is not unusably brittle;
6. lower across-seed variance, then lower mean ECE, then lower mean Brier score as tie-breakers.

The production pilot explicitly prefers `T10_robust`; `T10_clean` is retained
as the matched alternate diagnostic. Sort each recipe's three replicas in
ascending order by `(model_val_select bridge accuracy, same-consumer bridge
gain, negative bridge cross-entropy, seed ID)` and retain the middle replica.
The exact robust median checkpoint becomes the permanently frozen primary
teacher, while the clean median checkpoint is separately bound for the
alternate comparison; the lucky best seed is never chosen. The pre-confirmation
artifact is locked before one evaluation on `stack_val_consumer`. Performance
failure there is recorded and continuation remains authorized; hard integrity
failure writes a stopped-campaign report. The pipeline must not guess a
consumer, choose a runner-up, or refit either recipe.

## 12. `R10` reconstructor architecture

The primary reconstructor is a correction model over the frozen ordinary prediction:

```text
f0_live, h0 = stop_gradient(R0(x))
standardized_raw_delta[P] = HLG(x, standardized(f0_live), h0)
f_hat[P] = f0_live[P] + bounded(inverse_scale(standardized_raw_delta[P]))
f_hat[S] = f0_live[S]
```

`HLG` means local → hierarchical → global. It predicts physical residual-field corrections; the downstream tagger never receives unconstrained learned tokens.

### 12.1 Numerical spaces and immutable scalers

Four numerical spaces are distinct and must never be inferred from tensor names:

1. `physical_field`: raw repository residual-field coordinates. `f_true`, `f0`, every bridge, trust-bound computation, addition to `f0`, and every consumer input use this space.
2. `conditioning_standardized`: only the copy of `f0` supplied to HLG is transformed as `(f0 - mu_f0) / sigma_f0`.
3. `correction_standardized`: correction heads emit dimensionless corrections; channel scale `sigma_delta` maps them back to physical units before the trust `tanh` and addition.
4. `loss_standardized`: bridge/true Huber residuals are divided by `sigma_delta` and then balanced equally across the declared radius/field groups.

`mu_f0`, `sigma_f0`, `sigma_delta`, and trust statistics are fit only on valid particles in `stack_train_distill`, never on either validation parent or final-test. Use float64 accumulation, publish counts/means/scales, and hash the source manifest, `R0`, target schema, masks, and fitting code version. `sigma_f0` is the population standard deviation with a `1e-6` floor in the channel's physical units.

For each physical channel, resolve statistics in this exact order:

1. define `epsilon_j = max(1e-6, 1e-3 * q99(abs(f0_j)))` in physical units;
2. compute provisional `q99_j = q0.99(abs(b_0.10[j] - f0[j]))` over all valid entries;
3. if provisional `q99_j == 0` but any nonzero correction exists, replace `q99_j` with the 0.99 quantile over nonzero valid corrections and record `sparse_nonzero_fallback=true`;
4. if every valid correction is exactly zero, mark the channel inactive, set `q99_j=0`, and force its predicted correction exactly to zero;
5. only after that replacement/inactive decision, set both `sigma_delta_j=max(q99_j, epsilon_j)` and `trust_scale_j=max(2*q99_j, epsilon_j)`.

The same final resolved `q99_j` must feed loss scaling and trust scaling. No scaler is refit per seed or architecture.

Masked/padded entries are zeroed after every transform. Deployment bundles contain these immutable scaler buffers. Tests must reject a consumer fed standardized rather than physical fields and a correction added to `f0` before inverse scaling.

### 12.2 Base particle representation

Reuse the existing `local_particle_residual_field` feature construction and the frozen C0 representation. The current baseline has a 160-dimensional representation, four geometry-biased attention layers, jet context, and field-group output heads. Its exposed final per-particle state `h0` is a mandatory correction input, not recomputed by a second full C0 stack.

Inputs include only HLT particle features, valid masks, HLT jet context, standardized `f0_live`, and `stop_gradient(h0)`. Conditioning on `f0_live` is mandatory because the model must know the prediction it is correcting. A projected raw-HLT feature skip is fused with `h0` and `f0_live` before local processing.

### 12.3 Target-kernel local stage

Build one deterministic capped neighbor graph around valid HLT particles. For every edge, compute the same three Gaussian bandwidths used to define the targets:

```text
w_r(i, j) = exp(-0.5 * DeltaR(i, j)^2 / r^2)
r in {0.02, 0.05, 0.10}
```

For each bandwidth, apply a lightweight shared-edge message function using relative kinematics and source/target embeddings, multiply/condition aggregation by `w_r`, and retain a separate radius-specific particle representation. Padded particles are excluded, caps are deterministic, and any finite support cutoff must be wide enough to approximate the Gaussian tail and be recorded. Hard `DeltaR <= r` neighborhoods are retained only as an ablation.

Do not fuse the three radius streams into one representation before their matching output heads. Shared cross-radius/global context may be added later, but radius identity must remain recoverable by construction.

### 12.4 HLT-anchored hierarchical regions

Reuse the seeded soft-assignment machinery under `teacher_logit_reco/multiscale_subjet_part`. The primary profile is `few_subjets`:

- 4 small-radius regions;
- 4 medium-radius regions;
- 2 large-radius regions;
- 10 region tokens total.

Seeds and assignments are derived from HLT information only. No offline axis or target residual may determine a region at inference. Empty-region handling and assignment entropy must be reported.

Region tokens pool the locally processed particle embeddings, HLT regional summary features, assignment mass, scale/radius identity, a pooled summary of `f0_live`, and projected `h0` context.

### 12.5 Global region attention

Apply two transformer layers with four heads over the region tokens. Reuse the existing subjet pair-feature and attention-bias implementation where compatible. Pair features may include HLT-computable relative angle, mass/energy relations, shared-scale identity, and assignment overlap.

Global attention is performed over approximately ten stable region tokens rather than asking dense particle attention to rediscover the entire hierarchy at every layer.

### 12.6 Particle readback and optional refinement

Every valid particle cross-attends once to the globally updated region tokens. Fuse the readback with shared particle context and each preserved radius stream. Three radius-aware heads predict the corresponding 15-channel correction blocks. The five reliability channels are copied from `f0_live` in the primary model and have no trainable correction path.

The primary `D10_A3_hlg_primary` architecture uses one region-to-particle readback. `D10_A4_hlg_refine` adds one optional refinement pass:

1. update particle embeddings using the first readback;
2. repool regions once;
3. run one additional region attention/readback block.

No unbounded recurrence is permitted.

### 12.7 Locked numerical profile for `D10_A3_hlg_primary`

The following is the canonical A3 implementation, not a tuning menu. Any change creates a new run/configuration hash.

| Component | Locked value |
|---|---|
| Maximum particle width | Read from the source manifest; production expectation 128 |
| Base fusion | raw HLT token `14 -> 64`, standardized `f0` `50 -> 64`, `h0` `160 -> 96`, HLT jet context `-> 32`; concatenate 256 and project `256 -> 160` with GELU and LayerNorm |
| Neighbor graph | 32 nearest valid non-self particles within `DeltaR <= 0.30`; ties by source particle index |
| Self policy | No self edge; a separate residual/self projection is present in every local layer |
| Edge features | `DeltaEta`, `sin(DeltaPhi)`, `cos(DeltaPhi)`, `DeltaR`, clipped log `pT` ratio, clipped log energy ratio; log ratios clipped to `[-8,8]` |
| Gaussian aggregation | Weighted mean `sum(w*m)/(sum(w)+1e-6)`, independently at 0.02/0.05/0.10 |
| Local message stack | 2 residual message layers per preserved radius stream, width 160, two-layer width-160 edge MLP, GELU, dropout 0.05, pre-LayerNorm epsilon `1e-5` |
| Region seeds | Existing `few_subjets` radii: small 4 at 0.05-0.12 using leading-`pT`, medium 4 at 0.12-0.25 using local density, large 2 at 0.25-0.50 using farthest-point; density `pT` weight 1.0, self excluded, epsilon `1e-8`, ties by particle index |
| Soft assignment | Seeded-query mode, embed width 64, hidden width 128, temperature 0.50, geometry-bias strength 2.0, scale embedding on, radius floor 0.03, dead-token threshold `1e-3`; padded particles have exactly zero mass |
| Region/token dimension | 160 |
| Region transformer | 2 pre-norm layers, 4 heads, MLP width 320, GELU, residual/attention dropout 0.05, LayerNorm epsilon `1e-5`, mask value `-1e4` |
| Pair bias | Existing 19 HLT-only pair features and scale-pair embedding, hidden width 64, one learned scalar per attention head; feature scales/clips remain `eps=1e-6`, `DeltaR scale=5`, radius scale 0.5, max log 14, max log-ratio 8 |
| Particle readback | One 4-head cross-attention block, width 160, dropout 0.05, followed by residual fusion |
| Radius correction heads | Per radius concatenate base/radius/readback (`160+160+160=480`), then `480 -> 160 -> 128 -> 64 -> 15`, GELU, dropout 0.05, zero-initialized final projection |
| Primary refinement | None; A4 adds exactly one repool/region-attention/readback pass |
| Particle-control matching | `D10_A0M_capacity_particle` within ±5% trainable correction parameters and ±10% correction FLOPs of A3 |
| Direct-control matching | Each direct classifier within ±5% total deployed parameters and ±10% end-to-end inference FLOPs of the canonical `R0 + D10_A3_hlg_primary + selected_T10` bundle, measured at batch 1, manifest width, and the same valid mask |

For local layer `ell`, initialize all three streams with `h_i,r^0=h_i,base` and use:

```text
m_ij^ell = M_ell([LN(h_i,r^ell), LN(h_j,r^ell), edge_ij])
a_i,r^ell = sum_(i,j in E) w_r(i,j) * m_ij^ell
              / (sum_(i,j in E) w_r(i,j) + 1e-6)
h_i,r^(ell+1) = h_i,r^ell
              + Dropout(U_ell([LN(h_i,r^ell), a_i,r^ell]))
```

`M_ell` is `326 -> 160 -> 160` and `U_ell` is `320 -> 160 -> 160`, both with GELU between linear layers. A layer's `M_ell/U_ell` parameters are shared across the three radii; layers 0 and 1 do not share parameters. Radius identity arises only from `w_r` and the persistent stream states before region scale embeddings are added. Directed capped edges are used exactly as produced by the deterministic graph; no implicit reverse/self edges are inserted.

Map small/medium/large regions to the 0.02/0.05/0.10 local streams. With assignment `A_ik`, define each nonempty region input as the assignment-weighted mean of matching local state (160), standardized `f0` (50), frozen `h0` (160), and raw HLT token (14), concatenated with `log1p(sum_i A_ik)`, seed `pT` fraction, and a three-way scale one-hot: 389 values total. Project `389 -> 320 -> 160` with GELU/dropout 0.05, add a learned scale embedding, and apply LayerNorm. Regions with assignment mass below `1e-3` are zeroed and masked.

After global region attention, particle readback is exactly:

```text
c_i = MHA(query=LN(h_{i,base}), key=region_tokens, value=region_tokens)
readback_i = LN(h_{i,base} + Dropout(W_o(c_i)))
```

where `W_o` is `160 -> 160`. Each radius head receives `[h_{i,base}, h_{i,r}^2, readback_i]` as the declared 480-vector. A4 reuses the same seeds and assignment weights, repools the once-updated particle states, and performs exactly one additional region-transformer/readback pass; it does not reseed or recompute assignments.

The seeded region, soft-assignment, pair-feature, and transformer portions must instantiate the existing `SubjetSeedBuilderConfig`, `SoftSubjetAssignmentConfig`, `MultiScaleSubjetPairFeatureConfig`, and `MultiScaleSubjetTransformerConfig` contracts with the table values. Their normalized configuration payloads and implementation contract versions are hashed into every A2-and-later run. The `DeltaR <= 0.30` support leaves the largest Gaussian's three-sigma neighborhood available while bounding work. Deterministic neighbor selection is recomputed from HLT geometry only. `T10` dropout is disabled on both target and live-consumer forwards. HLG dropout is active only while training `R10` and disabled for every validation, confirmation, and deployment forward.

### 12.8 Zero-initialized correction and trust region

Every final radius-head correction projection is initialized to zero, so the initial deployed path exactly reproduces `f0_live`.

For field component `j`, use the final resolved training-partition statistics from Section 12.1:

```text
trust_scale_j = max(2 * q99_j, epsilon_j)
physical_raw_delta_j = sigma_delta_j * standardized_raw_delta_j
bounded_delta_j = trust_scale_j * tanh(physical_raw_delta_j / trust_scale_j)
```

Trust scales and addition to `f0` are in physical units. A valid correction is counted as saturated when `abs(bounded_delta_j) >= 0.99 * trust_scale_j`; `tanh` is not expected to equal the asymptotic bound exactly. The primary model has no learned sigmoid gate: zero initialization and the componentwise bound already preserve the anchor, while a gate would introduce scale degeneracy and could starve early gradients. A groupwise-gated ablation is allowed; its gates initialize open and are frozen during field warm-up. A separate ablation removes the trust bound.

### 12.9 Capacity and direct-learning controls

The architecture block must include a particle-only correction model widened/deepened to match HLG parameters and approximate FLOPs. This isolates hierarchy from raw capacity. It also includes two capacity-matched direct classifiers: one from raw HLT alone, and one using the frozen HLT-only `R0` outputs `f0/h0` with the same HLG backbone. Both emit logits directly with CE and have no bridge, KD, field output, or `T10` path. Both are matched once to the canonical-A3 bundle defined in Section 12.7 and are not resized after the eventual reconstructor selection. Together they distinguish extra raw-HLT capacity, useful residual-supervised `R0` representations, and the bridge curriculum itself.

### 12.10 Absolute-output diagnostic semantics

The two absolute-output runs are deliberately non-selectable diagnostics and do not use the correction-relative trust bound. On valid `stack_train_distill` physical channels, compute `lo_j=q0.001(b_0.10[j])`, `hi_j=q0.999(b_0.10[j])`, `center_j=(lo_j+hi_j)/2`, and `half_range_j=max((hi_j-lo_j)/2, epsilon_j)`. Their physical output is:

```text
f_hat[P] = center[P] + half_range[P] * tanh(raw_absolute[P])
f_hat[S] = f0[S]
```

Final absolute-head weights and biases initialize to zero, so the initial consumer field is exactly `[center[P], f0[S]]`. `D10_A5_hlg_absolute_conditioned` receives raw HLT, `f0`, and `h0` while emitting this absolute field. `D10_A5S_hlg_scratch_physical45` receives only raw HLT in its trainable physical-45 predictor; frozen `R0` is still invoked outside that predictor solely to supply the mandatory five-channel `f0[S]` pass-through. No `f0[P]` or `h0` tensor may enter A5S. A truly R0-free all-50 scratch predictor is a different future experiment.

## 13. Frozen-consumer distillation graph

After selection, freeze `T10` completely and set it to evaluation mode. Batch normalization/statistics, stochastic depth, and dropout must not update.

The selected checkpoint hash is immutable. The same checkpoint object/function produces cached targets, the live differentiable logits, and deployed logits. There is no post-cache refit.

For each training event:

```text
z_target = stop_gradient(T10(x, b_0.10))
f_hat    = R10(x)
z_live   = T10(x, f_hat)
```

`z_target` should normally come from the provenance-checked target-logit cache. `z_live` must be computed through a live frozen `T10` forward pass so gradients flow through the tagger into `f_hat`. Freezing parameters does not mean wrapping the live forward pass in `no_grad`.

An invariant test evaluates both paths at `f_hat = b_0.10` and requires matching logits and zero KD within numerical tolerance. This test would fail if different teacher checkpoints were accidentally used.

The first proof deploys the composition `R10 + T10` directly. A separately trained HLT student may be studied later only after this composition demonstrates recovery.

## 14. Loss definitions

Let `y` be the event label and `tau = 2` the default distillation temperature.

### Logit KD

```text
L_KD = tau^2 * KL(
    softmax(z_target / tau),
    log_softmax(z_live / tau)
)
```

Use the repository's established KL direction consistently and test it against a hand-computed example.

### Event cross-entropy

```text
L_CE = CE(z_live, y)
```

This protects the live path from merely matching teacher calibration artifacts.

### Bridge-field regression

The 45 physical channels form exactly twelve loss groups. For each ordered radius block `r in {0.02,0.05,0.10}`, use block-local indices `0:5=pt_density`, `5:8=centroid`, `8:10=multiplicity`, and `10:15=composition`. Equivalently, group `(r,s)` contains `15*radius_index + semantic_offsets[s]`. The five reliability channels are never members of a primary loss group.

```text
L_bridge = masked_group_balanced_Huber(
    (f_hat[P] - b_0.10[P]) / sigma_delta[P]
)
```

The Huber transition is `delta=1.0` in correction-standardized units. First average valid entries per channel, then channels per declared field group, then give every nonempty radius/field group equal weight so multiplicity and high-variance groups do not dominate. Primary training applies no bridge loss to the five pass-through reliability channels.

### True-field regression

```text
L_true = masked_group_balanced_Huber(
    (f_hat[P] - f_true[P]) / sigma_delta[P]
)
```

This is zero-weight in the primary pilot. It appears only in a targeted ablation because optimizing all the way to `f_true` can conflict with the deliberately small-step curriculum.

### Anchor regularization

```text
L_anchor = mean_valid((f_hat[P] - f0_live[P])^2 / (trust_scale[P]^2 + 1e-12))
```

### Gate regularization

`L_gate` exists only for `D10_A9_hlg_group_gate`. From each radius head's 160-dimensional fused state, a `160 -> 4` linear head predicts one gate for each semantic group at that radius. Gate-head weights initialize to zero and every bias initializes to `logit(0.95)=2.944438979`; therefore `g_i,r,s=sigmoid(a_i,r,s)` begins at 0.95. Broadcast the gate over that group's channels and apply it after the componentwise trust `tanh`:

```text
gated_delta_i,r,s = g_i,r,s * bounded_delta_i,r,s
L_gate = mean_valid_radius_semantic_groups((1 - g_i,r,s)^2)
```

Gate parameters are frozen for every field-warm-up step and unfreeze at the first Phase 2 optimizer step. A9 alone uses coefficient `0.005 * L_gate`; the primary architecture has no gate and records the component/coefficient as exactly zero. Report pre/post-gate correction norms, gate means, and gate saturation by all twelve groups and particle rank.

### Local smoothness

Let `u=(f_hat[P]-f0[P])/sigma_delta[P]` be the correction in correction-standardized units. Use the same directed, non-self, capped graph `E` and radius-specific Gaussian `w_r` declared in Section 12.7. For each nonempty radius×semantic group `g=(r,s)`, define:

```text
smooth_g = sum_(i,j in E) w_r(i,j) * ||u_i,g - u_j,g||_2^2
           / (channels_g * sum_(i,j in E) w_r(i,j) + 1e-6)
L_smooth = mean_(nonempty g)(smooth_g)
```

Edges touching padding are absent, and no term is formed for the five reliability channels. The graph/weight utility is implemented with the C0 losses in Step 5 and reused unchanged by A1-and-later architectures; C0 therefore does not depend on the later HLG module merely to compute this loss.

### Primary objective

The predeclared primary objective is:

```text
L_primary =
    1.00 * L_KD
  + 0.50 * L_CE
  + 0.20 * L_bridge
  + 0.00 * L_true
  + 0.02 * L_anchor
  + 0.00 * L_gate
  + 0.01 * L_smooth
```

Weights are initial defaults, not hidden tuning results. Raw and weighted component magnitudes must be logged so a term that is numerically irrelevant or overwhelming is immediately visible.

## 15. Training schedule

The primary schedule has two phases:

### Phase 1: field warm-up

- train the correction branch with `L_bridge + L_anchor + L_smooth`;
- keep `R0` and `T10` frozen;
- run exactly the immutable manifest's `field_warmup_steps`, identical across every warm-up-enabled comparison; validation loss is logged but cannot shorten or extend Phase 1;
- treat a non-finite loss/gradient or failed numerical invariant as a failed replica that never enters Phase 2, not as an early but valid warm-up completion;
- preserve the exact zero-correction checkpoint as a diagnostic.

### Phase 2: frozen-consumer distillation

- enable the declared loss combination for the run;
- keep `R0` and `T10` frozen;
- select epochs only with the exact `model_val_stop` rule below;
- evaluate the selected per-seed checkpoint once on `model_val_select`; never open either `stack_val` child during training or checkpoint selection.

Per-seed checkpoint choice is deterministic. For every post-teacher correction run, evaluate the run's bound consumer (`selected_T10`, `T10_all50`, or alternate `T10` as declared) on `f_hat`, let `best_accuracy` be the maximum `model_val_stop` accuracy, form the epoch pool with `best_accuracy-accuracy <= 0.0001`, choose the lowest event cross-entropy in that pool, and resolve cross-entropy differences `<=1e-6` with the earlier epoch. KD agreement is diagnostic and never selects a checkpoint.

Upstream A0/consumer runs and both direct classifiers apply the same accuracy-pool/cross-entropy/earliest rule to their own logits. Early `D10_L0_bridge_only` is the sole exception because no selected teacher exists: minimize standardized `L_bridge`, admit epochs satisfying `loss-best_loss <= max(1e-8, 1e-4*abs(best_loss))`, then choose lower normalized bridge MSE and finally the earlier epoch. Early-stopping patience for each family advances only when its first criterion improves beyond the declared tolerance.

The `KD-only` and `CE-only` ablations skip field warm-up unless explicitly marked otherwise; otherwise their names would not describe their actual supervision. The primary combined runs use warm-up.

Optimizer, learning-rate range, batch size, gradient clipping, mixed precision, and patience should inherit stable repository defaults. Any deviation is part of the run manifest. All comparisons in a block receive the same number of optimization steps or use the same clearly recorded early-stopping rule. Optimizer/scheduler state and rotating resume checkpoints live in verified job-local RAM. The three paired replicas publish all metrics and only the median replica's weights after aggregate evaluation.

## 16. Ablation campaign

All selectable post-teacher correction ablations use the same selected `T10`, virtual physical-45 bridge recipe, split manifests, initial `R0`, paired seed set, and evaluation code. Early `D10_L0_bridge_only` is trained before a teacher exists, then binds the selected `T10` only for its common deployable evaluation and possible final eligibility. Runs with nonzero KD use the same target-logit cache; zero-KD runs do not load privileged logits merely for uniformity. Only the named factor may change. Direct classifiers have no teacher path, and the all-50 semantic experiment has its own explicitly non-primary consumer, scaler, cache, and recipe lineage.

### 16.1 Loss ablations

The baseline architecture for this block is `D10_A0_c0_delta`, a simple particle correction branch conditioned on frozen `f0`, frozen `h0`, and the raw-HLT skip, with zero-initialized physical-45 delta/trust heads and no hierarchy.

| Run ID | KD | CE | Bridge regression | Other regularizers | Purpose |
|---|---:|---:|---:|---|---|
| `D10_L0_bridge_only` | 0 | 0 | 1.00 | anchor, smooth | Early physical-45 reachability diagnostic and direct transfer test |
| `D10_L1_ce_only` | 0 | 1.00 | 0 | none | Does end-task supervision alone find a useful field? |
| `D10_L2_kd_only` | 1.00 | 0 | 0 | none | Pure frozen-logit imitation requested by the hypothesis |
| `D10_L3_kd_ce` | 1.00 | 0.50 | 0 | none | Does label supervision stabilize KD? |
| `D10_L4_kd_bridge` | 1.00 | 0 | 0.20 | anchor, smooth | KD plus local physical target, without labels |
| `D10_L5_ce_bridge` | 0 | 0.50 | 0.20 | anchor, smooth | No teacher-logit supervision |
| `D10_L6_kd_ce_bridge` | 1.00 | 0.50 | 0.20 | none | Three main signals without regularization |
| `D10_L7_plus_anchor` | 1.00 | 0.50 | 0.20 | anchor | Isolate prediction-anchor regularization |
| `D10_L8_full_c0` | 1.00 | 0.50 | 0.20 | anchor, smooth | Add smoothness to form the primary C0 objective |
| `D10_L9_full_true_target` | 1.00 | 0.50 | 0.20 plus `0.05 * L_true` | primary full set | Does a weak pull toward full truth help or break the small step? |
| `D10_L10_no_trust` | 1.00 | 0.50 | 0.20 | primary except bounded trust | Test the correction trust region |

For `L0`, `L4` through `L10`, field warm-up follows Section 15. For `L1` through `L3`, training begins directly with the named objective.

### 16.2 Architecture ablations

This block uses `L_primary` for every correction run. The two direct classifiers necessarily use supervised CE with a matched training budget because they have no frozen-consumer or field output; that difference is part of the control definition, not an accidental loss change.

| Run ID | Architecture | Isolated question |
|---|---|---|
| `D10_A0_c0_delta` | Simple `f0+h0+raw-HLT` particle correction branch | Strong simple baseline |
| `D10_A0M_capacity_particle` | Widened/deeper particle-only correction matched to A3 parameters/FLOPs | Does HLG beat matched capacity? |
| `D10_A1_multiscale_local` | A0 plus three Gaussian target-kernel local streams | Does target-aligned locality help? |
| `D10_A1H_hard_radius` | A1 with hard 0.02/0.05/0.10 neighborhoods | Do Gaussian kernels matter? |
| `D10_A2_regions_no_global` | A1 plus HLT-seeded regional pooling/readback, no region transformer | Does hierarchy alone help? |
| `D10_A3_hlg_primary` | A2 plus two-layer global region attention and radius-aware heads | Primary local → hierarchical → global design |
| `D10_A4_hlg_refine` | A3 plus one repool/readback refinement pass | Is one iterative refinement worth its cost? |
| `D10_A5_hlg_absolute_conditioned` | A3 emits an absolute physical-45 field while still conditioned on `f0/h0/raw-HLT` | Is correction parameterization itself important? |
| `D10_A5S_hlg_scratch_physical45` | A3 HLG topology with all trainable `f0/h0` inputs removed emits absolute physical-45 fields from raw HLT only; frozen R0 supplies only the five-channel pass-through outside the predictor | Can the hierarchy predict the physical field without `f0/h0` conditioning? |
| `D10_A6_hlg_no_pair_bias` | A3 without region pair attention biases | Do explicit inter-region relations add value? |
| `D10_A7_hlg_no_h0` | A3 without frozen `R0` hidden features, retaining raw HLT and `f0` | Does representation reuse help? |
| `D10_A7F_hlg_no_f0` | A3 without `f0` conditioning, retaining raw HLT and `h0` | Must the model see the prediction it corrects? |
| `D10_A7X_hlg_no_raw_skip` | A3 without the raw-HLT skip, retaining `f0/h0` | Does `h0` form an information bottleneck? |
| `D10_A8_hlg_fused_radius_heads` | A3 fuses radius streams before one 45-channel head | Does preserving radius identity help? |
| `D10_A9_hlg_group_gate` | A3 with open, warm-up-frozen groupwise correction gates | Does gating help after removing its optimization hazard? |
| `D10_AS_hlg_regions_2_2_1` | A3 with 2/2/1 regions | Is the primary hierarchy unnecessarily granular? |
| `D10_AL_hlg_regions_8_8_4` | A3 with 8/8/4 regions | Is the primary hierarchy under-resolved? |
| `D10_AFIX_hlg_fixed_assignment` | A3 with deterministic fixed nearest-seed assignment instead of learned soft assignment | Is learned assignment useful? |
| `D10_ASAME_hlg_same_scale_only` | A3 masks global attention across scales | Does cross-scale reasoning add value? |
| `D10_AGLOBAL_hlg_one_global_token` | A3 pools the ten regions to one global token before readback | Are ten persistent region tokens necessary? |
| `A0_CAP500_direct_hlt` | Capacity-matched raw-HLT classifier on the full 500k union, with HLG processing and no `R0`/field/KD/`T10` path | Is gain due simply to added raw-HLT capacity? |
| `A0_CAP500_r0rep_direct` | Same HLG classifier using frozen `R0`'s HLT-only `f0/h0`, emitting logits directly with CE | Does `R0` representation learning explain the gain? |

`D10_L8_full_c0` and `D10_A0_c0_delta` are the same run if their complete manifests match; the campaign manager must deduplicate them rather than consume two jobs.

### 16.3 Loss × architecture interactions

The C0 loss sweep alone cannot establish that the same supervision behaves similarly in HLG. Run the following locked A3 subset:

| Run ID | A3 supervision/schedule | Purpose |
|---|---|---|
| `D10_XA3_bridge_only` | Bridge only, ordinary field warm-up | Can hierarchy improve direct bridge recovery? |
| `D10_XA3_ce_only` | CE only, no field warm-up | Can end-task supervision use A3 without privileged logits? |
| `D10_XA3_kd_only` | KD only, no field warm-up | Can A3 recover teacher behavior without field supervision? |
| `D10_XA3_kd_bridge` | KD + bridge + anchor + smooth | Does local supervision stabilize A3 KD? |
| `D10_XA3_kd_ce` | KD + CE, no field warm-up | Do labels stabilize A3 KD without a field target? |
| `D10_XA3_full_primary` | Exact A3 primary objective and schedule | Canonical A3; deduplicated with `D10_A3_hlg_primary` |
| `D10_XA3_full_no_warmup` | Primary objective from step 0 | Does warm-up cause the gain? |
| `D10_XA3_full_no_smooth` | Primary objective with `L_smooth=0` | Does the local smoothness prior help A3? |

### 16.4 Bridge-channel semantics

The primary consumer and every selectable deployment use `physical45`. Train one explicit all-50 consumer from the same `Tpred` initialization, select its median seed within its own recipe (without competing with the primary selector), and bind it in `teacher_binding_all50.json`. Run two deliberately different diagnostics using the already locked canonical `D10_A3_hlg_primary` architecture and the all-50 cache:

```text
T10_all50_clean
D10_B1_all50_fullhead
D10_B2_all50_physical45_only
```

`D10_B1_all50_fullhead` adds a separate `160 -> 64 -> 5` zero-initialized head, treating the five diagnostic coordinates as continuous only for this ablation and applying the same standardized-Huber/physical-trust rule without thresholding flags. It can therefore reproduce the complete all-50 bridge in principle. Its five extra correction scales/trust statistics are fit only on `stack_train_distill` and live exclusively in the all-50 binding. `D10_B2_all50_physical45_only` keeps the primary 45-channel correction and five-channel `f0` pass-through while imitating the all-50 teacher; its irreducible target/live mismatch is intentional and reported. The exact-zero-KD-at-equal-field invariant applies to B1 but is explicitly inapplicable to B2 because its live graph cannot supply the privileged five-channel endpoint.

Report both gains separately as shortcut-risk diagnostics. Neither can replace the primary selected teacher or become the selected deployment in this pilot. Compare them with a reliability-only response diagnostic to reveal whether the derived five channels dominate.

### 16.5 Alternate selected-teacher check

After the primary robust consumer is selected, train a matched simple/strong
architecture pair against the non-selected clean bridge consumer:

```text
D10_TALT_A0
D10_TALT_A3
```

`D10_TALT_A0` uses the same simple particle-only C0 graph and primary objective
as the robust-path A0 run; `D10_TALT_A3` uses the same canonical HLG graph and
primary objective as the robust-path A3 run. Together they test whether robust
bridge-consumer training changes distillability at both ends of the architecture
range. The
alternate recipe's complete `model_val_select` validity report and failed rules
remain attached to its predeclared median checkpoint, but performance-rule
failure does not suppress the run. The checkpoint is bound in
`teacher_binding_alt.json` with an explicit quality-warning override and has its
own cache namespace. Invalid provenance, a missing or changed checkpoint, or
non-finite confirmation metrics still fail the run. The alternate may not
retroactively alter the primary selection rule.

### 16.6 Trained negative controls

After a valid bridge teacher exists, train four separate, permanently non-selectable controls:

```text
D10_N0_shuffled_logit_kd
D10_N1_shuffled_bridge_field
D10_N2_shuffled_primary
D10_N3_nonprivileged_teacher_kd
```

`N0` is KD-only against event-shuffled privileged logits. `N1` is bridge-only against event-shuffled bridge corrections with groupwise marginal scale preserved. `N2` uses the primary objective with logits and bridge corrections shuffled by the same wrong-event map. `N3` uses the dedicated `physical45_selected_teacher_on_f0_control` cache from the selected checkpoint evaluated on `f0`, not `b_0.10`, and therefore measures non-privileged self-distillation. Each control has an explicit recipe/hash, paired seeds, and identical model/step budget to its corresponding positive run.

### 16.7 Campaign profiles and generated job count

The campaign registry is the sole source of truth for run IDs, deduplication, conditional parents, seed replicas, publication policy, and counts. Every row must explicitly set `scientific_role` and `selectable_for_primary_deployment`; there is no default value. Aliases inherit these fields from their canonical row, and registry construction fails if an ID is omitted from the following locked policy:

| Registry IDs | Scientific role | Selectable for primary deployment |
|---|---|---|
| `D10_L0_bridge_only`, `D10_L1_ce_only`, `D10_L2_kd_only`, `D10_L3_kd_ce`, `D10_L4_kd_bridge`, `D10_L5_ce_bridge`, `D10_L6_kd_ce_bridge`, `D10_L7_plus_anchor`, `D10_L8_full_c0`, `D10_L9_full_true_target` | Confirmatory physical-45 loss candidates | yes |
| `D10_L10_no_trust` | Unbounded safety diagnostic | no |
| `D10_A0_c0_delta`, `D10_A0M_capacity_particle`, `D10_A1_multiscale_local`, `D10_A1H_hard_radius`, `D10_A2_regions_no_global`, `D10_A3_hlg_primary`, `D10_A4_hlg_refine`, `D10_A6_hlg_no_pair_bias`, `D10_A7_hlg_no_h0`, `D10_A7F_hlg_no_f0`, `D10_A7X_hlg_no_raw_skip`, `D10_A8_hlg_fused_radius_heads`, `D10_A9_hlg_group_gate` | Confirmatory bounded physical-45 architecture candidates | yes |
| `D10_A5_hlg_absolute_conditioned`, `D10_A5S_hlg_scratch_physical45` | Absolute-output diagnostics | no |
| `D10_AS_hlg_regions_2_2_1`, `D10_AL_hlg_regions_8_8_4`, `D10_AFIX_hlg_fixed_assignment`, `D10_ASAME_hlg_same_scale_only`, `D10_AGLOBAL_hlg_one_global_token` | Exploratory hierarchy diagnostics | no |
| `D10_XA3_bridge_only`, `D10_XA3_ce_only`, `D10_XA3_kd_only`, `D10_XA3_kd_bridge`, `D10_XA3_kd_ce`, `D10_XA3_full_primary`, `D10_XA3_full_no_warmup`, `D10_XA3_full_no_smooth` | Confirmatory bounded A3 loss/schedule candidates | yes |
| `A0_CAP500_direct_hlt`, `A0_CAP500_r0rep_direct`, `A0_C250`, `A0_C250_LONG`, `A0_S500`, `Tpred`, `Tpred_continue`, `T10_clean`, `T10_robust`, `T10_all50_clean` | Baseline or teacher controls | no |
| `D10_B1_all50_fullhead`, `D10_B2_all50_physical45_only` | All-50 semantic diagnostics | no |
| `D10_TALT_A0`, `D10_TALT_A3` | Matched simple/strong alternate-teacher diagnostics | no |
| `D10_N0_shuffled_logit_kd`, `D10_N1_shuffled_bridge_field`, `D10_N2_shuffled_primary`, `D10_N3_nonprivileged_teacher_kd` | Negative controls | no |

These are complete IDs, not prefix rules. A new configuration must amend this table before any result exists. Neither `D10_TALT_A0` nor `D10_TALT_A3` may replace the primary teacher or enter the primary bundle in this pilot. The no-trust run records saturation as `not_applicable` and is excluded instead of receiving an invented alternate safety gate. Documentation/tests render counts from the registry; no submission script contains an independently maintained numeric total.

The current maximum configuration inventory is:

- 5 bridge-consumer runs including `T10_all50_clean`;
- 3 additional data/optimization-matched A0 runs (`A0_C250`, `A0_C250_LONG`, `A0_S500`); the capacity-matched A0 is counted in the architecture block;
- 11 loss configurations;
- 21 additional architecture configurations after `D10_A0_c0_delta`/L8 deduplication;
- 7 additional A3 loss-interaction configurations after canonical-A3 deduplication;
- 2 all-50 diagnostic reconstructors;
- 2 conditional alternate-teacher runs;
- 4 trained negative controls;
- 47 reconstruction-breadth configurations including early `D10_L0_bridge_only`;
- 46 post-teacher B6 configurations because L0 launches alongside B3;
- 55 maximum model-training configurations including the 8 upstream A0/consumer configurations, excluding `R0`, RAM/source audits, and logit jobs.

If the campaign trains rather than registers `R0`, record that additional
prerequisite training job separately. Both physical-45 consumers are trained
and provenance-audited before B4, so both TALT rows are reserved and runnable in
the production pilot. Registry cardinality and maximum executed configuration
count are both 55.

The required scientific profile is `paired3`: all valid configurations use seed IDs `{101,202,303}`, comparisons use paired seed IDs, selectors rank aggregate configuration results, and only median-replica weights are published. A one-seed `breadth_debug` profile may validate scheduling and tensor paths but is permanently non-selectable and cannot support scientific conclusions. GPU availability is not used as a reason to omit an informative ablation; persistent storage is controlled by metrics-only non-median replicas rather than by removing experiments.

## 17. Metrics and diagnostics

All classification metrics use the repository's established label conventions and sample weights. At minimum report discrimination, calibration, reconstruction, and resource metrics.

### 17.1 Bridge-consumer metrics

- multiclass cross-entropy and accuracy;
- one-vs-rest ROC AUC and background rejection at declared signal efficiencies;
- confusion matrix and classwise efficiency;
- expected calibration error and Brier score;
- response curve versus `rho` for the same consumer;
- negative-control response;
- results versus particle multiplicity, jet `pT`, jet mass, eta, matching quality, and flavor/process slice.

### 17.2 Reconstructor metrics

- all deployable classification metrics for `T10(x, f_hat)`;
- KD loss, logit cosine similarity, classwise logit bias, and prediction agreement with `T10(x, b_0.10)`;
- physical-45 group-balanced Huber/RMSE to `b_0.10`, `f0`, and `f_true`;
- out-of-sample explained variance, normalized MSE, Pearson/Spearman correlation, and cosine alignment to `0.10 * Delta_true[P]`, by radius and field group;
- correction magnitude divided by bridge-delta magnitude;
- fraction of corrections at the trust bound;
- gate statistics only for the declared gated ablation;
- correction-to-bridge alignment, per-group correction norms, and distance from the training bridge distribution;
- stability of logits and accuracy under predeclared small field perturbations;
- local smoothness and correction correlation by target radius;
- region assignment entropy, empty-region rate, token attention maps, and readback mass for hierarchical runs;
- comparison with the capacity-matched particle-only and direct-HLT controls;
- parameter count, approximate FLOPs, peak accelerator/host/RAM-workspace memory, training throughput, and inference latency;
- persistent bytes published, peak job-local bytes, RAM hit/miss/eviction telemetry, and confirmation that no dense derived field was written.

The small-field perturbation audit is fixed. For deterministic seeds `{9101,9102,9103,9104}`, draw independent standard-normal `eta` keyed by `(audit_seed, event_identity, valid_particle_index, physical_channel)`, set padding perturbations to zero, and apply:

```text
perturbation[P] = clip(0.05 * sigma_delta[P] * eta,
                       -0.10 * trust_scale[P],
                        0.10 * trust_scale[P])
f_hat_perturbed[P] = f_hat[P] + perturbation[P]
f_hat_perturbed[S] = f_hat[S]
```

For audit seed `s`, define `accuracy_loss_s = accuracy(T10(x,f_hat)) - accuracy(T10(x,f_hat_perturbed_s))`; a negative value is retained as an improvement, not clipped. Report accuracy/cross-entropy/logit-cosine changes for every seed, the arithmetic mean of the four accuracy losses, and `max_s accuracy_loss_s`. A selectable run passes only when mean accuracy loss is at most `0.002` and the maximum is at most `0.003`.

Correction-to-bridge cosine and groupwise distribution distances remain mandatory diagnostics and must be finite, but they are not automatic selection thresholds in this pilot. For the latter, persist only 1,001 float64 training-reference quantiles per physical channel of `(b_0.10-f0)/sigma_delta`; on an authorized validation split compute the same quantiles for `(f_hat-f0)/sigma_delta`, take the mean absolute quantile difference per channel, then average channels within each of the twelve groups. This is a compact one-dimensional Wasserstein approximation, not a dense field cache. Arbitrary post hoc alignment cutoffs would be another form of validation tuning, and neither diagnostic is evaluated on final-test.

### 17.3 Gain and recovery definitions

For a metric `M` where larger is better, compute on the same split and same frozen consumer:

```text
teacher_bridge_gain = M(T10(x, b_0.10)) - M(T10(x, f0))
deployable_gain     = M(T10(x, f_hat))   - M(T10(x, f0))
recovery_fraction  = deployable_gain / teacher_bridge_gain
```

If `teacher_bridge_gain <= 0`, `recovery_fraction` is undefined and must be recorded as `null`, never forced to zero or interpreted as success. For losses, use an explicitly sign-corrected improvement definition.

Confidence intervals must be paired by event. Use deterministic paired bootstrap resampling for headline differences and store the bootstrap seed and event manifest hash.

## 18. Gates and model selection

This is a ten-class pilot. The locked primary selection metric is `model_val_select` accuracy, higher is better, matching the existing local residual-field tagger protocol. `macro_per_class_accuracy` and cross-entropy are secondary guard/tie metrics. Binary FPR at 50% signal efficiency may be reported for explicitly defined binary slices but does not select this ten-class campaign. Every metric below is first computed per paired seed and then aggregated as an arithmetic mean with sample standard deviation; a selector never ranks the best individual seed.

### 18.1 Bridge-teacher quality diagnostics and continuation

Before distillation jobs are submitted, compute and publish all of the following
quality rules for each `T10` recipe:

1. mean `Delta_same = accuracy(T10(b_0.10)) - accuracy(T10(f0)) > 0` on `model_val_select`;
2. either the paired-bootstrap 90% lower bound for aggregate `Delta_same` is positive or mean `Delta_same >= 0.001` (0.1 percentage points);
3. mean `accuracy(T10(b_0.10)) - accuracy(Tpred_continue(f0)) > 0` on `model_val_select`;
4. at least two of three seed replicas have positive `Delta_same`, and the predeclared median replica itself has positive `Delta_same`;
5. mean `accuracy(T10(f0))` is no more than `0.002` below mean `accuracy(Tpred_continue(f0))` on `model_val_select`;
6. among the four adjacent response intervals, at most one may decrease by more than `0.0005`, and the `rho=0.10` point must be within `0.0005` of the maximum response accuracy;
7. each of the five Section 10 control gains is at most `max(0.00025, 0.25 * Delta_same)` on `model_val_select`;
8. checkpoint, physical-45 recipe, split, RAM audit, and response artifacts are valid.

Rules 1-7 are scientific quality diagnostics, not execution gates. Their
failure is recorded verbatim in the selection, binding, cache, and reports and
does not suppress downstream training. Rule 8 remains a hard integrity gate.
The production continuation policy explicitly selects `T10_robust` as the
primary consumer and retains `T10_clean` as the alternate comparison; this is a
declared preference, not a guessed fallback. The fixed median-replica rule in
Section 11 still chooses each recipe's checkpoint. Recipe, seed, checkpoint,
quality-policy name, eligibility boolean, and every failed rule are written to
the pre-confirmation artifact before `stack_val_consumer` is opened.

On `stack_val_consumer`, negative `Delta_same` is likewise a recorded quality
warning rather than a stop. Confirmation authorizes continuation whenever all
metrics are finite and provenance remains valid. Non-finite metrics, invalid
provenance, stale hashes, or a missing/changed checkpoint remain hard failures.
`stack_val_consumer` never ranks recipes, breaks ties, changes the selected
seed, or selects a runner-up. Full uncertainty and effect sizes remain
mandatory.

### 18.2 Reconstructor validity gate

A reconstructor configuration aggregate may enter final selection only when its immutable registry row has `selectable_for_primary_deployment=true` and it has:

- positive mean `deployable_gain` on `model_val_select`;
- at least two of three replicas with positive `deployable_gain` and a positive median-replica gain;
- positive `recovery_fraction` where defined;
- no provenance, leakage, mask, or reload-audit failure;
- no correction to the five primary pass-through reliability channels;
- no more than 1% of valid physical corrections at the trust bound;
- finite bridge-alignment and distribution-distance diagnostics as numerical-sanity checks, not ranking terms;
- mean accuracy loss no more than `0.002` and worst-seed loss no more than `0.003` under the exact four-seed small-field perturbation audit;
- published persistent bytes within the run reservation.

Failure of C0 does not block HLG training. These rules determine scientific validity and final selectability, not whether the breadth campaign receives GPU time. Non-primary all-50 and negative-control configurations remain non-selectable regardless of their metrics.

### 18.3 Final configuration selection

Filter first to valid registry rows with `selectable_for_primary_deployment=true`. Define `best_score=max(mean model_val_select deployable accuracy)` once and the order-independent `tie_pool={config: best_score-score <= 0.0005}`. Within that fixed pool prefer higher mean macro per-class accuracy, lower mean cross-entropy, lower across-seed deployable-gain standard deviation, fewer exact total deployed parameters, and finally the lexicographically smaller run ID. Inference latency remains a reported resource diagnostic but never changes selection because it is hardware- and load-dependent. Within the selected configuration, sort replicas ascending by `(model_val_select deployable accuracy, deployable gain, negative cross-entropy, seed ID)` and retain the middle replica.

The selector writes and hashes a pre-confirmation deployable manifest containing the configuration aggregate, paired seeds, median seed, epoch, checkpoint, scaler, teacher, and recipe hashes before `stack_val_deploy` is opened. The exact median checkpoint is then evaluated once on `stack_val_deploy`; it passes only with non-negative deployable gain, finite metrics, and valid provenance. Failure is the final pilot result and cannot promote a runner-up. After confirmation, the selector writes the locked deployable manifest before final-test. No confirmation or final-test result may change the selected run, seed, epoch, threshold, or teacher.

`A0_C250_LONG`, `A0_S500`, `A0_CAP500_direct_hlt`, and `A0_CAP500_r0rep_direct` are reported beside the selected model. They cannot masquerade as bridge recovery, but they determine whether the result survives matched consumer optimization, full 500k labeled data, raw-capacity matching, and direct use of `R0` representations.

## 19. Extension to the `rho` ladder

Only after the `0.10` pilot shows repeatable positive HLT-only recovery should the ladder extend:

```text
rho = 0.20 -> 0.30 -> 0.40 -> 0.50
```

The default bridge definition at every rung remains anchored to the original ordinary predictor:

```text
b_rho[P] = f0[P] + rho * (f_true[P] - f0[P])
b_rho[S] = f0[S]
```

This keeps rung meaning comparable. A moving-anchor curriculum such as `f_hat_previous + step * (f_true - f_hat_previous)` is a separate experiment because errors and model lineage then alter the target itself.

At each new rung:

1. warm-start the new privileged consumer from the previously selected consumer;
2. train it on the new bridge, plus a matched continued-training control on the old bridge;
3. select and freeze the consumer using same-consumer response curves;
4. initialize the new reconstructor from the previous selected reconstructor;
5. distill the new frozen target with the declared breadth ablations or a version explicitly reduced using the completed `rho=0.10` evidence;
6. stop the ladder if teacher gain disappears, HLT-only recovery reverses on confirmation data, or corrections become unstable.

Each rung writes its own selected-consumer artifact and explicit parent hashes. No job may infer lineage from directory names.

## 20. Reports

Generate three separate report sections or tables.

### 20.1 Baselines and deployable models

Contains the legacy A0 row, `A0_C250`, `A0_C250_LONG`, `A0_S500`, `A0_CAP500_direct_hlt`, `A0_CAP500_r0rep_direct`, `Tpred(x, f0)`, `Tpred_continue(x, f0)`, selected `T10(x, f0)`, and selected `T10(x, f_hat)` configurations. `T10(x, f0)` is the required denominator for bridge recovery. Every row must pass the HLT-only reload audit, and every A0 row states its exact training-manifest hash, unique jet count, optimizer-step budget, and three-seed aggregate.

### 20.2 Privileged/oracle diagnostics

Contains `T10(x, b_rho)`, the physical-45 ceiling `T10(x, [f_true[P], f0[S]])`, the full-50 oracle diagnostic `T10(x, f_true)`, zero-field diagnostics, all-50/alternate teacher lineages, and negative controls. These rows are clearly watermarked as non-deployable. The two ceiling rows must not be conflated because only the former preserves the primary pass-through semantics.

### 20.3 Ablation evidence

Contains paired differences for all loss, loss×architecture, bridge-channel, kernel, representation-reuse, hierarchy, head, gate, capacity, direct-HLT, and negative-control runs; resource/storage metrics; all three per-seed metrics, aggregate statistics, retained-median IDs, and failure states. Missing/failed/conditional-skipped runs remain visible rather than disappearing from the table. Exploratory hierarchy variants are labeled exploratory and cannot retroactively redefine the confirmatory selector.

The report must show both absolute performance and bridge-recovery fraction. A high recovery fraction of a negligible or negative bridge gain is not useful and must not be highlighted as a success.

## 21. Checkpoint packaging

Rotating training checkpoints may reference frozen parents and RAM recipes while their job is active. They remain job-local and are not published after successful completion. Persistent ablation artifacts contain only trainable best weights plus parent hashes; deployable packages contain one copy of each required model and no privileged dependency.

A deployable package includes:

- `R0` weights if embedded as a frozen submodule;
- HLG/correction weights and trust-scale buffers;
- selected frozen `T10` weights;
- HLT preprocessing and residual normalization statistics;
- target schema and class-order metadata;
- architecture/run manifest and parent hashes;
- a single HLT-only inference entry point.

It excludes:

- `f_true` and bridge fields;
- materialized bridge/response/control tensors;
- cached target logits;
- oracle/privileged checkpoint loaders;
- offline matching inputs.

The packaging test must copy or load the deployable bundle in an environment where privileged paths are absent, run a fixed HLT-only batch, and compare logits with the source checkpoint.

## 22. Proposed implementation surfaces

Reuse existing modules where possible. Suggested new files are organizational, not a mandate to duplicate working code.

### Python modules

- `teacher_logit_reco/local_particle_residual_field/bridge.py`
  - physical-45/all-50 virtual recipe contracts and batchwise construction;
  - response points and deterministic controls;
  - provenance validation.
- `teacher_logit_reco/local_particle_residual_field/bridge_ram.py`
  - verified single-node allocation-wide RAM workspace and locked byte ledger;
  - one-open-per-allocation NPZ staging, non-evictable 8,192-jet raw RAM shards, and rank partitioning;
  - streamed truth generation and frozen-R0 `f0/h0` provider;
  - derived-only bounded LRU/regeneration, matched block/bin shuffle, byte reservations, telemetry, and cleanup.
- `teacher_logit_reco/local_particle_residual_field/bridge_consumer.py`
  - clean/robust bridge input sampler;
  - same-consumer/matched-compute evaluator and numerical selector.
- `teacher_logit_reco/local_particle_residual_field/hierarchical_reconstructor.py`
  - Gaussian target-kernel local messages and preserved radius streams;
  - seeded regions, global region transformer, and particle readback;
  - locked A3 numerical profile, immutable scalers, and radius-aware zero-initialized physical-45 correction/trust heads.
- `teacher_logit_reco/local_particle_residual_field/bridge_distillation.py`
  - frozen-consumer live/target graph;
  - loss registry and metrics.
- `teacher_logit_reco/local_particle_residual_field/bridge_train.py`
  - phase scheduling, in-allocation RAM-local resume state, weights-only publication, and deployable export.
- `teacher_logit_reco/local_particle_residual_field/bridge_campaign.py`
  - generated 55-configuration maximum registry, 46-configuration post-teacher matrix, explicit selectability, conditional skips, paired seeds, deduplication, provisional/measured storage accounting, selector artifacts, and reports.

### Command-line entry points

- `scripts/audit_prediction_anchored_bridge_inputs.py`
- `scripts/write_prediction_anchored_bridge_recipe.py`
- `scripts/train_prediction_anchored_bridge_consumer.py`
- `scripts/select_prediction_anchored_bridge_consumer.py`
- `scripts/cache_prediction_anchored_bridge_logits.py`
- `scripts/train_prediction_anchored_bridge_reconstructor.py`
- `scripts/run_prediction_anchored_bridge_campaign.py`
- `scripts/evaluate_prediction_anchored_bridge_campaign.py`

Every CLI supports `--dry-run` or an equivalent manifest-only mode. Training commands validate parents before allocating an accelerator.

### Slurm entry points

- `sbatch/run_prepare_prediction_anchored_bridge_ram.sh`
- `sbatch/run_train_prediction_anchored_bridge_consumer.sh`
- `sbatch/run_cache_prediction_anchored_bridge_logits.sh`
- `sbatch/run_train_prediction_anchored_bridge_reconstructor.sh`
- `sbatch/submit_prediction_anchored_bridge_pilot.sh`

Tigris scripts must:

```bash
export PYTHONNOUSERSITE=1
```

and use the full account string:

```text
reu-aisocial
```

The submission script records every job ID and dependency. Packed training allocations request exactly one node; rank 0 is the sole staging/ledger leader, and every GPU rank joins the same allocation-wide RAM ledger. Consumer selection must be an `afterok` dependency for logit caching and reconstruction. Performance-rule failures publish warnings and continue; only hard integrity failures prevent downstream submission or cause downstream jobs to fail closed with an explicit stopped-campaign status. Every job requests and preflights host memory, creates a verified allocation/rank workspace, reserves persistent output bytes before training, and publishes no dense field tensor. Manifests record `cross_allocation_resume=false`; a failed/preempted allocation restarts the entire three-seed configuration.

## 23. Required tests

Tests should be small, deterministic, and CPU-capable unless explicitly marked as integration tests.

### Bridge tests

- exact physical-45 endpoint identities and unchanged five-channel pass-through;
- exact `rho = 0.10` physical-45 interpolation and separate all-50 semantics;
- mask preservation and non-finite rejection;
- event/particle order mismatch rejection;
- deterministic negative controls and scale-preservation checks;
- wrong-event controls are within-class/block, use frozen HLT multiplicity/`pT`/mass bins, are true derangements, and share the declared map across logits/fields;
- virtual recipe/hash round trip without a dense bridge file;
- streamed truth and `R0` regeneration agree with direct computation;
- compressed sources are opened/hashed once per allocation and staged-array hashes/rank ownership are checked;
- packed jobs reject multi-node allocation; allocation-wide locking prevents combined rank oversubscription;
- raw RAM shards are non-evictable, only derived shards enter the LRU, regeneration never reopens persistent NPZ, and 20% headroom/cleanup/`persistent_field_tensors_written=false` hold.

### Consumer tests

- clean and robust samplers use the declared probabilities under a fixed seed;
- paired seed IDs are exactly `101/202/303`, comparisons pair equal IDs, and selectors consume aggregates rather than the best replica;
- HLT-to-field input widening copies existing weights, zeroes only new columns, and reproduces reference logits before training;
- robust sampling includes exact `f0` at the declared 20% probability;
- corruptions never alter padding;
- `Tpred_continue` consumes exactly the matched extra-step budget;
- every Tpred continuation branch receives identical weights/optimizer state, batches, stochastic streams, evaluation cadence, and checkpoint opportunities except its declared field-condition sampler;
- `A0_C250` uses exactly the consumer child manifest, `A0_C250_LONG` uses no new jets and exactly the bridge fine-tuning continuation budget, and `A0_S500` uses the exact child-manifest union;
- both direct controls match the declared parameter/FLOP tolerance, use the 500k union, and contain no bridge/KD/`T10` path;
- same-consumer response evaluation uses one checkpoint for all fields;
- selector records aggregate quality rules, deterministically prefers robust,
  chooses each predeclared median replica, continues across performance
  warnings, and refuses hard provenance/non-finite/stale-lineage failures;
- consumer and final tie pools are defined against one maximum score and are invariant to registry iteration order;
- consumer calibration ties use lower mean ECE and then lower mean Brier score; final ties use exact deployed parameter count and never measured latency;
- selector enforces same-consumer and matched-compute gains plus every numerical tolerance in Section 18;
- `stack_val_consumer` is unreadable until consumer selection is locked and cannot change the selected recipe/seed;
- the selected checkpoint hash is identical in the selection artifact, target logits, live graph, and bundle;
- primary/all-50/alternate teacher bindings and cache namespaces cannot be interchanged;
- Stage B5 refuses to run without `selected_bridge_consumer.json`.

### Reconstructor architecture tests

- Gaussian local weights match the target formula at all three bandwidths and the hard-radius ablation is isolated;
- A3 uses the exact neighbor cap/support/self-edge/weighted-mean/local-width constants in Section 12.7;
- A3 local-message parameter sharing and equations, 389-value region pooling, empty-token mask, and readback residual equation match Section 12.7;
- `f0` and `h0` conditioning plus the raw-HLT skip are mandatory in primary configs;
- seeded region assignment uses HLT inputs only;
- empty regions remain finite;
- region count for `few_subjets` is `4 + 4 + 2 = 10`;
- soft-assignment temperature, token width, transformer depth/heads/MLP/dropout/norm, readback, and head dimensions match Section 12.7;
- pair biases and particle readback have correct shapes;
- zero-initialized correction returns `f_hat == f0_live`;
- primary outputs correct only channels 0-44 and copy channels 45-49 exactly from `f0_live`;
- radius-aware heads receive their matching local stream;
- the gated ablation initializes open, freezes gates during warm-up, and affects only the correction;
- the gated ablation uses exactly twelve gates, bias `2.944438979`, Phase 2 unfreeze, and coefficient `0.005`;
- trust bounds are respected componentwise;
- raw consumer fields, conditioning/correction standardization, inverse scaling, sparse/zero `q99`, inactive channels, and `0.99*trust_scale` saturation obey Section 12.1/12.8;
- A5/A5S initialize to `[center[P],f0[S]]`, use the absolute plausibility bound, and A5S prevents `f0[P]/h0` from entering its trainable predictor;
- gradients reach the correction branch through a frozen `T10` while no `T10` parameter receives a gradient update;
- the optional refinement pass performs exactly one repool;
- all-50 B1 satisfies equal-field zero KD, while B2 records the expected irreducible mismatch and can never be selected;
- parameter/FLOP matching for the particle-only and direct-HLT controls is within the declared tolerance.

### Loss and training tests

- each ablation run activates exactly its declared loss components;
- L6, L7, and L8 respectively contain no regularizer, anchor only, and anchor plus smoothness;
- all A3 loss-interaction runs activate their exact declared objective and warm-up policy;
- every warm-up-enabled run executes exactly `field_warmup_steps`; validation cannot terminate Phase 1, while a non-finite numerical failure prevents Phase 2;
- `KD-only` and `CE-only` do not silently receive bridge warm-up;
- masked/group-balanced Huber ignores padding;
- standardized Huber uses `delta=1.0`, then channel/group balancing in the declared order;
- the twelve radius×semantic groups and exact standardized, Gaussian-weighted directed-edge `L_smooth` formula match Section 14;
- per-family checkpoint selection obeys the fixed accuracy/CE/epoch rule or the sole L0 bridge-loss exception;
- KD temperature and KL direction match a hand calculation;
- cached and on-the-fly target logits agree;
- provenance is acyclic binding→cache→run, bindings contain no final cache hash, and N3 requires its dedicated `physical45_selected_teacher_on_f0_control` cache;
- `f_hat == b_0.10` gives matching target/live logits and zero KD using the exact same checkpoint;
- in-allocation RAM-local training resume preserves frozen parents and sampler state, while a new allocation restarts all three replicas;
- persistent checkpoints contain only retained median trainable weights and parent hashes, not optimizer state, non-median weights, or duplicate frozen parents;
- campaign deduplicates both declared loss/architecture aliases and renders counts from the registry.

### Deployment and reporting tests

- deployable export contains no privileged file dependency;
- HLT-only reload reproduces logits;
- final-test command rejects bridge/oracle evaluation flags;
- reports place deployable and privileged rows in separate sections;
- recovery is `null` for non-positive teacher gain;
- paired bootstrap is deterministic;
- `model_val_stop/select` and `stack_val_consumer/deploy` child manifests have exact 75k counts, full parent coverage, and zero overlap;
- stack confirmation failures cannot select a runner-up, and `stack_val_deploy` is inaccessible before a locked deployable selection;
- the registry renders 55 runnable configurations, 47 reconstruction-breadth configurations, and 46 post-teacher configurations, including the quality-warning-bound TALT pair;
- every registry row has an explicit scientific role/selectability value; direct, all-50, negative, absolute, exploratory, TALT, and no-trust rows cannot enter primary selection;
- storage preflight rejects projections over 5 GiB or the explicitly selected 6 GiB paired3 ceiling;
- production storage projection refuses any runnable `UNMEASURED` row and otherwise uses measured serialized state bytes, actual manifest particle width, and metrics-only non-median replicas;
- reports include persistent/RAM byte telemetry and adversarial-channel diagnostics;
- perturbation audit uses exactly four declared seeds, event/particle/channel keys, scales, and signed accuracy-loss aggregation; its mean/maximum thresholds are enforced without hidden alignment cutoffs;
- the 1,001-quantile groupwise distribution diagnostic is compact, deterministic, and inaccessible on final-test;
- Slurm scripts contain `PYTHONNOUSERSITE=1` and `reu-aisocial`;
- packed Slurm jobs request one node, one allocation leader/ledger, and declare restart-whole-configuration behavior with no cross-allocation resume;
- dependency graph prevents reconstruction after failed selection.

## 24. Expected failure modes and interpretations

### No `T10` bridge gain

The ten-percent bridge is too small for the consumer, the residual target does not carry useful task information, optimization did not adapt, or evaluation noise is too large. Diagnose response curves and matched continued training before increasing `rho`. Do not distill a consumer with no valid gain.

### `T10` gains, but `R10` cannot recover it

The correction may not be predictable from HLT information, the selected consumer may be too brittle, or the reconstructor/loss is inadequate. Loss ablations distinguish lack of physical predictability from failure of task-aware supervision; clean-versus-robust consumers test brittleness.

### Field error improves without tagger gain

Uniform field regression is spending capacity on residual dimensions irrelevant to the classifier. KD or field-group reweighting is indicated.

### KD improves agreement without label performance

The target logits may encode calibration/idiosyncrasies rather than useful ranking, or the teacher gain may be confined to low-impact examples. Inspect CE ablations and slice metrics.

### HLG does not beat C0

The correction may be predominantly local/per-particle, region assignment may be unstable, or the pilot lacks enough data. The result is still useful: prefer the simpler deployable model and retain hierarchy for later rungs only if diagnostics support it.

### Apparent gain also appears for shuffled controls

Treat this as invalid evidence. Suspect leakage, training-compute mismatch, distribution shift, or selection noise.

### All-50 gain greatly exceeds physical-45 gain

Suspect that oracle-derived flags/reliability channels are acting as privileged shortcuts. Keep the all-50 result diagnostic, inspect reliability-only response, and do not promote it into the primary deployable lineage without a new experiment.

### KD/CE gains with nonphysical field patterns

The reconstructor may be using the frozen consumer as an adversarial communication channel. Trust saturation and the exact perturbation thresholds are automatic selectability gates. Poor bridge alignment or bridge-distribution shift is prominently flagged and prevents a claim of physical bridge recovery, but remains diagnostic rather than causing an undeclared numerical rejection in this pilot. Compare with the direct-HLT capacity control and report both task gain and physical diagnostics.

### RAM generation becomes the throughput bottleneck

Increase verified full-allocation retention or LRU capacity within the 20% headroom rule, group related GPU runs within allocations that can share staged immutable sources, or increase target-generation workers. Do not solve the problem by silently publishing dense bridge/control tensors.

### Persistent budget would be exceeded

Refuse new jobs before allocation. Preserve parent checkpoints, metrics, and already selected candidates; publish only the predeclared median replica per configuration, reduce bounded log reservations using measured output, or require explicit selection of the 6 GiB paired3 ceiling. Never delete user artifacts automatically and never resolve quota by silently dropping a declared scientific configuration.

## 25. Implementation order

The system is divided into ten deliberately similar-sized implementation increments. Each step should be reviewable and testable on its own. A step is complete only when its implementation, targeted tests, command-line validation, and relevant documentation are present; creating placeholder APIs does not count.

Before every step, inspect the dirty working tree and preserve unrelated or partially completed work. After every step, run its targeted tests plus the previously completed bridge-curriculum tests.

### Step 1 of 10: contracts, child splits, registry, and provisional quota

Implement versioned configurations, canonical hashes, and the safety layer used by every later step. Create/bind the exact 250k/250k `stack_train` children, 75k/75k `model_val` children, and 75k/75k `stack_val` children; prove counts, class stratification, parent coverage, and zero overlap. Implement the declarative campaign registry, exact scientific-role/selectability fields, aliases, conditional skips, paired seed IDs, and formula-based provisional storage bounds. Not-yet-implemented architectures remain explicitly `UNMEASURED`; Step 1 must not instantiate placeholder A3 models or claim a final measured budget.

Completion requires split overlap/reordering/stale-parent tests, validation-access locks, generated 55/47/46 registry counts, complete selectability coverage, conditional TALT state, provisional-formula tests, particle-width-sensitive dense-cache projections, and refusal to submit production while any runnable row is `UNMEASURED`. The dry-run report contains every child hash, seed replica, retained-state rule, provisional byte category, and measurement status.

### Step 2 of 10: streamed truth, frozen `R0`, and virtual bridge provider

Implement the concrete single-node, one-open-per-allocation compressed-NPZ path: rank-0 verification/decompression, non-evictable raw 8,192-jet RAM shards, allocation-wide locked byte ledger, deterministic rank ranges, 20% headroom, derived-only LRU/regeneration, telemetry, cleanup, and matched block/bin shuffle controls. Generate `f_true` from resident HLT/offline sources, train/register frozen `R0`, and expose deterministic `f0,h0`. Implement physical-45/all-50 virtual recipes, response points, five controls, immutable conditioning/correction scalers, resolved sparse/zero-channel rules, and physical-space trust statistics without dense persistent output.

Completion requires one-open/hash instrumentation, single-node rejection, allocation-ledger concurrency/oversubscription tests, raw-shard non-eviction, derived-LRU regeneration without persistent reopen, staged-array/rank ownership checks, direct-versus-staged target and `R0` agreement, exact endpoint/pass-through identities, matched-derangement controls, scaler/inverse-scale/q99 ordering edge cases, and a production smoke proving that only recipes/scalers/metrics were written persistently.

### Step 3 of 10: consumer training recipes and matched compute

Implement three paired replicas of `A0_C250`, `A0_C250_LONG`, `A0_S500`, `Tpred`, exact-budget `Tpred_continue`, `T10_clean`, the 60/20/15/5 `T10_robust`, and `T10_all50_clean`. Copy the same reference HLT weights into A0/Tpred, use the HLT-anchored split input normalization above, zero only widened field-projection columns, and verify both initial-logit identity and a nonzero first-step field-column gradient. Branch every continuation from identical Tpred weights plus optimizer/scheduler state with paired batches/random streams/evaluation opportunities. Keep branch state in shared job-local RAM, record measured serialized sizes for every newly available upstream/consumer row, and publish all replica metrics plus only the ordered-median checkpoint.

Completion requires bitwise copied-weight/zero-column and initial-logit tests, exact branch optimizer/budget/batch/RNG equality, robust-mixture frequency tests including exact `f0`, paired-seed aggregation/median-retention tests, all-50 lineage separation, tiny-data overfit/in-allocation-resume checks, measured-state registry updates, and proof that publications contain neither optimizer/non-median weights nor generated fields.

### Step 4 of 10: response metrics, numerical selector, and exact-teacher logits

Implement per-seed and aggregate same-checkpoint response/control evaluation, matched-compute comparison, paired bootstrap, calibration/slice metrics, and every Section 18 rule. Publish all quality failures, deterministically prefer robust for the primary path, retain the ordered-median robust and clean replicas, lock `selected_bridge_consumer.json`, and only then perform the one-shot `stack_val_consumer` confirmation. Continue on performance warnings but stop on invalid provenance, non-finite metrics, stale hashes, or missing checkpoints. Create immutable primary/all-50/alternate bindings before any cache, then generate namespace-separated bridge targets plus the dedicated selected-teacher-on-`f0` N3 cache on `stack_train_distill`; every cache manifest points to its binding hash and refitting is forbidden.

Completion requires synthetic aggregate/median/tie cases, best-seed rejection, two-of-three quality-warning cases, warning-mode continuation through negative confirmation gain, hard failure for non-finite/provenance/stale/cross-lineage artifacts, no runner-up, identical hashes across selection/logits/live/bundle configurations, cached-versus-direct agreement, and exact zero KD at equal fields for every semantically reachable teacher path.

### Step 5 of 10: C0 correction path, losses, and reachability suite

Implement the `f0+h0+raw-HLT` C0 correction baseline, immutable numerical spaces/scalers, zero-initialized physical-45 trust heads, five-channel pass-through, frozen-live `T10` graph, two-phase `model_val_stop` schedule, and corrected `D10_L0` through `L10` mappings. Launch L0 without a selected teacher alongside B3 and promote its explained variance, normalized MSE, correlation, and cosine metrics by radius/group. Keep primary gating absent and implement no-trust/weak-truth variations declaratively.

Completion requires hand-checked twelve-group standardized loss/smoothness math, exact gate behavior, distinct L6/L7/L8 assertions, deterministic checkpoint rules including the L0 exception, zero initialization/pass-through/saturation tests, correct gradient flow through frozen `T10`, sealed validation access, exact run-to-loss mappings, in-allocation RAM-local resume, measured serialized sizes for every implemented C0 configuration, median-only publication, and a CPU miniature for all eleven configurations.

### Step 6 of 10: Gaussian local streams and particle-only capacity controls

Implement the exact Section 12.7 graph, shared-within-layer message/node equations, three preserved Gaussian radius streams, matching 15-channel heads, and shared global interfaces. Add `D10_A1_multiscale_local`, `D10_A1H_hard_radius`, and `D10_A0M_capacity_particle`. Implement parameter/FLOP measurement utilities and record measured serialized sizes for every C0/local/particle-only architecture now available. Full-HLG variants and both direct HLG classifiers remain `UNMEASURED` until Step 7.

Completion requires analytical capped-kernel/weighted-mean/message-update tests, deterministic tie/self policies, exact sharing/width/depth/head routing, pass-through/numerical-space semantics, A1/A1H factor isolation, particle-capacity tolerance, measured-state registry updates, and tiny-batch train/reload tests for every newly available non-hierarchical control.

### Step 7 of 10: HLT-seeded regions, global reasoning, and readback

Reuse/adapt the exact hashed `multiscale_subjet_part` contracts to produce the locked HLT-only 4/4/2 regions; implement the 389-value pooling equation, 160-wide two-layer/four-head region transformer, pair biases, and exact residual readback. Implement A2/A3/A4, no-pair/fused-head/gate/input-removal variants, both absolute-output diagnostics, 2/2/1 and 8/8/4 counts, fixed assignment, same-scale-only attention, and one-global-token pooling. Only after global HLG exists, implement and train/reload both full-HLG direct classifiers on the 500k union. Record measured serialized sizes and parameter/FLOP tolerances for every architecture row.

Completion requires exact assignment/pooling/token/transformer/readback equations, empty-token/mask handling, HLT-only provenance, pair-bias/readback shapes, cross-scale masks, A5/A5S initialization/pass-through semantics, gate timing/equation, exactly-one refinement, finite gradients, full-HLG direct-control train/reload/capacity tests, measured-state registry updates, and miniature paired-seed evaluation for every hierarchy variant.

### Step 8 of 10: semantic, adversarial-channel, and paired-seed evidence

Implement the eight-run A3 loss-interaction block with canonical-A3 deduplication; canonical-A3 all-50 semantics and reliability response; conditional matched `D10_TALT_A0`/`D10_TALT_A3` clean-consumer diagnostics; all four separately defined trained controls including the N3 cache namespace; exact four-seed perturbation, bridge-distribution/alignment, and adversarial-channel diagnostics. Measure any remaining special-run states, require every runnable row to be `MEASURED`, and execute the final measured 5/6 GiB preflight over the generated 55/47/46 registry without making HLG depend on C0 success.

Completion requires all-50 full-head versus physical-only invariants, immutable binding→cache provenance, N3 cache isolation, four-control matching, complete selectability/non-selectability, exact perturbation threshold tests, alignment diagnostics without hidden cutoffs, generated count/deduplication tests, zero `UNMEASURED` runnable rows, final measured budget refusal, post-teacher release after only a valid teacher gate, and a miniature three-seed comparison of HLG, particle-only, raw-HLT-direct, and `R0`-representation-direct controls.

### Step 9 of 10: campaign policy, reports, and deployable packaging

Implement B0-B6 state transitions, parent/quota reservations, conditional skips, aggregate final selection, pre-confirmation lock, one-shot `stack_val_deploy`, and separate deployable/privileged/ablation reports. Report both oracle ceilings plus `Tpred_continue(f0)` and selected `T10(f0)`. Publish only retained-median weights, bounded logs, metrics, recipes/scalers/bindings; build one locked `R0+R10+T10` bundle; and run a clean HLT-only reload audit with all privileged sources absent.

Completion requires synthetic success/failure/control/quota/confirmation campaigns, no-fallback behavior, correct null recovery, visible failed/skipped runs, per-seed/aggregate/median reporting, both oracle ceilings in privileged tables only, persistent/RAM telemetry, final-test rejection before confirmation lock, no duplicated parents, and tolerance-equivalent clean-bundle logits.

### Step 10 of 10: production CLIs, Tigris graph, and end-to-end rehearsal

Connect every audit, staging, recipe/scaler, consumer, selector/binding, logit, reconstructor, report, and export command. Implement dependency-aware Slurm submission for B0-B6/paired3, multi-GPU allocation packing by shared source/teacher, one-time source staging, job-ID ledgers, host-memory requests, measured storage reservations, sealed-validation permissions, and fail-closed selection. Enforce `PYTHONNOUSERSITE=1` and account `reu-aisocial`.

Completion requires command/dry-run/shell tests, simulated dependency/confirmation/preemption success and failure, registry-rendered 55/47/46 inspection, single-node/one-leader/allocation-ledger and one-open assertions, restart-whole-configuration behavior, paired seed/median publication, zero runnable `UNMEASURED` rows, measured 5/6 GiB budget modes, absence of dense-field output paths, and a local CPU rehearsal whose generated Tigris command performs no submission until explicitly executed.

### Operational sequence after implementation

The ten steps above implement and locally rehearse the system. Actual expensive execution is a separate, deliberate action:

1. require zero runnable `UNMEASURED` registry rows and pass the final measured 5 or 6 GiB preflight before submitting any production allocation;
2. submit B0-B4, including early L0 and three paired consumer replicas, then lock/confirm the physical-45 `rho = 0.10` teacher;
3. if valid, allow B5 and release the registry-derived 46-configuration post-teacher matrix; the full maximum registry contains 55 model configurations including the early/upstream runs;
4. inspect reachability, deployable recovery, all-50 shortcut, direct-HLT, adversarial, and negative-control evidence;
5. aggregate the already paired-three-seed matrix, retain each median checkpoint under the 5 or 6 GiB budget, and lock one configuration before opening `stack_val_deploy`;
6. pass one-shot deployment confirmation and the clean reload audit, and only then evaluate final-test HLT-only;
7. consider `rho = 0.20` only after the `0.10` result is repeatable.

## 26. Definition of success for the first pilot

The experiment is successful enough to justify the next rung when:

1. the physical-45 `b_0.10` gives reproducible positive same-consumer and matched-compute gains over `f0`;
2. at least one HLT-only `R10 + T10` aggregate recovers a positive portion of that gain on `model_val_select` and its locked median replica has non-negative gain on `stack_val_deploy`;
3. the result survives matched-compute and shuffled/random controls;
4. trust saturation and the exact four-seed perturbation gates pass, while finite bridge-alignment/distribution diagnostics are reported without any hidden selection threshold;
5. physical-45 versus all-50 results show whether derived reliability channels created shortcut gain;
6. `A0_C250_LONG`, `A0_S500`, capacity-matched particle-only, `A0_CAP500_direct_hlt`, and `A0_CAP500_r0rep_direct` establish whether HLG and privileged curriculum add value beyond extra steps, extra jets, extra raw capacity, or direct use of `R0` representations;
7. the deployable bundle reloads without any oracle artifact and the campaign remains within its persistent-storage budget;
8. all scientific comparisons use the paired three-seed aggregate and ordered-median checkpoint before final-test or `rho = 0.20` is launched.

A small positive result is valuable at this stage. The purpose of `rho = 0.10` is to test whether a reachable privileged staircase exists, not to maximize the oracle ceiling in one jump.

## 27. Research basis

The architecture is an engineering hypothesis grounded in established jet-learning ideas rather than a claim that any cited model directly solves this HLT reconstruction problem:

- [ParticleNet: Jet Tagging via Particle Clouds](https://arxiv.org/abs/1902.08570) motivates local geometric neighborhood processing for particle clouds.
- [LundNet: Jet Tagging with Lund-based Graph Neural Networks](https://arxiv.org/abs/2012.08526) motivates hierarchical, physically structured jet representations.
- [MIParT: A Multi-Scale Interactive Particle Transformer](https://arxiv.org/abs/2407.08682) provides relevant evidence for combining particle representations across scales.
- [PHAT-JeT: Particle Hierarchy and Adaptive Tokenization for Jet Tagging](https://arxiv.org/abs/2605.21789) is recent supporting evidence for local geometric processing plus hierarchical patch/token attention under resource constraints.

These references support the inductive biases. The ablation campaign, not architectural reputation, decides whether they help this residual bridge task.
