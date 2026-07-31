# HLT Offline-Structure Distillation Implementation Plan

Status: scientific and engineering implementation plan for a new, independent
campaign. Nothing in this document is evidence that the campaign is already
implemented or production-ready.

Short name:

```text
HOSD
```

HOSD asks whether an HLT-input Particle Transformer can classify better when
it is trained to predict stable physical structure that is available in the
paired offline jet. It separates three questions that must never be conflated:

```text
predictability:
  can HLT evidence predict a target?

auxiliary utility:
  does requiring that prediction improve jet classification?

feedback utility:
  does feeding the prediction into later model reasoning improve beyond
  auxiliary regularization?
```

The primary graph is:

```text
paired offline jet -- training-only target extractor --> structure targets
                                                          |
HLT particles --> early/complete HLT ParT ----------------+
                    |                                     |
                    +--> structure prediction heads ------+
                    |                                     |
                    +--> optional predicted-structure feedback
                    |         into later HLT-only reasoning
                    |
                    +--> deployable jet classifier
```

At deployment, only HLT inputs and quantities predicted from HLT inputs are
available. Offline particles, offline targets, target-validity masks, teacher
states, truth labels, and degradation construction lineage are forbidden.

This is a separate campaign from:

- `RELATIONAL_PARTICLE_TRANSFORMER_ATTENTION_BIAS_IMPLEMENTATION_PLAN.md`;
- `RELATION_EXPERT_TOKEN_BRIDGE_IMPLEMENTATION_PLAN.md`.

HOSD may authenticate and reuse compatible RPT or RETB artifacts, but it does
not change their run IDs, contracts, scientific meaning, selection, or final
test. It must also be runnable from source-bound HOSD artifacts when an RETB
scientific result is not available.

Relevant precedents are:

- Particle Transformer: <https://arxiv.org/abs/2202.03772>
- ATLAS GN2: <https://arxiv.org/abs/2505.19689>
- Set Transformer: <https://arxiv.org/abs/1810.00825>
- knowledge distillation for real-time jet tagging:
  <https://arxiv.org/abs/2311.14160>

GN2 shows that physics-aligned auxiliary track-origin and common-vertex tasks
can improve flavour classification. HOSD does not claim to implement GN2.
The current repository source does not contain the truth ancestry, production
vertex, or reconstructed secondary-vertex records needed to reproduce those
tasks literally.

---

## 1. Scientific motivation

An ordinary HLT classifier receives only the degraded HLT constituent view and
one jet-class label. That label says which answer is correct but does not say
which intermediate evidence should be recovered from the degraded jet.

Paired offline/HLT data provide a richer training signal. Without reconstructing
offline particles or matching constituents, the offline jet can supply:

- global jet and composition summaries;
- displacement and track-availability distributions;
- angular-tree and multiscale topology summaries;
- permutation-invariant summaries of physical relation families;
- frozen offline-teacher logits;
- exploratory pooled or token representations.

The hypothesis is not that all targets are recoverable. A useful target may be
only partially predictable, and a highly predictable target may be redundant
for classification. Therefore every registered target is evaluated in a
two-axis plane:

```text
target predictability x downstream tagging utility
```

The four scientifically distinct outcomes are:

| Predictability | Tagging utility | Interpretation |
|---|---|---|
| high | positive | ideal stable auxiliary target |
| high | none/negative | predictable but redundant or distracting |
| partial/low | positive | incomplete structure is still useful |
| low | none/negative | target is inaccessible or poorly parameterized |

Target error may describe a result, but it may never be the sole success
criterion or the sole reason to suppress a registered scientific arm.

---

## 2. Locked scientific questions

### Q1: which matching-free offline structures are predictable?

Can a frozen HLT representation recover offline global, track, topology,
relation, teacher-logit, or latent targets better than declared prior and
raw-summary baselines?

### Q2: which targets improve HLT classification?

Does joint classification-plus-structure training improve over an ordinary
HLT ParT after controlling parameters, FLOPs, label exposure, seeds, and
training updates?

### Q3: does predicted structure help later reasoning?

Does a later HLT-only transformer stage benefit from predicted structure
tokens, predicted conditioning, or predicted HLT-by-HLT pair biases beyond
the benefit of auxiliary regularization alone?

### Q4: is predicting a correction easier than predicting an absolute target?

For targets with an HLT analogue, compare:

```text
absolute offline structure
offline-minus-HLT residual
heteroscedastic residual distribution
```

### Q5: which target families are complementary?

Do track, topology, composition, relation, logit, and latent objectives provide
independent classification gains, or do they teach the same evidence?

### Q6: what explains any improvement?

Can gains be distinguished from:

- generic extra capacity;
- longer optimization;
- output-level knowledge distillation;
- regularization by arbitrary targets;
- class information encoded in target marginals;
- an unrestricted hidden branch that does not preserve semantic structure?

### Q7: do gains survive HLT-domain changes?

Do selected models remain useful across fixed, multi-realization, randomized,
kinematic-only, track-only, missing-only, mild, nominal, severe, and legacy
HLT-like degradation profiles?

### Q8: how does HOSD relate to RETB?

Does structured supervision improve:

- a standard HLT ParT;
- native HLT relation-expert tokens;
- the inputs to an offline-token predictor;
- final native token fusion?

RETB comparisons are optional authenticated comparators, not HOSD
prerequisites.

---

## 3. Scope, claims, and non-goals

The initial HOSD campaign uses the repository's controlled
`fixed_hlt_v3_track_dominant_proxy`. It must be described as an HLT-like proxy,
not real detector HLT data.

The initial campaign will not:

- reconstruct an offline particle list;
- expose offline-to-HLT constituent indices;
- use degradation keep/merge/source indices as supervision;
- perform nearest-neighbour, optimal-transport, Hungarian, or cross-view
  constituent matching;
- claim deterministic proxies are truth vertices;
- claim track-origin truth exists in the current JetClass schema;
- call a learned token a literal human-interpretable thought;
- declare success from target MSE without classification utility;
- feed an offline target or its availability mask into deployable inference;
- select target definitions, loss weights, or target transforms on
  `final_test`;
- stop future registered arms because an earlier scientific arm performs
  poorly.

The following language is exact:

`offline-derived`

: Calculated from the paired offline jet using a registered label-blind
  extractor.

`deterministic proxy`

: A reproducible structure built from available reconstructed quantities. It
  is not simulation truth and not a measured secondary vertex unless its
  contract explicitly says so.

`predicted offline structure`

: The HLT-only model's estimate of a registered offline-derived target. It is
  not an offline reconstruction at particle level.

`GN2-inspired`

: Auxiliary supervision aligned with track or vertex-like structure. It does
  not mean GN2 was reproduced.

---

## 4. Canonical model and target notation

### 4.1 Baselines

`H_BASE`

: Exact base-size HLT-input Particle Transformer with the repository-standard
  17 particle features and Weaver standard-four pair relation:
  `embed_dims=[128,512,128]`, `pair_embed_dims=[64,64,64]`, eight heads,
  eight particle blocks, two class blocks, GELU, zero attention/class/
  activation dropout, trimming enabled, and ten classes.

`H_BASE_LONG`

: Same graph and data as `H_BASE`, trained for the maximum label exposure and
  update budget used by any HOSD candidate.

`H_MONO_PARAM_<graph_hash>`

: Candidate-specific monolithic HLT ParT compiled only after the corresponding
  HOSD graph is locked. It minimizes absolute deployed-parameter mismatch in
  the predeclared grid in Section 16, then analytical FLOP mismatch, then the
  smaller sum of `(embed_dim,particle_blocks,class_blocks,attention_heads)`,
  then the lexicographically smaller tuple in that order.

`H_MONO_FLOP_<graph_hash>`

: Candidate-specific monolithic HLT ParT from the same grid. It minimizes
  analytical inference-FLOP mismatch, then parameter mismatch, then the same
  smaller-sum/lexicographic tie order. There is no circular campaign-wide
  capacity control selected before its comparison graph exists.

`H_PARTICLENET`

: Matched-input ParticleNet baseline trained with the same identities,
  degradation replicas, labels, checkpoint rule, and seed lineage.

`H_PARTICLENET_PARAM`

: Graph-specific closest parameter-matched ParticleNet control from the
  Section-16 locked multiplier grid; it is always emitted.

`H_KD_LOGIT`

: `H_BASE` trained with class CE plus temperature-2 KL against the locked
  offline `O_BASE` teacher logits.

`H_NATIVE_REL_AUX`

: HLT-only auxiliary that concatenates the seven
  `T_HLT_SELF_RELATION_<family>` summaries in canonical family/component
  order and predicts them from `TAP_LATE`. It uses no paired offline target.
  HOSD versions, normalizes, caches, and tests this exact extractor; RETB
  `S3_RELATION_AUX` is precedent, not an implementation shortcut.

`H_RETB_NATIVE_FUSION`

: Authenticated native HLT relation-expert token fusion comparator, when a
  complete compatible RETB artifact exists.

`H_RETB_BRIDGE`

: Authenticated RETB offline-token bridge comparator, when available.

### 4.2 Usage modes

For target family `t`:

`P_t_LINEAR`

: Frozen `H_BASE` encoder plus linear target probe.

`P_t_SHALLOW`

: Frozen `H_BASE` encoder plus the registered shallow target probe.

`A_t`

: Joint classification plus target auxiliary loss. Target heads are discarded
  at deployable inference.

`F_t`

: Joint classification plus target loss, with only HLT-predicted structure
  consumed by later reasoning.

`F_t_DETACHED`

: Same as `F_t`, but the feedback consumer cannot send gradients through the
  predicted structure into the predictor. This is the primary variant for a
  semantic-mechanism claim.

`F_t_END_TO_END`

: Performance-oriented variant in which classification gradients may update
  the predictor through feedback. It is not by itself evidence that the
  intermediate coordinates retained their declared meaning.

`F_t_ORACLE_SUB`

: Post-hoc exact-target substitution into a consumer trained on predictions.
  It is nondeployable and selection-ineligible, and is not an upper bound.

`F_t_ORACLE_TRAINED`

: A separately trained, nondeployable consumer that receives the exact target
  during training and inference. This is the oracle ceiling used in
  gap/room calculations.

### 4.3 Target parameterizations

`ABS`

: Predict the normalized absolute offline target.

`RES`

: Predict the normalized residual between the offline target and the same
  extractor applied to the HLT view.

`HET`

: Predict residual mean and bounded log variance. The reconstructed point
  estimate is HLT value plus predicted residual mean.

`HET` is defined only for continuous components with a valid HLT analogue.
Categorical feedback consumes raw softmax probabilities, never argmax
categories. No probability is described as calibrated unless a separately
registered train-only calibrator and its hash exist.

Undefinedness is represented by an ordered set of availability groups, not one
scalar per target family. Every component declares one group; components with
identical applicability may share it. The predictor emits one availability
logit per group. Group BCE is averaged over groups and jets; value losses
broadcast the corresponding offline group bit only as a training mask.
Deployable feedback consumes each HLT-predicted availability probability in
registry order, never an offline availability bit.

---

## 5. Data, splits, and access contract

HOSD reuses the exact RETB identity population and split roles:

| Logical role | Repository split | Count | Per class | Permitted use |
|---|---|---:|---:|---|
| `model_train` | `model_train` | 500,000 | 50,000 | model weights, target normalizers, train-only fitting |
| `val_stop` | first half of `model_val` | 50,000 | 5,000 | epoch/checkpoint selection only |
| `val_design` | second half of `model_val` | 50,000 | 5,000 | deterministically partitioned design selection and confirmation |
| unused | `stack_train` | 0 | 0 | reserved |
| `final_select` | `stack_val` | 50,000 | 5,000 | locked complete-graph finalist selection only |
| `final_test` | `final_test` | 300,000 | 30,000 | sealed final evaluation |

The `val_stop`/`val_design` identity partition is byte-for-byte the RETB
partition:

```text
sha256("retb_model_val_partition_v1" || canonical_jet_identity)
then canonical_jet_identity
```

The exact split-manifest hash is an HOSD campaign parent. HOSD must not create
a second semantically identical partition under a new hash. It may invoke the
shared RETB builders to reproduce the same versioned semantics under the
current source snapshot when no reusable artifact exists; absence of a prior
RETB result is not a blocker. Artifacts serialize all three names where they
differ:

```text
repository_split = "stack_val"
cache_logical_role = "stack_val"
access_role = "final_select"
```

To prevent repeated adaptive reuse, `val_design` is deterministically divided
within each class:

```text
h = sha256("hosd_val_design_partition_v1" || canonical_jet_identity)
design_select  = first 2,500 identities per class by (h, identity)
design_confirm = remaining 2,500 identities per class by (h, identity)
```

`design_select` chooses targets, parameterizations, feedback, and combinations.
`design_confirm` confirms the already locked mechanism/control bundle and
cannot change its architecture, target, loss, or hyperparameters.

Every target is paired to HLT input only by canonical jet identity. Training
code receives:

```text
HLT raw tokens and mask
class label
canonical jet identity
registered target vector/category and training-only loss mask
```

It never receives:

```text
offline raw particles
offline constituent index
HLT construction source index
nearest constituent
cross-view assignment
offline pair matrix aligned to HLT particles
```

### 5.1 HLT replicas

HOSD inherits the exact RETB replica cycle and profile semantics:

```text
R_FIXED
R_MULTI
R_RANDOM
```

The primary training domain is `R_MULTI`. A named fixed/profile robustness
domain uses its replica-zero view. `R_MULTI` and `R_RANDOM` robustness means
four separately reported deterministic evaluations at replicas `0,1,2,3`;
the primary metric is the arithmetic mean of per-replica metrics, and replica
zero is also reported separately. There is no logit averaging across replicas.
An inherited cache is reusable only after its contract, source hash, identity
order, profile, replica, and content hashes validate exactly.

### 5.2 Scale-up

The optional ceiling stage uses the predeclared balanced:

```text
scale_train = 3,000,000
```

It runs only after every registered 500k HOSD discovery, mechanism,
combination, and confirmation row has completed and the bounded shortlist is
locked. Scientific underperformance does not cancel scale-up: the selector
always emits the baseline and the best available registered candidates.

### 5.3 Role capabilities

Every executable declares one role:

```text
campaign_builder
target_builder
teacher_inference
train_worker
probe_worker
design_inference
design_selector
stack_inference
stack_selector
postlock_oracle_diagnostic
final_inference
reporter
label_auditor
```

The role-to-dataset matrix is serialized in the campaign specification.
Opening an unauthorized split or artifact type is a runtime error.

Before finalist locking:

- `stack_val` model inference is HLT-only and label-free;
- prediction shards contain identity, logits, and probabilities only;
- the selector joins labels from a separately authenticated label manifest;
- no `stack_val` offline targets, target masks, teacher states, or oracle
  outputs may be built;
- no `final_test` model-derived output of any kind is permitted.

After finalist locking, selection-ineligible `stack_val` oracle diagnostics
may run. `final_test` requires a separate execution lock and is evaluated
exactly once.

Every row has one immutable experiment role:

```text
scientific_candidate
reference_baseline
capacity_control
null_control
mechanism_control
oracle_diagnostic
report_only
```

Only `scientific_candidate` rows may win a HOSD family/combination selector.
`reference_baseline` rows may win the overall accuracy or rejection finalist
but are never relabeled as HOSD. Capacity, null, mechanism, oracle, and
report-only rows are selection-ineligible at every stage. These roles are
serialized before training and cannot be inferred from results.

Offline-derived `A_t`, deployable `F_t`, and their registered combinations
are scientific candidates. H_BASE/LONG, KD, ParticleNet, HLT-self auxiliaries,
direct-computed HLT pair bias, and authenticated RETB comparators are reference
baselines. STOP/DISABLED/mean/shuffle rows are null controls;
the predeclared primary DETACHED and END_TO_END feedback rows are scientific
candidates, while AUX_ONLY/mean-only/unrestricted comparison variants are
mechanism controls; exact-target consumers and
class/target oracles are oracle diagnostics.

---

## 6. Source-capability and target-admissibility audit

The current canonical JetClass reader exposes:

```text
particle four-vector
charge
five PID indicators
d0, d0err, dz, dzerr
jet class label
```

It does not expose:

```text
particle ancestry
production-vertex identity
track-origin truth category
reconstructed secondary-vertex collection
secondary-vertex mass or flight distance
```

Campaign bootstrap writes `target_capability_audit.json`. Every proposed
target receives exactly one availability class:

```text
AUTHENTIC_TRUTH
OFFLINE_RECO_DERIVED
HLT_RECO_DERIVED
DETERMINISTIC_PROXY
TEACHER_DERIVED
UNAVAILABLE
```

The audit records source branches, extractor entry point, semantic version,
label access, matching requirement, and evidence hash.

For the current source:

| Proposed target | Required status |
|---|---|
| offline jet/composition/track distributions | `OFFLINE_RECO_DERIVED` |
| offline C/A tree and relation aggregates | `OFFLINE_RECO_DERIVED` |
| HLT same-view summaries and pair relations | `HLT_RECO_DERIVED` |
| compatibility-component pseudo-vertices | `DETERMINISTIC_PROXY` |
| offline teacher logits/latents/tokens | `TEACHER_DERIVED` |
| true HLT track origin | `UNAVAILABLE` |
| true HLT common production vertex | `UNAVAILABLE` |
| literal reconstructed secondary vertices | `UNAVAILABLE` |

An unavailable target does not fail the current campaign. It is excluded from
the executable current-source manifest and remains listed in the future-data
suite. Supplying richer data requires:

- a new authenticated raw schema;
- new cache and campaign versions;
- a repeated capability audit;
- explicit truth-category and ambiguity policies;
- proof that HLT-native labels attach to HLT objects without offline matching.

The HLT-v3 degradation generator's internal source indices may never upgrade
an unavailable target. They remain inaccessible implementation details.

---

## 7. Target registry contract

Bootstrap writes:

```text
campaign/structure_target_registry.json
```

Every target component records:

- target family ID and semantic version;
- availability class;
- physical or proxy meaning;
- source artifact and source algorithm hash;
- exact component names and canonical order;
- shape, dtype, units, domain, and symmetry;
- transform, clipping, and inverse transform;
- applicability and missingness rule;
- loss mask and event reduction;
- normalizer population and normalizer hash;
- allowed `ABS`, `RES`, and `HET` parameterizations;
- permitted use modes;
- target-specific loss and metrics;
- `label_access=false` for deterministic builders;
- `constituent_matching_required=false`;
- whether a prediction may enter deployable feedback;
- cache contract and parent hashes.

Any semantic change requires a new target contract version. Changing only a
filename must not make two different target meanings look interchangeable.

### 7.1 Common numerical definitions

For a sorted finite vector `x` of length `n`, the quantile at `q` uses:

```text
r = (n - 1) * q
lo = floor(r)
hi = ceil(r)
Q(q) = x[lo] + (r - lo) * (x[hi] - x[lo])
```

Canonical quantiles are:

```text
q = 0.10, 0.25, 0.50, 0.75, 0.90
```

Standard deviation is population standard deviation with denominator `n`.
Angles are wrapped with `atan2(sin(delta), cos(delta))`. Exact ties preserve
canonical particle or tree ordering.

For an empty applicable set:

- the stored numeric value is zero;
- its target-loss mask is false;
- its availability bit is reported in diagnostics;
- neither that bit nor the offline loss mask may enter the classifier or a
  feedback module.

Continuous target normalizers are fit on `model_train` only:

```text
center = finite component median
scale = max((Q75 - Q25) / 1.349, 1e-6)
normalized = clip((value - center) / scale, -12, 12)
```

Clipping counts are recorded per component. Counts explicitly declared
categorical are not robust-standardized.

For a streamed node/pair family whose dense target is forbidden from
persistent storage, these statistics use the inherited deterministic
relation-normalization sampling coordinates: select at most 50,000 training
jets by salted canonical-identity order, then at most 64 applicable pairs per
selected jet by salted `(identity,left_index,right_index)` order. The salts,
limits, selected-identity hash, sampled-pair hash, exact valid counts, and
parent view hashes are serialized in
`hosd_streamed_target_normalizer_v1`. Sampling is fixed before any target
value is inspected, uses only `model_train` (or `scale_train` for Stage J),
and is identical for every candidate consuming that target. No sampled
identity or pair value is stored in the normalizer artifact.

### 7.2 Target-family reduction

Every auxiliary loss first averages over valid components within one jet.
It then averages over jets with at least one valid component. A target with
more particles, pairs, or channels therefore does not receive more weight
merely because it has more elements.

Target cache storage is compact. Dense `N x N` targets and full attention maps
are not persisted. Pair-derived targets are reduced during streaming or are
generated deterministically inside a worker from compact same-view data.

---

## 8. Required current-source target families

### 8.1 `T_OFFLINE_JET_10`

Using float64 sums over valid offline constituents, store:

```text
log1p(jet_pt)
jet_eta
sin(jet_phi)
cos(jet_phi)
log1p(jet_mass)
log1p(jet_energy)
log1p(valid_constituent_count)
leading_particle_pt_fraction
subleading_particle_pt_fraction
sqrt(sum_i pt_i^2) / (sum_i pt_i + epsilon)
```

The summed mass is `sqrt(max(E^2-px^2-py^2-pz^2,0))` in float64. If summed
transverse momentum is zero, `eta`, `sin(phi)`, and `cos(phi)` are stored as
zero and share a false `jet_direction` availability group; other finite
components remain valid. Empty jets store ten zeros with all value groups
false. A nonfinite source input is a source-audit failure, not silently
masked.

Jet four-vector components are summed before deriving jet coordinates and
mass. Empty jets are invalid for the first six components and have zero-valued
masked storage. Missing leading or subleading particles produce zero fractions
with valid masks because the fraction is physically defined as zero.

`ABS`, `RES`, and `HET` are required.

### 8.2 `T_OFFLINE_COMPOSITION_16`

Use the repository-canonical six PID categories:

```text
charged_hadron, neutral_hadron, photon, electron, muon, unknown
```

Store:

- six count fractions over valid particles;
- six scalar-`pT` fractions;
- fractions of negative, zero, and positive charge over valid particles;
- net charge divided by `max(number of nonzero-charge particles, 1)`.

All components are defined for nonempty jets. PID and charge consistency use
the inherited RPT rules.

`ABS`, `RES`, and `HET` are required for all 16 stored continuous
fraction/net-charge components. No discrete PID identity is stored in this
target.

### 8.3 `T_OFFLINE_TRACK_32`

Track validity uses the exact inherited RPT TRACK rule and uncertainty floors.
Store:

1. valid-track count divided by valid-particle count;
2. valid-track scalar-`pT` fraction;
3. unavailable charged-domain count fraction;
4. unavailable charged-domain scalar-`pT` fraction;
5. five signed quantiles of raw `d0` significance;
6. five signed quantiles of raw `dz` significance;
7. five quantiles of absolute `d0` significance;
8. five quantiles of absolute `dz` significance;
9. fractions with absolute `d0` significance greater than `1`, `2`, and `3`;
10. fractions with absolute `dz` significance greater than `1`, `2`, and `3`;
11. scalar-`pT`-weighted mean absolute `d0` significance;
12. scalar-`pT`-weighted mean absolute `dz` significance.

This is exactly 32 ordered components. Distribution-derived components are
masked when no valid track exists; the four availability components remain
valid. On an empty jet their denominators are one and their values are zero,
so they remain valid statements of observed absence.

`ABS`, `RES`, and `HET` are required.

### 8.4 `T_OFFLINE_DENSITY_22`

Run the exact RPT DENSITY node builder on the offline view. For each of its 22
registered node descriptors, store the valid-particle scalar-`pT`-weighted
mean. If the total valid scalar `pT` is zero, use the unweighted mean. Empty
jets are masked.

`ABS`, `RES`, and `HET` are required.

### 8.5 `T_OFFLINE_CA_TREE_26`

Use the exact compiled/Python-parity-validated RPT C/A tree contract. Store:

- valid-particle count divided by 128;
- node count divided by 255;
- maximum leaf depth divided by 127;
- scalar-`pT`-weighted mean leaf depth divided by 127;
- scalar-`pT`-weighted population standard deviation of leaf depth divided by
  127;
- for `K=2,4,8`, actual cluster count divided by `K`;
- for each `K`, the three largest cluster scalar-`pT` fractions, padded with
  zeros;
- for each `K`, the largest cluster mass divided by jet mass, with zero when
  jet mass is zero;
- for each `K`, normalized multiplicity entropy;
- for each `K`, normalized scalar-`pT` entropy.

Entropy is:

```text
-sum_c p_c * log(p_c) / log(max(actual_cluster_count, 2))
```

with zero contributions from zero-probability clusters. Clusters are ranked by
descending scalar `pT`, then canonical cluster ID. These definitions produce
26 ordered components.

`ABS`, `RES`, and `HET` are required.

### 8.6 `T_OFFLINE_RELATION_AGGREGATES`

For each family:

```text
BASE4, PT, TRACK, PID, CHARGE, DENSITY, REGION
```

the registry declares every raw channel as:

```text
node_continuous
node_binary
ordered_pair_continuous
unordered_pair_continuous
pair_binary
categorical
```

The authoritative tap is the raw engineered output before any learned encoder
and before `FeaturewiseNormalizer`; physical transforms already defined by a
family remain part of that raw schema. API bindings are exact:

- BASE4 calls `build_standard_four_pair_features` directly;
- DENSITY aggregates the 22-channel
  `build_density_node_features(...)[\"descriptor\"]`, not its pair expansion;
- REGION calls `build_region_raw_features` (or its batched equivalent) and
  binds the angular-tree resource and REGION normalizer;
- PT, TRACK, PID, and CHARGE use family-specific raw/detail APIs;
  `RelationalPairBuilder(return_details=True)` is not presumed to expose all
  raw categorical semantics.

For each continuous channel, store:

```text
mean, population_std, Q10, Q50, Q90
```

over its exact applicability mask. For binary channels, store the positive
fraction. For categorical channels, store the complete category-frequency
vector in canonical category order.

Self-pairs are excluded. Symmetric pair channels use only `i < j`; directed
channels use all valid `i != j` ordered pairs. The emitted component names,
dimensions, and order are materialized in the target registry rather than
inferred by consumers.

Each relation family is a separate single-family arm. An all-relation
aggregate is tested only in the bounded combination stage.

`ABS` and `RES` are required. `HET` is required only for continuous aggregate
components.

### 8.7 `T_OFFLINE_TRACK_COMPONENT_PROXY`

This is explicitly a deterministic proxy, not a secondary-vertex
reconstruction.

On valid offline tracks, construct an undirected graph with an edge when:

```text
TRACK chi2 <= 4
and deltaR <= 0.10
```

Find connected components deterministically. Components with fewer than two
tracks are excluded. Rank components by descending scalar `pT`, then smallest
canonical constituent index. Store:

- component count capped at 8 and divided by 8;
- for the four leading components, padded with zeros:
  - track count divided by 40;
  - scalar-`pT` fraction;
  - raw finite mean absolute `d0` significance;
  - raw finite mean absolute `dz` significance.

This produces 17 components. The cap and overflow count are audited. The
target is useful only as a controlled test of vertex-like grouping.

`ABS`, `RES`, and `HET` are required.

### 8.8 `T_OFFLINE_LOGITS`

Required frozen teachers:

```text
O_BASE
O_FULLREL
```

If complete compatible RETB relation-expert checkpoints exist, their seven
individual expert logits form a separate optional target family.

Teachers are locked before HOSD HLT training. Targets are float32 logits in
canonical class order. Distillation uses:

```text
temperature = 2
loss = T^2 * KL(softmax(z_off / T) || softmax(z_hlt / T))
```

`H_KD_LOGIT` uses only this target and is the mandatory teacher baseline.
Teacher logits are never called physical structure.

Both mandatory teachers use seed `101`, the exact Section-15 optimizer,
`model_train`, `val_stop` checkpointing, offline inputs, and the same
40-epoch budget as `H_BASE`. `O_BASE` is the exact base ParT; `O_FULLREL` is
the locked full-relation RPT graph and relation schemas. Their architecture,
normalizers, checkpoint bytes, selector trace, seed, and source hashes are
parents of every teacher-derived target ID. If a compatible checkpoint is
absent, `train_hosd_offline_teacher.py` produces it before
`lock_hosd_teachers.py`; locking alone is not a producer. Confirmation uses
these fixed seed-101 teachers for all student seeds. At 3M, both teachers are
retrained from their component-scoped seed-101 initialization on
`scale_train`; scale logits/latents/normalizers are rebuilt and the 500k
coordinates are not reused.

For all logit-KD rows:

```text
lambda_KD = 1.0
L = L_CE + lambda_KD * L_KD
```

CE and KL are each ordinary per-event means. `O_BASE` and `O_FULLREL` use the
same coefficient.

### 8.9 `T_OFFLINE_POOLED_LATENT`

The exact 128-dimensional normalized pre-classifier `O_BASE` pooled
representation is cached from the locked seed-101 teacher. Its coordinate
system is checkpoint-specific. Whitening is fit in float64 on `model_train`
with a symmetric eigensolver; eigenvalues are descending, eigenvector signs
make the largest-absolute component positive, and a numerically degenerate
eigenspace is fixed by deterministic projected canonical-basis
Gram-Schmidt. Eigenvalues are floored at `1e-5` times the largest. The loss is
equal-weight:

```text
normalized Huber + (1 - cosine similarity)
```

Report frozen teacher-head agreement and accuracy after replacing the teacher
state with the predicted state through a deterministic ridge adapter fit once
on `model_train` (`lambda=1e-4`, float64 normal equations, no labels and no
checkpoint selection). Its input is the predicted whitened state and its
target is the locked unwhitened teacher state.

This is exploratory and semantically abstract.

### 8.10 HLT-native matched auxiliaries and pair targets

Every offline physical family with a same-view HLT analogue has a matched
control:

```text
T_HLT_SELF_JET_10
T_HLT_SELF_COMPOSITION_16
T_HLT_SELF_TRACK_32
T_HLT_SELF_DENSITY_22
T_HLT_SELF_CA_TREE_26
T_HLT_SELF_RELATION_<BASE4|PT|TRACK|PID|CHARGE|DENSITY|REGION>
T_HLT_SELF_TRACK_COMPONENT_PROXY_17
```

It runs the identical extractor, normalization class, head, tap, loss weight,
and update budget on the HLT view. These `HLT_RECO_DERIVED` rows are the
required control for whether paired offline supervision adds value beyond
ordinary physics-aligned self-supervision. They are not substitutes for the
offline targets.

Two exact deployable same-view pair targets are registered:

`T_HLT_TRACK_PAIR_13`

: Over ordered valid HLT track pairs, store the exact inherited RPT TRACK
  compatibility schema in this order:
  `log1p_chi2`, `exp_minus_half_clipped_chi2`,
  `minimum_abs_d0_significance`, `maximum_abs_d0_significance`,
  `minimum_abs_dz_significance`, `maximum_abs_dz_significance`,
  `d0_significance_product`, `dz_significance_product`,
  `context_minus_query_normalized_d0`,
  `context_minus_query_normalized_dz`,
  `sin_query_minus_context_delta_phi`,
  `cos_query_minus_context_delta_phi`, and `log_delta_r`.
  Applicability, uncertainty floors, clipping, direction, and transforms are
  exactly `relational_part_track_relation_v1`; diagonal and non-valid-track
  pairs are inapplicable.

`T_HLT_REGION_PAIR_8`

: Over unordered valid HLT pairs, store same-cluster indicators for
  `K=2,4,8`, then LCA depth divided by 127, `log1p(merge_deltaR)`,
  `log1p(merge_kT)`, bounded merge-`z`, and
  `log1p(merge_mass/max(jet_mass,1e-6))`. It binds the exact inherited
  C/A-tree ordering, floors, and REGION resource. The three indicators use
  BCE and the five continuous channels use normalized Huber.

The direct-computed RPT relations
`FB_PAIR_EXACT_HLT_TRACK` and `FB_PAIR_EXACT_HLT_REGION` are eligible
reference baselines. They include measured builder cost and use the exact
same relation channels without prediction. A genuinely offline/truth pair
matrix is never aligned to HLT particles and has no current-source oracle row.

### 8.11 `T_RETB_SUMMARY_TOKENS`

This optional comparator consumes only a complete, authenticated, compatible
RETB target-coordinate lock. HOSD does not generate an unofficial token bank
under the RETB contract. If unavailable, the row is recorded as
`not_applicable_missing_authenticated_parent`, not as a failed run.

### 8.12 Future authenticated-data suite

The following remain registered but unavailable for the current source:

```text
T_HLT_TRACK_ORIGIN_TRUTH
T_HLT_COMMON_VERTEX_TRUTH
T_OFFLINE_SECONDARY_VERTEX_SET
```

When enabled by a new source version:

- HLT truth labels must attach to each HLT object through that object's own
  authenticated simulation association;
- ambiguous, material, pileup, merged, and unmatched HLT objects require
  explicit categories or masks;
- no label may be obtained by retaining HLT-v3 degradation source indices;
- offline vertex sets may be matched only to predicted vertex slots, never to
  HLT constituents, under a separately reviewed set-loss contract.

They are not part of the current production matrix.

---

## 9. Target caches, teachers, and normalizers

Target builders are label-blind and are not given a label field, label file,
or label-capable data handle. A separate `label_auditor` may join the already
published identity-only target manifest to the authenticated split label
manifest for correlation diagnostics and shuffle plans. Canonical target
shards contain no copied labels. The target manifest attests:

```text
label_access_for_extraction = false
```

Cache shards are:

- identity sorted;
- split specific;
- atomic and resumable;
- bound to source, campaign, split, offline input, extractor, relation
  normalizer, tree backend, teacher checkpoint, dtype, and target-registry
  hashes;
- complete only when identity coverage is exact with no duplicates.

Before finalist locking, canonical target and teacher-output production is
restricted to `model_train`, `val_stop`, and `val_design`. Stage J separately
builds `scale_train` targets and teacher outputs before scale training.
Selection-ineligible `stack_val` oracle targets may be produced only after
`locked_hosd_finalists.json`; final-test offline targets, if required for a
predeclared post-lock diagnostic, require the execution lock and a distinct
producer role.

Canonical targets and shuffled controls are distinct artifacts. A shuffle
builder may never overwrite or mutate the canonical target cache.

Target normalizers use only `model_train`. For `R_MULTI`, the offline target is
identical across replicas; HLT-analogue values are replica specific. Residual
targets therefore bind the exact HLT replica. Scale-up refits all target,
residual, whitening, and conditional-residual statistics on `scale_train`.

### 9.1 Conditional residual diagnostic

For continuous targets with an HLT analogue, a diagnostic
`COND_RES` parameterization subtracts:

```text
E_model_train[s_offline - s_HLT | coarse HLT bin]
```

Coarse bins are fixed train-only quantile bins in:

```text
log jet pt: 8 bins
absolute jet eta: 4 bins
valid multiplicity: 4 bins
valid-track fraction: 4 bins
```

Empty cells back off in the listed reverse order and finally to the global
train mean. Bin edges, backoff path, and cell counts are serialized.
`COND_RES` is a diagnostic parameterization and may enter a deployable
candidate only after its complete lookup table is embedded in the exported
HLT-only graph. That table has role
`train_fitted_identity_free_deployable_statistic`: it contains no event IDs or
event values, binds only model-train population statistics, and is permitted
as a runtime parameter. Its training provenance still reaches offline
targets, which is allowed and audited.

### 9.2 Storage policy

Before building caches, measure:

- bytes per jet by family;
- extraction jets/second;
- projected 500k and scale storage;
- maximum shard rebuild time;
- target-mask sparsity.

If projections exceed available storage, persist compact jet-level targets and
stream same-view node/pair auxiliaries. The storage decision is based only on
measured resource evidence, not model performance.

---

## 10. HLT encoder taps and prediction heads

HOSD begins from the parity-validated repository ParT implementation. The
standard `H_BASE` forward pass must remain logit-, gradient-, mask-, and
state-dictionary-compatible when no HOSD head or feedback module is active.

The base particle encoder contains eight particle-attention blocks. Registered
taps are:

```text
TAP_EARLY = after block 2
TAP_MID   = after block 4
TAP_LATE  = after block 8, before class attention
```

Block numbering is one-based in artifacts and reports. A tap exposes particle
states and the exact particle mask without mutating the forward pass.

The existing `ReferenceParticleStateTap` and `RetbParticleEncoder` do not
provide this exact standard-`H_BASE` split forward. HOSD therefore implements
a new versioned Weaver split-forward adapter that captures blocks 2, 4, and 8
and resumes ordinary blocks/class attention without changing keys or RNG.
Likewise, post-block-4 pair feedback requires a new provider rather than
assuming the existing eight-layer precomputed `LayerwisePairBiasProvider`.
Authoritative real-Weaver FP32 parity covers unsplit versus split logits,
gradients, masks, and state dictionaries at all three taps with mixed
precision disabled.

### 10.1 Global target head

The canonical global head uses four learned queries of dimension 128:

```text
particle states
  -> RMSNorm
  -> 4-query masked cross-attention
  -> concatenate query outputs
  -> Linear(512,256), GELU, RMSNorm
  -> target-specific output projection
```

The output projection is independent per target family. The shared head trunk
is not shared between scientifically distinct candidate models unless that
sharing is part of a registered combination graph.

For `HET`, the head emits:

```text
mean
log_variance clipped to [-8,5]
```

and trains with the exact diagonal Gaussian negative log likelihood in
normalized target space:

\[
L_{\mathrm{het}}
=
\frac{1}{2}
\left[
\exp(-\ell)(y-\mu)^2+\ell
\right].
\]

The clipping is applied inside the likelihood and recorded. Prediction
interval coverage at 50%, 68%, 90%, and 95% is reported.

Intervals are `mu +/- z*exp(0.5*clipped_log_variance)` with
`z={0.6744897501960817,0.994457883209753,1.6448536269514722,
1.959963984540054}` respectively. Primary coverage is evaluated in clipped
normalized space against the normalized target; inverse-transformed physical
interval endpoints and clipping/censoring counts are secondary reports.

For each registered availability group, the output additionally contains:

```text
availability_logit[group_id]
```

Availability trains with binary cross entropy averaged over groups and jets.
Every component maps to one group in the registry and its continuous value
loss is masked by that group's offline bit. Feedback uses:

```text
predicted_value[c] * sigmoid(availability_logit[group(c)])
and all sigmoid(availability_logit[g]) in canonical group order
```

so offline missingness never enters deployable reasoning.

### 10.2 Particle head

Current-source HOSD does not have authentic per-HLT-particle offline labels.
The particle head is therefore used only for HLT-native deterministic
auxiliaries or the future authenticated-data suite.

It is:

```text
RMSNorm(d)
Linear(d,d)
GELU
Linear(d,target_dimension)
```

Losses use the exact HLT particle mask and target-specific applicability mask.

### 10.3 Pair head

The pair head may predict only an HLT-by-HLT quantity. It consumes:

```text
h_i
h_j
h_i - h_j
h_i * h_j
```

after a shared `RMSNorm`. Symmetric targets use:

```text
h_i + h_j
abs(h_i - h_j)
h_i * h_j
```

and explicitly symmetrize outputs. Diagonal pairs are masked. The head is a
two-layer width-128 MLP.

Pair losses are reduced per jet. If dense evaluation is too expensive, train
with deterministic stratified pair sampling:

```text
all positive pairs up to 512
512 negative pairs
seed = sha256("hosd_pair_sample_v1" || epoch || identity || target_id)
```

Sampling uses no RNG library. For each canonical pair ID, compute:

```text
sha256("hosd_pair_sample_v1" || epoch || identity || target_id || pair_id)
```

and retain the 512 smallest `(hash,pair_id)` values separately for positive
and negative strata. For a mixed target, “positive” means any registered
binary channel is one; “negative” means all are zero. A continuous-only pair
target instead retains the 1,024 smallest hashes without stratification.
When a stratum has fewer than its cap, use all pairs.
Per-jet loss is `0.5*mean(positive)+0.5*mean(negative)` when both exist and the
single available stratum mean otherwise, so sampled prevalence cannot change
the objective. Validation uses all applicable pairs in streamed chunks.

### 10.4 Classification isolation

In `A_t`, the jet classifier must not consume:

- target predictions;
- target-head hidden activations;
- target masks;
- offline availability;
- target errors.

Only shared encoder gradients connect the auxiliary task to classification.
At export, all auxiliary-only heads are removed. Their parameters count toward
training capacity and checkpoint size but not deployed parameters/FLOPs.

In `F_t`, the classifier may consume only the declared HLT-predicted
structure interface. The exported graph must prove that target cache and
offline roles are unreachable.

---

## 11. Probe campaign: measuring predictability

The probe stage freezes one independently trained seed-`101` `H_BASE`
checkpoint before any target result is inspected.

Every current-source target family runs:

```text
P_PRIOR
P_RAW_MLP
P_t_LINEAR at TAP_EARLY, TAP_MID, TAP_LATE
P_t_SHALLOW at TAP_EARLY, TAP_MID, TAP_LATE
P_CLASS_CONDITIONAL_ORACLE
P_TARGET_TO_CLASS_ORACLE
```

`P_PRIOR` predicts train medians, category frequencies, or teacher class
priors.

`P_RAW_MLP` applies only when the same registered physical extractor exists on
the HLT view. It consumes that HLT summary plus:

```text
HLT jet pt, eta, mass, multiplicity, valid-track fraction
```

It establishes whether the target is a trivial transformation of HLT global
summaries. For teacher logits, teacher latents, and RETB coordinates it is
`not_applicable_no_raw_hlt_analogue`; the campaign does not silently run an
offline teacher on HLT input and call it raw.

`P_CLASS_CONDITIONAL_ORACLE` consumes the true jet class and train-only
class-conditional target statistics. It is nondeployable and
selection-ineligible. It estimates how much apparent target predictability is
explained by class correlation alone.

`P_TARGET_TO_CLASS_ORACLE` is a fixed
`LayerNorm -> Linear(d,128) -> GELU -> Linear(128,10)` classifier consuming
the exact offline target and declared availability vector. It is trained on
`model_train`, stopped on `val_stop`, and is nondeployable and
selection-ineligible. It measures the target's intrinsic class information;
it is not the reverse class-to-target conditional oracle.

Probe heads train on `model_train`; `val_stop` selects their checkpoint;
`design_select` reports predictability. No probe metric may cancel a later
registered single-family `AUX` arm.

### 11.1 Predictability metrics

Continuous scalar/vector targets report:

- normalized MAE and RMSE;
- coefficient of determination `R2`;
- Spearman rank correlation;
- median absolute error in physical units after inverse transform;
- valid-target coverage.

Heteroscedastic targets additionally report:

- negative log likelihood;
- 50%, 68%, 90%, and 95% interval coverage;
- interval width;
- error by predicted-uncertainty decile.

Categorical targets report:

- masked cross entropy;
- balanced accuracy;
- macro F1;
- complete confusion matrix;
- Brier score and target ECE.

Binary/pair targets report:

- per-jet and class-balanced ROC AUC;
- precision-recall AUC;
- balanced accuracy;
- prevalence and empirical probability-calibration error.

Teacher targets report:

- temperature-2 KL;
- top-1 agreement;
- per-class logit correlation;
- frozen-teacher decision preservation.

Latent targets report:

- normalized Huber and RMSE;
- cosine similarity;
- linear CKA;
- effective-rank retention;
- frozen-head accuracy and agreement.

Metrics are calculated per class and overall. A high overall score caused
only by class identity must be visible through the class-conditional oracle
and within-class metrics.

---

## 12. Joint auxiliary-supervision architecture

Every current-source target family runs an `A_t` single-family screen using
the canonical global head unless its registry requires a node or pair head.

The canonical global AUX tap is `TAP_LATE`; HLT pair prediction uses
`TAP_MID` so it can feed blocks 5-8. EARLY/MID/LATE global taps are probe
diagnostics only and are not adaptively selected for AUX. Every offline
physical family with an HLT analogue also runs its Section-8.10
`A_HLT_SELF_*` control with identical tap, head, loss dimensionality, weight,
and update budget.

The primary objective is always present:

\[
L =
L_{\mathrm{class}}
+ \lambda_{\mathrm{KD}} L_{\mathrm{KD}}
+ \sum_f \lambda_f L_f.
\]

Because every sampler is exactly class balanced, `L_class` is ordinary
unweighted per-event ten-class cross entropy. Each `L_f` is averaged first
within jet and then across jets as defined in Section 7.2.

Unless a target declares a categorical or distribution-specific likelihood,
continuous point prediction uses Huber loss with `delta=1` in normalized
target space. Availability uses binary cross entropy and contributes one
equal-weight subtask inside its family loss. `HET` replaces point Huber for
continuous values with Section-10.1 NLL but retains availability BCE.

### 12.1 Fixed auxiliary-weight screen

For every single family:

```text
lambda_f in {0.10, 0.30, 1.00}
```

The `lambda=0` graph is the disabled-head capacity control. Epoch selection
uses `val_stop` classification accuracy, never target error. The immutable
design selector uses `design_select` downstream tagging utility, then
target-specific predictability only as a late tie-breaker.

The selected single-family weight is frozen before any combination model is
trained.

### 12.2 Gradient controls

Every family includes:

`A_t_STOP_ENCODER`

: Target head trains, but auxiliary gradients are stopped before the shared
  encoder.

`A_t_DISABLED`

: Identical graph and initialized head with `lambda=0`.

`A_t_WITHIN_CLASS_SHUFFLE`

: Target identities are permuted independently within each class. This
  preserves class-target correlation and marginal masks while destroying
  event-specific structure.

`A_t_GLOBAL_SHUFFLE`

: Target identities are permuted across the complete balanced split,
  destroying both event-specific and most class-conditional structure.

`A_t_TARGET_MEAN`

: Every valid component uses the train-only unconditional target mean while
  preserving the original per-component loss mask.

Shuffle permutations are:

```text
sort by sha256(
  "hosd_target_shuffle_v1" ||
  shuffle_kind ||
  target_id ||
  split ||
  canonical_jet_identity
)
rotate by one position
```

Within-class shuffles operate separately in canonical class order. A
one-element group maps to itself and is reported.

### 12.3 Target parameterization screen

For `T_OFFLINE_JET_10`, `T_OFFLINE_COMPOSITION_16`, `T_OFFLINE_TRACK_32`,
`T_OFFLINE_DENSITY_22`, `T_OFFLINE_CA_TREE_26`, and
`T_OFFLINE_TRACK_COMPONENT_PROXY`, run:

```text
ABS
RES
HET
```

For relation aggregates, run `ABS` and `RES` for all families and `HET` for
the best continuous family after single-family `ABS/RES` completion.

Parameterization selection is based on classification utility. Lower target
error does not override a worse classifier.

---

## 13. Predicted-structure feedback

Feedback is tested after all single-family auxiliary rows finish. One
predeclared exemplar per interface type runs regardless of auxiliary
performance. Additional expensive feedback variants are bounded by the
selector in Section 17.

No primary feedback candidate uses teacher forcing. During both training and
inference, later layers consume only HLT-predicted structure. Oracle feedback
is a diagnostic in a distinct artifact role.

### 13.1 Global structure tokens

`FB_TOKEN` is the primary interface for jet-level offline targets.

At `TAP_MID`, the global target head emits a normalized target estimate. For
`HET`, concatenate mean and clipped log variance. A target-specific linear
map produces four structure tokens of dimension 128. Add learned:

- target-family embedding;
- parameterization embedding;
- structure-slot embedding;
- source embedding `predicted_from_hlt`.

The four tokens remain in a separate branch. They do not become valid members
of the H_BASE particle sequence. Before block 5, one masked residual
cross-attention adapter computes:

```text
delta_H = gamma * CrossAttention(Q=RMSNorm(H_particles), K=S, V=S)
H_particles <- H_particles + delta_H
gamma = 2 * tanh(raw_gamma), raw_gamma initialized exactly to zero
```

The adapter output projection has zero bias. Particle masks, sequence length,
pair-bias tensors, blocks 5-8, and class attention remain the ordinary H_BASE
path. Thus a zero gate is an algebraic no-op even when token embeddings and
projection biases are nonzero. This identity is required for logits and
gradients, not merely approximate numerics.

Required controls:

```text
FB_TOKEN_ZERO
FB_TOKEN_DISABLED_LOSS
FB_TOKEN_SHUFFLED_PREDICTION
FB_TOKEN_UNRESTRICTED
FB_TOKEN_MEAN_ONLY
FB_TOKEN_ORACLE_SUB
FB_TOKEN_ORACLE_TRAINED
```

`FB_TOKEN_ZERO` holds the residual gate at exactly zero.
`FB_TOKEN_DISABLED_LOSS` predicts and feeds structure with `lambda_f=0`.
`FB_TOKEN_SHUFFLED_PREDICTION` substitutes predicted structure using the
immutable split-level wrong-event mapping. The control first caches
identity-bound predicted structures, joins the mapped wrong identity before
batching, and then runs blocks 5-8. It is a selection-ineligible mechanism
diagnostic and is independent of batch layout.
`FB_TOKEN_UNRESTRICTED` replaces the target projection by the same four-query
trunk and a direct four-token output. A deterministic zero-parameter padding
ledger makes its trainable parameter count equal to the semantic branch; its
attention/readout widths, tap, gate, and update budget are otherwise exact.
`FB_TOKEN_MEAN_ONLY` removes HET log-variance channels while keeping a
parameter-matched dummy projection, testing variance as a label side channel.
The two oracle controls have the Section-4 substitution/independently-trained
meanings and are nondeployable.

The primary semantic candidate is `F_t_DETACHED`. On `design_confirm`, its
normalized target loss must be no more than `1.05` times the corresponding
`A_t` loss and each predicted availability rate must differ by at most `0.02`.
Failure does not terminate the run, but downgrades the interpretation to
“target-regularized hidden bottleneck.” `F_t_END_TO_END` remains an eligible
performance variant but cannot establish semantic mediation alone.

### 13.2 Feature-wise conditioning

`FB_FILM` maps the predicted global structure to bounded scale and shift for
the pre-normalized inputs of blocks 5 through 8:

```text
scale = 1 + 0.1 * tanh(s)
shift = 0.1 * tanh(b)
```

All final projection weights and biases initialize to zero, making the initial
graph exactly `H_BASE`. `FB_FILM` uses the same prediction head, target loss,
tap, and seed as `FB_TOKEN`.

### 13.3 Predicted HLT pair bias

`FB_PAIR` is permitted only for a registered HLT-by-HLT pair target. It may
use:

- `T_HLT_TRACK_PAIR_13`;
- `T_HLT_REGION_PAIR_8`;
- future authenticated HLT common-vertex truth.

It may not use an offline pair matrix.

At `TAP_MID`, predicted pair probabilities/features enter each later
attention head as:

\[
B_{ijh}
=
\alpha_h
\tanh(g_h(\hat r_{ij})).
\]

The `alpha_h` gates initialize to exactly zero and are bounded:

```text
alpha_h = 2 * tanh(raw_alpha_h)
```

The bias is symmetric for symmetric targets, zero on invalid pairs, and zero
on the diagonal unless the target contract explicitly defines a diagonal.
Gates remain fixed at zero for the first exact 5% of optimizer updates:

```text
gate_warmup_updates = ceil(0.05 * total_updates)
```

clipped to `[1,total_updates]` for nonempty training. After that point they
are trainable. The campaign records learned per-head scales.

Required controls:

```text
FB_PAIR_AUX_ONLY
FB_PAIR_ZERO_GATE
FB_PAIR_DETACHED
FB_PAIR_NO_SEMANTIC_LOSS
FB_PAIR_SHUFFLED
FB_PAIR_UNRESTRICTED_MLP
FB_PAIR_EXACT_HLT
```

`FB_PAIR_UNRESTRICTED_MLP` uses the identical pair input, two width-128 layers,
head projection, gate count, and update budget but has no semantic target
loss; zero-parameter ledger padding matches any output-dimension difference.
`FB_PAIR_EXACT_HLT` directly computes the registered TRACK or REGION
relation from runtime HLT input. It is deployable and selection-eligible as a
reference baseline, not an oracle. There is no current-source offline-pair
oracle because offline constituents are not aligned to HLT particles.

### 13.4 Feedback gradient path

For every selected feedback family:

- `END_TO_END` permits classification and target gradients through the
  predictor;
- `DETACHED` stops the consumer gradient at the predicted structure but
  preserves auxiliary gradients;
- `AUX_ONLY` removes the feedback consumer.

All three share component-scoped initialization, data order, label exposure,
and target cache. Shuffled feedback uses a frozen predictor checkpoint and
identity-bound prediction cache; a globally precomputed split permutation is
joined before batching. On-the-fly current-batch output is forbidden.

### 13.5 Probabilistic feedback

For `HET`, the structure-token and FiLM interfaces consume both predicted mean
and predicted log variance. They never sample a target during primary
training or inference. A sampling diagnostic uses a fixed identity-derived
normal draw and is selection-ineligible.

For categorical targets, feedback consumes the complete predicted probability
vector. For pairwise same-category compatibility it may use:

\[
\hat p_{\mathrm{same}}(i,j)
=
\sum_c p_i(c)p_j(c),
\]

only when the category semantics and applicability masks are authenticated.

---

## 14. Combination models and gradient conflict

After every single-family `AUX` and required feedback row completes, build a
bounded combination registry.

Mandatory combinations are:

```text
C_PHYSICAL
  one selected deployable target from each of
  JET, COMPOSITION, TRACK, DENSITY, CA_REGION, and OTHER_RELATIONS

C_TRACK_TOPOLOGY
  best TRACK-like target + best CA/REGION-like target

C_PHYSICAL_KD
  C_PHYSICAL + O_BASE logit KD

C_PHYSICAL_LATENT
  C_PHYSICAL + best eligible latent target

C_ALL_BEST
  one selected target from every eligible family group

C_NATIVE_OFFLINE
  H_NATIVE_REL_AUX + best offline-derived physical target
```

“Best” is defined by the deterministic Section-17 selector, not manual
inspection. If every member of a group is negative, the selector still emits
the best available member and marks its paired gain as negative.

### 14.1 Redundancy screen

On identity-paired `design_select` predictions and targets, report:

- residual Pearson and Spearman correlations;
- target-error correlation;
- shared-encoder gradient cosine;
- fraction of minibatches with negative gradient cosine;
- leave-one-family-out classification change;
- target-head representation linear CKA.

These diagnostics do not silently alter the mandatory combination list.

### 14.2 Loss weighting

The primary combination uses the selected single-family fixed weights, then
renormalizes them so:

```text
sum_f lambda_f = min(sum selected single-family lambdas, 1.0)
```

while preserving their ratios. Thus adding more targets cannot increase the
total auxiliary weight without bound.

For `C_ALL_BEST`, compare:

```text
W_FIXED
W_PCGRAD
```

PCGrad uses only shared-encoder gradients, applies tasks in canonical target
family order after a deterministic seed-based cyclic rotation per update, and
uses, for current gradient `g_i` and comparison `g_j`:

```text
if dot(g_i,g_j) < 0:
    g_i <- g_i - dot(g_i,g_j) / max(dot(g_j,g_j),1e-12) * g_j
```

Projected shared gradients are summed, divided by the number of tasks, then
global-norm clipped. Target-head-only parameters retain their own unprojected
gradients and join before the single optimizer step. The algorithm, ordering,
rotation, and clipping order are serialized.

Adaptive uncertainty weighting, GradNorm, and post-result manual weighting are
not part of the primary campaign.

---

## 15. Training protocol

Unless an exact inherited baseline contract says otherwise:

```text
optimizer = AdamW
betas = (0.9,0.999)
weight_decay = 1e-4
gradient_clip_norm = 1.0
maximum_epochs = 40
base_learning_rate = 1e-3
minimum_learning_rate = 1e-5
mixed_precision = BF16 on GH200
microbatch_size = 64
gradient_accumulation_steps = 2
effective_batch_size = 128
particle_attention_dropout = 0
residual_dropout = 0
class_attention_dropout = 0
activation_dropout = 0
num_workers = 0
```

For `T > 0` total optimizer updates:

```text
warmup_updates = min(T, max(1, ceil(0.05 * T)))
```

For one-based optimizer-update ordinal `u`:

```text
if u <= warmup_updates:
    lr = base_learning_rate * u / warmup_updates
elif T == warmup_updates:
    lr = base_learning_rate
else:
    p = (u - warmup_updates) / (T - warmup_updates)
    lr = minimum_learning_rate
         + 0.5 * (base_learning_rate - minimum_learning_rate)
           * (1 + cos(pi*p))
```

Thus the first update is `base_learning_rate/warmup_updates`; when
`warmup_updates=1`, including `T=1`, it uses base LR. This is the inherited
RETB one-based schedule. Resume restores optimizer, scheduler, sampler,
replica cycle, auxiliary head, and feedback-gate state exactly.

All rows derive RNG seeds independently by
`sha256("hosd_component_seed_v1" || campaign_seed || component_role ||
canonical_graph_id)`, so adding a head cannot shift shared-encoder
initialization. Full AUX/combination rows train for 40 epochs from the common
matched initialization. Feedback rows start from their locked `A_t`
checkpoint and receive exactly 40 additional epochs; their total label
exposure is therefore 80 epochs. `H_BASE_LONG` is exactly 80 epochs and
matches that maximum. Every artifact records microbatches, accumulation
boundaries, optimizer updates, and label presentations.

Checkpoint selection uses only `val_stop`:

1. maximum balanced classification accuracy;
2. lower classification cross entropy;
3. earlier epoch.

Target prediction quality never selects an epoch.

Parity tolerances are path-specific:

```text
authoritative Weaver FP32:
  logits abs=1e-6, rel=1e-5
  floating input/parameter gradients abs=2e-6, rel=2e-5
  masks, categories, identities, topology, state-dict keys/shapes exact

FP32 eager-to-export:
  logits abs=1e-5, rel=1e-5

BF16 eager-to-export:
  logits abs=1e-2, rel=1e-2
```

The authoritative test disables autocast/mixed precision. A newly added
module at its declared identity initialization must preserve shared
state-dictionary tensor values exactly.

### 15.1 Seeds

Discovery seed:

```text
101
```

Confirmation seeds:

```text
202, 303, 404
```

Each seed deterministically derives:

- model initialization;
- data ordering;
- pair sampling;
- target-control permutation;
- dropout;
- feedback initialization.

Teacher checkpoints and target caches are seed-independent unless their
contract explicitly names a teacher seed. Matched comparisons use identical
HLT input identities, replicas, batch order, and pipeline seed.

### 15.2 No performance-based termination

The following may fail a job:

- invalid provenance or forbidden access;
- nonfinite required inputs, loss, gradients, logits, or metrics;
- schema, shape, mask, or identity mismatch;
- missing required parent;
- resource exhaustion or runtime failure.

The following may not fail a job or cancel future registered jobs:

- worse accuracy than `H_BASE`;
- low target `R2`;
- negative gap closure;
- poor calibration;
- zero learned feedback gates;
- a target or combination being scientifically unhelpful.

Every selector emits the best available candidate and a negative-result trace
when all candidates lose.

---

## 16. Required controls

The complete 500k campaign includes:

### Architecture and compute

```text
H_BASE
H_BASE_LONG
H_PARTICLENET
H_MONO_PARAM_<graph_hash> for each confirmation graph
H_MONO_FLOP_<graph_hash> for each confirmation graph
H_PARTICLENET_PARAM_<graph_hash> for each confirmation graph
```

The monolithic ParT grid is the Cartesian product:

```text
embed_dim in {96,112,128,144,160,192}
particle_blocks in {6,8,10,12}
class_blocks in {1,2,3}
attention_heads in {4,8} where embed_dim % attention_heads == 0
```

For width `d`, `embed_dims=[d,4d,d]` and
`pair_embed_dims=[d/2,d/2,d/2]`. All other H_BASE feature, pair, activation,
normalization, trimming, and dropout settings remain fixed. Parameter counts
are exact trainable tensor counts. Analytical multiply-adds are counted as two FLOPs at batch one,
128 valid particles, ten classes, including input/pair encoders, attention
projections and score/value products, FFNs, class attention, feedback
adapters, and classifier; normalizations, activations, tree/relation builders,
and all non-multiply operations have separately reported operation and wall
time ledgers. The capacity compiler writes every term and uses the Section-4
tie rules without reading performance.

`H_PARTICLENET` is exactly
`jetclass_fresh.heterogeneous_hlt.ParticleNetHLTClassifier` wrapping Weaver
ParticleNet with the canonical 17 normalized HLT features, mask, repository
eta/phi points, `conv_params=[(16,(64,64,64)),
(16,(128,128,128)),(16,(256,256,256))]`,
`fc_params=[(256,0.1)]`, `use_fusion=false`, `use_fts_bn=true`,
`use_counts=true`, and ten outputs. The matched grid replaces every stage
width and FC width by a common multiplier in
`{0.50,0.75,1.00,1.25,1.50,2.00}` rounded to the nearest multiple of eight;
the closest parameter match is always emitted using parameter mismatch, FLOP
mismatch, smaller multiplier, then lexical config hash.

### Teacher and native auxiliaries

```text
H_KD_LOGIT using O_BASE
H_KD_LOGIT using O_FULLREL
H_NATIVE_REL_AUX
```

### Single-family structure

Every required current-source target in `AUX` mode, all registered weights,
and all declared parameterizations.

### Null and leakage controls

```text
disabled head
stop-encoder target head
target mean
global target shuffle
within-class target shuffle
class-conditional oracle probe
target-to-class oracle probe
```

The training-time within-class target shuffle is the AUX intervention.
“Wrong-event” is reserved for evaluation-time predicted-feedback substitution
from the frozen split prediction cache; there is no duplicate AUX
wrong-event-target row.

### Feedback mechanism

```text
AUX_ONLY
END_TO_END
DETACHED
ZERO feedback
semantic loss disabled
shuffled predicted structure
unrestricted matched-capacity feedback
oracle offline structure diagnostic
mean-only uncertainty feedback
exact HLT relation feedback
```

### Existing-system comparators

```text
H_RETB_NATIVE_FUSION when authenticated and compatible
H_RETB_BRIDGE when authenticated and compatible
```

Missing optional comparators are reported. Their absence cannot be represented
as a zero metric or an HOSD failure.

### 16.1 Numerically bounded 500k matrix

The current required family count is exactly 18:

```text
6 global physical = JET, COMPOSITION, TRACK, DENSITY, CA, TRACK_PROXY
7 offline relation aggregates
2 teacher-logit families
1 pooled-latent family
2 HLT-native pair families
```

Optional RETB coordinates never change these bounds.

- Stage C has at most `18 * 9 = 162` probe rows plus six named baselines.
- Stage D has 54 global ABS/RES/HET-weight rows, 42 relation ABS/RES rows,
  three locked-best relation-HET rows, six pair rows, three latent rows, two
  KD rows, 13 matched HLT-self rows, and at most 90 null/control rows:
  at most 213 rows.
- Stage E has at most eight base feedback graphs: mandatory TRACK global
  token and FiLM, mandatory TRACK and REGION pair, plus at most two promoted
  global targets through both global interfaces. END_TO_END and DETACHED run
  for each (16 rows). The complete control bundle runs only for the four
  mandatory exemplars and is capped at 30 further rows: at most 46 rows.
- The beam has width 12 across eight groups; 96 reduced-budget fits are the
  hard maximum. Four beam winners plus six mandatory full combinations are
  at most ten full fits.

The campaign compiler rejects any manifest exceeding a bound. Adding a future
target requires a new matrix-contract version and new explicit counts.

---

## 17. Screening and deterministic selection

### 17.1 Single-family utility score

After a candidate's epoch is selected on `val_stop`, evaluate on
`design_select`. First compute the maximum balanced accuracy among eligible
rows and freeze the eligible window as all rows within absolute `0.0001` of
that maximum; do not compare pairwise windows transitively. Within that fixed
set use:

1. higher Jeffreys-smoothed mean log QCD rejection;
2. lower cross entropy;
3. lower deployed analytical FLOPs;
4. fewer deployed parameters;
5. lower training GPU-hours;
6. lexicographically smaller candidate ID.

Target losses are not commensurate across families and never break a
cross-family tie. They remain reports and may break a within-family tie only
through standardized improvement over that family's `P_PRIOR`.

### 17.2 Feedback promotion

One mandatory exemplar runs for:

```text
T_OFFLINE_TRACK_32 through global structure tokens
T_OFFLINE_TRACK_32 through global FiLM
T_HLT_TRACK_PAIR_13 through predicted pair bias
T_HLT_REGION_PAIR_8 through predicted pair bias
```

Then promote at most two additional global target families total under the
single-family utility order; each runs through token and FiLM. Promotion
changes only the bounded expensive feedback wave.

### 17.3 Combination beam

To bound the Cartesian product, use beam width 12. Begin with every selected
single-family candidate. Traverse family groups in canonical order:

```text
JET
COMPOSITION
TRACK
DENSITY
CA_REGION
OTHER_RELATIONS
TEACHER
LATENT
```

At each step, expand by adding or omitting that group's selected target. The
proxy is the actual complete combination graph from the common initialization,
trained for exactly five epochs (the first `5*ceil(500000/128)` optimizer
updates) with the full losses, taps, and unfreeze policy. A five-epoch
`H_BASE_BEAM_BUDGET` is trained under identical label exposure. The beam ranks
this explicitly reduced-budget proxy; no claim is made that proxy and
full-budget rankings are identical. Complete candidates that duplicate
mandatory combinations are reused only after hash validation.

The top four complete beam candidates receive full 500k training. Mandatory
Section-14 combinations run even if absent from the top four.

### 17.4 Three-seed confirmation

Confirm:

- `H_BASE`;
- `H_PARTICLENET`;
- both logit-KD baselines;
- best physical `AUX`;
- best deployable `FEEDBACK`;
- best combination;
- `C_PHYSICAL_KD`;
- compatible RETB comparators.

After this graph set is locked, compile
`H_MONO_PARAM_<graph_hash>`, `H_MONO_FLOP_<graph_hash>`, and
`H_PARTICLENET_PARAM_<graph_hash>` for every HOSD graph in it, then confirm
those controls too. Seeds `202,303,404` run for every required row; discovery
seed 101 is reported separately and is not counted in confirmation. A missing
or failed seed makes the row incomplete, not scientifically worse.

### 17.5 Bounded scale shortlist

After complete three-seed 500k confirmation, construct the duplicate-free
union:

```text
H_BASE
best accuracy graph
best mean-log-rejection graph
best physical AUX graph
best feedback graph
best combination graph
H_PARTICLENET if not already present
```

Cap at seven graphs using accuracy order, then rejection order, then canonical
role order. The shortlist is emitted even when every HOSD candidate loses.
All shortlisted graphs retrain from scratch on `scale_train` with refit
train-derived statistics.

---

## 18. Campaign stages

### Stage A: bootstrap and capability audit

- bind source snapshot and exact split/HLT-v3 parent artifacts;
- validate or invoke the shared RETB-semantic producers for 500k identities,
  HLT-v3 caches/audit, four replicas, angular-tree shards/finalization, and
  relation/REGION/shared-HLT normalizers;
- write the target capability audit and complete target registry;
- measure storage/throughput;
- audit split disjointness and forbidden matching fields;
- publish the exhaustive experiment and producer registries.

No model training begins until Stage A validates.

### Stage B: target extraction and teacher lock

- train missing `O_BASE`/`O_FULLREL` teachers, then lock them without reading
  HOSD results;
- fit offline relation/REGION normalizers on `model_train`;
- build every required current-source target cache only for `model_train`,
  `val_stop`, and `val_design`;
- fit train-only target, residual, heteroscedastic, and whitening statistics;
- audit target finiteness, masks, coverage, clipping, class correlation, and
  degradation correlation;
- materialize canonical shuffle-control plans without modifying canonical
  targets.

Unusual target correlations are reported. They do not trigger manual target
redefinition after model results.

### Stage C: exact baselines and frozen probes

- train `H_BASE`, `H_BASE_LONG`, `H_PARTICLENET`, and logit-KD baselines;
- freeze the seed-101 probe encoder;
- train all prior, applicable raw-summary, linear, shallow,
  class-conditional, and target-to-class probes;
- publish a predictability matrix for every target and tap.

All target families continue to Stage D regardless of Stage-C predictability.

### Stage D: all single-family auxiliary arms

- train every target family, registered loss weight, and required
  parameterization;
- train disabled-head, stop-encoder, target-mean, global-shuffle, and
  within-class-shuffle controls;
- train every matched HLT-self auxiliary control;
- publish identity-bound `design_select` predictions and target metrics;
- select one weight/parameterization per family under Section 17.1.

### Stage E: predicted-feedback screen

- run mandatory global-token, FiLM, and HLT-native pair-bias exemplars;
- run the bounded promoted target/interface rows;
- run AUX-only, detached, zero, semantic-loss-disabled, shuffle,
  unrestricted, and oracle controls;
- publish learned gate scales and predicted-versus-oracle feedback gaps.

Oracle rows cannot enter deployable selection.

### Stage F: combinations and gradient conflict

- build mandatory combinations;
- run the bounded beam;
- compare fixed weighting with PCGrad on `C_ALL_BEST`;
- perform residual, gradient, and representation redundancy diagnostics;
- lock the best complete deployable combination definitions.

### Stage G: mechanism and causal controls

- leave one target family out of the selected combination;
- remove auxiliary-only heads at inference and prove exact logit parity;
- zero or shuffle predicted feedback;
- substitute target means and wrong-event predictions;
- compare semantic feedback to unrestricted matched-capacity feedback;
- compare point, residual, and uncertainty-aware reconstruction;
- evaluate whether classification gain tracks target error event by event.

These diagnostics use `design_confirm` and are selection-ineligible. They
confirm the already locked mechanism bundle and cannot reopen Stage-D/F
choices.

### Stage H: robustness

Evaluate only the named confirmation set from Section 17.4 on:

```text
D_OFFLINE_IDENTITY
D_KIN_ONLY
D_TRACK_ONLY
D_MISSING_ONLY
D_NOMINAL
D_MILD
D_SEVERE
D_LEGACY_V1
D_LEGACY_V2
R_FIXED
R_MULTI
R_RANDOM
```

Report by:

- class;
- jet `pT` and absolute `eta`;
- constituent multiplicity;
- valid-track fraction;
- offline-to-HLT target residual magnitude;
- predicted uncertainty.

Robustness results do not reopen target or model selection.

### Stage I: three-seed 500k confirmation

- train every required confirmation graph at seeds 202, 303, and 404;
- authenticate complete per-seed predictions;
- report paired target and classification statistics;
- build the bounded scale shortlist only after complete aggregation.

### Stage J: optional predeclared 3M scale-up

- refit every train-derived statistic on `scale_train`;
- retrain and lock seed-101 `O_BASE` and `O_FULLREL` scale teachers;
- build all scale target caches, teacher outputs, residual statistics, and
  whitening artifacts before scale student training;
- retrain every shortlisted graph from scratch;
- select epochs on `val_stop`;
- run no architecture, target, loss, or feedback redesign;
- publish complete scale checkpoints and HLT-only deployable exports.

This stage occurs after the complete 500k campaign and locked shortlist.

### Stage K: stack selection, controls, and sealed final test

1. run label-free HLT-only inference for every complete scaled graph on
   `stack_val`;
2. separately join authenticated labels and select accuracy and rejection
   finalists;
3. write `locked_hosd_finalists.json`;
4. build selection-ineligible offline-target/oracle diagnostics for finalists;
5. complete finalist semantic, capacity, export, and latency controls;
6. prepare identity-bound final inputs without model inference;
7. write `final_test_execution_lock.json`;
8. run all locked final rows on the common `final_test` exactly once;
9. publish JSON, Markdown, plots, and the completed job ledger.

The final row set is exact: both finalists, `H_BASE`, `H_BASE_LONG`,
candidate-specific parameter/FLOP/ParticleNet matches for each finalist,
ordinary `H_PARTICLENET`, both KD baselines, and each finalist's locked
semantic-control bundle (`AUX_ONLY`, DETACHED, semantic-loss-disabled,
unrestricted, shuffled prediction, zero feedback, and mean-only when HET).
The semantic bundle is the shared selected Stage-E feedback exemplar linked
separately to each finalist for paired comparison; it is not represented as a
candidate-specific architecture or capacity match. Candidate-specific
parameter/FLOP/ParticleNet rows provide that separate capacity comparison.
Oracle rows are reported separately as nondeployable and never enter the
deployable ranking.

If the scale stage is deliberately disabled in the campaign specification
before any scientific result, Stage K applies the identical workflow to
three-seed 500k graphs. A result cannot be used to disable scale-up after the
fact.

---

## 19. Classification, utility, and efficiency metrics

Every classifier reports:

- balanced and ordinary ten-class accuracy;
- cross entropy;
- per-class accuracy and efficiency;
- one-vs-rest AUC;
- complete confusion matrix;
- multiclass Brier score;
- deterministic 15-bin top-label ECE;
- QCD-versus-each-signal rejection at 30% and 50% signal efficiency.

The 15 ECE bins are:

```text
[0,1/15), [1/15,2/15), ..., [14/15,1]
```

Confidence is the largest softmax probability. Lowest class index breaks an
exact top-logit tie. Empty bins contribute zero. ECE is the sample-count
weighted absolute difference between mean confidence and accuracy in each
nonempty bin. It is top-label multiclass ECE, not one-vs-rest ECE.

For signal class `c`, use:

```text
discriminant = p_c / (p_c + p_QCD)
pass rule = score >= threshold
```

Evaluate thresholds in:

```text
{+infinity} union unique observed scores union {-infinity}
```

Choose the threshold whose achieved signal efficiency is closest to the
target. Break ties by larger achieved efficiency, then larger threshold.
Report achieved efficiency. Exact background efficiency is passing QCD count
divided by all QCD jets; rejection is its reciprocal. Display positive
infinity when no QCD jet passes.

For finite selection only, use:

\[
\widehat\epsilon_b
=
\frac{n_{\mathrm{pass}}+0.5}{N_{\mathrm{QCD}}+1},
\qquad
\widehat R
=
\frac{1}{\widehat\epsilon_b}.
\]

Unsmoothed counts and rejection remain the reported physics values.

Define the 18-term mean log rejection:

\[
M_R
=
\frac{1}{18}
\sum_{\substack{c\in\text{nine signals}\\
\epsilon_s\in\{0.30,0.50\}}}
\log \widehat R(c,\epsilon_s).
\]

### 19.1 Paired tagging utility

For candidate `m` and matched `H_BASE` seed:

\[
\Delta_{\mathrm{acc}}(m)
=
\operatorname{Acc}(m)-\operatorname{Acc}(H_{\mathrm{BASE}}),
\]

\[
\Delta_{R}(m)
=
M_R(m)-M_R(H_{\mathrm{BASE}}).
\]

Report both absolute differences and relative error reduction. Positive target
predictability does not modify these quantities.

### 19.2 Offline-gap closure

When `O_BASE` is better than `H_BASE`, define:

\[
G_{\mathrm{acc}}(m)
=
\frac{
\operatorname{Acc}(m)-\operatorname{Acc}(H_{\mathrm{BASE}})
}{
\operatorname{Acc}(O_{\mathrm{BASE}})
-\operatorname{Acc}(H_{\mathrm{BASE}})
}.
\]

For mean log rejection:

\[
G_R(m)
=
\frac{
M_R(m)-M_R(H_{\mathrm{BASE}})
}{
M_R(O_{\mathrm{BASE}})-M_R(H_{\mathrm{BASE}})
}.
\]

If a denominator is nonpositive, the corresponding closure is undefined and
reported as such. Values below zero and above one are not clipped.

### 19.3 Feedback decomposition

Report:

```text
auxiliary gain = metric(A_t) - metric(H_BASE)
feedback gain  = metric(F_t) - metric(A_t)
oracle room    = metric(F_t_ORACLE_TRAINED) - metric(F_t)
substitution   = metric(F_t_ORACLE_SUB) - metric(F_t)
semantic gain  = metric(F_t) - metric(F_t_UNRESTRICTED)
```

Use accuracy and mean log rejection separately. Oracle and unrestricted
controls are interpreted, not combined into one success score.

### 19.4 Efficiency

Report:

- complete and trainable parameters;
- deployed parameters after auxiliary heads are removed;
- analytical training and inference FLOPs;
- measured examples/second;
- peak GPU memory;
- target-cache bytes per jet;
- training GPU-hours;
- single-jet latency;
- batch-1, batch-128, and production-batch latency;
- export size and target-head removal savings.

An expensive model may be an accuracy upper bound, but it may not be called
HLT-deployable without measured resource evidence.

Latency is diagnostic, never a selector tie-break. Its artifact binds GPU
model/UUID, driver, CUDA, PyTorch, export backend, precision, input
multiplicity, and clock/power mode. After 200 synchronized warm-up calls it
records 1,000 synchronized repetitions and reports median, p90, and p95 for
batch 1, 128, and the measured production batch. Measurements from unlike
hardware/software contracts are not compared numerically.

---

## 20. Statistical reporting

All final comparisons use identity-paired predictions and matched seeds.

For every finalist versus every named baseline report:

- per-seed metrics and difference;
- three-seed mean and sample standard deviation;
- paired absolute accuracy difference;
- paired class-stratified bootstrap 95% interval;
- McNemar discordant counts;
- per-class paired differences;
- per-signal rejection intervals;
- paired mean-log-rejection difference interval;
- paired target-error difference when applicable.

The bootstrap is:

```text
10,000 resamples
seed = 917301
paired sampling unit = canonical jet identity
sample within each class at its original balanced count
2.5% and 97.5% quantiles
linear quantile interpolation using r=(n-1)q
```

The same resampled identities are used for both models and all targets.
Rejection thresholds and achieved efficiencies are recomputed inside every
resample. Use Jeffreys-smoothed log rejection for finite intervals and show
unsmoothed central values separately.

For a three-seed comparison, one shared identity resample is applied to every
seed, the paired metric difference is computed per seed, and the arithmetic
seed mean is the bootstrap replicate. Seed variation is also reported
separately as sample standard deviation. This is the inherited
`paired_statistics.py` seed/quantile contract.

Additional conventions are fixed globally:

- `R2=0` for a constant target when prediction SSE equals target SSE, and
  undefined when both target variance and prediction error are zero;
- Spearman uses average ranks for ties, is undefined with fewer than two
  finite pairs or a constant rank vector, and macro aggregation ignores only
  undefined components while reporting the count;
- target ECE uses the same 15 bins and endpoint rule as classification ECE;
- multiclass Brier is the per-event sum of ten squared probability errors,
  then averaged (not divided by ten);
- AUC uses average ranks for score ties and is undefined when either class is
  absent.

Target predictability intervals resample jets, not individual particles or
pairs. Per-jet target losses are computed before resampling, preventing
high-multiplicity jets from becoming multiple statistical units.

---

## 21. Finalist selection and final-test locks

### 21.1 Dual finalists

Every eligible complete graph must be:

- deployable from HLT inputs alone;
- complete at all three matched seeds;
- finite on required metrics;
- source-, cache-, normalizer-, target-, graph-, and checkpoint-valid;
- inference-export parity validated.

Two independent selectors operate on `stack_val`:

`ACCURACY_FINALIST`

: Highest three-seed mean balanced accuracy, then lower cross entropy, lower
inference FLOPs, fewer parameters, canonical graph ID.

`REJECTION_FINALIST`

: Highest three-seed mean `M_R`, then higher balanced accuracy, lower
inference FLOPs, fewer parameters, canonical graph ID.

The finalists may be identical. No minimum improvement is required.

### 21.2 Finalist lock

`selection/locked_hosd_finalists.json` binds:

- campaign specification and source snapshot;
- split, validation partition, and scale-pool hashes;
- HLT profile, replicas, cache, and audit hashes;
- target capability audit and target registry;
- the registry/audit hash and each graph's exact lineage-relevant target
  extractor, cache, mask, normalizer, residual, and teacher hashes; unrelated
  optional caches do not invalidate a graph;
- complete 500k and scale shortlist traces;
- every eligible graph definition and three-seed checkpoint hash;
- deployable export and parity artifacts;
- parameter, FLOP, latency, and training-cost artifacts;
- all label-free `stack_val` prediction shard hashes;
- separate label-manifest hash;
- selector metric artifacts and complete deterministic traces;
- accuracy and rejection finalist IDs;
- positive, zero, or negative gain status.

The lock authorizes post-selection oracle diagnostics but does not authorize
final-test model inference.

### 21.3 Execution lock

After finalist controls complete, write:

```text
selection/final_test_execution_lock.json
```

It binds:

- finalist lock;
- post-lock oracle-target and feedback-diagnostic hashes;
- final input/cache preparation hashes;
- capacity and matched-baseline control completion;
- semantic-control coverage;
- export parity;
- exact final row IDs;
- an unused exactly-once execution claim.

Final inference atomically consumes the claim. Restart may reuse a complete
authenticated final artifact but may not execute inference again. A partial
final shard set is an integrity incident requiring explicit recovery
authorization, not an automatic rerun.

Final-test metrics never choose a replacement model.

---

## 22. Artifact and provenance layout

Campaign root:

```text
checkpoints/hlt_offline_structure_distillation/<campaign_id>/
```

Required layout:

```text
campaign_spec.json
inputs/
  split_manifest.json.gz
  validation_partition_manifest.json.gz
  final_select_label_manifest.json.gz
  scale_train_manifest.json.gz
  raw_input_schema.json
  hlt_v3_profile.json
  hlt_v3_degradation_audit.json
  hlt_replica_manifest.json
  inherited_angular_tree_resource.json
  hlt_replicas/
  region_tree/
capability/
  target_capability_audit.json
registry/
  structure_target_registry.json
  experiment_registry.json
  loss_registry.json
  feedback_registry.json
  parameterization_registry.json
  producer_registry.json
  access_role_registry.json
  determinism.json
teachers/
  teacher_lock.json
  logits/
  latents/
targets/
  canonical/
  hlt_analogues/
  residuals/
  controls/
normalization/
  relation_500k/
  region_500k/
  hlt_shared_500k/
  target_500k/
  residual_500k/
  relation_scale/
  region_scale/
  hlt_shared_scale/
  target_scale/
  residual_scale/
baselines/
probes/
auxiliary/
feedback/
combinations/
mechanism_controls/
robustness/
confirmation_500k/
scale_up/
selection_predictions/
  stack_val/
postlock_oracle_diagnostics/
selection/
final_test/
reports/
job_ledgers/
```

Every immutable JSON artifact uses canonical sorted JSON, a content hash,
parent hashes, contract/schema version, producer entry point, and source
snapshot. Large arrays use identity-bound shard manifests and atomic
publication.

Artifact reuse requires exact agreement of:

- contract and schema;
- source and parent hashes;
- split and identity order;
- HLT profile and replica;
- target definition and normalizer;
- teacher checkpoint;
- graph, parameterization, loss, feedback mode, and seed;
- expected output content hashes.

Path existence is never sufficient.

### 22.1 Deployability audit

Deployability distinguishes two graphs:

1. complete training provenance, which must honestly retain labels, offline
   targets/masks, or teachers used to learn parameters;
2. runtime/export reachability, consisting only of tensors, modules, files,
   and constants loaded or computed during inference.

The audit fails only when the runtime/export graph reaches:

```text
offline_input
offline_target
target_loss_mask
truth_target
oracle_feedback
teacher_state
class_label
degradation_source_lineage
```

Predicted means, variances, probabilities, structure tokens, FiLM parameters,
and HLT-by-HLT pair biases are permitted only when produced inside the graph
from HLT inputs. A trained weight may have privileged training ancestry;
erasing that ancestry is itself an integrity failure. Identity-free
train-fitted deployable statistics from Section 9.1 are permitted runtime
constants with their full provenance. Tests prove that an AUX export with its
head removed passes while an export that opens a target cache fails. FEEDBACK
exports retain the HLT-side predictor and feedback module; only AUX-only
exports are target-head-free.

---

## 23. Storage, compute, and Tigris policy

Persistent dense pair targets and attention tensors are forbidden. Pair
metrics stream in deterministic chunks. Target cache shards publish
atomically and support exact resume after validating all completed hashes.

Before submission, the preflight reports:

- available and projected peak storage;
- projected target extraction time;
- projected GPU-hours by stage;
- maximum concurrent jobs;
- checkpoint and export sizes;
- Slurm walltime and memory requests from measured smoke evidence.

Tigris workers:

```text
source the declared conda environment
export PYTHONNOUSERSITE=1
prepend ${CONDA_PREFIX}/lib to LD_LIBRARY_PATH
set PYTHONDONTWRITEBYTECODE=1
print python executable and dependency versions
validate campaign source before artifact reuse
```

Wrappers validate source and campaign parents at startup even when every row
appears reusable. Array workers never assume the current shell environment.

Default production resources must come from authenticated miniature
measurements. The plan does not guess them.

Every mutable CLI supports `--dry-run`. Every Slurm submitter supports:

```text
--dry-run
--smoke-submit
--full-submit
```

Full submission requires a real miniature campaign using the same workers,
continuation logic, target extractors, feedback exports, and locks as
production.

---

## 24. Proposed implementation surfaces

Create:

```text
teacher_logit_reco/hlt_offline_structure_distillation/
  __init__.py
  contracts.py
  campaign.py
  capability.py
  target_registry.py
  target_common.py
  target_jet.py
  target_composition.py
  target_track.py
  target_density.py
  target_tree.py
  target_relations.py
  target_hlt_pair.py
  target_teacher.py
  target_controls.py
  target_cache.py
  target_normalization.py
  residuals.py
  teacher_lock.py
  encoder_taps.py
  split_forward.py
  heads.py
  probes.py
  feedback_tokens.py
  feedback_film.py
  feedback_pair.py
  losses.py
  pcgrad.py
  training.py
  baselines.py
  combinations.py
  robustness.py
  capacity.py
  evaluation.py
  statistics.py
  selection.py
  deployment.py
  provenance.py
  reporting.py
  workflow.py
```

Reuse rather than fork:

```text
jetclass_fresh.part_inputs
teacher_logit_reco.relational_part
teacher_logit_reco.relation_expert_token_bridge.splits
teacher_logit_reco.relation_expert_token_bridge.hlt_v3
teacher_logit_reco.relation_expert_token_bridge.hlt_cache
teacher_logit_reco.relation_expert_token_bridge.replicas
teacher_logit_reco.relation_expert_token_bridge.determinism
teacher_logit_reco.relation_expert_token_bridge.evaluation
teacher_logit_reco.relation_expert_token_bridge.paired_statistics
```

Reuse requires exact semantic and contract compatibility. HOSD does not import
an RETB scientific registry ID merely because a helper implementation is
shared. `particle_tap` and `layerwise_pair_bias` are references/tests, not
drop-in implementations for the new split-forward and post-block-4 feedback
contracts.

### 24.1 Command-line entry points

Create:

```text
scripts/build_hosd_campaign.py
scripts/build_hosd_shared_hlt_parents.py
scripts/build_hosd_tree_parents.py
scripts/fit_hosd_relation_normalizers.py
scripts/lock_hosd_inherited_parents.py
scripts/audit_hosd_target_capability.py
scripts/measure_hosd_storage.py
scripts/build_hosd_targets.py
scripts/audit_hosd_targets.py
scripts/fit_hosd_target_normalizers.py
scripts/train_hosd_offline_teacher.py
scripts/lock_hosd_teachers.py
scripts/infer_hosd_teacher_targets.py
scripts/train_hosd_baseline.py
scripts/train_hosd_probe.py
scripts/aggregate_hosd_predictability.py
scripts/train_hosd_auxiliary.py
scripts/select_hosd_single_targets.py
scripts/train_hosd_feedback.py
scripts/select_hosd_feedback.py
scripts/train_hosd_combination.py
scripts/analyze_hosd_gradient_conflict.py
scripts/run_hosd_mechanism_controls.py
scripts/evaluate_hosd_robustness.py
scripts/aggregate_hosd_confirmation.py
scripts/select_hosd_scale_shortlist.py
scripts/train_hosd_scale.py
scripts/prepare_hosd_scale_inputs.py
scripts/build_hosd_scale_tree.py
scripts/fit_hosd_scale_normalizers.py
scripts/compile_hosd_scale_teacher_adapters.py
scripts/build_hosd_scale_native_relations.py
scripts/infer_hosd_stack_val.py
scripts/select_hosd_finalists.py
scripts/build_hosd_postlock_oracles.py
scripts/audit_hosd_deployment.py
scripts/write_hosd_final_test_execution_lock.py
scripts/evaluate_hosd_final_test.py
scripts/write_hosd_report.py
scripts/monitor_hosd_campaign.py
scripts/materialize_hosd_runtime_inputs.py
scripts/write_hosd_runtime_config_template.py
scripts/prepare_hosd_execution.py
scripts/submit_hosd_slurm.py
scripts/snapshot_hosd_scheduler.py
scripts/interrupt_hosd_miniature_coordinate.py
scripts/capture_hosd_scheduler_evidence.py
scripts/verify_hosd_miniature.py
scripts/measure_hosd_miniature_resources.py
scripts/compile_hosd_resource_measurements.py
scripts/preflight_hosd_resources.py
scripts/authorize_hosd_campaign.py
scripts/complete_hosd_job_ledger.py
```

Every command resolves paths to absolute canonical locations, validates
source/parents before reuse, and prints or writes its immutable configuration.

### 24.2 Slurm entry points

Create:

```text
sbatch/hosd_common.sh
sbatch/run_hosd_bootstrap.sh
sbatch/run_hosd_hlt_cache.sh
sbatch/run_hosd_tree_shards.sh
sbatch/run_hosd_tree_finalize.sh
sbatch/run_hosd_relation_normalization.sh
sbatch/run_hosd_target_build.sh
sbatch/run_hosd_teacher_wave.sh
sbatch/run_hosd_baseline_array.sh
sbatch/run_hosd_probe_array.sh
sbatch/run_hosd_auxiliary_array.sh
sbatch/run_hosd_feedback_array.sh
sbatch/run_hosd_combination_array.sh
sbatch/run_hosd_controls_array.sh
sbatch/run_hosd_robustness_array.sh
sbatch/run_hosd_confirmation_array.sh
sbatch/run_hosd_scale_array.sh
sbatch/run_hosd_stack_val.sh
sbatch/run_hosd_final_test.sh
sbatch/submit_hosd_tigris_full.sh
```

The full DAG uses after-success dependencies for integrity, not performance.
Scientific selectors complete successfully with negative results. Failed
runtime rows block only dependent integrity consumers until repaired; they do
not alter the registered scientific matrix.

---

## 25. Required tests

### 25.1 Capability and data access

- current canonical source marks all three genuine GN2 target families
  unavailable;
- adding a branch name without authenticated schema evidence cannot enable a
  target;
- degradation construction indices are absent from dataset and target APIs;
- offline and HLT views pair only by canonical jet identity;
- source, event, and identity disjointness holds across splits;
- every role rejects unauthorized splits and artifact classes;
- a deployable worker cannot import or open target-cache readers.

### 25.2 Exact target semantics

- `T_OFFLINE_JET_10`, `COMPOSITION_16`, `TRACK_32`, `DENSITY_22`,
  `CA_TREE_26`, and proxy-17 dimensions and names are exact;
- float64 jet-vector sums match an independent reference;
- angle wrapping is continuous across `-pi/pi`;
- quantiles use the declared linear interpolation;
- empty and one-element cases produce exact masks and finite stored zeros;
- track validity and uncertainty floors match RPT;
- PID/charge categories match RPT exactly;
- C/A Python and compiled backends emit identical target summaries within the
  inherited continuous tolerance and exact topology;
- tree cluster ordering and entropy zero cases are deterministic;
- proxy component edges, connected components, ranking, caps, and overflow
  counts are exact;
- relation aggregation excludes diagonals, distinguishes directed from
  undirected pairs, and respects applicability masks;
- HLT-self extractors match their offline-family semantics on identical
  inputs;
- TRACK-pair 13 and REGION-pair eight channels, direction/symmetry, masks, and direct
  RPT controls are exact;
- builder APIs expose no labels and output bytes remain unchanged when the
  separate label manifest is permuted;
- semantic changes fail old contract validation.

### 25.3 Target caches and normalizers

- shards cover each expected identity exactly once;
- worker completion order does not alter canonical reassembled arrays or the
  identity-order semantic hash; production shard boundaries are fixed, and
  artifact bytes/hashes must match only under those fixed boundaries;
- interrupted shards resume identically;
- corrupt or stale hashes cannot be reused;
- no validation identity contributes to a target normalizer;
- `R_MULTI` offline targets are identical across replicas;
- residual targets bind the exact HLT replica;
- train/scale normalizer lineages cannot be interchanged;
- missing target masks never enter stored deployable inputs;
- canonical, global-shuffled, and within-class-shuffled caches are distinct
  immutable artifacts.

### 25.4 Encoder and auxiliary heads

- disabled taps preserve Weaver FP32 logits, gradients, masks, and state dict;
- real-Weaver split-forward parity is tested at blocks 2, 4, and 8 in FP32
  with mixed precision disabled;
- auxiliary-head removal preserves classifier logits exactly;
- `STOP_ENCODER` produces zero auxiliary gradient in shared parameters;
- per-jet loss reduction is invariant to padded particles and pair chunking;
- deterministic pair sampling is invariant to batch and worker layout;
- `lambda=0` matches the disabled-head control;
- `ABS`, `RES`, and reconstructed residual inverse transforms are exact;
- heteroscedastic log variance clips exactly and remains finite.
- availability-group order, broadcasting, BCE reduction, partial-family
  masks, and predicted-feedback probabilities are exact.

### 25.5 Feedback

- zero-initialized feedback graphs match `H_BASE` logits before training;
- zero-gated token feedback does not expand the live sequence, masks, or pair
  biases;
- FiLM scale/shift bounds are exact;
- pair feedback is symmetric when required and diagonal/invalid values are
  zero;
- gate warm-up uses the exact integer update rule for `T=1`, short, and normal
  runs;
- detached feedback blocks consumer gradients but preserves target gradients;
- primary training and export reject oracle targets;
- shuffled prediction controls are invariant to batch layout;
- probabilistic feedback uses means/probabilities, not argmax or unseeded
  sampling.

### 25.6 Controls and selection

- within-class shuffles preserve class marginals and destroy event identity;
- global shuffles use the declared complete balanced population;
- target-mean control preserves masks without exposing offline availability;
- every single-family row completes even after a poor probe;
- every selector emits a candidate when all gains are negative;
- null, capacity, mechanism, oracle, and report-only rows can never enter a
  scientific selector; reference baselines cannot be mislabeled HOSD;
- `design_select` alone chooses graphs and `design_confirm` cannot reopen a
  choice;
- target error cannot override a worse classifier before its declared
  tie-break position;
- beam width, family order, mandatory combinations, and duplicate reuse are
  deterministic;
- missing optional RETB parents yield `not_applicable`, not zero metrics;
- incomplete seed coverage cannot become an eligible finalist.
- the campaign compiler enforces every numerical row bound in Section 16.1.

### 25.7 Metrics and statistics

- accuracy, class order, ECE edges, endpoint inclusion, empty bins, and logit
  ties match the inherited deterministic contract;
- rejection discriminant, threshold set, pass ties, achieved efficiency, and
  zero-background display are exact;
- Jeffreys selection rejection is finite and distinguishes zero, one, and two
  passing QCD jets;
- all 18 mean-log-rejection terms recompute in every bootstrap;
- bootstrap resamples canonical jet identities within balanced classes using
  seed 917301 and declared linear quantiles;
- target statistics resample jets rather than particles or pairs;
- gap closure is undefined for nonpositive denominators and never clipped.

### 25.8 Provenance, export, and final seal

- wrapper startup validates source even when all artifacts are reusable;
- deployability preserves privileged training provenance but rejects every
  forbidden runtime/export dependency;
- research and exported logits agree under the dtype-specific tolerance;
- AUX exports are target-head-free; FEEDBACK exports retain their HLT-side
  predictor/feedback modules and both have the reported parameter/FLOP count;
- label-free `stack_val` shards contain no labels or targets;
- selector alone can join the authenticated label manifest;
- prelock `stack_val` oracle generation fails;
- all prelock final-test model inference fails;
- finalist lock binds every prediction, label, metric, checkpoint, target,
  normalizer, capacity, and trace hash;
- final inference requires both locks and consumes exactly one execution
  claim;
- negative scientific final results still produce a complete report and job
  ledger.

### 25.9 Real miniature acceptance

On Tigris, a miniature campaign must:

- use real JetClass input and the real HLT-v3 degradation;
- build at least one target from every current-source family group;
- train a baseline, probe, AUX, and feedback graph;
- exercise an all-negative synthetic selector fixture without cancelling
  continuation;
- interrupt and resume one target shard and one training row;
- validate an HLT-only export;
- traverse stack selection and the two final locks on miniature held-out
  identities;
- leave no manually injected row JSON or repaired path.

Unit tests and a synthetic DAG do not replace this acceptance.

---

## 26. Implementation order

### Step 1 of 12: contracts, campaign, and inherited parents

Implement canonical hashing, campaign specification, source binding, split
roles, producer registry, access roles, and exact validation of reusable
RETB/RPT parents, including current-source rebuild wrappers for HLT caches,
trees, and relation/REGION normalizers.

Done when a dry-run campaign can enumerate every Stage A-K artifact and job
without creating scientific output.

Implementation status: complete on 2026-07-30. The implementation lives in
`teacher_logit_reco/hlt_offline_structure_distillation/`, with campaign and
shared-parent wrapper CLIs in `scripts/build_hosd_*.py` and
`scripts/fit_hosd_relation_normalizers.py`. A distinct
`scripts/lock_hosd_inherited_parents.py` post-rebuild lock preserves the
immutable initial parent-status history. The focused Step-1 tests are
`tests/test_hlt_offline_structure_distillation_step1.py`.

### Step 2 of 12: capability audit

Implement raw-schema inspection, availability classes, forbidden matching
fields, future-data gates, and the immutable target registry compiler.

Done when current JetClass truth/vertex targets fail closed as unavailable
while current-source targets compile.

Implementation status: complete on 2026-07-30. The source-bound capability
audit and deterministic registry compiler live in
`teacher_logit_reco/hlt_offline_structure_distillation/capability.py`, with
canonical component/channel declarations in `target_schemas.py` and the
production wrapper in `scripts/audit_hosd_target_capability.py`. The current
14-field JetClass schema compiles 31 required current-source targets; the
optional authenticated RETB-token comparator remains non-executable without
its coordinate lock, and the three truth/vertex proposals remain fail-closed
behind explicit new-schema/cache/campaign gates. Numerical extraction remains
reserved for Step 3. Focused regressions are in
`tests/test_hlt_offline_structure_distillation_step2.py`.

### Step 3 of 12: physical target extractors

Implement and test jet, composition, track, density, tree, relation-aggregate,
track-component-proxy, HLT-self, and HLT TRACK/REGION pair extractors with
exact masks and numerical semantics.

Done when independent fixtures and real miniature jets validate every
component.

Implementation status: source-complete and locally validated on 2026-07-30;
the real-miniature portion of this acceptance criterion remains pending on
Tigris. The label-blind numerical
implementation is in
`teacher_logit_reco/hlt_offline_structure_distillation/extractors.py`.
It covers all 28 current physical target IDs, reuses one implementation for
each offline/HLT-self pair, calls the inherited RPT raw builders before learned
normalization, consumes the parity-validated C/A tree contract, and emits
zero-filled boolean-masked global or transient pair targets. The extractor
manifest is bound into the immutable Step-2 target registry. Independent
edge-case and reader-shaped miniature-jet coverage is in
`tests/test_hlt_offline_structure_distillation_step3.py`; teacher inference,
cache publication, and fitted target-normalizer hashes remain Step 4.

### Step 4 of 12: target cache, normalization, and teachers

Implement resumable identity-bound caches, HLT analogues, residuals,
heteroscedastic metadata, target controls, normalizers, teacher locking, and
teacher inference.

Done when complete 500k projections and a real miniature cache pass lineage
and resume tests.

Implementation status: source-complete and locally validated on 2026-07-30;
the authenticated real miniature, measured 500k projection, and filesystem
resume acceptance remain pending on Tigris. Step 4 is implemented in
`teacher_logit_reco/hlt_offline_structure_distillation/target_cache.py`,
`normalization.py`, `controls.py`, `storage.py`, and `teachers.py`, with
source-validating producers under `scripts/`. Cache shards are label-blind,
identity sorted, byte deterministic, immutable, fixed-boundary resumable, and
fail closed on stale lineage or bytes. The implementation includes exact
HLT-replica residual binding, model-train/scale-train-only robust statistics,
HET metadata, the fixed four-axis conditional-residual backoff, canonical
latent whitening, deterministic ridge fitting, distinct mean/global-shuffle/
within-class artifacts, 500k/3M storage projections, produced-checkpoint
teacher locks, FP32 teacher output capture, and the normalized 128-dimensional
`O_BASE` latent tap contract. Miniature lineage, corruption, resume,
worker-order, normalizer isolation, control, whitening, teacher-lock, and
storage tests are in
`tests/test_hlt_offline_structure_distillation_step4.py`.

### Step 5 of 12: ParT taps, baselines, and probes

Implement the new parity-safe standard-H_BASE split forward at blocks 2/4/8,
target heads, exact `H_BASE`/`H_BASE_LONG`,
ParticleNet and KD controls, and every frozen probe.

Done when disabled code paths preserve Weaver parity and Stage C runs
automatically.

Implementation status: source-complete and locally validated on 2026-07-30;
authoritative real-Weaver parity and automatic real-miniature Stage-C
completion remain pending on Tigris. The parity-safe opt-in Weaver
adapter and exact `H_BASE` wrapper are in
`teacher_logit_reco/hlt_offline_structure_distillation/taps.py`; it executes
Weaver's unmodified forward under temporary block hooks, so the disabled path
adds no module, state key, tensor operation, or RNG call. It exposes exact
post-block-2/4/8 states and masks and reserves the post-block-4 replacement
interface needed by Step 7. The locked global, particle, heteroscedastic, and
directional/symmetric pair heads plus SHA256-only stratified pair sampler are
in `heads.py`. Exact `H_BASE`, 80-epoch `H_BASE_LONG`, matched ParticleNet,
both temperature-2 KD rows, and the HLT-native seven-relation auxiliary are
frozen in `baselines.py`; fixed-budget resumable training is in
`stage_c_training.py`. Frozen linear/shallow/raw/statistical/
class-conditional/target-to-class probes and their metrics are in `probes.py`.
The source-bound Stage-C compiler emits all six baselines and 157 probe jobs
covering the exact 18 eligible target families without performance pruning.
Production CLIs are `scripts/train_hosd_baseline.py`,
`scripts/build_hosd_probe_taps.py`,
`scripts/materialize_hosd_probe_inputs.py`,
`scripts/materialize_hosd_pair_probe_inputs.py`,
`scripts/train_hosd_probe.py`, and
`scripts/aggregate_hosd_predictability.py`; authoritative real-Weaver FP32
logit, gradient, mask, and state parity is
`scripts/validate_hosd_weaver_parity.py`. Focused fake-Weaver, head, sampler,
baseline, KD, probe, and matrix-bound regressions live in
`tests/test_hlt_offline_structure_distillation_step5.py`.

### Step 6 of 12: auxiliary training

Implement per-event losses, fixed weights, `ABS/RES/HET`, checkpointing,
gradient controls, shuffles, target-mean controls, and single-family
selection.

Done when every Stage-D row completes regardless of target performance.

Implementation status: complete on 2026-07-31. The immutable 213-row
Stage-D matrix, phase-two resolution, global/pair/KD heads, ABS/RES/HET
losses, availability prediction, fixed weights, target shuffles,
target-mean and STOP_ENCODER controls, resumable fixed-budget training,
source-bound loader manifests, result aggregation, and fail-open scientific
selection are implemented under
`teacher_logit_reco/hlt_offline_structure_distillation/auxiliary.py`,
`stage_d_data_factory.py`, and `stage_d_training.py`. Production entry points
are `scripts/build_hosd_stage_d_loader_manifest.py`,
`scripts/train_hosd_auxiliary.py`, and
`scripts/select_hosd_single_targets.py`; focused acceptance coverage is in
`tests/test_hlt_offline_structure_distillation_step6.py`.

### Step 7 of 12: predicted feedback

Implement zero-gated residual token adapters, FiLM, predicted and
direct-computed HLT pair bias, zero initialization, gate
warm-up, detached/end-to-end paths, semantic/unrestricted controls, oracle
diagnostics, and HLT-only exports.

Done when Stage E passes parity, causal-control, and deployability tests.

Implementation status: source-complete and locally validated on 2026-07-31,
pending the required real Weaver/Tigris miniature. Token, FiLM, predicted-pair,
and direct-computed-pair feedback execute at the locked post-block-4 split;
zero gates are exact no-ops, detached/end-to-end gradients are explicit, and
warm-up is an integer optimizer-update rule. The unrestricted control uses a
literal four-query direct-token branch with exact inert trainable padding,
rather than semantic target coordinates. Runtime, loader, selection, oracle,
and HLT-only export contracts are in `feedback.py`, `stage_e_data_factory.py`,
and `stage_e_training.py`, with CLI and regression coverage in
`scripts/train_hosd_feedback.py`, `scripts/select_hosd_feedback.py`, and
`tests/test_hlt_offline_structure_distillation_step7.py`.
The production loader uses the graph-independent `hosd_data_order_v1` sampler
seed for discovery, confirmation, and scale manifests. Exact-HLT TRACK and
REGION rows are reference baselines, are ineligible to become
`BEST_FEEDBACK`, and remain separately eligible for later overall finalist
selection. `MEAN_ONLY` is exactly HET-parameter-matched with inert trainable
padding and a hashed capacity ledger. Exact builders publish separate hashed
non-multiply operation profiles; authoritative scale efficiency must add the
measured `hosd_exact_hlt_builder_timing_v1` evidence before production
selection lineage is complete.

### Step 8 of 12: combinations and mechanism controls

Implement mandatory combinations, bounded beam, fixed-weight normalization,
PCGrad control, redundancy metrics, leave-one-out, and mechanism
interventions.

Done when Stages F-G produce complete deterministic selection traces.

Implementation status: complete on 2026-07-31. Stage F implements the locked
family-order bounded beam, omit/reuse semantics, the six mandatory full fits,
and a separately registered selection-ineligible PCGrad control. The
`C_NATIVE_OFFLINE` graph now contains the actual seven-family 545-dimensional
native-relation auxiliary alongside the selected offline physical target.
Stage G has executable leave-one-out retrains, exact head-removal parity, and
identity-paired eventwise error/gain controls. Implementations and focused
tests are in `combinations.py`, `combination_runtime.py`,
`mechanism_runtime.py`, `scripts/train_hosd_combination.py`,
`scripts/run_hosd_mechanism_controls.py`, and
`tests/test_hlt_offline_structure_distillation_step8.py`.

### Step 9 of 12: metrics, robustness, and reporting

Implement exact classification/target/efficiency metrics, paired statistics,
degradation evaluations, plots, JSON, and Markdown reports.

Done when reports include positive and negative results without manual text
edits.

Implementation status: complete on 2026-07-31. Exact multiclass target
statistics, inherited classification/rejection metrics, shared paired
bootstrap, strict degradation/subgroup results, synchronized efficiency
profiling, deterministic plots, and JSON/Markdown reporting are implemented
in `metrics.py`, `efficiency.py`, and `reporting.py`. The production
robustness worker performs HLT-only deployable inference before a separate
label join. Entry points and focused coverage are
`scripts/evaluate_hosd_robustness.py`,
`scripts/profile_hosd_efficiency.py`, `scripts/plot_hosd_report.py`,
`scripts/write_hosd_report.py`, and
`tests/test_hlt_offline_structure_distillation_step9.py`.

### Step 10 of 12: confirmation and scale

Implement three-new-seed confirmation, graph-specific capacity controls,
complete aggregation, bounded scale shortlist, scale teacher/target refits,
retraining, and deployable export validation.

Done when an all-negative miniature still selects and executes its registered
scale analogue.

Implementation status: source-complete on 2026-07-31, pending the required
real miniature execution. Locked graph definitions, exact instantiated
capacity grids, analytical FLOP ledgers, graph-specific deterministic
controls, strict three-seed confirmation receipts, duplicate-free all-negative
shortlisting, scale teacher/target refits, exhaustive graph/seed receipts,
and all-seed HLT-only export auditing are implemented in
`graph_registry.py`, `capacity.py`, and `confirmation.py`. Entry points and
focused tests are under `scripts/*hosd*confirmation*`,
`scripts/compile_hosd_capacity_controls.py`,
`scripts/train_hosd_scale.py`, `scripts/prepare_hosd_scale_inputs.py`,
`scripts/build_hosd_scale_tree.py`,
`scripts/fit_hosd_scale_normalizers.py`,
`scripts/compile_hosd_scale_teacher_adapters.py`,
`scripts/build_hosd_scale_native_relations.py`,
`scripts/export_hosd_deployable_graph.py`,
and `tests/test_hlt_offline_structure_distillation_step10.py`.
Stage J no longer requires future scale views, tree caches, refit
normalizers, or teacher-adapter configurations to be injected as runtime
inputs: those artifacts are produced and authenticated in dependency order
after the locked 500k shortlist. Scale HLT views preserve the HLT-v3
measurement-state tensor, and the 3M NPZ publisher streams deterministic ZIP
members rather than assembling the complete archive in memory. The five
scale input views and five tree populations run as bounded array coordinates.
The four same-view native-relation targets are produced once in their own
exact-coverage wave before any scale graph starts; graph/seed workers cannot
race to create or overwrite them.
Scale offline, HLT, and native-relation arrays use authenticated NPY memory
maps. Tree construction reads those maps and retains only its current output
shard; physical-target construction publishes canonical target shards
directly and resolves only the tree shard or boundary-adjacent shards
intersecting the current identity range. A single shared authenticated
tree-split index verifies exact shard filenames/count, metadata and NPZ
hashes, parents, and canonical identity coverage before target, graph, or
resource consumers can obtain a lazy shard path; it revalidates shard bytes
again immediately before loading. Cache validation streams identity coverage instead of constructing a
population-wide identity list, set, dictionary, or ordered tree tuple.
Persisted target consumers and REGION-tree consumers likewise retain at most
one shard per bound family/replica. Graph workers bind REGION trees and
native-relation targets only when their locked graph definition requires
them. Production resource derivation requires completed representative tasks
for `scale_input_prepare`, `scale_tree_build`, `scale_target_build`, and the
worst-case `scale_graph_train`. Each source-bound projection is derived from
authenticated NPY/NPZ layout byte accounting, records its coordinate type and
resident shard size, and must remain within the registered Tigris limit.

### Step 11 of 12: stack selection and final seal

Implement label-free stack inference, separate label join, dual selectors,
finalist lock, post-lock oracle diagnostics, execution lock, exactly-once final
test, and final reporting.

Done when leakage, reuse, interruption, and exactly-once tests pass.

Implementation status: source-complete on 2026-07-31, pending the required
real miniature execution. Stack predictions are label-free and identity
bound; the selector alone joins labels. The first lock binds every
prediction, metric, checkpoint, export, and lineage hash. The second lock
derives the exact finalist graph/seed/checkpoint/export rows rather than
accepting caller-selected rows. Final execution creates one durable claim and
supports interruption-safe resume through immutable per-row receipts without
creating a second claim. Implementation and coverage are in
`finalization.py`, `deployment_runtime.py`, the Stage-K `scripts/` entry
points, and `tests/test_hlt_offline_structure_distillation_step11.py`.

### Step 12 of 12: production DAG and Tigris acceptance

Implement all Slurm workers, exhaustive plan factories, resource preflight,
monitoring, restart/recovery, smoke submission, and full authorization.

Done only after the real miniature acceptance in Section 25.9. Source
implementation alone is insufficient.

Implementation status: source-complete on 2026-07-31, pending the required
real Tigris miniature acceptance. Source-validating Slurm wrappers, immutable
submission attempts, registered plan factories for every node,
dependency-closure recovery, intentional target/training-coordinate
interruption evidence, measured Slurm resource derivation, strict runtime
configuration, resource preflight, real-JetClass/HLT-v3 label-blind view
materialization, child-job-waiting inherited-parent rebuilds, deterministic
plots, terminal job-ledger attestation, and full authorization bound to the
same-source miniature are implemented. The exact operator workflow is
`teacher_logit_reco/HOSD_RESEARCH_COMPUTE_RUNBOOK.md`. Production readiness
remains intentionally false until that real miniature traverses Sections
25.9 and 29; no local or synthetic test can set the acceptance artifact.
In particular, the full authorization contract is version 6 and fails closed
unless the resource preflight binds all four representative Stage-J memory
projections. Population-resident input preparation is projected to 3M from
authenticated per-event layout bytes. Bounded tree workers are projected at
the exact 10,000-event production shard, target workers at the exact
2,048-event production shard, and worst-case graph workers at the simultaneous
production tree/input/target resident combination. Tree bytes use the
analytical 128-leaf/255-node maximum and are checked against observed
miniature layouts, so a miniature final shard smaller than production cannot
reduce the authorized request. A missing coordinate, stale source or layout
hash, incorrect production resident-unit size, or projection above its
registered Tigris limit prevents authorization.
Runtime path values must be exact authenticated file/directory bindings;
literal paths, unresolved placeholders, incomplete replica keys, invalid
resource scalars, and bound-file drift are rejected before worker execution.
The Stage-A storage node directly measures every current physical extractor
on a real label-blind input and publishes its own source-bound evidence, so
neither storage nor Slurm resource policy accepts hand-authored measurement
values. The inherited RETB campaign-storage contract and the HOSD
target-cache storage measurement remain distinct artifacts; the latter enters
the HOSD resource preflight and is never passed as an RETB campaign parent.

Final source-closure audit on 2026-07-31: all 112 HOSD regression tests pass,
all 139 HOSD Python modules and entry points compile, and all 89 HOSD
command-line parsers complete `--help`. Stage J produces its scale inputs,
trees, offline/shared-HLT normalizers, scale teacher adapters, and optional
four-replica native-relation wave inside the authenticated DAG. Large NPZ and
runtime-input hashes are streamed, shared wave publication is race-safe, and
scale graph checkpoints bind the complete input/teacher/target/label
lineage. The sole unexecuted acceptance item is the deliberately external
real Tigris miniature and its two interruption/recovery exercises; their
absence cannot be replaced by a local artifact or marked passed in source.

---

## 27. Interpretation of possible outcomes

### Targets are predictable but do not improve tagging

The reconstructed quantities are redundant with evidence already used by
`H_BASE`, or the auxiliary constraint consumes useful capacity. Report the
negative utility result; do not promote predictability as success.

### Targets are only partly predictable but AUX improves

This supports the central representation-shaping hypothesis. Exact offline
reconstruction is unnecessary; conditional evidence is enough to regularize
useful HLT features.

### AUX improves but feedback does not

Structure prediction is valuable as a training objective, but its explicit
interface is a bottleneck or later layers already internalize the information.
The paper story becomes structured auxiliary distillation, not feedback.

### Feedback improves beyond AUX

This supports the stronger claim that inferred physical structure should
actively condition later transformer reasoning.

### Unrestricted feedback matches semantic feedback

The gain is attributable to an extra learned branch rather than the named
physical interface. The semantic novelty claim is weakened.

### Physical targets beat KD

Intermediate supervision adds information or optimization structure beyond
matching the teacher's final answer.

### KD matches all physical targets

The simplest useful offline supervision is output-level distillation.
Physical target results remain diagnostic but do not justify a more complex
deployment.

### Deterministic track-component proxy helps

Report it only as a proxy result. It motivates validation with authenticated
vertex data but does not establish true secondary-vertex reconstruction.

### Latent targets help while physical targets do not

Teacher representation transfer is useful, but the physical intermediate
targets are incomplete or poorly aligned. Avoid interpretability claims.

### ParticleNet remains stronger

HOSD may improve ParT without becoming the best matched HLT-input classifier.
Report both facts. A ParT-specific mechanism claim may remain valid.

### Nothing improves

Complete all registered rows, scale/final workflow as configured, and report
the upper bounds, predictability matrix, null controls, and negative paired
intervals. No result is replaced or hidden.

### Authenticated richer data become available

Create a new campaign/schema version. Do not mix truth-origin/vertex results
with deterministic-proxy results under one target ID or aggregate headline.

---

## 28. Definition of scientific success

HOSD is scientifically successful at increasing levels:

### Level 1: useful auxiliary supervision

- at least one physical offline-derived target improves over matched
  `H_BASE`;
- it also improves over the identical-budget HLT-self auxiliary for that
  family, or the claim is limited to physics-aligned self-supervision;
- the gain survives disabled-head, stop-encoder, target-mean, and shuffled
  controls;
- three-seed paired intervals support the direction of the gain.

### Level 2: structured mechanism

- semantic feedback improves beyond AUX-only;
- the primary DETACHED predictor satisfies the locked fidelity and
  availability tolerances on `design_confirm`;
- predicted feedback approaches its separately trained oracle ceiling;
- unrestricted matched-capacity feedback does not fully explain the gain.

### Level 3: competitive HLT-like classifier

- the complete graph improves accuracy and rejection over `H_BASE`,
  graph-specific `H_MONO_PARAM_*`, `H_MONO_FLOP_*`, `H_PARTICLENET`, and
  logit KD;
- gains persist across declared degradation controls;
- compute and latency are fully reported.

### Level 4: detector-relevant evidence

- a new authenticated paired HLT/offline dataset enables genuine structure
  targets;
- gains survive detector-domain and data/simulation validation;
- measured resources support the intended HLT use.

Failure to reach a higher level does not erase a lower-level result.

---

## 29. Definition of production readiness

The campaign is ready for full research-compute submission only when:

1. all twelve implementation steps are complete;
2. every current-source target has a real extractor, test, and producer;
3. every manifest-driven node has an executable plan factory;
4. source, split, cache, target, teacher, model, and selection lineage validate;
5. the deployability auditor proves HLT-only inference;
6. negative performance cannot cancel registered continuation;
7. storage and resources are measured rather than guessed;
8. restart, reuse, source-drift, partial-array, and interrupted-lock tests pass;
9. the real Tigris miniature traverses Stages A-K;
10. full-submission authorization binds the successful miniature and exact
    source snapshot.

A valid Markdown plan, unit tests, synthetic outputs, or a structurally valid
DAG is not production readiness.
