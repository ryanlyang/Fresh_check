# Grouped attention for ParT: physics, learned, and physics-guided groups

Date: 2026-10-17  
Status: initial four-arm implementation completed locally; queue preparation and GPU admission before production. No jobs submitted during implementation. See the [queue and recovery guide](../part_group_screen/README.md).

Implementation roadmap: [GROUPED_ATTENTION_IMPLEMENTATION_PLAN.md](GROUPED_ATTENTION_IMPLEMENTATION_PLAN.md), including code boundaries, control reuse, staged work packages, tests, and launch gates.

Quick navigation:

- [Comparison registry](#3-comparison-registry) and [recommended defaults](#4-shared-backbone-and-first-version-defaults).
- [Shared architecture and equations](#5-shared-grouped-attention-module).
- [Fixed physics groups](#6-candidate-a-fixed-physics-groups), [fully learned groups](#7-candidate-b-fully-learned-groups), [physics-guided groups](#8-candidate-c-physics-guided-learned-groups), and [random control](#9-control-d-random-groups-with-matched-sizes).
- [Fair comparisons](#10-fair-comparison-and-necessary-controls) and [training/evaluation](#11-training-evaluation-and-parallel-execution).
- [Stronger follow-ups](#12-stronger-extensions-to-retain-for-later), [acceptance checks](#13-diagnostics-and-implementation-acceptance-checks), and [implementation order](#14-implementation-order-and-remaining-architecture-details).
- [Prior work](#15-prior-work-and-claim-boundaries).

## 1. Purpose and what is decided

The central question is:

> Does explicitly organizing particles into groups improve ordinary ParT, and is it better to prescribe that organization, learn it, or combine both?

The study has three main candidates: fixed physics grouping, fully learned grouping, and physics-guided learned grouping. Ordinary ParT is the reference. A random-grouping control is the preferred additional condition because it tests whether meaningful organization matters beyond adding another summary-and-communication pathway.

**This study inherits the [Reusable Screen Protocol](../part_ev_screen/REUSABLE_SCREEN_PROTOCOL.md).** It is a new architectural experiment within that benchmark, not a separately chosen training recipe. Use its `ranger_short_v1` recipe, seed 101, data streams, precision/runtime contract, checkpoint selection, and full-validation evaluator. Reuse the donor's `PART_RANGER` results instead of submitting another baseline training run. The optional `EV_RANGER` comparison can also reuse existing results.

Control reuse means reusing verified comparison artifacts, **not warm-starting grouped models from trained ParT weights**. Every new model trains from scratch with the donor-matched initialization of unchanged backbone weights. The reusable protocol and donor's frozen contracts govern training and comparison; this document specifies the new architectures. The separate grouped launcher and control registry are implemented in `part_group_screen/` and `part_screen_common/`; the original seven-model screen is unchanged.

Implement the fixed-physics version first, then the two learned versions, using the same downstream group branch. Once validated, the candidates can train in parallel. No candidate needs to wait for another candidate's performance results before implementation.

**"Recommended version" below means a concrete, well-motivated starting design, not an empirically established best model.** Widths, group counts, routing iterations, and locality strength remain hypotheses. Freeze a complete version before collecting comparative results; revisions become separately identified experiments.

This is an offline, ten-class JetClass study. It is separate from:

- [The EdgeValue context/gating screen](CONTEXT_GATING_SCREEN_PLAN.md), which changes value messages rather than grouping.
- [The older multi-scale subjet plan](../teacher_logit_reco/MULTISCALE_SUBJET_PARTICLE_TRANSFORMER_PLAN.md), which targets a different HLT, binary-classification setting. Its data, warm starts, and training assumptions do not transfer automatically.

The initial grouped models do **not** include EV, selected PT/TRACK/REGION features, a new optimizer, or restricted particle attention. Those would be additional experimental factors.

## 2. Why this is a distinct hypothesis

ParT already learns contextual particle representations and uses physics-derived pairwise attention bias. It is not incapable of learning collective structure. The proposed change explicitly represents intermediate collections of particles and gives them a communication pathway inside the backbone.

A token need not represent one particle. Attention between two group tokens is ordinary pairwise attention operating on summaries of many particles. No three-index attention tensor or enumeration of all particle triplets is required.

This does not create new measured information. It changes how information already present in the same retained constituents is represented and processed. The possible benefit is a useful inductive bias, easier optimization, or a more effective allocation of model capacity.

The initial aim is better classification at measured additional cost, not a speedup claim: full particle attention remains present.

## 3. Comparison registry

The donor control IDs already exist. Grouped run IDs below are implemented in `part_group_screen/config.py`. Every row uses `ranger_short_v1` and seed 101.

| Run ID | Model / assignment mechanism | Role | New training? |
|---|---|---|---|
| `PART_RANGER` | Ordinary shared-bias ParT; no group branch | Required donor reference | **No: reuse** |
| `PART_GROUP_FIXED_RANGER` | ParT + fixed angular-agglomeration groups | Main candidate | Yes |
| `PART_GROUP_LEARNED_RANGER` | ParT + competitive soft neural groups | Main candidate | Yes |
| `PART_GROUP_HYBRID_RANGER` | Same neural router plus angular-locality preference | Main candidate | Yes |
| `PART_GROUP_RANDOM_RANGER` | ParT + random memberships with fixed-arm group sizes | Preferred extra control | Yes |
| `EV_RANGER` | Shared-bias ParT + EV; no group branch | Optional donor reference | **No: reuse** |

The implemented screen therefore has **four new training runs**, not five or six: three grouping strategies plus random grouping. There is no runtime switch to silently omit an arm. Each new run needs its own selected-checkpoint evaluation; completed donor evaluations are reused as well. The primary contrasts are each grouped model minus `PART_RANGER`, fixed minus random, and hybrid minus learned. A contrast against `EV_RANGER` compares two different mechanisms; it does not isolate the effect of adding grouping to EV.

All grouped arms use the same group-token construction, group self-attention, particle readback, insertion point, and residual connection. Fixed and random grouping have no trainable router. Learned and hybrid grouping should have identical trainable router shapes; the initial hybrid changes only the use of a fixed geometric penalty.

This does not make the conditions parameter- or compute-matched. Measure both. In particular, a random-group control does not replace a capacity control for a larger learned router.

## 4. Shared backbone and first-version defaults

Use the donor's frozen reference backbone, corresponding to [model.py](model.py), not a newly approximated transformer or an unverified newer implementation:

- Pinned Weaver `0.4.17` ParticleTransformer implementation.
- Official JetClass full-feature preprocessing, 17 particle input features, ten classes, and a maximum of 128 particles.
- Model width 128; eight attention heads; eight particle blocks; two class-attention blocks.
- Native particle FFN width 512.
- Original shared four-channel kinematic pairwise-bias pathway.
- Native particle embedding, normalization, dropout, padding, sequence trimming, and class-token readout.
- No EV contribution and no selected-relation feature families.

Verify source identities against the donor metadata when implementing rather than silently accepting another installed Weaver version. Inherit the screen's [precision context](../part_ev_screen/precision.py) and [data adapters](../part_ev_screen/data.py), including native trimming, continuous streams, and isolated initialization of added modules. The new branch must not perturb the unchanged backbone initialization or data/training RNG streams. The default runtime is the donor-matched single-GPU SPORC A100 environment, not the older Tigris full-scale setup; Section 11 specifies the reuse checks.

### Recommended initial choices

| Component | Proposed starting choice |
|---|---|
| Available group slots | `K = 8` |
| Active slots for low-multiplicity jets | `min(K, number of valid retained particles)` |
| Branch insertion | After particle block 4, before block 5 |
| Group width / attention heads | 128 / 8 |
| Group-token pooling | Assignment-weighted mean of particle hidden states |
| Physical summaries | Same seven features for every grouped arm; defined below |
| Group processing | One pre-normalized self-attention block with a 128→256→128 GELU FFN |
| Readback | One cross-attention operation: particle queries, group keys/values |
| Added branch dropout | 0 initially; native backbone dropout unchanged |
| Residual scale | One trainable scalar, initialized to `1e-3` |
| Additional group-pair attention bias | None initially |
| Learned routing | Three assignment rounds, temperature 1 |
| Hybrid locality coefficient | Fixed `lambda = 1` as an initial proposal |
| Auxiliary grouping loss | None initially; diagnose collapse before adding one |

These numerical defaults require correctness and numerical-stability probes. Necessary fixes can be made before performance screening, but must be frozen and documented consistently across affected conditions.

The middle insertion point gives the router contextual particle features while leaving later particle blocks to use the returned group information. A front-end router before the first particle block is a useful later ablation, not a separate placement used only by one primary candidate.

## 5. Shared grouped-attention module

### 5.1 Data flow

```text
Official retained particle inputs and mask
                    |
       Native ParT embedding + blocks 1–4
                    |
       Particle hidden states H, vectors p, mask
                    |
         +----------+--------------------------------+
         |                                           |
   Assignment matrix A                               |
   (fixed / learned / hybrid / random)                |
         |                                           |
   Pool hidden states and physical summaries         |
         |                                           |
   Up to eight group tokens                          |
         |                                           |
   Group self-attention + FFN                        |
         |                                           |
   Particle-query / group-key-value attention        |
         |                                           |
   Small learned residual correction ----------------+
                    |
          Native ParT blocks 5–8
                    |
       Original class-attention readout and logits
```

All real particles remain in the backbone. Particles can retrieve information from every active group, not just their own group. The final class token still reads particles; groups do not get an extra independent classifier in the initial version.

### 5.2 Assignment contract

For one jet, let `N` be its number of retained valid particles, `H` its `N x 128` hidden states, and `A` its `N x K` assignment matrix. For valid particles:

$$
A_{ik}\geq 0,\qquad \sum_k A_{ik}=1.
$$

Fixed and random groups use one-hot rows. Learned and hybrid groups use soft rows. Padded particles contribute zero to every group. Slots beyond the active slot count are masked before routing and attention.

This is a soft partition: a particle may contribute fractionally to several groups, but its total membership is one. Do not confuse it with ordinary cross-attention normalized over particles separately for each group, which does not enforce this partition property.

The slot number has no physical meaning. Do not add arbitrary group-index positional embeddings to the common downstream branch. Learned router queries may specialize, but a slot is not a labeled quark, prong, or decay product.

### 5.3 Pooling learned and physical information

Define effective occupancy, pooled hidden state, and group four-momentum:

$$
n_k=\sum_i A_{ik},\qquad
\bar h_k=\frac{\sum_i A_{ik}h_i}{n_k+\epsilon},\qquad
P_k^\mu=\sum_i A_{ik}p_i^\mu.
$$

Use the same assignment matrix in all three calculations. For a hard group, `n_k` is its constituent count; for a soft group, it is an effective occupancy. The group four-momenta sum to the retained input jet four-momentum when all slot contributions are included. This bookkeeping identity does not make a soft group a conventional reconstructed subjet.

Proposed seven-component physical descriptor:

1. Log of group transverse momentum divided by the scalar sum of retained constituent transverse momenta, with a positive floor.
2. `log(1 + group_mass / 1 GeV)`.
3. Group rapidity relative to the retained input jet axis.
4. Sine of the wrapped group-to-jet azimuthal difference.
5. Cosine of that difference.
6. `log(1 + effective_occupancy)`.
7. `log(1 + angular_width)`, with width measured in dimensionless angular units.

The group mass follows from `P_k`; its direction also follows from `P_k`. The width is an assignment- and constituent-pT-weighted RMS angular distance from that direction. The jet axis and all denominators are calculated from the same retained constituents, not a larger raw collection.

Use these descriptors for every grouped condition, including random groups. Clamp roundoff-induced negative mass-squared and specify stable zero-mass, zero-pT, and empty-group behavior. Check gradients at these limits for soft grouping. Coordinate differences must wrap azimuth properly.

An initial token builder is:

$$
G_k=\operatorname{LayerNorm}\left(W_h\bar h_k+
\operatorname{MLP}_{\mathrm{phys}}(f_k)\right),
$$

with a shared `7→64→128` GELU physical-feature MLP. Here "shared" means the same module is used for every group within a model; each training run has its own trained parameters. Newly fitted preprocessing statistics, if needed, must come from training data only and be identical across arms. The recommended first version uses the explicit transforms above without a new fitted scaler.

Nearly empty soft groups require guarded denominators. Use an identical, frozen numerical masking policy in both learned arms; do not tune an occupancy threshold separately to improve one model. Record how much assignment mass any numerical mask removes. Exact momentum bookkeeping refers to all groups before such masking.

### 5.4 Group interaction and readback

Use a standard residual pre-LayerNorm transformer block on `G` to produce `G_tilde`. Its FFN has hidden width 256. Then use particle hidden states as queries and updated group tokens as keys and values:

$$
M=\operatorname{MHA}_{\mathrm{cross}}
\left(\operatorname{LN}_H(H),
\operatorname{LN}_G(\widetilde G),
\operatorname{LN}_G(\widetilde G)\right),
\qquad H'=H+\gamma M.
$$

MHA includes its ordinary learned projections. There is no extra particle FFN in this added readback step. Return zero correction to padded queries; mask invalid group keys. Group self-attention likewise excludes invalid slots.

Initialize `gamma = 1e-3` to start near the baseline while allowing gradients into the whole branch. A forced `gamma = 0` is a correctness test, not the default training setting. Do not zero both the gate and every output path, which could prevent the branch from starting to learn.

## 6. Candidate A: fixed physics groups

### Mechanism

Use a deterministic, Cambridge/Aachen-inspired angular agglomeration with four-vector addition:

1. Start with one cluster per retained valid constituent.
2. Calculate rapidity–azimuth distances between current clusters.
3. Merge the closest pair, adding their four-vectors and combining memberships.
4. Recalculate distances involving the new cluster.
5. Stop when `min(8,N)` clusters remain.

This is a specified fixed-count angular merge tree. It should not be casually equated with an arbitrary inclusive FastJet radius call: beam distances, stopping rules, recombination, and treatment of low multiplicity must match the intended contract. Standard clustering tools and conventions are described in the [FastJet manual](https://arxiv.org/abs/1111.6097).

Membership depends on measured geometry, not learned hidden states or truth labels. It is fixed for a given retained input set. During training, native trimming can change that set, so assignments must remain aligned with the particles actually processed. Reuse the assignment within that forward pass; do not recluster separately for pooling and readback.

### Why this version

It gives groups an explicit spatial interpretation, keeps particle accounting simple, and needs no auxiliary training objective. Pooling the current hidden states still allows the content of each fixed group to be learned.

Eight groups are intermediate summaries, not an assumption of eight physical prongs. Their sizes need not be equal. A single global group or very coarse clustering could hide useful structure; excessively many groups would approach the original particle representation.

### Risks and stronger follow-ups

- Hard assignments can change abruptly near a clustering boundary.
- A fixed count can split a coherent core or retain a small soft cluster.
- Per-jet clustering can become a data-loading or device-synchronization bottleneck at 100M scale.
- Physical clusters are not guaranteed to be the optimal classification groups.

First follow-up: compare `K=4` and `K=8` using the same branch. The more ambitious extension is a nested `8→4` hierarchy from the same merge tree, described in Section 12. Radius-based or kT-based grouping is another later factor, not an unrecorded substitution after results arrive.

## 7. Candidate B: fully learned groups

### Mechanism

Use a small competitive router operating on the block-4 particle hidden states. This borrows the idea of differentiable assignments and iterative slots, without claiming to reproduce either published method exactly. [DiffPool](https://arxiv.org/abs/1806.08804) learns soft graph-cluster assignments; [Slot Attention](https://arxiv.org/abs/2006.15055) iteratively organizes inputs through competitive attention.

Recommended router:

- Eight distinct trainable initial slot vectors of width 128, shared across jets as initial parameters.
- Particle keys/values obtained by shared linear projections of normalized hidden states.
- Three assignment rounds in total, followed by common pooling using the final assignments.
- A shared GRU-cell slot update between rounds, driven by assignment-weighted particle values.
- Temperature `tau = 1`; no hard argmax or sampling in the initial version.
- No explicit angular penalty, radius restriction, or clustering-based initialization.

For each round, scores and assignments are:

$$
s_{ik}=\frac{q_k^\top k_i}{\sqrt{128}},\qquad
A_{ik}=\operatorname{softmax}_{k}(s_{ik}/\tau).
$$

The softmax is over **groups for each particle**. Slot updates use weighted averages over particles with occupancy normalization. Recompute assignments after updating slots. Gradients pass through the assignments, pooling, and recurrent updates; no detached routing or straight-through hard partition is needed.

Only the final assignment matrix enters the common group-token builder. Do not additionally inject learned slot states into the downstream group tokens in this first comparison: that would introduce an extra representational path absent from fixed and random grouping.

### Interpretation

"Fully learned" means no explicit clustering/locality rule. It does not mean physics-blind: the input particle states already contain kinematic information and have passed through ordinary ParT pairwise attention.

The classifier can learn spatial groups, particle-type groups, or other task-useful collections. Do not call the resulting slots physical subjets without checking them. Classification supervision does not uniquely identify a true partition.

### Risks and stronger follow-ups

Watch for one dominant group, duplicate slots, nearly uniform memberships, and unstable assignments. Start without a balancing loss; equal-sized groups are not a physical requirement. If collapse occurs, test a separately versioned weak diversity or assignment-sharpness regularizer and apply an equivalent policy to the hybrid router where appropriate.

Possible later improvements are additional refinement rounds, context-dependent slot initialization, or adaptive group usage. None is presumed to improve performance. Use the same tuning budget as the other candidates.

## 8. Candidate C: physics-guided learned groups

### Mechanism

Use the same trainable router, initialization, number of rounds, and temperature as Candidate B. Add a geometric preference to the assignment logits:

$$
s^{\mathrm{hybrid}}_{ik}
=\frac{q_k^\top k_i}{\sqrt{128}}
-\lambda\frac{\Delta R(i,c_k)^2}{2r_{\mathrm{jet}}^2}.
$$

Here `c_k` is a current soft group center, and `r_jet` is a common scale describing the retained jet's angular extent. The negative term favors proximity but does not forbid assigning a particle elsewhere.

Recommended initial definition:

1. Round 1 uses the neural-only assignments, exactly as in Candidate B.
2. Before rounds 2 and 3, calculate centers from the preceding round's assignments.
3. Calculate each center's rapidity by an assignment-pT-weighted mean and its azimuth by a weighted circular mean.
4. Use the retained jet's pT-weighted RMS angular radius, floored at `0.05`, as `r_jet`.
5. Start with fixed `lambda = 1` and `tau = 1`; these are proposed settings to freeze before comparison, not known optima.

The centers used by the locality term are routing centers; they need not equal the direction of the summed group four-vector used in the common physical descriptor. Keep these definitions distinct. Guard empty centers and degenerate circular means; use a documented jet-axis fallback. Use wrapped azimuthal distances throughout.

With `lambda = 0`, this router must reproduce Candidate B when weights and numerical settings match. This is a required equivalence test. Keep centers and their dependence on soft assignments differentiable in the nominal design, with finite-gradient safeguards.

### Interpretation

This tests whether a soft locality preference improves learned organization. It is neither a hard physical partition nor unrestricted learned routing. It must still be allowed to produce unequal occupancies and to use only a few groups strongly.

The first-round neural bootstrap avoids confounding the primary learned-versus-hybrid contrast with an additional clustering-based initialization. Physics-seeded slots are a legitimate later hybrid variant, but their initialization and runtime changes should be identified separately.

### Risks and stronger follow-ups

- Too large a locality penalty can overwhelm useful learned compatibility.
- Too small a penalty may be effectively indistinguishable from learned routing.
- Broad or coincident initial groups may yield unhelpful centers.
- A radius scale that behaves poorly on narrow jets can destabilize routing.

If this arm is promising, a small validation-only locality-strength sweep and a physics-seeded initialization are reasonable follow-ups. Learning the locality strength is another extension; if it can approach zero, report whether the model actually retained the prior.

## 9. Control D: random groups with matched sizes

### Mechanism

For each retained jet:

1. Compute the fixed-physics arm's group-size multiset.
2. Randomly partition the same constituents into those sizes, without using proximity to choose memberships.
3. Apply the identical group-token construction and attention branch used by Candidate A.

This preserves group count, occupancy distribution, and branch parameter count relative to fixed grouping. It intentionally destroys coherent geometric membership, not every trace of physics: the sizes themselves come from clustering, and the constituent inputs remain physical.

Use a deterministic, label-independent partition rule per event with a separately frozen grouping seed. It must be independent of worker scheduling and training RNG state and consistent under reordering of constituents. Original within-event constituent identifiers can be carried as routing metadata, never supplied as predictive features. Do not use class-encoded filenames or labels to construct memberships. If the loader cannot yet supply suitable stable identifiers, resolve that explicitly before claiming a reproducible control.

**Implemented resolution:** native trimming does not expose the original constituent indices, so version `canonical_post_trim_input_bits_v1` uses a stateless content key from the unchanged post-trim 17 input features and four-vectors. Canonical full-input ordering breaks distinguishable hash ties; an order-independent aggregate of particle hashes provides the jet key. Signed zero is canonicalized. Grouping seed is 731019, separate from the training RNG. No labels, filenames, batch positions, or learned hidden states enter this rule. Membership is fixed for the same retained set; native trimming can change that set. Identical duplicate inputs are interchangeable, with tests on physical membership and output tolerance. This preserves the original loader and trimmer rather than adding an unverified identifier channel.

Do not freshly randomize groups every optimization step in this control; that would also introduce a group-augmentation effect. A different random grouping seed can be a later sensitivity check.

### What the comparison establishes

Fixed versus random helps distinguish meaningful spatial organization from a generic extra group pathway. It is especially useful because related patch-based models can work with non-geometric grouping; PHAT-JeT examines ordering choices including random ordering in its own setting. This does not predict the outcome for our architecture. [PHAT-JeT](https://arxiv.org/html/2605.21789v1).

If fixed and random perform similarly, do not conclude grouping has no value. The useful effect could be generic intermediate summaries or extra processing rather than spatially coherent membership. Conversely, a fixed-over-random improvement is evidence about membership structure within this design, not a complete proof against every capacity explanation.

## 10. Fair comparison and necessary controls

### Shared input and compute accounting

Use the same retained constituent view, training/validation event populations, preprocessing, native backbone, global batch, optimizer, update budget, and checkpoint-selection policy. Neither grouped arm receives truth assignments, truth particle origins, extra detector measurements, or particles discarded by the baseline input contract.

Report:

- Trainable parameter counts, including every router, projection, gate, and normalization.
- Training and inference throughput at matched batch sizes.
- Peak GPU memory and CPU/data-loading cost.
- Grouping time separately where measurable, and end-to-end time including it.
- Attempted updates, successful updates, and AMP-overflow skips.

Group self-attention uses `K^2` scores and readback uses `NK`, in addition to native `N^2` particle attention. Pooling, projections, recurrent routing, and clustering also cost time. Do not infer total runtime from score counts alone or claim a linear-time full model.

### Capacity and mechanism controls

The initial priority is the reused ordinary baseline plus three new candidates and the new random control. If an apparent gain warrants confirmation:

1. Add an ordinary-ParT capacity control sized to the actual added parameter budget. The existing EV-matched baseline is not automatically matched to these new models.
2. Consider a comparable-compute control separately; equal parameters do not imply equal compute.
3. Test whether a single global summary (`K=1`) explains the improvement. This tests multiple groups versus one summary, not an exactly compute-matched replacement.
4. If needed, remove physical group descriptors from all compared grouped arms together. This tests whether gains arise from explicit aggregate observables, learned group communication, or both.
5. Generic summary-token attention without partition competition can test whether learned partitioning matters beyond a latent-token bottleneck.

Do not add all these conditions to the first screen by default. Choose follow-ups based on the specific mechanism claim that needs testing, and keep their development decisions off the final test set.

## 11. Training, evaluation, and parallel execution

### 11.1 Inherited recipe: no new epoch or LR decision

Follow the [Reusable Screen Protocol](../part_ev_screen/REUSABLE_SCREEN_PROTOCOL.md) and the donor's frozen `campaign.json`. The following is a convenience summary, not an independently maintained recipe. Any discrepancy must be resolved against those contracts before launch.

| Setting | Required first grouped screen |
|---|---|
| Training population | Official JetClass Pythia 100M, all files |
| Validation population | Official 5M validation jets |
| Seed | 101 for every new model and reused control |
| Global batch / hardware | 512; one donor-matched SPORC A100, no accumulation |
| Budget | 200,000 attempted updates; 102.4M jet exposures |
| Reporting epochs | 10 checks, one per 20,000 attempted updates |
| Checkpoint validation | 1,280,000 examples per check |
| Selection | Highest validation accuracy; first checkpoint wins exact ties |
| Early stopping | None |
| Final screening evaluation | Selected checkpoint on all 5M validation jets |
| Optimizer recipe | `ranger_short_v1`, unchanged |

Ranger uses the pinned Weaver RAdam plus state-complete Lookahead wrapper: LR `1e-3`, betas `(0.95,0.999)`, epsilon `1e-5`, weight decay zero, Lookahead `k=6` and `alpha=0.5`, no warmup or clipping. Apply the 15 LR drops at zero-based steps `140000,144000,...,196000`, ending at `1e-5`. Set LR before the attempted update: the first drop affects update 140,001. **Reuse [part_ev_screen.protocol.learning_rate](../part_ev_screen/protocol.py), not a new scheduler or the million-step schedule stopped early.**

Training and checkpoint-validation iterators remain continuous across checks. A reporting epoch means 10.24M exposures, not one full pass through 100M distinct jets. An AMP-skipped update still consumes its batch and advances the attempted-step/LR counters; record skips without adding replacement updates.

Reuse the complete loader and numerical contract: unchanged `JetClass_full.yaml`, feature/class ordering, 128-particle limit, native wrap padding/masks and trimming, data-stream seeds/order, two workers, and `fetch_step=0.01`. Train with FP16 autocast, FP32 parameters, GradScaler, and native single-GPU BatchNorm; validate/evaluate in FP32, with TF32 and cuDNN benchmarking off. Reuse the screen precision context for **both outer and Weaver-native autocast**. Match the donor GPU model/memory capacity and software environment, not just the label "A100".

The first screen does not add an optimizer sweep. A later AdamW arm would inherit the entire `adamw_short_v1` bundle and reuse `PART_ADAMW` (optionally `EV_ADAMW`), subject to the same checks. Do not compare a new AdamW model against `PART_RANGER` and call that a grouping-only contrast.

### 11.2 Donor controls: read-only, never training-array entries

Use this donor campaign:

```text
/home/ryreu/atlas/Fresh_check/checkpoints/part_ev_screen/context_gate_ampfix_20261016T215608Z
```

`PART_RANGER` is the required 2,143,486-parameter baseline. `EV_RANGER` is the optional 2,209,022-parameter EV reference. Both use seed 101. Do not substitute the historical small-data models, full-budget Tigris results, or plain/no-pair-bias models. Donor job IDs locate artifacts but do not establish completion; this plan makes no claim that the donor results have finished.

Create a read-only-reference `controls.json` in the **new grouped campaign**, following the reusable protocol's exporter/validator design:

1. Validate and record the donor campaign content hash, frozen source/preprocessing hashes, data inventory, complete recipe, task/seed, precision, and runtime. Verify the shared fields; a new architecture naturally has a different overall campaign hash.
2. Check `runs/<RUN_ID>/seed_101/training_complete.json`, completion at 200,000 attempted steps, all ten validation checks, `optimizer.json`, and the selected checkpoint hash. Verify highest-accuracy/first-tie selection rather than trusting a filename.
3. For reusable final metrics, validate `selection/locked_checkpoints.json`, `full_validation/metrics.json`, all prediction-file sidecars/hashes, and file/entry coverage of exactly 5M events, 500k per class. Retain logits/predictions and event identities for paired comparisons.
4. Record verified reference paths, hashes, selected check, recipe, and status. Missing or unfinished artifacts are **pending**, not zero-valued results or a passed control.

A pending control need not delay new-model training, but it blocks the final matched comparison. A completed, compatible control needs **neither retraining nor repeated inference** for the same population and metrics. Do not load its trained `best_model_val.pt`, `last.pt`, probe weights, or AMP-recovery scaler state as initialization for grouped training.

Control reuse is the planned workflow, not an optional afterthought. If compatibility fails, stop and resolve the mismatch; do not silently queue a replacement baseline. Changing seed, budget, batch/accumulation, recipe, preprocessing, or training precision requires a new matched control. Moving from the donor SPORC runtime to Tigris likewise requires reviewed comparability evidence or new controls. Queue concurrency and walltime can change without changing the recipe.

### 11.3 Validation, results, and interpretation

After each new model completes all ten checks, lock its accuracy-selected checkpoint and evaluate it on the same complete 5M validation population as the donor. Successive checkpoint checks can contain different subsets; the full-population evaluation provides the common comparison. Reuse the existing [screen driver](../scripts/run_part_ev_screen.py), [reporting](../part_ev_screen/reporting.py), and [metric definitions](metrics.py) rather than inventing a second evaluator.

Report accuracy, cross-entropy, macro one-versus-one AUC as the principal AUC summary, separately labeled macro one-versus-rest AUC, and per-class QCD rejection using the screen's score/threshold-tie conventions. Compute AUC over the full evaluation population, not by averaging minibatch AUCs. Report 30% and 50% signal efficiency for Hbb, Hcc, Hgg, H4q, Wqq, Zqq, and Tbqq, with Hqql at 99% and Tbl at 99.5% separately labeled. Keep accepted-background counts and achieved efficiencies; do not invent finite rejection values for zero QCD acceptances.

The combined report joins new and donor predictions by verified event identity. Accuracy differences are `100 * (new_accuracy - control_accuracy)` percentage points. Use one accuracy-selected checkpoint for every metric, not separate per-class winners. Include parameter counts and measured cost. While runs are incomplete, compare common completed check numbers or best-through-common-check accuracy, not best-so-far values from unequal budgets.

The full 5M population overlaps checkpoint-selection validation. It is a **screening validation comparison**, not an independent test; the official 20M test is not used here. A seed-101 comparison does not estimate across-seed variability. Later matched additional seeds or full-budget confirmation are separate studies with their own compatible controls; the present short-budget control cannot substitute for another seed or budget. No extra confirmation runs are included in this plan's initial queue.

### 11.4 Separate experiment registry and implementation contract

The grouped registry/launcher is implemented separately. The following is a conceptual summary, not a YAML file consumed by the launcher; `experiment_spec.json` is constructed and frozen by prepare. Use the [queue guide](../part_group_screen/README.md) for executable commands.

```yaml
schema: part_group_screen_v1
name: grouped_attention_seed101
donor_campaign: /home/ryreu/atlas/Fresh_check/checkpoints/part_ev_screen/context_gate_ampfix_20261016T215608Z
donor_campaign_sha256: "<verified campaign.json content_hash>"
recipe_ref: ranger_short_v1
seed: 101
controls:
  - PART_RANGER
  # Optionally also EV_RANGER; read-only, never a new training task.
new_models:
  - run_id: PART_GROUP_FIXED_RANGER
    architecture: fixed
  - run_id: PART_GROUP_LEARNED_RANGER
    architecture: learned
  - run_id: PART_GROUP_HYBRID_RANGER
    architecture: hybrid
  - run_id: PART_GROUP_RANDOM_RANGER
    architecture: random
initialization: fresh_seeded
evaluation: official_validation_5M
```

Import the complete donor recipe/data/runtime contracts, not only these configuration labels. Freeze the group settings from Sections 4–9 and their implementation separately. The implemented registry includes the random arm. Any later change to that registry requires a new version/campaign, not a reused approval. It does not create another baseline task.

Use a separate extensible model/task registry. **Do not append these models to the active seven-task screen's `TASKS`, change its frozen `source/`, or assume its hard-coded lock/report accepts new models.** Reuse shared scheduler, optimizer, loader, precision, checkpoint, and evaluation semantics while leaving the original campaign behavior unchanged. Existing optimizer code is in [optimizer.py](optimizer.py); isolated model initialization is demonstrated in the [screen model](../part_ev_screen/model.py).

### 11.5 Scheduling and isolation

Implement and validate sequentially, then allow the new models to train in parallel on the donor-matched runtime. The intended dependency sequence is:

`prepare -> new-model GPU checks -> review -> new-model training -> full-validation evaluation -> combined report`

Verified donor results are an additional prerequisite for the combined report, not new training-array or inference-array entries. Each grouped model needs its own admission checks; the donor's GPU/AMP-recovery approval does not automatically approve a new architecture.

Keep all pre-production walltimes **below six hours**. Measure sustained training, validation, and grouping throughput to choose production walltimes. Include real data loading and concurrent filesystem/CPU contention; do not reduce batch size, change precision, or add accumulation to fit a new model while still claiming the unchanged control contract.

Freeze new code/configuration and use separate run directories. Do not modify another chat's experiments, overwrite donor artifacts, or install over an actively used environment. Partition/account and concurrency must respect available permissions and donor-compatible hardware. Actual submission remains a later authorized action, not part of this documentation update.

## 12. Stronger extensions to retain for later

These are candidates for subsequent studies, not claims about known best settings.

### 12.1 Multi-scale groups: first priority after the initial screen

Represent fine and coarse organization simultaneously, such as eight groups followed by four parent groups. For fixed clustering, use nested cuts of one merge tree. For learned grouping, learn a second assignment from fine groups to coarse groups.

If `A` maps particles to fine groups and `B` maps fine to coarse groups, with both row-normalized, the composed assignment `C = AB` preserves the particle-membership sum. Build coarse four-momenta from fine four-momenta. When pooling hidden states, carry occupancies as well: simply averaging already normalized fine-group means incorrectly gives small and large groups equal constituent weight.

Allow fine–coarse communication and particle readback. Compare against a single-scale branch with a similar parameter/compute budget. A hierarchical gain should not be attributed solely to hierarchy if it also doubles the branch size.

### 12.2 Adaptive group usage

Allow up to eight slots but learn which are useful for each jet. Start by measuring occupancy and effective group number; avoid forcing equal usage. Explicit presence gates or sparsity mechanisms can be introduced later with a defined objective and empty-slot policy.

This might adapt to variable jet structure, but adds regularization and optimization choices. Apparent unused slots are not automatically a failure; universal collapse or redundant identical groups is more concerning.

### 12.3 Front-end versus mid-network grouping

Compare routing immediately after the input embedding against routing after block four, while controlling the rest of the branch. Early routing directly tests the proposed neural preprocessing idea; later routing has more contextual features. Do not compare these placements under different recipes and attribute the result only to grouping quality.

### 12.4 Recompute groups at multiple depths

Rebuild assignments and summaries at two or more depths so the grouping can evolve with particle representations. This is distinct from three internal router iterations at one insertion point. It adds compute and possibly parameters and requires its own control.

### 12.5 Physics bias between groups

Derive pairwise features from group four-momenta and add a separate group-attention bias. This could expose relationships among composite objects. Keep it separate initially: group four-vectors have a different distribution from particles, so blindly reusing the existing particle pair encoder and its normalization is not justified.

### 12.6 Combine grouping with EV only after isolating grouping

If grouping helps, test the factorial comparison ParT, ParT+EV, ParT+groups, and ParT+groups+EV under one matched protocol. The combination could be complementary or redundant. Do not assume the individual gains add.

## 13. Diagnostics and implementation acceptance checks

The plan should be implemented with a common assignment interface, a common group-feature builder, and one common communication branch. Avoid four independent copies of the model.

Before production training, require:

1. **Baseline preservation:** with the added residual contribution forced off, evaluation logits agree with the pinned baseline within a fixed numerical tolerance. Preserve backbone initialization with isolated RNG for added modules.
2. **Membership accounting:** valid rows sum to one; padded rows and unavailable columns are zero; hard variants have one valid membership per particle.
3. **Permutation behavior:** reorder particles, masks, four-vectors, and routing metadata together. Grouped outputs must remain equivalent up to slot relabeling/numerical tolerance. Clustering tie-breaking must not depend on arbitrary array order; exactly degenerate cases need documented handling.
4. **Physics bookkeeping:** group four-vectors sum to the retained jet vector before numerical slot masking. Test wrapped azimuth, narrow jets, single-particle jets, and nonuniform group sizes.
5. **Gradient checks:** learned assignments and hybrid centers receive finite gradients from the classification objective. No accidental detachment, hard argmax, or undefined empty-group derivatives.
6. **Hybrid equivalence:** with locality strength zero and matched weights, hybrid matches fully learned routing.
7. **No hidden inputs:** grouping does not read padding copies, truncated constituents, labels, class-coded identifiers, or privileged truth information.
8. **RNG and restart behavior:** grouping randomness is isolated; resume restores optimizer/Lookahead/router state and does not change the data or random-group assignment rules.
9. **Runtime checks:** benchmark complete training steps including grouping and real data loading, not just attention kernels.
10. **Parameter audit:** count all trainable parameters and check that all intended modules affect outputs/receive gradients. Do not pad counts with unused parameters.
11. **Control compatibility:** validate donor recipe/data/source/runtime/precision and completion artifacts. Reject incompatible controls; retain unfinished ones as pending and block the final report without blocking otherwise admitted new training.
12. **Inherited execution semantics:** test Ranger LR boundaries at zero-based steps 139999/140000 and 195999/196000, AMP skips and attempted counters, continuous input fingerprints, FP32 evaluation, selection ties, and full 5M coverage. Compare initial unchanged backbone weights and data-stream fingerprints against the donor contract.
13. **Queue isolation:** a dry-run/mock submission contains only the three main grouped tasks plus the optional random task. It never trains or reevaluates complete donor controls, changes the active seven-task registry, or writes into donor directories.

Record lightweight diagnostics on a fixed development sample: assignment entropy per particle, per-group occupancy and pT share, effective group count, within-group angular spread, slot similarity, residual-gate magnitude, and correction-to-backbone norm. Plot example assignments without using truth information to construct them; class labels may be used afterward for diagnostic summaries.

If random groups work as well as physical groups, or learned groups are diffuse but useful, report that result directly. If none helps, the conclusion is about these versions and budgets, not proof that all hierarchical representations are useless.

## 14. Implementation order and remaining architecture details

Recommended order:

1. Implement the reusable control exporter/validator and import the complete donor contracts into a separate grouped campaign. Mark unfinished controls pending; do not alter their jobs.
2. Common grouped module and baseline-preservation/isolated-initialization tests.
3. Fixed angular grouping and physical-feature tests.
4. Random grouping control, including stable identifier handling without perturbing the inherited data stream.
5. Fully learned soft router and gradient/collapse diagnostics.
6. Hybrid router and zero-locality equivalence tests.
7. Separate extensible registry and joined reporting, inherited-recipe tests, donor-runtime GPU checks, sustained benchmarks, and storage projection.
8. Freeze the complete screen and submit only the new models in parallel when authorized; reuse validated control results at reporting time.

The seed, update budget, reporting intervals, LR schedule, optimizer, selection policy, runtime compatibility requirements, and default control reuse are **already settled by the reusable protocol**. Do not reopen them as grouped-model hyperparameters. Before launch, resolve the remaining architecture details: numerical floors/masks, clustering tie rules and exact library behavior, router initialization/update equations, and any metadata needed for stable random grouping. These choices must be frozen before comparative screening, not adjusted after seeing which model wins.

Implementation resolutions are now in [config.py](../part_group_screen/config.py), the model/grouping modules, and the [architecture notes](../part_group_screen/README.md#frozen-architecture-and-routing-details). Actual counts are 2,368,127 (fixed/random) and 2,518,271 (learned/hybrid), versus 2,143,486 for the reused ParT control. Local correctness/regression checks do not establish GPU throughput or a performance gain; remote compatibility and all four admission probes remain required.

No remote deletion, data movement, job cancellation, environment replacement, paper-results changes, or training submission is part of writing this plan.

## 15. Prior work and claim boundaries

The general concepts are established. The proposed experiment is a controlled application and comparison; a novelty claim needs a separate literature assessment.

- **Particle Transformer:** the original physics-aware particle-attention baseline. [Qu, Li, and Qian, 2022](https://arxiv.org/abs/2202.03772).
- **FastJet:** reference for clustering algorithms, recombination, and implementation conventions. [Cacciari, Salam, and Soyez](https://arxiv.org/abs/1111.6097).
- **DiffPool:** differentiable soft cluster assignments for hierarchical graph representations. [Ying et al., 2018](https://arxiv.org/abs/1806.08804).
- **Slot Attention:** iterative competitive assignment to learned slot representations. The router above is inspired by this family, not asserted to be an exact implementation. [Locatello et al., 2020](https://arxiv.org/abs/2006.15055).
- **Set Transformer:** attention-based set summaries and inducing-point mechanisms, relevant to generic latent-token controls. [Lee et al., 2019](https://arxiv.org/abs/1810.00825).
- **PHAT-JeT:** local patch attention plus pooled patch-level communication, developed for resource-constrained jet tagging; a close structural precedent, not evidence that our full-capacity branch must improve. [Wang et al., 2026](https://arxiv.org/abs/2605.21789).
- **PLuM:** particle and Lund-splitting representations combined in a transformer; directly relevant to the question of explicit hierarchical physics information complementing particle inputs. [Gouskos and Maier, 2026](https://arxiv.org/abs/2605.26821).
- **Prong-assisted jet classification:** combines particle and reconstructed-subjet information using cross-attention with a different classifier backbone. [Hammad and Nojiri, 2024](https://arxiv.org/abs/2404.14677).

The immediate deliverable is a trustworthy answer to whether grouping helps in our matched ParT setting, what kind of grouping helps, and what it costs. Improved accuracy, interpretable groups, computational efficiency, and methodological novelty are separate claims and require separate evidence.
