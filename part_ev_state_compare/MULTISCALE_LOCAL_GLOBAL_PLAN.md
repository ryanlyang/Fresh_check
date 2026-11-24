# Multiscale local–global processing on full LW State-EV

Recorded 2026-11-23. **Detailed implementation specification; not implemented,
benchmarked, or queue-ready.** This supersedes the short multiscale note only.
It changes no model code, frozen campaigns, jobs, data, environments or other
idea plans.

The intended future deliverable is one locked scientific model with **independent
SPORC tier3 and Tigris benchmark/production pipelines**. Either or both may run.
Names of packages, files, schemas and CLI modes below are proposed contracts,
not existing commands. No numerical classification gain is promised.

## 1. Objective, hypothesis and boundaries

Add explicit **particle, fine-region and coarse-region resolutions** to the
complete noncompact, standard-relations, all-eight Layerwise State-EV backbone.
Build a nested geometric hierarchy, repeatedly summarize current particle
representations through it, exchange local/global information, and feed that
interpretation back to particles.

The hypothesis is compositional organization, not new information access:
native attention already communicates globally. The model may find it easier
to distinguish evidence within a region from the arrangement of several regions.
Every native particle remains in the original stream; pooled regions never
become a compulsory information bottleneck.

Use `part_ev_state_compare.model.build_model('state_layerwise', ...)`:

- Width 128, eight particle blocks, eight attention heads.
- One shared standard width-64 pair stem with eight independent final bias heads.
- Ordinary EV and width-32 State-EV in all eight particle blocks.
- Original particle FFNs, native head/residual scaling, dropout and normalizations.
- Native trimmer, embedding, two class-attention blocks, final norm and classifier.
- Existing state query chunk 64 and shared per-forward pair normalization.
- **2,327,462 trainable parameters** in the pinned reference.

Historical 86.3151% validation is context, not automatically a matched control,
independent test performance or a promise for this model. A two-A100/SyncBatchNorm
result is not automatically comparable to one-GH200/native-BatchNorm training.

This first method includes no free workspace tokens, learned grouping router,
experts, masked training, selected relations, richer pair inputs, pair memory,
new auxiliary loss, pretrained donor, distillation or ensemble.

## 2. Locked first model

Scientific identifier: `nested_multiscale_state_layerwise_v1`.
Candidate run ID: `MULTISCALE_STATE_EV_LAYERWISE`.
Matched control: `LW_STATE_EV_CONTROL`.

| Choice | First implementation |
| --- | --- |
| Resolutions | All native particles; at most 16 fine regions; at most four coarse regions |
| Hierarchy | One geometry-only angular agglomeration history with nested cuts |
| Fusion stages | After blocks 2, 4, 6, 8; zero-based indices 1, 3, 5, 7 |
| Stage weights | Independent across all four stages |
| Stage features | Rebuilt from current particle hidden states, not carried latent memory |
| Region width / attention | 128 / eight heads, width 16 per head |
| Region FFNs | Fine and coarse: biased `128 -> 512 -> 128`, exact GELU |
| New norms | Affine channelwise LayerNorm, epsilon `1e-5` |
| Particle pooling | Member mean and channelwise max, plus seven physical descriptors |
| Local fine communication | Self-attention among siblings of the same coarse parent |
| Coarse communication | Attention-pool fine children, then global coarse self-attention |
| Top-down | Every active fine region can attend to all active coarse regions |
| Particle feedback | Own fine token + own coarse-parent token + own normalized hidden state |
| Final readout | Separate class queries into final fine/coarse sets, residual fusion |
| Initial identity | Four feedback output linears and final class-fusion output linear zero |
| Added dropout / loss | None / none |
| Checkpointing | Four pure learned stages, non-reentrant; geometry built once outside |
| Inference | Original ten logits; no extra classifier or ensemble |

Sixteen fine regions and four coarse regions are not truth prong labels.
Full width, stage-specific weights and repeated feedback are accuracy-oriented
starting choices, not measured optima. This model is substantially larger than
the base; section 11 makes that explicit.

**Clarification to the short note:** geometrically identical four-vectors are
kept in indivisible atomic leaf groups. Thus active counts are
`min(number_of_distinct_geometric_atoms,16)` and `min(atoms,4)`, not necessarily
`min(particle_count,16/4)` on exact-duplicate inputs. Normal distinct inputs retain
the requested counts. This resolves a real symmetry ambiguity without dropping
particles or using arbitrary input positions to split identical geometry.

No automatic change of scales, stages, tie policy, width, pooling, dropout or
loss is allowed to fit resources or pass a test. Revised science needs a new
configuration identity and admission.

## 3. Tensor and hierarchy contract

Let B be local batch, N the actual native post-trim length, D=128, F=16, C=4,
and A=8 attention heads.

| Tensor | Shape | Meaning |
| --- | --- | --- |
| Native hidden states | `[N,B,D]` | Original sequence-first layout |
| Stage particle view H | `[B,N,D]` | Transpose, not a second particle stream |
| Retained four-vectors V | `[B,N,4]` | `px,py,pz,E`, physical inputs only |
| Valid mask | `[B,N]` | True for an actual retained particle |
| Fine membership Af | `[B,N,F]` | One-hot for valid particles, zero for padding |
| Coarse membership Ac | `[B,N,C]` | Same contract |
| Fine-to-coarse P | `[B,F,C]` | One parent per active fine region |
| Fine / coarse masks | `[B,F]` / `[B,C]` | Active slots are compact canonical prefixes |
| Fine / coarse descriptors | `[B,F,7]` / `[B,C,7]` | Detached physical features |
| Fine / coarse features | `[B,F,D]` / `[B,C,D]` | Differentiable learned activations |
| Particle-region IDs | `[B,N]` | Integer IDs, -1 only at padding |

For each valid particle:

```text
sum_f Af[i,f] = 1
sum_c Ac[i,c] = 1
Af @ P = Ac
fine_id and coarse_id are consistent with the same nested history
```

Every active fine/coarse region has positive original-particle occupancy.
No particle is discarded, assigned twice, or routed outside its own jet.

For gather operations only, map a padding ID of -1 to safe active slot zero,
then zero its post-projection correction with the particle mask. Never clamp an
out-of-range ID on a valid particle, or let negative indexing select a region.

Memberships/physical features are recomputed once per forward from the actual
trimmed input, then reused by all four stages. Learned region features are
recomputed at each stage from that stage's H; do not carry F/C activations from
a previous stage as persistent workspace tokens.

## 4. Deterministic geometry and nested cuts

### Coordinates and distance

Build geometry under no_grad with autocast disabled, using FP32 and epsilon
`1e-6`. Sanitize masked entries before any coordinate operation. Valid inputs
and all derived finite quantities must be checked; negative valid-particle
energy is an input failure. Do not silently remove an invalid valid particle.

Follow the physical conventions of `part_group_screen/geometry.py`:

```text
pt = sqrt(max(px*px + py*py, 0)), with a finite zero branch
y = 0.5 * [log(max(E+pz, eps)) - log(max(E-pz, eps))]
phi = atan2(py,px) when pt^2 > eps^2; otherwise phi = 0
wrap(a) = atan2(sin(a),cos(a))
d(a,b) = (y_a-y_b)^2 + wrap(phi_a-phi_b)^2
```

Merge the nearest active pair using **E-scheme four-vector addition**, recompute
the merged cluster coordinates, and continue. This is an exclusive angular
agglomeration history, not a radius-cut or learned clustering. There is no
beam distance or constituent-removal rule.

### Canonical geometric atoms

1. Gather only valid post-trim geometry into a fixed-capacity canonical layout;
   normalize signed zero to positive zero for key equality.
2. Lexicographically sort physical FP32 tuples `(px,py,pz,E)`. Use stable tensor
   sorts and integer validity keys; never sort a CUDA Boolean tensor directly.
3. Coalesce exactly equal tuples into an atom with integer multiplicity.
   Its vector is the representative vector times multiplicity in FP32.
   Keep an original-particle-to-atom mapping; hidden states are not coalesced
   or discarded from the particle stream.
4. Use fixed-capacity masks/prefix scans/integer reductions, not a variable-size
   GPU result copied to CPU. Inactive atom slots are zero and excluded.

“Equal” here means equal canonical finite FP32 components, not nearby angles,
rounded bins or an arbitrary tolerance. Distinct particles with equal geometry
may have different features and hidden states; they share a geometric leaf but
all remain in learned pooling.

Why this is necessary: fixed-count hard partitions of indistinguishable
geometric leaves cannot generally be permutation-equivariant if a cut splits
those leaves. A stable original-index sort does not solve this. Grouping exact
duplicates first makes the intended geometry-only symmetry explicit.

### Cluster keys and exact ties

An active cluster's key is its smallest canonical atom rank. Disjoint clusters
therefore have distinct keys, even if their summed four-vectors happen to match.

Keep the surviving slot/key of the smaller-key cluster on every merge. Resolve
an **exact computed distance tie** by the lexicographically smallest ordered
pair of cluster keys. A masked flattened argmin over canonical upper-triangular
slot pairs implements that order. No particle-index, label, filename, hidden
feature, learned score, random seed or hash collision decides a tie.

Near ties are not exact ties. Do not quantize distances, add epsilon perturbations
or silently use float64 geometry on one platform. CPU/GPU library rounding can
affect marginal cases; same-device replay requires exact mappings, while
cross-device oracle fixtures need documented distance margins or engineered
exact ties. Do not claim general rotation/boost invariance from this tie policy.

### One batched history, two snapshots

Record the fine cut as soon as active atom clusters reach 16, or immediately if
there were at most 16 atoms. Continue the **same history** to four clusters,
or keep the initial cut if there were at most four atoms.

A full root above four regions is not needed; a partial dendrogram through
the coarse cut is sufficient. Do not perform two independent reclusterings.

Use at most `max(N-4,0)` batched merge iterations with per-jet do_merge masks.
Jets already at their coarse target perform safe no-ops. On a no-merge row,
an all-infinity candidate argmin must not update vectors, labels or snapshots.
Update only the merged cluster's distance row/column, since other vectors did
not change. No `.item()`, CPU NumPy tree, Python jet loop or device/host
synchronization per merge.

Compact each snapshot by surviving canonical key. Derive fine-parent IDs from
actual common particle membership; verify every fine region lies wholly inside
one coarse region. Store occupancies using original-particle multiplicities,
not atom count.

Angular clustering can still be expensive. A dense candidate minimum across
up to N squared entries per merge makes the straightforward implementation
O(B*N^3) selection work in the worst case, despite row-only distance updates.
“No host sync” does not mean negligible cost. Profile it once per forward.

### Conservation and repeatability

Snapshot region vectors are the actual E-scheme cluster vectors. Define the
jet reference vector and scalar constituent-pT sums using a documented canonical
atom-order FP32 reduction; a fixed padded pairwise reduction is preferred.
Do not let floating atomic scatter sums determine distance ties.

Compare sums to original retained four-vectors with stated FP32 conservation
tolerance, not impossible bitwise equality across different addition trees.
Occupancies and nested integer mappings are exact. Tree construction never
uses hidden states and never repeats in checkpointed backward.

## 5. Seven physical descriptors, unchanged meanings

Compute descriptors once for both cuts from original retained particles,
their multiplicities/memberships and the snapshot four-vectors. Do not use
coarse child count as coarse particle occupancy.

For group g, with its E-scheme vector, original-particle occupancy n_g, and
jet reference direction:

```text
d0 = log(max(pt_group / max(sum_i pt_i, eps), eps))
d1 = log1p(sqrt(max(E_group^2 - |p_group|^2, 0)))
d2 = y_group - y_jet
d3 = sin(wrap(phi_group - phi_jet))
d4 = cos(wrap(phi_group - phi_jet))
d5 = log1p(n_group)
width = sqrt(max(sum_member pt_i * d(i,group) / max(sum_member pt_i,eps), 0))
d6 = log1p(width)
```

Use the repository's native physical units, finite zero-pt/mass branches and
coordinate epsilon. Inactive descriptors/vectors are zero, never NaN/-infinity.
No truth information, learned four-vectors, fitted normalization or new dataset
features are introduced.

A coarse width uses its original particles, not just the four-vectors of fine
children. Reuse coordinate/distance helpers after parity tests, but do not
assume the old grouping helper's independent fixed-count assignment supplies
these two nested cuts.

## 6. Differentiable, memory-bounded hidden pooling

At each stage define `U = particle_norm(H)`, with FP32 LayerNorm arithmetic.
Use U, not raw H, for member mean/max and particle-read keys/values.

For active fine region f:

```text
mean_f = sum_member U_i / n_f
max_f[channel] = maximum_member U_i[channel]
fine_seed_input_f = concat(mean_f, max_f, physical_f)   # width 263
```

The mean preserves distributed evidence; max preserves salient channel
responses. Gradients flow to hidden states through both. Geometry is detached.

Use small one-hot memberships and batched FP32 matrix products for means.
Do not multiply invalid NaNs by zero; sanitize the hidden view first. Division
uses actual positive occupancy; inactive denominator can be one before masking.

### Max values and equal-max gradients

The mathematical gradient convention is the one from
[torch.amax](https://docs.pytorch.org/docs/2.15/generated/torch.amax.html):
split a channel's upstream gradient equally among all member values tied for
the maximum. Do not use a dimension-wise argmax that sends every tie to the
first particle.

The intended efficient backend is an out-of-place segmented `scatter_reduce`
amax with source/index shapes matched, negative-infinity initialization, invalid
source entries masked to negative infinity and inactive outputs then set to zero.
It must match an independent masked-amax loop in **both values and tie gradients**.

PyTorch documents [scatter-reduce constraints and possible CUDA nondeterminism](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.scatter_reduce_.html).
Do not infer target-device support, deterministic gradients or higher-order
derivatives from CPU success. Verify the actual installed runtime, mixed
precision, inactive slots and checkpointed backward. No many-writer plain
`scatter` overwrite is acceptable.

Never materialize a retained `[B,16,N,128]` expansion of all masked hidden values.
At B=512,N=128 that is 512 MiB in FP32 before gradients, per stage. Keep ordinary
pooling storage O(B*N*D + B*F*D), plus the small membership arrays.

A bounded region-tiled masked-amax implementation is the correctness oracle/
diagnostic fallback, not an unmeasured automatic production fallback. If the
segmented backend fails target-device parity/restart gates, report and fix or
review the engineering choice; do not silently change gradients, weaken ties
or accept a memory-exploding implementation.

## 7. Exact learned fusion stage

Each of four stages has independent parameters. All new attention primitives
use separate biased Q,K,V,output width-128 linears, eight heads of width 16 and
fixed score scale `1/4`. All new dropout is zero.

For active tokens, attention means masked softmax of `Q K^T / 4` followed by
a weighted value sum and one output linear. No native pair bias, relative
position table, selected relations, learned temperature or invented momenta
enter these new attention modules.

### Named norms: fourteen per stage

Each name below is a distinct affine width-128 LayerNorm:

1. particle_norm.
2. fine_seed_norm.
3. fine_read_query_norm.
4. fine_local_attn_norm.
5. fine_ffn_norm.
6. coarse_seed_norm.
7. coarse_read_query_norm.
8. coarse_read_kv_norm.
9. coarse_global_attn_norm.
10. coarse_ffn_norm.
11. topdown_query_norm.
12. topdown_kv_norm.
13. feedback_fine_norm.
14. feedback_coarse_norm.

Particle-normalized U is reused directly for particle-read K/V. Do not add an
unlisted second particle K/V norm or share norms/weights between stages.

### Bottom-up and top-down equations

```text
U = particle_norm(H)
F0 = mask_f(fine_seed_norm(GELU(fine_seed_linear([mean_f(U), max_f(U), desc_f]))))

F1 = mask_f(F0 + ParticleRead(fine_read_query_norm(F0), U,
                            allowed=particle_belongs_to_fine))
F2 = mask_f(F1 + FineLocal(fine_local_attn_norm(F1),
                         allowed=active_siblings_including_self))
F3 = mask_f(F2 + FineFFN(fine_ffn_norm(F2)))

child_mean_c = sum_child(n_f * F3_f) / n_c
C0 = mask_c(coarse_seed_norm(GELU(coarse_seed_linear([child_mean_c, desc_c]))))

C1 = mask_c(C0 + ChildRead(coarse_read_query_norm(C0), coarse_read_kv_norm(F3),
                          allowed=fine_child_of_this_coarse))
C2 = mask_c(C1 + CoarseGlobal(coarse_global_attn_norm(C1),
                            allowed=all_active_coarse))
C3 = mask_c(C2 + CoarseFFN(coarse_ffn_norm(C2)))

F4 = mask_f(F3 + TopDown(topdown_query_norm(F3), topdown_kv_norm(C3),
                       allowed=all_active_coarse))

fine_for_i = gather_own_fine(feedback_fine_norm(F4))
coarse_for_i = gather_own_parent(feedback_coarse_norm(C3))
delta_i = feedback_output(GELU(feedback_input([U_i, fine_for_i, coarse_for_i])))
Hnext = H + mask_particle(cast_like(delta, H))

return Hnext, F4, C3, tensor_health
```

`fine_seed_linear` is biased `263 -> 128`; `coarse_seed_linear` is biased
`135 -> 128`. FineFFN and CoarseFFN each use biased `128 -> 512 -> 128` with GELU
and only the listed pre-norm. The feedback MLP is biased `384 -> 128 -> 128`
with GELU and no hidden norm/dropout.

Child-mean weights are original-particle occupancies, not a uniform mean of
unequally sized fine regions. The attention-pool can then learn a different
allocation using contextual tokens and descriptors.

Fine-local attention includes self and only siblings; it is not global fine
attention. Coarse-global and top-down reads see all active coarse tokens.
Each particle receives its **own** fine token and coarse parent, not a broadcast
mean of all regions. The full particle residual stays intact.

No extra fine FFN after top-down, extra particle norm after feedback, auxiliary
group head or gamma multiplier is included. Zeroing the feedback output alone
preserves initial identity without two nested zero gates.

Only the fourth stage's final F4/C3 feed the final class read. Earlier stage
region activations influence prediction through particle feedback and later
native State-EV processing. All gradients remain attached.

## 8. Final fine/coarse classifier reads

After the unchanged two native class blocks and final norm, let c be the native
`[B,1,128]` class representation.

Use two independent cross-attentions, with four new affine norms total:

```text
m_f = FineClassRead(LN_class_query_f(c), LN_class_fine(F_final),
                   keys=active_fine)
m_c = CoarseClassRead(LN_class_query_c(c), LN_class_coarse(C_final),
                     keys=active_coarse)
class_delta = fusion_output(GELU(fusion_input(concat(m_f,m_c))))
c_fused = c + cast_like(class_delta, c)
logits = original_classifier(c_fused.squeeze(1))   # [B,10], native classifier width
```

The fusion MLP is biased `256 -> 128 -> 128`. Its final output weight/bias start
zero; both attention output linears start nonzero. There is no extra final norm,
classifier MLP, concatenation to the classifier's input width or class-FFN change.

Separate softmax reads make both scales available without one joint token
population mechanically weighting the scale with more tokens. They do not
guarantee equal useful contribution; the learned fusion chooses its use.

These tokens are derived region representations, not the persistent workspace
method. Direct readout and feedback contributions still require later ablations.

## 9. Safe attention, masking and numerical arithmetic

All active attention query rows must have at least one allowed active key:

- Fine particle reads have actual members.
- Fine sibling attention includes self.
- Every active coarse token has fine children.
- Coarse/global and top-down see at least one coarse token.
- Both class reads have at least one region.

An active all-masked row is an error, not an empty softmax to repair silently.
For inactive query rows only, replace the score row by finite zeros before
softmax, evaluate safely, then zero the **post-output-linear** result and
residual token. Exclude those dummy numerical rows from diagnostics. They have
no semantic attention or gradient contribution. Never evaluate all-minus-infinity
softmax and multiply NaNs by zero afterward.

Mask invalid source features before new norms/projections; keep biases from
reviving inactive tokens by masking after every listed residual. Padded particle
corrections are exactly zero. Do not change the native padded-query/pair-BN rules.

Keep new LayerNorm arithmetic, means/maxima, descriptors, attention scores,
softmax, weighted reductions and region residual accumulations FP32. Keep master
parameters FP32; new learned linears follow normal FP16 training autocast.
Cast only the final correction to the native H dtype before residual addition.
FP32 validation disables both outer and native AMP scopes.

Small CPU FP64 oracle/gradcheck components retain float64 instead of downcasting;
production geometry remains FP32. No clipping of attention scores or descriptor
changes to pass tests. Intentional masked -infinity is not a nonfinite failure,
but valid projections/scores/residuals must all be finite.

Collect tensor-valued health without per-region GPU `.item()` calls. No added
distributed collective or rank-dependent new exception inside model forward;
inherited native SyncBN retains its normal training collectives. A bad new
branch must make the driver reject all ranks coherently before backward, not
let one rank stop between peers' SyncBN calls.

## 10. Initialization and learning sequence

Build the complete reference first with seed 101. Construct additions on CPU
under isolated versioned RNG scopes derived from seed/stage/component names,
not Python's randomized hash. Preserve every native starting tensor and the
post-build RNG state.

Initialize all new norms to weight one/bias zero, epsilon `1e-5`.
Initialize all non-identity new projection/MLP weights independently with
Xavier uniform, gain one, and zero biases. There are no learned region seed
vectors or slot/position embeddings.

Exactly five final linears start zero:

- Feedback output weight/bias in each of four stages.
- Final class-fusion output weight/bias.

Their upstream inputs, seed projections, all five attentions per stage, region
FFNs, class-read projections and fusion input start nonzero. No extra zero
scale, tiny gamma, warm-up freeze, pretrained load or custom LR.

At initialization, classify exactly through the original path up to computed-
tensor tolerances. New geometry/processing consumes no randomness, so native
dropout/trimming RNG and call order remain unchanged.

On the first CE backward, the five output linears should receive finite signals
on a nondegenerate fixture. Earlier upstream parameters can correctly have zero
gradients then. After outputs move, fine/coarse pooling, projections, attention,
FFNs and input states should receive finite learning signals.

One-particle or one-child attention can legitimately have zero Q/K selection
gradients. Require graph-connected finite gradients, not impossible nonzero
gradients in every degenerate fixture.

Use an actual-Ranger activation probe with eight successful updates within
twenty attempts, crossing rectification and a Lookahead boundary. Include a
mixed fixture with multiple fine regions and multiple coarse parents. Verify
later gradients through each pathway, not only output biases. A failure is a
diagnostic, not permission to change loss, initialization or stage count.

## 11. Parameter and computational cost

Analytical inventory for this exact full-width design:

| Component | Parameters |
| --- | ---: |
| One biased Q/K/V/output attention | `4*(128*128+128) = 66,048` |
| Five attentions per stage | 330,240 |
| Two `128 -> 512 -> 128` FFNs per stage | 263,424 |
| Fine seed `263 -> 128` | 33,792 |
| Coarse seed `135 -> 128` | 17,408 |
| Particle feedback `384 -> 128 -> 128` | 65,792 |
| Fourteen affine norms per stage | 3,584 |
| **One stage** | **714,240** |
| Four independent stages | 2,856,960 |
| Two final class-read attentions | 132,096 |
| Four final read norms | 1,024 |
| Final fusion `256 -> 128 -> 128` | 49,408 |
| **Total addition** | **3,039,488** |
| Reference LW State-EV | 2,327,462 |
| **Planned candidate** | **5,366,950** |

The candidate has about **130.6% more parameters** than the base, roughly 2.31
times its stored parameter count. It is accuracy-oriented, not a tiny add-on.
A module-arithmetic check supports the inventory; this is not a built/benchmarked
full scientific model. Enumerate actual names/shapes/counts before admission.

There are twenty new stage attentions and two class reads; the native eight
particle and two class attentions are retained. New particle-read scores scale
as O(B*A*F*N), not N squared, but native State-EV remains quadratic and tree
selection can be cubic. No linear-cost or speedup claim is justified.

At B=512,N=128, one FP32 fine particle-read score matrix
`[512,8,16,128]` is 32 MiB. That is not full peak memory: probabilities, hidden
pooling, four stages, gradients, recomputation and the native model also exist.
Measure device/host peaks and whole-step runtime on both platforms.

Report FLOPs with a declared multiply-add convention, separately for inference
and training, including new linears/reductions and checkpoint recomputation.
Report geometry comparisons/reductions and walltime separately if the FLOP
counter misses them. Do not infer runtime from attention-score sizes alone.

## 12. Concrete code integration

Proposed new package after checking names are unused:

```text
part_ev_multiscale/
  __init__.py
  config.py         # locked science, count and runtime identities
  geometry.py       # canonical atoms, one nested history, cuts and descriptors
  pooling.py        # masked means/segmented maxima and independent interfaces
  attention.py      # explicit safe attention with allowed-query/key masks
  model.py          # pure fusion stage, final read, native-compatible assembly
  diagnostics.py    # bounded maps/occupancy/norm/attention diagnostics
  restart.py        # complete-state and geometry-aware replay evidence
  protocol.py       # data/source/environment identity, storage, admission
  benchmark.py      # paired target-device correctness and performance
  engine.py         # explicit task/platform train/evaluate/report
  submission.py     # frozen snapshots, Slurm transactions, reviewed dispatch
  README.md         # real commands only after implementation/testing

scripts/run_part_ev_multiscale.py
scripts/submit_part_ev_multiscale.py
sbatch/run_part_ev_multiscale.sh
sbatch/submit_part_ev_multiscale.sh
tests/test_part_ev_multiscale_*.py
```

Register additions under `mod.multiscale.*`, including stages 0–3 and the final
read/fusion. Preserve original base state_dict keys. Geometry has no trainable
parameters or dataset-derived persistent cache.

Use a local native-compatible outer forward following the isolated-instance
assembly precedent in `part_ev_pair_readout/model.py`; do not copy its special
pair-readout final-attention code. Call all eight original particle Block.forward
methods and both original class Block.forward methods unchanged.

The forward order is:

1. Original trimmer once; obtain actual vectors/mask from its outputs.
2. New detached geometry/maps/descriptors once, from that retained input.
3. Original embedding and pair encoder once.
4. Original particle blocks with pure stages after 2,4,6,8.
5. Original class blocks and norm; final separate fine/coarse reads and fusion.
6. Original classifier and tensor-health rejection.
7. `finally` cleanup of native pair bindings/caches; all new activations local.

Keep the original outer API `(points,features,vectors,mask)`. Do not append
regions to native tensors/class keys, invent region momenta for native pair
bias, rerun trimming or native MHA, or replace original particle FFNs.

Reuse audited coordinate conventions and tests from `part_group_screen`,
not its hook-based live activation cache or separate single-cut assignment
as a drop-in hierarchy. Its fixed arm canonically sorts all raw features;
this method deliberately uses geometry-only atoms and a new identity.
Old grouping results are related comparisons, not equivalence proofs.

Other anchors: `part_ev_state_compare/model.py`,
`part_layerwise_screen/model.py`, `part_ev_state_speed/model.py`,
`part_ev_screen/precision.py`, `part_ev_fullscale/optimizer.py` and
`checkpoint.py`, `part_ev_state_production/engine.py`, and
`part_ev_rts_tigris/production.py`. Existing engines have hard-coded contracts;
use explicit platform adapters, not mutated protocol/task globals.

## 13. Autograd, memory lifetime and invariance

Pass H and immutable geometry tensors/masks explicitly into each stage.
Return Hnext, fine/coarse activations and health explicitly. No hook/module
attribute carries live hidden states, final regions or mutable stage counters.

Checkpoint each entire pure learned stage, non-reentrant, with
`preserve_rng_state=False` because there is no new random operation.
Geometry and native attention/BN/dropout stay outside these checkpoints.
Do not rebuild the tree, mutate maps, query cleared pair caches or reuse detached
region states during recomputation. Final class reads remain uncheckpointed.

No persistent full attention histories or per-event activation diagnostics.
Do not retain previous forwards' trees/regions or save them in state_dict.
A backward after forward cleanup must work.

Every module stays graph-connected even with inactive region slots, N=1 or a
single coarse parent. No rank may skip a stage based on occupancy. Use
`find_unused_parameters=False`; do not compensate for lost paths with a DDP flag.

The new branch is per jet. The native network retains training-time BN/SyncBN
batch dependence and its pair-padding conventions. Test branch isolation on
fixed hidden states and whole-model batch isolation in eval; do not claim
stronger training/raw-padding invariance than the base actually has.

Geometry mappings should be exactly particle-permutation equivariant on
same-device fixtures, including exact duplicates/ties. Hidden reductions and
outputs obey numerical tolerances, not bitwise equality under different floating
summation orders. Compare pairwise co-membership/mapped physical content rather
than arbitrary original particle index labels.

Hard memberships are not differentiable with respect to geometry. Hidden
mean/max, learned processing and feedback remain differentiable. Do not claim
IRC safety, full Lorentz equivariance, exact rotational invariance or robustness
to hard-boundary perturbations without independent evidence.

## 14. Full data, optimization and selection contract

Use original full `ranger_full_v1`, not the shortened 200k schedule:

| Field | Value |
| --- | --- |
| Training / validation pools | Official 100M / 5M JetClass jets |
| Independent test | Unopened during prepare, benchmark and routine production |
| Features / cap | Original 17 / 128 particles |
| Seed / initialization | Fresh 101, no donor or benchmark weights |
| Global batch | 512 |
| Training budget | 1,000,000 attempted updates, nominal 512M exposures |
| Validation | 50 checks, every 20,000 attempts, 1,280,000 events each |
| RAdam | LR 1e-3, betas (0.95,0.999), epsilon 1e-5, weight decay zero |
| Lookahead | Complete state, k=6, alpha=0.5 |
| Schedule | Original zero-based decay milestones 35–49, no new warm-up |
| Loss / regularization | Original ten-class CE/dropout, no new auxiliary or clipping |
| Training | Native/new linear FP16 autocast, FP32 parameters and stable reductions |
| Validation | FP32, highest accuracy, first tie wins |
| Early stopping | None |

Use pinned `part_ev_fullscale.protocol.learning_rate(step,recipe)`.
For zero-based step s, reporting epoch is `s//20000`. The original multiplier is
`(0.01 ** (1/15)) ** count(milestone <= epoch)`, milestones 35–49.
First decay is attempt 700,001. Attempt/LR cursors advance on scaler skips;
Lookahead advances only with actual optimizer steps. Preserve scaler defaults.

All new parameters use the same stable-order Ranger registration, with no
freezing, special LR, geometry loss, token weight decay or gradient multiplier.
Keep train and separate rank-zero validation streams continuous across checks.

The selected checkpoint is evaluated on full 5M validation: accuracy, macro
OVO/OVR AUC and current per-class QCD rejection points with acceptance counts
and explicit zero-background handling. Selection overlap remains; do not call
this independent test performance or select another checkpoint by full-5M AUC.

## 15. Independent execution on SPORC and Tigris

Each platform has independent preparation, paired benchmark, runtime identity,
source snapshot, output directory, storage reservation and Slurm ledger.
Neither waits for the other cluster or imports its environment.

| Setting | SPORC | Tigris |
| --- | --- | --- |
| Architecture / partition | Linux x86_64 / tier3 | Linux aarch64 / tigris |
| Default account | reu-aisocial | reu-aisocial |
| Training GPU | Two matching same-node A100s | One GH200, at least 80 GiB device memory |
| Global/local batch | 512 / 256 per rank | 512 / 512 |
| Native normalization | SyncBatchNorm | Native BatchNorm |
| Distribution | DDP, gradient_as_bucket_view | Ordinary one GPU |
| Loader workers | One per rank | Two |
| Torch threads | Two per rank | Four |
| Train/benchmark CPUs and host RAM | 8 / 128 GiB | 12 / 128 GiB |
| In-training validation | Unwrapped rank-zero eval, batch 512 | Ordinary eval, batch 512 |
| Final evaluation | One A100, keep saved SyncBN in eval | One GH200 |
| Interpreter | `/home/ryreu/miniconda3/envs/part_ev_screen_sporc/bin/python` | `/home/ryreu/miniforge3-aarch64/envs/part_ev_fullscale/bin/python` |

Verify actual GPU identity/memory, driver, pinned torch/CUDA/Weaver, TF32 disabled,
cuDNN benchmark False and approved NCCL settings on each platform. No package
installation or existing environment edits. No deleted selected-env dependency,
compiled tree backend or new CPU-to-GPU preprocessing service.

Both share scientific configuration, not identical batch partitioning, stream
order, BN or numerics. Do not migrate a checkpoint across world size/BN as if
it were an ordinary resume.

Default production is **candidate and fresh matched LW State-EV control on each
platform**: four full trainings if both platforms are authorized. Make that
cost visible. Control reuse is optional only after explicit authenticated
matching of science/exposure, data/source, seed, hardware/BN/workers/precision
and selection/evaluation identity.

Benchmark control weights are disposable, never production weights. Historical
86.31% is not automatically reused. Compare within platform; same seed 101 on
two different platforms is not two independent random seeds.

Both may run. No automatic first-start-winner lock, cross-cluster job dependency,
cancellation or requirement that the other control finish first.

## 16. Correctness tests and numerical acceptance

Independent tests must include:

1. **Tree oracle:** tiny readable CPU agglomeration with canonical atoms and
   pair-key ties versus batched implementation. N=1,2,4,16,17,128, non-prefix
   masks, exact duplicates, identical angles/different momenta, equal cluster
   vectors, symmetric ties and azimuth wrap.
2. **Nested cuts:** occupancies, active counts, one-parent membership,
   `Af @ P == Ac`, all retained particles exactly once, no padding membership.
3. **Geometry invariance:** particle permutations/duplicates, signed zero,
   no original-index tie dependence, finite zero-pt/zero-mass branches.
   Four-vector conservation within FP32 tolerances and exact multiplicities.
4. **Physical descriptors:** seven-feature independent formulas at both cuts;
   coarse original-particle counts/width, not fine-child approximations.
5. **Pooling:** member means/maxima and input gradients versus independent
   loops, mixed signs, all-equal max ties, inactive slots, valid NaN/Inf rejection.
6. **Attention:** explicit per-head FP64 oracle, output bias once, member/sibling/
   child/global masks, inactive safe rows, active empty-key failure and gradients.
7. **Full stage:** independently assembled active forward/gradients, top-down/
   feedback paths, all fourteen norms and independent stage weights.
8. **Initial native identity:** starting base tensors/buffers, trimmed inputs,
   masks, native RNG/trimmer and call counts exact; logits/gradients within
   limits. Include active nonzero original State-EV weights.
9. **Native integration:** one trim/embedding/pair build, eight native particle
   attentions/FFNs and two class attentions; four tree-free learned stages,
   twenty stage attentions plus two final reads. No repeated native MHA.
10. **Lifetime/recompute:** active checkpointed/uncheckpointed gradients,
    backward after forward cleanup, exceptions, repeated forwards, no retained
    live trees/regions/graphs and no geometry work in backward.
11. **Degeneracy/DDP:** N=1, all duplicate geometry, one coarse region, uneven
    lengths/occupancies between ranks, finite graph-connected zero gradients
    where attention selection is mathematically trivial.
12. **Learning/state:** real-Ranger activation and complete optimizer/scaler/
    RNG/trimmer/cursor roundtrip; strict missing-key/count/config rejection.
13. **Precision/memory:** expected operation dtypes, no hidden AMP in FP32 eval,
    no retained B*F*N*D expansion or per-merge host synchronization.

Proposed frozen parity thresholds:

| Comparison | rtol | atol |
| --- | ---: | ---: |
| Small FP64 learned/pooling oracle outputs/gradients | 1e-9 | 1e-10 |
| FP32 active outputs / initial logits | 1e-5 | 1e-6 |
| FP32 learned/input gradients | 1e-4 | 1e-5 |
| FP32 vector conservation / descriptors | 1e-5 | 1e-5 |
| FP32 global-DDP outputs | 5e-4 | 2e-5 |
| FP32 global-DDP gradients | 2e-3 | 2e-5 |
| FP16 checkpointed/uncheckpointed outputs | 2e-3 | 2e-3 |
| FP16 active gradients | 1e-2 | 1e-3 |

Mappings, counts, masks, initial tensors, native RNG and call counts are exact.
Different slot labels are compared through canonical mappings/physical content,
not used to excuse inconsistent nesting.

On CPU/GPU geometry comparisons use positive-margin fixtures or exact ties
engineered from equal computed coordinates. Report near-distance margins rather
than claiming arbitrary CPU/GPU transcendental bitwise agreement.
Descriptor comparisons use the specified FP32 arithmetic, not an FP64 mass
calculation substituted near a zero-mass cancellation. Test zero/near-zero mass
and denominator clamps separately for finite prescribed behavior; use
well-conditioned fixtures for the stated descriptor parity thresholds.

Failures require investigation, not loss changes, `strict=False`, weaker ties,
different geometry dtype, omitted coarse scale or tolerance inflation.

## 17. Actual-GPU, DDP and geometry-aware restart evidence

Require CUDA FP32 active full-model forward/input/parameter-gradient checks,
FP16 checkpointed/uncheckpointed parity, target-device segmented-max tie
gradients and real learning on **each** platform.

SPORC must use two actual A100s with SyncBN, unequal native lengths and region
counts, single-atom jets on one rank, finite connected gradients and replica
parameter equality. Tigris requires actual one-GH200/native-BN evidence.
CPU success cannot stand in for either.

For the DDP/global-batch numerical oracle, fix already-trimmed equal-length
inputs and disable original trimming/dropout only for that test. Compare against
an isolated ordinary-BN single-device global-batch reference, not rank-zero
training-mode SyncBN while a peer waits. Restore actual dropout/trimming for
learning, replay, restart and timing. Unequal native padding/BN populations
are separate health tests, not identical-global-reference promises.

### Complete-state restart

Populate nonzero feedback/class outputs and Ranger state before probing.
Immediate save/load compares exactly: model/buffers, RAdam options/moments/
counters, Lookahead slow weights/counter, scaler, rank RNG/trimmer, task/runtime,
step/validation and input-stream cursors.

Run seven rolling-batch updates for reference, uninterrupted repeat, disk-start
repeat, and save/resume after update three. Cross diagnostic validation and
Lookahead boundaries using a separate probe schema, not fake full-budget
production completion.

Subsequent floating model/Ranger tensors use a versioned bounded policy:

```text
state_abs_error <= min(1e-4, 1e-5 + 1e-4 * abs(reference))
loss_abs_error  <= min(2e-4, 1e-4 + 1e-5 * abs(reference))
```

But **canonical atom IDs, fine/coarse mappings, occupancy, parent masks,
input fingerprints and RNG/LR/cursor/skip/scale schedules stay exact** in
same-device trials. Record per-update mapping digests, merge-margin summaries,
losses, maximum state errors and worst paths for every rank. Settled fixtures
must have no AMP skips. No automatic threshold expansion from observed noise.

Also run a disposable bounded actual-ROOT replay probe: authenticate successive
train/validation batches, reconstruct/replay consumed batches without forward
or optimizer work, restore rank RNG and verify continuation/mappings.
Report synthetic-state and real-stream evidence separately. A short successful
probe does not establish bitwise-identical multi-day trajectories.

Coherent nonfinite, malformed checkpoint, stop and rank-local I/O failures must
terminate all ranks without a persistent collective hang.

## 18. Paired benchmark, profiling and mechanism diagnostics

One independent allocation per platform measures **fresh LW State-EV and fresh
multiscale**, sequentially, control first, with matched initial rank batches,
data fingerprints and recorded cold-start/order effects. No accuracy claim
from the disposable benchmark weights.

For each arm measure:

- 20 warm-up and 4,200 real-data attempts, windows 1,000/1,000/1,000/1,000/200.
- Fixed all-128 training, global batch 512: 10 warm-up and 40 measured attempts.
- 128,000 end-to-end FP32 validation events, batch 512.
- Explicit full-128 FP32 validation stress at batch 512.
- One-GPU FP32 inference batch 256: 10 warm-up and 20 timed batches;
  batch-one latency: 10 warm-up and 100 timed forwards.
- Full restart save/load and selected-weight write timings.
- Geometry plus learned-stage whole-step cost, AMP skips/scale, finite model/
  optimizer/scaler and exact active-mask/mapping checks.
- Per-GPU allocated/reserved peaks, per-process RSS and available whole-job/
  cgroup host-memory peak.

Include an adversarial all-duplicate-geometry case and a many-exact-distance-tie
distinct-atom case, not just random geometry. Verify finite model/backward,
correct counts and target memory on full-length inputs.

Training uses actual DDP/world size; rank-zero validation/inference uses
unwrapped eval, not DDP with idle peers. Shut down unused ROOT workers before
later synthetic stress/profiling without changing measured stream settings.

Admission requires no missing path, finite settled training, at least 10%
allocated-device-memory headroom on every GPU, and measured-window AMP skips
below 5%. Record reserved peaks too. Throughput denominator includes attempts,
including skipped optimizer updates.

Use a five-hour internal benchmark deadline, 5h30m Slurm request and early
signal. Incomplete evidence is failure/inconclusive, never a pass because
summary exited zero. No silent shortening/batch reduction/single-scale fallback.

### Bounded profiling

On at most two jets/32 particles, one train and one eval profile, aggregate:

- Canonicalization/duplicate atoms.
- Merge-distance selection/row updates and cut/map construction.
- Physical descriptors.
- Hidden mean/max pooling.
- Fine reads/local refinement, coarse pooling/global processing, top-down,
  feedback, final fine/coarse reads and fusion.
- Stage recomputation separately.

Require positive call counts, finite nonnegative CPU/device timings and at
least one positive timing per required scope. CPU-only or device-only attribution
is valid. Inclusive scopes cannot simply be summed. No unbounded profiler trace
over full batches/windows; small profiles are not full-batch speed evidence.

### Bounded scientific diagnostics

Log aggregate fine/coarse occupancy, duplicate multiplicity, number of children,
physical widths/pT shares, merge margins and tiny-region fractions.
On isolated small no-grad eval samples, inspect region RMS/cosine similarity,
attention entropy, correction/native norm ratios and final-scale read norms.

Disable feedback, top-down or final reads separately for bounded logit-difference
diagnostics, restoring weights/modes/RNG/caches/gradients afterward.
These interventions are not trained ablation results or proof of physical prongs.

Save no per-training-jet trees, hidden region features, attention movies or
large raw event dumps. Repeated grouping overhead must be visible in throughput.

## 19. Measured resource and walltime policy

| Phase | Initial request | SPORC | Tigris |
| --- | --- | --- | --- |
| Prepare | 2 h | CPU, 8 CPUs / 64 GiB | CPU, 8 CPUs / 64 GiB |
| Paired benchmark | 5 h 30 min | 2 A100s, 8 CPUs / 128 GiB | 1 GH200, 12 CPUs / 128 GiB |
| Summary | 15 min | CPU, 2 CPUs / 4 GiB | CPU, 2 CPUs / 4 GiB |
| Optional dispatch | 30 min | CPU, 2 CPUs / 4 GiB | CPU, 2 CPUs / 4 GiB |
| Train per arm | Measured | 2 A100s, 8 CPUs / 128 GiB | 1 GH200, 12 CPUs / 128 GiB |
| Evaluate per arm | Measured | 1 A100, 8 CPUs / 128 GiB | 1 GH200, 12 CPUs / 128 GiB |
| Report | 1 h | CPU, 2 CPUs / 8 GiB | CPU, 2 CPUs / 8 GiB |

These are planned requests, not statements of live availability. Check partition,
account/QoS, device memory/type, host RAM, maintenance and Slurm test-only requests
on each actual submission host. No silent GPU/platform substitution.

For each arm independently let s be slowest measured rank/window real-data
seconds/attempt, v the complete 128k validation seconds, c full-restart write
seconds and w selected-weight write seconds:

```text
T_compute = 1,000,000 * s
T_validation = 50 * 10 * v
T_writes = (1,000 + 100 + 2) * c + 51 * w
train_hours = ceil(1.30 * (T_compute + T_validation + T_writes
                         + measured_cold_start) / 3600)
```

Write allowance includes periodic/pre/post-validation restarts, improving
selected candidates and final best copy. Record formula/measurements.
Evaluation includes full-5M inference, checkpoint loading, prediction writes
and CPU merge/metrics; do not extrapolate GPU kernels alone.

Planned ceilings: 480 training hours and 24 evaluation hours, subject to stricter
live limits. If measured recommendation exceeds a limit, **do not submit**;
request a new execution/design decision. Do not cap down, silently drop stages,
change scales, shrink batch or split into three-day chains.

One long allocation per full training is intended. Graceful interruption-safe
resume remains required. The 30% margin is not a bound on replay, future
filesystem stalls or queue delays. Report allocated GPU-hours as walltime times
GPU count, separately from FLOPs and jet throughput.

## 20. Data/source admission, snapshots and storage

CPU preparation authenticates train/validation content/inventory receipts, entry
counts, class order, preprocessing YAML and pinned reference/data/optimizer
sources. No independent test access, ROOT copies, fitted new preprocessing,
disk trees or feature caches.

Freeze committed relevant source via safe git archive/transitive inventory;
reject symlinks/path traversal/live checkout imports. Disable user-site and
bytecode writes. New pushes/pulls cannot change old frozen workers.

Fingerprint distributions from stable interpreter installation roots, not lazy
setuptools-vendor paths. Emit actionable bounded mismatches; do not bypass real
torch/Weaver/source changes. Science identity covers every geometry, tie, scale,
norm, dtype, initialization, pooling and count choice. Runtime separately covers
platform/world size/BN, workers, precision, hardware and execution environment.

Proposed independent directories:

```text
checkpoints/part_ev_multiscale/
  sporc_<fresh-id>/
    source/ source_manifest.json launch.json campaign.json data_contents.json
    logs/ probes/ benchmark.json summary.json submitted_job_ids.tsv
    production/
      production.json production_job_ids.tsv
      runs/LW_STATE_EV_CONTROL/seed_101/
      runs/MULTISCALE_STATE_EV_LAYERWISE/seed_101/
      reports/
  tigris_<different-fresh-id>/
    ... separate equivalents ...
```

No mutable checkpoint, prediction chunk, mapping cache, ledger or source path
is shared between platform roots. Job IDs include the recorded cluster.

For P parameters, prospective production bytes per arm:

```text
weights = 51 * (8*P + 1 MiB)
restarts = 3 * (24*P + 16 MiB)
predictions = 5,000,000 * 48 + 64 MiB
misc = 1 GiB
arm_budget = weights + restarts + predictions + misc
```

Add 2 GiB per paired-benchmark platform and one shared 4-GiB emergency reserve.
With the planned base/candidate counts and two fresh arms on both platforms,
initial aggregate headroom is approximately **20.41 GiB before other outstanding
campaign budgets**. This is not available quota or reserved capacity.
Subtract already committed outputs appropriately; keep atomic-write allowance.

Retain latest complete restart, safe temporary allowance, at most 50 improving
selected weights plus final best, bounded logs/diagnostics and resumable
predictions. No restart retained for every update and no per-event tree archive.

Use shared-filesystem locks/registry across both platforms, including existing
registered campaigns and other method budgets. Reserve before submitting;
recheck before prepare/dispatch/heavy output. A local squeue absence cannot
release the other cluster's reservation. No automatic cleanup of environments,
datasets, old checkpoints or unrelated campaigns.

## 21. Launcher and automatic-production contract

Proposed launcher takes `--platform sporc|tigris`. Planned modes:

- `plan`: read-only provenance/resource/paired-task/storage/automation summary.
- `benchmark`: freeze new campaign, prepare -> paired benchmark -> summary.
- `review`: authenticate/recompute evidence/recommendations, no writes.
- `pipeline --reviewed`: benchmark chain plus explicitly authorized bounded
  automatic-production dispatch.
- `dispatch --reviewed`: separately submit from passed target-device evidence.
- `status`: read-only jobs/check histories/progress/reports.

Publish actual queue commands only after these interfaces exist and fake-Slurm
tests pass. A benchmark/summary completion alone does not authorize production.

Summary runs after benchmark termination to explain failures. Dispatch waits
for successful summary and independently rechecks model/geometry/pooling/
restart, data/source/runtime and resource/storage evidence. No inherited
“passed” field from another model/GPU, and no production on inconclusive evidence.

After admission, control and candidate train independently, each followed by an
afterok evaluation; paired report waits for both. No arbitrary concurrency cap.
Optional matching control reuse is explicit/authenticated and reported.

Protect canonical scoped paths, committed source, immutable snapshot imports,
live partition/space checks, shared registry and per-output submission locks.
Write durable flushed intent before each sbatch, record cluster/job/request
immediately afterward, and stop for reconciliation after ambiguous replies,
partial chains or duplicates. Never blindly resubmit a possibly live job.

Sanitize inherited SBATCH/SLURM/rank/PYTHONPATH and unapproved NCCL options at
submission; preserve scheduler-assigned GPU visibility within workers.
Use no-user-site/no-bytecode. Torchrun is for the two-A100 worker only, with no
elastic auto-restarts; Tigris uses ordinary world_size=1.

## 22. Production and evaluation transactions

Implement explicit task/platform functions, not global protocol mutation.
Fresh training never loads benchmark or historical weights.

Save atomically every 1,000 attempts, before/after validation and on a coordinated
early stop:

- Science/source/data/runtime/task identities and parameter inventory.
- All native and multiscale model parameters/buffers.
- Full RAdam/Lookahead options/moments/slow weights/counters.
- AMP scaler/cumulative skips and per-rank Python/NumPy/torch/CUDA RNG/trimmer.
- Attempt/check counters, selected accuracy/check/path/hash, continuous train/
  validation consumption and timing.
- Bounded aggregate geometry diagnostics schema/counters, not live graphs/maps.

Evolving region activations and input-specific trees are not restart state.
On resume rebuild loaders, replay consumed batches without forward/optimizer
work, restore rank RNG and regenerate geometry from the next input. Replay
consumes allocation time and is interruptible while preserving the prior complete
restart. Wrong platform/BN/world size, geometry/config/source/data or missing
multiscale keys reject resume.

Write a new improving selected checkpoint before transactional metadata commit;
do not destroy the previous best on an interrupted validation. Completion
requires 1M attempts **and** 50 completed checks, with the selected checkpoint
audited against accuracy-based records.

A graceful incomplete stop saves recoverable state and exits nonzero, blocking
afterok successors. No fake completion, automatic extra allocation, automatic
three-day chain or cross-cluster continuation. Resume is separately reviewed.

Log JSON heartbeats every 100 attempts, losses/skips/throughput, validation
boundaries/results, checkpoint/replay events and bounded occupancy/geometry
summaries. Use `check_XX.json`. Distinguish reaching the next training boundary
from a completed validation result; do not invent live validation percentages.

Final 5M evaluation writes bounded resumable predictions over all 50 validation
ROOT files. Authenticate checkpoint/campaign/data/chunk hashes and sidecars
before reuse, preserving class order and score/operating-point definitions.
One-GPU evaluation of SPORC state keeps its saved SyncBN modules in eval.

Reports include per-arm parameters, selected check, accuracy/AUC/rejections with
counts, timing/FLOPs convention, geometry overhead, device/host peaks and allocated
GPU-hours. Label single seed, selection overlap, platform/control-reuse differences
and missing outputs. Compare within platform, not pooled pseudo-seeds or historical
point estimates labeled as controlled gains.

## 23. Scientific controls, follow-up and prior art

Do not automatically queue these extras with the first design.

1. **Single scale:** fine-only regions, removing coarse construction/processing.
   Report changed capacity/compute; it is not a clean parameter-matched contrast.
2. **No sibling restriction:** global fine attention with the rest fixed;
   isolates the locality mask with unchanged parameters.
3. **No top-down:** remove coarse-to-fine attention, reporting the capacity change.
4. **Feedback-only / readout-only:** distinguish particle contextualization from
   the extra final summary route.
5. **One-shot versus repeated:** one post-block-four stage, with reduced parameters
   and compute explicitly reported.
6. **Shuffled nested memberships:** match both scale occupancies and parent nesting;
   do not independently shuffle the two cuts. Recompute physical descriptors from
   assigned particles and clearly declare the non-geometric control.
7. **Dense capacity control on LW State-EV:** widen all eight native particle FFNs
   to 1,979. Analytical count **5,367,086**, only 136 above the candidate. Verify
   actual count and disclose the awkward hardware width/changed active compute.
   Never silently substitute shared-bias ordinary ParT for this control.
8. **Compute/exposure control:** measured FLOPs-based extra baseline exposure
   with a declared stretched schedule, separately from equal-update comparisons.
9. **Additional matched seeds / independent test:** establish reproducibility
   and lock the final comparison before opening test data.

Interpret region collapse, tiny groups and boundary sensitivity alongside
accuracy, not as validation-based reasons to modify an existing run.
The hypothesis can fail through bad hard grouping, redundant global information,
gradient starvation, runtime overhead or simple added-capacity explanations.

[PHAT-JeT](https://arxiv.org/abs/2605.21789) already combines local geometric
processing with hierarchical patch attention. The
[earlier grouping branch](../part_group_screen/README.md) is also a necessary
internal comparison. This proposal retains the full dense State-EV backbone
and adds nested, repeated bottom-up/top-down exchanges; it does not inherit an
efficiency or novelty claim from those differences.

Hierarchy/global context is not new. Derived regions are not guaranteed physical
subjets, IRC-safe or Lorentz-equivariant. A contribution needs controlled gains
and a clear performance/cost/mechanism story; separate literature review remains
necessary before claiming novelty.

## 24. Implementation order and definition of done

1. Lock scientific/runtime schema, duplicate/tie semantics, parameter inventory,
   immutable map API and independent tiny tree/descriptor/pooling oracles.
2. Implement canonical geometry, two snapshots and nested maps; pass permutation/
   duplicate/tie/wrap/conservation tests before learned modules.
3. Implement safe attention and segmented max with correct tie gradients,
   inactive rows, dtype contracts and bounded memory.
4. Implement exact pure stage and final reads; prove active forward/gradients
   and all parameter/norm counts.
5. Integrate the native-compatible outer forward; prove identity, unchanged
   native calls/RNG, explicit map lifetime and checkpointed backward.
6. Prove actual-Ranger activation and full-state/mapping-aware restart, plus
   disposable real-stream replay and coherent worker failures.
7. Build independent platform preparation, immutable source/environment/data
   admission, cross-campaign storage reservation and read-only planning.
8. Implement actual-device paired benchmarks with full geometry cost, precision,
   SyncBN/DDP, degenerate input, max-tie gradients, memory, restart and I/O gates.
9. Fake-Slurm tests: wrong platform/interpreter, dirty source, path spaces,
   insufficient storage, forged/partial evidence, invalid dependencies, duplicate/
   ambiguous submissions and safe frozen imports after checkout changes.
10. **Benchmark-ready:** both locally tested interfaces/documented queue commands;
    local CPU success alone grants no target-GPU or production admission.
11. **Production-ready:** atomic complete-state engines, transactional selection,
    resumable evaluation/reporting and reviewed auto-dispatch implemented/tested.
12. **Admitted:** the actual platform's benchmark passes all gates and requested
    limits before authorized full paired trainings are submitted.

Handoff lists tests run/skipped, planned versus actual counts, untested GPU
assumptions, geometry/pooling/restart evidence, measured resource recommendations,
queue commands and which modes submit jobs. This document itself changes only
a plan and authorizes no job, environment modification or deletion.
