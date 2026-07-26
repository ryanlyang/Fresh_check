# Relational Particle Transformer Attention-Bias Campaign

Status: implementation-ready design for a new HLT-only architecture branch.

This campaign is independent of the residual-field, oracle-bridge, and
knowledge-distillation programs.  It does not use offline particles, oracle
fields, teacher logits, or privileged targets during training or inference.
Its purpose is direct and narrow:

> Train Particle Transformer variants from random initialization and determine
> whether richer, physically motivated particle-pair relations can beat the
> matched HLT Particle Transformer baseline.

The primary campaign uses one million training jets, 250,000 development
validation jets, and a sealed 500,000-jet final test.  It screens all controls,
all six single relation families, and all predeclared high-value combinations.
Every control and every single relation family, plus the strongest
combinations and predeclared architectural finalists, is confirmed across
three training seeds before the final test is opened.

---

## 1. Scientific motivation

The reference Particle Transformer (ParT) does not use only angular distance.
Its standard interaction bias is built from four pairwise kinematic quantities:

- logarithmic angular separation;
- logarithmic relative transverse momentum, \(k_T\);
- logarithmic momentum sharing, \(z\);
- logarithmic pair invariant mass squared.

The ordinary particle tokens also contain transverse momentum, energy, charge,
particle-identification indicators, transverse and longitudinal impact
parameters, and their uncertainties.  However, most of those quantities enter
only as individual token features.  They are not exposed to attention as
explicit particle-pair relations.

This leaves several plausible shortcuts for a better HLT tagger:

1. The standard \(k_T\) and \(z\) relations are symmetric, while the question
   "how much should context particle \(j\) influence query particle \(i\)?" is
   directional.
2. Heavy-flavor tagging should benefit from an explicit approximation to
   displaced-track or common-vertex compatibility.
3. PID and charge combinations are relational.  Same-type and mixed-type
   pairs can carry different information.
4. Local density and constituent rank may help attention distinguish hard
   prongs from diffuse radiation.
5. HLT-defined region or subjet membership may provide a useful hierarchy
   without requiring offline information.

The campaign therefore retains the full ParT token and classifier architecture
but extends the pairwise interaction representation.

Reference basis:

- Particle Transformer paper: <https://arxiv.org/abs/2202.03772>
- Official implementation: <https://github.com/jet-universe/particle_transformer>
- ATLAS GN2 overview, supporting learned track-origin and pair-compatibility
  reasoning: <https://cds.cern.ch/record/2933089?ln=en>
- PHAT-JeT, supporting local/hierarchical jet structure under resource
  constraints: <https://arxiv.org/abs/2605.21789>
- Particle-Lund Multimodality, supporting explicit clustering-hierarchy
  representations: <https://arxiv.org/abs/2605.26821>

These references motivate the inductive biases; none establishes that the
proposed HLT relations will improve this campaign.  That remains the
experiment.

---

## 2. Locked scientific questions

### Q1: do richer pair relations improve HLT tagging?

Does any from-scratch relational ParT variant improve 10-class held-out
accuracy and cross entropy relative to an exact from-scratch HLT ParT trained
on the same jets?

### Q2: which relation family provides the gain?

Can the improvement be attributed to directional transverse momentum, track
compatibility, PID, charge, local density, or HLT-defined regions?

### Q3: are useful relation families complementary?

Do predeclared pairs and higher-order combinations improve beyond the best
single relation family?

### Q4: is a gain merely additional capacity?

Does the result survive comparison to:

- a widened standard-four-feature pair encoder with matched parameter count;
- a full-shape model whose new relation inputs are identically zero?

### Q5: does the result survive training variation?

Does the advantage persist across seeds `101`, `202`, and `303`, rather than
arising from one favorable initialization?

---

## 3. Non-goals

This first campaign will not:

- use an offline or oracle teacher;
- distill logits;
- predict residual fields;
- warm-start from an existing HLT ParT checkpoint;
- fine-tune a frozen ParT;
- reconstruct missing particles;
- tune variants using final-test results;
- claim trigger readiness from accuracy alone;
- replace the particle set with a graph or region-only classifier;
- infer a shared physical decay vertex from information not present in
  JetClass.

Impact parameters are treated as approximate track compatibility information,
not as a complete track-helix or secondary-vertex reconstruction.

---

## 4. Locked data and split contract

The campaign uses balanced 10-class JetClass identities and a fixed HLT view.

| Logical role | Existing manifest split | Count | Purpose |
|---|---|---:|---|
| `train` | `model_train` | 1,000,000 | All model-weight training |
| `val_stop` | `model_val` | 125,000 | Checkpoint selection and early stopping |
| unused | `stack_train` | 0 | Reserved by the five-split manifest contract |
| `val_select` | `stack_val` | 125,000 | Architecture screening and confirmation |
| `final_test` | `final_test` | 500,000 | Sealed final evaluation |

Every nonzero split is exactly class-balanced.  Jet identities are sampled
without replacement and are disjoint across splits.

The first campaign uses the same fixed-HLT profile as the trusted HLT
architecture-comparison domain:

```text
hlt_profile = fixed_hlt_v1
hlt_degradation_strength = 0.6
max_constituents = 128
```

The bootstrap must serialize the complete HLT generator parameters, generator
seed, source-manifest hash, per-split identity hash, and per-split HLT content
hash.  A different HLT profile is a different campaign and must not be merged
into the same report.

The locked HLT seeds are the repository defaults:

```text
model_train = 1053
model_val = 1054
stack_val = 1056
final_test = 1057
```

`stack_train` is empty and is not cached.

Only HLT caches are persisted.  Offline arrays are read transiently while
building the fixed-HLT view and are not retained in the campaign root.

### Two-stage input audit

Before cache construction, normalization, or tree work, the cheap
preconstruction audit:

- inspects branch names, shapes, and dtypes for every source file without
  bulk-loading its event arrays;
- validates all manifest identities and class counts;
- fully loads the 1,000 lowest hashes per class from `model_train` and the 100
  lowest hashes per class from each other nonempty split, using
  `sha256("rpt_raw_audit_v1" || canonical_jet_identity)`;
- checks finite-value policy, the locked sentinel schema, PID/charge domains,
  constituent counts, and four-vector reconstructability on that sample.

It writes `preconstruction_raw_input_audit.json` and must pass before expensive
derived work begins.  After HLT caches, train-only normalizers, compiled
backend, and angular-tree split manifests exist, the full postconstruction
audit validates complete identity/hash coverage, cache/tree/runtime parity,
normalizer parents, storage, and HLT-only inference.  It writes
`postconstruction_input_audit.json`; screening depends on this second audit.

### Final-test seal

Training workers may load only `model_train` and `model_val`.

Screening and confirmation workers may additionally load `stack_val`.

There are two distinct final-test permissions:

1. A sealed data-preparation worker may read raw `final_test` identities and
   inputs before model training solely to construct and hash the fixed-HLT
   cache and compact angular-tree sidecar.  It may not load any checkpoint,
   run model inference, fit a normalizer, calculate a label-dependent metric,
   expose event-level summaries to selection workers, or write outside the
   sealed input subtree.
2. Only a final-evaluation worker may consume the prepared `final_test`
   tensors scientifically.  That worker must require:

- a locked confirmation artifact;
- a matching campaign-spec hash;
- a matching split-manifest hash;
- matching HLT-cache hashes;
- a checkpoint selected without final-test information.

All training, screening, confirmation, selection, normalization, diagnostic,
and reporting CLIs reject final-test split handles.  The pre-lock preparation
job records only provenance, counts, storage measurements, and pass/fail input
audits.  This permits operationally efficient cache/tree preparation without
opening final-test model-selection information.

---

## 5. Exact baseline architecture

`RPT_BASE` is the ordinary base-size HLT Particle Transformer:

```text
particle input dimension = 17
particle embedding dimensions = [128, 512, 128]
pair embedding dimensions = [64, 64, 64]
attention heads = 8
particle-attention blocks = 8
class-attention blocks = 2
standard pair feature count = 4
activation = GELU
```

The standard four pair features must be generated by the same Weaver
`pairwise_lv_fts` helper used by the reference model.  Their order, clipping,
and logarithmic conventions must not be reimplemented from memory.

The local wrapper will pass the generated pair tensor explicitly through
Weaver's `uu` interface.  Before any scientific run, an equivalence test must
show that:

```text
ParticleTransformer(x, v, mask)
```

and:

```text
ParticleTransformer(x, v, mask, uu=pairwise_lv_fts(v, 4))
```

produce equal logits, gradients, and padding behavior within the locked
floating-point tolerance when initialized with the same state dictionary.

All campaign models, including the baseline, use this explicit-pair code path.
This prevents a wrapper or preprocessing difference from being mistaken for a
relation-family gain.

All models are initialized randomly and trained from scratch.  New relations
are available in the first particle-attention block from the first optimizer
step.  There are no zero-initialized feature gates and no inherited baseline
weights in the primary campaign.

---

## 6. Relational attention interface

For particle-attention layer \(\ell\), head \(h\), query particle \(i\), and
context particle \(j\), attention has the ordinary form:

\[
A_{ij}^{(\ell,h)}
=
\frac{q_i^{(\ell,h)\top}k_j^{(\ell,h)}}{\sqrt{d_h}}
+
B_h(u_{ij}).
\]

The pair representation is:

\[
u_{ij}
=
\left[
u_{ij}^{\mathrm{base4}},
e_{ij}^{f_1},
\ldots,
e_{ij}^{f_m}
\right],
\]

where each enabled relation family \(f\) owns an optimized encoder producing a
small normalized representation \(e_{ij}^f\).  The concatenated tensor is passed
to the real ParT pair-embedding network, which outputs one additive bias per
attention head.

### Why use family encoders?

The relation families have different numerical structures:

- transverse-momentum and track relations are continuous and directional;
- PID and charge contain categorical pair states;
- density contains multiscale per-particle summaries;
- region relations contain categorical tree membership, merge history, and
  cluster descriptors.

Forcing every raw quantity directly into one unnormalized tensor would make
some families unnecessarily difficult to learn.  Each family therefore
normalizes its raw inputs and maps them to a compact representation before
the shared ParT pair encoder.

### Train-only featurewise normalization

Raw per-pair `LayerNorm` is forbidden.  It would mix validity flags,
uncertainties, ranks, angular quantities, and displacement significances
inside one pair and could erase precisely the absolute magnitude that some
families are intended to expose.

Instead:

- binary indicators remain exactly `0` or `1`;
- categorical values remain embedding indices;
- PT ranks/raw \(p_T\) fractions and any other channel explicitly registered
  as `fixed_scale` remain in their declared interval;
- every other continuous channel receives its own train-fitted robust center
  and scale.

For continuous feature \(x_c\):

\[
\hat x_c
=
\operatorname{clip}
\left[
\frac{x_c-\operatorname{median}(x_c)}
{\max(\operatorname{IQR}(x_c)/1.349,10^{-6})},
-8,8
\right].
\]

Statistics are fitted using `model_train` only.  To make pair-statistic fitting
deterministic and bounded:

1. select the 50,000 training jets with the lowest salted jet-identity hashes;
2. for pair-applicable channels, select at most 64 applicable directed pairs
   per selected jet by the lowest salted pair hashes;
3. for node-applicable channels, count each applicable particle exactly once,
   never once per incident pair;
4. calculate exact medians and quartiles over each channel's applicable
   bounded sample;
5. record sample identity hashes, counts, applicability rules, quantiles,
   zero fractions, clipping rates, and the scaler content hash.

Every continuous channel declares one immutable fit-applicability mask:

- `PT` pair quantities: every valid directed particle pair;
- `TRACK` node quantities: valid tracks only, counted once per track;
- `TRACK` compatibility quantities: valid-valid directed track pairs only;
- `DENSITY` descriptors: valid particles only, counted once per particle;
- `REGION` node descriptors: valid particles at a defined resolution, counted
  once per particle and resolution;
- `REGION` pair quantities: valid directed pairs at a defined resolution;
- `REGION` merge quantities: valid distinct directed pairs only;
- binary indicators and categorical states: never normalized.

Invalid, padded, or inapplicable values are excluded from fitting.  At runtime
they are transformed to zero only after continuous normalization and are then
masked again after the learned encoder.  Consequently, placeholder zeros
cannot dominate a median or collapse an IQR.  For every continuous channel,
`relation_normalization.json` records:

```text
feature_name
applicability_rule_id
applicable_count
median
q25
q75
robust_scale
applicable_zero_fraction
post_normalization_clip_fraction
```

Normalization is applied at the earliest unique semantic level.  TRACK node
features are normalized before the Siamese endpoint encoder.  DENSITY node
descriptors are normalized once per particle before constructing
`[d_i,d_j,d_j-d_i]`.  REGION cluster/particle descriptors are normalized once
per particle and resolution, while distinct-pair merge quantities use their
own pair-applicable scalers.  Learned embeddings and already bounded
binary/categorical or explicitly `fixed_scale` channels are not fitted again
after concatenation.

Per-track uncertainty floors are fitted separately over all valid positive
training-track errors as described in the TRACK section.

The scaler artifact is immutable:

```text
inputs/relation_normalization.json
```

Every training, validation, and test worker verifies its hash.  Validation or
test values never influence these statistics.

### Common family encoder

After physical transformations and featurewise normalization, continuous
family encoders use:

```text
Linear(raw_dimension, 32)
GELU
RMSNorm(32)
Linear(32, encoded_dimension)
```

Categorical features remain learned embeddings and are combined as specified
by their families.  No normalization is performed across the raw channels of
one pair.

### Common numerical policy

The campaign-wide numerical constant is:

```text
epsilon = 1e-6
```

Every equation written with \(\epsilon\) uses this exact value unless that
equation declares a different named floor explicitly.  It is serialized in
the campaign specification and model metadata.

Every family must:

- operate only on HLT tokens and the HLT valid-particle mask;
- zero invalid pairs after every learned encoder;
- use wrapped \(\Delta\phi\);
- replace NaN and infinity before learned layers;
- apply only each channel's declared physical transformation, followed by
  its registered robust normalization and clip;
- preserve query/context direction;
- be permutation equivariant over valid particles;
- produce no persistent \(N\times N\) cache;
- support mixed-precision execution without computing logarithms below
  `epsilon`;
- expose its raw feature names and encoded dimension in model metadata.

Dropout is zero inside pair preprocessing for the first campaign;
regularization remains in the common training policy.

---

## 7. Relation family `PT`: directional transverse momentum

The standard ParT \(k_T\) and \(z\) quantities describe pair hardness and
momentum sharing but are symmetric.  `PT` explicitly describes which particle
is the query and which particle supplies context.

For each valid particle:

\[
x_i=\log\frac{p_{T,i}+\epsilon}
{\sum_k p_{T,k}+\epsilon},
\]

where the denominator is the masked scalar sum of constituent \(p_T\).

Define normalized descending rank:

\[
r_i =
\begin{cases}
\mathrm{average\_rank}_{\downarrow}(p_{T,i})/(N_{\mathrm{valid}}-1),
& N_{\mathrm{valid}}>1,\\
0, & \text{otherwise}.
\end{cases}
\]

Exact ties receive their average rank.  Rank construction must therefore be
permutation equivariant rather than dependent on a stable-sort row index.

Also retain the bounded raw scalar-\(p_T\) fraction:

\[
f_i=\frac{p_{T,i}}{\sum_kp_{T,k}+\epsilon}.
\]

The raw directed pair vector is:

\[
\begin{aligned}
u_{ij}^{PT} = [&f_i,\ f_j,\ x_i,\ x_j,\ x_j-x_i,\\
&\log((p_{T,i}+p_{T,j})/\sum_kp_{T,k}+\epsilon),\\
&(p_{T,j}-p_{T,i})/(p_{T,j}+p_{T,i}+\epsilon),\\
&r_i,\ r_j,\ r_j-r_i].
\end{aligned}
\]

The context particle is always \(j\).  This lets different heads learn to
prefer hard context, soft context, similarly ranked particles, or asymmetric
hard-soft relationships.  No sign is hard-coded; a head may learn a positive,
negative, or negligible dependence on context hardness.

```text
raw dimension = 10
encoded dimension = 8
```

Required diagnostics:

- mean bias versus context-particle \(p_T\) rank;
- directional swap difference
  \(B(i,j)-B(j,i)\);
- head-wise bias for leading, subleading, and soft context particles.

---

## 8. Relation family `TRACK`: displacement and track compatibility

This family approximates whether two charged HLT particles exhibit compatible
prompt or displaced-track patterns.  It is a learned pair-compatibility
relation, not a claimed secondary-vertex reconstruction.

All displacement calculations begin from the raw HLT \(d_0\) and \(d_z\);
they are not passed through `tanh` before division by their uncertainties.
The uncertainty floors are fitted once from `model_train`:

\[
\sigma^{floor}_{d_0}
=
\max\left(Q_{0.01}(\sigma_{d_0}>0),10^{-6}\right)
\]

and analogously for \(d_z\).  Quantiles are computed only over valid tracks,
and the fit artifact records the count and the \(0.01,0.05,0.50,0.95,0.99\)
quantiles.  Define

\[
\sigma^{eff}_{d_0,i}
=
\sqrt{\sigma_{d_0,i}^2+(\sigma^{floor}_{d_0})^2},
\qquad
s_{0,i}
=
\operatorname{asinh}\left(
\frac{d_{0,i}}{\sigma^{eff}_{d_0,i}}
\right),
\]

with the corresponding definitions for \(d_z\) and \(s_z\).

`track_valid` is true only when all of these conditions hold:

- the canonical PID is charged hadron, electron, or muon;
- raw \(d_0,d_z,\sigma_{d_0},\sigma_{d_z}\) are finite;
- both uncertainties are strictly positive;
- none of those values equals a configured missing-value sentinel.

The sentinel policy is not inferred from observed distributions.  It comes
from the locked raw-input schema `jetclass_hlt_raw_v1`, which enumerates the
allowed missing encoding separately for \(d_0,d_z,\sigma_{d_0},\sigma_{d_z}\)
or explicitly declares `no_numeric_sentinel`.  Preflight audits observed
values against that policy and fails on an unexpected encoding.  A
concentration of legitimate zero displacement values must never be
heuristically reclassified as missingness.

### Shared per-track encoder

For each particle construct

\[
t_i=[
d_{0,i},d_{z,i},
\log\sigma^{eff}_{d_0,i},\log\sigma^{eff}_{d_z,i},
s_{0,i},s_{z,i},track\_valid_i
].
\]

Invalid continuous entries are zeroed after normalization, while the validity
bit is retained.  The same Siamese encoder is applied to query and context:

```text
Linear(7, 32) -> GELU -> RMSNorm(32) -> Linear(32, 16)
```

producing \(g_i\) and \(g_j\).

### Explicit pair compatibility

For a directed pair, append a 17-channel explicit vector \(c_{ij}\):

- a four-state one-hot validity relation:
  invalid-invalid, valid-invalid, invalid-valid, valid-valid;
- \(\log(1+\chi^2_{ij})\), where
  \[
  \chi^2_{ij}=
  \frac{(d_{0,i}-d_{0,j})^2}
       {(\sigma^{eff}_{d_0,i})^2+(\sigma^{eff}_{d_0,j})^2}
  +
  \frac{(d_{z,i}-d_{z,j})^2}
       {(\sigma^{eff}_{d_z,i})^2+(\sigma^{eff}_{d_z,j})^2};
  \]
- \(\exp[-\tfrac12\min(\chi^2_{ij},25)]\);
- minimum and maximum \(|s_0|\), and minimum and maximum \(|s_z|\);
- \(s_{0,i}s_{0,j}\) and \(s_{z,i}s_{z,j}\);
- signed context-minus-query normalized differences:
  \[
  \delta d_{0,ij}
  =
  \frac{d_{0,j}-d_{0,i}}
  {\sqrt{(\sigma^{eff}_{d_0,i})^2+
         (\sigma^{eff}_{d_0,j})^2}},
  \qquad
  \delta d_{z,ij}
  =
  \frac{d_{z,j}-d_{z,i}}
  {\sqrt{(\sigma^{eff}_{d_z,i})^2+
         (\sigma^{eff}_{d_z,j})^2}};
  \]
- \(\sin\Delta\phi_{ij}\), \(\cos\Delta\phi_{ij}\), and
  \(\log(\Delta R_{ij}+\epsilon)\).

All compatibility values other than the validity state are zero unless both
tracks are valid.  The final directed track input is

\[
[g_i,g_j,g_j-g_i,g_i\odot g_j,c_{ij}],
\]

and is encoded by:

```text
Linear(81, 48) -> GELU -> RMSNorm(48) -> Linear(48, 12)
```

```text
per-track raw dimension = 7
explicit pair dimension = 17
pair-encoder input dimension = 81
encoded dimension = 12
```

Required diagnostics:

- fraction of all four directed validity states;
- fitted uncertainty floors and quantile/count audit;
- distributions of raw displacement, transformed significance, and
  compatibility \(\chi^2\);
- bias versus minimum absolute displacement significance;
- bias for prompt-prompt, prompt-displaced, and displaced-displaced pairs;
- per-class gains for `Hbb`, `Hcc`, `Tbqq`, and `Tbl`.

---

## 9. Relation family `PID`: directional particle-type pairs

The five canonical HLT PID flags are:

```text
charged_hadron
neutral_hadron
photon
electron
muon
```

PID flags are interpreted with a locked threshold of `0.5`.  Values must be
finite and within `1e-6` of zero or one.  Exactly one flag at or above the
threshold selects that category; no selected flag maps to an explicit sixth
`unknown` category.  Multiple selected flags are an invalid multi-hot state
and fail production preflight.  Every directed pair maps:

```text
(pid_i, pid_j) -> integer in [0, 35]
```

The primary representation is factorized to share strength with rare
electron and muon pairs:

```text
query_embedding = Embedding(6, 8)
context_embedding = Embedding(6, 8)
pair_embedding = Embedding(36, 8)
encoded = tanh((query_embedding + context_embedding + pair_embedding) / sqrt(3))
```

The table is directional.  It does not assume that same-PID pairs are always
helpful.  Photon-to-hadron and hadron-to-photon can receive different learned
representations if attention benefits from that distinction.

```text
categorical states = 36
encoded dimension = 8
```

Required diagnostics:

- learned pair-state embedding norms;
- average attention bias for every populated PID pair;
- same-PID versus mixed-PID bias;
- population counts for all six categories and all 36 ordered pairs;
- explicit rare-state counts for electron and muon query/context pairs;
- zero-hot and multi-hot audit counts, with production multi-hot count
  required to be zero.

---

## 10. Relation family `CHARGE`: directional charge structure

Charge is quantized to `-1`, `0`, or `+1` after validating that the HLT cache
respects the expected tolerance.  A directional charge-pair index has nine
states and is represented by:

```text
Embedding(9, 4)
```

The categorical embedding is concatenated with:

- raw clipped \(q_i\) and \(q_j\);
- \(q_iq_j\);
- \(|q_i-q_j|/2\);
- both neutral;
- exactly one charged;
- same nonzero sign;
- opposite nonzero sign.

The concatenated representation passes through the common family encoder.

```text
raw continuous dimension = 8
categorical embedding dimension = 4
family-encoder input dimension = 12
encoded dimension = 6
```

This family tests relational charge patterns, not the hypothesis that a
larger charge magnitude should receive greater attention.

Required diagnostics:

- charge-state population table;
- head-wise bias for opposite-sign, same-sign, charged-neutral, and
  neutral-neutral pairs;
- gain after conditioning on PID.

---

## 11. Relation family `DENSITY`: multiscale local activity and composition

`DENSITY` describes the local environment around the query and context
particles.  It combines exact non-overlapping annuli with smooth log-radius
kernels so a hard radius boundary is not the sole representation.

The annular boundaries are:

```text
[0.00, 0.05]
(0.05, 0.10]
(0.10, 0.20]
(0.20, 0.40]
```

For each particle and annulus, compute from valid HLT constituents:

- `log1p(neighbor_count) / log1p(max_constituents)`;
- scalar annular \(p_T\) sum divided by the jet scalar \(p_T\) sum.

The query particle is excluded from both the neighbor count and the annular
\(p_T\) sum at every radius.

Also compute four Gaussian kernels in \(\log(\Delta R+\epsilon)\), with
physical radius centers

```text
R_a = 0.025, 0.071, 0.141, 0.283
epsilon = campaign global
sigma_logR = 0.45
```

For valid \(j\ne i\):

\[
K_a(i,j)
=
\exp\left[
-\frac{
\left(\log(\Delta R_{ij}+\epsilon)-\log R_a\right)^2
}{
2\sigma_{\log R}^2
}
\right].
\]

The two responses at every center are exactly:

\[
\rho^{count}_{i,a}
=
\frac{\sum_{j\ne i}K_a(i,j)}
{\max(N_{valid}-1,1)}
\]

and

\[
\rho^{p_T}_{i,a}
=
\frac{\sum_{j\ne i}K_a(i,j)p_{T,j}}
{\sum_{k\in valid}p_{T,k}+\epsilon}.
\]

Both are zero for \(N_{valid}\leq1\).  There is no hidden radial cutoff in
these smooth channels; the log-Gaussian weight supplies the falloff.

Within \(R\leq0.20\), excluding self, compute four local composition
fractions:

- charged-particle scalar-\(p_T\) fraction, where charged includes charged
  hadrons, electrons, and muons;
- neutral-hadron scalar-\(p_T\) fraction;
- photon scalar-\(p_T\) fraction;
- displaced-track scalar-\(p_T\) fraction, using a valid track with
  \(|d_0/\sigma^{eff}_{d_0}|\geq2\) or
  \(|d_z/\sigma^{eff}_{d_z}|\geq2\).

The denominator for the first three mutually exclusive PID fractions is the
local neighbor scalar-\(p_T\) sum.  The displaced fraction is an overlapping
track diagnostic with the same denominator.  All four fractions are exactly
zero when that denominator is zero.

Finally append:

- exact valid-neighbor fraction within \(R\leq0.40\):
  \(N_{neighbor}/\max(N_{valid}-1,1)\), defined as zero for
  \(N_{valid}\leq1\);
- the particle's self share of its \(R\leq0.20\) activity,
  \[
  selfshare_i=
  \begin{cases}
  \dfrac{p_{T,i}}
  {p_{T,i}+\sum_{j\ne i,\Delta R_{ij}\leq0.20}p_{T,j}},
  & \text{if the denominator}>0,\\
  0,&\text{otherwise}.
  \end{cases}
  \]

The resulting per-particle descriptor \(d_i\) has 22 channels: eight hard
annular, eight smooth-kernel, four local-composition, and two occupancy/share
channels.

For pair \((i,j)\), use:

\[
u_{ij}^{DENSITY}=[d_i,d_j,d_j-d_i].
\]

```text
per-particle dimension = 22
raw pair dimension = 66
encoded dimension = 12
```

Required diagnostics:

- annulus occupancy distributions;
- bias versus local activity fraction;
- performance binned by jet multiplicity;
- performance binned by leading-particle \(p_T\) fraction.

---

## 12. Relation family `REGION`: beam-free HLT angular-tree relations

`REGION` introduces a genuine HLT-only clustering hierarchy.  Its canonical
contract is a beam-free exclusive angular tree—not ordinary hadron-collider
FastJet C/A at radius `0.8`.

Starting from valid HLT four-vectors, repeatedly merge the pair with minimum

\[
d_{ij}=\Delta R_{ij}^{2}
\]

using E-scheme four-vector recombination until exactly one root remains.
There is no beam distance or beam removal.  `R_ref = 0.8` is retained only as
a dimensionless normalization/reference jet radius in relation features; it
does not decide whether a merge occurs.  Consequently every valid pair has a
lowest common ancestor even when its two leaves are separated by more than
`0.8`.  Cutting the single tree yields exclusive resolutions `K=2`, `K=4`,
and `K=8`.  No offline constituent, offline subjet, class label, or truth
object is used.

### Canonical ordering and tie contract

Before tree construction, valid leaves are ordered lexicographically by the
canonical byte representation of their complete raw HLT physics tuple:
four-vector followed by the locked raw token fields.  Signed zero and NaN
representations are canonicalized according to the input schema before the
comparison.  Equal merge distances are resolved by the ordered pair of the
two clusters' sorted leaf-key multisets, never by the input row index.
Leaves with identical complete tuples are physically interchangeable, and
permutation tests compare outputs after undoing that interchange.

The production backend is a repository-owned compiled extension with contract
ID `relational_ca_tree_v1`.  It implements the distance, canonicalization,
tie-breaking, and E-scheme rules above directly.  Its source hash, compiler
identity, build flags, and binary hash are recorded.  Preflight fails if this
backend is absent or stale and never substitutes ordinary FastJet C/A or the
old top-\(p_T\) anchor heuristic.  FastJet with a sufficiently large audited
radius may be used only as a test oracle on fixtures where it is proven
equivalent to the beam-free contract.

### Compiled-backend build and ABI contract

The backend is a CPU PyTorch `CppExtension`, compiled once per campaign in a
dedicated dependency job and then loaded read-only by cache workers.  No
worker performs opportunistic JIT compilation.  The production build command
is:

```bash
python -u scripts/build_relational_part_tree_backend.py \
  --contract relational_ca_tree_v1 \
  --build-dir "${CAMPAIGN_ROOT}/backend/build" \
  --output-dir "${CAMPAIGN_ROOT}/backend"
```

The compiler must support C++17 and OpenMP.  Locked numerical flags are:

```text
-O3
-fno-fast-math
-fno-associative-math
-ffp-contract=off
```

`-Ofast`, `-ffast-math`, unsafe reciprocal approximations, and contraction
that changes the declared arithmetic order are prohibited.  Input
four-vectors, \(\Delta R^2\), E-scheme recombination, tie comparisons,
transverse momentum, and mass reconstruction are computed in IEEE float64.
Tree indices are integer.  Persisted continuous node values are cast once to
IEEE float32 after the float64 tree topology and features are finalized; the
metadata records this storage cast and parity tolerances.

The extension exports `backend_manifest()` and `self_test()`.  Every loader
verifies:

```text
contract_id and schema_version
extension source SHA-256
compiled binary SHA-256
compiler identity and major version
complete compiler flags
platform architecture
Python major.minor
PyTorch version
PyTorch _GLIBCXX_USE_CXX11_ABI value
OpenMP availability
locked self-test vector/output hash
```

An ABI or self-test mismatch fails before reading a production shard.

The exact builder uses an all-pairs priority queue with stale-candidate
validation and canonical tie keys.  For \(N\leq128\), its declared worst-case
complexity is \(O(N^2\log N)\) time and \(O(N^2)\) temporary memory per jet.
Parallelism is across jets only; a jet's merge order contains no
nondeterministic parallel reduction.

### Compact tree resource

For every split, persist a compact \(O(N)\) beam-free angular-tree sidecar, not an
\(O(N^2)\) pair tensor:

```text
inputs/relation_tree_cache/<split>_exclusive_ca_v1/
  manifest.json
  shards/
    shard_00000.npz
    shard_00000.metadata.json
    ...
```

Jets remain in split-manifest identity order and are partitioned into
deterministic contiguous shards of at most 10,000 jets.  Each array task writes
to a job-unique temporary file, validates event identities and content, then
atomically renames both data and metadata to their final names.  On restart, a
shard is reused only if its contract, parent, identity, count, binary, schema,
and content hashes all match.  A partial or stale shard is never treated as
complete.  The split `manifest.json` is written atomically only after every
expected shard passes validation.

The sidecar stores leaf-to-node identity, parent/child links, node depth,
node four-vector, transverse momentum, mass, multiplicity, merge
\(\Delta R\), merge \(k_T\), merge momentum sharing \(z\), merge mass, and
exclusive leaf assignments at `K=2`, `K=4`, and `K=8`.

For requested resolution \(K\), the actual cluster count is
\(C_K=\min(K,N_{valid})\).  When \(N_{valid}<K\), all \(N_{valid}\) leaves are
singleton clusters; the resolution remains valid and its actual cluster
count is recorded.  Therefore there are no ambiguous masked-resolution
assignments and no extra resolution-validity bits are needed.
For the all-invalid safety case \(N_{valid}=0\), no REGION pair is valid and
the complete relation tensor is zero under the ordinary particle mask.

Metadata binds the resource to the exact HLT cache, ordered event identities,
tree schema, compiled-backend identity, canonicalization contract, and
clustering configuration.

### Mandatory throughput projection

Before submitting the full shard arrays, run the exact production binary on a
deterministic 20,000-jet `model_train` sample stratified by
\(N_{valid}\).  The ten locked strata are:

```text
[0, 8]
[9, 16]
[17, 24]
[25, 32]
[33, 40]
[41, 48]
[49, 64]
[65, 80]
[81, 96]
[97, 128]
```

The initial quota is 2,000 jets per stratum.  Within each stratum, select the
lowest unsigned SHA-256 values of:

```text
sha256("rpt_tree_probe_v1" || canonical_jet_identity)
```

If a stratum contains fewer than 2,000 jets, take all of it.  Let \(D\) be
the total deficit and \(R_s\) the unselected population remaining in every
nonempty stratum.  Redistribute \(D\) proportionally to \(R_s\) using the
largest-remainder method: allocate each floor first, then assign remaining
slots by descending fractional remainder and finally lower stratum index.
Select each stratum's additional lowest unused salted hashes.  Preflight fails
if `model_train` contains fewer than 20,000 total jets.

The 1,000 Python-reference parity jets are the lowest hashes within the final
probe under the independent salt `"rpt_tree_probe_parity_v1"`.  Throughput
and storage projections are reweighted by the audited full-`model_train`
population fraction of each stratum; they are not reported as though the
quota-balanced probe were the natural multiplicity distribution.  The probe
artifact records stratum populations, initial/final quotas, redistribution,
ordered identity hash, and both salts.

The probe records:

- projected jets per second overall and measured jets per second by locked
  multiplicity stratum;
- p50, p95, and maximum milliseconds per jet;
- peak resident memory;
- persisted bytes per jet;
- exact topology/feature parity against the Python reference on 1,000 probe
  jets;
- projected shard wall time, aggregate CPU node-hours, and campaign storage
  for all 1.75 million jets.

Default operational limits are:

```text
maximum projected shard wall time = 2 hours
maximum projected aggregate tree-build CPU node-hours = 48
tree storage must fit the Section 20 budget and free-space reserve
```

Exceeding a limit blocks bulk-cache submission unless the user supplies an
explicit recorded operational override.  The probe may reduce shard size or
increase bounded array concurrency without changing scientific identity.
It may not change numerical flags, topology, feature precision, or the tree
contract.

At training and inference, a batched tensor implementation obtains lowest
common ancestors by binary lifting from the compact tree.  The deployed
checkpoint may require this deterministic HLT-derived tree transform, but it
requires no persisted training sidecar, oracle object, or offline input:
the same transform can be run directly on live HLT constituents.

For every E-scheme node \(r\), reconstruct

\[
p_{T,r}=\sqrt{p_{x,r}^2+p_{y,r}^2},
\qquad
m_r=\sqrt{\max(E_r^2-p_{x,r}^2-p_{y,r}^2-p_{z,r}^2,0)}.
\]

The one-root E-scheme node is the REGION jet:
\(p_{T,jet}=p_{T,root}\) is the vector-summed jet transverse momentum, not the
constituent scalar-\(p_T\) sum, and \(m_{jet}=m_{root}\) uses the same
nonnegative reconstruction.  All cluster and merge masses use this formula
in float64 before persisted feature values are cast to their locked storage
precision.

### Pair relation

For directed leaves \(i,j\), the raw 41-channel relation contains:

- same-exclusive-cluster indicators at `K=2`, `K=4`, and `K=8`;
- normalized lowest-common-ancestor depth, where root depth is zero and the
  denominator is \(\max(\text{maximum leaf depth},1)\);
- lowest-common-ancestor merge quantities
  \[
  \log(\Delta R_{merge}/R_{ref}+\epsilon),\quad
  \log(k_{T,merge}/(p_{T,jet}R_{ref})+\epsilon),\quad
  z_{merge},\quad
  \log\frac{m_{merge}+\epsilon}{m_{jet}+\epsilon},
  \]
  where
  \(k_{T,merge}=\min(p_{T,a},p_{T,b})\Delta R_{ab}\) and
  \(z_{merge}=\min(p_{T,a},p_{T,b})/(p_{T,a}+p_{T,b}+\epsilon)\);
- for query and context clusters \(r\) at each of `K=2,4,8`:
  \[
  \log\frac{p_{T,r}+\epsilon}{p_{T,jet}+\epsilon},\qquad
  \log\frac{m_r+\epsilon}{m_{jet}+\epsilon},\qquad
  \frac{n_r}{N_{valid}};
  \]
- query and context particle \(p_T\) fractions within their clusters,
  \(p_{T,i}/(p_{T,r(i)}+\epsilon)\) and
  \(p_{T,j}/(p_{T,r(j)}+\epsilon)\), at each resolution;
- query and context \(\Delta R/R_{ref}\) to their E-scheme cluster axes at
  each resolution;
- signed normalized context-minus-query cluster-\(p_T\)-rank difference at
  each resolution:
  \[
  \frac{rank(r(j))-rank(r(i))}{\max(C_K-1,1)}.
  \]

Clusters use descending average \(p_T\) rank, so exact tied cluster \(p_T\)
receives the same rank and rank zero is hardest.  A positive signed difference
therefore means that the context cluster is softer than the query cluster.
For a diagonal pair \(i=j\), same-cluster and endpoint descriptors remain
defined, while merge-only quantities are zeroed after normalization and
excluded from their normalizer fits.

All continuous quantities use the common featurewise train-only robust
normalization.  The \(N<K\) singleton rule makes every requested resolution
unambiguous while preserving the 41-channel dimension.

```text
raw dimension = 41
encoded dimension = 12
```

Required diagnostics:

- angular-tree node and requested/actual exclusive-cluster count
  distributions;
- lowest-common-ancestor depth and merge-scale distributions;
- cluster \(p_T\), mass, and multiplicity fractions;
- same-cluster frequency at all three resolutions;
- performance binned by beam-free angular-tree depth and hard-prong count;
- inference-time ablation of each of `K=2`, `K=4`, and `K=8`;
- tree-sidecar/runtime recomputation parity on fixed events;
- equivalence to a trusted reference implementation on non-tied fixtures.

---

## 13. Combination and capacity policy

Enabled relation encodings are concatenated in this canonical order:

```text
base4, PT, TRACK, PID, CHARGE, DENSITY, REGION
```

Disabled families contribute no channels.  The standard pair-embedding depth
and activation remain unchanged.  The first pair-embedding layer expands only
as required by the enabled encoded dimension.

The campaign records:

- total parameters;
- trainable parameters;
- pair-encoder parameters;
- measured forward FLOPs at batch size one and 128 valid particles;
- measured peak GPU memory at the locked training batch size;
- median inference latency as a diagnostic, not a selection criterion.

`RPT_BASE_WIDE_MAX` matches active incremental capacity, not total-model
capacity.  Define:

\[
\Delta P_{FULL}
=P(RPT\_FULL\_ALL)-P(RPT\_BASE)
\]

and

\[
\Delta P_{WIDE}
=P(RPT\_BASE\_WIDE\_MAX)-P(RPT\_BASE),
\]

where \(P\) counts every trainable parameter that participates in the forward
path.  The control receives only `base4`; it widens the three hidden dimensions
of the ordinary pair encoder while keeping its depth, operations, activation,
eight-head output width, token path, and classifier unchanged.

Before any training, enumerate integer width triples:

```text
(w1, w2, w3) in [64, 256]^3
```

Each width is independently selectable and may not be below the reference
width `64`.  Use the exact symbolic parameter formula for the locked Weaver
pair-encoder modules—including biases and normalization parameters—to rank
candidates, instantiate the selected candidate, and verify its count from
`model.parameters()`.

Choose the triple minimizing
\(|\Delta P_{WIDE}-\Delta P_{FULL}|\).  Ties use:

1. lower analytically calculated pair-encoder FLOPs at 128 valid particles;
2. smaller \(w_1+w_2+w_3\);
3. lexicographically smaller \((w_1,w_2,w_3)\).

Registry construction fails unless:

\[
\frac{|\Delta P_{WIDE}-\Delta P_{FULL}|}
{\max(\Delta P_{FULL},1)}
\leq0.02.
\]

The registry records both incremental counts, the chosen widths, mismatch,
parameter-formula version, verified instantiated count, and FLOPs.  There are
no inactive padding parameters.

`RPT_FULL_ZERO_REL` instantiates the complete full-relation architecture and
all corresponding parameters but replaces every encoded new-relation channel
with zero before concatenation.  It is the exact-shape capacity and
optimization control.

### Confirmation-only architectures and matched controls

The 21-row screen keeps the standard shared pair-bias path so relation
semantics, rather than a wholesale attention rewrite, are compared first.
Two higher-upside relational architectures and two matching base4 controls are
nevertheless predeclared now and may be trained only after the screening
relations have been selected.  Their
`selected_relation_set` is the enabled family set of the highest-ranked
non-control screening row under the Section 16 screening rule; it is selected
even if that row loses to `RPT_BASE`.  The selected run ID and family-set hash
are written before either architectural finalist is submitted.

The mandatory incremental comparison uses exactly the same selected
relations:

```text
selected shared-bias screening model
RPT_SELECTED_LAYERWISE
RPT_SELECTED_EDGEVALUE
```

Within this subsection, \(u_{ij}\) is the canonical concatenation of `base4`
and the enabled encoded family channels.  Split the ordinary ParT
pair-embedding network immediately before its final head-output projection:
the unchanged shared prefix is `PairStem`, and
\(e_{ij}=PairStem(u_{ij})\) has the locked hidden width from the exact Weaver
implementation.  Family encoders and `PairStem` are evaluated once per batch.
For a base4 architectural control \(u_{ij}\) contains only `base4`; for a
selected relational row it contains `base4` plus exactly
`selected_relation_set`.

`RPT_SELECTED_LAYERWISE`

- compute each selected family embedding and the shared `PairStem` once per
  batch;
- preserve the same family encoders and concatenation order as the selected
  relational configuration;
- replace the one shared final head-output projection with an independent
  linear projection from \(e_{ij}\) to the attention-head biases in every
  transformer layer;
- train the complete model from random initialization;
- do not share a single projected bias tensor across all layers.

This allows early layers to prefer local detector compatibility and later
layers to prefer hard-prong or regional structure without duplicating the
expensive raw relation construction.

`RPT_BASE_LAYERWISE`

- applies the identical per-layer bias-projection topology to `base4` only;
- receives no PT, TRACK, PID, CHARGE, DENSITY, or REGION values;
- is trained from scratch at all three confirmation seeds;
- controls active capacity and optimization changes from independent
  per-layer bias projections.

`RPT_SELECTED_EDGEVALUE`

- use the same layer-specific attention-bias projections;
- additionally apply a learned per-layer, per-head relation-conditioned value
  projection.  For encoded relation width \(d_e\), every head owns
  \(W^{V,(\ell,h)}\in\mathbb{R}^{d_h\times d_e}\), with no bias term:
  \[
  V^{(\ell,h)}_{ij}
  =
  V^{(\ell,h)}_j+
  W^{V,(\ell,h)}e_{ij};
  \]
- zero relation features for every invalid query or key before aggregation;
- avoid materializing `[B,H,N,N,d_h]`.  Because the value projection is
  linear, first form
  \[
  \bar e^{(\ell,h)}_i
  =
  \sum_j a^{(\ell,h)}_{ij}e_{ij},
  \]
  then add \(W^{V,(\ell,h)}\bar e^{(\ell,h)}_i\) to the ordinary per-head
  value aggregate before head concatenation and output projection;
- train from scratch and report its added parameters, FLOPs, memory, and
  latency separately.

`RPT_BASE_EDGEVALUE`

- applies the identical layerwise-bias and per-head value-message topology to
  `base4` only;
- receives no new relation-family values;
- is trained from scratch at all three confirmation seeds;
- controls the benefit of the value-message architecture and its active
  capacity.

The edge-value path is a confirmation experiment, not part of the standard
screen.  It must pass exact zero-relation equivalence, padding, gradient, and
reference-attention parity tests before production submission.  Every
edge-value run records, per layer and head,

\[
\frac{\|W^V\bar e\|_2}
{\|\sum_j a_{ij}V_j\|_2+\epsilon}
\]

along with its distribution and masked-query count.

Claims are separated deliberately:

- the selected shared-bias model tests the new relation values alone against
  the ordinary ParT path and capacity controls;
- `SELECTED_LAYERWISE` tests the incremental layer-specific projection;
- `SELECTED_EDGEVALUE` tests the incremental value message;
- the two `BASE_*` rows test whether those architectural changes help without
  new relation values.

Layerwise or edge-value wins are labelled compound architecture results unless
their matched base4 control and incremental selected-relation comparison
support a narrower attribution.

No performance threshold is allowed to prevent a scientifically valid
variant from training.  Performance rules produce warnings in reports; only
invalid inputs, failed provenance, nonfinite training, or missing artifacts
stop dependencies.

---

## 14. Prespecified screening matrix

All rows below train from random initialization with seed `101` during
screening.

### Controls

| Run ID | Enabled relations | Purpose |
|---|---|---|
| `RPT_BASE` | base4 | Exact matched HLT ParT |
| `RPT_BASE_WIDE_MAX` | base4 | Parameter-matched capacity control |
| `RPT_FULL_ZERO_REL` | base4 plus zeroed full relation channels | Exact-shape control |

### Single relation families

| Run ID | Enabled relations |
|---|---|
| `RPT_PT` | base4 + PT |
| `RPT_TRACK` | base4 + TRACK |
| `RPT_PID` | base4 + PID |
| `RPT_CHARGE` | base4 + CHARGE |
| `RPT_DENSITY` | base4 + DENSITY |
| `RPT_REGION` | base4 + REGION |

### High-value pairs

| Run ID | Enabled relations |
|---|---|
| `RPT_PT_TRACK` | PT + TRACK |
| `RPT_TRACK_PID` | TRACK + PID |
| `RPT_TRACK_CHARGE` | TRACK + CHARGE |
| `RPT_PID_CHARGE` | PID + CHARGE |
| `RPT_PT_DENSITY` | PT + DENSITY |
| `RPT_PT_REGION` | PT + REGION |
| `RPT_TRACK_REGION` | TRACK + REGION |

### High-potential higher-order combinations

| Run ID | Enabled relations |
|---|---|
| `RPT_TRACK_PID_CHARGE` | TRACK + PID + CHARGE |
| `RPT_PT_TRACK_DENSITY` | PT + TRACK + DENSITY |
| `RPT_PT_TRACK_REGION` | PT + TRACK + REGION |
| `RPT_PT_TRACK_PID_CHARGE` | PT + TRACK + PID + CHARGE |
| `RPT_FULL_ALL` | PT + TRACK + PID + CHARGE + DENSITY + REGION |

The fixed screening matrix therefore contains 21 runs.  No row is added after
looking at `val_select` (`stack_val`).

After screening, the selector may also synthesize one
`RPT_SELECTED_UNION` configuration containing the union of relation families
present in the two highest-ranked single-family rows.  This is deterministic
even when every single loses to the baseline.  That configuration is eligible
only for confirmation; it must never retroactively replace a screening row.

### Configuration-role contract

Every registry row has exactly one immutable `configuration_role`:

| Role | Configurations | Eligible as nominal relational winner? |
|---|---|---|
| `reference_baseline` | `RPT_BASE` | No |
| `capacity_control` | `RPT_BASE_WIDE_MAX`, `RPT_FULL_ZERO_REL` | No |
| `architecture_control` | `RPT_BASE_LAYERWISE`, `RPT_BASE_EDGEVALUE` | No |
| `scientific_finalist` | every row with at least one active new relation family, including all singles/combinations, `RPT_FULL_ALL`, `RPT_SELECTED_UNION`, `RPT_SELECTED_LAYERWISE`, and `RPT_SELECTED_EDGEVALUE` | Yes |
| `semantic_control` | `RPT_SELECTED_UNARY` and inference perturbation records | No |

Roles affect selection and reporting, never whether a valid run is allowed to
finish.  Only `scientific_finalist` rows participate in relational-winner
selection.  The baseline and all controls remain mandatory matched
comparisons, are eligible for sealed final-test reporting when registered,
and can demonstrate that an apparent relation gain is actually capacity or
architecture.  If every scientific finalist loses to `RPT_BASE`, the selector
still names the highest-ranked scientific finalist as `best_available` and
marks `confirmation_gain_positive = false`; it never promotes a control as
the relational winner.  This flag is true exactly when the nominal winner's
mean matched-seed accuracy difference from `RPT_BASE` is strictly greater
than zero; it is a validation summary, not a substitute for the sealed-test
success criteria.

---

## 15. Training protocol

### Optimizer and schedule

All models use:

```text
loss = unweighted 10-class cross entropy
optimizer = AdamW
learning rate = 1.0e-3
betas = (0.9, 0.999)
weight decay = 1.0e-4
gradient clipping = 1.0
maximum epochs = 40
minimum epochs before early stopping = 12
early-stop patience = 8 validations
warm-up = first 5% of optimizer updates
schedule after warm-up = cosine decay
minimum learning rate = 1.0e-5
microbatch size = 64
gradient accumulation = 2
effective batch size = 128
num_workers = 0
mixed precision = BF16 on GH200
```

If a device does not support BF16, the worker may fall back to FP16 with a
gradient scaler, and the precision mode must be recorded.  Results from
different precision modes are not silently aggregated.

The training loader is deterministically shuffled per seed.  Every
configuration at a given seed receives the same training identity order.

### Checkpoint selection

At every completed epoch \(e\), evaluate `model_val` (`val_stop`) once and
store finite accuracy \(a_e\) and cross entropy \(c_e\) as canonical float64
values.  Given any completed eligible epoch set \(E\), checkpoint selection is
global:

1. compute \(a_{max}=\max_{e\in E}a_e\);
2. retain \(S=\{e\in E:a_{max}-a_e\leq0.0001\}\);
3. choose the epoch in \(S\) with minimum \(c_e\);
4. if multiple epochs have exactly the same serialized float64 cross entropy,
   choose the earliest epoch.

This is not an order-dependent comparison with the currently retained
checkpoint.  Nonfinite metrics make the run invalid rather than silently
removing an epoch from \(E\).

Early-stopping patience uses the same global selector.  After each validation,
recompute the preferred epoch over all epochs completed so far:

- initialize `patience_count = 0` after the first validation;
- reset it to zero exactly when the newly completed epoch becomes the globally
  preferred epoch;
- otherwise increment it by one;
- never stop before epoch 12 has completed;
- after epoch 12, stop immediately after a validation when
  `patience_count >= 8`;
- always recompute the final selected checkpoint over every completed epoch,
  including the stopping epoch.

The selected checkpoint is then evaluated exactly once on `stack_val`
(`val_select`).  `val_select` never selects epochs.

### Training-seed policy

Screening:

```text
seed = 101
all 21 fixed rows
```

Confirmation:

```text
seeds = 101, 202, 303
```

The seed-101 screening checkpoint is reused if its hashes match.  Seeds 202 and
303 are newly trained from scratch.

Mandatory confirmation rows:

- `RPT_BASE`;
- `RPT_BASE_WIDE_MAX`;
- `RPT_FULL_ZERO_REL`;
- all six single relation families, including `RPT_TRACK` whether or not it
  ranks among the top singles;
- best two predeclared non-full combination rows, ranked by the screening
  rule from the seven pair rows and four named higher-order rows other than
  `RPT_FULL_ALL`;
- `RPT_FULL_ALL`;
- `RPT_SELECTED_UNION` when distinct;
- `RPT_BASE_LAYERWISE`;
- `RPT_BASE_EDGEVALUE`;
- `RPT_SELECTED_LAYERWISE`;
- `RPT_SELECTED_EDGEVALUE`.

Thus every single-family claim and every capacity control receives all three
seeds.  Seed `101` is reused from screening after hash validation, while seeds
`202` and `303` are trained from scratch.  The two selected architectural
finalists and their two base4 architectural controls train all three seeds
because they have no screening checkpoint.

No final-test metric is available during confirmation.

---

## 16. Screening and confirmation selection

### Screening rank

Produce a complete role-labelled ranking for reporting, but select the
screening relation set and `best_available` relational row only from
`scientific_finalist` configurations.  For any candidate set, compute its
global maximum `val_select` accuracy and retain rows no more than `0.0001`
below that maximum.  Select within that global window by:

1. lower `val_select` cross entropy;
2. lower parameter count;
3. lexicographically smaller run ID.

Construct a complete ranking by removing the selected row and repeating this
global-window operation on the remaining rows.  This avoids non-transitive
pairwise “within tolerance” comparisons.

The selector always emits a result, even if every relation model is worse than
the baseline.  A negative result must not block confirmation of the mandatory
controls and best available rows.

### Confirmation aggregation

For each confirmed configuration report:

- mean and median `val_select` accuracy across three seeds;
- sample standard deviation;
- mean cross entropy;
- per-seed difference from the matched-seed `RPT_BASE`;
- mean paired accuracy difference;
- number of seeds that beat the matched baseline.

The primary finalist ordering is applied only to rows whose
`configuration_role == scientific_finalist`:

1. highest mean matched-seed accuracy difference;
2. lower mean cross entropy;
3. smaller accuracy standard deviation;
4. lower parameter count;
5. run ID.

The first row in that eligible ordering is the nominal relational winner.
`reference_baseline`, `capacity_control`, `architecture_control`, and
`semantic_control` rows are compared with it but cannot occupy that position.

Before final-test submission, write an immutable
`locked_finalists.json` containing:

- baseline ID;
- all locked scientific and control evaluation-row IDs;
- each row's configuration role and relational-selection eligibility;
- nominal relational-winner ID and `confirmation_gain_positive` flag;
- per-row mean matched-seed deltas and
  `capacity_control_reproduces_gain`;
- checkpoint hashes for all three seeds;
- campaign-spec hash;
- split-manifest hash;
- HLT-cache hashes;
- selection metrics;
- deterministic selection reason.

All confirmed rows may be final-tested after this lock is written.  The final
test is used for reporting, not further architecture selection.

---

## 17. Final-test and statistical reporting

The final evaluation uses the same 500,000 HLT jets for every locked
checkpoint.

For each seed and configuration report:

- configuration role and relational-selection eligibility;
- 10-class accuracy;
- cross entropy;
- macro per-class accuracy;
- per-class efficiency;
- one-vs-rest AUC;
- 15-bin expected calibration error;
- Brier score;
- confusion matrix;
- QCD-versus-each-signal background rejection at signal efficiencies 30% and
  50%;
- parameter count, FLOPs, peak memory, and measured latency.

Persist logits or probabilities and predicted labels in compressed,
identity-bound files so paired statistics can be recomputed without rerunning
the model.

For every finalist versus its matched baseline calculate:

- paired absolute accuracy difference;
- paired bootstrap 95% confidence interval with 10,000 deterministic
  resamples;
- McNemar discordant counts;
- relative error reduction;
- per-class paired accuracy differences;
- seed-level mean and spread.

The 500,000-event test reduces evaluation noise but does not replace the
three-seed confirmation.  Claims must present both sources of uncertainty.

---

## 18. Attention and relation diagnostics

Every variant records, per layer and head when accessible:

- pair-bias mean, standard deviation, absolute mean, and maximum;
- attention entropy;
- maximum attention weight;
- fraction of attention assigned to leading, subleading, and soft particles;
- fraction assigned within angular bands;
- gradient norm of each relation-family encoder;
- relation-family encoded activation norm.

Combination models also run inference-only family dropouts on `val_select`:

```text
full model
full model with PT zeroed
full model with TRACK zeroed
...
```

These ablations are diagnostic and do not retrain the model.  They help
distinguish a family that is merely present from one the trained model uses.

No attention visualization or bias statistic is itself evidence of a causal
physics mechanism.  The primary result remains held-out tagging performance.

### Semantic controls for the eventual winner

After confirmation selects the strongest relational configuration, run four
predeclared controls on `val_select`; they never inspect `final_test`:

1. **Within-jet shuffled relations.** Apply one non-identity permutation to
   valid particle indices and transform every selected relation matrix as
   \(E\mapsto PEP^T\), leaving tokens and padding positions fixed.  This
   preserves the relation distribution while breaking its association with
   the correct particle pair.  Jets with fewer than two valid particles are
   excluded from this perturbation and counted explicitly.
2. **Wrong-event relations.** Stratify jets by exact \(N_{valid}\), then use a
   deterministic class-blind derangement inside each stratum.  Never match an
   event to itself or to a different multiplicity.  Within each equal-length
   source/destination pair, align particles by descending average \(p_T\)
   rank, breaking nonidentical ties with the canonical physics-tuple ordering
   from the REGION contract.  If a stratum has fewer than two jets, exclude
   it from this control and report its event count; never silently borrow a
   different multiplicity.
3. **Directional swap.** Transpose only the new selected relation channels,
   \(E_{ij}\mapsto E_{ji}\), while retaining the standard four ParT channels.
4. **Unary endpoint control.** Train `RPT_SELECTED_UNARY` from scratch for
   seeds `101,202,303` under the exact contract below.

#### `RPT_SELECTED_UNARY` contract

Lock `unary_source_relation_set` to the active families of the nominal
relational winner.  Also lock `unary_reference_run_id` to the ordinary
shared-bias row with exactly that family set.  If the nominal winner is
`RPT_SELECTED_LAYERWISE` or `RPT_SELECTED_EDGEVALUE`, its reference is the
already trained selected shared-bias screening row; the unary control is not
matched to the compound architectural extension.

For each valid particle \(i\), concatenate endpoint-decomposable quantities
from the locked families in canonical family order:

- `PT`: \(f_i\), \(x_i\), and average normalized rank \(r_i\);
- `TRACK`: the exact seven-channel \(t_i\) vector from Section 8:
  \(d_0,d_z,\log\sigma^{eff}_{d_0},\log\sigma^{eff}_{d_z},s_0,s_z\), and
  `track_valid`;
- `PID`: one independently learned `Embedding(6, 8)` of the same canonical
  six-state PID category and unknown/multi-hot policy;
- `CHARGE`: quantized charge \(q_i\) plus one independently learned
  `Embedding(3, 4)` of charge state `-1/0/+1`;
- `DENSITY`: the complete normalized 22-channel per-particle descriptor
  \(d_i\);
- `REGION`: at each of `K=2,4,8`, the particle's cluster log-\(p_T\) fraction,
  log-mass fraction, multiplicity fraction, particle-within-cluster \(p_T\)
  fraction, distance to cluster axis, and normalized average cluster rank,
  for 18 channels total.

Numeric features use the same registered transformations, applicability
masks, and normalizer artifact as the pair model.  Unary PID and charge
embeddings are newly initialized and trained with the control; they never
reuse weights learned by a relational checkpoint.

Explicitly pair-only quantities are absent: TRACK \(\chi^2\) and endpoint
products, the PID directed residual table, charge-pair states, REGION
same-cluster/LCA/merge quantities, and every query-context difference.  This
is intentional.

Let the concatenated unary width be \(D_u\).  The adapter is:

```text
Linear(D_u, h1)
GELU
RMSNorm(h1)
Linear(h1, h2)
GELU
RMSNorm(h2)
Linear(h2, 128)
```

Its masked output is added to the ordinary 128-dimensional particle embedding
immediately before the first particle-attention block:

\[
x^{input}_i=x^{standard\ ParT}_i+UnaryAdapter(u_i).
\]

The standard `base4` pair path remains exact.  No new unary feature enters
query/key attention as an explicit pair bias, and there is no edge-value
message.

Parameter matching is based on active incremental trainable parameters, not
unused padding:

\[
\Delta P_{reference}
=P(unary\_reference\_run)-P(RPT\_BASE),
\]

\[
\Delta P_{unary}
=P(PID/charge\ unary\ embeddings)+P(UnaryAdapter).
\]

Before training, exhaustively search integer
\((h_1,h_2)\in[1,512]\times[1,512]\).  Choose the pair minimizing
\(|\Delta P_{unary}-\Delta P_{reference}|\); ties use lower measured adapter
FLOPs, then smaller \(h_1\), then smaller \(h_2\).  Registry construction
fails if the relative incremental mismatch exceeds 2%.  It records both
parameter equations, selected widths, total and incremental counts, relative
mismatch, and FLOPs.  Every selected parameter participates in the forward
path.

The first three are inference perturbations and therefore diagnose
association and directionality rather than standalone model quality.  The
unary row tests whether endpoint-decomposable information specifically
benefits from pairwise delivery rather than a matched token-side nonlinear
adapter.  It cannot by itself replace explicitly pair-only inputs that were
deliberately withheld.  All perturbation artifacts record the number of
permuted valid particles, fixed points, derangement hash, mask agreement, and
relation norms before and after transformation.

These controls do not create a performance gate, alter the prespecified
confirmation ranking, or cancel final evaluation.  The final lock contains
all confirmed scientific, baseline, capacity-control, and
architecture-control rows plus `RPT_SELECTED_UNARY` as an explicitly labelled
semantic control; the three inference perturbations remain validation-only
diagnostics.

---

## 19. Artifact and provenance contract

Campaign root:

```text
checkpoints/relational_particle_transformer/<campaign_id>/
```

Required layout:

```text
campaign_spec.json
backend/
  relational_ca_tree_v1.so
  backend_manifest.json
  throughput_probe.json
inputs/
  split_manifest.json.gz
  split_audit.json
  raw_input_schema.json
  preconstruction_raw_input_audit.json
  hlt_cache/
  hlt_cache_audit.json
  relation_normalization.json
  relation_tree_cache/
    <split>_exclusive_ca_v1/
      manifest.json
      shards/
        shard_<index>.npz
        shard_<index>.metadata.json
  postconstruction_input_audit.json
registry/
  relation_family_registry.json
  screening_registry.json
runs/
  <run_id>/
    seed_<seed>/
      config.json
      training_curves.json
      best_model_val.pt
      checkpoint_registration.json
      val_stop_metrics.json
      val_select_metrics.json
selection/
  screening_summary.json
  confirmation_registry.json
  confirmation_summary.json
  locked_finalists.json
  semantic_controls/
    perturbation_metrics.json
    unary_control_registry.json
final_test/
  <run_id>/
    seed_<seed>/
      metrics.json
      predictions.npz
reports/
  relational_part_report.json
  relational_part_report.md
job_ledgers/
```

Each JSON artifact has:

- a contract identifier;
- a schema version;
- canonical SHA-256 content hash;
- parent hashes;
- source commit and dirty-status hash;
- creation timestamp for audit only, excluded from scientific identity where
  appropriate.

Each checkpoint registration records:

- model state hash;
- configuration role and relational-selection eligibility;
- run registry hash;
- relation-family registry hash;
- locked raw-input/sentinel-schema hash;
- relation-normalization hash;
- angular-tree schema, canonicalization, source, compiler, and binary hashes
  when `REGION` is enabled;
- exact angular-tree split-manifest and shard-set hashes when `REGION` is
  enabled;
- HLT cache hashes;
- split hash;
- seed;
- selected epoch and selection metrics;
- parameter and FLOP profile;
- explicit `hlt_only_inference = true`;
- explicit `offline_or_teacher_required = false`.

The loader fails closed on stale or cross-campaign artifacts.

---

## 20. Storage and runtime policy

Pairwise relation tensors are generated on the GPU per batch and are never
cached persistently.  `REGION` stores only its compact \(O(N)\) beam-free tree
sidecar; full pair matrices are materialized transiently.  This avoids an
\(O(N^2)\) storage artifact.

The campaign persists:

- one compressed HLT cache for 1.75 million jets;
- compact identity-bound angular-tree shards and one authenticated manifest
  per split;
- selected and last checkpoints during training;
- small metrics files;
- final-test predictions only for locked finalists.

After a run completes successfully:

- retain `best_model_val.pt`;
- delete or overwrite `last.pt`;
- retain optimizer state only for actively resumable jobs;
- do not persist per-batch attention matrices;
- persist aggregated attention diagnostics only.

The bootstrap estimates projected storage and refuses submission only when the
filesystem cannot preserve a configurable safety margin.  The default campaign
budget is 20 GiB with a minimum 20 GiB free-space reserve after projection.
Preflight measures the tree-sidecar bytes per jet from a representative sample
and uses that measurement, not a hard-coded estimate, in the projection.
Completed hash-valid shards survive scheduler interruption and are reused;
temporary or unregistered shards do not count toward completion.

---

## 21. Proposed implementation surfaces

### Python package

Create:

```text
teacher_logit_reco/relational_part/
  __init__.py
  contracts.py
  splits.py
  normalization.py
  pair_base.py
  relation_pt.py
  relation_track.py
  relation_pid_charge.py
  relation_density.py
  relation_region.py
  ca_tree.py
  pair_builder.py
  model.py
  registry.py
  train.py
  evaluation.py
  semantic_controls.py
  selection.py
  reporting.py
  provenance.py
  csrc/
    relational_ca_tree_v1.cpp
```

Reuse numerical utilities from:

- `jetclass_fresh.part_inputs`;
- `jetclass_fresh.hlt_cache`;
- `teacher_logit_reco.local_graph_part.edge_features`;
- `teacher_logit_reco.subtoken_part.pairwise`;

but do not silently inherit either older experiment's scientific registry.
The relational campaign owns its own feature names, contracts, and hashes.

### Command-line entry points

Create:

```text
scripts/build_relational_part_campaign.py
scripts/audit_relational_part_raw_inputs.py
scripts/audit_relational_part_inputs.py
scripts/fit_relational_part_normalization.py
scripts/build_relational_part_tree_backend.py
scripts/probe_relational_part_tree_backend.py
scripts/build_relational_part_angular_tree_cache.py
scripts/finalize_relational_part_angular_tree_cache.py
scripts/train_relational_part.py
scripts/evaluate_relational_part.py
scripts/select_relational_part_screening.py
scripts/build_relational_part_confirmation_registry.py
scripts/aggregate_relational_part_confirmation.py
scripts/evaluate_relational_part_semantic_controls.py
scripts/evaluate_relational_part_final_test.py
scripts/write_relational_part_report.py
scripts/submit_relational_part_graph.py
```

All CLIs support `--dry-run` where applicable and print the resolved immutable
configuration before mutation.

### Slurm entry points

Create:

```text
sbatch/run_build_relational_part_splits.sh
sbatch/run_audit_relational_part_raw_inputs.sh
sbatch/run_build_relational_part_hlt_cache.sh
sbatch/run_audit_relational_part_inputs.sh
sbatch/run_fit_relational_part_normalization.sh
sbatch/run_build_relational_part_tree_backend.sh
sbatch/run_probe_relational_part_tree_backend.sh
sbatch/run_build_relational_part_angular_tree_shard.sh
sbatch/run_finalize_relational_part_angular_tree_cache.sh
sbatch/run_train_relational_part.sh
sbatch/run_select_relational_part_screening.sh
sbatch/run_submit_relational_part_confirmation.sh
sbatch/run_aggregate_relational_part_confirmation.sh
sbatch/run_submit_relational_part_final_test.sh
sbatch/run_evaluate_relational_part_final_test.sh
sbatch/run_write_relational_part_report.sh
sbatch/submit_relational_part_tigris_full.sh
```

---

## 22. Required tests

### Pair and model parity

- External standard-four pair features reproduce Weaver's internal pair
  features.
- `RPT_BASE` logits and gradients reproduce the reference ParT.
- Padding changes neither valid-token logits nor pair summaries.
- All-invalid and one-particle safety cases remain finite.
- Wrapped \(\Delta\phi\) is correct across the \(-\pi/\pi\) boundary.

### Relation-family semantics

- PT is directional and responds correctly when query/context are swapped.
- PT average-tied rank is permutation equivariant, and exact ties receive the
  same rank.
- Every unspecified \(\epsilon\) resolves to the serialized global `1e-6`
  constant.
- Featurewise robust-normalization statistics are fitted from `model_train`
  only, serialize deterministically, and reject a stale HLT parent.
- Every continuous normalizer excludes inapplicable values, records its
  applicability rule/count, and counts node features once per particle.
- Binary and categorical channels are not numerically standardized.
- Neutral particles never become valid tracks from zero-valued displacement
  fields.
- TRACK reads raw displacement before significance transformation.
- TRACK context-minus-query normalized-difference formulas match
  hand-calculated fixtures and reverse sign on endpoint swap.
- TRACK effective uncertainties and normalized differences remain finite at
  zero or tiny reported uncertainty.
- TRACK invalid, sentinel, and partial-validity states follow the locked
  four-state contract.
- TRACK audits a declared sentinel policy and never infers one from an
  observed zero concentration.
- The Siamese TRACK encoder shares its endpoint weights exactly.
- PID zero-hot handling is explicit and multi-hot input fails preflight.
- PID pair indexing is directional and complete.
- PID factorized endpoint embeddings receive gradients for rare states.
- CHARGE maps all nine states correctly.
- CHARGE passes exactly 12 channels into its family encoder.
- DENSITY excludes padding and self from both count and \(p_T\) summaries.
- DENSITY annuli are disjoint at boundaries.
- DENSITY smooth log-radius equations, zero-denominator behavior, and exact
  valid-neighbor fractions match hand-calculated fixtures.
- DENSITY self share is zero for a zero denominator.
- Beam-free REGION clustering, exclusive `K=2/4/8` assignments, and LCA
  extraction are deterministic.
- REGION's requested \(K>N\) resolution produces \(N\) valid singleton
  clusters and retains exactly 41 channels.
- REGION LCA depth, average tied cluster rank, and stable log mass fractions
  match hand-calculated trees.
- REGION uses root vector-summed \(p_T\), nonnegative float64 mass
  reconstruction, and context-minus-query rank signs.
- REGION outputs permute with particle permutations after identity undo.
- Compact REGION sidecars match direct runtime tree recomputation.
- Missing or mismatched compiled tree backends fail preflight.
- The backend rejects ABI, numerical-flag, binary-hash, and self-test
  mismatches.
- Shard interruption/resume reuses only atomically completed hash-valid
  shards, and finalization rejects missing, partial, duplicated, or stale
  shards.
- The throughput probe uses the locked binary and emits every required
  projection field before bulk submission.
- Probe sampling produces exact quotas, salted identities, largest-remainder
  redistribution, and population-weighted projections on undersized-stratum
  fixtures.
- Every family zeroes invalid pairs.

### Combination and capacity controls

- Canonical family concatenation order is stable.
- A run ID resolves to exactly its registered family set.
- `RPT_FULL_ZERO_REL` has the same parameter shapes as `RPT_FULL_ALL`.
- Zero-relation inputs are exactly zero at the shared pair encoder.
- `RPT_BASE_WIDE_MAX` width search and tie-breaks are deterministic, every
  added parameter is active, and incremental parameter mismatch is at most
  2%.
- Capacity-gain reproduction uses the exact three-seed mean-delta comparison
  from Section 26.
- Every registry row has exactly one valid configuration role, and only
  `scientific_finalist` rows are relational-selection eligible.
- Layer-specific projections reuse the computed family embedding but have
  distinct parameters and gradients in every layer.
- `RPT_BASE_LAYERWISE` and `RPT_BASE_EDGEVALUE` contain no new relation
  values and match the selected architectural topology.
- Edge-value attention is exactly reference-equivalent when its relation
  projection is zero.
- Edge-value masking prevents padded keys and queries from contributing.
- Efficient projected-after-aggregation edge messages equal explicit
  pair-conditioned values on a miniature fixture.

### Training and selection

- A miniature CPU run trains and resumes deterministically.
- Checkpoint selection computes the global maximum-accuracy window before CE
  and epoch tie-breaking.
- Early-stopping patience resets exactly when the current epoch becomes the
  globally preferred checkpoint.
- `val_select` cannot select checkpoints.
- The screening selector emits a result when all variants lose.
- Confirmation reuses only hash-matching seed-101 artifacts.
- Confirmation includes all six singles and all three controls at three
  seeds, independent of screening rank.
- Both base4 architectural controls and both selected architectural finalists
  train at all three seeds.
- Training-seed results are paired to the same-seed baseline.
- Semantic relation shuffling is non-identity on eligible jets and preserves
  pair-value distributions.
- Wrong-event relations use a class-blind, exact-\(N_{valid}\) derangement
  with no fixed point and report underpopulated excluded strata.
- Directional swap leaves the standard four ParT channels unchanged.
- `RPT_SELECTED_UNARY` receives no new pairwise channel.
- `RPT_SELECTED_UNARY` resolves the exact per-family endpoint registry,
  independently owns categorical embeddings, enters before the first
  particle-attention block, and meets the 2% active incremental-parameter
  tolerance without unused parameters.

### Final-test seal and provenance

- Training and selection CLIs reject `final_test`.
- The sealed preparation CLI may build final-test HLT/tree inputs but rejects
  checkpoints, inference requests, normalizer fitting, and metric outputs.
- Final evaluation rejects an absent or stale `locked_finalists.json`.
- A checkpoint from another split or HLT cache is rejected.
- Final-test rows load HLT arrays only.
- Checkpoints declare no offline or teacher dependency.

### Shell and Tigris contract

- All Tigris jobs export `PYTHONNOUSERSITE=1`.
- Account is exactly `reu-aisocial`.
- Conda root defaults to `/home/ryreu/miniforge3-aarch64`.
- Conda environment defaults to `atlas_kd_tigris`.
- GPU workers request `gpu:gh200:1`.
- The cheap raw-input/schema audit precedes HLT-cache construction,
  normalization fitting, backend probing, and tree shards.
- The full postconstruction audit depends on HLT cache, normalizer, and every
  finalized tree-split manifest and precedes screening.
- Backend build and throughput probe precede sharded angular-tree
  construction; its finalizer and normalization fitting precede every
  applicable relation run.
- Performance warnings do not create `DependencyNeverSatisfied`.
- Failed execution or provenance validation does block downstream jobs.

---

## 23. Failure modes and interpretations

### No single relation beats the baseline

The ordinary ParT query/key projections and standard kinematics may already
extract the available HLT relationships.  Confirm the capacity controls and
do not assume that adding more combinations will solve it.

### PT helps but track/PID do not

The main missing inductive bias is likely directional context importance, not
detector identity or displacement compatibility.

### TRACK helps heavy-flavor classes only

This is a scientifically useful result.  A class-conditional specialist or
later fusion may be more appropriate than requiring a global accuracy win.

### PID or CHARGE helps only in combination

Their information may be meaningful only when conditioned on displacement,
hardness, or region structure.  The predeclared combinations are designed to
test this.

### BASE_WIDE matches the relation models

The gain is likely pair-encoder capacity or optimization rather than relation
semantics.  Report that directly.

### FULL_ZERO_REL matches FULL_ALL

The added parameterization, not relation values, explains the gain.

### Region features are unstable

Audit compiled-backend identity, angular-tree hashes, exclusive-cluster
counts, tied-merge handling, sidecar/runtime parity, and permutation
equivariance.  Do not interpret a region result until deterministic behavior
is established.

### One seed wins and two lose

The architecture is not confirmed.  Report the seed spread and do not promote
the single lucky result.

### Validation improves but final test does not

Treat it as architecture-selection overfitting.  The final-test result is
reported without choosing a different model afterward.

---

## 24. Eight equal implementation steps

### Step 1 of 8: campaign contracts, split profile, and registry

Implement the immutable 1M/125k/125k/500k split contract, campaign-spec
hashing, relation-family registry, 21-row screening registry, artifact layout,
normalization contract, compact angular-tree resource contract,
confirmation-only architectural registry, configuration-role/selectability
contract, semantic-control registry, and measured storage projection.  Add
fail-closed split and HLT provenance checks.

Done when miniature manifests, registry hashes, and storage audits are covered
by tests and no training code is required to interpret a run ID.

### Step 2 of 8: exact ParT pair-path parity

Implement explicit generation and forwarding of Weaver's standard four pair
features.  Build `RPT_BASE` through the shared relational wrapper and prove
logit, gradient, mask, and state-dictionary parity with the reference HLT
ParT.

Done when the baseline cannot be distinguished numerically from the reference
within the locked tolerance.

### Step 3 of 8: PT, PID, and CHARGE relation families

Fit and register deterministic featurewise `model_train` robust-normalization
statistics.  Implement directional transverse-momentum features with
average-tied rank and raw fractions, factorized directional six-category PID
pair embeddings with strict zero/multi-hot rules, nine-state charge embeddings
with an explicit 12-channel MLP input, common masking, diagnostics, and
focused unit tests.

Done when all three singles can be instantiated, trained in a miniature run,
and permuted without changing event logits after undoing the permutation.

### Step 4 of 8: TRACK and DENSITY relation families

Implement measurement-aware track validity, train-fitted uncertainty floors,
raw-displacement `asinh` significance, the shared Siamese per-track encoder,
explicit \(\chi^2\) compatibility, disjoint annuli, smooth log-radius kernels,
local PID/displacement composition, exact neighbor fractions, and nonfinite
and boundary tests.

Done when track invalidity cannot leak through neutral particles and density
statistics reproduce hand-calculated fixtures.

### Step 5 of 8: REGION hierarchy and combination model

Implement the compiled deterministic HLT-only beam-free angular-tree backend,
locked ABI/numerical build, canonical leaf/tie handling, throughput probe,
resumable 10k-jet shards and atomic finalization, compact \(O(N)\) sidecars,
exclusive `K=2/4/8` partitions, lowest-common-ancestor and merge features,
region descriptors, canonical family concatenation, all predeclared
combinations, `BASE_WIDE_MAX`, and `FULL_ZERO_REL`.

Done when every one of the 21 screening rows builds from the registry and the
region path passes deterministic permutation, reference, ABI,
interruption/resume, and sidecar/runtime parity tests.

### Step 6 of 8: training, evaluation, and resource profiling

Implement the common from-scratch trainer, warm-up/cosine schedule,
deterministic data order, BF16 policy, global maximum-window checkpoint
selection, exact patience-reset behavior, resume behavior, `val_select`
evaluation, relation diagnostics, parameter/FLOP profiling, and bounded
checkpoint retention.  Implement the layer-specific relation-bias and
relation-conditioned edge-value attention paths, their base4 architectural
controls, memory-efficient message aggregation, norm diagnostics, and
zero-projection reference parity.

Done when a miniature end-to-end screening campaign produces registered
checkpoints and comparable metrics for every model family.

### Step 7 of 8: screening selector, three-seed confirmation, and sealed test

Implement deterministic screening rank, finalist registry,
`RPT_SELECTED_UNION`, all-single/all-control matched-seed confirmation,
`RPT_BASE_LAYERWISE`, `RPT_BASE_EDGEVALUE`, `RPT_SELECTED_LAYERWISE`,
`RPT_SELECTED_EDGEVALUE`, the four semantic controls including the trained
fully enumerated and incremental-parameter-matched unary endpoint row,
role-filtered confirmation aggregation, `locked_finalists.json`, HLT-only
final evaluation, paired bootstrap statistics, and the Markdown/JSON report.

Done when stale artifacts and premature final-test access fail closed while a
fully negative synthetic campaign still reaches a valid report.

### Step 8 of 8: Tigris preparation and full submission graph

Implement the production Slurm workers and one top-level Tigris submission
script.  The script must:

1. create a unique campaign root;
2. queue the 1M/125k/125k/500k split build;
3. run a cheap preconstruction raw-input/schema audit after the split,
   checking required branches, shapes, dtypes, finite-value policy, locked
   sentinels, PID/charge domains, complete manifest counts, and the exact
   salted sample from Section 4;
4. queue fixed-HLT cache construction only after that audit passes;
5. compile and authenticate `relational_ca_tree_v1` only after that audit;
6. run the locked 20k-jet throughput/storage/parity probe after the HLT cache
   and backend are ready;
7. submit resumable angular-tree shard arrays and their atomic split
   finalizers, allowing provenance-only sealed preparation workers to prepare
   `final_test` inputs without inference or metrics;
8. fit train-only relation normalizers after the raw audit and HLT cache,
   without opening `final_test`;
9. run the full postconstruction audit only after the HLT cache, normalizer,
   backend, and every split's tree finalizer are complete, validating
   identities, hashes, class balance, tree/runtime parity, storage, and
   HLT-only inputs;
10. queue all three screening controls, all six singles, all seven pairs, and
    all five higher-order rows after the full audit;
11. aggregate screening without treating poor performance as job failure;
12. submit seeds `202/303` for every screening control and single plus all
   three seeds for the selected and base4 layerwise/edge-value rows, while
   confirming the best two combinations and full/union configurations;
13. aggregate three-seed confirmation, train/evaluate the unary control, and
   run the relation shuffle, wrong-event, and directional-swap diagnostics;
14. write `locked_finalists.json` only after the confirmation and semantic
    controls;
15. submit final-test evaluation only after that lock;
16. write the final report and job ledger.

Done when `--dry-run` prints the complete DAG, a smoke campaign completes, and
the production command safely queues the entire workflow on Tigris.

---

## 25. Tigris production defaults

The top-level script defaults to:

```text
PROJECT_DIR=/home/ryreu/atlas/Fresh_check
DATA_DIR=/home/ryreu/atlas/PracticeTagging/data
OUTPUT_ROOT=/home/ryreu/atlas/Fresh_check/checkpoints
CONDA_BASE=/home/ryreu/miniforge3-aarch64
CONDA_ENV=atlas_kd_tigris
PYTHONNOUSERSITE=1
SBATCH_ACCOUNT=reu-aisocial
SBATCH_PARTITION=tigris
GPU_GRES=gpu:gh200:1
GPU_CPUS_PER_TASK=16
GPU_MEM=220G
CPU_CPUS_PER_TASK=16
CPU_MEM=192G
```

The bootstrap verifies that the configured data roots contain enough events
for every class before submitting the cache job.

Screening runs are submitted as a bounded Slurm array over the immutable
21-row registry.  Array concurrency is configurable so the user can exploit
available GPUs without changing scientific identity.

The eventual production command is:

```bash
cd /home/ryreu/atlas/Fresh_check
export PYTHONNOUSERSITE=1
bash sbatch/submit_relational_part_tigris_full.sh
```

The script prints:

- campaign root;
- source commit and dirty-status hash;
- split/cache job IDs;
- compiled-backend manifest and throughput-probe path;
- tree-shard array/finalizer job IDs and completion counts;
- screening array job ID;
- selector and continuation job IDs;
- ledger path;
- commands for `squeue` and `sacct`;
- commands for downloading reports and final metrics.

---

## 26. Definition of success

The campaign is scientifically successful if it cleanly determines whether
explicit HLT pair relations improve ParT, even if the answer is no.

For confirmation configuration \(C\), define its mean matched-seed validation
gain:

\[
\overline{\Delta}(C)
=
\frac{1}{3}
\sum_{s\in\{101,202,303\}}
\left[
accuracy(C,s)-accuracy(RPT\_BASE,s)
\right].
\]

For nominal relational winner \(W\):

```text
capacity_control_reproduces_gain =
    max(
        mean_delta(RPT_BASE_WIDE_MAX),
        mean_delta(RPT_FULL_ZERO_REL)
    )
    >= mean_delta(W)
```

This comparison uses canonical float64 aggregate values without a hidden
tolerance.  It is evaluated only as an explanation of a positive
\(\overline{\Delta}(W)\); a nonpositive winner already fails the positive-gain
criterion.

A positive architecture result requires:

1. a relation model beats the matched baseline in mean three-seed
   `val_select` accuracy;
2. `capacity_control_reproduces_gain == false`;
3. the locked model improves on the sealed 500,000-event final test;
4. the paired confidence interval and seed spread are reported;
5. inference remains HLT-only;
6. all model and data lineage is authenticated.

A particularly strong result is:

```text
mean matched-seed improvement >= +0.003 absolute accuracy
all three seeds beat their matched baseline
sealed final-test improvement is positive
capacity controls do not explain the gain
```

The stretch goal is a stable improvement of at least `+0.005` absolute
accuracy from richer pair relations alone.  If achieved, the next campaign
may combine the winning Relational ParT with the independently successful
residual particle-feature adapter and, separately, test transfer to the
realistic HLT-v2 profile.
