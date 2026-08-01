# Relation-Expert Token Bridge Campaign

Status: implementation-ready scientific and engineering plan for a new
offline-to-HLT Particle Transformer campaign.

Short name:

```text
RETB
```

The campaign asks whether several independently strong Particle Transformer
experts, each using a different particle-pair attention bias, can expose
complementary offline reasoning as small canonical token banks; whether HLT
jets can predict those offline token banks without constituent matching; and
whether a final token transformer can use the reconstructed offline evidence
plus native HLT evidence to outperform a matched HLT Particle Transformer.

The central object is not a reconstructed particle and is not an attributed
human-readable chain of reasoning.  It is a compact, task-relevant,
relation-specialized representation:

```text
complete offline particles
  -> relation-specialized ParT expert
  -> K x D canonical summary-token bank
  -> expert classifier

complete HLT particles
  -> HLT relation experts and native context
  -> HLT-to-offline token predictor
  -> K x D predicted/refined offline summary-token banks
  -> faithful frozen consumers and/or unrestricted deployable fusion
```

Pure-offline token coordinates remain the clean reference.  Bridge-aware
offline targets, bridge-aware HLT encoders, end-to-end fine-tuning, and an
unrestricted final fusion are separately named performance candidates rather
than silent changes to that reference.

Only jet-level offline/HLT identity pairing is used.  No offline constituent
index, nearest-neighbor match, source index, or particle assignment is exposed
to a predictor or deployable classifier.

---

## 1. Scientific motivation

The completed relational Particle Transformer campaign provides six
physically motivated relation families in addition to Weaver's standard four
pair features:

- directional transverse momentum (`PT`);
- track and displacement compatibility (`TRACK`);
- particle identity (`PID`);
- charge structure (`CHARGE`);
- multiscale local density (`DENSITY`);
- beam-free angular-tree regions (`REGION`).

A single model that concatenates every relation family into one pair tensor is
useful, but it does not guarantee that each relation obtains a clean
attention allocation.  Large or correlated relation families may dominate,
cancel, saturate, or simply be ignored by the shared pair encoder.  Even when
the combined model improves, it can be hard to establish whether distinct
relation families learned complementary event evidence.

RETB isolates the attention distributions rather than isolating the particle
fields.  Every expert receives the complete 17-channel Particle Transformer
particle input.  The experts differ in the relation supplied to attention:

```text
E_BASE4    standard Weaver four-feature pair relation
E_PT       base4 + PT
E_TRACK    base4 + TRACK
E_PID      base4 + PID
E_CHARGE   base4 + CHARGE
E_DENSITY  base4 + DENSITY
E_REGION   base4 + REGION
```

Thus a `PT`-routed expert may still read displacement, identity, charge, and
energy from the particles whose information it gathers.  A `TRACK`-routed
expert may still use the momentum and identity of compatible displaced
particles.  Particle fields remain colocated inside particle tokens, while
the routing prior is relation-specific.

Each expert must classify through a small learned-query token bottleneck.
The offline token banks become fixed jet-level targets.  HLT predictors infer
those targets from the HLT constituent set and HLT expert/context banks.  This
retains particle attention on both sides without requiring the two particle
sets to have equal lengths or one-to-one correspondence.

Relevant architectural precedents are:

- Particle Transformer: <https://arxiv.org/abs/2202.03772>
- Set Transformer learned attention pooling:
  <https://arxiv.org/abs/1810.00825>
- Perceiver latent bottlenecks: <https://arxiv.org/abs/2103.03206>
- joint-embedding representation prediction:
  <https://arxiv.org/abs/2301.08243>

These precedents motivate the components.  They do not establish the RETB
hypothesis; the campaign is designed to test it.

---

## 2. Locked scientific questions

### Q1: are relation-specialized offline experts individually strong?

Can every relation expert classify competitively when its prediction must pass
only through a registered \((K,D)\) bank of canonical summary tokens?

### Q2: are the experts complementary?

Does frozen-token fusion improve over:

- the best single expert;
- a logit average;
- an equal-compute ensemble of unbiased experts;
- a parameter/FLOP-matched standard ParT;
- one ParT receiving all relations in one shared bias path?

### Q3: how much token capacity is necessary?

How do slot count \(K\), slot dimension \(D\), equal-scalar alternatives, and
heterogeneous expert budgets affect individual expert strength, oracle offline
fusion, token stability, HLT predictability, final HLT performance, memory,
and latency?

### Q4: can HLT jets predict offline expert tokens without particle matching?

Can a predictor reproduce the offline token bank well enough that the frozen
offline expert head and frozen offline fusion preserve their oracle decisions?

### Q5: which predictor architecture and context are useful?

Do token translators benefit from:

- only the corresponding HLT expert tokens;
- corresponding tokens plus native HLT particle context;
- every HLT expert bank plus native HLT context?

Does a slot-query transformer decoder outperform affine, residual-MLP, and
token-self-attention controls?

### Q6: does token reconstruction add value beyond a native HLT multi-expert model?

Does the reconstructed-token path beat:

- the standard HLT ParT;
- a capacity-matched HLT ParT;
- native HLT relation-expert fusion;
- native HLT tokens passed through a dimensional adapter without an offline
  reconstruction objective?

### Q7: does native HLT evidence add residual value?

After offline token reconstruction, does a constrained native-HLT residual
branch improve classification without merely bypassing the reconstructed
path?

### Q8: where is the remaining information loss?

Which expert banks are easy or difficult to predict, and how much final
performance is recovered when individual predicted banks are replaced by
their oracle offline counterparts?

---

## 3. Non-goals and language discipline

The first campaign will not:

- reconstruct offline particles;
- infer or persist particle-to-particle matches;
- use a Hungarian or nearest-neighbor constituent loss;
- claim that latent tokens are literal human thoughts;
- force every token to have a named physical meaning;
- assume that low token MSE implies downstream utility;
- claim detector-realistic HLT simulation without paired detector evidence;
- tune any architecture on `final_test`;
- abort future scientific arms merely because an earlier arm performs poorly;
- treat a larger multi-expert model as evidence for relation specialization
  without capacity and ensemble controls;
- force expert diversity with an orthogonality penalty in the primary model;
- use expert dropout as the primary anti-redundancy mechanism.

The report may use "reasoning token" informally, but contracts and artifact
schemas use `summary_token`, `expert_token`, or `predicted_offline_token`.

---

## 4. Canonical terminology and model graph

### Offline objects

`O_BASE`

: Exact standard offline base-size Particle Transformer using the repository's
  canonical 17 particle features and Weaver standard-four pair path.  This is
  the primary ordinary offline baseline.

`O_WIDE`

: Existing RPT-style widened pair-encoder control.  This controls incremental
  pair-path capacity but is not claimed to match the complete seven-expert
  system.

`O_MONO_PARAM`

: Monolithic standard-four offline ParT selected from a locked depth/width
  grid to match the complete deployed offline expert-token graph's parameter
  count, including expert encoders, tokenizers, projections, refiner when
  active, and final consumer.

`O_MONO_FLOP`

: Monolithic standard-four offline ParT selected from the same grid to match
  the complete graph's analytical inference FLOPs.

`O_BASE_LONG`

: Ordinary `O_BASE` architecture trained with the same total labeled-example
  presentations as the complete expert-plus-fusion training program.

`O_FULLREL`

: One offline ParT using the existing `RPT_FULL_ALL` shared relation-bias path.

`OE_e_Ss`

: General offline expert notation when token shape \(S=s=(K,D)\) varies in
  both slot count and slot dimension.  Expert \(e\) is one of `BASE4`, `PT`,
  `TRACK`, `PID`, `CHARGE`, `DENSITY`, or `REGION`.  `OE_e_Kk` is shorthand
  only for a `D=128` row.

`OF_Ss`

: Offline fusion transformer consuming the frozen token banks from all seven
  `OE_e_Ss` experts.  `OF_Kk` is shorthand only for a uniform `D=128` row.

`OE_BRIDGE_e_Ss`

: Bridge-aware offline expert candidate.  Its forward pass still receives only
  offline particles, but its training objective includes the predictability of
  its token bank from paired HLT evidence.

`OE_PROJECT_e_Ss`

: Frozen pure-offline expert followed by a small learned bridge projection and
  a retrained offline consumer.  The projection, rather than the original
  offline token coordinates, is optimized for HLT predictability.

### HLT objects

`H_BASE`

: Exact standard base-size ParT trained from scratch on the campaign HLT view.

`H_WIDE`

: Existing RPT-style widened pair-encoder HLT control.

`H_MONO_PARAM`, `H_MONO_FLOP`, and `H_BASE_LONG`

: HLT analogues of the complete deployed HLT graph's capacity, inference
  compute, and label-exposure controls.  The matched graph includes every HLT
  encoder, predictor, projection, uncertainty head, refiner, and final
  consumer actually used at inference.

`HE_e_Ss`

: Native HLT relation expert with the same relation identity, particle
  architecture, summary-token shape, and classifier topology as
  `OE_e_Ss`, trained on HLT labels rather than offline targets.

Every selected shape has four registered HLT encoder modes:

```text
HE_SCRATCH_CE       random initialization, HLT CE only
HE_OFFLINE_INIT     corresponding offline particle encoder initialization
HE_DUAL_OBJECTIVE  HLT CE plus offline expert-logit/token alignment
HE_BRIDGE_TUNED     selected HLT evidence encoder partially unfrozen with predictor
```

`HF_NATIVE_Ss`

: Native HLT expert-token fusion with no offline token reconstruction.
  `HF_NATIVE_Kk` is shorthand only for a uniform `D=128` row.

`P_e_Ss_Aa_Cc`

: Predictor for offline expert \(e\), token shape \(S=s\), predictor
  architecture \(A=a\), and context policy \(C=c\).

`PF_FROZEN_Ss`

: Frozen offline fusion `OF_Ss` evaluated with predicted offline token banks.

`HF_ADAPTER_Ss`

: Deployable HLT fusion adapter consuming predicted offline banks, their
  uncertainty, and optional native HLT tokens.

`HF_UNRESTRICTED_Ss`

: Maximum-performance transformer classifier over predicted offline and native
  HLT token banks.  It is not constrained to be a residual correction to
  frozen offline logits.

`TR_REFINE_Ss`

: Native-HLT-conditioned token refiner placed after the first predicted
  offline token banks and before a frozen or trainable final fusion.

### Oracle and hybrid objects

`Z_e_off`

: Frozen offline expert token bank for expert \(e\), shape `[B,K,D]`.

`Z_e_hlt`

: Native HLT expert token bank, shape `[B,K,D]`.

`Zhat_e_off`

: Predicted offline token bank, shape `[B,K,D]`.

`HYBRID_e`

: Frozen offline fusion using `Zhat_e_off` for expert \(e\) and oracle
  `Z_off` for every other expert.

No deployable model may consume `Z_e_off`, offline particles, or any artifact
derived from them at inference.

---

## 5. Data and split contract

The production campaign uses balanced 10-class JetClass identities:

| Logical role | Repository split | Count | Per-class count | Permitted use |
|---|---|---:|---:|---|
| `train` | `model_train` | 500,000 | 50,000 | All model-weight training and train-only fitting |
| `val_stop` | first deterministic half of `model_val` | 50,000 | 5,000 | Epoch/checkpoint selection only |
| `val_design` | second deterministic half of `model_val` | 50,000 | 5,000 | Calibration, certification, component and beam selection |
| unused | `stack_train` | 0 | 0 | Reserved by the five-split manifest |
| `final_select` | `stack_val` | 50,000 | 5,000 | Complete-graph comparison and two-finalist selection only |
| `final_test` | `final_test` | 300,000 | 30,000 | Sealed final evaluation |

The user's requested 150,000 validation jets are intentionally divided into
50,000 checkpoint-selection jets, 50,000 design-selection jets, and 50,000
complete-graph finalist-selection jets.

Within each class of the repository `model_val`, sort identities by:

```text
sha256("retb_model_val_partition_v1" || canonical_jet_identity)
then canonical_jet_identity
```

The first 5,000 identities become `val_stop` and the remaining 5,000 become
`val_design`.  The partition manifest is immutable and source-bound.
`val_stop` may select epochs but no architecture, threshold, calibration, or
certification.  `val_design` may fit the explicitly label-free uncertainty
calibrator and select/certify components, but never selects an epoch.
`final_select` may compare only already complete, locked deployable graphs and
choose the two finalists; it may not alter a component inside a graph.

Campaign bootstrap also seals an identity-sorted
`final_select_label_manifest.json.gz` containing only canonical identity and
int64 class label, with the complete split manifest as its sole parent.
Stage-N selection inference cannot load it; only the immutable selector may
join it to label-free prediction shards.

Every split is sampled without replacement.  Source files, event identities,
and canonical jet identities are disjoint across splits.  The offline and HLT
versions of one jet always have the same split and ordered identity.

### Predeclared 3M scale-up pool

Architecture discovery remains the requested 500,000-jet campaign.  A
performance-ceiling stage is nevertheless fixed now:

```text
scale_train = 3,000,000 balanced training jets
```

`scale_train` contains the exact 500,000 `model_train` identities plus
2,500,000 additional balanced identities.  Added identities are disjoint from
`val_stop`, `val_design`, `stack_val`, and `final_test`.  Its manifest is built
and sealed at campaign bootstrap, before model results.

Only graph definitions in
`selection/locked_scale_shortlist.json` may train on `scale_train`.  The
scale-up may select epochs on `val_stop` but may not reopen
architecture, token-shape, loss, degradation, or predictor-family selection.
The 500k shortlist definitions and all shortlisted 3M checkpoints are
immutable before `stack_val`; the selected 3M finalists and their execution
dependencies are separately locked before the common `final_test` is opened.

Architecture-search training workers may load only `model_train` and
`val_stop`; Stage-M workers may load only `scale_train` and `val_stop`.
Design/calibration workers may additionally load `val_design`.
Only the Stage-N selection workflow may load `final_select/stack_val` before
finalist locking: its inference worker reads features but not labels, and its
selector reads the separately authenticated identity-to-label manifest but not
particle features.  After that lock, Stage-N diagnostic workers may read
`stack_val` only with `selection_eligible=false`.
Pre-lock final-test preparation may construct only identity-bound raw/degraded
inputs and deterministic sidecars without loading checkpoints.  Offline
targets and oracle outputs require the immutable scale-finalist lock.  The
sole pre-lock model-output exception is the label-free deployable `stack_val`
selection inference defined in Sections 15, 23, 25, and 29.  No pre-lock
`final_test` model output is permitted; scientific final-test inference
additionally requires the execution lock.

All primary models use:

```text
max_constituents = 128
raw token dimension = 14
Particle Transformer feature dimension = 17
number of classes = 10
```

The 14 raw columns are repository-canonical:

| Index | Raw field |
|---:|---|
| 0 | `pt` |
| 1 | `eta` |
| 2 | `phi` |
| 3 | `energy` |
| 4 | `charge` |
| 5:10 | five PID indicators |
| 10 | `d0` |
| 11 | `d0err` |
| 12 | `dz` |
| 13 | `dzerr` |

The 17 particle features and four-vectors must be produced only through
`jetclass_fresh.part_inputs`.  No campaign worker reimplements their
transforms.

### Jet-level pairing without constituent matching

The paired training dataset yields:

```text
offline raw tokens and mask
HLT raw tokens and mask
class label
canonical jet identity
```

It never yields:

```text
offline source constituent index
HLT source constituent index
merge parent
nearest offline constituent
nearest HLT constituent
constituent-level matching score
```

The HLT cache generator may internally transform an offline jet to construct a
controlled proxy, but all construction indices are discarded.  Cached HLT
arrays are sorted by HLT \(p_T\), authenticated, and are the only HLT-side
particle arrays available to model code.

### Training-realization contract

One stochastic proxy realization per training jet is not the production
bridge domain.  For `model_train`, create four deterministic realizations:

```text
replica_id = 0,1,2,3
```

The canonical training loader presents every jet exactly once per epoch and
chooses:

\[
r(e,j)=(e+h(j))\bmod 4,
\]

where \(e\) is the zero-based epoch and \(h(j)\) is the low two bits of
`sha256("retb_replica_cycle_v1" || canonical_jet_identity)`.  Thus each jet
cycles through all four realizations without multiplying label exposure.
Resume at an epoch boundary reproduces the same choices exactly.

`val_stop`, `val_design`, `stack_val`, and `final_test` each use only
`replica_id=0`.
Checkpoint and architecture comparisons therefore remain paired to one fixed
evaluation domain.

Two training-domain arms are registered:

```text
R_FIXED    replica 0 at every epoch
R_MULTI    four nominal-parameter random realizations, canonical primary
R_RANDOM   four fixed domain-randomized parameter realizations
```

`R_RANDOM` uses the following immutable multipliers:

| Replica | Kinematic severity | Track-loss severity | Track-core noise | Tail probability |
|---:|---:|---:|---:|---:|
| 0 | 1.00 | 1.00 | 1.00 | 1.00 |
| 1 | 0.80 | 1.20 | 1.10 | 0.75 |
| 2 | 1.20 | 0.80 | 0.90 | 1.25 |
| 3 | 1.00 | 1.00 | 1.20 | 1.50 |

The multipliers alter only the declared corruption family and are serialized
in each cache/stream manifest.  Labels and offline targets are identical
across replicas.

Replicas may be generated on demand or persisted in authenticated shards.
The production choice is made from measured CPU, storage, and I/O projections,
not classifier performance.  A loader must produce identical bytes either
way.

---

## 6. Track-dominant HLT degradation contract

The existing repository profile `fixed_hlt_v2_realistic/v1` primarily changes
constituent efficiency and kinematics and leaves raw charge, PID, and track
columns largely copied from their offline source.  RETB therefore requires a
new, non-interchangeable profile:

```text
profile_name = fixed_hlt_v3_track_dominant_proxy
profile_version = v1
nominal_strength = 1.0
```

This is explicitly an HLT-like controlled proxy unless it is later calibrated
against authenticated paired offline/HLT detector data.  Reports must not
rename it "real HLT."

### 6.1 Exact identity at strength zero

At `strength=0.0`:

```text
tokens_hlt == tokens_offline bitwise
mask_hlt   == mask_offline bitwise
ordering   == offline ordering
```

The implementation must short-circuit before random-number generation.

### 6.2 Deterministic random stream

Every event seed is derived from:

```text
sha256(
  "retb_hlt_v3_rng_v1" ||
  domain_seed ||
  replica_id ||
  canonical_jet_identity
)
```

Every corruption family owns a fixed substream ID.  Adding a diagnostic or a
later corruption family must not shift existing random draws.  Randomness may
not depend on batch size, worker count, shard boundaries, or array traversal
outside the canonical constituent order.

Domain seeds are:

```text
model_train = 3053
scale_train = 3053
val_stop    = 3054
val_design  = 3054
stack_val   = 3056
final_test  = 3057
```

`model_train` and `scale_train` intentionally share the `train_domain` seed
namespace.  For any of the original 500,000 identities, every HLT replica
must therefore be byte-identical in both manifests.  Scale-up changes the
identity population, not the degradation semantics of an existing identity.
`val_stop` and `val_design` intentionally share the validation-domain seed;
their disjoint identities, not a changed corruption definition, distinguish
the two roles.

### 6.3 Primitive-operation order

For every jet:

1. validate the offline raw-token schema;
2. apply the mild constituent threshold;
3. apply type-aware neutral local merging;
4. apply constituent efficiency loss;
5. apply mild kinematic response and local reassignment;
6. apply PID confusion;
7. enforce charge/PID consistency;
8. apply track-measurement loss;
9. smear surviving track measurements and reported uncertainties;
10. apply rare charge flips to eligible charged particles;
11. recompute energy while preserving each surviving constituent's declared
    mass hypothesis;
12. zero invalid/padded rows;
13. stable-sort by descending degraded \(p_T\), breaking exact ties by the
    pre-sort canonical index;
14. emit diagnostics and hashes.

Derived Particle Transformer features, jet axes, significances, ranks,
pairwise relations, density features, and REGION trees are rebuilt from the
degraded output.  They are never independently smeared.

### 6.4 Type-aware constituent and energy rules

The v3 implementation must not reuse
`merge_tokens_copy_dominant()` for arbitrary particle pairs.

Charged-domain particles (`charged_hadron`, `electron`, `muon`) are never
merged.  Charged-neutral pairs are never merged.  Neutral merging is eligible
only when both particles have the same PID category:

```text
neutral_hadron + neutral_hadron
photon + photon
```

For an eligible merge:

1. add the two Cartesian four-vectors in float64;
2. derive merged \(p_T,\eta,\phi,E\) from the summed four-vector;
3. retain the common one-hot PID;
4. set charge to zero;
5. set all four track fields to the explicit invalid-measurement sentinel;
6. record both source categories and the merge in diagnostics;
7. convert to float32 only after finite/domain checks.

Different neutral categories remain separate.  Exact tie order is canonical.

For a nonmerged surviving constituent, calculate the nonnegative source mass:

\[
m_{\mathrm{src}}^2
=
\max(E_{\mathrm{src}}^2-\|\vec p_{\mathrm{src}}\|^2,0)
\]

in float64 before corruption.  After \(p_T,\eta,\phi\) response, use:

\[
E_{\mathrm{new}}
=
\sqrt{
(p_T^{\mathrm{new}}\cosh\eta^{\mathrm{new}})^2
+m_{\mathrm{src}}^2
}.
\]

This preserves the source constituent mass instead of silently forcing every
particle massless.  A future PID-dependent mass-hypothesis profile requires a
new degradation version.

Fake, duplicate, and split constituents are disabled in v1.  They may be added
only under a new profile version backed by an authenticated response-evidence
artifact.  The absence of those mechanisms must be stated in reports.

### 6.5 Nominal mild kinematic component

At strength one, use the v2 mechanisms with a deliberately reduced kinematic
severity:

```text
hlt_pt_threshold             = 0.10
merge_radius                 = 0.0015
merge_probability            = 0.15
eff_plateau_barrel           = 0.9995
eff_plateau_endcap           = 0.9980
eff_turnon_pt_barrel         = 0.10
eff_turnon_pt_endcap         = 0.20
eff_width_pt_barrel          = 0.05
eff_width_pt_endcap          = 0.07
density_loss_scale           = 0.005
jet_quality_sigma            = 0.010
kinematic_smear_scale        = 0.080
kinematic_tail_base          = 0.0005
kinematic_tail_eta           = 0.0005
kinematic_tail_density       = 0.0005
local_reassign_scale         = 0.050
```

All kinematic terms scale continuously to zero with degradation strength.

The common parameters above are reference amplitudes, not a universal
particle response.  Apply the following immutable type multipliers:

| Current PID domain | Constituent-loss amplitude | Momentum response | Angular response | Local reassignment |
|---|---:|---:|---:|---:|
| charged hadron | 0.50 | 0.45 | 0.35 | 0.00 |
| electron | 0.60 | 0.55 | 0.45 | 0.00 |
| muon | 0.40 | 0.35 | 0.30 | 0.00 |
| photon | 0.90 | 0.85 | 0.90 | 1.00 |
| neutral hadron | 1.35 | 1.30 | 1.25 | 1.50 |
| unknown/zero-hot | 1.00 | 1.00 | 1.00 | 1.00 |

Constituent-loss probability means the loss term above the exact
strength-zero identity path; multiplying it cannot alter schema validation.
Charged particles receive track-like momentum/angular response and are never
locally reassigned.  Local reassignment is a calorimeter-like operation
applied only to photon, neutral-hadron, and unknown domains.  Track
measurement loss remains the distinct Section-6.6 mechanism and does not
remove the constituent.  Response diagnostics are reported separately for
all six domains so this proxy conditioning cannot be hidden by an inclusive
average.

The multiplier mapping is exact.  Let `a_loss`, `a_p`, `a_ang`, and
`a_reassign` be the four type-table values; let `s` be degradation strength;
and let `r_kin`, `r_track_loss`, `r_track_core`, and `r_tail` be the selected
`R_RANDOM` row, or one for `R_FIXED/R_MULTI`.  The inherited v2 helper first
computes its unscaled base terms from the original current-view particle and
pre-corruption local density.  RETB then uses:

| Mechanism | Exact RETB mapping |
|---|---|
| Constituent-efficiency loss above identity | `p_loss = clip01(p_loss_v2 * a_loss * s * r_kin)` |
| Core log-\(p_T\)/momentum sigma | `sigma_p = sigma_p_v2 * a_p * s * r_kin` |
| Kinematic-tail probability | `p_kin_tail = clip01(p_kin_tail_v2 * a_p * s * r_tail)` |
| Kinematic-tail residual magnitude | `delta_kin_tail = delta_kin_tail_v2 * a_p * s * r_kin` |
| Core \(\eta,\phi\) sigma | `sigma_ang = sigma_ang_v2 * a_ang * s * r_kin` |
| Reassignment probability | `p_reassign = clip01(p_reassign_v2 * a_reassign * s * r_kin)` |
| Reassignment displacement magnitude | `delta_reassign = delta_reassign_v2 * a_reassign * s * r_kin` |
| Section-6.6 track-measurement loss | `p_track_loss = clip01(p_track_loss_v3 * s * r_track_loss)` |
| Section-6.7 track core residual | `delta_track_core = delta_track_core_v3 * s * r_track_core` |
| Section-6.7 track-tail probability | `p_track_tail = clip01(p_track_tail_v3 * s * r_tail)` |
| Section-6.7 track-tail residual magnitude | `delta_track_tail = delta_track_tail_v3 * s * r_track_core` |

Here `p_loss_v2` contains exactly the inherited plateau, turn-on, endcap, and
`density_loss_scale` inefficiency terms.  `sigma_p_v2` contains the inherited
`jet_quality_sigma` and `kinematic_smear_scale` momentum terms.
`p_kin_tail_v2` contains exactly `kinematic_tail_base`,
`kinematic_tail_eta`, and `kinematic_tail_density`.  The inherited angular and
reassignment helpers supply the remaining named base terms; no multiplier is
applied to a mechanism not listed in the table.

The campaign spec binds the source/function hash that emits each named
`*_v2` base term.  Workers must call those exact base-term helpers and then
apply this table; they may not independently recreate the formula.  If the
current v2 implementation does not expose a base term separately, it is first
refactored behind a parity-tested interface with unchanged v2 output and a new
RETB adapter contract.

Multiplication order is base term, type multiplier, strength, replica-family
multiplier, then final clipping.  Probabilities clip to `[0,1]`; nonnegative
sigmas/magnitudes use the inherited v2 physical cap after all multipliers.
Random draws occur only after clipping.  Finally wrap \(\phi\), validate
finite \(p_T,\eta,\phi\), preserve or merge mass under Section 6.4, and
recompute energy from the degraded momentum and that mass.  There is no
independent energy-smearing draw.

### 6.6 Track-measurement availability

Track eligibility uses the existing RPT rule:

```text
valid particle
and charged PID in {charged_hadron, electron, muon}
and finite d0, d0err, dz, dzerr
and d0err > 0
and dzerr > 0
```

For an eligible track, define:

\[
p_{\mathrm{loss}} =
\operatorname{clip}\left[
0.030
+0.030 I(|\eta|\geq 1.5)
+0.080\,\sigma\left(\frac{0.80-p_T}{0.25}\right)
+0.020\min\left(\frac{\rho_{0.04}}{8},1\right),
0,0.35
\right],
\]

where \(\rho_{0.04}\) is the number of other valid constituents within
\(\Delta R<0.04\).  At strength \(s\), use
`clip(s * p_loss, 0, 1)`.

When the measurement is lost, keep the constituent but set:

```text
d0 = 0
d0err = 0
dz = 0
dzerr = 0
```

This is the explicit v3 invalid-measurement sentinel.  The v3 raw-input schema
must declare it; it must not be inferred from observed zeros.

### 6.7 Surviving-track response

For surviving tracks, reported uncertainties receive independent log-normal
scale factors whose medians are:

```text
median d0err scale = 1.35
median dzerr scale = 1.30
log-scale sigma    = 0.15
```

Let the original reported errors be \(\sigma_{d0}\) and \(\sigma_{dz}\).
Draw correlated standard-normal core residuals with:

```text
correlation(d0,dz) = 0.25
additional d0 noise scale = 0.75 * sigma_d0
additional dz noise scale = 0.65 * sigma_dz
```

The tail probability is:

\[
p_{\mathrm{tail}} =
\operatorname{clip}\left[
0.010
+0.005 I(|\eta|\geq1.5)
+0.002\min(\rho_{0.04},5),
0,0.08
\right].
\]

Tail residuals multiply both additional noise scales by `4.0`, preserving the
drawn sign.  Strength scales the additional variance, log-error displacement
from one, and tail probability to zero.  Outputs remain finite.  Positive
reported uncertainties are never clipped to zero.

### 6.8 PID and charge response

PID remains exactly one-hot after corruption.  At strength one, the
within-domain confusion probabilities are:

```text
charged_hadron -> electron       0.002
charged_hadron -> muon           0.002
electron       -> charged_hadron 0.010
muon           -> charged_hadron 0.010
electron       -> muon           0.001
muon           -> electron       0.001
neutral_hadron -> photon         0.010
photon         -> neutral_hadron 0.010
```

No nominal neutral-to-charged or charged-to-neutral PID transition is allowed
in v1 because the raw schema does not provide enough evidence to create a
consistent new track.  PID transition probabilities scale linearly with
strength.

Neutral PID forces charge to zero and invalidates all track measurements.
Charged-domain transitions retain the original charge.  Eligible charged
particles receive a charge-flip probability:

\[
p_{\mathrm{flip}} =
\operatorname{clip}\left[
0.002+0.002I(|\eta|\geq1.5)
+0.001\min(p_T/100,1),
0,0.01
\right].
\]

Charge flips negate only `-1` or `+1`; zero charge is not invented.

### 6.9 Strength and field-isolation profiles

The production registry includes:

```text
D_OFFLINE_IDENTITY  strength 0.0, all corruption families
D_KIN_ONLY          strength 1.0, kinematics/efficiency only
D_TRACK_ONLY        strength 1.0, track availability/response only
D_MISSING_ONLY      strength 1.0, constituent and track availability only
D_NOMINAL           strength 1.0, complete v3 profile
D_MILD              strength 0.5, complete v3 profile
D_SEVERE            strength 1.5, complete v3 profile
D_LEGACY_V1         fixed_hlt_v1 strength 0.6 comparison only
D_LEGACY_V2         fixed_hlt_v2_realistic strength 1.0 comparison only
```

`D_NOMINAL` is the sole primary training domain.  The other profiles are
controls or robustness evaluations.  Model selection may not choose a
degradation profile after viewing classifier performance.

### 6.10 Degradation validation

Before model training, the campaign writes an immutable degradation audit
covering:

- exact identity at strength zero;
- per-field changed/missing fractions;
- finite values and valid domains;
- constituent and valid-track count distributions;
- PID transition matrix;
- charge transition matrix;
- \(p_T,\eta,\phi,E,d0,dz,d0err,dzerr\) response quantiles;
- correlations of \(d0/dz\) residuals;
- response binned by \(p_T,\eta\), PID, multiplicity, and local density;
- before/after distributions and clipping fractions for all 17 transformed
  `pf_features` channels;
- before/after distributions for the standard four and every enabled RETB
  relation family, including explicit track-validity states;
- the fraction of raw `d0err/dzerr` changes erased or saturated by the
  canonical `[0,1]` transformed-feature clipping;
- monotonicity across strengths `0.0, 0.5, 1.0, 1.5`;
- absence of class-label input to the generator;
- exact cache reproducibility across batch and shard layouts.

If authenticated paired genuine HLT/offline response evidence becomes
available, it must be stored as a new parent artifact and requires a new
profile version.  Old proxy and empirically calibrated caches must never look
interchangeable.

---

## 7. Inherited Particle Transformer and relation contracts

RETB reuses, rather than reimplements:

- `jetclass_fresh.part_inputs`;
- `teacher_logit_reco.relational_part` standard-four `uu` path;
- the authoritative real-Weaver FP32 parity contract;
- relation encoders and normalization contracts for `PT`, `TRACK`, `PID`,
  `CHARGE`, `DENSITY`, and `REGION`;
- canonical relation-family ordering;
- compiled REGION tree backend and authenticated sidecars;
- exact mask, category, track-validity, and numerical policies;
- deterministic metric definitions from the relational campaign.

RETB creates new campaign, model, token, predictor, and degradation contract
versions.  An RETB checkpoint cannot be registered as an RPT checkpoint even
when its particle encoder begins from an RPT-compatible topology.

Every expert always includes `base4`.  Its additional relation is isolated:

| Expert | Pair input |
|---|---|
| `BASE4` | `base4` |
| `PT` | `base4 + PT` |
| `TRACK` | `base4 + TRACK` |
| `PID` | `base4 + PID` |
| `CHARGE` | `base4 + CHARGE` |
| `DENSITY` | `base4 + DENSITY` |
| `REGION` | `base4 + REGION` |

No primary expert receives two new relation families.  Multi-relation shared
biases are controls.

The six-category PID relation is inherited exactly:

```text
charged_hadron
neutral_hadron
photon
electron
muon
unknown
```

The raw schema contains five one-hot indicators.  A valid zero-hot particle
maps explicitly to `unknown`; multi-hot input fails preflight.

### Normalizer population and scale-up lineage

RETB retains the inherited estimator, valid-pair masks, clipping rules,
accumulation dtype, and channel order, but freezes the fitted populations:

```text
offline relation/REGION normalizer:
  offline model_train identities, one offline view per identity

shared HLT relation/REGION normalizer:
  model_train identities x nominal R_MULTI replicas {0,1,2,3}
  every identity-replica contributes with equal weight
```

The shared HLT normalizer is used unchanged by `R_FIXED`, `R_MULTI`, and
`R_RANDOM`, and by all fixed-severity robustness inputs.  `R_RANDOM` never
fits a severity-specific normalizer.  This makes realization-policy
comparisons change the degraded input rather than both the input and its
preprocessing.  Offline and HLT normalizers remain separate because their
fitted populations are different.

The pool is traversed in canonical `(jet_identity,replica_id)` order using the
inherited deterministic streaming estimator.  REGION statistics use the
REGION descriptors rebuilt from that exact view.  Relation statistics use
only valid ordered pairs under the inherited pair mask.  No validation,
`stack_val`, or final-test identity contributes to a fitted statistic.

At Stage M, refit rather than reuse every train-derived statistic:

```text
offline relation/REGION = offline scale_train
shared HLT relation/REGION = scale_train x nominal R_MULTI replicas {0,1,2,3}
target-token normalizers = new scale_train offline targets
any other fitted input standardizer = its corresponding locked scale_train view
```

Estimator code, clipping thresholds, dtype, masks, and replica weighting are
identical to the 500k recipe.  The new population legitimately produces new
content hashes.  The Stage-M uncertainty calibrator is also refitted
label-free on `val_design` after its new predictor checkpoint is selected on
`val_stop`.

Every normalizer artifact records its logical domain, exact identity-manifest
hash, replica IDs and weights, estimator/clip contract, valid-entry counts by
channel, parent schema/source hashes, fitted values, and content hash.
Checkpoint and graph reuse requires an exact normalizer parent; a 500k
normalizer cannot masquerade as a scale-up normalizer.

### Measurement-availability embedding

The ordinary 17-channel `O_BASE/H_BASE` baselines remain unchanged.  RETB also
registers `V_MEASUREMENT_EMBED`, which derives a three-state particle category
from the current view only:

```text
not_track_domain
track_measurement_available
track_measurement_missing
```

A learned 128-dimensional state embedding is added after the initial particle
embedding and before the first particle-attention block.  It does not alter
the canonical 17 input channels.  Offline experts derive the state from
offline fields; deployable HLT experts and predictors derive it from HLT
fields.  `TRACK` experts already expose `track_valid` through their relation,
so this embedding is evaluated both with and without `TRACK` to distinguish
token-level availability from pair-level validity.

`V_MEASUREMENT_EMBED` is a registered architecture candidate, not a silent
change to exact ParT controls.

### Relation-bias topology

The inherited RPT path concatenates `base4` and encoded relation channels,
then sends the concatenation through one nonlinear pair stem.  RETB calls this:

```text
B_CONCAT
```

It isolates experts from one another but does not mathematically isolate the
additional relation from `base4` inside an expert.

RETB therefore adds a true dual-path topology.  For expert \(e\), layer
\(\ell\), and head \(h\):

\[
B_{e,\ell,h}(i,j)
=
B_{\mathrm{base4},\ell,h}(i,j)
+\alpha_{e,\ell,h}
B_{\mathrm{relation},e,\ell,h}(i,j).
\]

The two terms own separate pair encoders and final per-layer/head projections.
They are added only after each has emitted attention logits.  The relation
cannot be suppressed merely by rescaling its raw inputs before a shared
nonlinearity.

Registered dual paths are:

```text
B_DUAL_FIXED   alpha[e,l,h] = 1
B_DUAL_GATED   alpha[e,l,h] = 2 * sigmoid(a[e,l,h]), initialized to 1
```

Gates are bounded in `(0,2)`, recorded per layer/head, and receive ordinary
gradients.  A `BASE4` dual-path capacity control supplies a second independently
parameterized base4 path in place of the additional relation.  A zero-relation
shape control retains every dual-path parameter but provides exactly zero
relation logits.

`B_CONCAT`, `B_DUAL_FIXED`, and `B_DUAL_GATED` are all scientific candidates
for the carried token shapes.  Claims of a clean relation-specific bias use
only a dual-path result; `B_CONCAT` remains the inherited efficiency/control
topology.

#### Exact layerwise Weaver interface

The dual path uses campaign-owned contract:

```text
retb_layerwise_pair_bias_v1
```

Each base4 and relation stem is shared across particle-attention layers and
emits one latent pair tensor:

```text
base latent      [B,C_base,N,N]
relation latent  [B,C_relation,N,N]
```

Every particle layer owns separate final linear/`1x1` projections from those
latents to its `H` attention heads.  A
`LayerwisePairBiasProvider.bias_for_layer(layer_index)` call returns exactly:

```text
[B,H,N,N]
```

for that layer, after the base/relation sum and gate, before the ordinary
attention mask is applied.  The explicit RETB Weaver wrapper passes the
provider to each particle block; no layer may silently reuse another layer's
projected bias.  The ordinary tensor-valued Weaver `uu` interface remains
unchanged for `O_BASE`, `H_BASE`, and `B_CONCAT`.

The implementation must not materialize `[B,L,H,N,N]`.  It retains the two
shared latent tensors, emits one layer slice immediately before its block,
and releases that slice after consumption.  During training, particle-block
activation checkpointing recomputes the layer projection in backward so all
layer slices are not retained simultaneously.  A streamed implementation is
parity-equivalent if it obeys the same projection order and dtype.

With the relation latent replaced by exact zeros, `B_DUAL_FIXED` and
`B_DUAL_GATED` must match the registered base-only dual-path control in
logits, gradients, masks, and emitted per-layer biases.  This is not required
to equal the lower-capacity ordinary one-stem `O_BASE`.

The state dictionary stores shared stems, every layer projection, and every
layer/head gate under versioned names.  It is incompatible with the existing
single-bias RPT state dictionary unless an explicit migration command records
the old and new contract hashes.  A migrated `B_CONCAT` checkpoint may not be
presented as a dual-path checkpoint.

---

## 8. Offline expert architecture

Each expert begins with the exact base-size RPT particle encoder:

```text
particle input dimension = 17
particle embedding dimensions = [128, 512, 128]
attention heads = 8
particle-attention blocks = 8
standard pair feature count = 4
enabled extra relation = expert-specific
activation = GELU
```

The expert processes all valid particles and returns the final particle hidden
states before Weaver's ordinary class-attention path:

```text
H_e shape = [B,N,128]
```

The implementation must add an explicit, tested particle-state tap.  Hooks
that depend on incidental module names are forbidden in production.  The
tap's logits and gradients must be parity-tested against the unmodified model
when the ordinary class head is selected.

Every expert is initialized independently and trained from scratch in the
primary offline screen.  Warm-start and shared-particle-encoder variants are
separate controls.

---

## 9. Canonical summary-token bottleneck

### 9.1 Learned slot queries

For token shape \(S=(K,D)\), expert \(e\) owns learned queries:

\[
Q_e\in\mathbb{R}^{K\times D}.
\]

Queries are expanded across the batch and receive:

- a learned expert-type embedding;
- a learned slot-index embedding;
- no particle index or constituent position.

The canonical tokenizer contains two blocks.  Each block performs:

1. pre-normalized self-attention among the \(K\) summary slots;
2. pre-normalized cross-attention from summary slots to valid particle hidden
   states, with learned key/value projections from particle width 128 to
   token width \(D\);
3. a pre-normalized GELU feed-forward residual.

Configuration:

```text
token dimension = D in {64,128}
tokenizer blocks = 2
attention heads = 4 when D=64, 8 when D=128
MLP expansion = 4
attention dropout = 0
residual dropout = 0.1
```

The output is:

```text
Z_e shape = [B,K,D]
```

Padded particles are never keys or values.  Summary slots are always valid.

### 9.2 Canonical slot identity

Slot identity is the learned query index.  There is no Hungarian matching
between token sets.  Once an offline expert is selected, its encoder,
tokenizer, slot queries, normalization, and classifier head are frozen.

Tokenwise reconstruction loss is valid only against this frozen coordinate
system.  Jointly moving offline targets and HLT predictors is forbidden in the
primary bridge.

### 9.3 No classification bypass

The individual expert classifier receives only `Z_e`.  It may not access:

- particle hidden states;
- the original class token;
- raw particles;
- jet-level features outside the token bank.

This ensures that the saved token bank is sufficient for that expert's own
decision.

### 9.4 Expert classification head

The head uses one learned class query and two class-attention blocks over
`Z_e`, followed by:

```text
RMSNorm(D)
Linear(D,10)
```

The class query is not part of the reconstructable token target.  The
reconstructable object is exactly the \(K\) summary tokens before the expert
class head.

### 9.5 Token-shape screen

The canonical shape registry is:

```text
S1_128  = (K=1,  D=128)
S2_128  = (K=2,  D=128)
S4_128  = (K=4,  D=128)
S8_128  = (K=8,  D=128)
S16_128 = (K=16, D=128)
S8_64   = (K=8,  D=64)
S16_64  = (K=16, D=64)
```

The equal-scalar comparison `S8_128` versus `S16_64` tests whether more
structured slots or more per-slot capacity is preferable.

Controls include:

- the ordinary Weaver class-token model with no new bottleneck;
- masked mean pooling to one vector;
- one learned query without summary self-attention;
- \(K\) learned queries without summary self-attention;
- the canonical two-block tokenizer;
- `TOK_MULTI_DEPTH`, which pools from particle blocks 4 and 8.

`TOK_MULTI_DEPTH` independently applies `RMSNorm + Linear(128,D)` to the
block-4 and final particle states, adds learned `intermediate`/`final` depth
embeddings, concatenates the two masked particle sequences, and applies the
same two-block learned-query tokenizer.  It has no direct classifier bypass.
This control tests whether late particle layers discard relation-specific
evidence useful to the bridge.

The canonical tokenizer is primary.  Pooling alternatives are controls and do
not silently replace it based on one result.

---

## 10. Individual expert training

The general offline expert objective is:

\[
L_{\mathrm{expert}}
=
L_{\mathrm{CE}}(y,\ell_e)
+\lambda_{\mathrm{KD}}\,T^2\,D_{\mathrm{KL}}
\left(
\operatorname{softmax}(\ell_{\mathrm{teacher}}/T)
\parallel
\operatorname{softmax}(\ell_e/T)
\right),
\]

with:

```text
temperature T = 2.0
teacher logits detached
```

Universal `O_BASE` KD is not the primary specialization policy because pulling
every expert toward the same logits may suppress complementary errors.  The
registered expert-loss candidates are:

```text
ELOSS_CE              lambda_KD=0.0
ELOSS_BASE_LOW        teacher=O_BASE,    lambda_KD=0.10
ELOSS_BASE            teacher=O_BASE,    lambda_KD=0.50
ELOSS_FULLREL         teacher=O_FULLREL, lambda_KD=0.50
ELOSS_ENSEMBLE        teacher=mean probability of locked O_BASE and O_FULLREL,
                      lambda_KD=0.50
ELOSS_KD_DOMINANT     teacher=selected strongest teacher,
                      lambda_KD=1.0, CE weight=0.25
```

`ELOSS_CE` is the natural-specialization reference.  Teacher checkpoints are
selected without RETB expert or final-test results.  `O_FULLREL` or the teacher
ensemble may be used only after their hashes and validation-only ordering are
locked.

Loss candidates are screened on representative experts, then `ELOSS_CE`,
`ELOSS_BASE_LOW`, and the best available teacher-KD candidate are trained for
every expert at every carried token shape.  The seven loss choices are made
jointly by Section 10.2's deterministic beam selector using all-bank fusion
utility, not seven independent expert or hybrid scores.  A globally shared
loss choice is retained as a simplicity control.

Each expert checkpoint records:

- token count and dimension;
- relation and normalization hashes;
- tokenizer configuration;
- teacher checkpoint hash;
- individual logits and metrics;
- token norm and utilization diagnostics;
- slot-to-particle attention sufficient statistics.

### 10.1 Bridge-aware offline target candidates

Pure offline tokens remain the primary semantic target.  RETB also tests
whether an equally strong but more HLT-predictable coordinate system improves
the deployable ceiling.

Registered target modes are:

```text
T0_PURE             selected offline expert tokens, no HLT objective
T1_ANCHORED_BRIDGE  co-designed tokens anchored to instance-level T0 content
T1_TASK_BRIDGE      co-designed task tokens without a T0-coordinate claim
T2_PROJECT          frozen pure tokens mapped to a predictable bridge space
T3_LOGIT            direct offline-logit distillation, no token-fidelity claim
```

Bridge-target construction is not part of the initial offline-expert stage.
It has exact forward dependencies:

1. lock the three-seed `T0_PURE` experts, fusion, and selected shapes;
2. train the seed-matched native HLT evidence encoders;
3. train `PILOT_T0` against frozen `T0_PURE`;
4. copy that pilot and co-design each `T1/T2` candidate;
5. freeze/version the new target semantics and rebuild target caches;
6. only then begin the full predictor architecture/loss campaign.

`PILOT_T0` is fixed globally rather than selected per target:

```text
HLT encoder = seed-matched HE_OFFLINE_INIT
HLT realization = R_MULTI
predictor = A3_SLOT_DECODER_DIRECT
context = C2_ALL
objective = W_TOKEN_HEAVY
uncertainty = U_SLOT
normalization = N_UNCLIPPED
learning rate = 5e-4
dropout = 0.0
```

The HLT encoder relation topology and token shape match the target expert.
Every bridge-target artifact binds the exact `T0`, HLT-encoder, and initial
`PILOT_T0` checkpoint hashes.  A different pilot creates a different target
contract and run ID.

For both `T1` modes, the offline forward pass still consumes only offline
particles.  Paired HLT evidence appears only in training losses.  The common
task-bridge objective is:

\[
L_{\mathrm{T1task}}
=
L_{\mathrm{offline\ expert}}
+\lambda_{\mathrm{pred}}L_{\mathrm{token}}
+0.50L_{\mathrm{offline\ fusion}}
+0.50L_{\mathrm{T0logit}},
\]

with:

```text
lambda_pred in {0.05,0.10,0.25}
```

`L_T0logit` is temperature-2 KL preservation through the frozen `T0` expert
head and frozen `T0` fusion after substituting the moving bank.  It prevents a
new consumer from hiding wholesale loss of the original decision.
During each per-expert co-design row, the other six banks are frozen
same-event `T0` banks.  `L_offline fusion` is ground-truth CE through a
candidate hybrid fusion initialized from `T0`; `L_T0logit` uses a separate
fully frozen copy.  Section 10.3 later retrains consumers for complete
seven-bank target tuples.

`T1_ANCHORED_BRIDGE` additionally minimizes:

\[
L_{\mathrm{T1anchor}}
=
L_{\mathrm{T1task}}
+0.25L_{\mathrm{T0anchor}}
+0.10L_{\mathrm{withinclass\ retrieval}}
+0.05L_{\mathrm{covariance}}.
\]

`L_T0anchor` is normalized Huber loss between the moving tokens and the frozen
same-event `T0` tokens.  For any `[K,D]` bank define:

```text
bank_vector = flatten(train-normalized bank in slot-major, then channel order)
dot and norms accumulated in FP32
cosine(a,b) = dot(a,b) / max(norm2(a)*norm2(b),1e-8)
similarity(a,b) = cosine(a,b) / 0.1
```

If either vector has exact FP32 norm zero, define its cosine similarity as
exactly zero.  InfoNCE uses FP32 `logsumexp`.  Retrieval candidates sort by
descending similarity, then ascending canonical identity for an exact tie.
There is no learned retrieval projection, pooling head, or slot permutation.
The retrieval loss is InfoNCE with the predicted HLT bank as query, the
same-event moving offline bank as positive, and 31 fixed within-class
wrong-event banks as negatives.  Negative identities are chosen without
replacement by
`sha256("retb_t1_negatives_v1" || pipeline_seed || jet_identity)` from a
canonical per-class `model_train` identity ring after explicitly removing the
query identity.  A ring with fewer than 32 distinct identities is an invalid
training fixture.  Using within-class negatives ensures a class prototype
cannot solve the retrieval objective.

Certification uses an independent canonical per-class `val_design` ring and
`sha256("retb_t1_cert_negatives_v1" || pipeline_seed || jet_identity)`, again
removing the query identity before choosing 31 negatives.  Training and
certification negative rings and hashes are therefore non-interchangeable.

For the global effective batch \(B\), cast train-normalized moving and `T0`
tokens to FP32 and define centered population covariance independently for
each slot:

\[
\mu_k=\frac{1}{B}\sum_{b=1}^{B}z_{b,k},
\qquad
C_k=\frac{1}{B}\sum_{b=1}^{B}
(z_{b,k}-\mu_k)(z_{b,k}-\mu_k)^\top.
\]

Then:

\[
L_{\mathrm{covariance}}
=
\frac{1}{K}\sum_{k=1}^{K}
\frac{\lVert C^{\mathrm{moving}}_k-C^{\mathrm{T0}}_k\rVert_F}
{\max(\lVert C^{\mathrm{T0}}_k\rVert_F,10^{-8})}.
\]

`T0` statistics are detached; moving statistics retain gradients.  \(B\)
includes every microbatch in the optimizer update and every distributed rank.
FP32 sums, outer-product sums, and counts are combined with a differentiable
all-reduce before centering and loss evaluation.  Accumulation steps retain
the required graph or differentiable sufficient statistics until that update;
they may not average independent microbatch covariance losses.  Global
effective batch size below two is invalid.

Training begins from `T0_PURE`.  The first candidate unfreezes only the offline
summary tokenizer and expert/fusion consumers.  A confirmation-only candidate
also unfreezes the final two offline particle blocks at one-tenth learning
rate.  Each candidate begins from a copy of `PILOT_T0`.  HLT predictor and
offline-target updates alternate; the predictor is detached during an
offline-target update, and the offline target is detached during a predictor
update.

For `T2_PROJECT`, the pure offline expert is frozen.  If `D_bridge=D`, use a
true residual projection:

```text
z_bridge = z_T0 + Linear(GELU(Linear(RMSNorm(z_T0),2D)),D)
```

If `D_bridge != D`, no dimension-changing residual is claimed:

```text
base = Linear(RMSNorm(z_T0),D_bridge)
z_bridge = base
         + Linear(GELU(Linear(RMSNorm(base),2*D_bridge)),D_bridge)
```

`D_bridge` is in `{64,128}`.  A new offline expert/fusion consumer and a copy
of `PILOT_T0` are trained against projected tokens with:

\[
L_{\mathrm{T2}}
=
L_{\mathrm{projected\ expert}}
+\lambda_{\mathrm{pred}}L_{\mathrm{token}}
+0.50L_{\mathrm{projected\ fusion}}
+0.25L_{\mathrm{T0project}}
+0.50L_{\mathrm{T0decodedlogit}}.
\]

`L_T0project` reconstructs `T0` through a learned linear decoder used only
during training; `L_T0decodedlogit` applies the frozen `T0` expert/fusion
logit-preservation loss to that decoded token.  Thus the formula is valid when
dimensions differ and `T2` is explicitly optimized for HLT predictability
rather than merely being a classifier-trained projection.

All `T1/T2` candidates are eligible for maximum-performance selection only
when their three-seed offline oracle fusion satisfies:

```text
mean accuracy deficit from T0_PURE <= 0.0020
mean cross-entropy increase <= 0.0050
worst per-class efficiency deficit <= 0.0100
```

Representation-preserving language additionally requires an immutable
`bridge_content_certified=true` result on `val_design`:

```text
frozen T0 expert/fusion class agreement >= 0.990
median per-slot/channel variance ratio to T0 >= 0.50
no more than 5% of slot/channels have variance ratio < 0.25
effective rank per expert >= 0.80 * T0 effective rank
relative per-slot covariance error <= 0.25
within-class 32-way same-event retrieval accuracy >= 0.20
```

For \(n\) evaluated events, the effective-rank matrix has exact shape
`[n_events,K*D]`, with each row equal to the slot-major train-normalized bank
vector above.  Cast the complete matrix to C-contiguous CPU float64 and compute
the exact economy singular values with
`numpy.linalg.svd(full_matrices=False,compute_uv=False)`.  The artifact binds
the NumPy and linked LAPACK versions.  Let \(s_{\max}\) be the largest singular
value.  If \(s_{\max}=0\), effective rank is zero.  Otherwise treat
`s_i <= 1e-12*s_max` as numerical zero, normalize the remaining singular
values to sum one, and define:

\[
\operatorname{erank}
=
\exp\left(-\sum_{i:p_i>0}p_i\log p_i\right).
\]

Certification retrieval uses the same unlearned FP32 cosine similarity and
canonical-identity tie rule.
Retrieval diagnostics also report mean reciprocal rank and nearest-target
identity accuracy by class.
For a dimension-changing `T2`, variance/covariance and frozen-logit
certification are computed after its training-only decoder maps back to `T0`
coordinates; effective rank and retrieval are additionally reported in the
bridge coordinates.  Thresholds and matrix conventions are serialized before
training.

`T1_ANCHORED_BRIDGE` is the primary representation candidate.
`T1_TASK_BRIDGE` remains eligible for maximum classification but may be called
representation bridging only if it independently passes the same content
certification.  `T2_PROJECT` is a learned bridge coordinate system and is not
claimed to preserve the original coordinates even when certified.

Candidates are selected using `val_design` only after the certification and
offline noninferiority flags are frozen on that population.  Failure does not
stop reporting or the pure-token bridge.

The report keeps `T0`, both `T1` modes, `T2`, and `T3` separate.  A
bridge-aware target is not described as a purely offline-discovered
representation.

### 10.2 Joint offline expert-loss bundle selection

Expert loss variants interact through fusion and are not selected by seven
independent leave-one-bank substitutions.  For every carried shape, cache the
identity-bound `model_train`, `val_stop`, and `val_design` tokens/logits from
each eligible loss variant.  Then:

1. choose the individually best variant for each expert as the deterministic
   default tuple;
2. traverse experts in canonical order
   `BASE4,PT,TRACK,PID,CHARGE,DENSITY,REGION`;
3. at each depth, expand every retained tuple with every eligible variant for
   the current expert while later experts retain their defaults;
4. train a fresh seed-`41701` `F_POOLED_MLP` readout on cached `model_train`,
   select its epoch on `val_stop`, and score the complete provisional tuple
   on `val_design`;
5. retain beam width `16` by accuracy, the global `0.0001` window, cross
   entropy, FLOPs, parameters, then lexicographic seven-variant tuple;
6. train canonical `F_TOKEN_TRANSFORMER` fusions for the final top four tuples
   and select one using the same ordering.

Readout weights are never shared across provisional tuples.  Also train every
globally shared loss tuple and the all-`ELOSS_CE` tuple as simplicity/natural
specialization controls.  The selected seven-expert tuple is shared across
pipeline seeds; only its learned checkpoints differ.  This joint selector,
not an individually best hybrid score, defines the primary offline bank.

### 10.3 Joint bridge-target coordinate systems

After `PILOT_T0` co-design, different experts may prefer different eligible
target modes.  A fusion trained on one coordinate tuple cannot silently
consume another.  Stage E therefore applies the Section-10.2 beam procedure
to the eligible per-expert target modes, using pilot all-predicted utility
instead of expert-loss utility.  Every provisional tuple receives its own
fresh `F_POOLED_MLP`; the final top four receive their own canonical
`F_TOKEN_TRANSFORMER`.  Beam width, seed, expert order, and tie rules are
identical except the readout seed is `41703`.

Freeze the top four mixed target tuples plus every homogeneous eligible
target-mode tuple.  Each frozen tuple owns a distinct offline fusion,
normalizer set, target-cache namespace, and contract hash.  The full predictor
campaign may compare these locked coordinate systems but may not construct a
new cross-mode tuple.  This prevents a predictor bundle from being evaluated
through a fusion that was never trained for its target coordinates.

---

## 11. Offline fusion architecture

For a fixed token shape \(S=(K,D)\), project every bank independently to the
fusion width 128 and concatenate the seven frozen expert banks:

```text
[Z_BASE4, Z_PT, Z_TRACK, Z_PID, Z_CHARGE, Z_DENSITY, Z_REGION]
```

Add learned:

- expert-type embedding;
- slot-index embedding;
- source embedding `oracle_offline`;

and prepend one fusion class token.

For `D=128`, the bank projection is initialized to identity.  For `D=64`, it
is a learned `RMSNorm(64) + Linear(64,128)` projection.  Projection parameters
belong to the fusion model, not the offline expert target.

The primary fusion transformer is:

```text
dimension = 128
layers = 3
heads = 8
MLP expansion = 4
pre-norm = true
attention dropout = 0
residual dropout = 0.1
classifier = RMSNorm + Linear(128,10)
```

The primary Stage-2 fusion freezes all expert encoders and tokenizers.  Only
the fusion transformer is trained.  This preserves independently learned
expert representations while complementarity is measured.

### Heterogeneous token budgets

Uniform shapes are the primary controlled comparisons.  Two heterogeneous
finalists are also required under a locked total of 56 summary slots:

```text
HET_PHYSICS:
  BASE4=4, PT=8, TRACK=16, PID=4, CHARGE=4, DENSITY=4, REGION=16

HET_SELECTED:
  K_e in {1,2,4,8,16}, sum_e K_e <= 56
```

`HET_SELECTED` begins with one slot per expert.  Repeatedly add the next
available slot-count increment to the expert with the largest three-seed
incremental `F_POOLED_MLP` fusion-accuracy gain per added slot on `val_design`.
For every greedy step, all candidate increments start from the identical
current allocation and use deterministic matched readout seeds.  Break ties
by lower fusion cross entropy, fewer added slots, then canonical expert order.
Stop at 56 slots or when no registered increment fits.  This fusion-utility
criterion accounts for redundant experts instead of rewarding only
stand-alone strength.  Slot dimension is 128 for this budgeted selection;
mixed dimensions are handled only by the explicit uniform shape screen.

Both heterogeneous assignments are frozen before their fusion transformer is
trained.  They do not use final fusion results to revise the allocation.

A maximum-performance allocation control, `HET_BEAM`, uses the same
`K_e in {1,2,4,8,16}`, `D=128`, and 56-slot limit.  It assigns experts in
canonical order with beam width `32`.  Every partial assignment is completed
with one slot for unassigned experts, discarded if the minimum completion
cannot fit, and scored by a separately trained seed-`41702`
`F_POOLED_MLP`.  Retention uses fusion accuracy, the global `0.0001` window,
cross entropy, fewer total slots, then lexicographic allocation.  The top four
complete allocations receive canonical token-transformer fusions.
`HET_SELECTED` remains the transparent greedy control; `HET_BEAM` tests
whether its local allocation missed a better complementary budget.

Required fusion controls:

```text
F_BEST_SINGLE
F_UNIFORM_LOGIT_MEAN
F_TRAINED_LOGIT_LINEAR
F_POOLED_MLP
F_TOKEN_TRANSFORMER
F_TOKEN_TRANSFORMER_LIGHT_FINETUNE
F_TOKEN_TRANSFORMER_FULL_FINETUNE
```

`LIGHT_FINETUNE` unfreezes only the summary tokenizers and uses one-tenth of
the fusion learning rate for them.  `FULL_FINETUNE` unfreezes complete experts
and is an architectural control, not the primary interpretation model.

---

## 12. Complementarity and capacity controls

The campaign must separate relation specialization from capacity, ensembling,
and optimization.

Required offline controls:

```text
O_BASE
O_WIDE
O_MONO_PARAM
O_MONO_FLOP
O_BASE_LONG
O_FULLREL
O_GROUPED_HEAD_REL
O_7X_UNBIASED_ENSEMBLE
O_7X_UNBIASED_TOKEN_FUSION
O_RELATION_EXPERT_TOKEN_FUSION
```

`O_WIDE` retains the completed RPT pair-encoder capacity control.  It is not a
total-system match.

`O_MONO_PARAM` and `O_MONO_FLOP` are selected only after the candidate
deployment graph is fully resolved.  Their target totals include all
inference-time expert encoders, summary tokenizers, dimension projections,
token refiners, reliability or uncertainty heads, and final consumers.
Training-only offline teachers and target caches are excluded from the HLT
deployment total.  The candidates come from a predeclared monolithic base4
grid over:

```text
particle hidden width
feed-forward expansion
attention-head count
particle-block count
class-block count
```

All candidates must divide hidden width exactly by head count and must use the
same particle inputs, optimizer, and token-free classification interface.
`O_MONO_PARAM` minimizes absolute total-parameter mismatch, then analytical
inference-FLOP mismatch, then smaller depth-plus-width sum, then
lexicographic configuration tuple.  `O_MONO_FLOP` reverses the first two
criteria.  HLT controls use the identical procedure against the complete
selected HLT graph.

Analytical FLOPs are reported for batch size 1 and 128 under the exact
deployed maximum-particle contract.  Measured latency at both batch sizes is
diagnostic and cannot replace the deterministic selector.  The selected
mismatches and whether both parameters and batch-1 FLOPs are within 5% are
reported; absence of a close dense match is not hidden.  Earlier encoder-only
matches are retained as historical capacity controls but cannot support a
claim against the complete deployed graph.

`O_BASE_LONG` controls repeated label exposure.  Its fixed optimizer-update
budget equals the sum of labeled examples presented while training seven
experts and the primary frozen fusion, divided by the `O_BASE` effective batch
size with ceiling rounding.  It uses the same warm-up fraction and cosine
schedule over that complete update count.  It does not stop early.

`H_BASE_LONG` uses the analogous sum over every selected HLT-side training
phase with a nonzero ground-truth CE term: native experts/fusion, predictors,
`J4`, `J5`, refiner, adapter, and final fusion.  A jet appearing once in one
joint optimizer update counts once even if that loss contains several CE
heads; separate optimizer updates count separately.  Pure offline teacher,
oracle-target generation, KD-only, and calibration-only updates do not count
as HLT-label presentations.  The exact component-wise ledger and ceiling
rounding are serialized before `H_BASE_LONG` starts.

`O_GROUPED_HEAD_REL` assigns disjoint attention-head groups to relation
families inside one model.  Each group receives a separate softmax and bias;
relations are not summed before softmax.  With eight heads the fixed mapping
is:

```text
heads 0,1 = base4
head 2    = base4 + PT
head 3    = base4 + TRACK
head 4    = base4 + PID
head 5    = base4 + CHARGE
head 6    = base4 + DENSITY
head 7    = base4 + REGION
```

`O_7X_UNBIASED_ENSEMBLE` trains seven independently seeded `BASE4` experts with
the same aggregate expert count and combines their logits.

`O_7X_UNBIASED_TOKEN_FUSION` uses the same token/fusion topology as the
relation-expert system but gives every expert only `base4`.

### Redundancy is measured, not presumed

The primary campaign does not force orthogonal tokens or repulsive attention
maps.  Shared class information is legitimately shared and should not be
removed to create visually different experts.

For frozen experts report:

- pairwise prediction disagreement;
- pairwise correct/error contingency tables;
- per-class error correlation;
- centered logit-residual correlation;
- linear CKA of token banks after flattening slots;
- slot-attention Jensen-Shannon divergence on common offline particles;
- performance of every single expert;
- performance of every pair and every frozen-token subset via cheap
  subset-specific linear/MLP readouts;
- leave-one-expert-out fusion performance;
- Shapley-style information contribution computed from all 128 expert subsets
  using the frozen subset readouts, not by masking an untrained fusion model;
- bias-zero and within-jet relation-shuffle sensitivity.

Optional specialization ablations are:

- `S0_NATURAL`: independent training, the primary policy;
- `S1_FIXED_SCALE`: use `B_DUAL_FIXED`, applying `alpha=1.0` to the emitted
  relation attention logits;
- `S2_BOUNDED_SCALE`: use `B_DUAL_GATED`, applying
  `alpha=2*sigmoid(a)` to emitted relation attention logits independently by
  layer/head;
- `S3_RELATION_AUX`: add a standardized relation-summary prediction head with
  total weight `0.10`;
- `S4_RESTRICTED_FIELDS`: hide non-relation particle fields as an
  interpretability control;
- `S5_CROSSCOV`: jointly fine-tune tokenizers with a `1e-3` cross-expert
  off-diagonal covariance penalty after per-bank centering.

`S3_RELATION_AUX` predicts fixed train-only-standardized summaries:

```text
BASE4   jet mass fraction, multiplicity, delta-R pair quartiles
PT      leading four pT fractions, pT entropy, scalar-pT concentration
TRACK   valid-track fraction, |d0|/|dz| significance quartiles,
        compatible-pair fractions at chi2 <= 1,4,9
PID     six PID pT fractions and six PID count fractions
CHARGE  positive/negative/neutral pT fractions, normalized net charge,
        opposite-sign and same-sign valid-pair fractions
DENSITY four annular mean counts, four annular pT fractions,
        displaced local-pT fraction
REGION  K=2/4/8 leading cluster pT fractions, cluster multiplicities,
        LCA-depth and merge-scale quartiles
```

Missing or inapplicable summaries are explicitly masked.  These targets are
diagnostic inductive biases, not privileged information.

`S5_CROSSCOV` is deliberately a secondary control.  It may remove shared
class information and cannot become the nominal winner merely because it
reduces representation similarity; it must improve the ordinary frozen-token
fusion selection metric.

No diversity regularizer is part of the primary model.  If experts are
redundant, that is a valid negative result.

### Expert dropout policy

Whole-bank expert dropout is:

```text
disabled in primary offline fusion
disabled as an anti-redundancy mechanism
```

A late robustness control may apply whole-bank dropout probabilities
`0.10` and `0.25` with expert encoders frozen.  This measures robustness to
missing evidence.  It is not described as producing expert diversity.

---

## 13. Two-stage token-shape selection

Offline performance alone does not determine the deployable token shape.

### 13.1 Offline high-capacity shape

Across confirmation seeds `101,202,303`, rank every registered uniform
\(S=(K,D)\) by frozen-expert `F_TOKEN_TRANSFORMER` on `val_design`:

The selector requires all seven uniform shape IDs at all three seeds.  A
missing or invalid shape artifact is an incomplete campaign, not an implicit
prefilter.  Seed-101 results may schedule controls but cannot remove a shape
from this ranking.

1. compute the maximum mean accuracy;
2. retain shapes within `0.0005` absolute accuracy;
3. choose lower mean cross entropy;
4. choose fewer total scalars \(K D\);
5. choose fewer slots \(K\);
6. choose smaller dimension \(D\);
7. choose lexicographic shape ID.

The result is the immutable pair `SHAPE_HIGH=(K,D)`.

### 13.2 Offline compact shape

Starting from increasing \(KD\), select the first shape satisfying:

```text
mean accuracy deficit from SHAPE_HIGH <= 0.0020
mean cross-entropy increase from SHAPE_HIGH <= 0.0050
worst per-class efficiency deficit <= 0.0100
all metrics finite at all three seeds
```

This is the immutable pair `SHAPE_COMPACT=(K,D)`.  If no smaller shape
qualifies, it equals `SHAPE_HIGH`.

The campaign writes the full ranking even if every multi-expert model loses to
`O_BASE`.  That outcome does not prevent HLT stages.

### 13.3 Carried shapes and heterogeneous finalists

Carry forward, with duplicates removed:

```text
S1_128
SHAPE_COMPACT
SHAPE_HIGH
HET_PHYSICS
HET_SELECTED
HET_BEAM
```

The heterogeneous finalists use the locked Section-11 assignments.  Predictor
architecture screening uses only uniform shapes; heterogeneous prediction is
assembled after per-expert predictors are selected.

### 13.4 Bridge-aware shape

After predictor confirmation, select `SHAPE_BRIDGE` from `SHAPE_COMPACT` and
`SHAPE_HIGH` using mean `val_design` performance through frozen offline
fusion:

1. higher mean paired gain over shape-matched `HF_NATIVE`;
2. lower mean frozen-fusion cross entropy;
3. lower mean normalized token error;
4. fewer total scalars \(KD\);
5. smaller \(K\), then \(D\).

The final report evaluates both uniform shapes and all three heterogeneous
finalists.
`SHAPE_BRIDGE` chooses the primary uniform deployable configuration; it does
not hide the compression/recoverability tradeoff.

---

## 14. Native HLT expert bank

For every carried shape, train `HE_e_Ss` with the same relation identity,
token shape, and classifier topology as its offline expert.  HLT inputs follow
the registered realization arm.

`HE_SCRATCH_CE` is the faithful native baseline:

```text
random initialization
unweighted 10-class HLT CE
no offline target
```

`HE_OFFLINE_INIT` copies all shape-compatible particle-encoder, relation,
tokenizer, and classifier parameters from the corresponding selected offline
expert.  It replaces offline fitted statistics with Section 7's one shared
nominal-R_MULTI HLT normalizer; it does not fit an expert- or
realization-specific normalizer.  Target-token normalizers remain bound to
their offline target coordinate system.  Newly introduced
availability embeddings initialize to zero.  It trains with HLT CE at:

```text
copied encoder learning rate = 1e-4
new parameter learning rate = 5e-4
```

`HE_DUAL_OBJECTIVE` begins from the same offline initialization and minimizes:

\[
L_{\mathrm{HE\_DUAL}}
=
L_{\mathrm{HLT\ CE}}
+\lambda_{\mathrm{token}}L_{\mathrm{token}}
+\lambda_{\mathrm{logit}}L_{\mathrm{expertKD}},
\]

over the paired frozen offline expert target.  The locked candidate weights
are:

```text
(lambda_token,lambda_logit) in
{(0.10,0.25),(0.25,0.25),(0.25,0.50)}
```

This objective makes the HLT evidence encoder bridge-aware before a separate
predictor is attached.  `HE_SCRATCH_CE` remains the named native-HLT control;
privileged `HE_DUAL_OBJECTIVE` is never presented as an HLT-only baseline.

Train:

- every individual HLT expert;
- every `HE_OFFLINE_INIT` expert at the two selected uniform shapes;
- every `HE_DUAL_OBJECTIVE` expert at the selected predictor-screen shapes;
- native logit mean;
- native learned logit fusion;
- native HLT expert-token fusion;
- standard `H_BASE`;
- `H_WIDE`;
- seven-expert unbiased HLT controls.

These establish whether multi-bias HLT processing alone explains later gains.

---

## 15. Offline target-token cache

After `SHAPE_HIGH`, `SHAPE_COMPACT`, `HET_PHYSICS`, `HET_SELECTED`, and
`HET_BEAM` are locked, run each required selected frozen offline expert over:

```text
model_train
val_stop
val_design
```

Targets are stored in identity-bound, resumable shards:

```text
float16 token payload when audited, otherwise float32
float32 per-channel normalization statistics
float32 expert logits
int64 labels
canonical identity indices
```

The target cache records:

- offline expert checkpoint hash;
- source split and identity hash;
- token shape and dtype;
- token content hash;
- expert-logit content hash;
- slot-query hash;
- source code snapshot;
- shard hashes and complete coverage.

Target caches are produced separately for every confirmation pipeline seed
and target mode:

```text
offline expert seed
offline fusion seed
target mode T0_PURE/T1_ANCHORED_BRIDGE/T1_TASK_BRIDGE/T2_PROJECT
token shape
locked seven-expert target-coordinate tuple and fusion hash
```

Float16 storage is attempted only after a float32 parity audit proves:

```text
max token round-trip absolute error <= 5e-4
expert logit max absolute error <= 2e-4
predicted class identities exact on the audit sample
```

If the audit fails for any expert/shape/seed, that target cache falls back
automatically to float32, records the failed float16 diagnostics, and
continues.  Mixed cache dtypes are permitted only when each shard manifest
declares its dtype and consumers convert to float32.  Float16 failure is not a
scientific or workflow failure.

Training always loads target tokens as float32.  No offline raw particles need
remain resident while HLT predictors train.

### Post-selection oracle and final-test targets

Before `selection/locked_scale_finalists.json` exists, Stage N may emit
identity-bound deployable `final_select/stack_val` logits and probabilities
for every shortlisted 3M graph and seed.  These selection-prediction shards:

```text
contain canonical identities, graph/seed IDs, float32 logits, and float32 probabilities
contain no labels, offline targets, oracle tokens, or oracle logits
consume only HLT inputs, deployable normalizers, and shortlisted 3M checkpoints
declare selection_eligible=true
```

Probabilities are the FP32 stable-softmax of the stored logits; shard
validation recomputes them and requires exact predicted-class identities and
the campaign's serialized FP32 probability tolerance.
The immutable selector joins labels only from the authenticated
`inputs/final_select_label_manifest.json.gz`, outside the prediction shards,
and writes separate content-hashed metric artifacts and selection traces.  The
label manifest is a canonical identity-sorted `(identity,int64_label)`
projection whose sole parent is the authenticated split manifest; only the
selector role may load it.  No other pre-lock `stack_val` model output is
allowed.

Before finalist locking, no command may produce a `final_test` model-derived
token, logit, probability, prediction, label joined to model output, or metric.
Pre-lock `stack_val`/`final_test` input preparation may write only:

```text
raw/offline input arrays
deterministically degraded HLT input arrays
relation/REGION input sidecars
identity and source manifests
```

These are input transformations, not model outputs, and their builders cannot
load a checkpoint.  The Stage-N selection-inference worker is a distinct
checkpoint-loading role authorized only for the label-free `stack_val`
selection shards above.

After the two 3M finalists are selected and
`locked_scale_finalists.json` is immutable, Stage N may build:

1. `final_select` offline targets for post-selection oracle/token-fidelity
   diagnostics only;
2. `final_test` offline targets required by locked oracle/token-fidelity
   report rows.

Every such target shard must use the exact corresponding Stage-M
scale-trained offline expert/fusion hashes and scale target normalizer.  A
500k teacher or pre-lock target cache is ineligible even if its shape matches.
Post-selection `final_select` diagnostics are marked
`selection_eligible=false` and cannot alter either finalist.

The post-lock cache includes tokens, expert logits, labels, and identities
only because the finalist selection is already frozen.  Its content hashes
become parents of `selection/final_test_execution_lock.json`; scientific
final-test inference cannot start before that second execution lock exists.

---

## 16. Predictor evidence banks

Predictor inputs are constructed once per HLT jet:

1. corresponding native HLT expert tokens `Z_e_hlt`;
2. every other native HLT expert token bank;
3. native unbiased HLT particle hidden states from `HE_BASE4`;
4. optional relation-specialized HLT particle hidden states.

Physical pair biases are applied inside particle encoders, where particle
pairs have defined geometry, momentum, displacement, and categories.  They are
not invented between an abstract summary query and a particle.

Evidence sequences receive:

- source-type embedding;
- expert-type embedding;
- slot-index embedding for summary banks;
- particle-versus-summary embedding;
- valid mask.

All HLT evidence is computed from HLT fields only.

---

## 17. Predictor architectures

All predictors output:

```text
predicted tokens [B,K,D]
log variance      [B,K,U]
gate              [B,K,1] where applicable
```

Here uncertainty width \(U\) follows the registered uncertainty head in
Section 20.  Log variance is clipped to `[-8,4]`.

### `A0_AFFINE`

Flatten neither particles nor slots.  Apply one shared affine projection to
each corresponding HLT token.  This is the cheapest token-alignment control.

### `A1_RESMLP`

For each HLT slot independently:

```text
RMSNorm(D)
Linear(D,2D)
GELU
Linear(2D,D)
```

Output is `Z_hlt + delta`.  Slots do not communicate.  This controls whether
cross-token attention is necessary.

### `A2_TOKEN_ENCODER`

Concatenate corresponding HLT expert tokens and run three self-attention
encoder blocks.  Project the first \(K\) outputs to offline tokens.  No raw
particle context is used in its self-only form.

### `A3_SLOT_DECODER_DIRECT`

The primary direct predictor owns \(K\) learned offline-target query
embeddings initialized from, but not weight-shared with, the frozen offline
slot queries.  Three decoder blocks perform:

1. self-attention among target queries;
2. cross-attention to the selected HLT evidence sequence;
3. GELU feed-forward residual.

Configuration:

```text
dimension = D
layers = 3
heads = 4 when D=64, 8 when D=128
MLP expansion = 4
dropout = 0.1
```

The decoder emits `Zhat_e_off` directly.

### `A4_SLOT_DECODER_GATED`

Use the same decoder to emit `delta` and gate:

\[
\hat Z_e^{off}
=
M_e\!\left(\operatorname{LN}(Z_e^{hlt})\right)
+\sigma(g)\odot\Delta Z_e.
\]

Here \(M_e\) is the learned affine HLT-to-offline anchor map.  The HLT token
residual anchor is therefore placed in the offline token coordinate system
before correction.
The gate bias initializes to `-2.0`, so early training begins near the mapped
HLT anchor rather than a large correction.

### Predictor parameter controls

For every selected predictor report:

- parameter count;
- analytical FLOPs;
- measured latency;
- peak memory;
- a widened residual MLP with matched incremental parameters;
- a decoder whose evidence tokens are zeroed but whose parameters remain
  active.

---

## 18. Predictor context policies

### `C0_SELF`

The predictor sees only `Z_e_hlt`.

### `C1_NATIVE`

The predictor sees:

- `Z_e_hlt`;
- unbiased HLT particle hidden states from `HE_BASE4`.

This tests whether the original HLT particle view can repair the corresponding
expert representation.

### `C2_ALL`

The predictor sees:

- `Z_e_hlt`;
- all seven HLT expert token banks;
- unbiased HLT particle hidden states.

This allows complementary relation experts to repair one another without
requiring a constituent match.

### `C3_ALL_PARTICLE`

Confirmation-only high-compute variant:

- all HLT expert token banks;
- particle hidden states from `BASE4`, `PT`, `TRACK`, and `REGION`.

This tests whether retaining several relation-specific particle streams
improves translation beyond summary-bank context.

The primary context candidate is `C2_ALL`.  `C0`, `C1`, and `C2` are mandatory
screening comparisons.  `C3` is confirmation-only because of memory cost.

---

## 19. Token normalization

For every offline expert and token shape, fit on `model_train` targets only:

```text
per expert
per slot
per channel
mean
standard deviation
```

Use:

\[
\tilde z_{e,k,d}
=
\frac{z_{e,k,d}-\mu_{e,k,d}}
{\max(\sigma_{e,k,d},10^{-4})},
\]

Predictors operate in normalized coordinates.  Frozen expert and fusion heads
consume inverse-transformed tokens in their original offline coordinate
system.

The primary normalization does not hard-clip finite targets or predictions.
Robust Huber loss handles extremes without erasing rare coordinates that may
matter for background rejection.  Required controls are:

```text
N_UNCLIPPED  finite standardized values, primary
N_CLIP16     clip normalized values to [-16,16]
N_CLIP8      clip normalized values to [-8,8]
```

Every mode records target and prediction counts beyond absolute standardized
values `8`, `16`, and `32`, plus their class and rejection contribution.
Nonfinite values remain execution failures rather than being hidden by
clipping.

The normalizer artifact records counts, zero-variance channels, tail counts,
control clipping fractions, target checkpoint hashes, and token-cache hashes.
Validation and test targets never fit normalization.

---

## 20. Predictor objectives

### 20.0 Uncertainty heads and calibration

Required uncertainty heads are:

```text
U_SLOT      one log variance per slot, U=1, primary simplicity control
U_GROUP4    four contiguous channel-group log variances per slot, U=4
U_DIAGONAL  one log variance per token channel, U=D
```

For `U_GROUP4`, channels are partitioned into four contiguous equal groups in
the frozen token coordinate order.  `D` is divisible by four for every
registered shape.

For notation below, \(s_{k,g(d)}\) is the log variance assigned to channel
\(d\).  Each predictor trains its uncertainty through the heteroscedastic
token loss.  After checkpoint selection, fit one additive log-variance offset
per expert and uncertainty group on `val_design` by minimizing held-out token
Gaussian NLL with predictor/token weights frozen.  Freeze that calibrator
before any `final_select/stack_val` inference.

The calibrated expected RMSE, uncertainty-head type, and group values are
embedded into final consumers.  Calibration uses no labels.  Reports include
coverage/error curves, NLL, error by uncertainty quantile, and whether
uncertainty improves final fusion over an equal-parameter constant-confidence
control.

### 20.1 Uncertainty-weighted normalized token loss

\[
L_{\mathrm{token}}
=
\frac{1}{KD}\sum_{k,d}
\left[
\exp(-s_{k,g(d)})\,
\operatorname{Huber}_{0.5}(\hat{\tilde z}_{k,d}-\tilde z_{k,d})
+s_{k,g(d)}
\right].
\]

`U_SLOT` shares one value across a slot; the other registered heads use the
group map above.

### 20.2 Directional agreement

\[
L_{\mathrm{cos}}
=
\frac{1}{K}\sum_k
\left[
1-\cos(\hat{\tilde z}_k,\tilde z_k)
\right].
\]

### 20.3 Token-relation loss

After row normalization, match within-bank Gram matrices:

\[
L_{\mathrm{rel}}
=
\frac{1}{K^2}
\left\|
\hat Z\hat Z^\top-ZZ^\top
\right\|_F^2.
\]

This preserves relationships among summary slots without particle matching.

### 20.4 Frozen expert-head distillation

Pass oracle and predicted tokens through the frozen offline expert head:

\[
L_{\mathrm{expertKD}}
=
T^2D_{\mathrm{KL}}
\left(
p_e^{off,T}\parallel\hat p_e^T
\right),
\qquad T=2.
\]

### 20.5 Oracle-substitution fusion distillation

For expert \(e\), construct the frozen offline fusion input with predicted
tokens for \(e\) and oracle tokens for every other expert:

\[
L_{\mathrm{swapKD}}
=
T^2D_{\mathrm{KL}}
\left(
p_{\mathrm{fusion,oracle}}^T
\parallel
p_{\mathrm{fusion,HYBRID}_e}^T
\right).
\]

### 20.6 Ground-truth classification

\[
L_{\mathrm{CE}}
=
\operatorname{CE}
\left(
y,
\ell_{\mathrm{fusion,HYBRID}_e}
\right).
\]

### 20.7 Canonical objective

The primary predictor objective is:

\[
L_P =
1.0L_{\mathrm{token}}
+0.25L_{\mathrm{cos}}
+0.10L_{\mathrm{rel}}
+0.50L_{\mathrm{expertKD}}
+0.50L_{\mathrm{swapKD}}
+0.25L_{\mathrm{CE}}.
\]

This is the globally shared simplicity objective.  Different expert banks may
have different downstream sensitivity, so the predeclared mixture registry is:

```text
                token  cosine  relation  expertKD  swapKD  CE
W_TOKEN_ONLY     1.00    0.00      0.00      0.00     0.00 0.00
W_TOKEN_HEAVY    1.00    0.25      0.10      0.25     0.25 0.10
W_CANONICAL      1.00    0.25      0.10      0.50     0.50 0.25
W_TASK_HEAVY     0.50    0.10      0.05      1.00     1.00 0.50
W_LOGIT_ONLY     0.00    0.00      0.00      1.00     1.00 0.50
```

All weights are frozen before results.  A confirmation-only `W_GRADNORM`
control begins from `W_CANONICAL` and applies deterministic gradient-norm
balancing among nonzero terms, with weights clipped to `[0.1,10]` times their
initial value and renormalized to constant sum.

Screen all fixed mixtures on representative `PT` and `TRACK` experts at both
selected shapes.  Carry `W_CANONICAL` plus the best mixture for every expert,
then apply Section 20.8's joint all-predicted bundle selector.  A globally
shared `W_CANONICAL` seven-expert bundle remains the simplicity control.

`W_LOGIT_ONLY` is a semantic control.  If it classifies well while token
fidelity collapses, the report describes it as task distillation, not offline
token reconstruction.

### 20.8 Joint predictor-bundle selector

Per-expert predictor choices are interacting components.  After all eligible
rows finish, publish identity-bound `val_design` predicted tokens, uncertainty,
and expert logits for every seed and candidate.  The primary bundle is chosen
by deterministic beam search:

1. define each expert's candidate as its complete architecture, context,
   objective, uncertainty, normalization, optimizer, HLT-evidence, target-mode,
   and shape tuple;
2. initialize the seven-expert default tuple from the individually best
   one-predicted/six-oracle hybrid candidates;
3. traverse experts in canonical order and expand each retained beam member
   with every eligible candidate for the current expert, leaving later experts
   at their defaults;
4. require the seven target modes to equal one of Section 10.3's locked
   coordinate tuples, then evaluate the complete all-predicted tuple through
   that tuple's exact frozen shape-matched offline fusion at all seeds;
5. retain beam width `32` by mean all-predicted accuracy, the global `0.0001`
   window, mean cross entropy, normalized token error, inference FLOPs,
   parameters, then lexicographic seven-candidate tuple.

No fusion or predictor is retrained during this search.  The final tuple and
all of its seed-specific checkpoint hashes form the immutable predictor
bundle.  Cross-mode tuples without a Stage-E fusion are ineligible rather than
being evaluated through mismatched coordinates.  Compare the result with:

```text
INDIVIDUAL_HYBRID_DEFAULT
GLOBAL_SHARED_CONFIGURATION
ALL_W_CANONICAL
```

The unrestricted final fusion, token refiner, and `J4/J5` training may begin
only after this bundle is locked.  They cannot feed back into predictor-bundle
selection.

---

## 21. Joint all-bank prediction

After individual predictors are selected, train/evaluate:

### Common-view invariant

Every `J1` through `J5` training example obeys:

```text
one canonical jet identity
one R_MULTI replica chosen by the Section-5 epoch/identity cycle
one degraded HLT particle array
all seven HLT experts and every predictor context consume that same array
```

Expert-specific relation tensors and REGION trees are rebuilt from that one
array.  No joint batch may combine tokens produced from different replica
arrays under one event identity.  Per-expert `R_FIXED` or `R_RANDOM`
pretraining remains part of checkpoint lineage, but once checkpoints enter a
joint graph their input policy is globally `R_MULTI`.  `J0` retains the
independent-pretraining controls; its all-bank deployment/evaluation still
uses the one fixed evaluation array for the identity.

### `J0_INDEPENDENT`

Seven independently trained predictors, one per expert.  Their outputs are
combined only by the frozen offline fusion.

### `J1_SHARED_CONTEXT`

Compute HLT evidence banks once, retain independent predictor decoders, and
train all decoders jointly with the sum of per-expert losses plus all-bank
fusion KD.

### `J2_COUPLED_DECODER`

Use all \(7K\) offline target queries in one decoder.  Queries carry
expert/slot identities and self-attend across experts before cross-attending to
HLT evidence.

### `J3_INDEPENDENT_PLUS_ADAPTER`

Freeze the independently selected predictors and train only a deployable final
HLT adapter.

### `J4_BRIDGE_FINETUNE`

Begin from selected HLT evidence encoders and predictors.  Unfreeze:

```text
HLT summary tokenizer
HLT relation encoder and dual-path gates
final N particle-attention blocks, N in {2,4}
predictor
```

Keep offline target encoders, offline expert heads, and offline fusion frozen.
Learning rates are:

```text
unfrozen HLT particle/relation parameters = 5e-5
HLT tokenizer = 1e-4
predictor = 2e-4
```

Train with HLT CE plus the selected token/expert/fusion objective.  The
`N=2/4` choice is screened on `val_design` after each candidate's epoch was
selected on `val_stop`; `N=2` remains the simplicity control.

### `J5_END_TO_END`

Initialize from `J4` and jointly fine-tune:

```text
all HLT expert evidence encoders
all predictors or the selected coupled decoder
token refiner when enabled
deployable final fusion
```

Offline target encoders and all oracle target caches remain frozen.  The
deployable fusion receives labels and offline-target distillation losses.
This is the unconstrained maximum-performance training path and is explicitly
privileged during training.

`J0` is the primary faithful frozen-evidence reconstruction path.  `J1/J2`
test cross-expert coordination, `J3` isolates downstream adaptation, `J4`
tests bridge-aware HLT representations, and `J5` tests the maximum deployable
ceiling.  Their claims remain separate.

---

## 22. Final token consumers

### 22.1 Frozen offline fusion

`PF_FROZEN_Ss` uses the exact frozen `OF_Ss` weights.  It accepts predicted
tokens only after inverse normalization.  This is the primary measure of
faithful target recovery.

### 22.2 Robust offline fusion control

After preliminary predictors exist, clone `OF_Ss` and retrain only its fusion
transformer on deterministic mixtures:

```text
25% all oracle banks
25% exactly one predicted bank
25% independently predicted/oracle bank choices with p=0.5
25% all predicted banks
```

Expert encoders, offline targets, and predictors remain frozen.  This is
`OF_ROBUST_Ss`.

The mixture schedule uses actual predictor outputs rather than generic expert
dropout.  A whole missing-bank token is used only for a separate robustness
control.

### 22.3 Deployable HLT adapter

`HF_ADAPTER_Ss` receives:

- every predicted offline token bank;
- predictor log variance as a learned reliability embedding;
- native unbiased HLT summary tokens;
- optional native HLT expert tokens;
- expert, slot, and source embeddings.

It contains two transformer layers and predicts a residual logit correction:

\[
\ell_{\mathrm{deploy}}
=
\ell_{\mathrm{frozen\ offline\ fusion}}
+\gamma\,\Delta\ell_{\mathrm{HLT}},
\]

where scalar \(\gamma\) initializes to zero.

Required variants:

```text
R0_PREDICTED_ONLY
R1_PREDICTED_PLUS_NATIVE_BASE
R2_PREDICTED_PLUS_ALL_NATIVE_EXPERTS
R3_NATIVE_ONLY_MATCHED_TO_R2
```

Native-evidence robustness modes are:

```text
ND0_NONE        p=0, maximum-ceiling candidate
ND1_FIXED       whole native branch dropped with p=0.10
ND2_CONFIDENCE  corrupt/drop native or predicted banks according to calibrated
                uncertainty while preserving expected corruption rate 0.10
```

`ND1/ND2` are bypass/robustness controls, not expert-diversity training.
`ND0_NONE` is mandatory so robustness regularization cannot silently lower the
ceiling.  Reports show frozen-path, residual-path, and combined logits
separately.

### 22.4 Native-conditioned token refiner

`TR_REFINE_Ss` receives predicted offline tokens, calibrated uncertainty, and
native HLT evidence.  Two cross-attention blocks emit a gated residual in the
offline token coordinate system:

\[
Z^{refined}_{e}
=
\hat Z^{off}_{e}
+\sigma(g_e)\odot\Delta Z_e.
\]

The refiner is trained through frozen expert/fusion token, logit, and CE
losses.  It has no oracle input at inference.  Required controls are:

```text
TR0_NONE
TR1_NATIVE_BASE
TR2_ALL_NATIVE
TR3_ZERO_NATIVE_SHAPE
```

Token fidelity is reported before and after refinement so a classifier-driven
off-manifold correction cannot be mistaken for improved reconstruction.

### 22.5 Unrestricted maximum-performance fusion

`HF_UNRESTRICTED_Ss` concatenates:

- predicted or refined offline token banks;
- all selected native HLT expert banks;
- calibrated uncertainty/reliability embeddings;
- expert, slot, source, and availability embeddings.

Required evidence variants are:

```text
F_TOKEN_ONLY
F_TOKEN_PLUS_EXPERT_LOGITS
F_TOKEN_ONLY_MATCHED
```

`F_TOKEN_PLUS_EXPERT_LOGITS` additionally appends one logit token for every
expert/source:

- native HLT expert logits;
- frozen offline expert-head logits computed from predicted or refined
  offline tokens.

Each 10-logit vector passes through `LayerNorm(10) + Linear(10,256)` and gains
its expert and `native_hlt`/`predicted_offline` source embeddings.  These
logits introduce no inference leakage: they are computed only from deployable
HLT-side or predicted token banks.  They remain differentiable in `J5` unless
their originating component is explicitly frozen.

`F_TOKEN_ONLY_MATCHED` receives no expert logits.  It adds an active
token-side residual MLP whose hidden width is selected from
`{64,128,192,256,320,384}` to minimize parameter mismatch to
`F_TOKEN_PLUS_EXPERT_LOGITS`, then inference-FLOP mismatch, then smaller width.
This prevents any gain from being attributed merely to the extra logit-token
parameters.

It uses four pre-normalized transformer layers, width 256, eight heads, MLP
expansion four, and a learned class token.  It predicts logits directly and
does not add them to frozen offline logits.

Reliability-conditioned gating is implemented as a bounded scalar per
expert/token multiplying only that token's residual contribution after
attention; gates do not modify masks or create hard event rejection.

Train `HF_UNRESTRICTED` with `ND0_NONE`, `ND1_FIXED`, and
`ND2_CONFIDENCE` for all three evidence variants after the joint predictor
bundle is locked.  It is the named maximum-performance classifier.
`PF_FROZEN`, `OF_ROBUST`, and `HF_ADAPTER` remain the faithful and constrained
controls.

---

## 23. Campaign matrix

The campaign is staged to avoid an uncontrolled full Cartesian product while
still running every important ablation.

### Stage A: data, degradation, and exact baselines

Run:

```text
offline split/validation-role/cache/scale-pool audit
four-replica HLT-v3 degradation and transformed-input audit
offline/shared-HLT relation and REGION normalizer fit/audit
real-Weaver explicit-uu parity
O_BASE
H_BASE
O_WIDE
H_WIDE
O_FULLREL
```

The complete-graph monolithic capacity controls are resolved later because
their matching targets do not exist until the final graph is selected.
No scientific arm depends on a baseline exceeding a performance threshold.
The access audit proves that no Stage B-K component worker can open
`final_select/stack_val`.

### Stage B: offline expert and token-shape screen

At seed `101`, train:

```text
7 experts x 7 canonical shapes = 49 CE/B_CONCAT screen rows
token-pooling controls at S1_128, S8_128, S16_128 for BASE4, PT, TRACK
TOK_MULTI_DEPTH at S8_128 for BASE4, PT, TRACK, REGION
B_DUAL_FIXED and B_DUAL_GATED at S8_128 for all seven experts
expert-loss registry at S8_128 for BASE4, PT, TRACK, REGION
initialization/LR/dropout controls for BASE4, PT, TRACK, REGION
```

The optimization screen uses `INIT_SCRATCH`, `INIT_OBASE_PARTICLE`, and
`INIT_ATTACH_AFTER_PRETRAIN`; learning rates `2e-4`, `5e-4`, and `1e-3`; and
dropout `0.0` and `0.1`.  The full Cartesian product is run only for `PT` and
`TRACK`; the winner, scratch reference, and attach-after-pretrain reference
are run for the other experts.  Training workers can read only `model_train`
and `val_stop`; they cannot load `val_design` or `stack_val`.  After every row and
model-validation checkpoint is immutable, the separate architecture selector
may read cached `val_design` predictions to choose the optimization candidate.
This screen cannot change the fixed 40-epoch budget.

Every row completes its fixed training schedule unless execution is invalid.

### Stage C: offline fusion and complementarity

At every canonical uniform shape, train frozen-token:

```text
logit mean
trained logit linear
pooled MLP
token transformer
```

At `S8_128` and the seed-101 best uniform shape, also train:

```text
light fine-tune
full fine-tune
grouped-head relation control
seven-unbiased token fusion
seven-unbiased logit ensemble
dual-path base4-only and zero-relation controls
S1 fixed relation scale
S2 bounded learned relation scale
S3 relation auxiliary objective
S4 restricted-field controls
S5 cross-covariance control
```

Confirm all seven individual `T0_PURE` experts and canonical token fusion for
all seven registered uniform shapes across seeds `101,202,303`.  No seed-101
performance prefilter is permitted.  Only after all 147 expert-shape-seed
rows and 21 fusion-shape-seed rows are immutable may Section 13 select
`SHAPE_HIGH` and `SHAPE_COMPACT`.  At those two shapes, train every carried
expert-loss candidate for all experts and seeds, then run the joint
expert-loss selector.  Freeze `HET_PHYSICS`, `HET_SELECTED`, and `HET_BEAM`
only after those complete three-seed inputs exist, then train their
shape-specific fusions.

### Stage D: native HLT controls

For `S1_128`, `SHAPE_COMPACT`, `SHAPE_HIGH`, and the three heterogeneous
assignments, train:

```text
all seven HE_SCRATCH_CE experts
HF_NATIVE
native logit mean
native trained logit fusion
unbiased multi-expert HLT controls
```

At `SHAPE_COMPACT` and `SHAPE_HIGH`, screen `HE_OFFLINE_INIT` and
`HE_DUAL_OBJECTIVE`, the registered dual-objective weights,
`V_MEASUREMENT_EMBED`, and realization policies `R_FIXED`, `R_MULTI`, and
`R_RANDOM`.  `R_MULTI` is the primary nominal multi-realization policy,
`R_RANDOM` is the high-performance severity-randomized candidate, and
`R_FIXED` remains the exact single-realization label-exposure control.

Confirm `H_BASE`, `H_WIDE`, every selected individual expert, both selected
native fusions, and the chosen HLT encoder modes at three matched pipeline
seeds.

### Stage E: locked bridge pilot and target co-design

For every selected uniform/heterogeneous target shape and pipeline seed:

1. train the exact Section-10.1 `PILOT_T0` against frozen `T0_PURE`;
2. lock its HLT encoder, predictor, target, and normalizer hashes;
3. co-design `T1_ANCHORED_BRIDGE`, `T1_TASK_BRIDGE`, and `T2_PROJECT`;
4. compute offline noninferiority and instance-content certification on
   `val_design`;
5. run Section 10.3's joint target-coordinate beam search on already eligible
   candidates using `val_design`;
6. publish distinct semantic versions/fusions for the top four mixed and
   homogeneous target tuples and rebuild their complete caches.

`T3_LOGIT` is trained as a task-distillation control.  It is never labeled
token reconstruction.  Full predictor architecture screening cannot start
until all required Stage-E caches are complete.

### Stage F: predictor architecture/context screen

Use representative experts:

```text
BASE4
PT
TRACK
REGION
```

At `SHAPE_COMPACT` and `SHAPE_HIGH`, seed `101`, run:

```text
A0 x C0
A1 x C0
A2 x C0
A3 x C0,C1,C2
A4 x C0,C1,C2
```

All rows use `W_CANONICAL`, `U_SLOT`, `N_UNCLIPPED`, `T0_PURE`, and the
selected native HLT evidence mode.  This is nine rows per shape per
representative expert.

Select the best direct architecture/context and best gated
architecture/context by frozen hybrid-fusion `val_design` accuracy, then lower
token error, parameters, and run ID.  For those two families on `PT` and
`TRACK`, also screen predictor learning rates `{2e-4,5e-4,1e-3}` and residual/
attention dropout `{0.0,0.1}`.  Carry the selected optimizer configuration plus
the fixed `5e-4/0.0` simplicity control.

### Stage G: predictor loss, uncertainty, and normalization screen

For selected direct and gated predictor families, run the complete predictor
loss registry, `U_SLOT`, `U_GROUP4`, and `U_DIAGONAL`, and `N_UNCLIPPED`,
`N_CLIP16`, and `N_CLIP8` on `PT` and `TRACK` at both selected shapes.
`W_GRADNORM` is fitted only on `model_train` and may not read validation
metrics while adapting weights.  Select by frozen hybrid/all-predicted fusion
utility.  `W_LOGIT_ONLY` is never eligible to be called faithful token
recovery, but remains a task-performance control.  Hard clipping is a control,
not the default.

### Stage H: all-expert predictor confirmation

Train the selected direct and gated families for all seven experts at seeds:

```text
101, 202, 303
```

Materialize them at `SHAPE_COMPACT`, `SHAPE_HIGH`, and every per-expert
`(K_e,128)` required by `HET_PHYSICS`, `HET_SELECTED`, or `HET_BEAM`;
heterogeneous rows reuse the selected architecture family but train
shape-specific weights.

Also train:

```text
C3_ALL_PARTICLE for PT, TRACK, REGION
matched widened residual-MLP controls
zero-evidence decoder controls
T0_PURE, eligible T1_ANCHORED_BRIDGE, T1_TASK_BRIDGE, and T2_PROJECT targets
selected HE_SCRATCH_CE, HE_OFFLINE_INIT, and HE_DUAL_OBJECTIVE evidence modes
```

Cache every eligible candidate's identity-bound `val_design` outputs and run
the Section-20.8 joint predictor-bundle beam selector.  The resulting
seven-expert configuration is selected from three-seed all-predicted utility
and is shared by all pipeline seeds; only learned weights and seed-bound
artifact hashes differ.  Per-seed or independent per-expert final selection is
forbidden.  Globally shared and individual-hybrid configurations remain
controls.

### Stage I: reconstruction and oracle substitutions

Evaluate, without retraining:

```text
no reconstruction
identity-projected HLT tokens
one predicted bank + six oracle banks
one oracle bank + six predicted banks
all predicted banks
all oracle banks
wrong-event offline targets
within-class wrong-event targets
slot-permuted targets
expert-bank-swapped targets
```

Wrong-event controls are evaluation-only and never enter model selection.
They run on `val_design`; Stage I cannot open `stack_val`.

### Stage J: joint prediction and final adapters

Train:

```text
J0_INDEPENDENT
J1_SHARED_CONTEXT
J2_COUPLED_DECODER
J3_INDEPENDENT_PLUS_ADAPTER
J4_BRIDGE_FINETUNE
J5_END_TO_END
TR_REFINE
PF_FROZEN
OF_ROBUST
HF_ADAPTER
HF_UNRESTRICTED
  x F_TOKEN_ONLY,F_TOKEN_PLUS_EXPERT_LOGITS,F_TOKEN_ONLY_MATCHED
  x ND0_NONE,ND1_FIXED,ND2_CONFIDENCE
```

Run these for both selected uniform shapes and eligible heterogeneous
assignments, then choose `SHAPE_BRIDGE`.  `J0` is the faithful independently
reconstructed reference; `J5` and `HF_UNRESTRICTED` are explicitly the
maximum-performance candidates rather than representation-faithfulness
claims.  Every `J1-J5` row uses one common `R_MULTI` degraded array per
identity for all experts, regardless of their pretraining policies.

### Stage K: degradation and robustness ablations

Using locked checkpoints where the input contract permits evaluation, report:

```text
D_OFFLINE_IDENTITY
D_KIN_ONLY
D_TRACK_ONLY
D_MISSING_ONLY
D_MILD
D_NOMINAL
D_SEVERE
D_LEGACY_V1
D_LEGACY_V2
```

Models are not retuned per degradation profile in the primary robustness
evaluation.  Separate retrained-domain controls are clearly labeled.
All Stage-K robustness diagnostics use `val_design` or declared training
splits, never `stack_val`.

### Stage L: 500k confirmation and bounded scale shortlist

Confirm all primary baselines, both uniform-shape finalists, heterogeneous
finalists, native HLT fusion, frozen reconstruction, token refiner, constrained
adapter, and unrestricted fusion across three matched pipeline seeds.
Every complete graph definition and component choice is immutable.  Stage L
cannot open `stack_val`; it ranks complete 500k graphs on `val_design`.

Build a predeclared bounded scale shortlist:

```text
top 3 by three-seed mean balanced accuracy
union
top 3 by three-seed mean Jeffreys-smoothed mean log rejection
remove duplicate graph IDs
maximum shortlist size = 6
```

Each ranking uses its Section-25 finalist metric and tie rules, except its
population is `val_design`.  If fewer than three graphs are eligible, include
all eligible graphs.  The union always emits at least the best available graph
and proceeds even if every graph loses to its baseline.  Write:

```text
selection/locked_scale_shortlist.json
```

The shortlist locks complete training and inference graph definitions but no
3M checkpoint and no final finalist identity.  For 500k reporting, resolve
graph-specific capacity and label-exposure controls for shortlisted graphs;
these controls cannot change shortlist membership.

### Stage M: predeclared 3M scale-up

Retrain every shortlisted graph definition on `scale_train`, across seeds
`101,202,303`, preserving every architecture, target semantics, loss,
degradation, replica, and inference decision.  Epoch selection may use
`val_stop`; it cannot reopen architecture selection.  Recompute complete
graph capacity controls if parameter counts are unchanged only to bind the new
training lineage, not to select a different topology.

Each locked graph definition contains both a training graph and a deployable
inference graph.  Stage M therefore retrains its required offline experts and
offline fusion, builds authenticated offline targets for all `scale_train`
identities, then trains the HLT encoders, predictors/refiner, and final
consumer.  Components shared across shortlisted graphs are trained once per
seed and referenced by every applicable manifest.  Offline teachers and target
generation are privileged training dependencies and remain excluded from
deployable inference accounting.  With no duplicates, the declared maximum is
six graph definitions times three seeds.

Stage M refits all Section-7 train-derived normalizers on `scale_train`,
refits target-token normalizers on the new scale targets, and refits the
label-free uncertainty calibrator on `val_design`.  These refits instantiate
the locked recipe and cannot change the graph definition.
No Stage-M worker may open `stack_val`.

### Stage N: one-time scale-finalist selection, locks, and final test

After every shortlisted 3M graph and seed is immutable:

1. run deployable classification inference on `final_select/stack_val`
   without constructing or consuming any oracle offline target, publishing
   label-free identity-bound logits/probability shards;
2. join labels from the authenticated identity-sorted
   `final_select_label_manifest`, publish separate metric artifacts, and
   select `ACCURACY_FINALIST` and `REJECTION_FINALIST` from the 3M graphs with
   the exact Section-25 selectors;
3. write immutable `selection/locked_scale_finalists.json`;
4. train/resolve the required 3M graph-specific capacity and `H_BASE_LONG`
   controls for the one or two locked finalists;
5. generate post-selection `final_select` oracle diagnostics and required
   `final_test` offline targets from the exact Stage-M teacher hashes;
6. write `selection/final_test_execution_lock.json`, binding those controls,
   targets, inputs, checkpoints, and both finalist-selection traces;
7. evaluate the locked finalists, named baselines, and required controls on
   the common 300,000-jet `final_test` exactly once.

Steps 1-2 are the only pre-lock access to `stack_val`.  Prediction shards are
label-free; only the selector process may join their canonical identities to
the locked label manifest.  Steps 4-5 are selection-ineligible and cannot
replace a finalist.  The two finalist selectors may choose the same graph.

---

## 24. Training protocol

Unless a component declares otherwise:

```text
optimizer = AdamW
betas = (0.9,0.999)
weight decay = 1e-4
gradient clipping = 1.0
maximum epochs = 40
warm-up = exact integer rule below
post-warm-up schedule = cosine
minimum learning rate = 1e-5
mixed precision = BF16 on GH200
num_workers = 0
```

Learning rates:

```text
complete ParT/expert training = 1e-3
frozen-token fusion = 5e-4
predictor = 5e-4
robust fusion = 2e-4
final residual adapter = 2e-4
lightly unfrozen tokenizer parameters = one tenth of fusion LR
```

Primary effective batch sizes:

```text
single expert = 128
frozen token fusion = 512
predictor = 256
joint all-bank predictor = largest power of two <= 128 that fits
```

If memory forces a smaller microbatch, gradient accumulation preserves the
effective batch.  A resource profile records the exact update count and
schedule.

For total optimizer-update count \(T>0\), the number of warm-up updates is
`min(T,max(1,ceil(0.05*T)))`.  The learning rate reaches its base value on the
last warm-up update; if `T=1`, that sole update uses the base value and there
is no cosine segment.  `T=0` is an invalid training row.  Resume recomputes
neither \(T\) nor the schedule.

### Optimization and attachment controls

The primary fixed-protocol reference is:

```text
INIT_SCRATCH
learning_rate = 1e-3
dropout = 0.0
```

The registered optimization candidates are:

```text
INIT_SCRATCH
INIT_OBASE_PARTICLE
INIT_ATTACH_AFTER_PRETRAIN
learning_rate in {2e-4,5e-4,1e-3}
dropout in {0.0,0.1}
```

For an offline expert, `INIT_OBASE_PARTICLE` copies compatible particle-encoder
weights from the seed-matched `O_BASE` checkpoint and randomly initializes
relation- and token-specific parameters.  HLT warm starts are separately
named `HE_OFFLINE_INIT` and copy the corresponding offline expert, never
`O_BASE` by implication.

`INIT_ATTACH_AFTER_PRETRAIN` first trains the ordinary particle classifier,
then attaches the relation path and tokenizer.  It freezes the particle
encoder for the first five attachment epochs, unfreezes the last four particle
blocks for the next five, and unfreezes the complete graph for the remaining
30.  The row still contains exactly 40 attachment-training epochs; pretraining
cost and label presentations are recorded and included in `O_BASE_LONG` or
`H_BASE_LONG` matching.  The screen cannot alter warm-up, checkpoint
selection, or total epochs after observing results.

### No performance-based run termination

Every scientific training row runs all 40 epochs.  Validation selects a
checkpoint but does not stop the run.  Worse-than-expected accuracy, negative
gain, or failure to meet a scientific success criterion is never an execution
failure and never blocks unrelated future rows.

Execution may fail closed only for:

- nonfinite loss, gradients, logits, or required metrics;
- corrupted or stale artifacts;
- schema/provenance mismatch;
- unavailable required resources;
- invalid masks/categories/identities;
- explicit scheduler or hardware failure.

Checkpoint selection copies the relational campaign's global
maximum-accuracy-window rule:

1. maximum `val_stop` accuracy;
2. retain epochs within `0.0001`;
3. lower cross entropy;
4. earliest epoch.

Neither `val_design` nor `stack_val` ever selects an epoch.

Training seeds are:

```text
screen = 101
confirmation = 101,202,303
```

At a fixed seed, all comparable rows receive the same training identity order.

### Pipeline-seed lineage

A primary confirmation row with pipeline seed \(s\) uses one coherent lineage:

```text
offline expert seed        = s
offline fusion seed        = s
offline target-cache seed  = s
native HLT expert seed     = s
predictor seed             = s
refiner/adapter seed       = s
final consumer seed        = s
```

The four HLT degradation replicas are fixed data artifacts and do not change
with model seed.  Their deterministic epoch/identity cycling is bound in the
campaign spec.  Screening may reuse seed-101 teachers, but such rows are
explicitly `fixed_teacher_screen` and cannot substitute for the matched-seed
primary confirmation.

For variance decomposition only, predictor seeds `202` and `303` may target
the seed-101 offline teacher while holding all else fixed.  These
`FIXED_TEACHER_CONTROL` rows are controls, not primary estimates.

Each selected per-expert predictor bundle is immutable and lists its pipeline
seed, teacher and HLT checkpoint hashes, target-cache and normalizer hashes,
shape, target mode, architecture, context, objective weights, uncertainty
head, normalization/clipping mode, optimizer/dropout configuration, HLT
encoder mode, and pretraining realization policy.  The bundle also records
one graph-level joint/deployment input policy, fixed to `R_MULTI`; this
graph-level field overrides per-expert pretraining policies whenever banks are
evaluated together.

The per-expert configuration is identical at primary seeds `101,202,303`;
each seed materializes a separate weight/hash bundle under that configuration.
Mixing seeds or bundle members at inference requires a separately registered
hybrid control.

---

## 25. Screening and selection rules

Every registry row has exactly one role:

```text
reference_baseline
capacity_control
architecture_control
scientific_candidate
semantic_control
robustness_control
```

Scientific selections consider only `scientific_candidate` rows.  Controls
are compared but cannot become the nominal scientific winner.

Within any component/design candidate pool, use `val_design`:

1. maximize the primary `val_design` accuracy or paired gain named by that
   stage;
2. retain rows within `0.0001` of the global maximum;
3. minimize cross entropy;
4. minimize normalized token error when the stage predicts tokens;
5. minimize measured FLOPs;
6. minimize parameter count;
7. choose lexicographically smaller run ID.

The sole exception is Stage N: it receives already locked, scale-trained
complete graph
definitions and applies the two explicit `stack_val` finalist selectors below.
No Stage-N result may feed back into a component choice.

The selector always emits a best available row, even when every row loses to
its baseline.  It also records:

```text
gain_positive
capacity_control_reproduces_gain
all_candidates_worse_than_baseline
```

These flags affect interpretation, not DAG continuation.

### Bounded 500k scale shortlist

Stage L ranks only complete deployable 500k graph definitions with all three
matched seeds.  Using `val_design`, create:

```text
ACC_SCALE_TOP3 = first three graphs under the accuracy ordering
REJ_SCALE_TOP3 = first three graphs under the Jeffreys mean-log-rejection
                 ordering
SCALE_SHORTLIST = canonical graph-ID union(ACC_SCALE_TOP3,REJ_SCALE_TOP3)
```

Accuracy uses the ordinary `0.0001` window and tie rules.  Rejection uses the
same 18-term score, `0.005` window, and tie rules as the final rejection
selector.  Duplicate graph IDs are removed without changing either source
ranking.  If a source list has fewer than three eligible graphs, use all of
them.  Scientific underperformance never removes a graph; only incomplete,
nonfinite, nondeployable, or lineage-invalid rows are ineligible.  The
shortlist has size one through six and is immutable before Stage M.

### Dual deployable finalists

Stage N runs two independent deterministic selectors over the same eligible
3M deployable graphs in `locked_scale_shortlist.json`.

`ACCURACY_FINALIST` uses the ordinary candidate ranking above with
`stack_val` macro balanced accuracy as its primary quantity.  Because
`stack_val` has exact equal class counts this equals ordinary overall accuracy;
both values and cross entropy are serialized.

`REJECTION_FINALIST` maximizes:

\[
\frac{1}{18}\sum_{c\in\mathrm{nine\ signals}}
\sum_{\epsilon_s\in\{0.30,0.50\}}
\log R_{c,\epsilon_s}^{\mathrm{selection}},
\]

then retains rows within `0.005` of the global maximum, maximizes accuracy,
minimizes cross entropy, minimizes analytical batch-1 inference FLOPs,
minimizes parameters, and uses lexicographic run ID.  The `0.005` window is
fixed before any campaign predictions exist.  The two selectors may emit the
same row; no artificial second model is required.

Only rows that are deployable without offline particles or oracle targets,
complete at all three matched seeds, finite on all required selector metrics,
and valid under the exact current source/cache contracts are eligible.
Their `stack_val` prediction artifacts may have only HLT input, shortlisted
3M checkpoint, and deployable-normalizer parents; an oracle target/cache parent
is forbidden.  Each shard contains identities, graph/seed IDs, logits, and
probabilities but no label field.  The selector joins labels from the
authenticated identity-sorted `final_select_label_manifest`, records that
manifest hash as a metric parent, and emits metric artifacts separately from
prediction shards.
Displayed positive-infinite zero-background rejection is permitted because
the selector uses the declared finite Jeffreys-smoothed quantity.  Poor
rejection relative to a baseline is recorded but never prevents selection or
scale-up.

---

## 26. Metrics

### 26.1 Classification

For every classifier:

- 10-class accuracy;
- cross entropy;
- macro per-class accuracy;
- per-class efficiency;
- one-vs-rest AUC;
- confusion matrix;
- Brier score;
- deterministic 15-bin top-label multiclass ECE;
- QCD-versus-each-signal rejection at 30% and 50% signal efficiency.

The 15 ECE bins are `[0,1/15), [1/15,2/15), ..., [14/15,1]`; confidence is the
largest softmax probability and correctness is the corresponding top-label
indicator with lowest class index breaking an exact logit tie.  Empty bins
contribute zero.  ECE is the sample-count-weighted absolute difference between
mean confidence and accuracy in each nonempty bin.  This is top-label
multiclass ECE, not one-vs-rest ECE.

For signal class \(c\), the QCD discriminant is
`p_c/(p_c+p_QCD)`.  A jet passes when its score is greater than or equal to the
threshold.  Evaluate every threshold in `{+infinity} U unique_observed_scores
U {-infinity}` and choose the threshold whose achieved signal efficiency is
closest to the target; break ties by larger achieved efficiency, then larger
threshold.  Report the achieved, not nominal, signal efficiency.  Background
efficiency is the passing-QCD count divided by all QCD jets and rejection is
its reciprocal.  If zero QCD jets pass, displayed rejection is positive
infinity.  The finite Stage-N selector uses the fixed Jeffreys-smoothed
quantity:

\[
\widehat\epsilon_b^{\mathrm{selection}}
=
\frac{n_{\mathrm{pass}}+0.5}{N_{\mathrm{QCD}}+1},
\qquad
R^{\mathrm{selection}}
=
\frac{1}{\widehat\epsilon_b^{\mathrm{selection}}}.
\]

Exact unsmoothed counts, efficiency, and rejection remain the reported
physics quantities.  The smoothing is used only for stable candidate ranking
and gives distinct values for zero, one, and two passing QCD jets.

These definitions, class order, endpoint inclusion, and all tie rules are
serialized into the RETB campaign spec and cannot vary by model.

### 26.2 Reconstruction

For each expert token bank:

- normalized Huber loss;
- normalized RMSE and MAE;
- cosine similarity per slot;
- Gram-matrix error;
- predicted uncertainty mean and calibration;
- empirical squared error by uncertainty quantile;
- frozen expert-logit KL;
- frozen expert prediction agreement;
- hybrid frozen-fusion KL;
- hybrid frozen-fusion accuracy;
- all-predicted frozen-fusion accuracy.

### 26.3 Complementarity

- best-single versus fusion gain;
- logit-mean versus token-fusion gain;
- relation-expert versus seven-unbiased fusion gain;
- pairwise disagreement and error correlation;
- leave-one-out loss;
- subset-readout contribution;
- CKA and attention-overlap diagnostics.

### 26.4 Bridge gains

Define:

\[
G_{\mathrm{native}}
=
\operatorname{Acc}(\mathrm{all\ predicted})
-
\operatorname{Acc}(\mathrm{HF\_NATIVE}),
\]

\[
R_{\mathrm{oracle}}
=
\frac{
\operatorname{Acc}(\mathrm{all\ predicted})
-
\operatorname{Acc}(\mathrm{HF\_NATIVE})
}{
\operatorname{Acc}(\mathrm{all\ oracle})
-
\operatorname{Acc}(\mathrm{HF\_NATIVE})
}.
\]

If the oracle denominator is nonpositive, `R_oracle` is undefined and is
reported as such rather than clamped.

### 26.5 Efficiency

- complete parameters and trainable parameters;
- analytical FLOPs;
- measured examples/second;
- peak GPU memory;
- token-cache bytes per jet;
- training GPU-hours;
- single-jet and batch latency.

---

## 27. Statistical reporting

Final comparisons use paired event predictions and matched training seeds.

For every locked finalist versus its named baseline report:

- per-seed difference;
- mean and sample standard deviation across seeds;
- paired absolute accuracy difference;
- paired class-stratified bootstrap 95% interval;
- McNemar discordant counts;
- relative error reduction;
- per-class paired differences;
- per-signal smoothed rejection intervals at 30% and 50% efficiency;
- mean-log-rejection interval;
- paired interval for the finalist-minus-baseline mean-log-rejection
  difference;
- when the two finalists differ, their paired rejection-score difference.

The paired bootstrap copies the relational deterministic definition:

```text
10,000 resamples
fixed serialized seed
paired sampling unit = jet identity
classes sampled in their fixed balanced counts
2.5% and 97.5% quantiles with the locked linear quantile method
```

For every rejection resample, use the same sampled jet identities for both
models, recompute the operating threshold and achieved signal efficiency, and
apply the Section-26 Jeffreys smoothing.  Per-signal intervals are computed in
log rejection and exponentiated at their endpoints.  Exact unsmoothed central
counts/rejections are shown alongside these finite intervals.  The
mean-log-rejection bootstrap recomputes all 18 terms inside each resample; it
does not average independently bootstrapped endpoints.

All final predictions are identity-bound so statistics can be recomputed
without inference.

---

## 28. Semantic and causal controls

The final report includes:

### Relation controls

- zero each expert's additional relation while retaining model parameters;
- shuffle the relation within a jet;
- use a wrong event's relation with matched valid-particle count;
- swap directional endpoints where defined;
- compare fixed versus learned bias scale.

### Token controls

- permute slot order without permuting slot embeddings;
- replace a bank with its within-class mean;
- replace a bank with a wrong event from the same class;
- replace a bank with a wrong event without class matching;
- zero one bank;
- add matched Gaussian noise;
- compare oracle/predicted token norm and covariance.

### Predictor controls

- zero HLT evidence;
- shuffle HLT evidence between events;
- remove native particle context;
- remove all non-corresponding expert banks;
- compare direct and gated prediction;
- evaluate `W_LOGIT_ONLY` for off-manifold token behavior.

### Native-branch bypass controls

- native branch removed;
- reconstructed branches removed;
- native branch dropped at evaluation;
- residual scalar \(\gamma=0\);
- source embeddings swapped.

No semantic perturbation is used to select a checkpoint.

The wrong-event relation control uses exact valid-particle multiplicity and
never self-matches.  A multiplicity stratum containing only one jet cannot be
deranged under those constraints, so the globally frozen policy excludes that
stratum from both the perturbed row and its paired active-reference row.  The
excluded identity hash, excluded count, eligible count, and eligible fraction
are serialized.  All other semantic controls retain the complete
`val_design` population.  Relation-zero is evaluated independently for each
of `PT`, `TRACK`, `PID`, `CHARGE`, `DENSITY`, and `REGION`.

---

## 29. Final-test seal

Stage L first writes:

```text
selection/locked_scale_shortlist.json
```

This source-bound object contains the union of the top-three 500k accuracy and
top-three 500k rejection graph definitions, duplicate removal, both
`val_design` ranking traces, and no 3M checkpoint or final finalist identity.
Stage M must instantiate every listed definition exactly.

After one-time deployable `stack_val` inference, Stage N writes:

```text
selection/locked_scale_finalists.json
```

It contains:

- campaign-spec and source hashes;
- shortlist hash and complete 500k/3M candidate lineage;
- split/validation-partition, scale-pool, HLT replica/cache, and
  realization-policy hashes;
- degradation profile/version/parameter hash;
- offline/shared-HLT/scale relation and REGION normalizer population, recipe,
  and content hashes;
- offline and HLT checkpoint hashes;
- target-token and token-normalizer hashes;
- `SHAPE_HIGH`, `SHAPE_COMPACT`, `SHAPE_BRIDGE`, and each heterogeneous
  assignment;
- token target mode, pilot-parent hashes, offline noninferiority, and
  bridge-content certification per expert;
- joint offline-loss, heterogeneous-allocation, and predictor-bundle beam
  traces;
- complete selected predictor bundle per expert and pipeline seed;
- selected loss, uncertainty, normalization, context, HLT encoder, and
  pretraining realization policy per expert;
- graph-level joint/deployment realization policy fixed to `R_MULTI`;
- selected refiner, consumer, and native-dropout mode;
- selected token-only/logit evidence mode and its matched control;
- complete deployed-graph parameter and inference-FLOP accounting separately
  for both finalist graphs;
- `ACCURACY_FINALIST` and `REJECTION_FINALIST` row IDs and deterministic
  3M `stack_val` selection traces;
- every shortlisted 3M checkpoint hash;
- every label-free 3M `stack_val` prediction-shard and shard-manifest hash;
- the authenticated `final_select` label-manifest hash;
- every selector metric-artifact hash and both complete finalist-selection
  trace hashes;
- every pre-final confirmation-prediction hash;
- deterministic selection reasons;
- flags for positive/negative gains;
- 500k and 3M pre-`stack_val` confirmation metrics only.

This lock authorizes, but does not yet contain, post-selection oracle target
generation.  After required finalist controls and post-selection target caches
exist, write:

```text
selection/final_test_execution_lock.json
```

It has `locked_scale_finalists.json` as a parent and additionally binds:

- scale-trained offline teacher/fusion and target-normalizer hashes;
- post-selection `final_select` oracle-diagnostic cache hashes;
- final-test target/input/sidecar hashes;
- graph-specific capacity and `H_BASE_LONG` control hashes;
- all final-test-eligible scientific/control row IDs.

Final evaluation rejects:

- absent or stale finalist or execution lock;
- a new checkpoint;
- a different source snapshot;
- mismatched target/cache identities;
- a different degradation artifact;
- unregistered final rows.

All locked rows may be reported on final test.  Test performance never selects
a replacement.  No artifact created before `locked_scale_finalists.json` may
contain final-test model outputs.

---

## 30. Artifact and provenance contract

Campaign root:

```text
checkpoints/relation_expert_token_bridge/<campaign_id>/
```

Required structure:

```text
campaign_spec.json
inputs/
  split_manifest.json.gz
  final_select_label_manifest.json.gz
  validation_partition_manifest.json.gz
  scale_train_manifest.json.gz
  split_audit.json
  scale_train_audit.json
  raw_input_schema.json
  hlt_v3_profile.json
  hlt_v3_degradation_audit.json
  hlt_replica_manifest.json
  hlt_replicas/
    replica_0/
    replica_1/
    replica_2/
    replica_3/
  region_tree/
  normalization/
    offline_500k/
      relation.json
      region.json
    hlt_shared_500k/
      relation.json
      region.json
    offline_scale/
      relation.json
      region.json
    hlt_shared_scale/
      relation.json
      region.json
    token/
registry/
  expert_registry.json
  expert_loss_registry.json
  token_registry.json
  token_shape_registry.json
  target_mode_registry.json
  relation_bias_topology_registry.json
  bridge_pilot_registry.json
  bridge_content_contract.json
  fusion_registry.json
  predictor_registry.json
  loss_registry.json
  uncertainty_registry.json
  realization_policy_registry.json
  degradation_domain_seed_registry.json
  normalizer_population_registry.json
  deployed_graph_registry.json
  campaign_stage_registry.json
  global_determinism.json
offline_experts/
offline_fusions/
hlt_experts/
native_hlt_fusions/
offline_targets/
post_selection_oracle_targets/
bridge_pilots/
bridge_target_candidates/
predictors/
predictor_bundles/
bundle_searches/
joint_predictors/
token_refiners/
final_adapters/
unrestricted_fusions/
capacity_controls/
scale_up/
selection_predictions/
  stack_val/
evaluations/
  stack_val_selection_metrics/
selection/
reports/
job_ledgers/
```

Every immutable JSON object uses canonical sorted JSON, a content hash, parent
hashes, and source snapshot.  Large shards use SHA-256 manifests and atomic
publication.

No artifact may be reused merely because a path exists.  Reuse requires exact
contract, parent, source, identity, replica/realization policy, target mode,
shape, dtype, pipeline seed, and content-hash agreement.

Changing token semantics, target coordinates, HLT degradation, relation
definitions, replica policy, uncertainty interpretation, deployed graph, or
selection rules requires a schema/contract version change.

---

## 31. Storage and runtime policy

The bootstrap measures rather than guesses:

- HLT-v3 compressed bytes per jet;
- REGION sidecar bytes per jet;
- one expert target bank bytes per jet at each registered \((K,D)\) and dtype;
- logits and identity bytes per jet;
- checkpoint sizes;
- projected peak concurrent storage;
- CPU degradation throughput;
- GPU expert and predictor throughput.

Persistent \(N\times N\) pair matrices and attention maps are forbidden.
Store only compact REGION sidecars and aggregated attention diagnostics.

Offline target caches are built only for selected token shapes and the locked
top-four/homogeneous target-coordinate tuples.  Screening token banks may be
streamed directly into fusion cache shards and deleted after authenticated
aggregation.

Storage projection before Stage N excludes `stack_val` and final-test model
outputs because their caches do not yet exist.  Stage N admits only the
label-free deployable `stack_val` selection shards and separate selector
metrics before finalist locking.  Post-selection oracle targets are projected
and admitted only after `locked_scale_finalists.json`; final-test model outputs
remain forbidden until the execution lock.

After successful training retain:

- selected checkpoint;
- training curve and metrics;
- configuration/provenance;
- no stale optimizer state unless the run is explicitly resumable.

The top-level submitter calculates free-space reserve and prints a projection.
Resource insufficiency may block jobs; poor scientific performance may not.

---

## 32. Proposed implementation surfaces

### Python package

Create:

```text
teacher_logit_reco/relation_expert_bridge/
  __init__.py
  contracts.py
  splits.py
  hlt_v3.py
  hlt_audit.py
  normalizer_lineage.py
  expert_registry.py
  token_shape_registry.py
  particle_tap.py
  layerwise_pair_bias.py
  summary_tokens.py
  expert_model.py
  expert_training.py
  complementarity.py
  offline_fusion.py
  target_cache.py
  bridge_targets.py
  bridge_content.py
  token_normalization.py
  hlt_experts.py
  realization_policy.py
  predictor_registry.py
  predictors.py
  predictor_losses.py
  predictor_training.py
  bundle_search.py
  joint_prediction.py
  token_refiner.py
  final_adapter.py
  capacity_controls.py
  evaluation.py
  selection.py
  reporting.py
  provenance.py
  workflow.py
```

Reuse authenticated code from:

```text
jetclass_fresh.part_inputs
teacher_logit_reco.relational_part
```

Do not fork relation formulas unless an upstream interface cannot support the
expert token tap.  Any necessary upstream change must retain RPT parity tests.

### Command-line entry points

Create:

```text
scripts/build_retb_campaign.py
scripts/build_retb_hlt_v3_cache.py
scripts/audit_retb_hlt_v3.py
scripts/fit_retb_normalizers.py
scripts/train_retb_offline_expert.py
scripts/train_retb_offline_fusion.py
scripts/analyze_retb_complementarity.py
scripts/select_retb_offline_shapes.py
scripts/search_retb_offline_loss_bundle.py
scripts/search_retb_heterogeneous_budget.py
scripts/train_retb_bridge_targets.py
scripts/certify_retb_bridge_content.py
scripts/search_retb_target_coordinate_bundle.py
scripts/build_retb_offline_target_cache.py
scripts/build_retb_postlock_oracle_targets.py
scripts/train_retb_hlt_expert.py
scripts/train_retb_native_hlt_fusion.py
scripts/train_retb_predictor.py
scripts/select_retb_predictors.py
scripts/search_retb_joint_predictor_bundle.py
scripts/evaluate_retb_oracle_substitutions.py
scripts/train_retb_joint_predictor.py
scripts/train_retb_token_refiner.py
scripts/train_retb_final_adapter.py
scripts/train_retb_unrestricted_fusion.py
scripts/build_retb_capacity_controls.py
scripts/aggregate_retb_confirmation.py
scripts/select_retb_scale_shortlist.py
scripts/train_retb_scale_shortlist.py
scripts/infer_retb_scale_stack_val.py
scripts/select_retb_scale_finalists.py
scripts/write_retb_final_test_execution_lock.py
scripts/evaluate_retb_final_test.py
scripts/write_retb_report.py
scripts/submit_retb_graph.py
```

Every CLI supports `--dry-run` where mutation is possible and prints its
resolved immutable configuration.

### Slurm entry points

Create:

```text
sbatch/run_retb_build_splits.sh
sbatch/run_retb_build_hlt_v3.sh
sbatch/run_retb_audit_inputs.sh
sbatch/run_retb_fit_normalizers.sh
sbatch/run_retb_train_offline_expert.sh
sbatch/run_retb_train_offline_fusion.sh
sbatch/run_retb_select_shapes.sh
sbatch/run_retb_train_bridge_pilot.sh
sbatch/run_retb_train_bridge_targets.sh
sbatch/run_retb_build_targets.sh
sbatch/run_retb_train_hlt_expert.sh
sbatch/run_retb_train_predictor.sh
sbatch/run_retb_train_joint.sh
sbatch/run_retb_train_refiner.sh
sbatch/run_retb_train_adapter.sh
sbatch/run_retb_train_unrestricted.sh
sbatch/run_retb_capacity_controls.sh
sbatch/run_retb_confirm.sh
sbatch/run_retb_scale_shortlist.sh
sbatch/run_retb_infer_scale_stack_val.sh
sbatch/run_retb_select_scale_finalists.sh
sbatch/run_retb_postlock_targets.sh
sbatch/run_retb_final_test.sh
sbatch/run_retb_report.sh
sbatch/submit_retb_tigris_full.sh
```

---

## 33. Required tests

### Data and degradation

- production split counts and per-class balance are exact;
- all split identities are disjoint;
- the hash partition assigns exactly 5,000 per class to each of `val_stop`
  and `val_design` and is invariant to file/shard order;
- offline/HLT identity order is exact;
- no constituent-match field reaches a dataset sample;
- HLT v3 strength zero is bitwise identity;
- random results are invariant to batch, shard, and worker layout;
- all four training replicas are identity-aligned, independently hashed, and
  deterministic;
- `R_FIXED`, `R_MULTI`, and `R_RANDOM` select exactly the registered replica
  and severity without changing label exposure;
- every shared `model_train`/`scale_train` identity has byte-identical replicas
  under the common domain seed;
- corruption substreams do not shift when unrelated diagnostics are added;
- operation order matches the contract;
- charged objects never merge and only equal neutral categories merge;
- merged four-vectors, mass, category, charge, and invalid track fields match
  hand-calculated float64 fixtures;
- nonmerged masses are preserved under kinematic smearing;
- charged, photon, neutral-hadron, and unknown response multipliers match the
  table, and charged particles never receive local reassignment;
- every degradation mechanism applies base, type, strength, replica
  multiplier, and clipping in the declared order;
- v2 base-term extraction matches the pre-refactor helper byte-for-byte and
  rejects a different source/function hash;
- energy changes only through degraded momentum plus preserved/merged mass;
- invalid track sentinels and validity masks agree;
- measurement-validity embeddings distinguish not-applicable, available, and
  missing states;
- PID remains one-hot and charge/PID states remain valid;
- strength scaling is monotonic for every registered diagnostic;
- derived ParT inputs use only degraded tokens;
- transformed 17-channel and relation-family audits catch upstream-value
  leakage and clipping saturation;
- cache metadata rejects v1/v2/v3 interchange;
- offline, shared-HLT, and scale-up normalizers use exactly their registered
  identities, replicas, masks, weighting, estimator, and clipping parents;
- `R_FIXED`, `R_MULTI`, and `R_RANDOM` load the same shared nominal HLT
  relation/REGION normalizer;
- Stage M refits relation, REGION, input, and target-token statistics on
  `scale_train` and rejects 500k normalizer substitution;
- the 3M scale pool contains the 500k train identities, adds exactly 2.5M
  identities, and is disjoint from every validation/test split;
- pre-lock final-test preparation cannot load a checkpoint, join labels to
  model outputs, or emit tokens, logits, predictions, or metrics; it may emit
  only authenticated raw/degraded inputs and deterministic identity/relation
  sidecars.

### Expert and token architecture

- real-Weaver baseline parity remains valid;
- particle tap reproduces ordinary logits/gradients when the reference head is
  used;
- every expert receives all 17 particle fields;
- every expert receives exactly its registered relation;
- summary tokens have exact `[B,K,D]` shape for every registered `(K,D)`;
- `S8_128` and `S16_64` expose the declared equal-scalar-budget comparison;
- token queries have stable expert and slot identities;
- particle padding cannot affect summary tokens;
- no expert classifier path bypasses summary tokens;
- every canonical uniform shape and heterogeneous assignment instantiates and
  backpropagates;
- `B_DUAL_FIXED` equals the sum of independently encoded base4 and relation
  logits after their separate pair encoders;
- `B_DUAL_GATED` initializes every post-encoder per-layer/head relation scale
  to one and never alters masks;
- the layerwise provider emits `[B,H,N,N]`, supplies the correct distinct
  projection to every layer, and never materializes `[B,L,H,N,N]`;
- zeroing the dual relation branch exactly reproduces its registered base4
  control in emitted biases, logits, gradients, masks, and identities;
- old single-bias state dictionaries cannot load as
  `retb_layerwise_pair_bias_v1` without explicit authenticated migration;
- `TOK_MULTI_DEPTH` reads exactly blocks 4 and 8, respects both masks, and has
  no classification bypass;
- bridge-aware target gradients reach only registered trainable components;
- `T0_PURE`, both `T1` modes, `T2_PROJECT`, and `T3_LOGIT` cannot be
  interchanged under one schema;
- all-empty HLT rows follow the canonical dummy-particle policy;
- token attention and logits remain finite in BF16 and FP32.

### Fusion and complementarity

- canonical expert concatenation order is stable;
- frozen fusion cannot update expert weights;
- light fine-tuning updates only tokenizers;
- grouped-head biases own separate softmax distributions;
- all-relation, unbiased ensemble, and capacity controls have correct active
  parameters;
- complete-graph capacity accounting includes encoders, predictors,
  projections, uncertainty, refiner, and final consumer;
- parameter- and inference-FLOP-matched monolithic selectors reproduce their
  deterministic tie-breaks at batch sizes 1 and 128;
- subset readouts cover all 128 expert subsets;
- complementarity metrics are deterministic;
- offline expert-loss beam search reproduces beam membership, provisional
  readouts, and the selected seven-expert tuple;
- primary fusion contains no expert dropout.

### Token selection

- `SHAPE_HIGH` uses the global tolerance window;
- `SHAPE_COMPACT` applies all three noninferiority limits;
- `SHAPE_BRIDGE` uses only eligible validation artifacts;
- all seven uniform shapes require complete three-seed results before
  `SHAPE_HIGH` or compact eligibility is evaluated;
- heterogeneous assignments obey the 56-slot budget and deterministic
  marginal-allocation rule;
- `HET_BEAM` reproduces its beam and never exceeds 56 slots;
- duplicate selected shapes are removed deterministically;
- all-negative campaigns still emit selections and continue.

### Target cache

- targets come from frozen selected offline experts;
- every `T1/T2` target binds the exact seed-matched `T0`, HLT encoder, and
  initial `PILOT_T0` parents;
- bridge-target jobs cannot run before the pure target, native HLT, and pilot
  parents exist;
- dimension-changing `T2` uses the direct-plus-bridge-residual formula while
  equal dimensions use the true residual formula;
- class-prototype synthetic targets fail variance, effective-rank,
  covariance, or within-class retrieval certification;
- content-certification thresholds and same-event/wrong-event identities
  reproduce exactly;
- retrieval uses unlearned slot-major flattened normalized-bank cosine,
  excludes the query identity, and rejects undersized negative rings;
- effective rank uses an exact `[n_events,K*D]` matrix and the declared
  zero-singular-value convention;
- covariance loss uses synchronized centered FP32 population covariance,
  exact relative-Frobenius normalization, and rejects effective batch size
  below two;
- cosine zero norms, exact retrieval ties, SVD backend/dtype, and numerical
  singular-value threshold match the frozen contract;
- joint target-mode tuples own distinct fusions and a cross-mode tuple cannot
  use an unmatched frozen fusion;
- target mode, shape, pipeline seed, slot-query, and checkpoint hashes are
  parents;
- pre-lock target builders accept only `model_train`, `val_stop`, and
  `val_design`;
- before `locked_scale_finalists.json`, the only permitted `stack_val` model
  outputs are identity-bound deployable logits/probabilities for every
  shortlisted 3M graph/seed, with no labels or oracle parents;
- the `stack_val` selector alone joins labels from the authenticated split
  manifest's identity-sorted label projection and publishes separate metric
  artifacts and selection traces; the inference worker cannot load that
  projection;
- the finalist lock binds every selection-prediction shard/manifest, label
  manifest, metric artifact, and selection-trace hash;
- no artifact created before `locked_scale_finalists.json` contains
  `stack_val` oracle tokens/logits, model-output labels, or any `final_test`
  model output;
- pre-lock final-test input preparation cannot load a checkpoint;
- post-lock `stack_val` oracle targets are selection-ineligible;
- final-test targets require exact Stage-M teacher/fusion/normalizer hashes and
  reject every 500k target cache;
- float16 round-trip audit enforces tolerances and failing banks automatically
  publish authenticated FP32 targets instead;
- mixed target-cache dtypes load to float32 without changing values outside the
  declared round-trip tolerance;
- shards publish atomically and resume by hash;
- missing, duplicated, stale, or reordered identities fail;
- validation/test data never fit token normalization.

### Predictors

- every architecture emits tokens, log variance, and valid gates;
- `U_SLOT`, `U_GROUP4`, and `U_DIAGONAL` expose exactly their registered
  uncertainty dimensions;
- additive uncertainty calibration fits `val_design` without labels and is
  frozen before `final_select/stack_val`;
- `C0`, `C1`, `C2`, and `C3` expose exactly their registered evidence;
- no offline input enters predictor forward;
- decoder target queries are fixed-index and non-autoregressive;
- direct and gated formulas match hand calculations;
- gate initialization is exactly `-2.0`;
- uncertainty clips to `[-8,4]`;
- `N_UNCLIPPED`, `N_CLIP16`, and `N_CLIP8` match exact tail fixtures and only
  registered hard-clip controls clip finite values;
- zero-evidence and shuffled-evidence controls are nontrivial;
- predictor gradients do not reach frozen offline experts or targets;
- every loss term matches a hand-calculated fixture;
- fixed-weight and GradNorm objectives use only their declared information;
- selected predictor bundles reject seed, teacher, target, HLT-mode, or
  normalization substitution;
- joint predictor beam search scores complete all-predicted tuples and
  reproduces its selected seven-expert configuration;
- `J1` through `J5` give all seven experts the same identity, replica, and
  degraded HLT particle array under canonical `R_MULTI`;
- logit-only targets cannot be reported as faithful token recovery.

### Final consumers

- frozen offline fusion weights remain unchanged;
- robust-fusion mixture proportions are deterministic;
- final adapter residual scalar initializes to zero;
- token refiner has an exact identity control and cannot read offline evidence;
- `HF_UNRESTRICTED` consumes the complete registered predicted/native banks;
- logit-augmented fusion receives only deployable native and predicted-token
  expert logits with exact source/expert identities;
- `F_TOKEN_ONLY_MATCHED` follows the declared parameter/FLOP selector and
  never receives expert logits;
- `ND0_NONE`, `ND1_FIXED`, and `ND2_CONFIDENCE` implement their exact
  training/inference semantics without event rejection;
- native-only and reconstructed-only paths are independently evaluable;
- native branch dropout applies only during adapter training;
- deployable inference loads HLT arrays and selected checkpoints only;
- oracle targets are impossible to request in deployable mode.

### Training, selection, and statistics

- every scientific run has a fixed 40-epoch schedule;
- poor validation performance does not create job failure;
- checkpoint selection matches the global window rule;
- only `val_stop` selects epochs; `val_design` performs calibration,
  certification, component selection, and 500k shortlisting; only Stage N may
  read `stack_val`;
- the label-free uncertainty calibrator is authorized on `val_design` and no
  label tensor reaches its fitting objective;
- Stage-B training workers cannot open `val_design` or `stack_val`; the
  immutable post-training design selector can open only `val_design`;
- seeds share data order;
- primary confirmation artifacts obey complete matched pipeline-seed lineage;
- fixed-teacher cross-seed rows are marked controls and cannot enter primary
  aggregation;
- accuracy and mean-log-rejection selectors can select different finalists
  and reproduce exact tolerance/tie rules;
- the 500k shortlist is the duplicate-free top-three accuracy/rejection union
  on `val_design`, contains at most six graphs, and still emits when all lose;
- Stage M trains every and only shortlisted graph across all three seeds;
- `stack_val` is opened once only after all shortlisted 3M graphs are
  immutable, and finalists are selected from those 3M predictions;
- zero-background rejection is displayed as infinity but uses the declared
  finite Jeffreys-smoothed selector quantity;
- paired rejection bootstraps recompute thresholds and all 18 mean-log terms
  within each common identity resample;
- finalist-specific capacity controls and `H_BASE_LONG` label-presentation
  counts include the complete declared inference/training graphs;
- scale-up accepts only shortlisted graph definitions and cannot reopen
  architecture selection;
- final-test model outputs require `locked_scale_finalists.json`, and
  scientific inference requires `final_test_execution_lock.json`;
- bootstrap, ECE, rejection, ties, and zero-background behavior copy the
  deterministic metric contract;
- wrong source, cache, degradation, target, or checkpoint hashes fail closed.

### Shell and Tigris

- jobs export `PYTHONNOUSERSITE=1`;
- conda and `LD_LIBRARY_PATH` handling follow the working RPT Tigris wrapper;
- sourced shell helpers resolve from the project directory, not Slurm spool;
- dynamic-job registration uses the activated Python 3.10 environment;
- optional performance warnings do not create unsatisfied dependencies;
- execution/provenance failure does block dependent consumers;
- `--dry-run` prints the complete DAG;
- smoke uses genuine miniature data and the compiled REGION backend;
- production submission never implicitly submits a different degradation
  profile.

---

## 34. Implementation order

### Step 1 of 15: campaign, split, replica, and scale contracts

Implement the 500k/50k/50k/50k/300k role binding, disjoint 3M scale pool, four
HLT training replicas, deterministic `val_stop`/`val_design` partition and
access roles, campaign spec, registries, artifact layout, deterministic
conventions, source provenance, storage measurements, and all run IDs.

Done when every future run and every replica choice can be resolved without
importing training code.

### Step 2 of 15: HLT v3 track-dominant proxy

Implement the versioned degradation, type-conditioned response, type-aware
mechanism equations and clipping order, merging and mass/energy rules, shared
train/scale domain seed, identity-bound RNG, field-isolation profiles,
transformed-input audits, realization policies, offline/shared-HLT normalizer
population recipes, cache metadata, and backward-incompatible cache checks.

Done when strength-zero identity, domain validity, monotonicity, replica and
shard-layout determinism, train/scale shared-identity equality, and
hand-calculated type/merge/normalizer fixtures pass.

### Step 3 of 15: particle-state tap, layerwise relation interface, and token shapes

Add the explicit pre-class particle-state interface, separate base4/relation
pair encoders, streamed per-layer Weaver bias provider, post-encoder dual bias
and gates, state-dictionary versioning, measurement-validity embedding,
single/multi-depth learned-query tokenizers, `(K,D)` registry, token-only
expert head, and reference-parity tests.

Done when every uniform and heterogeneous shape trains on miniature offline
and HLT samples with no classification bypass or `[B,L,H,N,N]`
materialization.

### Step 4 of 15: pure offline experts and optimization controls

Implement all seven expert definitions, expert-loss candidates,
initialization/attachment controls, learning-rate/dropout screen, diagnostics,
checkpoint retention, and run registry.

Done when the 49-row primary Stage-B shape screen and declared optimization
subset complete on miniature data.

### Step 5 of 15: offline fusion, joint loss selection, and shape selection

Implement frozen-token caches, fusion variants, ensemble/grouped-head and
relation-topology controls, joint expert-loss beam search, greedy/beam
heterogeneous assignments, subset readouts, redundancy diagnostics, and
complete three-seed aggregation for all seven uniform shapes.

Done when `SHAPE_HIGH` and `SHAPE_COMPACT` can be selected from a fully
negative synthetic fixture and no incomplete shape can enter selection.

### Step 6 of 15: native HLT evidence encoders

Implement `HE_SCRATCH_CE`, `HE_OFFLINE_INIT`, `HE_DUAL_OBJECTIVE`,
measurement-validity controls, native fusions, matched baselines, and
degradation/replica binding.

Done when native specialization, offline initialization, and privileged
alignment are separately measurable before any bridge target is co-designed.

### Step 7 of 15: locked pilot and bridge-aware offline targets

Implement seed-matched `PILOT_T0`, `T1_ANCHORED_BRIDGE`,
`T1_TASK_BRIDGE`, dimension-correct `T2_PROJECT`, `T3_LOGIT`, alternating
updates, offline noninferiority, collapse/content certification, and semantic
version separation, exact flattened-bank retrieval/effective rank, plus the
joint target-coordinate/fusion beam selector.

Done when missing pilot parents fail closed, class prototypes fail content
certification, and pure/task/representation claims cannot be interchanged.

### Step 8 of 15: target cache, normalization, and seed lineage

Implement selected-shape/mode target generation, FP16 audit with FP32
fallback, token normalizers, pilot/content parents, per-seed resumable shards,
sealed test preparation, and hash validation.

Done when target batches reproduce frozen expert logits after round trip and
cross-seed artifact mixing fails closed.

### Step 9 of 15: predictors, context, uncertainty, and objectives

Implement `A0` through `A4`, `C0` through `C3`, evidence-bank reuse,
`U_SLOT`/`U_GROUP4`/`U_DIAGONAL`, uncertainty calibration, normalization
controls, fixed and GradNorm objectives, direct/gated outputs, and capacity
controls.

Done when every registered predictor bundle trains and resumes on a miniature
paired dataset without constituent matches.

### Step 10 of 15: joint predictor bundle and oracle substitutions

Cache all eligible predicted banks, implement the canonical width-32
all-predicted bundle search, global/shared controls, hybrid/oracle
substitutions, and negative token controls.

Done when interacting predictor choices are selected as one immutable
seven-expert tuple and the search is deterministic on an all-negative fixture.

### Step 11 of 15: joint bridge optimization

Implement `J0` through `J5`, `HE_BRIDGE_TUNED`, shared/coupled prediction, and
end-to-end maximum-performance training from the already locked predictor
bundle under the one-identity/one-`R_MULTI`-view invariant.

Done when faithful, bridge-tuned, logit-distilled, and end-to-end candidates
remain separately labeled even if every row loses.

### Step 12 of 15: refiners and final consumers

Implement `TR_REFINE`, robust oracle/predicted mixtures, constrained adapters,
token/logit evidence variants, matched token-only controls,
`HF_UNRESTRICTED`, all native-dropout modes, bypass controls, deployable
export, and complete-graph parameter/FLOP accounting separately per graph.

Done when deployable inference proves it cannot load offline inputs or target
caches and monolithic matching covers the exact exported graph.

### Step 13 of 15: 500k confirmation and scale shortlist

Implement matched-seed 500k confirmation, bridge-aware shape selection,
bounded top-three accuracy/rejection union on `val_design`, deterministic
metrics/statistics, complete Markdown/JSON reports, and failure
interpretations.

Done when a source-bound shortlist of at most six complete graphs is emitted
even for an all-negative campaign and no `stack_val` or final-test prediction
exists.

### Step 14 of 15: shortlisted 3M scale-up, one-time selection, and final seal

Retrain every shortlisted graph on `scale_train`, aggregate three-seed
confirmation, refit every locked train-derived normalizer and label-free
calibrator on its declared scale/design population, open `stack_val` once for
label-free deployable 3M selection shards, join labels only inside the
authenticated selector, write `locked_scale_finalists.json`, build only then
the scale-teacher oracle/final-test targets and finalist controls, write
`final_test_execution_lock.json`, and run the common sealed final test exactly
once.

Done when architecture reselection, early oracle/test outputs, stale teacher
targets, labels embedded in selection-prediction shards, incomplete finalist
lock lineage, and premature final-test access fail closed while a negative
campaign completes normally.

### Step 15 of 15: Tigris production DAG

Implement CPU/GPU workers, resource probes, bounded arrays, resumable target
shards, dynamic continuation, smoke submission, full submission, monitoring
commands, and job ledger.

Done when a real miniature Tigris smoke completes and the production dry run
prints every Stage A-N dependency, including both selectors and scale-up.

---

## 35. Tigris production defaults

Use the verified repository defaults:

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
RETB_CPU_CACHE_CONCURRENCY=64
RETB_GPU_EXPERT_CONCURRENCY=64
RETB_GPU_PREDICTOR_CONCURRENCY=64
RETB_GPU_SCALE_CONCURRENCY=64
RETB_GPU_FINAL_CONCURRENCY=64
```

These are high-throughput array caps, not requested GPU counts. Each GPU row
still requests one GH200, while Slurm schedules up to 64 ready rows from each
wave subject to cluster/account availability. Internal predictor, joint,
consumer, export, robustness, and semantic-control phase arrays inherit the
same environment settings; they must not reintroduce smaller hidden caps.
Single-output selectors, aggregators, and the sealed exactly-once final-test
worker remain serialized.

The top-level production command will be:

```bash
cd /home/ryreu/atlas/Fresh_check
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
bash sbatch/submit_retb_tigris_full.sh
```

The final wrapper must activate conda before using `CONDA_PREFIX`; the example
above assumes an already activated interactive environment.  Worker wrappers
perform their own fail-closed activation.

The submitter prints:

- campaign root and source hash;
- degradation profile and cache hashes;
- every stage/array job ID;
- bounded concurrency;
- storage projection;
- monitoring commands;
- stale-job cancellation commands;
- report download paths.

---

## 36. Failure modes and interpretation

### Individual relation experts do not beat `O_BASE`

The relation bias may not improve a complete full-field expert.  Token
reconstruction can still be studied, but no claim of improved offline
reasoning is made.

### Oracle token fusion beats `O_BASE`, but capacity controls match it

The system benefits from capacity or ensembling, not demonstrated relation
specialization.

### Experts are individually strong but redundant

The biases lead to similar decisions.  Report the negative complementarity
result; do not force artificial diversity after the fact.

### Small \(K\) is competitive offline but predicts poorly

The compact slots may be overloaded or unstable.  Prefer `SHAPE_HIGH` if its
bridge-aware performance is better.

### Large \(K\) predicts better than small \(K\)

Additional slots may factorize evidence into easier targets.  Output dimension
alone is not reconstruction difficulty.

### Lower-dimensional or heterogeneous tokens win

Slot count was not the sole bottleneck.  Report both total scalar width and
per-expert allocation; do not translate the result back into a \(K\)-only
claim.

### Pure-offline tokens classify well but bridge-aware tokens deploy better

The original representation was class-optimal but unnecessarily hard to infer
from HLT.  This supports the bridge-aware target, provided its locked offline
noninferiority tests passed.  It is not evidence that the same latent
coordinates were reconstructed.

### Bridge-aware targets reduce offline performance

They are ineligible under the predeclared noninferiority contract.  Retain
`T0_PURE` and report the predictability/performance tradeoff rather than
relaxing the threshold.

### Task bridge classifies well but fails content certification

It learned predictable class/task prototypes rather than preserving
instance-level offline evidence.  It remains a maximum-performance
task-distillation candidate, but no faithful representation-recovery claim is
made.

### Joint bundle differs from every individually best predictor

Prediction errors interact across banks.  Report both the independent-hybrid
default and joint-beam result; the latter is the primary all-predicted
configuration and is not evidence that any individual predictor improved.

### Token error improves without frozen-consumer improvement

The loss overweights irrelevant coordinates.  Frozen expert/fusion utility,
not raw MSE, remains the predictor selection endpoint.

### Logit-only predictor performs best but token fidelity collapses

This is successful task distillation, not faithful token reconstruction.  It
is reported separately.

### Native HLT multi-expert fusion matches all reconstructed variants

Relation specialization helps HLT, but offline-token reconstruction adds no
deployable value.

### Oracle tokens are strong but predicted tokens recover little

The target contains information unavailable or too ambiguous in HLT.  Oracle
substitution locates which expert bank dominates the gap.

### Final HLT adapter improves while frozen fusion does not

The adapter learned useful HLT correction, but offline representation recovery
is weak.  Both facts are reported.

### `HF_UNRESTRICTED` wins but faithful consumers do not

The complete HLT system is useful, but the gain cannot be attributed to
recovering the offline token reasoning.  Treat unrestricted fusion as the
maximum-performance result and retain `PF_FROZEN`/`OF_ROBUST` as the
faithfulness evidence.

### Expert-logit tokens improve unrestricted fusion

The explicit expert decisions aid optimization or preserve evidence the final
token transformer did not easily recover.  Compare against
`F_TOKEN_ONLY_MATCHED` before attributing the gain to information rather than
capacity.

### Multi-realization training helps only robustness

The benefit is domain randomization rather than better fixed-proxy
reconstruction.  Report `R_FIXED` and `R_MULTI` together; do not silently make
the test degradation easier or harder.

### HLT/offline initialization helps while scratch does not

The bridge depends on representation alignment induced by initialization or
privileged training.  This is a valid deployable training result, but must be
distinguished from HLT-label-only learning.

### Track-dominant degradation causes a much larger gap than intended

This is not a job failure.  Report the domain severity and use predeclared
field-isolation controls.  Do not retune the nominal profile after classifier
results.

### The 500k and 3M graph rankings differ

This is an expected scale effect, not a contract failure.  Report which
shortlisted graphs moved and select the actual finalists only from the locked
3M `stack_val` predictions.  The bounded shortlist limits compute but can miss
a graph ranked outside both 500k top-three lists; record that design limitation
rather than retrospectively expanding the shortlist.

### No model improves

The campaign remains successful if it produces a provenance-complete negative
result identifying the limits of relation experts and latent reconstruction.

---

## 37. Definition of success

Engineering success requires:

- exact data, source, relation, degradation, token, and checkpoint lineage;
- no constituent-match dependency;
- no offline input in deployable inference;
- deterministic full campaign continuation after negative results;
- coherent matched pipeline-seed lineage and immutable per-expert bundles;
- no pre-finalist-lock final-test model output and no scientific final-test
  inference before both `locked_scale_finalists.json` and
  `final_test_execution_lock.json` are immutable;
- real-Weaver and compiled-REGION parity;
- a complete smoke and production DAG;
- reproducible reports and paired predictions.

Scientific success is not defined as a required gain.  The strongest positive
outcome would establish all of:

1. relation-expert offline token fusion improves over `O_BASE`, `O_WIDE`,
   shared-full-relation, and unbiased ensemble controls;
2. the gain is stable across seeds and experts show measurable complementary
   errors;
3. HLT predictors recover useful offline expert tokens through frozen
   consumers, with any representation-preservation claim passing the locked
   instance-content certification;
4. reconstruction improves over matched native HLT multi-expert fusion;
5. native HLT residual evidence adds further deployable gain without erasing
   the reconstructed path;
6. conclusions survive the track-dominant nominal domain and predeclared
   degradation controls;
7. the locked benefit survives the predeclared 3M scale-up and matched
   complete-graph capacity controls.

The campaign is also scientifically successful if it determines cleanly that
one or more of those hypotheses is false.




















Yes. I’d separate the remaining work into eight dependency-ordered tasks. This keeps each change reviewable and prevents another situation where structural DAG tests pass without an executable campaign.

### Task 1 — Production-manifest coverage contract

Create a central registry mapping every DAG node to:

- its worker;
- manifest producer;
- required inputs;
- expected outputs;
- resource type;
- static or dynamically generated rows.

Add a regression that fails whenever a manifest-driven node has no automatic producer. This would immediately catch the current `input_audit` gap.

### Task 2 — Complete Stage-A data preparation

Implement and wire:

- `input_audit.json`;
- actual offline and HLT relation normalizers;
- REGION tree caches and REGION normalizers;
- complete input lineage and identity coverage;
- miniature equivalents for smoke tests.

This task should leave Stage A producing every artifact needed to start model training.

### Task 3 — Static experiment manifest compiler

Generate concrete run rows for experiments whose matrix is known at campaign creation:

- offline expert training;
- initial offline controls;
- native HLT experts and controls;
- predetermined fusion configurations;
- fixed bridge pilots.

Each row should contain its full command, configuration hashes, input lineage, output locations, seed, and expected artifacts.

### Task 4 — Dynamic continuation through Stages B–E

Make selectors publish authenticated downstream manifests automatically:

- offline shape selection;
- optimization selection;
- offline fusion candidates;
- bridge-target candidates;
- bridge-content certification;
- target-coordinate selection.

The selector’s output and the generated downstream manifest should be hash-bound together.

### Task 5 — Target-cache and predictor continuation, Stages F–J

Wire the largest middle section:

- target-cache shards;
- target normalizers;
- independent predictor runs;
- uncertainty calibration;
- predictor-bundle selection;
- oracle substitutions;
- joint predictors;
- final consumers;
- deployable exports.

This should include resumable shard behavior and completeness validation before dependent jobs proceed.

### Task 6 — Controls, confirmation, and scale-up, Stages K–M

Implement automatic manifests and aggregation for:

- robustness and semantic controls;
- 500k seed confirmations;
- bridge-shape and scale-shortlist selection;
- 3M normalizer refits;
- all shortlisted 3M graph training;
- scale-completion attestation.

Poor scientific performance must be recorded, not treated as a runtime failure. Only malformed, incomplete, or provenance-invalid artifacts should stop dependencies.

### Task 7 — Stage-N sealed evaluation

Finish the two-lock final workflow:

- deployable `stack_val` inference;
- finalist selection;
- locked finalist lineage;
- post-lock oracle-target creation;
- finalist controls;
- final-test execution lock;
- exactly-once sealed final-test evaluation;
- final report and completed-job ledger.

This task should explicitly test that no performance threshold aborts the campaign.

### Task 8 — End-to-end operational validation

Run progressively:

1. Local synthetic DAG execution without Slurm.
2. Local miniature campaign through every Stage A–N worker.
3. `--smoke-simulate`.
4. Real miniature Tigris smoke.
5. Production dry run with authenticated storage measurements.
6. Only then, full submission.

Also test restart after a failed array element, stale-job cancellation, artifact reuse, source drift, and interrupted selector continuation.

I recommend implementing them in exactly this order. Tasks 1–2 are the immediate blockers; Tasks 3–7 complete the orchestration; Task 8 is the required proof that the plan is genuinely finished.








## Remaining production-execution closure blocks

The original implementation tasks produced the scientific contracts, model
components, worker CLIs, static experiment compiler, and production DAG, but
they did not fully connect every selector-dependent worker to concrete
production task rows.

At the time these closure blocks were defined, there were:

- 48 manifest-driven production targets;
- 13 targets with genuine campaign-bootstrap manifests;
- 35 targets whose concrete row plans still require production wiring;
- 31 producer-to-consumer transitions across those 35 targets.

The following five blocks are mandatory and dependency ordered. A block is not
complete merely because its contracts, worker CLI, or synthetic rows exist.
Completion requires real, source-bound task rows that invoke the intended
scientific worker with complete inputs and expected outputs.

### Closure Block 1 — Correct the readiness gate and plan-producer audit

Replace the current hook-presence audit with an execution-complete
plan-producer audit.

For every manifest-driven node, the audit must distinguish:

1. a manifest prepublished by the authenticated campaign bootstrap;
2. a downstream manifest plan emitted by the registered producer;
3. materialization of that plan into the graph-declared task-manifest path;
4. successful execution and authentication of every row;
5. complete-manifest attestation after all rows finish.

Merely registering a producer, calling the shared materialization hook, or
creating synthetic rows is not evidence of automatic production readiness.

Add a central plan-factory registry containing, for every non-bootstrap target:

- target node ID;
- producer node ID;
- plan-factory entry point;
- required producer artifacts and contracts;
- trigger artifact and trigger hash;
- row-count or deterministic row-count rule;
- allowed worker entry points;
- expected output contracts;
- static or dynamic resolution;
- dataset-access role;
- resource type;
- performance-independent continuation policy.

The local operational report must fail full-submission eligibility unless all
graph-declared manifest targets have genuine producer coverage. Synthetic DAG execution
remains useful for control-plane testing but cannot satisfy this requirement.

Version the affected operational-report, authorization, Step-15, graph, and
plan-producer contracts.

Done when:

- deleting any one real plan-factory registration fails readiness;
- a registered hook without an emitted plan fails readiness;
- a plan with missing rows, incorrect lineage, or synthetic worker outputs
  fails readiness;
- all 13 bootstrap manifests and all 35 dynamic/producer-generated manifests
  are accounted for separately;
- full-submission authorization cannot be created until all five closure
  blocks pass a real miniature campaign.

Implementation status (2026-07-29): implemented. The production graph now
embeds the exhaustive 35-entry non-bootstrap plan-factory registry, and the
versioned operational audit distinguishes bootstrap publication, plan
emission, manifest materialization, row/output authentication, and complete
manifest attestation. The shared hook and the synthetic DAG are explicitly
ineligible as producer evidence. Consequently the corrected local report
currently fails full-submission eligibility, as intended, until Closure
Blocks 2--5 provide and execute the real factories.

### Closure Block 2 — Complete real continuation through Stages B–E

Implement the seven remaining Stage-B–E manifest targets:

1. `offline_optimization_selector`;
2. `offline_shape_selector`;
3. `offline_complementarity`;
4. `offline_capacity_controls`;
5. `bridge_target_training`;
6. `bridge_content_certification`;
7. `target_coordinate_selector`.

Required Stage-B behavior:

- perform deployable expert inference on `val_design` for the complete locked
  PT/TRACK optimization grid;
- publish identity-bound prediction and metric artifacts;
- include parameter and FLOP evidence required by the locked optimizer
  selector;
- aggregate all required candidate rows before selection;
- always select the best available candidate, including when every candidate
  is worse than the baseline;
- publish the optimization selection and any predeclared follow-up rows without
  performance-based cancellation.

Required Stage-C behavior:

- require all 147 expert-confirmation checkpoints;
- require all 63 frozen-token caches;
- require all 21 canonical uniform-fusion confirmations;
- run canonical fusion inference on `val_design`;
- publish the complete 21-row uniform-shape metric artifact;
- select `SHAPE_HIGH` and `SHAPE_COMPACT` only after complete aggregation;
- train and aggregate all 128 subset readouts required for complementarity;
- produce pairwise, leave-one-out, Shapley, redundancy, and sensitivity
  diagnostics;
- execute all declared capacity controls with complete-graph parameter, FLOP,
  and label-exposure evidence.

Required Stage-E behavior:

- transform complete bridge-pilot outputs into concrete target-training rows;
- execute every predeclared target mode and negative/control row;
- publish content-certification inputs and certification rows;
- run noninferiority and instance-content certification without treating
  scientific failure as a runtime failure;
- publish the complete target-coordinate selector inputs;
- always select the best eligible target tuple or the locked fallback.

Every selector output must be hash-bound to the exact downstream manifest it
creates.

Done when a real miniature campaign progresses automatically from completed
Stage-B expert training through immutable target-coordinate selection without
manual row JSON, environment-variable injection, or path repair.

Implementation status (2026-07-30): the source implementation is complete.
The seven downstream factories now emit executable, source-bound manifests;
Stages B and C perform the complete optimization, shape, complementarity, and
capacity-control work; and Stage E executes every pilot/target row, emits
separate oracle and deployable predicted-coordinate artifacts, certifies
scientific outcomes without stopping on negative results, and selects from
fresh all-predicted readouts with an immutable T0 fallback. The pilot
`R_MULTI` path uses the global identity-dependent replica cycle. Coordinate
readout scores bind every consumed array, normalizer, predictor profile, and
fresh fusion checkpoint. Local contract/unit verification is complete; the
literal done criterion above remains an execution acceptance test until the
real miniature is run on research compute.

### Closure Block 3 — Complete Stages F–J target and predictor continuation

Implement the ten Stage-F–J manifest targets:

1. `target_cache_build`;
2. `target_normalizers`;
3. `predictor_training`;
4. `uncertainty_calibration`;
5. `predictor_bundle_selector`;
6. `oracle_substitutions`;
7. `joint_predictor_training`;
8. `joint_predictor_selector`;
9. `final_consumer_training`;
10. `deployable_export`.

Required behavior:

- build only selector-approved offline target coordinates;
- generate deterministic, resumable, identity-bound target-cache shards;
- authenticate complete shard coverage before normalizer fitting;
- fit target normalizers only from the globally locked training population;
- generate every independent predictor architecture/context/control row;
- preserve matched seeds, shapes, target modes, degradation replicas, and
  checkpoint lineage;
- perform uncertainty calibration only after complete predictor inference;
- select predictor bundles from complete frozen-consumer evidence;
- generate all oracle-substitution rows;
- generate and train all joint-predictor candidates;
- publish joint-selector inputs only after complete joint-candidate coverage;
- train every locked final-consumer variant;
- export deployable graphs with no offline inputs or targets;
- validate exported inference against the corresponding research graph.

Resumable cache or training rows may be reused only after revalidating all
expected output hashes. Negative predictor or consumer performance must never
suppress later rows.

Done when a real miniature campaign proceeds automatically from the locked
target-coordinate tuple through authenticated deployable exports.

Implementation status (2026-07-30): source implementation complete. The
public `predictor_training` and `joint_predictor_training` identities remain
unchanged, while their required internal topologies are frozen as:

```text
F architecture screen
  -> F architecture selection
  -> F optimizer screen
  -> F optimizer selection
  -> G objective/uncertainty/normalization screen
  -> G configuration selection
  -> H confirmation

J0--J4 candidate wave
  -> J4 block selector
  -> J5 end-to-end wave
```

The shared internal-phase executor publishes source-bound plans, per-row
byte-hash receipts, complete-phase attestations, and a final controller
attestation. Every later plan binds all earlier phase-completion hashes;
restarts revalidate every expected output; and the sign of a scientific
metric cannot suppress a later phase. Predictor training consumes the
identity/epoch-bound common `R_MULTI` replica correctly, T2 decoder queries
are projected into the actual bridge coordinate, and uncertainty fitting is
separated from predictor training. All ten F--J targets now have automatic
scientific-row producers and executable public-controller wiring. Local
contract and operational verification is complete. The literal miniature
done criterion remains an execution acceptance test until the real Tigris
smoke runs.

### Closure Block 4 — Complete Stages K–M controls, confirmation, and scale-up

Implement the ten Stage-K–M manifest targets:

1. `robustness_controls`;
2. `semantic_controls`;
3. `stage_l_graph_registration`;
4. `confirmation_500k`;
5. `confirmation_summary`;
6. `bridge_shape_selector`;
7. `scale_shortlist_selector`;
8. `scale_refits`;
9. `scale_graph_training`;
10. `scale_completion`.

Required behavior:

- execute every robustness profile and replica declared by the campaign;
- execute semantic bypass, substitution, and reconstruction controls;
- authenticate complete control coverage before Stage-L registration;
- register only complete deployable graphs;
- execute all required matched-seed 500k confirmations;
- aggregate confirmation metrics without performance gating;
- select the locked bridge shape from the complete confirmation artifact;
- construct the bounded scale shortlist exactly as predeclared;
- refit all required 3M normalizers from `scale_train`;
- train every shortlisted 3M graph and every required seed;
- reject incomplete scale waves but continue after scientifically negative
  results;
- publish scale completion only after every shortlisted graph, seed,
  checkpoint, capacity artifact, and required metric is present.

No 500k or 3M candidate may be omitted because it performed poorly.

Done when a real miniature scale analogue traverses all Stage-K–M nodes and
produces an immutable scale-completion artifact without manual intervention.

Implementation status (2026-07-30): the fail-closed Stage-K--M factory and
scientific-contract layer is implemented. All ten late targets are registered
with executable factory symbols; their source-bound producer inputs require
complete robustness-profile/replica, semantic-control, graph/seed, refit, and
scale-training coverage, and a scientific metric cannot remove a row. Stage-L
graph registration is now correctly pre-selection: it binds both completed
control waves and all compact/high candidate shapes, while `SHAPE_BRIDGE` is
selected only from the later complete confirmation artifact. The corrected
graph registry, confirmation summary, and Stage-L report use new contract
versions. Incomplete 500k or 3M waves fail closed, while wholly negative
scientific outcomes continue.

The concrete K--M execution layer is also complete: shared per-seed scale
refits feed every shortlisted graph plus the required named baseline; all
HLT capacity controls are trained as real models; `H_BASE_LONG` derives its
exact exposure ceiling only from authenticated nonzero ground-truth-CE
phases; and scale completion uses the versioned corrected contracts. Factory
inputs are emitted only from authenticated upstream completions, so a
manually placed factory-input JSON remains invalid. The literal miniature
done criterion remains an execution acceptance test until the real Tigris
smoke runs.

### Closure Block 5 — Complete sealed Stage-N evaluation

Implement the eight Stage-N manifest targets:

1. `prelock_final_inputs`;
2. `stack_val_inference`;
3. `accuracy_finalist_selector`;
4. `postlock_oracle_targets`;
5. `finalist_controls`;
6. `final_test_execution_lock`;
7. `sealed_final_test`;
8. `final_report`.

Required behavior:

- prelock preparation may construct only identity-bound raw/degraded inputs;
- no prelock `final_test` model-derived output is permitted;
- deployable `stack_val` inference must be label-free and must cover every
  shortlisted 3M graph;
- prediction shards must not contain labels or offline targets;
- selectors must join labels only from the authenticated label manifest;
- both accuracy and rejection selection traces must be complete and
  deterministic;
- `locked_scale_finalists.json` must bind all prediction, label-manifest,
  metric, capacity, checkpoint, and selection-trace hashes;
- post-lock oracle targets must be explicitly selection-ineligible;
- every finalist control must complete before the execution lock;
- `final_test_execution_lock.json` must bind the finalist lock, post-lock
  diagnostics, control coverage, sealed input preparation, and exactly-once
  execution claim;
- sealed final-test inference may run exactly once and only after both locks;
- the final report must include positive or negative scientific outcomes
  without changing the locked selections;
- the completed job ledger must be published only after the final report.

Done when a real miniature Stage-N run proves:

- no prelock final-test leakage;
- label-free prelock `stack_val` inference;
- deterministic two-selector finalist locking;
- correct post-lock oracle/control ordering;
- exactly-once sealed final-test execution;
- completion even when every scientific comparison is negative.

Implementation status (2026-07-30): the fail-closed Stage-N factory and
continuation-contract layer is implemented. All eight final targets have
registered executable factories, authenticated producer-input requirements,
exact shortlist/finalist graph-by-seed coverage checks, label-free `stack_val`
access rules, post-lock target/control ordering, complete execution-lock
coverage, and exactly-once sealed-evaluation assertions. The pre-lock input
node now invokes `prepare_retb_final_test_inputs.py`, whose artifact contract
matches the final execution lock; the former registry entry pointed to the
generic sealed-target-cache worker and would have produced an incompatible
artifact. Contract/schema versions were advanced so the corrected registry
and graph cannot be confused with earlier artifacts. Scientific
underperformance is explicitly non-blocking throughout.

The concrete Stage-N execution layer is complete. A single shared,
label-free HLT payload is published for each of `stack_val` and `final_test`;
per-graph inference bindings authenticate that payload without duplicating
it. The three independent pre-lock/oracle/control branches are joined by an
explicit authenticated evidence node before the execution lock, eliminating
the former multi-parent publication race. Direct HLT capacity-control
inference uses an explicit particle-batch interface, while RETB graphs use
the RETB HLT-input interface; internal model `TypeError`s cannot be hidden by
fallback dispatch. Hand-authored factory-input JSON remains invalid. The
literal miniature done criterion remains an execution acceptance test until
the real Tigris smoke runs.

## Final definition of production readiness

The campaign is production-ready only when all of the following are true:

1. all 63 manifest-driven nodes have genuine automatic producers;
2. all 50 non-bootstrap manifests are emitted without manual row JSON;
3. every manifest is source-, campaign-, graph-, trigger-, and parent-bound;
4. a real miniature campaign executes Stages A–N using the same workers and
   continuation mechanisms as production;
5. restart, partial-array failure, reuse, source drift, and interrupted
   continuation tests pass;
6. negative scientific performance never terminates the campaign;
7. the corrected readiness gate reports zero missing plan producers;
8. the authenticated production dry run passes;
9. full-submission authorization is generated only after the real Tigris smoke
   completes.

No remaining task may be declared complete based only on unit contracts,
synthetic outputs, registered entry points, or a structurally valid DAG.

## Submission-readiness closure (2026-07-30)

The source implementation is ready to submit for the required real Tigris
miniature smoke:

- the production graph has 84 nodes: 63 manifest-driven workers, 19 direct
  workers, and 2 virtual aliases;
- all 50 non-bootstrap manifests have genuine automatic producers;
- the local operational traversal resolves every Stage A--N node, verifies
  authenticated reuse and restart behavior, rejects source drift and
  incomplete arrays, and observes no performance-threshold abort;
- scale refits, scale graph training, named baselines, HLT capacity controls,
  shared inference payloads, the Stage-N evidence join, and sealed final
  inference are concrete executable work rather than placeholder rows;
- all scientific underperformance flags remain non-blocking: every
  predeclared row continues even if all available models are worse than their
  controls;
- the source-lock error now reports both expected and actual source records,
  which makes concurrent working-tree drift diagnosable without weakening
  the fail-closed comparison.

This is smoke-submission readiness, not production authorization. The next
required acceptance steps are:

1. commit and transfer one stable source snapshot to Tigris;
2. run the real miniature with `submit_retb_tigris_full.sh --smoke-submit`;
3. run the authenticated production dry run against its completed evidence;
4. generate full-submission authorization only if those operational checks
   pass.

Scientific underperformance in the miniature or full campaign must be
reported, but it does not block later rows or revoke execution authorization.
Only integrity, missing-artifact, source-drift, runtime, or scheduler failures
may stop dependent work.

## Frozen campaign source execution amendment (2026-07-31)

The mutable submission checkout is a control plane only. For every real smoke
or production submission, the launcher must:

1. resolve the committed `HEAD` of the submission checkout;
2. create a clean detached Git worktree outside that checkout;
3. re-enter the committed launcher from the detached worktree before creating
   campaign artifacts or submitting jobs;
4. bind the campaign and production graph to the detached worktree's commit
   and clean source-status hash;
5. export the detached worktree as `PROJECT_DIR` to every initial and dynamic
   Slurm job;
6. verify before each worker import that the detached worktree remains clean
   and at the bound commit.

Uncommitted changes in the mutable submission checkout are never campaign
inputs. After the detached worktree is created, the user may edit, commit,
push, pull, or create logs in the mutable checkout without changing or
invalidating the running campaign. Source-drift failures apply to the frozen
campaign checkout itself, not to the mutable submission checkout. The frozen
worktree is retained with the campaign evidence until deliberate archival or
cleanup.

The genuine miniature split cardinalities are frozen globally as 20
`model_train`, 20 `model_val` (10 `val_stop` plus 10 `val_design`), zero
`stack_train`, 10 `stack_val`, 20 `final_test`, and 40 `scale_train`
identities. The split worker must read these values from the authenticated
production graph; it may not maintain a second hard-coded miniature table.
Stage-A cache and REGION shard ranges must be derived from the same graph
cardinalities and must exactly match the authenticated cached views.

## REGION worker process-safety amendment (2026-08-01)

The compile-once REGION backend remains the sole production tree algorithm,
but parallel shard construction must not use `fork` after importing PyTorch or
loading the C++ extension. A genuine Tigris miniature row terminated by
`SIGSEGV` with no Python traceback under that process model, while all ten
input events were finite and independently passed canonical Python-reference
tree construction and validation.

Parallel REGION construction therefore uses clean `spawn` processes. Each
child must:

1. load the already-compiled campaign binary (worker JIT remains forbidden);
2. authenticate the binary against the campaign backend manifest and exact
   C++ source before processing a jet;
3. use one CPU thread so the process pool does not oversubscribe its Slurm CPU
   allocation; and
4. return the same canonical packed-tree contract used by serial execution.

Serial execution continues to authenticate and load the same binary in the
main process. This amendment changes only runtime isolation; tree topology,
floating-point rules, shard ordering, publication, scientific meaning, and
all artifact lineage remain unchanged. A repeated genuine miniature REGION
run is required to close the operational acceptance check.

The Stage-N prelock access guard must validate the exact worker interface and
campaign-relative input/output paths; it must not infer model access from
substrings in absolute path values. In particular, the repository-standard
`checkpoints/` directory name is not evidence that a row consumes a model
checkpoint. The permitted prelock command has only `--campaign-root`,
`--configuration`, and `--output`, with the configuration, attestation, and
four shared HLT payload artifacts fixed to their registered Stage-N paths.
Any additional checkpoint, logit, prediction, probability, or metric option
remains forbidden by this exact command-surface contract.

## Complete REGION role-coverage amendment (2026-08-01)

Stage A must materialize REGION trees for every role consumed anywhere in the
campaign, not merely the roles used by the initial 500k expert fits. The exact
cache universe is 18 views:

- six offline views: `model_train`, `val_stop`, `val_design`, `stack_val`,
  `final_test`, and `scale_train`;
- eight replicated HLT views: replicas 0--3 with `R_MULTI` for both
  `model_train` and `scale_train`; and
- four fixed HLT views: replica 0 with `R_FIXED` for `val_stop`, `val_design`,
  `stack_val`, and `final_test`.

The Stage-A REGION index is valid only when this coordinate set is exact and
complete. Missing, duplicate, or additional views fail closed. In particular,
pre-lock deployable `stack_val` preparation, sealed `final_test` preparation,
scale training, and post-lock oracle work may not rely on lazily generated or
unauthenticated REGION trees.

The pre-lock configuration must bind the authenticated HLT-cache metadata,
HLT array-content hash, identity-manifest hash, and REGION-tree manifest for
both `stack_val` and `final_test`. The live worker reloads those artifacts and
requires exact campaign-source, split-manifest, identity, and content lineage
before publishing either shared payload. The final-test REGION-sidecar hash is
the real finalized tree-manifest hash, not a synthetic placeholder derived
from the HLT-cache hash. Corrected tree indexes, configurations, pre-lock input
attestations, production graphs, and Step-15 bundles use new contract/schema
versions and are not interchangeable with the earlier nine-view artifacts.

Resource-probe artifacts have one canonical consumer-visible location:
`job_ledgers/resource_probes/{cpu,gpu}.json`. Probe workers must publish there,
and all training manifest factories must consume that same path. A successful
probe written to a different legacy filename does not satisfy the training
dependency.

## Stage-B prerequisite and installed-Weaver amendment (2026-08-01)

The genuine miniature established that four Stage-B execution paths required
additional explicit implementation. These corrections preserve every run ID,
scientific configuration, fixed epoch budget, and non-blocking treatment of
scientific underperformance.

Before the 147-row Stage-B array starts, Stage A must now train the ordinary
offline `O_BASE` and all-family `O_FULLREL` teachers on separate GH200 jobs so
they may execute in parallel. A dependent GPU finalizer must authenticate
both fixed-budget registrations and checkpoints, select
`SELECTED_STRONGEST` using `val_stop` only, publish the attachment-pretraining
record, and build source-bound teacher-logit views for every non-CE expert
loss. Stage-B KD rows consume the augmented `model_train` and `val_stop` NPZs,
not the teacher-free offline input files. Warm and attachment rows consume the
published `O_BASE` checkpoint. The Stage-B array depends on the finalizer, so
none of these rows can race missing prerequisites.

`TOK_WEAVER_CLASS` is an executable ordinary Particle Transformer control,
not a rejected placeholder. It receives the registered expert's full particle
view and relation family, classifies through Weaver's ordinary class-token
head, and does not introduce a RETB summary-token bottleneck. Its diagnostic
token tensor is an explicitly non-consumable zero sentinel at the registered
shape; only its logits are scientific outputs.

The directional pair stem must support the installed Weaver layout in which
the terminal head `Conv1d` is followed by `BatchNorm1d`. The per-layer
projection clone includes that post-projection normalization. Unsupported
post-projection modules still fail closed. Ordinary `O_BASE` adapter
checkpoints use the `classifier.mod.*` prefix, which is an authorized source
for particle-backbone-only warm initialization; relation and summary-token
parameters remain excluded.

The corrected production graph, node-execution registry, and Step-15 bundle
are versioned respectively as v33, v15, and v28. The two teacher-training
nodes, teacher-cache finalizer, and all Stage-B rows remain resumable. A failed
or weak metric never cancels future work; only execution or authenticated
artifact failure blocks an `afterok` dependency.

## Installed-Weaver and GH200 backward amendment (2026-08-01)

The second genuine Stage-B miniature identified three deterministic runtime
compatibility requirements. The installed Weaver configuration uses
`use_pre_activation_pair=False`, so its final pair-head convolution is
followed by both `BatchNorm1d` and the configured activation. Directional
layerwise projections must clone that complete Conv--normalization--activation
tail. The supported activation set is the explicit Weaver set used by the
campaign (`GELU` or `ReLU`, with `SiLU` accepted for compatible installed
variants); arbitrary post-projection modules remain rejected.

Teacher logits are nested mappings in collated batches. Device transfer must
therefore recurse through mappings, lists, and tuples so every KD target is on
the same device as its student logits. Keeping the containing mapping on CPU
is forbidden.

Torch 2.13 on GH200 selected a fused SDPA backward kernel that rejected
Weaver's transposed head layout for frozen-backbone attachment rows with an
`LSE ... strideH` error. All offline expert configurations now use one global
backend policy: Weaver SDPA dispatch is disabled and CUDA flash,
memory-efficient, and cuDNN SDPA are disabled while the mathematical backend
remains enabled. This changes only the kernel implementation of the same
attention equation, applies equally to every expert configuration, and is
serialized in the v2 offline-expert training contract, curves, and
registration. Old v1 and corrected v2 training artifacts are not
interchangeable.

## Stage-A teacher loss-contract dependency amendment (2026-08-01)

The Stage-A `O_BASE` and `O_FULLREL` teacher workers consume the immutable
`registry/retb_expert_losses.json` artifact. The production DAG must therefore
publish `step4_offline_training_contracts` before either teacher is submitted,
and both teacher nodes must declare that publication as an exact `afterok`
dependency. It is invalid to rely on submission timing or to recreate the
loss registry privately inside a teacher worker.

This is an execution-order correction only: teacher configurations, loss
definitions, run identities, seeds, access rules, and scientific meaning are
unchanged. The two teachers still become runnable together and retain full
parallelism after the short contract-publication barrier. The corrected
production graph, node-execution registry, and Step-15 bundle are versioned
respectively as v35, v16, and v30; their schemas are 32, 14, and 28.

## Tigris runtime-compiler pinning amendment (2026-08-01)

The GH200 Torch/Triton runtime may compile a small CUDA-driver helper during
the first attention backward pass. On Tigris, unconstrained compiler lookup
resolved `gcc` to `/tools/bin/blindfold/gcc`, causing both Stage-A teachers to
fail before their first optimizer update. Every RETB worker must therefore
pin `CC=/usr/bin/gcc` and `CXX=/usr/bin/c++` after conda activation and fail
closed if either executable is absent. The same paths are serialized in the
production graph and node-execution registry.

This correction changes only runtime tool discovery; model equations,
precision policy, kernels selected after helper initialization, seeds, data,
and scientific meaning remain unchanged. The corrected production graph,
node-execution registry, and Step-15 bundle are v36, v17, and v31, with
schemas 33, 15, and 29 respectively.

## Diagnostic model-device ownership amendment (2026-08-01)

Every call to `collect_expert_diagnostics(model, loader, device=...)` must
place both the model and recursively moved batch on the requested device
before inference. Callers may supply a freshly reconstructed CPU model loaded
from a CPU-mapped checkpoint; the diagnostic boundary owns device agreement
and may not assume that training-time placement persists across processes.

This corrects the Stage-B `offline_optimization_selector` CUDA evaluation and
all later users of the shared diagnostic routine. It changes no checkpoint,
model equation, metric definition, selection rule, run identity, or
scientific meaning.

## Parameter-free fusion dual-split execution amendment (2026-08-01)

Every Stage-C `F_BEST_SINGLE` and `F_UNIFORM_LOGIT_MEAN` row consumes the
authenticated `val_stop` and `val_design` frozen-token caches. Best-single
expert identity is selected only on `val_stop`; that locked choice is then
evaluated without parameter updates on both splits. Uniform logit mean is
likewise evaluated on both splits without fitting. Each row publishes a
separate immutable evaluation for each split, and best-single rows also
publish their immutable selection artifact.

The task interface must therefore accept `--val-design-cache`, bind both
cache manifests as inputs produced by `offline_fusion_cache`, and require all
corresponding outputs. The corrected static experiment plan and bundle are
versioned v6. Run identities, fusion definitions, selection rules, and
scientific meaning are unchanged.

## Optimization-selector checkpoint semantics amendment (2026-08-01)

The Stage-B optimization selector must reconstruct each expert with every
state-dictionary-bound semantic option from its registered configuration.
In particular, `particle_dropout` remains the exact screened value (`0.0` or
`0.1`) even though evaluation disables stochastic behavior through
`model.eval()`. Hard-coding zero dropout changes the module's serialized RETB
semantics and is forbidden. Evaluation may disable activation checkpointing,
which is explicitly not a learned or state-dictionary-bound semantic.

This correction enables strict loading of the existing registered
checkpoints and changes no parameters, inference equation, metric, selection
rule, run identity, or scientific meaning.

## Stage-C analysis and miniature-correlation amendment (2026-08-01)

Stage-C shape selection compares the complete mapping-valued campaign source
record from each inference and registration artifact directly with the
campaign source. An absent source remains permitted only where the existing
artifact contract explicitly permits it. Source dictionaries must never be
placed in a set or otherwise required to be hashable; a differing present
source still fails closed.

Pairwise complementarity correlations use one global deterministic policy.
The two flattened inputs must have exactly equal shapes or evaluation fails.
For equal-shaped inputs with fewer than two observations, the correlation is
mathematically undefined and is serialized as JSON `null`. A constant input
also yields `null`. Otherwise the ordinary finite Pearson correlation is
reported. This convention is especially relevant to the genuine miniature
`val_design` split, which has exactly one event per class, and is unchanged
for production classes with sufficient support.

The corrected complementarity report uses
`retb_offline_complementarity_v2`, schema version 2, and serializes the policy
inside the report. Version-1 and corrected version-2 evidence are not
interchangeable. This amendment changes no trained parameters, fusion logits,
selection metric, run identity, or scientific underperformance policy.

## Stage-C authoritative class-metric amendment (2026-08-01)

The authoritative classification evaluator serializes
`per_class_efficiency` as a mapping keyed by the frozen ten-class order. The
uniform-shape selector must require exactly those ten keys and canonicalize
the mapping in `QCD,Hbb,Hcc,Hgg,H4q,Hqql,Zqq,Wqq,Tbqq,Tbl` order. Iterating
mapping keys as if they were positional numeric values is forbidden. Missing,
additional, nonfinite, or out-of-range class efficiencies fail closed.

The uniform-shape metric artifact now carries the explicit class order and
uses `retb_uniform_shape_metrics_v2`, schema version 2. Its downstream shape
selection uses mapping-valued per-class means and
`retb_offline_shape_selection_v2`, schema version 2. Version-1 artifacts are
not interchangeable with the corrected contracts. The class metrics,
selection thresholds, tie rules, run identities, and non-blocking treatment
of scientific underperformance are unchanged.

## Stage-C capacity-control shape-lock path amendment (2026-08-01)

The authoritative shape-selection lock is
`selection/stage_c/locked_offline_shapes.json`, exactly as declared in the
shape-selector output manifest. Every Stage-C capacity-control lineage and
model-construction read must consume that path. The nonexistent legacy path
`selection/locked_offline_shapes.json` is forbidden. The broadly consumed
`selection/retb_offline_shapes.json` remains an immutable byte-identical
publication alias, but it does not replace the selector's canonical lock in
capacity-control lineage.

This is a producer/consumer path correction only. It changes no artifact
content, contract version, selected shape, capacity-control definition,
checkpoint semantics, run identity, or scientific underperformance policy.

## Stage-C capacity-evaluation identity-key amendment (2026-08-01)

Capacity-control evaluation consumes the production offline-expert collator
without a private batch schema. Event identities are therefore read from the
canonical `event_identities` field emitted by
`collate_offline_expert_batch`; a nonexistent `identities` batch field is
forbidden. The prediction artifact continues to serialize the collected
values under its established `identities` array field.

The production collator-to-capacity-evaluator boundary must be covered by an
integration regression with all ten classes and exact identity ordering. This
is a runtime interface correction only and changes no training update,
checkpoint, metric, identity value or order, contract version, control
definition, run identity, or scientific underperformance policy.
