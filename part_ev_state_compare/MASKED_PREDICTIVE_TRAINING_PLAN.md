# Masked predictive training on full LW State-EV

Recorded 2026-11-23. **Detailed implementation specification; not implemented,
benchmarked, or queue-ready.** This supersedes the short masked-training note
only. It changes no model code, existing campaigns/jobs, environments, data or
other idea plans.

The future deliverable is a classification-first training method with
**independent SPORC tier3 and Tigris benchmark/production pipelines**. Either
or both may run. Package names, schemas and CLI modes below are proposed
contracts, not existing commands. No classification gain is promised.

## 1. Objective and scientific hypothesis

Improve the representation learned by full, noncompact, standard-relations,
all-eight Layerwise State-EV using an additional **location-conditioned masked
latent prediction task**. Keep clean supervised classification on the original
full batch at every update, and finish with classification alone.

The hypothesis is better contextual representation learning, not that removing
important particles preserves a jet's label. Do not apply CE to masked jets,
force clean/masked logits to match, replace clean exposures with pretraining,
or distill predictions from a previously trained classifier.

The deployed model remains exactly ordinary LW State-EV:

- `part_ev_state_compare.model.build_model('state_layerwise', ...)`.
- Width 128, eight particle blocks, eight attention heads.
- Shared standard width-64 pair stem, eight independent final bias heads.
- Ordinary EV and width-32 State-EV in all eight particle blocks.
- Original FFNs, head/residual scales, dropout, BN/trimmer and class readout.
- Existing query chunk 64 and shared per-forward pair normalization.
- **2,327,462 deployed parameters**, with no extra inference module or forward.

No persistent workspace, experts, multiscale hierarchy, selected relations,
new pair features, pair-memory branch, ensemble or external teacher is included.

Historical 86.3151% validation is context only. It is not independent test
performance or automatically a matched control, especially across two-A100/
SyncBatchNorm and one-GH200/native-BatchNorm execution.

## 2. Locked first configuration

Science ID: `masked_latent_state_layerwise_v1`.
Candidate run ID: `MASKED_PREDICTIVE_STATE_EV_LAYERWISE`.
Control: `LW_STATE_EV_CONTROL`.

| Choice | First implementation |
| --- | --- |
| Student | Unchanged full LW State-EV, fresh seed 101 |
| Clean classification | All original 512 jets every attempted update |
| Auxiliary selection | 256 of those jets, evenly allocated across ranks |
| Selection | Uniform without replacement, independent of labels/eligibility |
| Mask amount | Nominal 25%; exact count defined in section 5 |
| Mask modes | Half selected slots uniform; half angular patches |
| Eligible jet | At least four retained valid particles and a finite nondegenerate visible reference |
| Teacher | Detached particle-only EMA copy; no class token/blocks/classifier |
| Targets | Non-affine-normalized post-block-6 and post-block-8 teacher states, averaged then normalized |
| Predictor | Width 128, two cross-attention/FFN blocks; four heads, FFN width 256 |
| Queries | Learned mask vector plus MLP of target angular location only |
| Source | Masked student's visible post-block-8 particle states |
| Loss | Clean CE plus scheduled normalized smooth-L1 latent loss |
| Teacher momentum | Cosine increase 0.996 -> 0.9999, attempted-update horizon |
| Auxiliary stochastic layers | Dropout/MHA dropout off; no second trim |
| Auxiliary BN | Eval arithmetic with immutable pre-clean-forward buffers |
| Microbatch | At most 32 selected jets per rank per auxiliary slot |
| Checkpointing | Existing student State-EV chunks retained; predictor uncheckpointed |
| EMA / optimizer | One optimizer/scaler decision per original clean batch |
| Selection/export | Clean student only, never teacher or predictor |

This is joint supervised/auxiliary training, not a two-stage SSL budget.
Auxiliary forwards add compute on existing jets; they do not add independent
training data or reduce the original supervised exposure.

Do not silently change mask fraction, teacher, loss weight, auxiliary population,
normalization, microbatch or schedule to fit memory or make tests pass.
A revised method needs a new science/runtime identity and fresh evidence.

## 3. Parameter inventory and ownership

The student contains all 2,327,462 reference parameters. The teacher retains only
particle embedding, pair encoder/bias heads, and the eight EV/State-EV particle
blocks. Exclude:

| Excluded student component | Parameters |
| --- | ---: |
| Two class-attention blocks | 399,376 |
| Final class norm | 256 |
| Classifier | 1,290 |
| Class token | 128 |
| Total excluded | 401,050 |
| **Frozen particle-only teacher** | **1,926,412** |

This count is checked against the current reference inventory; implementation
must enumerate keys and prove strict correspondence to that student subset.

Predictor counts, from the exact definitions in section 8:

| Predictor component | Parameters |
| --- | ---: |
| One biased Q/K/V/output attention | 66,048 |
| One biased `128 -> 256 -> 128` FFN | 65,920 |
| Three affine width-128 norms per block | 768 |
| One predictor block | 132,736 |
| Two blocks | 265,472 |
| Position MLP `3 -> 128 -> 128` | 17,024 |
| Learned mask embedding | 128 |
| Final output linear `128 -> 128` | 16,512 |
| **Training-only predictor** | **299,136** |
| **Student plus predictor, trainable** | **2,626,598** |
| **Trainable plus frozen teacher, resident parameters** | **4,553,010** |
| **Exported classifier** | **2,327,462** |

No optimizer state or gradients for teacher parameters. Teacher is outside DDP;
student and predictor are together inside the trainable wrapper/DDP.
Save teacher state in full restarts, but never in selected/exported classifier
weights. Stored parameter count is not a training FLOP/latency ratio.

## 4. Capture the real clean view, not a second augmentation

The clean forward must reproduce the original model and preprocessing exactly.
Use the unchanged official `JetClass_full.yaml` and loader; do not modify the
ROOT stream, class labels, feature transform or padding for CE/evaluation.

Expose an explicit clean-forward result containing logits plus **detached actual
post-native-trim inputs**: aligned 17 features, four-vectors and valid mask.
Capture them before embedding, not hidden activations. The native trimmer runs
once, with its original random permutation/truncation and counter.

Teacher and visible auxiliary views are built from this retained collection,
not the larger raw jet before trimming. They must never independently retrim,
select a different population or consume original trimmer/dropout randomness.

The teacher sees all valid particles in that retained view. The visible student
sees only the retained particles not selected as targets. Save an explicit
mapping from auxiliary rows/queries to retained particle positions for alignment;
do not infer target correspondence from tensor length or sorted particle rank.

All auxiliary target/source paths use fresh embeddings and fresh native pair
calculations. No clean-student hidden state, attention matrix, normalized pair
cache, class representation or current clean BN update may reach the predictor.

## 5. Selection, masks, geometry and RNG

### Selected jets

For each attempt with positive auxiliary coefficient, select exactly half of
each local clean batch without replacement: 128 of 256 per A100 rank, or 256
of 512 on one GH200. Selection does not depend on label, loss, multiplicity or
whether a jet will be eligible. Do not resample an ineligible choice.

Randomly ordered selected slots are split evenly into uniform-mask and patch-mask
modes. This balances nominal modes, not necessarily eligible modes. Log actual
eligible counts by mode.

Use independent, versioned **per-rank CPU torch generators** for jet selection,
particle priorities, patch centers and tie priorities. Draw fixed-capacity
random arrays for selected slots and 128 particle positions before eligibility
filtering; unused entries are ignored. Do not use global Python/NumPy/torch/CUDA
RNG or DataLoader-worker state. Store all generator states in full restarts.
Transfer each bounded set of random priorities once, then build masks/geometry
in batched tensor operations. Do not add a per-particle/jet GPU-to-CPU `.item()`
loop or a second data fetch to construct the views.

No auxiliary draws on coefficient-zero attempts. Every positive-coefficient
attempt consumes the declared draws even if no selected jet is eligible.
A skipped optimizer update still consumes that attempt's masks and is checkpointed
as such. Raw-stream replay does not draw masks.

### Exact mask count

For n retained valid particles:

```text
if n < 4: classification only
otherwise m = min(n-2, max(1, floor(n/4)))
```

Thus m is at most 32 and at least one, and at least two particles remain visible.
Do not round differently on different platforms or replace m with a Bernoulli
mask whose count occasionally removes the whole jet.

Uniform mode chooses m valid particles by independent random priorities.
Patch mode chooses a valid center uniformly, then takes m nearest valid particles
in squared pseudorapidity–wrapped-azimuth distance, including the center.

Mask geometry uses raw retained directions, not truth/PID or hidden states.
Use FP32 with finite zero-pt handling. The predictor's angular coordinates are
rebuilt later relative to the visible-only axis, not this temporary patch center.

### Ordering and patch ties

Assign random priorities in a canonical input-content order using four-vectors
and the ten jet-independent feature channels described in section 6.
Never use full-jet-relative feature values, labels, file names or learned states
as canonical metadata.

For patch distance ties, use the independent random tie priority assigned in
that canonical order, then canonical content to break a priority collision.
This avoids preferentially masking low/high energy just to resolve an equal
angular shell. Exactly identical content is interchangeable; selecting one of
identical copies has the same visible/target content multiset.

Mask augmentation is permutation-equivariant in distribution. Fixed-generator
tests on distinguishable canonical content should align exact selected content;
do not demand impossible distinct identities for truly identical copies.
Keep replay masks/indices exact on the same restored input/order.

Hold mask membership and supplied target angles fixed during leakage tests.
Re-running mask selection after perturbing hidden values tests a different
conditional task and is not a valid no-leakage comparison.

## 6. Visible-only auxiliary preprocessing

The existing loader returns already transformed/clipped features, not raw
invertible inputs. **Do not invert clipped features** to recover momenta or
reuse original relative-to-full-jet channels.

Sufficient data already exists: untransformed retained `pf_vectors` plus the
ten independent feature channels. No second ROOT read, raw-feature side dataset
or new loader/YAML is required for this locked design.

Rebuild these seven channel indices for both teacher and visible student:

| Index | Name | Auxiliary raw value before native transform |
| ---: | --- | --- |
| 0 | part_pt_log | `log(max(pt_i,eps))` |
| 1 | part_e_log | `log(max(E_i,eps))` |
| 2 | part_logptrel | `log(max(pt_i,eps)/pt_visible_jet)` |
| 3 | part_logerel | `log(max(E_i,eps)/E_visible_jet)` |
| 4 | part_deltaR | `hypot(deta_visible,dphi_visible)` |
| 15 | part_deta | Sign-oriented pseudorapidity difference to visible axis |
| 16 | part_dphi | Wrapped azimuth difference to visible axis |

Copy only indices **5–14** from the clean retained features: charge, five PID
indicators, tanh(d0), clipped d0 error, tanh(dz), clipped dz error. They contain
no per-jet normalization in the pinned YAML. Re-audit that assertion if the
source config changes.

### Common reference and coordinate formula

Let `Vvis` be the sum of actual visible four-vectors, with epsilon `1e-8`.
Use vector pT, not sum of scalar constituent pT:

```text
pt_visible_jet = hypot(sum_visible px, sum_visible py)
E_visible_jet  = sum_visible E
eta_i = asinh(pz_i / max(pt_i, eps))
eta_vis = asinh(Vvis.pz / pt_visible_jet)
phi_i = atan2(py_i, px_i), with phi=0 for a zero-pt particle
phi_vis = atan2(Vvis.py, Vvis.px)
sign_vis = +1 if eta_vis >= 0 else -1
deta_visible_i = sign_vis * (eta_i - eta_vis)
dphi_visible_i = atan2(sin(phi_i-phi_vis), cos(phi_i-phi_vis))
```

Require finite positive visible-jet pT/energy above epsilon. A finite but
degenerate visible reference makes that selected jet auxiliary-ineligible
(classification still runs); malformed/nonfinite valid input is an error,
not a silently excluded jet.

Teacher full-particle inputs and target-location queries use this **same
visible-defined** reference. Hidden momenta therefore cannot change visible
normalizations or axis. Teacher still receives true hidden features as targets;
the predictor/source do not.

Apply the frozen YAML's original manual transforms, channel order and clipping:
`(x-center)*scale` when center is specified, including native bounds.
For indices 0/1 centers/scales are 1.7/0.7 and 2.0/0.7; 2/3 use -4.7/0.7;
index 4 uses 0.2/4.0, all with default [-5,5]. Indices 15/16 have center=null
and are not newly standardized/clipped. Read/authenticate actual config values
rather than separately maintaining drifting magic constants.

This is an explicitly defined auxiliary re-expression, not a claim that every
raw dataset angular convention can be recovered by reversing preprocessing.
Clean CE and evaluation keep their original supplied features unchanged.

### Physical removal and padding

Compact visible tokens **before** student embedding, pair encoder, attention
bias, ordinary EV and every State-EV computation. The masked four-vectors,
PID, track attributes and original relative features are absent.

Rebuild wrap padding from that jet's visible tokens only, with constant-zero
valid mask. Teacher padding is rebuilt from its full retained collection.
Use aligned compaction/gather IDs; never reuse old wrapped slots that can contain
a hidden particle. No fake target token is appended to the student encoder.

Rebuild `pf_points` from the new angular channels too, even though ordinary
ParT ignores points. Do not pass jet observers, original jet axis/energy/pT,
particle rank, original valid-slot index or mask center to the predictor.

The task intentionally supplies hidden angular locations and target count.
It is **location-conditioned** prediction, not blind recovery of all missing
information; describe that conditioning honestly.

## 7. Teacher, targets and EMA transaction

Build teacher from a fresh isolated construction, strictly copy the initialized/
DDP-synchronized student's matching particle keys, remove class-only components,
set all teacher parameters requires_grad=False and keep it eval/no-grad.
No external trained weights, target optimizer, gradient or teacher CE.

Teacher features are the post-block-6 and post-block-8 particle states, after
the complete native block including FFN/residual, before any class read.
Use true teacher full-particle alignment to gather target positions:

```text
t6 = LN_no_affine(float32(H_teacher_block6), eps=1e-5)
t8 = LN_no_affine(float32(H_teacher_block8), eps=1e-5)
target = stop_gradient(LN_no_affine((t6+t8)/2, eps=1e-5))
```

Normalize over 128 channels per particle, not over particles or the batch.
No batch centering, codebook, label projection, contrastive negatives or target
cache is included.

Use FP32 teacher parameters, FP16 teacher forward, FP32 target normalization.
Teacher dropout is off; its disabled trimmer performs identity binding only,
with no permutation, truncation or counter/RNG advance. Capture block states
through explicit forward outputs, not persistent hook activations.

### EMA timing

For one-based attempted update u in 1..1,000,000:

```text
m(u) = 0.9999 - (0.9999-0.996) * (1+cos(pi*(u-1)/999999))/2
```

After **one successful complete student/predictor Ranger step**, including any
Lookahead interpolation, update matching teacher parameters in FP32:

`teacher = m(u)*teacher + (1-m(u))*student_particle_parameters`.

Teacher targets for update u use the **previous** teacher; do not update it after
clean backward but before auxiliary prediction. Skip parameter EMA on an
AMP-skipped optimizer step. Store successful-EMA count separately from u;
momentum is attempt-indexed, never silently re-indexed by successful count.

After every fully completed attempt, including a scaler skip with valid clean
forward, copy clean-student BN running means/variances and integer counters
directly into teacher. Do not EMA integer buffers or learn statistics from
teacher/visible views. This buffer-copy-on-skip policy is deliberate; teacher
parameters/EMA counter still do not advance on that skip.

EMA/buffer maintenance continues through warm-up and clean-only finish.
Fatal/incomplete attempts do not commit new teacher/progress state.

## 8. Exact predictor and prediction loss

For each eligible jet, visible student post-block-8 states provide K/V.
Each hidden-particle query contains only:

```text
location = [deta_visible, sin(dphi_visible), cos(dphi_visible)]
Q0 = learned_mask_vector + Wpos2(GELU(Wpos1(location)))
```

The position MLP is biased `3 -> 128 -> 128`, no positional rank embedding.
The mask vector is length 128. Targets do not attend to each other: no decoder
query self-attention or hidden-to-hidden pair features.

For each of two independently parameterized blocks:

```text
Q1 = Q + CrossAttention(LN_query(Q), LN_visible(Hvis), visible_key_mask)
Q2 = Q1 + Wff2(GELU(Wff1(LN_ffn(Q1))))
```

Each block has three independent affine LayerNorms, epsilon `1e-5`.
Attention has four heads of width 32, separate biased Q/K/V/output linears,
scale `1/sqrt(32)`, no dropout/bias terms from particle pairs.
FFN is biased `128 -> 256 -> 128` with GELU, no hidden norm or dropout.

After block two:

```text
prediction = LN_no_affine(Wprediction(Q2), eps=1e-5)
```

`Wprediction` is biased `128 -> 128`. No other predictor norm/projection/scale.
All new learned weights start nonzero Xavier-uniform, gain one, biases zero;
affine norms start one/zero; mask vector normal standard deviation 0.02.
Use isolated stable mechanism seeds; preserve native initialization/RNG.

Compute attention scores, softmax, weighted reductions, norms and loss in FP32;
linears use FP16 training autocast. CPU FP64 oracle mode retains float64.
Accumulate predictor query/residual states in FP32, casting linear outputs
before residual additions; do not silently keep the entire predictor in FP16.
Mask padded queries after every residual/output; no all-masked softmax on an
active query. Inactive rows may use finite-zero score rows then zeroed outputs.

Use smooth-L1 beta=1, then average **channels, hidden particles, eligible jets**:

```text
loss_jet_j = mean_hidden_targets(mean_channels(smooth_l1(prediction,target)))
L_mask = sum_eligible_j loss_jet_j / E_global
L_total = CE_clean + lambda(u)*L_mask
```

Do not weight long jets more by flattening all targets globally. E_global is
actual eligible selected jets after visible-reference checks. If E_global=0,
L_mask is zero and no target/source encoding is required.

Teacher targets are detached, but visible student states must not be detached:
auxiliary gradients must reach the student particle backbone. Predictor-only
learning is not the desired method.

## 9. Auxiliary coefficient and unchanged LR

One-based update u removes schedule ambiguity:

| Update interval | lambda |
| --- | --- |
| `1 <= u <= 20,000` | 0 |
| `20,000 < u < 40,000` | `0.1*(u-20000)/20000` |
| `40,000 <= u <= 800,000` | 0.1 |
| `800,000 < u < 900,000` | `0.1*(900000-u)/100000` |
| `900,000 <= u <= 1,000,000` | 0 |

First positive coefficient is u=20,001. There are **879,999** positive-coefficient
attempts and **120,001** clean-only attempts. This schedule is not optimizer
warm-up. Do not adapt lambda based on validation, collapse metrics or gradient
norm after seeing the run.

Keep original pinned Ranger/LR/global-batch/exposure. Use
`part_ev_fullscale.protocol.learning_rate(u-1,recipe)` with zero-based milestones
35–49, first decay at attempted update 700,001. Attempt cursors advance on scaler
skips; Lookahead advances only with successful optimizer steps.

One optimizer contains student and predictor in stable order, one GradScaler,
one LR setting, one final unscale/check/step/update per clean batch.
No teacher optimizer, predictor LR, weight decay, gradient clipping, freeze of
the student or new augmentation of clean CE.

## 10. BatchNorm isolation without version-counter bugs

This section is mandatory; “put the auxiliary model in eval” alone is insufficient.

At the start of each positive-coefficient attempt, let B0 be an immutable clone
of all student BN buffers before the current clean forward. With SyncBN, verify/bind the common
pre-clean buffer identity. Clean CE runs in original training mode and creates
updated buffers B1. **Finish clean backward before auxiliary work**, freeing
the clean activation graph.

Auxiliary student uses B0; teacher uses its previous clean buffer copy.
Neither may see B1 from the current batch, which can encode the missing particles.

Use a functional buffer substitution, e.g. public `torch.func.functional_call`
on the **inner student**, with the actual attached student parameters and an
immutable cloned buffer dictionary. Parameters must remain graph-connected and
identical objects; do not clone/detach student weights.

BN/SyncBN executes eval arithmetic with B0; native dropout modules and native MHA
dropout probability are zero, and trimmer.enabled=False. Do **not** indiscriminately
call `student.eval()`: keep state-message modules in training mode so their
existing pure checkpointed backward remains active. Scope and restore exact
per-module flags, dropout probabilities, trimmer state and native AMP controls.

Do not copy B0 into B1 tensors in place while autograd may have saved them.
Functional substitution restores registered B1 buffer references after forward,
while autograd retains unchanged B0 tensors. Hold the auxiliary mode context
through backward; release B0 graphs before any mutation. Restore on exception
too. Prove actual API/hook/recomputation behavior on the pinned torch version.

Scope native/outer AMP on the actual inner student and teacher that own `.mod`,
not a trainable bundle that lacks that attribute. Teacher eval/no-grad mode
does not imply FP32 here: its target forward deliberately uses FP16 autocast,
followed by FP32 normalization. Student validation/export is FP32.

At end of auxiliary work, all clean student buffers remain exactly B1, with
one clean-forward update/counter increment per attempt and no auxiliary updates.
Original trimmer counters advance only during their normal clean behavior.
No auxiliary dropout or random attention operation consumes native RNG.

Teacher is always eval/no-grad, with its own buffers/cache and no SyncBN
training collective. At the end of the completed attempt, copy B1 to teacher
as specified in section 7.
Check B1 for finiteness before committing/copying it; a finite loss alone is
not permission to save corrupted running statistics.

Snapshot buffers are small per-step objects, not persistent checkpoint payloads.
Never save a mid-auxiliary B0/B1 transaction as if it were a complete update.
Coefficient-zero attempts need no B0 snapshot or retained auxiliary view; clean
BN still updates normally and its end-of-attempt buffers are copied to teacher.

## 11. Sequential gradients, DDP and microbatching

### Wrapper and synchronization

Wrap **student + predictor**, not teacher, in one trainable bundle/DDP.
Provide explicit phase forwards; clean returns logits/retained inputs,
auxiliary returns attached prediction loss/statistics. DDP wraps every trainable
forward; calling an unwrapped custom training method bypasses its protocol.

SPORC uses `broadcast_buffers=False` for this wrapper: clean SyncBN synchronizes
its statistics, and the auxiliary phase must not broadcast/mutate B0/B1 buffers.
Record this choice and use the same clean SyncBN/buffer policy in fresh controls.
Verify initial and post-attempt buffer identities rather than assuming it.

Use `find_unused_parameters=False` and `static_graph=False`. Attach a mathematically
zero dependency on every trainable parameter to phase losses, including unexecuted
class/predictor components. A fixed stacked first-element dependency is sufficient
if tests verify full zero gradients. No dummy valid jet, fake target or extra
label should be encoded merely to satisfy DDP.

### Positive-coefficient update

1. Snapshot B0 and prepare original clean input.
2. Run clean forward **and backward inside DDP.no_sync()**; scale clean CE.
   Retain only detached trimmed inputs and accumulated gradients, not its graph.
3. Draw auxiliary selections/masks, build eligible views and reduce E_global
   once across ranks. Agree health before any new collective.
4. Process selected slots in fixed microbatches of at most 32: four per A100
   rank or eight on one GH200. Teacher no-grad targets and visible student/
   predictor graphs exist only for the current slot.
5. Each rank's slot loss for backward is
   `lambda(u) * world_size/E_global * sum_local_eligible(loss_jet)`.
6. Put all nonfinal slot forwards/backwards inside no_sync; the last slot is
   synchronized and flushes accumulated clean plus auxiliary gradients.
7. Unscale/check once, perform one Ranger/scaler decision, then teacher/buffer/
   progress maintenance.

DDP averages gradients, hence the world_size factor for the globally averaged
eligible-jet loss. Clean CE remains ordinary local-jet mean with equal local
batch sizes; do not rescale CE by particle/eligible count.

For a rank-local empty slot, call the same DDP phase with zero anchors/statistics,
without teacher/student encoding. The last slot still triggers all parameter
reduction hooks. No added model collective or training BN occurs in auxiliary
views; different local eligible counts therefore do not strand peers.

If E_global=0, complete exactly one synchronized zero-anchor phase to flush
the already accumulated clean gradients; no division or fake encoding.
All ranks take this branch together.

### Coefficient-zero and optimizer details

Coefficient-zero attempts use one ordinary synchronized clean forward/backward,
with predictor zero anchors and no auxiliary selection/encoding. After required
DDP reduction, set predictor grads to None when the auxiliary is globally inactive,
before optimizer unscale/step. This avoids advancing unused predictor RAdam
moments throughout warm-up; never base it on a rank-local empty subset.
All parameters remain registered in the same Ranger.

Native Lookahead still follows its global cadence, including registered slow
predictor weights; do not invent an independent predictor Lookahead counter.
On an active update a rank-local empty predictor receives reduced gradients
from peers normally. No reset of optimizer state at activation/fade-out.

Use one fixed scaler value for all backwards in an attempt, one unscale and one
update. Do not step between clean and auxiliary graphs, call two scaler updates
or average losses a second time over microbatch count.
Verify scale/skip decisions agree across ranks before EMA/progress commits;
an asymmetric scaler/optimizer step is a failure, not an EMA synchronization
strategy. Controlled overflow tests inject before gradient reduction so all
ranks see the same skipped update.

Public [DDP no_sync](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
requires including forwards inside its context. The full phase/anchor/functional-
buffer construction still needs actual two-process tests, including zero eligibility
and checkpoint recomputation. Do not adopt private/manual reducer APIs as a shortcut.

No checkpoint commits between these phases. Signals are handled at complete
attempt boundaries; aborts mid-attempt preserve the previous complete restart,
not partial BN/gradient/generator progress.

## 12. Integration and file organization

Proposed new package, after checking names are unused:

```text
part_ev_masked_predictive/
  __init__.py
  config.py          # science/phase schedules, counts and runtime schemas
  views.py           # isolated RNG, selection/masks/alignment and visible rebase
  predictor.py       # explicit cross-attention/FFN predictor and normalized loss
  model.py           # native-compatible clean/pretrimmed particle encoders
  teacher.py         # exact subset ownership, EMA and clean-buffer copies
  modes.py           # BN snapshot functional calls, dropout/trim/AMP restoration
  step.py            # sequential scaled backwards, DDP eligibility and anchors
  diagnostics.py     # bounded variance/leakage/shortcut/gradient diagnostics
  restart.py         # full-state, mask/EMA and real-stream probes
  protocol.py        # frozen data/source/environment/storage/admission
  benchmark.py       # phase-aware paired target-device checks and timing
  engine.py          # explicit platform train/student-evaluate/report
  submission.py      # immutable snapshots, Slurm ledgers and reviewed dispatch
  README.md          # tested actual commands only after implementation

scripts/run_part_ev_masked_predictive.py
scripts/submit_part_ev_masked_predictive.py
sbatch/run_part_ev_masked_predictive.sh
sbatch/submit_part_ev_masked_predictive.sh
tests/test_part_ev_masked_predictive_*.py
```

Student state_dict keys remain those of original LW State-EV. Predictor is a
separate bundle member; teacher outside it. Export student keys only and prove
strict load into the **unmodified** ordinary reference builder.

Use a local native-compatible explicit forward rather than mutable hidden hooks.
Clean forward calls original trimmer, embedding/pairs, all particle/class blocks
and classifier exactly once, returning detached retained inputs when requested.

Pretrimmed auxiliary/teacher encode calls the original trimmer only as a disabled
identity so existing post-trim pair-binding hooks receive the correct view.
It must not increment counters, permute or truncate. Then call original embedding,
pair encoder and eight particle blocks; no class blocks/FC. Source returns H8,
teacher explicitly H6/H8. Cache cleanup remains effective on success/failure.

Keep existing state chunk checkpointing; its backward uses explicit saved tensors
after cache cleanup. Do not reintroduce closures that query cleared/replaced BN
buffers or pair caches during recomputation.

Anchors: `part_ev_state_compare/model.py`, `part_ev_state_speed/model.py`,
`part_layerwise_screen/model.py`, `part_ev_fullscale/data.py`,
`particle_transformer/data/JetClass/JetClass_full.yaml`,
`part_ev_screen/precision.py`, `part_ev_fullscale/optimizer.py` and
`checkpoint.py`, `part_ev_state_production/engine.py` and
`part_ev_rts_tigris/production.py`.

Current generic step helpers assume one CE graph/model and one kind of restart.
They cannot implement this method by swapping a package name. Add explicit
task/platform/phase-aware code; never monkeypatch old protocol globals or change
frozen campaigns/installed Weaver.

## 13. Leakage contract and required adversarial tests

Allowed predictor information: visible p4/features, historical learned weights/
BN buffers, hidden angular queries and hidden-target count. Teacher may use
full retained features, but its **output is target-only**, detached from inputs
to the predictor.

On a fixed student/teacher/pre-clean state, fixed mask and fixed supplied query
angles, hold visible particles constant and independently perturb:

- Hidden pT/energy by scaling along fixed directions.
- Hidden PID/charge/track features.
- Original full-jet axis/energy/pT and original full-relative channels.
- Old padded/wrapped slots, including values copied from hidden particles.
- Clean-student hidden activations, pair-cache values and post-clean BN buffers.

Visible-view inputs, reference, predictor queries, source inputs and predictions
must remain unchanged within stated numerical bounds. Recompute full clean
features for the perturbed fixture so the test actually exposes old-feature reuse.
The teacher target may change; source/predictor must not. Never reselect masks
during this test.

Also verify clean BN B1 is unchanged by all auxiliary forward/backward work,
native RNG/trimmer state is restored, teacher BN receives no current-batch update
before its target forward, and no auxiliary cache references the clean graph.

Low prediction loss can result from a location-only shortcut or collapsed
teacher targets. These invariance tests alone do not establish semantic learning.
The intentionally supplied locations/multiplicity are not “zero information.”

## 14. Correctness tests and numerical policies

Required independent tests:

1. Predictor FP64 per-head/FFN oracle, mask/query padding, output bias once,
   input/parameter gradients and all-masked-active-key rejection.
2. Target tap indexing (post-block 6/8), target double-normalization, no teacher
   grad, exact strict teacher subset and no teacher class read.
3. Selection/mask cardinality, 50/50 nominal modes, no-replacement jet selection,
   canonical ties/duplicates, angular wrap, N<4 and visible-reference degeneracy.
4. Visible rebase formulas/YAML transforms, all seven replaced channels,
   untouched independent channels, visible-only padding, teacher/query alignment.
5. Every adversarial leakage test in section 13; weak masks are not enough.
6. Functional B0/B1 isolation through actual autograd, no in-place version errors,
   mode/AMP/dropout/trim restoration on normal/exception paths.
7. Exact unchanged clean initialization, native call/RNG/trimmer behavior,
   active nonzero original State-EV, and strict exported-student logits/gradients.
8. Loss mean over channels/targets/jets, unequal per-rank eligible counts,
   world-size-scaled gradient agreement with direct global oracle.
9. Sequential accumulation versus a small simultaneous CE+aux oracle using the
   same B0, masks/targets/modes/scaler; only graph lifetime differs.
10. DDP clean no_sync plus all auxiliary slots/final sync; one rank ineligible,
    empty last slots, globally zero eligible, coefficient-zero phases and
    every parameter graph-connected with finite zero gradients where appropriate.
11. Actual Ranger/predictor learning, delayed/missing gradient detection,
    one step/scaler update, inactive predictor moments and Lookahead cadence.
12. EMA after successful Ranger/Lookahead step, skip handling, attempt-indexed
    momentum, direct BN/counter copy-on-complete-attempt, no target-before-step leak.
13. Complete strict restart of student/predictor/Ranger/teacher/scaler/generators/
    progress, including schedule transitions and interrupted auxiliary work.
14. CPU/GPU dtype audit, finite validation/export, bounded graphs/microbatch
    memory and no target/cache/extra dataset writes.

Freeze first parity limits before GPU evidence:

| Comparison | rtol | atol |
| --- | ---: | ---: |
| Small FP64 predictor/loss oracle outputs/gradients | 1e-9 | 1e-10 |
| FP32 active forward / student export | 1e-5 | 1e-6 |
| FP32 gradient and sequential/simultaneous comparison | 1e-4 | 1e-5 |
| FP32 global-DDP outputs/gradients | 5e-4 / 2e-3 | 2e-5 |
| FP16 recomputation outputs | 2e-3 | 2e-3 |
| FP16 active gradients | 1e-2 | 1e-3 |

Masks, indices, generators, BN buffers at specified transaction points, initial
state, integer counters, call counts and metadata are exact. DDP local input
gradients have the expected world_size factor; shared parameter gradients match
after averaging.
Exact BN requirements here mean unchanged B0/B1 within an attempt and a direct
bitwise buffer copy to teacher. Subsequent floating BN trajectories across
separate GPU trials use the bounded restart policy, not an extra bitwise gate.

Leakage comparisons hold allowed inputs/state fixed; clean post-update parameters
are allowed to change from intended auxiliary learning. GPU library numerical
variation is not permission for hidden inputs to enter the source.

No `strict=False`, tolerance inflation, dropped teacher taps, disabled auxiliary
loss or invented BN replacement to turn a failed check into admission.

## 15. Restart evidence on actual target hardware

Require actual one-GH200/native-BN and two-A100/SyncBN correctness separately.
Exercise half-batch selection, microbatch 32, unequal eligibility, globally empty
views, frozen SyncBN auxiliary passes and checkpointed State-EV backward.
Never call a rank-zero training SyncBN reference while its peer waits.

For a global-batch DDP oracle, use fixed pretrimmed inputs, disable native dropout
only for that numerical plumbing test and compare against ordinary-BN global
reference. Restore real clean dropout/trimming for learning/replay/timing.
Cross-platform seed-101 runs are not numerically identical trajectories.

Populate student/predictor optimizer and teacher EMA state. Run reference,
uninterrupted repeat, disk-start repeat and resume-after-three trials over seven
rolling-batch updates with a separate diagnostic schema. Test warm-to-active
and fade-to-clean transitions separately by explicit probe schedule labels;
do not fake full production budgets or completion markers.

Immediate restored model/teacher/optimizer/scaler/generator/cursor state is exact.
For subsequent floating student/predictor/teacher/Ranger tensors use a versioned
bounded short-probe policy:

```text
state_abs_error <= min(1e-4, 1e-5 + 1e-4 * abs(reference))
loss_abs_error  <= min(2e-4, 1e-4 + 1e-5 * abs(reference))
```

Selection/mask/tap-alignment digests, RNG, BN copy policy/counters, lambda/
momentum/EMA-success/skip/scale schedules and scalar metadata remain exact.
Settled fixtures have no scaler skips; separately inject a controlled skip and
verify student/teacher parameter and BN behaviors.

Record every rank/trial, CE/aux/total losses, eligibility, generator/mask digests,
teacher subset/EMA checks, worst tensor paths and maximum errors. No adaptive
bounds from noisy repeats. A short probe is not a long-run accuracy guarantee.

A disposable actual-ROOT replay probe separately authenticates train/validation
batch fingerprints, loader replay without forwards/mask draws, restored rank/
auxiliary RNG and the next masks/targets. A small diagnostic boundary does not
replace production's 20k/1.28M validation budget.

## 16. Independent SPORC/Tigris execution and controls

| Setting | SPORC | Tigris |
| --- | --- | --- |
| Host / partition | Linux x86_64 / tier3 | Linux aarch64 / tigris |
| Default account | reu-aisocial | reu-aisocial |
| Training hardware | Two matching same-node A100s | One GH200, at least 80 GiB device memory |
| Clean global/local batch | 512 / 256 per rank | 512 / 512 |
| Auxiliary selected/local | 256 / 128 per rank | 256 / 256 |
| Auxiliary slots / microbatch | Four / at most 32 | Eight / at most 32 |
| Native clean BN | SyncBatchNorm | Native BatchNorm |
| Loader workers / torch threads | One / two per rank | Two / four |
| Train/benchmark CPU / host RAM | 8 / 128 GiB | 12 / 128 GiB |
| Final evaluation | One A100, saved SyncBN in eval | One GH200 |
| Interpreter | `/home/ryreu/miniconda3/envs/part_ev_screen_sporc/bin/python` | `/home/ryreu/miniforge3-aarch64/envs/part_ev_fullscale/bin/python` |

Each platform independently prepares, benchmarks, freezes source/runtime, reserves
storage and submits production. No dependency on the other environment or a
finished other-cluster benchmark. No install/environment mutation or deleted-env
dependency.

Verify actual GPU identity/memory, torch/CUDA/Weaver, driver, TF32 disabled,
cuDNN benchmark False and approved NCCL. Record DDP broadcast_buffers=False,
BN/worker/microbatch/scaler settings. No silent GPU/count/BN change.

Default production is candidate plus a fresh matched LW State-EV control on each
platform: **four full trainings if both are authorized**. Control uses identical
clean training/runtime and original CE only; no teacher/predictor/view overhead
is fabricated for its timing.

Optional control reuse must be explicitly authenticated by science, data/source,
seed, exposure, GPU/BN/worker/precision/buffer and selection/evaluation identity.
Historical 86.31% or disposable benchmark weights do not automatically qualify.

Both platform campaigns may run; no automatic winner/cancellation/cross-cluster
checkpoint migration. Compare candidate/control within platform first. Two
platforms with seed 101 are not two independent random-seed replicates.

## 17. Full training, validation and deployment contract

Use original full `ranger_full_v1`:

- Official 100M training pool, original 17 clean features, native cap 128.
- Fresh seed 101, global clean batch 512, 1M attempted updates: nominal 512M clean
  supervised exposures, not one pass over 100M unique jets.
- Pinned RAdam LR 1e-3, betas (0.95,0.999), epsilon 1e-5, weight decay zero;
  complete-state Lookahead k=6, alpha=0.5; original LR milestones/no added warm-up.
- Original clean dropout, native FP16/FP32-parameter training/GradScaler,
  no new clipping/smoothing/early stopping.
- 50 validation checks, every 20,000 attempts, 1.28M events each from the separate
  continuous validation stream. Keep original train/validation iterators alive.
- Highest **clean student** validation accuracy, first tie wins.
- Full-5M validation of selected student: accuracy, macro OVO/OVR AUC and current
  per-class QCD rejection operating points with counts/zero-background handling.
- No independent test access until a locked final comparison.

No teacher selection, ensemble averaging, masked inference, test-time predictor,
or switching exports to whichever model scores better.

Export only original student state_dict; load strictly into the unmodified
full LW State-EV builder. Prove FP32 logit parity on aligned eval/trim settings
and verify original parameter count, native calls and inference memory/throughput.
For a given exported student/config, deployed computation is baseline State-EV,
not the expensive multi-view training bundle.

Validation uses student eval/no-grad and consumes no auxiliary generators or
EMA updates. Preserve real trimmer/RNG behavior around validation.
The 5M population overlaps selection validation; it is not independent test.

## 18. Phase-aware benchmark: avoid testing only warm-up

A naive benchmark starting at u=1 sees lambda=0 and never exercises this method.
This is a mandatory failure mode to prevent.

Use a separately named **diagnostic schedule policy** with explicit schedule
labels for warm-up, active plateau, fade and clean finish, without pretending
the model has already trained 100k updates or mutating the production recipe.

In one allocation per platform, sequentially measure fresh LW State-EV control
and fresh masked-training bundle with matched initial ROOT fingerprints and
recorded cold-start/order effects.

- Control: 20 warm-up + 4,200 CE attempts, windows 1000/1000/1000/1000/200.
- Candidate active phase: 20 warm-up + 4,200 measured complete multi-view
  attempts, lambda=0.1 at explicitly recorded plateau schedule labels.
  Populate actual optimizer/EMA state normally; do not forge success counters.
- Candidate clean-only phase: 20 warm-up + 200 measured attempts including real
  teacher maintenance, anchors and predictor-inactive policy.
- Correctness probes at lambda boundaries, fade and E_global=0/one-rank eligible.
- Fixed full-128 active training at global 512, auxiliary 256/microbatch32:
  10 warm-up + 40 measured attempts, teacher full/visible-only source rebuilt.
- 128k end-to-end FP32 student validation, batch512; full-128 batch512 stress.
- Exported FP32 student inference batch256: 10 warm-up+20 measured;
  batch-one latency: 10 warm-up+100 timed.
- Full restart save/load and student selected-weight writes.

No production weights from these phases. Real production begins at u=1,
unmodified schedule, fresh student/predictor/teacher.

Measure whole-step time including clean graph release, B0 clones/functional
calls, view rebuilding, teacher, auxiliary forwards/backwards, eligibility
reductions, scaler/Ranger and EMA. No kernel-only estimate called training time.

Require finite settled loss/model/teacher/optimizer/scaler, actual auxiliary
backbone gradients, correct BN/RNG isolation, at least 10% allocated GPU headroom
on every GPU and measured-window AMP skips below 5%. Report reserved memory and
per-process/available whole-job host peaks. Throughput counts original clean jets
per attempted update; label auxiliary views separately.

Use unwrapped rank-zero student eval, never DDP with idle peers. Shut down unused
ROOT workers before later stress/profiling. Benchmark has a five-hour internal
deadline, 5h30m Slurm request and early signal. Missing active-phase evidence is
inconclusive/failure, not a pass or reason to reduce population/microbatch/batch.

## 19. Diagnostics and interpretation

Bound profiles to at most two jets/32 particles and one train/eval iteration
per scope. Retain aggregates, not full-batch profiler histories. Include clean,
BN snapshots, mask/rebase, teacher, visible source, predictor, each backward,
synchronization, EMA and export scopes; positive calls and finite nonnegative
CPU/device times with at least one positive attribution. Inclusive scopes are
not simply additive.

Log clean CE, normalized aux loss, lambda, actual eligible/target counts and
mask modes, teacher momentum/successes, skips, per-token target/prediction norms,
cross-token/jet variance and cosine similarity. Non-affine per-token LayerNorm
does **not** prevent all tokens collapsing to the same normalized direction.

On bounded isolated probes, verify auxiliary gradient reaches shared backbone
and inspect its scale versus CE without retaining two full production graphs.
Diagnostics must restore parameters/buffers/gradients/modes/RNG and not take an
extra optimizer step.

A zero-visible-KV prediction intervention probes sensitivity but is not a trained
location-only baseline. A separately trained location-only/no-context predictor
is a later control. Low masked loss, high target variance or attractive latent
plots do not prove classification gains.

Do not add variance/diversity penalties, stronger masks, teacher resets or
adaptive loss scaling after observing diagnostics in the same run.

No per-event target cache, raw-jet dump, latent database or stored masking history.
Bound aggregate logs and tiny fixture evidence only.

## 20. Resource projection and storage

| Phase | Initial walltime | SPORC | Tigris |
| --- | --- | --- | --- |
| Prepare | 2 h | CPU, 8 CPUs / 64 GiB | CPU, 8 CPUs / 64 GiB |
| Paired benchmark | 5 h 30 min | 2 A100s, 8 CPUs / 128 GiB | 1 GH200, 12 CPUs / 128 GiB |
| Summary | 15 min | CPU, 2 CPUs / 4 GiB | CPU, 2 CPUs / 4 GiB |
| Optional dispatch | 30 min | CPU, 2 CPUs / 4 GiB | CPU, 2 CPUs / 4 GiB |
| Train per arm | Measured | 2 A100s, 8 CPUs / 128 GiB | 1 GH200, 12 CPUs / 128 GiB |
| Student evaluate per arm | Measured | 1 A100, 8 CPUs / 128 GiB | 1 GH200, 12 CPUs / 128 GiB |
| Report | 1 h | CPU, 2 CPUs / 8 GiB | CPU, 2 CPUs / 8 GiB |

These are planned requests; verify live partition/account/QoS, memory/hardware,
maintenance and Slurm test-only requests. No silent fallback to another platform.

Let sa be slowest measured active-phase seconds/attempt, sc slowest clean-only
candidate seconds/attempt, v full128k student-validation seconds, c full restart
write seconds and w student selected-weight write seconds:

```text
T_compute_candidate = 879,999*sa + 120,001*sc
T_validation = 50*10*v
T_writes = (1,000+100+2)*c + 51*w
train_hours = ceil(1.30*(T_compute + T_validation + T_writes + cold_start)/3600)
```

Use per-rank/window maxima, not fastest mean. Baseline projects 1M ordinary CE
attempts. Save measurements/formulas and an all-active conservative alternative.
Validation/export has no predictor/teacher passes. Evaluate hours include full-5M
inference, checkpoint loading, writes and CPU merge/metrics, not kernels alone.

Planned ceilings: 480 train hours, 24 evaluate hours, subject to stricter live
limits. If recommendation exceeds a limit, stop admission and request a new
execution decision; do not cap down, silently shrink aux population or split
three-day chains. One long allocation per training is intended; interruption-safe
resume remains required. Queue, replay and future stalls are outside a guarantee.

### Storage

For deployed student P=2,327,462, trainable Q=2,626,598 and teacher R=1,926,412:

```text
selected_weights = 51*(8*P + 1 MiB)
candidate_restarts = 3*(24*Q + 4*R + 16 MiB)
predictions = 5,000,000*48 + 64 MiB
misc = 1 GiB
candidate_budget = selected_weights + candidate_restarts + predictions + misc
```

Reference control uses R=0,Q=P. Add 2 GiB per paired-benchmark platform and one
shared 4-GiB emergency reserve. Both paired platforms require approximately
**17.78 GiB initial aggregate headroom before other outstanding campaign budgets**.
This is planning arithmetic, not current free space or a quota reservation.

Retain latest complete restart with atomic temporary allowance, at most 50
improving student weights plus final best, bounded logs and prediction chunks.
No teacher/predictor copy in every selected checkpoint, no accumulating full
restarts per update and no latent/ROOT cache.

Shared-filesystem registry/locks include both platforms, existing campaigns
and the other method budgets. Reserve before submitting; recheck before prepare,
dispatch and heavy output. Local squeue absence cannot release another cluster's
reservation. No automatic deletion of environments, datasets or old campaigns.

## 21. Data/source/environment identity and submission

CPU preparation authenticates official train/validation inventory/content receipts,
entry counts, original YAML/class order and pinned model/data/optimizer sources.
No independent test scan or fitted new preprocessing.

Freeze committed relevant source via safe git archive/transitive inventory;
reject symlinks/path traversal/live imports. No-user-site/no-bytecode flags
prevent source inventory drift after ordinary imports.

Fingerprint packages from stable interpreter roots, not dynamically added
setuptools vendor paths. Real torch/Weaver/source mismatches produce actionable
bounded differences, not bypass switches.

Science identity includes view transformations, masks/generator policy, all
loss/EMA schedule formulas, predictor, BN/gradient transactions and counts.
Runtime identity includes platform/world size/BN, microbatch/slot counts,
workers/threads, AMP/scaler, hardware and DDP buffer behavior.

Proposed roots are independent:

```text
checkpoints/part_ev_masked_predictive/
  sporc_<fresh-id>/
    source/ source_manifest.json launch.json campaign.json data_contents.json
    logs/ probes/ benchmark.json summary.json submitted_job_ids.tsv
    production/
      production.json production_job_ids.tsv
      runs/LW_STATE_EV_CONTROL/seed_101/
      runs/MASKED_PREDICTIVE_STATE_EV_LAYERWISE/seed_101/
      reports/
  tigris_<different-fresh-id>/
    ... separate equivalents ...
```

Never share mutable restarts, selected exports, generators, predictions or
Slurm dependencies between platform roots. Record cluster with each job ID.

Proposed launcher `--platform sporc|tigris` modes:

- `plan`: read-only paired tasks, provenance/resources/storage/automation.
- `benchmark`: freeze new campaign, prepare -> phase-aware benchmark -> summary.
- `review`: authenticate/recompute evidence/recommendations, no writes.
- `pipeline --reviewed`: same chain plus explicit bounded automatic-production authority.
- `dispatch --reviewed`: separately submit from complete passed platform evidence.
- `status`: read-only jobs/checks/progress/reports.

Summary runs after benchmark termination to explain failures; dispatch waits
for successful summary and independently rechecks leakage/active-loss/BN/DDP/
EMA/restart/export/source/data/runtime/resource/storage evidence. Slurm completion
alone never authorizes production. Publish executable commands only after tested
interfaces exist.

After admission submit control/candidate trainings independently, afterok student
evaluation each, paired report after both. No arbitrary training concurrency cap.
Optional control reuse is explicit/authenticated/reported.

Canonical scoped paths, committed source, immutable imports, live resource/space
checks, shared registry and per-output submission locks are required.
Write flushed durable intent before each sbatch; record cluster/job/request
immediately. Partial/ambiguous replies or duplicates require reconciliation,
not blind resubmission.

Sanitize inherited SBATCH/SLURM/rank/PYTHONPATH and unapproved NCCL options at
submission; preserve worker scheduler GPU visibility. Torchrun only for SPORC
two ranks, no elastic automatic restarts. Old queued jobs keep their snapshots
after pushes/pulls.

## 22. Production checkpoints and evaluation transactions

Save only at complete attempt/validation boundaries, atomically every 1,000
attempts, before/after validation and on coordinated early stop:

- Science/source/data/runtime/task identities and all parameter inventories.
- Student and predictor parameters/buffers; complete particle-only teacher state.
- Full Ranger moments/options/slow weights/counter for all registered parameters.
- Scaler and cumulative skips; attempted/successful/EMA-success counts.
- Every rank's native Python/NumPy/torch/CUDA RNG/trimmer state and all dedicated
  selection/mask/patch generators.
- Validation count, best clean-student accuracy/check/path/hash, continuous train/
  validation consumed-batch counters, timing and current phase identity.

Do not persist B0/B1 temporary snapshots, gradients, live graphs, input-specific
masks/targets or pair caches. Wrong teacher subset, missing predictor/EMA/generator
state, changed view/phase/world size/BN/source/data rejects resume.

Replay original loader batches without model work or auxiliary draws, restore
all native/auxiliary RNG and continue from the next complete attempt.
Replay consumes allocation time and must preserve previous restart on interruption.

Never commit a checkpoint between clean and auxiliary backwards or between
optimizer and teacher/BN/progress updates. Partial failure uses the previous
complete state; no duplicated clean BN update on a falsely advanced cursor.

Write a new improving **student-only** candidate before selection metadata
commit. Preserve prior best on interruption; audit final selected check against
all 50 records. Completion requires 1M attempts and 50 completed validations.

Graceful incomplete stops save recoverable state and exit nonzero, blocking
afterok successors. No fake completion, automatic extra allocation, three-day
chain or cross-cluster migration. Resume is separately reviewed.

Log JSON heartbeats every 100 attempts, CE/aux/lambda/momentum, eligibility/skips,
throughput, phase/validation boundaries, checkpoint/replay events and bounded
diagnostics. Use `check_XX.json`. Training reaching a boundary is not validation
finishing; no invented validation percentage.

Full-5M student evaluation uses bounded resumable chunks over the 50 validation
ROOT files, authenticated checkpoint/campaign/data/prediction hashes and sidecars.
Preserve class order, score definitions and rejection/count conventions.
No teacher comparison to choose a better export.

Reports show training/deployment counts separately, selected check, accuracy/AUC/
rejection counts, phase-weighted runtime, teacher/view overhead, peak device/host
memory and allocated GPU-hours. Label single seed, control reuse/platform
differences and selection overlap. No pooled platform pseudo-seeds or historical
validation differences presented as controlled gains.

## 23. Scientific follow-up and prior art

Do not automatically queue these additional runs:

1. **Compute/exposure control:** original LW State-EV trained with an explicitly
   FLOPs-budgeted longer schedule; equal clean updates here are not equal compute.
2. **Stop-gradient visible source:** train predictor but remove auxiliary backbone
   gradients, testing whether representation updates are responsible.
3. **Location-only predictor:** a genuinely trained context-free task control,
   not an untrained zero-KV intervention.
4. **Mask pattern / amount:** uniform-only versus patch-only, then one reviewed
   fraction change, rather than many validation-selected masks.
5. **Clean-only finish:** test scheduled consolidation versus persistent auxiliary;
   no post-hoc schedule edits inside a run.
6. **Target/loss choices:** one layer versus two, auxiliary coefficient, teacher
   momentum or a same-view no-mask task, with explicit compute differences.
7. **Additional matched seeds and locked independent test:** establish
   reproducibility before interpreting small gains.

No parameter-matched larger inference control is necessary to explain a larger
deployed classifier: deployed capacity did not change. Training-only predictor/
teacher capacity and extra compute still require the above controls.

[I-JEPA](https://arxiv.org/abs/2301.08243) establishes latent-target prediction
with a target encoder; [masked particle modelling](https://arxiv.org/abs/2409.12589)
is already established for particle sets. Neither supplies a gain forecast for
this full-labelled classifier.

A [controlled jet pretraining study](https://arxiv.org/abs/2606.14870) found strong
pure classifier pretraining when downstream labels/capacity were plentiful.
Its transfer setting differs from this joint auxiliary experiment, but it
argues against assuming SSL automatically helps. No novelty claim follows from
combining familiar components; literature review and controlled gains are needed.

Main failure modes: irreducibly ambiguous hidden targets, angular/position
shortcuts, collapsed EMA latents, historical-statistics mismatch, competition
with CE and large training overhead for no classification benefit.

## 24. Implementation sequence and definition of done

1. Lock feature-channel audit, masks/conditioning, schedule/EMA/buffer transaction,
   student/teacher/predictor inventories and hashable identities.
2. Implement view construction and independent rebase/mask/alignment oracles;
   pass hidden-value/padding/full-axis perturbation tests first.
3. Implement predictor and detached target taps, FP64 math/loss oracles,
   normalization/gradient and strict teacher-subset checks.
4. Implement clean/pretrimmed native forward plus functional B0/B1 scopes;
   prove unchanged clean path/export, cache/RNG/mode restoration and no
   autograd version-counter failures.
5. Implement one-step sequential accumulation, global eligible normalization,
   all empty/inactive DDP paths, predictor moment policy and post-Lookahead EMA.
6. Prove full-state/mask/EMA restart across phase boundaries and disposable
   actual-loader replay, coherent stops/skips/rank-local failures.
7. Build independent platform preparation, immutable source/environment/data
   admission, storage reservations and read-only planning.
8. Implement phase-aware paired actual-device benchmarks, including positive
   auxiliary loss, leakage/BN isolation, target stability, memory and export.
9. Fake-Slurm tests for wrong architecture/interpreter, dirty source, path spaces,
   storage, forged/incomplete evidence, duplicates, partial/ambiguous submissions
   and frozen-source safety after checkout changes.
10. **Benchmark-ready:** locally tested/documented interfaces and real queue
    commands; no target-GPU or production claim from CPU tests alone.
11. **Production-ready:** complete atomic state, transactional student selection,
    resumable student evaluation/reporting and reviewed automatic dispatch.
12. **Admitted:** actual platform evidence passes every gate and resource/storage
    limit before authorized full paired production.

Handoff states tests run/skipped, planned versus actual counts, untested GPU
assumptions, leakage/BN/DDP/EMA/restart evidence, measured resources, commands
and which modes submit jobs. This document itself changes only a plan and
authorizes no external submission, environment modification or deletion.
