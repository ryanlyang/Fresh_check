# Persistent global workspace on full Layerwise State-EV

Recorded and implemented 2026-11-23. **Model and independent benchmark-gated
SPORC/Tigris production interfaces are implemented and locally queue-tested;
actual GPU admission, runtime and classification results remain pending.**
See [implementation, verification and queue commands](../part_ev_workspace/README.md).
This supersedes the short workspace design note, not existing model code,
frozen campaigns, jobs, or the other three idea plans.

The intended deliverable at implementation time is one scientific model with
**independent SPORC tier3 and Tigris benchmark/production pipelines**. Either or
both may run; neither depends on the other cluster starting or finishing.
The design contracts below are implemented in `part_ev_workspace`; the linked
README documents the concrete interfaces. The first version uses fresh controls
on both platforms; optional historical-control reuse remains a future reviewed
extension rather than an implicit behavior.

## 1. Objective, hypothesis, and boundaries

Add an explicit jet-level workspace to the complete noncompact, standard-
relations, all-eight Layerwise State-EV backbone. The workspace repeatedly reads
particle states, integrates evidence across several latent channels, and returns
context to particles. It also supplies a final summary to the existing classifier.

The hypothesis is **useful iterative organization of already accessible global
information**, not access to information that native attention cannot see.
Native particle attention already communicates globally. A small intermediate
workspace might make coherent jet-wide interpretations easier to form, preserve
and use in later particle/relation computations.

A useful interpretation is “revise this particle representation in light of the
jet-wide evidence.” Workspace tokens are not truth classes, physical subjets,
independent predictions, four-vectors, or guaranteed interpretable concepts.
They must never become a bottleneck through which all particle information passes.

The first experiment changes the architecture only. It adds no expert routing,
hierarchical grouping, selected relations, masked training, pair-memory branch,
state-conditioned logit bias, auxiliary loss, distillation, teacher, extra data,
optimizer group, or warm start. Do not combine the four proposed methods yet.

### Reference backbone

Use `part_ev_state_compare.model.build_model('state_layerwise', ...)`:

- Particle embedding width 128; eight particle blocks and eight attention heads.
- One shared standard width-64 pair stem and eight independent final bias heads.
- Ordinary EV and width-32 State-EV in every particle block.
- Original particle FFNs, native residual/head scaling, attention dropout,
  BatchNorm, trimmer, two class-attention blocks, final LayerNorm and classifier.
- Existing state-message query chunk 64 and shared per-forward pair normalization.
- **2,327,462 trainable parameters** in the current reference.

Historical 86.3151% at validation check 49 is a single-seed training-time result,
not an independent test result or a guaranteed matched control. In particular,
a historical two-A100/SyncBatchNorm run is not automatically a control for a
single-GH200/native-BatchNorm run.

## 2. Locked first model

Scientific architecture identifier: `workspace_state_layerwise_v1`.
Proposed run ID: `WORKSPACE_STATE_EV_LAYERWISE`.
Proposed matched-control run ID: `LW_STATE_EV_CONTROL`.

| Item | First implementation |
| --- | --- |
| Workspace | Eight learned initial tokens, width 128 |
| Exchange stages | After particle blocks 2, 4, 6, 8; zero-based indices 1, 3, 5, 7 |
| Stage parameters | Independent across all four stages |
| Within each attention | Eight heads, head width 16, scale `1/sqrt(16)` |
| Stage order | Particle read, workspace self-attention, workspace FFN, particle feedback |
| Workspace FFN | `128 -> 512 -> 128`, exact GELU, biases in both linears |
| New normalization | Affine channelwise LayerNorm, epsilon `1e-5` |
| New attention projections | Separate biased Q, K, V and output linears, width 128 |
| New dropout | Zero, including attention dropout |
| Final classifier addition | One class-query/workspace-key-value attention read |
| Identity outputs | Four feedback output linears and final class-read output linear zero |
| Workspace lifetime | One jet in one forward, with gradients across stages |
| Workspace checkpointing | Four pure stages, non-reentrant; final class read uncheckpointed |
| Native path | All original particle/class blocks and their mathematics retained |
| Inference outputs | Original ten classification logits; no ensemble or token-class heads |

Eight tokens offer multiple summaries without a narrow particle bottleneck.
Full width and independent stage weights allow different early/late roles.
Exchanging every two blocks gives feedback time to affect later native State-EV
processing. The last exchange can still help the original class-attention path.
These are performance-oriented starting choices, not measured optima.

Do not silently change token count, width, insertion frequency, weight sharing,
loss, normalization, dropout, or zero-output policy to fit resources or make a
test pass. A changed design needs a new config hash and fresh evidence.

## 3. Exact tensors and attention primitive

Use `B` for local batch, `N <= 128` for actual native post-trim sequence length,
`K=8` for workspace length, `D=128`, and `A=8` for head count.

| Tensor | Shape | Meaning |
| --- | --- | --- |
| Native particle states | `[N, B, D]` | Native sequence-first layout |
| Workspace particle view `H` | `[B, N, D]` | A transpose view at the stage boundary |
| Initial workspace parameter | `[1, K, D]` | One shared trainable initialization |
| Current workspace `Z` | `[B, K, D]` | Independent activations per jet |
| Valid-particle mask | `[B, N]` Boolean | True means an actual retained particle |
| New Q/K/V heads | `[B, A, sequence, 16]` | Never mix batch and sequence axes |
| Particle-read scores | `[B, A, K, N]` | Softmax over valid particle keys |
| Workspace-mix scores | `[B, A, K, K]` | Softmax over workspace keys |
| Particle-feedback scores | `[B, A, N, K]` | Softmax over workspace keys |
| Normalized native class token | `[B, 1, D]` | After both original class blocks and norm |
| Final class-read scores | `[B, A, 1, K]` | One class query per jet |

Implement a small explicit attention primitive in a new module rather than
overriding native MHA. For query `Qin` and key/value input `X`:

```text
Q = split_heads(Wq(Qin))
K = split_heads(Wk(X))
V = split_heads(Wv(X))
scores = Q @ transpose(K) / 4
scores[invalid_keys] = -infinity
weights = softmax(scores, axis=keys)
message = merge_heads(weights @ V)
output = Wo(message)
```

There is one output projection and one bias contribution, not an MHA output
followed by an accidentally duplicated output linear. No native pair bias,
positional embedding, learned temperature or physical relation term enters this
new attention. Q/K/V are width 128 before head splitting; all four projections
include biases. Self-attention uses the same primitive with a shared normalized
input, but still has independently learned Q/K/V matrices.

There are no invalid workspace tokens in v1. Particle-read attention masks
invalid particle keys. Particle-feedback results for invalid particle queries
are explicitly zeroed before residual addition.

### Numerical arithmetic

Keep parameters FP32. In CUDA training, small linear projections follow the
existing FP16 autocast policy. Compute new LayerNorm arithmetic, attention
scores, softmax and weighted reductions in FP32 with autocast disabled for
those operations; the output linear can use the normal projection context.
Cast a feedback correction to the native particle tensor's dtype before adding
it. Keep workspace residual accumulation FP32 and cast only where prescribed.
FP32 validation/inference has no hidden native `use_amp=True` scope.

For CPU FP64 oracle/gradcheck tests, retain float64 arithmetic instead of
unconditionally downcasting. This test mode is not a production option.

Use stable masked softmax, not clipped logits or post-hoc `nan_to_num`. Check
valid pre-mask scores and projections for finiteness. Intentional masked
`-infinity` entries are not nonfinite model failures. Empty jets are rejected;
there is therefore always at least one valid key for a particle read.

The initial implementation uses explicit dense new attention, whose sequence
dimensions are small. A fused backend is a later engineering option only after
forward/gradient parity, dropout/RNG and runtime evidence; never silently switch
scientific behavior or tolerances based on which GPU is available.

## 4. Four read–mix–feedback stages

Start each forward with an expanded, non-mutated view of the initial tokens.
Do not perform in-place operations on this expanded parameter view.

For stage `s`, use the particle states from the native blocks up to its insertion
in the fixed schedule. Stage 0 follows native block 2, stage 1 follows block 4,
and so on; there are two native blocks between exchanges.
Each stage has six distinct affine LayerNorms:

1. Particle-read query norm for Z.
2. Particle-read key/value norm for H.
3. Workspace self-attention norm.
4. Workspace FFN input norm.
5. Particle-feedback query norm for H.
6. Particle-feedback key/value norm for Z.

No norms or attention/FFN parameters are shared between stages.

```text
Hsafe = where(valid_particles, H, 0)

Z1 = Z  + Read_s(LN_read_query_s(Z), LN_read_kv_s(Hsafe), valid_particles)
Z2 = Z1 + Mix_s(LN_mix_s(Z1))
Z3 = Z2 + W2_s(GELU(W1_s(LN_ffn_s(Z2))))

delta_H = Write_s(LN_write_query_s(Hsafe), LN_write_kv_s(Z3))
Hnext = H + where(valid_particles, cast_like(delta_H, H), 0)

return Hnext, Z3, finite_health
```

This is a residual pre-norm transformer-style update of the workspace, followed
by particle-specific feedback. Feedback queries depend on each particle: do
not replace them with the same mean-workspace vector broadcast to every particle.

Carry Z3 into the next stage without detaching. The next two native particle
blocks process Hnext, including their unchanged State-EV mechanisms. Never
reinitialize the workspace from its learned seed between stages.

New invalid-particle corrections are zero, but do not alter the native model's
existing padded-query behavior. Sanitize the workspace's view before its norms
and projections so padded junk cannot affect workspace summaries.

No additional LayerNorm, FFN, dropout or gate is applied after the feedback
residual. Adding one would change both initialization identity and the first arm.

### What “persistent” means

Only the learned initial tokens are persistent model parameters. Evolving Z is
local, per-forward data, not a registered buffer, module attribute, cached
attention state, recurrent state across minibatches, or checkpoint payload.
Forward two must never read any activation created by forward one.

## 5. Final classifier access

Let `c` be the native normalized class representation after both original
class-attention blocks. Keep those blocks, final norm and final linear classifier
unchanged.

Add two new affine norms and one class-read attention primitive:

```text
class_delta = ClassRead(LN_class_query(c), LN_class_kv(Z_final))
c_fused = c + cast_like(class_delta, c)
logits = original_classifier(c_fused)
```

The class-read output weight and bias start at zero. Do not concatenate tokens
to the classifier input, change its width, replace its native query token, add
another classifier MLP, or normalize again after the residual.

This direct read makes workspace evidence accessible without forcing every
benefit through particle feedback. It is not the direct-pair readout method:
the source is the eight workspace tokens, not `r_ij`. Its contribution must be
separated from feedback in follow-up ablations.

## 6. Initialization and expected learning sequence

Build the full reference model on CPU using its ordinary seed-101 initialization
first. Only then construct `mod.global_workspace` inside an isolated RNG scope.
Use stable, versioned mechanism seeds derived from the experiment seed and
stage/component name; do not use Python's randomized `hash()`.

Initialize:

- Initial tokens from normal mean zero, standard deviation 0.02.
- All new LayerNorm weights one and biases zero.
- Non-identity new projection/FFN weights with Xavier uniform, gain one.
- Their biases zero.
- All four particle-feedback output weights/biases exactly zero.
- Final class-read output weight/bias exactly zero.
- Other workspace/read/mix/FFN/class-read QKV weights nonzero and independent.
- No additional zero scalar gate, tiny global multiplier, frozen backbone,
  copied trained weights or special learning rate.

This preserves the original function at initialization, up to the declared
computed-tensor tolerances. New stage computation consumes no training randomness,
so the native trimmer/dropout sequence must remain unchanged.

On the first backward, the five zero-output linears should receive a learning
signal on an appropriate nondegenerate fixture. Upstream workspace parameters
can correctly have zero classification gradients then. Once outputs move, all
new parameter families must develop finite learning signals. Do not require
every upstream tensor to move before that can mathematically happen.

The actual-Ranger learning probe targets eight successful updates within twenty
attempts, crossing rectification and a Lookahead boundary. Count AMP skips
separately. Verify finite gradients, output learning first, later upstream
learning and nonzero changes in each trainable family on a mixed multi-particle
fixture. A one-particle fixture can have legitimately zero Q/K selection
gradients; it must still produce graph-connected finite gradients.

An inability to meet this probe is evidence to investigate, not authority to
increase LR, change initialization, remove the shared path or loosen the test.

## 7. Parameter contract and scientific cost

The following is an analytical count for the locked module definitions, not a
claim that the future full model has already been built or benchmarked:

| Addition | Parameters |
| --- | ---: |
| One four-projection attention | `4 * (128*128 + 128) = 66,048` |
| One workspace FFN | `128*512+512 + 512*128+128 = 131,712` |
| Six affine norms per stage | `6 * 256 = 1,536` |
| One full stage | `3*66,048 + 131,712 + 1,536 = 331,392` |
| Four stages | 1,325,568 |
| Eight initial tokens | 1,024 |
| Final class read plus two norms | 66,560 |
| **Total addition** | **1,393,152** |
| **Planned full model** | **3,720,614** |

This is about **59.9% more parameters** than LW State-EV. Small token count does
not mean small parameter count. The implementation must enumerate actual names,
shapes and counts, verify both totals, and save that inventory in admission
metadata. Any discrepancy requires an explicit specification correction before
benchmarks, not removing an assertion.

Cross-attention projection work still scales with particle count and D squared.
The new scores cost O(B*A*N*K) and workspace self-attention O(B*A*K*K); the
native particle attention and state computation remain quadratic in N.

At B=512, N=128, one FP32 read or feedback score tensor has 4,194,304 elements,
16 MiB. That excludes probabilities, projections, gradients, checkpoint
recomputation, four stages and the entire original model. It is not a peak-memory
or speed forecast. Count train/inference FLOPs separately from walltime and
specify the multiply-add convention.

## 8. Concrete code integration

Proposed scientific package:

```text
part_ev_workspace/
  __init__.py
  config.py           # frozen science and parameter inventory contract
  attention.py        # explicit safe attention + independent oracle interface
  model.py            # WorkspaceStage, ClassRead, GlobalWorkspace, assembly
  diagnostics.py      # bounded, non-mutating workspace diagnostics
  protocol.py         # identities, source/environment/data admission, storage
  benchmark.py        # world=1/2 checks, timing, validation, restart evidence
  restart.py          # complete-state probe and numerical-policy validation
  engine.py           # explicit platform-aware fresh/train/evaluate/report
  submission.py       # read-only planning, snapshots, Slurm transactions
  README.md           # actual commands only once implemented and tested

scripts/run_part_ev_workspace.py
scripts/submit_part_ev_workspace.py
sbatch/run_part_ev_workspace.sh
sbatch/submit_part_ev_workspace.sh
tests/test_part_ev_workspace_*.py
```

Before implementation, check these names are still unused. Reuse authenticated
helpers rather than cloning entire old campaigns, but do not temporarily
monkeypatch another protocol module's global task or package map.

### Base and forward integration

Build the ordinary state_layerwise outer wrapper and preserve its existing
state_dict keys. Register additions only under `mod.global_workspace.*`,
including `initial_tokens`, `stages.0..3` and `class_read`.

Use a repository-local subclass with an explicit forward loop based on the
pinned ParticleTransformer, following the local-instance assembly precedent in
`part_ev_pair_readout/model.py`. Bind only this newly constructed instance to
the local forward implementation; never edit installed Weaver or patch its
global class. Do not transplant the pair-readout-specific final-attention code.

The new native-compatible forward must:

1. Run the original native trimmer exactly once; obtain its actual mask.
2. Run the original particle embedding and pair encoder exactly once.
3. Expand initial workspace tokens locally.
4. Call every original particle Block.forward exactly once, inserting the
   four pure workspace stages at the specified boundaries.
5. Run both original class Block.forward calls and final native norm once.
6. Add the final workspace class read, then call the unchanged classifier.
7. Clear native pair bindings and all transient caches in `finally` on success
   and failure; outer state-pair-cache hooks remain effective.

This preserves native `c_attn` layout/head scaling, residual scaling, FFNs and
dropout ordering by calling their forwards rather than approximating them.
Layerwise bias/EV wrappers must see the same particle sequence/mask shapes.

Keep the ordinary four-input outer API `(points, features, vectors, mask)`.
Do not append workspace tokens to those tensors, native class-attention keys,
pair-stem inputs, four-vectors or masks. Do not retrim for the workspace.

### Useful existing implementation anchors

- `part_ev_state_compare/model.py`: exact backbone assembly and count.
- `part_ev_state_speed/model.py`: explicit chunk arguments and cache cleanup.
- `part_layerwise_screen/model.py`: layerwise pair/mask binding.
- `part_ev_pair_readout/model.py`: isolated native-compatible forward assembly.
- `part_ev_screen/precision.py`: both native and outer AMP control.
- `part_ev_fullscale/optimizer.py` and `checkpoint.py`: complete Ranger state.
- `part_ev_state_production/engine.py`: two-GPU training/replay patterns.
- `part_ev_rts_tigris/production.py`: one-GH200 execution patterns.

The last two engines currently have their own hard-coded contracts. They are
references for a new explicit platform adapter, not interchangeable functions
that can be called after changing their module globals. Existing immutable
campaigns must continue to use their own source.

## 9. Autograd, masks, lifetime, and failure handling

Pass H, Z and the post-trim valid mask explicitly into each stage. Return Hnext,
Znext and tensor-valued health explicitly. Never use mutable hooks to carry
workspace activation data across blocks or into backward.

Checkpoint each whole workspace stage using non-reentrant checkpointing and
`preserve_rng_state=False` only because the new stage contains no random
operations. Native dropout, BatchNorm, pair construction and trimming remain
outside these checkpoints. The final class read is small and uncheckpointed.

Recomputation must use only passed inputs and immutable module configuration,
not `self.last_workspace`, hooks, cached native pair contexts or mutable lists.
A backward after forward cleanup must work, and checkpointed/uncheckpointed
results must agree. CPU deterministic checks also compare native RNG states.

Mask arbitrary valid layouts, not merely a valid prefix. Test N=1, partially
padded jets and unequal sequence lengths per DDP rank. Reject an all-invalid jet
before distributed native training using a coordinated input-health check.
There is no “no pairs, disable workspace” branch: one-particle jets remain valid.

Workspace processing is per jet. The full native network still has expected
BatchNorm/SyncBatchNorm batch dependence in training. Test workspace per-jet
isolation directly, and whole-model batch composition isolation in eval; do not
claim training-time independence that the base does not possess.

Native pair BatchNorm may also make raw padded-input perturbations affect the
base. Establish workspace masking invariance on fixed post-trim hidden states;
do not demand a stronger padded-input invariance from the inherited base than
it actually has. Preserve the original preprocessing/padding contract.

Accumulate finite-health flags without per-stage GPU `.item()` synchronization.
On bad new activations, return nonfinite logits through the existing coordinated
finite-loss rejection path; do not let one rank raise between SyncBN collectives
while peers enter the next block. No distributed collectives inside model forward:
rank-zero-only FP32 evaluation must remain possible. Input shape/identity/I/O
failures still need coherent worker-level propagation and bounded process-group
timeouts. Never replace failures by finite zero predictions.

## 10. Full data and optimization contract

Use the original full `ranger_full_v1` recipe rather than a shortened pilot:

| Field | Value |
| --- | --- |
| Training pool | Official 100M JetClass training jets |
| Validation pool | Official 5M validation jets |
| Independent test | Not opened by prepare, benchmark or routine production |
| Features / particle cap | Original 17 features; native cap 128 |
| Initial scientific seed | 101, fresh weights |
| Global batch | 512 on both clusters |
| Training budget | 1,000,000 attempted updates; nominal 512M exposures |
| Check interval | Every 20,000 attempts |
| Checks / population per check | 50 / 1,280,000 events |
| Optimizer | Pinned RAdam plus complete-state Lookahead wrapper |
| RAdam | LR 1e-3; betas (0.95, 0.999); epsilon 1e-5; weight decay zero |
| Lookahead | k=6; alpha=0.5 |
| LR schedule | Original zero-based epoch milestones 35–49; no new warm-up |
| Loss | Ordinary ten-class cross entropy; no new smoothing/auxiliary term |
| Precision | Native/outer FP16 training with FP32 parameters and GradScaler |
| Validation | FP32 logits and unchanged accuracy-based selection; first tie wins |
| Regularization | Original dropout; no added dropout or gradient clipping |

“Epoch” in logs remains a 20,000-update reporting interval, not a pass over
100M unique jets. Keep the training iterator continuous across validation
checks, preserve the separate continuous validation stream, and never restart
the ROOT scan at each check.

Call the pinned `part_ev_fullscale.protocol.learning_rate(step, recipe)` rather
than reimplementing milestone indexing. For zero-based update `step`, its epoch
is `step // 20000`; LR is `1e-3 * (0.01 ** (1/15)) ** decays`, where decays counts
milestones in 35 through 49 already reached. The first decay applies to update
attempt 700,001, not to the start of one-based check 35. LR/cursors advance with
attempts even when GradScaler skips an optimizer update; Lookahead advances only
when the wrapped optimizer actually steps. Preserve the existing scaler defaults.

Extend original optimizer registration to all workspace parameters in stable
order. No special workspace LR, freezing, token weight decay exception or
different parameter group is needed with the locked zero-weight-decay Ranger.

Final selected-checkpoint evaluation covers all 5M validation jets and reports
accuracy, macro OVO/OVR AUC and the existing per-class signal/QCD rejection
operating points with counts. Full validation overlaps selection validation;
do not label it independent test performance. Zero background acceptances need
counts/appropriate limits rather than an unqualified infinite improvement.

## 11. Independent execution on both clusters

Each platform has its own environment, Slurm ledger, benchmark, source/runtime
identity, paired production runs and storage reservation. They share science,
not a mutable training directory or admission from another GPU.

| Setting | SPORC | Tigris |
| --- | --- | --- |
| Submission host architecture | Linux x86_64 | Linux aarch64 |
| Partition | tier3 | tigris |
| Account default | reu-aisocial | reu-aisocial |
| Training hardware | Two matching same-node A100s | One GH200, at least 80 GiB device memory |
| Global / local batch | 512 / 256 per rank | 512 / 512 |
| Training model | DDP, gradient_as_bucket_view, no unused-parameter workaround | Ordinary one-GPU model |
| Native normalization | SyncBatchNorm conversion | Original native BatchNorm |
| Training loader workers | One per rank | Two |
| Torch threads per rank | Two | Four |
| Training CPUs / host RAM | 8 / 128 GiB | 12 / 128 GiB |
| Validation location / batch | Unwrapped rank-zero model / 512 | Unwrapped model / 512 |
| Final evaluation GPU | One A100, saved SyncBN modules in eval | One GH200 |
| Default interpreter | `/home/ryreu/miniconda3/envs/part_ev_screen_sporc/bin/python` | `/home/ryreu/miniforge3-aarch64/envs/part_ev_fullscale/bin/python` |

Verify actual hardware memory, driver, Python, torch/CUDA, source hashes,
TF32-disabled settings, cuDNN benchmark=False and approved NCCL environment.
Do not assume matching torch binaries across x86 and ARM; validate each frozen
environment and the scientific parity required on that platform.

Do not install packages, edit existing environments, depend on the removed
`part_selected_tigris` environment, or require a SPORC benchmark/preparation job
before Tigris preparation can run. Both prepare independently from the same
authenticated code/data contract. Historical donor metadata, if used, is read
only for inventory/preprocessing provenance, not to require a cross-architecture
package-list match or to load trained weights.

Moving a running checkpoint across GPU counts/BN implementations is not an
ordinary resume. No automatic cross-cluster migration is part of this design.

### Matched controls and replication

Default scientific production consists of **workspace plus a fresh matched
LW State-EV control on each platform**, each seed 101. That is four full trainings
if both cluster campaigns are authorized. Make this count visible in the plan.

A control may be reused only with explicit authenticated matching of model,
recipe, data, seed, GPU/BN/worker/precision configuration and selection/evaluation
procedure. Never silently substitute the 86.31% historical result or a benchmark's
disposable baseline weights for a production control.

SPORC and Tigris seed-101 results are not independent random seeds or pure
hardware comparisons: normalization, batch partitioning, stream ordering and
numerics differ. Compare within platform first. More seeds come after screening.

## 12. Data, source, environment and identity admission

Preparation is CPU-only. Validate the full official train/validation inventory,
file entry counts, preprocessing YAML and pinned Weaver model/data/optimizer
sources. No test inventory, new dataset, fitted preprocessing or training cache.
Coordinate reusable read-only data-hash receipts if available; size/mtime
shortcuts must not conceal a content identity change.

Freeze committed source before queueing workers. Use a complete, safe git archive
or a tested transitive source inventory: no symlinks, path traversal or live
checkout imports. Keep bytecode/user-site writes out of the frozen source.
Distinguish data/reference hashes, scientific configuration hash, source
manifest hash and platform runtime/campaign hash.

Fingerprint installed distributions from stable interpreter install roots,
not all dynamically added import paths; lazy setuptools vendor imports must
not look like new environment installations. Environment mismatches produce
a bounded, actionable diff. Never bypass a real torch/Weaver/source change.

Model config includes every choice in sections 2–6 plus actual parameter inventory.
Runtime config includes platform, world size, BN, workers, precision, GPU identity
and launch environment. Runtime changes cannot be disguised as the same training
state merely because the architecture matches.

Suggested independent paths:

```text
checkpoints/part_ev_workspace/
  sporc_<fresh-id>/
    source/  source_manifest.json  launch.json  campaign.json  data_contents.json
    logs/  probes/  benchmark.json  summary.json  submitted_job_ids.tsv
    production/
      production.json  production_job_ids.tsv
      runs/LW_STATE_EV_CONTROL/seed_101/
      runs/WORKSPACE_STATE_EV_LAYERWISE/seed_101/
      reports/
  tigris_<different-fresh-id>/
    ... separate equivalents ...
```

Never share `last.pt`, selected checkpoints, prediction chunks, mutable manifests,
module imports or job dependencies between these roots. Both may run without
cross-cluster cancellation. Slurm job IDs are scoped to their recorded cluster.

## 13. CPU and model correctness tests

Required suites should be small, deterministic and independent where possible:

1. **Attention oracle:** FP64 explicit per-jet/per-head loops versus the vectorized
   primitive; nontrivial masks, extreme finite logits, one valid key, output
   bias, Q/K/V/output gradients and softmax shift invariance.
2. **Full stage oracle:** separate direct implementation of read/mix/FFN/write;
   active nonzero feedback, input-H/input-Z gradients and every parameter family.
   An oracle must not simply call the production function with a different flag.
3. **Parameter inventory:** exact reference/addition/total counts and expected
   names; stage weights must have independent storage, not accidental aliases.
4. **Initialization:** all shared base weights/buffers, input tensors, actual
   trimmed tensors/masks and post-forward RNG/trimmer states exact. Computed
   native attention/block outputs and logits use declared FP32 tolerances.
   Exercise nonzero original State-EV weights, not just an initially zero base.
5. **Mechanism:** active feedback changes the input to later native blocks;
   removing final read still leaves a working feedback path; detaching/resetting
   Z changes the intended graph. First and delayed gradients follow section 6.
6. **Isolation/masking:** constituent permutation in eval, arbitrary padding
   patterns, masked-H invariance, N=1, empty-jet rejection, two successive forwards,
   batches of unrelated jets, and no cross-jet reduction in the new branch.
7. **Native calls:** one trim, one embedding/pair call, eight original particle
   attentions, two original class attentions, four workspace reads/mixes/writes,
   four workspace FFNs and one final workspace read. Never repeat native MHA.
8. **Autograd/lifetime:** checkpointed versus uncheckpointed outputs/gradients,
   backward after forward cleanup, exception cleanup and train/eval transitions.
   No workspace activation is a module buffer/state_dict entry or retained after
   no-grad evaluation.
9. **Health:** valid NaN/Inf in projections, pre-softmax scores or residuals must
   fail admission, including when zero output weights could hide the problem.
10. **State roundtrip:** CPU Ranger with populated moments/slow weights, scaler,
    native buffers, RNG/trimmer and progress; missing workspace keys or wrong
    configuration must fail strict loading.

Proposed numerical parity limits, frozen before GPU admission:

| Check | rtol | atol |
| --- | ---: | ---: |
| Small FP64 oracle outputs/gradients | 1e-9 | 1e-10 |
| FP32 initial identity / active outputs | 1e-5 | 1e-6 |
| FP32 active parameter/input gradients | 1e-4 | 1e-5 |
| FP32 DDP versus global-batch forward | 5e-4 | 2e-5 |
| FP32 DDP versus global-batch gradients | 2e-3 | 2e-5 |
| FP16 checkpoint/backend forward comparison | 2e-3 | 2e-3 |
| FP16 gradient comparison | 1e-2 | 1e-3 |

Integer metadata, masks, shared starting state, call counts and RNG are exact;
floating tolerances are not a license to accept changed inputs or missing paths.
Record maximum errors and offending paths. A failure requires investigation,
not a broad `strict=False`, tolerance inflation or disabling the state branch.

Use gradient-enabled mixed-length, nonzero-state fixtures. Zero-init identity
tests alone do not establish that the active mechanism is implemented correctly.
CPU success does not establish CUDA FP16, SyncBN, GPU fit or runtime.

## 14. Actual-GPU correctness, DDP and restart admission

Before sustained measurements, require:

- CUDA FP32 active full-model output/input/parameter-gradient parity against an
  uncheckpointed implementation, and FP16 checkpointed/uncheckpointed parity.
- Eight-successful-update native-Ranger activation test with AMP skips recorded.
- Two actual A100 ranks: SyncBN/DDP global-batch comparison with dropout disabled
  only for that plumbing test; native training dropout restored for learning
  and restart tests.
- Unequal rank sequence lengths, one rank with only one-particle jets, and
  finite graph-connected new parameters with `find_unused_parameters=False`.
- Single-GH200/native-BN equivalent tests on Tigris; never describe a CPU or
  SyncBN-disabled substitute as target-device evidence.
- Coordinated nonfinite, stop-signal, malformed-checkpoint and rank-local I/O
  failures that stop all workers without a permanent collective hang.

For the DDP/global-batch numerical oracle, hold already-trimmed inputs and masks
fixed at the same sequence length on both ranks, disable native trimming and
dropout only for this oracle, and compare against an isolated ordinary-BN
single-device reference with identical starting weights/buffers and global
batch. Do not call a training-mode SyncBN reference on rank zero while the other
rank waits: that introduces unmatched collectives. Unequal-rank-length tests are
separate graph/health checks, not a promise of equality to a differently padded
global-batch BN computation. Restore the actual native trimmer/dropout policy
before learning, restart, stream and throughput measurements.

### Restart test: exact state, bounded subsequent numerical variation

Use a model with populated optimizer state and nonzero learned workspace outputs.
The initial save and every immediate restoration must match exactly: model
parameters/buffers, RAdam moments/counters/cache metadata, Lookahead slow weights
and counter, AMP scaler, rank RNG/trimmer state, update/validation and stream
cursor bookkeeping. Loading successfully is not enough; compare snapshots.

Run a seven-update synthetic rolling-batch reference, uninterrupted repeat,
disk-start repeat, and a save/reload after update three. Cross a diagnostic
validation boundary and a Lookahead boundary. This fixture uses its own schema,
not a fake production completion marker or mutated production recipe.

For subsequent floating model tensors and Ranger moments/slow weights only,
use a separately versioned engineering tolerance with both elementwise and
absolute limits:

```text
state_abs_error <= min(1e-4, 1e-5 + 1e-4 * abs(reference))
loss_abs_error  <= min(2e-4, 1e-4 + 1e-5 * abs(reference))
```

Both uninterrupted and resumed comparisons must meet these fixed bounds.
Noisy repeats do not enlarge the bounds. RNG, LR, scalar metadata, counters,
skip/scale schedules and cursors stay exact. All values must remain finite;
the settled restart fixture must have no AMP skips. Record every rank/trial,
per-update loss/schedule, maximum state errors and worst tensor paths.
These are short-probe acceptance thresholds, not long-run accuracy guarantees.
Do not import another model's passed restart flag or turn a diagnostic into
production admission without this model's evidence.

Also test the **actual ROOT-stream replay plumbing** in a disposable bounded
probe: authenticate successive train/validation batch fingerprints, checkpoint,
recreate loaders, replay consumed batches without model/optimizer work, restore
rank RNG, and compare the continuation. A small fixture may force a validation
boundary using a separately declared probe policy; it must not pretend the full
20,000-update/1.28M-event production budget ran. Preserve the production recipe.

Admission records synthetic-state and real-stream evidence separately. Neither
proves bitwise-identical multi-day GPU trajectories, but missing/reordered input,
changed optimizer state or a replay cursor error must block production.

## 15. Benchmark recipe and performance evidence

One independent GPU benchmark per platform measures **both fresh LW State-EV
and fresh workspace**, sequentially on the same allocation. Baseline first,
workspace second; record that order and filesystem cold-start effects. Do not
reuse SPORC throughput as Tigris evidence.

For each arm, use the same authenticated inventory, seed, initial rank-batch
fingerprints and platform data-loader configuration:

- 20 real-data warm-up attempts and 4,200 measured attempts.
- Repeated-fetch timing windows: 1,000, 1,000, 1,000, 1,000, 200 attempts.
- Fixed-length 128-particle training: 10 warm-up and 40 measured attempts,
  global batch 512 with the platform's actual local batch.
- 128,000 events of end-to-end FP32 streaming validation, batch 512.
- Explicit full-128 FP32 validation at batch 512 on the rank-zero/evaluation GPU.
- FP32 inference on one GPU, batch 256, 10 warm-up and 20 timed batches;
  batch-one latency, 10 warm-up and 100 timed forwards.
- Full checkpoint save/load and selected-weight write timings.
- Model/optimizer/scaler finiteness, AMP skips/scales, replica parameter digests,
  hardware identity and peak allocated/reserved device memory.

On SPORC, training uses both ranks; rank-zero validation/inference uses the
unwrapped eval model, never DDP forward with idle peers. All ranks participate
in bounded outer coordination. Shut down idle train workers before later
synthetic stress/profiling phases, without changing the measured training stream.

Allocation admission requires at least 10% allocated-device-memory headroom on
every actual GPU, no unmatched hardware, and finite settled training. Record
reserved peaks too. Bound AMP skip fraction below 5% in measured windows;
do not hide skips by computing throughput only over successful attempts.

The paired benchmark has a five-hour internal deadline, a 5h30m Slurm request
and an early signal. Incomplete evidence is failure/inconclusive, never a pass
because a summary job exited zero. No silent reduction of batch, windows,
validation population or correctness coverage to meet the deadline.

### Bounded profiling and diagnostics

Measure stage read/mix/FFN/write and final-read timing separately from sustained
throughput on at most two jets/32 particles, one iteration per train/eval
profile. Include checkpoint recomputation scopes explicitly. Stage times are
inclusive and must not simply be summed. Require positive call counts, finite
nonnegative CPU/device time and at least one positive timing per required scope;
device-only or CPU-only attribution is valid.

Keep only aggregate profile records, not an unbounded profiler event history.
Record current/peak per-process RSS and, where available, job/cgroup memory
including workers. Never label one rank's RSS as whole-job host memory.
Small-profile timings are not full-batch speed/memory evidence.

On an isolated no-grad eval sample of at most eight jets/32 particles, log:

- Workspace RMS and norm growth across stages.
- Token cosine similarities and per-token variation across jets.
- Read/feedback attention entropy and token usage.
- Feedback/native-message and final-read/class-feature norm ratios.
- Logit changes when feedback alone or final read alone is disabled.

Diagnostics preserve weights, buffers, gradients, modes, RNG/trimmer and caches.
Redundant tokens are reported, not “fixed” with an unplanned diversity loss.
Save no training-event workspace activations. A benchmark makes no accuracy claim.

## 16. Measured walltime and resource planning

Proposed initial job resources:

| Phase | Walltime request | SPORC | Tigris |
| --- | --- | --- | --- |
| Prepare | 2 h | CPU, 8 CPUs / 64 GiB | CPU, 8 CPUs / 64 GiB |
| Paired benchmark | 5 h 30 min | 2 A100s, 8 CPUs / 128 GiB | 1 GH200, 12 CPUs / 128 GiB |
| Summary | 15 min | CPU, 2 CPUs / 4 GiB | CPU, 2 CPUs / 4 GiB |
| Optional dispatch | 30 min | CPU, 2 CPUs / 4 GiB | CPU, 2 CPUs / 4 GiB |
| Train, per arm | Measured recommendation | 2 A100s, 8 CPUs / 128 GiB | 1 GH200, 12 CPUs / 128 GiB |
| Evaluate, per arm | Measured recommendation | 1 A100, 8 CPUs / 128 GiB | 1 GH200, 12 CPUs / 128 GiB |
| Report | 1 h | CPU, 2 CPUs / 8 GiB | CPU, 2 CPUs / 8 GiB |

These are planned requests, not statements about current available resources.
Verify partition, account, QoS limits, GPU type, host RAM and `sbatch --test-only`
on the actual submission host. Do not change hardware to satisfy scheduling
without a matching benchmark and revised runtime identity.

For each arm separately, let s be the slowest measured real-data seconds/attempt
over all recorded rank/windows, v the complete 128k validation seconds, c the
full-state checkpoint seconds and w the selected-weight write seconds. Project:

```text
T_train_compute = 1,000,000 * s
T_validation    = 50 * (1,280,000 / 128,000) * v
T_writes        = (1,000 + 100 + 2) * c + 51 * w
train_hours     = ceil(1.30 * (T_train_compute + T_validation
                              + T_writes + measured_cold_start) / 3600)
```

The write allowance includes periodic plus pre/post-validation restart writes,
up to 50 improving candidates, final best copy and a small commit allowance.
Save the exact formula and measurements. Evaluation planning must include full
5M inference, checkpoint loading, prediction writes and CPU metric/merge cost;
do not scale GPU-only kernel time and call it end-to-end evaluation.

Use the larger of the measured recommendation and any explicit reviewed minimum.
Never cap a recommendation down to the partition limit. Planned upper ceilings
are 480 training hours and 24 evaluation hours, subject to stricter live limits.
If a recommendation exceeds a ceiling, **do not submit production**; request a
new resource/execution decision rather than silently split or shorten training.

Use one long allocation per full training, not automatic three-day segment chains.
Graceful stop/restart support is still required for interruptions.
The 30% margin is illustrative engineering allowance, not a bound. Queue delay,
future filesystem contention and restart replay can add time. Report allocated
GPU-hours as walltime times GPU count, separately from jet throughput/FLOPs.

## 17. Storage and artifact retention

**Operational amendment requested after initial implementation:** aggregate
reservations are now advisory for newly frozen workspace campaigns. Admission
enforces only the current campaign's remaining budget plus the shared 4-GiB
reserve against actual free bytes. Warnings retain the aggregate estimate,
including failed/stale reservations, but those estimates cannot block a
locally provisioned campaign. The budget/retention arithmetic below remains
conservative planning; it is not an enforced cross-campaign quota. Existing
frozen source, jobs and reservations are not rewritten. See the implementation
README for the current policy and monitoring responsibility.

All directories live on a shared home filesystem. A SPORC and a Tigris campaign
must account for one another even though their schedulers are separate.

For a model with P parameters, use the following conservative prospective
production budget in bytes, then subtract already present committed output:

```text
weights     = 51 * (8*P + 1 MiB)
restarts    = 3 * (24*P + 16 MiB)
predictions = 5,000,000 * 48 + 64 MiB
misc        = 1 GiB
arm_budget  = weights + restarts + predictions + misc
```

Reserve a 2-GiB paired-benchmark slot per platform and one shared 4-GiB emergency
reserve. With the planned baseline/workspace counts, two new paired platform
campaigns require about **18.94 GiB aggregate initial headroom**, before
outstanding budgets from every other registered campaign. That number is a
planning calculation, not current free space or a quota reservation.

Production retains the latest complete restart, safe atomic-write temporaries,
the improving selected-weight candidates (at most 50) and final best weights.
Only parameter/optimizer states, metadata, bounded logs and evaluation predictions
are persistent. No ROOT copies, pair/workspace feature caches or downloaded models.

Serialize submission/duplicate protection with a shared filesystem lock across
both platforms. Keep registry and reservation records for advisory estimates;
recheck actual local-campaign headroom before prepare, dispatch and heavy-output
phases. Neither an excessive aggregate estimate nor inability to read unrelated
reservation metadata is a new-workspace submission failure.

A missing job from the local `squeue` is not evidence that the other cluster's
reservation can be released. Reconcile using its recorded cluster and ledger.
Do not automatically delete old outputs, remove environments, release active
budgets, or count hard-linked caches as guaranteed reclaimable space.

## 18. Submission, production admission, and operational behavior

The proposed launcher accepts an explicit `--platform sporc|tigris` and selects
the matching architecture/interpreter/resources. Production uses only that
campaign's frozen science, never current live source or another platform's
benchmark. Record both cluster and job IDs.

Proposed modes:

- `plan`: validate arguments/provenance and print both scientific arms,
  resources, projected storage and requested automation; no files/jobs changed.
- `benchmark`: freeze a new campaign; submit prepare -> benchmark -> summary.
  Benchmark and summary never authorize training.
- `review`: read/authenticate/recompute evidence and recommendations without writes.
- `pipeline --reviewed`: same benchmark chain plus a bounded dispatch controller,
  with explicit automatic-production authority and resource/storage ceilings.
- `dispatch --reviewed`: separately submit production from complete passed evidence.
- `status`: read ledgers, Slurm, checks/progress and existing reports only.

These are specification targets. Do not publish copy-paste queue commands until
the new interfaces exist, fake-Slurm tests pass and real resource checks are done.

Summary runs after benchmark termination to explain failures; dispatch depends
on a successful summary **and independently rechecks all admission evidence**.
Slurm completion alone is never sufficient. Auto-production must be explicit,
not the default consequence of submitting a benchmark.

After admission, queue control and workspace trainings independently, each with
its own afterok evaluation, then a paired report after both complete evaluations.
Do not serialize their training just to impose an arbitrary concurrency cap.
No cross-cluster dependencies, “first-start winner” lock, automatic cancellation
or requirement that one cluster's control finish before the other can run.

The candidate and control must both have valid benchmark/resource evidence.
If reviewed control reuse is requested, bind that control's hashes/runtime
explicitly; record “reused” in the report and do not submit its duplicate.

Every submission:

1. Requires committed relevant source and a fresh canonical output child.
2. Authenticates the frozen source/data/config/environment inventory.
3. Checks architecture, interpreter, available space, partition/QoS and test-only
   Slurm requests without altering anything external.
4. Acquires registry/submission locks and checks for duplicate live campaigns.
5. Writes durable submission intent before each sbatch, then records returned
   job ID/request and flushes the ledger.
6. On ambiguous replies or partial chains, preserves intent and stops; requires
   reconciliation rather than blindly creating a second output/job chain.
7. Makes retry authority explicit and never overwrites old evidence.

Sanitize inherited `SBATCH_*`/`SLURM_*`/rank/PYTHONPATH and unapproved NCCL settings
at submission; preserve scheduler-assigned GPU visibility inside workers.
Set no-user-site/no-bytecode flags. Invoke torchrun only for the two-A100 worker,
with no automatic elastic restarts; native one-GH200 uses world_size=1.

## 19. Training, restart, evaluation, and reporting implementation

Use task-explicit functions with platform adapters, not global module mutation.
Within each run, acquire a nonblocking training lock and write optimizer metadata
before work. Fresh training never loads benchmark weights or a historical model.

Save atomically at least every 1,000 attempts, before/after validation and on a
coordinated early-stop signal. Each complete restart contains:

- Scientific config/hash, source/runtime identity, task and parameter inventory.
- All model state, including workspace seed/stages/class-read and native buffers.
- Full Ranger state: RAdam moments/options/counters, slow weights and Lookahead.
- AMP scaler and cumulative skips.
- Attempted update, validated-check count, best accuracy/check/path/hash,
  train/validation timing and exact stream-consumption counters.
- Rank-specific Python/NumPy/torch/CUDA RNG and native trimmer counters.

Never serialize evolving per-forward Z or require it for resume. Saving happens
only between completed updates/validation commits, not halfway through a forward.

Keep the existing deterministic loader replay principle: rebuild the same
continuous iterators, consume the recorded batches without model or optimizer
work, then restore rank RNG before the next forward. Replay is real allocated
time and can itself be interrupted; preserve the last complete restart while
replaying. Wrong world size, normalization, task, data/source hash or missing
workspace keys must reject resume.

Handle validation transactionally. Write a new improving candidate before
committing its metadata/restart, do not overwrite the prior selected model
prematurely, and audit final selected weights against the chosen check and all
50 saved validation records. Completion requires 1M attempts **and** 50 completed
checks, not merely reaching the final training boundary.

A planned/interrupt stop writes restart and exits nonzero/incomplete, blocking
afterok successors. Do not mark early stopping as completed production or
automatically spawn extra allocations. Resumption, if needed, is a reviewed
same-platform action. One-GPU final eval of SPORC weights preserves the saved
SyncBN modules in eval and may not resume their training.

Log JSON heartbeats every 100 attempts, validation check/step/accuracy/AUC,
throughput, AMP skips, checkpoint events and replay progress. Use the common
`check_XX.json` convention with explicit task, campaign and step fields. Progress
tools report “100% toward next training boundary; validation pending” separately
from a new validation result. Do not claim live validation percentages if the
engine does not log them.

Final evaluation binds checkpoint, campaign, data and prediction-chunk hashes.
Use resumable bounded chunks over all 50 validation ROOT files, with validated
sidecars and atomic prediction writes. A retry authenticates existing chunks
before reusing them. Preserve the current class order and score definitions;
do not choose a different checkpoint using full-5M AUC.

Reports show per-arm parameters, checks/selected check, accuracy, AUC,
rejection/counts, timing, FLOPs convention, device peaks and allocated GPU-hours.
Paired deltas are within the same platform first. Label single-seed validation,
control reuse, selection overlap, numerical/runtime differences and missing
outputs. Do not pool SPORC and Tigris as two random-seed replicates or claim a
new state of the art from historical validation point estimates.

## 20. Scientific follow-up after a promising first result

These are separate, explicitly configured ablations, not silent alternatives
that a launcher chooses after seeing results:

1. **Final-readout-only:** maintain particle reads and workspace mixing across
   stages but disable particle feedback. Tests whether the extra summary/readout
   explains the improvement.
2. **Feedback-only:** remove the final workspace class read. Tests whether
   contextual particle updates help without an extra classifier route.
3. **Reset workspace:** restart Z from the learned initial tokens at each stage.
   Keep stage parameters/counts and final read otherwise matched. Tests useful
   cross-stage persistence.
4. **One-shot:** one stage after block four, clearly reporting reduced parameter
   and compute counts; not a clean frequency-only contrast without controls.
5. **Ordinary added tokens / earlier grouped branch:** compare against relevant
   simpler established mechanisms, not only baseline ParT.
6. **Capacity control on LW State-EV:** retain its bias/state behavior and widen
   native particle FFNs. With the pinned affine hidden norms, width 1,184 gives
   an analytically estimated 3,719,846 parameters, 768 below the workspace model;
   implementation must count it, not assume a shared-bias wider ParT is the same.
7. **Compute control:** use measured training compute/FLOPs to specify additional
   baseline exposure and an explicitly stretched schedule, separately from the
   equal-update comparison. Similar parameters do not imply matched compute.
8. **Additional matched seeds:** confirm gains before interpreting tiny deltas;
   lock final independent-test comparisons before opening that split.

Do not make these all part of the initial campaign. First establish whether the
locked full model helps and fits. A larger model can improve without demonstrating
the proposed mechanism. Flat or worse performance is informative under the
tested design/budget, not proof that every workspace formulation fails.

Relevant prior art: [Perceiver](https://proceedings.mlr.press/v139/jaegle21a.html)
uses iterative latent attention; [OmniLearned, section II.5](https://arxiv.org/html/2510.24066v1#S2.SS5)
already places several learned tokens in a jet backbone. This proposal must not
claim that global tokens or latent attention are new. [L-GATr](https://arxiv.org/abs/2411.00446)
builds Lorentz-equivariant geometric-algebra processing; this ordinary latent
workspace does not add that equivariance. A separate literature review remains
necessary before a novelty claim.

## 21. Implementation order and definition of done

1. **Lock configuration/inventory:** implement exact modules/counting, native
   reference guard and hashable science/runtime schemas. Add CPU oracles first.
2. **Implement pure attention/stage/class read:** pass FP64 math, masks,
   gradients, per-jet isolation and health tests with active outputs.
3. **Assemble the native-compatible model:** pass initial identity, exact native
   call/RNG/trimming/buffer checks and checkpointed backward/lifetime tests.
4. **Prove learning and recovery locally:** real Ranger activation, full-state
   serialization, bounded trajectory comparisons and fake/two-process DDP
   failure/stop tests. No GPU claims from these results.
5. **Implement independent platform admission:** frozen source/import guards,
   data/environment checks, cross-platform storage accounting and read-only plan.
6. **Implement both GPU benchmarks:** common scientific checks, platform-specific
   normalization/DDP, paired timing, memory, real-stream/restart evidence,
   resource projection and non-mutating diagnostics.
7. **Test launchers without real Slurm:** temporary repositories, paths with
   spaces, wrong architecture/interpreter, dirty source, limited storage,
   invalid dependencies, forged evidence, ambiguous replies and concurrent
   duplicate dispatch. Verify pulling later code cannot change frozen workers.
8. **Benchmark-ready milestone:** both launch plans are locally tested and
   documented; no production approval yet. Queueing still requires user action.
9. **Production-ready milestone:** implement/authenticate both platform engines,
   atomic checkpoint/selection/evaluation/reporting and explicit reviewed
   benchmark-to-production dispatch; test stops/resumes and output budgets.
10. **Real admission:** each platform's actual benchmark passes its own gates;
    only then may that platform submit the approved full paired trainings.

The implementation handoff must list tests run/skipped, planned versus observed
parameter counts, untested GPU assumptions, actual queue commands, which modes
can submit jobs, and what still needs user review. This document itself changes
only the plan; it authorizes no external jobs or environment modifications.
