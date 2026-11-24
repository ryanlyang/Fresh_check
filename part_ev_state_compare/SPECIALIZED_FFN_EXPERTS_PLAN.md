# Specialized FFN experts on full Layerwise State-EV

Recorded 2026-11-23. **Detailed implementation specification; not implemented,
benchmarked, or queue-ready.** This replaces the short experts note only.
Existing code, frozen campaigns, jobs, environments and the other three plans
are not changed by this document.

The future deliverable is one locked scientific model with **independent SPORC
tier3 and Tigris benchmark/production pipelines**. Either or both may run.
Package names, schemas and CLI modes below are proposed implementation contracts,
not commands that already exist. No classification gain is promised.

## 1. Objective and scientific hypothesis

Test whether specializing **particle transformations after relation-aware
communication** improves on full, noncompact, standard-relations Layerwise
State-EV. Keep its original shared FFN in every block and add contextual,
particle-routed nonlinear corrections in the six middle/late particle blocks.

The shared FFN handles common processing; two of four specialists supply extra
transformations for each valid particle. Routing depends on both that particle's
contextual representation and a summary of its own jet. It does not assign
truth labels or prescribe physical expert roles.

This is not new information access: native attention already mixes information
across the jet. The hypothesis is a more useful allocation of nonlinear capacity.
It is also not a free-compute change, a replacement of State-EV, an ensemble,
or evidence that interpretable routes must improve classification.

Use `part_ev_state_compare.model.build_model('state_layerwise', ...)`:

- Width 128; eight particle blocks, eight attention heads.
- Shared standard width-64 pair stem, eight independent final attention-bias heads.
- Ordinary EV and width-32 State-EV in all eight particle blocks.
- Native width-512 particle FFNs, residual/head scales, dropout and normalizations.
- Original trimmer, particle embedding, two class-attention blocks and classifier.
- Existing state query chunk 64 and shared per-forward pair normalization.
- **2,327,462 trainable parameters** in the pinned reference.

Historical 86.3151% validation is context, not an automatically matched control
or independent test measurement. In particular, a two-A100/SyncBatchNorm run
cannot silently become a control for one-GH200/native-BatchNorm training.

No persistent workspace, multiscale branch, masked training, selected relations,
new pair features, pair memory, teacher or warm start enters this first method.

## 2. Locked first configuration

Scientific identifier: `contextual_ffn_experts_state_layerwise_v1`.
Proposed candidate run ID: `EXPERTS_STATE_EV_LAYERWISE`.
Matched control: `LW_STATE_EV_CONTROL`.

| Choice | First implementation |
| --- | --- |
| Modified particle blocks | 3–8, one-based; zero-based indices 2–7 |
| Original FFNs | Retained and always active in all eight blocks |
| Specialist bank | Four independent experts per modified block |
| Routes per valid particle | Exactly two distinct experts |
| Expert MLP | Biased `128 -> 256 -> 128`, exact GELU, affine hidden LayerNorm |
| Expert order | Input linear, GELU, hidden norm, output linear |
| Expert hidden norm | Width 256, epsilon `1e-5`; no additional input norm |
| Router context | Differentiable masked mean of native pre-FFN-normalized states, then affine width-128 LayerNorm |
| Router MLP | Biased `384 -> 32 -> 4`, exact GELU between linears |
| Router input | Concatenate `u_i`, this jet's `g`, and `u_i * g` in that order |
| Routing arithmetic | FP32; stable top-two with expert-ID tie breaking |
| Combination | Softmax renormalized over the selected two logits |
| Capacity policy | Dropless; no expert quota, overflow or rerouting |
| New dropout/noise | None, including router jitter |
| New residual output initialization | All 24 expert output weights and biases zero |
| Loss change | CE plus `0.001 * mean_over_6_blocks(L_balance)` |
| Checkpointing | Each pure expert MLP, non-reentrant; routing remains outside recomputation |
| Distribution | Replicated experts, ordinary data parallelism; no expert-parallel all-to-all |
| Inference | Original ten logits, no balancing adjustment or auxiliary head |

Blocks 1–2 first establish contextual features. Six later banks allow repeated
specialization without changing every early representation immediately. Shared
processing protects the existing strong backbone; top-two permits combinations
rather than an exclusive hard partition. These are accuracy-oriented defaults,
not measured optima.

Do not silently change expert count/width, blocks, top-k, loss, dispatch policy,
dropout or initialization to fit memory or make a benchmark pass. Such a change
needs an explicit design revision, identity and new admission evidence.

## 3. Exact native-block insertion

Let `h` denote the particle states **after** the native attention/EV/State-EV
sublayer, including its native output normalization, head scaling, attention
dropout and residual. Do not route the pre-attention state.

For each modified block, preserve the pinned native FFN computation:

```text
residual = h
u = native_pre_fc_norm(h)

dense = native_fc1(u)
dense = native_GELU(dense)
dense = native_activation_dropout(dense)
dense = native_post_fc_norm(dense)
dense = native_fc2(dense)

correction, router_stats, health = specialists(u, actual_post_trim_mask)
combined = dense + cast_like(correction, dense)
out = native_final_dropout(combined)
out = out + native_w_resid * residual

return out, router_stats, health
```

The final dropout is called **once on the sum**, at the original location.
The native activation dropout still acts on the shared FFN's hidden tensor
only. The specialists have no extra dropout. The learned native residual scale
multiplies the residual, not the correction. Do not add a new residual gate,
extra output norm, expert-specific residual or a second dense FFN call.

Copy the pinned native attention portion exactly for these modified blocks,
including its nontrivial `c_attn` layout and native post-attention norm/dropout.
Call the existing wrapped native attention once, so its ordinary EV, State-EV
and attention dropout realization remain intact.

The original two particle blocks and both class blocks remain original modules
with their original forwards. No routed class FFNs, extra class token, classifier
fusion or changed attention-bias calculation is part of v1.

## 4. Shapes, masks and contextual routing

Let B be local batch size, N the actual native post-trim sequence length,
D=128, E=4, K=2, and T the number of valid particles in this local batch.

| Tensor | Shape | Contract |
| --- | --- | --- |
| Native normalized states `u` | `[N, B, 128]` | Preserve native sequence-first layout |
| Bank view `U` | `[B, N, 128]` | Transpose at the bank boundary |
| Valid mask | `[B, N]` Boolean | True means retained particle |
| Valid packed states | `[T, 128]` | Stable batch-major particle packing |
| Jet summaries `g` | `[B, 128]` | Each row uses its own jet only |
| Router inputs | `[T, 384]` | `[u, g, u*g]` |
| Router logits/probabilities | `[T, 4]` | FP32 |
| Selected expert IDs/weights | `[T, 2]` | Distinct IDs; weights sum to one |
| Assignment-slot output | `[T, 2, 128]` | One result per requested route |
| Correction | `[B, N, 128]` | Zero at invalid positions |
| Per-block sufficient statistics | 4 probabilities, 4 counts, 1 token count | No persistent token history |

For each jet j:

```text
n_j = count(valid[j])
U_safe = where(valid, float32(U), 0)
mean_j = sum_over_particles(U_safe[j]) / n_j
g_j = context_LayerNorm(mean_j)

router_input_i = concat(float32(u_i), g_jet(i), float32(u_i) * g_jet(i))
logits_i = Wrouter2(GELU(Wrouter1(router_input_i)))
p_i = softmax(logits_i over all four experts)

ids_i = stable_descending_argsort(logits_i)[:2]
w_i = softmax(gather(logits_i, ids_i) over the two selected experts)
```

Do not detach `u` or `g`: classification and balancing gradients can train their
representation. The masked mean is recomputed at every modified block; it is
not a learned persistent workspace or a batch-wide summary.

Use `where`/masked selection rather than multiplying invalid NaNs by zero.
Exclude padding from the summary, router, experts, loss statistics and new
correction. Do not change the native padded-query behavior or pair-BN population.

Reject an empty jet through coordinated worker input checks. N=1 is valid.
Support arbitrary valid layouts, not just prefix masks. Obtain masks after the
original trimming/permutation; never infer validity from features or rerun trimming.

### Top-two and ties

Sort the four logits in their fixed expert-ID order using a stable descending
sort. Exact ties therefore prefer lower expert IDs. Do not add epsilon offsets,
random noise, particle-index tie breakers, batch quotas or capacity filtering.
PyTorch's [stable argsort contract](https://docs.pytorch.org/docs/stable/generated/torch.argsort.html)
is the intended operation; verify it in the actual installed version.

Top-k membership is piecewise discrete. Differentiate selected softmax weights,
not the integer selection operation; no straight-through estimator. The full
four-way softmax is used only for the balancing statistic and diagnostics.
Using its unrenormalized selected entries for the correction would be a
different model and is forbidden.

Record second-versus-third logit margins in bounded diagnostics. Near ties can
change routes under small GPU numerical differences; stable sorting resolves
exact ties, not arbitrary cross-device rounding differences.

## 5. Dropless expert execution and deterministic combination

Each expert computes:

```text
expert_e(u) = Wout_e(LayerNorm_256(GELU(Win_e(u))))
correction_i = w_i[0] * expert_ids_i[0](u_i)
             + w_i[1] * expert_ids_i[1](u_i)
```

Pack only valid particles. Flatten requested routes in token-major, route-slot
order. Each pair `(particle, slot)` is unique, even when an expert receives many
particles. Evaluate experts in fixed ID order 0, 1, 2, 3 using gathered inputs.

Write results into unique assignment slots, then combine the two slots in fixed
order. Prefer differentiable `index_copy` into disjoint slot indices over
`scatter_add`/`index_add` into shared particle destinations: avoid introducing
unnecessary many-writer floating atomics. Scatter the combined particle results
back to the original valid positions once. Never overwrite original particle
states in place or accidentally sum two output biases twice.

There are exactly 2T requested and executed assignments per bank, with:

```text
sum_e(requested_count[e]) = sum_e(executed_count[e]) = 2*T
requested_count[e] = executed_count[e] for every expert
valid_assignment_drop_count = 0
```

An overloaded expert processes all its assignments. Its behavior cannot depend
on which other jets share the batch. No truncation to a capacity factor, no
fallback shared-only path, no expert-choice token selection, and no inference
load-balancing intervention.

### Empty experts and DDP participation

All expert parameters must remain graph-connected on every training rank,
including an expert selected by zero local particles. The preferred fixed
four-expert loop also processes correctly shaped empty tensors and attaches
their empty outputs to the assignment buffer with differentiable operations.
Verify actual zero gradients, not absent gradients, through every parameter.

If the target runtime's empty-operation path omits a dependency, add an explicit
mathematically zero autograd dependency on that expert's parameters and test it;
never invent a dummy valid token, execute that expert on unrelated particles or
enable `find_unused_parameters=True` as the initial workaround.

A local empty expert may receive nonzero reduced gradients from the other rank.
A globally empty expert still participates with zero current gradients; existing
RAdam moments and Lookahead can update its weights normally. Do not skip its
optimizer state update based on route counts.

The four experts are replicated, not sharded. There is no expert-parallel
communication or routing all-to-all. Sparse indexing can still cause launch or
device/host synchronization overhead: measure it rather than promising speedup.

## 6. Precision, checkpointing and finite-health policy

Keep master parameters FP32 and the native FP16/GradScaler training recipe.

- Native pre-FFN normalization remains the native operation; cast its output
  to FP32 for context pooling, context norm and **both router linears**.
- Disable autocast for the complete router, full/selected softmax, auxiliary
  statistics and weighted route combination.
- Expert linears follow native FP16 training autocast; compute their new hidden
  LayerNorm arithmetic in FP32, then use the prescribed linear autocast context.
- Cast expert outputs to FP32 before weighting/combining. Cast the final new
  correction to the shared FFN output dtype before its original final dropout.
- Validation/inference uses FP32 throughout; disable both outer and native AMP.
- FP64 CPU oracle mode retains float64 rather than forcibly downcasting.

Precision audits must be module-path-aware: router linears intentionally run
FP32 while native/expert linears use FP16 training autocast. Do not apply an
existing blanket “all projections are FP16” assertion, or disable native
precision auditing globally to accommodate the router. Check both expected
families explicitly and authenticate those exceptions in the configuration.

Checkpoint each expert MLP with non-reentrant checkpointing,
`preserve_rng_state=False`, because that MLP has no random operation.
Pass gathered inputs explicitly. Keep routing, selected IDs/weights and auxiliary
statistics outside checkpoint recomputation. Do not reroute during backward.
Empty expert paths must also pass checkpointed/uncheckpointed gradient tests.

Do not checkpoint native dropout, BatchNorm, attention, pair preparation or a
whole modified block. Retain existing State-EV chunk checkpointing unchanged.

Accumulate tensor-valued finite-health flags for valid inputs, context, logits,
probabilities, expert outputs and corrections without a `.item()` per expert.
Do not clamp router logits or use `nan_to_num` to hide failures. Intentional
padding exclusion is not permission to hide invalid values in valid particles.

No added routing/statistic collectives or rank-dependent exceptions inside
model forward; inherited native SyncBN still has its ordinary training
collectives. Propagate bad health to nonfinite output/loss; the training driver
coherently rejects all ranks before backward/statistic collectives. Input/I/O
errors use bounded worker-level propagation. Rank-zero-only evaluation must
not enter collectives while the other training rank is idle.

## 7. Balancing loss: exact single-GPU definition

The only loss change is:

```text
L_total = L_CE + alpha * mean_b(L_balance_b)
alpha = 0.001
b ranges over the six modified particle blocks
```

Within one update, for each bank and the actual global valid-particle population:

```text
T = global count of valid particles
P_e = sum_valid_particles(p_i[e]) / T
C_e = count of requested top-two assignments to expert e
F_e = stop_gradient(C_e / (2*T))
L_balance = 4 * sum_e(P_e * F_e)
```

`sum_e P_e = sum_e F_e = 1`, and balanced uniform routing gives loss 1,
not zero. Log this baseline correctly. Assignments use detached hard counts;
full probabilities keep gradients through the router, context and backbone.

Normalize by particles, not jets, padded tensor size, local maximum sequence
length or a capacity-filtered count. Do not divide by top-k twice, omit the
factor four, or sum over six banks instead of taking their mean.

The coefficient is a conservative starting choice, not measured optimality.
No entropy bonus, z-loss, diversity penalty, router warm-up, jitter or
auxiliary-free bias controller is included. Do not change the coefficient after
looking at validation in the same run.

Validation selection uses classification accuracy only, never CE plus balance.
Evaluation predictions have no balance-dependent bias or cross-jet routing rule.

## 8. Two-GPU balancing: correct global values and gradients

This section is mandatory. Averaging local normalized balancing losses is
generally wrong when ranks contain unequal valid-token counts or load patterns.

On rank r, for each bank, return from forward:

- `S_r[e] = sum_local_valid p_i[e]`, attached to autograd.
- `C_r[e]`, detached integer requested assignment counts.
- `T_r`, detached integer valid-token count.
- Tensor health and bounded optional summaries; no cached “last loss” attribute.

The trainer, **outside model forward**, reduces detached copies of S, C and T
over all data-parallel ranks. Use fixed bank/expert order; pack the small
statistics instead of issuing per-expert collectives.

```text
T_global = sum_r T_r
F_global[e] = sum_r C_r[e] / (2*T_global)
P_global[e] = sum_r detach(S_r[e]) / T_global

true_value = 4 * sum_e(P_global[e] * F_global[e])

local_gradient_proxy_r =
    world_size * 4 / T_global * sum_e(F_global[e] * S_r[e])

loss_for_backward_r =
    local_gradient_proxy_r + stop_gradient(true_value - local_gradient_proxy_r)
```

Use this expression per bank, average six banks, multiply by 0.001 and add to
the ordinary local mean CE. Equal local jet batch sizes 256 make DDP-averaged
CE equal the global mean CE, independently of constituent counts.

The `world_size` factor is required because ordinary DDP averages gradients.
Averaging the proxy gradients yields the gradient of the intended global loss,
including gradients to shared backbone features. Detached reductions do not
discard local S gradients. The value-correction term makes logged balancing
values identical on ranks without changing gradients. This construction follows
the documented [DDP gradient averaging](https://docs.pytorch.org/docs/stable/notes/ddp.html);
it needs direct tests, not faith in an all-reduce API.

Unreduced local activation/input gradients are world_size times their slices
of the global mean-loss oracle; DDP does not average input tensors' gradients.
Apply that expected factor in input-gradient tests. Shared parameter gradients
must match after DDP averaging, including shared layers upstream of routing.

Do not all-reduce attached S with a non-autograd collective and assume its
gradient remains correct. Do not scale the entire CE by valid-particle count.
Do not make hard assignment counts differentiable.

For world_size=1 use the same formula without communication. Final evaluation
requests logits only and performs no balancing collective. Training health and
statistic shapes are agreed before reductions; identical collective order and
timeouts are required. No rank may skip a bank because it has an empty expert.

A changed world size is a different runtime identity, not an ordinary resume.

## 9. Initialization, symmetry and learning expectations

Build the exact reference first, with ordinary seed 101. Construct additions on
CPU under isolated, versioned RNG scopes derived from seed/block/expert name,
not Python's randomized `hash()`. Preserve native weights and post-build RNG.

Initialize:

- Expert input projections: independent Xavier-uniform weights, zero biases.
- Expert hidden norm: affine weights one/biases zero, epsilon `1e-5`.
- **All expert output matrices and biases: exactly zero.**
- Context LayerNorm: affine weights one/biases zero, epsilon `1e-5`.
- Router input linear: Xavier-uniform weights, zero biases.
- Router output linear: independent normal weights, standard deviation 0.01;
  zero biases. Avoid identical all-zero router heads.
- No output scale/gate, extra expert dropout, shared expert weights, pretrained
  donor, copied dense-expert initial weights or special learning rate.

Initial classification logits and the native dropout/trimmer sequence must
match the base within declared computed-tensor tolerances. Expert corrections
are exactly zero before training. Compute the native dense path in its original
order before the new bank; new operations consume no randomness.

**Total training gradients do not initially equal baseline gradients**: the
balancing term can immediately train router/context/backbone parameters.
For a CE-only initial parity test, router and upstream expert-input gradients
can be zero while selected output projections receive a signal. After outputs
move, CE should also train routing and expert inputs.

Never require every expert to receive nonzero CE gradient in a batch where it
was not selected. Use a deterministic coverage fixture and route-aware reports
to prove each bank/expert can learn; auxiliary gradients are not proof that
an expert's nonlinear transform contributes to classification.

Use the real Ranger in an activation probe with eight successful updates within
twenty attempts, spanning rectification and a Lookahead boundary. On a mixed,
multi-particle fixture, check:

- Finite gradients/scaler/optimizer state and exact requested/executed counts.
- Nonzero selected output learning before delayed CE upstream learning.
- All expert families learn on a separately constructed coverage fixture.
- Router CE and balancing gradients are measured separately.
- Adding zero-valued empty-expert graph dependencies changes no logits.

No token roles are hand-assigned. Independent weights, contextual logits,
top-two mixtures and observed gradients provide an opportunity for specialization,
not a guarantee against redundant experts.

## 10. Parameter and compute contract

Analytical counts for the locked biased linears and affine norms:

| Component | Parameters |
| --- | ---: |
| One expert input linear | `128*256 + 256 = 33,024` |
| One expert hidden LayerNorm | `2*256 = 512` |
| One expert output linear | `256*128 + 128 = 32,896` |
| One expert | **66,432** |
| Four experts per bank | 265,728 |
| Router `384 -> 32 -> 4` | `384*32+32 + 32*4+4 = 12,452` |
| Context norm per bank | 256 |
| One full specialist bank | **278,436** |
| Six banks / total addition | **1,670,616** |
| Reference LW State-EV | 2,327,462 |
| **Planned candidate** | **3,998,078** |

This is about **71.8% more stored parameters** than the reference. It is an
analytical plan plus module-arithmetic check, not a measured full scientific
implementation. Admission must enumerate actual names/shapes/counts and reject
unexpected aliases or totals.

Two selected width-256 experts add the same linear MAC count as one width-512
FFN **per valid particle in each of six blocks**, before router/indexing costs.
The original FFN remains active. Four stored experts do not imply four experts
executed per particle.

Counting one multiply-add as two FLOPs, the added linear work for a 128-valid-
particle jet is approximately **0.2204 GFLOPs**:
six banks times `128 * 2 * (2*2*128*256 + 384*32 + 32*4)`.
This excludes GELU/norms, context pooling, sorting/dispatch, original network,
backward, checkpoint recomputation and memory traffic. It is not total FLOPs,
a training-time ratio or a throughput forecast. At shorter actual sequences
the new expert work uses valid particles; native padded work is unchanged.

At local batch 512, N=128, a FP32 `[T,2,128]` assignment buffer alone is 64 MiB.
Expert activations, checkpointing and route packing add more. Measure full device
and host peaks; sparse small GEMMs/indexing may erase nominal arithmetic savings.

## 11. Code organization and safe integration

Proposed new files, after checking the names are unused:

```text
part_ev_ffn_experts/
  __init__.py
  config.py          # locked science/counts/runtime schema
  routing.py         # contextual router, tie policy, masks and sufficient stats
  dispatch.py        # dropless expert packing, disjoint slots and empty paths
  losses.py          # single-rank/global-DDP auxiliary value/gradient contract
  model.py           # Expert, bank, local native-compatible blocks/outer forward
  diagnostics.py     # bounded non-mutating usage/gradient/intervention diagnostics
  protocol.py        # data/source/environment identity, storage and admission
  restart.py         # complete-state and route-aware restart probes
  benchmark.py       # paired target-platform measurement and correctness evidence
  engine.py          # explicit platform-aware train/evaluate/report
  submission.py      # frozen snapshots, Slurm ledgers and reviewed dispatch
  README.md          # executable commands only after implementation/testing

scripts/run_part_ev_ffn_experts.py
scripts/submit_part_ev_ffn_experts.py
sbatch/run_part_ev_ffn_experts.sh
sbatch/submit_part_ev_ffn_experts.sh
tests/test_part_ev_ffn_experts_*.py
```

Register additions under `mod.blocks.{2..7}.specialists.*` with independent
`experts.0..3`, `router` and `context_norm`. Preserve every original state_dict
key; no wrapping existing modules in a new prefix that breaks checkpoint audits.

Use a local subclass for the six newly built block instances, with a
`forward_with_router_stats` entry point, plus a local native-compatible outer
forward. Never patch installed Weaver, a global class or an existing campaign.

The outer forward obtains the actual mask once, calls original blocks 1–2 and
the six explicit modified blocks, gathers tensor-valued statistics explicitly,
then uses unchanged class blocks/norm/classifier. Native pair binding/cleanup
and existing state-pair-cache hooks remain effective, with `finally` cleanup
on success/failure.

Provide a tensor-only public inference forward and an explicit keyword such as
`return_router_stats=True` on the **DDP-wrapped forward** for training. Return
logits, bank statistics and health as a structured collection of tensors.
Never call an unwrapped custom training forward to bypass DDP, or cache graph-
attached auxiliary loss/inputs in module attributes/hooks. The run's source
identity locks the exact API.

Useful anchors:

- `part_ev_state_compare/model.py`: actual baseline and capacity contracts.
- `part_layerwise_screen/model.py`: bias and post-trim mask binding.
- `part_ev_state_speed/model.py`: existing state recomputation/cache lifecycle.
- `part_ev_pair_readout/model.py`: local native-forward assembly and pinned
  particle block order; do not copy its pair-readout-specific attention.
- `part_ev_fullscale/optimizer.py` and `checkpoint.py`: complete Ranger/RNG state.
- `part_ev_gpu_scaling/benchmark.py`: paired-step patterns, **not** a ready-made
  expert loss step.
- `part_ev_state_production/engine.py`: two-A100 execution/replay patterns.
- `part_ev_rts_tigris/production.py`: one-GH200 patterns.

Existing production/benchmark step helpers assume CE-only outputs. Implement
and test an explicit experts-aware step with the auxiliary contract above.
Do not pretend that passing a different model into a CE-only helper enables
routing regularization, or monkeypatch another module's protocol/task globals.

## 12. Autograd, invariance and failure boundaries

Pass normalized states, valid packing indices and mask explicitly. New modules
perform no persistent caching and no mutable routing history. Reuse mask-derived
packing metadata within a forward only; do not reuse learned routes across blocks.

No expert activation, per-particle route table, context summary or autograd graph
is a persistent buffer or checkpoint field. Only learned expert/router/context
parameters and ordinary optimizer state persist across batches.

Test new-branch particle permutation equivariance, arbitrary padding masks,
jet-isolated routing/corrections, N=1, all particles choosing the same two experts,
and ranks with different local expert populations. Router predictions for a jet
cannot depend on another jet's expert load.

The inherited model has training-time BN/SyncBN batch dependence. Claim only
expert-branch isolation at fixed hidden inputs and whole-model
batch isolation in eval. Native pair BN may also depend on padded pair entries;
test new-branch masking on fixed post-trim hidden states rather than demanding
a stronger raw-input invariance than the base possesses.

Keep `find_unused_parameters=False`, `static_graph=False` and replicated experts.
All six banks participate every training update, including local empty experts.
Unequal numbers of valid tokens per rank must not change collective ordering.

On valid NaN/Inf or invalid route/count metadata, fail coherently before optimizer
steps; never replace failures by finite zeros, silently drop a particle or save
a corrupted state as a successful checkpoint. No per-expert distributed checks
inside model forward. Use process-group timeouts and bounded I/O/health handling.

## 13. Full training and selection recipe

Retain the original full `ranger_full_v1` recipe; the declared 0.001 balancing
term is the only optimization/loss change.

| Field | Value |
| --- | --- |
| Training pool | Official 100M JetClass training jets |
| Validation pool | Official 5M validation jets |
| Features / particle cap | Original 17 features / 128 |
| Seed / start | Fresh 101, no benchmark or historical weights |
| Global batch | 512 |
| Budget | 1,000,000 attempted updates, nominal 512M exposures |
| Validation | 50 checks, every 20,000 attempts, 1,280,000 events per check |
| Optimizer | Pinned RAdam plus complete-state Lookahead |
| RAdam | LR 1e-3; betas (0.95, 0.999); epsilon 1e-5; weight decay zero |
| Lookahead | k=6, alpha=0.5 |
| LR | Original zero-based milestones 35–49, no new warm-up |
| Loss | Ordinary ten-class CE plus the specified mean-bank balance term |
| Native regularization | Original dropout, no new clipping/smoothing |
| Training precision | FP16 native/experts, FP32 router/reductions/master parameters, GradScaler |
| Validation | FP32, highest classification accuracy, first tie wins |
| Early stopping | None |

Use pinned `part_ev_fullscale.protocol.learning_rate(step, recipe)`.
For zero-based step s, epoch is `s//20000` and the decay multiplier is
`(0.01 ** (1/15)) ** count(milestone <= epoch)` for milestones 35–49.
First decay is update attempt 700,001. Attempt/LR cursors advance on AMP skips;
Lookahead advances only with actual optimizer steps. Preserve scaler defaults.

Register every trainable parameter in stable Ranger order, with no per-expert LR,
weight decay exception, freezing or optimizer step conditioned on expert usage.
Keep train and separate rank-zero validation streams continuous across checks.

Validation is classification-only; no balancing backward, cross-rank balancing
collective or load-dependent inference rule. Log CE separately from balancing
and total training loss so their scales are visible.

Evaluate the accuracy-selected checkpoint on all 5M validation jets: accuracy,
macro OVO/OVR AUC and existing per-class signal-versus-QCD operating points,
with acceptance counts and explicit zero-background handling. This population
overlaps checkpoint selection; independent test data remains unopened until a
locked final comparison.

## 14. Independent SPORC and Tigris execution

Each platform gets its own frozen source/runtime identity, preparation,
paired benchmark, output tree, Slurm ledger, storage reservation and production.
Neither needs a completed preparation/benchmark on the other cluster.

| Setting | SPORC | Tigris |
| --- | --- | --- |
| Host architecture | Linux x86_64 | Linux aarch64 |
| Partition / default account | tier3 / reu-aisocial | tigris / reu-aisocial |
| Training hardware | Two matching same-node A100s | One GH200, at least 80 GiB device memory |
| Global / local batch | 512 / 256 per rank | 512 / 512 |
| Native normalization | SyncBatchNorm | Native BatchNorm |
| Training distribution | DDP, gradient_as_bucket_view | Ordinary one GPU |
| Loader workers | One per rank | Two |
| Torch threads | Two per rank | Four |
| Train/benchmark CPUs and host RAM | 8 / 128 GiB | 12 / 128 GiB |
| In-training validation | Unwrapped rank-zero eval, batch 512 | Ordinary eval, batch 512 |
| Final evaluation | One A100, preserve saved SyncBN in eval | One GH200 |
| Interpreter | `/home/ryreu/miniconda3/envs/part_ev_screen_sporc/bin/python` | `/home/ryreu/miniforge3-aarch64/envs/part_ev_fullscale/bin/python` |

Both use the same science; they are not runtime-identical. Verify actual GPU
memory/type, driver, torch/CUDA, pinned Weaver, TF32 disabled, cuDNN benchmark
False and approved NCCL settings separately. Do not add packages or alter
environments to make admission pass. Do not depend on deleted selected-env paths.

Ordinary evaluation performs no training collectives, including no balancing
reduction. Never call DDP forward with idle peers. One-GPU evaluation of SPORC
weights preserves SyncBN modules in eval and cannot resume their training.

Default production is **candidate plus a fresh, identically trained LW State-EV
control on each platform**: four full trainings if both platforms are authorized.
Make this count/cost explicit. Authenticated matching control reuse is optional
only after explicit review of science, exposure, seed, BN/GPU/worker/precision,
data, source and evaluation/selection identity. A benchmark control is disposable,
not a production checkpoint.

Seed 101 on two different platforms is not two independent random seeds.
Compare candidate/control within platform first. No automatic migration of
checkpoints, first-start-winner lock, cross-cluster dependency or cancellation.
Both campaigns may run if the user chooses.

## 15. Preparation, source and runtime identity

CPU preparation authenticates train/validation inventory, content receipts,
entry counts, preprocessing YAML, class order and pinned reference/data/optimizer
sources. Do not inspect independent test files, fit new preprocessing, copy ROOT
data or build disk-resident routed-feature caches.

Freeze committed relevant source through safe git archive/transitive inventory:
reject path traversal/symlinks and live checkout imports. Use no-user-site and
no-bytecode settings; do not let routine imports change frozen file inventories.
Old queued/running jobs continue using their own snapshots after pushes/pulls.

Fingerprint installed distributions only from stable interpreter installation
roots, not transient setuptools-vendor import paths. Emit bounded actionable
differences for true environment changes; do not bypass torch/Weaver/source drift.

Science identity includes all architecture, precision, tie, loss, checkpointing
and parameter-count choices. Runtime identity includes platform, world size,
BN, local batch, workers, threads, hardware, scaler and execution environment.
Record science, data, source-manifest and runtime/campaign hashes separately.

Proposed independent trees:

```text
checkpoints/part_ev_ffn_experts/
  sporc_<fresh-id>/
    source/ source_manifest.json launch.json campaign.json data_contents.json
    logs/ probes/ benchmark.json summary.json submitted_job_ids.tsv
    production/
      production.json production_job_ids.tsv
      runs/LW_STATE_EV_CONTROL/seed_101/
      runs/EXPERTS_STATE_EV_LAYERWISE/seed_101/
      reports/
  tigris_<different-fresh-id>/
    ... separate equivalents ...
```

Never share mutable checkpoints, prediction chunks, route statistics, ledgers
or production directories between platforms. Job IDs include their cluster.
Read-only historical provenance cannot authorize new hardware.

## 16. Correctness tests before GPU admission

Implement independent tests rather than checking a function against itself.

1. **Expert/router arithmetic:** small FP64 loop oracles, exact operation order,
   context means/norms, MLPs, selected/full softmax and gradients. Include distinct
   nonzero expert outputs, not just zero-init identity.
2. **Routing:** exactly two distinct IDs, normalized selected weights, stable
   expert-ID ties, near ties, permuted particles, arbitrary valid masks, N=1
   and coherent empty-jet rejection.
3. **Dropless dispatch:** loop over particles/slots versus packed execution.
   Verify requested/executed counts, unique slots, correct expert identity,
   gradients and no padding leakage. Overload one expert with all valid tokens.
4. **Empty experts:** zero-local and zero-global populations; finite zero rather
   than missing gradients for all parameters, checkpointed backward included.
5. **Loss oracle:** global single-device balance value and attached gradients;
   uniform case equals 1, top-two counts total 2T, six-bank mean and alpha exact.
6. **DDP loss algebra:** split the same fixed hidden tokens unequally, emulate
   global counts and average local proxy gradients. Compare shared router/context
   gradients and correctly world-scaled input gradients to a direct global
   oracle; show naive local averaging fails on a
   deliberate unequal-count fixture.
7. **Initial native parity:** base weights/buffers, native trimmed inputs/masks,
   call counts and RNG/trimmer state exact; logits/shared CE-only gradients
   within floating bounds. Explicitly distinguish total auxiliary gradients.
8. **Native integration:** one trim/embedding/pair computation, eight native
   particle attentions, two class attentions, eight shared FFNs and six banks.
   Exercise active nonzero State-EV and original dropout/residual/head scales.
9. **Lifetime/recompute:** checkpointed/uncheckpointed input/parameter gradients,
   backward after pair-cache cleanup, exceptions, repeated forwards, no stale
   stats/contexts, train/eval changes.
10. **Learning:** actual Ranger coverage/delayed-gradient tests from section 9.
11. **Strict state roundtrip:** all experts/router/context, populated Ranger slow
    weights/moments/counters, scaler, native RNG/trimmer and cursor metadata.
12. **Precision/health:** actual operation dtypes; NaN/Inf in valid states, logits,
    experts or stats rejected even if a zero output could conceal it.
13. **Count/config/API:** expected names/storage independence, strict counts,
    tensor-only inference, explicit stats training API and missing-key rejection.

Freeze these initial parity limits before target-GPU evidence:

| Comparison | rtol | atol |
| --- | ---: | ---: |
| FP64 loop oracle outputs/gradients | 1e-9 | 1e-10 |
| FP32 active dispatch/router outputs | 1e-5 | 1e-6 |
| FP32 gradients / initial shared CE gradient parity | 1e-4 | 1e-5 |
| FP32 global-DDP output comparison | 5e-4 | 2e-5 |
| FP32 global-DDP gradients | 2e-3 | 2e-5 |
| FP16 checkpointed/uncheckpointed outputs | 2e-3 | 2e-3 |
| FP16 active gradients | 1e-2 | 1e-3 |

Route IDs, hard counts, masks, input identity, initial tensors, RNG, metadata
and executed/drop counts are exact. Do not use a floating tolerance to accept
a changed selected expert on the same-device deterministic oracle.

FP16/FP32 route comparisons require a fixture with documented nonzero selection
margins; separately test close/exact ties under each prescribed dtype. Frozen
FP32 router arithmetic does not make upstream FP16 states equal to FP32 states.
Use nonzero-margin fixtures for permutation checks too: masked-mean summation
order can change final rounding, and exact-tie sorting alone cannot guarantee
identical discrete routes near a floating selection boundary.

Failures require evidence-led investigation, not `strict=False`, disabling loss/
experts, tolerance inflation, router noise or silently changing top-k.

## 17. Actual GPU, DDP and route-aware restart tests

Require target-device evidence on **each** platform:

- Full-model FP32 active forward/input/parameter-gradient checks against the
  independent uncheckpointed path and CUDA FP16 recomputation parity.
- Learning probe, finite optimizer/scaler, route coverage and no dropped routes.
- Actual two-A100 SyncBN/DDP run: different local valid-token counts and expert
  usage, empty-local experts, globally imbalanced routes, parameter equality
  after reduction and correctly scaled global balance gradients.
- Single-GH200/native-BN equivalent tests with world_size=1.
- Coherent nonfinite/stop/I/O handling without a collective hang.

For global-batch DDP model parity, fix already-trimmed equal-length inputs and
disable native trimming/dropout only for that plumbing oracle. Compare against
an isolated ordinary-BN single-device global-batch reference, not a rank-zero
training-mode SyncBN reference with peers waiting. Restore actual trimming and
dropout before learning, stream, restart and timing phases. Unequal-length native
BN populations need not equal a differently padded global reference.

### Full state plus discrete routes

Use nonzero trained expert outputs and populated Ranger state. Compare immediate
save/load exactly: model/buffers, optimizer moments/slow weights/options/counters,
scaler, per-rank RNG/trimmer, attempted update, validated checks and stream cursors.

Run four trials over seven synthetic rolling-batch updates: reference,
uninterrupted repeat, disk-start repeat, and resume after update three.
Cross a diagnostic validation and Lookahead boundary with a separate probe
schema, never fake full-production completion.

After restoration, floating model/Ranger tensors may use the separately
versioned bounded numerical policy:

```text
state_abs_error <= min(1e-4, 1e-5 + 1e-4 * abs(reference))
loss_abs_error  <= min(2e-4, 1e-4 + 1e-5 * abs(reference))
```

But **selected route IDs, requested/executed counts, input fingerprints,
LR/cursor/RNG/skip/scale schedules and scalar metadata must match exactly**
across these same-device trials. Persist bounded per-update route digests,
minimum top-two margins, CE/balance/total losses and worst tensor error paths
for all ranks. Settled restart fixtures must have no AMP skips.

Route changes can amplify tiny numeric differences. Use deterministic,
nondegenerate fixed fixtures and report their margins; do not accept changed
routes merely because losses are close or use artificial inference jitter.
A failed route-aware restart gate requires diagnosis, not relaxation based on
the observed maximum error. A short successful probe is not proof of bitwise
multi-day trajectories.

Separately authenticate actual ROOT train/validation replay using a disposable
bounded probe: batch fingerprints before/after recreate-and-replay, no forward/
optimizer work during replay, then exact restored rank RNG. Distinguish real
stream evidence from synthetic tensor-state evidence. Keep production's 20k/
1.28M boundaries unchanged; a separately declared small probe boundary is not
a claim that full validation ran.

## 18. Paired benchmark and diagnostic specification

One benchmark allocation per platform measures **fresh LW State-EV and fresh
experts**, sequentially, control first, with matched initial data fingerprints
and recorded cold-start/order effects. No accuracy claim comes from benchmark
training; weights are disposable.

For each arm:

- 20 real-data warm-up attempts plus 4,200 measured attempts, windows of 1,000,
  1,000, 1,000, 1,000 and 200. Include fetch/dispatch/auxiliary/DDP/optimizer time.
- Fixed all-128 training, global batch 512: 10 warm-up, 40 measured attempts.
- 128,000 end-to-end FP32 validation events, batch 512; explicit full-128 FP32
  validation stress at batch 512.
- FP32 one-GPU inference batch 256: 10 warm-up, 20 timed batches; batch-one
  latency: 10 warm-up, 100 timed forwards.
- Full checkpoint save/load and selected-weight write measurements.
- Per-rank GPU identity, peak allocated/reserved device memory, process RSS and
  available whole-job/cgroup host-memory peak.
- CE, balance, total loss, AMP skip/scale and all model/optimizer/scaler finiteness.
- Real-stream and route-aware restart evidence; no inherited pass flags.

Control timing has no fake routing loss. Candidate timing must actually include
its global balancing step and graph-connected empty experts. Run validation on
unwrapped eval models; do not call DDP with idle peers. Shut down unused ROOT
workers before later stress/profiling phases.

Fit gates: finite settled updates, no missing execution paths, zero valid route
drops, at least 10% allocated device-memory headroom on every actual GPU, and
measured-window AMP skip fraction below 5%. Record reserved memory separately.
Count throughput over attempts, not successful updates only.

The paired benchmark has a five-hour internal deadline, 5h30m Slurm request and
early signal. Incomplete evidence is inconclusive/failure, never an automatic
pass or silent reduction of population, batch, measured windows or correctness.

### Bounded profiling and routing diagnostics

Profile at most two jets/32 particles, one train and one eval iteration; retain
aggregate scopes, not an unbounded full-batch profiler history. Scopes include
context/router, stable selection, packing, expert MLP/recomputation and route
combination. Require positive call counts and finite nonnegative CPU/device
times, with at least one positive timing per required scope. Inclusive scopes
cannot simply be summed.

For bounded isolated eval samples and aggregate training counters, record:

- Requested/executed route counts and zero-drop assertion.
- Full probability means and primary/secondary expert usage per bank.
- Selected/full entropy, second–third margin summaries, balance and CE scales.
- Expert correction/shared-FFN RMS ratio; norm growth over modified blocks.
- On small covered fixtures, expert/router CE-only versus auxiliary gradient norms.
- Token-normalized global usage, not an average of local normalized usages.

Collapse/rarely used experts are diagnostics, not a mandatory equal-per-jet
constraint. Severe persistent non-use triggers a reported investigation, not
adaptive loss/noise changes. Correlations with particle properties or class do
not establish physical expert specialization.

Diagnostics restore weights/buffers/modes/RNG/trimmer/caches/gradients and use
bounded samples. Save no per-event route activations or jet feature databases.
A routed implementation need not be faster than a dense capacity control.

## 19. Resource projection and walltime policy

| Phase | Initial request | SPORC | Tigris |
| --- | --- | --- | --- |
| Prepare | 2 h | CPU, 8 CPUs / 64 GiB | CPU, 8 CPUs / 64 GiB |
| Paired benchmark | 5 h 30 min | 2 A100s, 8 CPUs / 128 GiB | 1 GH200, 12 CPUs / 128 GiB |
| Summary | 15 min | CPU, 2 CPUs / 4 GiB | CPU, 2 CPUs / 4 GiB |
| Optional dispatch | 30 min | CPU, 2 CPUs / 4 GiB | CPU, 2 CPUs / 4 GiB |
| Train per arm | Measured | 2 A100s, 8 CPUs / 128 GiB | 1 GH200, 12 CPUs / 128 GiB |
| Evaluate per arm | Measured | 1 A100, 8 CPUs / 128 GiB | 1 GH200, 12 CPUs / 128 GiB |
| Report | 1 h | CPU, 2 CPUs / 8 GiB | CPU, 2 CPUs / 8 GiB |

These are planned requests, not current cluster guarantees. Verify partition,
account/QoS, GPU/CPU/host memory, maintenance and test-only Slurm requests on
the actual submission host. Never choose a different GPU to make a queue start
without matching runtime evidence.

For each arm independently, let s be slowest measured real-data seconds/attempt
over rank/windows, v end-to-end 128k validation seconds, c full-restart write
seconds, and w selected-weight write seconds:

```text
T_compute = 1,000,000 * s
T_validation = 50 * 10 * v
T_writes = (1,000 + 100 + 2) * c + 51 * w
train_hours = ceil(1.30 * (T_compute + T_validation + T_writes
                         + measured_cold_start) / 3600)
```

The write allowance includes periodic and pre/post-validation states, improving
selected candidates and final best copy. Save measurements/formula in evidence.
Evaluation must include full-5M inference, checkpoint loading, prediction writes
and CPU metric/merge cost, not kernel throughput alone.

Use measured recommendations, not “experts should be cheap.” Planned ceilings
are 480 training hours and 24 evaluation hours, subject to stricter live limits.
If a recommendation exceeds a limit, stop production admission and request a
new execution decision; never cap it down, silently shrink the model or split
into three-day chains. One long allocation per full training is the intended
execution; interruption-safe resume still exists.

The 30% margin does not bound filesystem contention, queue time or restart
replay. Report GPU-hours as allocated walltime times GPU count separately from
FLOPs/jet and throughput.

## 20. Storage, retention and cross-campaign accounting

Model parameters, complete restarts and validation predictions dominate output;
there are no persistent routing tables or dataset copies.

For P parameters, prospective production bytes per arm are:

```text
weights = 51 * (8*P + 1 MiB)
restarts = 3 * (24*P + 16 MiB)
predictions = 5,000,000 * 48 + 64 MiB
misc = 1 GiB
arm_budget = weights + restarts + predictions + misc
```

Reserve 2 GiB per paired-benchmark platform and one shared 4-GiB emergency margin.
With the planned counts and two fresh arms on both platforms this is about
**19.19 GiB aggregate initial headroom**, **before other outstanding campaigns**.
This is a planning calculation, not available space or a quota reservation.
Subtract already present committed outputs from remaining budgets without
counting uncommitted/active files as free space.

Retain the latest complete restart with safe atomic-write temporary allowance,
at most 50 improving selected-weight candidates, final best weights, bounded
logs/diagnostics and resumable prediction chunks. No accumulating full restart
at every update or per-event expert caches.

Use shared-filesystem registry locks across both platforms, including existing
campaign registries already counted by current helpers and the other proposed
method budgets. Reserve before submission; recheck before prepare, dispatch
and large output phases.

A missing job from local squeue cannot release another cluster's reservation.
Record cluster/job ledger identity and reconcile explicitly. No automatic
environment/cache/dataset/checkpoint deletion, no pruning other campaigns and
no dependence on reclaiming hard-linked files.

## 21. Submission and automatic production contract

Proposed launcher `--platform sporc|tigris` selects explicit resources/interpreter.
Workers import only frozen campaign source. Planned modes:

- `plan`: read-only validation, paired tasks/resources/storage/automation summary.
- `benchmark`: freeze new source; submit prepare -> benchmark -> summary.
- `review`: authenticate/recompute evidence and recommendations without writes.
- `pipeline --reviewed`: same benchmark chain plus explicitly authorized,
  bounded automatic-production dispatch.
- `dispatch --reviewed`: separately submit production from passed target evidence.
- `status`: read-only jobs/progress/checks/reports.

These interfaces must be implemented/tested before publishing queue commands.
Benchmark/summary completion does not authorize production by itself.

Summary runs after benchmark termination to explain failures. Dispatch waits
for a successful summary and independently rechecks model/loss/route/restart,
data/source/runtime and resource/storage admission. A successful Slurm exit or
file named “passed” is not sufficient.

After admission, submit control and experts trainings independently, one afterok
evaluation each, then a paired report after both evaluations. No arbitrary
training concurrency cap. Optional authenticated control reuse must be explicit;
reports state reuse and its matching evidence.

Protect every submission with canonical scoped output paths, committed relevant
source, snapshot verification, live resource/storage checks, shared registry
lock and per-output submission lock. Write/flush durable intent before sbatch;
record cluster/job ID and full request immediately afterward. Ambiguous replies,
partial chains and duplicates require reconciliation, not blind retries.

Sanitize inherited SBATCH/SLURM/rank/PYTHONPATH and unapproved NCCL options at
submission; preserve scheduler-assigned GPU visibility inside workers.
Set no-user-site/no-bytecode flags. Use torchrun only for SPORC's two ranks,
with no elastic auto-restart; one GH200 is ordinary world_size=1.

Pushing/pulling later code cannot alter any existing frozen worker. Do not
modify current launchers or snapshots to redirect their imports to this package.

## 22. Production, restart and evaluation transactions

Implement an explicit task/platform-aware engine. No CE-only helper reuse that
forgets router statistics, no module-global protocol mutation.

On every training attempt:

1. Fetch the platform's continuous rank batch; agree input health.
2. Set pinned LR, zero gradients and run DDP/ordinary forward with explicit stats.
3. Agree finite logits/stats/health; reduce detached global balancing statistics.
4. Form CE plus the exactly scaled auxiliary proxy; agree finite loss.
5. Scale/backward, unscale/check gradients, normal GradScaler/Ranger step/update.
6. Record successful/skipped attempt, small aggregate routing/loss statistics.
7. Commit checkpoints/validation at boundaries; no live autograd graph retained.

Agree stop/failure signals coherently. Do not let rank-local exceptions skip
required collectives. Avoid per-expert CUDA-to-host synchronization for logging;
reduce/synchronize only bounded aggregate data in the normal step protocol.

Save atomically every 1,000 attempts, before/after validation and on early stop:

- Science/source/data/runtime/task identities and parameter inventory.
- All native and new expert/router/context model state.
- Complete RAdam/Lookahead moments, slow weights, options and counters.
- AMP scaler/cumulative skips and exact rank RNG/trimmer states.
- Attempted update, completed validation count, best check/accuracy/path/hash,
  continuous train/validation consumption counters and timing.
- Bounded cumulative routing counters/metric schema, not last-batch graphs.

No persistent per-forward router state is needed for resume. Reconstruct loaders,
replay consumed batches without model/optimizer work, then restore rank RNG
before the next forward. Replay consumes allocation time and must itself be
interruptible without destroying the previous complete checkpoint.

Reject wrong platform/world size/BN, missing expert keys, changed data/source/
loss/tie policy or inconsistent cursor. Do not resume one-GPU state as a two-GPU
job. A checkpoint's stored route diagnostics cannot override recomputed routes.

Validation selection is transactional: write a new improving candidate before
committing metadata; preserve the previous selected model on interruption.
Audit completion only after 1M attempts **and** 50 completed checks and verify the
selected checkpoint matches the recorded accuracy-selected check.

A graceful incomplete stop saves recoverable state and exits nonzero, blocking
afterok successors. No fake completion, automatic extra allocation or automatic
cross-cluster continuation. Resume is a separate reviewed same-runtime action.

Log JSON heartbeats every 100 attempts, CE/balance/total losses, skips, throughput,
validation boundaries/results, checkpoint/replay events and bounded routing
summaries. Use `check_XX.json` sidecars. Distinguish “100% toward next training
boundary; validation pending” from a completed validation result.

Full-5M evaluation uses bounded resumable prediction chunks over the 50 official
validation ROOT files. Authenticate checkpoint/campaign/data/prediction hashes
and sidecars before reuse; preserve class order and scoring conventions.
Never select a second checkpoint by full-5M AUC.

Reports show accuracy/AUC/rejections with counts, selected check, parameters,
FLOPs convention, timing, GPU/host peaks and allocated GPU-hours. Label single
seed, selection overlap, platform differences/control reuse and missing evidence.
Report within-platform deltas; do not pool the platforms as two random seeds.

## 23. Ablations, prior art and interpretation

Do not automatically queue these extras with the initial model. If promising:

1. **Particle-only router:** remove jet summary/product inputs; test whether
   contextual routing adds value, reporting the changed router parameter count.
2. **Uniform all-four mixture:** same stored experts, all four evaluated/averaged;
   a dense-mixture control with higher active compute, not a compute match.
3. **Fixed selected pair:** same two experts for every particle; clearly report
   dormant capacity. Not a clean stored-parameter-matched active model.
4. **Zero balancing:** isolate auxiliary influence without changing routing;
   no retrospective coefficient switch inside an existing run.
5. **Equal-update dense capacity control:** retain full LW State-EV but widen
   native FFNs in blocks 3–8 to 1,587. Analytically **3,998,012 parameters**,
   only 66 below the candidate. Confirm actual counts; this is a strong capacity
   control but activates more linear work and has awkward hardware alignment.
   An all-eight width-1,318 alternative is 3,997,494, with different placement.
6. **Approximate active-linear-compute control:** widen those six native FFNs
   from 512 to 1,024, retaining State-EV/LW bias; similar main linear MAC budget
   to shared-plus-two specialists before router/dispatch costs. Measure actual
   FLOPs/walltime; stored parameters and functional form differ.
7. **Compute/exposure control:** explicitly budget additional reference training
   from measured FLOPs with a declared stretched LR schedule, separate from the
   equal-update contrast.
8. **Extra matched seeds and locked independent test:** establish reproducibility
   before attributing small gains to specialization or announcing a new result.

Dense capacity controls should preserve LW State-EV, not silently use shared-bias
baseline ParT and remove the method being extended.

[Conditional Capacity and Routing in Mixture-of-Experts Particle Transformers](https://arxiv.org/abs/2610.02701)
already studies MoE ParT, including dropless and multi-active-expert settings.
It uses a restricted 188-class JetClass-II study, not this full 100M-jet
ten-class LW State-EV experiment. It supports investigating routing/active compute
separately, not forecasting our gains. [DeepSeekMoE](https://arxiv.org/abs/2401.06066)
provides shared/routed expert separation precedent. No claim that MoE or shared
experts are new is justified.

Potential contribution must be established by the particular contextual,
shared-plus-specialist addition, its controlled performance/cost trade-off and
evidence beyond capacity or regularization. Route correlations are not proof
of distinct physical roles. Flat results remain informative about this design/
budget, not all expert methods. A separate literature review is required before
claiming novelty.

## 24. Implementation sequence and definition of done

1. Lock hashable scientific/runtime config and parameter inventory, then write
   independent attention-free router/expert/loss oracles.
2. Implement FP64/FP32 contextual routing and dropless disjoint-slot dispatch,
   including empty experts and exact ties.
3. Prove single/global-DDP auxiliary values and gradients with unequal counts;
   test CE and auxiliary contributions separately.
4. Integrate local native-compatible blocks/outer forward; pass initial identity,
   active native call/order/RNG, masks, strict state and recomputation tests.
5. Prove Ranger activation/coverage and route-aware full-state recovery; implement
   bounded coordinated failures and disposable real-stream replay probes.
6. Build independent platform preparation, snapshot/env/data identities,
   cross-campaign storage reservation and read-only planning.
7. Implement paired actual-device benchmarks with mandatory auxiliary loss,
   CUDA precision/SyncBN/DDP/empty-expert correctness, fit, timing and restart.
8. Test all launch modes with fake Slurm: wrong architecture/interpreter,
   path spaces, dirty source, insufficient space, forged evidence, invalid
   dependencies, duplicate dispatch, partial/ambiguous submissions and frozen
   import safety after live checkout changes.
9. **Benchmark-ready:** both launchers locally tested/documented, actual queue
   commands supplied, no production approval inferred from local CPU success.
10. **Production-ready:** explicit experts-aware engines, complete checkpoints,
    transactional selection, resumable evaluation/reporting and reviewed
    automatic dispatch tested; actual target-platform evidence still required.
11. **Admitted production:** that platform passes its own real benchmark and
    requested resource/storage limits before any authorized full training.

The handoff lists tests run/skipped, actual versus planned counts, untested GPU
assumptions, loss/routing/restart evidence, resource recommendations, commands
and which modes submit jobs. This document itself changes only a plan; it
authorizes no external submission, environment mutation or deletion.
