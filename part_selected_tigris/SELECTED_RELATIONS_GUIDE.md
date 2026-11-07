# Selected relations: setups, optimizations, and code map

Repository guide, checked against the implementation on 2026-11-05.

For the subsequently agreed seven-model **full-schedule** comparison, use the
[new benchmark-stage README](../part_selected_fullscale/README.md) and
[campaign plan](FULLSCALE_SEVEN_RUN_CAMPAIGN_PLAN.md). Its separate launcher
queues diagnostics only; the four-model short-screen workflow below is unchanged.

After the seven benchmarks and restart retries pass, use the separate
[full-schedule production launcher](../part_selected_production/README.md).
It queues the seven individual training/evaluation runs using the original
frozen scientific snapshot and the measured per-model walltimes.

This is the starting point for understanding the PT/TRACK/REGION selected-relation
study and its faster execution candidate. It separates **changes to the model**
from **changes intended to run the same model more efficiently**.

The short version: selected models add three learned relation families to ParT's
existing four pair descriptors. We first reduced redundant tensor storage, then
added a separate implementation that batches and parallelizes REGION tree
construction. The latter still needs its own cluster parity and performance
results before we can claim a measured speedup. It does not precompute all jets
or freeze the learned relation networks.

All links below are relative to this file. The repository root is normally:

- Local: `C:\Users\22rya\ComputerScience\CERN\Fresh_check`
- Research Computing: `/home/ryreu/atlas/Fresh_check`

## 1. Which setup is which?

| Location / workflow | Purpose | Hardware / execution |
|---|---|---|
| [Historical relational code](../teacher_logit_reco/relational_part/) | Earlier relation definitions, selection studies, and offline transfer/ablations | Historical experiments; not the current screen launcher |
| [Selected screen](../part_selected_screen/README.md) | Four selected-relation models using the reusable short Ranger recipe | SPORC A100; includes the memory-efficient implementation |
| [Tigris implementation](protocol.py) | Same four model configurations on a larger-memory GPU | One GH200 per run; separate hardware-transfer contract |
| [Walltime recovery](../scripts/recover_part_selected_tigris_walltime.py) | Allow the existing Tigris campaign enough time after its runtime gate rejected the original reservation | Resource-only recovery; no model or speed optimization |
| [Fast candidate](../part_selected_fast/README.md) | Faster REGION tree construction/packing, optional cache/compilation, and measured admission checks | One GH200 per run; new campaign, not a patch to old jobs |

The detailed proposed work is in [OPTIMIZATION_PLAN.md](OPTIMIZATION_PLAN.md).
Its unchecked items include remote measurements and optional ideas, not just
missing code. This guide describes what is actually implemented.

The grouped-attention study, State-EV study, and `evseg_*` segmented DDP jobs are
separate experiments. In particular, a running `evseg_segment` job is not one of
these four selected-relation runs.

## 2. The four model configurations

Every new selected run uses seed 101. The four existing standard-relation
controls are reused as comparison results, not as initialization checkpoints.

| Task | Architecture key | Run ID | Reused control | Parameters (selected / control) |
|---:|---|---|---|---:|
| 0 | `selected_shared` | `SELECTED_PART_SHARED` | `PART_RANGER` | 2,153,406 / 2,143,486 |
| 1 | `selected_shared_ev` | `SELECTED_EV_SHARED` | `EV_RANGER` | 2,218,942 / 2,209,022 |
| 2 | `selected_layerwise` | `SELECTED_PART_LAYERWISE` | `PART_LAYERWISE_RANGER` | 2,157,158 / 2,147,238 |
| 3 | `selected_layerwise_ev` | `SELECTED_EV_LAYERWISE` | `EV_LAYERWISE_RANGER` | 2,222,694 / 2,212,774 |

Each selected model adds 9,920 parameters to its corresponding control. These
are therefore not parameter-matched comparisons. The fast implementation keeps
the selected model's parameter count and state-dict structure unchanged.

**Shared versus layerwise refers to scalar attention-bias heads.** The shared
case reuses one bias output across the eight particle-attention blocks. The
layerwise case uses a separate final bias head for each block, while sharing the
relation encoders and pair stem. In both EV configurations, the EV value
projections are already independent per layer. We are not comparing a shared EV
projection against independent EV projections here.

### What the selected relations contain

We retain the four native ParT kinematic pair descriptors and concatenate three
learned embeddings:

| Family | Information supplied | Learned encoder | Output channels |
|---|---|---|---:|
| PT | Transverse-momentum fractions, ranks, and directional pair combinations | 10 -> 32 -> 8 | 8 |
| TRACK | Raw impact parameters/errors, significances, track validity, and pair compatibility | Shared endpoint network 7 -> 32 -> 16; pair network 81 -> 48 -> 12 | 12 |
| REGION | Beam-free angular clustering, exclusive K=2/4/8 groups, common ancestors, cluster fractions, and geometry | 41 -> 32 -> 12 | 12 |

Each family network uses GELU and RMSNorm between its linear maps. Thus the pair
stem receives `4 + 8 + 12 + 12 = 36` channels, with its width still 64. The learned
embeddings and stem are computed once per forward pass and used across blocks;
they are not independently rebuilt eight times.

The data flow is:

```text
Native loader: 17 particle features + four raw track transport channels
  -> unchanged native particle permutation/trimming
  -> retain the original 17 inputs for the particle embedding
  -> compute standard pairs and PT/TRACK/REGION on the retained particles
  -> normalize fixed descriptors and run the learned family encoders
  -> concatenate 36 channels -> shared 64-wide pair stem
  -> shared or layerwise attention bias; optionally per-layer EV messages
  -> unchanged particle/class attention and classifier
```

Two distinctions matter when comparing to earlier studies or standard ParT:

- Directional selected features use dense directed pairs. Pair BatchNorm sees
  that population, including padded entries, instead of the standard control's
  symmetric triangular packing. The intervention includes this change as well
  as the extra information.
- The current REGION helper corrects an unequal-depth lowest-common-ancestor
  lookup in the historical vectorized implementation. It uses set-bit binary
  lifting, tested against parent walks. Historical files/artifacts remain
  unchanged; this is not bit-for-bit reproduction of the earlier lookup defect.

### Training and normalization

The scientific recipe comes from
[REUSABLE_SCREEN_PROTOCOL.md](../part_ev_screen/REUSABLE_SCREEN_PROTOCOL.md),
implemented by [recipes()](../part_ev_screen/protocol.py):

- Official 100M training population; batch 512; two original loader workers.
- `ranger_short_v1`: 200,000 attempted updates, or 102.4M training exposures.
  This is a short screen over the 100M population, not the full million-update
  recipe and not ten full dataset passes.
- RAdam + Lookahead; zero weight decay; initial LR `1e-3`. The compressed
  reference schedule decays at steps 140,000 through 196,000 every 4,000 steps,
  reaching `1e-5`. No AdamW substitution is part of this study.
- Native single-GPU BatchNorm; FP16 autocast training with FP32 master weights
  and GradScaler; FP32 evaluation; TF32 disabled.
- Ten checkpoint checks, every 20,000 updates, each on 1.28M examples from the
  continuous validation stream. Highest checkpoint-validation accuracy wins;
  the first occurrence wins ties.
- Evaluate that checkpoint on the official 5M validation set. This is not an
  independent 20M-test evaluation.

Normalization is fitted once using the first 50,000 jets of the seed-101
training stream, with up to 64 sampled directed pairs per jet. This subset is
for estimating scales, not the model's training population. Designated continuous
channels use median/IQR scaling and clipping to [-8, 8]; track uncertainty floors
come from valid positive errors in that subset. Constants are frozen for all
four models. The fast candidate copies and verifies the Tigris reference's
constants rather than fitting new ones.

Tigris selected runs reuse the completed SPORC controls, so their scientific
comparison is cross-hardware. Source/data checks do not imply bit-identical
ARM/x86 preprocessing or training. Fast-versus-reference speed measurements are
instead paired on the same GH200 and runtime.

## 3. Why the selected models are expensive

The small parameter increase understates the work. At batch 512 and 128
particles, there are 8,388,608 directed pair positions. Large intermediate
feature tables, transfers, and non-learned clustering can dominate even when
the MLPs themselves have few parameters.

The reference REGION path copies inputs to CPU, loops over jets to call the
compiled tree builder, returns per-jet dictionaries, validates/packs their
fields, and transfers packed fields back to GPU. PT/TRACK/REGION also produce
and process additional dense pair features. Moving to a GH200 addresses the
40-GiB A100 memory failures seen in this study, but does not by itself eliminate
CPU work, synchronization, or Python packing overhead.

These are code-visible sources of cost, not a measured ranking of bottlenecks.
The new profiles separate CPU tree/packing, transfers, model components, and
end-to-end throughput before deciding which optimization helps most.

## 4. Implemented optimizations

### A. Memory-efficient selected execution (`selected_memory_v1`)

These changes live in the original selected model package and are already part
of the Tigris reference used by the new fast candidate:

1. **One packed EV-stem allocation.** Store it physically as
   `[batch, query, key, channel]`, while exposing the logical axes expected by
   the original contraction. Blocks save views of that allocation instead of
   each making a separate contiguous copy. A full 512 x 128 x 128 x 64 FP16
   stem is 1 GiB; replacing eight copies with one removes 7 GiB of duplicate
   saved storage in that case. This is not a measurement of total peak-memory
   reduction.
2. **No unused EV message in non-EV models.** Bias-only models no longer
   allocate a value-message tensor they will never consume.
3. **Process raw families sequentially.** Release raw PT/TRACK/REGION tables
   when possible rather than retaining all three simultaneously. Autograd still
   retains what it needs for the learned networks.
4. **Avoid a large FP32 TRACK concatenation under AMP.** Cast the fixed
   descriptors to the following Linear's existing autocast dtype before
   concatenation. Descriptor calculations and normalization remain FP32; the
   effective input to the Linear is preserved.

Code: [model.py](../part_selected_screen/model.py), especially `packed_ev_stem`
and `SelectedPairs.forward`; [features.py](../part_selected_screen/features.py),
`Relations.forward`; [memory.py](../part_selected_screen/memory.py) for parity.

No activation checkpointing, CPU activation offload, smaller batch, or extra
backward recomputation was introduced. Shared-storage gradient accumulation
can round differently, so identical final training trajectories are not promised.

### B. Batched parallel trees and direct packing (`selected_fast_v1`)

This is the default candidate in the separate `part_selected_fast` package:

- One native call constructs the batch of trees. A bounded OpenMP team works
  across independent jets, preserving their output order. The order-sensitive
  operations within each tree are not parallelized.
- The per-tree core retains canonical tie handling, merge rules, float64 tree
  arithmetic, and final float32 outputs. The build disables fast-math,
  reassociation, and floating-point contraction.
- The native backend writes directly into contiguous integer/float arrays,
  avoiding the production path's per-jet Python dictionaries and repeated
  field conversion/packing. CPU indices are int32, expanded to the reference
  int64 indexing on device; feature storage remains float32.
- GPU REGION feature arithmetic is reused from the reference, with the packed
  tree input substituted. PT/TRACK computations and learned relation networks
  are unchanged. Transfers still exist; this is not a fully GPU tree builder.
- The adapter is instance-local. It does not modify the reference module's
  global functions, existing campaign source, or old checkpoints.

Code: [tree_core.h](../part_selected_fast/csrc/tree_core.h),
[selected_tree_batch.cpp](../part_selected_fast/csrc/selected_tree_batch.cpp),
[trees.py](../part_selected_fast/trees.py), and
[model.py](../part_selected_fast/model.py).

### C. Optional exact-view validation RAM cache

[cache.py](../part_selected_fast/cache.py) provides a bounded four-way FIFO byte
arena. [PackedTrees](../part_selected_fast/trees.py) keys entries by campaign
identity and exact ordered input content/mask. It stores fixed packed tree data,
not learned outputs or autograd tensors. Eviction or a miss rebuilds the same
view. Nothing spills to disk or enters a training checkpoint.

Caching is validation-only and **off by default** (`--cache-gib 0`). Each job
has its own cache. An 8-GiB cache, for example, is an optional experiment, not a
requirement or a promise of reuse across the 5M stream. A warm repeated-batch
microbenchmark is labeled separately from actual-stream performance.

### D. Optional compilation of GPU REGION descriptors

`--compile-region` uses `torch.compile` only on the pure GPU REGION descriptor
function, outside CPU clustering. It is off by default. Compilation cost,
numerical agreement, and end-to-end speed must be checked; failure stops the
candidate instead of silently falling back. Other family encoders and attention
are not rewritten by this option.

### What we deliberately have not implemented

We do not precompute all REGION trees or all three learned families at startup.
Native trimming changes the retained particle view, so an untrimmed tree cannot
simply be reused as though it were built from the current constituents. Learned
family embeddings also change as their weights train. More host RAM does not
remove either dependency, and this short stream does not guarantee useful
training-cache reuse.

Training-tree caching, ahead-of-model pipelining, TRACK linear-map fusion or
algebraic reassociation, and batching the layerwise bias heads remain deferred.
They require profiling and separate checks of AMP rounding, RNG order, and
restart behavior. There is no model simplification hidden in the fast path.

## 5. Where to edit or inspect the code

### Scientific model, data, and inherited definitions

| File | Responsibility |
|---|---|
| [part_selected_screen/model.py](../part_selected_screen/model.py) | Four architectures, pair-stem expansion, bias heads, EV binding, trimmer hook, memory layout |
| [part_selected_screen/features.py](../part_selected_screen/features.py) | Normalizers and learned PT/TRACK/REGION encoders |
| [part_selected_screen/region.py](../part_selected_screen/region.py) | Reference tree calls/packing, corrected LCA, GPU REGION features |
| [part_selected_screen/data.py](../part_selected_screen/data.py) | Raw track transport channels, unchanged native inputs, finite normalization prefix |
| [part_selected_screen/normalization.py](../part_selected_screen/normalization.py) | Training-only normalization fit and uncertainty floors |
| [part_selected_screen/protocol.py](../part_selected_screen/protocol.py) | Task IDs/counts, donor checks, model settings, benchmark requirements |
| [part_selected_screen/reporting.py](../part_selected_screen/reporting.py) | Control validation, selected/control metrics and contrasts |
| [part_ev_fullscale/model.py](../part_ev_fullscale/model.py) | Native ParT wrapper and original EV mechanism |
| [part_layerwise_screen/model.py](../part_layerwise_screen/model.py) | Scalar layerwise-bias attention wrapper |
| [part_ev_screen/protocol.py](../part_ev_screen/protocol.py) | Reusable short recipe and LR schedule |
| [part_ev_fullscale/optimizer.py](../part_ev_fullscale/optimizer.py) | RAdam/Lookahead implementation and restartable optimizer state |
| [part_screen_common/engine.py](../part_screen_common/engine.py) | Shared benchmark/train/checkpoint-lock/full-validation orchestration |
| [scripts/run_part_ev_screen.py](../scripts/run_part_ev_screen.py) | Reused training-step, validation, and metric functions |
| [part_screen_common/contracts.py](../part_screen_common/contracts.py), [controls.py](../part_screen_common/controls.py) | Provenance, donor compatibility, checkpoint and prediction checks |

Historical feature definitions are under
[teacher_logit_reco/relational_part/](../teacher_logit_reco/relational_part/):

- [relation_pt.py](../teacher_logit_reco/relational_part/relation_pt.py),
  [relation_track.py](../teacher_logit_reco/relational_part/relation_track.py),
  [relation_region.py](../teacher_logit_reco/relational_part/relation_region.py),
  and [normalization.py](../teacher_logit_reco/relational_part/normalization.py)
  define inherited features/channel names.
- [region_tree.py](../teacher_logit_reco/relational_part/region_tree.py) is the
  Python tree oracle/schema;
  [ca_tree.py](../teacher_logit_reco/relational_part/ca_tree.py) authenticates and
  loads the original compiled backend;
  [relational_ca_tree_v1.cpp](../teacher_logit_reco/relational_part/csrc/relational_ca_tree_v1.cpp)
  is that backend's source.
- [offline_transfer.py](../teacher_logit_reco/relational_part/offline_transfer.py)
  and [offline_ablations.py](../teacher_logit_reco/relational_part/offline_ablations.py)
  belong to the earlier offline campaigns. Do not use their launch logic for
  this reusable screen.

### Tigris transfer and fast execution

| File | Responsibility |
|---|---|
| [part_selected_tigris/protocol.py](protocol.py) | ARM/GH200 identity, source/environment checks, timing admission |
| [part_selected_tigris/inputs.py](inputs.py) | Native Tigris input-stream reference; no false ARM/x86 byte-parity claim |
| [part_selected_tigris/engine.py](engine.py), [reporting.py](reporting.py) | GH200 execution and explicitly cross-hardware reporting |
| [part_selected_fast/csrc/tree_core.h](../part_selected_fast/csrc/tree_core.h) | Per-jet native tree algorithm |
| [part_selected_fast/csrc/selected_tree_batch.cpp](../part_selected_fast/csrc/selected_tree_batch.cpp) | Batched OpenMP entry point and direct packing |
| [part_selected_fast/trees.py](../part_selected_fast/trees.py) | Compile/load, backend parity, packed transport, cache lookup |
| [part_selected_fast/cache.py](../part_selected_fast/cache.py) | Bounded RAM-only byte cache |
| [part_selected_fast/model.py](../part_selected_fast/model.py) | Instance-local adapter and optional REGION compilation |
| [part_selected_fast/parity.py](../part_selected_fast/parity.py) | Feature/gradient/optimizer/RNG/restart equivalence checks |
| [part_selected_fast/profiling.py](../part_selected_fast/profiling.py) | Scoped profiler instrumentation and host/GPU memory monitoring |
| [part_selected_fast/protocol.py](../part_selected_fast/protocol.py) | New immutable execution contract and speed/resource admission |
| [part_selected_fast/engine.py](../part_selected_fast/engine.py) | Fast model/loader binding into the existing training engine |
| [scripts/profile_part_selected_fast.py](../scripts/profile_part_selected_fast.py) | Paired reference/candidate timing and standard-model probes |
| [scripts/run_part_selected_fast.py](../scripts/run_part_selected_fast.py) | Prepare/profile/benchmark/gate/train/evaluate/report/status entry point |
| [scripts/submit_part_selected_fast.py](../scripts/submit_part_selected_fast.py) | Source snapshot, resources, dependency graph, job ledger, safe resume |

### Launchers and workers

| Setup | Submission script | Worker / additional driver |
|---|---|---|
| SPORC selected screen | [submit_part_selected_screen_sporc.sh](../sbatch/submit_part_selected_screen_sporc.sh) | [run_part_selected_screen_sporc.sh](../sbatch/run_part_selected_screen_sporc.sh), [run_part_selected_screen.py](../scripts/run_part_selected_screen.py) |
| Reference Tigris selected screen | [submit_part_selected_tigris.sh](../sbatch/submit_part_selected_tigris.sh) | [run_part_selected_tigris.sh](../sbatch/run_part_selected_tigris.sh), [run_part_selected_tigris.py](../scripts/run_part_selected_tigris.py) |
| Existing campaign's walltime recovery | [submit_part_selected_tigris_recovery.sh](../sbatch/submit_part_selected_tigris_recovery.sh) | [run_part_selected_tigris_recovery.sh](../sbatch/run_part_selected_tigris_recovery.sh), [recover_part_selected_tigris_walltime.py](../scripts/recover_part_selected_tigris_walltime.py) |
| New fast candidate | [submit_part_selected_fast_tigris.sh](../sbatch/submit_part_selected_fast_tigris.sh) | [run_part_selected_fast_tigris.sh](../sbatch/run_part_selected_fast_tigris.sh) |

[setup_part_selected_tigris_env.sh](../scripts/setup_part_selected_tigris_env.sh)
is the earlier environment setup helper. The fast launcher does **not** need to
clone/install another environment: it reads `SELECTED_PYTHON` from the reference
campaign's `launch.env` and verifies compatibility.

## 6. How the faster candidate is checked and queued

The complete copy/paste instructions are in the
[fast README](../part_selected_fast/README.md). Its default `--mode benchmark`
queues preparation, four paired profiles, four sustained benchmarks, then a gate.
It does not queue training. `--mode run` explicitly includes the gated training,
locking, evaluation, and reporting stages; `--mode train --resume` can add those
after successful diagnostics.

Defaults are 12 CPU cores, four tree threads, 128 GiB host RAM, cache off,
compilation off, and two concurrent GPU tasks. Training/evaluation reservations
are 168h/12h, not measured runtime promises. Preparation/profile/benchmark limits
are 5h each; gate/lock/report are 2h. Resource choices are frozen per campaign.

Before production, the implementation requires:

1. Exact packed-tree comparison with the authenticated original native backend,
   including different particle counts, ties, padding, and thread counts.
2. All-four-model FP32/FP16 checks of features, logits/loss, per-module gradients,
   BatchNorm state, native trimming, Lookahead, LR-boundary behavior, and a
   serialized restart with a cold cache.
3. Three alternating-order paired trials, each with 20 warmup and 200 measured
   updates, on the same GH200. The conservative reference/candidate timing
   ratio must be at least 1.05. This is an admission threshold, not an observed
   speedup or a confidence interval.
4. The sustained 5,000-update benchmark, full-length checks, 131,072 validation
   events, input-stream fingerprints, AMP-skip checks, host/GPU headroom, and a
   faster complete-job projection including startup. The existing 30% walltime
   margin remains in force.

`--mode review` prints the measurements; it does not replace the production gate.
A failed gate is not bypassed by `--resume`. Changes to cache/thread/compile
settings require a fresh candidate campaign. Existing production runs are never
silently resumed under the optimized implementation.

Local verification recorded on 2026-11-02 was **172 passed, nine skipped** across
the combined selected/fast/Tigris/recovery suite. Eight skips required CUDA and
one required the Linux native build. This guide's documentation update did not
rerun that suite. These results do not establish a GH200 speedup or identical
final model accuracy; consult the new campaign's measured reports for that.

Relevant tests:

- Fast implementation: [model](../tests/test_part_selected_fast_model.py),
  [native](../tests/test_part_selected_fast_native.py),
  [protocol](../tests/test_part_selected_fast_protocol.py),
  [launcher](../tests/test_part_selected_fast_launcher.py).
- Selected science/memory: [screen](../tests/test_part_selected_screen.py),
  [memory](../tests/test_part_selected_memory.py),
  [protocol](../tests/test_part_selected_protocol.py),
  [launcher](../tests/test_part_selected_launcher.py).
- Tigris/recovery: [transfer](../tests/test_part_selected_tigris.py),
  [inputs](../tests/test_part_selected_tigris_inputs.py),
  [launcher](../tests/test_part_selected_tigris_launcher.py),
  [recovery](../tests/test_part_selected_tigris_recovery.py),
  [recovery launcher](../tests/test_part_selected_tigris_recovery_launcher.py).

## 7. Campaign paths, outputs, and storage

The default read-only numerical reference for the fast candidate is:

```text
/home/ryreu/atlas/Fresh_check/checkpoints/part_selected_tigris/selected_gh200_20261023T073044Z
```

The shared and layerwise comparison donors are respectively:

```text
/home/ryreu/atlas/Fresh_check/checkpoints/part_ev_screen/context_gate_ampfix_20261016T215608Z
/home/ryreu/atlas/Fresh_check/checkpoints/part_layerwise_screen/layerwise_20261019T190751Z
```

New fast campaigns normally live under
`checkpoints/part_selected_fast/selected_fast_<UTC timestamp>`. These paths
describe configured references, not a fresh check of remote existence or job
status. Use the submission output and `launch.json` for the actual campaign.

Within a fast campaign:

| Path | What to inspect |
|---|---|
| `source/`, `snapshot.json`, `launch.json`, `experiment_spec.json` | Frozen code, settings, recipe and provenance |
| `inputs/selected.yaml`, `inputs/normalization.json`, `inputs/native_stream.json` | Input schema, fitted constants and native-stream identity |
| `backend/manifest.json`, `backend/parity.json` | New compiled backend identity and exact tree checks |
| `logs/`, `submitted_job_ids.tsv` | Job output/errors and exact submitted IDs |
| `profiles/SELECTED_*.json`, `.md`, `.trace.json` | Paired timing, parity, component traces and reference comparisons |
| `benchmarks/SELECTED_*.json` | Sustained timing, precision/memory results and runtime projections |
| `runs/<RUN_ID>/seed_101/validation/check_*.json` | Per-check validation metrics; absent files mean no recorded check yet |
| `runs/<RUN_ID>/seed_101/best_model_val.pt` | Selected model checkpoint |
| `selection/locked_checkpoints.json` | Checkpoint identities fixed before final evaluation |
| `runs/<RUN_ID>/seed_101/full_validation/` | Full-5M-validation predictions and metrics |
| `reports/selected_tigris_report.json` and `.md` | Final comparison; the fast workflow reuses this filename with its own execution identity |

The original Tigris walltime recovery additionally has its own
`recoveries/<recovery>/` request, source, logs, approval, and job ledger. It is not
the location of fast-candidate profiles. For any queued job, inspect its frozen
`source/`, not just today's mutable repository checkout.

No extra ROOT dataset, persistent tree/pair cache, or environment clone is
introduced by the fast candidate. Persistent additions include the source
snapshot, small input metadata, native build artifacts, diagnostics, benchmark
checkpoints, and normal model/prediction outputs if training/evaluation are
queued. Preparation checks for 6 GiB free as an initial allowance, not a complete
storage reservation. Optional Torch compilation can create its usual compiler
cache; leaving it and the RAM cache off is the simplest initial setup.

## 8. Recommended next use of this guide

For another optimization, start with the measured profile, change one execution
component, and create a new candidate. Keep the four architectures and reusable
recipe fixed, rerun parity/admission, then compare speed and full-screen results.
Near-baseline speed is an objective, not something the current code alone proves.

For a scientific change to relation definitions, encoder widths, trimming, or
training, use a new experiment rather than presenting it as an equivalent speed
optimization.
