# Independent compact State-EV campaign: runs 8 and 9

**Queue-ready stage: benchmarks only, on Tigris.** This does not cancel,
resubmit, modify, or depend on any of the existing seven production runs.
Production training/evaluation submission is a separate next step after review.
No training authorization is inferred from a successful benchmark.

## The two models

Both start fresh with seed **101**, the existing full 100M training pool,
batch 512, the original Ranger optimizer (RAdam + Lookahead), FP16 training
with FP32 parameters, FP32 validation, TF32 disabled, and one GH200. No selected
relations, additional particle features, or different data preprocessing.

| Arm | Attention bias | EV / state | Parameters | Planned updates |
|---|---|---|---:|---:|
| `COMPACT_STATE_LAYERWISE` | Layerwise | Ordinary EV plus compact state in all eight blocks | 2,262,950 | 1,000,000 |
| `PART_MATCH_COMPACT_COMPUTE` | Shared | None | 2,263,662 | Determined by benchmark FLOPs |

The control remains ordinary ParT: only its eight particle FFNs widen from
512 to **570**. Embedding width, attention heads, class blocks, class FFN width
512, and shared pair bias stay native. It has 712 more parameters (+0.031%),
the closest uniform integer FFN width. Baseline ParT has 2,143,486 parameters.

Both planned schedules have **50 validation checks**, each on 1.28M events
from the existing continuous validation stream. The control gets more updates
between checks, not more checkpoint-selection opportunities. Its learning-rate
schedule is stretched to the same relative milestones (70% flat, then the
original 15 decays). Selection remains highest validation accuracy, first tie.
Later full-5M validation overlaps checkpoint selection; it is not an independent
20M test. This launcher does not open the test split.
The compact arm's million updates represent 512M jet exposures; the 50 checks
are reporting intervals, not 50 full passes through the 100M dataset.

## What is compacted, and what could affect accuracy?

Keep ordinary EV and state conditioning in **all eight** blocks. Keep all eight
independent scalar-bias heads and every native backbone operation.

Two deliberate architectural changes need an accuracy experiment:

1. Reduce the state bottleneck from rank 32 to **16**.
2. Share a single pair-only projection `64 -> 16` across the eight blocks.
   Receiver and sender state projections and the state output projections
   remain independent in every block. Sharing the pair projection does **not**
   make the attention bias shared.

For block l, the state features are still

```text
g_ij = shared_pair_projection(LayerNorm(z_ij))
p_i  = receiver_l(LayerNorm(x_i))
q_j  = sender_l(LayerNorm(x_j))
r_ij = GELU(LayerNorm(g_ij + p_i + q_j + p_i*q_j))
state_message_i,h = output_l,h(sum_j attention_i,j,h * r_ij)
```

The state outputs start at zero; receiver/sender and shared pair maps are
Xavier-initialized under a forked CPU RNG. All native initial weights and the
caller RNG stream are preserved.

Execution changes, tested against a slower implementation of that SAME model:

- Normalize/project symmetric pairs once, using the lower triangle, and reuse
  the graph-connected result across all eight blocks.
- Compute state pooling in query chunks of 64, checkpointing those chunks.
  Backward receives explicit tensors; it never reads a cleared forward cache.
- Combine native, ordinary EV and state messages **before** the native output
  projection. One projection replaces two, with its bias added only once.
- Use the exact same post-dropout attention weights for all three messages.
  Receiver/sender conditioning is directional even though the pair stem is
  symmetric. Padding is excluded from the added messages.

No claim that rank 16 or tying weights preserves the trained rank-32 model.
Reassociation changes floating-point rounding, so FP32 and FP16 parity are
checked separately against fixed tolerances, including nonzero-state gradients.
Nothing is quantized, sparsified, pruned, or replaced by selected relations.

## FLOPs and the longer control schedule

The advertised approximately **1.41x** is a **forward dense-operator count at
128 particles**, not a promise of GPU speed or a complete all-operation count.
Locally checked values (multiply-add = two FLOPs):

| Model | Forward dense FLOPs / jet, N=128 |
|---|---:|
| Baseline ParT | 669,519,360 |
| Compact all-eight layerwise State-EV | 942,738,944 |
| Width-570 ordinary ParT control | 699,928,064 |

Budgeting uses **measured forward + backward + checkpoint recomputation**,
not simply three times a guessed forward ratio. At fixed N=128, those measured
training counts are 2,861,771,264 and 2,099,784,192 per jet respectively. That
would give **1,362,850** control updates, or **27,257** updates/check, versus
20,000/check for State-EV. This is illustrative, not the frozen production budget.

Each GH200 benchmark measures dense operator counts at multiple lengths,
checks independent points across chunk boundaries, and records the actual
native-trimmer length histogram over 4,200 training batches. Both arms must
match the donor's five raw-input fingerprints and each other's length histogram.
The summary averages each cost curve over this same observed population, then
rounds the control budget DOWN to a multiple of 50. It stores both planned
recipes and the evidence hashes before any production outcome is available.

Counting includes matrix multiplications and convolutions, including recompute.
It excludes normalization, activation, softmax, optimizer updates, indexing,
data movement and I/O. Validation work and walltime are reported separately.
This is an approximate **dense training-compute match**, not an assertion of
identical total hardware work. Full-run length distributions may differ from
the benchmark; eventual training should log realized compute as well.

## What the benchmarks check

- Frozen source, donor environment, native input/preprocessing sources, and
  the exact 1000 training / 50 validation-file inventory and metadata.
- CPU preparation needs no GPU and builds no selected-relation backend.
- Compact versus slow-oracle execution: FP32 evaluation, FP32 training and
  FP16 training; zero/nonzero state outputs; forward and gradient comparisons.
- Synchronized restart pilot: model, full Ranger/Lookahead state, GradScaler,
  Python/NumPy/Torch/CUDA RNG, trimmer, LR boundaries, and repeat/resume updates.
  This is an instrumented pilot, not a claim of bitwise production reproducibility.
- Full 128-particle, batch-512 memory/AMP probe with ten consecutive successful
  updates after calibration; native-length sustained throughput and skip rate.
- Observed training/validation activation precision, 131,072-event validation
  timing including metrics, and an atomic full-checkpoint write timing.
- Measured FLOPs, 30%-margin walltime projections, and whether each projected
  full run fits Tigris's previously observed 20-day maximum. Slurm validates
  current benchmark resource requests with `--test-only` before submission.

Disposable pilot accuracies are not production results. Benchmark weights are
never selected as production initialization. Summary fails closed if a model,
probe, fingerprint, restart, precision check or timing window is missing.

## Independence, disk space, and existing queued jobs

All implementation files are new. The existing model packages and launchers
are unchanged. Submission archives the committed repository into a **new**
campaign's `source/`, verifies its file inventory/hashes, and uses absolute
worker paths into that snapshot. Every worker verifies it again. Later pulls
in the live checkout cannot change this new snapshot or the old seven-run
snapshots. Do not manually edit their source directories or shared environment.

The old seven-run benchmark supplies **read-only metadata** (dataset inventory,
native stream fingerprints and Tigris environment), not selected features,
trained weights, cache files or a Slurm dependency. After prepare, this campaign
uses its own copied metadata. No data are duplicated or deleted.

Installed-distribution checks use explicit interpreter installation roots, so
optimizer imports adding `setuptools/_vendor` to `sys.path` do not produce the
previous false environment mismatch. The separate core-version and Weaver-source
checks remain enforced. No environment is created, upgraded or modified.

For compatibility, the inventory retains the donor's exact `dict(sorted(...))`
convention for `(lowercase package name, version string)` records. Duplicate
metadata records otherwise make an unsorted dictionary depend on enumeration
order. This fixes the prepare failure in `compact_20261109T233555Z`, where NumPy,
fsspec, Pillow, setuptools and typing_extensions each exposed two versions.
It is a legacy fingerprint convention, not a semantic-version resolver or proof
that duplicate metadata are harmless. The separate core-version/Weaver-source
comparisons are unchanged; no package/version check is bypassed. Do not delete
duplicate metadata or reinstall packages in an environment used by other jobs.

After this fix, retry with a **new output directory** (for example
`compact_inventoryfix_$(date -u +%Y%m%dT%H%M%SZ)`). Keep the failed campaign's
logs and frozen snapshot unchanged. A pull does not repair an existing snapshot.

Require **6 GiB free** at launch/prepare, and 4 GiB remaining at GPU start.
Benchmarks should use well below 1 GiB: one source snapshot, two disposable
restart probes, and bounded logs/JSON. No persistent pair cache, dataset cache,
large profiler traces or population predictions. These checks reserve no disk
space against other jobs. Future production storage must be reviewed separately.

No automatic submission retry: an uncertain `sbatch` response leaves an intent
journal. Inspect it and Slurm before creating another campaign. Reusing an
existing output directory is refused. No `scancel` or edits to other campaigns.

## Queue on Tigris after committing and pushing these new files

```bash
cd /home/ryreu/atlas/Fresh_check
git pull --ff-only

COMPACT_ROOT="/home/ryreu/atlas/Fresh_check/checkpoints/part_ev_compact/compact_$(date -u +%Y%m%dT%H%M%SZ)"

bash sbatch/submit_part_ev_compact_tigris.sh \
  --mode benchmark --output "$COMPACT_ROOT" --dry-run

bash sbatch/submit_part_ev_compact_tigris.sh \
  --mode benchmark --output "$COMPACT_ROOT"

squeue --me -o "%.18i %.30j %.12T %.12l %R"
```

Defaults (override with `--reference` / `--python` if explicitly relocated):

```text
Reference: /home/ryreu/atlas/Fresh_check/checkpoints/part_selected_fullscale/seven_buildfix_20261106T005700Z
Python:    /home/ryreu/miniforge3-aarch64/envs/part_selected_tigris/bin/python
```

Submission requires a clean committed worktree and Tigris/aarch64.
Resources: 12 CPUs / 128 GiB host RAM per job; GPU jobs use one GH200 each.
Both array elements are eligible concurrently (`--concurrency 2`, the default).
Use `--concurrency 1` only if desired for resource or contention reasons.

```text
CPU prepare, 5h
    -> afterok: GPU benchmark [0-1], 5h EACH
         -> afterany: CPU summary, 30min
```

Those are maximum allocations, not projected elapsed durations. Each long
phase has a 4h50m internal deadline; summary has 25 minutes. All checks are below
six hours. Failures produce diagnostics and never authorize training.

After completion, review read-only (the variable needs resetting in a new shell):

```bash
cat "$COMPACT_ROOT/submitted_job_ids.tsv"
bash sbatch/submit_part_ev_compact_tigris.sh \
  --mode review --output "$COMPACT_ROOT"
```

Successful summary: `reports/benchmark_summary.json`. Failures:
`diagnostics/*_failure.json` and `logs/evcompact_*.err`.
Review does not launch jobs or change campaign files. **Do not cancel the seven
existing runs:** these two benchmarks are independent. Use the results to choose
walltimes and implement the separate two-run production launch next.

## Local verification

```text
.venv/Scripts/python.exe -B -m pytest tests/test_part_ev_compact_model.py tests/test_part_ev_compact_protocol.py tests/test_part_ev_compact_prepare.py tests/test_part_ev_compact_launcher.py tests/test_part_ev_compact_review.py -q
```

CPU tests check actual pinned-Weaver model counts, masks, nonzero-state
forward/gradient parity, dropout, permutation invariance, packed/shared
projection reuse, cleanup on exceptions, complete optimizer restart, real
operator FLOPs, budget/schedule arithmetic and fail-closed report review.
Mock-Slurm tests check frozen-checkout independence, source corruption, resource
requests, dependencies, partial submissions and duplicate rejection. Actual
GH200 speed, AMP parity, memory headroom and accuracy retention require RC runs.
