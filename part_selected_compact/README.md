# Independent selected compact State-EV pair: benchmark stage

This is an additional two-model campaign on **Tigris**, separate from the seven
production models, the standard-relations compact pair, and compiler-recovery
jobs. **Only preparation, benchmarks and their summary can be queued here.**
There is no production training/evaluation mode and no automatic training
authorization. A passed benchmark is not evidence of retained accuracy.

## Models and matched comparison

Both models start fresh with seed 101, the official 100M training population,
global batch 512, native single-GPU BatchNorm, the full Ranger recipe (RAdam +
Lookahead), FP16 autocast training/FP32 parameters, FP32 validation and TF32 off.
The unchanged selected input loader transports four raw track channels alongside
the native 17 particle channels; only the native 17 enter the particle embedding.
PT/TRACK/REGION definitions and frozen training-fitted normalization are reused.

| Arm / run ID | Bias | EV/state | Parameters | Planned updates |
|---|---|---|---:|---:|
| `SELECTED_COMPACT_STATE_LAYERWISE` | Layerwise | Ordinary EV + compact state in all eight blocks | 2,272,870 | 1,000,000 |
| `SELECTED_PART_MATCH_COMPACT_COMPUTE` | Shared | None | 2,273,582 | Determined by selected-model benchmark FLOPs |

The control is ordinary **selected-relations ParT**, with only its eight particle
FFN widths increased from 512 to 570. Class-block FFNs stay 512, embedding width
128, and attention head count eight. It has 712 more parameters (about 0.031%);
570 is the closest uniform integer particle-FFN width. It is NOT layerwise,
does not have EV/state branches, and is not initialized from a trained model.

The compact model retains state conditioning in all eight blocks, reduces its
state bottleneck to 16, and ties one `64 -> 16` pair-only projection across
blocks. Receiver/sender and output projections stay independent per block.
These are architectural changes, so classification retention is an experiment,
not a guarantee. The branch outputs start at zero.

### Directed pairs: an essential difference from the standard compact model

Selected relation features are directional: `z_ij` need not equal `z_ji`.
The pair encoder still processes the complete dense directed population, with
the same pair BatchNorm as the existing selected models.

The new cache computes **all N*N directed state-pair projections once per
forward**, sharing their graph across blocks. It never mirrors a triangle.
The cache is released after forward, including exceptional exits; backward
checkpoint recomputation receives explicit tensors, not a mutable cleared cache.

Other compact execution changes are retained: chunked queries of 64, activation
checkpointing, and combining native/ordinary-EV/state messages before the native
output projection (adding its bias once). All messages use the same post-dropout
attention weights. The slow oracle uses the same model parameters, recomputes
directed projections per block and retains separate output projections.

Do not reuse the standard compact model's approximately 1.41x forward-FLOP ratio
or its control update budget. This selected model has different work.

## Compute matching and the future full schedule

The fixed arm gets 1,000,000 attempted updates (512M jet exposures) and 50 checks,
one per 20,000 updates. Each training-time validation check uses 1.28M examples
from the continuous validation stream. Check numbers are not full dataset passes.

Benchmarks measure dense forward + backward + activation-recomputation FLOPs at
lengths 1, 17, 32, 64, 65, 96, 127 and 128. Independent lengths/chunk boundaries
must agree with the fitted curve. N=1 uses its own measured value. Both arms must
have identical native input fingerprints, full selected-input fingerprints and
trimmed-length histograms over 4,200 measured real-data batches.

Average both training cost curves over that same observed length population:

```text
control_updates = floor_to_multiple_of_50(
    1,000,000 * selected_compact_training_cost / selected_control_training_cost)
```

The control also gets **50 checks**, not extra selection opportunities. Its check
spacing and original stepwise learning-rate milestones stretch with its total
budget. The actual recipes, evidence hashes and rounded budget are reported by
review before production results exist. No budget from the standard pair is
substituted. Full-run realized lengths should be accounted for in later production.

The convention counts dense matrix multiplications and convolutions (multiply-add
= two FLOPs), including the learned selected encoders and state recomputation.
It excludes CPU clustering, most descriptor arithmetic, normalization, softmax,
activations, optimizer updates, transfers/I/O and validation. This is an
**approximate dense training-compute match, not equal total hardware work or
walltime**. End-to-end timing includes selected tree work and data loading;
validation/metric computation and checkpoint writes are measured separately and
included in full-runtime projections. Queue delay and restart replay are excluded.

## Backend compatibility and independence

Defaults point to:

- Seven-run benchmark: `part_selected_fullscale/seven_buildfix_20261106T005700Z`.
- Passed recovery: `part_selected_compiler_recovery/compilerfix_omp_20261110T232529Z`.
- Python: `/home/ryreu/miniforge3-aarch64/envs/part_selected_tigris/bin/python`.

CPU preparation authenticates the donor inputs, normalization, numerical source,
dataset inventory and unchanged environment. It authenticates the existing
compiler-recovery approval, its exact original production/donor identity and the
reviewed loader implementation. Preparation and **each GPU worker** recheck the
existing binaries, compiler transition, loaded libraries, ABI, self-test, original
smoke hashes and batched tree parity against that approval. Neither native backend
is rebuilt or replaced. Changed libraries/compiler identity fail admission.

The new source snapshot is created from a clean committed checkout. No existing
package is edited, old manifests/snapshots/ledgers are not rewritten, and old jobs
are not cancelled, resubmitted or used as dependencies. A later `git pull` cannot
change these frozen workers. Do not edit their snapshots or shared environment.

No dataset copy, persistent tree/pair cache, large profiler trace or prediction
array is created. Persistent output is a source snapshot, small JSON/YAML evidence,
logs, and one disposable full restart probe per arm. Preparation requires 6 GiB
free; benchmark work retains a 4 GiB floor. These are checks, not quota reservations.
Summary gives a conservative future-production free-space estimate including
restart files, selected weights and two full-5M outputs. Existing concurrent jobs
can consume free space, so recheck it before future production submission.

## Checks and job graph

```text
CPU prepare (5h maximum)
  -> afterok: GPU benchmark array [0,1] (5h maximum per task)
  -> afterany: CPU summary (30min maximum)
```

Both GPU tasks are independently eligible by default (array concurrency two).
Each requests **one GH200, 12 CPUs and 128 GiB host RAM**. CPU jobs use the same
CPU/RAM request, with no GPU. All preproduction walltimes are below six hours.
Workers stop diagnostics before the allocation deadline; no partial success is
published on timeout, numerical failure, or insufficient memory.

Each GPU benchmark checks:

- Optimized selected model versus slow selected oracle: zero/nonzero state,
  FP32 evaluation/training and FP16 training, outputs, gradients and selected
  family connectivity. Tolerances are frozen before the jobs, not changed to pass.
- Exact synchronized repeat/resume pilot covering model, optimizer/Lookahead,
  scaler, RNG and trimmer state, including original full-schedule LR boundaries.
  This is not a claim of bitwise unsynchronized production determinism.
- Full 128-particle, batch-512 memory/AMP confirmation and sustained real-data
  training windows; no microbatch or schedule substitution to pass the gate.
- Precision audits, validation including metric computation, checkpoint I/O,
  host/GPU memory headroom, input streams and dense FLOP curves.

The summary requires both complete evidence records. It reports runtime estimates
with a 30% planning margin and whether each fits 480 hours, the previously observed
Tigris partition maximum. This threshold is not a live policy guarantee. Scheduler
`--test-only` validates the actual benchmark requests before any job is submitted.

Failed preparation/benchmarks produce diagnostics. A summary can run after failure
to explain missing evidence; it exits nonzero and does not authorize anything.
An existing output path is never silently reused: inspect its submission journals
after an ambiguous scheduler response. A deliberate retry gets a new output and
source snapshot. There is no automatic cancellation or hidden production chain.

## Queue on Tigris

Commit and push these new files locally first. Then, on **Tigris**:

```bash
cd /home/ryreu/atlas/Fresh_check
git pull --ff-only

SELECTED_COMPACT_ROOT="/home/ryreu/atlas/Fresh_check/checkpoints/part_selected_compact/selected_compact_$(date -u +%Y%m%dT%H%M%SZ)"

bash sbatch/submit_part_selected_compact_tigris.sh \
  --mode benchmark --output "${SELECTED_COMPACT_ROOT}" --dry-run

bash sbatch/submit_part_selected_compact_tigris.sh \
  --mode benchmark --output "${SELECTED_COMPACT_ROOT}"

squeue --me -o "%.18i %.30j %.12T %.12l %R"
```

The default donor and recovery paths above must still exist. Optional overrides
are `--reference`, `--recovery`, `--python`, `--account` and `--concurrency 1`.
Do not install packages into the existing environment or alter evidence to bypass
a failure. The freeze step rejects uncommitted/untracked checkout changes.

After the summary finishes (reuse the exact output printed at submission):

```bash
bash sbatch/submit_part_selected_compact_tigris.sh \
  --mode review --output "${SELECTED_COMPACT_ROOT}"
```

Review is read-only and recomputes the paired budget from authenticated evidence.
Check `passed`, both walltime/memory projections, disk estimate, and the resulting
control update count before implementing/admitting production. Benchmark weights
are disposable, never production initialization. No test split is read. Eventual
full-5M validation overlaps checkpoint selection and is not an independent test.

## Local verification

```powershell
.venv\Scripts\python.exe -B -m pytest tests/test_part_selected_compact_model.py tests/test_part_selected_compact_flops.py tests/test_part_selected_compact_protocol.py tests/test_part_selected_compact_benchmark.py tests/test_part_selected_compact_launcher.py -q
```

These tests exercise real CPU models/gradients/restarts/FLOP counts, independent
lengths, directed-cache semantics, frozen evidence, failure paths and a fake
scheduler. They do not replace the pending ARM/GH200/compiled-backend benchmarks.
