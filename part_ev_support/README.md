# Support-aware aggregation: independent SPORC benchmark

Implemented 2026-11-16; native-path parity check corrected 2026-11-20.
**Corrected benchmark queue-ready; remote GPU admission remains required.**
No jobs were submitted. Accuracy gains, GPU memory fit and runtime remain to be
measured. Science: [implementation plan](../part_ev_state_compare/SUPPORT_AWARE_AGGREGATION_PLAN.md).

## Model

The complete noncompact, standard-relations, all-eight Layerwise State-EV is
retained. Each particle block receives a small parallel message computed from
its existing directed width-32 state features. Four independent sigmoid gates
pool shared width-16 content over all valid **other** particles. Each channel
returns its weighted mean content, `log1p(sum(gates))`, and
`sum(gates)/number_of_neighbors`. These are soft support counts, not softmax
attention weights. A 72-to-32-to-128 GELU MLP combines the four channels.

| Model | Parameters |
| --- | ---: |
| Baseline ParT | 2,143,486 |
| Full Layerwise State-EV | 2,327,462 |
| Support-aware Layerwise State-EV | **2,384,070** |
| Added support parameters | 56,608 (2.43% over State-EV) |

Each of eight blocks adds 7,076 parameters. Only the new output matrices start
at zero. Gate/content/hidden projections use isolated per-layer Xavier seeds;
gate and hidden biases start at zero. No additional scale, dropout, loss,
learning-rate adjustment, optimizer group, selected relation or compaction is
introduced. The full original State-EV and ordinary EV messages remain active.

Original pre-update pair-state features are computed **once** in each receiver
chunk of 64 and serve both the original and support messages. The support sum
uses a contraction, not a `[batch, receiver, sender, channel, content]` product.
FP32 normalization, sigmoid, sums, counts and summary arithmetic protect the
reductions; small linear layers follow the existing FP16 training policy.
Evaluation is FP32. Non-reentrant checkpointing recomputes pure chunks, not
attention, BatchNorm, dropout or trimming. No new pair caches persist on disk.

Invalid/self pairs are excluded only from the new branch. One-particle jets
receive zero support messages even after hidden biases learn; their parameters
remain graph-connected for DDP. Gate preactivations are checked before sigmoid
can hide an infinity. Existing native attention, head scaling, output projection,
residuals, FFNs and class-attention readout are unchanged.

## Queue on sporcsubmit

Commit and push the implementation locally first. Copy only the commands:

```bash
(
  set -euo pipefail
  cd /home/ryreu/atlas/Fresh_check
  git pull --ff-only

  SUPPORT_ROOT="${PWD}/checkpoints/part_ev_support/support_$(date -u +%Y%m%dT%H%M%SZ)"

  bash sbatch/submit_part_ev_support.sh \
    --mode plan --output "${SUPPORT_ROOT}"

  bash sbatch/submit_part_ev_support.sh \
    --mode benchmark --output "${SUPPORT_ROOT}"

  printf '\nSave this benchmark path: %s\n' "${SUPPORT_ROOT}"
  squeue --me -o '%.18i %.35j %.12T %.12l %R'
)
```

Defaults: tier3, account `reu-aisocial`, existing SPORC interpreter
`/home/ryreu/miniconda3/envs/part_ev_screen_sporc/bin/python`, reference metadata
`/home/ryreu/atlas/Fresh_check/checkpoints/part_ev_layerwise_ddp/layerwise_2g_20261029T162243Z`.
No donor checkpoint weights are loaded. Every benchmark model starts fresh.

| Phase | Walltime | Resources | Dependency |
| --- | --- | --- | --- |
| `evsup_prepare` | 2 h | CPU, 8 CPUs / 64 GiB | None |
| `evsup_benchmark` | 5 h 30 min | **Two same-node A100s**, 8 CPUs / 64 GiB | Successful prepare |
| `evsup_summary` | 15 min | CPU, 2 CPUs / 4 GiB | Benchmark terminates |

All checks stay below six hours. The measurement deadline is five hours;
incomplete evidence fails admission. The model targets the unchanged seed-101
full Ranger recipe: global batch 512, 1,000,000 attempted updates, 50 checks of
1.28M validation events every 20,000 updates. These are not full dataset epochs.
This launcher does **not** expose production training.

`plan` is read-only. Submission verifies current partition limits, resource
requests using `sbatch --test-only`, committed relevant source, donor metadata,
and aggregate storage headroom. Each phase uses the new campaign's authenticated
frozen source, environment, data inventory, model configuration and actual
parameter inventory. Pulling future code will not change existing snapshots.
No existing job, campaign, environment or dataset is edited; the 20M test split
is not opened. No new dependency or compiled backend is required.

Storage accounts for a 2-GiB benchmark output slot, the shared 4-GiB reserve,
outstanding registered production/GH200-extension budgets, readout/triangle
benchmarks and other support benchmarks. This is admission accounting, not a
filesystem quota reservation. Output comprises frozen source, small JSON
reports and disposable restart checkpoints—not large pair caches or predictions.
No automatic deletion or environment installation is performed.

## Required evidence

- FP64 explicit-loop statistics and gradients, floor/zero-neighbor cases,
  constant-gate replication, channel layout and raw-statistic merging.
- Initialized baseline numerical identity; active dense-vs-chunked FP32 and CUDA
  FP16 output/gradient checks, masking, permutation, overflow and native-call checks.
- Native-path audit v2: exactly equal inputs, trimmed inputs, padding masks,
  shared base parameters and post-forward RNG/trimmer state; exactly eight native
  attention calls and one trim. Support output matrices must remain exactly zero
  for this initialization probe. Only computed attention outputs/weights use the
  existing FP32 `rtol=1e-5, atol=1e-6`. Nonfinite activations are rejected and all
  eight block maximum absolute differences are recorded. Both SPORC and Tigris
  admission require this audited evidence. Other numerical/restart gates remain
  unchanged; no model, optimizer, batch, initialization or schedule change.
- Actual two-rank SyncBatchNorm/DDP, an empty-support rank, unequal native lengths,
  and `find_unused_parameters=False`.
- Eight successful Ranger updates verify initial output learning, delayed
  upstream learning and finite movement/gradients in all 48 new parameter tensors.
  No learning-rate or initialization change is made to pass this test.
- Same-allocation fresh baseline State-EV and candidate: each receives 20 warm-up
  and 4,200 measured real-data steps with repeated-fetch windows, then 10 warm-up
  and 40 full-128 forward/backward steps. Initial rank-batch hashes must match.
- Complete-state seven-update restart fixture crossing Lookahead and a validation
  boundary, repeated twice: model, optimizer, scaler, rank RNG, trimmer, attempted/
  successful updates, skips, fixture cursor and accuracy-selected weights/check
  bookkeeping, including resuming again from after the selection boundary.
  This is **not** production ROOT-stream resume certification.
- Candidate 128k-event FP32 streaming validation plus a full-128 validation
  batch of 512. FP32 inference uses length 128, batch 256 and batch one.
- Finite state, per-GPU allocated/reserved peaks with at least 10% allocated
  headroom, checkpoint timing and per-training-process peak RSS. RSS does not
  include the entire job plus data-loader workers.

An isolated first-eight-events validation probe reports gate means, variation,
saturation, soft counts/fractions, denominator floors, channel correlations,
content similarity, count-versus-degree correlation and variation within exact
multiplicity bins. It measures support-message size against original State-EV
and the native attention message. Support-only disabling and count-coordinate
removal are diagnostic interventions, not separately trained ablations.
The probe preserves weights, buffers, gradients, RNG and mode flags; there are
no extra optimizer updates. Channel collapse is reported, not hidden by a gate
that demands favorable scientific results. These disposable weights and eight
events cannot establish gains, classwise rejection, or multiplicity robustness;
those belong to later properly sampled post-training evaluation.

Baseline is measured first, candidate second; ordering and cold-fetch time are
recorded. Cache warming and contention can affect the ratio. Full-training
projection uses conservative measured windows, all 50 validations, checkpoint
writes and 30% margin; queue delays and restart replay remain additional costs.

## Review

Use the exact printed path:

```bash
SUPPORT_ROOT="/home/ryreu/atlas/Fresh_check/checkpoints/part_ev_support/REPLACE_WITH_PRINTED_DIRECTORY"
bash /home/ryreu/atlas/Fresh_check/sbatch/submit_part_ev_support.sh \
  --mode review --output "${SUPPORT_ROOT}"
cat "${SUPPORT_ROOT}/submitted_job_ids.tsv"
```

Review is read-only. Inspect `logs/`, `benchmark.json`, `summary.json`,
`parameter_inventory.json` and `error_rank_*.json`. Slurm `COMPLETED` alone is not
admission; even passing evidence says `authorizes_training: false`.
Production requires a separate implementation and resource review.

No blind retry exists. An ambiguous submission keeps `submission_intent.json`;
inspect that, the job ledger and Slurm before retrying. Never alter an existing
snapshot or bypass failed environment, numerical, memory or restart checks.

## Retry the November 20 native-path failure on both clusters

Benchmark `22114967` / summary `22114968`, in
`support_20261117T033403Z`, failed an activation comparison with zero tolerance.
Its reported maximum difference was `1.7881393432617188e-7`. The correction uses
the already-declared FP32 tolerances and adds exact structural/RNG/weight audits;
it does not declare the failed benchmark passed or waive subsequent GPU checks.

1. Commit/push the correction, then pull on SPORC. Use the queue commands above
   with a **new** `SUPPORT_ROOT`. This submits a fresh prepare/benchmark/summary.
   Leave the failed source, hashes, reports and existing training jobs untouched.
2. Save that exact new path. After its **prepare** job succeeds, it can also serve
   as the Tigris donor; no need to wait for the SPORC GPU benchmark. The prepared
   directory must stay present and unchanged for both campaigns.
3. On Tigris, pull the correction and set `SUPPORT_ROOT` to the new SPORC path
   (shell variables do not transfer between login sessions). Then run:

```bash
(
  set -euo pipefail
  cd /home/ryreu/atlas/Fresh_check
  git pull --ff-only
  : "${SUPPORT_ROOT:?Set SUPPORT_ROOT to the NEW prepared SPORC support campaign}"
  test -f "${SUPPORT_ROOT}/campaign.json"

  SUPPORT_GH_ROOT="${PWD}/checkpoints/part_ev_rts_tigris/support_parityfix_$(date -u +%Y%m%dT%H%M%SZ)"
  bash sbatch/submit_part_ev_rts_tigris.sh \
    --mode plan --model support --benchmark "${SUPPORT_ROOT}" \
    --output "${SUPPORT_GH_ROOT}"
  bash sbatch/submit_part_ev_rts_tigris.sh \
    --mode pipeline --model support --benchmark "${SUPPORT_ROOT}" \
    --output "${SUPPORT_GH_ROOT}" --reviewed

  printf '\nNew Tigris support pipeline: %s\n' "${SUPPORT_GH_ROOT}"
  squeue --me -o '%.18i %.40j %.12T %.12l %R'
)
```

The Tigris command explicitly authorizes benchmark-gated automatic production,
using one GH200 per GPU job. New submissions reject donors predating audit v2.
Merely pulling code cannot fix a queued snapshot, and pointing a new Tigris
pipeline at the old SPORC donor still imports the old test. Do not edit either
snapshot in place. Inspect any old support pipeline's ledger/status and cancel
only its obsolete jobs if it is still queued; leave readout, triangles and the
running bias/multiply/memory chains alone.

To restore automatic **SPORC** production after the new benchmark, attach a new
[benchmark follow-up](../part_ev_benchmark_followup/README.md) to the new benchmark
path/summary job, with new follow-up and production output paths. The old
controller's dependency does not automatically move to the replacement summary.
The benchmark launcher itself remains benchmark-only.

## Local tests

```text
.venv/Scripts/python.exe -B -m pytest -q tests/test_part_ev_support_aggregation.py tests/test_part_ev_support_model.py tests/test_part_ev_support_benchmark.py tests/test_part_ev_support_protocol.py tests/test_part_ev_support_launcher.py
```

CPU mathematical/model tests, actual two-process Gloo and restart tests, and
fake-Slurm isolation tests are local evidence only. GPU admission and any
classification improvement remain unmeasured.
