# Triangle-aware updates: independent SPORC benchmark

Implemented 2026-11-16. **Benchmark queue-ready, not GPU-admitted or production-ready.**
No jobs have been submitted by this implementation. Accuracy gains and actual
GPU memory/runtime remain to be measured.

Science: [triangle-aware implementation plan](../part_ev_state_compare/TRIANGLE_AWARE_UPDATES_PLAN.md).

## Model

The full standard-relations, noncompact, all-eight Layerwise State-EV remains
intact. Only the last two particle blocks receive an additional width-16
triangle message. For receiver `i` and sender `j`, it combines directed
`i-k` and `k-j` current-state pair features; the target `i-j` gates the result.
The original post-dropout attention weights pool those messages. Geometry is
built once after native trimming, while each of the two blocks builds its own
current-state factors. No cross-layer pair memory is introduced.

| Model | Parameters |
| --- | ---: |
| Full Layerwise State-EV | 2,327,462 |
| Triangle-aware Layerwise State-EV | 2,334,662 |
| Addition | 7,200 (0.309%) |

Eight nearest-neighbor **units** are used. Exact FP32 cutoff ties share the
remaining weight across **all** tied particles, never an index-dependent subset.
The intermediary cannot be either endpoint; the mean divides by the actual
remaining fractional support, with no backfill. One/two-particle jets get a
zero triangle message while retaining connected gradients for DDP.

The two new head-output tensors start at zero; all factor/gate matrices use
isolated per-layer Xavier initialization. Base parameters, RNG consumption,
original messages, bias heads, native residual/head scaling and classifier are
preserved. No extra loss, dropout, pretraining, selected relations, readout,
compaction, support-aware branch, or training-recipe change is included.

## Execution and limits

- Stage A: checkpointed receiver chunks of 64 reuse the original width-32
  state feature once, returning its original State-EV message plus three
  width-16 FP32 factor caches. The full width-32 state tensor is not retained.
- Stage B: flat pair-index gathers, receiver tiles 16, sender tiles 64,
  intermediary tiles 4. Rows with more than eight contributors are packed
  into a separate fallback with at most 256 receivers per tile.
- FP32 normalization, bounded factors, products, reductions and attention
  pooling; AMP applies to the small projections. Valid preactivations are
  checked **before** tanh/sigmoid can hide infinities.
- Non-reentrant checkpointing does not rerun MHA, graph construction, trimming,
  BatchNorm or dropout. No triangle tensors persist on modules or disk.
- The default is `sparse_tiled`. A small `dense_oracle` backend exists solely
  for tests, not as an automatic runtime fallback or different scientific arm.

Ties can make support exceed eight particles, so worst-case computation is
not strictly eight-neighbor cost. At local batch 256 / length 128, each factor
cache is 256 MiB (three per triangle block). An ordinary gather tile is at
most 64 MiB and a packed tie gather tile 4 MiB. These are individual tensor
sizes, **not whole-model peak-memory estimates**. Backward, existing attention
and optimizer allocations must fit too; the GPU benchmark decides admission.

## Queue on sporcsubmit

Commit and push the implementation locally first. Copy only the commands,
not the Markdown fences or terminal prompt:

```bash
(
  set -euo pipefail
  cd /home/ryreu/atlas/Fresh_check
  git pull --ff-only

  TRIANGLE_ROOT="${PWD}/checkpoints/part_ev_triangles/triangles_$(date -u +%Y%m%dT%H%M%SZ)"

  bash sbatch/submit_part_ev_triangles.sh \
    --mode plan --output "${TRIANGLE_ROOT}"

  bash sbatch/submit_part_ev_triangles.sh \
    --mode benchmark --output "${TRIANGLE_ROOT}"

  printf '\nSave this benchmark path: %s\n' "${TRIANGLE_ROOT}"
  squeue --me -o '%.18i %.35j %.12T %.12l %R'
)
```

Defaults: `tier3`, account `reu-aisocial`, existing interpreter
`/home/ryreu/miniconda3/envs/part_ev_screen_sporc/bin/python`, reference metadata
`/home/ryreu/atlas/Fresh_check/checkpoints/part_ev_layerwise_ddp/layerwise_2g_20261029T162243Z`.
Override `--python` / `--reference` only deliberately. No reference weights are
loaded. Fresh seed 101, original full Ranger recipe, global batch 512:
1,000,000 attempted updates and 50 validation checks, not 50 dataset passes.

| Phase | Walltime limit | Resources | Dependency |
| --- | --- | --- | --- |
| `evtri_prepare` | 2 h | CPU, 8 CPUs / 64 GiB | None |
| `evtri_benchmark` | 5 h 30 min | **Two same-node A100s**, 8 CPUs / 64 GiB | Successful prepare |
| `evtri_summary` | 15 min | CPU, 2 CPUs / 4 GiB | Benchmark terminates |

All checks stay below six hours. The measurement deadline is five hours;
incomplete evidence fails admission. No production or cross-model dependencies
exist. No other campaign, source snapshot, environment, dataset or job is changed.

`plan` is read-only. Submission checks live partition limits, all resource
requests through `sbatch --test-only`, relevant committed source and aggregate
storage headroom. It freezes the committed repository and authenticates its
inventory on every phase. Preparation verifies the existing train/validation
inventory and pinned runtime. The independent test split is never opened.

A 2-GiB benchmark output slot is accounted for alongside the shared 4-GiB
reserve, outstanding registered production/extension and pair-readout budgets,
and other triangle benchmarks. This is a conservative guard, not a filesystem
quota reservation. No cleanup or package installation is performed. Output is
source snapshot, small JSON reports and a disposable restart checkpoint, not
pair caches or profiler traces.

## Required benchmark evidence

1. FP64 explicit-loop, dense and sparse directed-triangle output/gradient
   agreement, including ties, endpoint exclusion and fractional denominators.
2. Zero-output baseline identity, active sparse/dense forward and all-parameter
   gradient parity in FP32 and CUDA FP16, native call counts and post-trim geometry.
3. Actual two-rank SyncBatchNorm/DDP, an empty-triangle rank, a tie-only rank,
   unequal trimmed lengths, and `find_unused_parameters=False`.
4. Initial output signal, initially zero upstream gradients, then finite
   nonzero gradients and movement in **every** new tensor over eight successful
   Ranger updates. AMP skips are tracked separately. Eight crosses RAdam's
   early rectification and first Lookahead boundary: tiny early upstream
   updates can be below a FP32 matrix element's resolution. No LR or initial
   weight changes are made to make this test pass.
5. Fresh baseline State-EV and candidate on this same allocation: each gets
   20 warm-up + 4,200 measured real-data steps, repeated-fetch windows and
   10 warm-up + 40 full-128 forward/backward steps.
6. Candidate adversarial all-tied length-128 training at local batch 256:
   2 warm-up + 3 measured updates. Also explicit ordinary and all-tied FP32
   validation batches of 512. No silently smaller stress-test batch.
7. A complete-state seven-update restart fixture crossing Lookahead and a
   validation boundary, repeated from saved state twice. Model, optimizer,
   scaler, rank RNG, trimmer and fixture cursor are checked. This does **not**
   certify a future production ROOT-stream replay engine or promise bitwise
   CUDA reproducibility; declared parity tolerances remain enforced.
8. FP32 streaming validation of 128,000 events; FP32 inference batch 256 /
   length 128 and batch-one latency. Peak allocated/reserved GPU memory must
   leave at least 10% allocated-memory headroom on both matching A100s.
9. Real training/validation support histograms, tie fractions and support
   percentiles; all-tie fixture confirms 127 contributors and weight mass 8.
   Small active probes report gate/factor saturation, branch norms, disabling
   sensitivity, zero-denominator fraction and all ten new gradient norms.
10. Aggregate profiler records for geometry/packing, state/factor projection,
    ordinary gather/mixing, tie fallback and output projection. Train and
    validation profiles run separately from sustained timings. Nested scopes
    are inclusive; tile/factor scopes include checkpoint recomputation and
    must not simply be summed with their parent scopes. Host RSS is the peak
    per training process, **not** a measurement of the entire job plus workers.

Baseline is measured first and candidate second with separate deterministic
loaders and matching initial rank-batch hashes. Cold fetch times and ordering
are saved; filesystem warming/contention can affect the ratio. Adversarial
tie throughput is a stress result, not a representative physics throughput.
No speed or accuracy benefit is claimed from these disposable weights.

Full-training projection includes all 50 checks of 1.28M validation events and
checkpoint writes, plus 30% margin; queue delays and restart replay are excluded.

## Review

Use the exact printed path:

```bash
TRIANGLE_ROOT="/home/ryreu/atlas/Fresh_check/checkpoints/part_ev_triangles/REPLACE_WITH_PRINTED_DIRECTORY"
bash /home/ryreu/atlas/Fresh_check/sbatch/submit_part_ev_triangles.sh \
  --mode review --output "${TRIANGLE_ROOT}"
cat "${TRIANGLE_ROOT}/submitted_job_ids.tsv"
```

Review is read-only. Inspect `logs/`, `benchmark.json`, `summary.json` and
`error_rank_*.json` for failures. A passed benchmark still says
`authorizes_training: false`; production follows a separate implementation and
resource review. A changed backend or science definition requires new evidence.

No blind retry exists. An ambiguous submission retains `submission_intent.json`;
inspect it, the ledger and scheduler before retrying. Never edit frozen source
or bypass a failed correctness, environment, memory or restart gate.

## Local tests

```text
.venv/Scripts/python.exe -B -m pytest -q tests/test_part_ev_triangles_math.py tests/test_part_ev_triangles_model.py tests/test_part_ev_triangles_benchmark.py tests/test_part_ev_triangles_protocol.py tests/test_part_ev_triangles_launcher.py
```

CPU tests include real two-process Gloo and optimizer restart checks, independent
mathematical oracles, active gradients, cache/RNG/BN lifetime and fake-Slurm
isolation/failure tests. They do not establish CUDA FP16/DDP compatibility,
40-GiB fit, throughput, or classification gains; those remain benchmark/training work.
