# Evolving pair memory: independent SPORC benchmark

Implements the [pair-memory plan](../part_ev_state_compare/EVOLVING_PAIR_MEMORY_PLAN.md).
The model and benchmark are implemented; **GPU feasibility and accuracy are not
established**. This launcher never submits production, changes another campaign,
installs packages, or cancels jobs. The attention-bias and multiplication arms
remain separate.

## Model contract

Start with unchanged noncompact, standard-relations, all-eight layerwise State-EV,
width 32. Add a directed `[batch, receiver, sender, 32]` memory local to one
forward pass:

```text
block 0:       original State-EV; initialize memory from its state features
blocks 1–6:    original State-EV + read incoming memory; gated write for next block
block 7:       original State-EV + read incoming memory; no unused final writer
```

For each writer, `candidate = tanh(W LN(original_state_features) + bias)`.
Writers have independent Xavier weights and zero biases, initialized in isolated
per-layer RNG scopes. Update gates are
`sigmoid(a*candidate + b*memory + d*(candidate*memory) + beta)`; the next memory
is `(1-gate)*memory + gate*candidate`. The four gate vectors are stored as one
`[4,32]` parameter per gate. Initial write fractions are 0.2 for half the channels
and 0.5 for the other half. All four vectors remain learned.

Each reader pools **incoming** memory using the same native, post-dropout
attention weights, then applies its own zero-initialized `[8,16,32]` projection.
The message is `(ordinary_EV + original_State_EV) + memory_readout`, with the
existing output-projection weight applied once, without a second bias.
Native attention, layerwise physics bias, class blocks and original state paths
are retained. Valid self pairs remain; any padding pair is zeroed.

**2,364,294 parameters:** 36,832 more than the 2,327,462-parameter reference
(about 1.58%). Seven writers, six gates and seven readers; no dead first reader
or last writer. This is not a matched-capacity control or a combination with
either other extension.

### Explicit flow and numerical safeguards

The private `PairMemoryTransformer` forward and `particle_block` adapter carry
memory explicitly. They reuse native modules and state-dict names; installed
Weaver is not edited. Construction verifies the pinned Weaver version and source
hash. Native trimming/permutation occurs once, before pair memory exists.

Normalized pairs are shared once per forward. Each block projects endpoints
once and reuses original state features inside pure 64-query chunks. Non-reentrant
checkpointing receives incoming memory and every other activation explicitly.
No hooks supply mutable recurrent memory, no detach, no in-place memory writes,
and no stochastic/BatchNorm operations inside checkpointed chunks.

Writer dense projections, attention and pooling follow the original AMP policy.
New normalization, candidate `tanh`, gate logits/sigmoid, recurrence, and carried
memory use FP32 (FP64 in the independent mathematical oracle). Explicit checks
reject nonfinite preactivations even if bounded nonlinearities would hide them.
One scalar health check is synchronized per model forward, not per chunk.
TF32/BF16/compilation are not enabled for cluster execution.

Zero memory readers recover the original model, including when the existing
State-EV output weights are nonzero. The first successful update trains readers;
writers and gates initially receive zero gradients. Later successful updates
must reach every writer/gate, including the first writer through the final
reader. No branch is skipped because its reader happens to be zero.

## Queue on sporcsubmit

Commit/push these new files and documentation, then pull on RC. Do not modify
any existing campaign's `source/` directory. Each new job uses a separate frozen
snapshot of committed `HEAD`, with its own config, environment-variable prefix,
checksums, logs, and output directory. Later pulls cannot alter that snapshot.

```bash
cd /home/ryreu/atlas/Fresh_check
git pull --ff-only

MEM_ROOT="/home/ryreu/atlas/Fresh_check/checkpoints/part_ev_pair_memory/memory_$(date -u +%Y%m%dT%H%M%SZ)"
MEM_REFERENCE="/home/ryreu/atlas/Fresh_check/checkpoints/part_ev_layerwise_ddp/layerwise_2g_20261029T162243Z"

bash sbatch/submit_part_ev_pair_memory.sh \
  --mode prepare --output "${MEM_ROOT}" --reference "${MEM_REFERENCE}" \
  --partition tier3 --dry-run

bash sbatch/submit_part_ev_pair_memory.sh \
  --mode prepare --output "${MEM_ROOT}" --reference "${MEM_REFERENCE}" \
  --partition tier3

squeue --me -o "%.18i %.30j %.12T %.12l %R"
```

Defaults: account `reu-aisocial`, existing Python
`/home/ryreu/miniconda3/envs/part_ev_screen_sporc/bin/python`. Optional `--python`,
`--account`, `--partition`, and `--reference` overrides are frozen at submission.
The requested partition must be UP and allow the walltime; Slurm checks account
and QoS access. No automatic routing to another partition.

| Job | Resources | Walltime | Dependency |
| --- | --- | --- | --- |
| `evmem_prepare` | CPU only, 8 CPUs, 64 GiB RAM | 5 hours | None |
| `evmem_benchmark` | 2 same-node A100s, 8 CPUs, 64 GiB RAM | 5 hours 30 minutes | Successful preparation |
| `evmem_summary` | CPU only, 2 CPUs, 4 GiB RAM | 15 minutes | Benchmark terminates |

All checks request less than six hours. The GPU benchmark has a five-hour
internal budget. Failures, timeouts, missing evidence, or an invalid dependency
do not authorize training. There is no production or segmented-training mode.

## What must pass on the GPUs

- Fixed-warmup initial equivalence in eval FP32, train FP32 and train FP16, in
  both execution orders, with identical old weights and RNG. Active-original-
  state zero-reader output/gradient parity is checked separately.
- Independent dense FP64 recurrence forward/backward oracle across all eight
  blocks; actual history dependence while later inputs are fixed; final-reader
  gradient back to the first writer.
- Active chunked/checkpointed versus unchunked gradient checks across the
  64-query boundary, unchanged same-input attention/ordinary EV, and active
  two-rank SyncBatchNorm/DDP versus a single global-batch reference.
- Reader-first then writer/gate learning over three successful Ranger updates
  (at most 20 attempts for AMP calibration). Skipped attempts do not count.
- Fresh disposable real-data model: 20 warmup, 4,200 measured updates, repeated
  fetch windows; then 10 warmup and 40 all-valid 128-particle updates. Global
  batch 512, local 256, one training worker per rank, unchanged Ranger.
- FP16 activation audit; finite loss/model/optimizer; AMP skips and matching
  rank parameters/scaler/Lookahead counters; both A100s must retain at least
  10% allocated-memory headroom. Reserved memory is reported too.
- Atomic full model, optimizer/Lookahead, scaler and per-rank RNG/trimmer
  checkpoint roundtrip; seven continuation updates crossing a Lookahead
  boundary. This retained-batch probe checks exact stochastic replay state,
  **not production-scale data-cursor replay time**.
- Rank-zero FP32 validation on 128k events at batch 512, plus fixed-128 FP32
  throughput at batch 256 and latency at batch one. Inference excludes loading;
  validation includes loading/metric overhead. All peaks include these phases.
- Fixed small no-grad diagnostics: write fractions/saturation, memory and
  candidate RMS, update RMS, history/current-state norm ratio, writer/reader
  gradient norms. They restore buffers, modes, RNG and trimmer state and do not
  advance the training stream or save pair tensors.

Gate-fraction diagnostics apply to blocks 1–6; their zero placeholders in blocks
0 and 7 mean no learned gate, not a learned decision to retain all history.

The local suite also tests masks, both-axis permutation, directed/self pairs,
one-particle and rejected-empty inputs, gate endpoints, overflow detection,
exceptions, multiple outstanding forwards, train/eval interleaving, and real
two-process Gloo recovery. CPU results cannot establish CUDA fit or speed.
Tolerances and the model are frozen before submission, not relaxed on failure.

## Resource risk and storage

At local batch 256 and length 128, **one FP32 memory is 0.5 GiB**. Seven training
boundaries alone total 3.5 GiB, before their gradients, the original model's
activations, temporary copies and workspaces. Query chunking does not remove
this quadratic carry. The benchmark may fail the memory gate; that is useful
evidence, not permission to detach history or silently shrink width/batch.
Smaller chunks can be an explicitly reviewed same-formula retry if needed.

Inference retains current/next memory, not a list of layer histories. Extra
parameters are small; training activations are never persisted to disk. Allow
at least 6 GiB free for this benchmark and headroom; it is not a reservation
against other jobs' writes. Preparation rechecks train/validation bytes against
staging receipts, without copying data. The benchmark saves ordinary probe
checkpoints and small JSON/log files, not full-5M predictions. No cleanup is
performed and the 20M independent test population is never opened.

## Review before production

```bash
bash /home/ryreu/atlas/Fresh_check/sbatch/submit_part_ev_pair_memory.sh \
  --mode review --output "${MEM_ROOT}"
```

Evidence is in `benchmark.json`, `summary.json`, `error_rank_*.json`, `logs/`,
and `submitted_job_ids.tsv`. Preserve failed evidence; code changes require a
fresh output/snapshot. Review is read-only; the queued summary writes only its
own summary artifact.

Projections use the slowest measured training window, **all 50 validations**,
checkpoint writes and 30% margin. The intended recipe remains fresh seed 101,
1M update attempts, nominally 512M exposures from the 100M pool, and 1.28M-event
validation every 20k updates. AMP skips reduce successful updates. Eventual
checkpoint selection is highest accuracy, first tie retained; the full-5M
evaluation is overlapping validation, not independent testing.

Queue delay and long restart/data-replay costs are excluded from the estimate.
Twelve days is provisional. A later production launcher should use one long
allocation with resumable full-state saves, not a planned three-day chain.
Benchmark weights never become production weights, and benchmark success
never automatically submits production. Historical control reuse needs its
own runtime/provenance audit; no replacement control is silently trained.

## Local verification

```powershell
$memoryTests = (Get-ChildItem tests/test_part_ev_pair_memory_*.py).FullName
& .\.venv\Scripts\python.exe -B -m pytest @memoryTests -q
```

These tests include CPU mathematics, native-model equivalence, distributed
recovery and fake Slurm. CUDA/AMP/SyncBatchNorm performance and feasibility
remain to be measured on SPORC; no accuracy gain is claimed.
