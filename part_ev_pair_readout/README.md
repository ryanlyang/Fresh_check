# Direct pair-to-classifier readout: independent SPORC benchmark

Implemented 2026-11-16. **Benchmark queue-ready; GPU evidence and accuracy gains
are not yet established.** No jobs have been submitted by this implementation.
There is intentionally no production mode.

Science: [detailed design](../part_ev_state_compare/DIRECT_PAIR_CLASSIFIER_READOUT_PLAN.md).

## What is being tested

The full, noncompact, standard-relations, all-eight Layerwise State-EV backbone
is retained, including its original particle and two class-attention blocks.
Two learned queries pool the final block's directed pair-state features into
two width-32 summaries. A small residual fusion MLP makes those summaries
available to the original classifier. Only the new readout excludes diagonal
pairs; the original State-EV messages are unchanged.

| Model | Parameters |
| --- | ---: |
| Original Layerwise State-EV | 2,327,462 |
| With direct pair readout | 2,349,702 |
| Addition | 22,240 (0.956%) |

This is a new information path, not a replacement for the original classifier.
The fusion output starts at zero, while the queries, key/value projections and
hidden fusion layer start nonzero. The initial model function and all base
parameters are preserved. Once the output learns, the new path can send
classification gradients directly to the final pair-state features.

No selected relations, compaction, triangle branch, support branch, pair memory,
new attention-bias mechanism, auxiliary loss, or pretrained initialization is
included. The locked design has not been tuned using validation results.

## Queue on sporcsubmit

First commit and push these changes locally, then pull them on SPORC. The
launcher rejects dirty relevant source and snapshots the committed repository.
Existing queued campaigns use their own sources; this launcher does not change
their files, dependencies, environments or jobs.

Copy only the commands inside this block, not Markdown fences or prompt text:

```bash
(
  set -euo pipefail
  cd /home/ryreu/atlas/Fresh_check
  git pull --ff-only

  READOUT_ROOT="${PWD}/checkpoints/part_ev_pair_readout/readout_$(date -u +%Y%m%dT%H%M%SZ)"

  bash sbatch/submit_part_ev_pair_readout.sh \
    --mode plan --output "${READOUT_ROOT}"

  bash sbatch/submit_part_ev_pair_readout.sh \
    --mode benchmark --output "${READOUT_ROOT}"

  printf '\nSave this benchmark path: %s\n' "${READOUT_ROOT}"
  squeue --me -o '%.18i %.35j %.12T %.12l %R'
)
```

Defaults (override only deliberately with `--reference` / `--python`):

- Reference metadata: `/home/ryreu/atlas/Fresh_check/checkpoints/part_ev_layerwise_ddp/layerwise_2g_20261029T162243Z`.
- Interpreter: `/home/ryreu/miniconda3/envs/part_ev_screen_sporc/bin/python`.
- Partition/account: `tier3` / `reu-aisocial`. No automatic partition substitution.
- Fresh seed 101; full `ranger_full_v1` projection: 1,000,000 update attempts,
  global batch 512, 50 validations of 1.28M events at 20,000-update intervals.
- No reference checkpoint is loaded. Only its authenticated recipe, runtime,
  environment and train/validation inventory are reused.

`plan` is read-only. Actual submission checks live partition limits, runs
`sbatch --test-only` for all resource requests, and checks aggregate storage.
It never installs packages or deletes artifacts to pass those checks.

## The three-job chain

| Phase | Walltime limit | Resources | Dependency |
| --- | --- | --- | --- |
| `evread_prepare` | 2 h | CPU, 8 CPUs / 64 GiB | None |
| `evread_benchmark` | 5 h 30 min | **Two same-node A100s**, 8 CPUs / 64 GiB | Successful prepare |
| `evread_summary` | 15 min | CPU, 2 CPUs / 4 GiB | Benchmark terminates |

All preliminary jobs are below six hours. No cross-model or production
dependencies are added. A five-hour internal measurement deadline and an early
Slurm signal leave shutdown time. Interrupted/incomplete measurements fail
admission; a completed summary job is not itself proof of success.

Preparation rehashes the existing 100M training / 5M validation inventory,
checks pinned Weaver/preprocessing/environment identities and writes only this
new campaign. It never reads the independent test split.

Submission reserves a conservative **2-GiB output slot**, plus the shared
4-GiB reserve and outstanding registered extension/production budgets.
Existing files in each slot are accounted for. This is a guard, not a quota
reservation: unrelated concurrent writers can still consume disk space.
Snapshots, compact JSON evidence, and one overwritten disposable restart
checkpoint are saved; pair activations are never saved to disk.

## What the benchmark measures and must pass

1. FP64 dense-softmax oracle, including gradients, tied/extreme scores, empty
   pair sets, and global rather than chunk-local normalization.
2. Original-function identity with active original State-EV messages;
   active readout chunk/checkpoint forward and gradient parity in FP32 and FP16.
3. Two-rank SyncBatchNorm/DDP comparison, a rank with no readout pairs, and
   unequal native trimmed lengths, with `find_unused_parameters=False`.
4. Correct first-output-then-upstream learning over three successful Ranger/AMP
   updates. Skipped scaler-calibration attempts are counted separately.
5. **Both unchanged State-EV and the candidate**, separately initialized, on
   this allocation: 20 warm-up + 4,200 measured real-data training steps each,
   repeated-fetch windows, plus full-128-particle forward/backward checks.
6. Full-state restart smoke test: model, optimizer/Lookahead, scaler, rank RNG,
   trimmer, update/validation counters, and deterministic fixture cursor;
   seven updates cross Lookahead and validation boundaries. Repeatability and
   resumed trajectories are both checked. This bounded fixture test does
   **not** certify a future production ROOT-stream replay engine.
7. Candidate FP32 streaming validation (128,000 events), explicit worst-length
   validation batch 512, FP32 inference batch 256 / 128 particles, batch-one
   latency, and per-GPU allocated/reserved peaks with a 90% allocation guard.
8. Small isolated eval diagnostics: query entropy, maximum pair weight,
   effective support, key/value/summary RMS, query norms, summary/weight
   similarity and correction size. Disabling, swapping or zeroing summaries
   gives descriptive logit-sensitivity probes, **not** trained-control accuracy
   results. The seven new parameters' gradient norms are recorded from an
   already-required optimizer step, without diagnostic backward passes.

Normal forward passes do not retain diagnostic tensors or summaries on modules.
Only an explicit small no-grad eval call may materialize dense diagnostic
weights; it restores modes and RNG and leaves BN buffers and gradients intact.
Training pools and chunk merges stay FP32 even inside AMP. A nonfinite valid
readout produces nonfinite logits so the existing coordinated finite-loss check
rejects before backward; it is never silently clipped into a valid prediction.

The baseline is measured first, then the candidate, on separate deterministic
loaders with matched initial rank-batch hashes. Cold fetch times and ordering
are saved. Filesystem cache warming/contention can affect wall time, so these
ratios are measurements from this allocation, not universal speed guarantees.

The summary projects all 50 full training-time validations and checkpoint I/O
with 30% margin. It excludes queue delay and restart replay. Model performance
cannot be inferred from disposable benchmark losses.

## Review and failure handling

The 2026-11-18 retry fixes the native particle-path diagnostic: computed
attention outputs/weights use the already-declared FP32 tolerances
(`rtol=1e-5`, `atol=1e-6`), not bitwise equality. Raw inputs, trimmed inputs,
padding masks, shared parameter values, RNG/trimmer state and native call
counts remain exact checks. Per-block absolute differences are reported;
nonfinite values, larger discrepancies and all existing gradient/AMP/restart
gates still fail. This changes benchmark checks, not the model or recipe.

After committing/pushing/pulling the fix, use the queue block above with a
**fresh output directory**. Do not edit or reuse the failed campaign's snapshot.
Existing queued jobs retain their frozen code and resource requests.

Use the exact path printed on submission:

```bash
READOUT_ROOT="/home/ryreu/atlas/Fresh_check/checkpoints/part_ev_pair_readout/REPLACE_WITH_PRINTED_DIRECTORY"

bash /home/ryreu/atlas/Fresh_check/sbatch/submit_part_ev_pair_readout.sh \
  --mode review --output "${READOUT_ROOT}"

cat "${READOUT_ROOT}/submitted_job_ids.tsv"
```

Review verifies frozen evidence; it submits nothing. Inspect `logs/`,
`benchmark.json`, `summary.json`, and any `error_rank_*.json` on failure. A pass
still reports `authorizes_training: false`: production needs a separate
implementation/admission after resource review.

There is no blind resume/retry mode. If submission is interrupted or Slurm's
reply is ambiguous, inspect `submission_intent.json`, the job ledger and queue
before doing anything else. Do not create a duplicate campaign just because
the terminal did not print all three IDs. A source/science/runtime change
requires a fresh independent benchmark, never editing its frozen snapshot.

## Local verification

```text
.venv/Scripts/python.exe -B -m pytest -q tests/test_part_ev_pair_readout_pooling.py tests/test_part_ev_pair_readout_model.py tests/test_part_ev_pair_readout_protocol.py tests/test_part_ev_pair_readout_launcher.py tests/test_part_ev_pair_readout_benchmark.py
```

CPU tests cover mathematical/gradient equivalence, zero-initialized and active
models, source isolation, mask edges, permutation/batch behavior, RNG/BN/cache
lifetime, Ranger restart and fake-Slurm submission/failure paths. They are not
a substitute for the queued CUDA FP16, two-A100 DDP, memory and throughput checks.
