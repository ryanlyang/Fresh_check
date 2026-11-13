# Relation-state multiplication: independent SPORC benchmark

Implements the [relation-state multiplication plan](../part_ev_state_compare/RELATION_STATE_MULTIPLICATION_PLAN.md).
The model and benchmark are implemented. This launcher is **benchmark-only**:
it does not submit production training, modify another campaign, install packages,
or change the queued attention-bias experiment.

For a completed, passed benchmark, use the separate
[production launcher and queue instructions](../part_ev_state_production/README.md).
The benchmark launcher itself remains unchanged and benchmark-only.

## Model and scientific contract

The starting point is noncompact, standard-relations, all-eight layerwise
State-EV with width 32. Only its state-message modules are extended:

```text
g = E LN(z_ij)
p = P LN(x_i)                 # receiver state
q = Q LN(x_j)                 # sender state
h = g + p + q + p*q           # original operation order
gamma = tanh(A p + C q)
r_new = GELU(LN(h + g*gamma))
```

Each block has independent bias-free `A` and `C`, both `32 x 32`, initialized
to zero without consuming RNG. The relation channel coefficient is `1+gamma`,
bounded between zero and two. The head-specific state-output projections,
ordinary EV, native attention, layerwise physics bias, and dropout are unchanged.
This is **not** a combination with the attention-bias extension or pair memory.

**2,343,846 parameters**, an addition of 16,384 (about 0.70%) to the reference's
2,327,462. No selected relations, private encoder, compact path, or extra exposure.

Endpoint projections `A p` and `C q` run once per particle, not once per pair.
The model retains the forward-local normalized-pair cache and checkpointed
query chunks of 64. Chunk backward uses explicit tensors, not mutable cache state.
No full gate tensors are logged or cached on disk.

Numerical contract: preserve the native AMP operation order for the original
`h`; evaluate the two new per-particle projections, `tanh`, product and residual
addition in FP32 (FP64 in mathematical tests), followed by the existing FP32
LayerNorm/GELU and unchanged aggregation AMP policy. At zero modulation, the
already-computed original `h` is simply cast to its usual normalization precision.

### Expected delayed activation

The existing State-EV output weights also start at zero. Thus the first successful
update should train those output weights while both new gate gradients remain
zero. Subsequent successful updates should activate the gates. The benchmark
checks **three successful Ranger updates**, allowing at most 20 attempts for AMP
calibration. Skips do not count as successful updates. Missing activation fails
the check; it does not trigger tuning, a bypass, or a changed initialization.

## Queue from sporcsubmit

Commit/push the new package, driver, launchers, tests and plan updates, then pull
on RC. Do not edit any existing campaign's `source/` directory. Every new job
executes a frozen snapshot of committed `HEAD`; subsequent repository pulls cannot
change its model or worker. No dependency on the attention-bias campaign is added.

```bash
cd /home/ryreu/atlas/Fresh_check
git pull --ff-only

MUL_ROOT="/home/ryreu/atlas/Fresh_check/checkpoints/part_ev_state_multiply/state_multiply_$(date -u +%Y%m%dT%H%M%SZ)"
MUL_REFERENCE="/home/ryreu/atlas/Fresh_check/checkpoints/part_ev_layerwise_ddp/layerwise_2g_20261029T162243Z"

bash sbatch/submit_part_ev_state_multiply.sh \
  --mode prepare --output "${MUL_ROOT}" --reference "${MUL_REFERENCE}" \
  --partition tier3 --dry-run

bash sbatch/submit_part_ev_state_multiply.sh \
  --mode prepare --output "${MUL_ROOT}" --reference "${MUL_REFERENCE}" \
  --partition tier3

squeue --me -o "%.18i %.30j %.12T %.12l %R"
```

The default Python is
`/home/ryreu/miniconda3/envs/part_ev_screen_sporc/bin/python`, account
`reu-aisocial`, partition `tier3`. `--python`, `--account` and `--partition`
overrides are recorded in the frozen launch contract. No environment changes.
The exact requested partition must be UP and allow the benchmark walltime;
Slurm validates account/QoS access. There is no silent rerouting.

| Job | Resources | Walltime | Dependency |
| --- | --- | --- | --- |
| `evmul_prepare` | CPU only, 8 CPUs, 64 GiB RAM | 5 hours | None |
| `evmul_benchmark` | 2 same-node A100s, 8 CPUs, 64 GiB RAM | 5 hours 30 minutes | Successful preparation |
| `evmul_summary` | CPU only, 2 CPUs, 4 GiB RAM | 15 minutes | Benchmark terminates |

Each requested check is under six hours. The GPU probe has a five-hour internal
budget plus scheduler allowance. A failure, timeout or incomplete report never
authorizes production. The launcher has no production, cancellation, automatic
resubmission or three-day segment mode.

## Benchmark checks and measurements

- Zero-modulation identity with controlled initial weights, RNG and fixed warmup
  in eval FP32, train FP32 and train AMP; trained-like nonzero State-EV output
  weights for the gradient comparison. Existing project tolerances are frozen,
  not relaxed on failure.
- Delayed activation and finite nonzero gradients for every new matrix; active
  chunked/checkpointed versus dense-sized forward/backward checks across the
  64-query boundary. A separate local FP64 test checks an independently written
  dense mathematical formula and its gradients.
- Same-input native attention and ordinary EV invariance, plus active-state
  SyncBatchNorm/DDP versus a global-batch gradient reference.
- Fresh disposable training model, 20 warmup and 4,200 measured real-data updates
  across repeated fetch windows, followed by 10 warmup and 40 fixed-128 updates.
  Global batch 512, local batch 256, one input worker per rank.
- FP16 activation audit, finite model/optimizer state, AMP skip counts, distributed
  parameter/scaler/Lookahead agreement and at least 10% allocated-memory headroom
  on both A100s. Fixed-length and validation peaks are included.
- Atomic full model/optimizer/Lookahead/scaler and per-rank RNG/trimmer checkpoint
  roundtrip, followed by seven updates across a Lookahead boundary. Compare
  losses, full state and final RNG. This model-replay probe uses a retained batch;
  it does not time production-scale data-cursor replay.
- 128k-event rank-zero FP32 validation at batch 512 and fixed-128 FP32 inference.
  Validation includes loading/metric overhead; fixed inference excludes loading.
- A small fixed diagnostic sample: gate mean/spread, saturation, residual-to-
  original-preactivation norm, state-message norm, matrix and gradient norms.
  Diagnostics restore RNG, buffers, trimmer counters and modes and do not advance
  the training stream. A nonzero gate is not evidence of improved classification.

The original full recipe is frozen for projections: fresh seed 101, unchanged
Ranger, 1M update attempts, nominally 512M exposures from the 100M training pool,
and 50 checks of 1.28M streaming-validation events every 20k updates. Highest
accuracy, first tie wins. No benchmark weights become training checkpoints.
The eventual full-5M evaluation overlaps checkpoint-selection validation and is
not independent test evidence. The 20M test population is never opened here.

## Read-only review

```bash
bash /home/ryreu/atlas/Fresh_check/sbatch/submit_part_ev_state_multiply.sh \
  --mode review --output "${MUL_ROOT}"
```

`benchmark.json` contains detailed evidence; `summary.json` contains the review,
with logs under `logs/`, failures in `error_rank_*.json`, and IDs in
`submitted_job_ids.tsv`. Preparation failures are explained by their logs; the
summary reports missing evidence rather than claiming success.

The projection uses the slowest measured training window, all 50 validations,
checkpoint I/O and a declared 30% margin. Queue time and long data-replay costs
are excluded. Twelve days remains provisional. Review these measurements before
implementing/admitting production as one long allocation with resumable saves.
No historical accuracy result is automatically approved as a matched control;
its recipe, runtime and segmented restart history still need disclosure/audit.

Preparation validates the old source/recipe/runtime and current environment,
then rechecks the official train/validation file bytes against staging receipts.
This reads the data without copying it. Interpreter-root package enumeration
avoids treating lazy setuptools vendor imports as newly installed packages.
Keep at least 6 GiB free for benchmark artifacts/headroom; this is not a disk
reservation against concurrently running jobs. Production storage is a later
check. No datasets, old checkpoints or other jobs are removed automatically.

For a code change or failed probe, preserve the evidence and use a new output
directory. The launcher refuses existing outputs and changed frozen files.

## Local verification

```powershell
$multiplyTests = (Get-ChildItem tests/test_part_ev_state_multiply_*.py).FullName
& .\.venv\Scripts\python.exe -B -m pytest @multiplyTests -q
```

CPU mathematical/model tests, two-process Gloo recovery and fake-Slurm tests
are local checks. CUDA/AMP/SyncBatchNorm memory and timing must pass on SPORC.
