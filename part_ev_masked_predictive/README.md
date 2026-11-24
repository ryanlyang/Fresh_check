# Masked predictive training on full Layerwise State-EV

Implemented 2026-11-24 from
[the scientific specification](../part_ev_state_compare/MASKED_PREDICTIVE_TRAINING_PLAN.md).
Independent SPORC and Tigris pipelines are ready for benchmark submission.
Local tests do **not** establish CUDA fit, speed, numerical admission, or an
accuracy gain. Those decisions belong to each platform's actual benchmark.
Nothing here submits jobs on import or changes an existing campaign.

## What runs

Each platform gets a fresh, seed-101 matched pair:

| Arm | Training | Deployed parameters | Trainable parameters | Frozen teacher parameters |
| --- | --- | ---: | ---: | ---: |
| `control` | Original full all-eight LW State-EV, clean CE only | 2,327,462 | 2,327,462 | 0 |
| `masked` | Same student, clean CE plus masked latent prediction | 2,327,462 | 2,626,598 | 1,926,412 |

The masked arm adds a 299,136-parameter predictor **during training only**.
Both export strict-loadable weights for the original, unmodified LW State-EV
model: no predictor, teacher, extra inference branch, or compact backbone.

Both arms use the original 100M-pool Ranger recipe: global clean batch 512,
1,000,000 attempted updates, 50 validations every 20,000 updates, 1.28M events
per check, and the original learning-rate milestones. Checkpoint selection
uses clean validation accuracy, retaining the first maximum. Final evaluation
uses balanced full-5M validation; it overlaps checkpoint-selection validation
and is **not an independent test**.

| Platform | Benchmark / training | Local clean batch | BN | CPUs / host RAM | Existing Python environment |
| --- | --- | ---: | --- | --- | --- |
| SPORC `tier3` | Two same-node A100s | 256 / rank | SyncBatchNorm | 8 / 128 GiB | `/home/ryreu/miniconda3/envs/part_ev_screen_sporc/bin/python` |
| Tigris | One GH200, at least 80 GiB | 512 | Native BatchNorm | 12 / 128 GiB | `/home/ryreu/miniforge3-aarch64/envs/part_ev_fullscale/bin/python` |

Evaluation uses one GPU of the platform's type. These are separate matched
comparisons, not runtime-identical cross-platform replicas. Running both
pipelines authorizes **four full trainings**, not two. Neither pipeline
cancels, depends on, migrates, or selects a winner against the other.

## Queue the automatic pipeline

First commit/push the new code locally. On the relevant login node, pull it
and use a **fresh absolute output path**, directly under
`checkpoints/part_ev_masked_predictive`. Do not reuse another campaign's output.
The launcher freezes committed source; dirty/uncommitted Python, shell or YAML
files block submission. No new environment or dataset copy is made.

SPORC:

```bash
(
  set -euo pipefail
  cd /home/ryreu/atlas/Fresh_check
  git pull --ff-only

  MASKED_ROOT="${PWD}/checkpoints/part_ev_masked_predictive/sporc_$(date -u +%Y%m%dT%H%M%SZ)"
  printf 'Campaign: %s\n' "${MASKED_ROOT}"

  bash sbatch/submit_part_ev_masked_predictive.sh \
    --mode plan --platform sporc --output "${MASKED_ROOT}"

  bash sbatch/submit_part_ev_masked_predictive.sh \
    --mode pipeline --platform sporc --output "${MASKED_ROOT}" \
    --max-train-hours 480 --max-evaluate-hours 24 --reviewed

  squeue --me -o '%.18i %.48j %.12T %.12l %R'
)
```

Tigris:

```bash
(
  set -euo pipefail
  cd /home/ryreu/atlas/Fresh_check
  git pull --ff-only

  MASKED_ROOT="${PWD}/checkpoints/part_ev_masked_predictive/tigris_$(date -u +%Y%m%dT%H%M%SZ)"
  printf 'Campaign: %s\n' "${MASKED_ROOT}"

  bash sbatch/submit_part_ev_masked_predictive.sh \
    --mode plan --platform tigris --output "${MASKED_ROOT}"

  bash sbatch/submit_part_ev_masked_predictive.sh \
    --mode pipeline --platform tigris --output "${MASKED_ROOT}" \
    --max-train-hours 480 --max-evaluate-hours 24 --reviewed

  squeue --me -o '%.18i %.48j %.12T %.12l %R'
)
```

`--reviewed` explicitly authorizes the paired production jobs **only if the
complete benchmark evidence passes** and measured recommendations fit the
reviewed ceilings and live Slurm account/QoS/partition/resource checks.
Defaults are account `reu-aisocial` and the existing interpreters above.
There is no concurrency cap, automatic cancellation, shortened recipe, or
three-day segmentation.

The initial chain is CPU prepare (2h) → paired GPU benchmark (5h30m) → CPU
summary (15m) → CPU dispatch (30m). Summary runs after any benchmark outcome,
but returns nonzero on incomplete/failed admission, blocking dispatch.
Dispatch submits two independent training jobs, an `afterok` evaluation for
each, and a one-hour CPU report after both evaluations succeed. Training and
evaluation walltimes are measured **separately for each arm**, not guessed in
advance or clamped to fit. Exceeding 480h training / 24h evaluation, or stricter
live limits, blocks production and leaves evidence for review.

For **benchmarks only**, use `--mode benchmark` instead of `--mode pipeline`
and omit `--reviewed`. It creates prepare → benchmark → summary only. After
reviewing successful evidence, production may be submitted separately:

```bash
# Set MASKED_ROOT to the exact printed campaign path first.
bash sbatch/submit_part_ev_masked_predictive.sh \
  --mode review --output "${MASKED_ROOT}"
bash sbatch/submit_part_ev_masked_predictive.sh \
  --mode dispatch --output "${MASKED_ROOT}" --reviewed
```

Do not run a second fresh pipeline merely because dependent jobs are pending.
Recorded submission IDs and durable pre-submit intentions prevent blind
duplicate submissions. An ambiguous or partial submission requires reconciling
the journal with Slurm before another attempt.

## What the auxiliary task does

Clean classification always uses all 512 jets. When the auxiliary weight is
positive, exactly half the local clean batch is independently selected; half
the selected slots use uniform masking and half use a spatial patch. Eligible
jets have at least four retained particles; roughly one-quarter are targets,
with at least two visible particles left. Four versioned per-rank CPU generators
control selection, particles, centers and ties without consuming native model
RNG. Masking begins from the **actual clean post-trim collection**, never a
second independently trimmed jet.

The visible student sees physically compacted visible particles only. All
seven jet-relative feature channels are rebuilt from the visible vector-sum
four-momentum using the pinned preprocessing YAML. Teacher full-jet features
use that same visible-defined reference. Independent PID/charge/track features
are retained; no hidden particle can influence visible normalization, padding,
pair computation, or old cache state. Query locations contain only signed
visible-relative eta and wrapped phi sine/cosine.

A frozen particle-only EMA teacher supplies normalized block-6/block-8 targets.
Two independently parameterized cross-attention/FFN predictor blocks map masked
locations and visible student states to those targets, without query
self-attention. Smooth-L1 averages channels, then targets, then globally
eligible jets; large jets do not receive larger loss weights.

Auxiliary weight is zero through update 20,000, ramps to 0.1 at 40,000, stays
there through 800,000, and decays to zero at 900,000. The final 100,001 attempts
are clean-only. Overall, 879,999 attempts are auxiliary-active and 120,001 are
clean-only. Teacher parameter momentum rises from 0.996 to 0.9999; parameter
EMA follows successful Ranger/Lookahead updates only, while clean BN buffers
are copied after every complete attempt, including a scaler skip.

The clean forward/backward completes before auxiliary graphs are built. Each
auxiliary microbatch has at most 32 selected jets and uses immutable pre-clean
BN buffers through `functional_call`, with attached student parameters, frozen
BN arithmetic/dropout/trimming and unchanged checkpointed State-EV training
arithmetic. Live post-clean buffers are never replaced. DDP uses public
`no_sync` for clean/nonfinal work and a final synchronized flush, with zero
anchors covering all parameters even for empty slots. One complete attempt
means one scaler decision, Ranger step and teacher transaction.

## Benchmark gates and projections

The benchmark uses disposable fresh weights and does not create production
checkpoints. It measures the control and masked arms on the real ROOT stream:

- Twenty warmup plus 4,200 measured complete attempts per main arm; masked
  attempts execute the real teacher, views, predictor and sequential backwards.
- A separate masked clean-only phase: twenty warmup plus 200 measured attempts.
- Full-128-particle stress, batch-512 FP32 validation, 128k streaming validation,
  exported original-model FP32 throughput at batch 256 and batch-one latency.
- Complete restart/student checkpoint I/O, prediction I/O and a timed synthetic
  full-5M CPU merge/metric pass. No test data are opened.

Diagnostic auxiliary schedule labels are explicit and **are not production
cursors or a claim of training 100,000 updates**. Projection uses the slowest
measured windows, weighted by 879,999 active plus 120,001 clean-only masked
attempts, all 50 validations, checkpoint writes and a 30% margin. Queue waits
and restart replay are excluded. Peak allocated/reserved GPU memory and host
RSS/cgroup evidence are recorded; allocated memory must remain below 90%.

Admission also requires actual-device FP16/FP32 observations, independent FP64
predictor/gradient and visible-feature oracles, hidden-input/cache/BN leakage
interventions, clean/auxiliary gradient parity, chunk-checkpoint backward
parity beyond 64 visible particles, teacher tap/gradient checks, eight real
learning steps, controlled overflow recovery and actual CUDA DDP/SyncBN phase
tests. CPU Gloo health probes cannot stand in for those GPU gates.

Complete restart roundtrips are exact; seven-update repeat/disk/resume trials
use fixed bounded floating-point tolerances and exact RNG, mask, scaler,
schedule and cursor checks, including phase boundaries and Lookahead/validation.
Separate ROOT replay checks authenticate the next inputs/masks and a bounded
next-target comparison using a fresh frozen teacher. Replay itself performs no
model work or augmentation draws. Synthetic restart tests, not that fresh
teacher probe, authenticate continuation of the **trained** EMA state.

Small train/export profiles report CPU/device attribution and partial operator
FLOPs, not an invented complete training FLOP count. Latent variance, norms,
cosine, backbone gradients and a zero-KV intervention are diagnostic only; they
do not adapt the loss or alter the scientific recipe.

## Progress, artifacts and interruption

```bash
bash sbatch/submit_part_ev_masked_predictive.sh \
  --mode status --output "${MASKED_ROOT}"
```

Status is read-only and includes recorded validation checks, the latest logged
update, progress toward the next 20k boundary, queue/accounting state and GPU
allocation hours including retries. Reaching a boundary does not mean its
validation has finished; training-time metrics are not full-5M/test results.

Within the printed campaign directory:

- `source/`, `source_manifest.json`, `launch.json`: immutable source/environment
  and authority. Later pulls do not change queued workers.
- `campaign.json`, `benchmark.json`, `summary.json`: preparation, complete raw
  evidence and recomputed admission. Failure files/logs never authorize training.
- `production.json`, `production/campaign.json`: admitted full paired recipe.
- `production/runs/{LW_STATE_EV_CONTROL,MASKED_PREDICTIVE_STATE_EV_LAYERWISE}/seed_101/`:
  `last.pt`, student-only `selected/` and `best_model_val.pt`, `validation/`,
  `training_complete.json` and resumable hashed `full_validation/` chunks.
- `production/reports/masked_report.{json,md}`: authenticated matched results,
  AUC, QCD rejection with counts, deployment/training sizes and measured costs.

`last.pt` is atomic and contains the complete student, predictor, teacher,
Ranger moments/Lookahead/cache, scaler, per-rank native and auxiliary RNG,
trimmer counters and train/validation cursors. It is written every 1,000
complete attempts, around validation and on a cooperative stop; no partial
auxiliary attempt can replace it. Stop/deadline exits are nonzero (75), blocking
dependent evaluation. There is no silent reset or automatic extra allocation.

Explicit same-platform recovery is available after inspecting terminal Slurm
states and the authenticated complete checkpoint:

```bash
bash sbatch/submit_part_ev_masked_predictive.sh \
  --mode resume --output "${MASKED_ROOT}" --arm masked --reviewed
# Use --arm control for the other arm; never change platforms on resume.
```

Active, uncertain or partially recorded old jobs block this operation; reconcile
obsolete dependencies first. A completed training resumes evaluation only.
Continuous ROOT train/validation streams are replayed without model/augmentation
work, then saved native RNG is restored. Replay can be expensive and consumes
the new allocation; a retry is not guaranteed to finish within its walltime.

## Storage and local verification

No dataset, environment, persistent target cache or profiler trace is copied.
The paired campaign's nominal output budget is about **6.89 GiB**, plus a shared
**4-GiB reserve**: approximately **10.89 GiB** before outputs exist. A
benchmark-only campaign initially requires 2 GiB plus reserve and rechecks the
production budget at dispatch. Retained outputs are subtracted from the
remaining budget. Other campaigns' speculative reservations are advisory only,
as requested; they cannot block an adequately provisioned masked campaign.
They can still consume real space, so monitor `df` while jobs run. Nothing is
deleted or reservations silently rewritten.

Local verification covers the model/teacher/predictor, masking/rebasing/leakage,
sequential and checkpointed gradients, real Ranger/EMA restart transactions,
two-process CPU Gloo edge cases, raw-evidence forgery rejection, fake-Slurm
submission chains, frozen-source isolation, explicit recovery and read-only
status. Run the focused tests with the repository's Python environment:

```bash
python -B -m pytest tests/test_part_ev_masked_predictive_*.py -q -p no:cacheprovider
```

On 2026-11-24, the final focused suite passed **91 tests**, including a real
two-process CPU Gloo test. A broader regression run across workspace, experts,
multiscale and native State-EV/training/precision/reporting code passed **411
tests with 9 skipped**. These runs overlap; their counts should not be added.
Python syntax, Bash syntax and whitespace checks also passed.

This implementation was tested locally with CPU Torch. GPU checks deliberately
remain mandatory on each cluster; neither speed nor accuracy is assumed from
the local tests. No real jobs were submitted and no environments, datasets or
existing campaign artifacts were changed.
