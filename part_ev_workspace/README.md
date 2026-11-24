# Persistent global workspace: paired full-schedule experiment

Implemented from [the scientific plan](../part_ev_state_compare/PERSISTENT_GLOBAL_WORKSPACE_PLAN.md).
The model, paired benchmark, production admission, training, recovery, full-5M
evaluation and reporting interfaces are implemented. **Actual CUDA fit, runtime,
DDP correctness and accuracy are not established locally.** Those are measured
by the queued benchmark; CPU tests cannot authorize production.

## What runs

Each platform independently benchmarks and, if explicitly authorized, trains:

| Arm | Architecture | Parameters |
|---|---|---:|
| `control` | Full, noncompact, all-eight Layerwise State-EV | 2,327,462 |
| `workspace` | Same backbone + persistent global workspace | 3,720,614 |

There are eight learned width-128 workspace tokens and four independent
read/mix/FFN/feedback stages after particle blocks 2, 4, 6 and 8. The final native
class representation also reads the workspace before its original classifier.
Feedback and final-read output projections start at zero, preserving the native
initial logits/RNG, while real Ranger probes check that upstream paths learn.
Workspace state is local to one forward, not a persistent event cache.

Both arms start fresh at seed 101 with the unchanged original Ranger recipe:
100M training pool, global batch 512, 1,000,000 **attempted updates**, 50 validation
checks every 20,000 updates, and the original learning-rate milestones. Each
check consumes 1.28M events from a continuous validation stream. The first best
accuracy selects the checkpoint; evaluation uses the full balanced 5M validation
population. This overlaps checkpoint selection and is **not an independent test**.

| Platform | Benchmark / each training | Each evaluation |
|---|---|---|
| SPORC `tier3` | Two same-node A100s, local batch 256, SyncBatchNorm, 8 CPUs / 128 GiB | One A100, saved SyncBN in eval, 8 CPUs / 128 GiB |
| Tigris | One GH200 with at least 80 GiB, local batch 512, native BatchNorm, 12 CPUs / 128 GiB | One GH200, 12 CPUs / 128 GiB |

Training uses FP16 AMP with FP32 master parameters; workspace norms, attention
scores/softmax/reductions and workspace residuals are FP32. Validation is FP32.
Whole workspace exchanges use nonreentrant activation checkpointing. Native
particle attention, pair encoding, dropout, trimming and class attention remain.

Default control reuse is deliberately **not** enabled: this first interface runs
a fresh matched control on each platform. Launching pipelines on both platforms
can therefore produce **four full trainings**, not two. Neither platform waits
for, cancels, or locks out the other. No arbitrary concurrency cap is imposed.

## Queue prepare → benchmark → automatic production

First commit **all new workspace implementation files**, push, and pull on RC.
The launcher archives committed source; it rejects uncommitted Python/shell/YAML
changes. Do not edit frozen `source/` directories. Later pulls do not change these
workers and this addition does not patch existing campaign code.

Run this on **sporcsubmit**:

```bash
(
  set -euo pipefail
  cd /home/ryreu/atlas/Fresh_check
  git pull --ff-only
  WORKSPACE_ROOT="$(pwd -P)/checkpoints/part_ev_workspace/sporc_$(date -u +%Y%m%dT%H%M%SZ)"
  printf 'Campaign: %s\n' "${WORKSPACE_ROOT}"
  bash sbatch/submit_part_ev_workspace.sh \
    --mode plan --platform sporc --output "${WORKSPACE_ROOT}"
  bash sbatch/submit_part_ev_workspace.sh \
    --mode pipeline --platform sporc --output "${WORKSPACE_ROOT}" \
    --max-train-hours 480 --max-evaluate-hours 24 --reviewed
  squeue --me -o '%.18i %.48j %.12T %.12l %R'
)
```

Run this separately on **Tigris**:

```bash
(
  set -euo pipefail
  cd /home/ryreu/atlas/Fresh_check
  git pull --ff-only
  WORKSPACE_ROOT="$(pwd -P)/checkpoints/part_ev_workspace/tigris_$(date -u +%Y%m%dT%H%M%SZ)"
  printf 'Campaign: %s\n' "${WORKSPACE_ROOT}"
  bash sbatch/submit_part_ev_workspace.sh \
    --mode plan --platform tigris --output "${WORKSPACE_ROOT}"
  bash sbatch/submit_part_ev_workspace.sh \
    --mode pipeline --platform tigris --output "${WORKSPACE_ROOT}" \
    --max-train-hours 480 --max-evaluate-hours 24 --reviewed
  squeue --me -o '%.18i %.48j %.12T %.12l %R'
)
```

These are user-run instructions, not a statement that any jobs were submitted.
The printed campaign path is the authoritative path for subsequent commands.
No timestamp placeholders need editing in these submission commands.

Default existing interpreters (not installed/modified by the launcher):

- SPORC: `/home/ryreu/miniconda3/envs/part_ev_screen_sporc/bin/python`
- Tigris: `/home/ryreu/miniforge3-aarch64/envs/part_ev_fullscale/bin/python`
- Data: `/home/ryreu/atlas/datasets/JetClass`; account: `reu-aisocial`.

An explicitly different existing interpreter can be supplied using `--python`
on a **fresh** campaign; architecture, pinned Weaver and complete environment
still have to pass. Never change an environment used by active jobs.

The chain initially has four jobs:

```text
prepare → paired benchmark → summary → admission/dispatch
                                      ├─ control train   → control evaluate   ─┐
                                      └─ workspace train → workspace evaluate ─┴─ report
```

Prepare requests 2 CPU hours; the paired benchmark requests 5h30m and enforces a
5-hour internal deadline; summary 15m; dispatch 30m; paired report 1h. Training
and evaluation walltimes are **measured per arm**, including 50 validations,
checkpoint I/O and 30% margin. The 480h/24h flags are reviewed upper ceilings,
**not** walltimes automatically requested for every job. A recommendation above
either ceiling or the live partition/account/QoS limit blocks submission; it is
never silently shortened, split, or capped down.

`pipeline --reviewed` authorizes automatic production only after raw scientific,
numerical, runtime and storage evidence passes. Slurm `COMPLETED`, an edited
`passed` flag, or a summary file alone cannot authorize production. Both control
and candidate must pass. A failed benchmark leaves evidence and no training.

## Benchmark-only / review / later production

To inspect the benchmark before deciding, replace `--mode pipeline` with
`--mode benchmark` and omit `--reviewed`. This submits only prepare, benchmark
and summary. It does **not** reserve the complete production budget or train.

With the printed campaign path assigned to `WORKSPACE_ROOT`, on its own cluster:

```bash
cd /home/ryreu/atlas/Fresh_check
bash sbatch/submit_part_ev_workspace.sh --mode review --output "${WORKSPACE_ROOT}"
bash sbatch/submit_part_ev_workspace.sh --mode status --output "${WORKSPACE_ROOT}"
```

Both are read-only. To separately authorize production after a passed benchmark:

```bash
bash sbatch/submit_part_ev_workspace.sh \
  --mode dispatch --output "${WORKSPACE_ROOT}" --reviewed
```

Review recomputes timing/admission from `benchmark.json`; dispatch authenticates
source, environment, data inventory, raw evidence and live Slurm requests again.
All later phases execute the original frozen interpreter and source.
Status prints all recorded checks and the last logged percentage toward the
next 20k-update boundary; reaching that boundary is distinguished from having
finished validation. It also displays this campaign's queue rows and journals.

## What the benchmark proves

The disposable benchmark runs the control first, then workspace, with identical
first rank-batch fingerprints and independent fresh initialization. Per arm:

- Twenty warmups and 4,200 real-stream attempts, preserving the 100M file shards,
  trimming, rank/worker policy and batch size. Timings retain 1,000/1,000/1,000/
  1,000/200-attempt windows; the slowest rank/window drives the projection.
- Full-length 128-particle training and FP32 validation stress; 128k streamed
  validation; FP32 batch-256 throughput and batch-1 latency; actual checkpoint
  write/load timing and a synthetic full-5M CPU metric-cost measurement.
- Observed activation/master precision audits, synchronized scaler/Lookahead
  schedules, finite states, less than 5% measured AMP skips, and 10% allocated
  GPU-memory headroom. Host RSS/high-water/cgroup evidence is recorded too.
- Independent double-precision attention **and complete-stage** forward/gradient
  oracles; native initial identity/RNG/call counts; active checkpoint backward;
  padding, permutation, per-jet isolation, one-particle and unequal-rank lengths;
  actual SyncBN/DDP gradients versus a global-batch reference on SPORC.
- Real Ranger learning through initially zero output gates; seven-update repeat,
  disk and resumed trajectories crossing validation and Lookahead boundaries.
  Immediate state/RNG/counters restore exactly; subsequent floating-point state
  and losses meet the plan's fixed bounds. No adaptive tolerance relaxation.
- Actual ROOT iterator replay and rank-RNG restoration; rank-coordinated malformed
  input/nonfinite/I/O/stop rejection; bounded two-event profiles and eight-event
  workspace collapse/entropy/feedback diagnostics. These make no accuracy claim.

Root-only DDP validation/CPU metrics exchange bounded coordination heartbeats,
so peers do not spend an entire long metric calculation in one NCCL broadcast.

Profiler FLOPs are explicitly partial visible-operator counts (2 FLOPs/MAC) at
the bounded profile shape, not total model or corpus-training FLOPs. Timing is
not inferred from parameter count. CPU checks do not supply fake GPU evidence.

## Storage and isolation

Outputs live only under `checkpoints/part_ev_workspace/<fresh-name>/`:

```text
source/                          committed frozen source
launch.json, source_manifest.json, snapshot.sha256
campaign.json, data_contents.json
benchmark.json, summary.json, probes/, logs/
pipeline_submissions.json, pipeline_job_ids.tsv
production.json, production_submissions.json, production_job_ids.tsv
production/runs/<run-id>/seed_101/
  optimizer.json, last.pt, selected/check_XX.pt, best_model_val.pt
  validation/check_XX.json, training_complete.json
  full_validation/file_XXX.npz + .json, metrics.json
production/reports/workspace_report.json + .md
```

At most 50 improving selected candidates are retained per arm, plus final best
weights and one atomic latest full restart (with bounded write temporaries).
No dataset copies, per-event workspace cache, downloaded models or new conda
environment. Probe checkpoints stay in the separate bounded benchmark slot.

Two paired pipelines on the shared filesystem prospectively need about
**18.94 GiB initial headroom including one shared 4-GiB emergency reserve**,
before other registered outstanding output budgets. This is conservative
planning, not a quota or a claim about currently free space. Benchmark-only
slots reserve 2 GiB each, then production checks the larger budget at dispatch.

Admission uses shared locks and counts both new platform reservations plus
existing registered campaign budgets. Old frozen launchers cannot learn about
new registry types retroactively: avoid simultaneously submitting unrelated old
launchers while budgeting tightly. Unregistered users/files can still consume
space. No output/environment/data is deleted automatically, and missing jobs
from one cluster's queue do not release the other cluster's reservation.

## Interruption and explicit recovery

Training saves full model/Ranger slow weights/moments/caches/scaler and per-rank
RNG/trimmer, attempted step, validated checks, best candidate/hash and timing at
least every 1,000 attempts and before/after validation. A USR1/TERM or monotonic
deadline coordinates both ranks and saves between updates. An incomplete stop
exits nonzero (75 for planned stops), blocking `afterok` successors; it is never
reported as a completed 1M/50-check training.

The next **reviewed same-platform** allocation rebuilds/replays the exact recorded
training and validation iterator cursors before restoring RNG and continuing.
Replay consumes walltime and can itself be interrupted. No benchmark weights,
best-only weights, altered normalization, changed GPU count, or cross-cluster
checkpoint migration is accepted as a training restart.

If recovery is needed, inspect accounting and journals first. Cancel only the
obsolete dependent evaluation/report jobs for that arm after verifying their
recorded IDs; the launcher never cancels them or the other arm automatically.
Then:

```bash
bash sbatch/submit_part_ev_workspace.sh \
  --mode resume --output "${WORKSPACE_ROOT}" --arm workspace --reviewed
```

Use `--arm control` for the control. Recovery requires terminal accounting for
that arm's prior training/evaluation and the obsolete paired report. Missing or
active accounting stops it. It authenticates the latest complete restart, queues
only that arm's remaining training/evaluation and relinks a paired report to the
other arm's latest evaluation. Completed training is audited and not repeated.
The original resource recommendation remains; no automatic extra allocations.

Every submission writes durable **unknown intent before sbatch**. Partial
chains or ambiguous replies stop; inspect the journal and actual scheduler before
any manual reconciliation. Do not blindly rerun with a new timestamp: that can
duplicate allocations. A same-platform production-owner record also prevents a
second fresh comparison from silently duplicating this one.

## Verification and interpretation

Local checks:

Verified 2026-11-23 on Windows / PyTorch 2.12.1 CPU: **54 new tests passed**.
The selected combined workspace/backbone/launcher/precision/full-schedule
regression suite passed **136 tests with 6 existing GPU/environment skips**.
Both read-only platform plans and Bash syntax passed. No real Slurm requests,
cluster dataset reads, CUDA benchmarks or production allocations were executed.

```powershell
.venv\Scripts\python.exe -B -m pytest tests/test_part_ev_workspace_model.py tests/test_part_ev_workspace_runtime.py tests/test_part_ev_workspace_protocol.py tests/test_part_ev_workspace_launcher.py tests/test_part_ev_workspace_status.py tests/test_part_ev_workspace_coordination.py -q
```

Includes CPU functional math, real Ranger learning/restart, strict checkpoint
identity/cursor checks, bounded diagnostics, aggregate headroom, raw-evidence
rejection, fake-Slurm submission/dispatch/recovery, read-only plans, frozen-source
isolation, two-process CPU/Gloo error/stop coordination and shell syntax. They do not execute Slurm, read RC datasets, verify
actual CUDA/DDP behavior or measure classification gains.

The final report audits selected weights/all 50 checks and full-5M chunk hashes,
recomputes metrics, checks paired event order, and reports accuracy/AUC/rejection,
learning curves, recorded timing, skips and benchmark costs. Interpret within
platform first. The two platforms are not two independent random seeds, and
historical runtime/normalization contrasts do not isolate architecture alone.
