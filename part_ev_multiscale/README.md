# Nested multiscale local-global State-EV

Implementation of [the scientific plan](../part_ev_state_compare/MULTISCALE_LOCAL_GLOBAL_PLAN.md).
Model, paired benchmarks, benchmark-gated production, restart, full-5M evaluation,
reporting and read-only status are implemented. Actual A100/GH200 fit, throughput,
CUDA numerical gates and classification gains are **not established locally**.
The queued benchmark must establish the device-specific evidence first.

## Model and controls

| Arm | Run ID | Parameters |
|---|---|---:|
| `control` | `LW_STATE_EV_CONTROL` | 2,327,462 |
| `multiscale` | `MULTISCALE_STATE_EV_LAYERWISE` | 5,366,950 |

The candidate retains the entire noncompact, standard-relations, all-eight
Layerwise State-EV backbone. One detached FP32 E-scheme angular merge history
constructs nested cuts at up to 16 fine / 4 coarse regions after native trimming.
Exactly duplicate four-vectors form indivisible atoms. Canonical physical keys
and exact-distance tie rules are independent of particle order; signed zeros
are normalized. Regions have no radius/beam removal or discarded particles.

Four independently parameterized stages after native blocks 2/4/6/8 rebuild
region tokens from the current particle states: member mean/max + seven physical
descriptors, fine member reads, coarse-sibling fine attention, occupancy-weighted
coarse seeds and child reads, global coarse attention, top-down attention, and
own-parent feedback to particles. The final native class representation reads
both final region scales before the unchanged classifier. These tokens are not
carried between events or stages. No selected relations, experts, auxiliary loss,
new dropout, warm start or compact State-EV substitution is added.

Exactly four feedback outputs and the final fusion output start at zero; all
other new linears use isolated Xavier initialization. Native initial tensors,
RNG, call counts, logits and CE gradients are checked. Whole learned stages use
nonreentrant checkpointing; geometry is outside recomputation. Segmented amax
pooling uses linear-size storage and distributes gradients across tied maxima.
FP32 norms/scores/reductions/region residuals coexist with FP16 learned linears;
evaluation is FP32. Small FP64 independent stage/pooling/attention oracles remain.

Each platform runs a **fresh matched control and fresh candidate**, seed 101.
The original Ranger recipe remains: 100M training pool, global batch 512,
1,000,000 attempted updates and 50 checks every 20,000 attempts, 1.28M validation
events per check, original LR milestones, zero weight decay, CE only. The first
maximum validation accuracy selects weights. Full balanced 5M validation follows;
it overlaps selection and is not an independent test.

## Queue on SPORC

Commit all new implementation files, push, then run on **sporcsubmit**:

```bash
(
  set -euo pipefail
  cd /home/ryreu/atlas/Fresh_check
  git pull --ff-only
  MULTISCALE_ROOT="$(pwd -P)/checkpoints/part_ev_multiscale/sporc_$(date -u +%Y%m%dT%H%M%SZ)"
  printf 'Campaign: %s\n' "${MULTISCALE_ROOT}"
  bash sbatch/submit_part_ev_multiscale.sh \
    --mode plan --platform sporc --output "${MULTISCALE_ROOT}"
  bash sbatch/submit_part_ev_multiscale.sh \
    --mode pipeline --platform sporc --output "${MULTISCALE_ROOT}" \
    --max-train-hours 480 --max-evaluate-hours 24 --reviewed
  squeue --me -o '%.18i %.48j %.12T %.12l %R'
)
```

## Queue separately on Tigris

```bash
(
  set -euo pipefail
  cd /home/ryreu/atlas/Fresh_check
  git pull --ff-only
  MULTISCALE_ROOT="$(pwd -P)/checkpoints/part_ev_multiscale/tigris_$(date -u +%Y%m%dT%H%M%SZ)"
  printf 'Campaign: %s\n' "${MULTISCALE_ROOT}"
  bash sbatch/submit_part_ev_multiscale.sh \
    --mode plan --platform tigris --output "${MULTISCALE_ROOT}"
  bash sbatch/submit_part_ev_multiscale.sh \
    --mode pipeline --platform tigris --output "${MULTISCALE_ROOT}" \
    --max-train-hours 480 --max-evaluate-hours 24 --reviewed
  squeue --me -o '%.18i %.48j %.12T %.12l %R'
)
```

These are instructions, not jobs submitted by the implementation. Each platform
is independent: both pipelines can produce **four full trainings total**.
Neither automatically cancels, migrates, waits for or reuses the other's runs.
There is no arbitrary concurrency cap or three-day segmentation.

| Resource | SPORC | Tigris |
|---|---|---|
| Benchmark / each training | Two same-node A100s, local batch 256, SyncBN; 8 CPUs, 128 GiB host RAM | One GH200 >=80 GiB, local batch 512, native BN; 12 CPUs, 128 GiB host RAM |
| Each evaluation | One A100, saved SyncBN in eval; 8 CPUs, 128 GiB | One GH200; 12 CPUs, 128 GiB |
| Existing Python | `/home/ryreu/miniconda3/envs/part_ev_screen_sporc/bin/python` | `/home/ryreu/miniforge3-aarch64/envs/part_ev_fullscale/bin/python` |

Defaults: data `/home/ryreu/atlas/datasets/JetClass`, account `reu-aisocial`.
No environment installation, mutation, compiled selected-tree backend or test
data access. A different existing interpreter is possible only for a fresh
campaign via `--python`, subject to platform/source/environment admission.

Initial chain: CPU prepare (2h) -> paired GPU benchmark (5h30m, internal 5h
deadline) -> summary (15m) -> reviewed automatic dispatch (30m). Summary runs
after benchmark termination; only a successful independently validated summary
admits dispatch. Dispatch then submits:

```text
control train    -> control evaluate    --+
multiscale train -> multiscale evaluate --+-> paired report (1h CPU)
```

Train/evaluate walltimes are measured **per arm**, not the ceiling flags. The
train estimate includes slowest rank/window real attempts, all 50 validations,
periodic/boundary/selected checkpoint writes, cold fetch and 30% margin. Final
evaluation includes measured streaming and full-population CPU metrics. Above
480h train / 24h evaluate or stricter live Slurm limits: stop, do not cap down,
shrink batch, drop a scale, split allocations or submit unreviewed alternatives.
The hierarchy is not claimed cheap: merge selection remains cubic worst-case;
added capacity is substantial. Only actual benchmarks determine its cost.

## Benchmark-only, review, status and later production

For benchmarks without automatic production use `--mode benchmark` instead of
`--mode pipeline` and omit `--reviewed`. On the same cluster, retain the printed
campaign path in `MULTISCALE_ROOT`:

```bash
bash sbatch/submit_part_ev_multiscale.sh --mode review --output "${MULTISCALE_ROOT}"
bash sbatch/submit_part_ev_multiscale.sh --mode status --output "${MULTISCALE_ROOT}"
# Separate explicit production authorization after passed evidence:
bash sbatch/submit_part_ev_multiscale.sh --mode dispatch --output "${MULTISCALE_ROOT}" --reviewed
```

Review/status do not write or submit. Status prints all recorded check accuracies,
last logged progress toward the next 20k boundary, queue rows and journals.
Reaching a training boundary is distinguished from completing its validation;
there is no fabricated live validation percentage.

Benchmark evidence includes, for each arm:

- 20 warmup + 4,200 real attempts in 1,000/1,000/1,000/1,000/200 windows;
  identical first rank-batch fingerprints, control first, fresh disposable models.
- Full-128 train stress (10 warmup/40 measured), 128k streamed FP32 validation,
  full-128 batch-512 FP32 stress, single-GPU batch-256 throughput, batch-one
  latency, full checkpoint I/O, 100k prediction-chunk write/read timings and
  synthetic 5M concatenation/CPU metric cost.
- Candidate full-length duplicate-atom and exact-distance-tie backward stress.
- Independent CPU dictionary hierarchy versus target-device batched hierarchy
  through N=128, exact nested retention/duplicates/ties/permutation, independent
  FP64 complete-stage/attention/pooling values and gradients, equal-max ties.
- Native initial identity, active FP32/AMP input and parameter gradients,
  checkpointed/uncheckpointed parity, actual SyncBN/DDP global-batch reference,
  one-particle/unequal-length graphs and real Ranger pathway activation.
- Seven-update reference/repeat/disk/resumed full-state probes, with exact
  canonical mapping digests, counters/RNG/scaler/Lookahead schedules; fixed
  bounded float state/loss thresholds. Actual ROOT train/val cursor replay
  separately verifies input fingerprints and bounded pretrim geometry.
- Coordinated input/nonfinite/I/O/stop failures, observed dtype audits, finite
  model/optimizer/scaler, measured AMP skip fraction <5%, allocated-device peak
  <90%, reserved peak and host RSS/cgroup peak. CPU success is not GPU proof.
- Small train/eval profiles with geometry, pooling, bottom-up/top-down/feedback
  and final-read scopes including recompute; eight-event geometry/occupancy,
  region RMS, entropy, feedback and bounded intervention diagnostics.

Profiler-visible FLOPs use 2/MAC and explicitly omit unsupported operators and
geometry comparisons. Small-profile counts are not full-model/corpus FLOPs.
Disposable benchmarks make no classification accuracy claim.

## Isolation, storage and recovery

All outputs are new, scoped children of `checkpoints/part_ev_multiscale/`.
Source is frozen from committed Git content, fully inventoried and hash-checked
by every worker. Uncommitted Python/shell/YAML files, symlinks, changed inventory,
live-checkout imports and changed environment/data are rejected. Later pulls do
not change frozen workers. Existing campaigns are not edited or requeued.

Storage admission follows the user's updated policy: only **this campaign's
remaining output budget plus a shared 4-GiB emergency reserve** is a hard gate.
Initially about **12.21 GiB** for paired automatic production, **6 GiB** for
benchmark-only. Already written bytes reduce the remaining budget. Other
campaigns' future reservations are advisory, including unreadable/stale foreign
accounting. Both platform pipelines together estimate about **20.41 GiB** with
one shared reserve, not an actual quota reservation. Monitor free space; no
automatic dataset/environment/checkpoint deletion is performed.

Per arm: one latest complete full restart with atomic temporary allowance,
at most 50 improving selected checkpoints plus final best, bounded diagnostics,
resumable full-5M prediction chunks. No per-jet tree/region cache or graph dump.
Production logs constant-size rank geometry summaries every 100 attempts.

Save full model/RAdam/Lookahead/scaler/rank RNG/trimmer, attempt/validation
cursors, selected metadata/hash, timing and geometry diagnostic cursor every
1,000 attempts, before/after validation and coordinated shutdown. Incomplete
planned stops save and exit 75, blocking afterok successors. Full completion
requires both 1M attempts and 50 completed validations. Resume rebuilds/replays
loaders before restoring rank RNG; replay consumes allocation time and remains
interruptible without destroying the prior complete restart.

After checking scheduler accounting/journals and manually cancelling only the
obsolete dependent jobs for the interrupted arm, explicit same-platform recovery:

```bash
bash sbatch/submit_part_ev_multiscale.sh \
  --mode resume --output "${MULTISCALE_ROOT}" --arm multiscale --reviewed
```

`--arm control` selects the control. Terminal accounting and complete checkpoint
identity are required. The other arm is untouched; completed training is audited,
not repeated. No automatic cross-cluster migration or extra allocation.

Every sbatch has durable unknown intent **before** submission and a recorded
cluster/job/request afterward. Ambiguous replies or partial chains stop for
manual reconciliation; do not blindly retry with a fresh timestamp. Shared
submission locks and a same-platform science owner prevent duplicates.

Artifacts: `launch.json`, `source/`, source manifests, `campaign.json`, data
evidence, `benchmark.json`, `summary.json`, journals, probes/logs; then
`production.json`, `production/runs/<run-id>/seed_101/` with `last.pt`,
`selected/check_XX.pt`, `validation/check_XX.json`, completion and full-validation
chunks; `production/reports/multiscale_report.json` and `.md`. Reports audit all
checks/weights/chunks, recompute metrics and show within-platform accuracy,
AUC/rejection counts, timing/memory/FLOP caveats and available scheduler GPU-hours.
One-GH200/native-BN and two-A100/SyncBN results are not pooled as independent seeds
or described as architecture-only cross-platform comparisons.

## Local verification

Windows / PyTorch 2.12.1 CPU. Target-device gates are implemented but CUDA/Slurm/
RC datasets are unavailable locally. No real jobs, environment changes, old
campaign writes or dataset reads were performed. Run the independent suite with:

Verified 2026-11-23: **94 independent multiscale tests passed**. During development,
the selected cross-family backbone/workspace/experts/precision/full-schedule
regression run passed **321 tests, with 9 existing GPU/environment skips**.

```powershell
.venv\Scripts\python.exe -B -m pytest tests/test_part_ev_multiscale_geometry.py tests/test_part_ev_multiscale_model.py tests/test_part_ev_multiscale_runtime.py tests/test_part_ev_multiscale_protocol.py tests/test_part_ev_multiscale_launcher.py tests/test_part_ev_multiscale_status.py tests/test_part_ev_multiscale_coordination.py tests/test_part_ev_multiscale_benchmark.py -q -p no:cacheprovider
```

Read-only plans and Bash syntax are tested for both platforms. Passing fake
Slurm, CPU/Gloo, geometry/math and restart tests does not establish CUDA fit,
runtime, or validation gains. Publication claims require additional controlled
capacity/compute ablations, seeds and an independently locked test comparison.
