# Contextual FFN experts on full Layerwise State-EV

Implements [the specialized-FFN plan](../part_ev_state_compare/SPECIALIZED_FFN_EXPERTS_PLAN.md)
as an independent model, paired benchmark and benchmark-gated full-production
pipeline on SPORC and Tigris. **Local CPU checks do not establish CUDA fit,
target-device speed, numerical admission or classification gains.** Actual
production admission requires this platform's complete benchmark evidence.

## Locked experiment

| Arm | Architecture | Parameters | Training loss |
|---|---|---:|---|
| `control` | Full noncompact all-eight LW State-EV | 2,327,462 | CE only |
| `experts` | Same backbone plus six contextual specialist banks | 3,998,078 | CE + 0.001 × mean global bank balance |

Particle blocks 3–8 retain their original width-512 FFNs and add four independent
`128 → 256 → LayerNorm → 128` specialists each, with exact GELU before the norm.
Each valid particle chooses exactly two specialists. The router sees its native
pre-FFN-normalized state, its own jet's differentiable masked-mean context and
their product. Stable descending sorting breaks exact ties by expert ID; selected
logits are renormalized over the two routes. Dispatch is dropless, with disjoint
assignment slots and graph-connected empty experts—no token quota, artificial
token, jitter, inference balancing or expert-parallel all-to-all.

All 24 expert output projections start at zero. Native weights, initial logits,
dropout/trimmer RNG, eight attentions/shared FFNs and two class blocks are retained.
The final native dropout is applied once to the sum of dense and specialist
outputs. New modules live under `mod.blocks.{2..7}.specialists.*`; original keys
are preserved. No persistent workspace or other proposed extension is included.

The candidate adds 1,670,616 stored parameters (about 71.8%). This is an initial
architectural screen, not a parameter-/compute-matched capacity comparison or
a novelty claim. No extra ablations are automatically submitted.

The full recipe remains fresh seed 101, the official 100M training pool, global
batch 512, original RAdam/Lookahead and LR milestones, **1,000,000 attempted
updates**, and **50 validation checks every 20,000 attempts**. Each check uses
1.28M events from a continuous validation stream. First highest classification
accuracy selects the checkpoint; full evaluation uses all balanced 5M validation
events. That population overlaps selection and is **not an independent test**.

Native/expert linears train with FP16 AMP and FP32 master parameters. Router
linears, context/hidden norms, softmaxes and weighted combinations are FP32.
The audit authenticates these module-path exceptions; it does not turn off
native precision checks. Validation/inference is FP32. Only pure expert MLPs
are nonreentrantly checkpointed; routing is not recomputed during backward.

## Independent platform resources

| Platform | Paired benchmark / each training | Each final evaluation |
|---|---|---|
| SPORC `tier3` | 2 matching same-node A100s, local 256, SyncBN/DDP, 8 CPUs / 128 GiB | 1 A100, saved SyncBN in eval, 8 CPUs / 128 GiB |
| Tigris | 1 GH200 ≥80 GiB, local 512, native BN, 12 CPUs / 128 GiB | 1 GH200, 12 CPUs / 128 GiB |

Both arms start fresh on each platform. If both pipelines are authorized, that
can produce **four full trainings**, not two. Historical-control reuse is not
implemented implicitly. There is no cross-platform dependency, automatic
cancellation, first-start-winner mechanism, checkpoint migration or concurrency
cap. Compare within platform; seed 101 on different hardware is not two seeds.

Existing interpreters are used without installing or modifying environments:

- SPORC: `/home/ryreu/miniconda3/envs/part_ev_screen_sporc/bin/python`
- Tigris: `/home/ryreu/miniforge3-aarch64/envs/part_ev_fullscale/bin/python`
- Data root: `/home/ryreu/atlas/datasets/JetClass`; account: `reu-aisocial`.

Prepare requests 2 CPU hours; the paired benchmark 5h30m (5-hour internal
deadline); summary 15m; optional admission/dispatch 30m; paired report 1h.
Train/evaluation walltimes are measured per arm with 30% margin, including the
50 validations and checkpoint I/O. Reviewed ceilings are 480h/24h, subject to
live partition/account/QoS limits. Exceeding a ceiling blocks admission; no
silent shortening, model shrinkage or three-day split. Queue and replay are
not covered by an engineering guarantee.

## Queue prepare → benchmark → gated production

First commit/push all new `part_ev_ffn_experts/`, driver, shell and test files
and this documentation. The launcher archives committed source and rejects
uncommitted relevant Python/shell/YAML changes. Workers verify that complete
snapshot and import only frozen source. Later pulls cannot change these workers
or any existing workspace/other campaign's frozen source.

On **sporcsubmit**:

```bash
(
  set -euo pipefail
  cd /home/ryreu/atlas/Fresh_check
  git pull --ff-only
  EXPERTS_ROOT="$(pwd -P)/checkpoints/part_ev_ffn_experts/sporc_$(date -u +%Y%m%dT%H%M%SZ)"
  printf 'Campaign: %s\n' "${EXPERTS_ROOT}"
  bash sbatch/submit_part_ev_ffn_experts.sh \
    --mode plan --platform sporc --output "${EXPERTS_ROOT}"
  bash sbatch/submit_part_ev_ffn_experts.sh \
    --mode pipeline --platform sporc --output "${EXPERTS_ROOT}" \
    --max-train-hours 480 --max-evaluate-hours 24 --reviewed
  squeue --me -o '%.18i %.48j %.12T %.12l %R'
)
```

Separately on **Tigris**:

```bash
(
  set -euo pipefail
  cd /home/ryreu/atlas/Fresh_check
  git pull --ff-only
  EXPERTS_ROOT="$(pwd -P)/checkpoints/part_ev_ffn_experts/tigris_$(date -u +%Y%m%dT%H%M%SZ)"
  printf 'Campaign: %s\n' "${EXPERTS_ROOT}"
  bash sbatch/submit_part_ev_ffn_experts.sh \
    --mode plan --platform tigris --output "${EXPERTS_ROOT}"
  bash sbatch/submit_part_ev_ffn_experts.sh \
    --mode pipeline --platform tigris --output "${EXPERTS_ROOT}" \
    --max-train-hours 480 --max-evaluate-hours 24 --reviewed
  squeue --me -o '%.18i %.48j %.12T %.12l %R'
)
```

These are instructions for the user; no real jobs were submitted during local
implementation. Save the printed campaign path. Initially four jobs are queued:

```text
prepare → paired benchmark → summary → reviewed admission/dispatch
                                      ├─ control train → control evaluation ─┐
                                      └─ experts train → experts evaluation ─┴─ paired report
```

Summary runs after benchmark termination but succeeds only on complete admitted
evidence. Dispatch independently recomputes all gates. A Slurm `COMPLETED`,
boolean `passed`, or a historical result alone cannot admit production.
Scientific/precision/route/restart/GPU-memory gates remain strict.

For benchmark-only mode, use `--mode benchmark` instead of `pipeline` and omit
`--reviewed`; no full training is authorized. Read-only review/status and a
separately reviewed later production dispatch, on the campaign's own cluster:

```bash
bash sbatch/submit_part_ev_ffn_experts.sh --mode review --output "${EXPERTS_ROOT}"
bash sbatch/submit_part_ev_ffn_experts.sh --mode status --output "${EXPERTS_ROOT}"
bash sbatch/submit_part_ev_ffn_experts.sh --mode dispatch --output "${EXPERTS_ROOT}" --reviewed
```

Status prints every recorded check, last logged progress toward the next 20k
boundary, this campaign's queue rows and submission journals. Reaching a training
boundary is not the same as finishing validation; validation has no live percent.

## Mandatory global balancing and target-device admission

For bank b, let T be global valid-particle count, S the sum of four-way route
probabilities and C the hard counts of requested top-two assignments:

```text
P = S / T
F = stop_gradient(C / (2*T))
balance_b = 4 * sum(P * F)
loss = mean jet CE + 0.001 * mean_six_banks(balance_b)
```

Uniform routing has balance 1, not 0. Rank-local normalization is not averaged.
Outside model forward, the trainer reduces detached probability sums and exact
int64 assignment/token counts in fixed packed collectives. Each rank backprops
`world_size * 4/T * sum(F_global * S_local)`, value-corrected to the global
balance. DDP averaging produces the global parameter gradient; unreduced local
input gradients are world-size times global-reference slices. Full global CE
is logged separately using the existing schedule gather, without changing the
ordinary local-mean CE backward or adding fake control routing work.

Target benchmarks require, per arm:

- Fresh matched first data fingerprints; 20 warmups and 4,200 real-stream
  attempts in 1,000/1,000/1,000/1,000/200 windows. Timing includes auxiliary,
  dispatch, data, DDP and optimizer costs. Slowest rank/window drives projection.
- Full-128 global-512 training stress (10 warmups/40 measured); 128k streaming
  FP32 validation; full-128 batch-512 validation stress; FP32 batch-256 inference
  and batch-one latency; actual restart/weight I/O and full-population metric cost.
- Path-aware observed precision; all states finite; measured AMP skip fraction
  below 5%; zero valid assignment drops; 10% allocated GPU-memory headroom.
- Independent FP64 per-particle routing/expert/loss oracles, exact routes/counts,
  native trim/call/RNG/shared-CE parity with active State-EV/head/residual scales,
  active checkpointed/uncheckpointed forward/input/parameter gradients.
- Actual SPORC SyncBN/DDP global-model parity plus replicated-bank gradients
  with unequal counts, local/global empty experts and upstream shared layers.
  GH200 checks the corresponding one-GPU path. CPU/Gloo tests do not replace these.
- Real Ranger activation: eight successful updates within twenty attempts;
  separate deterministic CE-only coverage of all 24 expert transforms and
  separately measured router CE/auxiliary gradients.
- Seven rolling-update reference/repeat/disk/resume trials with exact immediate
  state/RNG/counters, exact route digests/requested/executed counts/input/LR/
  skip/scaler schedules, fixed bounded subsequent floating state/losses and
  recorded margins/worst error paths. Probe-only separated router biases make
  this declared fixture nondegenerate; production routing is never changed.
- Actual ROOT iterator recreate/replay fingerprints and restored rank RNG,
  coordinated bad-input/nonfinite/bad-stat/I/O/stop rejection, bounded profiles
  (2 jets/17 particles) and diagnostics (8 jets/17 particles). Profiles cover
  context/router, stable selection, packing, expert/recomputation and combination.

Diagnostics preserve weights/buffers/modes/RNG/trimmer/caches/gradients. Usage,
entropy, margins, correction/dense ratios and CE-versus-auxiliary norms are
reported, not used to adapt losses, choose routes or drop tokens.

## Storage and isolation

New outputs live only under `checkpoints/part_ev_ffn_experts/<fresh-name>/`:

```text
source/ + source_manifest.json + snapshot.sha256 + launch.json
campaign.json + data_contents.json + benchmark.json + summary.json
logs/ + probes/ + pipeline_submissions.json + pipeline_job_ids.tsv
production.json + production_submissions.json + production_job_ids.tsv
production/campaign.json
production/runs/<run-id>/seed_101/
  optimizer.json + last.pt + selected/check_XX.pt + best_model_val.pt
  validation/check_XX.json + training_complete.json
  full_validation/file_XXX.npz + .json + metrics.json
production/reports/experts_report.json + .md
```

At most 50 improving selected candidates, final best and one complete latest
restart plus atomic-write allowance are retained per arm. There are no ROOT
copies, learned-feature caches, persistent per-particle routing tables or new
conda environments. The benchmark has a separate bounded 2-GiB slot.

As requested, **other campaigns' reservations are advisory, not a hard gate**.
The local gate is this campaign's remaining budget plus a 4-GiB shared emergency
reserve: initially about **11.59 GiB** for a paired pipeline, or 6 GiB for
benchmark-only. The two paired platform outputs together have about 19.19 GiB
prospective initial planning including one shared reserve; that is not an
enforced aggregate quota. Other jobs may consume space; monitor
`df -h /home/ryreu/atlas` and manage cleanup yourself. No data/environments/checkpoints
are deleted, no old ledgers are released and no frozen policy is rewritten.

Shared submission locks and durable pre-sbatch intent prevent blind duplicates.
Unknown replies or partial chains require reconciliation with the recorded
cluster/job IDs. Never blindly repeat with a fresh timestamp. Same-platform
production-owner records prevent duplicate comparisons; the other platform has
a separate owner and output tree.

## Recovery, evaluation and interpretation

Training commits full native/new model, Ranger moments/slow weights/cache/
counter, scaler, all rank RNG/trimmer, attempted/validated cursors, selected
candidate/hash, timing and detached cumulative route/loss counters at least
every 1,000 attempts, around validation and on coordinated stop. Wrong source,
platform, world size, BN, loss policy, missing expert state or malformed counters
reject recovery. No per-forward routing graph needs serialization.

USR1/TERM and a monotonic pre-walltime deadline stop at safe boundaries; latest
complete state is retained and incomplete jobs exit nonzero, blocking afterok
evaluation. No automatic extra allocations or split chains. Reviewed same-platform
recovery rebuilds/replays both continuous streams, restores RNG, then continues.
Replay itself takes allocation time and is interruptible.

After inspecting accounting and cancelling only obsolete dependent evaluation/
report jobs for the affected arm, an explicit recovery command is:

```bash
bash sbatch/submit_part_ev_ffn_experts.sh \
  --mode resume --output "${EXPERTS_ROOT}" --arm experts --reviewed
```

Use `--arm control` for the control. Prior arm/report accounting must be terminal;
uncertainty stops rather than creating a duplicate. Recovery authenticates the
checkpoint, skips already completed training and relinks evaluation/report.
Nothing automatically cancels or changes the other arm's jobs.

Completion requires 1M attempts **and** 50 committed checks. Selection is
transactional and audited against the first maximum accuracy and checkpoint
hash. Final evaluation authenticates/resumes bounded prediction chunks across
all 50 official validation ROOT files. Reporting recomputes metrics/hashes,
checks paired event order and shows accuracy/AUC/rejections with counts,
learning curves, global route usage, separate CE/balance/total means, timing,
AMP skips, measured peaks and actual scheduler GPU-hours when available.

FLOP convention is 2 FLOPs/MAC. The added main linear-work calculation is about
0.2204 GFLOPs per 128-valid-particle jet, excluding nonlinearities, dispatch,
native network, backward and recomputation. Bounded profiler FLOPs are explicitly
partial visible-operator counts, not total model or corpus-training compute.
Do not infer speed or novelty from stored parameters, usage plots or historical
validation. Parameter/compute controls and extra matched seeds remain follow-up.

## Local verification

Verified locally on Windows / PyTorch 2.12.1 CPU. The combined experts,
workspace and selected backbone/launcher/precision/full-schedule regressions
passed **214 tests with 6 existing GPU/environment-dependent skips**. The
experts-specific suite has 64 tests, including real two-process CPU/Gloo and
instrumented final-boundary/first-tie/report tests. CPU AMP emulation is not a
substitute for CUDA admission. Both read-only platform plans and Bash syntax
also pass; no real Slurm commands, RC dataset reads, GPU benchmarks, environment
changes, deletions or production submissions were performed.

```powershell
.venv\Scripts\python.exe -B -m pytest tests/test_part_ev_ffn_experts_math.py tests/test_part_ev_ffn_experts_model.py tests/test_part_ev_ffn_experts_runtime.py tests/test_part_ev_ffn_experts_protocol.py tests/test_part_ev_ffn_experts_launcher.py tests/test_part_ev_ffn_experts_status.py tests/test_part_ev_ffn_experts_coordination.py tests/test_part_ev_ffn_experts_production.py -q
```

These tests exercise actual CPU math/Ranger/restart/precision and two-process
Gloo gradient/empty-expert behavior, plus synthetic raw-evidence rejection and
fake Slurm. They do not submit real jobs, open RC data, establish NCCL/SyncBN
CUDA behavior, measure GH200/A100 runtime, or establish classification gains.
