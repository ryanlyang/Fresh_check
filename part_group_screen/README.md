# Grouped-attention screen

This is the implementation of the [study](../part_ev_fullscale/GROUPED_ATTENTION_STUDY_PLAN.md)
and [implementation plan](../part_ev_fullscale/GROUPED_ATTENTION_IMPLEMENTATION_PLAN.md).
It uses the [Reusable Screen Protocol](../part_ev_screen/REUSABLE_SCREEN_PROTOCOL.md)
without changing the original seven-model screen or its active campaigns.

Status: **locally implemented and tested; ready to queue preparation and GPU admission**.
Real donor compatibility, A100 throughput, FP16 stability, memory, and production
walltimes must still pass the queued checks. No cluster jobs were submitted during
implementation. CPU tests are not a substitute for GPU admission.

## What runs

| Array index | Run ID | Assignment | Parameters |
|---:|---|---|---:|
| 0 | `PART_GROUP_FIXED_RANGER` | Fixed angular agglomeration | 2,368,127 |
| 1 | `PART_GROUP_LEARNED_RANGER` | Learned competitive soft groups | 2,518,271 |
| 2 | `PART_GROUP_HYBRID_RANGER` | Learned groups with angular locality | 2,518,271 |
| 3 | `PART_GROUP_RANDOM_RANGER` | Random memberships, fixed-arm group sizes | 2,368,127 |

All four train **from scratch**, seed 101, under the complete donor
`ranger_short_v1` recipe: 200,000 attempted updates, batch 512, one compatible
SPORC A100, no accumulation, ten checks of 1.28M validation jets, then the
accuracy-selected checkpoint on all 5M validation jets. Training uses all official
100M training files, not a new small subset. The official 20M test is not used.

`PART_RANGER` (2,143,486 parameters) is a required **result reference**, never a
training/evaluation array entry. `EV_RANGER` (2,209,022) can optionally be included
as another reference. These are not the earlier Tigris full-budget or small-data
controls. The default donor is:

```text
/home/ryreu/atlas/Fresh_check/checkpoints/part_ev_screen/context_gate_ampfix_20261016T215608Z
```

Pending donor results do not prevent admitted new training. Missing or incompatible
control evidence prevents the completed comparison. No donor weights initialize a
new model; no completed donor inference is repeated. The tiny fresh-initialization
ParT parity probe is a correctness check, not another control training run.

## Queue on SPORC

First review, commit, and push the implementation from your local checkout, then
pull it on RC. The launcher intentionally rejects uncommitted relevant changes;
do not discard other work to get past that check. Keep the donor environment and
frozen source unchanged. This workflow does not install packages or modify them.

On **sporcsubmit**, using the donor-compatible existing environment:

```bash
cd /home/ryreu/atlas/Fresh_check
git pull --ff-only

GROUP_ROOT="/home/ryreu/atlas/Fresh_check/checkpoints/part_group_screen/groups_$(date -u +%Y%m%dT%H%M%SZ)"
GROUP_DONOR="/home/ryreu/atlas/Fresh_check/checkpoints/part_ev_screen/context_gate_ampfix_20261016T215608Z"

bash sbatch/submit_part_group_screen_sporc.sh \
  --mode prepare --output "${GROUP_ROOT}" --donor "${GROUP_DONOR}" \
  --partition tier3 --concurrency 2 --dry-run

bash sbatch/submit_part_group_screen_sporc.sh \
  --mode prepare --output "${GROUP_ROOT}" --donor "${GROUP_DONOR}" \
  --partition tier3 --concurrency 2
```

Save the printed campaign path. After reconnecting, set `GROUP_ROOT` to that
**same existing path**, not another timestamp. The default Python is
`/home/ryreu/miniconda3/envs/part_ev_screen_sporc/bin/python`; override `--python`
at preparation only if that is not the donor's actual environment. Account defaults
to `reu-aisocial`. Requested GPU defaults to `a100`, 8 CPUs and 64 GiB host memory.
The partition must exist and you must have permission to use it; scheduler rejection
does not trigger a fallback to debug or Tigris.

Add `--include-ev-control` to **both prepare commands** above if you want verified
EV comparisons too. It adds no new jobs but makes EV results required for the
final report. The initial registry always contains all four grouped arms.

Preparation freezes a committed source snapshot and checks source, recipe,
environment, data metadata, donor evidence, and at least 4 GiB free artifact space.
It queues one CPU prepare job followed by a four-task GPU benchmark array.
Each pre-production job requests **two hours**, below the six-hour limit.

After all four benchmarks complete:

```bash
bash sbatch/submit_part_group_screen_sporc.sh \
  --mode review --output "${GROUP_ROOT}"
```

Review parity, actual activation precision, data fingerprints, AMP skips,
full-length memory, and sustained timing. A failed check cannot be approved by
adding a flag. The benchmarks use 20 warmup + 4,200 measured updates, three
worst-length steps at batch 512, 128,000 validation examples, and checkpoint I/O.
All production models/scalers/optimizers start fresh after admission.

If that evidence passes and is acceptable:

```bash
bash sbatch/submit_part_group_screen_sporc.sh \
  --mode train --output "${GROUP_ROOT}" --reviewed --concurrency 2
```

This queues the entire downstream chain:

```text
train[4] -> selected-checkpoint lock -> full-validation[4] -> comparative report
```

Training/evaluation walltimes come from the slowest measured arm with the inherited
30% planning margin, not guessed constants. `--train-hours N` can increase, but
not undercut, the recommendation. Restart stream replay and future filesystem
contention can add time; the projection is not a guarantee. Concurrency can be
1 through 4 without changing the recipe. Do not submit a second copy to make an
already-active array more concurrent.

## Read-only progress and outputs

```bash
cat "${GROUP_ROOT}/submitted_job_ids.tsv"
squeue --me -o "%.18i %.32j %.2t %.10M %.12l %R"

bash sbatch/submit_part_group_screen_sporc.sh \
  --mode status --output "${GROUP_ROOT}"
```

Status reads all available new-model and requested donor check records. It shows
every recorded check, their common contiguous cutoff, and best accuracy **through
that cutoff**, including the matched difference from ParT. It does not deserialize
checkpoints, refresh registries, write donor files, or verify final results anew.

Key artifacts:

```text
experiment_spec.json                 frozen architecture/data/recipe/runtime/source
controls.json                        pending/verified/incompatible donor references
control_validation_history/          immutable refresh records
benchmarks/<RUN_ID>.json              new-model admission measurements
production_approval/approval.json     explicit reviewed admission and evidence hashes
runs/<RUN_ID>/seed_101/
    last.pt                          complete resume state, not model selection
    best_model_val.pt                highest validation accuracy; first exact tie
    validation/check_*.json
    diagnostics/check_*.json
    training_complete.json
    full_validation/file_*.npz        FP32 logits and labels
    full_validation/file_*.json       input-file/entry/checkpoint identity and hash
    full_validation/metrics.json
selection/locked_checkpoints.json
comparison_lock.json                 verified donor/new inputs to final comparison
reports/grouped_screen_report.json
reports/grouped_screen_report.md
reports/learning_curves.pdf
reports/grouping_diagnostics/*.pdf
logs/grp_<phase>_<job>_<task>.out/.err
```

The report includes accuracy, macro OVO and OVR AUC separately, cross-entropy,
QCD rejection, paired accuracy counts, and measured costs. Contrasts are each new
model versus ParT, fixed versus random, hybrid versus learned, and optionally
versus EV. A comparison against EV is **not** grouping added to EV. Zero accepted
QCD is recorded without inventing a finite rejection.

Rejection uses the existing evaluator: 30%/50% points for the hadronic signals,
99% for Hqql and 99.5% for Tbl. Group diagnostic plots include occupancy, entropy,
effective count, angular width, slot similarity, gamma, correction magnitude,
pT share, and bounded assignment examples. Diagnostics do not select checkpoints.

## Pending controls and recovery

If preparation fails with `Invalid metadata hash/schema` for the donor's
`production_approval/approval.json`, first inspect the prepare stderr. The original
grouped reader incorrectly used the compact campaign JSON hash encoding for a v1
recovery approval, whose producer uses default (spaced) JSON separators. The reader
now validates that approval with its producer's encoding; campaign hashes and all
approval policy/source/evidence checks remain enforced. Regression coverage uses
the real approval writer and rejects tampered and wrongly encoded approvals.

For a campaign frozen with the old reader, commit/push the fix and pull it on RC,
then run `--mode prepare` with a **new output path** after the failed campaign's
dependent jobs are terminal. `--resume` on the old path still runs the old frozen
reader. Preserve the old logs; do not rehash donor files, remove its approval, or
edit either frozen source snapshot. Preparation failure before
`experiment_spec.json` means `--mode review` cannot run yet; inspect the prepare
logs instead. No grouped training has started at that stage.

If required controls are still pending when the report starts, it writes
`reports/progress.json`, exits with code 3, and does not publish a completed
comparison. Once their own workflow has finished, and no grouped jobs are active:

```bash
bash sbatch/submit_part_group_screen_sporc.sh \
  --mode refresh-controls --output "${GROUP_ROOT}"
```

Wait for that CPU job to finish, inspect `controls.json`, then:

```bash
bash sbatch/submit_part_group_screen_sporc.sh \
  --mode report --output "${GROUP_ROOT}" --resume
```

Reporting also revalidates donor references itself. Corruption or incompatible
contracts are errors, not pending completion. No read-only control needs to be
retrained by this launcher. An optional EV reference chosen at prepare cannot be
silently dropped later to produce a favorable comparison.

For terminal interrupted stages, inspect logs/ledger and wait for any dependent
jobs to terminate before retrying the relevant command:

```bash
# Terminal/incomplete probes (passing probe artifacts are reused):
bash sbatch/submit_part_group_screen_sporc.sh \
  --mode benchmark --output "${GROUP_ROOT}" --resume

# Interrupted production: resumes checkpoints and queues lock/evaluation/report:
bash sbatch/submit_part_group_screen_sporc.sh \
  --mode train --output "${GROUP_ROOT}" --reviewed --resume

# Training complete but evaluation interrupted:
bash sbatch/submit_part_group_screen_sporc.sh \
  --mode evaluate --output "${GROUP_ROOT}" --resume
```

Successful training tasks are checked and skipped on retry. Resume restores model,
Ranger slow buffers/counter, scaler, RNG, and native trimmer counters, replaying
the deterministic data cursors before continuing. Evaluation reuses verified
committed prediction files; a file missing its commit sidecar is recomputed.
Committed hash mismatches stop the run instead of overwriting evidence.

A **recorded failed numerical benchmark** is preserved and requires investigation
and a separately identified corrected campaign. No code path widens tolerances,
changes precision/batch size, or promotes probe weights to production. Do not patch
frozen snapshots. After a source/configuration change, prepare a new output path.
The submitter rejects active duplicates and never cancels jobs or deletes data.

## Frozen architecture and routing details

The native shared-bias ParT embedding, eight particle blocks, pairwise encoder,
two class blocks, and classifier remain unchanged. After particle block four,
the branch builds up to eight group tokens from the **actual native-trimmed**
constituents, processes them with one width-128/eight-head attention block and
128-to-256-to-128 FFN, then returns information by particle-to-group cross-attention:
`H + gamma * correction`, initially `gamma=0.001`. The added dropout is zero;
native dropout remains unchanged.

The common builder combines assignment-weighted hidden means with seven physical
descriptors (relative pT, mass, relative rapidity, sine/cosine azimuth, occupancy,
angular width). Sensitive geometry, routing scores/softmax, and reductions use
FP32; learned projections use the donor FP16 training context. Parameters and
validation remain FP32. Floors/initialization/settings live in `config.py` and the
frozen source. There is no fitted normalization or extra loss.

- **Fixed:** repeatedly merge the closest rapidity-azimuth cluster pair, add its
  four-vectors, and stop at `min(8,N)`. This is an explicit fixed-count angular
  tree, not an inclusive radius-based clustering substitution. A batched device
  implementation is tested against a small independent CPU oracle.
- **Learned:** softmax across groups for each particle, three assignment rounds
  and two shared GRU updates, using contextual block-four hidden states. Final
  assignments, not hidden slot states, enter the common builder.
- **Hybrid:** identical learnable shapes/initialization; the first round is fully
  learned, then scores receive an angular penalty from the previous assignments'
  weighted centers. Locality is 1, radius floor 0.05. Setting locality to zero
  yields exactly the learned router.
- **Random:** same occupancy multiset as fixed groups, but memberships are
  stateless hash-ordered, using seed 731019. No training/global RNG is consumed.

**Resolved metadata choice:** native trimming does not expose original constituent
indices. Instead of changing its loader/trimmer, canonical keys use the actual
post-trim 17 input features and four-vector FP32 bits. A jet key is an
order-independent combination of constituent hashes; ties use the full canonical
input ordering. No labels, filenames, batch positions, or learned hidden features
enter the keys. The same retained input set has the same partition under worker,
restart, or constituent-order changes. A changed retained set from native trimming
can change the partition. Exactly identical constituents are interchangeable;
tests compare physical membership as well as numerical outputs. This is not
assignment by unstable array index, nor a claim of cross-platform bitwise parity.

Padded rows/slots are masked. Low-multiplicity nonempty jets are supported; empty
jets are rejected. Soft assignments have no occupancy-pruning threshold, so no
assignment mass is silently discarded. Common branch modules initialize identically
across arms, and added initialization leaves both CPU and CUDA RNG untouched.

## Verification and limits

Local tests use CPU PyTorch, synthetic data, genuine native models/optimizers, and
a fake Slurm scheduler. They cover initialization/parity, geometry/gradients,
grouping/permutation, donor contracts, pending controls, precision interfaces,
checkpoint/resume, checkpoint ties, per-file evaluation recovery, report joining,
plots, source freezing, admission, dependency graphs, and duplicate guards.
The original screen/full-scale regression tests are run separately.

Verification recorded on 2026-10-17 (Windows, Python 3.14.3, PyTorch 2.12.1+cpu):

| Check | Observed result |
|---|---|
| Combined grouped/control and original screen/full-scale suites | 189 passed, 6 CUDA-only skipped |
| Final protocol recheck, including the added no-write donor-destination guard | 11 passed |
| Both Bash scripts: syntax; actual prepare dry-run; mocked Slurm graph | Passed; no real submissions |
| Python compile and 3.10 syntax compatibility | Passed for all 16 new runtime Python files |

The combined command was the new-test command below plus
`tests/test_part_ev_screen_{protocol,model,data,precision,training,reporting,launcher}.py`
and `tests/test_part_fullscale_{model,training}.py` (brace expansion in Bash).
The final path-guard test was added after that combined run and is included in
the subsequent protocol recheck. This is local evidence, not a claim that the
SPORC production Python/CUDA stack has already passed.

```bash
python -m pytest tests/test_part_group_model.py tests/test_part_group_training.py \
  tests/test_part_group_protocol.py tests/test_part_group_reporting.py \
  tests/test_part_screen_controls.py tests/test_part_group_launcher.py -q
```

The CUDA-specific RNG test skips without CUDA. Actual batch-512 FP16/memory/input
parity and timing are checked by each queued A100 benchmark, not certified by the
local test result. Tests can shrink budgets internally; production campaign
validation rejects those fixture recipes.

Dataset evidence is the donor's immutable file/entry inventory plus matching paths,
sizes, and mtimes, **not** a SHA-256 of each ROOT payload. Same-location metadata is
rechecked during prepare and by the reused loaders. Dataset relocation/content
changes need explicit verification; they are not accepted by changing a path flag.
Full prediction files and sidecars are checksummed for result reuse.

Four raw 5M-by-10 FP32 prediction sets require 0.8 GB before labels/checkpoints;
prepare requires 4 GiB free including temporary writes/headroom. Donor predictions
are referenced, never copied. There is no per-training-jet grouping cache.

These are single-seed **screening validation** comparisons, not independent test
or multi-seed significance estimates. Later multi-scale grouping, new seeds,
AdamW, capacity controls, or full-budget training are separate experiments; none
is silently added to this queue.
