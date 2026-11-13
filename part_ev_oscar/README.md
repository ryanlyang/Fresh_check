# Independent Oscar full-test evaluation

Evaluate **standard shared ParT** (seed 101, selected check 46) and **standard
all-eight layerwise State-EV** (seed 101, selected check 49) on the official
20M JetClass test population. No Selected relations, last-four variant, compact
variant, retraining, checkpoint reselection, or changes to existing campaigns.

The test archive has already been uploaded and extracted on Oscar:

```
/oscar/scratch/rlyang/part_ev_test/incoming/JetClass_Pythia_test_20M.tar
/oscar/scratch/rlyang/part_ev_test/data/JetClass/Pythia/test_20M
```

Approved archive SHA-256:
`0545be548366fbc8c4131ac8147c87660bdc038ad35189e01ec4867ab6db76a4`.
The new preparation phase independently verifies every extracted file against
the tar payload and checks exactly 200 official filenames and 100,000 entries
per ROOT TTree. It never deletes or extracts anything.

## Isolation and scientific contract

- Existing RIT evaluators are **read-only dependencies**, copied into an
  independent tool snapshot. Each RIT export uses the original frozen source,
  interpreter, CPU-readable environment and recorded hardware-admission checks,
  selected-checkpoint audit, and all 50 validation records. No new GPU hardware
  check or model inference runs on RIT; GPU parity is checked on Oscar.
- The two original campaigns and the saved validation evaluation must be
  inactive. Nothing is cancelled automatically.
- Exports retain original provenance and checkpoint bytes, plus a portable
  tensor-only state dictionary (all parameters **and buffers**). Oscar loads
  that dictionary with `weights_only=True`. No optimizer/restart state or training
  dataset needs to be transferred. Only transfer bundles produced by your trusted
  RIT jobs: they contain executable model source, not untrusted downloads.
- Each model has its own frozen donor source tree. One model is loaded per fresh
  process, so historical Python namespaces cannot collide.
- RIT exports 512 **scored** validation events from each class (5,120 total),
  raw TTree prefixes with padding context, exact preprocessed inputs, labels,
  and FP32 logits. Weaver's `wrap` padding tiles the constituent values of the
  whole fetched file, not each jet independently. Each prefix therefore retains
  at least the first `512 * 128` constituent values (or the whole original file
  if shorter), even if that requires more than 512 raw jets. The manifest records
  `raw_entries` and `padding_context`. Extra context jets are preprocessed but
  **not scored**. All first-512 tensors, including masked padding, must still
  match the original full-file loader exactly.
- Reference logits come from the **already completed full-5M GPU evaluation**,
  not a CPU approximation. CPU export verifies all 50 prediction payload hashes
  and sidecars per model, their selected checkpoint, FP32/batch-512 contract,
  labels, coverage, and aggregate accuracy. It retains only the ten 512-event
  logit slices. Missing, modified, or ambiguous references block export; there
  is no automatic GPU fallback or relaxed parity tolerance.
- Oscar processes the raw probes with the **same loader used for test inference**.
  Input tolerances are rtol=1e-6, atol=1e-6. Logit tolerances are rtol=1e-4,
  atol=2e-5, with maximum absolute error <=1e-3 and **zero probe argmax changes**.
  These are predeclared engineering checks, not a proof of identical test metrics.
  Failure blocks evaluation; there is no tolerance-widening/retry switch.
- FP32, no AMP, no TF32, no `torch.compile`, no quantization. Default batch 512;
  64/128/256 are explicit alternatives **before admission**, not automatic OOM
  fallbacks. Native BatchNorm and saved SyncBatchNorm remain in evaluation mode.
- Donor Python major/minor, scientific package versions, CUDA build version, and
  critical Weaver source hashes are checked. Python patch versions and the
  PyTorch packaging `+cu...` suffix may differ; full actual versions are recorded.
  An admitted campaign subsequently requires the same recorded environment,
  GPU model/capability, driver version, precision, and batch size on every worker.
- Every inference worker repeats validation parity on its allocated node before
  reading test events. Test labels are used for final metrics, never model tuning.
- Each 100k-event prediction file has a SHA-256, label hash, event/file identity,
  precision audit, and atomic commit sidecar. Retry reuses verified committed
  files. Orphan prediction files without sidecars are recomputed. Missing or
  modified committed files fail closed. Per-model locks prevent simultaneous writes.
- Pulling later repo changes does not alter queued jobs: workers use explicit
  frozen `--tool-root` paths, including when Slurm runs its spool-directory copy.

## 1. CPU-only export on SPORC (not Oscar)

Commit/push the new files locally, then pull on RIT. Keep the original environment
and source snapshots unchanged. The new source directory contains only the
explicit tool manifest; unrelated untracked research plans are not copied.

```bash
cd /home/ryreu/atlas/Fresh_check
git pull --ff-only

export EV_EXPORT="/home/ryreu/atlas/Fresh_check/checkpoints/part_ev_oscar_exports/test_pair_cpu_$(date -u +%Y%m%dT%H%M%SZ)"

python3 -B scripts/part_ev_oscar.py export-plan --output "$EV_EXPORT"
python3 -B scripts/part_ev_oscar.py export-submit --output "$EV_EXPORT"
```

`export-plan` is read-only and inspects metadata, not the 5M prediction payloads.
Check that it reports two **CPU-only** jobs and the intended completed reference.
`export-submit` snapshots the tooling and submits two independent jobs: `tier3`,
account `reu-aisocial`, 4 CPUs, 64 GiB, **2h**, **zero GPUs** each. CPU jobs still
wait for scheduler resources; the 2h limit is not a measured duration. Use
`--cpu-partition debug` on both commands if appropriate for your allocation.
`--gres` is rejected. The worker hides CUDA devices and uses each donor's
recorded Python; the submitter itself only needs Python >=3.9. Default donors:

```
part_ev_state_fullscale/state_full_20261024T185049Z
part_ev_layerwise_ddp/layerwise_2g_20261029T162243Z
```

Paths are under `/home/ryreu/atlas/Fresh_check/checkpoints`. `--part`/`--state`
can identify relocated original campaigns; the checkpoint/architecture checks
remain mandatory. Original donor audits may still require original dataset
metadata to be present on RIT; no such training-data dependency is transferred.

The default reference discovery requires exactly one matching completed pair
under `checkpoints/part_ev_rejection_eval`. If there are multiple candidates,
identify the one from jobs **22082035/22082036**, then pass its exact directory
with `--validation-results` to both `export-plan` and `export-submit`. For example:

```bash
EV_VALIDATION_LOG="$(find /home/ryreu/atlas/Fresh_check/checkpoints/part_ev_rejection_eval \
  -maxdepth 3 -type f -name 'evrej_part_22082035.out' -print -quit)"
if [[ -n "$EV_VALIDATION_LOG" ]]; then
  EV_VALIDATION="$(dirname "$(dirname "$EV_VALIDATION_LOG")")"
  python3 -B scripts/part_ev_oscar.py export-plan --output "$EV_EXPORT" \
    --validation-results "$EV_VALIDATION"
fi
```

This only inspects the plan. Add the same `--validation-results "$EV_VALIDATION"`
argument to `export-submit` after checking it. Reference predictions and original
ROOT validation files must still be present on RIT. They are read, never modified.

After both exports complete:

```bash
python3 -B "$EV_EXPORT/source/scripts/part_ev_oscar.py" export-review --output "$EV_EXPORT"
```

This verifies both exports and writes the bundle commit marker `bundle.json`.
Do not transfer an incomplete export. If submission itself was interrupted,
`export-resume --output "$EV_EXPORT"` submits missing exports once all recorded
jobs are inactive. An interrupted **partially written model export** is preserved
for diagnosis and requires a fresh export directory; no recursive cleanup is done.

### Recovery from the original GPU export failure

The original exporter could fail with `Validation ROOT slice changed
preprocessing/order`: cutting to exactly 512 raw jets changed file-wide wrap
padding. This was an export-probe bug, not evidence of changed model weights.
The corrected exporter preserves the context above; it does not modify the
official loader, model, checkpoint, or equality/parity tolerances. Regression
tests use nonconstant, variable-length constituents (constant fixtures had
hidden the difference).

Use the CPU commands above with a **new** export root, not `export-resume` on
the old GPU export or a partially written bundle. The new launcher does not
convert an old frozen GPU plan in place. It does not need a replacement SPORC
GPU job: the completed validation predictions supply the reference logits.
The old export, trained checkpoints, saved evaluation, and unrelated queued
jobs remain unchanged. Only finalize/transfer after both CPU exports succeed.

## 2. Transfer the complete bundle (WSL on your computer)

Replace the example export basename with the exact directory printed above.
Keep the trailing slash on the source directory. Do not use `--delete`.

```bash
EV_EXPORT_NAME="test_pair_cpu_REPLACE_WITH_EXPORTED_TIMESTAMP"
LOCAL_BUNDLE="/mnt/c/Users/22rya/ComputerScience/CERN/a_download_checkpoints/oscar_test_pair/${EV_EXPORT_NAME}"
mkdir -p -- "$LOCAL_BUNDLE"

rsync -rt --partial --inplace --checksum --info=progress2 \
  "ryreu@sporcsubmit.rc.rit.edu:/home/ryreu/atlas/Fresh_check/checkpoints/part_ev_oscar_exports/${EV_EXPORT_NAME}/" \
  "$LOCAL_BUNDLE/"

ssh rlyang@ssh.ccv.brown.edu \
  'mkdir -p /oscar/scratch/rlyang/part_ev_test/models/rit_pair'

rsync -rt --partial --inplace --checksum --info=progress2 \
  "$LOCAL_BUNDLE/" \
  rlyang@ssh.ccv.brown.edu:/oscar/scratch/rlyang/part_ev_test/models/rit_pair/
```

Resume an interrupted transfer with the same command. Password/Duo stays
interactive; no credentials are saved by these tools. The bundle has source,
weights, metadata, and small validation probes, not the 100M training dataset.
All transferred artifacts are rehashed before use on Oscar.

The raw validation probes are already inside `rit_pair/part/probes/` and
`rit_pair/state/probes/`, so this transfer puts them in **Oscar scratch too**.
Do not upload the entire 5M validation set or its saved prediction files. Only
their small audit metadata and selected logit slices are included. Context
prefix sizes depend on constituent multiplicity; inspect the completed bundle
with `du -sh -- "$EV_EXPORT"` on RIT before transferring.

## 3. New scratch environment on Oscar

Do **not** copy a Conda environment directory from RIT. Choose an available
Python matching the exported major/minor (expected 3.10), via Oscar modules or
an existing Python installation. First locate it, e.g. `command -v python3.10`
or `module spider python`. Pass its actual executable below. No module name
or GPU-account membership is assumed by the tooling.

```bash
OSCAR_ROOT=/oscar/scratch/rlyang/part_ev_test
BUNDLE="$OSCAR_ROOT/models/rit_pair"

bash "$BUNDLE/source/scripts/setup_part_ev_oscar.sh" \
  --bundle "$BUNDLE" \
  --base-python /ABSOLUTE/PATH/TO/python3.10 \
  --venv "$OSCAR_ROOT/envs/part_ev_test_v1"
```

The setup refuses an existing environment. It derives pinned packages and the
official PyTorch CUDA wheel index from the export, installs only into the new
venv, and redirects pip caches/temp storage to scratch. If the donor wheel is
unavailable or installation fails, retain the log and use a new environment
path after resolving it; do not weaken the dependency/parity checks.

## 4. Prepare and admit Oscar, without test inference

```bash
OSCAR_ROOT=/oscar/scratch/rlyang/part_ev_test
BUNDLE="$OSCAR_ROOT/models/rit_pair"
EV_OSCAR_PYTHON="$OSCAR_ROOT/envs/part_ev_test_v1/bin/python"
export EV_TEST="$OSCAR_ROOT/results/test_$(date -u +%Y%m%dT%H%M%SZ)"

"$EV_OSCAR_PYTHON" -B "$BUNDLE/source/scripts/part_ev_oscar.py" init \
  --bundle "$BUNDLE" --output "$EV_TEST" --python "$EV_OSCAR_PYTHON" \
  --partition gpu --gres gpu:nvidia_rtx_a6000:1 \
  --batch-size 512 --evaluate-hours 24

"$EV_OSCAR_PYTHON" -B "$EV_TEST/source/scripts/part_ev_oscar.py" prepare-submit \
  --output "$EV_TEST"
```

Default GPU type comes from your Oscar `sinfo`: RTX A6000. Availability is not a
guarantee of immediate scheduling. `--account` is optional; Oscar uses your
default association. Change partition/GRES at `init` only if your account has
access. GPU jobs need one visible GPU; there is no DDP or global queue throttle.
If using another GPU type, it must support the pinned CUDA/PyTorch environment.

`init` submits nothing. `prepare-submit` queues:

| Phase | Resources | Requested limit | Dependency |
|---|---|---|---|
| Test inventory/byte verification | CPU, 4 CPUs, 64 GiB | 2h | None |
| ParT validation parity | 1 GPU, 4 CPUs, 64 GiB | 2h | Preparation |
| State-EV validation parity | 1 GPU, 4 CPUs, 64 GiB | 2h | Preparation |

No test predictions are produced in these phases. Both parity jobs can run
independently. A failed dependency blocks downstream jobs. Failed parity is
retained; investigate it rather than resubmitting production.

After completion:

```bash
"$EV_OSCAR_PYTHON" -B "$EV_TEST/source/scripts/part_ev_oscar.py" review --output "$EV_TEST"
```

Review reports actual parity errors (full details in `parity_part.json` and
`parity_state.json`), memory, and small-probe timing. Timing includes loading the
small files and is **not** a guaranteed 20M runtime estimate. Default evaluation
walltime is a provisional 24h allocation, not an observed Oscar benchmark result.
Use a fresh `init` if resource/batch settings need changing before inference.

## 5. Run the two full-test evaluations

Only after review passes:

```bash
"$EV_OSCAR_PYTHON" -B "$EV_TEST/source/scripts/part_ev_oscar.py" evaluate-submit \
  --output "$EV_TEST" --reviewed

squeue --me -o '%.18i %.30j %.12T %.12l %R'
```

This writes a test-admission record and queues two independent 24h GPU jobs,
a CPU metrics job after **each** model (4 CPUs, 64 GiB, 5h), and a final CPU report
after both metric jobs (1h). There is no dependency between the two GPU evaluations.
All caches, temporary files, predictions, and logs remain in the scratch campaign.

```bash
"$EV_OSCAR_PYTHON" -B "$EV_TEST/source/scripts/part_ev_oscar.py" status --output "$EV_TEST"
cat "$EV_TEST/report.md"
```

If interrupted, inspect the logs and `jobs.json`, wait for that campaign's jobs
to become terminal, then:

```bash
"$EV_OSCAR_PYTHON" -B "$EV_TEST/source/scripts/part_ev_oscar.py" resume \
  --output "$EV_TEST" --reviewed
```

The 5-minute pre-timeout signal stops between batches (exit 75); the unfinished
ROOT file is recomputed next time, while verified complete files are reused.
Real failures also block dependent jobs. `resume` does not cancel or silently
change precision/batching. Completed inference gets only CPU result audits/metrics,
not another GPU allocation. A changed GPU/driver/environment blocks continuation
instead of mixing numeric runtimes inside one prediction population.

## Results and storage

`report.md` and sealed `report.json` contain full-test accuracy, global macro
OVO and OVR AUC, accuracy differences in percentage points, raw AUC differences,
and QCD rejection/change percentages. Rejection includes 30% and 50% for all
signals, plus Hlvqq/Hqql at 99% and Tblv/Tbl at 99.5%. Thresholds use sorted
signal-vs-QCD log odds, include ties, and retain acceptance counts/achieved
efficiencies. Zero QCD acceptances are `null`/`ZB`, never measured infinity.
These are ROC operating-point estimates on test, not fixed deployment cuts.

Metrics are calculated over all 20M events, not averaged over chunks. Both
models must have the same file/entry ordering and label hashes. There is no
automatic "winner" selection or tuning based on the test results. It remains
a single-seed comparison with the one-GPU/native-BN versus two-GPU/SyncBN
training confound explicitly noted.

Two raw prediction sets require approximately 1.92 GB total (float32 10-class
logits plus int64 labels). AUC is CPU work and requests ample host RAM; it does
not need another GPU. Archive + extracted files are roughly 60.7 GB decimal,
plus environment, caches, small bundles and output. Budget about 70–80 GB or
check actual environment size; **scratch quota**, not `df` filesystem free space,
is authoritative. No cleanup is automatic.

Download the complete result directory and keep the local export bundle before
Oscar scratch retention expires. Keep the original local test tar. Do not delete
the scratch dataset/archive while these jobs or retries still need them.

Official Oscar references:
[storage](https://docs.ccv.brown.edu/oscar/managing-files/filesystem),
[transfer](https://docs.ccv.brown.edu/oscar/managing-files/filetransfer),
[GPU submission](https://docs.ccv.brown.edu/oscar/gpu-computing/gpus).
