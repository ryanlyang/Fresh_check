# Selected-production compiler recovery

This is an independent recovery overlay for selected tasks **2, 5 and 6** of
`part_selected_production/seven_full_20261107`. It addresses the observed
`/usr/bin/c++` banner change from Red Hat GCC `11.5.0-14` to `11.5.0-15`.
It does not rebuild a backend, edit an old manifest, pretend the compiler is
unchanged, or relax any numerical tolerance.

## Admission and scope

A one-hour CPU-only Tigris check loads the **existing** reference and optimized
backend binaries. It verifies:

- Original production/source/benchmark identities and Python environment.
- Reference manifest contract, binary and C++ source hashes, flags, architecture,
  Python/PyTorch ABI, embedded compiler identity and OpenMP availability.
- The exact supported driver-banner transition, with current executable hash.
- Original backend self-test and both historical canonical smoke-tree hashes.
- Exact optimized-versus-original packed-tree parity at particle widths
  1, 2, 11, 32, 64 and 128 using 1, 2 and 4 threads, including masked/nonfinite
  padding, duplicate/tied particles, and rejection of nonfinite valid inputs.
- Hashes of the actually loaded C/C++/math/OpenMP runtime libraries.

The check uses synthetic CPU fixtures, not training/validation event payloads.
Dataset metadata and existing campaign JSON are still checked. No GPU is needed.
No successful compatibility record is written on failure. Each GPU recovery
worker repeats the checks and requires identical compatibility evidence, so a
different node's compiler/library drift cannot inherit approval silently.
This is bounded compatibility evidence, not proof for every possible input.

The recovery uses a private `Bench` constructor with the two authenticated
backend modules supplied explicitly. The original modules' globals are not
modified. Models, loaders, normalization, optimizers, training steps, checkpoint
format, AMP policy, budgets, validation and reporting remain the original frozen
implementations. In particular, the original backend was built with `-14`; its
embedded build identity continues to be compared against that **old** manifest.

## Queue the compatibility check

Commit/push the recovery implementation locally first. On **Tigris**:

```bash
cd /home/ryreu/atlas/Fresh_check
git pull --ff-only

SEVEN_ROOT="/home/ryreu/atlas/Fresh_check/checkpoints/part_selected_production/seven_full_20261107"
RECOVERY_ROOT="/home/ryreu/atlas/Fresh_check/checkpoints/part_selected_compiler_recovery/compilerfix_$(date -u +%Y%m%dT%H%M%SZ)"

bash sbatch/submit_part_selected_compiler_recovery.sh \
  --mode plan --campaign "${SEVEN_ROOT}" --output "${RECOVERY_ROOT}"

bash sbatch/submit_part_selected_compiler_recovery.sh \
  --mode check --campaign "${SEVEN_ROOT}" --output "${RECOVERY_ROOT}"
```

Keep the printed recovery path. This freezes only the small recovery source and
metadata, not copies of datasets, backend binaries or checkpoints. The check
requests 12 CPUs, 128 GiB host memory, no GPU, and one hour. It submits **no
training** and does not change the original production files or job ledger.

After the check completes, using the same `RECOVERY_ROOT`:

```bash
bash sbatch/submit_part_selected_compiler_recovery.sh \
  --mode review --output "${RECOVERY_ROOT}"
```

Do not edit `compatibility.json` or the old manifests to force approval. A failed
check needs investigation; a new check uses a new recovery directory.

## Queue only the failed selected runs

After a passed review:

```bash
bash sbatch/submit_part_selected_compiler_recovery.sh \
  --mode submit --output "${RECOVERY_ROOT}" --reviewed

squeue --me -o "%.18i %.35j %.12T %.12l %R"
```

| Task | Model | Training limit | Evaluation limit |
|---|---|---:|---:|
| 2 | Selected parameter-matched shared-bias ParT | 192 h (8 days) | 3 h |
| 5 | Selected shared-bias ParT | 192 h (8 days) | 3 h |
| 6 | Selected all-eight layerwise State-EV | 336 h (14 days) | 4 h |

Each GPU job requests one GH200, 12 CPUs and 128 GiB host RAM. There is no
training concurrency cap; each evaluation depends on its own training with
`afterok`. There are no new performance benchmarks or altered time estimates.

Existing run directories and their original campaign identity are reused.
`last.pt`, if present and valid, resumes the complete training state with the
original exact-stream replay; selected best weights are not used as restart
state. Completed selected training is not rerun, but its checkpoint is audited
before evaluation. All three still use seed 101, 1,000,000 updates, batch 512 and
50 validation checks. This is full-5M validation, not independent test inference.

Submission refuses active/pending selected retries, duplicate reporting jobs,
unresolved submission attempts, insufficient original storage headroom, or a
second recovery owner for the same production. Active tasks 3/4 and the compact
campaign do not block it and are never cancelled or modified. An uncertain
`sbatch` result remains explicitly unresolved; inspect the scheduler before
manual reconciliation, never blindly retry it.

New IDs are recorded both in the recovery ledger and **appended** to the original
mutable production ledger; both submission-intent journals record the attempt.
This lets the old production launcher also detect these active jobs. No old
ledger records, snapshots, campaign JSON, backend manifests, libraries or
benchmark results are overwritten. Compatibility/session provenance lives in
the recovery directory. Normal recovered training/evaluation outputs go into
the original run directories.

The new final report waits for the recovered evaluations and any still-pending
original evaluations for the other four models. If an unrelated evaluation is
missing and has no live recorded job, the launcher explicitly defers the report
instead of rerunning that model. Once those evaluations are queued or complete:

```bash
bash sbatch/submit_part_selected_compiler_recovery.sh \
  --mode report --output "${RECOVERY_ROOT}" --reviewed
```

The original report writer audits all seven checkpoints and prediction files.
No report is treated as complete merely because a Slurm job exited successfully.

## Local verification

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_part_selected_compiler_recovery.py tests/test_part_selected_compiler_recovery_launcher.py tests/test_part_selected_production.py tests/test_part_selected_production_launcher.py
```

Local tests cover compiler/ABI rejection, immutable evidence, private loader
binding, model counts/recipe identity, original restart behavior, resource
requests, dependencies and duplicate/partial-submission safety. The actual
Linux/aarch64 binary/library parity must still pass on Tigris.
