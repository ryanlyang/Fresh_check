# Automatic benchmark-to-production follow-ups on SPORC

Attach one independent CPU controller to each already-queued readout, triangle,
or support benchmark. Once its summary succeeds, the controller recomputes the
benchmark's scientific admission checks and, **only if they pass**, submits:

```text
existing prepare -> existing benchmark -> existing summary
                                                 |
                                              afterok
                                                 v
                                 CPU controller: verify + submit
                                                 |
                       fresh train -> afterok:evaluate -> afterok:report
```

Arming is explicit permission to submit that model's future production chain.
It does not train immediately, change existing jobs, or waive failed checks.
The three chains have no cross-model dependencies or concurrency cap. Closing
the SSH session is fine. Scheduler capacity still determines when each job runs.

## Scientific setup and resources

All three reuse the production engine in `part_ev_state_production` and the exact
scientific code, data inventory, environment and model settings frozen by their
own benchmark. No new GPU preflight, new training recipe or model changes.

- Fresh seed 101; 1,000,000 update attempts, global batch 512 / local batch 256.
- Two same-node A100s, DDP/SyncBatchNorm, unchanged full Ranger schedule.
- 50 validation checks, each after 20,000 updates, each on 1.28M validation jets.
- Accuracy-selected checkpoint; final FP32 evaluation on the full 5M validation
  population, using one A100. No independent test-set evaluation.
- One long training allocation, not successive three-day segments. Existing
  latest-full-state checkpoints remain available for explicit recovery.

| Kind (`--model`) | Parameters | GPU-job host RAM | Training walltime |
|---|---:|---:|---|
| `readout` | 2,349,702 | 64 GiB | Measured recommendation |
| `triangles` | 2,334,662 | 128 GiB | Measured recommendation |
| `support` | 2,384,070 | 64 GiB | Measured recommendation |

Training/evaluation request 8 CPUs. Each controller requests **2 hours, 4 CPUs,
16 GiB RAM, no GPUs**; each final report requests 1 hour, 8 CPUs, 64 GiB, no GPUs.
No production profiler is enabled. Triangle's larger host-memory request is
retained from the fixed benchmark.

Training and evaluation walltimes are taken from each passed benchmark, not
guessed before it completes. Its training recommendation includes the 50
validations, checkpoint overhead and 30% margin. Arming defaults to approval
ceilings of **480 hours training and 24 hours evaluation**; these are ceilings,
not allocation requests. Lower them explicitly with `--max-train-hours` and
`--max-evaluate-hours` if desired. The measured requests must also fit the live
`tier3` partition and pass `sbatch --test-only`. Nothing is silently shortened.
If a model needs more than the approved or live limit, its controller stops
without submitting training.

## Attach to the currently queued benchmarks

Commit/push the implementation locally, then `git pull --ff-only` in
`/home/ryreu/atlas/Fresh_check` on **sporcsubmit**, not Tigris. The targeted
follow-up/production files must be committed and clean. Existing jobs use their
own snapshots and are not modified by this pull or by arming controllers.

The known summary job IDs from the current queue are:

**Support update (November 20):** `22114967` / `22114968` failed the original
zero-tolerance native-path check. For a retry, use the
[fresh support benchmark procedure](../part_ev_support/README.md#retry-the-november-20-native-path-failure-on-both-clusters)
and replace the support summary ID below with the **new** one. Use fresh
follow-up/production paths; an old controller remains bound to its old summary.
Do not rerun the all-model example to duplicate readout or triangle controllers.

| Model | Benchmark | Summary |
|---|---:|---:|
| Readout (fixed) | 22138550 | 22138551 |
| Triangles (fixed) | 22138556 | 22138557 |
| Support | 22114967 | 22114968 |

The following **read-only** command locates their unique campaign directories
from submission ledgers (works even before log files exist), checks the plans,
then prints the three commands to arm them. If benchmarks were replaced, update
the summary IDs first. It refuses missing or ambiguous matches.

```bash
cd /home/ryreu/atlas/Fresh_check
git pull --ff-only

python3 -B - <<'PY'
from pathlib import Path
import shlex
import subprocess

repo = Path('/home/ryreu/atlas/Fresh_check')
targets = (
    ('readout', 'part_ev_pair_readout', '22138551'),
    ('triangles', 'part_ev_triangles', '22138557'),
    ('support', 'part_ev_support', '22114968'),
)
commands = []
for kind, family, summary in targets:
    matches = []
    for ledger in (repo / 'checkpoints' / family).glob('*/submitted_job_ids.tsv'):
        if ['summary', summary] in [line.split() for line in ledger.read_text().splitlines()]:
            matches.append(ledger.parent.resolve())
    if len(matches) != 1:
        raise SystemExit(f'{kind}: expected one ledger for summary {summary}, found {matches}')
    output = repo / 'checkpoints/part_ev_benchmark_followup' / f'{kind}_{summary}'
    production = repo / 'checkpoints/part_ev_state_production' / f'{kind}_full_seed101'
    common = ['--model', kind, '--benchmark', str(matches[0]),
              '--output', str(output), '--production', str(production)]
    launcher = ['bash', str(repo / 'sbatch/submit_part_ev_benchmark_followup.sh')]
    subprocess.run([*launcher, '--mode', 'plan', *common], check=True)
    commands.append([*launcher, '--mode', 'arm', '--reviewed', *common])

print('\nAfter reviewing all three plans, run these commands to arm production:')
for command in commands:
    print(shlex.join(command))
print('\nRead-only: no jobs submitted or campaign files changed.')
PY
```

Run the printed `--mode arm --reviewed` commands to queue the controllers.
Each queues only one CPU job at this stage, dependent on its own summary.
If a summary already completed (or aged out of Slurm), arming requires complete
passed benchmark evidence immediately; the controller then has no obsolete job
dependency. The worker still rechecks everything before production submission.
This avoids relying on an expired Slurm dependency; see the official
[Slurm dependency documentation](https://github.com/SchedMD/slurm/blob/master/doc/man/man1/sbatch.1).

Do not also manually submit these same production campaigns. Duplicate benchmark
owners, existing production directories and uncertain submissions are rejected.
If arming one controller fails, earlier successfully armed controllers remain
independent; inspect their ledgers instead of blindly repeating all commands.

## Check status

```bash
squeue --me -o '%.18i %.40j %.12T %.12l %R'

bash /home/ryreu/atlas/Fresh_check/sbatch/submit_part_ev_benchmark_followup.sh \
  --mode status \
  --output /home/ryreu/atlas/Fresh_check/checkpoints/part_ev_benchmark_followup/readout_22138551
```

Use `triangles_22138557` or `support_22114968` for the others. Status is read-only
metadata, not a live Slurm audit. Controller logs are `logs/evfollow_<kind>_<id>.*`.
After dispatch, `result.json` records the production path, three submitted job
IDs and walltimes; the production campaign also has `submitted_job_ids.tsv`.

Important distinctions:

- **No `result.json` yet:** the controller may be pending, cancelled by a failed
  dependency, or have failed before admission. Check Slurm and its logs.
- **`submitted`:** the three production jobs were queued, not necessarily started
  or completed. A rerun of the worker is a no-op, not a second training.
- **`blocked`:** admission/resource checks failed before any production output
  was created. Nothing was submitted by the production launcher.
- **`dispatch_started` / `needs_reconciliation`:** do not retry blindly. A
  partial/uncertain production submission may exist. Inspect both submission
  journals, ledgers and Slurm before explicit recovery.

For a known inactive controller blocked before production creation (for example,
storage was subsequently freed), explicit retry uses the same frozen intent:

```bash
bash /home/ryreu/atlas/Fresh_check/sbatch/submit_part_ev_benchmark_followup.sh \
  --mode retry --reviewed \
  --output /home/ryreu/atlas/Fresh_check/checkpoints/part_ev_benchmark_followup/readout_22138551
```

Active, unknown, partial or already-submitted handoffs cannot use this shortcut.
There is no automatic retry loop, cancellation, deletion or resource escalation.
Production restart, if needed later, uses the existing production launcher's
explicit `--mode resume --reviewed --output <production>` after inspecting Slurm.

## Storage, source safety and limitations

Arming freezes only a small orchestration snapshot, not another dataset, model
checkpoint or full repository archive. It requires 4 GiB reserve plus 64 MiB
headroom. At dispatch, production checks remaining budgets of other registered
state-extension productions and budgeted related benchmarks, counting the shared
4 GiB reserve once. Each new model's full output allowance is roughly 2.5 GiB
plus that shared reserve. Existing filesystem use already reduces free space.
There are no persistent pair caches or dataset copies, and nothing is deleted.
This accounting is not a filesystem quota reservation against unrelated jobs.

Controllers serialize only the short production admission/submission operation
under the shared production lock so simultaneous benchmark completions do not
race their output budgets. This is **not** a cap on GPU training concurrency.

The controller freezes committed orchestration at arming. Production copies
from that authenticated snapshot, not whatever the checkout contains later.
Scientific imports come from the original benchmark snapshot in a separate
clean interpreter; benchmark files remain untouched. Every worker rechecks
source, benchmark evidence and environment. Later pushes/pulls cannot change
these queued sources, but deleting snapshots/data or changing the pinned Python
environment can still break them.

This requires `sbatch` access from a tier3 CPU batch job. Local tests simulate
Slurm and exercise the real production admission reviewers, CPU model builders,
checkpoint/resume engine and source isolation. They do not establish site policy
or GPU performance; the already queued cluster benchmarks provide that evidence.
