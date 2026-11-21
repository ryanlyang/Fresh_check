# Readout, triangles and support: independent one-GH200 pipelines

Each model has its own frozen pipeline:

`prepare → benchmark → summary → dispatch → train → evaluate → report`

`pipeline --reviewed` authorizes automatic submission of **fresh full training** after
verified benchmark success. Merely having a Slurm `COMPLETED` result is not admission.
There are no cross-model dependencies, concurrency caps, arrays or segmented jobs.
No SPORC jobs, model implementations, environments, data or frozen sources are changed.

## Scientific/runtime contract

| Model option | Parameters | Frozen scientific implementation |
|---|---:|---|
| `readout` | 2,349,702 | `part_ev_pair_readout` |
| `triangles` | 2,334,662 | `part_ev_triangles` |
| `support` | 2,384,070 | `part_ev_support` |

All remain additions to all-eight-block layerwise State-EV. The model-specific
numerical tolerances, native-path checks, delayed learning checks and restart
probes remain in force; they are not bypassed to get a passing benchmark.

- Seed 101, Ranger, global/local batch 512, 1,000,000 updates, 50 validations
  every 20,000 updates; 1.28M validation events/check. No shortened recipe.
- Exactly **one GH200** for benchmark, training and evaluation; native BatchNorm,
  two loader workers, FP16 training and FP32 validation, TF32 disabled.
- The donor is an already **prepared** SPORC benchmark. Its immutable model,
  recipe, source inventory and dataset metadata are authenticated read-only.
  Its GPU benchmark need not have run or passed. Later SPORC benchmark/summary
  writes do not invalidate the independent Tigris campaign.
- ARM environment: `/home/ryreu/miniforge3-aarch64/envs/part_ev_fullscale/bin/python`.
  No packages are installed/changed. Weaver source identity must match the donor;
  the ARM environment is independently frozen, not equated to the x86 environment.
- Native BN and single-rank stream ordering differ from the two-A100/SyncBN runs.
  These are fresh experiments, not cross-platform bitwise replicas. Never resume
  a SPORC checkpoint in this pipeline or pool hardware results as identical runs.

## Resources and gates

| Phase | Requested walltime | GPUs | CPUs | Host RAM |
|---|---|---:|---:|---:|
| Prepare | 2h | 0 | 8 | 128 GiB |
| Benchmark | 5h30m | 1 GH200 | 8 | 128 GiB |
| Summary | 15m | 0 | 4 | 16 GiB |
| Dispatch | 30m | 0 | 4 | 16 GiB |
| Train | Measured recommendation | 1 GH200 | 8 | 128 GiB |
| Evaluate | Measured recommendation | 1 GH200 | 8 | 128 GiB |
| Report | 1h | 0 | 4 | 16 GiB |

The benchmark compares the candidate with native one-GH200 layerwise State-EV:
4,200 measured real-data updates per arm after 20 warmups, full-128-particle
batch-512 training/memory checks, checkpoint/restart, 128k FP32 validation and
batch-256 inference plus batch-1 latency. Triangle all-tied stress checks remain
full length; **only diagnostic profiling** is bounded to 2 jets × 32 particles.
Training workers are shut down before synthetic probes/profiling/validation.

The slowest observed sustained window projects the full training budget; all 50
validations, checkpoint writes and cold fetch are included with a 30% margin.
Queue delays, replay and future contention are not guaranteed by this margin.
Default approved ceilings are **480h training / 24h evaluation**, further limited
by live Slurm partition/account checks. Smaller ceilings can be set at submission.
A failed, incomplete, changed or over-budget report submits **no production**.

Aggregate free-space checks include outstanding registered SPORC and these Tigris
outputs, with a shared 4 GiB reserve. Benchmark budgets are about 1 GiB each;
production reserves roughly another 2.5 GiB per model (versioned best candidates,
atomic restart writes, predictions and metadata), checked again at dispatch.
These are accounting safeguards, **not quota reservations**. Future unrelated
jobs can still consume space. No dataset copies or persistent pair caches.

## Queue all three on Tigris

Commit/push the new files locally, then on **Tigris** pull from a clean checkout.
The following finds the prepared references by their recorded benchmark job IDs;
change the IDs only if you intentionally chose different prepared references.
Missing or ambiguous ledgers stop the commands. A failed SPORC GPU benchmark is not
a blocker, provided preparation succeeded and its frozen model is the intended one.

**Support retry:** benchmark `22114967` froze the old zero-tolerance native-path
check. First prepare a corrected SPORC support campaign as described in the
[support retry instructions](../part_ev_support/README.md#retry-the-november-20-native-path-failure-on-both-clusters).
Set `EVRTS_SUPPORT_BENCHMARK_JOB` to its **new** benchmark job ID before using the
all-three example below. Only preparation must finish; the SPORC GPU benchmark
may still be pending. New Tigris submissions reject the old support donor.
If readout and triangles are already queued, do not resubmit all three: use the
support-only retry commands instead. Existing frozen jobs are never patched.

```bash
(
  set -euo pipefail
  cd /home/ryreu/atlas/Fresh_check
  git pull --ff-only

  evrts_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  evrts_base="/home/ryreu/atlas/Fresh_check/checkpoints/part_ev_rts_tigris"

  evrts_reference() {
    local evrts_family="$1" evrts_job="$2" evrts_ledger
    local -a evrts_refs
    # Ledgers exist even while the SPORC GPU benchmark is still pending.
    mapfile -t evrts_refs < <(
      while IFS= read -r evrts_ledger; do
        if awk -v job="${evrts_job}" \
          '$1 == "benchmark" && $2 == job {found=1} END {exit !found}' \
          "${evrts_ledger}"; then
          dirname -- "${evrts_ledger}"
        fi
      done < <(find "checkpoints/${evrts_family}" -maxdepth 2 \
        -type f -name submitted_job_ids.tsv -print)
    )
    [[ "${#evrts_refs[@]}" == 1 ]] || {
      echo "Need exactly one prepared reference for job ${evrts_job}" >&2
      return 1
    }
    realpath -- "${evrts_refs[0]}"
  }

  evrts_readout="$(evrts_reference part_ev_pair_readout 22138550)"
  evrts_triangles="$(evrts_reference part_ev_triangles 22138556)"
  : "${EVRTS_SUPPORT_BENCHMARK_JOB:?Set this to the NEW corrected SPORC support benchmark job ID}"
  evrts_support="$(evrts_reference part_ev_support "${EVRTS_SUPPORT_BENCHMARK_JOB}")"

  # Inspect all three before submitting any. Plan is read-only.
  for evrts_model in readout triangles support; do
    evrts_variable="evrts_${evrts_model}"
    bash sbatch/submit_part_ev_rts_tigris.sh \
      --mode plan --model "${evrts_model}" \
      --benchmark "${!evrts_variable}" \
      --output "${evrts_base}/${evrts_model}_${evrts_stamp}"
  done

  # Explicit opt-in to benchmark-gated full training.
  for evrts_model in readout triangles support; do
    evrts_variable="evrts_${evrts_model}"
    bash sbatch/submit_part_ev_rts_tigris.sh \
      --mode pipeline --model "${evrts_model}" \
      --benchmark "${!evrts_variable}" \
      --output "${evrts_base}/${evrts_model}_${evrts_stamp}" \
      --reviewed
  done

  squeue --me -o '%.18i %.40j %.12T %.12l %R'
)
```

Initially there are **four jobs/model**: prepare, benchmark, summary and dispatch.
Summary runs after the benchmark ends so failures can be reported. Dispatch depends
on **successful** summary and independently recomputes the complete admission.
It then submits train → evaluate → report with `afterok` dependencies. Failed
dependencies are cancelled rather than hanging forever.

All orchestration files are snapshotted from committed Git at submission. Later
push/pull changes cannot alter queued workers. The scientific donor directory must
also remain unchanged and present; do not delete it while either campaign uses it.

## Inspect or retry a handoff safely

Use the exact path printed at submission:

```bash
bash sbatch/submit_part_ev_rts_tigris.sh --mode status --output "/absolute/campaign/path"
bash sbatch/submit_part_ev_rts_tigris.sh --mode review --output "/absolute/campaign/path"
```

`pipeline_job_ids.tsv` records the original four jobs; `production_job_ids.tsv`
records the dynamically submitted training/evaluation/report jobs. Check the
actual scheduler as well. Results live under `production/runs/<run_id>/seed_101/`;
the final audited report is `production/reports/production_report.md`.

If dispatch stopped **before any production submission** (for example, storage
became tight), clear the external blocker and explicitly retry its frozen handoff:

```bash
bash sbatch/submit_part_ev_rts_tigris.sh --mode dispatch \
  --output "/absolute/campaign/path" --reviewed
```

A completed dispatch is idempotent. An unknown `sbatch` response or partial chain
blocks blind retry; reconcile the intent journal, ledger and Slurm first. No
automatic retries, cancellation, or replacement of existing chains are performed.
Checkpoint recovery is implemented (latest complete optimizer/scaler/RNG and exact
data-stream replay); it does not automatically submit another job after a timeout.

The two clusters remain independent. If you want to keep only whichever starts
first, inspect both queues and explicitly cancel the unwanted chain yourself.
Nothing here cancels the other cluster or shares its training output.

## Local verification

The new tests cover native model counts, CPU oracle/correctness paths, benchmark
orchestration, malformed/missing evidence, bounds, immutable imports/snapshots,
fake-Slurm dependencies and partial submissions, and exact toy training recovery
before/after/during validation. Local CPU tests cannot establish GH200 numerical
parity, memory headroom, throughput or production accuracy: those remain the
mandatory queued benchmark's job.
