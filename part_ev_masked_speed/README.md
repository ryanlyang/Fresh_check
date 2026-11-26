# Masked-predictive training: isolated GH200 speed study

The original masked benchmark passed, but its 745-hour recommended allocation
exceeded the 480-hour ceiling. Its active auxiliary step took about 2.215 s,
versus 0.411 s for a clean-only step. This diagnostic tests execution changes,
not a cheaper learning objective. It does **not** submit production.

## What changes

All five variants run sequentially on one GH200 in the same allocation, using
fresh disposable seed-101 models and matching train-stream fingerprints:

| Variant | Auxiliary batch | Serial slots for 256 selected jets | Execution |
| --- | ---: | ---: | --- |
| baseline32 | 32 | 8 | Unmodified original step |
| fast32 | 32 | 8 | Batched EMA operations and consolidated exact buffer audit |
| fast64 | 64 | 4 | Same optimizations, larger auxiliary batches |
| fast128 | 128 | 2 | Same optimizations, larger auxiliary batches |
| fast256 | 256 | 1 | Same optimizations, larger auxiliary batch |

The auxiliary teacher and visible-student passes use fixed BN statistics and
disabled dropout. Grouping more selected jets together should therefore preserve
the objective, while reducing serial work and synchronization overhead. It may
also increase padding and memory, so a bigger batch is not automatically faster.
Teacher EMA retains multiply-then-add ordering, successful-update semantics and
the original skipped-update buffer-copy rule. The buffer check remains exact;
it uses one host decision instead of one decision per buffer.

Unchanged: clean batch 512, all clean classification updates, 50% auxiliary jet
selection, both masking modes, target counts, teacher targets, predictor, loss
normalization, BN isolation, optimizer, LR/auxiliary/EMA schedules, and the
1M-update/50-validation production recipe. No extra parameters or inference
operations are introduced. No layers, views or auxiliary updates are removed.
The optimized step currently supports **single-GPU diagnostics only**; it is not
installed into SPORC DDP or any production runner.

## Evidence collected

- FP32 and AMP full-step comparisons against the original path, with nonzero
  State-EV outputs, variable particle counts, full batch 512, exact masks/RNG/
  scaler/BN buffers, and bounded gradients/model/optimizer/teacher differences.
  Existing numerical tolerances are retained, not widened.
- Seven-update save/restore probes through auxiliary schedule boundaries and a
  Lookahead update, including restoration into a fresh session. These are
  synthetic transaction checks, not proof of full ROOT-stream resume.
- Twenty warmup steps, 300 measured real-data active steps in three windows,
  100 measured clean-only steps, and a full-128-particle memory/throughput test.
  Timing includes data delivery and batch fingerprinting; the slowest window
  determines each conservative step-time estimate. Warmup is excluded.
- AMP skips, full-state finiteness, a 10% device-memory reserve, and separate
  clean/teacher/predictor/backward/optimizer/EMA profiling scopes. Profiling is
  outside the timing windows. Profiles include nested scopes: do not sum their
  inclusive times as though they were disjoint.

A rejected candidate is reported and does not prevent testing the others.
Failure of the original-path baseline stops comparisons. An interrupted study
keeps its partial results and is not admitted as complete.

Provisional allocation estimates weight 879,999 active and 120,001 clean steps,
reuse the original measured validation/checkpoint overhead, and add 30% margin.
They exclude queues/replay. Short windows, disposable weights and reused
validation costs make these **estimates**, not full production admission.
If no validated candidate is faster, the report says so. No accuracy-retention
claim is made: even mathematically equivalent regrouping can change AMP rounding
and the subsequent training trajectory.

## Isolation and resources

New output: `checkpoints/part_ev_masked_speed/<fresh-name>` with its own committed
source snapshot and submission journal. The donor's source/evidence is read-only
and hashed. Relevant reference model/data/optimizer files must still match the
donor byte-for-byte. The original masked package and all queued campaigns remain
unchanged. No datasets, environments, old checkpoints or jobs are modified.

- Benchmark: **one GH200, 12 CPUs, 128 GiB host RAM, 5h30m**.
- Summary: **CPU only, 2 CPUs, 4 GiB RAM, 15 minutes**, `afterany` the benchmark.
  No advance signal is attached to this short CPU job.
- Local storage gate: remaining **2 GiB diagnostic budget plus 4 GiB reserve**.
  Other campaigns can consume this free space; this is not a disk reservation.
- Only small restart probes and JSON evidence are retained; no ROOT cache or
  evaluation predictions are created. The original environment/data inventory
  and GH200/driver identity are checked. No new environment is installed.

## Queue on Tigris

Commit/push the new files first. On Tigris:

```bash
(
  set -euo pipefail
  cd /home/ryreu/atlas/Fresh_check
  git pull --ff-only

  masked_log="$(find checkpoints/part_ev_masked_predictive \
    -maxdepth 3 -type f \
    -name 'evmp_tigris_benchmark_1263609.out' -print -quit)"
  [[ -n "${masked_log}" ]] || { echo "Original benchmark log not found."; exit 1; }
  MASKED_ROOT="$(realpath "$(dirname "$(dirname "${masked_log}")")")"
  SPEED_ROOT="${PWD}/checkpoints/part_ev_masked_speed/speed_$(date -u +%Y%m%dT%H%M%SZ)"
  printf 'Donor: %s\nDiagnostic: %s\n' "${MASKED_ROOT}" "${SPEED_ROOT}"

  bash sbatch/submit_part_ev_masked_speed.sh \
    --mode plan --from-campaign "${MASKED_ROOT}" --output "${SPEED_ROOT}"
  bash sbatch/submit_part_ev_masked_speed.sh \
    --mode benchmark --from-campaign "${MASKED_ROOT}" --output "${SPEED_ROOT}"

  squeue --me -o '%.18i %.40j %.12T %.12l %R'
)
```

Save the printed diagnostic path. After completion, review that exact path:

```bash
bash /home/ryreu/atlas/Fresh_check/sbatch/submit_part_ev_masked_speed.sh \
  --mode review --output /absolute/path/printed/above
```

This queues only a benchmark and its summary, not a control training or production
job. Keep existing jobs as they are. If submission stops after recording a job,
inspect `diagnostic_submissions.json` and Slurm before taking further action;
never blindly repeat submission under another name.

## Next decision

Choose a numerically validated faster candidate using the measured real-data
and fixed-length results, not batch size alone. Then implement/admit that path
for production with real-stream restart and full validation checks. A result
under 480 hours here does not bypass those checks. If execution changes are not
enough, reducing auxiliary work would be a separate scientific experiment and
requires an explicit decision; this diagnostic does not do that.
