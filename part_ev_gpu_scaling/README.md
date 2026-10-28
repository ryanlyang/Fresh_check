# One versus two A100s: layerwise all-eight State-EV

This is a disposable performance diagnostic, not another training campaign.
It reads the prepared `part_ev_state_compare` campaign and never changes its
source, benchmark decisions, jobs, checkpoints, or approval. No test data is used.

Both jobs use seed 101, width 32, all eight state blocks, query chunks of 64,
2,327,462 parameters, FP16 AMP with FP32 parameters, and the full Ranger recipe.
The global batch remains 512: 512 on one GPU, 256 per GPU on two. Two GPUs use
single-node DistributedDataParallel (DDP), synchronized BatchNorm, identical
initial weights, and synchronized gradient updates. No learning-rate scaling,
microbatch accumulation, reduced width, or altered state math is introduced.
There are two data-loader workers and eight allocated CPUs **total** in either job.

## Queue

Commit/push these files locally, then pull on SPORC. Use the existing environment;
do not install or upgrade packages for this diagnostic.

```bash
cd /home/ryreu/atlas/Fresh_check
git pull --ff-only
bash sbatch/submit_part_ev_gpu_scaling.sh \
  --campaign /home/ryreu/atlas/Fresh_check/checkpoints/part_ev_state_compare/state_compare_20261027T224700Z
```

Add `--dry-run` to inspect requests without creating files or submitting jobs.
The launcher prints the new diagnostic directory and two job IDs. Both jobs have
04:30:00 walltimes. The two-GPU job requests two A100s **on one node**, and runs
after the one-GPU job finishes (even if that job fails). Neither job cancels or
depends on production training. Submission requires a fresh output directory;
partial submissions and failures retain their evidence, not an automatic retry.

After both finish:

```bash
bash sbatch/submit_part_ev_gpu_scaling.sh --mode review --output /absolute/diagnostic/path
```

## Measurements and checks

1. A small FP32, nonzero-state forward/backward check compares the distributed
   model against a global-batch native-BatchNorm reference with dropout/trimming
   disabled. This tests DDP gradient averaging and SyncBatchNorm integration.
2. Real ROOT streaming: 20 warmup and 4,200 timed steps, repeated timing windows,
   same global exposure count and original per-rank stochastic trimming. Input
   transfer, loading stalls, Ranger updates, AMP and DDP communication are timed.
3. Fixed workload: 10 warmup and 40 timed steps on the same deterministic global
   512-by-128 synthetic particle batch, split across ranks. Every slot is valid,
   trimming is off and inputs are resident on the GPUs. This isolates compute
   and synchronization from loading and varying particle lengths. It is not an
   accuracy evaluation or a replacement for real-data throughput.

The report includes throughput, per-GPU allocated/reserved peak memory, AMP skips,
model/optimizer synchronization evidence, walltime speedup and relative GPU-hours.
For example, a 1.5x throughput speedup costs 2/1.5 = 1.33x as many training
GPU-hours. Training-only time projections exclude validation/checkpoint overhead,
restarts and queue time; they are **not** full-job walltime recommendations.

The historical 90% allocated-memory threshold is shown separately, not relaxed.
A completed diagnostic can report insufficient headroom without throwing away its
timing. Nothing here authorizes production, even if both benchmarks pass.

## Limitations

Distributed file sharding changes event order, and native trimming acts per rank.
SyncBatchNorm aggregates statistics, but this is not a claim of bitwise-identical
training or proven accuracy retention. The fixed workload controls particle count
and data to make scaling easier to interpret. Precision is explicitly scoped on
the underlying model as well as outer autocast: the DDP wrapper alone does not
expose Weaver's internal AMP flag.

GPU names, memory sizes, hostname, NCCL settings, topology (in Slurm stdout), and
source/environment identities are retained. Different GPU models or memory sizes
prevent a headline scaling comparison. Different nodes, clocks and competing
workloads can still affect the numbers: repeat close calls. Two GPUs can also wait
longer in the queue. Do not silently reuse this runtime in the existing
single-GPU production campaign.

Implementation follows PyTorch's [DDP guidance](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html)
and [non-reentrant checkpoint support](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html).
Local CPU tests do not establish actual A100/NCCL scaling; the queued checks do that.
