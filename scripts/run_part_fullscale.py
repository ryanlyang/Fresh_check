#!/usr/bin/env python3
"""Full JetClass ParT / shared-bias ParT-EV: prepare, benchmark, train, test.

Run through sbatch/submit_part_fullscale.sh on Tigris. No phase submits jobs.
The test phase is deliberately opt-in and cannot select training checkpoints.
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import timedelta
import json
import os
from pathlib import Path
import signal
import statistics
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from part_ev_fullscale import protocol as P

STOP_REQUESTED = False


def progress(stage, **values):
    print(json.dumps({"stage": stage, **values}, sort_keys=True, allow_nan=False), flush=True)


def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def prepare(args):
    from part_ev_fullscale.data import inspect_dataset
    from part_ev_fullscale.model import reference_metadata
    output = args.output.resolve()
    with P.file_lock(output / ".prepare.lock", blocking=False):
        if (output / "campaign.json").exists():
            P.load_campaign(output)
            progress("already_prepared", output=str(output))
            return
        env = P.environment()
        reference_metadata()
        progress("inspecting_dataset_headers", data_root=str(args.data_root))
        inventory = inspect_dataset(args.data_root, check_entries=True)
        spec = {"schema": P.SCHEMA, "recipe": P.RECIPE, "tasks": P.TASKS,
                "environment": env, "inventory": inventory,
                "data_config_sha256": P.file_sha256(REPO / "particle_transformer/data/JetClass/JetClass_full.yaml"),
                "reference": {"paper": "https://arxiv.org/abs/2202.03772",
                              "launcher": "particle_transformer/train_JetClass.sh",
                              "weaver_tag": "v0.4.17"},
                "test_is_opt_in": True}
        spec["content_hash"] = P.digest(spec)
        P.write_json(output / "campaign.json", spec)
        progress("prepared", campaign_sha256=spec["content_hash"], training_runs=len(P.TASKS),
                 training_events=inventory["splits"]["train"]["event_count"],
                 validation_events=inventory["splits"]["val"]["event_count"],
                 test_submitted=False)


def distributed_setup():
    import torch
    import torch.distributed as dist
    if not torch.cuda.is_available():
        raise RuntimeError("GPU phase requires CUDA; use the Slurm launcher, not a login-node run")
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    P.local_batch_size(world)
    torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group("nccl", rank=rank, world_size=world, timeout=timedelta(hours=6))
    # Explicit, identical math policy for all six runs. FP16 AMP matches --use-amp.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    for name in ("SIGUSR1", "SIGTERM"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), request_stop)
    return rank, world, torch.device("cuda", local)


def barrier(world):
    if world > 1:
        import torch.distributed as dist
        dist.barrier()


def broadcast(value, rank, world):
    if world == 1:
        return value
    import torch.distributed as dist
    values = [value if rank == 0 else None]
    dist.broadcast_object_list(values, src=0)
    return values[0]


def stop_requested(device, world):
    import torch
    value = torch.tensor(int(STOP_REQUESTED), device=device)
    if world > 1:
        import torch.distributed as dist
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return bool(value.item())


def make_model(architecture, seed, device, world):
    import torch
    from part_ev_fullscale.model import build_model
    from part_ev_fullscale.checkpoint import seed_everything
    seed_everything(seed)
    base = build_model(architecture, use_amp=True).to(device)
    if world > 1:
        base = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base)
        wrapped = torch.nn.parallel.DistributedDataParallel(
            base, device_ids=[device.index], output_device=device.index,
            find_unused_parameters=True)
    else:
        wrapped = base
    return base, wrapped


def inputs_and_labels(batch, device):
    values, labels, _ = batch
    inputs = [values[name].to(device, non_blocking=True)
              for name in ("pf_points", "pf_features", "pf_vectors", "pf_mask")]
    if "_label_" not in labels:
        raise ValueError("Official simple-classification target _label_ missing")
    truth = labels["_label_"].long().to(device, non_blocking=True)
    return inputs, truth


def train_step(model, optimizer, scaler, batch, device):
    import torch
    inputs, truth = inputs_and_labels(batch, device)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16):
        logits = model(*inputs)
        loss = torch.nn.functional.cross_entropy(logits, truth)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Nonfinite training loss; last good checkpoint is retained")
    old_scale = scaler.get_scale()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return float(loss.detach()), scaler.get_scale() < old_scale


def save_checkpoint(path, base, optimizer, scaler, state, spec, runtime, rank, world):
    import torch.distributed as dist
    from part_ev_fullscale.checkpoint import capture_rank_state, atomic_torch_save
    local_state = capture_rank_state(base)
    rank_states = [None] * world
    if world > 1:
        dist.all_gather_object(rank_states, local_state)
    else:
        rank_states[0] = local_state
    if rank == 0:
        atomic_torch_save({"schema": "part_fullscale_checkpoint_v1", "campaign_sha256": spec["content_hash"],
                           "runtime": runtime, "model": base.state_dict(), "optimizer": optimizer.state_dict(),
                           "scaler": scaler.state_dict(), "rank_states": rank_states, "state": state}, path)
    barrier(world)


def replay(iterator, count, phase, rank):
    started = time.monotonic()
    for index in range(count):
        next(iterator)
        if rank == 0 and ((index + 1) % 1000 == 0 or index + 1 == count):
            progress("resume_data_replay", stream=phase, batches=index + 1, total=count,
                     seconds=round(time.monotonic() - started, 1))
        if STOP_REQUESTED:
            raise SystemExit(75)  # Previous complete checkpoint remains untouched.


def validate(base, iterator, device):
    import torch
    base.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for _ in range(P.RECIPE["validation_events_per_epoch"] // 512):
            if STOP_REQUESTED:
                return None
            inputs, truth = inputs_and_labels(next(iterator), device)
            logits = base(*inputs)
            if not bool(torch.isfinite(logits).all()):
                raise FloatingPointError("Nonfinite validation logits")
            correct += int((logits.argmax(1) == truth).sum())
            total += len(truth)
            loss_sum += float(torch.nn.functional.cross_entropy(logits.float(), truth, reduction="sum"))
    if total != P.RECIPE["validation_events_per_epoch"]:
        raise ValueError("Validation sample count differs from the reference budget")
    return {"accuracy": correct / total, "loss": loss_sum / total, "events": total}


def runtime_contract(spec, rank, world, device):
    import torch
    result = {"world_size": world, "local_batch_size": P.local_batch_size(world),
              "workers_per_rank": P.RECIPE["workers_per_rank"],
              "batchnorm": "SyncBatchNorm" if world > 1 else "BatchNorm",
              "gpu": torch.cuda.get_device_name(device),
              "tf32": False, "cudnn_benchmark": False,
              "campaign_sha256": spec["content_hash"]}
    if world > 1:
        import torch.distributed as dist
        records = [None] * world
        dist.all_gather_object(records, result)
        if any(row != result for row in records):
            raise ValueError("All ranks must use identical GPU and batch policies")
    return result


def freeze_runtime(output, runtime):
    path = output / "training_runtime.json"
    with P.file_lock(output / ".runtime.lock"):
        if path.exists():
            if P.read_json(path) != runtime:
                raise ValueError("All six production runs and their resumes must use the same runtime/GPU count")
        else:
            P.write_json(path, runtime)


def require_benchmark(output, runtime):
    path = output / "benchmarks" / f"gpus_{runtime['world_size']}.json"
    if not path.is_file():
        raise RuntimeError("Complete and review the GPU benchmark for this world size before production training")
    result = P.read_json(path)
    if (result.get("runtime") != runtime or result.get("passed") is not True
            or set(result.get("models", {})) != {"part", "part_ev"}):
        raise ValueError("Successful benchmark from this exact campaign/runtime is required")


def train(args):
    import torch
    from part_ev_fullscale.checkpoint import atomic_torch_save, restore_rank_state, seed_everything, trusted_load
    from part_ev_fullscale.data import build_loader, shutdown_loader
    from part_ev_fullscale.optimizer import Ranger
    output = args.output.resolve()
    spec = P.load_campaign(output)
    task_dir = P.task_directory(output, args.task_index)
    task = P.TASKS[args.task_index]
    rank, world, device = distributed_setup()
    runtime = runtime_contract(spec, rank, world, device)
    if rank == 0:
        require_benchmark(output, runtime)
        freeze_runtime(output, runtime)
    barrier(world)
    # Only one job may write this seed. Other ranks wait until rank0 has the lock.
    guard = P.file_lock(task_dir / ".train.lock", blocking=False) if rank == 0 else contextlib.nullcontext()
    with guard:
        complete = broadcast((task_dir / "training_complete.json").exists(), rank, world)
        if complete:
            if rank == 0:
                progress("training_already_complete", **task)
            return
        base, model = make_model(task["architecture"], task["seed"], device, world)
        optimizer = Ranger(base.parameters())
        scaler = torch.amp.GradScaler("cuda")
        state = {"task": task, "step": 0, "validated_epochs": 0, "best_accuracy": -1.0,
                 "best_step": None, "amp_skipped_steps": 0}
        last = task_dir / "last.pt"
        checkpoint = trusted_load(last) if last.exists() else None
        if checkpoint:
            if (checkpoint["schema"] != "part_fullscale_checkpoint_v1"
                    or checkpoint["campaign_sha256"] != spec["content_hash"]
                    or checkpoint["runtime"] != runtime or checkpoint["state"]["task"] != task
                    or len(checkpoint["rank_states"]) != world):
                raise ValueError("Restart checkpoint belongs to another task/protocol/runtime")
            base.load_state_dict(checkpoint["model"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer"])
            scaler.load_state_dict(checkpoint["scaler"])
            state = checkpoint["state"]
        train_loader, _ = build_loader("train", spec["inventory"], P.local_batch_size(world),
                                       rank=rank, world_size=world, seed=task["seed"])
        val_loader = None
        try:
            train_iter = iter(train_loader)
            if rank == 0:
                val_loader, _ = build_loader("val", spec["inventory"], 512, seed=task["seed"])
                val_iter = iter(val_loader)
            else:
                val_iter = None
            if checkpoint:
                replay(train_iter, state["step"], "train", rank)
                if rank == 0:
                    replay(val_iter, state["validated_epochs"] * (P.RECIPE["validation_events_per_epoch"] // 512), "val", rank)
                restore_rank_state(base, checkpoint["rank_states"][rank])
            else:
                # Equal initial base weights across ranks/models; separate perrank stochastic streams.
                seed_everything(task["seed"] + rank)
            barrier(world)
            del checkpoint
            clock = time.monotonic()
            start_step = state["step"]
            while True:
                completed_epochs = state["step"] // P.RECIPE["steps_per_epoch"]
                if state["validated_epochs"] < completed_epochs:
                    # Preserve a pre-validation restart point if walltime ends during validation.
                    save_checkpoint(last, base, optimizer, scaler, state, spec, runtime, rank, world)
                    metric = validate(base, val_iter, device) if rank == 0 else None
                    metric = broadcast(metric, rank, world)
                    if metric is None:
                        raise SystemExit(75)
                    state["validated_epochs"] += 1
                    if metric["accuracy"] > state["best_accuracy"]:
                        state["best_accuracy"], state["best_step"] = metric["accuracy"], state["step"]
                        if rank == 0:
                            atomic_torch_save({"model": base.state_dict(), "task": task,
                                               "campaign_sha256": spec["content_hash"], "runtime": runtime,
                                               "step": state["step"], "validation": metric}, task_dir / "best_model_val.pt")
                    if rank == 0:
                        P.write_json(task_dir / "validation" / f"epoch_{state['validated_epochs']:02d}.json",
                                     {"step": state["step"], **metric})
                        progress("validation", **task, **state, metric=metric)
                    save_checkpoint(last, base, optimizer, scaler, state, spec, runtime, rank, world)
                if state["step"] == P.RECIPE["total_steps"]:
                    if rank == 0:
                        P.write_json(task_dir / "training_complete.json", {"campaign_sha256": spec["content_hash"],
                                     "runtime": runtime, **state, "best_checkpoint_sha256": P.file_sha256(task_dir / "best_model_val.pt")})
                        progress("training_complete", **state)
                    break
                model.train()
                lr = P.learning_rate(state["step"])
                for group in optimizer.param_groups:
                    group["lr"] = lr
                loss, skipped = train_step(model, optimizer, scaler, next(train_iter), device)
                state["step"] += 1
                state["amp_skipped_steps"] += int(skipped)
                if rank == 0 and state["step"] % 100 == 0:
                    progress("training", **task, step=state["step"], lr=lr, local_loss=loss,
                             global_jets_per_second=512 * (state["step"] - start_step) / (time.monotonic() - clock),
                             amp_skipped_steps=state["amp_skipped_steps"])
                stopping = stop_requested(device, world) if state["step"] % 10 == 0 else False
                if stopping or state["step"] % 1000 == 0:
                    save_checkpoint(last, base, optimizer, scaler, state, spec, runtime, rank, world)
                if stopping:
                    if rank == 0:
                        progress("checkpointed_for_resume", **state)
                    raise SystemExit(75)
        finally:
            shutdown_loader(train_loader)
            if val_loader is not None:
                shutdown_loader(val_loader)


def benchmark(args):
    import torch
    from part_ev_fullscale.data import build_loader, shutdown_loader
    from part_ev_fullscale.optimizer import Ranger
    spec = P.load_campaign(args.output)
    rank, world, device = distributed_setup()
    result = {"runtime": runtime_contract(spec, rank, world, device), "models": {}}
    for architecture in ("part", "part_ev"):
        base, model = make_model(architecture, 101, device, world)
        optimizer = Ranger(base.parameters())
        scaler = torch.amp.GradScaler("cuda")
        loader, _ = build_loader("train", spec["inventory"], P.local_batch_size(world),
                                  rank=rank, world_size=world, seed=101)
        try:
            iterator = iter(loader)
            start = time.monotonic()
            first = next(iterator)
            cold_seconds = time.monotonic() - start
            model.train()
            skipped = 0
            for index in range(args.warmup):
                _, skip = train_step(model, optimizer, scaler, first if index == 0 else next(iterator), device)
                skipped += int(skip)
            torch.cuda.synchronize()
            barrier(world)
            torch.cuda.reset_peak_memory_stats()
            clock = time.monotonic()
            for _ in range(args.steps):
                loss, skip = train_step(model, optimizer, scaler, next(iterator), device)
                skipped += int(skip)
            torch.cuda.synchronize()
            barrier(world)
            seconds = time.monotonic() - clock
            if skipped == args.steps + args.warmup:
                raise FloatingPointError("Every benchmark optimizer step was skipped; AMP is not usable")
            memory = torch.tensor(torch.cuda.max_memory_allocated(), device=device, dtype=torch.float64)
            if world > 1:
                import torch.distributed as dist
                dist.all_reduce(memory, op=dist.ReduceOp.MAX)
            result["models"][architecture] = {
                "measured_steps": args.steps, "warmup_steps": args.warmup, "seconds": seconds,
                "global_jets_per_second": args.steps * 512 / seconds,
                "training_only_estimate_hours": seconds / args.steps * 1000000 / 3600,
                "peak_allocated_gib_per_gpu": float(memory.item() / 1024**3),
                "cold_first_batch_seconds_rank0": cold_seconds, "last_local_loss": loss,
                "amp_skipped_steps_rank0": skipped,
                "parameters": sum(p.numel() for p in base.parameters()),
            }
            # Exercise the rank-zero, unwrapped eval path used at checkpoints;
            # other ranks wait here instead of entering SyncBN/DDP collectives.
            if rank == 0:
                base.eval()
                with torch.no_grad():
                    inputs, truth = inputs_and_labels(first, device)
                    values = base(*inputs)
                    if not bool(torch.isfinite(values).all()) or values.shape != (len(truth), 10):
                        raise FloatingPointError("GPU evaluation smoke failed")
            barrier(world)
            if rank == 0:
                progress("benchmark_model_complete", architecture=architecture, **result["models"][architecture])
        finally:
            shutdown_loader(loader)
            del model, base, optimizer, scaler, loader
            torch.cuda.empty_cache()
    if rank == 0:
        result["passed"] = True
        P.write_json(args.output / "benchmarks" / f"gpus_{world}.json", result)
        progress("benchmark_complete", gpus=world, production_submitted=False)


def inspect_test(args, spec):
    from part_ev_fullscale.data import inspect_test_dataset
    manifest = inspect_test_dataset(args.test_data, check_entries=True)
    payload = {"campaign_sha256": spec["content_hash"], "inventory": manifest}
    path = args.output / "test_inventory.json"
    with P.file_lock(args.output / ".test_inventory.lock"):
        if path.exists() and P.read_json(path) != payload:
            raise ValueError("Test population differs from the existing frozen inventory")
        if not path.exists():
            P.write_json(path, payload)
    return manifest


def test(args):
    import numpy as np
    import torch
    from part_ev_fullscale.checkpoint import trusted_load
    from part_ev_fullscale.data import build_loader, shutdown_loader
    from part_ev_fullscale.metrics import calculate_metrics
    from part_ev_fullscale.model import build_model
    output = args.output.resolve()
    spec = P.load_campaign(output)
    task_dir = P.task_directory(output, args.task_index)
    task = P.TASKS[args.task_index]
    rank, world, device = distributed_setup()
    if world != 1:
        raise ValueError("Each of the six final-test tasks uses one GPU")
    # Freeze ALL six checkpoints before the first model can inspect test events.
    with P.file_lock(output / ".finalists.lock"):
        finalists = []
        for index in range(6):
            directory = P.task_directory(output, index)
            done = P.read_json(directory / "training_complete.json")
            if done["campaign_sha256"] != spec["content_hash"] or done["step"] != 1000000 or done["validated_epochs"] != 50:
                raise ValueError("All six training runs must finish before final testing")
            sha = P.file_sha256(directory / "best_model_val.pt")
            if sha != done["best_checkpoint_sha256"]:
                raise ValueError("Best checkpoint changed after training completion")
            finalists.append({"task": P.TASKS[index], "sha256": sha, "step": done["best_step"]})
        path = output / "locked_finalists.json"
        if path.exists() and P.read_json(path) != finalists:
            raise ValueError("Finalists were changed after the test lock")
        if not path.exists():
            P.write_json(path, finalists)
    inventory = inspect_test(args, spec)
    test_dir = task_dir / "test"
    with P.file_lock(test_dir / ".test.lock", blocking=False):
        if (test_dir / "metrics.json").exists():
            progress("test_already_complete", **task)
            return
        checkpoint = trusted_load(task_dir / "best_model_val.pt")
        base = build_model(task["architecture"], use_amp=True).to(device)
        base.load_state_dict(checkpoint["model"], strict=True)
        base.eval()
        # Set warmed evaluation trim behavior consistently; training checkpoints
        # store weights, not the trimmer's transient Python counter.
        for module in base.modules():
            if hasattr(module, "_counter"):
                module._counter = 5
        n_events = inventory["splits"]["test"]["event_count"]
        logits_path, truth_path = test_dir / "logits.npy", test_dir / "labels.npy"
        state_path = test_dir / "progress.json"
        state = P.read_json(state_path) if state_path.exists() else {"files": 0, "events": 0,
                        "checkpoint_sha256": finalists[args.task_index]["sha256"], "inventory_sha256": P.digest(inventory)}
        if state["checkpoint_sha256"] != finalists[args.task_index]["sha256"] or state["inventory_sha256"] != P.digest(inventory):
            raise ValueError("Test resume inputs changed")
        mode = "r+" if state_path.exists() else "w+"
        logits = np.lib.format.open_memmap(logits_path, mode=mode, dtype=np.float32, shape=(n_events, 10))
        labels = np.lib.format.open_memmap(truth_path, mode=mode, dtype=np.int8, shape=(n_events,))
        with torch.no_grad():
            for index, row in enumerate(inventory["splits"]["test"]["files"]):
                if index < state["files"]:
                    continue
                single_file = {"splits": {"test": {"files": [row]}}}
                loader, _ = build_loader("test", single_file, 512, workers=0, seed=task["seed"])
                offset = state["events"]
                start = offset
                try:
                    for batch in loader:
                        if STOP_REQUESTED:
                            logits.flush()
                            labels.flush()
                            raise SystemExit(75)  # Repeat only this incomplete ROOT file.
                        inputs, truth = inputs_and_labels(batch, device)
                        scores = base(*inputs).float().cpu().numpy()
                        if not np.isfinite(scores).all():
                            raise FloatingPointError("Nonfinite final-test logits")
                        end = offset + len(truth)
                        logits[offset:end] = scores
                        labels[offset:end] = truth.cpu().numpy()
                        offset = end
                finally:
                    shutdown_loader(loader)
                if offset - start != row["num_entries"]:
                    raise ValueError("Incomplete ROOT test file")
                logits.flush()
                labels.flush()
                state.update(files=index + 1, events=offset)
                P.write_json(state_path, state)
                progress("test_file_complete", **task, files=state["files"], events=offset)
        if state["events"] != n_events:
            raise ValueError("Incomplete test population")
        metrics = calculate_metrics(logits, labels)
        if set(metrics["class_support"].values()) != {2000000}:
            raise ValueError("Official test must have exactly 2M examples of each class")
        P.write_json(test_dir / "metrics.json", {"task": task, "campaign_sha256": spec["content_hash"],
                     "checkpoint_sha256": state["checkpoint_sha256"], "inventory_sha256": P.digest(inventory),
                     "metrics": metrics})
        progress("test_complete", **task, accuracy=metrics["accuracy"], macro_auc_ovo=metrics["macro_auc_ovo"])


def report(args):
    spec = P.load_campaign(args.output)
    results = []
    for index, task in enumerate(P.TASKS):
        result = P.read_json(P.task_directory(args.output, index) / "test/metrics.json")
        if result["campaign_sha256"] != spec["content_hash"] or result["task"] != task or result["metrics"]["events"] != 20000000:
            raise ValueError("Report needs all six full-population final-test results")
        results.append(result)
    if len({r["inventory_sha256"] for r in results}) != 1:
        raise ValueError("Results use different test populations")
    models = {}
    for architecture in ("part", "part_ev"):
        rows = [r["metrics"] for r in results if r["task"]["architecture"] == architecture]
        models[architecture] = {key: {"mean": statistics.mean(r[key] for r in rows),
                               "seed_std": statistics.stdev(r[key] for r in rows)}
                               for key in ("accuracy", "macro_auc_ovo", "macro_auc_ovr")}
        rejections = {}
        for signal_name, targets in rows[0]["qcd_rejection"].items():
            rejections[signal_name] = {}
            for efficiency in targets:
                values = [r["qcd_rejection"][signal_name][efficiency]["rejection"] for r in rows]
                finite = all(v is not None for v in values)
                rejections[signal_name][efficiency] = {
                    "per_seed": dict(zip(map(str, P.SEEDS), values)),
                    "mean": statistics.mean(values) if finite else None,
                    "seed_std": statistics.stdev(values) if finite else None,
                    "zero_background_seeds": sum(v is None for v in values)}
        models[architecture]["qcd_rejection"] = rejections
    paired_deltas = [results[i + 3]["metrics"]["accuracy"] - results[i]["metrics"]["accuracy"] for i in range(3)]
    summary = {"campaign_sha256": spec["content_hash"], "models": models, "per_seed_results": results,
               "paired_seed_accuracy_gains": paired_deltas, "mean_accuracy_gain": statistics.mean(paired_deltas)}
    P.write_json(args.output / "reports/fullscale_report.json", summary)
    progress("report_complete", models=models, mean_accuracy_gain=summary["mean_accuracy_gain"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "benchmark", "train", "test", "report"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--test-data", type=Path, help="Directory containing the 200 official test ROOT files")
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.phase == "prepare" and args.data_root is None:
        parser.error("prepare requires --data-root")
    if args.phase in ("train", "test") and (args.task_index is None or not 0 <= args.task_index < 6):
        parser.error("train/test requires --task-index 0..5")
    if args.phase == "test" and args.test_data is None:
        parser.error("test requires --test-data; training never opens test files")
    if args.steps < 1 or args.warmup < 1:
        parser.error("benchmark steps/warmup must be positive")
    try:
        globals()[args.phase](args)
    finally:
        if "torch.distributed" in sys.modules:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
