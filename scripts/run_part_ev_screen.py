#!/usr/bin/env python3
"""Isolated SPORC screen: prepare, benchmark, train, full-validation, report."""
from __future__ import annotations

import argparse
import copy
from contextlib import nullcontext
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from part_ev_screen import protocol as P
from part_ev_screen.model import build_model, diagnostic_mode, diagnostics
from part_ev_screen.precision import model_precision, audit_precision, POLICY as PRECISION_POLICY
from part_ev_screen.data import (build_loader, finite_validation_loader, batch_fingerprint,
                                 shutdown_loader)
from part_ev_fullscale.checkpoint import (atomic_torch_save, trusted_load, seed_everything,
    capture_rank_state, restore_rank_state)
from part_ev_fullscale.optimizer import build_optimizer, optimizer_metadata

STOP = False


def progress(stage, **fields):
    print(json.dumps(dict(stage=stage, **fields), allow_nan=False), flush=True)


def stop_signal(*_):
    global STOP
    STOP = True


def check_stop():
    if STOP:
        raise SystemExit(75)


def gpu_setup(spec):
    if platform.machine() != "x86_64" or not torch.cuda.is_available():
        raise RuntimeError("Screen GPU phases require x86_64 SPORC CUDA")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or torch.cuda.device_count() != 1:
        raise RuntimeError("Request exactly one visible GPU per task")
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    for name in ("SIGTERM", "SIGUSR1"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), stop_signal)
    device = torch.device("cuda:0")
    runtime = dict(gpu=torch.cuda.get_device_name(0), gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
        world_size=1, batch_size=512, workers=2, batchnorm="native", tf32=False,
        cudnn_benchmark=False, precision=PRECISION_POLICY, campaign_sha256=spec["content_hash"])
    return device, runtime


def make_model(task, device):
    seed_everything(task["seed"])
    # Default to FP32 for evaluation; training scopes BOTH native and outer AMP.
    return build_model(task["architecture"], seed=task["seed"], trim=True, use_amp=False).to(device)


def initial_parity(model, task, batch, device):
    """Same-platform native/control parity, never historical Tigris logits."""
    from part_ev_fullscale.model import build_model as original_model
    rng = capture_rank_state(model)
    was_training = model.training
    try:
        seed_everything(task["seed"])
        architecture = "part" if task["architecture"] == "part" else "part_ev"
        reference = original_model(architecture, trim=True, use_amp=False).to(device).eval()
        for key, tensor in reference.state_dict().items():
            if not torch.equal(model.state_dict()[key], tensor):
                raise ValueError(f"Initial native/EV parameter mismatch: {key}")
        model.eval()
        values, _ = tensors(batch, device)
        with torch.no_grad(), model_precision(model, device, training=False):
            expected = reference(*(v[:8] for v in values))
            actual = model(*(v[:8] for v in values))
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        return dict(passed=True, maximum_absolute_error=float((actual-expected).abs().max()), events=len(actual))
    finally:
        restore_rank_state(model, rng)
        model.train(was_training)


def tensors(batch, device):
    values, labels = batch[:2]
    return ([values[k].to(device, non_blocking=True) for k in
             ("pf_points", "pf_features", "pf_vectors", "pf_mask")],
            labels["_label_"].to(device, non_blocking=True).long())


def training_step(model, optimizer, scaler, batch, device, recipe, step, *, full_length=False,
                  check_precision=False):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    lr = P.learning_rate(step, recipe)
    for group in optimizer.param_groups:
        group["lr"] = lr
    values, truth = tensors(batch, device)
    if full_length:
        # A synthetic worst-length memory probe ONLY. Replicate a real particle
        # into padding, avoiding unphysical zero four-vectors in this probe.
        mask = values[-1].bool()
        values = [torch.where(mask, x, x[:, :, :1].expand_as(x)) for x in values[:-1]] + [torch.ones_like(values[-1])]
        if values[-1].shape[-1] != 128:
            raise ValueError("Full-length benchmark requires 128 particle slots")
    old_scale = scaler.get_scale()
    expected = torch.float16 if device.type == "cuda" else torch.float32
    audit = audit_precision(model, expected_dtype=expected) if check_precision else nullcontext()
    with model_precision(model, device, training=True), audit as precision:
        logits = model(*values)
        loss = torch.nn.functional.cross_entropy(logits, truth)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Nonfinite training loss")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    limit = recipe["gradient_clipping"]
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf") if limit is None else limit)
    finite = bool(torch.isfinite(norm))
    scaler.step(optimizer)
    scaler.update()
    skipped = scaler.get_scale() < old_scale
    if not finite and not skipped:
        raise FloatingPointError("Nonfinite gradients without an AMP-skipped update")
    result = dict(loss=float(loss.detach()), lr=lr, amp_skipped=skipped,
                  gradient_norm=float(norm) if finite else None,
                  clipped=bool(finite and limit is not None and float(norm) > limit))
    if check_precision:
        result["precision"] = precision
    return result


def classification_metrics(logits, labels, *, rejection=False):
    from scipy.special import softmax, logsumexp
    from sklearn.metrics import roc_auc_score
    if logits.shape != (len(labels), 10) or not np.isfinite(logits).all() or not len(labels):
        raise ValueError("Invalid or nonfinite validation predictions")
    counts = np.bincount(labels, minlength=10)
    if counts.shape != (10,) or np.any(counts == 0):
        raise ValueError("Validation must contain every class")
    scores = logits.astype(np.float64)
    row = dict(events=len(labels), accuracy=float(np.mean(scores.argmax(1) == labels)),
        loss=float(np.mean(logsumexp(scores, axis=1) - scores[np.arange(len(labels)), labels])),
        class_counts={name: int(n) for name, n in zip(P.CLASS_NAMES, counts)})
    if rejection:
        from part_ev_fullscale.metrics import calculate_metrics
        # Reuse the established tie/count convention, including finite=None
        # when zero background passes. Never replace that with a finite mean.
        row.update(calculate_metrics(logits, labels))
    else:
        probability = softmax(scores, axis=1)
        row.update(macro_auc_ovo=float(roc_auc_score(labels, probability, multi_class="ovo", average="macro")),
                   macro_auc_ovr=float(roc_auc_score(labels, probability, multi_class="ovr", average="macro")))
    return row


def context_diagnostic(model, batch, device):
    # A fixed, separately loaded validation batch; restore stochastic/trimmer
    # state so collecting diagnostics cannot change subsequent optimization.
    rng = capture_rank_state(model)
    was_training = model.training
    hooks, ordinary, control_records = [], {}, {}
    try:
        from part_ev_fullscale.model import EdgeValueAttention
        # Observe existing EV controls without replacing their implementation.
        for name, module in model.named_modules():
            if isinstance(module, EdgeValueAttention):
                def native_hook(_module, _inputs, output, key=name):
                    ordinary[key] = output[0]
                def ev_hook(_module, _inputs, output, key=name):
                    native = ordinary.pop(key)
                    control_records[key] = dict(correction_to_native_norm=float(
                        (output[0]-native).float().norm() / native.float().norm().clamp_min(1e-12)))
                hooks.append(module.reference.register_forward_hook(native_hook))
                hooks.append(module.register_forward_hook(ev_hook))
        model.eval()
        diagnostic_mode(model, True)
        with torch.no_grad(), model_precision(model, device, training=False):
            values, _ = tensors(batch, device)
            model(*values)
        return dict(input_fingerprint=batch_fingerprint(batch), layers={**control_records, **diagnostics(model)})
    finally:
        for hook in hooks:
            hook.remove()
        diagnostic_mode(model, False)
        restore_rank_state(model, rng)
        model.train(was_training)


def validate(model, iterator, device, events, batch_size=512, *, check_precision=False):
    model.eval()
    logits = np.empty((events, 10), np.float32)
    labels = np.empty(events, np.int64)
    offset = 0
    started = time.monotonic()
    precision = None
    with torch.no_grad(), model_precision(model, device, training=False):
        for index in range(events // batch_size):
            check_stop()
            values, truth = tensors(next(iterator), device)
            audit = (audit_precision(model, expected_dtype=torch.float32)
                     if check_precision and index == 0 else nullcontext())
            with audit as observed:
                prediction = model(*values).float().cpu().numpy()
            if observed is not None:
                precision = observed
            n = len(truth)
            logits[offset:offset+n] = prediction
            labels[offset:offset+n] = truth.cpu().numpy()
            offset += n
    if offset != events:
        raise ValueError("Validation sample budget differs")
    inference_seconds = time.monotonic() - started
    result = classification_metrics(logits, labels)
    result.update(inference_seconds=inference_seconds, seconds=time.monotonic() - started)
    if check_precision:
        if precision is None:
            raise RuntimeError("Validation precision audit did not run")
        result["precision"] = precision
    return result


def prepare(args):
    from part_ev_fullscale.data import inspect_dataset
    output = args.output.resolve()
    launch = P.read_json(output / "launch.json")
    if platform.machine() != "x86_64" or torch.version.cuda is None:
        raise RuntimeError("Prepare requires the dedicated CUDA-enabled x86_64 environment")
    with P.file_lock(output / ".prepare.lock", blocking=False):
        if (output / "campaign.json").exists():
            P.load_campaign(output)
            progress("already_prepared")
            return
        inventory = inspect_dataset(launch["data_root"], check_entries=True, splits=("train", "val"))
        if shutil.disk_usage(output).free < 8 * 1024**3:
            raise OSError("Keep at least 8 GiB free for this screen's checkpoints, logits and headroom")
        for task in P.TASKS:
            seed_everything(task["seed"])
            model = build_model(task["architecture"], seed=task["seed"])
            progress("model_contract", **task, parameters=sum(p.numel() for p in model.parameters()))
        spec = P.create_spec(inventory, Path(__file__).resolve().parents[1], launch)
        P.write_json(output / "campaign.json", spec)
        progress("prepared", campaign_sha256=spec["content_hash"])


def benchmark(args):
    spec = P.load_campaign(args.output)
    task = spec["tasks"][args.task_index]
    recipe, limits = spec["recipes"][task["recipe"]], spec["benchmark"]
    device, runtime = gpu_setup(spec)
    directory = args.output / "benchmarks"
    path = directory / (task["run_id"] + ".json")
    with P.file_lock(directory / (task["run_id"] + ".lock"), blocking=False):
        if path.exists():
            existing = P.read_json(path)
            if existing.get("passed") and existing.get("runtime") == runtime:
                progress("benchmark_already_complete", **task)
                return
            raise ValueError("Failed/different benchmark artifact; inspect and use a new campaign")
        loader, _ = build_loader("train", spec["inventory"], 512, seed=task["seed"])
        val_loader = None
        try:
            model = make_model(task, device)
            optimizer = build_optimizer(model, recipe)
            scaler = torch.amp.GradScaler("cuda")
            iterator = iter(loader)
            seed_everything(task["seed"])
            started = time.monotonic()
            first = next(iterator)
            cold = time.monotonic() - started
            parity = initial_parity(model, task, first, device)
            fingerprints = {"0": batch_fingerprint(first)}
            window_start = time.monotonic()
            windows, skipped = [], 0
            torch.cuda.reset_peak_memory_stats()
            warmup, measured = limits["warmup_steps"], limits["measured_steps"]
            progress("benchmark_training_start", **task, batch_size=512, precision=PRECISION_POLICY)
            for index in range(warmup + measured):
                check_stop()
                batch = first if index == 0 else next(iterator)
                if index in (warmup, 1020, 2020, 4020):
                    fingerprints[str(index)] = batch_fingerprint(batch)
                # Benchmark actual early schedule; separate full-length probe
                # below also tests the peak AdamW LR.
                row = training_step(model, optimizer, scaler, batch, device, recipe, index,
                                    check_precision=index == 0)
                if index == 0:
                    training_precision = row["precision"]
                    progress("benchmark_precision", **task, passed=training_precision["passed"],
                             activation_dtype=training_precision["expected_activation_dtype"],
                             observed_modules=len(training_precision["observed_activations"]),
                             peak_allocated_bytes=torch.cuda.max_memory_allocated())
                if index >= warmup:
                    skipped += row["amp_skipped"]
                if index + 1 == warmup:
                    torch.cuda.synchronize()
                    window_start = measured_start = time.monotonic()
                if index >= warmup and ((index+1-warmup) % limits["window_steps"] == 0 or index+1 == warmup+measured):
                    torch.cuda.synchronize()
                    stop = time.monotonic()
                    count = (index+1-warmup) % limits["window_steps"] or limits["window_steps"]
                    windows.append(dict(steps=count, seconds=stop-window_start))
                    window_start = stop
                    progress("benchmark_window", **task, measured_steps=index+1-warmup, **row, seconds=windows[-1]["seconds"])
            seconds = time.monotonic() - measured_start
            native_memory = torch.cuda.max_memory_allocated()
            # Use the very same model and optimizer; no synthetic records ever
            # enter production or validation. Force the trimmer to retain 128.
            for module in model.modules():
                if type(module).__name__ == "SequenceTrimmer":
                    module.enabled = False
            torch.cuda.reset_peak_memory_stats()
            full_skips = 0
            diagnostic_mode(model, True)
            for offset in range(limits["full_length_steps"]):
                peak_step = recipe.get("warmup_steps", 1) - 1 + offset
                row = training_step(model, optimizer, scaler, first, device, recipe, peak_step, full_length=True)
                full_skips += row["amp_skipped"]
            diagnostic_mode(model, False)
            full_memory = torch.cuda.max_memory_allocated()
            checkpoint_clock = time.monotonic()
            # Keep this separate benchmark artifact, never a production weight.
            save_checkpoint(directory / (task["run_id"] + "_probe.pt"), model, optimizer, scaler,
                            dict(task=task, step=warmup+measured), spec, runtime)
            checkpoint_seconds = time.monotonic() - checkpoint_clock
            val_loader, _ = build_loader("val", spec["inventory"], 512, seed=task["seed"])
            torch.cuda.reset_peak_memory_stats()
            val = validate(model, iter(val_loader), device, limits["validation_events"], check_precision=True)
            validation_memory = torch.cuda.max_memory_allocated()
            # Use the slowest complete fetch window, not just a warm cached batch.
            per_step = max(w["seconds"]/w["steps"] for w in windows)
            projected = (per_step*recipe["total_steps"] + val["seconds"] * 10 * recipe["validation_events_per_epoch"]/val["events"] + 220*checkpoint_seconds + cold) / 3600
            passed = (skipped / measured < limits["max_skip_fraction"] and full_skips == 0
                      and training_precision["passed"] and val["precision"]["passed"]
                      and max(native_memory, full_memory, validation_memory) < runtime["gpu_memory_bytes"] * limits["memory_fraction_limit"])
            result = dict(passed=passed, task=task, campaign_sha256=spec["content_hash"], runtime=runtime,
                input_fingerprints=fingerprints, cold_fetch_seconds=cold, measured_steps=measured,
                initial_parity=parity,
                training_precision=training_precision,
                seconds=seconds, jets_per_second=512*measured/seconds, amp_skipped_steps=skipped,
                checkpoint_seconds=checkpoint_seconds,
                full_length_amp_skips=full_skips, native_peak_bytes=native_memory, full_length_peak_bytes=full_memory,
                validation_peak_bytes=validation_memory,
                windows=windows, validation=val, projected_training_hours=projected)
            P.write_json(path, result)
            progress("benchmark_complete", **result)
            if not passed:
                raise RuntimeError("Benchmark did not pass memory/numerical gates; no production submission")
        finally:
            shutdown_loader(loader)
            if val_loader is not None:
                shutdown_loader(val_loader)


def save_checkpoint(path, model, optimizer, scaler, state, spec, runtime):
    atomic_torch_save(dict(schema=P.SCHEMA, campaign_sha256=spec["content_hash"], runtime=runtime,
        model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
        state=copy.deepcopy(state), rng=capture_rank_state(model)), path)


def load_checkpoint(path, model, optimizer, scaler, spec, runtime, task):
    checkpoint = trusted_load(path)
    if (checkpoint.get("schema") != P.SCHEMA or checkpoint["campaign_sha256"] != spec["content_hash"]
            or checkpoint["runtime"] != runtime or checkpoint["state"]["task"] != task):
        raise ValueError("Checkpoint belongs to another task/campaign/runtime")
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint["state"], checkpoint["rng"]


def replay(iterator, batches):
    for index in range(batches):
        check_stop()
        next(iterator)
        if (index+1) % 10000 == 0:
            progress("resume_data_replay", batches=index+1, total=batches)


def train(args):
    spec = P.load_campaign(args.output)
    task = spec["tasks"][args.task_index]
    recipe = spec["recipes"][task["recipe"]]
    device, runtime = gpu_setup(spec)
    P.benchmark_summary(args.output, spec, runtime)
    run = P.task_directory(args.output, args.task_index)
    with P.file_lock(run / ".train.lock", blocking=False):
        if (run / "training_complete.json").exists():
            complete = P.read_json(run / "training_complete.json")
            if (complete["best_checkpoint_sha256"] != P.file_sha256(run / "best_model_val.pt")
                    or complete["campaign_sha256"] != spec["content_hash"] or complete["task"] != task
                    or complete["runtime"] != runtime or complete["step"] != recipe["total_steps"]):
                raise ValueError("Completed checkpoint changed")
            progress("already_trained", **task)
            return
        model = make_model(task, device)
        optimizer = build_optimizer(model, recipe)
        scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
        P.write_json(run / "optimizer.json", optimizer_metadata(model, optimizer))
        state = dict(task=task, step=0, validated_checks=0, best_accuracy=-1., best_check=None,
                     amp_skipped_steps=0, clipped_steps=0, train_seconds=0., validation_seconds=0.)
        last, rng = run / "last.pt", None
        if last.exists():
            state, rng = load_checkpoint(last, model, optimizer, scaler, spec, runtime, task)
        train_loader, _ = build_loader("train", spec["inventory"], 512, seed=task["seed"])
        val_loader, _ = build_loader("val", spec["inventory"], 512, seed=task["seed"])
        try:
            iterator, val_iterator = iter(train_loader), iter(val_loader)
            diagnostic_loader, _ = build_loader("val", spec["inventory"], 512, workers=0, seed=task["seed"])
            diagnostic_batch = next(iter(diagnostic_loader))
            del diagnostic_loader
            if rng is not None:
                replay(iterator, state["step"])
                replay(val_iterator, state["validated_checks"] * (recipe["validation_events_per_epoch"] // 512))
                restore_rank_state(model, rng)
            else:
                seed_everything(task["seed"])
            while True:
                if STOP:
                    save_checkpoint(last, model, optimizer, scaler, state, spec, runtime)
                    raise SystemExit(75)
                if state["step"] // recipe["steps_per_epoch"] > state["validated_checks"]:
                    save_checkpoint(last, model, optimizer, scaler, state, spec, runtime)
                    metric = validate(model, val_iterator, device, recipe["validation_events_per_epoch"])
                    state["validated_checks"] += 1
                    state["validation_seconds"] += metric["seconds"]
                    metric.update(check=state["validated_checks"], step=state["step"])
                    if metric["accuracy"] > state["best_accuracy"]:
                        state["best_accuracy"], state["best_check"] = metric["accuracy"], metric["check"]
                        atomic_torch_save(dict(model=model.state_dict(), task=task, runtime=runtime,
                            campaign_sha256=spec["content_hash"], validation=metric), run / "best_model_val.pt")
                    P.write_json(run / "validation" / f"check_{metric['check']:02d}.json", metric)
                    P.write_json(run / "diagnostics" / f"check_{metric['check']:02d}.json",
                        dict(check=metric["check"], **context_diagnostic(model, diagnostic_batch, device)))
                    progress("validation", **task, **metric)
                    save_checkpoint(last, model, optimizer, scaler, state, spec, runtime)
                if state["step"] == recipe["total_steps"]:
                    P.write_json(run / "training_complete.json", dict(state,
                        campaign_sha256=spec["content_hash"], runtime=runtime,
                        best_checkpoint_sha256=P.file_sha256(run / "best_model_val.pt")))
                    progress("training_complete", **task)
                    return
                started = time.monotonic()
                row = training_step(model, optimizer, scaler, next(iterator), device, recipe, state["step"])
                state["train_seconds"] += time.monotonic() - started
                state["step"] += 1
                state["amp_skipped_steps"] += int(row["amp_skipped"])
                state["clipped_steps"] += int(row["clipped"])
                if state["step"] % 100 == 0:
                    progress("training", **task, step=state["step"], **row,
                        jets_per_second=state["step"]*512/state["train_seconds"],
                        amp_skipped_steps=state["amp_skipped_steps"])
                if state["step"] % 1000 == 0:
                    save_checkpoint(last, model, optimizer, scaler, state, spec, runtime)
        finally:
            shutdown_loader(train_loader)
            shutdown_loader(val_loader)


def lock_finalists(output, spec):
    path = output / "selection" / "locked_checkpoints.json"
    with P.file_lock(output / ".selection.lock"):
        rows = []
        for index, task in enumerate(spec["tasks"]):
            run = P.task_directory(output, index)
            complete = P.read_json(run / "training_complete.json")
            checkpoint = run / "best_model_val.pt"
            recipe = spec["recipes"][task["recipe"]]
            if (complete["campaign_sha256"] != spec["content_hash"] or complete["step"] != recipe["total_steps"]
                    or complete["validated_checks"] != recipe["epochs"] or complete["task"] != task
                    or P.file_sha256(checkpoint) != complete["best_checkpoint_sha256"]):
                raise ValueError("Every training must complete with an unchanged selected checkpoint")
            rows.append(dict(task=task, checkpoint=str(checkpoint), sha256=complete["best_checkpoint_sha256"]))
        payload = dict(campaign_sha256=spec["content_hash"], checkpoints=rows)
        if path.exists() and P.read_json(path) != payload:
            raise ValueError("Selected checkpoints changed after full-validation lock")
        if not path.exists():
            P.write_json(path, payload)
        return payload


def atomic_npz(path, **arrays):
    temporary = path.with_suffix(".tmp")
    with open(temporary, "wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def evaluate(args):
    spec = P.load_campaign(args.output)
    task = spec["tasks"][args.task_index]
    device, runtime = gpu_setup(spec)
    P.benchmark_summary(args.output, spec, runtime)
    selected = lock_finalists(args.output, spec)["checkpoints"][args.task_index]
    run = P.task_directory(args.output, args.task_index)
    directory = run / "full_validation"
    with P.file_lock(directory / ".evaluation.lock", blocking=False):
        model = make_model(task, device).eval()
        checkpoint = trusted_load(selected["checkpoint"])
        if checkpoint["task"] != task or checkpoint["runtime"] != runtime or checkpoint["campaign_sha256"] != spec["content_hash"]:
            raise ValueError("Selected checkpoint contract mismatch")
        model.load_state_dict(checkpoint["model"], strict=True)
        for module in model.modules():
            if hasattr(module, "_counter"):
                module._counter = 5
        prediction_parts, label_parts, records = [], [], []
        for index, record in enumerate(sorted(spec["inventory"]["splits"]["val"]["files"], key=lambda r: r["name"])):
            check_stop()
            path = directory / f"file_{index:03d}.npz"
            sidecar = path.with_suffix(".json")
            contract = dict(campaign_sha256=spec["content_hash"], checkpoint_sha256=selected["sha256"],
                            file=record, entry_start=0, entry_stop=record["num_entries"])
            reusable = path.exists() and sidecar.exists()
            if reusable:
                saved = P.read_json(sidecar)
                if saved["contract"] != contract or saved["sha256"] != P.file_sha256(path):
                    raise ValueError("Committed validation prediction changed")
            else:
                loader = finite_validation_loader(spec["inventory"], record)
                predictions, labels = [], []
                with torch.no_grad(), model_precision(model, device, training=False):
                    for batch in loader:
                        check_stop()
                        values, truth = tensors(batch, device)
                        prediction = model(*values).float().cpu().numpy()
                        if not np.isfinite(prediction).all():
                            raise FloatingPointError("Nonfinite full-validation prediction")
                        predictions.append(prediction)
                        labels.append(truth.cpu().numpy())
                logits, truth = np.concatenate(predictions), np.concatenate(labels)
                if len(truth) != record["num_entries"]:
                    raise ValueError("Incomplete finite validation file")
                atomic_npz(path, logits=logits, labels=truth)
                P.write_json(sidecar, dict(contract=contract, sha256=P.file_sha256(path)))
            with np.load(path, allow_pickle=False) as saved:
                prediction_parts.append(saved["logits"])
                label_parts.append(saved["labels"])
            records.append(P.read_json(sidecar))
            progress("full_validation_file", **task, completed=index+1, total=50)
        logits, truth = np.concatenate(prediction_parts), np.concatenate(label_parts)
        population = spec["recipes"][task["recipe"]]["evaluation_population"]
        if len(truth) != population or not np.array_equal(np.bincount(truth, minlength=10), np.full(10, population//10)):
            raise ValueError("Full validation must contain exactly 500k events in each of ten classes")
        metrics = classification_metrics(logits, truth, rejection=True)
        P.write_json(directory / "metrics.json", dict(task=task, campaign_sha256=spec["content_hash"],
            checkpoint_sha256=selected["sha256"], identity_manifest=records, metrics=metrics))
        progress("full_validation_complete", **task, accuracy=metrics["accuracy"], macro_auc_ovo=metrics["macro_auc_ovo"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("prepare", "benchmark", "review", "train", "evaluate", "report"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--task-index", type=int)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.phase in ("benchmark", "train", "evaluate"):
        P.task_directory(args.output, args.task_index)
    if args.phase == "review":
        spec = P.load_campaign(args.output)
        print(json.dumps(P.benchmark_summary(args.output, spec), indent=2))
    elif args.phase == "report":
        from part_ev_screen.reporting import report
        report(args.output)
    else:
        globals()[args.phase](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
