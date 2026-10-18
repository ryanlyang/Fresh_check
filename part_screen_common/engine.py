"""Explicit protocol/model callbacks; no mutation of the seven-task screen.

Reuse its actual pure step, validation and metric functions. Orchestration here
is protocol-parameterized; the historical driver and its globals stay untouched.
"""
from __future__ import annotations
import copy
import signal
import time
from pathlib import Path
import numpy as np
import torch

from scripts.run_part_ev_screen import (training_step, validate, classification_metrics,
    tensors, atomic_npz, progress, gpu_setup as reference_gpu_setup)
from part_ev_screen.data import build_loader, shutdown_loader, finite_validation_loader, batch_fingerprint
from part_ev_screen.precision import model_precision
from part_ev_fullscale.optimizer import build_optimizer, optimizer_metadata
from part_ev_fullscale.checkpoint import (seed_everything, capture_rank_state, restore_rank_state,
    atomic_torch_save, trusted_load)
from .contracts import runtime_identity
from .controls import selected_training


class ScreenEngine:
    def __init__(self, protocol, model_factory, parity, diagnostics):
        self.P, self.model_factory = protocol, model_factory
        self.parity, self.diagnostics = parity, diagnostics
        self.stopping = False

    def signal(self, *_):
        self.stopping = True

    def check_stop(self):
        if self.stopping:
            raise SystemExit(75)

    def gpu_setup(self, spec):
        device, runtime = reference_gpu_setup(spec)
        if runtime_identity(runtime) != spec["donor"]["runtime"]:
            raise ValueError("GPU/runtime differs from frozen donor contract")
        for name in ("SIGTERM", "SIGUSR1"):
            if hasattr(signal, name):
                signal.signal(getattr(signal, name), self.signal)
        return device, runtime

    def make_model(self, task, device):
        seed_everything(task["seed"])
        return self.model_factory(task["architecture"], seed=task["seed"], trim=True, use_amp=False).to(device)

    def save(self, path, model, optimizer, scaler, state, spec, runtime):
        atomic_torch_save(dict(schema=self.P.SCHEMA, campaign_sha256=spec["content_hash"], runtime=runtime,
            model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
            state=copy.deepcopy(state), rng=capture_rank_state(model)), path)

    def load(self, path, model, optimizer, scaler, spec, runtime, task):
        row = trusted_load(path)
        if (row.get("schema") != self.P.SCHEMA or row.get("campaign_sha256") != spec["content_hash"]
                or row.get("runtime") != runtime or row["state"].get("task") != task):
            raise ValueError("Checkpoint belongs to another task/campaign/runtime")
        model.load_state_dict(row["model"], strict=True)
        optimizer.load_state_dict(row["optimizer"])
        scaler.load_state_dict(row["scaler"])
        return row["state"], row["rng"]

    def replay(self, iterator, count):
        for i in range(count):
            self.check_stop()
            next(iterator)
            if (i + 1) % 10000 == 0:
                progress("resume_data_replay", batches=i + 1, total=count)

    def benchmark(self, args):
        P = self.P
        spec = P.load_campaign(args.output)
        task = spec["tasks"][args.task_index]
        recipe, limits = spec["recipes"][task["recipe"]], spec["benchmark"]
        device, runtime = self.gpu_setup(spec)
        directory = args.output / "benchmarks"
        path = directory / f"{task['run_id']}.json"
        with P.file_lock(directory / f"{task['run_id']}.lock", blocking=False):
            if path.exists():
                previous = P.read_json(path)
                if (previous.get("passed") is True and previous.get("runtime") == runtime
                        and previous.get("campaign_sha256") == spec["content_hash"] and previous.get("task") == task):
                    progress("benchmark_already_complete", **task)
                    return
                raise ValueError("Failed/different benchmark: preserve evidence and use a new campaign")
            loader, _ = build_loader("train", spec["inventory"], 512, seed=task["seed"])
            val_loader = None
            try:
                model = self.make_model(task, device)
                if sum(p.numel() for p in model.parameters()) != spec["parameters"][task["architecture"]]:
                    raise ValueError("Parameter count differs from prepare")
                optimizer = build_optimizer(model, recipe)
                scaler = torch.amp.GradScaler("cuda")
                iterator = iter(loader)
                seed_everything(task["seed"])
                begin = time.monotonic()
                first = next(iterator)
                cold = time.monotonic() - begin
                parity = self.parity(model, task, first, device)
                fingerprints = {"0": batch_fingerprint(first)}
                warmup, measured = limits["warmup_steps"], limits["measured_steps"]
                windows, skipped = [], 0
                torch.cuda.reset_peak_memory_stats()
                for index in range(warmup + measured):
                    self.check_stop()
                    batch = first if index == 0 else next(iterator)
                    if index in (warmup, 1020, 2020, 4020):
                        fingerprints[str(index)] = batch_fingerprint(batch)
                    row = training_step(model, optimizer, scaler, batch, device, recipe, index,
                                        check_precision=index == 0)
                    if index == 0:
                        training_precision = row["precision"]
                    if index >= warmup:
                        skipped += row["amp_skipped"]
                    if index + 1 == warmup:
                        torch.cuda.synchronize()
                        measured_start = window_start = time.monotonic()
                    if index >= warmup and ((index + 1 - warmup) % limits["window_steps"] == 0 or index + 1 == warmup + measured):
                        torch.cuda.synchronize()
                        stop = time.monotonic()
                        count = (index + 1 - warmup) % limits["window_steps"] or limits["window_steps"]
                        windows.append(dict(steps=count, seconds=stop - window_start))
                        window_start = stop
                        progress("benchmark_window", **task, measured_steps=index + 1 - warmup, **row, window=windows[-1])
                seconds = time.monotonic() - measured_start
                native_memory = torch.cuda.max_memory_allocated()
                for module in model.modules():
                    if type(module).__name__ == "SequenceTrimmer":
                        module.enabled = False
                torch.cuda.reset_peak_memory_stats()
                full_skips = 0
                for index in range(limits["full_length_steps"]):
                    self.check_stop()
                    row = training_step(model, optimizer, scaler, first, device, recipe, index, full_length=True)
                    full_skips += row["amp_skipped"]
                full_memory = torch.cuda.max_memory_allocated()
                start = time.monotonic()
                self.save(directory / f"{task['run_id']}_probe.pt", model, optimizer, scaler,
                          dict(task=task, step=warmup + measured), spec, runtime)
                checkpoint_seconds = time.monotonic() - start
                val_loader, _ = build_loader("val", spec["inventory"], 512, seed=task["seed"])
                torch.cuda.reset_peak_memory_stats()
                validation = validate(model, iter(val_loader), device, limits["validation_events"], check_precision=True)
                validation_memory = torch.cuda.max_memory_allocated()
                diagnostics = self.diagnostics(model, first, device)
                per_step = max(w["seconds"] / w["steps"] for w in windows)
                projected = (per_step * recipe["total_steps"] + validation["seconds"] * recipe["epochs"]
                    * recipe["validation_events_per_epoch"] / validation["events"] + 220 * checkpoint_seconds + cold) / 3600
                inputs_match = fingerprints == spec["donor"]["input_fingerprints"]
                passed = (inputs_match and skipped / measured < limits["max_skip_fraction"] and full_skips == 0
                    and training_precision["passed"] and validation["precision"]["passed"]
                    and max(native_memory, full_memory, validation_memory) < runtime["gpu_memory_bytes"] * limits["memory_fraction_limit"])
                result = dict(passed=passed, task=task, campaign_sha256=spec["content_hash"], runtime=runtime,
                    initial_parity=parity, input_fingerprints=fingerprints, inputs_match_donor=inputs_match,
                    training_precision=training_precision, validation=validation, diagnostics=diagnostics,
                    measured_steps=measured, seconds=seconds, jets_per_second=512 * measured / seconds,
                    cold_fetch_seconds=cold, windows=windows, amp_skipped_steps=skipped, full_length_amp_skips=full_skips,
                    native_peak_bytes=native_memory, full_length_peak_bytes=full_memory,
                    validation_peak_bytes=validation_memory, checkpoint_seconds=checkpoint_seconds,
                    projected_training_hours=projected)
                P.write_json(path, result)
                progress("benchmark_complete", **result)
                if not passed:
                    raise RuntimeError("Grouped benchmark failed; no production admission")
            finally:
                shutdown_loader(loader)
                if val_loader is not None:
                    shutdown_loader(val_loader)

    def train(self, args):
        P = self.P
        spec = P.load_campaign(args.output)
        task = spec["tasks"][args.task_index]
        recipe = spec["recipes"][task["recipe"]]
        device, runtime = self.gpu_setup(spec)
        P.require_approval(args.output, spec, runtime)
        run = P.task_directory(args.output, args.task_index)
        with P.file_lock(run / ".train.lock", blocking=False):
            if (run / "training_complete.json").exists():
                selected_training(run, task, spec["content_hash"], recipe, runtime_identity(runtime))
                progress("already_trained", **task)
                return
            model = self.make_model(task, device)
            optimizer = build_optimizer(model, recipe)
            scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
            P.write_json(run / "optimizer.json", optimizer_metadata(model, optimizer))
            state = dict(task=task, step=0, validated_checks=0, best_accuracy=-1., best_check=None,
                amp_skipped_steps=0, clipped_steps=0, train_seconds=0., validation_seconds=0.)
            last, rng = run / "last.pt", None
            if last.exists():
                state, rng = self.load(last, model, optimizer, scaler, spec, runtime, task)
            train_loader, _ = build_loader("train", spec["inventory"], 512, seed=task["seed"])
            val_loader, _ = build_loader("val", spec["inventory"], 512, seed=task["seed"])
            try:
                iterator, val_iterator = iter(train_loader), iter(val_loader)
                diagnostic_loader, _ = build_loader("val", spec["inventory"], 512, workers=0, seed=task["seed"])
                diagnostic_batch = next(iter(diagnostic_loader))
                shutdown_loader(diagnostic_loader)
                del diagnostic_loader
                if rng is not None:
                    self.replay(iterator, state["step"])
                    self.replay(val_iterator, state["validated_checks"] * (recipe["validation_events_per_epoch"] // 512))
                    restore_rank_state(model, rng)
                else:
                    seed_everything(task["seed"])
                while True:
                    if self.stopping:
                        self.save(last, model, optimizer, scaler, state, spec, runtime)
                        raise SystemExit(75)
                    if state["step"] // recipe["steps_per_epoch"] > state["validated_checks"]:
                        self.save(last, model, optimizer, scaler, state, spec, runtime)
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
                            dict(check=metric["check"], **self.diagnostics(model, diagnostic_batch, device)))
                        progress("validation", **task, **metric)
                        self.save(last, model, optimizer, scaler, state, spec, runtime)
                    if state["step"] == recipe["total_steps"]:
                        P.write_json(run / "training_complete.json", dict(state, campaign_sha256=spec["content_hash"],
                            runtime=runtime, best_checkpoint_sha256=P.file_sha256(run / "best_model_val.pt")))
                        progress("training_complete", **task)
                        return
                    start = time.monotonic()
                    row = training_step(model, optimizer, scaler, next(iterator), device, recipe, state["step"])
                    state["train_seconds"] += time.monotonic() - start
                    state["step"] += 1
                    state["amp_skipped_steps"] += int(row["amp_skipped"])
                    state["clipped_steps"] += int(row["clipped"])
                    if state["step"] % 100 == 0:
                        progress("training", **task, step=state["step"], **row,
                            jets_per_second=state["step"] * 512 / state["train_seconds"],
                            amp_skipped_steps=state["amp_skipped_steps"])
                    if state["step"] % 1000 == 0:
                        self.save(last, model, optimizer, scaler, state, spec, runtime)
            finally:
                shutdown_loader(train_loader)
                shutdown_loader(val_loader)

    def lock(self, output, spec):
        P = self.P
        with P.file_lock(output / ".selection.lock"):
            rows = []
            for index, task in enumerate(spec["tasks"]):
                selected = selected_training(P.task_directory(output, index), task, spec["content_hash"],
                    spec["recipes"][task["recipe"]], spec["donor"]["runtime"])
                rows.append(dict(task=task, checkpoint=selected["checkpoint"], sha256=selected["checkpoint_sha256"]))
            record = dict(campaign_sha256=spec["content_hash"], checkpoints=rows)
            path = output / "selection/locked_checkpoints.json"
            if path.exists() and P.read_json(path) != record:
                raise ValueError("Selected checkpoints changed after lock")
            if not path.exists():
                P.write_json(path, record)
            return record

    def evaluate(self, args):
        P = self.P
        spec = P.load_campaign(args.output)
        task = spec["tasks"][args.task_index]
        device, runtime = self.gpu_setup(spec)
        P.require_approval(args.output, spec, runtime)
        selected = self.lock(args.output, spec)["checkpoints"][args.task_index]
        run = P.task_directory(args.output, args.task_index)
        directory = run / "full_validation"
        with P.file_lock(directory / ".evaluation.lock", blocking=False):
            model = self.make_model(task, device).eval()
            checkpoint = trusted_load(selected["checkpoint"])
            if checkpoint["task"] != task or checkpoint["runtime"] != runtime or checkpoint["campaign_sha256"] != spec["content_hash"]:
                raise ValueError("Selected checkpoint identity differs")
            model.load_state_dict(checkpoint["model"], strict=True)
            for module in model.modules():
                if hasattr(module, "_counter"):
                    module._counter = 5
            parts, labels, records = [], [], []
            for index, file in enumerate(sorted(spec["inventory"]["splits"]["val"]["files"], key=lambda r: r["name"])):
                self.check_stop()
                path = directory / f"file_{index:03d}.npz"
                sidecar = path.with_suffix(".json")
                contract = dict(campaign_sha256=spec["content_hash"], checkpoint_sha256=selected["sha256"],
                                file=file, entry_start=0, entry_stop=file["num_entries"])
                if path.exists() and sidecar.exists():
                    saved = P.read_json(sidecar)
                    if saved.get("contract") != contract or saved.get("sha256") != P.file_sha256(path):
                        raise ValueError("Committed validation predictions changed")
                else:
                    loader = finite_validation_loader(spec["inventory"], file)
                    logits_list, truth_list = [], []
                    with torch.no_grad(), model_precision(model, device, training=False):
                        for batch in loader:
                            self.check_stop()
                            values, truth = tensors(batch, device)
                            logits = model(*values).float().cpu().numpy()
                            if not np.isfinite(logits).all():
                                raise FloatingPointError("Nonfinite full-validation logits")
                            logits_list.append(logits)
                            truth_list.append(truth.cpu().numpy())
                    logits, truth = np.concatenate(logits_list), np.concatenate(truth_list)
                    if len(truth) != file["num_entries"]:
                        raise ValueError("Incomplete validation file")
                    atomic_npz(path, logits=logits, labels=truth)
                    P.write_json(sidecar, dict(contract=contract, sha256=P.file_sha256(path)))
                with np.load(path, allow_pickle=False) as arrays:
                    parts.append(arrays["logits"])
                    labels.append(arrays["labels"])
                records.append(P.read_json(sidecar))
                progress("full_validation_file", **task, completed=index + 1)
            logits, truth = np.concatenate(parts), np.concatenate(labels)
            total = spec["recipes"][task["recipe"]]["evaluation_population"]
            if len(truth) != total or not np.array_equal(np.bincount(truth, minlength=10), np.full(10, total // 10)):
                raise ValueError("Full-validation population/class balance differs")
            metrics = classification_metrics(logits, truth, rejection=True)
            P.write_json(directory / "metrics.json", dict(task=task, campaign_sha256=spec["content_hash"],
                checkpoint_sha256=selected["sha256"], identity_manifest=records, metrics=metrics))
            progress("full_validation_complete", **task, accuracy=metrics["accuracy"], macro_auc_ovo=metrics["macro_auc_ovo"])
