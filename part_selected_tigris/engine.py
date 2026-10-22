"""GH200 admission; the common training and evaluation loops stay unchanged."""
import functools
import os
import platform
import signal
import time
import torch
from part_selected_screen import data
from part_selected_screen.data import shutdown_loader, batch_fingerprint
from scripts.run_part_ev_screen import training_step, validate, progress
from part_ev_fullscale.optimizer import build_optimizer
from part_ev_fullscale.checkpoint import seed_everything
from part_screen_common.controls import selected_training
from . import protocol as P


def gpu_setup(self, spec):
    if platform.machine() != 'aarch64' or not torch.cuda.is_available():
        raise RuntimeError('Run GPU phases on Tigris ARM CUDA')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1 or torch.cuda.device_count() != 1:
        raise RuntimeError('Request exactly one visible GPU')
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    for name in ('SIGTERM', 'SIGUSR1'):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), self.signal)
    runtime = dict(gpu=torch.cuda.get_device_name(0),
        gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
        world_size=1, batch_size=512, workers=2, batchnorm='native', tf32=False,
        cudnn_benchmark=False, precision=P.S.POLICY, campaign_sha256=spec['content_hash'])
    P.validate_runtime(runtime)
    return torch.device('cuda:0'), runtime


def benchmark(self, args):
    P = self.P
    build_loader = functools.partial(data.build_loader, config=args.output/"inputs/selected.yaml")
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
            inputs_match = fingerprints == spec["input_reference"]["fingerprints"]
            passed = (inputs_match and skipped / measured < limits["max_skip_fraction"] and full_skips == 0
                and training_precision["passed"] and validation["precision"]["passed"]
                and max(native_memory, full_memory, validation_memory) < runtime["gpu_memory_bytes"] * limits["memory_fraction_limit"])
            result = dict(passed=passed, task=task, campaign_sha256=spec["content_hash"], runtime=runtime,
                initial_parity=parity, input_fingerprints=fingerprints, inputs_match_native_tigris=inputs_match,
                inputs_match_sporc_donor=fingerprints == spec["donor"]["input_fingerprints"],
                training_precision=training_precision, validation=validation, diagnostics=diagnostics,
                measured_steps=measured, seconds=seconds, jets_per_second=512 * measured / seconds,
                cold_fetch_seconds=cold, windows=windows, amp_skipped_steps=skipped, full_length_amp_skips=full_skips,
                native_peak_bytes=native_memory, full_length_peak_bytes=full_memory,
                validation_peak_bytes=validation_memory, checkpoint_seconds=checkpoint_seconds,
                projected_training_hours=projected)
            P.write_json(path, result)
            progress("benchmark_complete", **result)
            if not passed:
                raise RuntimeError("GH200 selected benchmark failed; no production admission")
        finally:
            shutdown_loader(loader)
            if val_loader is not None:
                shutdown_loader(val_loader)


def lock(self, output, spec):
    P = self.P
    expected_runtime = P.training_runtime(output, spec)
    with P.file_lock(output / ".selection.lock"):
        rows = []
        for index, task in enumerate(spec["tasks"]):
            selected = selected_training(P.task_directory(output, index), task, spec["content_hash"],
                spec["recipes"][task["recipe"]], expected_runtime)
            rows.append(dict(task=task, checkpoint=selected["checkpoint"], sha256=selected["checkpoint_sha256"]))
        record = dict(campaign_sha256=spec["content_hash"], checkpoints=rows)
        path = output / "selection/locked_checkpoints.json"
        if path.exists() and P.read_json(path) != record:
            raise ValueError("Selected checkpoints changed after lock")
        if not path.exists():
            P.write_json(path, record)
        return record


def engine_for(output):
    from scripts.run_part_selected_screen import engine_for as selected_engine
    original = selected_engine(output)
    # The original factory already isolates loader bindings from all donors.
    # Override only hardware admission, benchmark input reference and lock runtime.
    class TigrisEngine(type(original)):
        pass
    TigrisEngine.gpu_setup = gpu_setup
    TigrisEngine.benchmark = benchmark
    TigrisEngine.lock = lock
    return TigrisEngine(P, original.model_factory, original.parity, original.diagnostics)
