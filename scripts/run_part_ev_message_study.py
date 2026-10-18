#!/usr/bin/env python3
"""Five-model SPORC study, with independently verified reused controls."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import platform
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from part_ev_message_study import protocol as P, controls as C
from part_ev_message_study.engine import create_engine
from part_ev_message_study.model import build_model
from part_ev_fullscale.data import inspect_dataset
from part_ev_fullscale.checkpoint import seed_everything, capture_rank_state, restore_rank_state
from part_ev_fullscale.model import build_model as reference_model
from scripts.check_part_ev_screen_amp_recovery import finite_state

D = create_engine()


def initial_parity(model, task, batch, device):
    rng, original = capture_rank_state(model), copy.deepcopy(model.state_dict())
    training = model.training
    trimmers = [(m, m.enabled) for m in model.modules() if type(m).__name__ == "SequenceTrimmer"]
    try:
        seed_everything(task["seed"])
        reference = reference_model("part_ev", trim=False, use_amp=False).to(device)
        for name, tensor in reference.state_dict().items():
            if not torch.equal(model.state_dict()[name], tensor):
                raise ValueError(f"Changed initial EV/backbone parameter: {name}")
        for module, _ in trimmers:
            module.enabled = False
        values, _ = D.tensors(batch, device)
        values = [v[:8] for v in values]
        errors = {}
        for mode in (False, True):
            # Isolate BN changes in this diagnostic. All initial buffers as well
            # as weights are restored before the actual benchmark starts.
            model.load_state_dict(original)
            reference.load_state_dict({k: original[k] for k in reference.state_dict()})
            model.train(mode)
            reference.train(mode)
            seed_everything(task["seed"] + 771)
            with torch.no_grad(), D.model_precision(reference, device, training=mode):
                expected = reference(*values)
            after_reference = capture_rank_state(reference)
            seed_everything(task["seed"] + 771)
            with torch.no_grad(), D.model_precision(model, device, training=mode):
                actual = model(*values)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            after_model = capture_rank_state(model)
            if (not torch.equal(after_model["torch"], after_reference["torch"])
                    or any(not torch.equal(a,b) for a,b in zip(after_model["cuda"], after_reference["cuda"]))):
                raise ValueError("New mechanism consumes extra dropout/training RNG")
            errors["train" if mode else "eval"] = float((actual-expected).abs().max())
        return dict(passed=True, maximum_absolute_errors=errors, events=len(actual))
    finally:
        model.load_state_dict(original)
        for module, enabled in trimmers:
            module.enabled = enabled
        restore_rank_state(model, rng)
        model.train(training)


def gpu_setup(spec):
    device, runtime = D.gpu_setup(spec)
    if C.runtime_identity(runtime) != spec["donor"]["runtime"]:
        raise ValueError("Allocated GPU/memory/runtime differs from reused controls")
    return device, runtime


def prepare(args):
    if platform.machine() != "x86_64" or torch.version.cuda is None:
        raise RuntimeError("Prepare on SPORC using the donor's x86 CUDA environment")
    output = args.output
    with P.file_lock(output / ".prepare.lock", blocking=False):
        if (output / "campaign.json").exists():
            P.load_campaign(output)
            D.progress("already_prepared")
            return
        launch = P.read_json(output / "launch.json")
        donor, contract = C.donor_contract(launch["donor_campaign"])
        inventory = inspect_dataset(launch["data_root"], check_entries=True, splits=("train", "val"))
        if inventory != donor["inventory"]:
            raise ValueError("Data inventory differs from donor; do not reuse controls across a changed population")
        if shutil.disk_usage(output).free < 12 * 1024**3:
            raise OSError("Keep 12 GiB free for five-model checkpoints, validation logits and headroom")
        D.progress("verifying_staged_ROOT_bytes", files=1050)
        contents = C.verify_data_contents(inventory)
        registry, _ = C.registry(launch["donor_campaign"], donor, contract)
        for task in P.TASKS:
            seed_everything(task["seed"])
            model = build_model(task["architecture"], seed=task["seed"])
            D.progress("model_contract", **task, parameters=sum(p.numel() for p in model.parameters()))
        spec = P.create_spec(inventory, launch, contract, contents)
        P.write_json(output / "data_contents.json", contents)
        P.write_json(output / "controls.json", registry)
        P.write_json(output / "campaign.json", spec)
        D.progress("prepared", campaign_sha256=spec["content_hash"],
                   controls={k:v["status"] for k,v in registry["controls"].items()})


def benchmark(args):
    spec = P.load_campaign(args.output)
    P.verify_inventory_metadata(spec)
    task = spec["tasks"][args.task_index]
    recipe, limits = spec["recipes"][task["recipe"]], spec["benchmark"]
    device, runtime = gpu_setup(spec)
    directory = args.output / "benchmarks"
    path = directory / (task["run_id"] + ".json")
    with P.file_lock(directory / (task["run_id"] + ".lock"), blocking=False):
        if path.exists():
            row = C.checked_hash(P.read_json(path))
            if row["passed"] and row["runtime"] == runtime and row["campaign_sha256"] == spec["content_hash"]:
                D.progress("benchmark_already_complete", **task)
                return
            raise ValueError("Failed benchmark retained; inspect evidence and prepare a new campaign")
        loader, _ = D.build_loader("train", spec["inventory"], 512, seed=task["seed"])
        val_loader = None
        try:
            model = D.make_model(task, device)
            optimizer = D.build_optimizer(model, recipe)
            scaler = torch.amp.GradScaler("cuda")
            iterator = iter(loader)
            seed_everything(task["seed"])
            cold_start = time.monotonic()
            first = next(iterator)
            cold = time.monotonic() - cold_start
            parity = initial_parity(model, task, first, device)
            fingerprints = {"0": D.batch_fingerprint(first)}
            windows, skipped = [], 0
            warmup, measured = limits["warmup_steps"], limits["measured_steps"]
            torch.cuda.reset_peak_memory_stats()
            D.progress("benchmark_training_start", **task)
            for index in range(warmup + measured):
                D.check_stop()
                batch = first if index == 0 else next(iterator)
                if index in (warmup, 1020, 2020, 4020):
                    fingerprints[str(index)] = D.batch_fingerprint(batch)
                row = D.training_step(model, optimizer, scaler, batch, device, recipe, index,
                                      check_precision=index == 0)
                if index == 0:
                    precision = row["precision"]
                if index >= warmup:
                    skipped += int(row["amp_skipped"])
                if index+1 == warmup:
                    torch.cuda.synchronize()
                    window_start = measured_start = time.monotonic()
                if index >= warmup and ((index+1-warmup) % limits["window_steps"] == 0 or index+1 == warmup+measured):
                    torch.cuda.synchronize()
                    stop = time.monotonic()
                    count = (index+1-warmup) % limits["window_steps"] or limits["window_steps"]
                    windows.append(dict(steps=count, seconds=stop-window_start))
                    window_start = stop
                    D.progress("benchmark_window", **task, measured_steps=index+1-warmup, **row, **windows[-1])
            seconds = time.monotonic() - measured_start
            native_memory = torch.cuda.max_memory_allocated()
            trimmers = [(m,m.enabled) for m in model.modules() if type(m).__name__ == "SequenceTrimmer"]
            for m, _ in trimmers:
                m.enabled = False
            torch.cuda.reset_peak_memory_stats()
            attempts = []
            for index in range(limits["calibration_attempts"] + limits["confirmation_attempts"]):
                D.check_stop()
                before = scaler.get_scale()
                # Synthetic full-128 batches only for admission; no probe state
                # ever enters production. Peak LR for every calibration update.
                row = D.training_step(model, optimizer, scaler, first, device, recipe, 0, full_length=True)
                row.update(attempt=index+1, scale_before=before, scale_after=scaler.get_scale(),
                           state_finite=finite_state(torch, (model.state_dict(), optimizer.state_dict(), scaler.state_dict())))
                attempts.append(row)
                D.progress("full_length_confirmation" if index >= limits["calibration_attempts"] else
                           "full_length_calibration", **task, **row)
            full_memory = torch.cuda.max_memory_allocated()
            for m, enabled in trimmers:
                m.enabled = enabled
            checkpoint_start = time.monotonic()
            D.save_checkpoint(directory / (task["run_id"] + "_probe.pt"), model, optimizer, scaler,
                dict(task=task, step=warmup+measured, full_length_probe_attempts=len(attempts)), spec, runtime)
            checkpoint_seconds = time.monotonic() - checkpoint_start
            val_loader, _ = D.build_loader("val", spec["inventory"], 512, seed=task["seed"])
            torch.cuda.reset_peak_memory_stats()
            val = D.validate(model, iter(val_loader), device, limits["validation_events"], check_precision=True)
            val_memory = torch.cuda.max_memory_allocated()
            per_step = max(w["seconds"]/w["steps"] for w in windows)
            projected = (per_step*recipe["total_steps"] + val["seconds"]*10*
                         recipe["validation_events_per_epoch"]/val["events"] + 220*checkpoint_seconds+cold)/3600
            confirmed = attempts[limits["calibration_attempts"]-1:]
            passed = (skipped < measured*limits["max_skip_fraction"]
                      and all(a["state_finite"] for a in attempts)
                      and all(not a["amp_skipped"] and a["gradient_norm"] is not None for a in confirmed)
                      and max(native_memory,full_memory,val_memory) < runtime["gpu_memory_bytes"]*limits["memory_fraction_limit"]
                      and precision["passed"] and val["precision"]["passed"])
            result = dict(passed=passed, task=task, campaign_sha256=spec["content_hash"], runtime=runtime,
                initial_parity=parity, input_fingerprints=fingerprints, training_precision=precision,
                cold_fetch_seconds=cold, measured_steps=measured, seconds=seconds,
                jets_per_second=512*measured/seconds, amp_skipped_steps=skipped,
                full_length_attempts=attempts, native_peak_bytes=native_memory,
                full_length_peak_bytes=full_memory, validation_peak_bytes=val_memory,
                checkpoint_seconds=checkpoint_seconds, windows=windows, validation=val,
                projected_training_hours=projected)
            result["content_hash"] = P.digest(result)
            P.write_json(path, result)
            D.progress("benchmark_complete", **task, passed=passed, projected_training_hours=projected)
            if not passed:
                raise RuntimeError("New architecture admission failed; training is blocked")
        finally:
            D.shutdown_loader(loader)
            if val_loader is not None:
                D.shutdown_loader(val_loader)


def refresh_controls(output, spec, *, require_complete=False, recompute=False):
    donor, contract = C.donor_contract(spec["donor"]["path"])
    if contract != spec["donor"] or donor["inventory"] != spec["inventory"]:
        raise ValueError("Donor control contract changed")
    result, labels = C.registry(contract["path"], donor, contract,
        require_complete=require_complete, recompute=D.classification_metrics if recompute else None)
    P.write_json(output / "controls.json", result)
    return result, labels


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("prepare","benchmark","review","train","evaluate","report","controls"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--approve", action="store_true")
    parser.add_argument("--reviewed", action="store_true")
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    if args.phase in ("benchmark", "train", "evaluate"):
        P.task_directory(args.output, args.task_index)
    if args.phase == "prepare":
        prepare(args)
    elif args.phase == "benchmark":
        benchmark(args)
    else:
        spec = P.load_campaign(args.output)
        if args.phase == "review":
            result = P.approve(args.output, spec, reviewed=args.reviewed) if args.approve else P.benchmark_summary(args.output, spec)
            print(json.dumps(result, indent=2))
        elif args.phase == "controls":
            result, _ = refresh_controls(args.output, spec)
            print(json.dumps({k:v["status"] for k,v in result["controls"].items()}, indent=2))
        elif args.phase == "report":
            from part_ev_message_study.reporting import report
            report(args.output, spec, D, refresh_controls)
        else:
            P.verify_inventory_metadata(spec)
            P.require_approval(args.output, spec)
            gpu_setup(spec)  # Compare to donor before entering unchanged engine.
            getattr(D, args.phase)(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
