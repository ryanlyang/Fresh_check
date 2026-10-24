#!/usr/bin/env python3
"""Fresh seven-run full schedule on SPORC; no historical campaign dependencies."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import platform
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from part_ev_state_fullscale import protocol as P
from part_ev_state_fullscale.engine import create_engine
from part_ev_state_fullscale.model import build_model
from part_ev_fullscale.data import inspect_dataset
from part_ev_fullscale.checkpoint import seed_everything
from part_ev_fullscale.model import build_model as reference_model
from part_ev_message_study import controls as C
from part_ev_message_study.parity import initial_parity as isolated_parity, ParityError
from part_ev_state_speed.benchmark import gradient_equivalence
from scripts.check_part_ev_screen_amp_recovery import finite_state

D = create_engine()
gpu_setup = D.gpu_setup


def initial_parity(model, task, batch, device):
    def reference_factory(_architecture, **kwargs):
        return reference_model("part" if task["architecture"] == "part" else "part_ev", **kwargs)
    report = isolated_parity(D, model, task, batch, device, reference_factory)
    if task["architecture"] == "state_late4":
        rng = D.capture_rank_state(model)
        try:
            seed_everything(task["seed"] + 991)
            report["state_gradients"] = gradient_equivalence("state_late4", device)
        except Exception as error:
            report.update(passed=False, error=str(error))
            raise ParityError(str(error), report) from error
        finally:
            D.restore_rank_state(model, rng)
    return report


def prepare(args):
    if platform.machine() != "x86_64" or torch.version.cuda is None:
        raise RuntimeError("Prepare on SPORC in the existing CUDA-enabled environment")
    output = args.output
    with P.file_lock(output / ".prepare.lock", blocking=False):
        if (output / "campaign.json").exists():
            P.load_campaign(output)
            D.progress("already_prepared")
            return
        launch = P.read_json(output / "launch.json")
        inventory = inspect_dataset(launch["data_root"], check_entries=True, splits=("train", "val"))
        if shutil.disk_usage(output).free < 16 * 1024**3:
            raise OSError("Keep 16 GiB free for seven runs, validation logits and headroom")
        D.progress("verifying_staged_ROOT_bytes", files=1050)
        contents = C.verify_data_contents(inventory)
        contents["provenance"] = "staging_receipts_and_actual_ROOT_bytes_verified_now"
        # No optimizer construction before capturing the startup environment.
        for task in P.TASKS:
            seed_everything(task["seed"])
            model = build_model(task["architecture"], seed=task["seed"])
            D.progress("model_contract", **task, parameters=sum(p.numel() for p in model.parameters()))
            del model
        spec = P.create_spec(inventory, launch, contents)
        P.write_json(output / "data_contents.json", contents)
        P.write_json(output / "campaign.json", spec)
        D.progress("prepared", campaign_sha256=spec["content_hash"], runs=7, validation_checks=50)


def benchmark(args):
    if args.task_index not in P.BENCHMARK_INDICES:
        raise ValueError("Benchmark only seed-101 tasks 0, 3, 6")
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
            try:
                parity = initial_parity(model, task, first, device)
            except ParityError as error:
                # Separate evidence: never turn a failed probe into an admitted benchmark.
                D.progress("initial_parity_failed", **task, evidence=error.report)
                raise
            D.progress("initial_parity_complete", **task, evidence=parity)
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
            projected = P.projected_hours(recipe, windows, val, checkpoint_seconds, cold)
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



def review_view(summary):
    """Keep console review short; complete admission evidence stays in benchmark JSON."""
    return dict(passed=summary["passed"],
        models=[dict(model=row["task"]["run_id"], seed=row["task"]["seed"],
            train_jets_per_second=row["jets_per_second"],
            projected_training_hours=row["projected_training_hours"],
            peak_gib=max(row[k] for k in ("native_peak_bytes", "full_length_peak_bytes", "validation_peak_bytes"))/1024**3,
            amp_skipped_steps=row["amp_skipped_steps"]) for row in summary["models"]],
        recommended_train_hours=summary["recommended_train_hours"],
        train_hours_by_architecture=summary["train_hours_by_architecture"],
        recommended_evaluate_hours=summary["recommended_evaluate_hours"],
        note="Fresh 1M-step training; 50 validation checks. Projections exclude queue and restart replay.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("prepare", "benchmark", "review", "train", "evaluate", "report"))
    parser.add_argument("--output", type=Path, required=True)
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
        # Check environment before optimizer lazy imports expose vendored distributions.
        spec = P.load_campaign(args.output)
        if args.phase == "review":
            summary = (P.approve(args.output, spec, reviewed=args.reviewed) if args.approve
                       else P.benchmark_summary(args.output, spec))
            print(json.dumps(review_view(summary), indent=2))
        else:
            P.verify_inventory_metadata(spec)
            P.require_approval(args.output, spec)
            if args.phase == "report":
                from part_ev_state_fullscale.reporting import report
                report(args.output, spec, D)
            else:
                getattr(D, args.phase)(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

