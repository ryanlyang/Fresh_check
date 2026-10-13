#!/usr/bin/env python3
"""Bounded, isolated timing probe of an existing frozen full-scale campaign.

This is not a training phase: no production checkpoint, benchmark, runtime lock,
submission ledger, or scheduler setting is changed. All writes stay in a new
diagnostics directory. Models and data come from the campaign's frozen source.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path
import time


def load_driver(campaign):
    source = campaign / "source"
    path = source / "scripts/run_part_fullscale.py"
    spec = importlib.util.spec_from_file_location("_frozen_part_timing_runtime", path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    if Path(driver.P.__file__).resolve() != (source / "part_ev_fullscale/protocol.py").resolve():
        raise RuntimeError("Timing probe imported a different campaign's code")
    return driver


def project(model, recipe):
    """Project measured components, without counting checkpoint writes twice."""
    train = model.get("training_windows", [])
    val = model.get("validation", {})
    full = model.get("full_checkpoint_seconds", [])
    best = model.get("best_checkpoint_seconds", [])
    steps = sum(row["steps"] for row in train)
    # Require several complete fetch populations, not just the initial prefetch.
    min_steps = math.ceil(4 * recipe["training_population"] * recipe["fetch_step"]
                          / recipe["global_batch_size"])
    reasons = []
    if steps < min_steps:
        reasons.append(f"Need at least {min_steps} training steps across repeated data fetches")
    if len(train) < 2:
        reasons.append("Need multiple independent timing windows")
    if val.get("events", 0) != recipe["validation_events_per_epoch"]:
        reasons.append("A full production-sized validation pass was not measured")
    if not full or not best:
        reasons.append("Checkpoint timing is incomplete")
    if model.get("error"):
        reasons.append("Probe encountered an error")
    if reasons:
        return {"usable": False, "reasons": reasons}
    train_seconds = sum(row["seconds"] - row["checkpoint_seconds"] for row in train)
    full_count = recipe["total_steps"] // 1000 + 2 * recipe["epochs"]
    checkpoint_seconds = (full_count * sum(full) / len(full)
                          + recipe["epochs"] * sum(best) / len(best))
    other = (model["setup_seconds"] + recipe["epochs"] * val["seconds"] + checkpoint_seconds)
    hours = (recipe["total_steps"] * train_seconds / steps + other) / 3600
    # Excluding the first 1000-step window is a sensitivity, not a confidence bound.
    late = train[1:]
    late_steps = sum(row["steps"] for row in late)
    late_rate = sum(row["seconds"] - row["checkpoint_seconds"] for row in late) / late_steps
    late_hours = (recipe["total_steps"] * late_rate + other) / 3600
    planning_hours = max(hours, late_hours)
    sensitivity = planning_hours * 1.25
    if planning_hours > 72:
        assessment = "72h_at_risk"
    elif sensitivity > 72:
        assessment = "72h_limited_headroom"
    else:
        assessment = "72h_plausible_not_guaranteed"
    return {
        "usable": True, "observed_projection_hours": hours,
        "excluding_first_training_window_hours": late_hours,
        "planning_projection_hours": planning_hours,
        "with_25_percent_slowdown_hours": sensitivity,
        "hours_remaining_under_72h": 72 - planning_hours, "assessment": assessment,
        "projected_full_checkpoint_count": full_count,
        "projected_best_checkpoint_count_upper_bound": recipe["epochs"],
        "projected_checkpoint_hours": checkpoint_seconds / 3600,
        "projected_validation_hours": recipe["epochs"] * val["seconds"] / 3600,
        "training_jets_per_second_including_periodic_checkpoints":
            steps * recipe["global_batch_size"] / sum(row["seconds"] for row in train),
        "assumptions": [
            "Whole observed training window includes cold start and repeated ROOT loads",
            "One complete validation pass is scaled to all reporting epochs",
            "All epochs are conservatively allowed a best-checkpoint write",
            "25 percent slowdown is a sensitivity scenario, not a statistical interval",
            "Six simultaneous jobs' storage contention is not measured",
            "No queue waiting, maintenance, or restart data replay is included",
        ],
    }


def write_report(driver, output, report):
    for model in report["models"].values():
        model["projection"] = project(model, report["recipe"])
    report["complete"] = (len(report["models"]) == 2
                          and all(row.get("complete", False) for row in report["models"].values()))
    report["usable"] = (len(report["models"]) == 2
                        and all(row["projection"]["usable"] for row in report["models"].values()))
    driver.P.write_json(output / "report.json", report)
    lines = ["# Full-scale runtime diagnostic", "",
             "Production jobs, checkpoints, and benchmarks were left unchanged.", "",
             "| Model | Train steps | Validation events | All windows (h) | Excluding first window (h) | With 25% slowdown | Assessment |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for name, model in report["models"].items():
        estimate = model["projection"]
        values = (f"{estimate['observed_projection_hours']:.2f}",
                  f"{estimate['excluding_first_training_window_hours']:.2f}",
                  f"{estimate['with_25_percent_slowdown_hours']:.2f}", estimate["assessment"]) if estimate["usable"] else (
                      "unavailable", "unavailable", "unavailable", "inconclusive")
        lines.append(f"| {name} | {sum(w['steps'] for w in model['training_windows'])} | "
                     f"{model.get('validation', {}).get('events', 0)} | {' | '.join(values)} |")
    lines.extend(["", "These are projections, not guarantees or confidence intervals. The probe does not "
                  "measure six-job filesystem contention, queue delays, or restart data replay.",
                  "Planning uses the larger of the two projections. The 25% slowdown is illustrative, not a bound.",
                  "A failed probe or one without enough training/validation data is inconclusive. A probe can "
                  "miss its full training target yet still collect enough repeated-fetch windows; see complete "
                  "and usable separately in the JSON report.", ""])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def probe_model(driver, spec, args, architecture, deadline, model_report, persist):
    import torch
    from part_ev_fullscale.checkpoint import atomic_torch_save, seed_everything
    from part_ev_fullscale.data import build_loader, shutdown_loader
    from part_ev_fullscale.optimizer import Ranger

    started = time.monotonic()
    loaders = []
    checkpoint = args.output / f"{architecture}_timing_checkpoint.pt"
    best_checkpoint = args.output / f"{architecture}_timing_best.pt"
    # Only these freshly created diagnostic files may be replaced and removed.
    if checkpoint.exists() or best_checkpoint.exists():
        raise FileExistsError("Diagnostic checkpoint paths already exist")
    base = model = optimizer = scaler = None
    try:
        base, model = driver.make_model(architecture, 101, args.device, 1)
        optimizer = Ranger(base.parameters())
        scaler = torch.amp.GradScaler("cuda")
        train_loader, _ = build_loader("train", spec["inventory"], 512, seed=101)
        loaders.append(train_loader)
        val_loader, _ = build_loader("val", spec["inventory"], 512, seed=101)
        loaders.append(val_loader)
        # Diagnostic-only failure bound: do not wait indefinitely for a dead worker.
        # No batch contents, worker counts, sampling policy, or model math changes.
        for loader in loaders:
            loader.timeout = 300
        train_iter, val_iter = iter(train_loader), iter(val_loader)
        seed_everything(101)
        state = {"task": {"architecture": architecture, "seed": 101}, "step": 0,
                 "validated_epochs": 0, "best_accuracy": -1.0, "best_step": None,
                 "amp_skipped_steps": 0}
        model_report["setup_seconds"] = time.monotonic() - started
        model_report["parameters"] = sum(p.numel() for p in base.parameters())
        model_report["loader_timeout_seconds"] = 300
        torch.cuda.reset_peak_memory_stats()

        def stopping(limit):
            return driver.STOP_REQUESTED or time.monotonic() >= limit

        def save_full():
            torch.cuda.synchronize()
            tick = time.monotonic()
            driver.save_checkpoint(checkpoint, base, optimizer, scaler, state, spec,
                                   args.runtime, 0, 1)
            elapsed = time.monotonic() - tick
            model_report["full_checkpoint_seconds"].append(elapsed)
            return elapsed

        # Reserve the last quarter of each architecture's budget for validation.
        train_deadline = started + 0.75 * (deadline - started)
        tick = time.monotonic()
        batch_wait = checkpoint_seconds = 0.0
        window_steps = skipped = 0
        model.train()
        while state["step"] < args.train_steps and not stopping(train_deadline):
            for group in optimizer.param_groups:
                group["lr"] = driver.P.learning_rate(state["step"])
            before_batch = time.monotonic()
            batch = next(train_iter)
            batch_wait += time.monotonic() - before_batch
            if stopping(train_deadline):
                break
            loss, skip = driver.train_step(model, optimizer, scaler, batch, args.device)
            state["step"] += 1
            state["amp_skipped_steps"] += int(skip)
            skipped += int(skip)
            window_steps += 1
            if state["step"] % 1000 == 0:
                checkpoint_seconds += save_full()
            if state["step"] % 1000 == 0 or state["step"] == args.train_steps:
                torch.cuda.synchronize()
                model_report["training_windows"].append({
                    "step_end": state["step"], "steps": window_steps,
                    "seconds": time.monotonic() - tick, "batch_wait_seconds": batch_wait,
                    "checkpoint_seconds": checkpoint_seconds, "amp_skipped_steps": skipped,
                    "last_loss": loss,
                })
                driver.progress("sustained_training", architecture=architecture,
                                **model_report["training_windows"][-1])
                persist()
                tick = time.monotonic()
                batch_wait = checkpoint_seconds = 0.0
                window_steps = skipped = 0
        if window_steps:
            torch.cuda.synchronize()
            model_report["training_windows"].append({
                "step_end": state["step"], "steps": window_steps,
                "seconds": time.monotonic() - tick, "batch_wait_seconds": batch_wait,
                "checkpoint_seconds": checkpoint_seconds, "amp_skipped_steps": skipped,
            })
        model_report["training_target_complete"] = state["step"] == args.train_steps
        if state["step"] and state["amp_skipped_steps"] == state["step"]:
            raise FloatingPointError("All measured optimizer steps were skipped")
        persist()
        if stopping(deadline):
            model_report["stop_reason"] = "signal_or_time_budget"
            return
        save_full()  # Production saves immediately before validation.
        base.eval()
        tick = time.monotonic()
        total = correct = 0
        loss_sum = 0.0
        with torch.no_grad():
            for index in range(args.validation_steps):
                if stopping(deadline):
                    break
                batch = next(val_iter)
                if stopping(deadline):
                    break
                inputs, truth = driver.inputs_and_labels(batch, args.device)
                logits = base(*inputs)
                if not bool(torch.isfinite(logits).all()):
                    raise FloatingPointError("Nonfinite validation logits")
                correct += int((logits.argmax(1) == truth).sum())
                total += len(truth)
                loss_sum += float(torch.nn.functional.cross_entropy(logits.float(), truth, reduction="sum"))
                if (index + 1) % 250 == 0 or index + 1 == args.validation_steps:
                    torch.cuda.synchronize()
                    model_report["validation"] = {"events": total, "steps": index + 1,
                                                  "seconds": time.monotonic() - tick}
                    driver.progress("sustained_validation", architecture=architecture,
                                    **model_report["validation"])
                    persist()
        torch.cuda.synchronize()
        model_report["validation"] = {"events": total, "steps": total // 512,
                                      "seconds": time.monotonic() - tick}
        model_report["validation_target_complete"] = total == args.validation_steps * 512
        if total:
            metric = {"accuracy": correct / total, "loss": loss_sum / total, "events": total}
            tick = time.monotonic()
            atomic_torch_save({"model": base.state_dict(), "task": state["task"],
                               "campaign_sha256": spec["content_hash"], "runtime": args.runtime,
                               "step": state["step"], "validation": metric}, best_checkpoint)
            model_report["best_checkpoint_seconds"].append(time.monotonic() - tick)
            save_full()
        model_report["complete"] = (model_report["training_target_complete"]
                                    and model_report["validation_target_complete"])
        model_report["stop_reason"] = "targets_complete" if model_report["complete"] else "signal_or_time_budget"
    finally:
        model_report["elapsed_seconds"] = time.monotonic() - started
        model_report["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024**3
        for loader in loaders:
            shutdown_loader(loader)
        # No production paths can reach these deletion targets.
        checkpoint.unlink(missing_ok=True)
        best_checkpoint.unlink(missing_ok=True)
        del model, base, optimizer, scaler
        torch.cuda.empty_cache()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-steps", type=int, default=10000)
    parser.add_argument("--validation-steps", type=int, default=2500)
    parser.add_argument("--budget-minutes", type=int, default=100)
    args = parser.parse_args(argv)
    if not (1 <= args.train_steps <= 20000 and 1 <= args.validation_steps <= 2500
            and 1 <= args.budget_minutes <= 100):
        parser.error("Targets must be train 1..20000, validation 1..2500, budget 1..100 minutes")
    args.campaign = args.campaign.resolve(strict=True)
    args.output = args.output.resolve(strict=True)
    diagnostics = (args.campaign / "diagnostics").resolve(strict=True)
    if diagnostics != args.campaign / "diagnostics":
        parser.error("Campaign diagnostics directory must not redirect through a symbolic link")
    if args.output.parent != diagnostics or args.output == args.campaign:
        parser.error("Output must be a new direct child of the campaign diagnostics directory")
    # A fresh unique directory is required for every submission; never overwrite a report.
    with (args.output / "started.marker").open("x") as handle:
        handle.write("This directory has been used; submit a fresh diagnostic for another attempt.\n")
    driver = load_driver(args.campaign)
    spec = driver.P.load_campaign(args.campaign)
    rank, world, args.device = driver.distributed_setup()
    if rank != 0 or world != 1:
        raise RuntimeError("Timing diagnostic requires one GPU and one rank")
    args.runtime = driver.runtime_contract(spec, rank, world, args.device)
    driver.require_benchmark(args.campaign, args.runtime)
    report = {"schema": "part_fullscale_sustained_timing_v1", "campaign": str(args.campaign),
              "campaign_sha256": spec["content_hash"], "runtime": args.runtime,
              "recipe": spec["recipe"], "models": {}, "finished": False,
              "changes_production": False, "authorizes_training": False,
              "targets": {"train_steps": args.train_steps, "validation_steps": args.validation_steps,
                          "budget_minutes": args.budget_minutes},
              "diagnostic_sha256": driver.P.file_sha256(Path(__file__))}
    persist = lambda: write_report(driver, args.output, report)
    persist()
    start = time.monotonic()
    deadline = start + args.budget_minutes * 60
    for index, architecture in enumerate(("part", "part_ev")):
        if driver.STOP_REQUESTED or time.monotonic() >= deadline:
            break
        row = {"training_windows": [], "full_checkpoint_seconds": [],
               "best_checkpoint_seconds": [], "complete": False}
        report["models"][architecture] = row
        model_deadline = time.monotonic() + (deadline - time.monotonic()) / (2 - index)
        try:
            probe_model(driver, spec, args, architecture, model_deadline, row, persist)
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            import traceback
            traceback.print_exc()
        persist()
    report["finished"] = True
    report["elapsed_seconds"] = time.monotonic() - start
    persist()
    driver.progress("sustained_timing_complete", report=str(args.output / "report.json"),
                    usable=report["usable"], complete=report["complete"], changes_production=False)
    return 1 if any(row.get("error") for row in report["models"].values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
