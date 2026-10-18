#!/usr/bin/env python3
"""Independent grouped screen; production is gated on its own reviewed probes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from part_group_screen import protocol as P
from part_group_screen.model import build_model
from part_group_screen.diagnostics import initial_parity, collect
from part_group_screen.config import TASKS
from part_screen_common.controls import refresh_controls
from part_screen_common.engine import ScreenEngine, progress
from part_ev_fullscale.checkpoint import seed_everything

ENGINE = ScreenEngine(P, build_model, initial_parity, collect)


def prepare(args):
    output = args.output.resolve()
    launch = P.read_json(output / "launch.json")
    donor = Path(launch["donor"]).resolve()
    # Check disjoint paths before even creating a lock in the destination.
    if output == donor or output.is_relative_to(donor) or donor.is_relative_to(output):
        raise ValueError("Grouped campaign must be separate from donor")
    with P.file_lock(output / ".prepare.lock", blocking=False):
        path = output / "experiment_spec.json"
        if path.exists():
            spec = P.load_campaign(output)
            if spec["launch"] != launch:
                raise ValueError("Frozen launch configuration changed")
            refresh_controls(output, spec["donor"])
            progress("already_prepared", campaign_sha256=spec["content_hash"])
            return
        parameters = {}
        for task in TASKS:
            seed_everything(task["seed"])
            model = build_model(task["architecture"], seed=task["seed"])
            parameters[task["architecture"]] = sum(p.numel() for p in model.parameters())
            del model
        spec = P.create_spec(donor, launch["controls"], Path(__file__).resolve().parents[1], launch, parameters)
        # Four 5M logit sets + labels, checkpoints, temporary writes and margin.
        required = 4 * 1024 ** 3
        free = shutil.disk_usage(output).free
        if free < required:
            raise OSError("Need at least 4 GiB free for grouped artifacts/headroom; nothing deleted")
        P.write_json(path, spec)
        registry = refresh_controls(output, spec["donor"])
        progress("prepared", campaign_sha256=spec["content_hash"], parameters=parameters,
                 controls_ready=registry["ready"], free_bytes=free)


def status(output):
    spec = P.load_campaign(output, check_environment=False)
    rows = []
    entries = [(task, P.task_directory(output, i), False) for i, task in enumerate(spec["tasks"])]
    for name, task in spec["donor"]["tasks"].items():
        entries.append((task, Path(spec["donor"]["path"]) / "runs" / name / f"seed_{task['seed']}", True))
    for task, run, reused in entries:
        checks = [P.read_json(p) for p in sorted((run / "validation").glob("check_*.json"))]
        contiguous = 0
        for row in checks:
            if row.get("check") != contiguous + 1:
                break
            contiguous += 1
        rows.append(dict(task=task, reused=reused, contiguous_checks=contiguous,
                         completed=(run / "training_complete.json").is_file(), checks=checks,
                         evaluated=(run / "full_validation/metrics.json").is_file()))
    common = min((r["contiguous_checks"] for r in rows), default=0)
    for row in rows:
        selected = max(row["checks"][:common], key=lambda c:c["accuracy"]) if common else None
        row["best_through_common_check"] = selected
    base = next(r for r in rows if r["task"]["run_id"] == "PART_RANGER")["best_through_common_check"]
    differences = {r["task"]["run_id"]:100 * (r["best_through_common_check"]["accuracy"] - base["accuracy"])
                   for r in rows if not r["reused"]} if base else {}
    print(json.dumps(dict(campaign=spec["content_hash"], common_check=common, models=rows,
        best_through_common_delta_pp=differences,
        note="Read-only progress, including donor controls; not final artifact verification."), indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("prepare", "benchmark", "review", "approve", "train", "lock", "evaluate", "refresh-controls", "report", "status"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--reviewed", action="store_true")
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    if args.phase in ("benchmark", "train", "evaluate"):
        P.task_directory(args.output, args.task_index)
        getattr(ENGINE, args.phase)(args)
    elif args.phase == "prepare":
        prepare(args)
    elif args.phase == "status":
        status(args.output)
    elif args.phase == "report":
        from part_group_screen.reporting import report
        return 0 if report(args.output) else 3
    else:
        spec = P.load_campaign(args.output)
        if args.phase == "review":
            result = P.benchmark_summary(args.output, spec)
        elif args.phase == "approve":
            if not args.reviewed:
                raise ValueError("Explicit --reviewed is required for admission")
            result = P.approve(args.output, spec)
        elif args.phase == "lock":
            P.require_approval(args.output, spec)
            result = ENGINE.lock(args.output, spec)
        else:
            result = refresh_controls(args.output, spec["donor"])
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
