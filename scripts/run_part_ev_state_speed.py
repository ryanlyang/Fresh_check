#!/usr/bin/env python3
"""Tier3 state-EV speed study; no writes to reused campaigns."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from part_ev_state_speed import protocol as P, benchmark as B
from part_ev_state_speed.engine import create_engine, benchmark_driver
from part_ev_message_study import controls as C

D = create_engine()


def benchmark(args, driver, spec):
    # Reuse the campaign checked before optimizer construction. Lazy optimizer
    # imports can expose setuptools._vendor distributions on sys.path without
    # changing the installed environment. A fresh process still checks it fully.
    driver.benchmark(args)  # Same sustained training/memory/AMP gate as the donor.
    task = spec["tasks"][args.task_index]
    path = args.output / "benchmarks" / (task["run_id"] + "_speed.json")
    with P.file_lock(path.with_suffix(".lock"), blocking=False):
        if path.exists():
            B.validate_speed_report(C.checked_hash(P.read_json(path)), spec, task)
            return
        device, runtime = driver.gpu_setup(spec)
        D.seed_everything(task["seed"])
        loader, _ = D.build_loader("train", spec["inventory"], 512, seed=task["seed"])
        try:
            batch = next(iter(loader))
            row = P.read_json(args.output / "benchmarks" / (task["run_id"] + ".json"))
            if D.batch_fingerprint(batch) != row["input_fingerprints"]["0"]:
                raise ValueError("Timing inputs differ from the common benchmark input")
            result = B.speed_probe(D, spec, task, batch, device, runtime)
            P.write_json(path, result)
            D.progress("matched_inference_timing_complete", **task, models=result["models"])
        finally:
            D.shutdown_loader(loader)


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

    def parity(candidate, task, batch, device):
        return B.initial_parity(D, spec, candidate, task, batch, device)

    driver = benchmark_driver(D, parity)
    if args.phase == "prepare":
        driver.prepare(args)
    else:
        # Validate before any optimizer/lazy imports, then use the same verified
        # spec for both parity and inference timing. Do not weaken the shared
        # environment fingerprint or mutate the reused donor protocols.
        spec = P.load_campaign(args.output)
        if args.phase == "benchmark":
            benchmark(args, driver, spec)
        elif args.phase == "review":
            summary = (P.approve(args.output, spec, reviewed=args.reviewed) if args.approve else
                       P.benchmark_summary(args.output, spec))
            rows = []
            for task, bench in zip(P.TASKS, summary["models"]):
                speed = P.read_json(args.output / "benchmarks" / (task["run_id"] + "_speed.json"))["models"]
                own = speed[task["run_id"]]
                rows.append(dict(model=task["run_id"], train_jets_per_second=bench["jets_per_second"],
                    projected_training_hours=bench["projected_training_hours"],
                    peak_gib=max(bench[k] for k in ("native_peak_bytes", "full_length_peak_bytes", "validation_peak_bytes"))/1024**3,
                    inference_fp32_jets_per_second=own["512"]["jets_per_second"],
                    inference_time_vs_EV=own["512"]["seconds_per_batch"]/speed["EV"]["512"]["seconds_per_batch"],
                    inference_time_vs_original_state=own["512"]["seconds_per_batch"]/speed["STATE_original"]["512"]["seconds_per_batch"],
                    latency_batch1_ms=own["1"]["milliseconds_per_batch"],
                    latency_time_vs_EV=own["1"]["seconds_per_batch"]/speed["EV"]["1"]["seconds_per_batch"]))
            print(json.dumps(dict(passed=summary["passed"], models=rows,
                recommended_train_hours=summary["recommended_train_hours"],
                recommended_evaluate_hours=summary["recommended_evaluate_hours"],
                note="Time ratios below 1 are faster. Accuracy retention is not yet established."), indent=2))
        elif args.phase == "report":
            from part_ev_state_speed.reporting import report
            report(args.output, spec, D)
        else:
            P.verify_inventory_metadata(spec)
            P.require_approval(args.output, spec)
            driver.gpu_setup(spec)
            getattr(D, args.phase)(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
