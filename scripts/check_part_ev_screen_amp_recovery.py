#!/usr/bin/env python3
"""Read-only probe continuation; never changes or authorizes production training."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

SCHEMA = "part_ev_screen_amp_recovery_v1"
POLICY = dict(calibration_attempts=8, confirmation_attempts=10,
              batch_size=512, particles=128, lr="constant_original_peak",
              initial_state="saved_post_stress_probe_including_optimizer_scaler_rng",
              input="original_first_training_batch_with_padding_filled_by_first_particle",
              authorizes_training=False)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def create_json(path, value):
    # Unique diagnostic directories; never overwrite an earlier result.
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def frozen_driver(campaign):
    source = campaign / "source"
    for name, module in tuple(sys.modules.items()):
        if name.split(".")[0] in ("part_ev_screen", "part_ev_fullscale"):
            location = getattr(module, "__file__", None)
            if not location or not Path(location).resolve().is_relative_to(source):
                raise RuntimeError("Refusing already-imported non-campaign model code")
    sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location("frozen_screen_driver", source / "scripts/run_part_ev_screen.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def eligible_report(row, spec, task):
    limits = spec["benchmark"]
    if (row["task"] != task or row["campaign_sha256"] != spec["content_hash"]
            or row["initial_parity"]["passed"] is not True
            or row["measured_steps"] != limits["measured_steps"]):
        raise ValueError("Wrong campaign/task or incomplete original benchmark")
    for audit, dtype in ((row["training_precision"], "torch.float16"),
                         (row["validation"]["precision"], "torch.float32")):
        observed = audit.get("observed_activations", {})
        if (audit.get("passed") is not True or audit.get("parameter_dtypes") != ["torch.float32"]
                or audit.get("expected_activation_dtype") != dtype or not observed
                or any(value != dtype for value in observed.values())):
            raise ValueError("Original precision audit did not pass")
    skips = row["amp_skipped_steps"]
    full_skips = row["full_length_amp_skips"]
    if (type(skips) is not int or not 0 <= skips < row["measured_steps"] * limits["max_skip_fraction"]
            or type(full_skips) is not int or not 0 <= full_skips <= limits["full_length_steps"]
            or row["passed"] is not (full_skips == 0)):
        raise ValueError("Original benchmark failed for reasons other than full-length AMP skips")
    peaks = [row[k] for k in ("native_peak_bytes", "full_length_peak_bytes", "validation_peak_bytes")]
    if not all(0 < x < row["runtime"]["gpu_memory_bytes"] * limits["memory_fraction_limit"] for x in peaks):
        raise ValueError("Original memory gate did not pass")


def prepare(campaign, output):
    if not output.is_relative_to(campaign / "diagnostics") or output == campaign / "diagnostics":
        raise ValueError("Use a new directory beneath campaign/diagnostics")
    D = frozen_driver(campaign)
    spec = D.P.load_campaign(campaign)
    if read(output / "launch.json") != spec["launch"]:
        raise ValueError("Launcher resources differ from the frozen campaign")
    artifacts, reports = {}, []
    for task in spec["tasks"]:
        prefix = "benchmarks/" + task["run_id"]
        row = read(campaign / (prefix + ".json"))
        eligible_report(row, spec, task)
        reports.append(row)
        for suffix in (".json", "_probe.pt"):
            relative = prefix + suffix
            artifacts[relative] = sha(campaign / relative)
    if any(row["runtime"] != reports[0]["runtime"] or
           row["input_fingerprints"] != reports[0]["input_fingerprints"] for row in reports):
        raise ValueError("Original benchmarks used different runtimes/input streams")
    plan = dict(schema=SCHEMA, campaign=str(campaign), campaign_sha256=spec["content_hash"],
                tasks=spec["tasks"], runtime=reports[0]["runtime"], policy=POLICY,
                artifacts=artifacts, driver_sha256=sha(__file__),
                launch_sha256=sha(output / "launch.env"))
    plan["content_hash"] = digest(plan)
    create_json(output / "recovery.json", plan)
    print(json.dumps(dict(stage="recovery_prepared", output=str(output), authorizes_training=False)), flush=True)


def load_plan(output):
    plan = read(output / "recovery.json")
    payload = {k: v for k, v in plan.items() if k != "content_hash"}
    if (digest(payload) != plan.get("content_hash") or plan["schema"] != SCHEMA
            or plan["policy"] != POLICY or plan["driver_sha256"] != sha(__file__)
            or plan["launch_sha256"] != sha(output / "launch.env")):
        raise ValueError("Recovery diagnostic source/configuration changed")
    return plan


def verify_artifacts(campaign, plan, task):
    for suffix in (".json", "_probe.pt"):
        key = "benchmarks/" + task["run_id"] + suffix
        if sha(campaign / key) != plan["artifacts"][key]:
            raise ValueError("Original benchmark artifact changed: " + key)


def finite_number(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def assess_attempt(row, *, confirmation):
    """Only calibration may skip an update, and its scaler must back off."""
    if not finite_number(row["loss"]) or not row["state_finite"]:
        return "nonfinite loss, parameters, buffers, or optimizer state"
    before, after = row["scale_before"], row["scale_after"]
    if not all(finite_number(x) and x > 0 for x in (before, after)):
        return "invalid GradScaler scale"
    if row["amp_skipped"]:
        if after >= before or row["optimizer_steps"] != 0:
            return "skipped-update/scaler/optimizer disagreement"
        if confirmation:
            return "AMP skipped an update during confirmation"
    elif (after < before or row["optimizer_steps"] != 1
          or not finite_number(row["gradient_norm"])):
        return "update did not have finite gradients and one optimizer step"
    return None


def finite_state(torch, value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_state(torch, x) for x in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite_state(torch, x) for x in value)
    return not isinstance(value, float) or math.isfinite(value)


def run_probe(D, spec, task, runtime, device, campaign, output, report):
    torch = D.torch
    recipe = spec["recipes"][task["recipe"]]
    model = D.make_model(task, device)
    optimizer = D.build_optimizer(model, recipe)
    scaler = torch.amp.GradScaler("cuda")
    path = campaign / "benchmarks" / (task["run_id"] + "_probe.pt")
    state, rng = D.load_checkpoint(path, model, optimizer, scaler, spec, runtime, task)
    limits = spec["benchmark"]
    if state["step"] != limits["warmup_steps"] + limits["measured_steps"]:
        raise ValueError("Not the completed benchmark's post-stress checkpoint")
    report.update(checkpoint_state=state, initial_scaler=scaler.state_dict())
    if not finite_state(torch, (model.state_dict(), optimizer.state_dict())):
        raise FloatingPointError("Nonfinite saved model/optimizer state")
    loader, _ = D.build_loader("train", spec["inventory"], 512, seed=task["seed"])
    hook = None
    try:
        # Match original benchmark ordering, then restore the checkpoint RNG so
        # recreating a loader/first batch does not change the resumed model RNG.
        iterator = iter(loader)
        D.seed_everything(task["seed"])
        batch = next(iterator)
        fingerprint = D.batch_fingerprint(batch)
        original = read(campaign / "benchmarks" / (task["run_id"] + ".json"))
        if fingerprint != original["input_fingerprints"]["0"]:
            raise ValueError("Recreated first training batch differs from the benchmark")
        if batch[0]["pf_mask"].shape != (512, 1, 128):
            raise ValueError("Recovery requires the original 512 x 128 padded batch")
        report["input_fingerprint"] = fingerprint
        D.restore_rank_state(model, rng)
        for module in model.modules():
            if type(module).__name__ == "SequenceTrimmer":
                module.enabled = False
        D.diagnostic_mode(model, True)
        optimizer_steps = [0]

        def count_step(*_):
            optimizer_steps[0] += 1

        hook = optimizer.register_step_post_hook(count_step)
        peak_step = recipe.get("warmup_steps", 1) - 1
        report["lr_schedule_index"] = peak_step
        torch.cuda.reset_peak_memory_stats()
        with (output / (task["run_id"] + ".steps.jsonl")).open("x", buffering=1) as log:
            for index in range(POLICY["calibration_attempts"] + POLICY["confirmation_attempts"]):
                D.check_stop()
                confirmation = index >= POLICY["calibration_attempts"]
                before, count_before = scaler.get_scale(), optimizer_steps[0]
                started = time.monotonic()
                step = D.training_step(model, optimizer, scaler, batch, device, recipe, peak_step,
                                       full_length=True, check_precision=index == 0)
                torch.cuda.synchronize()
                row = dict(step, attempt=index + 1, phase="confirmation" if confirmation else "calibration",
                           scale_before=before, scale_after=scaler.get_scale(),
                           optimizer_steps=optimizer_steps[0] - count_before,
                           state_finite=finite_state(torch, (model.state_dict(), optimizer.state_dict())),
                           seconds=time.monotonic() - started,
                           peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                           peak_reserved_bytes=torch.cuda.max_memory_reserved())
                failure = assess_attempt(row, confirmation=confirmation)
                if row["peak_allocated_bytes"] >= runtime["gpu_memory_bytes"] * limits["memory_fraction_limit"]:
                    failure = "recovery exceeded original peak allocated-memory limit"
                if index == POLICY["calibration_attempts"] - 1 and row["amp_skipped"]:
                    failure = "calibration did not end with a successful update"
                row["failure"] = failure
                report["attempts"].append(row)
                log.write(json.dumps(row, allow_nan=False) + "\n")
                print(json.dumps(dict(stage="amp_recovery_attempt", run_id=task["run_id"], **row), allow_nan=False), flush=True)
                if failure:
                    raise FloatingPointError(failure)
        report["final_scaler"] = scaler.state_dict()
        report["passed"] = True
    finally:
        if hook is not None:
            hook.remove()
        D.diagnostic_mode(model, False)
        D.shutdown_loader(loader)


def run(output, index):
    plan = load_plan(output)
    if type(index) is not int or not 0 <= index < len(plan["tasks"]):
        raise ValueError("Need a task index 0..6")
    campaign, task = Path(plan["campaign"]), plan["tasks"][index]
    D = frozen_driver(campaign)
    with D.P.file_lock(output / (task["run_id"] + ".lock"), blocking=False):
        result = output / (task["run_id"] + ".json")
        if result.exists() or (output / (task["run_id"] + ".steps.jsonl")).exists():
            raise ValueError("Diagnostic task already attempted; preserve it and inspect before retrying")
        report = dict(schema=SCHEMA, recovery_sha256=plan["content_hash"], task=task,
                      passed=False, authorizes_training=False, attempts=[])
        started = time.monotonic()
        try:
            spec = D.P.load_campaign(campaign)
            if spec["content_hash"] != plan["campaign_sha256"] or spec["tasks"] != plan["tasks"]:
                raise ValueError("Frozen campaign changed")
            verify_artifacts(campaign, plan, task)
            device, runtime = D.gpu_setup(spec)
            report["runtime"] = runtime
            if runtime != plan["runtime"]:
                raise ValueError("GPU/runtime differs from original benchmark")
            run_probe(D, spec, task, runtime, device, campaign, output, report)
            verify_artifacts(campaign, plan, task)
        except BaseException as exc:
            report.update(passed=False, error=dict(type=type(exc).__name__, message=str(exc)))
            raise
        finally:
            report["seconds"] = time.monotonic() - started
            create_json(result, report)


def summarize(output):
    plan = load_plan(output)
    rows = []
    print("Recovery check only: does NOT authorize training. Original benchmarks are unchanged.\n")
    print(f"{'Model':29s} {'Status':10s} {'Cal skips':>10s} {'Confirmed':>10s} {'Final scale':>12s}")
    for task in plan["tasks"]:
        path = output / (task["run_id"] + ".json")
        row = read(path) if path.exists() else None
        if row is not None and (row["task"] != task or row["recovery_sha256"] != plan["content_hash"]
                                or row["authorizes_training"] is not False):
            raise ValueError("Mismatched diagnostic result")
        attempts = row["attempts"] if row else []
        skips = sum(x["amp_skipped"] for x in attempts if x["phase"] == "calibration")
        confirmed = sum(x["phase"] == "confirmation" and x["failure"] is None for x in attempts)
        status = "MISSING" if row is None else ("PASS" if row["passed"] else "FAIL")
        scale = attempts[-1]["scale_after"] if attempts else "--"
        print(f"{task['run_id']:29s} {status:10s} {skips:10d} {confirmed:7d}/10 {str(scale):>12s}")
        if row and row.get("error"):
            print("  " + row["error"]["message"])
        rows.append(row)
    return all(row is not None and row["passed"] for row in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "run", "summary"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--task-index", type=int)
    args = parser.parse_args()
    output = args.output.resolve(strict=True)
    if args.phase == "prepare":
        if args.campaign is None:
            parser.error("prepare requires --campaign")
        prepare(args.campaign.resolve(strict=True), output)
    elif args.phase == "run":
        run(output, args.task_index)
    else:
        return 0 if summarize(output) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
