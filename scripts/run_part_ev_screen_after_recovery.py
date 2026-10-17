#!/usr/bin/env python3
"""Explicit recovery approval around the unchanged, frozen screen trainer.

Only the admission check is replaced in memory. Models, optimizers, AMP,
initialization, checkpoint selection, data, and training budget are unchanged.
"""
from __future__ import annotations

from contextlib import contextmanager
import importlib.util
import json
import math
from pathlib import Path
import sys

_helper = importlib.util.spec_from_file_location("screen_recovery_checker",
    Path(__file__).with_name("check_part_ev_screen_amp_recovery.py"))
R = importlib.util.module_from_spec(_helper)
_helper.loader.exec_module(R)

SCHEMA = "part_ev_screen_reviewed_recovery_approval_v1"
BASIS = "original_memory_precision_parity_throughput_plus_bounded_amp_recovery"


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def validate_recovery(row, plan, spec, task, original, learning_rate):
    if (row.get("schema") != R.SCHEMA or row.get("passed") is not True or row.get("error")
            or row.get("authorizes_training") is not False or row.get("task") != task
            or row.get("recovery_sha256") != plan["content_hash"] or row.get("runtime") != plan["runtime"]
            or row.get("input_fingerprint") != original["input_fingerprints"]["0"]
            or row.get("checkpoint_state") != dict(task=task,
                step=spec["benchmark"]["warmup_steps"] + spec["benchmark"]["measured_steps"])):
        raise ValueError("Recovery result identity, checkpoint, runtime or status mismatch")
    recipe = spec["recipes"][task["recipe"]]
    peak_step = recipe.get("warmup_steps", 1) - 1
    if row["lr_schedule_index"] != peak_step:
        raise ValueError("Recovery did not use the prescribed peak LR index")
    attempts = row["attempts"]
    calibration = R.POLICY["calibration_attempts"]
    if len(attempts) != calibration + R.POLICY["confirmation_attempts"]:
        raise ValueError("Recovery must include all 18 attempts")
    previous_scale = row["initial_scaler"]["scale"]
    if not positive(previous_scale):
        raise ValueError("Invalid initial recovery scale")
    for index, attempt in enumerate(attempts):
        confirmation = index >= calibration
        if (attempt["attempt"] != index + 1
                or attempt["phase"] != ("confirmation" if confirmation else "calibration")
                or attempt.get("failure") is not None or attempt["state_finite"] is not True
                or type(attempt["amp_skipped"]) is not bool
                or type(attempt["optimizer_steps"]) is not int
                or attempt["scale_before"] != previous_scale
                or R.assess_attempt(attempt, confirmation=confirmation) is not None):
            raise ValueError("Recovery has incomplete, nonfinite, skipped or inconsistent updates")
        if index == calibration - 1 and attempt["amp_skipped"]:
            raise ValueError("Calibration must end with a successful update")
        if (not positive(attempt["lr"]) or not math.isclose(attempt["lr"], learning_rate(peak_step, recipe),
                                                         rel_tol=1e-12, abs_tol=1e-15)):
            raise ValueError("Recovery LR differs from original recipe peak")
        allocated, reserved = attempt["peak_allocated_bytes"], attempt["peak_reserved_bytes"]
        memory = plan["runtime"]["gpu_memory_bytes"]
        if (not positive(allocated) or not positive(reserved) or reserved < allocated or reserved > memory
                or allocated >= memory * spec["benchmark"]["memory_fraction_limit"]):
            raise ValueError("Recovery memory gate failed")
        previous_scale = attempt["scale_after"]
    if row["final_scaler"]["scale"] != previous_scale:
        raise ValueError("Recovery final scaler differs from the logged updates")
    audit = attempts[0].get("precision", {})
    observed = audit.get("observed_activations", {})
    if (audit.get("passed") is not True or audit.get("parameter_dtypes") != ["torch.float32"]
            or audit.get("expected_activation_dtype") != "torch.float16" or not observed
            or any(dtype != "torch.float16" for dtype in observed.values())):
        raise ValueError("Recovery lacks a passing observed FP16 precision audit")


def evidence(campaign, recovery, D, spec):
    if not recovery.is_relative_to(campaign / "diagnostics"):
        raise ValueError("Recovery must belong to this campaign's diagnostics directory")
    plan = R.load_plan(recovery)
    if (Path(plan["campaign"]).resolve() != campaign or plan["campaign_sha256"] != spec["content_hash"]
            or plan["tasks"] != spec["tasks"] or R.read(recovery / "launch.json") != spec["launch"]):
        raise ValueError("Recovery belongs to a different campaign/protocol")
    models, hashes = [], {}
    for task in spec["tasks"]:
        R.verify_artifacts(campaign, plan, task)
        original = R.read(campaign / "benchmarks" / (task["run_id"] + ".json"))
        R.eligible_report(original, spec, task)
        if original["runtime"] != plan["runtime"]:
            raise ValueError("Original runtime differs from recovery")
        path = recovery / (task["run_id"] + ".json")
        row = R.read(path)
        validate_recovery(row, plan, spec, task, original, D.P.learning_rate)
        steps = path.with_suffix(".steps.jsonl")
        if [json.loads(line) for line in steps.read_text().splitlines()] != row["attempts"]:
            raise ValueError("Recovery step log and report differ")
        for item in (path, steps):
            hashes[item.name] = R.sha(item)
        validation = original["validation"]
        if (not positive(original["projected_training_hours"]) or not positive(validation["seconds"])
                or validation["events"] != spec["benchmark"]["validation_events"]):
            raise ValueError("Missing original training/validation timing evidence")
        models.append(original)
    if any(row["input_fingerprints"] != models[0]["input_fingerprints"] for row in models):
        raise ValueError("Original input streams differ")
    margin = spec["benchmark"]["planning_margin"]
    summary = dict(passed=True, admission_basis=BASIS, runtime=plan["runtime"], models=models,
        recommended_train_hours=max(1, math.ceil(margin * max(x["projected_training_hours"] for x in models))),
        recommended_evaluate_hours=max(2, math.ceil(margin * max(
            x["validation"]["seconds"] * spec["recipes"][x["task"]["recipe"]]["evaluation_population"]
            / x["validation"]["events"] for x in models) / 3600)))
    return plan, hashes, summary


def approve(campaign, recovery, approval, *, reviewed):
    if not reviewed:
        raise ValueError("Explicit --reviewed approval is required")
    if approval != campaign / "production_approval":
        raise ValueError("Use campaign/production_approval for the isolated admission wrapper")
    if any(p.is_file() for p in (campaign / "runs").rglob("*")):
        raise ValueError("Initial approval requires no existing production run artifacts")
    D = R.frozen_driver(campaign)
    spec = D.P.load_campaign(campaign)
    if R.read(approval / "launch.json") != spec["launch"]:
        raise ValueError("Production launch resources differ from the campaign")
    plan, hashes, summary = evidence(campaign, recovery, D, spec)
    if R.sha(approval / "launch.env") != plan["launch_sha256"]:
        raise ValueError("Production shell resources differ from the recovered benchmark")
    payload = dict(schema=SCHEMA, campaign=str(campaign), campaign_sha256=spec["content_hash"],
        recovery=str(recovery), recovery_plan_sha256=R.sha(recovery / "recovery.json"),
        recovery_results_sha256=hashes, authorizes_training=True, reviewed=True, basis=BASIS,
        initialization="fresh_original_seed_model_optimizer_and_grad_scaler",
        runtime=plan["runtime"], driver_sha256=R.sha(__file__), helper_sha256=R.sha(R.__file__),
        launch_sha256=R.sha(approval / "launch.env"), summary=summary)
    payload["content_hash"] = R.digest(payload)
    R.create_json(approval / "approval.json", payload)
    return summary


def load_approval(approval):
    payload = R.read(approval / "approval.json")
    if (payload.get("schema") != SCHEMA or payload.get("basis") != BASIS
            or payload.get("authorizes_training") is not True or payload.get("reviewed") is not True
            or R.digest({k: v for k, v in payload.items() if k != "content_hash"}) != payload.get("content_hash")
            or payload["driver_sha256"] != R.sha(__file__) or payload["helper_sha256"] != R.sha(R.__file__)
            or payload["launch_sha256"] != R.sha(approval / "launch.env")
            or payload["initialization"] != "fresh_original_seed_model_optimizer_and_grad_scaler"):
        raise ValueError("Approval source, policy or configuration changed")
    campaign, recovery = Path(payload["campaign"]), Path(payload["recovery"])
    if approval != campaign / "production_approval" or R.sha(recovery / "recovery.json") != payload["recovery_plan_sha256"]:
        raise ValueError("Approval/recovery location or plan changed")
    D = R.frozen_driver(campaign)
    spec = D.P.load_campaign(campaign)
    if spec["content_hash"] != payload["campaign_sha256"] or R.read(approval / "launch.json") != spec["launch"]:
        raise ValueError("Approved campaign/resources changed")
    plan, hashes, summary = evidence(campaign, recovery, D, spec)
    if (hashes != payload["recovery_results_sha256"] or summary != payload["summary"]
            or plan["runtime"] != payload["runtime"] or plan["launch_sha256"] != payload["launch_sha256"]):
        raise ValueError("Approved recovery evidence or timing changed")
    return payload, D, spec, summary


@contextmanager
def admission(D, campaign, spec, summary):
    """Replace only the entry gate, for this invocation, with verified evidence."""
    original = D.P.benchmark_summary

    def approved(output, supplied_spec, runtime=None):
        if Path(output).resolve() != campaign or supplied_spec != spec:
            raise ValueError("Approval cannot admit another campaign")
        if runtime is not None and runtime != summary["runtime"]:
            raise ValueError("Production GPU/runtime differs from approved benchmarks")
        return summary

    D.P.benchmark_summary = approved
    try:
        yield
    finally:
        D.P.benchmark_summary = original


def execute(approval, phase, index, *, resume=False):
    payload, D, spec, summary = load_approval(approval)
    campaign = Path(payload["campaign"])
    if phase in ("train", "evaluate"):
        run = D.P.task_directory(campaign, index)
        if phase == "train" and not resume and any(p.is_file() for p in run.rglob("*")):
            raise ValueError("Production files exist; inspect and explicitly --resume")
    elif phase != "report":
        raise ValueError("Only train/evaluate/report are admitted")
    print(json.dumps(dict(stage="reviewed_recovery_admission", phase=phase, task_index=index,
                         approval_sha256=payload["content_hash"], basis=BASIS, resume=resume)), flush=True)
    with admission(D, campaign, spec, summary):
        # Dispatch through the original CLI, including its task, environment,
        # source and checkpoint checks. No probe state is loaded into production.
        previous = sys.argv
        try:
            sys.argv = [str(campaign / "source/scripts/run_part_ev_screen.py"), "--phase", phase,
                        "--output", str(campaign)]
            if phase in ("train", "evaluate"):
                sys.argv += ["--task-index", str(index)]
            D.main()
        finally:
            sys.argv = previous


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("approve", "review", "train", "evaluate", "report"), required=True)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--recovery", type=Path)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--reviewed", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    approval = args.approval.resolve(strict=True)
    if args.phase == "approve":
        if args.campaign is None or args.recovery is None:
            parser.error("Approval needs --campaign and --recovery")
        result = approve(args.campaign.resolve(strict=True), args.recovery.resolve(strict=True),
                         approval, reviewed=args.reviewed)
    elif args.phase == "review":
        result = load_approval(approval)[3]
    else:
        execute(approval, args.phase, args.task_index, resume=args.resume)
        return 0
    # A small machine-readable summary; original report rows are kept in approval.json.
    print(json.dumps({k: v for k, v in result.items() if k != "models"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
