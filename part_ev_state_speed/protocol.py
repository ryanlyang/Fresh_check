"""Reuse admission machinery in a private namespace, with a new frozen contract."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from part_ev_message_study import protocol as historical, controls as C
from .model import CONFIG, COUNTS

# Relative imports still resolve to the historical helper package. Assignments
# below affect only this private module, never the historical protocol object.
_spec = importlib.util.spec_from_file_location("part_ev_message_study._speed_protocol", historical.__file__)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
SCHEMA = "part_ev_state_speed_v1"
ROOT = Path(__file__).resolve().parents[1]
TASKS = [dict(run_id=run, architecture=arch, recipe="ranger_short_v1", seed=101)
         for run, arch in (("EV_STATE_FAST_RANGER", "state_fast"),
                          ("EV_STATE_LATE4_RANGER", "state_late4"),
                          ("EV_STATE_LATE4_R16_RANGER", "state_late4_r16"))]
BENCHMARK = copy.deepcopy(historical.BENCHMARK)
SPEED_POLICY = dict(version=1, batch_sizes=[1, 512], warmup=5, repeats=3, steps=20,
                    precision="fp32", timing="GPU_resident_forward_including_launch_overhead",
                    equivalent_events=64, gradient_rtol=1e-4, gradient_atol=1e-5)
DEFAULT_STATE_DONOR = "/home/ryreu/atlas/Fresh_check/checkpoints/part_ev_message_study/messages_parityfix_20261019T053047Z"


def source_manifest(root):
    names = set(historical.source_manifest(root))
    names.update(p.relative_to(root).as_posix() for p in (Path(root) / "part_ev_state_speed").glob("*.py"))
    names.update(("scripts/run_part_ev_state_speed.py", "sbatch/run_part_ev_state_speed.sh",
                  "sbatch/submit_part_ev_state_speed.sh"))
    return {name: historical.file_sha256(Path(root) / name) for name in sorted(names)}


def task_directory(output, index):
    if type(index) is not int or not 0 <= index < len(TASKS):
        raise ValueError("Task index must be 0..2")
    task = TASKS[index]
    return Path(output) / "runs" / task["run_id"] / f"seed_{task['seed']}"


def state_reference(path, donor):
    root = Path(path).resolve(strict=True)
    spec = historical.load_campaign(root)
    historical.require_approval(root, spec)
    if spec["donor"] != donor:
        raise ValueError("State reference must use the same verified ParT/EV donor")
    task = next(t for t in spec["tasks"] if t["run_id"] == "EV_STATE_RANGER")
    trained = C.validate_training(root, spec, task, donor["runtime"])
    checkpoint = root / "runs/EV_STATE_RANGER/seed_101/best_model_val.pt"
    return spec, trained, dict(path=str(root), campaign_sha256=spec["content_hash"],
        checkpoint=str(checkpoint), checkpoint_sha256=trained["checkpoint_sha256"],
        artifact_sha256=trained["artifact_sha256"])


def create_spec(inventory, launch, donor, data_contents):
    spec = _base.create_spec(inventory, launch, donor, data_contents)
    state, _, reference = state_reference(launch["state_donor"], donor)
    if state["inventory"] != inventory:
        raise ValueError("State reference uses a different population")
    spec.pop("content_hash")
    spec.update(state_reference=reference, speed_policy=SPEED_POLICY)
    spec["content_hash"] = historical.digest(spec)
    return spec


def load_campaign(output, *, check_environment=True):
    spec = _base.load_campaign(output, check_environment=check_environment)
    if spec.get("speed_policy") != SPEED_POLICY:
        raise ValueError("Speed benchmark policy changed")
    ref = spec["state_reference"]
    for name, checksum in ref["artifact_sha256"].items():
        if historical.file_sha256(Path(ref["path"]) / name) != checksum:
            raise ValueError("Read-only state reference artifact changed")
    if (ref["path"] != str(Path(spec["launch"]["state_donor"]).resolve())
            or historical.file_sha256(ref["checkpoint"]) != ref["checkpoint_sha256"]):
        raise ValueError("State checkpoint identity changed")
    return spec


def validate_parity(report):
    historical.validate_parity(report)
    evidence = report.get("state_speed_equivalence", {})
    if (evidence.get("passed") is not True or evidence.get("nonzero_branch") is not True
            or evidence.get("policy") != SPEED_POLICY
            or evidence.get("trained_checkpoint", {}).get("passed") is not True
            or evidence.get("gradients", {}).get("passed") is not True):
        raise ValueError("Missing nonzero-state forward/gradient equivalence evidence")
    g = evidence["gradients"]
    if (g.get("checkpoint_backward") is not True or g.get("tensors", 0) < 1
            or g.get("forward", {}).get("passed") is not True):
        raise ValueError("Missing checkpoint-backward evidence")


for _name, _value in dict(SCHEMA=SCHEMA, ROOT=ROOT, TASKS=TASKS, COUNTS=COUNTS,
                         CONFIG=CONFIG, BENCHMARK=BENCHMARK, source_manifest=source_manifest,
                         task_directory=task_directory, validate_parity=validate_parity).items():
    setattr(_base, _name, _value)


def benchmark_summary(output, spec, runtime=None):
    summary = _review_core(output, spec, runtime)
    from .benchmark import validate_speed_report
    for row in summary["models"]:
        path = Path(output) / "benchmarks" / (row["task"]["run_id"] + "_speed.json")
        speed = C.checked_hash(historical.read_json(path))
        validate_speed_report(speed, spec, row["task"])
        trained = row["initial_parity"]["state_speed_equivalence"]["trained_checkpoint"]
        if (trained.get("checkpoint_sha256") != spec["state_reference"]["checkpoint_sha256"]
                or trained.get("events") != SPEED_POLICY["equivalent_events"]
                or speed.get("input_fingerprint") != row["input_fingerprints"]["0"]):
            raise ValueError("Trained parity/timing inputs do not match this campaign")
        summary["benchmark_sha256"][row["task"]["run_id"] + "_speed"] = historical.file_sha256(path)
    return summary


# The inherited approval functions call the augmented review, including timings.
_review_core = _base.benchmark_summary
_base.benchmark_summary = benchmark_summary


def __getattr__(name):
    return getattr(_base, name)
