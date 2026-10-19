"""Frozen five-task protocol, shared-recipe matching, and admission gates."""
from __future__ import annotations

import copy
import math
from pathlib import Path
from part_ev_screen import protocol as S
from part_ev_fullscale.protocol import (CLASS_NAMES, digest, file_lock, file_sha256,
                                        read_json, write_json)
from .model import CONFIG, COUNTS
from .controls import checked_hash, runtime_identity
from .parity import POLICY as PARITY_POLICY, validate_report as validate_parity

SCHEMA = "part_ev_message_study_v1"
PRECISION_POLICY = S.PRECISION_POLICY
learning_rate = S.learning_rate
runtime_environment = S.runtime_environment
TASKS = [dict(run_id=name, architecture=arch, recipe="ranger_short_v1", seed=101)
         for name, arch in (("EV_STATE_RANGER", "ev_state"),
             ("EV_SEPARATE_RANGER", "ev_separate"),
             ("EV_STATE_SEPARATE_RANGER", "ev_state_separate"),
             ("EV_ROUTING_RANGER", "ev_routing"),
             ("EV_VARIANCE_RANGER", "ev_variance"))]
BENCHMARK = dict(warmup_steps=20, measured_steps=4200, window_steps=1000,
                 validation_events=128000, calibration_attempts=8, confirmation_attempts=10,
                 max_skip_fraction=.05, memory_fraction_limit=.90, planning_margin=1.30)
ROOT = Path(__file__).resolve().parents[1]


def recipes():
    return {"ranger_short_v1": S.recipes()["ranger_short_v1"]}


def task_directory(output, index):
    if type(index) is not int or not 0 <= index < len(TASKS):
        raise ValueError("Task index must be 0..4")
    task = TASKS[index]
    return Path(output) / "runs" / task["run_id"] / f"seed_{task['seed']}"


def source_manifest(root):
    root = Path(root)
    names = set(S.source_manifest(root))
    names.update(p.relative_to(root).as_posix() for p in (root / "part_ev_message_study").glob("*.py"))
    names.update(("scripts/run_part_ev_message_study.py",
                  "scripts/run_part_ev_screen_after_recovery.py",
                  "scripts/check_part_ev_screen_amp_recovery.py",
                  "sbatch/submit_part_ev_message_study.sh", "sbatch/run_part_ev_message_study.sh"))
    return {name: file_sha256(root / name) for name in sorted(names)}


def create_spec(inventory, launch, donor, data_contents):
    spec = dict(schema=SCHEMA, tasks=copy.deepcopy(TASKS), recipes=recipes(),
                parameters=COUNTS, model=CONFIG, benchmark=BENCHMARK, precision=PRECISION_POLICY,
                parity_policy=copy.deepcopy(PARITY_POLICY),
                inventory=inventory, launch=launch, environment=runtime_environment(),
                source=source_manifest(ROOT), data_config_sha256=file_sha256(S.DATA_CONFIG),
                donor=donor, data_contents_sha256=digest(data_contents))
    spec["content_hash"] = digest(spec)
    return spec


def load_campaign(output, *, check_environment=True):
    output = Path(output)
    spec = checked_hash(read_json(output / "campaign.json"))
    for key, expected in dict(schema=SCHEMA, tasks=TASKS, recipes=recipes(), parameters=COUNTS,
                              model=CONFIG, benchmark=BENCHMARK, precision=PRECISION_POLICY,
                              parity_policy=PARITY_POLICY,
                              source=source_manifest(ROOT)).items():
        if spec.get(key) != expected:
            raise ValueError(f"Frozen message study {key} differs; use a new campaign")
    if (spec["data_config_sha256"] != file_sha256(S.DATA_CONFIG)
            or spec["launch"] != read_json(output / "launch.json")
            or spec["data_contents_sha256"] != digest(read_json(output / "data_contents.json"))):
        raise ValueError("Data or launch contract changed")
    checked_hash(spec["inventory"], "inventory_sha256")
    if check_environment and spec["environment"] != runtime_environment():
        raise ValueError("Use the unchanged donor environment")
    donor = Path(spec["donor"]["path"])
    if checked_hash(read_json(donor / "campaign.json"))["content_hash"] != spec["donor"]["campaign_sha256"]:
        raise ValueError("Donor campaign changed")
    return spec


def verify_inventory_metadata(spec):
    for split in ("train", "val"):
        for row in spec["inventory"]["splits"][split]["files"]:
            path = Path(row["path"])
            stat = path.lstat()
            if path.is_symlink() or (stat.st_size, stat.st_mtime_ns) != (row["size_bytes"], row["mtime_ns"]):
                raise ValueError(f"Input dataset changed after prepare: {path}")


def precision_ok(row, dtype):
    observed = row.get("observed_activations", {})
    return (row.get("passed") is True and row.get("parameter_dtypes") == ["torch.float32"]
            and row.get("expected_activation_dtype") == dtype and bool(observed)
            and all(value == dtype for value in observed.values()))


def benchmark_summary(output, spec, runtime=None):
    rows, hashes = [], {}
    for task in TASKS:
        path = Path(output) / "benchmarks" / (task["run_id"] + ".json")
        row = checked_hash(read_json(path))
        validate_parity(row.get("initial_parity", {}))
        if (row["task"] != task or row["campaign_sha256"] != spec["content_hash"]
                or row.get("passed") is not True or row["initial_parity"].get("passed") is not True
                or row["measured_steps"] != BENCHMARK["measured_steps"]
                or not precision_ok(row["training_precision"], "torch.float16")
                or not precision_ok(row["validation"]["precision"], "torch.float32")):
            raise ValueError("All five exact-campaign memory/precision/parity benchmarks must pass")
        if not (0 <= row["amp_skipped_steps"] < BENCHMARK["measured_steps"] * BENCHMARK["max_skip_fraction"]):
            raise ValueError("Too many sustained AMP skips")
        attempts = row["full_length_attempts"]
        calibration = BENCHMARK["calibration_attempts"]
        if len(attempts) != calibration + BENCHMARK["confirmation_attempts"]:
            raise ValueError("Incomplete full-length calibration/confirmation")
        for index, attempt in enumerate(attempts):
            if (attempt["attempt"] != index + 1 or not math.isfinite(attempt["loss"])
                    or attempt["state_finite"] is not True
                    or (index >= calibration - 1 and (attempt["amp_skipped"]
                        or attempt["gradient_norm"] is None or not math.isfinite(attempt["gradient_norm"])))):
                raise ValueError("Full-length AMP did not settle into finite successful updates")
        expected = runtime or (rows[0]["runtime"] if rows else row["runtime"])
        if (row["runtime"] != expected or runtime_identity(row["runtime"]) != spec["donor"]["runtime"]
                or row["runtime"]["campaign_sha256"] != spec["content_hash"]):
            raise ValueError("GPU/runtime differs from donor or other benchmarks")
        for name in ("native_peak_bytes", "full_length_peak_bytes", "validation_peak_bytes"):
            if not 0 < row[name] < expected["gpu_memory_bytes"] * BENCHMARK["memory_fraction_limit"]:
                raise ValueError("GPU memory headroom check failed")
        if not math.isfinite(row["projected_training_hours"]) or row["projected_training_hours"] <= 0:
            raise ValueError("Invalid runtime projection")
        if (row["validation"]["events"] != BENCHMARK["validation_events"]
                or not math.isfinite(row["validation"]["seconds"]) or row["validation"]["seconds"] <= 0):
            raise ValueError("Invalid validation timing/population")
        if rows and row["input_fingerprints"] != rows[0]["input_fingerprints"]:
            raise ValueError("New model input streams differ")
        donor_row = read_json(Path(spec["donor"]["path"]) / "benchmarks/EV_RANGER.json")
        if row["input_fingerprints"] != donor_row["input_fingerprints"]:
            raise ValueError("New model input stream differs from reused EV control")
        rows.append(row)
        hashes[task["run_id"]] = file_sha256(path)
    return dict(passed=True, runtime=rows[0]["runtime"], models=rows, benchmark_sha256=hashes,
                recommended_train_hours=max(1, math.ceil(BENCHMARK["planning_margin"] *
                                                        max(r["projected_training_hours"] for r in rows))),
                recommended_evaluate_hours=max(2, math.ceil(BENCHMARK["planning_margin"] *
                    max(r["validation"]["seconds"] * 5000000 / r["validation"]["events"] for r in rows)/3600)))


def approve(output, spec, *, reviewed):
    if not reviewed:
        raise ValueError("Review benchmarks and pass --reviewed before training")
    summary = benchmark_summary(output, spec)
    payload = dict(schema=SCHEMA, campaign_sha256=spec["content_hash"],
                   benchmark_sha256=summary["benchmark_sha256"], reviewed=True,
                   initialization="fresh_seeded_model_optimizer_scaler")
    payload["content_hash"] = digest(payload)
    path = Path(output) / "production_approval.json"
    with file_lock(Path(output) / ".approval.lock"):
        if path.exists() and read_json(path) != payload:
            raise ValueError("Reviewed benchmark evidence changed")
        if not path.exists():
            write_json(path, payload)
    return summary


def require_approval(output, spec):
    approval = checked_hash(read_json(Path(output) / "production_approval.json"))
    summary = benchmark_summary(output, spec)
    if (approval.get("reviewed") is not True or approval["campaign_sha256"] != spec["content_hash"]
            or approval["benchmark_sha256"] != summary["benchmark_sha256"]
            or approval["initialization"] != "fresh_seeded_model_optimizer_scaler"):
        raise ValueError("Valid reviewed admission required")
    return summary
