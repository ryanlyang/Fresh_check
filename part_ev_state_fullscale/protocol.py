"""Independent full-budget contract; no reused controls or screen checkpoints."""
from __future__ import annotations

import copy
import math
from pathlib import Path
from part_ev_fullscale.protocol import (CLASS_NAMES, SEEDS, digest, file_lock,
    file_sha256, read_json, write_json, recipe_for, learning_rate)
from part_ev_screen import protocol as S
from part_ev_message_study.controls import checked_hash
from part_ev_message_study.protocol import precision_ok, verify_inventory_metadata
from part_ev_message_study.parity import POLICY as PARITY_POLICY, validate_report
from .model import CONFIG, COUNTS

SCHEMA = "part_ev_state_fullscale_v1"
ROOT = Path(__file__).resolve().parents[1]
PRECISION_POLICY = S.PRECISION_POLICY
runtime_environment = S.runtime_environment
TASKS = [dict(run_id=name, architecture=arch, seed=seed, recipe="ranger_full_v1")
         for name, arch, seeds in (
             ("PART_RANGER", "part", SEEDS),
             ("EV_STATE_LATE4_RANGER", "state_late4", SEEDS),
             ("EV_RANGER", "part_ev", (101,))) for seed in seeds]
BENCHMARK_INDICES = (0, 3, 6)
BENCHMARK = dict(warmup_steps=20, measured_steps=4200, window_steps=1000,
    validation_events=128000, calibration_attempts=8, confirmation_attempts=10,
    max_skip_fraction=.05, memory_fraction_limit=.90, planning_margin=1.30)


def recipes():
    recipe = recipe_for(training_recipe="original")
    for field in ("ddp_batchnorm", "ddp_trimming", "ddp_schedule_adaptation"):
        recipe.pop(field)
    recipe.update(batchnorm="native_single_gpu", evaluation_population=5000000,
                  evaluation_split="official_validation", checkpoint_auc="not_used_for_selection")
    return {"ranger_full_v1": recipe}


def task_key(task):
    return f"{task['run_id']}_seed_{task['seed']}"


def task_directory(output, index):
    if type(index) is not int or not 0 <= index < len(TASKS):
        raise ValueError("Task index must be 0..6")
    task = TASKS[index]
    return Path(output) / "runs" / task["run_id"] / f"seed_{task['seed']}"


def source_manifest(root):
    root = Path(root)
    names = set(S.source_manifest(root))
    for package in ("part_ev_message_study", "part_ev_state_speed", "part_ev_state_fullscale"):
        names.update(p.relative_to(root).as_posix() for p in (root / package).glob("*.py"))
    names.update(("scripts/run_part_ev_state_fullscale.py", "scripts/check_part_ev_screen_amp_recovery.py",
                  "sbatch/run_part_ev_state_fullscale.sh", "sbatch/submit_part_ev_state_fullscale.sh"))
    return {name: file_sha256(root / name) for name in sorted(names)}


def create_spec(inventory, launch, data_contents):
    spec = dict(schema=SCHEMA, tasks=copy.deepcopy(TASKS), recipes=recipes(),
        model=copy.deepcopy(CONFIG), parameters=COUNTS, benchmark=BENCHMARK,
        benchmark_indices=list(BENCHMARK_INDICES), precision=PRECISION_POLICY,
        parity_policy=PARITY_POLICY, inventory=inventory, launch=launch,
        source=source_manifest(ROOT), environment=runtime_environment(),
        data_config_sha256=file_sha256(S.DATA_CONFIG), data_contents_sha256=digest(data_contents))
    spec["content_hash"] = digest(spec)
    return spec


def load_campaign(output, *, check_environment=True):
    output = Path(output)
    spec = checked_hash(read_json(output / "campaign.json"))
    expected = dict(schema=SCHEMA, tasks=TASKS, recipes=recipes(), model=CONFIG,
        parameters=COUNTS, benchmark=BENCHMARK, benchmark_indices=list(BENCHMARK_INDICES),
        precision=PRECISION_POLICY, parity_policy=PARITY_POLICY, source=source_manifest(ROOT))
    for key, value in expected.items():
        if spec.get(key) != value:
            raise ValueError(f"Frozen full-schedule {key} differs; use a new campaign")
    if (spec["launch"] != read_json(output / "launch.json")
            or spec["data_config_sha256"] != file_sha256(S.DATA_CONFIG)
            or spec["data_contents_sha256"] != digest(read_json(output / "data_contents.json"))):
        raise ValueError("Data or launch contract changed")
    checked_hash(spec["inventory"], "inventory_sha256")
    if check_environment and spec["environment"] != runtime_environment():
        raise ValueError("Use the unchanged prepared environment")
    return spec


def projected_hours(recipe, windows, validation, checkpoint_seconds, cold_seconds):
    # Fifty validations and full-budget checkpoint I/O, not the screen's hardcoded ten.
    per_step = max(w["seconds"] / w["steps"] for w in windows)
    saves = recipe["total_steps"] // 1000 + 3 * recipe["epochs"] + 2
    return (per_step * recipe["total_steps"] + validation["seconds"] * recipe["epochs"] *
            recipe["validation_events_per_epoch"] / validation["events"] +
            saves * checkpoint_seconds + cold_seconds) / 3600


def benchmark_summary(output, spec, runtime=None):
    rows, hashes = [], {}
    for index in BENCHMARK_INDICES:
        task = TASKS[index]
        path = Path(output) / "benchmarks" / (task["run_id"] + ".json")
        row = checked_hash(read_json(path))
        validate_report(row.get("initial_parity", {}))
        if (row.get("passed") is not True or row["task"] != task
                or row["campaign_sha256"] != spec["content_hash"]
                or row["measured_steps"] != BENCHMARK["measured_steps"]
                or not precision_ok(row["training_precision"], "torch.float16")
                or not precision_ok(row["validation"]["precision"], "torch.float32")):
            raise ValueError("All three exact-campaign benchmarks must pass")
        if task["architecture"] == "state_late4":
            gradient = row["initial_parity"].get("state_gradients", {})
            if (gradient.get("passed") is not True or gradient.get("checkpoint_backward") is not True
                    or gradient.get("forward", {}).get("passed") is not True
                    or gradient.get("tensors", 0) < 1
                    or not math.isfinite(gradient.get("max_absolute_error", math.inf))):
                raise ValueError("Nonzero state gradient/checkpoint parity missing")
        expected_runtime = runtime or (rows[0]["runtime"] if rows else row["runtime"])
        if row["runtime"] != expected_runtime or row["runtime"]["campaign_sha256"] != spec["content_hash"]:
            raise ValueError("GPU/runtime differs between benchmark and production")
        if not 0 <= row["amp_skipped_steps"] < BENCHMARK["measured_steps"] * BENCHMARK["max_skip_fraction"]:
            raise ValueError("Too many AMP skips")
        attempts = row["full_length_attempts"]
        calibration = BENCHMARK["calibration_attempts"]
        if len(attempts) != calibration + BENCHMARK["confirmation_attempts"]:
            raise ValueError("Incomplete full-length confirmation")
        for number, attempt in enumerate(attempts):
            if (attempt["attempt"] != number + 1 or not math.isfinite(attempt["loss"])
                    or attempt["state_finite"] is not True
                    or (number >= calibration - 1 and (attempt["amp_skipped"]
                        or attempt["gradient_norm"] is None or not math.isfinite(attempt["gradient_norm"])))):
                raise ValueError("Full-length AMP did not recover")
        for key in ("native_peak_bytes", "full_length_peak_bytes", "validation_peak_bytes"):
            if not 0 < row[key] < expected_runtime["gpu_memory_bytes"] * BENCHMARK["memory_fraction_limit"]:
                raise ValueError("Insufficient GPU memory headroom")
        val = row["validation"]
        windows = row["windows"]
        if (val["events"] != BENCHMARK["validation_events"] or not math.isfinite(val["seconds"])
                or val["seconds"] <= 0 or not windows
                or sum(w["steps"] for w in windows) != BENCHMARK["measured_steps"]
                or any(w["steps"] <= 0 or not math.isfinite(w["seconds"]) or w["seconds"] <= 0 for w in windows)
                or any(not math.isfinite(row[k]) or row[k] < 0 for k in ("checkpoint_seconds", "cold_fetch_seconds"))):
            raise ValueError("Invalid benchmark timing evidence")
        projected = projected_hours(recipes()[task["recipe"]], windows, val,
                                    row["checkpoint_seconds"], row["cold_fetch_seconds"])
        if not math.isclose(row["projected_training_hours"], projected, rel_tol=1e-12):
            raise ValueError("Projection does not cover the full schedule")
        if set(row["input_fingerprints"]) != {"0", "20", "1020", "2020", "4020"}:
            raise ValueError("Missing input stream fingerprints")
        if rows and row["input_fingerprints"] != rows[0]["input_fingerprints"]:
            raise ValueError("Seed-101 input streams differ across architectures")
        rows.append(row)
        hashes[task["run_id"]] = file_sha256(path)
    by_arch = {row["task"]["architecture"]: math.ceil(row["projected_training_hours"] *
               BENCHMARK["planning_margin"]) for row in rows}
    return dict(passed=True, runtime=rows[0]["runtime"], models=rows, benchmark_sha256=hashes,
        recommended_train_hours=max(by_arch.values()), train_hours_by_architecture=by_arch,
        recommended_evaluate_hours=max(5, math.ceil(BENCHMARK["planning_margin"] *
            max(r["validation"]["seconds"] * 5000000 / r["validation"]["events"] for r in rows) / 3600)))


def approve(output, spec, *, reviewed):
    if not reviewed:
        raise ValueError("Review preflight results and supply --reviewed")
    summary = benchmark_summary(output, spec)
    row = dict(schema=SCHEMA, campaign_sha256=spec["content_hash"], reviewed=True,
        benchmark_sha256=summary["benchmark_sha256"], initialization="fresh_model_optimizer_scaler")
    row["content_hash"] = digest(row)
    path = Path(output) / "production_approval.json"
    with file_lock(Path(output) / ".approval.lock"):
        if path.exists() and read_json(path) != row:
            raise ValueError("Previously approved evidence changed")
        if not path.exists():
            write_json(path, row)
    return summary


def require_approval(output, spec):
    row = checked_hash(read_json(Path(output) / "production_approval.json"))
    summary = benchmark_summary(output, spec)
    if (row.get("schema") != SCHEMA or row.get("reviewed") is not True
            or row["campaign_sha256"] != spec["content_hash"]
            or row["benchmark_sha256"] != summary["benchmark_sha256"]
            or row["initialization"] != "fresh_model_optimizer_scaler"):
        raise ValueError("Valid reviewed full-schedule approval required")
    return summary
