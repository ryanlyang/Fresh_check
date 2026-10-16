"""Separate, immutable seven-cell exploratory protocol; no historical mutations."""
from __future__ import annotations

import copy
import importlib.metadata
import math
from pathlib import Path
from part_ev_fullscale.protocol import (CLASS_NAMES, digest, environment, file_lock,
    file_sha256, read_json, write_json, recipe_for)
from part_ev_fullscale.data import DATA_CONFIG
from .model import CONFIG, COUNTS

SCHEMA = "part_ev_context_gate_screen_v1"
TASKS = [dict(run_id=name, architecture=architecture, recipe=recipe, seed=101)
    for name, architecture, recipe in (
        ("PART_RANGER", "part", "ranger_short_v1"),
        ("EV_RANGER", "part_ev", "ranger_short_v1"),
        ("EV_CONTEXT_RANGER", "part_ev_context", "ranger_short_v1"),
        ("EV_GATE_RANGER", "part_ev_relation_gate", "ranger_short_v1"),
        ("EV_CONTEXT_GATE_RANGER", "part_ev_context_relation_gate", "ranger_short_v1"),
        ("PART_ADAMW", "part", "adamw_short_v1"),
        ("EV_ADAMW", "part_ev", "adamw_short_v1"))]

BENCHMARK = dict(warmup_steps=20, measured_steps=4200, window_steps=1000,
    validation_events=128000, full_length_steps=3, max_skip_fraction=0.05,
    memory_fraction_limit=0.90, planning_margin=1.30)


def runtime_environment():
    result = environment()
    result["installed_distributions"] = dict(sorted((d.metadata["Name"].lower(), d.version)
        for d in importlib.metadata.distributions() if d.metadata["Name"]))
    return result


def recipes():
    result = {}
    for name, original in (("ranger_short_v1", "original"), ("adamw_short_v1", "adamw")):
        recipe = recipe_for(training_recipe=original)
        for field in ("test_population", "ddp_batchnorm", "ddp_trimming", "ddp_schedule_adaptation", "lr_decay_epochs"):
            recipe.pop(field, None)
        recipe.update(epochs=10, total_steps=200000, batchnorm="native_single_gpu",
                      evaluation_population=5000000, evaluation_split="official_validation",
                      checkpoint_auc="macro_ovo_and_ovr_not_used_for_selection")
        if original == "original":
            recipe.update(lr_schedule="compressed_reference_milestones", lr_decay_steps=list(range(140000, 200000, 4000)))
        else:
            recipe["warmup_steps"] = 10000
        result[name] = recipe
    return result


def learning_rate(step, recipe):
    if not 0 <= step < recipe["total_steps"]:
        raise ValueError("Update outside the frozen screen budget")
    if recipe["lr_schedule"] == "compressed_reference_milestones":
        return recipe["start_lr"] * .01 ** (sum(step >= m for m in recipe["lr_decay_steps"]) / 15)
    if recipe["lr_schedule"] != "linear_warmup_then_cosine":
        raise ValueError("Unknown schedule")
    ordinal, warmup = step + 1, recipe["warmup_steps"]
    if ordinal <= warmup:
        return recipe["start_lr"] * ordinal / warmup
    fraction = (ordinal - warmup) / (recipe["total_steps"] - warmup)
    return recipe["final_lr"] + .5 * (recipe["start_lr"] - recipe["final_lr"]) * (1 + math.cos(math.pi * fraction))


def task_directory(output, index):
    if type(index) is not int or not 0 <= index < len(TASKS):
        raise ValueError("Task index must be 0..6")
    task = TASKS[index]
    return Path(output) / "runs" / task["run_id"] / f"seed_{task['seed']}"


def source_manifest(root):
    root = Path(root)
    paths = [*root.glob("part_ev_screen/*.py"), *root.glob("part_ev_fullscale/*.py"),
             root / "scripts/run_part_ev_screen.py", root / "particle_transformer/data/JetClass/JetClass_full.yaml"]
    return {p.relative_to(root).as_posix(): file_sha256(p) for p in sorted(paths)}


def create_spec(inventory, source_root, launch):
    spec = dict(schema=SCHEMA, tasks=copy.deepcopy(TASKS), recipes=recipes(),
                model=CONFIG, parameters=COUNTS, benchmark=BENCHMARK,
                inventory=inventory, environment=runtime_environment(), launch=launch,
                source=source_manifest(source_root), data_config_sha256=file_sha256(DATA_CONFIG))
    spec["content_hash"] = digest(spec)
    return spec


def load_campaign(output, *, check_environment=True):
    spec = read_json(Path(output) / "campaign.json")
    payload = {k: v for k, v in spec.items() if k != "content_hash"}
    if digest(payload) != spec.get("content_hash") or spec.get("schema") != SCHEMA:
        raise ValueError("Screen campaign metadata hash/schema mismatch")
    if (spec["tasks"] != TASKS or spec["recipes"] != recipes() or spec["model"] != CONFIG
            or spec["parameters"] != COUNTS or spec["benchmark"] != BENCHMARK):
        raise ValueError("Frozen screen protocol differs")
    if spec["source"] != source_manifest(Path(__file__).resolve().parents[1]):
        raise ValueError("Frozen campaign source differs; do not patch active campaigns")
    if spec["data_config_sha256"] != file_sha256(DATA_CONFIG):
        raise ValueError("Input configuration changed")
    if check_environment and spec["environment"] != runtime_environment():
        raise ValueError("Use the unchanged, dedicated prepare environment")
    return spec


def benchmark_summary(output, spec, runtime=None):
    rows = []
    for task in TASKS:
        row = read_json(Path(output) / "benchmarks" / (task["run_id"] + ".json"))
        if (row.get("passed") is not True or row.get("campaign_sha256") != spec["content_hash"]
                or row.get("task") != task or row.get("initial_parity", {}).get("passed") is not True):
            raise ValueError("All seven exact-campaign benchmarks must pass before training")
        rows.append(row)
    expected = runtime or rows[0]["runtime"]
    if any(row["runtime"] != expected for row in rows):
        raise ValueError("Benchmarks and production must use one identical GPU/runtime")
    fingerprints = [row["input_fingerprints"] for row in rows]
    if any(f != fingerprints[0] for f in fingerprints):
        raise ValueError("Matched task input streams differ")
    hours = max(row["projected_training_hours"] for row in rows)
    return dict(passed=True, runtime=expected, models=rows,
                recommended_train_hours=math.ceil(hours * BENCHMARK["planning_margin"]))
