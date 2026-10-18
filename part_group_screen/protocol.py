"""Immutable new campaign; donor results can become ready independently."""
from __future__ import annotations
import copy
import math
from pathlib import Path
from part_ev_fullscale.protocol import read_json, write_json, file_sha256, file_lock
from part_ev_screen.protocol import recipes, learning_rate, runtime_environment, BENCHMARK
from part_ev_screen.precision import POLICY
from part_screen_common.contracts import (checked_json, hashed, donor_contract, verify_source,
                                           verify_dataset_metadata, runtime_identity)
from .config import CONFIG, TASKS

SCHEMA = "part_group_screen_v1"
DONOR_DEFAULT = "/home/ryreu/atlas/Fresh_check/checkpoints/part_ev_screen/context_gate_ampfix_20261016T215608Z"


def task_directory(output, index, tasks=None):
    tasks = TASKS if tasks is None else tasks
    if type(index) is not int or not 0 <= index < len(tasks):
        raise ValueError("New-model task index out of range; controls are not tasks")
    t = tasks[index]
    return Path(output) / "runs" / t["run_id"] / f"seed_{t['seed']}"


def source_manifest(root):
    root = Path(root)
    paths = []
    for package in ("part_group_screen", "part_screen_common", "part_ev_screen", "part_ev_fullscale"):
        paths.extend((root / package).glob("*.py"))
    paths.extend(root / name for name in ("scripts/run_part_group_screen.py", "scripts/run_part_ev_screen.py",
        "scripts/export_part_screen_controls.py", "sbatch/run_part_group_screen_sporc.sh",
        "sbatch/submit_part_group_screen_sporc.sh", "particle_transformer/data/JetClass/JetClass_full.yaml"))
    return {p.relative_to(root).as_posix(): file_sha256(p) for p in sorted(paths)}


def create_spec(donor, controls, source_root, launch, parameters):
    contract = donor_contract(donor, controls, local_source=source_root)
    if contract["environment"] != runtime_environment():
        raise ValueError("Use donor-matched unchanged SPORC Python environment")
    if any(t["recipe"] != "ranger_short_v1" for t in contract["tasks"].values()):
        raise ValueError("Initial grouped screen only reuses Ranger controls")
    if "PART_RANGER" not in controls:
        raise ValueError("PART_RANGER is required")
    verify_dataset_metadata(contract["inventory"])
    return hashed(dict(schema=SCHEMA, tasks=copy.deepcopy(TASKS), model=copy.deepcopy(CONFIG),
        recipes={"ranger_short_v1": contract["recipes"]["ranger_short_v1"]}, donor=contract,
        environment=contract["environment"], inventory=contract["inventory"], precision=POLICY,
        benchmark=copy.deepcopy(BENCHMARK), parameters=parameters, launch=launch,
        source=source_manifest(source_root), dataset_evidence="unchanged_donor_paths_sizes_mtimes_and_entry_inventory"))


def load_campaign(output, *, check_environment=True):
    spec = checked_json(Path(output) / "experiment_spec.json", SCHEMA)
    if (spec["tasks"] != TASKS or spec["model"] != CONFIG or spec["precision"] != POLICY
            or spec["recipes"] != {"ranger_short_v1": recipes()["ranger_short_v1"]}
            or spec["benchmark"] != BENCHMARK):
        raise ValueError("Grouped frozen protocol differs")
    if spec["source"] != source_manifest(Path(__file__).resolve().parents[1]):
        raise ValueError("Frozen source differs; use the campaign snapshot")
    if check_environment and spec["environment"] != runtime_environment():
        raise ValueError("Grouped environment differs from donor")
    return spec


def benchmark_summary(output, spec, runtime=None):
    rows = []
    for task in spec["tasks"]:
        row = read_json(Path(output) / "benchmarks" / f"{task['run_id']}.json")
        if (row.get("passed") is not True or row.get("task") != task
                or row.get("campaign_sha256") != spec["content_hash"]
                or row.get("initial_parity", {}).get("passed") is not True
                or runtime_identity(row["runtime"]) != spec["donor"]["runtime"]
                or row["input_fingerprints"] != spec["donor"]["input_fingerprints"]):
            raise ValueError("All exact-campaign grouped benchmarks must pass donor/runtime/input parity")
        if runtime is not None and row["runtime"] != runtime:
            raise ValueError("Production runtime differs from reviewed benchmarks")
        for audit, dtype in ((row["training_precision"], "torch.float16"),
                             (row["validation"]["precision"], "torch.float32")):
            observed = audit.get("observed_activations", {})
            if (audit.get("passed") is not True or audit.get("parameter_dtypes") != ["torch.float32"]
                    or audit.get("expected_activation_dtype") != dtype or not observed
                    or any(v != dtype for v in observed.values())):
                raise ValueError("Observed training/validation precision gate failed")
        if (row["amp_skipped_steps"] / row["measured_steps"] >= spec["benchmark"]["max_skip_fraction"]
                or row["full_length_amp_skips"] != 0
                or max(row[k] for k in ("native_peak_bytes", "full_length_peak_bytes", "validation_peak_bytes"))
                >= row["runtime"]["gpu_memory_bytes"] * spec["benchmark"]["memory_fraction_limit"]):
            raise ValueError("Grouped benchmark numerical/memory admission failed")
        if not math.isfinite(row["projected_training_hours"]) or row["projected_training_hours"] <= 0:
            raise ValueError("Missing sustained timing evidence")
        rows.append(row)
    margin = spec["benchmark"]["planning_margin"]
    return dict(passed=True, runtime=rows[0]["runtime"], models=rows,
        recommended_train_hours=max(1, math.ceil(margin * max(r["projected_training_hours"] for r in rows))),
        recommended_evaluate_hours=max(2, math.ceil(margin * max(
            r["validation"]["seconds"] * 5000000 / r["validation"]["events"] for r in rows) / 3600)))


def approve(output, spec):
    summary = benchmark_summary(output, spec)
    record = hashed(dict(schema="part_group_admission_v1", campaign_sha256=spec["content_hash"],
        initialization="fresh_seed_model_optimizer_scaler", reviewed=True, summary=summary,
        benchmarks={t["run_id"]: file_sha256(Path(output) / "benchmarks" / f"{t['run_id']}.json") for t in spec["tasks"]}))
    path = Path(output) / "production_approval/approval.json"
    with file_lock(Path(output) / ".approval.lock"):
        if path.exists() and checked_json(path) != record:
            raise ValueError("Existing approval differs; do not overwrite")
        write_json(path, record)
    return summary


def require_approval(output, spec, runtime=None):
    record = checked_json(Path(output) / "production_approval/approval.json", "part_group_admission_v1")
    summary = benchmark_summary(output, spec, runtime)
    if (record.get("campaign_sha256") != spec["content_hash"] or record.get("reviewed") is not True
            or record.get("initialization") != "fresh_seed_model_optimizer_scaler" or record.get("summary") != summary
            or record["benchmarks"] != {t["run_id"]: file_sha256(Path(output) / "benchmarks" / f"{t['run_id']}.json") for t in spec["tasks"]}):
        raise ValueError("Production admission changed")
    return summary
