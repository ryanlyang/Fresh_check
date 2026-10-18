"""Validate donor identities without importing or altering their frozen trainer."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from part_ev_fullscale.protocol import digest, read_json, file_sha256
from part_ev_screen import protocol as reference
from part_ev_screen.precision import POLICY
from part_ev_fullscale.data import CLASSES, SPLITS

DONOR_SCHEMA = "part_ev_context_gate_screen_v2"
CONTROL_TASKS = {t["run_id"]: t for t in reference.TASKS
                 if t["run_id"] in ("PART_RANGER", "EV_RANGER", "PART_ADAMW", "EV_ADAMW")}
REQUIRED_SOURCE = {
    "part_ev_fullscale/model.py", "part_ev_fullscale/optimizer.py",
    "part_ev_fullscale/checkpoint.py", "part_ev_fullscale/data.py",
    "part_ev_screen/model.py", "part_ev_screen/protocol.py", "part_ev_screen/data.py",
    "part_ev_screen/precision.py", "scripts/run_part_ev_screen.py",
    "particle_transformer/data/JetClass/JetClass_full.yaml",
}


def hashed(payload):
    return dict(payload, content_hash=digest(payload))


def checked_json(path, schema=None, *, hash_function=digest):
    row = read_json(path)
    if (not isinstance(row, dict) or row.get("content_hash") != hash_function(
            {k: v for k, v in row.items() if k != "content_hash"})
            or (schema is not None and row.get("schema") != schema)):
        raise ValueError(f"Invalid metadata hash/schema: {path}")
    return row


def _recovery_approval_digest(payload):
    """Match the v1 approval producer, not the compact campaign JSON encoding.

    scripts/check_part_ev_screen_amp_recovery.py uses JSON's default separators;
    run_part_ev_screen_after_recovery.py hashes approvals with that helper. Keep
    this format-specific and do not import/execute a donor's frozen source.
    """
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def child(root, relative):
    root = Path(root).resolve(strict=True)
    relative = Path(relative)
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root) or path == root:
        raise ValueError(f"Unsafe artifact path: {relative}")
    return path


def runtime_identity(runtime):
    """Only campaign bookkeeping differs across otherwise matched runtimes."""
    return {k: v for k, v in runtime.items() if k != "campaign_sha256"}


def verify_source(root, manifest):
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("Missing frozen source manifest")
    for name, checksum in manifest.items():
        path = child(root, name)
        if not path.is_file() or file_sha256(path) != checksum:
            raise ValueError(f"Frozen source differs: {name}")


def donor_contract(root, controls, *, local_source=None):
    root = Path(root).resolve(strict=True)
    controls = list(controls)
    if not controls or len(set(controls)) != len(controls) or any(x not in CONTROL_TASKS for x in controls):
        raise ValueError("Unknown, duplicate or empty control selection")
    spec = checked_json(root / "campaign.json", DONOR_SCHEMA)
    if spec.get("precision") != POLICY:
        raise ValueError("Donor precision differs")
    tasks = {t["run_id"]: t for t in spec["tasks"]}
    if len(tasks) != len(spec["tasks"]):
        raise ValueError("Duplicate donor task identity")
    expected_recipes = reference.recipes()
    for name in controls:
        task = CONTROL_TASKS[name]
        if tasks.get(name) != task or spec["recipes"].get(task["recipe"]) != expected_recipes[task["recipe"]]:
            raise ValueError(f"Donor task/recipe differs: {name}")
        if spec.get("parameters", {}).get(task["architecture"]) != reference.COUNTS[task["architecture"]]:
            raise ValueError("Donor control parameter count differs")
    if not REQUIRED_SOURCE.issubset(spec["source"]):
        raise ValueError("Donor source manifest omits required numerical/data code")
    verify_source(root / "source", spec["source"])
    config = "particle_transformer/data/JetClass/JetClass_full.yaml"
    if spec.get("data_config_sha256") != spec["source"].get(config):
        raise ValueError("Donor preprocessing identity differs")
    # All existing reference sources must remain identical. New packages do not
    # invalidate this comparison, and no donor code is loaded dynamically.
    if local_source is not None:
        verify_source(local_source, spec["source"])
    inventory = spec["inventory"]
    if inventory.get("inventory_sha256") != digest({k: v for k, v in inventory.items() if k != "inventory_sha256"}):
        raise ValueError("Donor inventory hash differs")
    for split, count in (("train", 1000), ("val", 50)):
        rows = inventory["splits"][split]["files"]
        expected_names = {f"{name}_{i:03d}.root" for name in CLASSES for i in SPLITS[split][1]}
        if (len(rows) != count or len({r["name"] for r in rows}) != count
                or {r["name"] for r in rows} != expected_names
                or len({r["path"] for r in rows}) != count
                or any(r.get("num_entries") != 100000 or Path(r["path"]).name != r["name"] for r in rows)):
            raise ValueError("Donor inventory lacks exact official entry coverage")
    runtimes, fingerprints, evidence = [], [], {}
    for name in controls:
        path = root / "benchmarks" / f"{name}.json"
        row = read_json(path)
        if row.get("task") != tasks[name] or row.get("campaign_sha256") != spec["content_hash"]:
            raise ValueError("Donor benchmark identity differs")
        runtime = row["runtime"]
        if (runtime.get("campaign_sha256") != spec["content_hash"] or runtime.get("world_size") != 1
                or runtime.get("batch_size") != 512 or runtime.get("workers") != 2
                or runtime.get("batchnorm") != "native" or runtime.get("precision") != POLICY
                or runtime.get("tf32") is not False or runtime.get("cudnn_benchmark") is not False
                or "A100" not in runtime.get("gpu", "") or runtime.get("gpu_memory_bytes", 0) <= 0):
            raise ValueError("Donor GPU/runtime contract differs")
        for audit, dtype in ((row["training_precision"], "torch.float16"),
                             (row["validation"]["precision"], "torch.float32")):
            observed = audit.get("observed_activations", {})
            if (audit.get("passed") is not True or audit.get("parameter_dtypes") != ["torch.float32"]
                    or audit.get("expected_activation_dtype") != dtype or not observed
                    or any(v != dtype for v in observed.values())):
                raise ValueError("Donor lacks observed precision evidence")
        if row.get("initial_parity", {}).get("passed") is not True or not row.get("input_fingerprints"):
            raise ValueError("Donor lacks parity/input fingerprint evidence")
        runtimes.append(runtime_identity(runtime))
        fingerprints.append(row["input_fingerprints"])
        evidence[f"benchmarks/{name}.json"] = file_sha256(path)
    if any(r != runtimes[0] for r in runtimes) or any(f != fingerprints[0] for f in fingerprints):
        raise ValueError("Requested donor controls differ in runtime/input stream")
    approval_path = root / "production_approval/approval.json"
    if approval_path.is_file():
        approval = checked_json(approval_path, "part_ev_screen_reviewed_recovery_approval_v1",
                                hash_function=_recovery_approval_digest)
        if (approval.get("campaign_sha256") != spec["content_hash"]
                or Path(approval["campaign"]).resolve() != root
                or approval.get("basis") != "original_memory_precision_parity_throughput_plus_bounded_amp_recovery"
                or approval.get("authorizes_training") is not True or approval.get("reviewed") is not True
                or runtime_identity(approval["runtime"]) != runtimes[0]
                or approval.get("summary", {}).get("passed") is not True
                or approval.get("initialization") != "fresh_original_seed_model_optimizer_and_grad_scaler"):
            raise ValueError("Donor production approval differs")
        for key, name in (("driver_sha256", "run_part_ev_screen_after_recovery.py"),
                          ("helper_sha256", "check_part_ev_screen_amp_recovery.py"),
                          ("launch_sha256", "launch.env")):
            if file_sha256(approval_path.parent / name) != approval[key]:
                raise ValueError("Donor approval source/launch evidence changed")
        approved_models = {r["task"]["run_id"]: r for r in approval.get("summary", {}).get("models", [])}
        if any(approved_models.get(name) != read_json(root / "benchmarks" / f"{name}.json") for name in controls):
            raise ValueError("Donor benchmark differs from reviewed approval evidence")
        recovery = Path(approval["recovery"]).resolve(strict=True)
        if not recovery.is_relative_to(root / "diagnostics"):
            raise ValueError("Recovery evidence is outside donor diagnostics")
        if file_sha256(recovery / "recovery.json") != approval["recovery_plan_sha256"]:
            raise ValueError("Donor recovery plan changed")
        required_results = {name + suffix for name in controls for suffix in (".json", ".steps.jsonl")}
        if not required_results.issubset(approval["recovery_results_sha256"]):
            raise ValueError("Donor recovery approval lacks requested-control evidence")
        for name, checksum in approval["recovery_results_sha256"].items():
            if file_sha256(child(recovery, name)) != checksum:
                raise ValueError("Donor recovery evidence changed")
        evidence["production_approval/approval.json"] = file_sha256(approval_path)
    else:
        if any(read_json(root / "benchmarks" / f"{name}.json").get("passed") is not True for name in controls):
            raise ValueError("Donor lacks passing benchmark or reviewed production approval")
    return copy.deepcopy(dict(path=str(root), campaign_sha256=spec["content_hash"], controls=controls,
        tasks={n: tasks[n] for n in controls}, recipes=spec["recipes"], source=spec["source"],
        parameters={n: spec["parameters"][tasks[n]["architecture"]] for n in controls},
        inventory=inventory, environment=spec["environment"], precision=spec["precision"],
        runtime=runtimes[0], input_fingerprints=fingerprints[0], evidence=evidence,
        data_config_sha256=spec["data_config_sha256"]))


def verify_dataset_metadata(inventory):
    """Same-location metadata evidence, NOT a claim of payload hashing."""
    for split in ("train", "val"):
        for row in inventory["splits"][split]["files"]:
            path = Path(row["path"])
            stat = path.stat()
            if path.is_symlink() or stat.st_size != row["size_bytes"] or stat.st_mtime_ns != row["mtime_ns"]:
                raise ValueError(f"Dataset metadata changed; verify contents before reuse: {path}")
