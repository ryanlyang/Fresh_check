"""Fail-closed donor and artifact validation. Never write into the donor."""
from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys
import numpy as np
from part_ev_screen import protocol as S
from part_ev_fullscale.protocol import digest, file_sha256, read_json, write_json
from part_ev_fullscale.checkpoint import trusted_load

CONTROL_IDS = ("PART_RANGER", "EV_RANGER")
DEFAULT_DONOR = "/home/ryreu/atlas/Fresh_check/checkpoints/part_ev_screen/context_gate_ampfix_20261016T215608Z"
ROOT = Path(__file__).resolve().parents[1]


def checked_hash(payload, field="content_hash"):
    if payload.get(field) != digest({k: v for k, v in payload.items() if k != field}):
        raise ValueError("Artifact content hash mismatch")
    return payload


def runtime_identity(runtime):
    return {key: value for key, value in runtime.items() if key != "campaign_sha256"}


def donor_contract(donor, *, check_environment=True):
    donor = Path(donor).resolve(strict=True)
    spec = checked_hash(read_json(donor / "campaign.json"))
    if (spec["schema"] != S.SCHEMA or spec["tasks"] != S.TASKS
            or spec["recipes"] != S.recipes() or spec["model"] != S.CONFIG
            or spec["parameters"] != S.COUNTS or spec["precision"] != S.PRECISION_POLICY
            or spec["data_config_sha256"] != file_sha256(S.DATA_CONFIG)):
        raise ValueError("Donor screen/data/precision protocol differs")
    if spec["source"] != S.source_manifest(ROOT):
        raise ValueError("Shared trainer/model/data source differs from frozen donor")
    for name, checksum in spec["source"].items():
        path = (donor / "source" / name).resolve(strict=True)
        if not path.is_relative_to(donor / "source") or file_sha256(path) != checksum:
            raise ValueError("Donor frozen source changed")
    checked_hash(spec["inventory"], "inventory_sha256")
    if check_environment and spec["environment"] != S.runtime_environment():
        raise ValueError("Use the donor's unchanged dedicated SPORC environment")
    if read_json(donor / "launch.json") != spec["launch"]:
        raise ValueError("Donor launch contract changed")
    approval = donor / "production_approval"
    if (approval / "approval.json").exists():
        # Run the existing approval verifier in a clean process: its frozen
        # imports must not share the new campaign's Python module namespace.
        for name in ("run_part_ev_screen_after_recovery.py", "check_part_ev_screen_amp_recovery.py"):
            if file_sha256(approval / name) != file_sha256(ROOT / "scripts" / name):
                raise ValueError("Unrecognized donor approval verifier")
        result = subprocess.run([sys.executable, str(approval / "run_part_ev_screen_after_recovery.py"),
            "--phase", "review", "--approval", str(approval)], capture_output=True, text=True, check=True)
        summary = json.loads(result.stdout)
        admission = dict(path=str(approval / "approval.json"), sha256=file_sha256(approval / "approval.json"))
    else:
        summary = S.benchmark_summary(donor, spec)
        admission = dict(benchmark_hashes={task["run_id"]: file_sha256(
            donor / "benchmarks" / (task["run_id"] + ".json")) for task in spec["tasks"]})
    runtime = summary["runtime"]
    if summary.get("passed") is not True or runtime["campaign_sha256"] != spec["content_hash"]:
        raise ValueError("Donor has no valid training admission")
    return spec, dict(path=str(donor), campaign_sha256=spec["content_hash"],
                     runtime=runtime_identity(runtime), admission=admission)


def verify_data_contents(inventory):
    """Compare actual ROOT bytes to the upload/extraction SHA-256 receipts."""
    root = Path(inventory["data_root"])
    files = {str(Path(row["path"]).relative_to(root)).replace("\\", "/"): row
             for split in ("train", "val") for row in inventory["splits"][split]["files"]}
    members, receipts = {}, {}
    names = [f"JetClass_Pythia_train_100M_part{i}.tar.json" for i in range(10)]
    names += ["JetClass_Pythia_val_5M.tar.json"]
    for name in names:
        path = root / "receipts" / name
        receipt = read_json(path)
        if (receipt.get("contract") != "jetclass_archive_stage_v1"
                or receipt.get("archive") != name.removesuffix(".json")):
            raise ValueError("Unrecognized staging receipt")
        receipts[name] = file_sha256(path)
        for member in receipt["members"]:
            relative = member["path"]
            if relative not in files or relative in members:
                raise ValueError("Unexpected/duplicate ROOT entry in staging receipts")
            row = files[relative]
            if member["size"] != row["size_bytes"] or file_sha256(row["path"]) != member["sha256"]:
                raise ValueError(f"ROOT content differs from verified staging receipt: {relative}")
            stat = Path(row["path"]).stat()
            if (stat.st_size, stat.st_mtime_ns) != (row["size_bytes"], row["mtime_ns"]):
                raise ValueError("Dataset changed while verifying bytes")
            members[relative] = member["sha256"]
    if members.keys() != files.keys():
        raise ValueError("Staging receipts do not cover the full train/validation population")
    return dict(receipt_sha256=receipts, file_sha256=members,
                provenance="staged_bytes_verified_now_and_unchanged_donor_inventory")


def validate_training(root, spec, task, runtime):
    run = root / "runs" / task["run_id"] / f"seed_{task['seed']}"
    complete = read_json(run / "training_complete.json")
    recipe = spec["recipes"][task["recipe"]]
    if (complete["task"] != task or complete["campaign_sha256"] != spec["content_hash"]
            or complete["step"] != recipe["total_steps"]
            or complete["validated_checks"] != recipe["epochs"]
            or runtime_identity(complete["runtime"]) != runtime
            or complete["runtime"]["campaign_sha256"] != spec["content_hash"]):
        raise ValueError("Training completion/recipe/runtime mismatch")
    checks = [read_json(run / "validation" / f"check_{i:02d}.json") for i in range(1, 11)]
    for i, row in enumerate(checks, 1):
        if (row["check"] != i or row["step"] != i * recipe["steps_per_epoch"]
                or row["events"] != recipe["validation_events_per_epoch"]
                or not math.isfinite(row["accuracy"]) or not 0 <= row["accuracy"] <= 1):
            raise ValueError("Invalid/incomplete validation check sequence")
    best = max(checks, key=lambda row: row["accuracy"])  # Python retains first tie.
    if complete["best_check"] != best["check"] or complete["best_accuracy"] != best["accuracy"]:
        raise ValueError("Selected check is not the first highest-accuracy check")
    checkpoint = run / "best_model_val.pt"
    checksum = file_sha256(checkpoint)
    if checksum != complete["best_checkpoint_sha256"]:
        raise ValueError("Selected checkpoint changed")
    saved = trusted_load(checkpoint)
    if (saved["task"] != task or saved["campaign_sha256"] != spec["content_hash"]
            or saved["runtime"] != complete["runtime"] or saved["validation"] != best):
        raise ValueError("Selected checkpoint metadata differs from validated training")
    optimizer = read_json(run / "optimizer.json")
    if (optimizer["optimizer"] != "Ranger"
            or optimizer["parameters"] != spec["parameters"][task["architecture"]]):
        raise ValueError("Control optimizer/parameter count differs")
    for group in optimizer["groups"]:
        for field in ("weight_decay", "betas", "eps"):
            if group[field] != recipe[field]:
                raise ValueError("Control optimizer settings differ")
    hashes = {str(path.relative_to(root)): file_sha256(path) for path in
              [root / "campaign.json", run / "training_complete.json",
               run / "optimizer.json", checkpoint, *sorted((run / "validation").glob("check_*.json"))]}
    return dict(task=task, checkpoint_sha256=checksum, best_check=best["check"],
                training=complete, validation_checks=checks, artifact_sha256=hashes)


def validate_evaluation(root, spec, task, training, *, recompute=None):
    """Verify sidecars, full coverage, logits and labels; optionally recompute metrics."""
    run = root / "runs" / task["run_id"] / f"seed_{task['seed']}"
    selected = read_json(root / "selection/locked_checkpoints.json")
    if selected["campaign_sha256"] != spec["content_hash"]:
        raise ValueError("Wrong finalist lock campaign")
    matching = [row for row in selected["checkpoints"] if row["task"] == task]
    if (len(matching) != 1 or matching[0]["sha256"] != training["checkpoint_sha256"]
            or Path(matching[0]["checkpoint"]).resolve() != (run / "best_model_val.pt").resolve()):
        raise ValueError("Wrong finalist lock/checkpoint")
    directory = run / "full_validation"
    row = read_json(directory / "metrics.json")
    if (row["task"] != task or row["campaign_sha256"] != spec["content_hash"]
            or row["checkpoint_sha256"] != training["checkpoint_sha256"]):
        raise ValueError("Metrics identity mismatch")
    files = sorted(spec["inventory"]["splits"]["val"]["files"], key=lambda r: r["name"])
    if len(row["identity_manifest"]) != len(files):
        raise ValueError("Incomplete prediction manifest")
    predictions, labels = [], []
    hashes = dict(training["artifact_sha256"])
    for i, (record, file) in enumerate(zip(row["identity_manifest"], files)):
        path = directory / f"file_{i:03d}.npz"
        sidecar = path.with_suffix(".json")
        expected = dict(campaign_sha256=spec["content_hash"], checkpoint_sha256=training["checkpoint_sha256"],
                        file=file, entry_start=0, entry_stop=file["num_entries"])
        if (record["contract"] != expected or read_json(sidecar) != record
                or record["sha256"] != file_sha256(path)):
            raise ValueError("Prediction sidecar/hash/entry coverage mismatch")
        with np.load(path, allow_pickle=False) as saved:
            y, logits = saved["labels"], saved["logits"]
            if (y.shape != (file["num_entries"],) or not np.issubdtype(y.dtype, np.integer)
                    or np.any((y < 0) | (y >= 10)) or logits.shape != (len(y), 10)
                    or not np.isfinite(logits).all()):
                raise ValueError("Malformed validation predictions")
            labels.append(y)
            if recompute is not None:
                predictions.append(logits)
        hashes[str(path.relative_to(root))] = record["sha256"]
        hashes[str(sidecar.relative_to(root))] = file_sha256(sidecar)
    truth = np.concatenate(labels)
    population = spec["recipes"][task["recipe"]]["evaluation_population"]
    if (population != 5000000 or len(truth) != population or row["metrics"]["events"] != population
            or not np.array_equal(np.bincount(truth, minlength=10), np.full(10, population//10))):
        raise ValueError("Expected exactly 5M validation events, 500k per class")
    if recompute is not None:
        metrics = recompute(np.concatenate(predictions), truth, rejection=True)
        if metrics != row["metrics"]:
            raise ValueError("Stored metrics differ from recomputed prediction metrics")
    for path in (root / "selection/locked_checkpoints.json", directory / "metrics.json"):
        hashes[str(path.relative_to(root))] = file_sha256(path)
    return dict(training, status="complete", metrics=row["metrics"], artifact_sha256=hashes), truth


def registry(donor, spec, contract, *, require_complete=False, recompute=None):
    root = Path(donor)
    records, labels = {}, {}
    for name in CONTROL_IDS:
        task = next(task for task in spec["tasks"] if task["run_id"] == name)
        run = root / "runs" / name / "seed_101"
        if not (run / "training_complete.json").exists():
            row = dict(status="pending_training", task=task)
        else:
            row = validate_training(root, spec, task, contract["runtime"])
            if not (run / "full_validation/metrics.json").exists():
                row["status"] = "pending_evaluation"
            else:
                row, labels[name] = validate_evaluation(root, spec, task, row, recompute=recompute)
        records[name] = row
    if require_complete and any(row["status"] != "complete" for row in records.values()):
        raise ValueError("Donor controls are pending; final comparison blocked (do not retrain them)")
    result = dict(schema="ev_message_controls_v1", donor=contract, controls=records)
    result["content_hash"] = digest(result)
    return result, labels
