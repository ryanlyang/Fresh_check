"""Control completion/evaluation validation; never writes to the donor."""
from __future__ import annotations
from pathlib import Path
import math
import numpy as np

from part_ev_fullscale.protocol import read_json, write_json, file_sha256, file_lock, CLASS_NAMES
from .contracts import checked_json, donor_contract, hashed, runtime_identity


def selected_training(run, task, campaign_hash, recipe, expected_runtime):
    run = Path(run)
    complete = read_json(run / "training_complete.json")
    if (complete.get("task") != task or complete.get("campaign_sha256") != campaign_hash
            or complete.get("step") != recipe["total_steps"]
            or complete.get("validated_checks") != recipe["epochs"]
            or runtime_identity(complete.get("runtime", {})) != expected_runtime):
        raise ValueError("Training completion identity/budget/runtime differs")
    checks = []
    for i in range(1, recipe["epochs"] + 1):
        check = read_json(run / "validation" / f"check_{i:02d}.json")
        if (check.get("check") != i or check.get("step") != i * recipe["steps_per_epoch"]
                or check.get("events") != recipe["validation_events_per_epoch"]
                or not isinstance(check.get("accuracy"), (int, float))
                or not math.isfinite(check["accuracy"]) or not 0 <= check["accuracy"] <= 1):
            raise ValueError("Incomplete/invalid checkpoint validation history")
        checks.append(check)
    best = max(checks, key=lambda c: c["accuracy"])
    if complete.get("best_check") != best["check"] or complete.get("best_accuracy") != best["accuracy"]:
        raise ValueError("Selection differs from highest accuracy / first tie")
    checkpoint = run / "best_model_val.pt"
    checksum = file_sha256(checkpoint)
    if checksum != complete.get("best_checkpoint_sha256"):
        raise ValueError("Selected checkpoint changed")
    optimizer = read_json(run / "optimizer.json")
    expected_name = "Ranger" if recipe["optimizer"] == "Weaver_RAdam_plus_Lookahead" else "AdamW"
    if optimizer.get("optimizer") != expected_name or not optimizer.get("groups"):
        raise ValueError("Missing/mismatched optimizer audit")
    names = []
    for group in optimizer["groups"]:
        expected_decay = (0. if group.get("name") == "no_decay" else recipe["weight_decay"])
        if (group.get("betas") != recipe["betas"] or group.get("eps") != recipe["eps"]
                or group.get("lr") != recipe["start_lr"] or group.get("weight_decay") != expected_decay
                or group.get("parameter_tensors") != len(group.get("parameter_names", []))
                or type(group.get("parameters")) is not int or group["parameters"] <= 0):
            raise ValueError("Optimizer group audit differs from recipe")
        names.extend(group["parameter_names"])
    if (not names or len(set(names)) != len(names)
            or optimizer.get("parameters") != sum(g["parameters"] for g in optimizer["groups"])):
        raise ValueError("Optimizer parameter coverage differs")
    return dict(checkpoint=str(checkpoint.resolve()), checkpoint_sha256=checksum, best_check=best["check"],
                checks=checks, complete=complete, optimizer=optimizer,
                completion_sha256=file_sha256(run / "training_complete.json"),
                optimizer_sha256=file_sha256(run / "optimizer.json"))


def validate_predictions(run, task, campaign_hash, checkpoint_hash, files, *, population=5000000):
    """Stream verification; file/entry sidecars define identity, not NPZ names."""
    run = Path(run)
    directory = run / "full_validation"
    metric_path = directory / "metrics.json"
    result = read_json(metric_path)
    if (result.get("task") != task or result.get("campaign_sha256") != campaign_hash
            or result.get("checkpoint_sha256") != checkpoint_hash):
        raise ValueError("Full-validation metric identity differs")
    expected = {r["name"]: r for r in files}
    records, seen, labels_by_file = [], set(), {}
    correct, total, counts = 0, 0, np.zeros(10, np.int64)
    entries = []
    for sidecar in sorted(directory.glob("file_*.json")):
        row = read_json(sidecar)
        contract = row.get("contract", {})
        name = contract.get("file", {}).get("name")
        if name not in expected or name in seen:
            raise ValueError("Unexpected/duplicated prediction file identity")
        file_row = expected[name]
        if (contract.get("campaign_sha256") != campaign_hash
                or contract.get("checkpoint_sha256") != checkpoint_hash
                or contract.get("file") != file_row or contract.get("entry_start") != 0
                or contract.get("entry_stop") != file_row["num_entries"]):
            raise ValueError("Prediction file/entry contract differs")
        npz = sidecar.with_suffix(".npz")
        if file_sha256(npz) != row.get("sha256"):
            raise ValueError("Prediction checksum differs")
        with np.load(npz, allow_pickle=False) as arrays:
            logits, labels = arrays["logits"], arrays["labels"]
            if (logits.shape != (file_row["num_entries"], 10) or logits.dtype != np.float32
                    or labels.shape != (len(logits),) or labels.dtype.kind not in "iu"
                    or not np.isfinite(logits).all() or (labels < 0).any() or (labels >= 10).any()):
                raise ValueError("Invalid prediction shape/dtype/values")
            correct += int((logits.argmax(1) == labels).sum())
            total += len(labels)
            counts += np.bincount(labels.astype(np.int64), minlength=10)
            import hashlib
            labels_by_file[name] = hashlib.sha256(labels.astype("<i8").tobytes()).hexdigest()
        records.append(row)
        entries.append(dict(name=name, path=str(npz.resolve()), sha256=row["sha256"],
                            sidecar_sha256=file_sha256(sidecar), entries=file_row["num_entries"]))
        seen.add(name)
    if (seen != set(expected) or total != population or population % 10
            or not np.array_equal(counts, np.full(10, population // 10))):
        raise ValueError("Incomplete full-validation file/entry/class coverage")
    if result.get("identity_manifest") != records:
        raise ValueError("Metrics prediction manifest differs")
    metrics = result["metrics"]
    if (metrics.get("events") != total or metrics.get("accuracy") != correct / total
            or metrics.get("class_counts") != dict(zip(CLASS_NAMES, map(int, counts)))):
        raise ValueError("Metrics disagree with committed prediction counts/accuracy")
    for key in ("loss", "macro_auc_ovo", "macro_auc_ovr"):
        if not isinstance(metrics.get(key), (float, int)) or not math.isfinite(metrics[key]):
            raise ValueError("Missing/nonfinite evaluation metric")
    return dict(metrics=metrics, metrics_sha256=file_sha256(metric_path), predictions=entries,
                labels_by_file=labels_by_file, population=total)


def inspect_control(contract, name):
    root = Path(contract["path"])
    task = contract["tasks"][name]
    recipe = contract["recipes"][task["recipe"]]
    run = root / "runs" / name / f"seed_{task['seed']}"
    base = dict(run_id=name, task=task, path=str(run), donor_campaign_sha256=contract["campaign_sha256"])
    if not (run / "training_complete.json").is_file():
        return dict(base, status="pending_training")
    try:
        selected = selected_training(run, task, contract["campaign_sha256"], recipe, contract["runtime"])
        if selected["optimizer"]["parameters"] != contract["parameters"][name]:
            raise ValueError("Completed donor parameter count differs")
        base["training"] = selected
        lock_path = root / "selection/locked_checkpoints.json"
        if not lock_path.is_file() or not (run / "full_validation/metrics.json").is_file():
            return dict(base, status="pending_evaluation")
        lock = read_json(lock_path)
        match = [r for r in lock.get("checkpoints", []) if r.get("task") == task]
        if (lock.get("campaign_sha256") != contract["campaign_sha256"] or len(match) != 1
                or match[0].get("sha256") != selected["checkpoint_sha256"]
                or Path(match[0]["checkpoint"]).resolve() != Path(selected["checkpoint"]).resolve()):
            raise ValueError("Donor finalist lock differs")
        evaluation = validate_predictions(run, task, contract["campaign_sha256"], selected["checkpoint_sha256"],
            contract["inventory"]["splits"]["val"]["files"], population=recipe["evaluation_population"])
        return dict(base, status="verified", evaluation=evaluation)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return dict(base, status="incompatible", reason=str(exc))


def refresh_controls(output, expected_contract):
    output = Path(output).resolve()
    donor = Path(expected_contract["path"]).resolve()
    if output == donor or output.is_relative_to(donor) or donor.is_relative_to(output):
        raise ValueError("New campaign and donor must be separate directories")
    current = donor_contract(donor, expected_contract["controls"])
    if current != expected_contract:
        raise ValueError("Frozen donor contract changed; cannot refresh status")
    with file_lock(output / ".controls.lock"):
        rows = {name: inspect_control(current, name) for name in current["controls"]}
        path = output / "controls.json"
        if path.exists():
            old = checked_json(path, "part_reused_controls_v1")
            for name, before in old["controls"].items():
                if before.get("status") == "verified" and rows.get(name) != before:
                    raise ValueError(f"Previously verified control changed: {name}")
        record = hashed(dict(schema="part_reused_controls_v1", donor_campaign_sha256=current["campaign_sha256"],
            controls=rows, ready=all(r["status"] == "verified" for r in rows.values())))
        write_json(output / "control_validation_history" / f"{record['content_hash']}.json", record)
        write_json(path, record)
        if any(r["status"] == "incompatible" for r in rows.values()):
            raise ValueError("Incompatible control artifacts; inspect controls.json")
        return record
