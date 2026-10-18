#!/usr/bin/env python3
"""Isolated, inference-only evaluation of the six completed Ranger models.

All model/preprocessing code and the Python environment belong to the original
frozen campaign. Only this new evaluation directory is writable. Neither the
training streams nor the test population are read. Validation is not held-out
test performance: it was already used to select the checkpoints.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
SCHEMA = "part_fullscale_full_validation_v1"
SEEDS = (101, 202, 303)
TASKS = [{"architecture": model, "seed": seed}
         for model in ("part", "part_ev") for seed in SEEDS]
FILE_EVENTS = 100_000
FILE_COUNT = 50
VALIDATION_EVENTS = 5_000_000
CLASS_EVENTS = 500_000
FILE_LABELS = {"ZJetsToNuNu": 0, "HToBB": 1, "HToCC": 2, "HToGG": 3,
               "HToWW4Q": 4, "HToWW2Q1L": 5, "ZToQQ": 6, "WToQQ": 7,
               "TTBar": 8, "TTBarLep": 9}
POLICY = {"batch_size": 512, "native_model_use_amp": True,
          "outer_autocast": False, "stored_logits_dtype": "float32",
          "tf32": False, "cudnn_benchmark": False,
          "trimmer_counter": 5, "loader": "finite_unshuffled_all_validation_entries"}
METADATA = ("campaign.json", "source.sha256", "launch.env", "training_runtime.json")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def progress(stage, **values):
    print(json.dumps({"stage": stage, **values}, sort_keys=True, allow_nan=False), flush=True)


def separate_output(output, campaign):
    output, campaign = Path(output).resolve(), Path(campaign).resolve(strict=True)
    if output == campaign or output in campaign.parents or campaign in output.parents:
        raise ValueError("Evaluation output must be separate from (not inside/above) the training campaign")
    return output, campaign


def verify_frozen_source(campaign):
    source = (campaign / "source").resolve(strict=True)
    seen = set()
    for line in (campaign / "source.sha256").read_text().splitlines():
        if not line:
            continue
        checksum, relative = line[:64], line[66:]
        if (len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum)
                or line[64:66] not in ("  ", " *")):
            raise ValueError("Invalid original source checksum manifest")
        path = source / relative
        resolved = path.resolve(strict=True)
        if path.is_symlink() or source not in resolved.parents or resolved in seen:
            raise ValueError("Unsafe or duplicate original source path")
        if file_sha256(path) != checksum:
            raise ValueError(f"Original frozen source changed: {relative}")
        seen.add(resolved)
    required = ("scripts/run_part_fullscale.py", "part_ev_fullscale/protocol.py",
                "part_ev_fullscale/model.py", "part_ev_fullscale/data.py",
                "part_ev_fullscale/metrics.py", "part_ev_fullscale/checkpoint.py",
                "particle_transformer/data/JetClass/JetClass_full.yaml")
    if not all((source / name).resolve() in seen for name in required):
        raise ValueError("Original source manifest does not cover the evaluation dependencies")
    return source


def load_driver(campaign):
    source = verify_frozen_source(campaign)
    for name, module in list(sys.modules.items()):
        if name == "part_ev_fullscale" or name.startswith("part_ev_fullscale."):
            path = getattr(module, "__file__", None)
            if path is not None and source not in Path(path).resolve().parents:
                raise RuntimeError("Already imported a different campaign's part_ev_fullscale code")
    path = source / "scripts/run_part_fullscale.py"
    module_spec = importlib.util.spec_from_file_location("_frozen_full_validation_driver", path)
    driver = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(driver)
    if Path(driver.P.__file__).resolve() != source / "part_ev_fullscale/protocol.py":
        raise RuntimeError("Validation imported the wrong frozen protocol")
    return driver


def support():
    path = Path(__file__).with_name("part_fullscale_validation_support.py")
    module_spec = importlib.util.spec_from_file_location("_full_validation_support", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def evaluator_hashes():
    return {name: file_sha256(Path(__file__).with_name(name)) for name in
            ("evaluate_part_fullscale_validation.py", "part_fullscale_validation_support.py")}


def validate_profile(spec):
    if (spec.get("comparison", "standard") != "standard"
            or spec.get("training_recipe", "original") != "original"
            or spec["tasks"] != TASKS
            or spec["recipe"]["optimizer"] != "Weaver_RAdam_plus_Lookahead"
            or spec["recipe"]["pair_bias"] != "shared"
            or spec["recipe"]["total_steps"] != 1_000_000
            or spec["recipe"]["epochs"] != 50):
        raise ValueError("This evaluator requires the six original shared-bias Ranger ParT/ParT-EV runs")


def lock_checkpoints(campaign, spec, protocol):
    """Read and cross-check existing selections; never create a donor lock/file."""
    selected = []
    runtime = read_json(campaign / "training_runtime.json")
    if (runtime["world_size"] != 1 or runtime["campaign_sha256"] != spec["content_hash"]
            or runtime["tf32"] is not False or runtime["cudnn_benchmark"] is not False):
        raise ValueError("Expected the original single-GPU, TF32-disabled training runtime")
    recipe = spec["recipe"]
    for task in spec["tasks"]:
        run = campaign / "runs" / task["architecture"] / f"seed_{task['seed']}"
        done = read_json(run / "training_complete.json")
        if (done["campaign_sha256"] != spec["content_hash"] or done["task"] != task
                or done["step"] != recipe["total_steps"]
                or done["validated_epochs"] != recipe["epochs"]
                or done["runtime"] != runtime):
            raise ValueError(f"All six runs must have completed their full training budget: {task}")
        expected = {f"epoch_{i:02d}.json" for i in range(1, recipe["epochs"] + 1)}
        if {p.name for p in (run / "validation").glob("epoch_*.json")} != expected:
            raise ValueError(f"Incomplete validation history: {task}")
        history = [read_json(run / "validation" / f"epoch_{i:02d}.json")
                   for i in range(1, recipe["epochs"] + 1)]
        for check, row in enumerate(history, 1):
            if (row["step"] != check * recipe["steps_per_epoch"]
                    or row["events"] != recipe["validation_events_per_epoch"]
                    or not math.isfinite(row["accuracy"]) or not 0 <= row["accuracy"] <= 1):
                raise ValueError(f"Invalid validation history: {task}, check {check}")
        best = max(range(len(history)), key=lambda i: history[i]["accuracy"])
        if done["best_step"] != history[best]["step"] or done["best_accuracy"] != history[best]["accuracy"]:
            raise ValueError(f"Best checkpoint is not the first maximum-validation-accuracy selection: {task}")
        path = run / "best_model_val.pt"
        sha = file_sha256(path)
        if sha != done["best_checkpoint_sha256"]:
            raise ValueError(f"Checkpoint changed since training completed: {path}")
        selected.append({"task": task, "path": str(path), "sha256": sha,
                         "step": history[best]["step"], "selected_check": best + 1,
                         "validation": history[best], "history_sha256": digest(history),
                         "completion_sha256": file_sha256(run / "training_complete.json")})
    return selected


def root_entries(path):
    import uproot
    with uproot.open(path) as root:
        if "tree" not in root or getattr(root["tree"], "classname", "") != "TTree":
            raise ValueError(f"Expected an official ROOT TTree: {path}")
        return int(root["tree"].num_entries)


def validation_inventory(spec):
    """Validate only recorded val files; train and test directories need not exist."""
    from part_ev_fullscale.data import rank_file_dict
    val = copy.deepcopy(spec["inventory"]["splits"]["val"])
    rows = val["files"]
    expected = {f"{name}_{index:03d}.root" for name in FILE_LABELS for index in range(120, 125)}
    if len(rows) != FILE_COUNT or {row["name"] for row in rows} != expected:
        raise ValueError("Validation inventory must contain exactly the 50 official validation ROOT files")
    inventory = {"splits": {"val": val}}
    rank_file_dict("val", inventory)  # Reject changed/missing/linked recorded files.
    for row in rows:
        count = root_entries(row["path"])
        if count != FILE_EVENTS or row.get("num_entries", count) != count:
            raise ValueError(f"Wrong validation entry count: {row['path']}")
        row["num_entries"] = count
        row["sha256"] = file_sha256(row["path"])
    val["files"] = sorted(rows, key=lambda r: r["name"])
    val["event_count"] = sum(row["num_entries"] for row in rows)
    if val["event_count"] != VALIDATION_EVENTS:
        raise ValueError("Validation population must be exactly 5M jets")
    return inventory


def prepare(args):
    output, campaign = separate_output(args.output, args.campaign)
    driver = load_driver(campaign)
    protocol = driver.P
    spec = protocol.load_campaign(campaign)
    validate_profile(spec)
    # Require completed training before reading validation data or running inference.
    checkpoints = lock_checkpoints(campaign, spec, protocol)
    for data_path in (Path(r["path"]).parent.resolve() for r in spec["inventory"]["splits"]["val"]["files"]):
        if output == data_path or output in data_path.parents or data_path in output.parents:
            raise ValueError("Evaluation output must not overlap the validation data directory")
    output.mkdir(parents=True, exist_ok=True)
    with protocol.file_lock(output / ".prepare.lock", blocking=False):
        if (output / "evaluation.json").exists():
            plan, _ = load_evaluation(output)
            if plan["campaign"] != str(campaign):
                raise ValueError("Existing evaluation belongs to another campaign")
            progress("already_prepared", output=str(output))
            return
        progress("validation_inventory", files=FILE_COUNT, test_read=False)
        inventory = validation_inventory(spec)
        required = VALIDATION_EVENTS * len(TASKS) * 41 + 2 * 1024**3
        if shutil.disk_usage(output).free < required:
            raise OSError(f"Need at least {required / 1024**3:.2f} GiB free for validation logits and headroom; nothing deleted")
        plan = {"schema": SCHEMA, "campaign": str(campaign), "output": str(output),
                "campaign_sha256": spec["content_hash"], "split": "validation",
                "events": VALIDATION_EVENTS, "tasks": TASKS, "checkpoints": checkpoints,
                "inventory": inventory, "inventory_sha256": digest(inventory),
                "original_metadata": {name: file_sha256(campaign / name) for name in METADATA},
                "evaluator_sha256": evaluator_hashes(), "policy": POLICY,
                "training_runtime": read_json(campaign / "training_runtime.json"),
                "selection": "existing_best_checkpoint_validation_accuracy_first_tie",
                "test_accessed": False}
        plan["content_hash"] = digest(plan)
        protocol.write_json(output / "evaluation.json", plan)
        progress("prepared_validation", output=str(output), models=len(TASKS), events=VALIDATION_EVENTS)


def load_evaluation(output):
    output = Path(output).resolve()
    plan = read_json(output / "evaluation.json")
    sha = plan.pop("content_hash")
    if digest(plan) != sha or plan["schema"] != SCHEMA:
        raise ValueError("Evaluation plan changed or is incompatible")
    plan["content_hash"] = sha
    _, campaign = separate_output(output, plan["campaign"])
    if (plan["output"] != str(output) or plan["tasks"] != TASKS or plan["split"] != "validation"
            or plan["policy"] != POLICY or plan["events"] != VALIDATION_EVENTS
            or plan["inventory_sha256"] != digest(plan["inventory"])
            or plan["evaluator_sha256"] != evaluator_hashes()):
        raise ValueError("Evaluation source, path, tasks, policy or inventory differs")
    for name, sha in plan["original_metadata"].items():
        if file_sha256(campaign / name) != sha:
            raise ValueError(f"Original campaign metadata changed: {name}")
    driver = load_driver(campaign)
    spec = driver.P.load_campaign(campaign)
    validate_profile(spec)
    if (spec["content_hash"] != plan["campaign_sha256"]
            or lock_checkpoints(campaign, spec, driver.P) != plan["checkpoints"]):
        raise ValueError("Original checkpoint selections changed")
    return plan, driver


def file_identity(plan, task_index, file_index):
    row = plan["inventory"]["splits"]["val"]["files"][file_index]
    return {"evaluation_sha256": plan["content_hash"], "task": plan["tasks"][task_index],
            "checkpoint_sha256": plan["checkpoints"][task_index]["sha256"],
            "inventory_sha256": plan["inventory_sha256"], "file_index": file_index,
            "source_sha256": row["sha256"], "source_name": row["name"],
            "entry_start": 0, "entry_stop": row["num_entries"]}


def prediction_paths(directory, index):
    return directory / f"file_{index:03d}.npz", directory / f"file_{index:03d}.json"


def validate_arrays(logits, labels, row):
    import numpy as np
    if (logits.shape != (row["num_entries"], 10) or logits.dtype != np.float32
            or labels.shape != (row["num_entries"],) or labels.dtype != np.int8
            or not np.isfinite(logits).all()):
        raise ValueError("Incomplete or nonfinite validation predictions")
    label = FILE_LABELS[row["name"].rsplit("_", 1)[0]]
    if not np.all(labels == label):
        raise ValueError(f"Wrong labels for validation file {row['name']}")


def read_prediction(directory, plan, task_index, file_index):
    import numpy as np
    npz, sidecar = prediction_paths(directory, file_index)
    record = read_json(sidecar)
    if (record.get("identity") != file_identity(plan, task_index, file_index)
            or record.get("sha256") != file_sha256(npz)):
        raise ValueError(f"Validation prediction identity/checksum differs: {npz}")
    with np.load(npz, allow_pickle=False) as saved:
        if set(saved.files) != {"logits", "labels"}:
            raise ValueError("Unexpected validation prediction fields")
        logits, labels = saved["logits"], saved["labels"]
    validate_arrays(logits, labels, plan["inventory"]["splits"]["val"]["files"][file_index])
    return logits, labels


def write_prediction(directory, plan, task_index, file_index, logits, labels, protocol):
    import numpy as np
    row = plan["inventory"]["splits"]["val"]["files"][file_index]
    validate_arrays(logits, labels, row)
    npz, sidecar = prediction_paths(directory, file_index)
    fd, temporary = tempfile.mkstemp(prefix=npz.name + ".", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez(handle, logits=logits, labels=labels)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, npz)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    protocol.write_json(sidecar, {"identity": file_identity(plan, task_index, file_index),
                                 "sha256": file_sha256(npz)})


def expected_metric_identity(plan, index):
    return {"task": plan["tasks"][index], "evaluation_sha256": plan["content_hash"],
            "checkpoint_sha256": plan["checkpoints"][index]["sha256"],
            "inventory_sha256": plan["inventory_sha256"], "split": "validation"}


def evaluate(args):
    import numpy as np
    import torch
    output = Path(args.output).resolve()
    plan, driver = load_evaluation(output)
    index = args.task_index
    if type(index) is not int or not 0 <= index < len(TASKS):
        raise ValueError("Task index must be 0..5")
    from part_ev_fullscale.checkpoint import trusted_load
    from part_ev_fullscale.data import rank_file_dict, shutdown_loader
    from part_ev_fullscale.metrics import calculate_metrics
    rank_file_dict("val", plan["inventory"])
    task = plan["tasks"][index]
    directory = output / "runs" / task["architecture"] / f"seed_{task['seed']}"
    directory.mkdir(parents=True, exist_ok=True)
    with driver.P.file_lock(directory / ".evaluation.lock", blocking=False):
        prediction_dir = directory / "predictions"
        prediction_dir.mkdir(exist_ok=True)
        rank, world, device = driver.distributed_setup()
        if rank != 0 or world != 1 or torch.cuda.get_device_name(device) != plan["training_runtime"]["gpu"]:
            raise ValueError("Validation requires one GPU of the original Tigris training type")
        base = None
        helper = support()
        for file_index, row in enumerate(plan["inventory"]["splits"]["val"]["files"]):
            if driver.STOP_REQUESTED:
                raise SystemExit(75)
            # Verify actual input content before either reusing or creating predictions.
            if file_sha256(row["path"]) != row["sha256"]:
                raise ValueError(f"Validation ROOT file changed: {row['path']}")
            npz, sidecar = prediction_paths(prediction_dir, file_index)
            if sidecar.exists():
                read_prediction(prediction_dir, plan, index, file_index)
                continue
            if base is None:
                selected = plan["checkpoints"][index]
                checkpoint = trusted_load(selected["path"])
                if (checkpoint["task"] != task or checkpoint["campaign_sha256"] != plan["campaign_sha256"]
                        or checkpoint["step"] != selected["step"]
                        or checkpoint["runtime"] != plan["training_runtime"]
                        or checkpoint["validation"]["accuracy"] != selected["validation"]["accuracy"]):
                    raise ValueError("Loaded checkpoint metadata differs from the frozen selection")
                # Same native AMP and math policy as the original validation routine.
                base, _ = driver.make_model(task["architecture"], task["seed"], device, 1)
                base.load_state_dict(checkpoint["model"], strict=True)
                base.eval()
                for module in base.modules():
                    if hasattr(module, "_counter"):
                        module._counter = POLICY["trimmer_counter"]
                del checkpoint
            logits = np.empty((row["num_entries"], 10), dtype=np.float32)
            labels = np.empty(row["num_entries"], dtype=np.int8)
            loader = helper.finite_validation_loader(plan["inventory"], row, POLICY["batch_size"])
            offset = 0
            try:
                with torch.no_grad():
                    for batch in loader:
                        if driver.STOP_REQUESTED:
                            raise SystemExit(75)
                        inputs, truth = driver.inputs_and_labels(batch, device)
                        scores = base(*inputs).float().cpu().numpy()
                        end = offset + len(truth)
                        if end > row["num_entries"]:
                            raise ValueError("Validation loader repeated entries")
                        logits[offset:end] = scores
                        labels[offset:end] = truth.cpu().numpy()
                        offset = end
            finally:
                shutdown_loader(loader)
            if offset != row["num_entries"]:
                raise ValueError("Validation loader omitted entries")
            write_prediction(prediction_dir, plan, index, file_index, logits, labels, driver.P)
            progress("validation_file_complete", **task, files=file_index + 1, total=FILE_COUNT)
        del base
        torch.cuda.empty_cache()
        # Recheck every committed file, including on resume; do not trust a metrics marker alone.
        logits = np.empty((VALIDATION_EVENTS, 10), dtype=np.float32)
        labels = np.empty(VALIDATION_EVENTS, dtype=np.int8)
        offset = 0
        prediction_hashes = []
        for file_index, row in enumerate(plan["inventory"]["splits"]["val"]["files"]):
            scores, truth = read_prediction(prediction_dir, plan, index, file_index)
            end = offset + len(truth)
            logits[offset:end], labels[offset:end] = scores, truth
            offset = end
            prediction_hashes.append(file_sha256(prediction_paths(prediction_dir, file_index)[1]))
        if offset != VALIDATION_EVENTS or not np.all(np.bincount(labels, minlength=10) == CLASS_EVENTS):
            raise ValueError("Validation coverage must be exactly 5M jets, 500k per class")
        metrics = calculate_metrics(logits, labels)
        record = {**expected_metric_identity(plan, index), "metrics": metrics,
                  "selected_check": plan["checkpoints"][index]["selected_check"],
                  "prediction_sidecar_sha256": prediction_hashes, "policy": POLICY}
        record["content_hash"] = digest(record)
        driver.P.write_json(directory / "metrics.json", record)
        progress("full_validation_complete", **task, accuracy=metrics["accuracy"],
                 macro_auc_ovo=metrics["macro_auc_ovo"])


def report(args):
    output = Path(args.output).resolve()
    plan, driver = load_evaluation(output)
    with driver.P.file_lock(output / ".report.lock", blocking=False):
        records = []
        for index, task in enumerate(plan["tasks"]):
            directory = output / "runs" / task["architecture"] / f"seed_{task['seed']}"
            record = read_json(directory / "metrics.json")
            content_hash = record.pop("content_hash", None)
            if digest(record) != content_hash:
                raise ValueError(f"Validation metric record changed: {task}")
            record["content_hash"] = content_hash
            if any(record.get(key) != value for key, value in expected_metric_identity(plan, index).items()):
                raise ValueError(f"Mismatched validation result: {task}")
            if (record.get("policy") != POLICY
                    or record.get("selected_check") != plan["checkpoints"][index]["selected_check"]):
                raise ValueError("Mixed inference precision policies or checkpoint selections")
            hashes = []
            for file_index in range(len(plan["inventory"]["splits"]["val"]["files"])):
                read_prediction(directory / "predictions", plan, index, file_index)
                hashes.append(file_sha256(prediction_paths(directory / "predictions", file_index)[1]))
            if record.get("prediction_sidecar_sha256") != hashes:
                raise ValueError("Result does not describe the committed validation predictions")
            records.append(record)
        helper = support()
        result = helper.build_report(records)
        result.update(campaign=plan["campaign"], campaign_sha256=plan["campaign_sha256"],
                      policy=POLICY, selection=plan["selection"], checkpoints=plan["checkpoints"],
                      per_seed_results=records, test_accessed=False)
        driver.P.write_json(output / "reports/validation_report.json", result)
        text = helper.markdown_report(result)
        path = output / "reports/validation_report.md"
        fd, temporary = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        progress("validation_report_complete", report=str(path))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "evaluate", "report"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--task-index", type=int)
    args = parser.parse_args(argv)
    if args.phase == "prepare" and args.campaign is None:
        parser.error("prepare requires --campaign")
    if args.phase != "prepare" and args.campaign is not None:
        parser.error("--campaign is only supplied at prepare")
    if args.phase == "evaluate" and args.task_index is None:
        parser.error("evaluate requires --task-index")
    if args.phase != "evaluate" and args.task_index is not None:
        parser.error("--task-index is only valid for evaluate")
    globals()[args.phase](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
