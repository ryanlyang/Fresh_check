#!/usr/bin/env python3
"""Supplemental, inference-only evaluation of the twelve locked offline RPT models.

Run this standalone driver with the ORIGINAL campaign's pinned --source-root.
No original campaign artifact is changed. ROOT files are streamed, trees are
shared across checkpoints, and only logits are persisted. Commands are also
usable without Slurm; see RELATIONAL_PARTICLE_TRANSFORMER_FULL_TEST.md.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import gzip
import hashlib
import importlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time


VERSION = "relational_part_offline_full_test_v1"
MODELS = (
    "OFF_RPT_BASE", "OFF_RPT_BASE_EDGEVALUE",
    "OFF_RPT_SELECTED_LAYERWISE", "OFF_RPT_SELECTED_EDGEVALUE",
)
SEEDS = (101, 202, 303)
PREFIXES = (
    "ZJetsToNuNu", "HToBB", "HToCC", "HToGG", "HToWW4Q",
    "HToWW2Q1L", "ZToQQ", "WToQQ", "TTBar", "TTBarLep",
)
CLASS_NAMES = ("QCD", "Hbb", "Hcc", "Hgg", "H4q", "Hqql", "Zqq", "Wqq", "Tbqq", "Tbl")
TARGETS = (0.30, 0.50, 0.75, 0.99, 0.995)
NAMES = {f"{prefix}_{i:03d}.root" for prefix in PREFIXES for i in range(100, 120)}


def digest(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"Absent or symlink artifact: {path}")
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def hashed(value):
    return {**value, "content_hash": hashlib.sha256(canonical(value)).hexdigest()}


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    unbound = {k: v for k, v in value.items() if k != "content_hash"}
    if value.get("content_hash") != hashed(unbound)["content_hash"]:
        raise ValueError(f"JSON authentication failed: {path}")
    return value


@contextmanager
def atomic_file(path, binary=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb" if binary else "w", **({} if binary else {"encoding": "utf-8"})) as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_json(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise FileExistsError(f"Refusing to replace a different artifact: {path}")
        return
    with atomic_file(path) as stream:
        stream.write(canonical(value) + b"\n")


@contextmanager
def exclusive(path):
    """Kernel-released locks survive neither failures nor Slurm timeouts."""
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def progress(stage, **fields):
    print(json.dumps({"stage": stage, **fields}, sort_keys=True), flush=True)


def inspect_archive(path):
    with tarfile.open(path, "r:") as archive:
        members = archive.getmembers()
    files = [m for m in members if m.isfile()]
    names = [m.name for m in files]
    expected = {f"test_20M/{name}" for name in NAMES}
    if set(names) != expected or len(names) != len(expected):
        raise ValueError("Archive must contain exactly the 200 official test_20M ROOT filenames")
    for member in members:
        if member.isdir() and member.name.rstrip("/") == "test_20M":
            continue
        if not member.isfile() or member.name not in expected:
            raise ValueError(f"Unsafe/unexpected archive member: {member.name}")
    return {"files": len(files), "unpacked_bytes": sum(m.size for m in files), "names": sorted(names)}


def source_snapshot(source):
    """Track actual runtime bytes, not unrelated commits or generated pyc files."""
    paths = []
    for directory in ("teacher_logit_reco", "jetclass_fresh"):
        paths.extend(p for p in (source / directory).rglob("*") if p.suffix in {".py", ".cpp"})
    paths.append(source / "jetclass_fixed_hlt.py")
    return {p.relative_to(source).as_posix(): digest(p) for p in sorted(paths)}


def activate_source(source):
    source = Path(source).resolve()
    sys.path.insert(0, str(source))
    import teacher_logit_reco
    if not Path(teacher_logit_reco.__file__).resolve().is_relative_to(source):
        raise RuntimeError("Another repository's teacher_logit_reco is already imported")


def environment():
    import numpy as np
    import torch
    import awkward
    import uproot
    weaver = importlib.import_module("weaver.nn.model.ParticleTransformer")
    return {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
            "awkward": awkward.__version__, "uproot": uproot.__version__,
            "weaver_particle_transformer_sha256": digest(weaver.__file__)}


def parent_artifacts(parent):
    """Portable relative paths: archived metadata can still contain RC paths."""
    campaign = read_json(parent / "campaign_spec.json")
    lock = read_json(parent / "selection/locked_finalists.json")
    if campaign["contract"] != "relational_part_offline_transfer_campaign_v1":
        raise ValueError("Expected the offline transfer campaign")
    if campaign.get("class_order") != list(CLASS_NAMES) or campaign.get("seeds") != list(SEEDS):
        raise ValueError("Original class order or seeds differ")
    if lock["contract"] != "relational_part_offline_transfer_final_lock_v1" or lock["campaign_sha256"] != campaign["content_hash"]:
        raise ValueError("Original offline lock/campaign mismatch")
    wanted = [(name, seed) for name in MODELS for seed in SEEDS]
    locked = {(row["run_id"], row["seed"]): row for row in lock["locked_rows"]}
    if len(lock["locked_rows"]) != 12 or set(locked) != set(wanted):
        raise ValueError("Original lock must cover exactly four models by three seeds")
    relative = ["campaign_spec.json", "selection/locked_finalists.json",
                "inputs/relation_normalization.json", "inputs/region_normalization.json",
                "inputs/split_manifest.json.gz", "registry/screening_registry.json",
                "backend/backend_manifest.json"]
    relation = read_json(parent / relative[2])
    region = read_json(parent / relative[3])
    if digest(parent / "inputs/split_manifest.json.gz") != campaign["split_manifest"]["file_sha256"]:
        raise ValueError("Original split manifest bytes differ from the campaign")
    backend = read_json(parent / "backend/backend_manifest.json")
    binary = "backend/" + backend["binary_filename"]
    if Path(binary).name != backend["binary_filename"] or digest(parent / binary) != backend["binary_sha256"]:
        raise ValueError("Original compiled tree backend differs")
    relative.append(binary)
    tasks = []
    for name, seed in wanted:
        row = locked[(name, seed)]
        checkpoint = f"runs/{name}/seed_{seed}/best_model_val.pt"
        regpath = f"runs/{name}/seed_{seed}/checkpoint_registration.json"
        contractpath = f"registry/model_contracts/{name}.json"
        registration, contract = read_json(parent / regpath), read_json(parent / contractpath)
        expected_families = [] if name in MODELS[:2] else ["PT", "TRACK", "REGION"]
        if (registration["content_hash"] != row["checkpoint_registration_sha256"]
                or contract["content_hash"] != row["model_contract_sha256"]
                or registration["model_contract_sha256"] != contract["content_hash"]
                or registration.get("offline_tagger_inference") is not True
                or contract.get("run_id") != name
                or contract.get("relation_families") != expected_families
                or contract["campaign_sha256"] != campaign["content_hash"]
                or contract["relation_normalization_sha256"] != relation["content_hash"]
                or contract["region_normalization_sha256"] != region["content_hash"]
                or registration["checkpoint_sha256"] != row["checkpoint_sha256"]
                or digest(parent / checkpoint) != row["checkpoint_sha256"]):
            raise ValueError(f"Original checkpoint lineage mismatch: {name}/{seed}")
        relative.extend([checkpoint, regpath, contractpath])
        tasks.append({"run_id": name, "seed": seed, "checkpoint": checkpoint,
                      "checkpoint_sha256": row["checkpoint_sha256"],
                      "model_contract_sha256": contract["content_hash"],
                      "relation_families": contract["relation_families"]})
    return campaign, tasks, {name: digest(parent / name) for name in sorted(set(relative))}


def overlap_audit(split_path, test_names):
    with gzip.open(split_path, "rt", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if int(manifest["max_constits"]) != 128:
        raise ValueError("Expected original 128-particle input convention")
    splits = manifest["splits"]
    for name in ("model_train", "model_val", "stack_val", "final_test"):
        if not splits.get(name):
            raise ValueError(f"Original manifest lacks {name}")
    result = {}
    for split, identities in splits.items():
        # Standard JetClass filenames retain identity after directory relocation.
        used = {PurePosixPath(row["file"].replace("\\", "/")).name for row in identities}
        common = sorted(used & set(test_names))
        result[split] = {"original_event_count": len(identities), "overlapping_filenames": common}
        if common:
            raise ValueError(f"Test files overlap original {split}: {common}")
    return {"identity_rule": "official_ROOT_basename_plus_entry; whole-file disjointness",
            "all_original_splits_file_disjoint": True, "splits": result}


def file_assignment(files, jobs):
    if not 1 <= jobs <= len(files):
        raise ValueError("Job count must be between 1 and file count")
    # File order interleaves classes; each task handles several classes.
    return [list(range(i * len(files) // jobs, (i + 1) * len(files) // jobs)) for i in range(jobs)]


def prepare(args):
    import uproot
    parent, source, output, data = (p.resolve() for p in (args.parent, args.source_root, args.output, args.data_dir))
    if output == parent or output.is_relative_to(parent):
        raise ValueError("Supplemental output must be separate from the original campaign")
    campaign, tasks, artifacts = parent_artifacts(parent)
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if commit != campaign["source"]["commit"]:
        raise ValueError("Use the original offline campaign's pinned source commit")
    changed = subprocess.check_output(["git", "-C", str(source), "diff", "--name-only", "HEAD", "--", "*.py", "*.cpp"], text=True)
    if changed.strip():
        raise ValueError("Pinned source has modified Python/C++ files")
    untracked = subprocess.check_output(["git", "-C", str(source), "ls-files", "--others", "--exclude-standard"], text=True)
    if any(Path(name).suffix in {".py", ".cpp"} for name in untracked.splitlines()):
        raise ValueError("Pinned source has untracked Python/C++ files")
    activate_source(source)
    env = environment()
    paths = sorted(data.glob("*.root"))
    if {p.name for p in paths} != NAMES or len(paths) != 200:
        raise ValueError("Data directory must contain exactly the 200 official ROOT files (100--119)")
    audit = overlap_audit(parent / "inputs/split_manifest.json.gz", NAMES)
    files = []
    for path in sorted(paths, key=lambda p: (p.stem.rsplit("_", 1)[1], PREFIXES.index(p.stem.rsplit("_", 1)[0]))):
        label = PREFIXES.index(path.stem.rsplit("_", 1)[0])
        with uproot.open(path) as root:
            entries = int(root["tree"].num_entries)
        if entries != 100_000:
            raise ValueError(f"Expected 100,000 events in {path}, found {entries}")
        file_digest = digest(path)
        if any(row["sha256"] == file_digest for row in files):
            raise ValueError(f"Duplicate ROOT bytes under different test filenames: {path}")
        files.append({"name": path.name, "label": label, "entries": entries,
                      "sha256": file_digest, "bytes": path.stat().st_size})
        progress("authenticate_test_files", completed=len(files), total=200)
    plan = hashed({"contract": VERSION, "schema_version": 1,
                   "parent": str(parent), "source_root": str(source), "data_dir": str(data),
                   "parent_campaign_sha256": campaign["content_hash"], "parent_artifacts": artifacts,
                   "source_commit": commit, "source_files": source_snapshot(source),
                   "driver_sha256": digest(__file__), "environment": env,
                   "tasks": tasks, "files": files, "assignments": file_assignment(files, args.jobs),
                   "event_count": 20_000_000, "class_order": list(CLASS_NAMES),
                   "batch_size": 64, "chunk_size": 10_000, "input_view": "offline",
                   "precision": "FP32; original model AMP disabled; no outer autocast",
                   "transient_trimmer_policy": "restore freshly loaded _counter at every 10000-event chunk; flags/weights unchanged",
                   "overlap_audit": audit, "performance_gate": False,
                   "checkpoints_and_normalizers_refitted": False,
                   "rejection_targets": list(TARGETS),
                   "metric_policy": {"ROC": "original logit_s-logit_QCD, ceil rank, >= threshold; global not shard-averaged",
                                     "zero_background": "null rejection with infinite-empirical flag and conditional Wilson bound",
                                     "rejection_interval": "95% Wilson binomial interval conditional on empirical signal threshold; threshold uncertainty excluded",
                                     "accuracy_bootstrap": "class-stratified paired event resampling via sufficient-category multinomial counts",
                                     "bootstrap_seed": 917301, "bootstrap_replicates": 10000,
                                     "bootstrap_rng": "numpy.Generator(PCG64); same seed reset for each comparison",
                                     "bootstrap_quantiles": "0.025,0.975; numpy method=linear",
                                     "bootstrap_note": "same sampling distribution as event bootstrap, not same draws as original campaign",
                                     "ECE": "original float64 15-bin top-label ECE; [left,right), final bin right-inclusive; empty=0"}})
    write_json(output / "evaluation_plan.json", plan)
    progress("prepared", output=str(output), gpu_tasks=len(plan["assignments"]), models=12)


def load_plan(output):
    plan = read_json(output / "evaluation_plan.json")
    if plan["contract"] != VERSION or digest(__file__) != plan["driver_sha256"]:
        raise ValueError("Supplemental driver/contract differs from the frozen evaluation")
    source = Path(plan["source_root"])
    if source_snapshot(source) != plan["source_files"]:
        raise ValueError("Pinned runtime source bytes changed")
    parent = Path(plan["parent"])
    for name, expected in plan["parent_artifacts"].items():
        if digest(parent / name) != expected:
            raise ValueError(f"Original artifact changed: {name}")
    activate_source(source)
    if environment() != plan["environment"]:
        raise ValueError("Evaluation runtime package versions/Weaver bytes changed")
    return plan


def chunks(row, plan):
    return [(start, min(start + plan["chunk_size"], row["entries"]))
            for start in range(0, row["entries"], plan["chunk_size"])]


def chunk_metadata(plan, index, start, stop):
    return {"contract": plan.get("contract", VERSION) + "_predictions", "plan_sha256": plan["content_hash"],
            "file_index": index, "source_sha256": plan["files"][index]["sha256"],
            "entry_start": start, "entry_stop": stop,
            "model_order": [[r["run_id"], r["seed"]] for r in plan["tasks"]]}


def read_chunk(path, expected, *, keys=None):
    import numpy as np
    sidecar = read_json(path.with_suffix(".json"))
    if sidecar["metadata"] != expected or sidecar["npz_sha256"] != digest(path):
        raise ValueError(f"Prediction shard lineage/bytes differ: {path}")
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {f"logits_{i:02d}" for i in range(12)}:
            raise ValueError("Prediction shard does not contain all twelve models")
        arrays = {key: data[key] for key in (data.files if keys is None else keys)}
    for values in arrays.values():
        if values.shape != (expected["entry_stop"] - expected["entry_start"], 10) or values.dtype != np.float32 or not np.isfinite(values).all():
            raise ValueError("Invalid prediction shape, precision or nonfinite logits")
    return arrays, sidecar


def chunk_path(output, index, start):
    return output / "predictions" / f"file_{index:03d}" / f"chunk_{start:06d}.npz"


def models_and_backend(plan, device):
    import torch
    from teacher_logit_reco.relational_part.runtime import build_runtime_model
    from teacher_logit_reco.relational_part.ca_tree import load_tree_backend
    parent = Path(plan["parent"])
    relation = read_json(parent / "inputs/relation_normalization.json")
    region = read_json(parent / "inputs/region_normalization.json")
    screening = read_json(parent / "registry/screening_registry.json")
    backend_manifest = read_json(parent / "backend/backend_manifest.json")
    backend = load_tree_backend(parent / "backend" / backend_manifest["binary_filename"],
                                parent / "backend/backend_manifest.json",
                                source_path=Path(plan["source_root"]) / "teacher_logit_reco/relational_part/csrc/relational_ca_tree_v1.cpp")
    models = []
    for task in plan["tasks"]:
        model = build_runtime_model(task["run_id"], screening_registry=screening,
                                    normalization_artifact=relation, region_normalization_artifact=region,
                                    selected_families=task["relation_families"])
        checkpoint = torch.load(parent / task["checkpoint"], map_location="cpu", weights_only=False)
        if checkpoint["model_contract_sha256"] != task["model_contract_sha256"]:
            raise ValueError("Checkpoint payload has another model contract")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        models.append(model.to(device).eval())
    return models, backend


def infer_arrays(arrays, label, models, backend, device):
    """The exact original tokenizer, particle inputs, tree backend and forwards."""
    import numpy as np
    import torch
    from jetclass_fresh.jetclass_data import _tokens_from_arrays, _verify_label_chunk
    from jetclass_fresh.part_inputs import build_particle_transformer_inputs_from_tokens
    from teacher_logit_reco.relational_part.ca_tree import build_compiled_tree
    from teacher_logit_reco.relational_part.evaluation import model_forward
    _verify_label_chunk(arrays, np.arange(len(arrays)), label, Path("official_test_chunk"))
    reset_transient_trimmers(models)
    tokens, mask = _tokens_from_arrays(arrays, 128)
    results = [[] for _ in models]
    with torch.no_grad():
        for start in range(0, len(tokens), 64):
            raw, valid = tokens[start:start + 64], mask[start:start + 64]
            inputs = build_particle_transformer_inputs_from_tokens(raw, valid, source_view="offline")
            vectors = inputs.pf_vectors.transpose(0, 2, 1)
            trees = [build_compiled_tree(backend, vectors[i], raw[i], valid[i]) for i in range(len(raw))]
            tensors = {"points": inputs.pf_points, "features": inputs.pf_features,
                       "lorentz_vectors": inputs.pf_vectors, "mask": inputs.pf_mask.astype(bool),
                       "raw_tokens": raw}
            batch = {key: torch.from_numpy(value).to(device) for key, value in tensors.items()}
            batch["region_trees"] = trees
            for i, model in enumerate(models):
                # A Weaver trimmer must not modify another checkpoint's inputs.
                independent = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                value = model_forward(model, independent).detach().float().cpu().numpy()
                if value.shape != (len(raw), 10) or not np.isfinite(value).all():
                    raise ValueError("Invalid model outputs")
                results[i].append(value)
    return {f"logits_{i:02d}": np.concatenate(rows) for i, rows in enumerate(results)}


def reset_transient_trimmers(models):
    """Unserialized Weaver warm-up counters must not depend on job restarts."""
    for model in models:
        for name, module in model.named_modules():
            if name.split(".")[-1] == "trimmer" and hasattr(module, "_counter"):
                if not hasattr(module, "_rpt_full_test_initial_counter"):
                    module._rpt_full_test_initial_counter = copy.deepcopy(module._counter)
                module._counter = copy.deepcopy(module._rpt_full_test_initial_counter)


def run(args, *, frozen_plan=None, model_loader=None):
    import numpy as np
    import torch
    import uproot
    output = args.output.resolve()
    plan = load_plan(output) if frozen_plan is None else frozen_plan
    from jetclass_fresh.jetclass_data import PARTICLE_READ_BRANCHES, LABEL_BRANCHES
    if not 0 <= args.task_index < len(plan["assignments"]):
        raise IndexError("GPU task index outside frozen assignments")
    if not torch.cuda.is_available():
        raise RuntimeError("Submit inference to a GPU node; CPU inference is not enabled")
    torch.set_num_threads(min(4, int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))))
    device = torch.device("cuda")
    with exclusive(output / "locks" / f"gpu_{args.task_index}.lock"):
        models = backend = None
        for index in plan["assignments"][args.task_index]:
            row = plan["files"][index]
            path = Path(plan["data_dir"]) / row["name"]
            if digest(path) != row["sha256"]:
                raise ValueError(f"ROOT file bytes changed: {path}")
            records = []
            with uproot.open(path) as root:
                for start, stop in chunks(row, plan):
                    destination = chunk_path(output, index, start)
                    expected = chunk_metadata(plan, index, start, stop)
                    if destination.exists() and destination.with_suffix(".json").exists():
                        _, record = read_chunk(destination, expected)
                        progress("reuse_chunk", file=row["name"], start=start, stop=stop)
                    else:
                        if destination.with_suffix(".json").exists():
                            raise ValueError(f"Committed prediction data is missing: {destination}")
                        # An NPZ without its commit sidecar is an interrupted write,
                        # not a reusable result. Its exact chunk is safely recomputed.
                        if models is None:
                            models, backend = (models_and_backend if model_loader is None else model_loader)(plan, device)
                        clock = time.monotonic()
                        progress("starting_chunk", file=row["name"], start=start, stop=stop, models=12)
                        arrays = root["tree"].arrays(PARTICLE_READ_BRANCHES + LABEL_BRANCHES,
                                                    entry_start=start, entry_stop=stop, library="ak")
                        predictions = infer_arrays(arrays, row["label"], models, backend, device)
                        with atomic_file(destination) as stream:
                            np.savez_compressed(stream, **predictions)
                        record = hashed({"metadata": expected, "npz_sha256": digest(destination),
                                         "execution": {"host": platform.node(),
                                                       "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                                                       "gpu": torch.cuda.get_device_name(device)}})
                        write_json(destination.with_suffix(".json"), record)
                        progress("evaluated_chunk", file=row["name"], start=start, stop=stop,
                                 models=12, seconds=round(time.monotonic() - clock, 2))
                    records.append(record["content_hash"])
            write_json(output / "predictions" / f"file_{index:03d}" / "complete.json",
                       hashed({"contract": plan["contract"] + "_file_complete", "plan_sha256": plan["content_hash"],
                               "file_index": index, "chunks": records}))
        progress("gpu_task_complete", task_index=args.task_index)


def average_ranks(values):
    """Vectorized equivalent of the original stable-sort average-tie ranks."""
    import numpy as np
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
    stops = np.r_[starts[1:], len(values)]
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.repeat(0.5 * (starts + 1 + stops), stops - starts)
    return ranks


def paired_accuracy(candidate, reference, truth):
    """Exact bootstrap distribution for paired accuracy using sufficient counts.

Unlike the old event-index bootstrap, this does not generate 20M x 10k indices.
The RNG realization differs, so the policy has its own supplemental contract.
"""
    import numpy as np
    delta = (candidate == truth).astype(np.int8) - (reference == truth).astype(np.int8)
    rng = np.random.Generator(np.random.PCG64(917301))
    draws = np.zeros(10000, dtype=np.int64)
    counts = []
    for label in range(10):
        values = delta[truth == label]
        if not len(values):
            raise ValueError("Paired bootstrap requires all ten classes")
        counts.append(np.bincount(values + 1, minlength=3))
        sample = rng.multinomial(len(values), counts[-1] / len(values), size=len(draws))
        draws += sample[:, 2] - sample[:, 0]
    if len({int(row.sum()) for row in counts}) != 1:
        raise ValueError("Paired bootstrap requires exactly balanced classes")
    return {"mean_accuracy_difference": float(delta.mean()),
            "confidence_interval_95": np.quantile(draws / len(truth), [0.025, 0.975], method="linear").tolist(),
            "classwise_counts_loss_tie_win": [v.tolist() for v in counts],
            "sampling_unit": "paired_event_identity", "stratification": "true_class_fixed_counts",
            "seed": 917301, "replicates": 10000, "quantile_method": "linear"}


def wilson_interval(passed, total):
    if total <= 0 or not 0 <= passed <= total:
        raise ValueError("Invalid binomial counts")
    z = statistics.NormalDist().inv_cdf(0.975)
    p, z2 = passed / total, z * z
    center = (p + z2 / (2 * total)) / (1 + z2 / total)
    radius = z * math.sqrt(p * (1 - p) / total + z2 / (4 * total * total)) / (1 + z2 / total)
    return [max(0.0, center - radius), min(1.0, center + radius)]


def calculate_metrics(logits, labels):
    import numpy as np
    from teacher_logit_reco.relational_part import evaluation as ev
    values = np.asarray(logits, dtype=np.float64)
    # Same statistic and tie convention, removing a Python loop over 20M ranks.
    original = ev._average_ranks
    try:
        ev._average_ranks = average_ranks
        metrics = ev.evaluate_logits(values, labels, split="official_test_20M")
    finally:
        ev._average_ranks = original
    metrics.pop("content_hash")
    metrics["contract"] = VERSION + "_metrics"
    for label in range(1, 10):
        for target in TARGETS:
            row = ev.qcd_signal_rejection(values, labels, signal_index=label, target_efficiency=target)
            bounds = wilson_interval(row["qcd_false_positive_count"], row["qcd_support"])
            row["conditional_qcd_fpr_wilson_95"] = bounds
            row["conditional_rejection_wilson_95"] = [1 / bounds[1], None if bounds[0] == 0 else 1 / bounds[0]]
            metrics["qcd_signal_rejection"][CLASS_NAMES[label]][str(target)] = row
    return hashed(metrics)


def aggregate(args):
    import numpy as np
    output = args.output.resolve()
    plan = load_plan(output)
    index = args.task_index
    if not 0 <= index < 12:
        raise IndexError("Metric task index outside twelve-model matrix")
    task = plan["tasks"][index]
    # Only one model's 20M logits is held at a time; never concatenate all twelve.
    values = np.empty((plan["event_count"], 10), dtype=np.float32)
    labels = np.empty(plan["event_count"], dtype=np.int16)
    reference = np.empty(plan["event_count"], dtype=np.int16)
    layerwise = np.empty(plan["event_count"], dtype=np.int16) if index >= 9 else None
    base_index = SEEDS.index(task["seed"])
    keys = {f"logits_{index:02d}", f"logits_{base_index:02d}"}
    if layerwise is not None:
        keys.add(f"logits_{6 + base_index:02d}")
    offset, file_hashes = 0, []
    for file_index, row in enumerate(plan["files"]):
        done = read_json(output / "predictions" / f"file_{file_index:03d}" / "complete.json")
        records = []
        for start, stop in chunks(row, plan):
            arrays, record = read_chunk(chunk_path(output, file_index, start),
                                        chunk_metadata(plan, file_index, start, stop), keys=keys)
            end = offset + stop - start
            values[offset:end] = arrays[f"logits_{index:02d}"]
            labels[offset:end] = row["label"]
            reference[offset:end] = arrays[f"logits_{base_index:02d}"].argmax(axis=1)
            if layerwise is not None:
                layerwise[offset:end] = arrays[f"logits_{6 + base_index:02d}"].argmax(axis=1)
            offset = end
            records.append(record["content_hash"])
        if done != hashed({"contract": VERSION + "_file_complete", "plan_sha256": plan["content_hash"],
                           "file_index": file_index, "chunks": records}):
            raise ValueError("File completion/coverage differs from predictions")
        file_hashes.append(done["content_hash"])
    if offset != plan["event_count"] or not np.array_equal(np.bincount(labels, minlength=10), np.full(10, 2_000_000)):
        raise ValueError("Full-test coverage is incomplete or unbalanced")
    progress("calculate_global_metrics", run_id=task["run_id"], seed=task["seed"], event_count=offset)
    metrics = calculate_metrics(values, labels)
    prediction = values.argmax(axis=1)
    paired = {"OFF_RPT_BASE": paired_accuracy(prediction, reference, labels)}
    if layerwise is not None:
        paired["OFF_RPT_SELECTED_LAYERWISE"] = paired_accuracy(prediction, layerwise, labels)
    result = hashed({"contract": VERSION + "_result", "plan_sha256": plan["content_hash"],
                     "task": task, "file_completion_hashes": file_hashes, "metrics": metrics,
                     "paired_accuracy": paired, "used_for_model_selection": False})
    write_json(output / "metrics" / f"model_{index:02d}.json", result)
    progress("metrics_complete", task_index=index, accuracy=metrics["accuracy"])


def report(args):
    output = args.output.resolve()
    plan = load_plan(output)
    results = [read_json(output / "metrics" / f"model_{i:02d}.json") for i in range(12)]
    for i, result in enumerate(results):
        if result["plan_sha256"] != plan["content_hash"] or result["task"] != plan["tasks"][i] or result["file_completion_hashes"] != results[0]["file_completion_hashes"]:
            raise ValueError("Report mixes evaluations, checkpoints or test populations")
    rows = {}
    for name in MODELS:
        metrics = [r["metrics"] for r in results if r["task"]["run_id"] == name]
        accuracies = [m["accuracy"] for m in metrics]
        rows[name] = {"mean_accuracy": statistics.mean(accuracies),
                      "seed_sample_standard_deviation": statistics.stdev(accuracies),
                      "per_seed_accuracy": dict(zip(map(str, SEEDS), accuracies)), "qcd_signal_rejection": {}}
        for signal in CLASS_NAMES[1:]:
            rows[name]["qcd_signal_rejection"][signal] = {}
            for target in TARGETS:
                values = [m["qcd_signal_rejection"][signal][str(target)]["background_rejection"] for m in metrics]
                rows[name]["qcd_signal_rejection"][signal][str(target)] = {
                    "per_seed_rejection": dict(zip(map(str, SEEDS), values)),
                    "mean_finite_background_rejection": None if any(v is None for v in values) else statistics.mean(values)}
    for name in MODELS:
        rows[name]["difference_vs_baseline"] = rows[name]["mean_accuracy"] - rows[MODELS[0]]["mean_accuracy"]
    result = hashed({"contract": VERSION + "_report", "plan_sha256": plan["content_hash"],
                     "event_count_per_model": plan["event_count"], "models": rows,
                     "per_model_results": results, "inference_only": True,
                     "interpretation": "1M-training-jet models; supplemental official 20M test evaluation; not 100M training or a new model-selection round"})
    write_json(output / "reports/full_test_report.json", result)
    lines = ["# Offline RPT: supplemental official 20M-jet test", "",
             "Frozen checkpoints; no retraining or model selection. Three seeds per configuration.", "",
             "| Model | Accuracy | Difference vs base | Seed SD |", "|---|---:|---:|---:|"]
    for name, row in rows.items():
        lines.append(f"| {name} | {100*row['mean_accuracy']:.4f}% | {100*row['difference_vs_baseline']:+.4f} pp | {100*row['seed_sample_standard_deviation']:.4f} pp |")
    for target in TARGETS:
        lines.extend(["", f"## QCD rejection at {100*target:g}% signal efficiency", "",
                      "Three-seed arithmetic mean; percent change relative to BASE. Zero-background seeds give no finite mean.", "",
                      "| Signal | BASE | BASE_EDGEVALUE | SELECTED_LAYERWISE | SELECTED_EDGEVALUE |", "|---|---:|---:|---:|---:|"])
        for signal in CLASS_NAMES[1:]:
            means = [rows[name]["qcd_signal_rejection"][signal][str(target)]["mean_finite_background_rejection"] for name in MODELS]
            cells = []
            for i, mean in enumerate(means):
                cell = "no finite mean" if mean is None else f"{mean:.2f}"
                if i and mean is not None and means[0] is not None:
                    cell += f" ({100*(mean/means[0]-1):+.1f}%)"
                cells.append(cell)
            lines.append(f"| {signal} | " + " | ".join(cells) + " |")
    lines.extend(["", "Per-seed thresholds, achieved efficiencies, background counts, conditional Wilson intervals and paired accuracy intervals are in JSON.",
                  "Wilson intervals condition on the empirical signal threshold; they do not include its sampling uncertainty. Seed SD is not a confidence interval.",
                  "These models were trained on 1M jets, not the full 100M training sample.", ""])
    text = "\n".join(lines)
    path = output / "reports/full_test_report.md"
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise FileExistsError("Existing Markdown report differs")
    with atomic_file(path, binary=False) as stream:
        stream.write(text)
    progress("report_complete", path=str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    archive = sub.add_parser("inspect-archive")
    archive.add_argument("archive", type=Path)
    p = sub.add_parser("prepare")
    for name in ("parent", "source-root", "data-dir", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--jobs", type=int, default=20)
    for command in ("run", "aggregate", "report"):
        p = sub.add_parser(command)
        p.add_argument("--output", type=Path, required=True)
        if command != "report":
            p.add_argument("--task-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "-1")))
    args = parser.parse_args()
    if args.command == "inspect-archive":
        print(json.dumps(inspect_archive(args.archive), indent=2))
    else:
        {"prepare": prepare, "run": run, "aggregate": aggregate, "report": report}[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
