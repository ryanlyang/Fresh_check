#!/usr/bin/env python3
"""Inference-only x86 migration of the locked offline factorial campaign.

The original training campaign and Tigris predictions are read-only. This
adapter copies its authenticated Python/C++ source, records a separate runtime,
and requires numerical validation before writing any production predictions.
"""
from __future__ import annotations

import argparse
import copy
import gc
import math
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import evaluate_relational_part_offline_full_test as F

VERSION = "relational_part_sporc_inference_v1"
ADAPTER_FILES = (
    "scripts/run_relational_part_sporc_inference.py",
    "scripts/relational_part_sporc_reporting.py",
    "scripts/build_relational_part_tree_backend.py",
    "sbatch/run_relational_part_sporc_inference.sh",
)
TREE_SOURCE = "teacher_logit_reco/relational_part/csrc/relational_ca_tree_v1.cpp"
# Freeze before observing migration outputs; never tune per model or class.
VALIDATION_POLICY = {
    "golden_events_per_class": 640,  # Ten batch-64 forwards, including active trimming.
    "golden_file_selection": "first file in original plan for each class; first chunk",
    "new_cpu_gpu_events": 8,
    "new_cpu_gpu_trimmer_states": ["fresh", "after_ten_full_batches"],
    "tree_events_per_class": 4,
    "logit_atol": 5e-5, "logit_rtol": 5e-5,
    "tree_continuous_atol": 2e-6,
    "tree_topology_categories_masks": "exact",
    "golden_models": 12, "new_models": 12,
    "precision": "FP32, no model AMP or outer autocast",
    "performance_gate": False,
}
MATH_OPTIONS = {
    "matmul_allow_tf32": False, "cudnn_allow_tf32": True,
    "cudnn_benchmark": False, "cudnn_deterministic": False,
}


def runtime_environment():
    import torch
    return {**F.environment(), "architecture": platform.machine(),
            "torch_cuda_build": torch.version.cuda,
            "torch_cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI)}


def safe_relative(name):
    path = PurePosixPath(name)
    if not path.parts or path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
        raise ValueError(f"Unsafe artifact path: {name}")
    return path


def authenticate_files(root, expected):
    for name, digest in expected.items():
        if F.digest(root / safe_relative(name)) != digest:
            raise ValueError(f"Artifact/source changed: {root / name}")


def require_separate_output(output, inputs):
    for path in inputs:
        path = path.resolve()
        if output.is_relative_to(path) or path.is_relative_to(output):
            raise ValueError(f"Migration output must be separate from read-only input: {path}")


def choose_probe_files(plan):
    selected = []
    for label in range(10):
        matches = [i for i, row in enumerate(plan["files"]) if row["label"] == label]
        if not matches:
            raise ValueError("Golden plan lacks a class")
        index = matches[0]
        if plan["files"][index]["entries"] < VALIDATION_POLICY["golden_events_per_class"]:
            raise ValueError("Golden probe file is too short")
        selected.append(index)
    return selected


def check_storage(output, source_campaign, event_count):
    # Account for both our new predictions and the still-queued Tigris campaign.
    budget = event_count * 12 * 10 * 4
    roots = (output / "test/predictions", source_campaign / "test/predictions")
    remaining = sum(max(0, budget - sum(p.stat().st_size for p in root.rglob("*.npz")))
                    for root in roots)
    available = shutil.disk_usage(output if output.exists() else output.parent).free
    if available < remaining + 6 * 1024**3:
        raise OSError("Insufficient space for SPORC and pending Tigris logits plus 6 GiB headroom; nothing deleted")


def bootstrap(args):
    source_campaign, output = args.source_campaign.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError("Output exists; use --resume, never overwrite a migration")
    if not 1 <= args.jobs <= 200:
        raise ValueError("Use between 1 and 200 inference jobs")
    source_spec = F.read_json(source_campaign / "campaign_spec.json")
    if source_spec["contract"] != "relational_part_offline_factorial_followup_v1":
        raise ValueError("Expected the trained offline factorial campaign")
    previous, original = Path(source_spec["previous_evaluation"]), Path(source_spec["parent"])
    require_separate_output(output, [source_campaign, previous, original, Path(source_spec["test_data"])])
    authenticate_files(source_campaign / "source", source_spec["source_files"])
    source_plan = F.read_json(source_campaign / "test/evaluation_plan.json")
    reference_plan = F.read_json(source_campaign / "reference_plan.json")
    reference_report = F.read_json(source_campaign / "reference_report.json")
    if (reference_plan["content_hash"] != source_spec["reference_plan_sha256"]
            or reference_report["content_hash"] != source_spec["reference_report_sha256"]
            or reference_report["plan_sha256"] != reference_plan["content_hash"]
            or source_plan["campaign_sha256"] != source_spec["content_hash"]
            or source_plan["event_count"] != 20_000_000):
        raise ValueError("Source evaluation/reference lineage differs")
    environment = runtime_environment()
    if environment["architecture"] != "x86_64" or not environment["torch_cuda_build"]:
        raise ValueError("Bootstrap must run in the x86 CUDA-enabled SPORC environment")
    if environment["weaver_particle_transformer_sha256"] != source_spec["environment"]["weaver_particle_transformer_sha256"]:
        raise ValueError("SPORC Weaver source must match the trained campaign exactly")
    check_storage(output, source_campaign, source_plan["event_count"])
    output.mkdir()
    frozen = output / "source"
    for name, expected in source_spec["source_files"].items():
        destination = frozen / safe_relative(name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_campaign / "source" / name, destination)
        if F.digest(destination) != expected:
            raise ValueError("Frozen source copy differs")
    for name in ADAPTER_FILES:
        destination = frozen / name
        if destination.exists():
            raise ValueError(f"Adapter would overwrite original source: {name}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / name, destination)
    F.write_json(output / "reference_plan.json", reference_plan)
    F.write_json(output / "reference_report.json", reference_report)
    manifest = F.hashed({
        "contract": VERSION, "schema_version": 1, "output": str(output),
        "source_campaign": str(source_campaign), "source_campaign_sha256": source_spec["content_hash"],
        "source_test_plan_sha256": source_plan["content_hash"],
        "previous_evaluation": str(previous), "original_parent": str(original),
        "reference_plan_sha256": reference_plan["content_hash"],
        "reference_report_sha256": reference_report["content_hash"],
        "source_files": {**source_spec["source_files"], **{n: F.digest(frozen / n) for n in ADAPTER_FILES}},
        "source_environment": source_spec["environment"], "environment": environment,
        "models": source_spec["models"], "seeds": source_spec["seeds"], "jobs": args.jobs,
        "validation_policy": VALIDATION_POLICY, "math_options": MATH_OPTIONS,
        "probe_file_indices": choose_probe_files(reference_plan),
        "inference_only": True, "performance_gate": False, "automatic_deletion": False,
    })
    F.write_json(output / "sporc_campaign.json", manifest)
    (output / "logs").mkdir()
    # Authenticate the original lock using its OWN frozen implementation, not
    # potentially newer source from the interactive checkout.
    subprocess.run([sys.executable, "-s", str(frozen / ADAPTER_FILES[0]),
                    "check", "--output", str(output)], check=True, cwd=frozen)
    F.progress("sporc_bootstrapped", output=str(output), jobs=args.jobs)


def load_spec(output):
    from scripts import run_relational_part_offline_ablations as A
    spec = F.read_json(output / "sporc_campaign.json")
    if (spec["contract"] != VERSION or spec["schema_version"] != 1
            or spec["output"] != str(output) or REPO != output / "source"
            or spec["environment"] != runtime_environment()
            or spec["validation_policy"] != VALIDATION_POLICY or spec["math_options"] != MATH_OPTIONS
            or not 1 <= spec["jobs"] <= 200 or not spec["inference_only"]
            or spec["performance_gate"] is not False):
        raise ValueError("SPORC source/runtime/validation contract differs")
    authenticate_files(REPO, spec["source_files"])
    actual = F.source_snapshot(REPO)
    if actual != {n: h for n, h in spec["source_files"].items()
                  if n == "jetclass_fixed_hlt.py" or n.startswith(("teacher_logit_reco/", "jetclass_fresh/"))}:
        raise ValueError("Frozen model source inventory changed")
    source_campaign = Path(spec["source_campaign"])
    source = A.load_campaign(source_campaign, runtime=False)
    original_plan = A.test_plan(source_campaign, source)
    if (source["content_hash"] != spec["source_campaign_sha256"]
            or original_plan["content_hash"] != spec["source_test_plan_sha256"]
            or source["models"] != spec["models"] or source["seeds"] != spec["seeds"]
            or source["parent"] != spec["original_parent"]
            or source["previous_evaluation"] != spec["previous_evaluation"]):
        raise ValueError("Original locked campaign changed")
    for name in ("reference_plan", "reference_report"):
        if F.read_json(output / (name + ".json"))["content_hash"] != spec[name + "_sha256"]:
            raise ValueError("Migration reference artifact changed")
    if spec["probe_file_indices"] != choose_probe_files(F.read_json(output / "reference_plan.json")):
        raise ValueError("Deterministic portability probe selection changed")
    return spec, source, original_plan


def new_plan(spec, original, backend):
    body = {k: copy.deepcopy(v) for k, v in original.items() if k != "content_hash"}
    body.update(contract=VERSION + "_test", campaign_sha256=spec["content_hash"],
                source_campaign_sha256=spec["source_campaign_sha256"],
                source_test_plan_sha256=original["content_hash"],
                backend_manifest_sha256=backend["content_hash"], environment=spec["environment"],
                validation_policy=spec["validation_policy"], math_options=spec["math_options"],
                assignments=F.file_assignment(original["files"], spec["jobs"]))
    return F.hashed(body)


def load_backend(output):
    from teacher_logit_reco.relational_part.ca_tree import load_tree_backend
    manifest = F.read_json(output / "backend/backend_manifest.json")
    module = load_tree_backend(output / "backend" / manifest["binary_filename"],
                              output / "backend/backend_manifest.json", source_path=REPO / TREE_SOURCE)
    return module, manifest


def load_ready(output):
    spec, source, original = load_spec(output)
    backend, manifest = load_backend(output)
    plan = F.read_json(output / "test/evaluation_plan.json")
    if plan != new_plan(spec, original, manifest):
        raise ValueError("SPORC inference plan changed")
    return spec, source, plan, backend


def build(args):
    spec, source, original = load_spec(args.output)
    old_manifest = F.read_json(Path(spec["original_parent"]) / "backend/backend_manifest.json")
    if F.digest(REPO / TREE_SOURCE) != old_manifest["source_sha256"]:
        raise ValueError("Tree algorithm source differs from original backend")
    with F.exclusive(args.output / "locks/build.lock"):
        if not (args.output / "backend/backend_manifest.json").exists():
            subprocess.run([sys.executable, "-s", str(REPO / "scripts/build_relational_part_tree_backend.py"),
                            "--contract", "relational_ca_tree_v1", "--build-dir", str(args.output / "build"),
                            "--output-dir", str(args.output / "backend")], check=True)
        _, manifest = load_backend(args.output)
        if manifest["platform_architecture"] != "x86_64":
            raise ValueError("Rebuilt backend is not x86")
        F.write_json(args.output / "test/evaluation_plan.json", new_plan(spec, original, manifest))
    F.progress("sporc_backend_ready", backend_sha256=manifest["content_hash"])


def configure_math():
    import torch
    torch.set_num_threads(min(4, int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))))
    torch.backends.cuda.matmul.allow_tf32 = MATH_OPTIONS["matmul_allow_tf32"]
    torch.backends.cudnn.allow_tf32 = MATH_OPTIONS["cudnn_allow_tf32"]
    torch.backends.cudnn.benchmark = MATH_OPTIONS["cudnn_benchmark"]
    torch.backends.cudnn.deterministic = MATH_OPTIONS["cudnn_deterministic"]


def gpu_identity():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("This phase requires a Slurm GPU allocation")
    return {"name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "total_memory": torch.cuda.get_device_properties(0).total_memory}


def assert_logits(actual, expected):
    import numpy as np
    actual, expected = np.asarray(actual), np.asarray(expected)
    if not actual.size or actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("Invalid parity logit shape or nonfinite values")
    error = float(np.max(np.abs(actual.astype(np.float64) - expected)))
    if not np.allclose(actual, expected, atol=VALIDATION_POLICY["logit_atol"],
                       rtol=VALIDATION_POLICY["logit_rtol"]):
        raise ValueError(f"FP32 portability parity failed (maximum absolute error {error}); tolerances are frozen")
    return error


def trimmer_state(model):
    """Snapshot the non-checkpointed counters from disposable probe models."""
    return {name: copy.deepcopy(module._counter) for name, module in model.named_modules()
            if name.split(".")[-1] == "trimmer" and hasattr(module, "_counter")}


def probe_trimmer_state(model, state):
    """Use matching warmed counters on CPU and GPU without changing flags.

    F.infer_arrays resets counters on entry; set its reset cache too. Only the
    disposable validation models receive this override, never inference models.
    """
    import torch
    modules = {name: module for name, module in model.named_modules()
               if name.split(".")[-1] == "trimmer" and hasattr(module, "_counter")}
    if set(modules) != set(state):
        raise ValueError("CPU/GPU trimmer inventories differ")
    for name, module in modules.items():
        value = state[name]
        if isinstance(module._counter, torch.Tensor):
            module._counter.copy_(torch.as_tensor(value).to(module._counter.device))
        else:
            module._counter = copy.deepcopy(value)
        module._rpt_full_test_initial_counter = copy.deepcopy(module._counter)


def load_models(spec, source, tasks, device, *, historical=False):
    import torch
    from scripts import run_relational_part_offline_ablations as A
    from teacher_logit_reco.relational_part.runtime import build_runtime_model
    parent = Path(spec["original_parent"] if historical else spec["source_campaign"])
    normalization_root = Path(spec["original_parent"])
    relation = F.read_json(normalization_root / "inputs/relation_normalization.json")
    region = F.read_json(normalization_root / "inputs/region_normalization.json")
    screening = F.read_json(normalization_root / "registry/screening_registry.json") if historical else None
    models = []
    for task in tasks:
        path = parent / safe_relative(task["checkpoint"])
        if F.digest(path) != task["checkpoint_sha256"]:
            raise ValueError("Checkpoint bytes changed")
        model = (build_runtime_model(task["run_id"], screening_registry=screening,
                    normalization_artifact=relation, region_normalization_artifact=region,
                    selected_families=task["relation_families"]) if historical else A.model_for(source, task["run_id"]))
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint["model_contract_sha256"] != task["model_contract_sha256"]:
            raise ValueError("Checkpoint payload contract differs")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        if any(getattr(module, "use_amp", False) for module in model.modules()):
            raise ValueError("FP32 inference requires model AMP disabled")
        models.append(model.to(device).eval())
        del checkpoint
    return models


def golden_chunk(spec, reference_plan, reference_report, index):
    # Bind sampled bytes to the ALREADY REPORTED file completion, not merely
    # to a newly self-hashed prediction sidecar.
    previous = Path(spec["previous_evaluation"])
    row = reference_plan["files"][index]
    records = []
    for start, stop in F.chunks(row, reference_plan):
        record = F.read_json(F.chunk_path(previous, index, start).with_suffix(".json"))
        if record["metadata"] != F.chunk_metadata(reference_plan, index, start, stop):
            raise ValueError("Golden prediction lineage differs")
        records.append(record["content_hash"])
    done = F.read_json(previous / "predictions" / f"file_{index:03d}/complete.json")
    expected = F.hashed({"contract": reference_plan["contract"] + "_file_complete",
                         "plan_sha256": reference_plan["content_hash"], "file_index": index, "chunks": records})
    if done != expected or any(r["file_completion_hashes"][index] != done["content_hash"]
                               for r in reference_report["per_model_results"]):
        raise ValueError("Golden predictions differ from reported evaluation")
    start, stop = F.chunks(row, reference_plan)[0]
    values, record = F.read_chunk(F.chunk_path(previous, index, start),
                                  F.chunk_metadata(reference_plan, index, start, stop))
    return values, record["content_hash"]


def validation_receipt(output, spec, plan, *, device=None):
    receipt = F.read_json(output / "portability_validation.json")
    if (receipt["contract"] != VERSION + "_validation" or receipt["passed"] is not True
            or receipt["campaign_sha256"] != spec["content_hash"]
            or receipt["plan_sha256"] != plan["content_hash"]
            or receipt["policy"] != VALIDATION_POLICY
            or receipt["new_model_order"] != [[t["run_id"], t["seed"]] for t in plan["tasks"]]
            or receipt["probe_file_indices"] != spec["probe_file_indices"]
            or len(receipt["golden_checks"]) != 10 or len(receipt["new_cpu_gpu_checks"]) != 12
            or len(receipt["new_timings"]) != 10
            or (device is not None and receipt["gpu"] != device)):
        raise ValueError("Missing, stale or incompatible SPORC portability attestation")
    def nonnegative(value):
        return type(value) in (int, float) and math.isfinite(value) and value >= 0

    probes = spec["probe_file_indices"]
    golden = receipt["golden_checks"]
    keys = {f"logits_{i:02d}" for i in range(VALIDATION_POLICY["golden_models"])}
    if ([row.get("file_index") for row in golden] != probes
            or any(row.get("passed") is not True
                   or not isinstance(row.get("golden_chunk_sha256"), str)
                   or len(row["golden_chunk_sha256"]) != 64
                   or set(row.get("per_model_max_absolute_error", {})) != keys
                   or not all(nonnegative(v) for v in row["per_model_max_absolute_error"].values())
                   for row in golden)):
        raise ValueError("Incomplete or invalid historical golden parity coverage")
    cpu = receipt["new_cpu_gpu_checks"]
    if ([[row.get("run_id"), row.get("seed")] for row in cpu] != receipt["new_model_order"]
            or any(row.get("passed") is not True or not nonnegative(row.get("maximum_absolute_error"))
                   or set(row.get("per_state_maximum_absolute_error", {})) != set(VALIDATION_POLICY["new_cpu_gpu_trimmer_states"])
                   or not all(nonnegative(v) for v in row["per_state_maximum_absolute_error"].values())
                   or row["maximum_absolute_error"] != max(row["per_state_maximum_absolute_error"].values())
                   for row in cpu)):
        raise ValueError("Incomplete or invalid new-checkpoint CPU/GPU parity coverage")
    trees = receipt.get("tree_checks", [])
    expected_trees = [(index, entry) for index in probes
                      for entry in range(VALIDATION_POLICY["tree_events_per_class"])]
    if ([(row.get("file_index"), row.get("entry")) for row in trees] != expected_trees
            or any(any(row.get(flag) is not True for flag in (
                    "passed", "topology_and_categories_exact", "continuous_shapes_exact", "continuous_values_finite"))
                   or row.get("continuous_absolute_tolerance") != VALIDATION_POLICY["tree_continuous_atol"]
                   or not nonnegative(row.get("maximum_continuous_absolute_error"))
                   or row["maximum_continuous_absolute_error"] > VALIDATION_POLICY["tree_continuous_atol"]
                   for row in trees)):
        raise ValueError("Incomplete or invalid real-event tree parity coverage")
    timings = receipt["new_timings"]
    if ([row.get("label") for row in timings] != list(range(10))
            or any(row.get("events") != VALIDATION_POLICY["golden_events_per_class"]
                   or not nonnegative(row.get("seconds")) or row["seconds"] == 0 for row in timings)):
        raise ValueError("Incomplete or invalid all-class inference timing coverage")
    return receipt


def validate(args):
    import torch
    import uproot
    from jetclass_fresh.jetclass_data import PARTICLE_READ_BRANCHES, LABEL_BRANCHES, _tokens_from_arrays
    from jetclass_fresh.part_inputs import build_particle_transformer_inputs_from_tokens
    from teacher_logit_reco.relational_part.ca_tree import build_compiled_tree
    from teacher_logit_reco.relational_part.region_tree import build_reference_tree
    from scripts.build_relational_part_tree_backend import _canonical_smoke_parity
    spec, source, plan, backend = load_ready(args.output)
    configure_math()
    gpu = gpu_identity()
    with F.exclusive(args.output / "locks/validation.lock"):
        if (args.output / "portability_validation.json").exists():
            validation_receipt(args.output, spec, plan, device=gpu)
            F.progress("sporc_validation_reused")
            return
        reference = F.read_json(args.output / "reference_plan.json")
        report = F.read_json(args.output / "reference_report.json")
        if len(report["per_model_results"]) != 12:
            raise ValueError("Incomplete historical model coverage")
        for task, row in zip(reference["tasks"], report["per_model_results"]):
            if (F.hashed({k: v for k, v in row.items() if k != "content_hash"}) != row
                    or row["task"] != task or row["plan_sha256"] != reference["content_hash"]):
                raise ValueError("Historical result lineage differs")
        arrays_by_class, golden, tree_checks = [], [], []
        n = VALIDATION_POLICY["golden_events_per_class"]
        for index in spec["probe_file_indices"]:
            row = plan["files"][index]
            path = Path(plan["data_dir"]) / row["name"]
            if F.digest(path) != row["sha256"]:
                raise ValueError("Probe ROOT bytes changed")
            with uproot.open(path) as root:
                arrays = root["tree"].arrays(PARTICLE_READ_BRANCHES + LABEL_BRANCHES,
                                             entry_start=0, entry_stop=n, library="ak")
            if len(arrays) != n:
                raise ValueError("Incomplete portability probe")
            raw, mask = _tokens_from_arrays(arrays[:VALIDATION_POLICY["tree_events_per_class"]], 128)
            inputs = build_particle_transformer_inputs_from_tokens(raw, mask, source_view="offline")
            for i in range(len(raw)):
                compiled = build_compiled_tree(backend, inputs.pf_vectors[i].T, raw[i], mask[i])
                python = build_reference_tree(inputs.pf_vectors[i].T, raw[i], mask[i])
                result = _canonical_smoke_parity(compiled, python)
                if not result["passed"]:
                    raise ValueError(f"Real-event tree parity failed: {row['name']}:{i}: {result}")
                tree_checks.append({"file_index": index, "entry": i, **result})
            arrays_by_class.append((row["label"], arrays))
            golden.append(golden_chunk(spec, reference, report, index))
        device = torch.device("cuda")
        old_models = load_models(spec, source, reference["tasks"], device, historical=True)
        golden_checks = []
        for index, (label, arrays), (expected, receipt_hash) in zip(spec["probe_file_indices"], arrays_by_class, golden):
            actual = F.infer_arrays(arrays, label, old_models, backend, device)
            errors = {}
            for i, task in enumerate(reference["tasks"]):
                key = f"logits_{i:02d}"
                try:
                    errors[key] = assert_logits(actual[key], expected[key][:n])
                except ValueError as exc:
                    raise ValueError(f"Historical {task['run_id']} seed {task['seed']}, "
                                     f"file {plan['files'][index]['name']}: {exc}") from exc
            golden_checks.append({"file_index": index, "golden_chunk_sha256": receipt_hash,
                                  "per_model_max_absolute_error": errors, "passed": True})
            F.progress("sporc_golden_parity", label=label, maximum_absolute_error=max(errors.values()))
        del old_models, actual, golden
        gc.collect()
        torch.cuda.empty_cache()
        models = load_models(spec, source, plan["tasks"], device)
        torch.cuda.reset_peak_memory_stats()
        timings = []
        for label, arrays in arrays_by_class:
            torch.cuda.synchronize()
            start = time.monotonic()
            values = F.infer_arrays(arrays, label, models, backend, device)
            torch.cuda.synchronize()
            elapsed = time.monotonic() - start
            timings.append({"label": label, "events": n, "seconds": elapsed})
            F.progress("sporc_new_checkpoint_probe", label=label, events=n, seconds=round(elapsed, 2))
        # New shared+EV cells have no historical test predictions. Check each
        # against its own CPU FP32 forward in addition to historical golden parity.
        label, arrays = arrays_by_class[0]
        sample = arrays[:VALIDATION_POLICY["new_cpu_gpu_events"]]
        warmed_states = [trimmer_state(model) for model in models]
        gpu_values = F.infer_arrays(sample, label, models, backend, device)
        for model, state in zip(models, warmed_states):
            probe_trimmer_state(model, state)
        warmed_gpu_values = F.infer_arrays(sample, label, models, backend, device)
        cpu_checks = []
        for i, task in enumerate(plan["tasks"]):
            cpu_model = load_models(spec, source, [task], torch.device("cpu"))
            cpu_values = F.infer_arrays(sample, label, cpu_model, backend, torch.device("cpu"))
            probe_trimmer_state(cpu_model[0], warmed_states[i])
            warmed_cpu_values = F.infer_arrays(sample, label, cpu_model, backend, torch.device("cpu"))
            try:
                error = assert_logits(gpu_values[f"logits_{i:02d}"], cpu_values["logits_00"])
                warmed_error = assert_logits(warmed_gpu_values[f"logits_{i:02d}"], warmed_cpu_values["logits_00"])
            except ValueError as exc:
                raise ValueError(f"CPU/GPU {task['run_id']} seed {task['seed']}: {exc}") from exc
            cpu_checks.append({"run_id": task["run_id"], "seed": task["seed"],
                               "passed": True,
                               "maximum_absolute_error": max(error, warmed_error),
                               "per_state_maximum_absolute_error": {
                                   "fresh": error, "after_ten_full_batches": warmed_error}})
            del cpu_model
        max_seconds_per_event = max(t["seconds"] / t["events"] for t in timings)
        largest_assignment = max(sum(plan["files"][i]["entries"] for i in group) for group in plan["assignments"])
        receipt = F.hashed({"contract": VERSION + "_validation", "passed": True,
            "campaign_sha256": spec["content_hash"], "plan_sha256": plan["content_hash"],
            "policy": VALIDATION_POLICY, "gpu": gpu, "probe_file_indices": spec["probe_file_indices"],
            "new_model_order": [[t["run_id"], t["seed"]] for t in plan["tasks"]],
            "tree_checks": tree_checks, "golden_checks": golden_checks,
            "new_cpu_gpu_checks": cpu_checks, "new_timings": timings,
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
            "largest_assignment_events": largest_assignment,
            "estimated_largest_task_seconds_excluding_io": largest_assignment * max_seconds_per_event,
            "timing_is_estimate_not_performance_gate": True})
        F.write_json(args.output / "portability_validation.json", receipt)
        F.progress("sporc_validation_passed", receipt=str(args.output / "portability_validation.json"),
                   estimated_task_hours=receipt["estimated_largest_task_seconds_excluding_io"] / 3600)


def infer(args):
    spec, source, plan, backend = load_ready(args.output)
    configure_math()
    validation_receipt(args.output, spec, plan, device=gpu_identity())
    check_storage(args.output, Path(spec["source_campaign"]), plan["event_count"])
    F.run(argparse.Namespace(output=args.output / "test", task_index=args.task_index),
          frozen_plan=plan,
          model_loader=lambda frozen, device: (load_models(spec, source, frozen["tasks"], device), backend))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("bootstrap", "check", "build", "validate", "infer", "aggregate", "report"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-campaign", type=Path)
    parser.add_argument("--jobs", type=int, default=40)
    parser.add_argument("--task-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "-1")))
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.phase == "bootstrap":
        if args.source_campaign is None:
            parser.error("bootstrap needs --source-campaign")
        bootstrap(args)
    elif args.phase == "check":
        load_spec(args.output)
        F.progress("sporc_inputs_authenticated")
    elif args.phase in ("aggregate", "report"):
        from scripts import relational_part_sporc_reporting as reporting
        spec, _, plan, _ = load_ready(args.output)
        validation_receipt(args.output, spec, plan)
        if args.phase == "aggregate":
            if not 0 <= args.task_index < 12:
                parser.error("aggregate task index must be 0..11")
            reporting.aggregate(args.output, spec, plan, args.task_index)
        else:
            reporting.report(args.output, spec, plan)
    else:
        globals()[args.phase](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
