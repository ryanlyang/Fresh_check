#!/usr/bin/env python3
"""Fresh, single-runtime SPORC evaluation of all eight cells / 24 checkpoints.

This is NOT a relaxation or continuation of portability-v1. Historical scores
are comparison evidence only; every primary prediction is computed on SPORC.
"""
from __future__ import annotations

import argparse
import copy
import gc
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import run_relational_part_sporc_inference as D
from scripts import relational_part_sporc_reporting as R
F = D.F

VERSION = "relational_part_sporc_unified_evaluation_v1"
FOLLOWUP_MODELS = ("OFF_RPT_BASE_LAYERWISE", "OFF_RPT_BASE_SHARED_EDGEVALUE",
                   "OFF_RPT_SELECTED_SHARED_EDGEVALUE", "OFF_RPT_SELECTED_SHARED")
MATH_OPTIONS = {**D.MATH_OPTIONS, "cudnn_allow_tf32": False,
                "float32_matmul_precision": "highest"}
POLICY = {
    "probe_events_per_class": 640, "probe_classes": list(range(10)),
    "probe_selection": "first file per class, first 640 entries, ten full batch-64 forwards",
    "cpu_gpu_events_per_class": 8, "states": ["fresh", "after_ten_full_batches"],
    "required": ["checkpoint_source_data_identity", "finite_logits_exact_shapes",
                 "same_gpu_repeatability", "compiled_reference_tree_agreement", "complete_24_checkpoint_coverage"],
    "same_gpu_repeat_atol": 5e-5, "same_gpu_repeat_rtol": 5e-5,
    "diagnostic_atol": 5e-5, "diagnostic_rtol": 5e-5,
    "cpu_gpu_agreement": "diagnostic_only_not_an_authorization_condition",
    "historical_tigris_agreement": "diagnostic_only_not_an_authorization_condition",
    "tree_events_per_class": 4, "tree_continuous_atol": 2e-6,
    "tree_topology_categories_masks": "exact", "performance_gate": False,
    "gpu_compatibility": "same CUDA compute capability; GPU name/memory recorded",
}
ADAPTER_FILES = (*D.ADAPTER_FILES,
    "scripts/run_relational_part_sporc_unified.py", "scripts/relational_part_sporc_unified_reporting.py",
    "scripts/diagnose_relational_part_sporc_parity.py", "sbatch/run_relational_part_sporc_unified.sh")


def combined_tasks(original, followup):
    """Keep run IDs, seeds, checkpoint bytes and their owning roots unambiguous."""
    tasks = []
    for origin, plan, names in (("historical", original, F.MODELS), ("followup", followup, FOLLOWUP_MODELS)):
        R._matrix(plan["tasks"], names, origin)
        tasks.extend({**copy.deepcopy(task), "origin": origin} for task in plan["tasks"])
    if len({(t["run_id"], t["seed"]) for t in tasks}) != 24:
        raise ValueError("Unified evaluation requires 24 distinct checkpoints")
    for key in ("files", "event_count", "class_order", "batch_size", "chunk_size", "metric_policy", "rejection_targets"):
        if original[key] != followup[key]:
            raise ValueError(f"Historical and follow-up test protocols differ: {key}")
    return tasks


def validate_reference(plan, report):
    R._authenticated(plan, "Reference plan")
    R._authenticated(report, "Reference report")
    R._matrix(plan["tasks"], F.MODELS, "Historical")
    if (plan["contract"] != F.VERSION or report["contract"] != F.VERSION + "_report"
            or report["plan_sha256"] != plan["content_hash"]
            or report["event_count_per_model"] != plan["event_count"]
            or len(report["per_model_results"]) != 12):
        raise ValueError("Historical reference report binding differs")
    for task, row in zip(plan["tasks"], report["per_model_results"]):
        R._authenticated(row, "Historical nested result")
        R._metric(row["metrics"], plan)
        if (row["contract"] != F.VERSION + "_result" or row["task"] != task
                or row["plan_sha256"] != plan["content_hash"] or row["used_for_model_selection"] is not False
                or len(row["file_completion_hashes"]) != len(plan["files"])
                or row["file_completion_hashes"] != report["per_model_results"][0]["file_completion_hashes"]):
            raise ValueError("Historical nested result coverage differs")


def storage_projection(output, source_campaign, event_count):
    remaining = []
    for root, count in ((output / "test/predictions", 24), (source_campaign / "test/predictions", 12)):
        # Reserve uncompressed float32 bytes for not-yet-committed chunks. Do
        # not subtract compressed file size from an uncompressed total.
        committed_events = 0
        if root.exists():
            for path in root.glob("file_*/chunk_*.json"):
                receipt = F.read_json(path)
                if path.with_suffix(".npz").is_file():
                    meta = receipt["metadata"]
                    committed_events += meta["entry_stop"] - meta["entry_start"]
        remaining.append(max(0, event_count - committed_events) * count * 10 * 4)
    free = shutil.disk_usage(output if output.exists() else output.parent).free
    required = sum(remaining) + 6 * 1024**3
    result = {"free_bytes": free, "required_free_bytes": required,
              "unified_remaining_raw_logit_bytes": remaining[0], "tigris_remaining_raw_logit_bytes": remaining[1],
              "headroom_bytes": 6 * 1024**3, "automatic_deletion": False}
    if free < required:
        raise OSError(f"Need {required/1024**3:.2f} GiB free for unified + pending Tigris logits/headroom; "
                      f"have {free/1024**3:.2f} GiB. Nothing deleted.")
    return result


def bootstrap(args):
    source_campaign, output = args.source_campaign.resolve(), args.output.resolve()
    if output.exists() or not 1 <= args.jobs <= 200:
        raise ValueError("Use a NEW output and 1..200 inference tasks")
    source = F.read_json(source_campaign / "campaign_spec.json")
    if source["contract"] != "relational_part_offline_factorial_followup_v1":
        raise ValueError("Expected the locked offline factorial source campaign")
    old = F.read_json(source_campaign / "reference_plan.json")
    old_report = F.read_json(source_campaign / "reference_report.json")
    followup = F.read_json(source_campaign / "test/evaluation_plan.json")
    validate_reference(old, old_report)
    tasks = combined_tasks(old, followup)
    if (source["reference_plan_sha256"] != old["content_hash"]
            or source["reference_report_sha256"] != old_report["content_hash"]
            or followup["campaign_sha256"] != source["content_hash"] or old["event_count"] != 20_000_000):
        raise ValueError("Source campaign lineage differs")
    D.require_separate_output(output, [source_campaign, Path(source["parent"]),
        Path(source["previous_evaluation"]), Path(source["test_data"])])
    D.authenticate_files(source_campaign / "source", source["source_files"])
    environment = D.runtime_environment()
    if (environment["architecture"] != "x86_64" or not environment["torch_cuda_build"]
            or environment["weaver_particle_transformer_sha256"] != source["environment"]["weaver_particle_transformer_sha256"]):
        raise ValueError("Use the x86 CUDA SPORC environment with the trained Weaver source")
    storage = storage_projection(output, source_campaign, old["event_count"])
    output.mkdir()
    frozen = output / "source"
    for name, expected in source["source_files"].items():
        destination = frozen / D.safe_relative(name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_campaign / "source" / name, destination)
        if F.digest(destination) != expected:
            raise ValueError("Frozen source copy differs")
    for name in ADAPTER_FILES:
        destination = frozen / name
        if destination.exists():
            raise ValueError(f"Adapter would replace historical code: {name}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / name, destination)
    for name, value in (("reference_plan", old), ("reference_report", old_report)):
        F.write_json(output / (name + ".json"), value)
    spec = F.hashed({"contract": VERSION, "schema_version": 1, "output": str(output),
        "source_campaign": str(source_campaign), "source_campaign_sha256": source["content_hash"],
        "source_test_plan_sha256": followup["content_hash"], "source_models": source["models"],
        "original_parent": source["parent"], "previous_evaluation": source["previous_evaluation"],
        "reference_plan_sha256": old["content_hash"], "reference_report_sha256": old_report["content_hash"],
        "source_files": {**source["source_files"], **{name: F.digest(frozen / name) for name in ADAPTER_FILES}},
        "environment": environment, "source_environment": source["environment"], "tasks": tasks,
        "models": list(dict.fromkeys(t["run_id"] for t in tasks)), "seeds": list(F.SEEDS), "jobs": args.jobs,
        "math_options": MATH_OPTIONS, "validation_policy": POLICY, "inference_only": True,
        "performance_gate": False, "used_for_model_selection": False, "automatic_deletion": False,
        "primary_comparison": "all eight cells versus the newly evaluated SPORC baseline",
        "historical_predictions_reused": False, "platform_selection_by_scores": False})
    F.write_json(output / "unified_campaign.json", spec)
    F.write_json(output / "storage_projection.json", F.hashed(storage))
    (output / "logs").mkdir()
    subprocess.run([sys.executable, "-s", str(frozen / "scripts/run_relational_part_sporc_unified.py"),
                    "check", "--output", str(output)], cwd=frozen, check=True)
    F.progress("unified_bootstrapped", output=str(output), models=24, jobs=args.jobs)


def load_spec(output):
    from scripts import run_relational_part_offline_ablations as A
    spec = F.read_json(output / "unified_campaign.json")
    if (spec["contract"] != VERSION or spec["schema_version"] != 1 or spec["output"] != str(output)
            or REPO != output / "source" or spec["environment"] != D.runtime_environment()
            or spec["math_options"] != MATH_OPTIONS or spec["validation_policy"] != POLICY
            or not 1 <= spec["jobs"] <= 200 or spec["seeds"] != list(F.SEEDS)
            or spec["historical_predictions_reused"] is not False or spec["performance_gate"] is not False
            or spec["used_for_model_selection"] is not False or spec["platform_selection_by_scores"] is not False):
        raise ValueError("Unified source/runtime/policy differs")
    D.authenticate_files(REPO, spec["source_files"])
    if F.source_snapshot(REPO) != {n: h for n, h in spec["source_files"].items()
            if n == "jetclass_fixed_hlt.py" or n.startswith(("teacher_logit_reco/", "jetclass_fresh/"))}:
        raise ValueError("Frozen source inventory changed")
    source = A.load_campaign(Path(spec["source_campaign"]), runtime=False)
    followup = A.test_plan(Path(spec["source_campaign"]), source)
    old, report = (F.read_json(output / (name + ".json")) for name in ("reference_plan", "reference_report"))
    validate_reference(old, report)
    if (source["content_hash"] != spec["source_campaign_sha256"]
            or source["models"] != spec["source_models"] or source["parent"] != spec["original_parent"]
            or source["previous_evaluation"] != spec["previous_evaluation"]
            or followup["content_hash"] != spec["source_test_plan_sha256"]
            or old["content_hash"] != spec["reference_plan_sha256"] or report["content_hash"] != spec["reference_report_sha256"]
            or spec["tasks"] != combined_tasks(old, followup)
            or spec["models"] != list(dict.fromkeys(t["run_id"] for t in spec["tasks"]))):
        raise ValueError("Unified checkpoint/source/reference lineage differs")
    # Authenticate original checkpoint identities against the original lock too.
    _, tasks, _ = F.parent_artifacts(Path(spec["original_parent"]))
    if tasks != old["tasks"]:
        raise ValueError("Historical checkpoint lock changed")
    return spec, source, followup


def new_plan(spec, followup, backend):
    body = {k: copy.deepcopy(v) for k, v in followup.items() if k != "content_hash"}
    body.update(contract=VERSION + "_test", campaign_sha256=spec["content_hash"],
        tasks=spec["tasks"], source_campaign_sha256=spec["source_campaign_sha256"],
        backend_manifest_sha256=backend["content_hash"], environment=spec["environment"],
        validation_policy=POLICY, math_options=MATH_OPTIONS,
        primary_baseline_runtime="SPORC", historical_predictions_reused=False,
        assignments=F.file_assignment(followup["files"], spec["jobs"]))
    return F.hashed(body)


def load_ready(output):
    spec, source, followup = load_spec(output)
    backend, manifest = D.load_backend(output)
    plan = F.read_json(output / "test/evaluation_plan.json")
    if plan != new_plan(spec, followup, manifest):
        raise ValueError("Unified test plan changed")
    return spec, source, plan, backend


def build(args):
    spec, _, followup = load_spec(args.output)
    original = F.read_json(Path(spec["original_parent"]) / "backend/backend_manifest.json")
    if F.digest(REPO / D.TREE_SOURCE) != original["source_sha256"]:
        raise ValueError("Tree algorithm source differs")
    with F.exclusive(args.output / "locks/build.lock"):
        if not (args.output / "backend/backend_manifest.json").exists():
            subprocess.run([sys.executable, "-s", str(REPO / "scripts/build_relational_part_tree_backend.py"),
                "--contract", "relational_ca_tree_v1", "--build-dir", str(args.output / "build"),
                "--output-dir", str(args.output / "backend")], check=True)
        _, manifest = D.load_backend(args.output)
        F.write_json(args.output / "test/evaluation_plan.json", new_plan(spec, followup, manifest))


def configure_math():
    import torch
    D.configure_math()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load_models(spec, source, tasks, device):
    models = []
    for task in tasks:
        if task["origin"] not in ("historical", "followup"):
            raise ValueError("Unknown checkpoint origin")
        models.extend(D.load_models(spec, source, [task], device, historical=task["origin"] == "historical"))
    return models


def compare_logits(actual, expected):
    from scripts.diagnose_relational_part_sporc_parity import logit_difference
    row = logit_difference(actual, expected, atol=POLICY["diagnostic_atol"], rtol=POLICY["diagnostic_rtol"])
    if not row.get("finite") or not row.get("shape_matches") or not row.get("elements"):
        raise ValueError("Nonfinite, empty or malformed diagnostic logits")
    return row


def repeatability(actual, repeated):
    row = compare_logits(actual, repeated)
    if not row["passed"]:
        raise ValueError(f"Same-GPU repeated inference differs: {row}")
    return row


def validation_receipt(output, spec, plan, *, gpu=None):
    receipt = F.read_json(output / "runtime_validation.json")
    order = [[t["run_id"], t["seed"]] for t in plan["tasks"]]
    if (receipt["contract"] != VERSION + "_validation" or receipt["passed"] is not True
            or receipt["campaign_sha256"] != spec["content_hash"] or receipt["plan_sha256"] != plan["content_hash"]
            or receipt["policy"] != POLICY or receipt["model_order"] != order
            or receipt["cross_platform_equivalence_claimed"] is not False
            or (gpu is not None and gpu["capability"] != receipt["gpu"]["capability"])):
        raise ValueError("Unified runtime validation is missing or incompatible")
    probes = D.choose_probe_files(plan)
    checks = receipt["gpu_checks"]
    expected = {(index, model) for index in probes for model in range(24)}
    if len(checks) != 240 or {(v["file_index"], v["model_index"]) for v in checks} != expected:
        raise ValueError("Incomplete all-class/all-model GPU validation")
    for row in checks:
        check = row["repeatability"]
        if (check["passed"] is not True or check["finite"] is not True or check["shape_matches"] is not True
                or check["elements"] != POLICY["probe_events_per_class"] * 10
                or check["outside_tolerance_count"] != 0):
            raise ValueError("Invalid repeated-inference validation")
    cpu = receipt["cpu_gpu_diagnostics"]
    wanted = {(i, m, s) for i in probes for m in range(24) for s in POLICY["states"]}
    if len(cpu) != len(wanted) or {(v["file_index"], v["model_index"], v["state"]) for v in cpu} != wanted:
        raise ValueError("Incomplete CPU/GPU diagnostics")
    for row in cpu:
        if not row["comparison"]["finite"] or not row["comparison"]["shape_matches"] or row["comparison"]["elements"] != 80:
            raise ValueError("Malformed CPU/GPU diagnostic")
    trees = receipt["tree_checks"]
    if (len(trees) != len(probes) * POLICY["tree_events_per_class"]
            or {(v["file_index"], v["entry"]) for v in trees} != {(i, j) for i in probes for j in range(POLICY["tree_events_per_class"])}
            or any(v["passed"] is not True or v["topology_and_categories_exact"] is not True
                   or not v["continuous_shapes_exact"] or not v["continuous_values_finite"]
                   or not 0 <= v["maximum_continuous_absolute_error"] <= POLICY["tree_continuous_atol"] for v in trees)):
        raise ValueError("Incomplete compiled/reference tree validation")
    return receipt


def validate(args):
    import numpy as np
    import torch
    import uproot
    from jetclass_fresh.jetclass_data import PARTICLE_READ_BRANCHES, LABEL_BRANCHES, _tokens_from_arrays
    from jetclass_fresh.part_inputs import build_particle_transformer_inputs_from_tokens
    from teacher_logit_reco.relational_part.ca_tree import build_compiled_tree
    from teacher_logit_reco.relational_part.region_tree import build_reference_tree
    from scripts.build_relational_part_tree_backend import _canonical_smoke_parity
    spec, source, plan, backend = load_ready(args.output)
    configure_math()
    gpu, device = D.gpu_identity(), torch.device("cuda")
    with F.exclusive(args.output / "locks/validation.lock"):
        if (args.output / "runtime_validation.json").exists():
            validation_receipt(args.output, spec, plan, gpu=gpu)
            return
        models = load_models(spec, source, plan["tasks"], device)
        torch.cuda.reset_peak_memory_stats()
        samples, warmed, checks, trees, timings = [], [], [], [], []
        gpu_small = {}
        probes = D.choose_probe_files(plan)
        for index in probes:
            row = plan["files"][index]
            path = Path(plan["data_dir"]) / row["name"]
            if F.digest(path) != row["sha256"]:
                raise ValueError("Probe ROOT bytes changed")
            with uproot.open(path) as root:
                arrays = root["tree"].arrays(PARTICLE_READ_BRANCHES + LABEL_BRANCHES,
                    entry_start=0, entry_stop=POLICY["probe_events_per_class"], library="ak")
            if len(arrays) != POLICY["probe_events_per_class"]:
                raise ValueError("Incomplete probe")
            raw, mask = _tokens_from_arrays(arrays[:POLICY["tree_events_per_class"]], 128)
            inputs = build_particle_transformer_inputs_from_tokens(raw, mask, source_view="offline")
            for entry in range(len(raw)):
                compiled = build_compiled_tree(backend, inputs.pf_vectors[entry].T, raw[entry], mask[entry])
                reference = build_reference_tree(inputs.pf_vectors[entry].T, raw[entry], mask[entry])
                comparison = _canonical_smoke_parity(compiled, reference)
                if not comparison["passed"]:
                    raise ValueError(f"Compiled/reference tree mismatch: {row['name']}:{entry}: {comparison}")
                trees.append(dict(file_index=index, entry=entry, **comparison))
            # F.infer_arrays resets fresh trimmer counters at every call.
            torch.cuda.synchronize()
            start = time.monotonic()
            actual = F.infer_arrays(arrays, row["label"], models, backend, device)
            torch.cuda.synchronize()
            timings.append(dict(file_index=index, events=len(arrays), seconds=time.monotonic() - start))
            states = [D.trimmer_state(model) for model in models]
            repeated = F.infer_arrays(arrays, row["label"], models, backend, device)
            for i in range(24):
                key = f"logits_{i:02d}"
                checks.append(dict(file_index=index, model_index=i, repeatability=repeatability(actual[key], repeated[key])))
            sample = arrays[:POLICY["cpu_gpu_events_per_class"]]
            fresh_states = [copy.deepcopy({name: getattr(module, "_rpt_full_test_initial_counter")
                for name, module in model.named_modules() if hasattr(module, "_rpt_full_test_initial_counter")}) for model in models]
            gpu_small[index, "fresh"] = F.infer_arrays(sample, row["label"], models, backend, device)
            for model, state in zip(models, states):
                D.probe_trimmer_state(model, state)
            gpu_small[index, "after_ten_full_batches"] = F.infer_arrays(sample, row["label"], models, backend, device)
            for model, state in zip(models, fresh_states):
                D.probe_trimmer_state(model, state)
            samples.append((index, row["label"], sample))
            warmed.append(states)
            F.progress("unified_probe", file=row["name"], models=24, repeatability_passed=True)
        peak = dict(allocated=torch.cuda.max_memory_allocated(), reserved=torch.cuda.max_memory_reserved())
        del models, actual, repeated
        gc.collect()
        torch.cuda.empty_cache()
        cpu_rows = []
        for i, task in enumerate(plan["tasks"]):
            cpu_models = load_models(spec, source, [task], torch.device("cpu"))
            initial = D.trimmer_state(cpu_models[0])
            for (index, label, sample), states in zip(samples, warmed):
                for state in POLICY["states"]:
                    D.probe_trimmer_state(cpu_models[0], initial if state == "fresh" else states[i])
                    values = F.infer_arrays(sample, label, cpu_models, backend, torch.device("cpu"))
                    cpu_rows.append(dict(file_index=index, model_index=i, state=state,
                        comparison=compare_logits(gpu_small[index, state][f"logits_{i:02d}"], values["logits_00"])))
            del cpu_models
            F.progress("unified_cpu_diagnostic", model_index=i, gates_on_agreement=False)
        largest = max(sum(plan["files"][i]["entries"] for i in group) for group in plan["assignments"])
        receipt = F.hashed({"contract": VERSION + "_validation", "campaign_sha256": spec["content_hash"],
            "plan_sha256": plan["content_hash"], "passed": True, "policy": POLICY, "gpu": gpu,
            "model_order": [[t["run_id"], t["seed"]] for t in plan["tasks"]],
            "gpu_checks": checks, "tree_checks": trees, "cpu_gpu_diagnostics": cpu_rows,
            "timings": timings, "peak_gpu_memory_bytes": peak, "cross_platform_equivalence_claimed": False,
            "estimated_largest_task_seconds_excluding_io": largest * max(t["seconds"] / t["events"] for t in timings),
            "timing_is_estimate_not_performance_gate": True})
        F.write_json(args.output / "runtime_validation.json", receipt)
        validation_receipt(args.output, spec, plan, gpu=gpu)
        F.progress("unified_runtime_validated", cpu_gpu_outside_tolerance=sum(not r["comparison"]["passed"] for r in cpu_rows))


def read_chunk(path, metadata, *, keys=None):
    import numpy as np
    record = F.read_json(path.with_suffix(".json"))
    if record["metadata"] != metadata or F.digest(path) != record["npz_sha256"]:
        raise ValueError("Unified prediction lineage/bytes differ")
    wanted = {f"logits_{i:02d}" for i in range(24)}
    if len(metadata["model_order"]) != 24 or (keys is not None and not set(keys) <= wanted):
        raise ValueError("Expected unified 24-checkpoint metadata/keys")
    with np.load(path, allow_pickle=False) as values:
        if len(values.files) != 24 or set(values.files) != wanted:
            raise ValueError("Unified chunk must contain exactly 24 predictions")
        result = {key: values[key] for key in (wanted if keys is None else keys)}
    for value in result.values():
        if value.shape != (metadata["entry_stop"] - metadata["entry_start"], 10) or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError("Invalid unified prediction shape/dtype/finiteness")
    return result, record


def infer(args):
    import numpy as np
    import torch
    import uproot
    from jetclass_fresh.jetclass_data import PARTICLE_READ_BRANCHES, LABEL_BRANCHES
    spec, source, plan, backend = load_ready(args.output)
    configure_math()
    gpu = D.gpu_identity()
    receipt = validation_receipt(args.output, spec, plan, gpu=gpu)
    if not 0 <= args.task_index < len(plan["assignments"]):
        raise IndexError("Inference task outside frozen assignments")
    storage_projection(args.output, Path(spec["source_campaign"]), plan["event_count"])
    output = args.output / "test"
    with F.exclusive(output / "locks" / f"gpu_{args.task_index}.lock"):
        models = None
        for index in plan["assignments"][args.task_index]:
            row = plan["files"][index]
            data = Path(plan["data_dir"]) / row["name"]
            if F.digest(data) != row["sha256"]:
                raise ValueError("ROOT file bytes changed")
            records = []
            with uproot.open(data) as root:
                for start, stop in F.chunks(row, plan):
                    path = F.chunk_path(output, index, start)
                    metadata = F.chunk_metadata(plan, index, start, stop)
                    if path.exists() and path.with_suffix(".json").exists():
                        _, record = read_chunk(path, metadata, keys=set())
                    else:
                        if path.with_suffix(".json").exists():
                            raise ValueError("Committed predictions are missing; refusing silent replacement")
                        if models is None:
                            models = load_models(spec, source, plan["tasks"], torch.device("cuda"))
                        clock = time.monotonic()
                        F.progress("starting_chunk", file=row["name"], start=start, stop=stop, models=24)
                        arrays = root["tree"].arrays(PARTICLE_READ_BRANCHES + LABEL_BRANCHES,
                            entry_start=start, entry_stop=stop, library="ak")
                        values = F.infer_arrays(arrays, row["label"], models, backend, torch.device("cuda"))
                        with F.atomic_file(path) as stream:
                            np.savez_compressed(stream, **values)
                        record = F.hashed({"metadata": metadata, "npz_sha256": F.digest(path),
                            "runtime_validation_sha256": receipt["content_hash"], "gpu": gpu,
                            "slurm_job_id": os.environ.get("SLURM_JOB_ID")})
                        F.write_json(path.with_suffix(".json"), record)
                        F.progress("evaluated_chunk", file=row["name"], start=start, stop=stop, models=24,
                                   seconds=time.monotonic() - clock)
                    if record["runtime_validation_sha256"] != receipt["content_hash"]:
                        raise ValueError("Prediction chunk has another runtime validation")
                    records.append(record["content_hash"])
            F.write_json(output / "predictions" / f"file_{index:03d}/complete.json", F.hashed({
                "contract": plan["contract"] + "_file_complete", "plan_sha256": plan["content_hash"],
                "file_index": index, "chunks": records}))


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
    elif args.phase in ("aggregate", "report"):
        from scripts import relational_part_sporc_unified_reporting as reporting
        spec, _, plan, _ = load_ready(args.output)
        receipt = validation_receipt(args.output, spec, plan)
        if args.phase == "aggregate":
            reporting.aggregate(args.output, spec, plan, receipt, args.task_index)
        else:
            reporting.report(args.output, spec, plan, receipt)
    else:
        globals()[args.phase](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
