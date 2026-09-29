#!/usr/bin/env python3
"""Restartable, storage-bounded follow-up to the completed offline RPT study.

All mutations are confined to a NEW campaign. Existing checkpoints, reports,
predictions, raw ROOT files, and the historical runtime are read-only inputs.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import random
import shutil
import statistics
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import evaluate_relational_part_offline_full_test as F

VERSION = "relational_part_offline_factorial_followup_v1"
NEW_FILES = (
    "teacher_logit_reco/relational_part/offline_ablations.py",
    "teacher_logit_reco/relational_part/offline_ablation_data.py",
    "scripts/run_relational_part_offline_ablations.py",
    "scripts/evaluate_relational_part_offline_full_test.py",
    "sbatch/run_rpt_offline_ablations.sh",
)
EXTRA_PARENTS = ("inputs/offline_cache_binding.json", "registry/global_determinism.json",
                 "registry/relation_family_registry.json")


def source_files(source):
    return {**F.source_snapshot(source),
            **{name: F.digest(source / name) for name in NEW_FILES}}


def bootstrap(args):
    """Copy the old authenticated runtime, overlay only the new implementation."""
    from teacher_logit_reco.relational_part.offline_ablations import MODEL_SPECS
    parent, previous, output = (p.resolve() for p in (args.parent, args.previous, args.output))
    if output.exists():
        raise FileExistsError("New output already exists; use the submitter's --resume")
    if output.is_relative_to(parent) or output.is_relative_to(previous):
        raise ValueError("Supplemental campaign must be separate from both original campaigns")
    campaign, original_tasks, artifacts = F.parent_artifacts(parent)
    plan = F.read_json(previous / "evaluation_plan.json")
    report = F.read_json(previous / "reports/full_test_report.json")
    if (plan["contract"] != F.VERSION or plan["parent_campaign_sha256"] != campaign["content_hash"]
            or report["plan_sha256"] != plan["content_hash"] or plan["tasks"] != original_tasks
            or report["event_count_per_model"] != 20_000_000):
        raise ValueError("Original 20M report/plan is not bound to the offline checkpoints")
    for relative, expected in plan["parent_artifacts"].items():
        if F.digest(parent / relative) != expected:
            raise ValueError(f"Original evaluation parent differs: {relative}")
    for relative in EXTRA_PARENTS:
        F.read_json(parent / relative)
        artifacts[relative] = F.digest(parent / relative)
    # Authenticate all old completion receipts now; logits are rehashed at aggregation.
    if len(report["per_model_results"]) != 12:
        raise ValueError("Original report lacks all twelve results")
    for task, row in zip(original_tasks, report["per_model_results"]):
        if (row != F.hashed({k: v for k, v in row.items() if k != "content_hash"}) or row["task"] != task
                or row["plan_sha256"] != plan["content_hash"]
                or row["metrics"] != F.hashed({k: v for k, v in row["metrics"].items() if k != "content_hash"})):
            raise ValueError("Original report nested result authentication differs")
    completions = []
    for index, row in enumerate(plan["files"]):
        if not (args.test_data / row["name"]).is_file():
            raise FileNotFoundError(f"Missing extracted test file: {row['name']}")
        done = F.read_json(previous / "predictions" / f"file_{index:03d}" / "complete.json")
        records = []
        for start, stop in F.chunks(row, plan):
            path = F.chunk_path(previous, index, start)
            sidecar = F.read_json(path.with_suffix(".json"))
            if not path.is_file() or sidecar["metadata"] != F.chunk_metadata(plan, index, start, stop):
                raise ValueError("Original prediction files/lineage are absent or inconsistent")
            records.append(sidecar["content_hash"])
        if done != F.hashed({"contract": F.VERSION + "_file_complete", "plan_sha256": plan["content_hash"],
                             "file_index": index, "chunks": records}):
            raise ValueError("Original prediction completion differs")
        completions.append(done["content_hash"])
    if any(r["file_completion_hashes"] != completions for r in report["per_model_results"]):
        raise ValueError("Original results differ from prediction coverage")
    old_source = Path(plan["source_root"])
    for name, expected in plan["source_files"].items():
        if F.digest(old_source / name) != expected:
            raise ValueError(f"Original pinned runtime changed: {name}")
    if shutil.disk_usage(output.parent).free < 25 * 1024**3:
        raise OSError("Need at least 25 GiB free to stage shared caches and new logits; trees are budgeted after caching")
    output.mkdir()
    source = output / "source"
    for name in plan["source_files"]:
        destination = source / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(old_source / name, destination)
    for name in NEW_FILES:
        destination = source / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / name, destination)
    F.write_json(output / "reference_plan.json", plan)
    F.write_json(output / "reference_report.json", report)
    spec = F.hashed({"contract": VERSION, "schema_version": 1,
        "parent": str(parent), "previous_evaluation": str(previous),
        "train_data": str(args.train_data.resolve()), "test_data": str(args.test_data.resolve()),
        "parent_artifacts": artifacts, "parent_campaign_sha256": campaign["content_hash"],
        "reference_plan_sha256": plan["content_hash"], "reference_report_sha256": report["content_hash"],
        "original_source_files": plan["source_files"], "source_files": source_files(source),
        "environment": plan["environment"], "models": MODEL_SPECS, "seeds": list(F.SEEDS),
        "training": "exact original TrainingConfig, event identities/order, BF16, 1M jets; from scratch",
        "normalization": "reuse original offline training-only artifacts unchanged",
        "selection": "original model_val checkpoint rule; all 12 runs evaluated regardless of metrics",
        "performance_gate": False, "followup_after_inspection_of_500k_and_20M_results": True,
        "shared_ev": "original shared PairEmbed packing, BN population and Weaver trimmer; independent layer/head EV projections",
        "layerwise": "unchanged original confirmation implementation; historical pipeline differences are retained",
        "test_jobs": 20, "prediction_bytes_uncompressed": 20_000_000 * 12 * 10 * 4,
        "headroom_bytes": 6 * 1024**3, "automatic_dataset_deletion": False})
    F.write_json(output / "campaign_spec.json", spec)
    (output / "logs").mkdir()
    F.progress("bootstrapped", output=str(output), training_tasks=12, test_tasks=20)


def load_campaign(output, *, runtime=True):
    from teacher_logit_reco.relational_part.offline_ablations import MODEL_SPECS
    spec = F.read_json(output / "campaign_spec.json")
    if spec["contract"] != VERSION or spec["models"] != MODEL_SPECS or spec["seeds"] != list(F.SEEDS):
        raise ValueError("Follow-up contract/matrix differs")
    if source_files(output / "source") != spec["source_files"]:
        raise ValueError("Frozen follow-up source changed")
    if runtime and (REPO != output / "source" or F.environment() != spec["environment"]):
        raise ValueError("Use the campaign's frozen source and original Conda/Weaver environment")
    for name, expected in spec["parent_artifacts"].items():
        if F.digest(Path(spec["parent"]) / name) != expected:
            raise ValueError(f"Read-only original artifact changed: {name}")
    for name, expected in (("reference_plan", spec["reference_plan_sha256"]),
                           ("reference_report", spec["reference_report_sha256"])):
        if F.read_json(output / (name + ".json"))["content_hash"] != expected:
            raise ValueError("Copied reference artifact differs")
    return spec


def parent_json(spec, name):
    return F.read_json(Path(spec["parent"]) / name)


def backend_for(spec):
    from teacher_logit_reco.relational_part.ca_tree import load_tree_backend
    manifest = parent_json(spec, "backend/backend_manifest.json")
    parent = Path(spec["parent"])
    return load_tree_backend(parent / "backend" / manifest["binary_filename"],
                             parent / "backend/backend_manifest.json",
                             source_path=REPO / "teacher_logit_reco/relational_part/csrc/relational_ca_tree_v1.cpp")


def model_for(spec, run_id):
    from teacher_logit_reco.relational_part.offline_ablations import build_ablation_model
    return build_ablation_model(run_id,
        normalization_artifact=parent_json(spec, "inputs/relation_normalization.json"),
        region_normalization_artifact=parent_json(spec, "inputs/region_normalization.json"))


def prepare(args):
    from dataclasses import asdict
    from teacher_logit_reco.relational_part.train import TrainingConfig
    from teacher_logit_reco.relational_part.offline_ablations import MODEL_SPECS, SELECTED
    spec = load_campaign(args.output)
    determinism = parent_json(spec, "registry/global_determinism.json")
    for seed in F.SEEDS:
        old = parent_json(spec, f"runs/OFF_RPT_BASE/seed_{seed}/checkpoint_registration.json")
        if TrainingConfig(seed=seed).artifact(global_determinism_sha256=determinism["content_hash"])["content_hash"] != old["training_contract_sha256"]:
            raise ValueError("Training protocol no longer matches original checkpoint")
        if old["precision_mode"] != "bf16":
            raise ValueError("Expected original CUDA BF16 training")
    binding = parent_json(spec, "inputs/offline_cache_binding.json")
    if binding["campaign_sha256"] != spec["parent_campaign_sha256"]:
        raise ValueError("Original offline cache binding belongs to another campaign")
    contracts, tasks = {}, []
    for run_id, architecture in MODEL_SPECS.items():
        contract = F.hashed({"contract": VERSION + "_model", "run_id": run_id, **architecture,
            "campaign_sha256": spec["content_hash"], "relation_families": list(SELECTED) if architecture["features"] == "selected" else [],
            "relation_normalization_sha256": parent_json(spec, "inputs/relation_normalization.json")["content_hash"],
            "region_normalization_sha256": parent_json(spec, "inputs/region_normalization.json")["content_hash"],
            "edge_value_projection_sharing": "independent_per_layer_per_head" if architecture["edge_value"] else "absent"})
        F.write_json(args.output / "registry" / (run_id + ".json"), contract)
        contracts[run_id] = contract["content_hash"]
        for seed in F.SEEDS:
            tasks.append({"task_index": len(tasks), "run_id": run_id, "seed": seed,
                          "model_contract_sha256": contract["content_hash"], "relation_families": contract["relation_families"],
                          "training_config": asdict(TrainingConfig(seed=seed))})
    F.write_json(args.output / "training_tasks.json", F.hashed({"contract": VERSION + "_tasks",
                 "campaign_sha256": spec["content_hash"], "tasks": tasks, "performance_gate": False}))
    backend_for(spec)  # Authentication and self-test, without compiling/rebuilding.
    F.progress("prepared", tasks=len(tasks))


def cache(args):
    import numpy as np
    from jetclass_fresh.jetclass_data import load_split_manifest, load_offline_view
    from jetclass_fresh.hlt_cache import hash_arrays, jet_identity_hash
    from teacher_logit_reco.relational_part.offline_ablation_data import SPLITS, view_arrays, open_arrays
    spec = load_campaign(args.output)
    split = list(SPLITS)[args.task_index]
    destination = args.output / "inputs/cache" / split
    expected = parent_json(spec, "inputs/offline_cache_binding.json")["splits"][split]
    with F.exclusive(destination / "build.lock"):
        receipt_path = destination / "receipt.json"
        if receipt_path.exists():
            receipt = F.read_json(receipt_path)
            if receipt["campaign_sha256"] != spec["content_hash"] or receipt["original_binding"] != expected:
                raise ValueError("Existing cache lineage differs")
            open_arrays(destination, receipt)
            F.progress("cache_reused", split=split)
            return
        manifest = load_split_manifest(Path(spec["parent"]) / "inputs/split_manifest.json.gz")
        if manifest.max_constits != 128 or len(manifest.splits[split]) != SPLITS[split]:
            raise ValueError("Original split size/particle width differs")
        F.progress("cache_loading_root", split=split)
        view = load_offline_view(manifest, split, data_dir=spec["train_data"], verify_label_branches=True,
                                 read_chunk_size=10_000)
        files, arrays = view_arrays(view)
        if (hash_arrays(arrays) != expected["offline_content_sha256"]
                or jet_identity_hash(view.jet_ids) != expected["event_identity_sha256"]):
            raise ValueError("Rebuilt tokens/identity order differ from ORIGINAL offline training bytes")
        digests = {}
        for name, values in arrays.items():
            path = destination / (name + ".npy")
            with F.atomic_file(path) as stream:
                np.save(stream, values, allow_pickle=False)
            digests[path.name] = F.digest(path)
        # Conservative uncompressed tree bound; NPZ storage is normally smaller.
        nodes = int(np.maximum(2 * arrays["mask"].sum(axis=1, dtype=np.int64) - 1, 0).sum())
        receipt = F.hashed({"contract": VERSION + "_cache", "campaign_sha256": spec["content_hash"],
            "split": split, "original_binding": expected, "jet_files": files, "files": digests,
            "tree_storage_bound_bytes": nodes * 64 + len(view.tokens) * (128 * 4 * 4 + 1024),
            "event_count": len(view.tokens)})
        F.write_json(receipt_path, receipt)
        F.progress("cache_complete", split=split, events=len(view.tokens))


def storage(args):
    from teacher_logit_reco.relational_part.offline_ablation_data import SPLITS
    spec = load_campaign(args.output)
    required_trees = sum(F.read_json(args.output / "inputs/cache" / split / "receipt.json")["tree_storage_bound_bytes"] for split in SPLITS)
    existing = sum(p.stat().st_size for p in (args.output / "inputs/trees").rglob("*.npz"))
    predictions = sum(p.stat().st_size for p in (args.output / "test/predictions").rglob("*.npz"))
    remaining = max(0, required_trees - existing) + max(0, spec["prediction_bytes_uncompressed"] - predictions) + spec["headroom_bytes"]
    free = shutil.disk_usage(args.output).free
    F.progress("storage_budget", remaining_bytes=remaining, available_bytes=free,
               cache_copies=1, persistent_dense_pair_features=False)
    if free < remaining:
        raise OSError("Insufficient remaining space for compact trees, test logits and 6 GiB headroom; no data was deleted")


def trees(args):
    import numpy as np
    from jetclass_fresh.part_inputs import build_particle_transformer_inputs_from_tokens
    from teacher_logit_reco.relational_part.ca_tree import build_compiled_tree, pack_tree_shard
    from teacher_logit_reco.relational_part.offline_ablation_data import tree_tasks, open_arrays, identity_at
    spec = load_campaign(args.output)
    task = tree_tasks()[args.task_index]
    directory = args.output / "inputs/cache" / task["split"]
    receipt = F.read_json(directory / "receipt.json")
    if receipt["campaign_sha256"] != spec["content_hash"]:
        raise ValueError("Tree input cache belongs to another campaign")
    path = args.output / "inputs/trees" / task["split"] / f"shard_{task['shard']:03d}.npz"
    lineage = {"campaign_sha256": spec["content_hash"], "task": task, "cache_sha256": receipt["content_hash"]}
    with F.exclusive(path.with_suffix(".lock")):
        if path.with_suffix(".json").exists():
            old = F.read_json(path.with_suffix(".json"))
            if old["lineage"] != lineage or old["npz_sha256"] != F.digest(path):
                raise ValueError("Existing tree shard is stale")
            return
        # A 126-task array must not hash/decompress all 9GB of tokens 126 times.
        # Cache was authenticated when built; final bind rehashes it before training.
        arrays = open_arrays(directory, receipt, verify=False)
        start, stop = task["start"], task["stop"]
        raw, mask = arrays["tokens"][start:stop], arrays["mask"][start:stop]
        inputs = build_particle_transformer_inputs_from_tokens(raw, mask, source_view="offline")
        vectors = inputs.pf_vectors.transpose(0, 2, 1)
        backend = backend_for(spec)
        built = [build_compiled_tree(backend, vectors[i], raw[i], mask[i]) for i in range(stop - start)]
        identities = [identity_at(arrays, receipt["jet_files"], i) for i in range(start, stop)]
        with F.atomic_file(path) as stream:
            np.savez_compressed(stream, **pack_tree_shard(built, identities))
        F.write_json(path.with_suffix(".json"), F.hashed({"contract": VERSION + "_trees",
                     "lineage": lineage, "npz_sha256": F.digest(path)}))
        F.progress("trees_complete", **task)


def bind(args):
    from teacher_logit_reco.relational_part.offline_ablation_data import SPLITS, tree_tasks, open_arrays
    spec = load_campaign(args.output)
    splits = {}
    for split in SPLITS:
        directory = args.output / "inputs/cache" / split
        receipt = F.read_json(directory / "receipt.json")
        if (receipt["campaign_sha256"] != spec["content_hash"]
                or receipt["original_binding"] != parent_json(spec, "inputs/offline_cache_binding.json")["splits"][split]):
            raise ValueError("Token cache differs from original binding")
        open_arrays(directory, receipt)
        hashes = []
        for task in (t for t in tree_tasks() if t["split"] == split):
            path = args.output / "inputs/trees" / split / f"shard_{task['shard']:03d}.npz"
            tree = F.read_json(path.with_suffix(".json"))
            if (tree["lineage"] != {"campaign_sha256": spec["content_hash"], "task": task, "cache_sha256": receipt["content_hash"]}
                    or tree["npz_sha256"] != F.digest(path)):
                raise ValueError("Tree coverage or source changed")
            hashes.append(tree["content_hash"])
        splits[split] = {"cache": receipt["content_hash"], "trees": hashes}
    F.write_json(args.output / "data_binding.json", F.hashed({"contract": VERSION + "_data_binding",
                 "campaign_sha256": spec["content_hash"], "splits": splits}))
    F.progress("inputs_bound", splits=list(splits))


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def registered_tasks(output, spec):
    from dataclasses import asdict
    from teacher_logit_reco.relational_part.train import TrainingConfig
    from teacher_logit_reco.relational_part.offline_ablations import MODEL_SPECS
    registry = F.read_json(output / "training_tasks.json")
    wanted = [(name, seed) for name in MODEL_SPECS for seed in F.SEEDS]
    if (registry["campaign_sha256"] != spec["content_hash"] or registry["performance_gate"] is not False
            or [(t["run_id"], t["seed"]) for t in registry["tasks"]] != wanted):
        raise ValueError("Follow-up task registry differs")
    for index, task in enumerate(registry["tasks"]):
        contract = F.read_json(output / "registry" / (task["run_id"] + ".json"))
        architecture = MODEL_SPECS[task["run_id"]]
        families = ["PT", "TRACK", "REGION"] if architecture["features"] == "selected" else []
        if (task["task_index"] != index or task["training_config"] != asdict(TrainingConfig(seed=task["seed"]))
                or task["relation_families"] != families or contract["relation_families"] != families
                or contract["campaign_sha256"] != spec["content_hash"]
                or any(contract[key] != value for key, value in architecture.items())
                or task["model_contract_sha256"] != contract["content_hash"]):
            raise ValueError("Task architecture/training contract differs")
    return registry


def verify_registration(output, spec, task):
    from teacher_logit_reco.relational_part.train import TrainingConfig
    path = output / "runs" / task["run_id"] / f"seed_{task['seed']}"
    reg = F.read_json(path / "checkpoint_registration.json")
    contract = F.read_json(output / "registry" / (task["run_id"] + ".json"))
    binding = F.read_json(output / "data_binding.json")
    if (reg["run_id"] != task["run_id"] or reg["seed"] != task["seed"]
            or reg["model_contract_sha256"] != task["model_contract_sha256"]
            or contract["content_hash"] != task["model_contract_sha256"]
            or contract["campaign_sha256"] != spec["content_hash"]
            or reg["lineage_hashes"]["campaign_spec"] != spec["content_hash"]
            or reg["run_registry_sha256"] != registered_tasks(output, spec)["content_hash"]
            or binding["campaign_sha256"] != spec["content_hash"]
            or reg["lineage_hashes"]["data_binding"] != binding["content_hash"]
            or any(reg["lineage_hashes"][name] != parent_json(spec, "inputs/" + name + ".json")["content_hash"]
                   for name in ("relation_normalization", "region_normalization"))
            or reg["training_contract_sha256"] != TrainingConfig(seed=task["seed"]).artifact(
                global_determinism_sha256=parent_json(spec, "registry/global_determinism.json")["content_hash"])["content_hash"]
            or reg["checkpoint_sha256"] != F.digest(path / "best_model_val.pt")
            or reg["offline_tagger_inference"] is not True or reg["precision_mode"] != "bf16"
            or reg["val_select_used_for_checkpoint_selection"] is not False):
        raise ValueError(f"Checkpoint registration/lineage differs: {path}")
    metrics = F.read_json(path / "val_select_metrics.json")
    if metrics["content_hash"] != reg["val_select_metrics_sha256"]:
        raise ValueError("Validation metric registration differs")
    return reg


def train(args):
    import torch
    from teacher_logit_reco.relational_part.data import make_relational_loader
    from teacher_logit_reco.relational_part.offline_ablation_data import AblationDataset, SPLITS
    from teacher_logit_reco.relational_part.profiling import profile_model_resources
    from teacher_logit_reco.relational_part.train import TrainingConfig, train_relational_model, resolve_precision
    spec = load_campaign(args.output)
    registry = registered_tasks(args.output, spec)
    task = registry["tasks"][args.task_index]
    ready = F.read_json(args.output / "runtime_validation.json")
    if ready["campaign_sha256"] != spec["content_hash"] or not ready["passed"]:
        raise ValueError("Real-Weaver architecture validation has not passed")
    destination = args.output / "runs" / task["run_id"] / f"seed_{task['seed']}"
    with F.exclusive(destination / "training.lock"):
        if (destination / "checkpoint_registration.json").exists():
            verify_registration(args.output, spec, task)
            F.progress("training_reused", run_id=task["run_id"], seed=task["seed"])
            return
        if not torch.cuda.is_available() or resolve_precision(torch.device("cuda"))["mode"] != "bf16":
            raise RuntimeError("Matched training requires a BF16-capable CUDA GPU")
        binding = F.read_json(args.output / "data_binding.json")
        if binding["campaign_sha256"] != spec["content_hash"]:
            raise ValueError("Training data binding differs")
        seed_all(task["seed"])
        model = model_for(spec, task["run_id"])
        loaders = [make_relational_loader(AblationDataset(args.output, split,
                   uses_region=bool(task["relation_families"]), binding=binding), seed=task["seed"],
                   training=split == "model_train") for split in SPLITS]
        profile_path = destination / "resource_profile.json"
        if profile_path.exists():
            profile = F.read_json(profile_path)
        else:
            profile = profile_model_resources(model, next(iter(loaders[1])), device="cuda",
                                               model_contract_sha256=task["model_contract_sha256"])
            F.write_json(profile_path, profile)
        if profile["model_contract_sha256"] != task["model_contract_sha256"]:
            raise ValueError("Resource profile model binding differs")
        train_relational_model(model=model, train_loader=loaders[0], val_stop_loader=loaders[1],
            val_select_loader=loaders[2], output_dir=destination, run_id=task["run_id"],
            model_contract_sha256=task["model_contract_sha256"], run_registry_sha256=registry["content_hash"],
            relation_registry_sha256=parent_json(spec, "registry/relation_family_registry.json")["content_hash"],
            global_determinism_sha256=parent_json(spec, "registry/global_determinism.json")["content_hash"],
            lineage_hashes={"campaign_spec": spec["content_hash"], "data_binding": binding["content_hash"],
                "relation_normalization": parent_json(spec, "inputs/relation_normalization.json")["content_hash"],
                "region_normalization": parent_json(spec, "inputs/region_normalization.json")["content_hash"]},
            config=TrainingConfig(seed=task["seed"]), device="cuda", resource_profile=profile,
            resume=True, inference_input_role="offline_tagger")
        verify_registration(args.output, spec, task)
        F.progress("training_complete", run_id=task["run_id"], seed=task["seed"])


def zero_message_parity(reference, batch, *, device):
    """FP32 reference equality, including active trimmer and input/weight gradients."""
    import torch
    from teacher_logit_reco.relational_part.offline_ablations import SharedBiasEdgeValue
    from teacher_logit_reco.relational_part.evaluation import model_forward
    # Python module objects (the installed Weaver import) cannot be pickled by
    # deepcopy. They are immutable interfaces, not model state to duplicate.
    interface = reference._weaver_module
    candidate = SharedBiasEdgeValue(copy.deepcopy(reference, memo={id(interface): interface})).to(device)
    reference = reference.to(device)
    for edge in candidate.edge_attention:
        edge.edge_projection.data.zero_()
    trimmers = [reference.mod.trimmer, candidate.reference.mod.trimmer]
    if not all(t.enabled for t in trimmers):
        raise AssertionError("Parity must exercise enabled reference trimming")
    for training in (False, True):
        reference.train(training)
        candidate.train(training)
        for model in (reference, candidate):
            for name, child in model.named_modules():
                if name.split(".")[-1] == "trimmer" and hasattr(child, "_counter"):
                    if isinstance(child._counter, torch.Tensor):
                        child._counter.fill_(6)
                    else:
                        child._counter = 6
        outputs, inputs, gradients = [], [], []
        for model in (reference, candidate):
            model.zero_grad(set_to_none=True)
            independent = {k: v.detach().clone().to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            independent["features"].requires_grad_(True)
            # Weaver's trimming quantile uses Python random; its permutation
            # and attention dropout use Torch. Reset both paths identically.
            seed_all(90210)
            with torch.autocast(torch.device(device).type, enabled=False):
                output = model_forward(model, independent)
                output.square().mean().backward()
            outputs.append(output.detach())
            inputs.append(independent["features"].grad)
            gradients.append([p.grad for name, p in model.named_parameters() if not name.endswith("edge_projection")])
        torch.testing.assert_close(outputs[0], outputs[1], atol=5e-5, rtol=5e-5)
        torch.testing.assert_close(inputs[0], inputs[1], atol=1e-4, rtol=1e-4)
        if len(gradients[0]) != len(gradients[1]):
            raise AssertionError("Shared EV introduced non-edge backbone parameters")
        for left, right in zip(*gradients):
            if left is None or right is None:
                assert left is right
            else:
                torch.testing.assert_close(left, right, atol=1e-4, rtol=1e-4)
        assert all(edge._relation_stem is None for edge in candidate.edge_attention)
        assert all(t.enabled for t in trimmers)
        torch.testing.assert_close(torch.as_tensor(trimmers[0]._counter), torch.as_tensor(trimmers[1]._counter), atol=0, rtol=0)
        left_buffers, right_buffers = list(reference.buffers()), list(candidate.buffers())
        assert len(left_buffers) == len(right_buffers)
        for left, right in zip(left_buffers, right_buffers):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
    return {"zero_message_logits_and_input_and_parameter_gradients": True,
            "batchnorm_buffers_and_trimmer_counters_exact": True,
            "active_training_trimmer": True, "fp32_atol_rtol_logits": 5e-5,
            "fp32_atol_rtol_gradients": 1e-4}


def validate(args):
    import numpy as np
    import torch
    from jetclass_fresh.part_inputs import build_particle_transformer_inputs_from_tokens
    from teacher_logit_reco.relational_part.ca_tree import build_compiled_tree
    from teacher_logit_reco.relational_part.offline_ablation_data import open_arrays
    from teacher_logit_reco.relational_part.offline_ablations import MODEL_SPECS
    from teacher_logit_reco.relational_part.model import RelationalParticleTransformer
    from teacher_logit_reco.relational_part.evaluation import model_forward
    spec = load_campaign(args.output)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Runtime validation requires the production BF16 CUDA device")
    if (args.output / "runtime_validation.json").exists():
        existing = F.read_json(args.output / "runtime_validation.json")
        if existing["campaign_sha256"] != spec["content_hash"] or not existing["passed"]:
            raise ValueError("Existing runtime validation is stale")
        return
    directory = args.output / "inputs/cache/model_train"
    arrays = open_arrays(directory, F.read_json(directory / "receipt.json"))
    raw = np.array(arrays["tokens"][:8], copy=True)
    mask = np.array(arrays["mask"][:8], copy=True)
    for row, count in enumerate((7, 3, 1, 5, 8, 2, 4, 6)):
        mask[row, count:] = False
    raw[~mask] = 0
    inputs = build_particle_transformer_inputs_from_tokens(raw, mask, source_view="offline")
    backend = backend_for(spec)
    trees_batch = [build_compiled_tree(backend, inputs.pf_vectors[i].T, raw[i], mask[i]) for i in range(8)]
    batch = {"points": torch.from_numpy(inputs.pf_points), "features": torch.from_numpy(inputs.pf_features),
             "lorentz_vectors": torch.from_numpy(inputs.pf_vectors), "mask": torch.from_numpy(inputs.pf_mask).bool(),
             "raw_tokens": torch.from_numpy(raw), "region_trees": trees_batch}
    parity = {}
    for selected in (False, True):
        seed_all(101)
        reference = model_for(spec, "OFF_RPT_SELECTED_SHARED") if selected else RelationalParticleTransformer()
        parity[str(selected)] = zero_message_parity(reference, batch, device="cuda")
    forward = {}
    for name in MODEL_SPECS:
        seed_all(101)
        model = model_for(spec, name).cuda().train()
        moved = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model_forward(model, moved)
            loss = torch.nn.functional.cross_entropy(logits, torch.arange(8, device="cuda"))
        loss.backward()
        assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        edges = [p for key, p in model.named_parameters() if key.endswith("edge_projection")]
        if MODEL_SPECS[name]["edge_value"]:
            assert len(edges) == 8 and all(p.grad is not None and p.grad.abs().sum() > 0 for p in edges)
        restored = model_for(spec, name).cuda().eval()
        restored.load_state_dict(model.state_dict(), strict=True)
        model.eval()
        with torch.no_grad():
            expected = model_forward(model, moved)
            actual = model_forward(restored, moved)
        torch.testing.assert_close(actual, expected, atol=5e-5, rtol=5e-5)
        forward[name] = {"bf16_backward_finite": True, "strict_state_reload": True,
                         "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
        del model, restored
        torch.cuda.empty_cache()
    F.write_json(args.output / "runtime_validation.json", F.hashed({"contract": VERSION + "_runtime_validation",
                 "campaign_sha256": spec["content_hash"], "passed": True, "parity": parity,
                 "models": forward, "device": torch.cuda.get_device_name(0)}))
    F.progress("runtime_validation_passed")


def lock(args):
    spec = load_campaign(args.output)
    tasks, rows = [], []
    for task in registered_tasks(args.output, spec)["tasks"]:
        registration = verify_registration(args.output, spec, task)
        path = f"runs/{task['run_id']}/seed_{task['seed']}"
        tasks.append({"run_id": task["run_id"], "seed": task["seed"], "checkpoint": path + "/best_model_val.pt",
                      "checkpoint_sha256": registration["checkpoint_sha256"],
                      "model_contract_sha256": task["model_contract_sha256"], "relation_families": task["relation_families"]})
        rows.append({"run_id": task["run_id"], "seed": task["seed"], "registration_sha256": registration["content_hash"],
                     "checkpoint_sha256": registration["checkpoint_sha256"], "selected_epoch": registration["selected_epoch"],
                     "validation": F.read_json(args.output / path / "val_select_metrics.json")})
    artifact = F.hashed({"contract": VERSION + "_lock", "campaign_sha256": spec["content_hash"],
                         "rows": rows, "performance_gate": False, "all_twelve_included": True})
    F.write_json(args.output / "selection/locked_models.json", artifact)
    old = F.read_json(args.output / "reference_plan.json")
    plan = F.hashed({"contract": VERSION + "_test", "campaign_sha256": spec["content_hash"],
        "lock_sha256": artifact["content_hash"], "tasks": tasks, "parent": str(args.output),
        "data_dir": spec["test_data"], "files": old["files"], "event_count": old["event_count"],
        "class_order": old["class_order"], "chunk_size": old["chunk_size"], "batch_size": old["batch_size"],
        "assignments": F.file_assignment(old["files"], 20), "reference_plan_sha256": old["content_hash"],
        "precision": old["precision"], "metric_policy": old["metric_policy"], "rejection_targets": old["rejection_targets"],
        "performance_gate": False, "used_for_model_selection": False})
    F.write_json(args.output / "test/evaluation_plan.json", plan)
    F.progress("all_checkpoints_locked", count=len(tasks))


def test_plan(output, spec):
    plan = F.read_json(output / "test/evaluation_plan.json")
    locked = F.read_json(output / "selection/locked_models.json")
    old = F.read_json(output / "reference_plan.json")
    if (plan["contract"] != VERSION + "_test" or plan["campaign_sha256"] != spec["content_hash"]
            or plan["lock_sha256"] != locked["content_hash"] or locked["campaign_sha256"] != spec["content_hash"]
            or any(plan[key] != old[key] for key in ("files", "chunk_size", "batch_size", "precision", "metric_policy", "rejection_targets", "class_order"))
            or plan["reference_plan_sha256"] != old["content_hash"]
            or plan["assignments"] != F.file_assignment(old["files"], 20)
            or plan["parent"] != str(output) or plan["data_dir"] != spec["test_data"]
            or plan["performance_gate"] is not False or plan["used_for_model_selection"] is not False
            or locked["performance_gate"] is not False or locked["all_twelve_included"] is not True
            or plan["event_count"] != 20_000_000 or len(plan["tasks"]) != 12 or len(locked["rows"]) != 12):
        raise ValueError("Frozen test plan differs")
    for index, (task, bound) in enumerate(zip(registered_tasks(output, spec)["tasks"], locked["rows"])):
        registration = verify_registration(output, spec, task)
        expected_task = {"run_id": task["run_id"], "seed": task["seed"],
            "checkpoint": f"runs/{task['run_id']}/seed_{task['seed']}/best_model_val.pt",
            "checkpoint_sha256": registration["checkpoint_sha256"],
            "model_contract_sha256": task["model_contract_sha256"], "relation_families": task["relation_families"]}
        if (registration["content_hash"] != bound["registration_sha256"] or plan["tasks"][index] != expected_task
                or bound["run_id"] != task["run_id"] or bound["seed"] != task["seed"]
                or bound["validation"]["content_hash"] != registration["val_select_metrics_sha256"]):
            raise ValueError("Locked checkpoint registration changed")
    return plan


def infer(args):
    import torch
    spec = load_campaign(args.output)
    plan = test_plan(args.output, spec)
    def loader(frozen, device):
        models = []
        for task in frozen["tasks"]:
            model = model_for(spec, task["run_id"])
            checkpoint = torch.load(args.output / task["checkpoint"], map_location="cpu", weights_only=False)
            if checkpoint["model_contract_sha256"] != task["model_contract_sha256"]:
                raise ValueError("New checkpoint payload contract differs")
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            models.append(model.to(device).eval())
        return models, backend_for(spec)
    F.run(argparse.Namespace(output=args.output / "test", task_index=args.task_index),
          frozen_plan=plan, model_loader=loader)


def aggregate(args):
    import numpy as np
    spec = load_campaign(args.output)
    plan = test_plan(args.output, spec)
    previous_plan = F.read_json(args.output / "reference_plan.json")
    task = plan["tasks"][args.task_index]
    # Keep all same-seed argmax predictions for exact paired comparisons across
    # the complete eight-cell factorial matrix, but only one model's full logits.
    references = {t["run_id"]: np.empty(plan["event_count"], dtype=np.int8)
                  for t in [*previous_plan["tasks"], *plan["tasks"]] if t["seed"] == task["seed"]}
    values = np.empty((plan["event_count"], 10), dtype=np.float32)
    labels = np.empty(plan["event_count"], dtype=np.int8)
    offset = 0
    completions = {"old": [], "new": []}
    for index, row in enumerate(plan["files"]):
        records = {"old": [], "new": []}
        for start, stop in F.chunks(row, plan):
            end = offset + stop - start
            for tag, current, root in (("old", previous_plan, Path(spec["previous_evaluation"])),
                                       ("new", plan, args.output / "test")):
                indices = [i for i, t in enumerate(current["tasks"]) if t["seed"] == task["seed"]]
                arrays, receipt = F.read_chunk(F.chunk_path(root, index, start), F.chunk_metadata(current, index, start, stop),
                                               keys={f"logits_{i:02d}" for i in indices})
                records[tag].append(receipt["content_hash"])
                for i in indices:
                    references[current["tasks"][i]["run_id"]][offset:end] = arrays[f"logits_{i:02d}"].argmax(axis=1)
                if tag == "new":
                    values[offset:end] = arrays[f"logits_{args.task_index:02d}"]
            labels[offset:end] = row["label"]
            offset = end
        for tag, current, root in (("old", previous_plan, Path(spec["previous_evaluation"])),
                                   ("new", plan, args.output / "test")):
            done = F.read_json(root / "predictions" / f"file_{index:03d}" / "complete.json")
            if done != F.hashed({"contract": current["contract"] + "_file_complete", "plan_sha256": current["content_hash"],
                                 "file_index": index, "chunks": records[tag]}):
                raise ValueError("Prediction file completion/coverage differs")
            completions[tag].append(done["content_hash"])
    # test_plan has already required the official 20M population. Express the
    # coverage check through that bound plan for small synthetic regression tests.
    if offset != plan["event_count"] or not np.array_equal(np.bincount(labels, minlength=10), np.full(10, plan["event_count"] // 10)):
        raise ValueError("Official test coverage is incomplete/unbalanced")
    old_report = F.read_json(args.output / "reference_report.json")
    if any(r["file_completion_hashes"] != completions["old"] for r in old_report["per_model_results"]):
        raise ValueError("Old predictions differ from the already published metrics")
    metric = F.calculate_metrics(values, labels)
    prediction = values.argmax(axis=1)
    paired = {name: F.paired_accuracy(prediction, reference, labels)
              for name, reference in references.items() if name != task["run_id"]}
    F.write_json(args.output / "test/metrics" / f"model_{args.task_index:02d}.json", F.hashed({
        "contract": VERSION + "_result", "plan_sha256": plan["content_hash"], "task": task,
        "file_completion_hashes": completions, "metrics": metric, "paired_accuracy": paired,
        "used_for_model_selection": False}))
    F.progress("metric_complete", run_id=task["run_id"], seed=task["seed"], accuracy=metric["accuracy"])


def report(args):
    spec = load_campaign(args.output)
    plan = test_plan(args.output, spec)
    old = F.read_json(args.output / "reference_report.json")
    results = [F.read_json(args.output / "test/metrics" / f"model_{i:02d}.json") for i in range(12)]
    for i, result in enumerate(results):
        if (result["plan_sha256"] != plan["content_hash"] or result["task"] != plan["tasks"][i]
                or result["file_completion_hashes"] != results[0]["file_completion_hashes"]):
            raise ValueError("Report mixes checkpoints or event populations")
    all_results = old["per_model_results"] + results
    models = {}
    for name in (*F.MODELS, *spec["models"]):
        matching = {r["task"]["seed"]: r for r in all_results if r["task"]["run_id"] == name}
        if set(matching) != set(F.SEEDS):
            raise ValueError("Report is missing a seed")
        accuracies = [matching[seed]["metrics"]["accuracy"] for seed in F.SEEDS]
        models[name] = {"mean_accuracy": statistics.mean(accuracies),
            "seed_sample_standard_deviation": statistics.stdev(accuracies),
            "per_seed_accuracy": {str(seed): matching[seed]["metrics"]["accuracy"] for seed in F.SEEDS},
            "qcd_signal_rejection": {}}
        for signal in F.CLASS_NAMES[1:]:
            models[name]["qcd_signal_rejection"][signal] = {}
            for target in F.TARGETS:
                numbers = [matching[s]["metrics"]["qcd_signal_rejection"][signal][str(target)]["background_rejection"] for s in F.SEEDS]
                models[name]["qcd_signal_rejection"][signal][str(target)] = {
                    "per_seed": dict(zip(map(str, F.SEEDS), numbers)),
                    "mean": None if any(x is None for x in numbers) else statistics.mean(numbers)}
    contrasts = {}
    for features, shared, layer, shared_ev, layer_ev in (
        ("standard", "OFF_RPT_BASE", "OFF_RPT_BASE_LAYERWISE", "OFF_RPT_BASE_SHARED_EDGEVALUE", "OFF_RPT_BASE_EDGEVALUE"),
        ("selected", "OFF_RPT_SELECTED_SHARED", "OFF_RPT_SELECTED_LAYERWISE", "OFF_RPT_SELECTED_SHARED_EDGEVALUE", "OFF_RPT_SELECTED_EDGEVALUE")):
        contrast = {}
        for label, positive, negative in (("EV_at_shared", shared_ev, shared), ("EV_at_layerwise", layer_ev, layer),
                                          ("layerwise_without_EV", layer, shared), ("layerwise_with_EV", layer_ev, shared_ev)):
            diffs = [models[positive]["per_seed_accuracy"][str(s)] - models[negative]["per_seed_accuracy"][str(s)] for s in F.SEEDS]
            contrast[label] = {"positive": positive, "negative": negative, "mean_accuracy_difference": statistics.mean(diffs),
                               "per_seed_difference": dict(zip(map(str, F.SEEDS), diffs))}
        contrast["EV_by_layerwise_interaction"] = {
            "definition": "(layerwise_EV-layerwise_no_EV)-(shared_EV-shared_no_EV)",
            "mean_accuracy_difference": contrast["EV_at_layerwise"]["mean_accuracy_difference"] - contrast["EV_at_shared"]["mean_accuracy_difference"]}
        contrasts[features] = contrast
    contrasts["selected_minus_standard"] = {}
    for label, positive, negative in (
        ("shared_without_EV", "OFF_RPT_SELECTED_SHARED", "OFF_RPT_BASE"),
        ("shared_with_EV", "OFF_RPT_SELECTED_SHARED_EDGEVALUE", "OFF_RPT_BASE_SHARED_EDGEVALUE"),
        ("layerwise_without_EV", "OFF_RPT_SELECTED_LAYERWISE", "OFF_RPT_BASE_LAYERWISE"),
        ("layerwise_with_EV", "OFF_RPT_SELECTED_EDGEVALUE", "OFF_RPT_BASE_EDGEVALUE")):
        differences = {str(s): models[positive]["per_seed_accuracy"][str(s)] - models[negative]["per_seed_accuracy"][str(s)] for s in F.SEEDS}
        contrasts["selected_minus_standard"][label] = {"positive": positive, "negative": negative,
            "mean_accuracy_difference": statistics.mean(differences.values()), "per_seed_difference": differences}
    for name, row in models.items():
        row["difference_vs_baseline"] = row["mean_accuracy"] - models["OFF_RPT_BASE"]["mean_accuracy"]
    result = F.hashed({"contract": VERSION + "_report", "campaign_sha256": spec["content_hash"],
        "event_count_per_model": 20_000_000, "models": models, "contrasts": contrasts,
        "per_model_results": all_results, "new_training_runs": 12, "existing_checkpoints_retrained": False,
        "followup_after_test_inspection": True, "training_jets": 1_000_000,
        "interpretation": "Matched protocol, not universally parameter/compute matched. Shared retains original packing/trimmer; layerwise retains historical confirmation path."})
    F.write_json(args.output / "reports/offline_factorial_report.json", result)
    lines = ["# Offline RPT follow-up: complete eight-cell matrix", "",
             "20M official test jets; 1M training jets; three seeds. Four new follow-up cells were fixed AFTER inspection of earlier test results.", "",
             "| Model | Accuracy | Delta base (pp) | Seed SD (pp) |", "|---|---:|---:|---:|"]
    for name, row in models.items():
        lines.append(f"| {name} | {100*row['mean_accuracy']:.4f}% | {100*row['difference_vs_baseline']:+.4f} | {100*row['seed_sample_standard_deviation']:.4f} |")
    lines.extend(["", "## Matched architectural contrasts", "", "| Features | Contrast | Accuracy delta (pp) |", "|---|---|---:|"])
    for family, values in contrasts.items():
        for label, row in values.items():
            lines.append(f"| {family} | {label} | {100*row['mean_accuracy_difference']:+.4f} |")
    for target in F.TARGETS:
        lines.extend(["", f"## QCD rejection at {100*target:g}% signal efficiency", "",
                      "| Model | " + " | ".join(F.CLASS_NAMES[1:]) + " |", "|---|" + "---:|" * 9])
        for name, row in models.items():
            cells = []
            for signal in F.CLASS_NAMES[1:]:
                value = row["qcd_signal_rejection"][signal][str(target)]["mean"]
                base = models["OFF_RPT_BASE"]["qcd_signal_rejection"][signal][str(target)]["mean"]
                cell = "SAT" if value is None else f"{value:.2f}"
                if name != "OFF_RPT_BASE" and value is not None and base is not None:
                    cell += f" ({100*(value/base-1):+.1f}%)"
                cells.append(cell)
            lines.append("| " + name + " | " + " | ".join(cells) + " |")
    lines.extend(["", "SAT means at least one zero-background seed, not infinite population rejection.",
        "Means are over independently evaluated seeds, not ensemble predictions. JSON retains passing counts, conditional Wilson intervals and paired accuracy intervals.",
        "The family encoders and EV introduce extra parameters/compute. These ablations do not establish parameter-matched causality.",
        "Shared-bias models retain original Weaver trimming and pair-normalization packing; layerwise models retain the earlier confirmation implementation.", ""])
    text = "\n".join(lines)
    path = args.output / "reports/offline_factorial_report.md"
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise FileExistsError("Existing combined Markdown differs")
    with F.atomic_file(path, binary=False) as stream:
        stream.write(text)
    F.progress("complete", report=str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("bootstrap", "prepare", "cache", "storage", "trees", "bind",
                                          "validate", "train", "lock", "infer", "aggregate", "report"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--train-data", type=Path)
    parser.add_argument("--test-data", type=Path)
    parser.add_argument("--task-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "-1")))
    args = parser.parse_args()
    args.output = args.output.resolve()
    limits = {"cache": 3, "trees": 126, "train": 12, "infer": 20, "aggregate": 12}
    if args.phase in limits and not 0 <= args.task_index < limits[args.phase]:
        parser.error("task index outside frozen array")
    globals()[args.phase](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
