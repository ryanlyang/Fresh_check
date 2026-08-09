#!/usr/bin/env python3
"""Train one immutable offline RPT transfer task."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relational_part.contracts import load_hashed_json, write_immutable_json  # noqa: E402
from teacher_logit_reco.relational_part.profiling import profile_model_resources  # noqa: E402
from teacher_logit_reco.relational_part.normalization import RELATION_NORMALIZATION_ARTIFACT_CONTRACT_V3  # noqa: E402
from teacher_logit_reco.relational_part.offline_transfer import (  # noqa: E402
    OFFLINE_TRANSFER_MODEL_CONTRACT,
    OFFLINE_TRANSFER_TASK_REGISTRY_CONTRACT,
    validate_offline_task_registry,
)
from teacher_logit_reco.relational_part.relation_region import REGION_NORMALIZATION_CONTRACT  # noqa: E402
from teacher_logit_reco.relational_part.runtime import build_cached_loaders, build_runtime_model  # noqa: E402
from teacher_logit_reco.relational_part.train import TrainingConfig, train_relational_model  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--task-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "-1")))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    root = args.campaign_root.resolve()
    campaign = load_hashed_json(root / "campaign_spec.json")
    tasks = load_hashed_json(root / "registry" / "training_tasks.json", expected_contract=OFFLINE_TRANSFER_TASK_REGISTRY_CONTRACT)
    validate_offline_task_registry(tasks)
    if args.task_index < 0 or args.task_index >= len(tasks["tasks"]):
        raise IndexError("offline training task index differs")
    task = tasks["tasks"][args.task_index]
    run_id, seed = str(task["run_id"]), int(task["seed"])
    model_contract = load_hashed_json(root / task["model_contract_path"], expected_contract=OFFLINE_TRANSFER_MODEL_CONTRACT)
    if model_contract["content_hash"] != task["model_contract_sha256"]:
        raise ValueError("offline model contract differs from task")
    relation = load_hashed_json(root / "inputs" / "relation_normalization.json", expected_contract=RELATION_NORMALIZATION_ARTIFACT_CONTRACT_V3)
    region = load_hashed_json(root / "inputs" / "region_normalization.json", expected_contract=REGION_NORMALIZATION_CONTRACT)
    screening = load_hashed_json(root / "registry" / "screening_registry.json")
    relation_registry = load_hashed_json(root / "registry" / "relation_family_registry.json")
    determinism = load_hashed_json(root / "registry" / "global_determinism.json")
    binding = load_hashed_json(root / "inputs" / "offline_cache_binding.json")
    families = tuple(task["relation_families"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    model = build_runtime_model(
        run_id,
        screening_registry=screening,
        normalization_artifact=relation,
        region_normalization_artifact=region,
        selected_families=families,
    )
    train_loader, val_loader, select_loader, hashes = build_cached_loaders(
        cache_dir=root / "inputs" / "offline_cache",
        seed=seed,
        families=families,
        tree_root=root / "inputs" / "relation_tree_cache",
        input_view="offline",
    )
    expected_hashes = {name: row["offline_content_sha256"] for name, row in binding["splits"].items() if name != "final_test"}
    if hashes != expected_hashes:
        raise ValueError("opened offline training caches differ from binding")
    output = root / "runs" / run_id / f"seed_{seed}"
    profile_path = output / "resource_profile.json"
    if profile_path.is_file():
        profile = load_hashed_json(profile_path)
    else:
        profile = profile_model_resources(
            model,
            next(iter(val_loader)),
            device=device,
            model_contract_sha256=model_contract["content_hash"],
        )
        write_immutable_json(profile_path, profile)
    lineage = {
        "campaign_spec": campaign["content_hash"],
        "offline_cache_binding": binding["content_hash"],
        "relation_normalization": relation["content_hash"],
        "region_normalization": region["content_hash"],
        **{f"offline_{name}": digest for name, digest in hashes.items()},
    }
    for split in ("model_train", "model_val", "stack_val"):
        lineage[f"angular_tree_{split}"] = load_hashed_json(
            root / "inputs" / "relation_tree_cache" / f"{split}_exclusive_ca_v1" / "manifest.json"
        )["content_hash"]
    config = TrainingConfig(seed=seed)
    registration = train_relational_model(
        model=model,
        train_loader=train_loader,
        val_stop_loader=val_loader,
        val_select_loader=select_loader,
        output_dir=output,
        run_id=run_id,
        model_contract_sha256=model_contract["content_hash"],
        run_registry_sha256=tasks["content_hash"],
        relation_registry_sha256=relation_registry["content_hash"],
        global_determinism_sha256=determinism["content_hash"],
        lineage_hashes=lineage,
        config=config,
        device=device,
        resource_profile=profile,
        resume=True,
        inference_input_role="offline_tagger",
    )
    print(registration["content_hash"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
