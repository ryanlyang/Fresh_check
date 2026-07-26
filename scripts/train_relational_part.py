#!/usr/bin/env python3
"""Train one immutable relational Particle Transformer run/seed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relational_part import (  # noqa: E402
    RELATION_FAMILY_REGISTRY_CONTRACT,
    RELATION_NORMALIZATION_ARTIFACT_CONTRACT,
    REGION_NORMALIZATION_CONTRACT,
    RESOURCE_PROFILE_CONTRACT,
    SCREENING_REGISTRY_CONTRACT,
    TrainingConfig,
    build_cached_loaders,
    build_global_determinism_contract,
    build_runtime_model,
    load_hashed_json,
    profile_model_resources,
    train_relational_model,
    validate_content_hash,
    validate_campaign_source,
    write_immutable_json,
)

import torch  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, choices=(101, 202, 303), required=True)
    parser.add_argument("--model-contract", type=Path, required=True)
    parser.add_argument("--tree-root", type=Path)
    parser.add_argument("--selected-families", nargs="*", default=[])
    parser.add_argument("--unary-registry", type=Path)
    parser.add_argument(
        "--run-registry",
        type=Path,
        help="Immutable registry governing this run; defaults to screening.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    registry_dir = args.campaign_root / "registry"
    inputs_dir = args.campaign_root / "inputs"
    campaign = load_hashed_json(args.campaign_root / "campaign_spec.json")
    validate_campaign_source(campaign, repo_root=REPO_ROOT)
    screening = load_hashed_json(
        registry_dir / "screening_registry.json",
        expected_contract=SCREENING_REGISTRY_CONTRACT,
    )
    relation_registry = load_hashed_json(
        registry_dir / "relation_family_registry.json",
        expected_contract=RELATION_FAMILY_REGISTRY_CONTRACT,
    )
    normalization = load_hashed_json(
        inputs_dir / "relation_normalization.json",
        expected_contract=RELATION_NORMALIZATION_ARTIFACT_CONTRACT,
    )
    region_path = inputs_dir / "region_normalization.json"
    region_normalization = (
        load_hashed_json(
            region_path, expected_contract=REGION_NORMALIZATION_CONTRACT
        )
        if region_path.is_file()
        else None
    )
    model_contract = load_hashed_json(args.model_contract)
    if model_contract.get("run_id") != args.run_id:
        raise ValueError("model contract run_id differs from --run-id")
    determinism = build_global_determinism_contract()
    expected_determinism = campaign["parent_artifact_hashes"][
        "global_determinism"
    ]
    if determinism["content_hash"] != expected_determinism:
        raise ValueError("campaign global deterministic policy drifted")
    config = TrainingConfig(seed=args.seed)
    training_contract = config.artifact(
        global_determinism_sha256=determinism["content_hash"]
    )
    resolved = {
        "run_id": args.run_id,
        "seed": args.seed,
        "selected_families": list(args.selected_families),
        "device": args.device,
        "model_contract_sha256": model_contract["content_hash"],
        "screening_registry_sha256": screening["content_hash"],
        "relation_registry_sha256": relation_registry["content_hash"],
        "normalization_sha256": normalization["content_hash"],
        "region_normalization_sha256": (
            None
            if region_normalization is None
            else region_normalization["content_hash"]
        ),
        "training_contract": training_contract,
        "dry_run": bool(args.dry_run),
    }
    unary_registry = (
        load_hashed_json(args.unary_registry)
        if args.unary_registry is not None
        else None
    )
    run_registry = (
        load_hashed_json(args.run_registry)
        if args.run_registry is not None
        else screening
    )
    if args.run_id == "RPT_SELECTED_UNARY" and unary_registry is None:
        raise ValueError("RPT_SELECTED_UNARY requires --unary-registry")
    if unary_registry is not None:
        resolved["unary_registry_sha256"] = unary_registry["content_hash"]
    resolved["run_registry_sha256"] = run_registry["content_hash"]
    print(json.dumps(resolved, indent=2, sort_keys=True))
    if args.dry_run:
        return 0

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    model = build_runtime_model(
        args.run_id,
        screening_registry=screening,
        normalization_artifact=normalization,
        region_normalization_artifact=region_normalization,
        selected_families=args.selected_families,
        unary_registry=unary_registry,
    )
    active_families = (
        tuple(args.selected_families)
        if args.run_id.startswith("RPT_SELECTED_")
        else tuple(model_contract.get("new_relation_families", ()))
    )
    train_loader, val_stop_loader, val_select_loader, hlt_hashes = (
        build_cached_loaders(
            cache_dir=args.cache_dir,
            seed=args.seed,
            families=active_families,
            tree_root=args.tree_root,
        )
    )
    output = (
        args.campaign_root / "runs" / args.run_id / f"seed_{args.seed}"
    )
    profile_path = output / "resource_profile.json"
    if profile_path.is_file():
        profile = load_hashed_json(
            profile_path, expected_contract=RESOURCE_PROFILE_CONTRACT
        )
    else:
        example = next(iter(val_stop_loader))
        profile = profile_model_resources(
            model,
            example,
            device=device,
            model_contract_sha256=model_contract["content_hash"],
        )
        write_immutable_json(profile_path, profile)
    lineage = {
        "campaign_spec": campaign["content_hash"],
        "split_manifest": campaign["split_manifest_hash"],
        "raw_input_schema": load_hashed_json(
            inputs_dir / "raw_input_schema.json"
        )["content_hash"],
        "hlt_binding": load_hashed_json(
            inputs_dir / "hlt_cache_audit.json"
        )["content_hash"],
        "postconstruction_input_audit": load_hashed_json(
            inputs_dir / "postconstruction_input_audit.json"
        )["content_hash"],
        "relation_normalization": normalization["content_hash"],
        **{f"hlt_{name}": value for name, value in hlt_hashes.items()},
    }
    if region_normalization is not None and "REGION" in active_families:
        lineage["region_normalization"] = region_normalization["content_hash"]
        lineage["angular_tree_resource"] = load_hashed_json(
            inputs_dir / "angular_tree_resource_contract.json"
        )["content_hash"]
        lineage["angular_tree_backend"] = load_hashed_json(
            args.campaign_root / "backend" / "backend_manifest.json"
        )["content_hash"]
        lineage["angular_tree_throughput_probe"] = load_hashed_json(
            args.campaign_root / "backend" / "throughput_probe.json"
        )["content_hash"]
        for split in ("model_train", "model_val", "stack_val"):
            lineage[f"angular_tree_{split}"] = load_hashed_json(
                args.tree_root
                / f"{split}_exclusive_ca_v1"
                / "manifest.json"
            )["content_hash"]
    registration = train_relational_model(
        model=model,
        train_loader=train_loader,
        val_stop_loader=val_stop_loader,
        val_select_loader=val_select_loader,
        output_dir=output,
        run_id=args.run_id,
        model_contract_sha256=model_contract["content_hash"],
        run_registry_sha256=(
            unary_registry["content_hash"]
            if unary_registry is not None
            else run_registry["content_hash"]
        ),
        relation_registry_sha256=relation_registry["content_hash"],
        global_determinism_sha256=determinism["content_hash"],
        lineage_hashes=lineage,
        config=config,
        device=device,
        resource_profile=profile,
        resume=True,
    )
    validate_content_hash(registration)
    print(json.dumps(registration, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
