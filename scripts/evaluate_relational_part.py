#!/usr/bin/env python3
"""Evaluate a relational checkpoint without permitting checkpoint reselection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relational_part import (  # noqa: E402
    RELATION_NORMALIZATION_ARTIFACT_CONTRACT,
    REGION_NORMALIZATION_CONTRACT,
    SCREENING_REGISTRY_CONTRACT,
    build_cached_loaders,
    build_runtime_model,
    evaluate_model,
    load_hashed_json,
    write_immutable_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, choices=(101, 202, 303), required=True)
    parser.add_argument("--model-contract", type=Path, required=True)
    parser.add_argument("--split", choices=("model_val", "stack_val"), required=True)
    parser.add_argument("--tree-root", type=Path)
    parser.add_argument("--selected-families", nargs="*", default=[])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_dir = (
        args.campaign_root / "runs" / args.run_id / f"seed_{args.seed}"
    )
    if args.split == "stack_val":
        registered = run_dir / "val_select_metrics.json"
        if registered.is_file():
            payload = load_hashed_json(registered)
            print(json.dumps(
                {
                    "status": "authenticated_existing_single_evaluation",
                    "checkpoint_reselection_attempted": False,
                    "metrics": payload,
                },
                indent=2,
                sort_keys=True,
            ))
            return 0
        raise ValueError(
            "stack_val is evaluated exactly once by the trainer; its missing "
            "registered artifact makes the run incomplete rather than eligible "
            "for an ad-hoc second evaluation"
        )

    screening = load_hashed_json(
        args.campaign_root / "registry" / "screening_registry.json",
        expected_contract=SCREENING_REGISTRY_CONTRACT,
    )
    normalization = load_hashed_json(
        args.campaign_root / "inputs" / "relation_normalization.json",
        expected_contract=RELATION_NORMALIZATION_ARTIFACT_CONTRACT,
    )
    region_path = args.campaign_root / "inputs" / "region_normalization.json"
    region = (
        load_hashed_json(
            region_path, expected_contract=REGION_NORMALIZATION_CONTRACT
        )
        if region_path.is_file()
        else None
    )
    model_contract = load_hashed_json(args.model_contract)
    resolved = {
        "run_id": args.run_id,
        "seed": args.seed,
        "split": args.split,
        "checkpoint": str(run_dir / "best_model_val.pt"),
        "model_contract_sha256": model_contract["content_hash"],
        "checkpoint_reselection_allowed": False,
        "dry_run": bool(args.dry_run),
    }
    print(json.dumps(resolved, indent=2, sort_keys=True))
    if args.dry_run:
        return 0

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = build_runtime_model(
        args.run_id,
        screening_registry=screening,
        normalization_artifact=normalization,
        region_normalization_artifact=region,
        selected_families=args.selected_families,
    )
    checkpoint = torch.load(
        run_dir / "best_model_val.pt", map_location="cpu", weights_only=False
    )
    if checkpoint.get("model_contract_sha256") != model_contract["content_hash"]:
        raise ValueError("checkpoint belongs to another model contract")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    families = (
        tuple(args.selected_families)
        if args.run_id.startswith("RPT_SELECTED_")
        else tuple(model_contract.get("new_relation_families", ()))
    )
    _, val_stop_loader, _, _ = build_cached_loaders(
        cache_dir=args.cache_dir,
        seed=args.seed,
        families=families,
        tree_root=args.tree_root,
    )
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    model.to(device)
    metrics = evaluate_model(
        model, val_stop_loader, split="val_stop", device=device
    )
    write_immutable_json(args.output, metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
