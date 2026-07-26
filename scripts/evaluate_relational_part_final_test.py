#!/usr/bin/env python3
"""Evaluate one locked checkpoint on the HLT-only sealed final test."""

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
    LOCKED_FINALISTS_CONTRACT,
    RELATION_NORMALIZATION_ARTIFACT_CONTRACT,
    REGION_NORMALIZATION_CONTRACT,
    SCREENING_REGISTRY_CONTRACT,
    build_final_test_loader,
    build_runtime_model,
    evaluate_locked_finalist,
    load_hashed_json,
    sha256_file,
    validate_locked_finalists,
    write_immutable_json,
    validate_campaign_source,
)
from teacher_logit_reco.relational_part.workflow import (  # noqa: E402
    parse_named_hashes,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locked-finalists", type=Path, required=True)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    parser.add_argument("--split-manifest-sha256", required=True)
    parser.add_argument("--hlt-cache-hash", action="append", default=[])
    parser.add_argument("--screening-registry", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--region-normalization", type=Path)
    parser.add_argument("--unary-registry", type=Path)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--tree-root", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, choices=(101, 202, 303), required=True)
    parser.add_argument("--model-contract", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-registration", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-event-count", type=int, default=500_000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    lock = load_hashed_json(
        args.locked_finalists, expected_contract=LOCKED_FINALISTS_CONTRACT
    )
    campaign = load_hashed_json(args.campaign_spec)
    validate_campaign_source(campaign, repo_root=REPO_ROOT)
    if (
        str(campaign.get("campaign_profile", "")).startswith("production_")
        and args.expected_event_count != 500_000
    ):
        raise ValueError(
            "production final evaluation requires exactly 500,000 events"
        )
    hlt_hashes = parse_named_hashes(args.hlt_cache_hash)
    validate_locked_finalists(
        lock,
        campaign_spec_sha256=campaign["content_hash"],
        split_manifest_sha256=args.split_manifest_sha256,
        hlt_cache_hashes=hlt_hashes,
    )
    locked_rows = {row["run_id"]: row for row in lock["evaluation_rows"]}
    if args.run_id not in locked_rows:
        raise ValueError("requested run is absent from the finalist lock")
    families = tuple(locked_rows[args.run_id]["new_relation_families"])
    registration = load_hashed_json(args.checkpoint_registration)
    model_contract = load_hashed_json(args.model_contract)
    normalization = load_hashed_json(
        args.normalization,
        expected_contract=RELATION_NORMALIZATION_ARTIFACT_CONTRACT,
    )
    region = (
        load_hashed_json(
            args.region_normalization,
            expected_contract=REGION_NORMALIZATION_CONTRACT,
        )
        if args.region_normalization is not None
        else None
    )
    unary = (
        load_hashed_json(args.unary_registry)
        if args.unary_registry is not None
        else None
    )
    if args.run_id == "RPT_SELECTED_UNARY":
        if unary is None or unary["content_hash"] != lock[
            "unary_control_registry_sha256"
        ]:
            raise ValueError("unary registry is absent or differs from the lock")
    resolved = {
        "run_id": args.run_id,
        "seed": args.seed,
        "families": list(families),
        "split": "final_test",
        "locked_finalists_sha256": lock["content_hash"],
        "checkpoint_sha256": registration["checkpoint_sha256"],
        "hlt_only": True,
        "final_test_used_for_selection": False,
        "expected_event_count": args.expected_event_count,
        "dry_run": bool(args.dry_run),
    }
    print(json.dumps(resolved, indent=2, sort_keys=True))
    if args.dry_run:
        return 0
    if sha256_file(args.checkpoint) != registration["checkpoint_sha256"]:
        raise ValueError("final checkpoint file hash mismatch")
    screening = load_hashed_json(
        args.screening_registry, expected_contract=SCREENING_REGISTRY_CONTRACT
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = build_runtime_model(
        args.run_id,
        screening_registry=screening,
        normalization_artifact=normalization,
        region_normalization_artifact=region,
        selected_families=families,
        unary_registry=unary,
    )
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    if checkpoint.get("model_contract_sha256") != model_contract["content_hash"]:
        raise ValueError("final checkpoint model contract mismatch")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    loader, final_hlt_hash = build_final_test_loader(
        cache_dir=args.cache_dir,
        seed=args.seed,
        families=families,
        tree_root=args.tree_root,
    )
    if final_hlt_hash != hlt_hashes.get("final_test"):
        raise ValueError("opened final-test HLT cache differs from the lock")
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    model.to(device)
    result = evaluate_locked_finalist(
        model,
        loader,
        run_id=args.run_id,
        seed=args.seed,
        checkpoint_registration=registration,
        locked_finalists=lock,
        campaign_spec_sha256=campaign["content_hash"],
        split_manifest_sha256=args.split_manifest_sha256,
        hlt_cache_hashes=hlt_hashes,
        output_dir=args.output_dir,
        device=device,
        expected_event_count=args.expected_event_count,
    )
    write_immutable_json(args.output_dir / "metrics.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
