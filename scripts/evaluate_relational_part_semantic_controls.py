#!/usr/bin/env python3
"""Build the unary registry or evaluate the three val-select perturbations."""

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
    CONFIRMATION_SUMMARY_CONTRACT,
    RELATION_NORMALIZATION_ARTIFACT_CONTRACT,
    REGION_NORMALIZATION_CONTRACT,
    SCREENING_REGISTRY_CONTRACT,
    build_runtime_model,
    build_semantic_perturbation_artifact,
    build_unary_control_registry,
    build_unary_model_contract,
    build_val_select_loader,
    evaluate_semantic_perturbations,
    load_hashed_json,
    sha256_file,
    write_immutable_json,
)
from teacher_logit_reco.relational_part.workflow import (  # noqa: E402
    reject_final_test_paths,
)


def _common_normalizers(args):
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
    return normalization, region


def _build_unary(args) -> int:
    reject_final_test_paths(
        (args.confirmation_summary, args.normalization, args.output)
    )
    summary = load_hashed_json(
        args.confirmation_summary,
        expected_contract=CONFIRMATION_SUMMARY_CONTRACT,
    )
    normalization, _ = _common_normalizers(args)
    base_model_contract = load_hashed_json(args.base_model_contract)
    if summary["nominal_relational_winner_id"] != args.nominal_winner_run_id:
        raise ValueError("unary winner differs from confirmation")
    registry = build_unary_control_registry(
        nominal_winner_run_id=args.nominal_winner_run_id,
        unary_reference_run_id=args.unary_reference_run_id,
        families=args.families,
        reference_incremental_parameters=args.reference_incremental_parameters,
        reference_total_parameters=args.reference_total_parameters,
        base_total_parameters=args.base_total_parameters,
        confirmation_summary_sha256=summary["content_hash"],
        relation_normalization_sha256=normalization["content_hash"],
    )
    model_contract = build_unary_model_contract(
        unary_registry_sha256=registry["content_hash"],
        base_model_contract_sha256=base_model_contract["content_hash"],
        relation_normalization_sha256=normalization["content_hash"],
    )
    print(json.dumps(
        {"unary_registry": registry, "unary_model_contract": model_contract},
        indent=2,
        sort_keys=True,
    ))
    if not args.dry_run:
        write_immutable_json(args.output, registry)
        write_immutable_json(args.model_contract_output, model_contract)
    return 0


def _perturb(args) -> int:
    reject_final_test_paths(
        (
            args.cache_dir,
            args.checkpoint,
            args.checkpoint_registration,
            args.output,
        )
    )
    summary = load_hashed_json(
        args.confirmation_summary,
        expected_contract=CONFIRMATION_SUMMARY_CONTRACT,
    )
    registration = load_hashed_json(args.checkpoint_registration)
    model_contract = load_hashed_json(args.model_contract)
    screening = load_hashed_json(
        args.screening_registry, expected_contract=SCREENING_REGISTRY_CONTRACT
    )
    normalization, region = _common_normalizers(args)
    resolved = {
        "run_id": args.run_id,
        "seed": args.seed,
        "families": list(args.families),
        "split": "val_select",
        "checkpoint_sha256": registration["checkpoint_sha256"],
        "confirmation_summary_sha256": summary["content_hash"],
        "final_test_accessed": False,
        "dry_run": bool(args.dry_run),
    }
    print(json.dumps(resolved, indent=2, sort_keys=True))
    if args.dry_run:
        return 0
    if summary["nominal_relational_winner_id"] != args.run_id:
        raise ValueError("semantic perturbations must use the nominal winner")
    if sha256_file(args.checkpoint) != registration["checkpoint_sha256"]:
        raise ValueError("semantic checkpoint file hash mismatch")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = build_runtime_model(
        args.run_id,
        screening_registry=screening,
        normalization_artifact=normalization,
        region_normalization_artifact=region,
        selected_families=args.families,
    )
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    if checkpoint.get("model_contract_sha256") != model_contract["content_hash"]:
        raise ValueError("semantic checkpoint model contract mismatch")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    loader, _ = build_val_select_loader(
        cache_dir=args.cache_dir,
        seed=args.seed,
        families=args.families,
        tree_root=args.tree_root,
    )
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    model.to(device)
    metrics, diagnostics = evaluate_semantic_perturbations(
        model, loader, device=device
    )
    artifact = build_semantic_perturbation_artifact(
        nominal_winner_run_id=args.run_id,
        nominal_checkpoint_sha256=registration["checkpoint_sha256"],
        confirmation_summary_sha256=summary["content_hash"],
        metrics=metrics,
        diagnostics=diagnostics,
    )
    write_immutable_json(args.output, artifact)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    unary = subparsers.add_parser("build-unary-registry")
    unary.add_argument("--confirmation-summary", type=Path, required=True)
    unary.add_argument("--normalization", type=Path, required=True)
    unary.add_argument("--base-model-contract", type=Path, required=True)
    unary.add_argument("--region-normalization", type=Path)
    unary.add_argument("--nominal-winner-run-id", required=True)
    unary.add_argument("--unary-reference-run-id", required=True)
    unary.add_argument("--families", nargs="+", required=True)
    unary.add_argument("--reference-incremental-parameters", type=int, required=True)
    unary.add_argument("--reference-total-parameters", type=int, required=True)
    unary.add_argument("--base-total-parameters", type=int, required=True)
    unary.add_argument("--output", type=Path, required=True)
    unary.add_argument("--model-contract-output", type=Path, required=True)
    unary.add_argument("--dry-run", action="store_true")
    unary.set_defaults(handler=_build_unary)

    perturb = subparsers.add_parser("perturb")
    perturb.add_argument("--confirmation-summary", type=Path, required=True)
    perturb.add_argument("--screening-registry", type=Path, required=True)
    perturb.add_argument("--normalization", type=Path, required=True)
    perturb.add_argument("--region-normalization", type=Path)
    perturb.add_argument("--cache-dir", type=Path, required=True)
    perturb.add_argument("--tree-root", type=Path)
    perturb.add_argument("--run-id", required=True)
    perturb.add_argument("--seed", type=int, choices=(101, 202, 303), required=True)
    perturb.add_argument("--families", nargs="+", required=True)
    perturb.add_argument("--model-contract", type=Path, required=True)
    perturb.add_argument("--checkpoint", type=Path, required=True)
    perturb.add_argument("--checkpoint-registration", type=Path, required=True)
    perturb.add_argument("--device", default="auto")
    perturb.add_argument("--output", type=Path, required=True)
    perturb.add_argument("--dry-run", action="store_true")
    perturb.set_defaults(handler=_perturb)
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
