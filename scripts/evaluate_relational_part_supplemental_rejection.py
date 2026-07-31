#!/usr/bin/env python3
"""Compute a clearly post-hoc QCD-rejection working point on val_select."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Any, Mapping


def _source_root(argv: list[str]) -> Path:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--source-root", type=Path, required=True)
    parsed, _ = bootstrap.parse_known_args(argv)
    root = parsed.source_root.resolve()
    if not (root / "teacher_logit_reco" / "relational_part").is_dir():
        raise FileNotFoundError(f"relational-part source is absent: {root}")
    sys.path.insert(0, str(root))
    return root


SOURCE_ROOT = _source_root(sys.argv[1:])

import numpy as np  # noqa: E402
import torch  # noqa: E402

from teacher_logit_reco.relational_part import (  # noqa: E402
    CLASS_NAMES,
    RELATION_NORMALIZATION_ARTIFACT_CONTRACT,
    REGION_NORMALIZATION_CONTRACT,
    SCREENING_REGISTRY_CONTRACT,
    build_runtime_model,
    build_val_select_loader,
    load_hashed_json,
    qcd_signal_rejection,
    sha256_file,
    validate_campaign_source,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.relational_part.evaluation import (  # noqa: E402
    model_forward,
)


CONTRACT = "relational_part_posthoc_val_select_rejection_v1"


def _move_batch(
    batch: Mapping[str, Any], device: torch.device
) -> dict[str, Any]:
    return {
        name: value.to(device) if isinstance(value, torch.Tensor) else value
        for name, value in batch.items()
    }


def _output_name(target_efficiency: float) -> str:
    token = format(float(target_efficiency), ".12g").replace(".", "p")
    return f"qcd_rejection_signal_efficiency_{token}.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--target-efficiency", type=float, default=0.75)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tree-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    source_root = args.source_root.resolve()
    if source_root != SOURCE_ROOT:
        raise ValueError("--source-root changed after import bootstrap")
    campaign_root = args.campaign_root.resolve()
    if not 0.0 < args.target_efficiency <= 1.0:
        raise ValueError("--target-efficiency must lie in (0,1]")

    campaign = load_hashed_json(campaign_root / "campaign_spec.json")
    validate_campaign_source(campaign, repo_root=source_root)
    screening = load_hashed_json(
        campaign_root / "registry" / "screening_registry.json",
        expected_contract=SCREENING_REGISTRY_CONTRACT,
    )
    normalization = load_hashed_json(
        campaign_root / "inputs" / "relation_normalization.json",
        expected_contract=RELATION_NORMALIZATION_ARTIFACT_CONTRACT,
    )
    region_path = campaign_root / "inputs" / "region_normalization.json"
    region = (
        load_hashed_json(
            region_path, expected_contract=REGION_NORMALIZATION_CONTRACT
        )
        if region_path.is_file()
        else None
    )
    model_contract = load_hashed_json(
        campaign_root
        / "registry"
        / "model_contracts"
        / f"{args.run_id}.json"
    )
    run_root = (
        campaign_root / "runs" / args.run_id / f"seed_{int(args.seed)}"
    )
    registration = load_hashed_json(
        run_root / "checkpoint_registration.json",
        expected_contract="relational_part_checkpoint_registration_v2",
    )
    official_metrics = load_hashed_json(
        run_root / "val_select_metrics.json",
        expected_contract="relational_part_evaluation_v1",
    )
    checkpoint_path = run_root / str(registration["checkpoint_file"])

    if model_contract.get("run_id") != args.run_id:
        raise ValueError("model contract run_id mismatch")
    if registration.get("run_id") != args.run_id:
        raise ValueError("checkpoint registration run_id mismatch")
    if int(registration.get("seed", -1)) != int(args.seed):
        raise ValueError("checkpoint registration seed mismatch")
    if registration.get("model_contract_sha256") != model_contract[
        "content_hash"
    ]:
        raise ValueError("checkpoint registration model-contract mismatch")
    if registration.get("val_select_metrics_sha256") != official_metrics[
        "content_hash"
    ]:
        raise ValueError("official val_select metrics hash mismatch")
    if registration.get("val_select_used_for_checkpoint_selection") is not False:
        raise ValueError("val_select was unexpectedly used for checkpoint selection")
    if sha256_file(checkpoint_path) != registration["checkpoint_sha256"]:
        raise ValueError("checkpoint file hash mismatch")

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

    families = tuple(model_contract.get("new_relation_families", ()))
    tree_root = (
        args.tree_root
        if args.tree_root is not None
        else campaign_root / "inputs" / "relation_tree_cache"
    )
    model = build_runtime_model(
        args.run_id,
        screening_registry=screening,
        normalization_artifact=normalization,
        region_normalization_artifact=region,
    )
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if checkpoint.get("model_contract_sha256") != model_contract["content_hash"]:
        raise ValueError("checkpoint payload model-contract mismatch")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    loader, hlt_hash = build_val_select_loader(
        cache_dir=campaign_root / "inputs" / "hlt_cache",
        seed=args.seed,
        families=families,
        tree_root=tree_root,
    )
    if hlt_hash != registration["lineage_hashes"]["hlt_stack_val"]:
        raise ValueError("opened stack_val cache hash mismatch")

    logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for raw in loader:
            batch = _move_batch(raw, device)
            output = model_forward(model, batch)
            logits.append(output.detach().float().cpu().numpy())
            labels.append(batch["labels"].detach().long().cpu().numpy())
    if not logits:
        raise ValueError("stack_val loader is empty")
    all_logits = np.concatenate(logits, axis=0)
    all_labels = np.concatenate(labels, axis=0)

    rejection = {
        CLASS_NAMES[index]: qcd_signal_rejection(
            all_logits,
            all_labels,
            signal_index=index,
            target_efficiency=args.target_efficiency,
        )
        for index in range(1, len(CLASS_NAMES))
    }
    artifact = with_content_hash(
        {
            "contract": CONTRACT,
            "schema_version": 1,
            "scientific_status": "supplemental_post_hoc_diagnostic_only",
            "official_campaign_metric": False,
            "eligible_for_model_selection": False,
            "changes_campaign_dependencies": False,
            "split": "val_select",
            "run_id": args.run_id,
            "seed": int(args.seed),
            "target_signal_efficiency": float(args.target_efficiency),
            "event_count": int(len(all_labels)),
            "class_order": list(CLASS_NAMES),
            "qcd_signal_rejection": rejection,
            "campaign_spec_sha256": campaign["content_hash"],
            "model_contract_sha256": model_contract["content_hash"],
            "checkpoint_registration_sha256": registration["content_hash"],
            "checkpoint_sha256": registration["checkpoint_sha256"],
            "official_val_select_metrics_sha256": official_metrics[
                "content_hash"
            ],
            "hlt_stack_val_sha256": hlt_hash,
            "campaign_source": model_contract.get("source"),
            "diagnostic_script_sha256": sha256_file(Path(__file__)),
            "calculation_dtype": "float64",
        }
    )
    output = args.output or (
        campaign_root
        / "supplemental_diagnostics"
        / args.run_id
        / f"seed_{int(args.seed)}"
        / _output_name(args.target_efficiency)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file():
        existing = json.loads(output.read_text(encoding="utf-8"))
        validate_content_hash(existing, expected_contract=CONTRACT)
        if existing != artifact:
            raise FileExistsError(
                f"supplemental output exists with different content: {output}"
            )
    else:
        write_immutable_json(output, artifact)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
