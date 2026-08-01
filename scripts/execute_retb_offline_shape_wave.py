#!/usr/bin/env python3
"""Aggregate all 21 locked canonical fusion design rows and select shapes."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    canonical_sha256,
    load_hashed_json,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.fusion import (  # noqa: E402
    build_fusion_model,
)
from teacher_logit_reco.relation_expert_token_bridge.fusion_cache import (  # noqa: E402
    load_frozen_token_cache,
    publish_frozen_token_cache,
)
from teacher_logit_reco.relation_expert_token_bridge.fusion_training import (  # noqa: E402
    OfflineFusionTrainingConfig,
    evaluate_fusion,
    infer_fusion_val_design,
    make_fusion_loader,
    train_frozen_fusion,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.selection import (  # noqa: E402
    HET_PHYSICS,
    build_uniform_shape_metrics,
    select_heterogeneous_allocations,
    select_offline_shapes,
)
from teacher_logit_reco.relation_expert_token_bridge.registry import (  # noqa: E402
    EXPERT_ORDER,
    TOKEN_SHAPES,
)
from teacher_logit_reco.relation_expert_token_bridge.step5 import (  # noqa: E402
    validate_stage_c_run_registry,
)
from teacher_logit_reco.relation_expert_token_bridge.step6 import (  # noqa: E402
    STAGE_D_SHAPES,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    authorize_dataset_access,
    load_and_validate_campaign_source,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _link(source: Path, target: Path) -> None:
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.is_symlink() or _sha256(target) != _sha256(source):
            raise FileExistsError(f"shape alias differs: {target}")
        return
    os.link(source, target)


def _optional_source_matches(observed: object, expected: object) -> bool:
    """Compare mapping-valued source records without requiring hashability."""

    return observed is None or observed == expected


def _shape_for(k: int, d: int = 128) -> str:
    matches = [
        shape
        for shape, row in TOKEN_SHAPES.items()
        if int(row["K"]) == int(k) and int(row["D"]) == int(d)
    ]
    if len(matches) != 1:
        raise ValueError(f"uniform shape for K={k}, D={d} differs")
    return matches[0]


def _stage_d_parent_allocations(
    *,
    selection: Mapping[str, object],
    heterogeneous: Mapping[str, object],
) -> dict[str, dict[str, int]]:
    allocations = {
        "S1_128": {name: 1 for name in EXPERT_ORDER},
        "SHAPE_COMPACT": {
            name: int(selection["SHAPE_COMPACT"]["K"])
            for name in EXPERT_ORDER
        },
        "SHAPE_HIGH": {
            name: int(selection["SHAPE_HIGH"]["K"])
            for name in EXPERT_ORDER
        },
        "HET_PHYSICS": {
            name: int(HET_PHYSICS[name]) for name in EXPERT_ORDER
        },
        "HET_SELECTED": {
            name: int(
                heterogeneous["HET_SELECTED"]["allocation"][name]
            )
            for name in EXPERT_ORDER
        },
        "HET_BEAM": {
            name: int(heterogeneous["HET_BEAM"]["allocation"][name])
            for name in EXPERT_ORDER
        },
    }
    if tuple(allocations) != STAGE_D_SHAPES:
        raise RuntimeError("Stage-D offline parent shape coverage differs")
    return allocations


def _analytical_fusion_flops(
    arrays: dict, *, variant: str
) -> int:
    banks = arrays["token_banks"]
    projections = sum(
        2 * int(values.shape[1]) * int(values.shape[2]) * 128
        for values in banks.values()
    )
    if variant == "F_POOLED_MLP":
        pooling = sum(
            max(0, int(values.shape[1]) - 1) * 128
            for values in banks.values()
        )
        return int(
            projections
            + pooling
            + 2 * (7 * 128) * 512
            + 2 * 512 * 10
        )
    if variant != "F_TOKEN_TRANSFORMER":
        raise ValueError("heterogeneous fusion FLOP variant differs")
    sequence = 1 + sum(int(values.shape[1]) for values in banks.values())
    width = 128
    block = (
        8 * sequence * width * width
        + 4 * sequence * sequence * width
        + 16 * sequence * width * width
    )
    return int(projections + 3 * block + 2 * width * 10)


def _combined_arrays(
    root: Path,
    *,
    allocation: dict[str, int],
    pipeline_seed: int,
    split: str,
) -> tuple[dict, dict]:
    manifests = {}
    token_banks = {}
    expert_logits = {}
    identities = labels = None
    for expert in EXPERT_ORDER:
        shape = _shape_for(allocation[expert])
        manifest, arrays = load_frozen_token_cache(
            root
            / "inputs"
            / "fusion_cache"
            / "offline"
            / shape
            / f"seed_{pipeline_seed}"
            / split
            / f"{split}_frozen_tokens.json"
        )
        manifests[expert] = manifest
        if identities is None:
            identities, labels = arrays["identities"], arrays["labels"]
        elif not np.array_equal(identities, arrays["identities"]) or not np.array_equal(
            labels, arrays["labels"]
        ):
            raise ValueError("heterogeneous source-cache identities differ")
        token_banks[expert] = arrays["token_banks"][expert]
        expert_logits[expert] = arrays["expert_logits"][expert]
    metadata = {
        "identity_manifest_sha256": manifests["BASE4"][
            "identity_manifest_sha256"
        ],
        "label_manifest_sha256": manifests["BASE4"]["label_manifest_sha256"],
        "expert_checkpoint_hashes": {
            expert: manifests[expert]["expert_checkpoint_hashes"][expert]
            for expert in EXPERT_ORDER
        },
        "expert_registration_hashes": {
            expert: manifests[expert]["expert_registration_hashes"][expert]
            for expert in EXPERT_ORDER
        },
    }
    return metadata, {
        "identities": identities,
        "labels": labels,
        "token_banks": token_banks,
        "expert_logits": expert_logits,
    }


def _fit_search_readout(
    *,
    root: Path,
    allocation: dict[str, int],
    readout_seed: int,
    bank_seed: int,
    variant: str,
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> dict:
    identity = {
        "allocation": allocation,
        "readout_seed": int(readout_seed),
        "bank_seed": int(bank_seed),
        "variant": variant,
    }
    output = (
        root
        / "runs"
        / "stage_c"
        / "heterogeneous_search"
        / canonical_sha256(identity)[:24]
    )
    evidence_path = output / "score.json"
    if evidence_path.is_file():
        return load_hashed_json(
            evidence_path,
            expected_contract="retb_heterogeneous_search_score_v2",
        )
    _, train = _combined_arrays(
        root,
        allocation=allocation,
        pipeline_seed=bank_seed,
        split="model_train",
    )
    _, stop = _combined_arrays(
        root,
        allocation=allocation,
        pipeline_seed=bank_seed,
        split="val_stop",
    )
    _, design = _combined_arrays(
        root,
        allocation=allocation,
        pipeline_seed=bank_seed,
        split="val_design",
    )
    torch.manual_seed(int(readout_seed))
    model = build_fusion_model(
        variant,
        bank_dimensions={
            expert: int(train["token_banks"][expert].shape[-1])
            for expert in EXPERT_ORDER
        },
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=5.0e-4, weight_decay=1.0e-4
    )
    train_loader = make_fusion_loader(
        train, batch_size=batch_size, seed=readout_seed, training=True
    )
    stop_loader = make_fusion_loader(
        stop, batch_size=batch_size, seed=0, training=False
    )
    best = None
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            banks = {
                name: value.to(device)
                for name, value in batch["token_banks"].items()
            }
            logits = {
                name: value.to(device)
                for name, value in batch["expert_logits"].items()
            }
            labels = batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            output_logits = model(
                token_banks=banks, expert_logits=logits
            )
            loss = torch.nn.functional.cross_entropy(output_logits, labels)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError(
                    "heterogeneous search gradient is nonfinite"
                )
            optimizer.step()
        metrics, _ = evaluate_fusion(
            model, stop_loader, device=device, split="val_stop"
        )
        key = (-metrics["accuracy"], metrics["cross_entropy"], epoch)
        if best is None or key < best[0]:
            best = (
                key,
                {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                },
            )
    model.load_state_dict(best[1], strict=True)
    design_loader = make_fusion_loader(
        design, batch_size=batch_size, seed=0, training=False
    )
    metrics, _ = evaluate_fusion(
        model, design_loader, device=device, split="val_design"
    )
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "best_model_val.pt"
    torch.save(
        {
            "contract": "retb_heterogeneous_search_readout_v1",
            "schema_version": 1,
            **identity,
            "model_state_dict": best[1],
        },
        checkpoint,
    )
    evidence = bind_source(
        with_content_hash(
            {
                "contract": "retb_heterogeneous_search_score_v2",
                "schema_version": 2,
                **identity,
                "accuracy": metrics["accuracy"],
                "cross_entropy": metrics["cross_entropy"],
                "parameter_count": sum(
                    value.numel() for value in model.parameters()
                ),
                "measured_flops": float(
                    _analytical_fusion_flops(design, variant=variant)
                ),
                "flop_method": (
                    "analytical_dense_matmul_multiply_add_v1"
                ),
                "checkpoint_sha256": _sha256(checkpoint),
                "readout_sha256": _sha256(checkpoint),
                "fusion_sha256": _sha256(checkpoint),
                "fixed_epoch_budget_completed": True,
                "performance_based_termination": False,
            }
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    write_immutable_json(evidence_path, evidence)
    return evidence


def _heterogeneous_selection(
    *,
    root: Path,
    miniature: bool,
    batch_size: int,
    device: torch.device,
) -> dict:
    epochs = 2 if miniature else 40
    memo = {}

    def score(
        kind: str, allocation: dict[str, int], seed: int
    ) -> dict:
        bank_seed = int(seed) if int(seed) in {101, 202, 303} else 101
        variant = (
            "F_TOKEN_TRANSFORMER"
            if kind == "F_TOKEN_TRANSFORMER"
            else "F_POOLED_MLP"
        )
        key = (
            kind,
            tuple(allocation[name] for name in EXPERT_ORDER),
            int(seed),
        )
        if key not in memo:
            memo[key] = _fit_search_readout(
                root=root,
                allocation=dict(allocation),
                readout_seed=int(seed),
                bank_seed=bank_seed,
                variant=variant,
                epochs=epochs,
                batch_size=batch_size,
                device=device,
            )
        return memo[key]

    selection = select_heterogeneous_allocations(
        greedy_scorer=lambda values, seed: score(
            "GREEDY_POOLED", values, seed
        ),
        beam_pooled_scorer=lambda values, seed: score(
            "BEAM_POOLED", values, seed
        ),
        beam_transformer_scorer=lambda values, seed: score(
            "F_TOKEN_TRANSFORMER", values, seed
        ),
    )
    result = dict(selection)
    result["search_score_artifact_hashes"] = sorted(
        {row["content_hash"] for row in memo.values()}
    )
    result["search_readouts_trained"] = len(memo)
    result.pop("content_hash")
    return bind_source(
        with_content_hash(result),
        source_snapshot=source_snapshot(REPO_ROOT),
    )


def _alias_experts(
    *,
    root: Path,
    alias: str,
    allocation: dict[str, int],
    registry: dict,
) -> None:
    lookup = {
        (
            str(row["configuration"]["shape_id"]),
            str(row["configuration"]["expert_id"]),
            int(row["seed"]),
        ): str(row["run_id"])
        for row in registry["expert_confirmation_rows"]
    }
    for seed in (101, 202, 303):
        for expert in EXPERT_ORDER:
            source_shape = _shape_for(allocation[expert])
            run_id = lookup[(source_shape, expert, seed)]
            source = (
                root
                / "runs"
                / "stage_c"
                / "offline_experts"
                / run_id
                / f"seed_{seed}"
            )
            target = (
                root
                / "selection"
                / "offline_experts"
                / alias
                / expert
                / f"seed_{seed}"
            )
            _link(
                source / "checkpoint_registration.json",
                target / "checkpoint_registration.json",
            )
            _link(
                source / "best_model_val.pt",
                target / "best_model_val.pt",
            )


def _train_heterogeneous_fusions(
    *,
    root: Path,
    alias: str,
    allocation: dict[str, int],
    registry: dict,
    campaign: dict,
    miniature: bool,
    batch_size: int,
    device: torch.device,
) -> None:
    architecture_sha = load_hashed_json(
        root / "registry" / "retb_offline_fusion.json"
    )["content_hash"]
    for seed in (101, 202, 303):
        cache_paths = {}
        for split in ("model_train", "val_stop", "val_design"):
            metadata, arrays = _combined_arrays(
                root,
                allocation=allocation,
                pipeline_seed=seed,
                split=split,
            )
            cache_root = (
                root
                / "inputs"
                / "fusion_cache"
                / "offline"
                / alias
                / f"seed_{seed}"
                / split
            )
            publish_frozen_token_cache(
                output_dir=cache_root,
                split=split,
                pipeline_seed=seed,
                shape_id=alias,
                identities=arrays["identities"],
                labels=arrays["labels"],
                token_banks=arrays["token_banks"],
                expert_logits=arrays["expert_logits"],
                expert_checkpoint_hashes=metadata[
                    "expert_checkpoint_hashes"
                ],
                expert_registration_hashes=metadata[
                    "expert_registration_hashes"
                ],
                identity_manifest_sha256=metadata[
                    "identity_manifest_sha256"
                ],
                label_manifest_sha256=metadata["label_manifest_sha256"],
                source_snapshot=source_snapshot(REPO_ROOT),
            )
            cache_paths[split] = (
                cache_root / f"{split}_frozen_tokens.json"
            )
        output = (
            root
            / "selection"
            / "offline_fusions"
            / alias
            / f"seed_{seed}"
        )
        model = build_fusion_model(
            "F_TOKEN_TRANSFORMER",
            bank_dimensions={expert: 128 for expert in EXPERT_ORDER},
        )
        train_frozen_fusion(
            model=model,
            model_train_manifest=cache_paths["model_train"],
            val_stop_manifest=cache_paths["val_stop"],
            output_dir=output,
            run_id=f"HET_FUSION_{alias}_seed_{seed}",
            run_registry_sha256=registry["content_hash"],
            global_determinism_sha256=campaign[
                "parent_artifact_hashes"
            ]["global_determinism"],
            fusion_architecture_sha256=architecture_sha,
            config=OfflineFusionTrainingConfig(
                seed=seed,
                maximum_epochs=2 if miniature else 40,
                batch_size=batch_size,
                campaign_profile=(
                    "miniature_test" if miniature else "production"
                ),
            ),
            device=device,
        )
        infer_fusion_val_design(
            model=model,
            checkpoint_path=output / "best_model_val.pt",
            val_design_manifest=cache_paths["val_design"],
            output_path=output / "val_design_inference.json",
            device=device,
        )


def _alias_uniform_fusions(
    *,
    root: Path,
    alias: str,
    source_shape: str,
    registry: dict,
) -> None:
    lookup = {
        (str(row["configuration"]["shape_id"]), int(row["seed"])): str(
            row["run_id"]
        )
        for row in registry["canonical_fusion_rows"]
    }
    for seed in (101, 202, 303):
        source = root / "runs" / "stage_c" / lookup[(source_shape, seed)]
        target = (
            root
            / "selection"
            / "offline_fusions"
            / alias
            / f"seed_{seed}"
        )
        for name in (
            "fusion_registration.json",
            "best_model_val.pt",
            "val_design_inference.json",
            "val_design_predictions.npz",
        ):
            _link(source / name, target / name)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    authorize_dataset_access(
        worker_role="design_worker", requested_resource="val_design"
    )
    registry = load_hashed_json(
        args.campaign_root / "registry" / "retb_stage_c_runs.json"
    )
    registry_sha = validate_stage_c_run_registry(registry)
    rows = []
    label_sha = None
    for member in registry["canonical_fusion_rows"]:
        run_id = str(member["run_id"])
        inference = load_hashed_json(
            args.campaign_root
            / "runs"
            / "stage_c"
            / run_id
            / "val_design_inference.json"
        )
        registration = load_hashed_json(
            args.campaign_root
            / "runs"
            / "stage_c"
            / run_id
            / "fusion_registration.json"
        )
        if (
            not _optional_source_matches(
                inference.get("source"), campaign.get("source")
            )
            or not _optional_source_matches(
                registration.get("source"), campaign.get("source")
            )
            or inference["checkpoint_sha256"]
            != registration["checkpoint_sha256"]
        ):
            raise ValueError("canonical fusion design lineage differs")
        current_label = inference["label_manifest_sha256"]
        if label_sha is None:
            label_sha = current_label
        elif label_sha != current_label:
            raise ValueError("canonical fusion design label lineage differs")
        metrics = inference["metrics"]
        rows.append(
            {
                "shape_id": member["configuration"]["shape_id"],
                "pipeline_seed": int(member["seed"]),
                "split": "val_design",
                "fusion_variant": "F_TOKEN_TRANSFORMER",
                "accuracy": metrics["accuracy"],
                "cross_entropy": metrics["cross_entropy"],
                "per_class_efficiency": metrics["per_class_efficiency"],
                "fusion_checkpoint_sha256": registration[
                    "checkpoint_sha256"
                ],
                "fusion_registration_sha256": registration["content_hash"],
                "frozen_cache_sha256": inference[
                    "cache_manifest_sha256"
                ],
                "metrics_artifact_sha256": inference["content_hash"],
                "label_manifest_sha256": current_label,
            }
        )
    metrics_artifact = bind_source(
        build_uniform_shape_metrics(
            rows=rows,
            stage_c_run_registry_sha256=registry_sha,
            val_design_label_manifest_sha256=label_sha,
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    selection = bind_source(
        select_offline_shapes(metrics_artifact),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    output = args.campaign_root / "selection" / "stage_c"
    write_immutable_json(
        output / "uniform_shape_metrics.json", metrics_artifact
    )
    write_immutable_json(
        output / "locked_offline_shapes.json", selection
    )
    write_immutable_json(
        args.campaign_root / "selection" / "retb_offline_shapes.json",
        selection,
    )
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    miniature = campaign["campaign_profile"] == "miniature_test"
    heterogeneous = _heterogeneous_selection(
        root=args.campaign_root,
        miniature=miniature,
        batch_size=args.batch_size,
        device=device,
    )
    write_immutable_json(
        args.campaign_root
        / "selection"
        / "retb_heterogeneous_shapes.json",
        heterogeneous,
    )
    allocations = _stage_d_parent_allocations(
        selection=selection,
        heterogeneous=heterogeneous,
    )
    for alias, allocation in allocations.items():
        _alias_experts(
            root=args.campaign_root,
            alias=alias,
            allocation=allocation,
            registry=registry,
        )
        if alias in {"S1_128", "SHAPE_COMPACT", "SHAPE_HIGH"}:
            _alias_uniform_fusions(
                root=args.campaign_root,
                alias=alias,
                source_shape=(
                    "S1_128"
                    if alias == "S1_128"
                    else selection[alias]["shape_id"]
                ),
                registry=registry,
            )
        else:
            _train_heterogeneous_fusions(
                root=args.campaign_root,
                alias=alias,
                allocation=allocation,
                registry=registry,
                campaign=campaign,
                miniature=miniature,
                batch_size=args.batch_size,
                device=device,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
