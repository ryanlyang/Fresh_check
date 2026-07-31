#!/usr/bin/env python3
"""Publish one identity-bound seven-expert frozen-token fusion cache."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.contracts import (
    load_hashed_json,
    require_sha256,
)
from teacher_logit_reco.relation_expert_token_bridge.expert_model import (
    RetbExpertModel,
    RetbParticleEncoder,
)
from teacher_logit_reco.relation_expert_token_bridge.expert_training import (
    OfflineExpertDataset,
    make_offline_expert_loader,
)
from teacher_logit_reco.relation_expert_token_bridge.fusion_cache import (
    publish_frozen_token_cache,
)
from teacher_logit_reco.relation_expert_token_bridge.registry import EXPERT_ORDER
from teacher_logit_reco.relation_expert_token_bridge.step5 import (
    resolve_expert_confirmation_training_run,
    validate_stage_c_run_registry,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import source_snapshot
from teacher_logit_reco.relation_expert_token_bridge.workflow import (
    authorize_dataset_access,
    load_and_validate_campaign_source,
)
from teacher_logit_reco.relational_part.ca_tree import unpack_tree_shard


def _named_paths(rows: Sequence[str]) -> dict[str, Path]:
    output = {}
    for row in rows:
        if "=" not in row:
            raise ValueError("--expert-registration requires EXPERT=PATH")
        name, path = row.split("=", 1)
        if name in output:
            raise ValueError("expert registration is duplicated")
        output[name] = Path(path)
    if set(output) != set(EXPERT_ORDER):
        raise ValueError("exactly seven expert registrations are required")
    return output


def _checkpoint_paths(rows: Sequence[str]) -> dict[str, Path]:
    output = {}
    for row in rows:
        if "=" not in row:
            raise ValueError("--expert-checkpoint requires EXPERT=PATH")
        name, path = row.split("=", 1)
        if name in output:
            raise ValueError("expert checkpoint is duplicated")
        output[name] = Path(path)
    if output and set(output) != set(EXPERT_ORDER):
        raise ValueError("exactly seven expert checkpoints are required")
    return output


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_trees(
    root: Path,
    *,
    split: str,
    identities: Sequence[str],
) -> list[dict]:
    split_root = root / f"{split}_exclusive_ca_v1" / "shards"
    by_identity = {}
    for shard in sorted(split_root.glob("shard_*.npz")):
        shard_identities, trees = unpack_tree_shard(shard)
        for identity, tree in zip(shard_identities, trees):
            key = str(identity)
            if key in by_identity:
                raise ValueError("REGION tree identity is duplicated")
            by_identity[key] = tree
    missing = [value for value in identities if value not in by_identity]
    if missing:
        raise ValueError(
            f"REGION trees lack {len(missing)} requested identities"
        )
    return [by_identity[value] for value in identities]


def _infer_expert(
    *,
    expert: str,
    run: dict,
    registration: dict,
    checkpoint_path: Path,
    arrays: dict[str, np.ndarray],
    relation_normalization: dict,
    region_normalization: dict,
    region_tree_root: Path,
    split: str,
    batch_size: int,
    device: torch.device,
    collect_relation_sensitivity: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    configuration = run["configuration"]
    if (
        registration.get("run_id") != run["run_id"]
        or registration.get("expert_id") != expert
        or registration.get("shape_id") != configuration["shape_id"]
        or registration.get("seed") != run["seed"]
        or registration.get("checkpoint_sha256")
        != _file_sha256(checkpoint_path)
    ):
        raise ValueError(f"{expert} registration/checkpoint lineage differs")
    identities = [str(value) for value in arrays["identities"].tolist()]
    trees = (
        _load_trees(
            region_tree_root,
            split=split,
            identities=identities,
        )
        if expert == "REGION"
        else None
    )
    dataset = OfflineExpertDataset(
        tokens=arrays["tokens"],
        mask=arrays["mask"],
        labels=arrays["labels"],
        identities=identities,
        region_trees=trees,
    )
    loader = make_offline_expert_loader(
        dataset,
        seed=int(run["seed"]),
        training=False,
        batch_size=int(batch_size),
    )
    weaver = importlib.import_module("weaver.nn.model.ParticleTransformer")
    encoder = RetbParticleEncoder(
        expert_id=expert,
        topology=configuration["topology"],
        weaver_module=weaver,
        normalization_artifact=(
            None if expert == "BASE4" else relation_normalization
        ),
        region_normalization_artifact=(
            region_normalization if expert == "REGION" else None
        ),
        measurement_embedding=configuration["measurement_embedding"],
        dual_base4_capacity_control=False,
        activation_checkpointing=False,
        particle_dropout=0.0,
    )
    model = RetbExpertModel(
        particle_encoder=encoder,
        shape_id=configuration["shape_id"],
        tokenizer_mode=configuration["tokenizer_mode"],
    )
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    token_rows = []
    logit_rows = []
    seen = []
    with torch.no_grad():
        for raw in loader:
            inputs = {}
            for target, candidates in {
                "features": ("features",),
                "vectors": ("vectors", "lorentz_vectors"),
                "mask": ("mask",),
                "raw_tokens": ("raw_tokens", "tokens"),
                "region_trees": ("region_trees",),
            }.items():
                for candidate in candidates:
                    if candidate in raw:
                        value = raw[candidate]
                        inputs[target] = (
                            value.to(device)
                            if isinstance(value, torch.Tensor)
                            else value
                        )
                        break
            output = model(return_details=True, **inputs)
            tokens = output["tokens"].detach().float().cpu()
            logits = output["logits"].detach().float().cpu()
            if not (
                bool(torch.isfinite(tokens).all())
                and bool(torch.isfinite(logits).all())
            ):
                raise FloatingPointError(
                    f"{expert} frozen-cache inference is nonfinite"
                )
            token_rows.append(tokens.numpy())
            logit_rows.append(logits.numpy())
            seen.extend(str(value) for value in raw["event_identities"])
    if seen != identities:
        raise ValueError(f"{expert} inference identity order differs")
    sensitivity: dict[str, np.ndarray] = {}
    if collect_relation_sensitivity and expert != "BASE4":
        for mode in ("zero", "within_jet_cyclic"):
            model.particle_encoder.set_semantic_relation_transform(mode)
            current = []
            with torch.no_grad():
                for raw in loader:
                    inputs = {}
                    for target, candidates in {
                        "features": ("features",),
                        "vectors": ("vectors", "lorentz_vectors"),
                        "mask": ("mask",),
                        "raw_tokens": ("raw_tokens", "tokens"),
                        "region_trees": ("region_trees",),
                    }.items():
                        for candidate in candidates:
                            if candidate in raw:
                                value = raw[candidate]
                                inputs[target] = (
                                    value.to(device)
                                    if isinstance(value, torch.Tensor)
                                    else value
                                )
                                break
                    output = model(return_details=True, **inputs)
                    current.append(
                        output["logits"].detach().float().cpu().numpy()
                    )
            sensitivity[mode] = np.concatenate(current, axis=0)
        model.particle_encoder.set_semantic_relation_transform("active")
    return (
        np.concatenate(token_rows, axis=0),
        np.concatenate(logit_rows, axis=0),
        sensitivity,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument(
        "--split",
        required=True,
        choices=("model_train", "scale_train", "val_stop", "val_design"),
    )
    parser.add_argument("--pipeline-seed", required=True, type=int)
    parser.add_argument("--shape-id", required=True)
    parser.add_argument(
        "--heterogeneous-allocation",
        action="store_true",
        help=(
            "Allow each expert registration to carry its own locked uniform "
            "shape while publishing one named heterogeneous bank."
        ),
    )
    parser.add_argument("--input-npz", required=True, type=Path)
    parser.add_argument("--input-manifest", type=Path)
    parser.add_argument("--expert-registration", action="append", default=[])
    parser.add_argument("--expert-checkpoint", action="append", default=[])
    parser.add_argument("--relation-normalization", type=Path)
    parser.add_argument("--region-normalization", type=Path)
    parser.add_argument("--region-tree-root", type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--identity-manifest-sha256")
    parser.add_argument("--label-manifest-sha256")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    role = (
        "scale_training_worker"
        if args.split == "scale_train"
        else "training_worker"
        if args.split != "val_design"
        else "design_worker"
    )
    authorize_dataset_access(worker_role=role, requested_resource=args.split)
    paths = _named_paths(args.expert_registration)
    checkpoints = _checkpoint_paths(args.expert_checkpoint)
    registrations = {
        name: load_hashed_json(path) for name, path in paths.items()
    }
    for name, registration in registrations.items():
        if (
            registration.get("contract") != "retb_offline_expert_registration_v1"
            or registration.get("expert_id") != name
            or (
                not args.heterogeneous_allocation
                and registration.get("shape_id") != args.shape_id
            )
            or registration.get("seed") != args.pipeline_seed
            or registration.get("fixed_epoch_budget_completed") is not True
        ):
            raise ValueError(f"expert registration {name} is ineligible")
    resolved = {
        "dry_run": bool(args.dry_run),
        "split": args.split,
        "pipeline_seed": args.pipeline_seed,
        "shape_id": args.shape_id,
        "input_npz": str(args.input_npz.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "expert_registration_hashes": {
            name: value["content_hash"] for name, value in registrations.items()
        },
    }
    if args.dry_run:
        print(json.dumps(resolved, indent=2, sort_keys=True))
        return 0
    with np.load(args.input_npz, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    prepared_fields = {"identities", "labels"} | {
        f"{kind}_{name}"
        for name in EXPERT_ORDER
        for kind in ("tokens", "logits")
    }
    if checkpoints:
        if set(arrays) != {"tokens", "mask", "labels", "identities"}:
            raise ValueError("raw frozen-cache input NPZ fields differ")
        if (
            args.input_manifest is None
            or args.relation_normalization is None
            or args.region_normalization is None
            or args.region_tree_root is None
        ):
            raise ValueError(
                "raw cache inference requires input and normalization lineage"
            )
        input_manifest = load_hashed_json(args.input_manifest)
        if (
            input_manifest.get("logical_role") != args.split
            or input_manifest.get("npz_sha256")
            != _file_sha256(args.input_npz)
            or input_manifest.get("source") != campaign.get("source")
        ):
            raise ValueError("offline input manifest lineage differs")
        relation = load_hashed_json(args.relation_normalization)
        region = load_hashed_json(args.region_normalization)
        registry = load_hashed_json(
            args.campaign_root / "registry" / "retb_stage_c_runs.json"
        )
        validate_stage_c_run_registry(registry)
        resolved_device = torch.device(
            "cuda"
            if args.device == "auto" and torch.cuda.is_available()
            else "cpu"
            if args.device == "auto"
            else args.device
        )
        token_banks = {}
        expert_logits = {}
        sensitivity_arrays = {}
        for name in EXPERT_ORDER:
            run = next(
                (
                    resolve_expert_confirmation_training_run(
                        registry,
                        run_id=str(row["run_id"]),
                        _registry_validated=True,
                    )
                    for row in registry["expert_confirmation_rows"]
                    if row["configuration"]["expert_id"] == name
                    and row["configuration"]["shape_id"]
                    == registrations[name]["shape_id"]
                    and int(row["seed"]) == args.pipeline_seed
                ),
                None,
            )
            if run is None:
                raise ValueError("expert confirmation parent is absent")
            (
                token_banks[name],
                expert_logits[name],
                sensitivity,
            ) = _infer_expert(
                expert=name,
                run=run,
                registration=registrations[name],
                checkpoint_path=checkpoints[name],
                arrays=arrays,
                relation_normalization=relation,
                region_normalization=region,
                region_tree_root=args.region_tree_root,
                split=args.split,
                batch_size=args.batch_size,
                device=resolved_device,
                collect_relation_sensitivity=args.split == "val_design",
            )
            for mode, values in sensitivity.items():
                sensitivity_arrays[f"{mode}_{name}"] = values
        identity_sha = input_manifest["identity_manifest_sha256"]
        label_sha = input_manifest["content_hash"]
    else:
        if set(arrays) != prepared_fields:
            raise ValueError("frozen-token input NPZ fields differ")
        token_banks = {
            name: arrays[f"tokens_{name}"] for name in EXPERT_ORDER
        }
        expert_logits = {
            name: arrays[f"logits_{name}"] for name in EXPERT_ORDER
        }
        identity_sha = args.identity_manifest_sha256
        label_sha = args.label_manifest_sha256
    manifest = publish_frozen_token_cache(
        output_dir=args.output_dir,
        split=args.split,
        pipeline_seed=args.pipeline_seed,
        shape_id=args.shape_id,
        identities=arrays["identities"],
        labels=arrays["labels"],
        token_banks=token_banks,
        expert_logits=expert_logits,
        expert_checkpoint_hashes={
            name: require_sha256(
                registrations[name]["checkpoint_sha256"],
                name=f"{name}.checkpoint_sha256",
            )
            for name in EXPERT_ORDER
        },
        expert_registration_hashes={
            name: registrations[name]["content_hash"] for name in EXPERT_ORDER
        },
        identity_manifest_sha256=identity_sha,
        label_manifest_sha256=label_sha,
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    if checkpoints and args.split == "val_design":
        expected = {
            f"{mode}_{expert}"
            for expert in EXPERT_ORDER
            if expert != "BASE4"
            for mode in ("zero", "within_jet_cyclic")
        }
        if set(sensitivity_arrays) != expected:
            raise RuntimeError(
                "val_design relation sensitivity coverage differs"
            )
        sensitivity_path = (
            args.output_dir / "val_design_relation_sensitivity.npz"
        )
        if sensitivity_path.exists():
            with np.load(sensitivity_path, allow_pickle=False) as prior:
                if set(prior.files) != expected or any(
                    not np.array_equal(
                        prior[name], sensitivity_arrays[name]
                    )
                    for name in expected
                ):
                    raise FileExistsError(
                        "relation sensitivity arrays differ on reuse"
                    )
        else:
            np.savez_compressed(
                sensitivity_path,
                **sensitivity_arrays,
            )
    print(json.dumps({**resolved, "manifest": manifest}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
