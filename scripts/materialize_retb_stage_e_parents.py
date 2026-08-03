#!/usr/bin/env python3
"""Materialize immutable seed/shape/expert parent bundles for Stage E."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.bridge_targets import (  # noqa: E402
    fit_bridge_token_normalizer,
)
from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    load_hashed_json,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.fusion import (  # noqa: E402
    build_fusion_model,
)
from teacher_logit_reco.relation_expert_token_bridge.fusion_cache import (  # noqa: E402
    load_frozen_token_cache,
)
from teacher_logit_reco.relation_expert_token_bridge.fusion_training import (  # noqa: E402
    evaluate_fusion,
    make_fusion_loader,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.registry import (  # noqa: E402
    EXPERT_ORDER,
    TOKEN_SHAPES,
)
from teacher_logit_reco.relation_expert_token_bridge.step6 import (  # noqa: E402
    validate_stage_d_run_registry,
)
from teacher_logit_reco.relation_expert_token_bridge.step7 import (  # noqa: E402
    validate_stage_e_template_registry,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    authorize_dataset_access,
    load_and_validate_campaign_source,
)


PARENT_BUNDLE_CONTRACT = "retb_stage_e_parent_bundle_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _link(source: Path, target: Path) -> None:
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"Stage-E parent source is absent: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.is_symlink() or _sha256(target) != _sha256(source):
            raise FileExistsError(f"Stage-E parent link differs: {target}")
        return
    os.link(source, target)


def _publish_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp.npz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **arrays)
        if path.exists():
            if path.is_symlink() or path.read_bytes() != temporary.read_bytes():
                raise FileExistsError(f"Stage-E dataset differs: {path}")
        else:
            os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return _sha256(path)


def _allocation(
    *,
    alias: str,
    uniform: Mapping[str, Any],
    heterogeneous: Mapping[str, Any],
) -> dict[str, tuple[int, int]]:
    if alias in {"SHAPE_COMPACT", "SHAPE_HIGH"}:
        row = uniform[alias]
        return {
            expert: (int(row["K"]), int(row["D"]))
            for expert in EXPERT_ORDER
        }
    if alias == "HET_PHYSICS":
        values = heterogeneous["HET_PHYSICS"]
    elif alias in {"HET_SELECTED", "HET_BEAM"}:
        values = heterogeneous[alias]["allocation"]
    else:
        raise ValueError("Stage-E shape alias differs")
    return {expert: (int(values[expert]), 128) for expert in EXPERT_ORDER}


def _uniform_shape(k: int, d: int) -> str:
    matches = [
        name
        for name, value in TOKEN_SHAPES.items()
        if int(value["K"]) == int(k) and int(value["D"]) == int(d)
    ]
    if len(matches) != 1:
        raise ValueError(f"no unique uniform cache for K={k}, D={d}")
    return matches[0]


def _offline_banks(
    root: Path,
    *,
    allocation: Mapping[str, tuple[int, int]],
    seed: int,
    split: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]:
    metadata: dict[str, Any] = {}
    tokens: dict[str, np.ndarray] = {}
    logits: dict[str, np.ndarray] = {}
    identities = labels = None
    for expert in EXPERT_ORDER:
        shape = _uniform_shape(*allocation[expert])
        manifest, arrays = load_frozen_token_cache(
            root
            / "inputs"
            / "fusion_cache"
            / "offline"
            / shape
            / f"seed_{seed}"
            / split
            / f"{split}_frozen_tokens.json"
        )
        if identities is None:
            identities = arrays["identities"]
            labels = arrays["labels"]
            metadata = {
                "identity_manifest_sha256": manifest[
                    "identity_manifest_sha256"
                ],
                "label_manifest_sha256": manifest["label_manifest_sha256"],
            }
        elif not np.array_equal(identities, arrays["identities"]) or not np.array_equal(
            labels, arrays["labels"]
        ):
            raise ValueError("Stage-E offline source caches are misaligned")
        tokens[expert] = arrays["token_banks"][expert]
        logits[expert] = arrays["expert_logits"][expert]
    metadata.update({"identities": identities, "labels": labels})
    return metadata, tokens, logits


def _bridge_parent_rows(
    registry: Mapping[str, Any],
) -> dict[tuple[str, str, int], Mapping[str, Any]]:
    rows = {}
    for row in registry["bridge_parent_expert_rows"]:
        config = row["configuration"]
        key = (
            str(config["shape_id"]),
            str(config["expert_id"]),
            int(row["seed"]),
        )
        if key in rows:
            raise ValueError("Stage-E HLT parent identity is duplicated")
        rows[key] = row
    return rows


def _native_banks(
    root: Path,
    *,
    parent_rows: Mapping[tuple[str, str, int], Mapping[str, Any]],
    alias: str,
    seed: int,
    split: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
    dict[str, str],
]:
    identities = labels = None
    banks: dict[str, np.ndarray] = {}
    particle_states: dict[str, np.ndarray] = {}
    particle_masks: dict[str, np.ndarray] = {}
    hashes: dict[str, str] = {}
    replica_ids = (0, 1, 2, 3) if split == "model_train" else (0,)
    for expert in EXPERT_ORDER:
        row = parent_rows[(alias, expert, int(seed))]
        parent = (
            root
            / "runs"
            / "stage_d"
            / "hlt_experts"
            / row["run_id"]
            / f"seed_{seed}"
        )
        manifest = load_hashed_json(parent / "native_output_manifest.json")
        registration = load_hashed_json(parent / "checkpoint_registration.json")
        if (
            manifest.get("contract") != "retb_native_hlt_expert_outputs_v5"
            or manifest.get("expert_id") != expert
            or manifest.get("expert_registration_sha256")
            != registration["content_hash"]
        ):
            raise ValueError("Stage-E native expert output lineage differs")
        hashes[expert] = manifest["content_hash"]
        replicas = []
        state_replicas = []
        mask_replicas = []
        for replica in replica_ids:
            record = manifest["files"][f"{split}_replica_{replica}"]
            path = parent / record["relative_path"]
            if _sha256(path) != record["file_sha256"]:
                raise ValueError("Stage-E native expert output bytes differ")
            with np.load(path, allow_pickle=False) as payload:
                current_ids = np.asarray(payload["identities"])
                current_labels = np.asarray(payload["labels"], dtype=np.int64)
                replicas.append(np.asarray(payload["tokens"], dtype=np.float32))
                state_replicas.append(
                    np.asarray(payload["particle_states"], dtype=np.float32)
                )
                mask_replicas.append(
                    np.asarray(payload["particle_mask"], dtype=bool)
                )
            if identities is None:
                identities, labels = current_ids, current_labels
            elif not np.array_equal(identities, current_ids) or not np.array_equal(
                labels, current_labels
            ):
                raise ValueError("Stage-E native expert populations differ")
        banks[expert] = (
            replicas[0] if len(replicas) == 1 else np.stack(replicas)
        )
        particle_states[expert] = (
            state_replicas[0]
            if len(state_replicas) == 1
            else np.stack(state_replicas)
        )
        particle_masks[expert] = (
            mask_replicas[0]
            if len(mask_replicas) == 1
            else np.stack(mask_replicas)
        )
    return (
        identities,
        labels,
        banks,
        particle_states,
        particle_masks,
        hashes,
    )


def _fusion_logits(
    *,
    checkpoint: Path,
    token_banks: Mapping[str, np.ndarray],
    expert_logits: Mapping[str, np.ndarray],
    identities: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    dimensions = {
        expert: int(token_banks[expert].shape[-1]) for expert in EXPERT_ORDER
    }
    model = build_fusion_model(
        "F_TOKEN_TRANSFORMER", bank_dimensions=dimensions
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    arrays = {
        "identities": identities,
        "labels": labels,
        "token_banks": token_banks,
        "expert_logits": expert_logits,
    }
    metrics, prediction = evaluate_fusion(
        model,
        make_fusion_loader(arrays, batch_size=512, seed=0, training=False),
        device=device,
        split="model_train",
    )
    del metrics
    if list(prediction["identities"]) != [
        str(value) for value in identities.tolist()
    ]:
        raise ValueError("Stage-E fusion identity order differs")
    return np.asarray(prediction["logits"], dtype=np.float32)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    templates = load_hashed_json(
        args.campaign_root / "registry" / "retb_stage_e_templates.json"
    )
    validate_stage_e_template_registry(templates)
    stage_d = load_hashed_json(
        args.campaign_root / "registry" / "retb_stage_d_runs.json"
    )
    validate_stage_d_run_registry(stage_d)
    uniform = load_hashed_json(
        args.campaign_root / "selection" / "retb_offline_shapes.json"
    )
    heterogeneous = load_hashed_json(
        args.campaign_root / "selection" / "retb_heterogeneous_shapes.json"
    )
    parent_rows = _bridge_parent_rows(stage_d)
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    for split, role in (
        ("model_train", "training_worker"),
        ("val_stop", "training_worker"),
        ("val_design", "design_worker"),
    ):
        authorize_dataset_access(worker_role=role, requested_resource=split)
    completed = 0
    for alias in templates["shapes"]:
        allocation = _allocation(
            alias=alias, uniform=uniform, heterogeneous=heterogeneous
        )
        for seed in templates["pipeline_seeds"]:
            split_payloads = {}
            for split in ("model_train", "val_stop", "val_design"):
                meta, offline_tokens, offline_logits = _offline_banks(
                    args.campaign_root,
                    allocation=allocation,
                    seed=int(seed),
                    split=split,
                )
                (
                    hlt_ids,
                    hlt_labels,
                    hlt_banks,
                    particle_states,
                    particle_masks,
                    hlt_hashes,
                ) = _native_banks(
                    args.campaign_root,
                    parent_rows=parent_rows,
                    alias=alias,
                    seed=int(seed),
                    split=split,
                )
                if not np.array_equal(meta["identities"], hlt_ids) or not np.array_equal(
                    meta["labels"], hlt_labels
                ):
                    raise ValueError("Stage-E offline/HLT populations differ")
                fusion_root = (
                    args.campaign_root
                    / "selection"
                    / "offline_fusions"
                    / alias
                    / f"seed_{seed}"
                )
                hybrid = _fusion_logits(
                    checkpoint=fusion_root / "best_model_val.pt",
                    token_banks=offline_tokens,
                    expert_logits=offline_logits,
                    identities=meta["identities"],
                    labels=meta["labels"],
                    device=device,
                )
                split_payloads[split] = (
                    meta,
                    offline_tokens,
                    offline_logits,
                    hlt_banks,
                    particle_states,
                    particle_masks,
                    hlt_hashes,
                    hybrid,
                )
            for expert in EXPERT_ORDER:
                parent = (
                    args.campaign_root
                    / "selection"
                    / "stage_e_parents"
                    / alias
                    / expert
                    / f"seed_{seed}"
                )
                offline_parent = (
                    args.campaign_root
                    / "selection"
                    / "offline_experts"
                    / alias
                    / expert
                    / f"seed_{seed}"
                )
                fusion_parent = (
                    args.campaign_root
                    / "selection"
                    / "offline_fusions"
                    / alias
                    / f"seed_{seed}"
                )
                hlt_row = parent_rows[(alias, expert, int(seed))]
                hlt_parent = (
                    args.campaign_root
                    / "runs"
                    / "stage_d"
                    / "hlt_experts"
                    / hlt_row["run_id"]
                    / f"seed_{seed}"
                )
                base_row = parent_rows[(alias, "BASE4", int(seed))]
                base_parent = (
                    args.campaign_root
                    / "runs"
                    / "stage_d"
                    / "hlt_experts"
                    / base_row["run_id"]
                    / f"seed_{seed}"
                )
                links = {
                    "t0_registration.json": offline_parent
                    / "checkpoint_registration.json",
                    "t0_best_model_val.pt": offline_parent / "best_model_val.pt",
                    "hlt_encoder_registration.json": hlt_parent
                    / "checkpoint_registration.json",
                    "hlt_encoder_best_model_val.pt": hlt_parent
                    / "best_model_val.pt",
                    "unbiased_particle_encoder_registration.json": base_parent
                    / "checkpoint_registration.json",
                    "unbiased_particle_encoder_best_model_val.pt": base_parent
                    / "best_model_val.pt",
                    "t0_fusion_registration.json": fusion_parent
                    / "fusion_registration.json",
                    "t0_fusion_best_model_val.pt": fusion_parent
                    / "best_model_val.pt",
                }
                for name, source in links.items():
                    _link(source, parent / name)
                datasets = {}
                for split, values in split_payloads.items():
                    (
                        meta,
                        offline_tokens,
                        offline_logits,
                        hlt_banks,
                        particle_states,
                        particle_masks,
                        hlt_hashes,
                        hybrid,
                    ) = values
                    arrays = {
                        "identities": meta["identities"],
                        "labels": meta["labels"],
                        "unbiased_particle_states": particle_states[
                            "BASE4"
                        ],
                        "particle_mask": particle_masks["BASE4"],
                        "target_tokens": offline_tokens[expert],
                        "target_expert_logits": offline_logits[expert],
                        "target_hybrid_logits": hybrid,
                        **{
                            f"hlt_tokens_{name}": hlt_banks[name]
                            for name in EXPERT_ORDER
                        },
                        **{
                            f"t0_tokens_{name}": offline_tokens[name]
                            for name in EXPERT_ORDER
                        },
                        **{
                            f"relation_particle_states_{name}": (
                                particle_states[name]
                            )
                            for name in ("PT", "TRACK", "REGION")
                        },
                        **{
                            f"relation_particle_mask_{name}": (
                                particle_masks[name]
                            )
                            for name in ("PT", "TRACK", "REGION")
                        },
                    }
                    path = parent / f"{split}_pilot_dataset.npz"
                    datasets[split] = {
                        "file_sha256": _publish_npz(path, arrays),
                        "identity_manifest_sha256": meta[
                            "identity_manifest_sha256"
                        ],
                        "label_manifest_sha256": meta[
                            "label_manifest_sha256"
                        ],
                        "native_output_manifest_hashes": hlt_hashes,
                    }
                train_values = split_payloads["model_train"]
                train_meta, train_tokens = train_values[0], train_values[1]
                t0_registration = load_hashed_json(
                    parent / "t0_registration.json"
                )
                normalizer = bind_source(
                    fit_bridge_token_normalizer(
                        train_tokens[expert],
                        expert_id=expert,
                        shape_id=alias,
                        target_checkpoint_sha256=t0_registration[
                            "checkpoint_sha256"
                        ],
                        token_cache_sha256=datasets["model_train"][
                            "file_sha256"
                        ],
                        identity_manifest_sha256=train_meta[
                            "identity_manifest_sha256"
                        ],
                    ),
                    source_snapshot=source_snapshot(REPO_ROOT),
                )
                write_immutable_json(parent / "target_normalizer.json", normalizer)
                bundle = bind_source(
                    with_content_hash(
                        {
                            "contract": PARENT_BUNDLE_CONTRACT,
                            "schema_version": 1,
                            "shape_id": alias,
                            "expert_id": expert,
                            "pipeline_seed": int(seed),
                            "allocation": {
                                name: list(allocation[name])
                                for name in EXPERT_ORDER
                            },
                            "link_hashes": {
                                name: _sha256(parent / name)
                                for name in sorted(links)
                            },
                            "dataset_evidence": datasets,
                            "target_normalizer_sha256": normalizer[
                                "content_hash"
                            ],
                            "R_MULTI_train_replica_count": 4,
                            "fixed_validation_replica": 0,
                        }
                    ),
                    source_snapshot=source_snapshot(REPO_ROOT),
                )
                write_immutable_json(parent / "parent_bundle.json", bundle)
                completed += 1
    if completed != 105:
        raise RuntimeError("Stage-E parent bundle coverage differs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
