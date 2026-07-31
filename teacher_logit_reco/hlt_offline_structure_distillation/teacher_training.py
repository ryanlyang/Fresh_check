"""Numerical production of the two mandatory HOSD offline teachers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import hashlib
import numpy as np
import torch
from jetclass_fresh.jetclass_data import (
    JetIdentity,
    JetView,
    load_offline_view,
    load_split_manifest,
)
from teacher_logit_reco.relational_part import (
    CHECKPOINT_REGISTRATION_CONTRACT,
    TrainingConfig,
    build_runtime_model,
    load_hashed_json,
    profile_model_resources,
    train_relational_model,
)
from teacher_logit_reco.relational_part.data import (
    RelationalJetDataset,
    make_relational_loader,
)
from teacher_logit_reco.relational_part.runtime import load_region_tree_split

from .contracts import canonical_sha256
from .contracts import INPUT_VIEW_MANIFEST_CONTRACT
from .teachers import MANDATORY_TEACHERS, TEACHER_SEED


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _subset_view(
    view: JetView,
    *,
    identity_rows: list[Mapping[str, Any]],
    logical_split: str,
) -> JetView:
    requested = [
        f"{row['file']}#{int(row['entry'])}" for row in identity_rows
    ]
    positions = {
        identity.key(): index for index, identity in enumerate(view.jet_ids)
    }
    if len(positions) != len(view.jet_ids):
        raise ValueError("offline view contains duplicate identities")
    missing = [identity for identity in requested if identity not in positions]
    if missing:
        raise ValueError(f"{logical_split} identities are absent from offline view")
    indices = [positions[identity] for identity in requested]
    return JetView(
        tokens=view.tokens[indices],
        mask=view.mask[indices],
        labels=view.labels[indices],
        jet_ids=[view.jet_ids[index] for index in indices],
        split=logical_split,
        metadata={
            **view.metadata,
            "logical_role": logical_split,
            "identity_order_sha256": canonical_sha256(requested),
        },
    )


def _subset_trees(
    trees: list[Mapping[str, Any]],
    *,
    full_view: JetView,
    subset_view: JetView,
) -> list[Mapping[str, Any]]:
    positions = {
        identity.key(): index for index, identity in enumerate(full_view.jet_ids)
    }
    return [trees[positions[identity.key()]] for identity in subset_view.jet_ids]


def _load_scale_training_view(
    input_npz: str | Path,
    labels_npz: str | Path,
    *,
    expected_source: Mapping[str, Any],
) -> tuple[JetView, str, str]:
    input_path = Path(input_npz)
    manifest = load_hashed_json(
        input_path.with_suffix(input_path.suffix + ".json"),
        expected_contract=INPUT_VIEW_MANIFEST_CONTRACT,
    )
    input_sha = _file_sha256(input_path)
    if (
        manifest.get("source") != dict(expected_source)
        or manifest.get("split") != "scale_train"
        or manifest.get("view_kind") != "canonical_offline"
        or manifest.get("replica_id") is not None
        or manifest.get("npz_sha256") != input_sha
    ):
        raise ValueError("scale offline training input lineage differs")
    with np.load(input_path, allow_pickle=False) as payload:
        if not {"identity", "raw_tokens", "mask"}.issubset(payload.files):
            raise ValueError("scale offline training input fields differ")
        identities = tuple(str(value) for value in payload["identity"].tolist())
        tokens = np.asarray(payload["raw_tokens"], dtype=np.float32)
        mask = np.asarray(payload["mask"], dtype=bool)
    labels_path = Path(labels_npz)
    with np.load(labels_path, allow_pickle=False) as payload:
        if not {"identities", "labels"}.issubset(payload.files):
            raise ValueError("scale offline training label fields differ")
        label_identities = tuple(
            str(value) for value in payload["identities"].tolist()
        )
        labels = np.asarray(payload["labels"], dtype=np.int64)
    if (
        identities != label_identities
        or labels.shape != (len(identities),)
        or len(identities) != len(set(identities))
    ):
        raise ValueError("scale offline training identity/label join differs")
    jet_ids = []
    for identity, label in zip(identities, labels):
        file_name, separator, entry = identity.rpartition("#")
        if not separator or not file_name or not entry.isdigit():
            raise ValueError("scale offline training identity is noncanonical")
        jet_ids.append(
            JetIdentity(file=file_name, entry=int(entry), label=int(label))
        )
    view = JetView(
        tokens=tokens,
        mask=mask,
        labels=labels,
        jet_ids=jet_ids,
        split="scale_train",
        metadata={
            "logical_role": "scale_train",
            "identity_order_sha256": canonical_sha256(list(identities)),
        },
    )
    return (
        view,
        manifest["content_hash"],
        _file_sha256(labels_path),
    )


def train_offline_teacher(
    *,
    teacher_id: str,
    campaign: Mapping[str, Any],
    offline_manifest_path: str | Path,
    data_dir: str | Path | None,
    validation_partition: Mapping[str, Any],
    screening_registry: Mapping[str, Any],
    relation_registry: Mapping[str, Any],
    relation_normalizer: Mapping[str, Any],
    region_normalizer: Mapping[str, Any] | None,
    global_determinism: Mapping[str, Any],
    model_contract: Mapping[str, Any],
    output_dir: str | Path,
    tree_root: str | Path | None,
    device: str,
    miniature: bool,
    training_split: str = "model_train",
    validation_tree_root: str | Path | None = None,
    training_input_npz: str | Path | None = None,
    training_labels_npz: str | Path | None = None,
) -> dict[str, Any]:
    """Train one real offline-input teacher and return its registration."""

    if teacher_id not in MANDATORY_TEACHERS:
        raise ValueError("unknown mandatory HOSD teacher")
    if training_split not in {"model_train", "scale_train"}:
        raise ValueError("offline teacher training split differs")
    manifest = load_split_manifest(offline_manifest_path)
    if training_split == "scale_train":
        if training_input_npz is None or training_labels_npz is None:
            raise ValueError(
                "scale teacher training requires authenticated input and labels"
            )
        train_view, scale_input_manifest_sha, scale_labels_sha = (
            _load_scale_training_view(
                training_input_npz,
                training_labels_npz,
                expected_source=campaign["source"],
            )
        )
    else:
        if training_input_npz is not None or training_labels_npz is not None:
            raise ValueError("500k teacher cannot consume scale training inputs")
        train_view = load_offline_view(
            manifest,
            training_split,
            data_dir=data_dir,
            verify_label_branches=False,
        )
    full_val = load_offline_view(
        manifest,
        "model_val",
        data_dir=data_dir,
        verify_label_branches=False,
    )
    val_stop = _subset_view(
        full_val,
        identity_rows=list(validation_partition["roles"]["val_stop"]),
        logical_split="val_stop",
    )
    val_design = _subset_view(
        full_val,
        identity_rows=list(validation_partition["roles"]["val_design"]),
        logical_split="val_design",
    )
    uses_region = teacher_id == "O_FULLREL"
    if uses_region and tree_root is None:
        raise ValueError("O_FULLREL requires the authenticated offline tree root")
    train_trees = val_stop_trees = val_design_trees = None
    if uses_region:
        train_trees = load_region_tree_split(
            tree_root,
            split=training_split,
            expected_identities=train_view.jet_ids,
        )
        validation_root = validation_tree_root or tree_root
        val_stop_trees = load_region_tree_split(
            validation_root,
            split="val_stop",
            expected_identities=val_stop.jet_ids,
        )
        val_design_trees = load_region_tree_split(
            validation_root,
            split="val_design",
            expected_identities=val_design.jet_ids,
        )
    datasets = {
        training_split: RelationalJetDataset(
            train_view, region_trees=train_trees
        ),
        "val_stop": RelationalJetDataset(
            val_stop,
            region_trees=val_stop_trees,
        ),
        "val_design": RelationalJetDataset(
            val_design,
            region_trees=val_design_trees,
        ),
    }
    loaders = {
        role: make_relational_loader(
            dataset,
            seed=TEACHER_SEED,
            training=role == training_split,
        )
        for role, dataset in datasets.items()
    }
    run_id = "RPT_BASE" if teacher_id == "O_BASE" else "RPT_FULL_ALL"
    resolved_device = (
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if device == "auto" else device
    model = build_runtime_model(
        run_id,
        screening_registry=screening_registry,
        normalization_artifact=relation_normalizer,
        region_normalization_artifact=region_normalizer,
    )
    config = (
        TrainingConfig(
            seed=TEACHER_SEED,
            maximum_epochs=2,
            minimum_epochs=2,
            early_stop_patience=2,
            campaign_profile="miniature_test",
        )
        if miniature
        else TrainingConfig(
            seed=TEACHER_SEED,
            minimum_epochs=40,
            accuracy_window=0.0,
            campaign_profile="hosd_teacher",
        )
    )
    example = next(iter(loaders["val_stop"]))
    resource_profile = profile_model_resources(
        model,
        example,
        device=resolved_device,
        model_contract_sha256=model_contract["content_hash"],
    )
    lineage = {
        "campaign_spec": campaign["content_hash"],
        "split_manifest": validation_partition["source_manifest_sha256"],
        "training_split": canonical_sha256(training_split),
        "validation_partition": validation_partition["content_hash"],
        "relation_normalization": relation_normalizer["content_hash"],
        "global_determinism": global_determinism["content_hash"],
    }
    if training_split == "scale_train":
        lineage["scale_training_input_view"] = scale_input_manifest_sha
        lineage["scale_training_labels"] = scale_labels_sha
    if region_normalizer is not None and uses_region:
        lineage["region_normalization"] = region_normalizer["content_hash"]
        roots = {
            training_split: Path(tree_root),
            "val_stop": Path(validation_tree_root or tree_root),
            "val_design": Path(validation_tree_root or tree_root),
        }
        for split, split_root in roots.items():
            tree_manifest = load_hashed_json(
                split_root
                / f"{split}_exclusive_ca_v1"
                / "manifest.json"
            )
            lineage[f"offline_region_tree_{split}"] = tree_manifest[
                "content_hash"
            ]
    registration_path = Path(output_dir) / "checkpoint_registration.json"
    if registration_path.is_file():
        registration = load_hashed_json(
            registration_path,
            expected_contract=CHECKPOINT_REGISTRATION_CONTRACT,
        )
        if (
            registration.get("run_id") != run_id
            or registration.get("seed") != TEACHER_SEED
            or registration.get("model_contract_sha256")
            != model_contract["content_hash"]
            or registration.get("inference_input_role") != "offline_teacher"
        ):
            raise ValueError("reusable teacher registration lineage differs")
        checkpoint = Path(output_dir) / registration["checkpoint_file"]
        if not checkpoint.is_file():
            raise FileNotFoundError("reusable teacher checkpoint is absent")
        import hashlib

        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != registration[
            "checkpoint_sha256"
        ]:
            raise ValueError("reusable teacher checkpoint bytes drifted")
        return registration
    return train_relational_model(
        model=model,
        train_loader=loaders[training_split],
        val_stop_loader=loaders["val_stop"],
        val_select_loader=loaders["val_design"],
        output_dir=output_dir,
        run_id=run_id,
        model_contract_sha256=model_contract["content_hash"],
        run_registry_sha256=screening_registry["content_hash"],
        relation_registry_sha256=relation_registry["content_hash"],
        global_determinism_sha256=global_determinism["content_hash"],
        lineage_hashes=lineage,
        config=config,
        device=resolved_device,
        resource_profile=resource_profile,
        resume=True,
        inference_input_role="offline_teacher",
    )


__all__ = ["train_offline_teacher"]
