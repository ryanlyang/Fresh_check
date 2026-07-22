"""Training and immutable checkpoints for frozen-feature R0--R4 fusion heads."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np
import torch
from torch.nn import functional as functional

from jetclass_fresh.fusion import load_prediction_block, validate_prediction_alignment
from jetclass_fresh.hlt_baseline import resolve_device
from jetclass_fresh.jetclass_data import LABEL_NAMES

from .fusion_atomic import publish_temporary_file
from .fusion_campaign import (
    FUSION_FAMILY_REPRESENTATION,
    FUSION_FIT_SPLIT,
    FUSION_HEAD_SEEDS,
    FUSION_SELECTION_SPLIT,
    default_fusion_candidate_specs,
    default_fusion_group_specs,
    stable_fusion_json_hash,
)
from .fusion_features import (
    LOCAL_RESIDUAL_FIELD_FUSION_FEATURE_MANIFEST_CONTRACT,
    LOCAL_RESIDUAL_FIELD_FUSION_FEATURE_CONTRACT,
    require_development_prediction_sources,
)
from .fusion_metrics import (
    local_residual_field_binary_projection_metrics,
    local_residual_field_multiclass_metrics,
)
from .fusion_models import (
    LOCAL_RESIDUAL_FIELD_REPRESENTATION_FUSION_MODEL_CONTRACT,
    FrozenRepresentationFusionHead,
    RepresentationFusionOutput,
    representation_fusion_diagnostics,
)
from .fusion_seed_control import sha256_file


LOCAL_RESIDUAL_FIELD_REPRESENTATION_FUSION_TRAIN_CONTRACT = (
    "local_residual_field_representation_fusion_train_v1"
)
LOCAL_RESIDUAL_FIELD_REPRESENTATION_FUSION_CHECKPOINT_CONTRACT = (
    "local_residual_field_representation_fusion_checkpoint_v1"
)
REPRESENTATION_FUSION_SPLITS = (FUSION_FIT_SPLIT, FUSION_SELECTION_SPLIT)


@dataclass(frozen=True)
class RepresentationFusionSplitData:
    embedding_a: np.ndarray
    embedding_b: np.ndarray
    logits_a: np.ndarray
    logits_b: np.ndarray
    labels: np.ndarray

    def __post_init__(self) -> None:
        rows = len(self.labels)
        if np.asarray(self.labels).shape != (rows,):
            raise ValueError("labels must have shape [N]")
        for name in ("embedding_a", "embedding_b", "logits_a", "logits_b"):
            value = np.asarray(getattr(self, name))
            if value.ndim != 2 or value.shape[0] != rows or not np.isfinite(value).all():
                raise ValueError(f"{name} must be a finite aligned matrix")
        if np.asarray(self.logits_a).shape != (rows, len(LABEL_NAMES)):
            raise ValueError("member A logits do not use the ten-class schema")
        if np.asarray(self.logits_b).shape != (rows, len(LABEL_NAMES)):
            raise ValueError("member B logits do not use the ten-class schema")


@dataclass
class RepresentationFusionTrainConfig:
    campaign_id: str
    group_id: str
    candidate_id: str
    feature_root: str
    prediction_sources: str
    source_artifact_audit: str
    checkpoint_path: str
    report_path: str
    seed: int = 5101
    hidden_width: int = 64
    dropout: float = 0.0
    weight_decay: float = 1.0e-5
    learning_rate: float = 1.0e-3
    epochs: int = 80
    patience: int = 10
    batch_size: int = 2048
    device: str = "auto"
    overwrite: bool = False

    def __post_init__(self) -> None:
        if int(self.seed) not in FUSION_HEAD_SEEDS:
            raise ValueError(f"representation head seed must be one of {FUSION_HEAD_SEEDS}")
        if int(self.hidden_width) not in {64, 128}:
            raise ValueError("hidden_width must be 64 or 128")
        if float(self.dropout) not in {0.0, 0.1}:
            raise ValueError("dropout must be 0.0 or 0.1")
        if float(self.weight_decay) not in {1.0e-5, 1.0e-4}:
            raise ValueError("weight_decay must be one of the locked grid values")
        if int(self.epochs) <= 0 or int(self.patience) <= 0 or int(self.batch_size) <= 0:
            raise ValueError("epochs, patience, and batch_size must be positive")


def _validate_hashed_json(path: Path, *, contract: str, hash_key: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("contract") != contract:
        raise ValueError(f"artifact contract mismatch: {path}")
    unsigned = dict(payload)
    stored = unsigned.pop(hash_key, None)
    if stored != stable_fusion_json_hash(unsigned):
        raise ValueError(f"artifact hash mismatch: {path}")
    return dict(payload)


def _load_member_split(
    feature_root: Path,
    member: str,
    split: str,
    *,
    prediction: Any,
    prediction_sources_hash: str,
    source_artifact_audit_hash: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    member_root = feature_root / member
    manifest = _validate_hashed_json(
        member_root / "representation_manifest.json",
        contract=LOCAL_RESIDUAL_FIELD_FUSION_FEATURE_MANIFEST_CONTRACT,
        hash_key="manifest_hash",
    )
    if tuple(manifest.get("development_splits") or ()) != REPRESENTATION_FUSION_SPLITS:
        raise ValueError(f"{member} feature manifest is not development-only")
    if manifest.get("final_test_opened") is not False:
        raise ValueError(f"{member} feature manifest opened final-test")
    if manifest.get("prediction_sources_hash") != prediction_sources_hash:
        raise ValueError(f"{member} feature manifest prediction-source hash mismatch")
    if manifest.get("source_artifact_audit_hash") != source_artifact_audit_hash:
        raise ValueError(f"{member} feature manifest source-audit hash mismatch")
    metadata_path = member_root / f"{split}_representations_metadata.json"
    metadata = _validate_hashed_json(
        metadata_path, contract=LOCAL_RESIDUAL_FIELD_FUSION_FEATURE_CONTRACT, hash_key="metadata_hash",
    )
    manifest_split = manifest.get("splits", {}).get(split, {})
    if not isinstance(manifest_split, Mapping) or manifest_split.get("metadata_hash") != metadata.get("metadata_hash"):
        raise ValueError(f"feature manifest does not bind metadata for {member}/{split}")
    if metadata.get("member_id") != member or metadata.get("split") != split:
        raise ValueError(f"representation metadata identity mismatch for {member}/{split}")
    if metadata.get("prediction_content_hash") != prediction.metadata.get("prediction_content_hash"):
        raise ValueError(f"representation/prediction content mismatch for {member}/{split}")
    if metadata.get("jet_identity_hash") != prediction.metadata.get("jet_identity_hash"):
        raise ValueError(f"representation/prediction identity mismatch for {member}/{split}")
    representation_path = Path(str(metadata["representation_path"]))
    if not representation_path.is_file():
        representation_path = member_root / f"{split}_representations.npz"
    if sha256_file(representation_path) != metadata.get("representation_sha256"):
        raise ValueError(f"representation bytes changed for {member}/{split}")
    if manifest_split.get("representation_sha256") != metadata.get("representation_sha256"):
        raise ValueError(f"feature manifest does not bind representation bytes for {member}/{split}")
    with np.load(representation_path, allow_pickle=False) as payload:
        embedding = np.asarray(payload["jet_embedding"], dtype=np.float32)
        labels = np.asarray(payload["labels"], dtype=np.int64)
        file_indices = np.asarray(payload["jet_file_indices"], dtype=np.int64)
        entries = np.asarray(payload["jet_entries"], dtype=np.int64)
    if not np.array_equal(labels, prediction.labels):
        raise ValueError(f"representation labels do not align for {member}/{split}")
    files = list(metadata.get("jet_files") or ())
    identities = [(files[int(index)], int(entry)) for index, entry in zip(file_indices, entries)]
    expected = [(str(identity.file), int(identity.entry)) for identity in prediction.jet_ids]
    if identities != expected:
        raise ValueError(f"representation jet identities do not align for {member}/{split}")
    return embedding, {
        "manifest_hash": manifest["manifest_hash"],
        "metadata_hash": metadata["metadata_hash"],
        "representation_sha256": metadata["representation_sha256"],
        "checkpoint_hash": metadata["checkpoint_hash"],
        "prediction_sha256": metadata["prediction_sha256"],
    }


def load_representation_fusion_development_data(
    *,
    group_id: str,
    feature_root: str | Path,
    prediction_sources: str | Path,
    source_artifact_audit: str | Path,
) -> tuple[dict[str, RepresentationFusionSplitData], dict[str, Any]]:
    """Load exactly stack_train/stack_val with full cache and identity validation."""

    registry = require_development_prediction_sources(
        prediction_sources, source_artifact_audit=source_artifact_audit,
    )
    groups = {group.group_id: group for group in default_fusion_group_specs()}
    if group_id not in groups:
        raise ValueError(f"unknown fusion group {group_id!r}")
    member_a, member_b = groups[group_id].member_ids
    output: dict[str, RepresentationFusionSplitData] = {}
    hashes: dict[str, Any] = {"members": {member_a: {}, member_b: {}}}
    split_identities: dict[str, set[tuple[str, int, int]]] = {}
    for split in REPRESENTATION_FUSION_SPLITS:
        block_a = load_prediction_block(registry["members"][member_a]["prediction_root"], member_a, split, verify_hash=True)
        block_b = load_prediction_block(registry["members"][member_b]["prediction_root"], member_b, split, verify_hash=True)
        validate_prediction_alignment([block_a, block_b])
        embedding_a, hash_a = _load_member_split(
            Path(feature_root), member_a, split, prediction=block_a,
            prediction_sources_hash=registry["manifest_hash"],
            source_artifact_audit_hash=registry["source_artifact_audit_hash"],
        )
        embedding_b, hash_b = _load_member_split(
            Path(feature_root), member_b, split, prediction=block_b,
            prediction_sources_hash=registry["manifest_hash"],
            source_artifact_audit_hash=registry["source_artifact_audit_hash"],
        )
        output[split] = RepresentationFusionSplitData(
            embedding_a=embedding_a, embedding_b=embedding_b,
            logits_a=block_a.logits.astype(np.float32), logits_b=block_b.logits.astype(np.float32),
            labels=block_a.labels.astype(np.int64),
        )
        hashes["members"][member_a][split] = hash_a
        hashes["members"][member_b][split] = hash_b
        split_identities[split] = {
            (str(identity.file), int(identity.entry), int(identity.label)) for identity in block_a.jet_ids
        }
    if split_identities[FUSION_FIT_SPLIT].intersection(split_identities[FUSION_SELECTION_SPLIT]):
        raise ValueError("representation fusion stack_train and stack_val identities overlap")
    for field_name in ("embedding_a", "embedding_b"):
        if getattr(output[FUSION_FIT_SPLIT], field_name).shape[1] != getattr(
            output[FUSION_SELECTION_SPLIT], field_name
        ).shape[1]:
            raise ValueError(f"representation dimension changed across development splits for {field_name}")
    hashes.update({
        "member_ids": [member_a, member_b],
        "prediction_sources_hash": registry["manifest_hash"],
        "source_artifact_audit_hash": registry["source_artifact_audit_hash"],
    })
    return output, hashes


def _batch_tensors(data: RepresentationFusionSplitData, indices: np.ndarray, device: Any) -> tuple[Any, ...]:
    return (
        torch.as_tensor(data.embedding_a[indices], dtype=torch.float32, device=device),
        torch.as_tensor(data.embedding_b[indices], dtype=torch.float32, device=device),
        torch.as_tensor(data.logits_a[indices], dtype=torch.float32, device=device),
        torch.as_tensor(data.logits_b[indices], dtype=torch.float32, device=device),
        torch.as_tensor(data.labels[indices], dtype=torch.long, device=device),
    )


def _evaluate(
    model: FrozenRepresentationFusionHead,
    data: RepresentationFusionSplitData,
    *,
    device: Any,
    batch_size: int,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    model.eval()
    logits_rows, gate_rows, correction_rows, a_rows = [], [], [], []
    with torch.no_grad():
        for start in range(0, len(data.labels), batch_size):
            indices = np.arange(start, min(start + batch_size, len(data.labels)))
            embedding_a, embedding_b, logits_a, logits_b, labels = _batch_tensors(data, indices, device)
            output = model(embedding_a, embedding_b, logits_a, logits_b)
            logits_rows.append(output.logits.cpu())
            a_rows.append(logits_a.cpu())
            if output.gate is not None:
                gate_rows.append(output.gate.cpu())
            if output.correction is not None:
                correction_rows.append(output.correction.cpu())
    logits = torch.cat(logits_rows)
    combined = RepresentationFusionOutput(
        logits=logits,
        gate=torch.cat(gate_rows) if gate_rows else None,
        correction=torch.cat(correction_rows) if correction_rows else None,
    )
    diagnostics = representation_fusion_diagnostics(combined, logits_a=torch.cat(a_rows))
    loss = float(functional.cross_entropy(logits, torch.as_tensor(data.labels, dtype=torch.long)).item())
    return logits.numpy().astype(np.float32), loss, diagnostics


def predict_representation_fusion_head(
    model: FrozenRepresentationFusionHead,
    data: RepresentationFusionSplitData,
    *,
    device: str = "cpu",
    batch_size: int = 2048,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply a frozen fusion-only head to an aligned cached split."""

    active_device = resolve_device(device)
    model = model.to(active_device)
    logits, _loss, diagnostics = _evaluate(
        model, data, device=active_device, batch_size=int(batch_size),
    )
    return logits, diagnostics


def train_representation_fusion_head(
    candidate_id: str,
    train: RepresentationFusionSplitData,
    validation: RepresentationFusionSplitData,
    *,
    seed: int = 5101,
    hidden_width: int = 64,
    dropout: float = 0.0,
    weight_decay: float = 1.0e-5,
    learning_rate: float = 1.0e-3,
    epochs: int = 80,
    patience: int = 10,
    batch_size: int = 2048,
    device: str = "cpu",
) -> tuple[FrozenRepresentationFusionHead, dict[str, Any], dict[str, np.ndarray]]:
    """Train a fusion-only head; input representations are immutable non-gradient tensors."""

    if int(seed) not in FUSION_HEAD_SEEDS:
        raise ValueError(f"seed must be one of {FUSION_HEAD_SEEDS}")
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    active_device = resolve_device(device)
    model = FrozenRepresentationFusionHead(
        candidate_id, train.embedding_a.shape[1], train.embedding_b.shape[1],
        hidden_width=hidden_width, dropout=dropout,
    ).to(active_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    generator = np.random.default_rng(int(seed))
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    stale = 0
    curve: list[dict[str, Any]] = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        permutation = generator.permutation(len(train.labels))
        losses: list[float] = []
        for start in range(0, len(permutation), int(batch_size)):
            indices = permutation[start : start + int(batch_size)]
            embedding_a, embedding_b, logits_a, logits_b, labels = _batch_tensors(train, indices, active_device)
            if any(tensor.requires_grad for tensor in (embedding_a, embedding_b, logits_a, logits_b)):
                raise RuntimeError("frozen fusion inputs unexpectedly require gradients")
            optimizer.zero_grad(set_to_none=True)
            output = model(embedding_a, embedding_b, logits_a, logits_b)
            loss = functional.cross_entropy(output.logits, labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        _, validation_loss, _ = _evaluate(model, validation, device=active_device, batch_size=int(batch_size))
        curve.append({"epoch": epoch, "train_cross_entropy": float(np.mean(losses)), "validation_cross_entropy": validation_loss})
        if validation_loss < best_loss - 1.0e-8:
            best_loss, best_epoch = validation_loss, epoch
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            stale = 0
        else:
            stale += 1
            if stale >= int(patience):
                break
    if best_state is None:
        raise RuntimeError("representation fusion training produced no checkpoint")
    model.load_state_dict(best_state)
    model.to(active_device)
    train_logits, train_loss, train_diagnostics = _evaluate(model, train, device=active_device, batch_size=int(batch_size))
    val_logits, val_loss, val_diagnostics = _evaluate(model, validation, device=active_device, batch_size=int(batch_size))
    audit = {
        "seed": int(seed), "selected_epoch": best_epoch, "best_validation_cross_entropy": best_loss,
        "curve": curve, "train_cross_entropy": train_loss, "validation_cross_entropy": val_loss,
        "diagnostics": {FUSION_FIT_SPLIT: train_diagnostics, FUSION_SELECTION_SPLIT: val_diagnostics},
        "trainable_parameter_count": model.trainable_parameter_count,
        "backbone_parameter_count": model.backbone_parameter_count,
        "backbone_gradients_present": False,
        "backbone_optimizer_state_present": False,
        "optimizer_state_included_in_checkpoint": False,
    }
    return model.cpu(), audit, {FUSION_FIT_SPLIT: train_logits, FUSION_SELECTION_SPLIT: val_logits}


def _atomic_torch_save(path: Path, payload: Mapping[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        publish_temporary_file(temporary, path, overwrite=overwrite)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, payload: Mapping[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        publish_temporary_file(temporary, path, overwrite=overwrite)
    finally:
        if temporary.exists():
            temporary.unlink()


def train_representation_fusion_campaign_candidate(config: RepresentationFusionTrainConfig) -> dict[str, Any]:
    """Train one group/candidate/seed and write a fusion-only checkpoint plus stack report."""

    specs = {
        spec.candidate_id: spec for spec in default_fusion_candidate_specs()
        if spec.family == FUSION_FAMILY_REPRESENTATION
    }
    if config.candidate_id not in specs:
        raise ValueError(f"candidate is not a locked representation candidate: {config.candidate_id!r}")
    data, source_hashes = load_representation_fusion_development_data(
        group_id=config.group_id, feature_root=config.feature_root,
        prediction_sources=config.prediction_sources, source_artifact_audit=config.source_artifact_audit,
    )
    model, training, logits = train_representation_fusion_head(
        config.candidate_id, data[FUSION_FIT_SPLIT], data[FUSION_SELECTION_SPLIT],
        seed=config.seed, hidden_width=config.hidden_width, dropout=config.dropout,
        weight_decay=config.weight_decay, learning_rate=config.learning_rate,
        epochs=config.epochs, patience=config.patience, batch_size=config.batch_size, device=config.device,
    )
    spec = specs[config.candidate_id]
    checkpoint = {
        "contract": LOCAL_RESIDUAL_FIELD_REPRESENTATION_FUSION_CHECKPOINT_CONTRACT,
        "model_contract": LOCAL_RESIDUAL_FIELD_REPRESENTATION_FUSION_MODEL_CONTRACT,
        "campaign_id": config.campaign_id,
        "group_id": config.group_id,
        "member_ids": source_hashes["member_ids"],
        "candidate_id": config.candidate_id,
        "candidate_spec": spec.to_dict(),
        "candidate_spec_hash": stable_fusion_json_hash(spec.to_dict()),
        "model_config": {
            "embedding_dim_a": int(data[FUSION_FIT_SPLIT].embedding_a.shape[1]),
            "embedding_dim_b": int(data[FUSION_FIT_SPLIT].embedding_b.shape[1]),
            "num_classes": len(LABEL_NAMES), "hidden_width": int(config.hidden_width), "dropout": float(config.dropout),
        },
        "state_dict": model.state_dict(),
        "normalization": {"embeddings": "per_jet_l2_after_optional_learned_projection", "logits": "unchanged"},
        "feature_formula": "[h_A,h_B,abs(h_A-h_B),h_A*h_B,z_A,z_B,z_B-z_A]",
        "source_hashes": source_hashes,
        "training": training,
        "final_test_opened": False,
        "backbone_state_included": False,
        "optimizer_state_included": False,
        "runtime_inputs": "HLT_only",
        "uses_true_fields": False,
        "uses_offline_particles": False,
        "uses_teacher_logits_at_runtime": False,
        "deployable": True,
    }
    checkpoint_path = Path(config.checkpoint_path)
    _atomic_torch_save(checkpoint_path, checkpoint, overwrite=bool(config.overwrite))
    checkpoint_hash = sha256_file(checkpoint_path)
    report: dict[str, Any] = {
        "ok": True,
        "contract": LOCAL_RESIDUAL_FIELD_REPRESENTATION_FUSION_TRAIN_CONTRACT,
        "campaign_id": config.campaign_id,
        "group_id": config.group_id,
        "member_ids": source_hashes["member_ids"],
        "candidate_id": config.candidate_id,
        "candidate_spec": spec.to_dict(),
        "candidate_spec_hash": stable_fusion_json_hash(spec.to_dict()),
        "hyperparameters": {
            "seed": int(config.seed), "hidden_width": int(config.hidden_width), "dropout": float(config.dropout),
            "weight_decay": float(config.weight_decay), "learning_rate": float(config.learning_rate),
        },
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_hash": checkpoint_hash,
        "source_hashes": source_hashes,
        "normalization": checkpoint["normalization"],
        "training": training,
        "fit_split": FUSION_FIT_SPLIT,
        "selection_split": FUSION_SELECTION_SPLIT,
        "development_splits": list(REPRESENTATION_FUSION_SPLITS),
        "final_test_opened": False,
        "metrics": {
            split: {
                "multiclass": local_residual_field_multiclass_metrics(logits[split], data[split].labels, label_names=LABEL_NAMES),
                "binary_projection": local_residual_field_binary_projection_metrics(logits[split], data[split].labels, label_names=LABEL_NAMES),
            }
            for split in REPRESENTATION_FUSION_SPLITS
        },
        "runtime_inputs": "HLT_only",
        "uses_true_fields": False,
        "uses_offline_particles": False,
        "uses_teacher_logits_at_runtime": False,
        "deployable": True,
        "config": asdict(config),
    }
    report["artifact_hash"] = stable_fusion_json_hash(report)
    _atomic_json(Path(config.report_path), report, overwrite=bool(config.overwrite))
    return report


def load_representation_fusion_head_from_checkpoint(
    path: str | Path,
    *,
    device: str = "cpu",
) -> tuple[FrozenRepresentationFusionHead, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("contract") != LOCAL_RESIDUAL_FIELD_REPRESENTATION_FUSION_CHECKPOINT_CONTRACT:
        raise ValueError("representation fusion checkpoint contract mismatch")
    if payload.get("backbone_state_included") is not False or payload.get("optimizer_state_included") is not False:
        raise ValueError("fusion checkpoint unexpectedly contains backbone or optimizer state")
    model_config = payload["model_config"]
    model = FrozenRepresentationFusionHead(
        payload["candidate_id"], model_config["embedding_dim_a"], model_config["embedding_dim_b"],
        num_classes=model_config["num_classes"], hidden_width=model_config["hidden_width"],
        dropout=model_config["dropout"],
    )
    model.load_state_dict(payload["state_dict"])
    model.to(resolve_device(device)).eval()
    return model, payload


__all__ = [
    "LOCAL_RESIDUAL_FIELD_REPRESENTATION_FUSION_TRAIN_CONTRACT",
    "LOCAL_RESIDUAL_FIELD_REPRESENTATION_FUSION_CHECKPOINT_CONTRACT",
    "REPRESENTATION_FUSION_SPLITS",
    "RepresentationFusionSplitData",
    "RepresentationFusionTrainConfig",
    "load_representation_fusion_development_data",
    "train_representation_fusion_head",
    "predict_representation_fusion_head",
    "train_representation_fusion_campaign_candidate",
    "load_representation_fusion_head_from_checkpoint",
]
