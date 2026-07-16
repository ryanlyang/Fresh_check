"""Immutable stack-only logit fusion for ABPH G-tier variants."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .config import canonical_hash


ABPH_FROZEN_FUSION_CONTRACT = "adaptive_binary_pseudooffline_frozen_fusion_v1"
ABPH_FUSION_FIT_SPLIT = "stack_train"
ABPH_FUSION_SELECTION_SPLIT = "stack_val"
ABPH_FUSION_APPLICATION_SPLITS: tuple[str, ...] = (
    "stack_train",
    "stack_val",
    "final_test",
)


def _array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    hasher = hashlib.sha256()
    hasher.update(str(array.dtype).encode("utf-8"))
    hasher.update(str(tuple(array.shape)).encode("utf-8"))
    if array.dtype.kind == "O":
        hasher.update(
            json.dumps(array.tolist(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
    else:
        hasher.update(array.tobytes())
    return hasher.hexdigest()


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(exp.sum(axis=-1, keepdims=True), 1.0e-12)


def _cross_entropy(logits: np.ndarray, labels: np.ndarray) -> float:
    probabilities = _softmax(logits)
    rows = np.arange(labels.shape[0])
    return float(-np.log(np.maximum(probabilities[rows, labels], 1.0e-12)).mean())


def _project_simplex(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("simplex projection expects one dimension")
    ordered = np.sort(values)[::-1]
    cumulative = np.cumsum(ordered) - 1.0
    support = ordered - cumulative / np.arange(1, values.size + 1) > 0.0
    if not support.any():
        return np.full_like(values, 1.0 / values.size)
    rho = int(np.nonzero(support)[0][-1])
    theta = cumulative[rho] / float(rho + 1)
    return np.maximum(values - theta, 0.0)


@dataclass(frozen=True)
class LogitPredictionBlock:
    """One model/split prediction block with identity-bound provenance."""

    member: str
    split: str
    logits: np.ndarray
    labels: np.ndarray
    jet_ids: np.ndarray
    checkpoint_hash: str
    resolved_config_hash: str
    provenance: Mapping[str, Any]

    def validate(self) -> None:
        logits = np.asarray(self.logits)
        labels = np.asarray(self.labels)
        jet_ids = np.asarray(self.jet_ids)
        if not self.member or self.split not in ABPH_FUSION_APPLICATION_SPLITS:
            raise ValueError("fusion block has an invalid member or split")
        if logits.ndim != 2 or logits.shape[1] < 2:
            raise ValueError("fusion logits must have shape [jets,classes]")
        if labels.shape != (logits.shape[0],) or jet_ids.shape[0] != logits.shape[0]:
            raise ValueError("fusion labels/identities differ from logits")
        if labels.dtype.kind not in "iu" or np.any(labels < 0) or np.any(labels >= logits.shape[1]):
            raise ValueError("fusion labels are invalid")
        if not np.isfinite(logits).all():
            raise ValueError("fusion logits contain nonfinite values")
        if not self.checkpoint_hash or not self.resolved_config_hash:
            raise ValueError("fusion block lacks checkpoint/config identity")
        required = (
            "source_manifest_hash",
            "jet_identity_hash",
            "label_hash",
            "class_mapping_hash",
            "hlt_content_hash",
        )
        missing = [name for name in required if not self.provenance.get(name)]
        if missing:
            raise ValueError(f"fusion block lacks provenance {missing}")
        if self.provenance.get("teacher_logits_loaded") is not False:
            raise ValueError("fusion predictions must be teacher-logit free")

    @property
    def prediction_hash(self) -> str:
        self.validate()
        payload = {
            "member": self.member,
            "split": self.split,
            "logits_hash": _array_hash(np.asarray(self.logits, dtype=np.float32)),
            "labels_hash": _array_hash(np.asarray(self.labels, dtype=np.int64)),
            "jet_ids_hash": _array_hash(np.asarray(self.jet_ids)),
            "checkpoint_hash": self.checkpoint_hash,
            "resolved_config_hash": self.resolved_config_hash,
            "provenance": dict(self.provenance),
        }
        return canonical_hash(payload)


def _validate_aligned_blocks(
    blocks: Sequence[LogitPredictionBlock],
    *,
    split: str,
    members: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    rows = tuple(blocks)
    expected = tuple(str(name) for name in members)
    if tuple(row.member for row in rows) != expected:
        raise ValueError(
            f"fusion membership/order differs from frozen declaration: "
            f"{tuple(row.member for row in rows)} != {expected}"
        )
    if not rows:
        raise ValueError("fusion requires at least one member")
    for row in rows:
        row.validate()
        if row.split != split:
            raise ValueError(f"fusion expected {split}, received {row.split}")
    labels = np.asarray(rows[0].labels, dtype=np.int64)
    jet_ids = np.asarray(rows[0].jet_ids)
    classes = int(np.asarray(rows[0].logits).shape[1])
    shared_fields = (
        "source_manifest_hash",
        "jet_identity_hash",
        "label_hash",
        "class_mapping_hash",
        "hlt_content_hash",
    )
    for row in rows[1:]:
        if not np.array_equal(labels, np.asarray(row.labels, dtype=np.int64)):
            raise ValueError("fusion member labels are not aligned")
        if not np.array_equal(jet_ids, np.asarray(row.jet_ids)):
            raise ValueError("fusion member jet identities are not aligned")
        if int(np.asarray(row.logits).shape[1]) != classes:
            raise ValueError("fusion member class dimensions differ")
        for field in shared_fields:
            if row.provenance.get(field) != rows[0].provenance.get(field):
                raise ValueError(f"fusion member provenance conflicts on {field}")
    return np.stack([np.asarray(row.logits, dtype=np.float64) for row in rows], axis=1), labels


def _fit_weights(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    classwise: bool,
    updates: int,
    learning_rate: float,
) -> np.ndarray:
    jets, members, classes = logits.shape
    weights = (
        np.full((members, classes), 1.0 / members, dtype=np.float64)
        if classwise
        else np.full((members,), 1.0 / members, dtype=np.float64)
    )
    rows = np.arange(jets)
    for update in range(int(updates)):
        if classwise:
            fused = np.einsum("bmc,mc->bc", logits, weights)
        else:
            fused = np.einsum("bmc,m->bc", logits, weights)
        residual = _softmax(fused)
        residual[rows, labels] -= 1.0
        if classwise:
            gradient = np.einsum("bc,bmc->mc", residual, logits) / jets
            step = float(learning_rate) / np.sqrt(update + 1.0)
            for class_index in range(classes):
                weights[:, class_index] = _project_simplex(
                    weights[:, class_index] - step * gradient[:, class_index]
                )
        else:
            gradient = np.einsum("bc,bmc->m", residual, logits) / jets
            step = float(learning_rate) / np.sqrt(update + 1.0)
            weights = _project_simplex(weights - step * gradient)
    return weights


def _fuse(logits: np.ndarray, weights: np.ndarray, kind: str) -> np.ndarray:
    if kind == "scalar_simplex":
        return np.einsum("bmc,m->bc", logits, weights)
    if kind == "classwise_simplex":
        return np.einsum("bmc,mc->bc", logits, weights)
    raise ValueError(f"unknown fusion kind {kind!r}")


@dataclass(frozen=True)
class FrozenFusionArtifact:
    fusion_variant: str
    members: tuple[str, ...]
    member_checkpoint_hashes: Mapping[str, str]
    member_config_hashes: Mapping[str, str]
    kind: str
    weights: tuple[Any, ...]
    stack_train_prediction_hashes: Mapping[str, str]
    stack_val_prediction_hashes: Mapping[str, str]
    stack_train_cross_entropy: float
    stack_val_cross_entropy: float
    selection_candidates: Mapping[str, float]
    membership_hash: str

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "contract": ABPH_FROZEN_FUSION_CONTRACT,
            "fusion_variant": self.fusion_variant,
            "members": list(self.members),
            "member_checkpoint_hashes": dict(self.member_checkpoint_hashes),
            "member_config_hashes": dict(self.member_config_hashes),
            "kind": self.kind,
            "weights": list(self.weights),
            "fit_split": ABPH_FUSION_FIT_SPLIT,
            "selection_split": ABPH_FUSION_SELECTION_SPLIT,
            "stack_train_prediction_hashes": dict(self.stack_train_prediction_hashes),
            "stack_val_prediction_hashes": dict(self.stack_val_prediction_hashes),
            "stack_train_cross_entropy": float(self.stack_train_cross_entropy),
            "stack_val_cross_entropy": float(self.stack_val_cross_entropy),
            "selection_candidates": dict(self.selection_candidates),
            "membership_hash": self.membership_hash,
            "final_test_loaded_during_fit": False,
            "teacher_logits_loaded": False,
        }
        payload["artifact_hash"] = canonical_hash(payload)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FrozenFusionArtifact":
        data = dict(payload)
        if data.get("contract") != ABPH_FROZEN_FUSION_CONTRACT:
            raise ValueError("frozen fusion contract mismatch")
        expected_hash = data.pop("artifact_hash", None)
        if expected_hash != canonical_hash(data):
            raise ValueError("frozen fusion artifact hash mismatch")
        if data.get("fit_split") != ABPH_FUSION_FIT_SPLIT:
            raise ValueError("fusion was not fit on stack_train")
        if data.get("selection_split") != ABPH_FUSION_SELECTION_SPLIT:
            raise ValueError("fusion was not selected on stack_val")
        if data.get("final_test_loaded_during_fit") is not False:
            raise ValueError("fusion fit accessed final_test")
        return cls(
            fusion_variant=str(data["fusion_variant"]),
            members=tuple(str(value) for value in data["members"]),
            member_checkpoint_hashes=dict(data["member_checkpoint_hashes"]),
            member_config_hashes=dict(data["member_config_hashes"]),
            kind=str(data["kind"]),
            weights=tuple(data["weights"]),
            stack_train_prediction_hashes=dict(data["stack_train_prediction_hashes"]),
            stack_val_prediction_hashes=dict(data["stack_val_prediction_hashes"]),
            stack_train_cross_entropy=float(data["stack_train_cross_entropy"]),
            stack_val_cross_entropy=float(data["stack_val_cross_entropy"]),
            selection_candidates=dict(data["selection_candidates"]),
            membership_hash=str(data["membership_hash"]),
        )


def fit_frozen_stack_fusion(
    fusion_variant: str,
    declared_members: Sequence[str],
    stack_train_blocks: Sequence[LogitPredictionBlock],
    stack_val_blocks: Sequence[LogitPredictionBlock],
    *,
    candidate_kinds: Sequence[str] = ("scalar_simplex", "classwise_simplex"),
    updates: int = 500,
    learning_rate: float = 0.25,
) -> FrozenFusionArtifact:
    """Fit on stack_train, select form on stack_val, then freeze everything."""

    members = tuple(str(name) for name in declared_members)
    if len(members) < 2 or len(set(members)) != len(members):
        raise ValueError("fusion declaration must contain at least two unique members")
    train_logits, train_labels = _validate_aligned_blocks(
        stack_train_blocks, split=ABPH_FUSION_FIT_SPLIT, members=members
    )
    val_logits, val_labels = _validate_aligned_blocks(
        stack_val_blocks, split=ABPH_FUSION_SELECTION_SPLIT, members=members
    )
    kinds = tuple(str(kind) for kind in candidate_kinds)
    if not kinds or not set(kinds).issubset({"scalar_simplex", "classwise_simplex"}):
        raise ValueError("unsupported fusion candidate kind")
    candidates: dict[str, tuple[np.ndarray, float, float]] = {}
    for kind in kinds:
        weights = _fit_weights(
            train_logits,
            train_labels,
            classwise=kind == "classwise_simplex",
            updates=int(updates),
            learning_rate=float(learning_rate),
        )
        train_ce = _cross_entropy(_fuse(train_logits, weights, kind), train_labels)
        val_ce = _cross_entropy(_fuse(val_logits, weights, kind), val_labels)
        candidates[kind] = (weights, train_ce, val_ce)
    selected = min(candidates, key=lambda name: (candidates[name][2], name))
    weights, train_ce, val_ce = candidates[selected]
    first_train = stack_train_blocks[0]
    checkpoint_hashes = {row.member: row.checkpoint_hash for row in stack_train_blocks}
    config_hashes = {row.member: row.resolved_config_hash for row in stack_train_blocks}
    for row in stack_val_blocks:
        if row.checkpoint_hash != checkpoint_hashes[row.member]:
            raise ValueError(f"{row.member} checkpoint changed between stack_train and stack_val")
        if row.resolved_config_hash != config_hashes[row.member]:
            raise ValueError(f"{row.member} config changed between stack_train and stack_val")
    membership_payload = {
        "fusion_variant": str(fusion_variant),
        "members": list(members),
        "member_checkpoint_hashes": checkpoint_hashes,
        "member_config_hashes": config_hashes,
        "source_manifest_hash": first_train.provenance["source_manifest_hash"],
        "class_mapping_hash": first_train.provenance["class_mapping_hash"],
    }
    return FrozenFusionArtifact(
        fusion_variant=str(fusion_variant),
        members=members,
        member_checkpoint_hashes=checkpoint_hashes,
        member_config_hashes=config_hashes,
        kind=selected,
        weights=tuple(np.asarray(weights).tolist()),
        stack_train_prediction_hashes={row.member: row.prediction_hash for row in stack_train_blocks},
        stack_val_prediction_hashes={row.member: row.prediction_hash for row in stack_val_blocks},
        stack_train_cross_entropy=train_ce,
        stack_val_cross_entropy=val_ce,
        selection_candidates={name: float(value[2]) for name, value in candidates.items()},
        membership_hash=canonical_hash(membership_payload),
    )


def apply_frozen_fusion(
    artifact: FrozenFusionArtifact,
    blocks: Sequence[LogitPredictionBlock],
) -> np.ndarray:
    """Apply a frozen artifact; callers cannot substitute or reorder members."""

    rows = tuple(blocks)
    if not rows:
        raise ValueError("no fusion blocks were supplied")
    split = rows[0].split
    logits, _ = _validate_aligned_blocks(rows, split=split, members=artifact.members)
    for row in rows:
        if row.checkpoint_hash != artifact.member_checkpoint_hashes.get(row.member):
            raise ValueError(f"{row.member} does not match the frozen checkpoint")
        if row.resolved_config_hash != artifact.member_config_hashes.get(row.member):
            raise ValueError(f"{row.member} does not match the frozen configuration")
    return _fuse(logits, np.asarray(artifact.weights, dtype=np.float64), artifact.kind)


def write_frozen_fusion_artifact(path: str | Path, artifact: FrozenFusionArtifact) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(artifact.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(destination)


def load_frozen_fusion_artifact(path: str | Path) -> FrozenFusionArtifact:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("frozen fusion artifact must be a JSON mapping")
    return FrozenFusionArtifact.from_dict(payload)


__all__ = [
    "ABPH_FROZEN_FUSION_CONTRACT",
    "ABPH_FUSION_APPLICATION_SPLITS",
    "ABPH_FUSION_FIT_SPLIT",
    "ABPH_FUSION_SELECTION_SPLIT",
    "FrozenFusionArtifact",
    "LogitPredictionBlock",
    "apply_frozen_fusion",
    "fit_frozen_stack_fusion",
    "load_frozen_fusion_artifact",
    "write_frozen_fusion_artifact",
]
