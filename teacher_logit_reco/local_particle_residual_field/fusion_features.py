"""Strict frozen ParT representation caches for the P7b fusion campaign."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping

import numpy as np

from jetclass_fresh.fusion import load_prediction_block
from jetclass_fresh.hlt_baseline import amp_autocast_context, require_torch, resolve_device
from jetclass_fresh.jetclass_data import LABEL_NAMES

from .curriculum import LocalResidualFieldCurriculumJointModel
from .data import move_local_particle_residual_field_batch_to_device
from .fusion import (
    LOCAL_RESIDUAL_FIELD_PREDICTION_CONTRACT,
    LocalResidualFieldPredictionConfig,
    _make_prediction_loader,
    _prediction_dataset,
    load_local_residual_field_tagger_from_checkpoint,
)
from .fusion_atomic import publish_temporary_file
from .fusion_campaign import (
    FUSION_MEMBER_A0,
    FUSION_MEMBER_A0_SEED1,
    FUSION_MEMBER_P7B,
    stable_fusion_json_hash,
)
from .fusion_seed_control import sha256_file
from .fusion_sources import (
    FUSION_DEVELOPMENT_SPLITS,
    LOCAL_RESIDUAL_FIELD_FUSION_PREDICTION_SOURCES_CONTRACT,
    require_fusion_source_artifact_audit,
)


LOCAL_RESIDUAL_FIELD_FUSION_FEATURE_CONTRACT = "local_residual_field_fusion_features_v1"
LOCAL_RESIDUAL_FIELD_FUSION_FEATURE_MANIFEST_CONTRACT = "local_residual_field_fusion_feature_manifest_v1"
LOCAL_RESIDUAL_FIELD_FUSION_REPRESENTATION_NAME = "pre_classifier_cls_embedding"
LOCAL_RESIDUAL_FIELD_FUSION_REPRESENTATION_SOURCE = "final_linear_forward_pre_hook"
FUSION_FEATURE_MEMBERS = (FUSION_MEMBER_A0, FUSION_MEMBER_A0_SEED1, FUSION_MEMBER_P7B)
FUSION_FEATURE_FP32_LOGITS_ATOL = 2.0e-3
FUSION_FEATURE_AMP_LOGITS_ATOL = 1.0e-2
FUSION_FEATURE_LOGITS_RTOL = 2.0e-3


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return dict(payload)


def _atomic_json(path: str | Path, payload: Mapping[str, Any], *, overwrite: bool) -> None:
    output = Path(path)
    if output.exists() and not bool(overwrite):
        raise FileExistsError(f"refusing to overwrite immutable artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=str(output.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        publish_temporary_file(temporary, output, overwrite=overwrite)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_npz(path: str | Path, *, overwrite: bool, **arrays: Any) -> None:
    output = Path(path)
    if output.exists() and not bool(overwrite):
        raise FileExistsError(f"refusing to overwrite immutable artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=str(output.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        publish_temporary_file(temporary, output, overwrite=overwrite)
    finally:
        if temporary.exists():
            temporary.unlink()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(tuple(array.shape)).encode("utf-8"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _model_logits(output: Any) -> Any:
    if hasattr(output, "student_logits"):
        return output.student_logits
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, tuple) and output:
        return output[0]
    return output


def _campaign_forward(model: Any, batch: Mapping[str, Any]) -> Any:
    if isinstance(model, LocalResidualFieldCurriculumJointModel):
        if model.oracle_consumer is not None:
            raise ValueError("deployable P7b representation model unexpectedly contains an oracle consumer")
        return model(
            batch["points"],
            batch["features"],
            batch["lorentz_vectors"],
            batch["mask"],
            tokens=batch["tokens"],
            raw_mask=batch["raw_mask"],
            indices=batch["indices"],
            target_fields=None,
            return_outputs=True,
        )
    return model(
        batch["points"],
        batch["features"],
        batch["lorentz_vectors"],
        batch["mask"],
        tokens=batch["tokens"],
        raw_mask=batch["raw_mask"],
        indices=batch["indices"],
        target_fields=None,
        oracle_fields=None,
        return_outputs=True,
    )


def _part_classifier(model: Any) -> Any:
    student = model.student if isinstance(model, LocalResidualFieldCurriculumJointModel) else model
    classifier = getattr(student, "part_model", None)
    if classifier is None:
        raise ValueError("member does not expose the deployable student Part classifier")
    return classifier


class PreClassifierEmbeddingCapture:
    """Capture the unique final class head input, failing closed on layout drift."""

    def __init__(self, model: Any, *, num_classes: int = len(LABEL_NAMES)) -> None:
        torch = require_torch()
        self.model = model
        self.num_classes = int(num_classes)
        self.part_classifier = _part_classifier(model)
        candidates = [
            (str(name), module)
            for name, module in self.part_classifier.named_modules()
            if name and isinstance(module, torch.nn.Linear) and int(module.out_features) == self.num_classes
        ]
        if not candidates:
            raise ValueError(
                f"could not locate a {self.num_classes}-class Linear head inside the deployable Part classifier"
            )
        if len(candidates) != 1:
            names = ", ".join(name for name, _ in candidates)
            raise ValueError(f"ambiguous {self.num_classes}-class Part classifier heads: {names}")
        self.final_head_name, self.final_head = candidates[0]
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()

    @property
    def trainable_parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad))

    def capture(self, forward: Callable[[], Any]) -> tuple[Any, Any]:
        torch = require_torch()
        captured_inputs: list[Any] = []
        captured_outputs: list[Any] = []

        def pre_hook(module: Any, inputs: tuple[Any, ...]) -> None:
            del module
            if inputs:
                captured_inputs.append(inputs[0])

        def post_hook(module: Any, inputs: tuple[Any, ...], output: Any) -> None:
            del module, inputs
            captured_outputs.append(output)

        pre_handle = self.final_head.register_forward_pre_hook(pre_hook)
        post_handle = self.final_head.register_forward_hook(post_hook)
        try:
            with torch.no_grad():
                logits = _model_logits(forward())
        finally:
            pre_handle.remove()
            post_handle.remove()
        if len(captured_inputs) != 1 or len(captured_outputs) != 1:
            raise RuntimeError(
                "final Part classifier head hook must fire exactly once per forward; "
                f"captured inputs={len(captured_inputs)}, outputs={len(captured_outputs)}"
            )
        embedding = captured_inputs[0]
        head_logits = captured_outputs[0]
        if not isinstance(logits, torch.Tensor) or not isinstance(embedding, torch.Tensor):
            raise TypeError("captured logits and embedding must be torch tensors")
        if int(logits.ndim) != 2 or int(logits.shape[1]) != self.num_classes:
            raise ValueError(f"member logits must have shape [batch,{self.num_classes}], got {tuple(logits.shape)}")
        if int(embedding.ndim) != 2 or int(embedding.shape[0]) != int(logits.shape[0]):
            raise ValueError(
                "pre-classifier representation must have shape [batch,embedding_dim], "
                f"got {tuple(embedding.shape)}"
            )
        if tuple(head_logits.shape) != tuple(logits.shape) or not torch.allclose(
            head_logits, logits, atol=1.0e-5, rtol=1.0e-4
        ):
            raise ValueError("captured final-head output does not reproduce member logits")
        if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(embedding).all()):
            raise FloatingPointError("member logits or pre-classifier embedding contains non-finite values")
        return logits.detach(), embedding.detach()

    def preflight(self, forward: Callable[[], Any]) -> dict[str, Any]:
        torch = require_torch()
        with torch.no_grad():
            normal_logits = _model_logits(forward()).detach()
        captured_logits, embedding = self.capture(forward)
        agrees = bool(torch.allclose(normal_logits, captured_logits, atol=1.0e-5, rtol=1.0e-4))
        if not agrees:
            raise ValueError("normal and representation-capture forwards do not reproduce identical logits")
        return {
            "ok": True,
            "representation_name": LOCAL_RESIDUAL_FIELD_FUSION_REPRESENTATION_NAME,
            "representation_source": LOCAL_RESIDUAL_FIELD_FUSION_REPRESENTATION_SOURCE,
            "final_head_name": self.final_head_name,
            "embedding_dim": int(embedding.shape[1]),
            "logits_shape": list(captured_logits.shape),
            "embedding_shape": list(embedding.shape),
            "normal_capture_logits_agree": True,
            "trainable_parameter_count": self.trainable_parameter_count,
            "model_training": bool(self.model.training),
        }


@dataclass
class FusionFeatureCacheConfig:
    checkpoint: str
    member_id: str
    output_dir: str
    prediction_sources: str
    source_artifact_audit: str
    hlt_cache_dir: str
    manifest_path: str
    splits: tuple[str, ...] = FUSION_DEVELOPMENT_SPLITS
    batch_size: int = 128
    num_workers: int = 0
    device: str = "auto"
    amp: bool = True
    storage_dtype: str = "float16"
    logits_atol: float | None = None
    logits_rtol: float = FUSION_FEATURE_LOGITS_RTOL
    overwrite: bool = False

    def __post_init__(self) -> None:
        for name in (
            "checkpoint", "member_id", "output_dir", "prediction_sources",
            "source_artifact_audit", "hlt_cache_dir", "manifest_path",
        ):
            value = str(getattr(self, name) or "").strip()
            if not value:
                raise ValueError(f"{name} must be non-empty")
            setattr(self, name, value)
        if self.member_id not in FUSION_FEATURE_MEMBERS:
            raise ValueError(f"member_id must be one of {FUSION_FEATURE_MEMBERS}")
        self.splits = tuple(str(split) for split in self.splits)
        if self.splits != FUSION_DEVELOPMENT_SPLITS:
            raise ValueError(f"feature-cache splits are locked to {FUSION_DEVELOPMENT_SPLITS}")
        if self.storage_dtype not in {"float16", "float32"}:
            raise ValueError("storage_dtype must be float16 or float32")
        self.batch_size = int(self.batch_size)
        self.num_workers = int(self.num_workers)
        if self.batch_size <= 0 or self.num_workers < 0:
            raise ValueError("batch_size must be positive and num_workers non-negative")
        self.amp = bool(self.amp)
        self.logits_atol = float(
            FUSION_FEATURE_AMP_LOGITS_ATOL
            if self.logits_atol is None and self.amp
            else FUSION_FEATURE_FP32_LOGITS_ATOL
            if self.logits_atol is None
            else self.logits_atol
        )
        self.logits_rtol = float(self.logits_rtol)
        if self.logits_atol < 0.0 or self.logits_rtol < 0.0:
            raise ValueError("logit reproduction tolerances must be non-negative")
        self.overwrite = bool(self.overwrite)


def require_development_prediction_sources(
    path: str | Path,
    *,
    source_artifact_audit: str | Path,
) -> dict[str, Any]:
    registry = _read_json(path)
    if registry.get("contract") != LOCAL_RESIDUAL_FIELD_FUSION_PREDICTION_SOURCES_CONTRACT:
        raise ValueError("development prediction-source contract mismatch")
    if registry.get("ok") is not True or registry.get("final_test_opened") is not False:
        raise ValueError("development prediction-source registry is not an unopened successful artifact")
    unsigned = dict(registry)
    stored_hash = unsigned.pop("manifest_hash", None)
    if stored_hash != stable_fusion_json_hash(unsigned):
        raise ValueError("development prediction-source manifest hash mismatch")
    audit = require_fusion_source_artifact_audit(source_artifact_audit)
    if registry.get("source_artifact_audit_hash") != audit.get("audit_hash"):
        raise ValueError("prediction sources do not bind the active source-artifact audit")
    if tuple(registry.get("development_splits") or ()) != FUSION_DEVELOPMENT_SPLITS:
        raise ValueError("prediction-source development splits changed")
    members = registry.get("members")
    if not isinstance(members, Mapping) or set(members) != set(FUSION_FEATURE_MEMBERS):
        raise ValueError(f"prediction sources must contain exactly {FUSION_FEATURE_MEMBERS}")
    for member, row in members.items():
        if not isinstance(row, Mapping):
            raise ValueError(f"prediction-source row for {member} is malformed")
        split_rows = row.get("splits")
        if not isinstance(split_rows, Mapping) or set(split_rows) != set(FUSION_DEVELOPMENT_SPLITS):
            raise ValueError(f"prediction-source splits for {member} are incomplete")
        for split, split_row in split_rows.items():
            if not isinstance(split_row, Mapping):
                raise ValueError(f"prediction-source row for {member}/{split} is malformed")
            for key, hash_key in (
                ("prediction_path", "prediction_sha256"),
                ("metadata_path", "metadata_sha256"),
            ):
                artifact = Path(str(split_row.get(key) or ""))
                if not artifact.is_file() or sha256_file(artifact) != split_row.get(hash_key):
                    raise ValueError(f"prediction source changed or disappeared: {member}/{split}/{key}")
    return registry


def _identity_arrays(jet_ids: list[Any] | tuple[Any, ...]) -> tuple[list[str], np.ndarray, np.ndarray]:
    files: list[str] = []
    file_to_index: dict[str, int] = {}
    file_indices: list[int] = []
    entries: list[int] = []
    for identity in jet_ids:
        file_name = str(identity.file)
        if file_name not in file_to_index:
            file_to_index[file_name] = len(files)
            files.append(file_name)
        file_indices.append(file_to_index[file_name])
        entries.append(int(identity.entry))
    return files, np.asarray(file_indices, dtype=np.int32), np.asarray(entries, dtype=np.int64)


def cache_local_residual_field_fusion_features(config: FusionFeatureCacheConfig) -> dict[str, Any]:
    torch = require_torch()
    audit = require_fusion_source_artifact_audit(config.source_artifact_audit)
    registry = require_development_prediction_sources(
        config.prediction_sources,
        source_artifact_audit=config.source_artifact_audit,
    )
    checkpoint_hash = sha256_file(config.checkpoint)
    if config.member_id in audit.get("checkpoint_hashes", {}):
        expected = audit["checkpoint_hashes"].get(config.member_id)
        if checkpoint_hash != expected:
            raise ValueError(f"{config.member_id} checkpoint does not match source-artifact audit")
    source_row = registry["members"][config.member_id]
    prediction_root = source_row["prediction_root"]
    device = resolve_device(config.device)
    amp_enabled = bool(config.amp and getattr(device, "type", str(device)) == "cuda")
    model, _payload = load_local_residual_field_tagger_from_checkpoint(config.checkpoint, device=device)
    model.to(device)
    if isinstance(model, LocalResidualFieldCurriculumJointModel) and model.oracle_consumer is not None:
        raise ValueError("P7b feature extraction cannot load an oracle consumer")
    capture = PreClassifierEmbeddingCapture(model, num_classes=len(LABEL_NAMES))
    if capture.trainable_parameter_count != 0 or bool(model.training):
        raise ValueError("feature extraction requires a frozen eval-mode member")

    member_dir = Path(config.output_dir) / config.member_id
    split_reports: dict[str, Any] = {}
    all_identities: dict[str, set[tuple[str, int, int]]] = {}
    for split in config.splits:
        prediction = load_prediction_block(prediction_root, config.member_id, split, verify_hash=True)
        if prediction.metadata.get("contract") != LOCAL_RESIDUAL_FIELD_PREDICTION_CONTRACT:
            raise ValueError(f"{config.member_id}/{split} prediction contract mismatch")
        prediction_checkpoint_hash = prediction.metadata.get("checkpoint_hash") or prediction.metadata.get(
            "student_checkpoint_hash"
        )
        if prediction_checkpoint_hash != checkpoint_hash:
            raise ValueError(f"{config.member_id}/{split} prediction checkpoint hash mismatch")
        if prediction.metadata.get("runtime_inputs") != "HLT_only" or any(
            bool(prediction.metadata.get(key))
            for key in ("uses_true_fields", "uses_offline_particles", "uses_teacher_logits_at_runtime")
        ) or prediction.metadata.get("deployable") is not True:
            raise ValueError(f"{config.member_id}/{split} is not a deployable HLT-only prediction")
        prediction_config = LocalResidualFieldPredictionConfig(
            checkpoint=config.checkpoint,
            prediction_dir=str(prediction_root),
            model_name=config.member_id,
            hlt_cache_dir=config.hlt_cache_dir,
            target_cache_dir=None,
            manifest_path=config.manifest_path,
            splits=(split,),
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            device=config.device,
            amp=config.amp,
            confirm_final_test=False,
        )
        dataset = _prediction_dataset(prediction_config, split, model=model)
        loader = _make_prediction_loader(
            dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            seed=0,
            hlt_only=True,
        )
        first_batch = next(iter(loader))
        first_batch = move_local_particle_residual_field_batch_to_device(first_batch, device)
        with amp_autocast_context(amp_enabled):
            preflight = capture.preflight(lambda: _campaign_forward(model, first_batch))

        embedding_rows: list[np.ndarray] = []
        logits_rows: list[np.ndarray] = []
        labels_rows: list[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                batch = move_local_particle_residual_field_batch_to_device(batch, device)
                with amp_autocast_context(amp_enabled):
                    logits, embedding = capture.capture(lambda: _campaign_forward(model, batch))
                logits_rows.append(logits.float().cpu().numpy())
                embedding_rows.append(embedding.float().cpu().numpy())
                labels_rows.append(batch["labels"].detach().cpu().numpy().astype(np.int64))
        logits = np.concatenate(logits_rows, axis=0).astype(np.float32, copy=False)
        embeddings = np.concatenate(embedding_rows, axis=0).astype(np.float32, copy=False)
        labels = np.concatenate(labels_rows, axis=0).astype(np.int64, copy=False)
        if not np.array_equal(labels, prediction.labels):
            raise ValueError(f"{config.member_id}/{split} feature labels do not match prediction labels")
        if logits.shape != prediction.logits.shape or not np.allclose(
            logits,
            prediction.logits,
            atol=config.logits_atol,
            rtol=config.logits_rtol,
        ):
            max_abs = float(np.max(np.abs(logits - prediction.logits))) if logits.shape == prediction.logits.shape else None
            raise ValueError(
                f"{config.member_id}/{split} feature forward does not reproduce cached logits; max_abs={max_abs}"
            )
        jet_files, file_indices, entries = _identity_arrays(prediction.jet_ids)
        storage = embeddings.astype(np.float16 if config.storage_dtype == "float16" else np.float32)
        npz_path = member_dir / f"{split}_representations.npz"
        metadata_path = member_dir / f"{split}_representations_metadata.json"
        _atomic_npz(
            npz_path,
            overwrite=config.overwrite,
            jet_embedding=storage,
            labels=labels,
            jet_file_indices=file_indices,
            jet_entries=entries,
        )
        alignment = prediction.metadata.get("dataset_metadata", {}).get("alignment_report", {})
        metadata = {
            "ok": True,
            "contract": LOCAL_RESIDUAL_FIELD_FUSION_FEATURE_CONTRACT,
            "member_id": config.member_id,
            "split": split,
            "representation_name": LOCAL_RESIDUAL_FIELD_FUSION_REPRESENTATION_NAME,
            "representation_source": LOCAL_RESIDUAL_FIELD_FUSION_REPRESENTATION_SOURCE,
            "final_head_name": capture.final_head_name,
            "representation_path": str(npz_path.resolve()),
            "representation_sha256": sha256_file(npz_path),
            "representation_content_hash": _array_sha256(storage),
            "representation_dtype": str(storage.dtype),
            "representation_shape": list(storage.shape),
            "embedding_dim": int(storage.shape[1]),
            "labels_hash": _array_sha256(labels),
            "jet_files": jet_files,
            "checkpoint": str(Path(config.checkpoint).resolve()),
            "checkpoint_hash": checkpoint_hash,
            "prediction_path": source_row["splits"][split]["prediction_path"],
            "prediction_sha256": source_row["splits"][split]["prediction_sha256"],
            "prediction_content_hash": prediction.metadata.get("prediction_content_hash"),
            "jet_identity_hash": prediction.metadata.get("jet_identity_hash"),
            "source_manifest_hash": alignment.get("source_manifest_hash"),
            "hlt_content_hash": alignment.get("hlt_content_hash"),
            "source_artifact_audit_hash": audit["audit_hash"],
            "prediction_sources_hash": registry["manifest_hash"],
            "runtime_inputs": "HLT_only",
            "uses_true_fields": False,
            "uses_offline_particles": False,
            "uses_teacher_logits_at_runtime": False,
            "deployable": True,
            "model_eval": not bool(model.training),
            "trainable_parameter_count": capture.trainable_parameter_count,
            "stochastic_field_corruption": False,
            "amp_enabled": amp_enabled,
            "preflight": preflight,
            "recomputed_logits_max_abs_diff": float(np.max(np.abs(logits - prediction.logits))),
        }
        metadata["metadata_hash"] = stable_fusion_json_hash(metadata)
        _atomic_json(metadata_path, metadata, overwrite=config.overwrite)
        split_reports[split] = metadata
        all_identities[split] = {
            (str(identity.file), int(identity.entry), int(identity.label)) for identity in prediction.jet_ids
        }
    if all_identities[FUSION_DEVELOPMENT_SPLITS[0]].intersection(all_identities[FUSION_DEVELOPMENT_SPLITS[1]]):
        raise ValueError(f"{config.member_id} representation development splits overlap")
    manifest = {
        "ok": True,
        "contract": LOCAL_RESIDUAL_FIELD_FUSION_FEATURE_MANIFEST_CONTRACT,
        "member_id": config.member_id,
        "config": asdict(config),
        "source_artifact_audit_path": str(Path(config.source_artifact_audit).resolve()),
        "source_artifact_audit_hash": audit["audit_hash"],
        "prediction_sources_path": str(Path(config.prediction_sources).resolve()),
        "prediction_sources_hash": registry["manifest_hash"],
        "checkpoint_path": str(Path(config.checkpoint).resolve()),
        "checkpoint_hash": checkpoint_hash,
        "development_splits": list(FUSION_DEVELOPMENT_SPLITS),
        "final_test_opened": False,
        "splits": split_reports,
    }
    manifest["manifest_hash"] = stable_fusion_json_hash(manifest)
    manifest_path = member_dir / "representation_manifest.json"
    _atomic_json(manifest_path, manifest, overwrite=config.overwrite)
    return manifest


__all__ = [
    "LOCAL_RESIDUAL_FIELD_FUSION_FEATURE_CONTRACT",
    "LOCAL_RESIDUAL_FIELD_FUSION_FEATURE_MANIFEST_CONTRACT",
    "LOCAL_RESIDUAL_FIELD_FUSION_REPRESENTATION_NAME",
    "LOCAL_RESIDUAL_FIELD_FUSION_REPRESENTATION_SOURCE",
    "FUSION_FEATURE_MEMBERS",
    "PreClassifierEmbeddingCapture",
    "FusionFeatureCacheConfig",
    "require_development_prediction_sources",
    "cache_local_residual_field_fusion_features",
]
