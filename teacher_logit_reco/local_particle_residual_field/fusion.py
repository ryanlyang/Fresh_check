"""Prediction caching and fusion for local particle residual-field taggers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from jetclass_fresh.fusion import (
    PredictionBlock,
    classification_metrics_from_logits,
    load_prediction_block,
    save_prediction_block,
    softmax_np,
    validate_prediction_alignment,
)
from jetclass_fresh.hlt_baseline import (
    amp_autocast_context,
    require_torch,
    resolve_device,
    save_json,
)
from jetclass_fresh.hlt_cache import load_cached_hlt_view
from jetclass_fresh.jetclass_data import LABEL_NAMES, load_split_manifest, manifest_hash
from jetclass_fresh.part_inputs import build_particle_transformer_inputs_from_tokens

from .data import (
    LocalParticleResidualFieldDatasetConfig,
    load_local_particle_residual_field_dataset,
    make_local_particle_residual_field_loader,
    move_local_particle_residual_field_batch_to_device,
)
from .curriculum import (
    LOCAL_RESIDUAL_FIELD_CURRICULUM_DEPLOYABLE_CONTRACT,
    LOCAL_RESIDUAL_FIELD_CURRICULUM_JOINT_CONTRACT,
    LocalResidualFieldCurriculumJointModel,
)
from .tagger import (
    LOCAL_RESIDUAL_FIELD_TAGGER_CONTRACT,
    ORACLE_RESIDUAL_FIELD_SOURCES,
    RESIDUAL_FIELD_SOURCE_JOINT_RECONSTRUCTOR,
    RESIDUAL_FIELD_SOURCE_FROZEN_RECONSTRUCTOR,
    RESIDUAL_FIELD_SOURCE_HLT_ONLY,
    RESIDUAL_FIELD_SOURCE_LEARNED_NO_TARGET,
    RESIDUAL_FIELD_SOURCE_ZERO,
    LocalResidualFieldAugmentedParT,
)
from .train import _jsonable, _torch_load_checkpoint


LOCAL_RESIDUAL_FIELD_PREDICTION_CONTRACT = "local_particle_residual_field_predictions_v1"
LOCAL_RESIDUAL_FIELD_FUSION_CONTRACT = "local_particle_residual_field_fusion_v1"
LOCAL_RESIDUAL_FIELD_PARTICLE_VIEW_FUSION_CONTRACT = "local_particle_residual_field_particle_view_fusion_v1"

LOCAL_RESIDUAL_FIELD_FUSION_MODE_UNIFORM_LOGIT_MEAN = "uniform_logit_mean"
LOCAL_RESIDUAL_FIELD_FUSION_MODE_SCALAR_WEIGHTED_LOGIT_MEAN = "scalar_weighted_logit_mean"
LOCAL_RESIDUAL_FIELD_FUSION_MODES = (
    LOCAL_RESIDUAL_FIELD_FUSION_MODE_UNIFORM_LOGIT_MEAN,
    LOCAL_RESIDUAL_FIELD_FUSION_MODE_SCALAR_WEIGHTED_LOGIT_MEAN,
)

LOCAL_RESIDUAL_FIELD_FUSION_DEFAULT_SPLITS = ("stack_train", "stack_val", "final_test")
LOCAL_RESIDUAL_FIELD_FUSION_FIT_SPLIT = "stack_train"


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class LocalResidualFieldPredictionConfig:
    """Cache logits from one trained local residual-field tagger."""

    checkpoint: str
    prediction_dir: str
    model_name: str
    hlt_cache_dir: str
    target_cache_dir: str | None = None
    manifest_path: str | None = None
    splits: tuple[str, ...] = LOCAL_RESIDUAL_FIELD_FUSION_DEFAULT_SPLITS
    batch_size: int = 128
    num_workers: int = 0
    device: str = "auto"
    amp: bool = True
    max_jets: int | None = None
    confirm_final_test: bool = False
    allow_oracle_final_test: bool = False
    overwrite: bool = False
    verify_hash: bool = True
    require_manifest_match: bool = True

    def __post_init__(self) -> None:
        self.checkpoint = str(self.checkpoint)
        self.prediction_dir = str(self.prediction_dir)
        self.model_name = str(self.model_name).strip()
        if not self.model_name:
            raise ValueError("model_name must be non-empty")
        self.hlt_cache_dir = str(self.hlt_cache_dir)
        self.target_cache_dir = None if not self.target_cache_dir else str(self.target_cache_dir)
        self.manifest_path = None if not self.manifest_path else str(self.manifest_path)
        self.splits = tuple(str(split) for split in self.splits)
        if not self.splits:
            raise ValueError("at least one split is required")
        if "final_test" in self.splits and not bool(self.confirm_final_test):
            raise ValueError("final_test prediction caching requires confirm_final_test=True")
        self.batch_size = int(self.batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.num_workers = int(self.num_workers)
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        self.max_jets = None if self.max_jets is None else int(self.max_jets)
        if self.max_jets is not None and self.max_jets <= 0:
            raise ValueError("max_jets must be positive when provided")
        self.confirm_final_test = bool(self.confirm_final_test)
        self.allow_oracle_final_test = bool(self.allow_oracle_final_test)
        self.overwrite = bool(self.overwrite)
        self.verify_hash = bool(self.verify_hash)
        self.require_manifest_match = bool(self.require_manifest_match)


@dataclass
class LocalResidualFieldFusionConfig:
    """Run late-logit fusion over cached prediction blocks."""

    prediction_dir: str
    output_dir: str
    groups: Mapping[str, Sequence[str]]
    splits: tuple[str, ...] = LOCAL_RESIDUAL_FIELD_FUSION_DEFAULT_SPLITS
    fusion_modes: tuple[str, ...] = LOCAL_RESIDUAL_FIELD_FUSION_MODES
    fit_split: str = LOCAL_RESIDUAL_FIELD_FUSION_FIT_SPLIT
    scalar_weight_trials: int = 128
    control_seed: int = 4079
    confirm_final_test: bool = False
    verify_hash: bool = True

    def __post_init__(self) -> None:
        self.prediction_dir = str(self.prediction_dir)
        self.output_dir = str(self.output_dir)
        normalized_groups: dict[str, tuple[str, ...]] = {}
        for name, members in dict(self.groups).items():
            clean_name = str(name).strip()
            clean_members = tuple(str(member).strip() for member in members if str(member).strip())
            if not clean_name:
                raise ValueError("fusion group names must be non-empty")
            if not clean_members:
                raise ValueError(f"fusion group {clean_name!r} has no members")
            if len(clean_members) != len(set(clean_members)):
                raise ValueError(f"fusion group {clean_name!r} contains duplicate members")
            normalized_groups[clean_name] = clean_members
        if not normalized_groups:
            raise ValueError("at least one fusion group is required")
        self.groups = normalized_groups
        self.splits = tuple(str(split) for split in self.splits)
        if "final_test" in self.splits and not bool(self.confirm_final_test):
            raise ValueError("final_test fusion requires confirm_final_test=True")
        self.fusion_modes = tuple(str(mode) for mode in self.fusion_modes)
        unknown = sorted(set(self.fusion_modes) - set(LOCAL_RESIDUAL_FIELD_FUSION_MODES))
        if unknown:
            raise ValueError(f"unknown fusion modes: {unknown}")
        self.fit_split = str(self.fit_split)
        self.scalar_weight_trials = int(self.scalar_weight_trials)
        if self.scalar_weight_trials <= 0:
            raise ValueError("scalar_weight_trials must be positive")
        self.control_seed = int(self.control_seed)
        self.confirm_final_test = bool(self.confirm_final_test)
        self.verify_hash = bool(self.verify_hash)


@dataclass(frozen=True)
class LocalResidualFieldParticleViewFusionOutput:
    """Output from the particle-level residual-view gate."""

    fused_fields: Any
    gates: Any
    diagnostics: Mapping[str, Any]


class LocalResidualFieldParticleViewFusion(torch.nn.Module):
    """Learn per-particle gates over multiple residual-field views."""

    def __init__(self, *, field_dim: int, num_views: int, hidden_dim: int = 128, dropout: float = 0.05) -> None:
        super().__init__()
        require_torch()
        self.gate = torch.nn.Sequential(
            torch.nn.LayerNorm(int(field_dim) * int(num_views)),
            torch.nn.Linear(int(field_dim) * int(num_views), int(hidden_dim)),
            torch.nn.GELU(),
            torch.nn.Dropout(float(dropout)),
            torch.nn.Linear(int(hidden_dim), int(num_views)),
        )
        self.field_dim = int(field_dim)
        self.num_views = int(num_views)

    def forward(self, view_fields: Any, mask: Any | None = None) -> LocalResidualFieldParticleViewFusionOutput:
        if view_fields.ndim != 4:
            raise ValueError("view_fields must have shape [B, V, P, F]")
        if int(view_fields.shape[1]) != self.num_views:
            raise ValueError(f"view count {view_fields.shape[1]} != configured {self.num_views}")
        if int(view_fields.shape[-1]) != self.field_dim:
            raise ValueError(f"field dim {view_fields.shape[-1]} != configured {self.field_dim}")
        batch, views, particles, field_dim = view_fields.shape
        flat = view_fields.permute(0, 2, 1, 3).reshape(batch, particles, views * field_dim)
        gates = torch.softmax(self.gate(flat), dim=-1)
        fused = (view_fields * gates.permute(0, 2, 1).unsqueeze(-1)).sum(dim=1)
        if mask is not None:
            if mask.ndim == 3:
                mask = mask.squeeze(1)
            mask = mask.to(device=fused.device, dtype=torch.bool)
            fused = fused * mask.unsqueeze(-1).to(dtype=fused.dtype)
            gates = gates * mask.unsqueeze(-1).to(dtype=gates.dtype)
        diagnostics = {
            "contract": LOCAL_RESIDUAL_FIELD_PARTICLE_VIEW_FUSION_CONTRACT,
            "num_views": int(views),
            "field_dim": int(field_dim),
            "gate_mean": float(gates.detach().mean().cpu().item()) if int(gates.numel()) else 0.0,
        }
        return LocalResidualFieldParticleViewFusionOutput(fused_fields=fused, gates=gates, diagnostics=diagnostics)


class _HLTOnlyPredictionDataset:
    def __init__(self, hlt_view: Any, *, max_jets: int | None = None) -> None:
        n_rows = int(len(hlt_view.labels))
        if max_jets is not None:
            n_rows = min(n_rows, int(max_jets))
        self.tokens = np.asarray(hlt_view.tokens[:n_rows], dtype=np.float32)
        self.mask = np.asarray(hlt_view.mask[:n_rows], dtype=bool)
        self.labels = np.asarray(hlt_view.labels[:n_rows], dtype=np.int64)
        self.jet_ids = tuple(hlt_view.jet_ids[:n_rows])
        self.metadata = {
            "contract": "local_particle_residual_field_hlt_only_prediction_dataset_v1",
            "allowed_inputs": "HLT_particles_only_deployable_final_test",
            "split": str(hlt_view.split),
            "n_jets": int(n_rows),
            "max_particles": int(self.tokens.shape[1]),
            "raw_token_dim": int(self.tokens.shape[2]),
            "target_fields_present": False,
            "teacher_logits_present": False,
            "alignment_report": {
                "hlt_content_hash": hlt_view.metadata.get("hlt_content_hash"),
                "jet_identity_hash": hlt_view.metadata.get("jet_identity_hash"),
                "source_manifest_hash": hlt_view.metadata.get("source_manifest_hash"),
            },
            "hlt_metadata": {
                "view": hlt_view.metadata.get("view"),
                "hlt_content_hash": hlt_view.metadata.get("hlt_content_hash"),
                "jet_identity_hash": hlt_view.metadata.get("jet_identity_hash"),
                "source_manifest_hash": hlt_view.metadata.get("source_manifest_hash"),
                "hlt_profile": hlt_view.metadata.get("hlt_profile"),
                "hlt_profile_version": hlt_view.metadata.get("hlt_profile_version"),
                "hlt_degradation_strength": hlt_view.metadata.get("hlt_degradation_strength"),
            },
        }

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "tokens": self.tokens[index],
            "mask": self.mask[index],
            "labels": np.int64(self.labels[index]),
            "indices": np.int64(index),
        }


def _collate_hlt_only_prediction_batch(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    torch = require_torch()
    tokens = np.stack([np.asarray(sample["tokens"], dtype=np.float32) for sample in samples], axis=0)
    raw_mask = np.stack([np.asarray(sample["mask"], dtype=bool) for sample in samples], axis=0)
    labels = np.asarray([sample["labels"] for sample in samples], dtype=np.int64)
    part_inputs = build_particle_transformer_inputs_from_tokens(
        tokens,
        raw_mask,
        labels=labels,
        source_view="fixed_hlt",
    )
    return {
        "tokens": torch.from_numpy(tokens).float(),
        "raw_mask": torch.from_numpy(raw_mask).bool(),
        "points": torch.from_numpy(part_inputs.pf_points).float(),
        "features": torch.from_numpy(part_inputs.pf_features).float(),
        "lorentz_vectors": torch.from_numpy(part_inputs.pf_vectors).float(),
        "mask": torch.from_numpy(part_inputs.pf_mask).bool(),
        "labels": torch.from_numpy(labels).long(),
        "indices": torch.from_numpy(np.asarray([sample["indices"] for sample in samples], dtype=np.int64)).long(),
    }


def _make_prediction_loader(dataset: Any, *, batch_size: int, num_workers: int, seed: int, hlt_only: bool = False):
    torch = require_torch()
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate_hlt_only_prediction_batch if hlt_only else None,
        generator=generator,
    )


def _metrics_from_logits(logits: np.ndarray, labels: np.ndarray, *, label_names: Sequence[str]) -> dict[str, Any]:
    base = classification_metrics_from_logits(logits, labels)
    preds = np.argmax(logits, axis=1).astype(np.int64)
    names = tuple(str(name) for name in label_names)
    n_classes = int(logits.shape[1])
    if len(names) != n_classes:
        names = tuple(str(index) for index in range(n_classes))
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    for true_label, pred_label in zip(labels, preds):
        if 0 <= int(true_label) < n_classes and 0 <= int(pred_label) < n_classes:
            confusion[int(true_label), int(pred_label)] += 1
    per_class = []
    for index, name in enumerate(names):
        support = int(confusion[index].sum())
        correct = int(confusion[index, index])
        per_class.append(
            {
                "class_index": int(index),
                "class_name": str(name),
                "support": support,
                "correct": correct,
                "accuracy": correct / float(support) if support else 0.0,
            }
        )
    base["confusion_matrix"] = confusion.astype(int).tolist()
    base["per_class_accuracy"] = per_class
    base["macro_per_class_accuracy"] = float(np.mean([row["accuracy"] for row in per_class])) if per_class else 0.0
    return base


def load_local_residual_field_tagger_from_checkpoint(
    checkpoint: str | Path,
    *,
    device: Any = "cpu",
) -> tuple[Any, Mapping[str, Any]]:
    payload = _torch_load_checkpoint(checkpoint, map_location=device)
    if not isinstance(payload, Mapping):
        raise ValueError("local residual-field checkpoint must contain a mapping")
    if payload.get("contract") == LOCAL_RESIDUAL_FIELD_CURRICULUM_DEPLOYABLE_CONTRACT:
        model = LocalResidualFieldCurriculumJointModel.from_deployable_checkpoint(payload, device=device)
        return model, payload
    model_config = dict(payload.get("model_config") or {})
    model_config.pop("contract", None)
    model_config.pop("augmented_feature_dim", None)
    model = LocalResidualFieldAugmentedParT(model_config).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, payload


def _requires_target_fields_for_prediction(model: LocalResidualFieldAugmentedParT) -> bool:
    return model.config.field_source not in {
        RESIDUAL_FIELD_SOURCE_HLT_ONLY,
        RESIDUAL_FIELD_SOURCE_ZERO,
        RESIDUAL_FIELD_SOURCE_FROZEN_RECONSTRUCTOR,
        RESIDUAL_FIELD_SOURCE_JOINT_RECONSTRUCTOR,
        RESIDUAL_FIELD_SOURCE_LEARNED_NO_TARGET,
    }


def _load_hlt_only_prediction_dataset(
    config: LocalResidualFieldPredictionConfig,
    split: str,
) -> _HLTOnlyPredictionDataset:
    hlt_view = load_cached_hlt_view(config.hlt_cache_dir, str(split), verify_hash=bool(config.verify_hash))
    if config.manifest_path and bool(config.require_manifest_match):
        manifest = load_split_manifest(config.manifest_path)
        expected = tuple(manifest.splits[str(split)])
        if tuple(hlt_view.jet_ids) != expected:
            raise ValueError(f"HLT-only prediction split {split} does not match split manifest")
        expected_hash = manifest_hash(manifest)
        hlt_manifest_hash = hlt_view.metadata.get("source_manifest_hash")
        if hlt_manifest_hash not in (None, expected_hash):
            raise ValueError(f"HLT source_manifest_hash {hlt_manifest_hash} != manifest hash {expected_hash}")
    return _HLTOnlyPredictionDataset(hlt_view, max_jets=config.max_jets)


def _prediction_dataset(config: LocalResidualFieldPredictionConfig, split: str, *, model: LocalResidualFieldAugmentedParT):
    split_name = str(split)
    if isinstance(model, LocalResidualFieldCurriculumJointModel):
        return _load_hlt_only_prediction_dataset(config, split)
    deployable_no_target = (
        model.config.field_source not in ORACLE_RESIDUAL_FIELD_SOURCES
        and not _requires_target_fields_for_prediction(model)
    )
    if deployable_no_target:
        return _load_hlt_only_prediction_dataset(config, split)
    if split_name == "final_test" and model.config.field_source not in ORACLE_RESIDUAL_FIELD_SOURCES:
        raise ValueError(
            f"refusing deployment final_test prediction for target-dependent field source "
            f"{model.config.field_source!r}"
        )
    if not config.target_cache_dir:
        raise ValueError(
            f"target_cache_dir is required for target-dependent prediction source {model.config.field_source!r}"
        )
    return load_local_particle_residual_field_dataset(
        LocalParticleResidualFieldDatasetConfig(
            hlt_cache_dir=config.hlt_cache_dir,
            target_cache_dir=config.target_cache_dir,
            split=str(split),
            manifest_path=config.manifest_path,
            max_jets=config.max_jets,
            include_oracle_fields=model.config.field_source in ORACLE_RESIDUAL_FIELD_SOURCES,
            allow_final_test_targets=bool(split == "final_test" and config.confirm_final_test),
            verify_hash=bool(config.verify_hash),
            require_manifest_match=bool(config.require_manifest_match),
        )
    )


def _collect_prediction_logits(
    model: Any,
    loader: Any,
    *,
    device: Any,
    amp_enabled: bool,
) -> tuple[np.ndarray, np.ndarray]:
    torch = require_torch()
    logits_chunks: list[np.ndarray] = []
    labels_chunks: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = move_local_particle_residual_field_batch_to_device(batch, device)
            with amp_autocast_context(bool(amp_enabled)):
                if isinstance(model, LocalResidualFieldCurriculumJointModel):
                    if model.oracle_consumer is not None:
                        raise ValueError("deployable curriculum prediction model unexpectedly contains an oracle consumer")
                    output = model(
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
                else:
                    output = model(
                        batch["points"],
                        batch["features"],
                        batch["lorentz_vectors"],
                        batch["mask"],
                        tokens=batch["tokens"],
                        raw_mask=batch["raw_mask"],
                        indices=batch["indices"],
                        target_fields=batch.get("target_fields"),
                        oracle_fields=batch.get("oracle_fields"),
                        return_outputs=True,
                    )
            output_logits = output.student_logits if hasattr(output, "student_logits") else output.logits
            logits = output_logits.detach().float().cpu().numpy()
            if not np.isfinite(logits).all():
                raise FloatingPointError("local residual-field prediction logits contain non-finite values")
            logits_chunks.append(logits)
            labels_chunks.append(batch["labels"].detach().cpu().numpy().astype(np.int64))
    if not logits_chunks:
        raise ValueError("prediction loader produced no batches")
    return np.concatenate(logits_chunks, axis=0), np.concatenate(labels_chunks, axis=0)


def cache_local_residual_field_tagger_predictions(config: LocalResidualFieldPredictionConfig) -> dict[str, Any]:
    torch = require_torch()
    device = resolve_device(str(config.device))
    amp_enabled = bool(config.amp and getattr(device, "type", str(device)) == "cuda")
    model, payload = load_local_residual_field_tagger_from_checkpoint(config.checkpoint, device=device)
    curriculum_deployable = isinstance(model, LocalResidualFieldCurriculumJointModel)
    checkpoint_path = Path(config.checkpoint)
    checkpoint_hash = _sha256_file(checkpoint_path)
    teacher_config_path = checkpoint_path.with_name("teacher_config.json")
    teacher_config: dict[str, Any] = {}
    if teacher_config_path.exists():
        loaded_teacher_config = json.loads(teacher_config_path.read_text(encoding="utf-8"))
        if not isinstance(loaded_teacher_config, Mapping):
            raise ValueError(f"teacher_config.json is not an object: {teacher_config_path}")
        teacher_config = dict(loaded_teacher_config)
    if (
        not curriculum_deployable
        and model.config.field_source in ORACLE_RESIDUAL_FIELD_SOURCES
        and "final_test" in config.splits
        and not bool(config.allow_oracle_final_test)
    ):
        raise ValueError("refusing to cache final_test oracle predictions without allow_oracle_final_test=True")
    output_dir = Path(config.prediction_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_reports: dict[str, Any] = {}
    for split in config.splits:
        dataset = _prediction_dataset(config, split, model=model)
        if isinstance(dataset, _HLTOnlyPredictionDataset):
            loader = _make_prediction_loader(
                dataset,
                batch_size=int(config.batch_size),
                num_workers=int(config.num_workers),
                seed=0,
                hlt_only=True,
            )
        else:
            loader = make_local_particle_residual_field_loader(
                dataset,
                batch_size=int(config.batch_size),
                shuffle=False,
                num_workers=int(config.num_workers),
                seed=0,
            )
        logits, labels = _collect_prediction_logits(model, loader, device=device, amp_enabled=bool(amp_enabled))
        if not np.array_equal(labels, dataset.labels):
            raise ValueError(f"prediction labels do not match dataset labels for split {split}")
        checkpoint_metadata = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {}
        oracle_teacher_diagnostic = bool(
            not curriculum_deployable and teacher_config.get("role") == "oracle_teacher_candidate"
        )
        uses_true_fields = bool(
            not curriculum_deployable and model.config.field_source in ORACLE_RESIDUAL_FIELD_SOURCES
        )
        field_source = "curriculum_predicted" if curriculum_deployable else str(model.config.field_source)
        model_contract = (
            LOCAL_RESIDUAL_FIELD_CURRICULUM_JOINT_CONTRACT
            if curriculum_deployable
            else LOCAL_RESIDUAL_FIELD_TAGGER_CONTRACT
        )
        student_checkpoint_hash = checkpoint_hash
        predictor_checkpoint_hash = checkpoint_hash if curriculum_deployable else None
        predictor_checkpoint = None
        if not curriculum_deployable:
            train_config = payload.get("config") if isinstance(payload.get("config"), Mapping) else {}
            predictor_checkpoint = train_config.get("reconstructor_checkpoint")
            if predictor_checkpoint and Path(str(predictor_checkpoint)).is_file():
                predictor_checkpoint_hash = _sha256_file(str(predictor_checkpoint))
            elif model.config.field_source in {
                RESIDUAL_FIELD_SOURCE_FROZEN_RECONSTRUCTOR,
                RESIDUAL_FIELD_SOURCE_JOINT_RECONSTRUCTOR,
            }:
                predictor_checkpoint_hash = checkpoint_hash
        teacher_used_during_training = (
            checkpoint_metadata.get("teacher_used_during_training")
            if curriculum_deployable
            else teacher_config.get("teacher_id")
        )
        runtime_inputs = (
            "HLT_plus_true_residual_fields"
            if uses_true_fields
            else "HLT_plus_zero_residual_fields"
            if oracle_teacher_diagnostic and model.config.field_source == RESIDUAL_FIELD_SOURCE_ZERO
            else "HLT_only"
        )
        metadata = {
            "contract": LOCAL_RESIDUAL_FIELD_PREDICTION_CONTRACT,
            "model_contract": model_contract,
            "checkpoint": str(config.checkpoint),
            "checkpoint_hash": checkpoint_hash,
            "student_checkpoint_hash": student_checkpoint_hash,
            "predictor_checkpoint": predictor_checkpoint,
            "predictor_checkpoint_hash": predictor_checkpoint_hash,
            "component_checkpoint_storage": (
                "joint_deployable_checkpoint" if curriculum_deployable else "tagger_checkpoint"
            ),
            "checkpoint_epoch": payload.get("epoch"),
            "field_source": field_source,
            "model_config": payload.get("model_config") if curriculum_deployable else model.config.to_dict(),
            "dataset_metadata": _jsonable(dataset.metadata),
            "run_id": checkpoint_metadata.get("run_id") if curriculum_deployable else None,
            "teacher_used_during_training": teacher_used_during_training,
            "teacher_id": teacher_config.get("teacher_id") if not curriculum_deployable else None,
            "teacher_role": teacher_config.get("role"),
            "oracle_teacher_diagnostic": oracle_teacher_diagnostic,
            "teacher_config": str(teacher_config_path) if teacher_config else None,
            "teacher_config_hash": _sha256_file(teacher_config_path) if teacher_config else None,
            "teacher_reuse_contract_hash": (
                teacher_config.get("reuse_contract", {}).get("reuse_contract_hash")
                if isinstance(teacher_config.get("reuse_contract"), Mapping)
                else None
            ),
            "runtime_inputs": runtime_inputs,
            "uses_true_fields": uses_true_fields,
            "uses_offline_particles": False,
            "uses_teacher_logits_at_runtime": False,
            "deployable": bool(not oracle_teacher_diagnostic and not uses_true_fields),
            "split": str(split),
            "selection_allowed": str(split) != "final_test",
        }
        block = PredictionBlock(
            model_name=str(config.model_name),
            split=str(split),
            logits=logits,
            probs=softmax_np(logits),
            labels=labels,
            jet_ids=list(dataset.jet_ids),
            metadata=metadata,
        )
        split_reports[str(split)] = save_prediction_block(block, output_dir, overwrite=bool(config.overwrite))
    report = {
        "ok": True,
        "contract": LOCAL_RESIDUAL_FIELD_PREDICTION_CONTRACT,
        "prediction_dir": str(output_dir),
        "model_name": str(config.model_name),
        "checkpoint": str(config.checkpoint),
        "checkpoint_hash": checkpoint_hash,
        "student_checkpoint_hash": checkpoint_hash,
        "predictor_checkpoint_hash": checkpoint_hash if curriculum_deployable else None,
        "teacher_used_during_training": (
            payload.get("metadata", {}).get("teacher_used_during_training")
            if curriculum_deployable and isinstance(payload.get("metadata"), Mapping)
            else teacher_config.get("teacher_id")
        ),
        "runtime_inputs": "HLT_only" if curriculum_deployable else None,
        "curriculum_deployable_checkpoint": bool(curriculum_deployable),
        "splits": list(config.splits),
        "split_reports": _jsonable(split_reports),
    }
    save_json(output_dir / str(config.model_name) / "prediction_manifest.json", report)
    return report


def _load_group_blocks(prediction_dir: str | Path, members: Sequence[str], split: str, *, verify_hash: bool) -> list[PredictionBlock]:
    blocks: list[PredictionBlock] = []
    missing: list[str] = []
    for member in members:
        try:
            blocks.append(load_prediction_block(prediction_dir, str(member), str(split), verify_hash=bool(verify_hash)))
        except FileNotFoundError:
            missing.append(str(member))
    if missing:
        raise FileNotFoundError(f"Missing prediction blocks for split {split}: {', '.join(missing)}")
    validate_prediction_alignment(blocks)
    return blocks


def _preflight_fusion_blocks(config: LocalResidualFieldFusionConfig) -> dict[tuple[str, str], list[PredictionBlock]]:
    """Resolve every selected group member/split before any fusion output is written."""

    cache: dict[tuple[str, str], list[PredictionBlock]] = {}
    missing: list[str] = []
    invalid: list[str] = []
    required_splits = tuple(dict.fromkeys((str(config.fit_split), *config.splits)))
    for group_name, members in config.groups.items():
        for split in required_splits:
            try:
                cache[(str(group_name), str(split))] = _load_group_blocks(
                    config.prediction_dir,
                    members,
                    split,
                    verify_hash=bool(config.verify_hash),
                )
            except FileNotFoundError as exc:
                missing.append(f"{group_name}/{split}: {exc}")
            except ValueError as exc:
                invalid.append(f"{group_name}/{split}: {exc}")
    if missing:
        raise FileNotFoundError("Missing prediction blocks for selected fusion members: " + "; ".join(missing))
    if invalid:
        raise ValueError("Invalid selected-member prediction alignment: " + "; ".join(invalid))
    return cache


def _fused_runtime_metadata(blocks: Sequence[PredictionBlock]) -> dict[str, Any]:
    metadata = [dict(block.metadata or {}) for block in blocks]
    runtime_values = {str(item.get("runtime_inputs") or "unknown") for item in metadata}
    return {
        "runtime_inputs": next(iter(runtime_values)) if len(runtime_values) == 1 else "mixed",
        "uses_true_fields": any(bool(item.get("uses_true_fields")) for item in metadata),
        "uses_offline_particles": any(bool(item.get("uses_offline_particles")) for item in metadata),
        "uses_teacher_logits_at_runtime": any(
            bool(item.get("uses_teacher_logits_at_runtime")) for item in metadata
        ),
        "deployable": all(item.get("deployable") is True for item in metadata),
        "selection_allowed": all(item.get("selection_allowed") is not False for item in metadata),
        "student_checkpoint_hashes": {
            str(block.model_name): item.get("student_checkpoint_hash")
            for block, item in zip(blocks, metadata)
        },
        "predictor_checkpoint_hashes": {
            str(block.model_name): item.get("predictor_checkpoint_hash")
            for block, item in zip(blocks, metadata)
        },
        "teachers_used_during_training": {
            str(block.model_name): item.get("teacher_used_during_training")
            for block, item in zip(blocks, metadata)
        },
    }


def _uniform_logits(blocks: Sequence[PredictionBlock]) -> np.ndarray:
    return np.mean(np.stack([block.logits for block in blocks], axis=0), axis=0).astype(np.float32)


def _sample_simplex(rng: np.random.Generator, n_members: int, n_trials: int) -> np.ndarray:
    if n_members == 1:
        return np.ones((1, 1), dtype=np.float32)
    weights = rng.dirichlet(np.ones((n_members,), dtype=np.float64), size=int(n_trials)).astype(np.float32)
    uniform = np.full((1, n_members), 1.0 / float(n_members), dtype=np.float32)
    return np.concatenate([uniform, weights], axis=0)


def _weighted_logits(blocks: Sequence[PredictionBlock], weights: np.ndarray) -> np.ndarray:
    stacked = np.stack([block.logits for block in blocks], axis=0)
    return np.tensordot(weights.astype(np.float32), stacked.astype(np.float32), axes=(0, 0)).astype(np.float32)


def _fit_scalar_weights(blocks: Sequence[PredictionBlock], *, trials: int, seed: int) -> dict[str, Any]:
    validate_prediction_alignment(blocks)
    rng = np.random.default_rng(int(seed))
    candidates = _sample_simplex(rng, len(blocks), int(trials))
    best_weights = candidates[0]
    best_metrics = _metrics_from_logits(_weighted_logits(blocks, best_weights), blocks[0].labels, label_names=LABEL_NAMES)
    best_ce = float(best_metrics["cross_entropy"])
    for weights in candidates[1:]:
        metrics = _metrics_from_logits(_weighted_logits(blocks, weights), blocks[0].labels, label_names=LABEL_NAMES)
        ce = float(metrics["cross_entropy"])
        if np.isfinite(ce) and ce < best_ce:
            best_ce = ce
            best_weights = weights
            best_metrics = metrics
    return {
        "weights": [float(value) for value in best_weights],
        "fit_metrics": _jsonable(best_metrics),
        "trials": int(candidates.shape[0]),
    }


def _write_fusion_metric_table(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group", "mode", "split", "accuracy", "cross_entropy", "macro_per_class_accuracy", "n_jets",
        "members", "weights", "runtime_inputs", "uses_true_fields", "uses_offline_particles",
        "uses_teacher_logits_at_runtime", "deployable", "selection_allowed",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_local_residual_field_fusion(config: LocalResidualFieldFusionConfig) -> dict[str, Any]:
    preflight_blocks = _preflight_fusion_blocks(config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "ok": True,
        "contract": LOCAL_RESIDUAL_FIELD_FUSION_CONTRACT,
        "prediction_dir": str(config.prediction_dir),
        "output_dir": str(output_dir),
        "splits": list(config.splits),
        "fit_split": str(config.fit_split),
        "selected_member_predictions_complete": True,
        "preflight_split_count": len(set((str(config.fit_split), *config.splits))),
        "groups": {},
    }
    metric_rows: list[dict[str, Any]] = []
    for group_name, members in config.groups.items():
        group_report: dict[str, Any] = {"members": list(members), "fusion_modes": {}}
        fit_blocks = preflight_blocks[(str(group_name), str(config.fit_split))]
        group_report["member_provenance"] = _fused_runtime_metadata(fit_blocks)
        scalar_fit: dict[str, Any] | None = None
        if LOCAL_RESIDUAL_FIELD_FUSION_MODE_SCALAR_WEIGHTED_LOGIT_MEAN in config.fusion_modes:
            scalar_fit = _fit_scalar_weights(
                fit_blocks,
                trials=int(config.scalar_weight_trials),
                seed=int(config.control_seed),
            )
        for mode in config.fusion_modes:
            mode_report: dict[str, Any] = {"metrics": {}, "available": True}
            weights = None
            if mode == LOCAL_RESIDUAL_FIELD_FUSION_MODE_SCALAR_WEIGHTED_LOGIT_MEAN:
                assert scalar_fit is not None
                weights = np.asarray(scalar_fit["weights"], dtype=np.float32)
                mode_report["fit"] = scalar_fit
            for split in config.splits:
                blocks = preflight_blocks[(str(group_name), str(split))]
                logits = _uniform_logits(blocks) if weights is None else _weighted_logits(blocks, weights)
                metrics = _metrics_from_logits(logits, blocks[0].labels, label_names=LABEL_NAMES)
                runtime_metadata = _fused_runtime_metadata(blocks)
                mode_report["metrics"][split] = _jsonable({**metrics, **runtime_metadata, "split": str(split)})
                metric_rows.append(
                    {
                        "group": group_name,
                        "mode": mode,
                        "split": split,
                        "accuracy": metrics.get("accuracy"),
                        "cross_entropy": metrics.get("cross_entropy"),
                        "macro_per_class_accuracy": metrics.get("macro_per_class_accuracy"),
                        "n_jets": metrics.get("n_jets"),
                        "members": " ".join(members),
                        "weights": "" if weights is None else " ".join(f"{float(value):.8g}" for value in weights),
                        **{
                            key: runtime_metadata[key]
                            for key in (
                                "runtime_inputs", "uses_true_fields", "uses_offline_particles",
                                "uses_teacher_logits_at_runtime", "deployable", "selection_allowed",
                            )
                        },
                    }
                )
            group_report["fusion_modes"][mode] = mode_report
        report["groups"][group_name] = group_report
    save_json(output_dir / "fusion_report.json", report)
    _write_fusion_metric_table(output_dir / "fusion_metrics.csv", metric_rows)
    return report


__all__ = [
    "LOCAL_RESIDUAL_FIELD_PREDICTION_CONTRACT",
    "LOCAL_RESIDUAL_FIELD_FUSION_CONTRACT",
    "LOCAL_RESIDUAL_FIELD_PARTICLE_VIEW_FUSION_CONTRACT",
    "LOCAL_RESIDUAL_FIELD_FUSION_MODE_UNIFORM_LOGIT_MEAN",
    "LOCAL_RESIDUAL_FIELD_FUSION_MODE_SCALAR_WEIGHTED_LOGIT_MEAN",
    "LOCAL_RESIDUAL_FIELD_FUSION_MODES",
    "LOCAL_RESIDUAL_FIELD_FUSION_DEFAULT_SPLITS",
    "LOCAL_RESIDUAL_FIELD_FUSION_FIT_SPLIT",
    "LocalResidualFieldPredictionConfig",
    "LocalResidualFieldFusionConfig",
    "LocalResidualFieldParticleViewFusion",
    "LocalResidualFieldParticleViewFusionOutput",
    "cache_local_residual_field_tagger_predictions",
    "load_local_residual_field_tagger_from_checkpoint",
    "run_local_residual_field_fusion",
]
