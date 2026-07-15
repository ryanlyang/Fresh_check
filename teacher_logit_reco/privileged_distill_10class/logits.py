"""PD10 Step 4 teacher-logit cache utilities."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from jetclass_fresh.fusion import (
    PredictionBlock,
    load_prediction_block,
    prediction_paths,
    sanitize_prediction_logits,
    save_prediction_block,
    softmax_np,
)
from jetclass_fresh.heterogeneous_hlt import balanced_limit_jet_view
from jetclass_fresh.hlt_baseline import (
    JetViewTorchDataset,
    ParticleViewTorchDataset,
    make_data_loader,
    require_torch,
    resolve_device,
    save_json,
)
from jetclass_fresh.hlt_cache import load_cached_hlt_view
from jetclass_fresh.jetclass_data import JetView, LABEL_NAMES, load_offline_view, load_split_manifest, manifest_hash
from teacher_logit_reco.architecture_view_part import load_cached_offline_view

from .config import (
    PD10_EXPERIMENT_NAME,
    PD10_HLT_DEGRADATION_STRENGTH,
    PD10_MANIFEST_SPLIT_ORDER,
    PD10_NUM_CLASSES,
    PD10_SPLIT_ORDER,
    PD10_SPLIT_SIZES,
    PD10_TEACHER_ALLOWED_INPUTS,
    PD10_TEACHER_HLT,
    PD10_TEACHER_OFFLINE,
    default_pd10_experiment_layout,
)
from .inputs import pd10_hlt_params_dict
from .teachers import (
    PD10_PART_TEACHER_TARGETS,
    load_pd10_part_teacher_model_from_checkpoint,
    normalize_pd10_part_teacher_target,
    pd10_part_teacher_model_name,
    sha256_file,
)


PD10_STEP4_EXPERIMENT_STEP = "pd10_step4_teacher_logit_cache"
PD10_TEACHER_LOGIT_CACHE_CONTRACT = "pd10_teacher_logit_cache_v1"
PD10_TEACHER_LOGIT_CACHE_MANIFEST = "teacher_logit_manifest.json"
PD10_TEACHER_LOGIT_CACHE_REPORT = "teacher_logit_cache_report.json"
PD10_TEACHER_LOGIT_SPLITS: tuple[str, ...] = PD10_MANIFEST_SPLIT_ORDER


@dataclass(frozen=True)
class PD10TeacherLogitCacheConfig:
    """Configuration for caching one PD10 teacher's logits on model-facing splits."""

    teacher_target: str
    checkpoint: str
    output_dir: str
    manifest_path: str
    hlt_cache_dir: str
    offline_cache_dir: str | None = None
    data_dir: str | None = None
    splits: tuple[str, ...] = field(default_factory=lambda: PD10_TEACHER_LOGIT_SPLITS)
    batch_size: int = 128
    num_workers: int = 0
    device: str = "auto"
    max_model_train_jets: int | None = PD10_SPLIT_SIZES["model_train"]
    max_model_val_jets: int | None = PD10_SPLIT_SIZES["model_val"]
    max_final_test_jets: int | None = PD10_SPLIT_SIZES["final_test"]
    overwrite: bool = False
    skip_existing: bool = True
    confirm_final_test: bool = False
    verify_label_branches: bool = False
    read_chunk_size: int = 50_000
    control_seed: int = 7207
    verify_hlt_hash: bool = True

    def __post_init__(self) -> None:
        target = normalize_pd10_part_teacher_target(self.teacher_target)
        splits = tuple(str(split) for split in self.splits)
        if not splits:
            raise ValueError("at least one split is required")
        unknown = [split for split in splits if split not in PD10_TEACHER_LOGIT_SPLITS]
        if unknown:
            raise ValueError(f"unknown PD10 teacher-logit splits: {unknown}")
        if "final_test" in splits and not bool(self.confirm_final_test):
            raise ValueError("Refusing to cache final_test teacher logits without confirm_final_test=True")
        if int(self.batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        for split, value in (
            ("model_train", self.max_model_train_jets),
            ("model_val", self.max_model_val_jets),
            ("final_test", self.max_final_test_jets),
        ):
            if value is not None and int(value) > int(PD10_SPLIT_SIZES[split]):
                raise ValueError(f"max jets for {split} cannot exceed {PD10_SPLIT_SIZES[split]}")
        object.__setattr__(self, "teacher_target", target)
        object.__setattr__(self, "splits", splits)

    @property
    def model_name(self) -> str:
        return pd10_part_teacher_model_name(self.teacher_target)

    @property
    def source_view(self) -> str:
        return "fixed_hlt" if self.teacher_target == PD10_TEACHER_HLT else "offline"

    @property
    def teacher_output_dir(self) -> Path:
        return Path(self.output_dir) / self.model_name


def _amp_state_targets(model: Any) -> list[Any]:
    targets: list[Any] = []
    seen: set[int] = set()
    for obj in (model, getattr(model, "mod", None), getattr(model, "module", None)):
        if obj is None or not hasattr(obj, "use_amp"):
            continue
        obj_id = id(obj)
        if obj_id in seen:
            continue
        targets.append(obj)
        seen.add(obj_id)
    return targets


@contextmanager
def _disable_model_amp_for_eval(model: Any):
    targets = _amp_state_targets(model)
    previous = [(target, getattr(target, "use_amp")) for target in targets]
    try:
        for target, _state in previous:
            setattr(target, "use_amp", False)
        yield bool(previous)
    finally:
        for target, state in previous:
            setattr(target, "use_amp", state)


def pd10_teacher_logit_cache_dir(
    teacher_target: str,
    *,
    output_root: str | Path = "checkpoints",
) -> Path:
    layout = default_pd10_experiment_layout(output_root=output_root)
    return layout.teacher_logit_cache_dir(normalize_pd10_part_teacher_target(teacher_target))


def pd10_teacher_logit_prediction_paths(
    output_dir: str | Path,
    teacher_target: str,
    split: str,
) -> tuple[Path, Path]:
    return prediction_paths(output_dir, pd10_part_teacher_model_name(teacher_target), split)


def _max_jets_for_split(config: PD10TeacherLogitCacheConfig, split: str) -> int | None:
    if split == "model_train":
        return config.max_model_train_jets
    if split == "model_val":
        return config.max_model_val_jets
    if split == "final_test":
        return config.max_final_test_jets
    return None


def _expected_size_for_split(split: str, view: JetView) -> int:
    if split in PD10_SPLIT_SIZES:
        return int(PD10_SPLIT_SIZES[split])
    return int(len(view.labels))


def pd10_teacher_logit_selection_seed(config: PD10TeacherLogitCacheConfig, split: str) -> int:
    return int(config.control_seed) + 1009 * (PD10_TEACHER_LOGIT_SPLITS.index(split) + 1)


def _load_source_view(config: PD10TeacherLogitCacheConfig, split: str) -> tuple[JetView, dict[str, Any]]:
    if config.teacher_target == PD10_TEACHER_HLT:
        view = load_cached_hlt_view(config.hlt_cache_dir, split, verify_hash=bool(config.verify_hlt_hash))
        if view.metadata.get("view") not in (None, "fixed_hlt"):
            raise ValueError(f"Expected fixed_hlt cache for {split}, got {view.metadata.get('view')!r}")
        if view.metadata.get("hlt_params") != pd10_hlt_params_dict():
            raise ValueError(
                f"HLT cache for {split} does not match configured PD10 fixed-HLT params "
                f"(strength={PD10_HLT_DEGRADATION_STRENGTH:g})"
            )
        metadata = {
            "source_view": "fixed_hlt",
            "source_manifest_hash": view.metadata.get("source_manifest_hash"),
            "hlt_content_hash": view.metadata.get("hlt_content_hash"),
            "hlt_seed": view.metadata.get("seed"),
            "hlt_degradation_strength": float(PD10_HLT_DEGRADATION_STRENGTH),
            "expected_hlt_params": pd10_hlt_params_dict(),
            "no_offline_inputs_loaded": True,
            "offline_privileged_inputs_loaded": False,
        }
        return view, metadata

    manifest = load_split_manifest(config.manifest_path)
    if config.offline_cache_dir:
        view = load_cached_offline_view(config.offline_cache_dir, split, verify_hash=True)
        offline_source = "cached_offline"
    else:
        view = load_offline_view(
            manifest,
            split,
            data_dir=config.data_dir,
            verify_label_branches=config.verify_label_branches,
            read_chunk_size=config.read_chunk_size,
        )
        offline_source = "raw_offline"
    if view.metadata.get("view") not in (None, "offline"):
        raise ValueError(f"Expected offline view for {split}, got {view.metadata.get('view')!r}")
    metadata = {
        "source_view": "offline",
        "source_manifest_hash": manifest_hash(manifest),
        "hlt_content_hash": None,
        "no_hlt_inputs_loaded": True,
        "offline_privileged_inputs_loaded": True,
        "offline_source": offline_source,
        "offline_cache_dir": str(config.offline_cache_dir) if config.offline_cache_dir else None,
        "offline_content_hash": view.metadata.get("offline_content_hash"),
        "offline_jet_identity_hash": view.metadata.get("jet_identity_hash"),
    }
    return view, metadata


def collect_pd10_teacher_logits_for_view(
    model,
    view: JetView,
    *,
    teacher_target: str,
    split: str,
    model_name: str,
    batch_size: int,
    num_workers: int,
    device,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    torch = require_torch()
    target = normalize_pd10_part_teacher_target(teacher_target)
    if target == PD10_TEACHER_HLT:
        dataset = JetViewTorchDataset(view)
        source_view = "fixed_hlt"
    else:
        dataset = ParticleViewTorchDataset(view, expected_view="offline")
        source_view = "offline"
    loader = make_data_loader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        seed=int(seed),
        source_view=source_view,
    )
    logits_rows: list[np.ndarray] = []
    labels_rows: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        with _disable_model_amp_for_eval(model) as amp_disabled:
            for batch in loader:
                batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                logits = model(batch["points"], batch["features"], batch["lorentz_vectors"], batch["mask"])
                logits_rows.append(logits.detach().cpu().numpy().astype(np.float32))
                labels_rows.append(batch["labels"].detach().cpu().numpy().astype(np.int64))
    logits_np = (
        np.concatenate(logits_rows, axis=0).astype(np.float32, copy=False)
        if logits_rows
        else np.zeros((0, PD10_NUM_CLASSES), dtype=np.float32)
    )
    labels_np = (
        np.concatenate(labels_rows, axis=0).astype(np.int64, copy=False)
        if labels_rows
        else np.zeros((0,), dtype=np.int64)
    )
    if logits_np.ndim != 2 or int(logits_np.shape[1]) != PD10_NUM_CLASSES:
        raise ValueError(f"PD10 teacher logits must have shape [N, {PD10_NUM_CLASSES}], got {logits_np.shape}")
    if int(labels_np.shape[0]) != int(logits_np.shape[0]):
        raise ValueError("labels/logits row count mismatch")
    logits_np, sanitization = sanitize_prediction_logits(
        logits_np,
        model_name=model_name,
        split=split,
    )
    sanitization["amp_disabled_for_eval"] = bool(amp_disabled)
    return logits_np, labels_np, sanitization


def build_pd10_teacher_logit_block(
    config: PD10TeacherLogitCacheConfig,
    split: str,
    *,
    logits: np.ndarray,
    labels: np.ndarray,
    view: JetView,
    source_metadata: Mapping[str, Any],
    checkpoint_payload: Mapping[str, Any],
    logit_sanitization: Mapping[str, Any] | None = None,
) -> PredictionBlock:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    logits = np.asarray(logits, dtype=np.float32)
    if len(view.jet_ids) != int(labels.shape[0]):
        raise ValueError("jet_ids length does not match labels/logits")
    identity_labels = np.asarray([int(identity.label) for identity in view.jet_ids], dtype=np.int64)
    if not np.array_equal(labels, identity_labels):
        raise ValueError("labels and jet ids are not aligned")
    metadata = {
        "contract": PD10_TEACHER_LOGIT_CACHE_CONTRACT,
        "experiment_name": PD10_EXPERIMENT_NAME,
        "experiment_step": PD10_STEP4_EXPERIMENT_STEP,
        "teacher_target": config.teacher_target,
        "model_name": config.model_name,
        "model_kind": "pd10_part_teacher_logits",
        "architecture": "part",
        "source_view": config.source_view,
        "allowed_inputs": PD10_TEACHER_ALLOWED_INPUTS[config.teacher_target],
        "split": split,
        "split_expected_size": _expected_size_for_split(split, view),
        "max_jets": _max_jets_for_split(config, split),
        "checkpoint": str(config.checkpoint),
        "checkpoint_sha256": sha256_file(config.checkpoint),
        "checkpoint_epoch": checkpoint_payload.get("epoch"),
        "checkpoint_experiment_step": checkpoint_payload.get("experiment_step"),
        "label_names": list(LABEL_NAMES),
        "num_classes": PD10_NUM_CLASSES,
        "student_deployment_inputs": "HLT_only",
        "teacher_logits_train_time_only": True,
        "logit_sanitization": dict(logit_sanitization or {}),
        **dict(source_metadata),
    }
    return PredictionBlock(
        model_name=config.model_name,
        split=split,
        logits=logits,
        probs=softmax_np(logits),
        labels=labels,
        jet_ids=list(view.jet_ids),
        metadata=metadata,
    )


def validate_pd10_teacher_logit_metadata(
    metadata: Mapping[str, Any],
    *,
    teacher_target: str,
    split: str | None = None,
) -> None:
    target = normalize_pd10_part_teacher_target(teacher_target)
    model_name = pd10_part_teacher_model_name(target)
    if metadata.get("contract") != PD10_TEACHER_LOGIT_CACHE_CONTRACT:
        raise ValueError("PD10 teacher logit cache contract mismatch")
    if metadata.get("experiment_step") != PD10_STEP4_EXPERIMENT_STEP:
        raise ValueError("PD10 teacher logit cache step mismatch")
    if metadata.get("teacher_target") != target:
        raise ValueError(f"teacher_target mismatch: {metadata.get('teacher_target')} != {target}")
    if metadata.get("model_name") != model_name:
        raise ValueError(f"model_name mismatch: {metadata.get('model_name')} != {model_name}")
    if split is not None and metadata.get("split") != split:
        raise ValueError(f"split mismatch: {metadata.get('split')} != {split}")
    if int(metadata.get("num_classes", -1)) != PD10_NUM_CLASSES:
        raise ValueError("PD10 teacher logit cache must contain 10 classes")
    if int(metadata.get("n_jets", -1)) <= 0:
        raise ValueError("PD10 teacher logit cache contains no rows")
    if metadata.get("allowed_inputs") != PD10_TEACHER_ALLOWED_INPUTS[target]:
        raise ValueError(
            f"allowed_inputs mismatch: {metadata.get('allowed_inputs')} != {PD10_TEACHER_ALLOWED_INPUTS[target]}"
        )
    if target == PD10_TEACHER_HLT:
        if metadata.get("source_view") != "fixed_hlt":
            raise ValueError("HLT teacher logit cache must use fixed_hlt source_view")
        if not bool(metadata.get("no_offline_inputs_loaded")):
            raise ValueError("HLT teacher logit cache must declare no_offline_inputs_loaded")
        if metadata.get("hlt_content_hash") is None:
            raise ValueError("HLT teacher logit cache is missing hlt_content_hash")
    if target == PD10_TEACHER_OFFLINE:
        if metadata.get("source_view") != "offline":
            raise ValueError("offline teacher logit cache must use offline source_view")
        if not bool(metadata.get("no_hlt_inputs_loaded")):
            raise ValueError("offline teacher logit cache must declare no_hlt_inputs_loaded")


def load_pd10_teacher_logit_block(
    output_dir: str | Path,
    teacher_target: str,
    split: str,
    *,
    verify_hash: bool = True,
) -> PredictionBlock:
    block = load_prediction_block(
        output_dir,
        pd10_part_teacher_model_name(teacher_target),
        split,
        verify_hash=verify_hash,
    )
    validate_pd10_teacher_logit_metadata(block.metadata, teacher_target=teacher_target, split=split)
    if block.logits.ndim != 2 or int(block.logits.shape[1]) != PD10_NUM_CLASSES:
        raise ValueError(f"PD10 teacher logits must have shape [N, {PD10_NUM_CLASSES}], got {block.logits.shape}")
    return block


def _existing_metadata_if_valid(config: PD10TeacherLogitCacheConfig, split: str) -> dict[str, Any] | None:
    npz_path, metadata_path = prediction_paths(config.output_dir, config.model_name, split)
    if not npz_path.exists() or not metadata_path.exists():
        return None
    if not config.skip_existing or config.overwrite:
        return None
    block = load_pd10_teacher_logit_block(config.output_dir, config.teacher_target, split)
    return dict(block.metadata)


def cache_pd10_teacher_logits(config: PD10TeacherLogitCacheConfig) -> dict[str, Any]:
    torch = require_torch()
    device = resolve_device(config.device)
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    model, payload = load_pd10_part_teacher_model_from_checkpoint(
        config.checkpoint,
        device=device,
        fallback_model_size="base",
    )

    split_reports: dict[str, Any] = {}
    prediction_rows: list[dict[str, Any]] = []
    for split in config.splits:
        existing = _existing_metadata_if_valid(config, split)
        if existing is not None:
            split_reports[split] = {"skipped_existing": True, "metadata": existing}
            prediction_rows.append(existing)
            continue
        view, source_metadata = _load_source_view(config, split)
        view, selection_report = balanced_limit_jet_view(
            view,
            _max_jets_for_split(config, split),
            seed=pd10_teacher_logit_selection_seed(config, split),
        )
        source_metadata = {**source_metadata, "subset_selection": selection_report}
        logits, labels, logit_sanitization = collect_pd10_teacher_logits_for_view(
            model,
            view,
            teacher_target=config.teacher_target,
            split=split,
            model_name=config.model_name,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            device=device,
            seed=pd10_teacher_logit_selection_seed(config, split),
        )
        block = build_pd10_teacher_logit_block(
            config,
            split,
            logits=logits,
            labels=labels,
            view=view,
            source_metadata=source_metadata,
            checkpoint_payload=payload,
            logit_sanitization=logit_sanitization,
        )
        metadata = save_prediction_block(block, config.output_dir, overwrite=bool(config.overwrite))
        validate_pd10_teacher_logit_metadata(metadata, teacher_target=config.teacher_target, split=split)
        split_reports[split] = {"skipped_existing": False, "metadata": metadata}
        prediction_rows.append(metadata)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    manifest = {
        "ok": True,
        "contract": PD10_TEACHER_LOGIT_CACHE_CONTRACT,
        "experiment_name": PD10_EXPERIMENT_NAME,
        "experiment_step": PD10_STEP4_EXPERIMENT_STEP,
        "teacher_target": config.teacher_target,
        "model_name": config.model_name,
        "allowed_inputs": PD10_TEACHER_ALLOWED_INPUTS[config.teacher_target],
        "checkpoint": str(config.checkpoint),
        "checkpoint_sha256": sha256_file(config.checkpoint),
        "output_dir": str(config.output_dir),
        "teacher_logit_dir": str(config.teacher_output_dir),
        "splits": list(config.splits),
        "split_sizes": {
            str(row.get("split")): int(row.get("split_expected_size", row.get("n_jets", 0)))
            for row in prediction_rows
        },
        "prediction_rows": prediction_rows,
        "config": asdict(config),
    }
    save_json(config.teacher_output_dir / PD10_TEACHER_LOGIT_CACHE_MANIFEST, manifest)
    save_json(config.teacher_output_dir / PD10_TEACHER_LOGIT_CACHE_REPORT, manifest)
    return manifest


__all__ = [
    "PD10_STEP4_EXPERIMENT_STEP",
    "PD10_TEACHER_LOGIT_CACHE_CONTRACT",
    "PD10_TEACHER_LOGIT_CACHE_MANIFEST",
    "PD10_TEACHER_LOGIT_CACHE_REPORT",
    "PD10_TEACHER_LOGIT_SPLITS",
    "PD10TeacherLogitCacheConfig",
    "build_pd10_teacher_logit_block",
    "cache_pd10_teacher_logits",
    "collect_pd10_teacher_logits_for_view",
    "load_pd10_teacher_logit_block",
    "pd10_teacher_logit_cache_dir",
    "pd10_teacher_logit_prediction_paths",
    "pd10_teacher_logit_selection_seed",
    "validate_pd10_teacher_logit_metadata",
]
