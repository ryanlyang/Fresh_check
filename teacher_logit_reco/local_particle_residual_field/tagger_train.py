"""Training loop for augmented local residual-field Particle Transformer taggers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import csv
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from jetclass_fresh.hlt_baseline import (
    amp_autocast_context,
    amp_grad_scaler,
    require_torch,
    resolve_device,
    save_json,
    set_training_seed,
)
from jetclass_fresh.jetclass_data import LABEL_NAMES
from teacher_logit_reco.set_matching.five_view_train import classification_metrics_from_predictions

from .data import (
    LocalParticleResidualFieldDataset,
    LocalParticleResidualFieldDatasetConfig,
    LocalParticleResidualFieldHLTOnlyDataset,
    load_local_particle_residual_field_hlt_only_dataset,
    load_local_particle_residual_field_dataset,
    make_local_particle_residual_field_loader,
    move_local_particle_residual_field_batch_to_device,
)
from .model import LocalResidualFieldReconstructorConfig
from .tagger import (
    LOCAL_RESIDUAL_FIELD_TAGGER_CONTRACT,
    RESIDUAL_FIELD_SOURCE_FROZEN_RECONSTRUCTOR,
    RESIDUAL_FIELD_SOURCE_HLT_ONLY,
    RESIDUAL_FIELD_SOURCE_JOINT_RECONSTRUCTOR,
    RESIDUAL_FIELD_SOURCE_CROSS_JET_SHUFFLE,
    RESIDUAL_FIELD_SOURCE_LEARNED_NO_TARGET,
    RESIDUAL_FIELD_SOURCE_ORACLE,
    RESIDUAL_FIELD_SOURCES,
    LocalResidualFieldControlConfig,
    LocalResidualFieldAugmentedParT,
    LocalResidualFieldTaggerConfig,
    load_local_residual_reconstructor_from_checkpoint,
    normalize_residual_field_source,
    warm_start_local_residual_field_tagger_part,
)
from .train import (
    _jsonable,
    _torch_load_checkpoint,
    compute_local_residual_reconstruction_loss,
    resolve_local_residual_field_indices,
)


LOCAL_RESIDUAL_FIELD_TAGGER_TRAIN_CONTRACT = "local_particle_residual_field_augmented_part_train_v1"
LOCAL_RESIDUAL_FIELD_TAGGER_SELECTION_METRICS = (
    "accuracy",
    "loss",
    "cross_entropy",
    "macro_per_class_accuracy",
)
LOWER_IS_BETTER_TAGGER_METRICS = {"loss", "cross_entropy"}


@dataclass
class LocalResidualFieldTaggerTrainConfig:
    """Configuration for Step 5 augmented residual-field ParT training."""

    output_dir: str
    hlt_cache_dir: str
    target_cache_dir: str
    manifest_path: str | None = None
    train_split: str = "model_train"
    val_split: str = "model_val"
    stack_val_split: str = "stack_val"
    seed: int = 20421
    batch_size: int = 64
    eval_batch_size: int = 128
    epochs: int = 45
    part_lr: float = 3.0e-5
    reconstructor_lr: float = 3.0e-4
    weight_decay: float = 1.0e-4
    num_workers: int = 0
    device: str = "auto"
    amp: bool = True
    grad_clip_norm: float = 1.0
    early_stop_patience: int = 6
    max_train_jets: int | None = None
    max_val_jets: int | None = None
    max_stack_val_jets: int | None = None
    num_classes: int = len(LABEL_NAMES)
    label_names: tuple[str, ...] = tuple(LABEL_NAMES)
    model_size: str = "base"
    field_source: str = RESIDUAL_FIELD_SOURCE_FROZEN_RECONSTRUCTOR
    reconstructor_checkpoint: str | None = None
    baseline_checkpoint: str | None = None
    require_baseline_warm_start: bool = False
    residual_field_scale: float = 1.0
    field_dropout: float = 0.0
    control_seed: int = 9173
    control_noise_scale: float = 1.0
    control_random_match_target_std: bool = True
    learned_control_hidden_dim: int = 128
    learned_control_dropout: float = 0.05
    field_subset: tuple[str, ...] = ()
    reconstructor_loss_weight: float = 0.0
    reconstructor_field_group_weights: Mapping[str, float] = field(default_factory=dict)
    reconstructor_huber_beta: float = 0.1
    reconstructor_uncertainty_loss_weight: float = 1.0
    reconstructor_consistency_loss_weight: float = 0.0
    kd_loss_weight: float = 0.0
    kd_temperature: float = 2.0
    teacher_logits_dir: str | None = None
    teacher_logits_train_path: str | None = None
    teacher_logits_val_path: str | None = None
    teacher_logits_stack_val_path: str | None = None
    selection_metric: str = "accuracy"
    min_selection_valid_fraction: float = 0.99
    verify_hash: bool = True
    require_manifest_match: bool = True
    save_last_checkpoint: bool = True

    def __post_init__(self) -> None:
        self.output_dir = str(self.output_dir)
        self.hlt_cache_dir = str(self.hlt_cache_dir)
        self.target_cache_dir = str(self.target_cache_dir)
        self.manifest_path = None if not self.manifest_path else str(self.manifest_path)
        self.field_source = normalize_residual_field_source(self.field_source)
        if self.field_source not in RESIDUAL_FIELD_SOURCES:
            raise ValueError(f"unknown field_source {self.field_source!r}")
        for name in ("batch_size", "eval_batch_size", "epochs", "num_classes"):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            setattr(self, name, value)
        if int(self.num_workers) < 0:
            raise ValueError("num_workers cannot be negative")
        if int(self.early_stop_patience) < -1:
            raise ValueError("early_stop_patience must be -1 or greater")
        for name in ("part_lr", "reconstructor_lr", "kd_temperature"):
            value = float(getattr(self, name))
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
            setattr(self, name, value)
        for name in (
            "weight_decay",
            "grad_clip_norm",
            "reconstructor_loss_weight",
            "kd_loss_weight",
            "residual_field_scale",
            "reconstructor_uncertainty_loss_weight",
            "reconstructor_consistency_loss_weight",
        ):
            value = float(getattr(self, name))
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            setattr(self, name, value)
        if float(self.reconstructor_huber_beta) <= 0.0:
            raise ValueError("reconstructor_huber_beta must be positive")
        dropout = float(self.field_dropout)
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("field_dropout must be in [0, 1)")
        self.field_dropout = dropout
        self.control_seed = int(self.control_seed)
        self.control_noise_scale = float(self.control_noise_scale)
        if self.control_noise_scale < 0.0:
            raise ValueError("control_noise_scale must be non-negative")
        self.control_random_match_target_std = bool(self.control_random_match_target_std)
        self.learned_control_hidden_dim = int(self.learned_control_hidden_dim)
        if self.learned_control_hidden_dim <= 0:
            raise ValueError("learned_control_hidden_dim must be positive")
        self.learned_control_dropout = float(self.learned_control_dropout)
        if self.learned_control_dropout < 0.0 or self.learned_control_dropout >= 1.0:
            raise ValueError("learned_control_dropout must be in [0, 1)")
        self.field_subset = tuple(str(value).strip() for value in self.field_subset if str(value).strip())
        self.reconstructor_field_group_weights = {
            str(key): float(value)
            for key, value in dict(self.reconstructor_field_group_weights or {}).items()
        }
        self.label_names = tuple(str(name) for name in self.label_names)
        if len(self.label_names) != int(self.num_classes):
            raise ValueError("label_names length must match num_classes")
        self.selection_metric = str(self.selection_metric)
        if self.selection_metric not in LOCAL_RESIDUAL_FIELD_TAGGER_SELECTION_METRICS:
            raise ValueError(f"selection_metric must be one of {LOCAL_RESIDUAL_FIELD_TAGGER_SELECTION_METRICS}")
        self.min_selection_valid_fraction = float(self.min_selection_valid_fraction)
        if self.min_selection_valid_fraction <= 0.0 or self.min_selection_valid_fraction > 1.0:
            raise ValueError("min_selection_valid_fraction must be in (0, 1]")
        self.max_train_jets = _optional_positive_int(self.max_train_jets, field_name="max_train_jets")
        self.max_val_jets = _optional_positive_int(self.max_val_jets, field_name="max_val_jets")
        self.max_stack_val_jets = _optional_positive_int(self.max_stack_val_jets, field_name="max_stack_val_jets")
        self.baseline_checkpoint = None if not self.baseline_checkpoint else str(self.baseline_checkpoint)
        self.reconstructor_checkpoint = None if not self.reconstructor_checkpoint else str(self.reconstructor_checkpoint)
        self.teacher_logits_dir = None if not self.teacher_logits_dir else str(self.teacher_logits_dir)
        self.teacher_logits_train_path = None if not self.teacher_logits_train_path else str(self.teacher_logits_train_path)
        self.teacher_logits_val_path = None if not self.teacher_logits_val_path else str(self.teacher_logits_val_path)
        self.teacher_logits_stack_val_path = None if not self.teacher_logits_stack_val_path else str(self.teacher_logits_stack_val_path)
        self.verify_hash = bool(self.verify_hash)
        self.require_manifest_match = bool(self.require_manifest_match)
        self.save_last_checkpoint = bool(self.save_last_checkpoint)


def _optional_positive_int(value: int | None, *, field_name: str) -> int | None:
    if value is None:
        return None
    output = int(value)
    if output <= 0:
        raise ValueError(f"{field_name} must be positive when provided")
    return output


def _teacher_logits_path(config: LocalResidualFieldTaggerTrainConfig, split: str) -> str | None:
    direct = {
        config.train_split: config.teacher_logits_train_path,
        config.val_split: config.teacher_logits_val_path,
        config.stack_val_split: config.teacher_logits_stack_val_path,
    }.get(str(split))
    if direct:
        return str(direct)
    if not config.teacher_logits_dir:
        return None
    root = Path(config.teacher_logits_dir)
    candidate_roots = (root, root / "offline_part_teacher_10class")
    for name in (
        f"{split}_teacher_logits.npz",
        f"{split}_logits.npz",
        f"{split}.npz",
        f"{split}_predictions.npz",
    ):
        for candidate_root in candidate_roots:
            candidate = candidate_root / name
            if candidate.exists():
                return str(candidate)
    return None


def _required_teacher_logits_splits(config: LocalResidualFieldTaggerTrainConfig) -> tuple[str, ...]:
    ordered = (str(config.train_split), str(config.val_split), str(config.stack_val_split))
    output: list[str] = []
    seen: set[str] = set()
    for split in ordered:
        if split not in seen:
            seen.add(split)
            output.append(split)
    return tuple(output)


def _validate_required_teacher_logits(config: LocalResidualFieldTaggerTrainConfig) -> dict[str, str]:
    """Fail closed when KD is requested but teacher logits are incomplete."""

    if float(config.kd_loss_weight) <= 0.0:
        return {}
    paths: dict[str, str] = {}
    missing: list[str] = []
    for split in _required_teacher_logits_splits(config):
        path = _teacher_logits_path(config, split)
        if not path or not Path(path).exists():
            missing.append(split)
            continue
        paths[split] = str(path)
    if missing:
        raise FileNotFoundError(
            "KD was requested for local residual-field tagger training "
            f"(kd_loss_weight={float(config.kd_loss_weight):g}), but teacher logits are missing for "
            f"splits {missing}. Provide LOCAL_RESIDUAL_FIELD_TEACHER_LOGITS_DIR with "
            f"{config.train_split}/{config.val_split}/{config.stack_val_split} logits, or set kd_loss_weight=0."
        )
    return paths


def _load_tagger_dataset(
    config: LocalResidualFieldTaggerTrainConfig,
    split: str,
    *,
    max_jets: int | None,
) -> LocalParticleResidualFieldDataset | LocalParticleResidualFieldHLTOnlyDataset:
    dataset_config = LocalParticleResidualFieldDatasetConfig(
        hlt_cache_dir=config.hlt_cache_dir,
        target_cache_dir=config.target_cache_dir,
        split=str(split),
        manifest_path=config.manifest_path,
        max_jets=max_jets,
        teacher_logits_path=_teacher_logits_path(config, split),
        include_oracle_fields=config.field_source == RESIDUAL_FIELD_SOURCE_ORACLE,
        verify_hash=bool(config.verify_hash),
        require_manifest_match=bool(config.require_manifest_match),
        require_teacher_logits_metadata=_teacher_logits_path(config, split) is not None,
    )
    if config.field_source == RESIDUAL_FIELD_SOURCE_HLT_ONLY:
        return load_local_particle_residual_field_hlt_only_dataset(dataset_config)
    return load_local_particle_residual_field_dataset(
        dataset_config
    )


def _selection_score(metrics: Mapping[str, Any], metric_name: str) -> tuple[float, float]:
    value = metrics.get(metric_name)
    if value is None and metric_name == "cross_entropy":
        value = metrics.get("loss")
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float("-inf"), float("nan")
    if not np.isfinite(numeric):
        return float("-inf"), numeric
    if metric_name in LOWER_IS_BETTER_TAGGER_METRICS:
        return -numeric, numeric
    return numeric, numeric


def _metrics_valid_for_selection(
    metrics: Mapping[str, Any],
    *,
    expected_n_jets: int,
    min_valid_fraction: float,
) -> tuple[bool, str]:
    expected = int(expected_n_jets)
    if expected <= 0:
        return False, "expected validation jet count is zero"
    try:
        seen = int(metrics.get("n_jets", 0) or 0)
    except (TypeError, ValueError):
        seen = 0
    min_seen = int(math.ceil(float(expected) * float(min_valid_fraction)))
    if seen < min_seen:
        return (
            False,
            f"finite validation coverage {seen}/{expected} below required {min_seen} "
            f"({float(min_valid_fraction):.4f})",
        )
    try:
        loss = float(metrics.get("loss"))
    except (TypeError, ValueError):
        return False, "loss is missing or not numeric"
    if not np.isfinite(loss):
        return False, "loss is not finite"
    return True, ""


def _write_epoch_metrics_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flattened: list[dict[str, Any]] = []
    for row in rows:
        payload: dict[str, Any] = {"epoch": row.get("epoch")}
        split_keys = (
            ("train", "train"),
            ("model_val", "model_val"),
            ("stack_val", "stack_val"),
            ("best_model_val_reloaded_stack_val", "best_model_val_reloaded_stack_val"),
        )
        for split_key, split_prefix in split_keys:
            metrics = row.get(split_key)
            if not isinstance(metrics, Mapping):
                continue
            for key in (
                "loss",
                "cross_entropy",
                "kd_loss",
                "reconstructor_loss",
                "accuracy",
                "macro_per_class_accuracy",
                "n_jets",
                "attempted_jets",
                "valid_fraction",
                "total_batches",
                "finite_batches",
                "nonfinite_batches",
                "nonfinite_grad_batches",
                "nonfinite_fraction",
                "valid_for_selection",
                "selection_valid_fraction_required",
                "selection_expected_n_jets",
                "selection_rejection_reason",
            ):
                if key in metrics:
                    payload[f"{split_prefix}_{key}"] = metrics.get(key)
        flattened.append(payload)
    fieldnames: list[str] = []
    for row in flattened:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["epoch"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flattened)


def _remap_field_groups(
    field_groups: Mapping[str, Sequence[int]],
    selected_indices: Sequence[int],
) -> dict[str, tuple[int, ...]]:
    old_to_new = {int(old): int(new) for new, old in enumerate(selected_indices)}
    remapped: dict[str, tuple[int, ...]] = {}
    for group, indices in dict(field_groups).items():
        values = tuple(old_to_new[int(index)] for index in indices if int(index) in old_to_new)
        if values:
            remapped[str(group)] = values
    if not remapped:
        remapped["all"] = tuple(range(len(tuple(selected_indices))))
    return remapped


def _subset_batch_fields(batch: Mapping[str, Any], selected_indices: Sequence[int]) -> dict[str, Any]:
    output = dict(batch)
    torch = require_torch()
    if "target_fields" not in batch:
        if tuple(int(index) for index in selected_indices):
            raise ValueError("cannot select residual fields from a batch without target_fields")
        return output
    indices = torch.as_tensor(tuple(int(index) for index in selected_indices), device=batch["target_fields"].device, dtype=torch.long)
    if int(indices.numel()) == 0:
        raise ValueError("selected_indices must not be empty")
    for key in ("target_fields", "oracle_fields"):
        value = output.get(key)
        if value is not None:
            output[key] = value.index_select(dim=-1, index=indices)
    for key in ("target_features", "oracle_features"):
        value = output.get(key)
        if value is not None:
            output[key] = value.index_select(dim=1, index=indices)
    return output


def _dataset_alignment_fields(metadata: Mapping[str, Any]) -> dict[str, Any]:
    alignment = metadata.get("alignment_report") if isinstance(metadata.get("alignment_report"), Mapping) else {}
    hlt_metadata = metadata.get("hlt_metadata") if isinstance(metadata.get("hlt_metadata"), Mapping) else {}
    target_metadata = metadata.get("target_metadata") if isinstance(metadata.get("target_metadata"), Mapping) else {}
    return {
        "source_manifest_hash": alignment.get("source_manifest_hash") or hlt_metadata.get("source_manifest_hash"),
        "hlt_content_hash": alignment.get("hlt_content_hash") or hlt_metadata.get("hlt_content_hash"),
        "offline_content_hash": alignment.get("offline_content_hash") or target_metadata.get("offline_content_hash"),
        "target_content_hash": alignment.get("target_content_hash") or target_metadata.get("target_content_hash"),
        "jet_identity_hash": alignment.get("jet_identity_hash") or hlt_metadata.get("jet_identity_hash"),
    }


def _validate_reconstructor_checkpoint_payload(
    payload: Mapping[str, Any] | None,
    *,
    current_dataset_metadata: Mapping[str, Any],
    full_field_names: Sequence[str],
    selected_field_names: Sequence[str],
) -> None:
    if payload is None:
        return
    model_config = payload.get("model_config") if isinstance(payload, Mapping) else None
    if isinstance(model_config, Mapping):
        checkpoint_field_names = tuple(str(name) for name in model_config.get("field_names", ()))
        if checkpoint_field_names and checkpoint_field_names not in {
            tuple(str(name) for name in full_field_names),
            tuple(str(name) for name in selected_field_names),
        }:
            raise ValueError("reconstructor checkpoint field_names do not match current target field ordering")
    checkpoint_metadata = payload.get("dataset_metadata") if isinstance(payload, Mapping) else None
    if not isinstance(checkpoint_metadata, Mapping):
        raise ValueError("reconstructor checkpoint is missing dataset_metadata for provenance validation")
    for split in ("model_train", "model_val", "stack_val"):
        current = current_dataset_metadata.get(split)
        checkpoint = checkpoint_metadata.get(split)
        if not isinstance(current, Mapping) or not isinstance(checkpoint, Mapping):
            raise ValueError(f"reconstructor checkpoint missing dataset metadata for split {split}")
        current_fields = _dataset_alignment_fields(current)
        checkpoint_fields = _dataset_alignment_fields(checkpoint)
        for key, current_value in current_fields.items():
            checkpoint_value = checkpoint_fields.get(key)
            if current_value in (None, ""):
                continue
            if checkpoint_value in (None, ""):
                raise ValueError(f"reconstructor checkpoint is missing provenance for {split}.{key}")
            if str(current_value) != str(checkpoint_value):
                raise ValueError(
                    f"reconstructor checkpoint provenance mismatch for {split}.{key}: "
                    f"{checkpoint_value} != {current_value}"
                )


def _build_model(
    config: LocalResidualFieldTaggerTrainConfig,
    *,
    full_field_names: Sequence[str],
    selected_indices: Sequence[int],
    selected_field_names: Sequence[str],
    selected_field_groups: Mapping[str, Sequence[int]],
    device: Any,
) -> tuple[LocalResidualFieldAugmentedParT, Mapping[str, Any] | None]:
    reconstructor = None
    reconstructor_payload: Mapping[str, Any] | None = None
    reconstructor_config: LocalResidualFieldReconstructorConfig | None = None
    if config.reconstructor_checkpoint:
        reconstructor, reconstructor_payload = load_local_residual_reconstructor_from_checkpoint(
            config.reconstructor_checkpoint,
            map_location=device,
        )
        reconstructor_config = getattr(reconstructor, "config", None)
    elif config.field_source in {
        RESIDUAL_FIELD_SOURCE_FROZEN_RECONSTRUCTOR,
        RESIDUAL_FIELD_SOURCE_JOINT_RECONSTRUCTOR,
    }:
        reconstructor_config = LocalResidualFieldReconstructorConfig(
            field_dim=len(selected_field_names),
            field_names=tuple(selected_field_names),
            field_groups=selected_field_groups,
        )
    source_field_indices: tuple[int, ...] = ()
    if reconstructor_config is not None:
        recon_field_dim = int(getattr(reconstructor_config, "field_dim", len(selected_field_names)))
        if recon_field_dim != len(selected_field_names):
            source_field_indices = tuple(int(index) for index in selected_indices)
            if source_field_indices and max(source_field_indices) >= recon_field_dim:
                raise ValueError(
                    f"selected field index {max(source_field_indices)} is outside reconstructor output dim {recon_field_dim}"
                )
    tagger_config = LocalResidualFieldTaggerConfig(
        num_classes=int(config.num_classes),
        field_dim=len(selected_field_names),
        field_source=str(config.field_source),
        model_size=str(config.model_size),
        residual_field_scale=float(config.residual_field_scale),
        field_dropout=float(config.field_dropout),
        field_names=tuple(selected_field_names),
        field_groups=selected_field_groups,
        source_field_indices=source_field_indices,
        reconstructor_config=reconstructor_config,
        control_config=LocalResidualFieldControlConfig(
            seed=int(config.control_seed),
            noise_scale=float(config.control_noise_scale),
            random_match_target_std=bool(config.control_random_match_target_std),
            learned_hidden_dim=int(config.learned_control_hidden_dim),
            learned_dropout=float(config.learned_control_dropout),
            field_names=tuple(selected_field_names),
        ),
    )
    model = LocalResidualFieldAugmentedParT(tagger_config, reconstructor=reconstructor).to(device)
    return model, reconstructor_payload


def _optimizer(model: LocalResidualFieldAugmentedParT, config: LocalResidualFieldTaggerTrainConfig):
    torch = require_torch()
    groups: list[dict[str, Any]] = []
    part_params = [param for param in model.part_model.parameters() if param.requires_grad]
    if part_params:
        groups.append({"params": part_params, "lr": float(config.part_lr), "name": "part"})
    if model.reconstructor is not None and config.field_source == RESIDUAL_FIELD_SOURCE_JOINT_RECONSTRUCTOR:
        reco_params = [param for param in model.reconstructor.parameters() if param.requires_grad]
        if reco_params:
            groups.append({"params": reco_params, "lr": float(config.reconstructor_lr), "name": "reconstructor"})
    if model.control_generator is not None and config.field_source == RESIDUAL_FIELD_SOURCE_LEARNED_NO_TARGET:
        control_params = [param for param in model.control_generator.parameters() if param.requires_grad]
        if control_params:
            groups.append({"params": control_params, "lr": float(config.reconstructor_lr), "name": "learned_no_target_control"})
    if not groups:
        raise ValueError("no trainable parameters are active for this tagger run")
    return torch.optim.AdamW(groups, weight_decay=float(config.weight_decay))


def _kd_loss(logits: Any, teacher_logits: Any, *, temperature: float) -> Any:
    torch = require_torch()
    temp = float(temperature)
    return torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(logits / temp, dim=-1),
        torch.nn.functional.softmax(teacher_logits.to(device=logits.device, dtype=logits.dtype) / temp, dim=-1),
        reduction="batchmean",
    ) * (temp * temp)


def _model_gradients_are_finite(model: LocalResidualFieldAugmentedParT) -> bool:
    torch = require_torch()
    status = None
    for param in model.parameters():
        grad = param.grad
        if grad is None:
            continue
        finite = torch.isfinite(grad).all()
        status = finite if status is None else status & finite
    if status is None:
        return True
    return bool(status.detach().cpu().item())


def _run_epoch(
    model: LocalResidualFieldAugmentedParT,
    loader: Any,
    *,
    device: Any,
    optimizer: Any | None,
    scaler: Any | None,
    amp_enabled: bool,
    grad_clip_norm: float,
    config: LocalResidualFieldTaggerTrainConfig,
    field_names: Sequence[str],
    field_groups: Mapping[str, Sequence[int]],
    selected_indices: Sequence[int],
) -> dict[str, Any]:
    torch = require_torch()
    training = optimizer is not None
    model.train(training)
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    logits_chunks: list[np.ndarray] = []
    labels_chunks: list[np.ndarray] = []
    loss_sum = 0.0
    ce_sum = 0.0
    kd_sum = 0.0
    reco_sum = 0.0
    seen = 0
    attempted_jets = 0
    total_batches = 0
    finite_batches = 0
    nonfinite_batches = 0
    nonfinite_grad_batches = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            batch = move_local_particle_residual_field_batch_to_device(batch, device)
            batch = _subset_batch_fields(batch, selected_indices)
            batch_size = int(batch["labels"].numel())
            attempted_jets += batch_size
            total_batches += 1
            if training:
                optimizer.zero_grad(set_to_none=True)
            with amp_autocast_context(bool(amp_enabled)):
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
                logits = output.logits
                ce_loss = criterion(logits, batch["labels"])
                loss = ce_loss / float(max(batch_size, 1))
                kd_value = logits.new_zeros(())
                if float(config.kd_loss_weight) > 0.0:
                    if "teacher_logits" not in batch:
                        raise RuntimeError(
                            "KD was requested, but the batch does not contain teacher_logits. "
                            "Check teacher-logit cache paths and dataset metadata."
                        )
                    kd_value = _kd_loss(logits, batch["teacher_logits"], temperature=float(config.kd_temperature))
                    loss = loss + float(config.kd_loss_weight) * kd_value
                reco_value = logits.new_zeros(())
                if (
                    float(config.reconstructor_loss_weight) > 0.0
                    and output.reconstructor_output is not None
                    and config.field_source == RESIDUAL_FIELD_SOURCE_JOINT_RECONSTRUCTOR
                ):
                    reco_value, _ = compute_local_residual_reconstruction_loss(
                        output.reconstructor_output,
                        batch,
                        field_names=field_names,
                        field_groups=field_groups,
                        selected_indices=selected_indices,
                        field_group_weights=dict(config.reconstructor_field_group_weights),
                        huber_beta=float(config.reconstructor_huber_beta),
                        uncertainty_loss_weight=float(config.reconstructor_uncertainty_loss_weight),
                        consistency_loss_weight=float(config.reconstructor_consistency_loss_weight),
                    )
                    loss = loss + float(config.reconstructor_loss_weight) * reco_value
            if not bool(torch.isfinite(loss).detach().cpu().item()) or not bool(torch.isfinite(logits).all().detach().cpu().item()):
                nonfinite_batches += 1
                if training:
                    optimizer.zero_grad(set_to_none=True)
                continue
            if training:
                assert optimizer is not None
                if scaler is not None and bool(amp_enabled):
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if float(grad_clip_norm) > 0.0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
                        gradients_finite = bool(torch.isfinite(grad_norm).detach().cpu().item())
                    else:
                        gradients_finite = _model_gradients_are_finite(model)
                    if not gradients_finite:
                        nonfinite_grad_batches += 1
                        optimizer.zero_grad(set_to_none=True)
                        scaler.update()
                        continue
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if float(grad_clip_norm) > 0.0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
                        gradients_finite = bool(torch.isfinite(grad_norm).detach().cpu().item())
                    else:
                        gradients_finite = _model_gradients_are_finite(model)
                    if not gradients_finite:
                        nonfinite_grad_batches += 1
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    optimizer.step()
            labels = batch["labels"].detach().cpu().numpy().astype(np.int64)
            logits_np = logits.detach().float().cpu().numpy()
            logits_chunks.append(logits_np)
            labels_chunks.append(labels)
            batch_size = int(labels.shape[0])
            seen += batch_size
            finite_batches += 1
            ce_sum += float(ce_loss.detach().cpu().item())
            kd_sum += float(kd_value.detach().cpu().item()) * batch_size
            reco_sum += float(reco_value.detach().cpu().item()) * batch_size
            loss_sum += float(loss.detach().cpu().item()) * batch_size
    attempted = int(attempted_jets)
    total = int(total_batches)
    nonfinite_total = int(nonfinite_batches + nonfinite_grad_batches)
    valid_fraction = float(seen) / float(attempted) if attempted > 0 else 0.0
    nonfinite_fraction = float(nonfinite_total) / float(total) if total > 0 else 0.0
    base_counts = {
        "attempted_jets": attempted,
        "valid_fraction": valid_fraction,
        "total_batches": total,
        "finite_batches": int(finite_batches),
        "nonfinite_batches": int(nonfinite_batches),
        "nonfinite_grad_batches": int(nonfinite_grad_batches),
        "nonfinite_fraction": nonfinite_fraction,
    }
    if seen == 0:
        return {"n_jets": 0, "loss": float("nan"), "accuracy": 0.0, **base_counts}
    logits_all = np.concatenate(logits_chunks, axis=0)
    labels_all = np.concatenate(labels_chunks, axis=0)
    preds = np.argmax(logits_all, axis=1).astype(np.int64)
    metrics = classification_metrics_from_predictions(
        preds=preds,
        labels=labels_all,
        loss_sum=ce_sum,
        logits=logits_all if int(config.num_classes) == 2 else None,
        label_names=tuple(config.label_names),
    )
    metrics["loss"] = loss_sum / float(seen)
    metrics["cross_entropy"] = ce_sum / float(seen)
    metrics["kd_loss"] = kd_sum / float(seen)
    metrics["reconstructor_loss"] = reco_sum / float(seen)
    metrics.update(base_counts)
    return metrics


def _checkpoint_payload(
    *,
    model: LocalResidualFieldAugmentedParT,
    optimizer: Any | None,
    epoch: int,
    config: LocalResidualFieldTaggerTrainConfig,
    metrics: Mapping[str, Any],
    dataset_metadata: Mapping[str, Any],
    selected_indices: Sequence[int],
    warm_start_report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "checkpoint_contract": LOCAL_RESIDUAL_FIELD_TAGGER_TRAIN_CONTRACT,
        "model_contract": LOCAL_RESIDUAL_FIELD_TAGGER_CONTRACT,
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "config": _jsonable(asdict(config)),
        "model_config": model.config.to_dict(),
        "metrics": _jsonable(metrics),
        "dataset_metadata": _jsonable(dataset_metadata),
        "selected_field_indices": [int(index) for index in selected_indices],
        "warm_start_report": _jsonable(warm_start_report),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    return payload


def train_local_residual_field_tagger(config: LocalResidualFieldTaggerTrainConfig) -> dict[str, Any]:
    torch = require_torch()
    set_training_seed(int(config.seed))
    output_dir = Path(config.output_dir)
    diagnostics_dir = output_dir / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(str(config.device))
    amp_enabled = bool(config.amp and getattr(device, "type", str(device)) == "cuda")
    teacher_logits_paths = _validate_required_teacher_logits(config)

    train_dataset = _load_tagger_dataset(config, config.train_split, max_jets=config.max_train_jets)
    val_dataset = _load_tagger_dataset(config, config.val_split, max_jets=config.max_val_jets)
    stack_val_dataset = _load_tagger_dataset(config, config.stack_val_split, max_jets=config.max_stack_val_jets)
    field_names = tuple(train_dataset.field_names)
    field_groups = {
        str(group): tuple(int(index) for index in indices)
        for group, indices in dict(train_dataset.field_groups).items()
    }
    selected_indices = resolve_local_residual_field_indices(
        field_names=field_names,
        field_groups=field_groups,
        subset=tuple(config.field_subset),
    )
    selected_field_names = tuple(field_names[int(index)] for index in selected_indices)
    selected_field_groups = _remap_field_groups(field_groups, selected_indices)
    if tuple(val_dataset.field_names) != field_names or tuple(stack_val_dataset.field_names) != field_names:
        raise ValueError("field_names differ across tagger splits")
    dataset_metadata = {
        "model_train": train_dataset.metadata,
        "model_val": val_dataset.metadata,
        "stack_val": stack_val_dataset.metadata,
    }
    model, reconstructor_payload = _build_model(
        config,
        full_field_names=field_names,
        selected_indices=selected_indices,
        selected_field_names=selected_field_names,
        selected_field_groups=selected_field_groups,
        device=device,
    )
    _validate_reconstructor_checkpoint_payload(
        reconstructor_payload,
        current_dataset_metadata=dataset_metadata,
        full_field_names=field_names,
        selected_field_names=selected_field_names,
    )
    warm_start_report = None
    if config.baseline_checkpoint:
        warm_start_report = warm_start_local_residual_field_tagger_part(
            model,
            config.baseline_checkpoint,
            map_location=device,
            require=bool(config.require_baseline_warm_start),
        )
        save_json(diagnostics_dir / "warm_start_report.json", warm_start_report)
    optimizer = _optimizer(model, config)
    scaler = amp_grad_scaler(bool(amp_enabled))

    train_loader = make_local_particle_residual_field_loader(
        train_dataset,
        batch_size=int(config.batch_size),
        shuffle=True,
        num_workers=int(config.num_workers),
        seed=int(config.seed),
    )
    shuffle_eval_for_control = config.field_source == RESIDUAL_FIELD_SOURCE_CROSS_JET_SHUFFLE
    val_loader = make_local_particle_residual_field_loader(
        val_dataset,
        batch_size=int(config.eval_batch_size),
        shuffle=bool(shuffle_eval_for_control),
        num_workers=int(config.num_workers),
        seed=int(config.seed) + 1,
    )
    stack_val_loader = make_local_particle_residual_field_loader(
        stack_val_dataset,
        batch_size=int(config.eval_batch_size),
        shuffle=bool(shuffle_eval_for_control),
        num_workers=int(config.num_workers),
        seed=int(config.seed) + 2,
    )
    source_metadata = {
        "contract": LOCAL_RESIDUAL_FIELD_TAGGER_TRAIN_CONTRACT,
        "config": _jsonable(asdict(config)),
        "model_config": model.config.to_dict(),
        "dataset_metadata": _jsonable(dataset_metadata),
        "reconstructor_checkpoint_payload_present": bool(reconstructor_payload is not None),
        "warm_start_report": _jsonable(warm_start_report),
        "selected_field_indices": [int(index) for index in selected_indices],
        "selected_field_names": list(selected_field_names),
        "selected_field_groups": _jsonable(selected_field_groups),
        "teacher_logits_paths": dict(teacher_logits_paths),
    }
    save_json(output_dir / "source_metadata.json", source_metadata)

    curves: list[dict[str, Any]] = []
    best_epoch = -1
    best_score = float("-inf")
    best_metric_value = float("nan")
    best_metrics: Mapping[str, Any] = {}
    epochs_without_improvement = 0
    expected_val_n_jets = len(val_dataset)
    expected_stack_val_n_jets = len(stack_val_dataset)
    for epoch in range(int(config.epochs)):
        train_metrics = _run_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=bool(amp_enabled),
            grad_clip_norm=float(config.grad_clip_norm),
            config=config,
            field_names=selected_field_names,
            field_groups=selected_field_groups,
            selected_indices=tuple(range(len(selected_field_names))),
        )
        with torch.no_grad():
            val_metrics = _run_epoch(
                model,
                val_loader,
                device=device,
                optimizer=None,
                scaler=None,
                amp_enabled=bool(amp_enabled),
                grad_clip_norm=0.0,
                config=config,
                field_names=selected_field_names,
                field_groups=selected_field_groups,
                selected_indices=tuple(range(len(selected_field_names))),
            )
        val_ok, val_rejection_reason = _metrics_valid_for_selection(
            val_metrics,
            expected_n_jets=int(expected_val_n_jets),
            min_valid_fraction=float(config.min_selection_valid_fraction),
        )
        val_metrics = dict(val_metrics)
        val_metrics["valid_for_selection"] = bool(val_ok)
        val_metrics["selection_valid_fraction_required"] = float(config.min_selection_valid_fraction)
        val_metrics["selection_expected_n_jets"] = int(expected_val_n_jets)
        if val_rejection_reason:
            val_metrics["selection_rejection_reason"] = str(val_rejection_reason)
        row = {"epoch": int(epoch), "train": train_metrics, "model_val": val_metrics}
        curves.append(row)
        score, raw_value = _selection_score(val_metrics, str(config.selection_metric))
        if bool(val_ok) and np.isfinite(score) and score > best_score:
            best_epoch = int(epoch)
            best_score = float(score)
            best_metric_value = float(raw_value)
            best_metrics = val_metrics
            epochs_without_improvement = 0
            torch.save(
                _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    config=config,
                    metrics=val_metrics,
                    dataset_metadata=dataset_metadata,
                    selected_indices=tuple(range(len(selected_field_names))),
                    warm_start_report=warm_start_report,
                ),
                output_dir / "best_model_val.pt",
            )
        else:
            epochs_without_improvement += 1
        if bool(config.save_last_checkpoint):
            torch.save(
                _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    config=config,
                    metrics=val_metrics,
                    dataset_metadata=dataset_metadata,
                    selected_indices=tuple(range(len(selected_field_names))),
                    warm_start_report=warm_start_report,
                ),
                output_dir / "last.pt",
            )
        save_json(output_dir / "training_curves.json", {"epochs": curves, "selection_metric": config.selection_metric})
        _write_epoch_metrics_csv(diagnostics_dir / "epoch_metrics.csv", curves)
        if int(config.early_stop_patience) >= 0 and epochs_without_improvement > int(config.early_stop_patience):
            break

    if best_epoch < 0 or not (output_dir / "best_model_val.pt").exists():
        raise RuntimeError(
            "tagger training did not produce best_model_val.pt; no validation epoch met "
            f"the minimum finite-coverage requirement ({float(config.min_selection_valid_fraction):.4f})."
        )
    best_payload = _torch_load_checkpoint(output_dir / "best_model_val.pt", map_location=device)
    model.load_state_dict(best_payload["model_state_dict"])
    with torch.no_grad():
        stack_val_metrics = _run_epoch(
            model,
            stack_val_loader,
            device=device,
            optimizer=None,
            scaler=None,
            amp_enabled=bool(amp_enabled),
            grad_clip_norm=0.0,
            config=config,
            field_names=selected_field_names,
            field_groups=selected_field_groups,
            selected_indices=tuple(range(len(selected_field_names))),
        )
    stack_ok, stack_rejection_reason = _metrics_valid_for_selection(
        stack_val_metrics,
        expected_n_jets=int(expected_stack_val_n_jets),
        min_valid_fraction=float(config.min_selection_valid_fraction),
    )
    stack_val_metrics = dict(stack_val_metrics)
    stack_val_metrics["valid_for_selection"] = bool(stack_ok)
    stack_val_metrics["selection_valid_fraction_required"] = float(config.min_selection_valid_fraction)
    stack_val_metrics["selection_expected_n_jets"] = int(expected_stack_val_n_jets)
    if stack_rejection_reason:
        stack_val_metrics["selection_rejection_reason"] = str(stack_rejection_reason)
    if curves:
        curves[-1]["best_model_val_reloaded_stack_val"] = stack_val_metrics
        save_json(output_dir / "training_curves.json", {"epochs": curves, "selection_metric": config.selection_metric})
        _write_epoch_metrics_csv(diagnostics_dir / "epoch_metrics.csv", curves)
    if not bool(stack_ok):
        raise RuntimeError(
            "best_model_val checkpoint produced invalid stack_val coverage: "
            f"{stack_rejection_reason}"
        )
    report = {
        "ok": True,
        "contract": LOCAL_RESIDUAL_FIELD_TAGGER_TRAIN_CONTRACT,
        "model_contract": LOCAL_RESIDUAL_FIELD_TAGGER_CONTRACT,
        "output_dir": str(output_dir),
        "field_source": str(config.field_source),
        "best_epoch": int(best_epoch),
        "selection_metric": str(config.selection_metric),
        "best_model_selection_metric_value": float(best_metric_value),
        "best_model_val": _jsonable(best_metrics),
        "stack_val": _jsonable(stack_val_metrics),
        "checkpoint": str(output_dir / "best_model_val.pt"),
        "last_checkpoint": str(output_dir / "last.pt") if bool(config.save_last_checkpoint) else None,
        "training_curves": str(output_dir / "training_curves.json"),
        "diagnostic_csv": str(diagnostics_dir / "epoch_metrics.csv"),
        "source_metadata": str(output_dir / "source_metadata.json"),
        "warm_start_report": _jsonable(warm_start_report),
        "selected_field_indices": [int(index) for index in selected_indices],
        "selected_field_names": list(selected_field_names),
        "selected_field_groups": _jsonable(selected_field_groups),
        "teacher_logits_paths": dict(teacher_logits_paths),
        "model_config": model.config.to_dict(),
        "dataset_metadata": _jsonable(dataset_metadata),
    }
    save_json(output_dir / "run_report.json", report)
    return report


__all__ = [
    "LOCAL_RESIDUAL_FIELD_TAGGER_TRAIN_CONTRACT",
    "LOCAL_RESIDUAL_FIELD_TAGGER_SELECTION_METRICS",
    "LocalResidualFieldTaggerTrainConfig",
    "train_local_residual_field_tagger",
]
