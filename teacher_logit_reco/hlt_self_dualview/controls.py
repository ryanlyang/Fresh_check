"""Step 6 controls for deployable HLT self-dualview studies."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from jetclass_fresh.fusion import PredictionBlock, load_hlt_model_from_checkpoint, save_prediction_block, softmax_np
from jetclass_fresh.hlt_baseline import (
    ParticleViewTorchDataset,
    amp_grad_scaler,
    build_hlt_classifier,
    make_data_loader,
    require_torch,
    resolve_device,
    run_epoch,
    save_json,
    set_training_seed,
)
from jetclass_fresh.hlt_cache import jet_identity_hash, load_cached_hlt_view
from jetclass_fresh.jetclass_data import JetView, LABEL_NAMES
from jetclass_fresh.part_inputs import PF_FEATURE_NAMES

from teacher_logit_reco.privileged_distill_10class.config import (
    PD10_NUM_CLASSES,
    PD10_SPLIT_ORDER,
    PD10_SPLIT_SIZES,
)
from teacher_logit_reco.privileged_distill_10class.metrics import pd10_prediction_metrics_from_logits

from .config import (
    HLT_SDV_ALLOWED_INPUTS,
    HLT_SDV_DEPLOYMENT_INPUTS,
    HLT_SDV_EXPERIMENT_NAME,
    HLT_SDV_VARIANT_HLT2_ONLY,
    HLT_SDV_VARIANT_TTA,
    default_hlt_sdv_experiment_layout,
    hlt_sdv_strength_from_variant,
    hlt_sdv_strength_tag,
    normalize_hlt_sdv_variant,
)
from .data import load_hlt_sdv_dataset, make_hlt_sdv_data_loader, move_hlt_sdv_batch_to_device
from .hlt2_cache import default_hlt2_cache_dir
from .model import sha256_file, strip_compile_prefix_from_state_dict


HLT_SDV_STEP6_EXPERIMENT_STEP = "hlt_sdv_step6_tta_hlt2_controls"
HLT_SDV_HLT2_ONLY_CONTRACT = "hlt_self_dualview_hlt2_only_part_control_v1"
HLT_SDV_TTA_CONTRACT = "hlt_self_dualview_hlt_part_logit_average_control_v1"
HLT_SDV_HLT2_ONLY_MODEL_KIND = "hlt2_only_part"
HLT_SDV_TTA_MODEL_KIND = "hlt_part_hlt_plus_hlt2_logit_average"


def _require_finite_logits_array(logits: np.ndarray, *, context: str) -> None:
    finite = np.isfinite(logits)
    if bool(finite.all()):
        return
    bad_rows = np.where(~np.all(finite, axis=1))[0]
    first_bad = int(bad_rows[0]) if bad_rows.size else None
    finite_values = logits[finite]
    finite_min = float(np.min(finite_values)) if finite_values.size else None
    finite_max = float(np.max(finite_values)) if finite_values.size else None
    raise FloatingPointError(
        f"{context} produced non-finite logits: "
        f"shape={tuple(logits.shape)}, first_bad_row={first_bad}, "
        f"finite_min={finite_min}, finite_max={finite_max}"
    )


@dataclass(frozen=True)
class HLT2OnlyTrainConfig:
    """Training/eval config for the HLT2-only ParT control."""

    output_dir: str
    hlt2_cache_dir: str
    hlt_teacher_checkpoint: str
    variant_name: str = HLT_SDV_VARIANT_HLT2_ONLY
    train_split: str = "model_train"
    val_split: str = "model_val"
    final_test_split: str = "final_test"
    seed: int = 8811
    batch_size: int = 128
    eval_batch_size: int = 128
    epochs: int = 10
    lr: float = 3.0e-5
    weight_decay: float = 1.0e-4
    num_workers: int = 0
    device: str = "auto"
    amp: bool = True
    grad_clip_norm: float = 1.0
    early_stop_patience: int = 3
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    max_final_test_batches: int | None = None
    max_train_jets: int | None = PD10_SPLIT_SIZES["model_train"]
    max_val_jets: int | None = PD10_SPLIT_SIZES["model_val"]
    max_final_test_jets: int | None = PD10_SPLIT_SIZES["final_test"]
    model_size: str = "base"
    compile_model: bool = False
    initialize_from_hlt_checkpoint: bool = True
    evaluate_model_val_predictions: bool = True
    evaluate_final_test: bool = True
    confirm_final_test: bool = False
    overwrite: bool = False

    def __post_init__(self) -> None:
        variant = normalize_hlt_sdv_variant(self.variant_name)
        if (self.train_split, self.val_split, self.final_test_split) != PD10_SPLIT_ORDER:
            raise ValueError(f"HLT2-only split order must be {PD10_SPLIT_ORDER}")
        if not str(self.hlt2_cache_dir).strip():
            raise ValueError("hlt2_cache_dir is required for the HLT2-only control")
        if bool(self.initialize_from_hlt_checkpoint) and not str(self.hlt_teacher_checkpoint).strip():
            raise ValueError("hlt_teacher_checkpoint is required when warm-starting HLT2-only control")
        if self.evaluate_final_test and not bool(self.confirm_final_test):
            raise ValueError("HLT2-only final-test evaluation requires confirm_final_test=True")
        if int(self.batch_size) <= 0 or int(self.eval_batch_size) <= 0:
            raise ValueError("batch_size and eval_batch_size must be positive")
        if int(self.epochs) <= 0:
            raise ValueError("epochs must be positive")
        if float(self.lr) <= 0.0 or float(self.weight_decay) < 0.0:
            raise ValueError("lr must be positive and weight_decay cannot be negative")
        if self.model_size not in {"tiny", "base", "large"}:
            raise ValueError("model_size must be 'tiny', 'base', or 'large'")
        for split, value in (
            ("model_train", self.max_train_jets),
            ("model_val", self.max_val_jets),
            ("final_test", self.max_final_test_jets),
        ):
            if value is not None and int(value) > int(PD10_SPLIT_SIZES[split]):
                raise ValueError(f"max jets for {split} cannot exceed {PD10_SPLIT_SIZES[split]}")
        object.__setattr__(self, "variant_name", variant)
        object.__setattr__(self, "seed", int(self.seed))

    @property
    def model_name(self) -> str:
        return self.variant_name

    @property
    def checkpoint_path(self) -> Path:
        return Path(self.output_dir) / "best_model_val.pt"

    @property
    def last_checkpoint_path(self) -> Path:
        return Path(self.output_dir) / "last.pt"

    @property
    def prediction_dir(self) -> Path:
        return Path(self.output_dir) / "predictions"

    @property
    def hlt2_strength(self) -> float | None:
        return hlt_sdv_strength_from_variant(self.variant_name)


@dataclass(frozen=True)
class HLTTTAControlConfig:
    """Config for the HLT ParT logit-average TTA control."""

    output_dir: str
    hlt_cache_dir: str
    hlt2_cache_dir: str
    hlt_teacher_checkpoint: str
    variant_name: str = HLT_SDV_VARIANT_TTA
    val_split: str = "model_val"
    final_test_split: str = "final_test"
    seed: int = 8821
    batch_size: int = 128
    num_workers: int = 0
    device: str = "auto"
    max_val_jets: int | None = PD10_SPLIT_SIZES["model_val"]
    max_final_test_jets: int | None = PD10_SPLIT_SIZES["final_test"]
    max_val_batches: int | None = None
    max_final_test_batches: int | None = None
    evaluate_final_test: bool = True
    confirm_final_test: bool = False
    overwrite: bool = False

    def __post_init__(self) -> None:
        variant = normalize_hlt_sdv_variant(self.variant_name)
        if self.val_split != "model_val" or self.final_test_split != "final_test":
            raise ValueError("TTA control evaluates model_val and final_test only")
        if self.evaluate_final_test and not bool(self.confirm_final_test):
            raise ValueError("TTA final-test evaluation requires confirm_final_test=True")
        if int(self.batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        for split, value in (("model_val", self.max_val_jets), ("final_test", self.max_final_test_jets)):
            if value is not None and int(value) > int(PD10_SPLIT_SIZES[split]):
                raise ValueError(f"max jets for {split} cannot exceed {PD10_SPLIT_SIZES[split]}")
        object.__setattr__(self, "variant_name", variant)
        object.__setattr__(self, "seed", int(self.seed))

    @property
    def model_name(self) -> str:
        return self.variant_name

    @property
    def prediction_dir(self) -> Path:
        return Path(self.output_dir) / "predictions"

    @property
    def hlt2_strength(self) -> float | None:
        return hlt_sdv_strength_from_variant(self.variant_name)


def _limit_view(view: JetView, max_jets: int | None) -> JetView:
    if max_jets is None or int(max_jets) >= int(len(view.labels)):
        return view
    limit = max(int(max_jets), 0)
    return JetView(
        tokens=np.asarray(view.tokens[:limit], dtype=np.float32),
        mask=np.asarray(view.mask[:limit], dtype=bool),
        labels=np.asarray(view.labels[:limit], dtype=np.int64),
        jet_ids=list(view.jet_ids[:limit]),
        split=view.split,
        metadata={**dict(view.metadata), "hlt_sdv_control_max_jets": int(limit)},
    )


def _max_jets_for_hlt2_config(config: HLT2OnlyTrainConfig, split: str) -> int | None:
    if split == config.train_split:
        return config.max_train_jets
    if split == config.val_split:
        return config.max_val_jets
    if split == config.final_test_split:
        return config.max_final_test_jets
    return None


def _load_hlt2_control_view(config: HLT2OnlyTrainConfig, split: str) -> JetView:
    view = load_cached_hlt_view(config.hlt2_cache_dir, split)
    source_view = view.metadata.get("view")
    if source_view not in (None, "hlt2"):
        raise ValueError(f"HLT2-only control requires hlt2 cache rows, got {source_view!r}")
    if bool(view.metadata.get("uses_offline_particles")):
        raise ValueError("HLT2-only control cannot consume a cache that used offline particles")
    return _limit_view(view, _max_jets_for_hlt2_config(config, split))


def _make_hlt2_loader(view: JetView, *, batch_size: int, shuffle: bool, num_workers: int, seed: int):
    dataset = ParticleViewTorchDataset(view, expected_view="hlt2")
    return make_data_loader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        seed=seed,
        source_view="hlt2",
    )


def _checkpoint_model(model):
    return getattr(model, "_orig_mod", model)


def _build_hlt2_only_model(config: HLT2OnlyTrainConfig, *, device, model=None) -> tuple[Any, dict[str, Any]]:
    if model is not None:
        return model.to(device), {"provided_model": True, "initialized_from_hlt_checkpoint": False}
    if config.initialize_from_hlt_checkpoint:
        model, payload = load_hlt_model_from_checkpoint(config.hlt_teacher_checkpoint, device=device)
        model.train()
        return model, {
            "provided_model": False,
            "initialized_from_hlt_checkpoint": True,
            "hlt_teacher_checkpoint": str(config.hlt_teacher_checkpoint),
            "hlt_teacher_checkpoint_sha256": sha256_file(config.hlt_teacher_checkpoint),
            "hlt_teacher_epoch": payload.get("epoch"),
            "hlt_teacher_experiment_step": payload.get("experiment_step"),
        }
    model = build_hlt_classifier(num_classes=PD10_NUM_CLASSES, model_size=config.model_size)
    return model.to(device), {"provided_model": False, "initialized_from_hlt_checkpoint": False}


def _single_part_checkpoint_payload(
    model,
    optimizer,
    *,
    epoch: int,
    config: HLT2OnlyTrainConfig,
    metrics: Mapping[str, Any],
    initialization: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint_model = _checkpoint_model(model)
    return {
        "contract": HLT_SDV_HLT2_ONLY_CONTRACT,
        "experiment_name": HLT_SDV_EXPERIMENT_NAME,
        "experiment_step": HLT_SDV_STEP6_EXPERIMENT_STEP,
        "variant_name": config.variant_name,
        "model_name": config.model_name,
        "model_kind": HLT_SDV_HLT2_ONLY_MODEL_KIND,
        "epoch": int(epoch),
        "model_state_dict": checkpoint_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "config": asdict(config),
        "metrics": dict(metrics),
        "model_config": getattr(checkpoint_model, "config", {}),
        "initialization": dict(initialization),
        "label_names": list(LABEL_NAMES),
        "pf_feature_names": list(PF_FEATURE_NAMES),
        "allowed_inputs": HLT_SDV_ALLOWED_INPUTS,
        "student_deployment_inputs": HLT_SDV_DEPLOYMENT_INPUTS,
        "deployment_inputs": HLT_SDV_DEPLOYMENT_INPUTS,
        "requires_offline_inputs": False,
        "requires_teacher_features": False,
        "requires_deterministic_hlt2_transform": True,
        "returns_offline_particles": False,
        "inference_export_requires_teacher_features": False,
        "selection_metric": "model_val_cross_entropy",
        "selection_tie_breaker": "model_val_accuracy",
        "no_final_test_used_for_selection": True,
    }


def _load_hlt2_only_model_from_checkpoint(checkpoint: str | Path, *, device):
    try:
        return load_hlt_model_from_checkpoint(checkpoint, device=device)
    except Exception:
        torch = require_torch()
        payload = torch.load(checkpoint, map_location=device)
        model = build_hlt_classifier(num_classes=PD10_NUM_CLASSES, model_size="base")
        state_dict = strip_compile_prefix_from_state_dict(payload["model_state_dict"])
        model.load_state_dict(state_dict, strict=True)
        return model.to(device).eval(), payload


def _collect_part_logits_for_view(
    model,
    view: JetView,
    *,
    batch_size: int,
    num_workers: int,
    device,
    seed: int,
    max_batches: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[Any], dict[str, Any]]:
    torch = require_torch()
    loader = _make_hlt2_loader(view, batch_size=batch_size, shuffle=False, num_workers=num_workers, seed=seed)
    logits_chunks: list[np.ndarray] = []
    labels_chunks: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            logits = model(batch["points"], batch["features"], batch["lorentz_vectors"], batch["mask"])
            logits_chunks.append(logits.detach().cpu().float().numpy())
            labels_chunks.append(batch["labels"].detach().cpu().long().numpy())
    if not logits_chunks:
        raise ValueError("No HLT2-only prediction batches were produced")
    logits_np = np.concatenate(logits_chunks, axis=0).astype(np.float32)
    labels_np = np.concatenate(labels_chunks, axis=0).astype(np.int64)
    jet_ids = list(view.jet_ids[: int(labels_np.shape[0])])
    metrics = pd10_prediction_metrics_from_logits(logits_np, labels_np)
    return logits_np, labels_np, jet_ids, metrics


def _hlt2_prediction_block(
    *,
    config: HLT2OnlyTrainConfig,
    split: str,
    view: JetView,
    logits: np.ndarray,
    labels: np.ndarray,
    jet_ids: list[Any],
    checkpoint_payload: Mapping[str, Any] | None,
    metrics: Mapping[str, Any],
) -> PredictionBlock:
    metadata = {
        "contract": HLT_SDV_HLT2_ONLY_CONTRACT,
        "experiment_name": HLT_SDV_EXPERIMENT_NAME,
        "experiment_step": HLT_SDV_STEP6_EXPERIMENT_STEP,
        "variant_name": config.variant_name,
        "model_name": config.model_name,
        "model_kind": HLT_SDV_HLT2_ONLY_MODEL_KIND,
        "architecture": "part",
        "split": split,
        "source_view": "hlt2",
        "hlt2_strength": config.hlt2_strength,
        "checkpoint": str(config.checkpoint_path),
        "checkpoint_sha256": sha256_file(config.checkpoint_path),
        "checkpoint_epoch": None if checkpoint_payload is None else checkpoint_payload.get("epoch"),
        "checkpoint_experiment_step": None if checkpoint_payload is None else checkpoint_payload.get("experiment_step"),
        "hlt2_cache_dir": config.hlt2_cache_dir,
        "hlt2_content_hash": view.metadata.get("hlt2_content_hash") or view.metadata.get("hlt_content_hash"),
        "hlt2_jet_identity_hash": jet_identity_hash(view.jet_ids),
        "rich_metrics": dict(metrics),
        "allowed_inputs": HLT_SDV_ALLOWED_INPUTS,
        "student_deployment_inputs": HLT_SDV_DEPLOYMENT_INPUTS,
        "deployment_inputs": HLT_SDV_DEPLOYMENT_INPUTS,
        "requires_offline_inputs": False,
        "requires_teacher_features": False,
        "requires_deterministic_hlt2_transform": True,
        "returns_offline_particles": False,
        "inference_export_requires_teacher_features": False,
        "no_final_test_used_for_selection": True,
    }
    return PredictionBlock(
        model_name=config.model_name,
        split=split,
        logits=np.asarray(logits, dtype=np.float32),
        probs=softmax_np(logits),
        labels=np.asarray(labels, dtype=np.int64),
        jet_ids=list(jet_ids),
        metadata=metadata,
    )


def _save_hlt2_predictions(
    model,
    view: JetView,
    *,
    config: HLT2OnlyTrainConfig,
    split: str,
    device,
    checkpoint_payload: Mapping[str, Any] | None,
    batch_size: int,
    max_batches: int | None,
    seed: int,
    validation_thresholds_by_class: Mapping[str, Any] | None = None,
    validation_binary_thresholds: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    logits, labels, jet_ids, metrics = _collect_part_logits_for_view(
        model,
        view,
        batch_size=batch_size,
        num_workers=config.num_workers,
        device=device,
        seed=seed,
        max_batches=max_batches,
    )
    if validation_thresholds_by_class is not None or validation_binary_thresholds is not None:
        metrics = pd10_prediction_metrics_from_logits(
            logits,
            labels,
            validation_thresholds_by_class=validation_thresholds_by_class,
            validation_binary_thresholds=validation_binary_thresholds,
        )
    block = _hlt2_prediction_block(
        config=config,
        split=split,
        view=view,
        logits=logits,
        labels=labels,
        jet_ids=jet_ids,
        checkpoint_payload=checkpoint_payload,
        metrics=metrics,
    )
    metadata = save_prediction_block(block, config.prediction_dir, overwrite=bool(config.overwrite))
    return metrics, metadata


def train_hlt2_only_control(
    config: HLT2OnlyTrainConfig,
    *,
    model=None,
) -> dict[str, Any]:
    torch = require_torch()
    set_training_seed(config.seed)
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir)
    if config.checkpoint_path.exists() and not config.overwrite:
        raise FileExistsError(f"HLT2-only checkpoint already exists: {config.checkpoint_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_view = _load_hlt2_control_view(config, config.train_split)
    val_view = _load_hlt2_control_view(config, config.val_split)
    train_loader = _make_hlt2_loader(
        train_view,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        seed=int(config.seed) + 101,
    )
    val_loader = _make_hlt2_loader(
        val_view,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        seed=int(config.seed) + 202,
    )

    provided_model = model is not None
    model, initialization = _build_hlt2_only_model(config, device=device, model=model)
    checkpoint_model = model
    if config.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)
        checkpoint_model = getattr(model, "_orig_mod", checkpoint_model)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.lr), weight_decay=float(config.weight_decay))
    scaler = amp_grad_scaler(bool(config.amp and device.type == "cuda"))
    run_metadata = {
        "ok": True,
        "contract": HLT_SDV_HLT2_ONLY_CONTRACT,
        "experiment_name": HLT_SDV_EXPERIMENT_NAME,
        "experiment_step": HLT_SDV_STEP6_EXPERIMENT_STEP,
        "variant_name": config.variant_name,
        "model_name": config.model_name,
        "model_kind": HLT_SDV_HLT2_ONLY_MODEL_KIND,
        "architecture": "part",
        "config": asdict(config),
        "initialization": dict(initialization),
        "train_n_jets": int(len(train_view.labels)),
        "val_n_jets": int(len(val_view.labels)),
        "train_hlt2_content_hash": train_view.metadata.get("hlt2_content_hash") or train_view.metadata.get("hlt_content_hash"),
        "val_hlt2_content_hash": val_view.metadata.get("hlt2_content_hash") or val_view.metadata.get("hlt_content_hash"),
        "allowed_inputs": HLT_SDV_ALLOWED_INPUTS,
        "student_deployment_inputs": HLT_SDV_DEPLOYMENT_INPUTS,
        "deployment_inputs": HLT_SDV_DEPLOYMENT_INPUTS,
        "requires_offline_inputs": False,
        "requires_teacher_features": False,
        "requires_deterministic_hlt2_transform": True,
        "returns_offline_particles": False,
        "inference_export_requires_teacher_features": False,
        "control_role": "single_view_hlt2_control_for_dual_fusion_interpretation",
        "selection_metric": "model_val_cross_entropy",
        "selection_tie_breaker": "model_val_accuracy",
        "no_final_test_used_for_selection": True,
    }
    save_json(output_dir / "config.json", run_metadata)

    curves: list[dict[str, Any]] = []
    best_val_ce = float("inf")
    best_val_accuracy = -1.0
    best_epoch = -1
    best_state_dict = None
    best_row: dict[str, Any] | None = None
    epochs_without_improvement = 0
    for epoch in range(1, int(config.epochs) + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            amp=config.amp,
            grad_clip_norm=config.grad_clip_norm,
            max_batches=config.max_train_batches,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            device=device,
            criterion=criterion,
            amp=False,
            max_batches=config.max_val_batches,
        )
        row = {"epoch": int(epoch), "train": train_metrics, "model_val": val_metrics}
        curves.append(row)
        save_json(output_dir / "training_curves.json", {"epochs": curves, "selection_metric": "model_val_cross_entropy"})
        val_ce = float(val_metrics["loss"])
        val_accuracy = float(val_metrics["accuracy"])
        improved = val_ce < best_val_ce or (np.isclose(val_ce, best_val_ce) and val_accuracy > best_val_accuracy)
        torch.save(
            _single_part_checkpoint_payload(
                checkpoint_model,
                optimizer,
                epoch=epoch,
                config=config,
                metrics=row,
                initialization=initialization,
            ),
            config.last_checkpoint_path,
        )
        if improved:
            best_val_ce = val_ce
            best_val_accuracy = val_accuracy
            best_epoch = int(epoch)
            best_row = row
            epochs_without_improvement = 0
            torch.save(
                _single_part_checkpoint_payload(
                    checkpoint_model,
                    optimizer,
                    epoch=epoch,
                    config=config,
                    metrics=row,
                    initialization=initialization,
                ),
                config.checkpoint_path,
            )
            if provided_model:
                best_state_dict = copy.deepcopy(_checkpoint_model(model).state_dict())
        else:
            epochs_without_improvement += 1
        if config.early_stop_patience >= 0 and epochs_without_improvement >= int(config.early_stop_patience):
            break

    if provided_model:
        if best_state_dict is not None:
            _checkpoint_model(model).load_state_dict(best_state_dict, strict=True)
        selected_model = model
        selected_payload = {
            "epoch": best_epoch,
            "experiment_step": HLT_SDV_STEP6_EXPERIMENT_STEP,
            "model_config": getattr(_checkpoint_model(model), "config", {}),
        }
    else:
        selected_model, selected_payload = _load_hlt2_only_model_from_checkpoint(config.checkpoint_path, device=device)

    model_val_prediction_metrics = None
    model_val_prediction_metadata = None
    validation_thresholds = None
    validation_binary_thresholds = None
    if config.evaluate_model_val_predictions:
        model_val_prediction_metrics, model_val_prediction_metadata = _save_hlt2_predictions(
            selected_model,
            val_view,
            config=config,
            split=config.val_split,
            device=device,
            checkpoint_payload=selected_payload,
            batch_size=config.eval_batch_size,
            max_batches=config.max_val_batches,
            seed=int(config.seed) + 303,
        )
        validation_thresholds = model_val_prediction_metrics.get("score_thresholds_by_class")
        validation_binary_thresholds = model_val_prediction_metrics.get("binary_score_thresholds")

    final_test_metrics = None
    final_test_prediction_metadata = None
    if config.evaluate_final_test:
        final_view = _load_hlt2_control_view(config, config.final_test_split)
        final_test_metrics, final_test_prediction_metadata = _save_hlt2_predictions(
            selected_model,
            final_view,
            config=config,
            split=config.final_test_split,
            device=device,
            checkpoint_payload=selected_payload,
            batch_size=config.eval_batch_size,
            max_batches=config.max_final_test_batches,
            seed=int(config.seed) + 404,
            validation_thresholds_by_class=validation_thresholds,
            validation_binary_thresholds=validation_binary_thresholds,
        )

    report = {
        **run_metadata,
        "best_epoch": int(best_epoch),
        "best_model_val_cross_entropy": float(best_val_ce),
        "best_model_val_accuracy": float(best_val_accuracy),
        "best_model_val_training_metrics": None if best_row is None else dict(best_row["model_val"]),
        "epochs_completed": int(len(curves)),
        "final_epoch": curves[-1] if curves else None,
        "checkpoint": str(config.checkpoint_path),
        "checkpoint_sha256": sha256_file(config.checkpoint_path),
        "last_checkpoint": str(config.last_checkpoint_path),
        "model_val_prediction_metrics": model_val_prediction_metrics,
        "model_val_prediction_metadata": model_val_prediction_metadata,
        "final_test_metrics": final_test_metrics,
        "final_test_prediction_metadata": final_test_prediction_metadata,
    }
    save_json(output_dir / "model_val_report.json", report)
    save_json(output_dir / "run_report.json", report)
    if final_test_metrics is not None:
        save_json(output_dir / "final_test_report.json", {**report, "metrics": final_test_metrics})
    return report


def _load_tta_dataset(config: HLTTTAControlConfig, split: str):
    max_jets = config.max_val_jets if split == config.val_split else config.max_final_test_jets
    return load_hlt_sdv_dataset(
        config.hlt_cache_dir,
        split,
        hlt2_cache_dir=config.hlt2_cache_dir,
        max_jets=max_jets,
    )


def _load_tta_model(config: HLTTTAControlConfig, *, device, model=None) -> tuple[Any, dict[str, Any]]:
    if model is not None:
        return model.to(device), {"provided_model": True}
    model, payload = load_hlt_model_from_checkpoint(config.hlt_teacher_checkpoint, device=device)
    return model, dict(payload)


def _collect_tta_logits(
    model,
    dataset,
    *,
    batch_size: int,
    num_workers: int,
    device,
    seed: int,
    variant_name: str,
    max_batches: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[Any], dict[str, Any], dict[str, Any]]:
    torch = require_torch()
    loader = make_hlt_sdv_data_loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        seed=seed,
    )
    averaged_chunks: list[np.ndarray] = []
    hlt_chunks: list[np.ndarray] = []
    hlt2_chunks: list[np.ndarray] = []
    labels_chunks: list[np.ndarray] = []
    jet_ids: list[Any] = []
    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            batch = move_hlt_sdv_batch_to_device(batch, device)
            hlt_logits = model(
                batch["hlt_inputs"]["points"],
                batch["hlt_inputs"]["features"],
                batch["hlt_inputs"]["lorentz_vectors"],
                batch["hlt_inputs"]["mask"],
            )
            hlt2_logits = model(
                batch["hlt2_inputs"]["points"],
                batch["hlt2_inputs"]["features"],
                batch["hlt2_inputs"]["lorentz_vectors"],
                batch["hlt2_inputs"]["mask"],
            )
            averaged = 0.5 * (hlt_logits + hlt2_logits)
            averaged_chunks.append(averaged.detach().cpu().float().numpy())
            hlt_chunks.append(hlt_logits.detach().cpu().float().numpy())
            hlt2_chunks.append(hlt2_logits.detach().cpu().float().numpy())
            labels_chunks.append(batch["labels"].detach().cpu().long().numpy())
            jet_ids.extend(batch["jet_ids"])
    if not averaged_chunks:
        raise ValueError("No TTA prediction batches were produced")
    logits = np.concatenate(averaged_chunks, axis=0).astype(np.float32)
    hlt_logits_np = np.concatenate(hlt_chunks, axis=0).astype(np.float32)
    hlt2_logits_np = np.concatenate(hlt2_chunks, axis=0).astype(np.float32)
    labels = np.concatenate(labels_chunks, axis=0).astype(np.int64)
    _require_finite_logits_array(logits, context=f"{variant_name}:averaged_tta")
    _require_finite_logits_array(hlt_logits_np, context=f"{variant_name}:hlt_component")
    _require_finite_logits_array(hlt2_logits_np, context=f"{variant_name}:hlt2_component")
    metrics = pd10_prediction_metrics_from_logits(logits, labels)
    component_metrics = {
        "hlt_only_component": pd10_prediction_metrics_from_logits(hlt_logits_np, labels),
        "hlt2_only_component_same_checkpoint": pd10_prediction_metrics_from_logits(hlt2_logits_np, labels),
    }
    return logits, labels, jet_ids, metrics, component_metrics


def _tta_prediction_block(
    *,
    config: HLTTTAControlConfig,
    split: str,
    dataset,
    logits: np.ndarray,
    labels: np.ndarray,
    jet_ids: list[Any],
    checkpoint_payload: Mapping[str, Any] | None,
    metrics: Mapping[str, Any],
    component_metrics: Mapping[str, Any],
) -> PredictionBlock:
    metadata = {
        "contract": HLT_SDV_TTA_CONTRACT,
        "experiment_name": HLT_SDV_EXPERIMENT_NAME,
        "experiment_step": HLT_SDV_STEP6_EXPERIMENT_STEP,
        "variant_name": config.variant_name,
        "model_name": config.model_name,
        "model_kind": HLT_SDV_TTA_MODEL_KIND,
        "architecture": "part_logit_average",
        "split": split,
        "source_views": ["HLT", "HLT2"],
        "logit_combination": "0.5 * HLT_logits + 0.5 * HLT2_logits",
        "hlt2_strength": config.hlt2_strength,
        "checkpoint": str(config.hlt_teacher_checkpoint),
        "checkpoint_sha256": sha256_file(config.hlt_teacher_checkpoint),
        "checkpoint_epoch": None if checkpoint_payload is None else checkpoint_payload.get("epoch"),
        "checkpoint_experiment_step": None if checkpoint_payload is None else checkpoint_payload.get("experiment_step"),
        "dataset": dataset.to_metadata(),
        "hlt_cache_dir": config.hlt_cache_dir,
        "hlt2_cache_dir": config.hlt2_cache_dir,
        "hlt_jet_identity_hash": jet_identity_hash(dataset.jet_ids),
        "rich_metrics": dict(metrics),
        "component_metrics": dict(component_metrics),
        "allowed_inputs": HLT_SDV_ALLOWED_INPUTS,
        "student_deployment_inputs": HLT_SDV_DEPLOYMENT_INPUTS,
        "deployment_inputs": HLT_SDV_DEPLOYMENT_INPUTS,
        "requires_offline_inputs": False,
        "requires_teacher_features": False,
        "requires_deterministic_hlt2_transform": True,
        "returns_offline_particles": False,
        "inference_export_requires_teacher_features": False,
        "no_final_test_used_for_selection": True,
    }
    return PredictionBlock(
        model_name=config.model_name,
        split=split,
        logits=np.asarray(logits, dtype=np.float32),
        probs=softmax_np(logits),
        labels=np.asarray(labels, dtype=np.int64),
        jet_ids=list(jet_ids),
        metadata=metadata,
    )


def _save_tta_predictions(
    model,
    *,
    config: HLTTTAControlConfig,
    split: str,
    device,
    checkpoint_payload: Mapping[str, Any] | None,
    validation_thresholds_by_class: Mapping[str, Any] | None = None,
    validation_binary_thresholds: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    dataset = _load_tta_dataset(config, split)
    max_batches = config.max_val_batches if split == config.val_split else config.max_final_test_batches
    seed = int(config.seed) + (303 if split == config.val_split else 404)
    logits, labels, jet_ids, metrics, component_metrics = _collect_tta_logits(
        model,
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        device=device,
        seed=seed,
        variant_name=config.variant_name,
        max_batches=max_batches,
    )
    if validation_thresholds_by_class is not None or validation_binary_thresholds is not None:
        metrics = pd10_prediction_metrics_from_logits(
            logits,
            labels,
            validation_thresholds_by_class=validation_thresholds_by_class,
            validation_binary_thresholds=validation_binary_thresholds,
        )
    block = _tta_prediction_block(
        config=config,
        split=split,
        dataset=dataset,
        logits=logits,
        labels=labels,
        jet_ids=jet_ids,
        checkpoint_payload=checkpoint_payload,
        metrics=metrics,
        component_metrics=component_metrics,
    )
    metadata = save_prediction_block(block, config.prediction_dir, overwrite=bool(config.overwrite))
    return metrics, metadata


def run_hlt_tta_control(
    config: HLTTTAControlConfig,
    *,
    model=None,
) -> dict[str, Any]:
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, payload = _load_tta_model(config, device=device, model=model)
    model_val_metrics, model_val_metadata = _save_tta_predictions(
        model,
        config=config,
        split=config.val_split,
        device=device,
        checkpoint_payload=payload,
    )
    validation_thresholds = model_val_metrics.get("score_thresholds_by_class")
    validation_binary_thresholds = model_val_metrics.get("binary_score_thresholds")
    final_test_metrics = None
    final_test_metadata = None
    if config.evaluate_final_test:
        final_test_metrics, final_test_metadata = _save_tta_predictions(
            model,
            config=config,
            split=config.final_test_split,
            device=device,
            checkpoint_payload=payload,
            validation_thresholds_by_class=validation_thresholds,
            validation_binary_thresholds=validation_binary_thresholds,
        )

    report = {
        "ok": True,
        "contract": HLT_SDV_TTA_CONTRACT,
        "experiment_name": HLT_SDV_EXPERIMENT_NAME,
        "experiment_step": HLT_SDV_STEP6_EXPERIMENT_STEP,
        "variant_name": config.variant_name,
        "model_name": config.model_name,
        "model_kind": HLT_SDV_TTA_MODEL_KIND,
        "architecture": "part_logit_average",
        "config": asdict(config),
        "checkpoint": str(config.hlt_teacher_checkpoint),
        "checkpoint_sha256": sha256_file(config.hlt_teacher_checkpoint),
        "model_val_metrics": model_val_metrics,
        "model_val_prediction_metadata": model_val_metadata,
        "final_test_metrics": final_test_metrics,
        "final_test_prediction_metadata": final_test_metadata,
        "allowed_inputs": HLT_SDV_ALLOWED_INPUTS,
        "student_deployment_inputs": HLT_SDV_DEPLOYMENT_INPUTS,
        "deployment_inputs": HLT_SDV_DEPLOYMENT_INPUTS,
        "requires_offline_inputs": False,
        "requires_teacher_features": False,
        "requires_deterministic_hlt2_transform": True,
        "returns_offline_particles": False,
        "inference_export_requires_teacher_features": False,
        "control_role": "logit_average_tta_control_for_dual_fusion_interpretation",
        "no_final_test_used_for_selection": True,
    }
    save_json(output_dir / "run_report.json", report)
    save_json(output_dir / "model_val_report.json", report)
    if final_test_metrics is not None:
        save_json(output_dir / "final_test_report.json", {**report, "metrics": final_test_metrics})
    return report


def default_hlt2_only_config_for_pd10_root(
    pd10_root: str | Path,
    *,
    strength: float = 0.20,
    output_dir: str | Path | None = None,
    hlt2_cache_dir: str | Path | None = None,
    confirm_final_test: bool = False,
    overwrite: bool = False,
) -> HLT2OnlyTrainConfig:
    pd10_root = Path(pd10_root)
    layout = default_hlt_sdv_experiment_layout(
        output_root=pd10_root.parent,
        pd10_experiment_name=pd10_root.name,
    )
    variant = HLT_SDV_VARIANT_HLT2_ONLY if abs(float(strength) - 0.20) <= 1.0e-12 else f"hlt2_only_part_{hlt_sdv_strength_tag(strength)}"
    return HLT2OnlyTrainConfig(
        output_dir=str(Path(output_dir) if output_dir else layout.variant_dir(variant)),
        hlt2_cache_dir=str(Path(hlt2_cache_dir) if hlt2_cache_dir else default_hlt2_cache_dir(pd10_root, float(strength))),
        hlt_teacher_checkpoint=str(layout.hlt_teacher_checkpoint),
        variant_name=variant,
        confirm_final_test=confirm_final_test,
        overwrite=overwrite,
    )


def default_tta_config_for_pd10_root(
    pd10_root: str | Path,
    *,
    strength: float = 0.20,
    output_dir: str | Path | None = None,
    hlt2_cache_dir: str | Path | None = None,
    confirm_final_test: bool = False,
    overwrite: bool = False,
) -> HLTTTAControlConfig:
    pd10_root = Path(pd10_root)
    layout = default_hlt_sdv_experiment_layout(
        output_root=pd10_root.parent,
        pd10_experiment_name=pd10_root.name,
    )
    variant = HLT_SDV_VARIANT_TTA if abs(float(strength) - 0.20) <= 1.0e-12 else f"tta_hlt_part_hlt_plus_hlt2_{hlt_sdv_strength_tag(strength)}"
    return HLTTTAControlConfig(
        output_dir=str(Path(output_dir) if output_dir else layout.variant_dir(variant)),
        hlt_cache_dir=str(pd10_root / "hlt_cache"),
        hlt2_cache_dir=str(Path(hlt2_cache_dir) if hlt2_cache_dir else default_hlt2_cache_dir(pd10_root, float(strength))),
        hlt_teacher_checkpoint=str(layout.hlt_teacher_checkpoint),
        variant_name=variant,
        confirm_final_test=confirm_final_test,
        overwrite=overwrite,
    )


__all__ = [
    "HLT2OnlyTrainConfig",
    "HLTTTAControlConfig",
    "HLT_SDV_HLT2_ONLY_CONTRACT",
    "HLT_SDV_HLT2_ONLY_MODEL_KIND",
    "HLT_SDV_STEP6_EXPERIMENT_STEP",
    "HLT_SDV_TTA_CONTRACT",
    "HLT_SDV_TTA_MODEL_KIND",
    "default_hlt2_only_config_for_pd10_root",
    "default_tta_config_for_pd10_root",
    "run_hlt_tta_control",
    "train_hlt2_only_control",
]
