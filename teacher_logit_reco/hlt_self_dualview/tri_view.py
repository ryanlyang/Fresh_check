"""Deployable HLT tri-view source and fusion training."""

from __future__ import annotations

from contextlib import nullcontext
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
from jetclass_fresh.jetclass_data import JetIdentity, JetView, LABEL_NAMES
from jetclass_fresh.part_inputs import PF_FEATURE_NAMES, build_particle_transformer_inputs_from_tokens

from teacher_logit_reco.privileged_distill_10class.config import (
    PD10_NUM_CLASSES,
    PD10_REPRESENTATION_DIM,
    PD10_SPLIT_ORDER,
    PD10_SPLIT_SIZES,
)
from teacher_logit_reco.privileged_distill_10class.metrics import pd10_prediction_metrics_from_logits

from .config import HLT_SDV_ALLOWED_INPUTS, HLT_SDV_DEPLOYMENT_INPUTS, HLT_SDV_EXPERIMENT_NAME
from .model import (
    HLT_SDV_DEFAULT_DROPOUT,
    HLT_SDV_DEFAULT_FUSION_HIDDEN_DIM,
    HLT_SDV_DEFAULT_MODEL_SIZE,
    HLTSDVParticleTransformerEmbeddingBranch,
    hlt_sdv_branch_dim_from_config,
    hlt_sdv_embedding_branch_config,
    load_matching_branch_weights,
    sha256_file,
)

try:  # Keep imports lightweight on machines without torch.
    import torch as _torch
except ImportError:  # pragma: no cover - environment dependent
    _torch = None

if _torch is None:  # pragma: no cover - environment dependent
    class _ModuleBase:
        pass
else:
    _ModuleBase = _torch.nn.Module


HLT_TRIVIEW_EXPERIMENT_NAME = "hlt_self_triview_10class"
HLT_TRIVIEW_STEP = "hlt_self_triview_source_and_fusion_v1"
HLT_TRIVIEW_SOURCE_CONTRACT = "hlt_self_triview_source_part_v1"
HLT_TRIVIEW_FUSION_CONTRACT = "hlt_self_triview_particle_fusion_v1"
HLT_TRIVIEW_PREDICTION_CONTRACT = "hlt_self_triview_prediction_cache_v1"
HLT_TRIVIEW_MODEL_ARCHITECTURE = "deployable_hlt_hlt2s035_hlt2s100_part_concat_pairwise_abs_product"
HLT_TRIVIEW_MODEL_NAME = "tri_hlt_hlt2_s0p35_s1p00"
HLT_TRIVIEW_DEPLOYMENT_INPUTS = "HLT_plus_deterministic_HLT2_s0p35_and_s1p00"
HLT_TRIVIEW_DEFAULT_TRAIN_JETS = 1_000_000
HLT_TRIVIEW_DEFAULT_VAL_JETS = 250_000
HLT_TRIVIEW_DEFAULT_FINAL_TEST_JETS = 500_000


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
        metadata={**dict(view.metadata), "hlt_triview_max_jets": int(limit)},
    )


def _identity_key(identity: JetIdentity) -> tuple[str, int, int]:
    return str(identity.file), int(identity.entry), int(identity.label)


def _view_index_by_identity(view: JetView) -> dict[tuple[str, int, int], int]:
    result: dict[tuple[str, int, int], int] = {}
    for index, identity in enumerate(view.jet_ids):
        key = _identity_key(identity)
        if key in result:
            raise ValueError(f"duplicate jet identity in {view.split}: {identity}")
        result[key] = int(index)
    return result


def _align_secondary_view(parent: JetView, secondary: JetView, *, label: str) -> JetView:
    if parent.split != secondary.split:
        raise ValueError(f"{label} split mismatch: {parent.split} != {secondary.split}")
    index_by_identity = _view_index_by_identity(secondary)
    indices: list[int] = []
    for row, identity in enumerate(parent.jet_ids):
        key = _identity_key(identity)
        if key not in index_by_identity:
            raise ValueError(f"{label} view is missing parent HLT identity at row {row}: {identity}")
        secondary_row = index_by_identity[key]
        if int(secondary.labels[secondary_row]) != int(parent.labels[row]):
            raise ValueError(f"{label} label mismatch at parent row {row}")
        indices.append(secondary_row)
    aligned = JetView(
        tokens=np.asarray(secondary.tokens[indices], dtype=np.float32),
        mask=np.asarray(secondary.mask[indices], dtype=bool),
        labels=np.asarray(secondary.labels[indices], dtype=np.int64),
        jet_ids=[secondary.jet_ids[index] for index in indices],
        split=secondary.split,
        metadata={**dict(secondary.metadata), "aligned_to_hlt_triview_parent": True, "view_label": label},
    )
    if aligned.jet_ids != parent.jet_ids:
        raise ValueError(f"{label} aligned jet identities do not match parent HLT")
    if not np.array_equal(aligned.labels, parent.labels):
        raise ValueError(f"{label} aligned labels do not match parent HLT labels")
    return aligned


def _check_view(view: JetView, *, expected_view: str, label: str) -> None:
    source_view = view.metadata.get("view")
    if source_view not in (None, expected_view):
        raise ValueError(f"{label} expected view {expected_view!r}, got {source_view!r}")
    if bool(view.metadata.get("uses_offline_particles")):
        raise ValueError(f"{label} cache metadata says it used offline particles")
    if not np.isfinite(view.tokens).all():
        raise FloatingPointError(f"{label} tokens contain non-finite values")
    if view.mask.shape != view.tokens.shape[:2]:
        raise ValueError(f"{label} mask shape does not match tokens")


def _require_finite_metrics(metrics: Mapping[str, Any], *, context: str) -> None:
    for key in ("loss", "cross_entropy", "accuracy"):
        if key not in metrics:
            continue
        value = metrics.get(key)
        if value is None:
            continue
        try:
            finite = np.isfinite(float(value))
        except (TypeError, ValueError):
            finite = False
        if not bool(finite):
            raise FloatingPointError(f"{context} produced non-finite metric {key}={value!r}")


def _require_finite_logits(logits: np.ndarray, *, context: str) -> None:
    finite = np.isfinite(logits)
    if bool(finite.all()):
        return
    bad_rows = np.where(~np.all(finite, axis=1))[0]
    first_bad = int(bad_rows[0]) if int(bad_rows.shape[0]) else -1
    finite_logits = logits[finite]
    if finite_logits.size:
        finite_min = float(np.min(finite_logits))
        finite_max = float(np.max(finite_logits))
    else:
        finite_min = None
        finite_max = None
    raise FloatingPointError(
        f"{context} produced non-finite logits: "
        f"bad_rows={int(bad_rows.shape[0])}/{int(logits.shape[0])}, "
        f"first_bad_row={first_bad}, finite_min={finite_min}, finite_max={finite_max}"
    )


@dataclass(frozen=True)
class HLTTriViewSourceConfig:
    """Training config for one single-view source ParT branch."""

    output_dir: str
    cache_dir: str
    source_name: str
    source_view: str
    warm_start_checkpoint: str | None = None
    train_split: str = "model_train"
    val_split: str = "model_val"
    final_test_split: str = "final_test"
    seed: int = 9101
    batch_size: int = 128
    eval_batch_size: int = 128
    epochs: int = 10
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    num_workers: int = 0
    device: str = "auto"
    amp: bool = True
    grad_clip_norm: float = 1.0
    early_stop_patience: int = 3
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    max_final_test_batches: int | None = None
    max_train_jets: int | None = HLT_TRIVIEW_DEFAULT_TRAIN_JETS
    max_val_jets: int | None = HLT_TRIVIEW_DEFAULT_VAL_JETS
    max_final_test_jets: int | None = HLT_TRIVIEW_DEFAULT_FINAL_TEST_JETS
    model_size: str = HLT_SDV_DEFAULT_MODEL_SIZE
    compile_model: bool = False
    evaluate_model_val_predictions: bool = True
    evaluate_final_test: bool = True
    confirm_final_test: bool = False
    overwrite: bool = False

    def __post_init__(self) -> None:
        if (self.train_split, self.val_split, self.final_test_split) != PD10_SPLIT_ORDER:
            raise ValueError(f"HLT tri-view source split order must be {PD10_SPLIT_ORDER}")
        if self.source_view not in {"fixed_hlt", "hlt2"}:
            raise ValueError("source_view must be 'fixed_hlt' or 'hlt2'")
        if any(ch.isspace() for ch in str(self.source_name)) or not str(self.source_name).strip():
            raise ValueError("source_name must be a non-empty path-safe token")
        if self.evaluate_final_test and not bool(self.confirm_final_test):
            raise ValueError("HLT tri-view source final-test evaluation requires confirm_final_test=True")
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
        object.__setattr__(self, "seed", int(self.seed))

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
    def model_name(self) -> str:
        return self.source_name


def _max_jets_for_source(config: HLTTriViewSourceConfig, split: str) -> int | None:
    if split == config.train_split:
        return config.max_train_jets
    if split == config.val_split:
        return config.max_val_jets
    if split == config.final_test_split:
        return config.max_final_test_jets
    return None


def _load_source_view(config: HLTTriViewSourceConfig, split: str) -> JetView:
    view = load_cached_hlt_view(config.cache_dir, split, verify_hash=True)
    _check_view(view, expected_view=config.source_view, label=config.source_name)
    return _limit_view(view, _max_jets_for_source(config, split))


def _source_checkpoint_payload(
    model,
    optimizer,
    *,
    epoch: int,
    config: HLTTriViewSourceConfig,
    metrics: Mapping[str, Any],
    initialization: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint_model = getattr(model, "_orig_mod", model)
    return {
        "contract": HLT_TRIVIEW_SOURCE_CONTRACT,
        "experiment_name": HLT_TRIVIEW_EXPERIMENT_NAME,
        "experiment_step": HLT_TRIVIEW_STEP,
        "source_name": config.source_name,
        "model_name": config.model_name,
        "model_kind": "single_view_part_source",
        "source_view": config.source_view,
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
        "deployment_inputs": HLT_TRIVIEW_DEPLOYMENT_INPUTS,
        "requires_offline_inputs": False,
        "requires_teacher_features": False,
        "requires_deterministic_hlt2_transform": config.source_view == "hlt2",
        "returns_offline_particles": False,
        "selection_metric": "model_val_cross_entropy",
        "selection_tie_breaker": "model_val_accuracy",
        "no_final_test_used_for_selection": True,
    }


def _build_source_model(config: HLTTriViewSourceConfig, *, device, model=None) -> tuple[Any, dict[str, Any]]:
    if model is not None:
        return model.to(device), {"provided_model": True}
    if config.warm_start_checkpoint:
        loaded, payload = load_hlt_model_from_checkpoint(config.warm_start_checkpoint, device=device)
        loaded.train()
        return loaded, {
            "provided_model": False,
            "warm_start_checkpoint": str(config.warm_start_checkpoint),
            "warm_start_checkpoint_sha256": sha256_file(config.warm_start_checkpoint),
            "warm_start_epoch": payload.get("epoch"),
            "warm_start_experiment_step": payload.get("experiment_step"),
        }
    return build_hlt_classifier(num_classes=PD10_NUM_CLASSES, model_size=config.model_size).to(device), {
        "provided_model": False,
        "warm_start_checkpoint": None,
    }


def _collect_source_logits(
    model,
    view: JetView,
    *,
    config: HLTTriViewSourceConfig,
    batch_size: int,
    device,
    seed: int,
    max_batches: int | None,
) -> tuple[np.ndarray, np.ndarray, list[Any], dict[str, Any]]:
    torch = require_torch()
    dataset = ParticleViewTorchDataset(view, expected_view=config.source_view)
    loader = make_data_loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        seed=seed,
        source_view=config.source_view,
    )
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
        raise ValueError(f"No prediction batches were produced for {config.source_name}")
    logits_np = np.concatenate(logits_chunks, axis=0).astype(np.float32)
    _require_finite_logits(logits_np, context=f"{config.source_name}:{view.split}:prediction")
    labels_np = np.concatenate(labels_chunks, axis=0).astype(np.int64)
    jet_ids = list(view.jet_ids[: int(labels_np.shape[0])])
    metrics = pd10_prediction_metrics_from_logits(logits_np, labels_np)
    return logits_np, labels_np, jet_ids, metrics


def _save_source_predictions(
    model,
    view: JetView,
    *,
    config: HLTTriViewSourceConfig,
    split: str,
    checkpoint_payload: Mapping[str, Any] | None,
    device,
    batch_size: int,
    max_batches: int | None,
    seed: int,
    validation_thresholds_by_class: Mapping[str, Any] | None = None,
    validation_binary_thresholds: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    logits, labels, jet_ids, metrics = _collect_source_logits(
        model,
        view,
        config=config,
        batch_size=batch_size,
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
    block = PredictionBlock(
        model_name=config.model_name,
        split=split,
        logits=logits,
        probs=softmax_np(logits),
        labels=labels,
        jet_ids=jet_ids,
        metadata={
            "contract": HLT_TRIVIEW_PREDICTION_CONTRACT,
            "experiment_name": HLT_TRIVIEW_EXPERIMENT_NAME,
            "experiment_step": HLT_TRIVIEW_STEP,
            "model_name": config.model_name,
            "model_kind": "single_view_part_source",
            "source_view": config.source_view,
            "split": split,
            "cache_dir": config.cache_dir,
            "checkpoint": str(config.checkpoint_path),
            "checkpoint_sha256": sha256_file(config.checkpoint_path),
            "checkpoint_epoch": None if checkpoint_payload is None else checkpoint_payload.get("epoch"),
            "rich_metrics": dict(metrics),
            "view_content_hash": view.metadata.get("hlt2_content_hash") or view.metadata.get("hlt_content_hash"),
            "jet_identity_hash": jet_identity_hash(view.jet_ids),
            "allowed_inputs": HLT_SDV_ALLOWED_INPUTS,
            "deployment_inputs": HLT_TRIVIEW_DEPLOYMENT_INPUTS,
            "requires_offline_inputs": False,
            "requires_teacher_features": False,
            "requires_deterministic_hlt2_transform": config.source_view == "hlt2",
            "no_final_test_used_for_selection": True,
        },
    )
    metadata = save_prediction_block(block, config.prediction_dir, overwrite=bool(config.overwrite))
    return metrics, metadata


def train_hlt_triview_source(config: HLTTriViewSourceConfig, *, model=None) -> dict[str, Any]:
    torch = require_torch()
    set_training_seed(config.seed)
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir)
    if config.checkpoint_path.exists() and not config.overwrite:
        raise FileExistsError(f"HLT tri-view source checkpoint already exists: {config.checkpoint_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_view = _load_source_view(config, config.train_split)
    val_view = _load_source_view(config, config.val_split)
    train_dataset = ParticleViewTorchDataset(train_view, expected_view=config.source_view)
    val_dataset = ParticleViewTorchDataset(val_view, expected_view=config.source_view)
    train_loader = make_data_loader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        seed=int(config.seed) + 101,
        source_view=config.source_view,
    )
    val_loader = make_data_loader(
        val_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        seed=int(config.seed) + 202,
        source_view=config.source_view,
    )
    model, initialization = _build_source_model(config, device=device, model=model)
    checkpoint_model = model
    if config.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)
        checkpoint_model = getattr(model, "_orig_mod", checkpoint_model)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.lr), weight_decay=float(config.weight_decay))
    scaler = amp_grad_scaler(bool(config.amp and device.type == "cuda"))
    run_metadata = {
        "ok": True,
        "contract": HLT_TRIVIEW_SOURCE_CONTRACT,
        "experiment_name": HLT_TRIVIEW_EXPERIMENT_NAME,
        "experiment_step": HLT_TRIVIEW_STEP,
        "source_name": config.source_name,
        "model_name": config.model_name,
        "model_kind": "single_view_part_source",
        "source_view": config.source_view,
        "architecture": "part",
        "config": asdict(config),
        "initialization": dict(initialization),
        "train_n_jets": int(len(train_view.labels)),
        "val_n_jets": int(len(val_view.labels)),
        "train_content_hash": train_view.metadata.get("hlt2_content_hash") or train_view.metadata.get("hlt_content_hash"),
        "val_content_hash": val_view.metadata.get("hlt2_content_hash") or val_view.metadata.get("hlt_content_hash"),
        "allowed_inputs": HLT_SDV_ALLOWED_INPUTS,
        "deployment_inputs": HLT_TRIVIEW_DEPLOYMENT_INPUTS,
        "requires_offline_inputs": False,
        "requires_teacher_features": False,
        "requires_deterministic_hlt2_transform": config.source_view == "hlt2",
        "returns_offline_particles": False,
        "selection_metric": "model_val_cross_entropy",
        "selection_tie_breaker": "model_val_accuracy",
        "no_final_test_used_for_selection": True,
    }
    save_json(output_dir / "config.json", run_metadata)

    curves: list[dict[str, Any]] = []
    best_val_ce = float("inf")
    best_val_accuracy = -1.0
    best_epoch = -1
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
        _require_finite_metrics(train_metrics, context=f"{config.source_name}:epoch{epoch}:train")
        val_metrics = run_epoch(
            model,
            val_loader,
            device=device,
            criterion=criterion,
            amp=False,
            max_batches=config.max_val_batches,
        )
        _require_finite_metrics(val_metrics, context=f"{config.source_name}:epoch{epoch}:model_val")
        row = {"epoch": int(epoch), "train": train_metrics, "model_val": val_metrics}
        curves.append(row)
        save_json(output_dir / "training_curves.json", {"epochs": curves, "selection_metric": "model_val_cross_entropy"})
        torch.save(
            _source_checkpoint_payload(
                checkpoint_model,
                optimizer,
                epoch=epoch,
                config=config,
                metrics=row,
                initialization=initialization,
            ),
            config.last_checkpoint_path,
        )
        val_ce = float(val_metrics["loss"])
        val_accuracy = float(val_metrics["accuracy"])
        improved = val_ce < best_val_ce or (np.isclose(val_ce, best_val_ce) and val_accuracy > best_val_accuracy)
        if improved:
            best_val_ce = val_ce
            best_val_accuracy = val_accuracy
            best_epoch = int(epoch)
            best_row = row
            epochs_without_improvement = 0
            torch.save(
                _source_checkpoint_payload(
                    checkpoint_model,
                    optimizer,
                    epoch=epoch,
                    config=config,
                    metrics=row,
                    initialization=initialization,
                ),
                config.checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
        if int(config.early_stop_patience) >= 0 and epochs_without_improvement >= int(config.early_stop_patience):
            break

    selected_model, selected_payload = load_hlt_model_from_checkpoint(config.checkpoint_path, device=device)
    model_val_prediction_metrics = None
    model_val_prediction_metadata = None
    validation_thresholds = None
    validation_binary_thresholds = None
    if config.evaluate_model_val_predictions:
        model_val_prediction_metrics, model_val_prediction_metadata = _save_source_predictions(
            selected_model,
            val_view,
            config=config,
            split=config.val_split,
            checkpoint_payload=selected_payload,
            device=device,
            batch_size=config.eval_batch_size,
            max_batches=config.max_val_batches,
            seed=int(config.seed) + 303,
        )
        validation_thresholds = model_val_prediction_metrics.get("score_thresholds_by_class")
        validation_binary_thresholds = model_val_prediction_metrics.get("binary_score_thresholds")

    final_test_metrics = None
    final_test_prediction_metadata = None
    if config.evaluate_final_test:
        final_view = _load_source_view(config, config.final_test_split)
        final_test_metrics, final_test_prediction_metadata = _save_source_predictions(
            selected_model,
            final_view,
            config=config,
            split=config.final_test_split,
            checkpoint_payload=selected_payload,
            device=device,
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


class HLTTriViewDataset:
    """Aligned HLT, HLT2 s0p35, and HLT2 s1p00 dataset."""

    def __init__(
        self,
        hlt_view: JetView,
        hlt2_s0p35_view: JetView,
        hlt2_s1p00_view: JetView,
        *,
        max_jets: int | None = None,
    ) -> None:
        require_torch()
        parent = _limit_view(hlt_view, max_jets)
        hlt2_s0p35 = _limit_view(hlt2_s0p35_view, max_jets)
        hlt2_s1p00 = _limit_view(hlt2_s1p00_view, max_jets)
        _check_view(parent, expected_view="fixed_hlt", label="hlt")
        _check_view(hlt2_s0p35, expected_view="hlt2", label="hlt2_s0p35")
        _check_view(hlt2_s1p00, expected_view="hlt2", label="hlt2_s1p00")
        hlt2_s0p35 = _align_secondary_view(parent, hlt2_s0p35, label="hlt2_s0p35")
        hlt2_s1p00 = _align_secondary_view(parent, hlt2_s1p00, label="hlt2_s1p00")
        self.hlt_tokens = np.asarray(parent.tokens, dtype=np.float32)
        self.hlt_mask = np.asarray(parent.mask, dtype=bool)
        self.hlt2_s0p35_tokens = np.asarray(hlt2_s0p35.tokens, dtype=np.float32)
        self.hlt2_s0p35_mask = np.asarray(hlt2_s0p35.mask, dtype=bool)
        self.hlt2_s1p00_tokens = np.asarray(hlt2_s1p00.tokens, dtype=np.float32)
        self.hlt2_s1p00_mask = np.asarray(hlt2_s1p00.mask, dtype=bool)
        self.labels = np.asarray(parent.labels, dtype=np.int64)
        self.jet_ids = list(parent.jet_ids)
        self.split = parent.split
        self.hlt_metadata = dict(parent.metadata)
        self.hlt2_s0p35_metadata = dict(hlt2_s0p35.metadata)
        self.hlt2_s1p00_metadata = dict(hlt2_s1p00.metadata)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "hlt_tokens": self.hlt_tokens[index],
            "hlt_mask": self.hlt_mask[index],
            "hlt2_s0p35_tokens": self.hlt2_s0p35_tokens[index],
            "hlt2_s0p35_mask": self.hlt2_s0p35_mask[index],
            "hlt2_s1p00_tokens": self.hlt2_s1p00_tokens[index],
            "hlt2_s1p00_mask": self.hlt2_s1p00_mask[index],
            "label": np.int64(self.labels[index]),
            "jet_id": self.jet_ids[index],
        }

    def to_metadata(self) -> dict[str, Any]:
        return {
            "contract": "hlt_self_triview_dataset_v1",
            "split": self.split,
            "n_jets": int(len(self.labels)),
            "source_views": ["fixed_hlt", "hlt2_s0p35", "hlt2_s1p00"],
            "hlt_content_hash": self.hlt_metadata.get("hlt_content_hash"),
            "hlt2_s0p35_content_hash": self.hlt2_s0p35_metadata.get("hlt2_content_hash")
            or self.hlt2_s0p35_metadata.get("hlt_content_hash"),
            "hlt2_s1p00_content_hash": self.hlt2_s1p00_metadata.get("hlt2_content_hash")
            or self.hlt2_s1p00_metadata.get("hlt_content_hash"),
            "jet_identity_hash": jet_identity_hash(self.jet_ids),
            "allowed_inputs": HLT_SDV_ALLOWED_INPUTS,
            "deployment_inputs": HLT_TRIVIEW_DEPLOYMENT_INPUTS,
            "requires_offline_inputs": False,
            "requires_teacher_features": False,
        }


def _particle_input_dict(tokens: np.ndarray, mask: np.ndarray, labels: np.ndarray, *, source_view: str) -> dict[str, Any]:
    torch = require_torch()
    inputs = build_particle_transformer_inputs_from_tokens(tokens, mask, labels=labels, source_view=source_view)
    return {
        "points": torch.from_numpy(inputs.pf_points).float(),
        "features": torch.from_numpy(inputs.pf_features).float(),
        "lorentz_vectors": torch.from_numpy(inputs.pf_vectors).float(),
        "mask": torch.from_numpy(inputs.pf_mask).bool(),
    }


def collate_hlt_triview_batch(samples: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("cannot collate an empty HLT tri-view batch")
    torch = require_torch()
    labels = np.asarray([sample["label"] for sample in samples], dtype=np.int64)
    hlt_tokens = np.stack([np.asarray(sample["hlt_tokens"], dtype=np.float32) for sample in samples], axis=0)
    hlt_mask = np.stack([np.asarray(sample["hlt_mask"], dtype=bool) for sample in samples], axis=0)
    hlt2_s0p35_tokens = np.stack(
        [np.asarray(sample["hlt2_s0p35_tokens"], dtype=np.float32) for sample in samples],
        axis=0,
    )
    hlt2_s0p35_mask = np.stack(
        [np.asarray(sample["hlt2_s0p35_mask"], dtype=bool) for sample in samples],
        axis=0,
    )
    hlt2_s1p00_tokens = np.stack(
        [np.asarray(sample["hlt2_s1p00_tokens"], dtype=np.float32) for sample in samples],
        axis=0,
    )
    hlt2_s1p00_mask = np.stack(
        [np.asarray(sample["hlt2_s1p00_mask"], dtype=bool) for sample in samples],
        axis=0,
    )
    jet_ids = [sample["jet_id"] for sample in samples]
    return {
        "hlt_inputs": _particle_input_dict(hlt_tokens, hlt_mask, labels, source_view="fixed_hlt"),
        "hlt2_s0p35_inputs": _particle_input_dict(
            hlt2_s0p35_tokens,
            hlt2_s0p35_mask,
            labels,
            source_view="hlt2",
        ),
        "hlt2_s1p00_inputs": _particle_input_dict(
            hlt2_s1p00_tokens,
            hlt2_s1p00_mask,
            labels,
            source_view="hlt2",
        ),
        "labels": torch.from_numpy(labels).long(),
        "jet_ids": jet_ids,
        "jet_files": [identity.file for identity in jet_ids],
        "jet_entries": torch.tensor([int(identity.entry) for identity in jet_ids], dtype=torch.long),
        "jet_identity_labels": torch.tensor([int(identity.label) for identity in jet_ids], dtype=torch.long),
        "allowed_inputs": HLT_SDV_ALLOWED_INPUTS,
        "deployment_inputs": HLT_TRIVIEW_DEPLOYMENT_INPUTS,
        "requires_offline_inputs": False,
        "requires_teacher_features": False,
    }


def make_hlt_triview_data_loader(
    dataset: HLTTriViewDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    seed: int = 12345,
):
    torch = require_torch()
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_hlt_triview_batch,
        generator=generator,
    )


def move_hlt_triview_batch_to_device(batch: Mapping[str, Any], device) -> dict[str, Any]:
    torch = require_torch()
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, Mapping):
            moved[key] = {
                sub_key: sub_value.to(device, non_blocking=True) if torch.is_tensor(sub_value) else sub_value
                for sub_key, sub_value in value.items()
            }
        else:
            moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def load_hlt_triview_dataset(
    hlt_cache_dir: str | Path,
    hlt2_s0p35_cache_dir: str | Path,
    hlt2_s1p00_cache_dir: str | Path,
    split: str,
    *,
    max_jets: int | None = None,
) -> HLTTriViewDataset:
    return HLTTriViewDataset(
        load_cached_hlt_view(hlt_cache_dir, split, verify_hash=True),
        load_cached_hlt_view(hlt2_s0p35_cache_dir, split, verify_hash=True),
        load_cached_hlt_view(hlt2_s1p00_cache_dir, split, verify_hash=True),
        max_jets=max_jets,
    )


class HLTSelfTriViewFusionModel(_ModuleBase):
    """Three-branch HLT / HLT2-s0p35 / HLT2-s1p00 particle fusion model."""

    def __init__(
        self,
        *,
        num_classes: int = PD10_NUM_CLASSES,
        model_size: str = HLT_SDV_DEFAULT_MODEL_SIZE,
        branch_config: Mapping[str, Any] | None = None,
        fusion_hidden_dim: int = HLT_SDV_DEFAULT_FUSION_HIDDEN_DIM,
        representation_dim: int = PD10_REPRESENTATION_DIM,
        dropout: float = HLT_SDV_DEFAULT_DROPOUT,
    ) -> None:
        torch = require_torch()
        nn = torch.nn
        super().__init__()
        if int(num_classes) != PD10_NUM_CLASSES:
            raise ValueError(f"HLT tri-view is a 10-class model; got num_classes={num_classes}")
        cfg = hlt_sdv_embedding_branch_config(model_size=model_size, overrides=branch_config)
        branch_dim = hlt_sdv_branch_dim_from_config(cfg)
        self.hlt_branch = HLTSDVParticleTransformerEmbeddingBranch(**cfg)
        self.hlt2_s0p35_branch = HLTSDVParticleTransformerEmbeddingBranch(**cfg)
        self.hlt2_s1p00_branch = HLTSDVParticleTransformerEmbeddingBranch(**cfg)
        self.branch_dim = int(branch_dim)
        fusion_input_dim = self.branch_dim * 9
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_input_dim),
            nn.Linear(fusion_input_dim, int(fusion_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(fusion_hidden_dim), int(representation_dim)),
            nn.GELU(),
            nn.LayerNorm(int(representation_dim)),
        )
        self.classifier = nn.Linear(int(representation_dim), int(num_classes))
        self.config = {
            "contract": HLT_TRIVIEW_FUSION_CONTRACT,
            "experiment_name": HLT_TRIVIEW_EXPERIMENT_NAME,
            "experiment_step": HLT_TRIVIEW_STEP,
            "architecture": HLT_TRIVIEW_MODEL_ARCHITECTURE,
            "num_classes": int(num_classes),
            "model_size": str(model_size),
            "branch_config": dict(cfg),
            "branch_dim": int(self.branch_dim),
            "fusion_input_dim": int(fusion_input_dim),
            "fusion_terms": [
                "h_hlt",
                "h_hlt2_s0p35",
                "h_hlt2_s1p00",
                "abs_hlt_minus_s0p35",
                "abs_hlt_minus_s1p00",
                "abs_s0p35_minus_s1p00",
                "hlt_times_s0p35",
                "hlt_times_s1p00",
                "s0p35_times_s1p00",
            ],
            "fusion_hidden_dim": int(fusion_hidden_dim),
            "representation_dim": int(representation_dim),
            "dropout": float(dropout),
            "allowed_inputs": HLT_SDV_ALLOWED_INPUTS,
            "deployment_inputs": HLT_TRIVIEW_DEPLOYMENT_INPUTS,
            "requires_offline_inputs": False,
            "requires_teacher_features": False,
            "requires_deterministic_hlt2_transform": True,
        }

    def branch_parameters(self):
        yield from self.hlt_branch.parameters()
        yield from self.hlt2_s0p35_branch.parameters()
        yield from self.hlt2_s1p00_branch.parameters()

    def head_parameters(self):
        yield from self.fusion.parameters()
        yield from self.classifier.parameters()

    def set_branches_trainable(self, trainable: bool) -> None:
        for parameter in self.branch_parameters():
            parameter.requires_grad_(bool(trainable))

    def forward(self, hlt_inputs: Mapping[str, Any], hlt2_s0p35_inputs: Mapping[str, Any], hlt2_s1p00_inputs: Mapping[str, Any]):
        torch = require_torch()
        h0 = self.hlt_branch(hlt_inputs)
        h1 = self.hlt2_s0p35_branch(hlt2_s0p35_inputs)
        h2 = self.hlt2_s1p00_branch(hlt2_s1p00_inputs)
        for name, embedding in (("hlt", h0), ("hlt2_s0p35", h1), ("hlt2_s1p00", h2)):
            if int(embedding.shape[-1]) != self.branch_dim:
                raise ValueError(f"{name} embedding must have dim {self.branch_dim}; got {tuple(embedding.shape)}")
        fusion_input = torch.cat(
            [
                h0,
                h1,
                h2,
                torch.abs(h0 - h1),
                torch.abs(h0 - h2),
                torch.abs(h1 - h2),
                h0 * h1,
                h0 * h2,
                h1 * h2,
            ],
            dim=1,
        )
        return self.classifier(self.fusion(fusion_input))

    def forward_batch(self, batch: Mapping[str, Any]):
        return self(batch["hlt_inputs"], batch["hlt2_s0p35_inputs"], batch["hlt2_s1p00_inputs"])


def initialize_hlt_triview_branches_from_checkpoints(
    model: HLTSelfTriViewFusionModel,
    *,
    hlt_checkpoint: str | Path,
    hlt2_s0p35_checkpoint: str | Path,
    hlt2_s1p00_checkpoint: str | Path,
    device,
) -> dict[str, Any]:
    return {
        "hlt_branch": load_matching_branch_weights(
            model.hlt_branch,
            hlt_checkpoint,
            device=device,
            branch_label="hlt_branch",
        ),
        "hlt2_s0p35_branch": load_matching_branch_weights(
            model.hlt2_s0p35_branch,
            hlt2_s0p35_checkpoint,
            device=device,
            branch_label="hlt2_s0p35_branch",
        ),
        "hlt2_s1p00_branch": load_matching_branch_weights(
            model.hlt2_s1p00_branch,
            hlt2_s1p00_checkpoint,
            device=device,
            branch_label="hlt2_s1p00_branch",
        ),
        "source": "three_single_view_part_source_checkpoints",
        "hlt_checkpoint": str(hlt_checkpoint),
        "hlt2_s0p35_checkpoint": str(hlt2_s0p35_checkpoint),
        "hlt2_s1p00_checkpoint": str(hlt2_s1p00_checkpoint),
    }


@dataclass(frozen=True)
class HLTTriViewTrainConfig:
    """Training config for the tri-view particle fusion model."""

    output_dir: str
    hlt_cache_dir: str
    hlt2_s0p35_cache_dir: str
    hlt2_s1p00_cache_dir: str
    hlt_source_checkpoint: str
    hlt2_s0p35_source_checkpoint: str
    hlt2_s1p00_source_checkpoint: str
    model_name: str = HLT_TRIVIEW_MODEL_NAME
    train_split: str = "model_train"
    val_split: str = "model_val"
    final_test_split: str = "final_test"
    seed: int = 9201
    batch_size: int = 64
    eval_batch_size: int = 96
    epochs: int = 10
    head_warmup_epochs: int = 1
    head_warmup_lr: float = 1.0e-3
    branch_lr: float = 2.0e-5
    head_lr: float = 3.0e-4
    weight_decay: float = 1.0e-4
    num_workers: int = 0
    device: str = "auto"
    amp: bool = True
    grad_clip_norm: float = 1.0
    early_stop_patience: int = 3
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    max_final_test_batches: int | None = None
    max_train_jets: int | None = HLT_TRIVIEW_DEFAULT_TRAIN_JETS
    max_val_jets: int | None = HLT_TRIVIEW_DEFAULT_VAL_JETS
    max_final_test_jets: int | None = HLT_TRIVIEW_DEFAULT_FINAL_TEST_JETS
    model_size: str = HLT_SDV_DEFAULT_MODEL_SIZE
    compile_model: bool = False
    fusion_hidden_dim: int = HLT_SDV_DEFAULT_FUSION_HIDDEN_DIM
    representation_dim: int = PD10_REPRESENTATION_DIM
    dropout: float = HLT_SDV_DEFAULT_DROPOUT
    evaluate_model_val_predictions: bool = True
    evaluate_final_test: bool = True
    confirm_final_test: bool = False
    overwrite: bool = False

    def __post_init__(self) -> None:
        if (self.train_split, self.val_split, self.final_test_split) != PD10_SPLIT_ORDER:
            raise ValueError(f"HLT tri-view split order must be {PD10_SPLIT_ORDER}")
        if self.evaluate_final_test and not bool(self.confirm_final_test):
            raise ValueError("HLT tri-view final-test evaluation requires confirm_final_test=True")
        if int(self.batch_size) <= 0 or int(self.eval_batch_size) <= 0:
            raise ValueError("batch_size and eval_batch_size must be positive")
        if int(self.epochs) <= 0:
            raise ValueError("epochs must be positive")
        if int(self.head_warmup_epochs) < 0:
            raise ValueError("head_warmup_epochs cannot be negative")
        for split, value in (
            ("model_train", self.max_train_jets),
            ("model_val", self.max_val_jets),
            ("final_test", self.max_final_test_jets),
        ):
            if value is not None and int(value) > int(PD10_SPLIT_SIZES[split]):
                raise ValueError(f"max jets for {split} cannot exceed {PD10_SPLIT_SIZES[split]}")
        object.__setattr__(self, "seed", int(self.seed))

    @property
    def checkpoint_path(self) -> Path:
        return Path(self.output_dir) / "best_model_val.pt"

    @property
    def last_checkpoint_path(self) -> Path:
        return Path(self.output_dir) / "last.pt"

    @property
    def prediction_dir(self) -> Path:
        return Path(self.output_dir) / "predictions"


def _max_jets_for_triview(config: HLTTriViewTrainConfig, split: str) -> int | None:
    if split == config.train_split:
        return config.max_train_jets
    if split == config.val_split:
        return config.max_val_jets
    if split == config.final_test_split:
        return config.max_final_test_jets
    return None


def load_hlt_triview_dataset_for_config(config: HLTTriViewTrainConfig, split: str) -> HLTTriViewDataset:
    return load_hlt_triview_dataset(
        config.hlt_cache_dir,
        config.hlt2_s0p35_cache_dir,
        config.hlt2_s1p00_cache_dir,
        split,
        max_jets=_max_jets_for_triview(config, split),
    )


def _optimizer_for_triview_stage(model, config: HLTTriViewTrainConfig, *, stage: str):
    torch = require_torch()
    if stage == "head_warmup":
        model.set_branches_trainable(False)
        return torch.optim.AdamW(
            [parameter for parameter in model.head_parameters() if parameter.requires_grad],
            lr=float(config.head_warmup_lr),
            weight_decay=float(config.weight_decay),
        )
    model.set_branches_trainable(True)
    return torch.optim.AdamW(
        [
            {"params": list(model.branch_parameters()), "lr": float(config.branch_lr)},
            {"params": list(model.head_parameters()), "lr": float(config.head_lr)},
        ],
        weight_decay=float(config.weight_decay),
    )


def _amp_autocast_context(torch, enabled: bool):
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


def run_hlt_triview_epoch(
    model,
    loader,
    *,
    device,
    optimizer=None,
    scaler=None,
    amp: bool = False,
    grad_clip_norm: float = 0.0,
    max_batches: int | None = None,
) -> dict[str, Any]:
    torch = require_torch()
    criterion = torch.nn.CrossEntropyLoss()
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            batch = move_hlt_triview_batch_to_device(batch, device)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            autocast_enabled = bool(amp and device.type == "cuda")
            with _amp_autocast_context(torch, autocast_enabled):
                logits = model.forward_batch(batch)
                loss = criterion(logits, batch["labels"])
            if not bool(torch.isfinite(logits).all()):
                raise FloatingPointError(
                    f"HLT tri-view {'train' if is_train else 'eval'} batch {batch_index} produced non-finite logits"
                )
            if not bool(torch.isfinite(loss).all()):
                raise FloatingPointError(
                    f"HLT tri-view {'train' if is_train else 'eval'} batch {batch_index} produced non-finite loss"
                )
            if is_train:
                if scaler is not None and autocast_enabled:
                    scaler.scale(loss).backward()
                    if float(grad_clip_norm) > 0.0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if float(grad_clip_norm) > 0.0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
                    optimizer.step()
            labels = batch["labels"]
            batch_size = int(labels.numel())
            total_loss += float(loss.detach().cpu().item()) * batch_size
            total_correct += int((logits.detach().argmax(dim=1) == labels).sum().detach().cpu().item())
            total_seen += batch_size
    if total_seen == 0:
        return {"loss": float("nan"), "cross_entropy": float("nan"), "accuracy": 0.0, "n_jets": 0}
    ce = total_loss / float(total_seen)
    metrics = {"loss": float(ce), "cross_entropy": float(ce), "accuracy": total_correct / float(total_seen), "n_jets": int(total_seen)}
    _require_finite_metrics(metrics, context="HLT tri-view epoch")
    return metrics


def _triview_checkpoint_payload(
    model,
    optimizer,
    *,
    epoch: int,
    config: HLTTriViewTrainConfig,
    metrics: Mapping[str, Any],
    branch_initialization: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint_model = getattr(model, "_orig_mod", model)
    return {
        "contract": HLT_TRIVIEW_FUSION_CONTRACT,
        "experiment_name": HLT_TRIVIEW_EXPERIMENT_NAME,
        "experiment_step": HLT_TRIVIEW_STEP,
        "model_name": config.model_name,
        "architecture": HLT_TRIVIEW_MODEL_ARCHITECTURE,
        "epoch": int(epoch),
        "model_state_dict": checkpoint_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "config": asdict(config),
        "metrics": dict(metrics),
        "model_config": getattr(checkpoint_model, "config", {}),
        "branch_initialization": dict(branch_initialization),
        "label_names": list(LABEL_NAMES),
        "pf_feature_names": list(PF_FEATURE_NAMES),
        "allowed_inputs": HLT_SDV_ALLOWED_INPUTS,
        "deployment_inputs": HLT_TRIVIEW_DEPLOYMENT_INPUTS,
        "requires_offline_inputs": False,
        "requires_teacher_features": False,
        "requires_deterministic_hlt2_transform": True,
        "selection_metric": "model_val_cross_entropy",
        "selection_tie_breaker": "model_val_accuracy",
        "no_final_test_used_for_selection": True,
    }


def load_hlt_triview_model_from_checkpoint(checkpoint: str | Path, *, device):
    torch = require_torch()
    payload = torch.load(checkpoint, map_location=device)
    model_config = dict(payload.get("model_config") or {})
    model = HLTSelfTriViewFusionModel(
        num_classes=int(model_config.get("num_classes", PD10_NUM_CLASSES)),
        model_size=str(model_config.get("model_size", HLT_SDV_DEFAULT_MODEL_SIZE)),
        branch_config=model_config.get("branch_config"),
        fusion_hidden_dim=int(model_config.get("fusion_hidden_dim", HLT_SDV_DEFAULT_FUSION_HIDDEN_DIM)),
        representation_dim=int(model_config.get("representation_dim", PD10_REPRESENTATION_DIM)),
        dropout=float(model_config.get("dropout", HLT_SDV_DEFAULT_DROPOUT)),
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()
    return model, payload


def collect_hlt_triview_outputs(
    model,
    dataset: HLTTriViewDataset,
    *,
    batch_size: int,
    num_workers: int,
    device,
    seed: int,
    max_batches: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[Any], dict[str, Any]]:
    torch = require_torch()
    loader = make_hlt_triview_data_loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        seed=seed,
    )
    logits_chunks: list[np.ndarray] = []
    labels_chunks: list[np.ndarray] = []
    jet_ids: list[Any] = []
    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            batch = move_hlt_triview_batch_to_device(batch, device)
            logits = model.forward_batch(batch)
            logits_chunks.append(logits.detach().cpu().float().numpy())
            labels_chunks.append(batch["labels"].detach().cpu().long().numpy())
            jet_ids.extend(batch["jet_ids"])
    if not logits_chunks:
        raise ValueError("No HLT tri-view prediction batches were produced")
    logits_np = np.concatenate(logits_chunks, axis=0).astype(np.float32)
    _require_finite_logits(logits_np, context=f"HLT tri-view:{dataset.split}:prediction")
    labels_np = np.concatenate(labels_chunks, axis=0).astype(np.int64)
    return logits_np, labels_np, jet_ids, pd10_prediction_metrics_from_logits(logits_np, labels_np)


def collect_and_save_hlt_triview_predictions(
    model,
    dataset: HLTTriViewDataset,
    *,
    config: HLTTriViewTrainConfig,
    split: str,
    checkpoint_payload: Mapping[str, Any] | None,
    batch_size: int,
    num_workers: int,
    device,
    seed: int,
    max_batches: int | None = None,
    validation_thresholds_by_class: Mapping[str, Any] | None = None,
    validation_binary_thresholds: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    logits, labels, jet_ids, metrics = collect_hlt_triview_outputs(
        model,
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
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
    block = PredictionBlock(
        model_name=config.model_name,
        split=split,
        logits=logits,
        probs=softmax_np(logits),
        labels=labels,
        jet_ids=jet_ids,
        metadata={
            "contract": HLT_TRIVIEW_PREDICTION_CONTRACT,
            "experiment_name": HLT_TRIVIEW_EXPERIMENT_NAME,
            "experiment_step": HLT_TRIVIEW_STEP,
            "model_name": config.model_name,
            "architecture": HLT_TRIVIEW_MODEL_ARCHITECTURE,
            "split": split,
            "checkpoint": str(config.checkpoint_path),
            "checkpoint_sha256": sha256_file(config.checkpoint_path),
            "checkpoint_epoch": None if checkpoint_payload is None else checkpoint_payload.get("epoch"),
            "dataset": dataset.to_metadata(),
            "rich_metrics": dict(metrics),
            "allowed_inputs": HLT_SDV_ALLOWED_INPUTS,
            "deployment_inputs": HLT_TRIVIEW_DEPLOYMENT_INPUTS,
            "requires_offline_inputs": False,
            "requires_teacher_features": False,
            "requires_deterministic_hlt2_transform": True,
            "no_final_test_used_for_selection": True,
        },
    )
    metadata = save_prediction_block(block, config.prediction_dir, overwrite=bool(config.overwrite))
    return metrics, metadata


def train_hlt_triview_model(config: HLTTriViewTrainConfig) -> dict[str, Any]:
    torch = require_torch()
    set_training_seed(config.seed)
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir)
    if config.checkpoint_path.exists() and not config.overwrite:
        raise FileExistsError(f"HLT tri-view checkpoint already exists: {config.checkpoint_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = load_hlt_triview_dataset_for_config(config, config.train_split)
    val_dataset = load_hlt_triview_dataset_for_config(config, config.val_split)
    train_loader = make_hlt_triview_data_loader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        seed=int(config.seed) + 101,
    )
    val_loader = make_hlt_triview_data_loader(
        val_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        seed=int(config.seed) + 202,
    )

    model = HLTSelfTriViewFusionModel(
        num_classes=PD10_NUM_CLASSES,
        model_size=config.model_size,
        fusion_hidden_dim=config.fusion_hidden_dim,
        representation_dim=config.representation_dim,
        dropout=config.dropout,
    ).to(device)
    branch_initialization = initialize_hlt_triview_branches_from_checkpoints(
        model,
        hlt_checkpoint=config.hlt_source_checkpoint,
        hlt2_s0p35_checkpoint=config.hlt2_s0p35_source_checkpoint,
        hlt2_s1p00_checkpoint=config.hlt2_s1p00_source_checkpoint,
        device=device,
    )
    checkpoint_model = model
    if config.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)
        checkpoint_model = getattr(model, "_orig_mod", checkpoint_model)

    scaler = None
    if bool(config.amp and device.type == "cuda"):
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            scaler = torch.amp.GradScaler("cuda", enabled=True)
        else:
            scaler = torch.cuda.amp.GradScaler(enabled=True)

    run_metadata = {
        "ok": True,
        "contract": HLT_TRIVIEW_FUSION_CONTRACT,
        "experiment_name": HLT_TRIVIEW_EXPERIMENT_NAME,
        "experiment_step": HLT_TRIVIEW_STEP,
        "model_name": config.model_name,
        "architecture": HLT_TRIVIEW_MODEL_ARCHITECTURE,
        "config": asdict(config),
        "train_dataset": train_dataset.to_metadata(),
        "model_val_dataset": val_dataset.to_metadata(),
        "branch_initialization": branch_initialization,
        "allowed_inputs": HLT_SDV_ALLOWED_INPUTS,
        "deployment_inputs": HLT_TRIVIEW_DEPLOYMENT_INPUTS,
        "requires_offline_inputs": False,
        "requires_teacher_features": False,
        "requires_deterministic_hlt2_transform": True,
        "selection_metric": "model_val_cross_entropy",
        "selection_tie_breaker": "model_val_accuracy",
        "no_final_test_used_for_selection": True,
    }
    save_json(output_dir / "config.json", run_metadata)

    curves: list[dict[str, Any]] = []
    best_val_ce = float("inf")
    best_val_accuracy = -1.0
    best_epoch = -1
    best_row: dict[str, Any] | None = None
    epochs_without_improvement = 0
    current_stage: str | None = None
    optimizer = None
    for epoch in range(1, int(config.epochs) + 1):
        stage = "head_warmup" if epoch <= int(config.head_warmup_epochs) else "full_finetune"
        if optimizer is None or stage != current_stage:
            optimizer = _optimizer_for_triview_stage(model, config, stage=stage)
            current_stage = stage
        train_metrics = run_hlt_triview_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            amp=config.amp,
            grad_clip_norm=config.grad_clip_norm,
            max_batches=config.max_train_batches,
        )
        val_metrics = run_hlt_triview_epoch(
            model,
            val_loader,
            device=device,
            amp=False,
            max_batches=config.max_val_batches,
        )
        row = {"epoch": int(epoch), "stage": stage, "train": train_metrics, "model_val": val_metrics}
        curves.append(row)
        save_json(output_dir / "training_curves.json", {"epochs": curves, "selection_metric": "model_val_cross_entropy"})
        torch.save(
            _triview_checkpoint_payload(
                checkpoint_model,
                optimizer,
                epoch=epoch,
                config=config,
                metrics=row,
                branch_initialization=branch_initialization,
            ),
            config.last_checkpoint_path,
        )
        val_ce = float(val_metrics["cross_entropy"])
        val_accuracy = float(val_metrics["accuracy"])
        improved = val_ce < best_val_ce or (np.isclose(val_ce, best_val_ce) and val_accuracy > best_val_accuracy)
        if improved:
            best_val_ce = val_ce
            best_val_accuracy = val_accuracy
            best_epoch = int(epoch)
            best_row = row
            epochs_without_improvement = 0
            torch.save(
                _triview_checkpoint_payload(
                    checkpoint_model,
                    optimizer,
                    epoch=epoch,
                    config=config,
                    metrics=row,
                    branch_initialization=branch_initialization,
                ),
                config.checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
        if int(config.early_stop_patience) >= 0 and epochs_without_improvement >= int(config.early_stop_patience):
            break

    selected_model, selected_payload = load_hlt_triview_model_from_checkpoint(config.checkpoint_path, device=device)
    model_val_prediction_metrics = None
    model_val_prediction_metadata = None
    validation_thresholds = None
    validation_binary_thresholds = None
    if config.evaluate_model_val_predictions:
        model_val_prediction_metrics, model_val_prediction_metadata = collect_and_save_hlt_triview_predictions(
            selected_model,
            val_dataset,
            config=config,
            split=config.val_split,
            checkpoint_payload=selected_payload,
            batch_size=config.eval_batch_size,
            num_workers=config.num_workers,
            device=device,
            seed=int(config.seed) + 303,
            max_batches=config.max_val_batches,
        )
        validation_thresholds = model_val_prediction_metrics.get("score_thresholds_by_class")
        validation_binary_thresholds = model_val_prediction_metrics.get("binary_score_thresholds")

    final_test_metrics = None
    final_test_prediction_metadata = None
    if config.evaluate_final_test:
        final_test_dataset = load_hlt_triview_dataset_for_config(config, config.final_test_split)
        final_test_metrics, final_test_prediction_metadata = collect_and_save_hlt_triview_predictions(
            selected_model,
            final_test_dataset,
            config=config,
            split=config.final_test_split,
            checkpoint_payload=selected_payload,
            batch_size=config.eval_batch_size,
            num_workers=config.num_workers,
            device=device,
            seed=int(config.seed) + 404,
            max_batches=config.max_final_test_batches,
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


__all__ = [
    "HLTTriViewSourceConfig",
    "HLTTriViewTrainConfig",
    "HLTSelfTriViewFusionModel",
    "HLT_TRIVIEW_DEFAULT_FINAL_TEST_JETS",
    "HLT_TRIVIEW_DEFAULT_TRAIN_JETS",
    "HLT_TRIVIEW_DEFAULT_VAL_JETS",
    "HLT_TRIVIEW_DEPLOYMENT_INPUTS",
    "HLT_TRIVIEW_FUSION_CONTRACT",
    "HLT_TRIVIEW_MODEL_ARCHITECTURE",
    "HLT_TRIVIEW_MODEL_NAME",
    "HLT_TRIVIEW_SOURCE_CONTRACT",
    "collate_hlt_triview_batch",
    "initialize_hlt_triview_branches_from_checkpoints",
    "load_hlt_triview_dataset",
    "load_hlt_triview_model_from_checkpoint",
    "make_hlt_triview_data_loader",
    "move_hlt_triview_batch_to_device",
    "run_hlt_triview_epoch",
    "train_hlt_triview_model",
    "train_hlt_triview_source",
]
