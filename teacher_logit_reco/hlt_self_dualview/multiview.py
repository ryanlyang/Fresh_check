"""Deployable HLT N-view particle fusion training.

This module generalizes the tri-view particle fusion model to HLT plus an
arbitrary number of deterministic HLT2 views.  It intentionally keeps the same
training contract as the tri-view code: source branches are initialized from
single-view ParT checkpoints, the fusion head warms up first, selection is done
on model-val cross entropy, and final-test is evaluated only after selection.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from jetclass_fresh.fusion import PredictionBlock, save_prediction_block, softmax_np
from jetclass_fresh.hlt_baseline import (
    make_data_loader,
    require_torch,
    resolve_device,
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

from .config import HLT_SDV_ALLOWED_INPUTS, HLT_SDV_EXPERIMENT_NAME
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
from .tri_view import _align_secondary_view, _check_view, _limit_view

try:  # Keep imports lightweight on machines without torch.
    import torch as _torch
except ImportError:  # pragma: no cover - environment dependent
    _torch = None

if _torch is None:  # pragma: no cover - environment dependent
    class _ModuleBase:
        pass
else:
    _ModuleBase = _torch.nn.Module


HLT_MULTIVIEW_EXPERIMENT_NAME = "hlt_self_multiview_10class"
HLT_MULTIVIEW_STEP = "hlt_self_multiview_particle_fusion_v1"
HLT_MULTIVIEW_FUSION_CONTRACT = "hlt_self_multiview_particle_fusion_v1"
HLT_MULTIVIEW_PREDICTION_CONTRACT = "hlt_self_multiview_prediction_cache_v1"
HLT_MULTIVIEW_MODEL_ARCHITECTURE = "deployable_hlt_multiview_pairwise_abs_product"
HLT_MULTIVIEW_DEPLOYMENT_INPUTS = "HLT_plus_deterministic_HLT2_multiview"
HLT_MULTIVIEW_REPORT = "hlt_multiview_report.json"


def _normalize_view_name(name: str) -> str:
    value = str(name).strip()
    if not value:
        raise ValueError("view name cannot be empty")
    if any(ch.isspace() for ch in value):
        raise ValueError(f"view name cannot contain whitespace: {name!r}")
    return value


@dataclass(frozen=True)
class HLTMultiViewSpec:
    """One deployable particle view and its branch initializer."""

    name: str
    source_view: str
    cache_dir: str
    checkpoint: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _normalize_view_name(self.name))
        source_view = str(self.source_view).strip()
        if source_view not in {"fixed_hlt", "hlt2"}:
            raise ValueError("source_view must be 'fixed_hlt' or 'hlt2'")
        object.__setattr__(self, "source_view", source_view)
        if not str(self.cache_dir).strip():
            raise ValueError("cache_dir cannot be empty")
        if not str(self.checkpoint).strip():
            raise ValueError("checkpoint cannot be empty")


class HLTMultiViewDataset:
    """Aligned dataset for HLT plus deterministic HLT2 views."""

    def __init__(
        self,
        views: Sequence[JetView],
        specs: Sequence[HLTMultiViewSpec],
        *,
        max_jets: int | None = None,
    ) -> None:
        require_torch()
        if len(views) != len(specs):
            raise ValueError("views/specs length mismatch")
        if len(views) < 2:
            raise ValueError("HLT multi-view fusion needs at least two views")
        if specs[0].source_view != "fixed_hlt":
            raise ValueError("first HLT multi-view spec must be the deployable fixed_hlt parent")
        parent = _limit_view(views[0], max_jets)
        _check_view(parent, expected_view="fixed_hlt", label=specs[0].name)
        aligned = [parent]
        for view, spec in zip(views[1:], specs[1:]):
            limited = _limit_view(view, max_jets)
            _check_view(limited, expected_view=spec.source_view, label=spec.name)
            aligned.append(_align_secondary_view(parent, limited, label=spec.name))
        self.view_names = tuple(spec.name for spec in specs)
        self.source_views = tuple(spec.source_view for spec in specs)
        self.tokens_by_view = {
            spec.name: np.asarray(view.tokens, dtype=np.float32)
            for spec, view in zip(specs, aligned)
        }
        self.mask_by_view = {
            spec.name: np.asarray(view.mask, dtype=bool)
            for spec, view in zip(specs, aligned)
        }
        self.metadata_by_view = {
            spec.name: dict(view.metadata)
            for spec, view in zip(specs, aligned)
        }
        self.labels = np.asarray(parent.labels, dtype=np.int64)
        self.jet_ids = list(parent.jet_ids)
        self.split = parent.split

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "views": {
                name: {
                    "tokens": self.tokens_by_view[name][index],
                    "mask": self.mask_by_view[name][index],
                }
                for name in self.view_names
            },
            "label": np.int64(self.labels[index]),
            "jet_id": self.jet_ids[index],
        }

    def to_metadata(self) -> dict[str, Any]:
        content_hashes = {}
        for name, metadata in self.metadata_by_view.items():
            content_hashes[name] = metadata.get("hlt2_content_hash") or metadata.get("hlt_content_hash")
        return {
            "contract": "hlt_self_multiview_dataset_v1",
            "split": self.split,
            "n_jets": int(len(self.labels)),
            "view_names": list(self.view_names),
            "source_views": list(self.source_views),
            "content_hashes": content_hashes,
            "jet_identity_hash": jet_identity_hash(self.jet_ids),
            "allowed_inputs": HLT_SDV_ALLOWED_INPUTS,
            "deployment_inputs": HLT_MULTIVIEW_DEPLOYMENT_INPUTS,
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


def collate_hlt_multiview_batch(samples: list[Mapping[str, Any]], *, specs: Sequence[HLTMultiViewSpec]) -> dict[str, Any]:
    if not samples:
        raise ValueError("cannot collate an empty HLT multi-view batch")
    torch = require_torch()
    labels = np.asarray([sample["label"] for sample in samples], dtype=np.int64)
    view_inputs = {}
    for spec in specs:
        tokens = np.stack(
            [np.asarray(sample["views"][spec.name]["tokens"], dtype=np.float32) for sample in samples],
            axis=0,
        )
        mask = np.stack(
            [np.asarray(sample["views"][spec.name]["mask"], dtype=bool) for sample in samples],
            axis=0,
        )
        view_inputs[spec.name] = _particle_input_dict(tokens, mask, labels, source_view=spec.source_view)
    jet_ids = [sample["jet_id"] for sample in samples]
    return {
        "view_inputs": view_inputs,
        "labels": torch.from_numpy(labels).long(),
        "jet_ids": jet_ids,
        "jet_files": [identity.file for identity in jet_ids],
        "jet_entries": torch.tensor([int(identity.entry) for identity in jet_ids], dtype=torch.long),
        "jet_identity_labels": torch.tensor([int(identity.label) for identity in jet_ids], dtype=torch.long),
        "allowed_inputs": HLT_SDV_ALLOWED_INPUTS,
        "deployment_inputs": HLT_MULTIVIEW_DEPLOYMENT_INPUTS,
        "requires_offline_inputs": False,
        "requires_teacher_features": False,
    }


def make_hlt_multiview_data_loader(
    dataset: HLTMultiViewDataset,
    specs: Sequence[HLTMultiViewSpec],
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
        collate_fn=partial(collate_hlt_multiview_batch, specs=tuple(specs)),
        generator=generator,
    )


def move_hlt_multiview_batch_to_device(batch: Mapping[str, Any], device) -> dict[str, Any]:
    torch = require_torch()
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, Mapping):
            moved[key] = {
                sub_key: (
                    {
                        leaf_key: leaf_value.to(device, non_blocking=True) if torch.is_tensor(leaf_value) else leaf_value
                        for leaf_key, leaf_value in sub_value.items()
                    }
                    if isinstance(sub_value, Mapping)
                    else sub_value.to(device, non_blocking=True)
                    if torch.is_tensor(sub_value)
                    else sub_value
                )
                for sub_key, sub_value in value.items()
            }
        else:
            moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def load_hlt_multiview_dataset(
    specs: Sequence[HLTMultiViewSpec],
    split: str,
    *,
    max_jets: int | None = None,
) -> HLTMultiViewDataset:
    views = [load_cached_hlt_view(spec.cache_dir, split, verify_hash=True) for spec in specs]
    return HLTMultiViewDataset(views, specs, max_jets=max_jets)


class HLTSelfMultiViewFusionModel(_ModuleBase):
    """N-branch HLT / HLT2 particle fusion model."""

    def __init__(
        self,
        *,
        view_names: Sequence[str],
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
            raise ValueError(f"HLT multi-view is a 10-class model; got num_classes={num_classes}")
        names = tuple(_normalize_view_name(name) for name in view_names)
        if len(names) < 2:
            raise ValueError("HLT multi-view fusion needs at least two views")
        if len(set(names)) != len(names):
            raise ValueError(f"view names must be unique, got {names!r}")
        cfg = hlt_sdv_embedding_branch_config(model_size=model_size, overrides=branch_config)
        branch_dim = hlt_sdv_branch_dim_from_config(cfg)
        self.view_names = names
        self.branches = nn.ModuleDict(
            {name: HLTSDVParticleTransformerEmbeddingBranch(**cfg) for name in names}
        )
        self.branch_dim = int(branch_dim)
        n_views = len(names)
        pair_count = n_views * (n_views - 1) // 2
        fusion_terms = n_views + 2 * pair_count
        fusion_input_dim = self.branch_dim * fusion_terms
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
            "contract": HLT_MULTIVIEW_FUSION_CONTRACT,
            "experiment_name": HLT_MULTIVIEW_EXPERIMENT_NAME,
            "experiment_step": HLT_MULTIVIEW_STEP,
            "architecture": HLT_MULTIVIEW_MODEL_ARCHITECTURE,
            "num_classes": int(num_classes),
            "model_size": str(model_size),
            "view_names": list(names),
            "n_views": int(n_views),
            "branch_config": dict(cfg),
            "branch_dim": int(self.branch_dim),
            "fusion_input_dim": int(fusion_input_dim),
            "fusion_terms": int(fusion_terms),
            "pair_count": int(pair_count),
            "fusion_hidden_dim": int(fusion_hidden_dim),
            "representation_dim": int(representation_dim),
            "dropout": float(dropout),
            "allowed_inputs": HLT_SDV_ALLOWED_INPUTS,
            "deployment_inputs": HLT_MULTIVIEW_DEPLOYMENT_INPUTS,
            "requires_offline_inputs": False,
            "requires_teacher_features": False,
            "requires_deterministic_hlt2_transform": True,
        }

    def branch_parameters(self):
        yield from self.branches.parameters()

    def head_parameters(self):
        yield from self.fusion.parameters()
        yield from self.classifier.parameters()

    def set_branches_trainable(self, trainable: bool) -> None:
        for parameter in self.branch_parameters():
            parameter.requires_grad_(bool(trainable))

    def forward(self, view_inputs: Mapping[str, Mapping[str, Any]]):
        torch = require_torch()
        embeddings = []
        for name in self.view_names:
            embedding = self.branches[name](view_inputs[name])
            if int(embedding.shape[-1]) != self.branch_dim:
                raise ValueError(f"{name} embedding must have dim {self.branch_dim}; got {tuple(embedding.shape)}")
            embeddings.append(embedding)
        fusion_chunks = list(embeddings)
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                fusion_chunks.append(torch.abs(embeddings[i] - embeddings[j]))
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                fusion_chunks.append(embeddings[i] * embeddings[j])
        fusion_input = torch.cat(fusion_chunks, dim=1)
        return self.classifier(self.fusion(fusion_input))

    def forward_batch(self, batch: Mapping[str, Any]):
        return self(batch["view_inputs"])


def initialize_hlt_multiview_branches_from_checkpoints(
    model: HLTSelfMultiViewFusionModel,
    specs: Sequence[HLTMultiViewSpec],
    *,
    device,
) -> dict[str, Any]:
    initialization = {
        "source": f"{len(specs)}_single_view_part_source_checkpoints",
        "view_names": [spec.name for spec in specs],
        "checkpoints": {spec.name: str(spec.checkpoint) for spec in specs},
    }
    for spec in specs:
        initialization[f"{spec.name}_branch"] = load_matching_branch_weights(
            model.branches[spec.name],
            spec.checkpoint,
            device=device,
            branch_label=f"{spec.name}_branch",
        )
    return initialization


@dataclass(frozen=True)
class HLTMultiViewTrainConfig:
    """Training config for N-view particle fusion."""

    output_dir: str
    view_specs: tuple[HLTMultiViewSpec, ...]
    model_name: str
    train_split: str = "model_train"
    val_split: str = "model_val"
    final_test_split: str = "final_test"
    seed: int = 9301
    batch_size: int = 48
    eval_batch_size: int = 64
    epochs: int = 10
    head_warmup_epochs: int = 1
    head_warmup_lr: float = 3.0e-4
    branch_lr: float = 2.0e-5
    head_lr: float = 3.0e-4
    weight_decay: float = 1.0e-4
    num_workers: int = 0
    device: str = "auto"
    amp: bool = False
    grad_clip_norm: float = 1.0
    early_stop_patience: int = 3
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    max_final_test_batches: int | None = None
    max_train_jets: int | None = int(PD10_SPLIT_SIZES["model_train"])
    max_val_jets: int | None = int(PD10_SPLIT_SIZES["model_val"])
    max_final_test_jets: int | None = int(PD10_SPLIT_SIZES["final_test"])
    model_size: str = HLT_SDV_DEFAULT_MODEL_SIZE
    compile_model: bool = False
    fusion_hidden_dim: int = HLT_SDV_DEFAULT_FUSION_HIDDEN_DIM
    representation_dim: int = PD10_REPRESENTATION_DIM
    dropout: float = HLT_SDV_DEFAULT_DROPOUT
    evaluate_model_val_predictions: bool = True
    evaluate_final_test: bool = True
    confirm_final_test: bool = False
    overwrite: bool = False
    experiment_name: str = HLT_MULTIVIEW_EXPERIMENT_NAME
    experiment_step: str = HLT_MULTIVIEW_STEP
    fusion_contract: str = HLT_MULTIVIEW_FUSION_CONTRACT
    prediction_contract: str = HLT_MULTIVIEW_PREDICTION_CONTRACT
    allowed_inputs: str = HLT_SDV_ALLOWED_INPUTS
    deployment_inputs: str = HLT_MULTIVIEW_DEPLOYMENT_INPUTS

    def __post_init__(self) -> None:
        if (self.train_split, self.val_split, self.final_test_split) != PD10_SPLIT_ORDER:
            raise ValueError(f"HLT multi-view split order must be {PD10_SPLIT_ORDER}")
        if len(self.view_specs) < 2:
            raise ValueError("HLT multi-view fusion needs at least two view specs")
        names = tuple(spec.name for spec in self.view_specs)
        if len(set(names)) != len(names):
            raise ValueError(f"view names must be unique, got {names!r}")
        if self.view_specs[0].source_view != "fixed_hlt":
            raise ValueError("first view spec must be the deployable fixed_hlt parent")
        if self.evaluate_final_test and not bool(self.confirm_final_test):
            raise ValueError("HLT multi-view final-test evaluation requires confirm_final_test=True")
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
        object.__setattr__(self, "model_name", _normalize_view_name(self.model_name))
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


def _max_jets_for_multiview(config: HLTMultiViewTrainConfig, split: str) -> int | None:
    if split == config.train_split:
        return config.max_train_jets
    if split == config.val_split:
        return config.max_val_jets
    if split == config.final_test_split:
        return config.max_final_test_jets
    return None


def load_hlt_multiview_dataset_for_config(config: HLTMultiViewTrainConfig, split: str) -> HLTMultiViewDataset:
    return load_hlt_multiview_dataset(
        config.view_specs,
        split,
        max_jets=_max_jets_for_multiview(config, split),
    )


def _optimizer_for_multiview_stage(model, config: HLTMultiViewTrainConfig, *, stage: str):
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
    raise FloatingPointError(
        f"{context} produced non-finite logits: bad_rows={int(bad_rows.shape[0])}/"
        f"{int(logits.shape[0])}, first_bad_row={first_bad}"
    )


def run_hlt_multiview_epoch(
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
            batch = move_hlt_multiview_batch_to_device(batch, device)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            autocast_enabled = bool(amp and device.type == "cuda")
            with _amp_autocast_context(torch, autocast_enabled):
                logits = model.forward_batch(batch)
                loss = criterion(logits, batch["labels"])
            if not bool(torch.isfinite(logits).all()):
                raise FloatingPointError(
                    f"HLT multi-view {'train' if is_train else 'eval'} batch {batch_index} produced non-finite logits"
                )
            if not bool(torch.isfinite(loss).all()):
                raise FloatingPointError(
                    f"HLT multi-view {'train' if is_train else 'eval'} batch {batch_index} produced non-finite loss"
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
    _require_finite_metrics(metrics, context="HLT multi-view epoch")
    return metrics


def _multiview_checkpoint_payload(
    model,
    optimizer,
    *,
    epoch: int,
    config: HLTMultiViewTrainConfig,
    metrics: Mapping[str, Any],
    branch_initialization: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint_model = getattr(model, "_orig_mod", model)
    return {
        "contract": HLT_MULTIVIEW_FUSION_CONTRACT,
        "experiment_name": HLT_MULTIVIEW_EXPERIMENT_NAME,
        "experiment_step": HLT_MULTIVIEW_STEP,
        "model_name": config.model_name,
        "architecture": HLT_MULTIVIEW_MODEL_ARCHITECTURE,
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
        "deployment_inputs": HLT_MULTIVIEW_DEPLOYMENT_INPUTS,
        "requires_offline_inputs": False,
        "requires_teacher_features": False,
        "requires_deterministic_hlt2_transform": True,
        "selection_metric": "model_val_cross_entropy",
        "selection_tie_breaker": "model_val_accuracy",
        "no_final_test_used_for_selection": True,
    }


def load_hlt_multiview_model_from_checkpoint(checkpoint: str | Path, *, device):
    torch = require_torch()
    payload = torch.load(checkpoint, map_location=device)
    model_config = dict(payload.get("model_config") or {})
    model = HLTSelfMultiViewFusionModel(
        view_names=tuple(model_config["view_names"]),
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


def collect_hlt_multiview_outputs(
    model,
    dataset: HLTMultiViewDataset,
    specs: Sequence[HLTMultiViewSpec],
    *,
    batch_size: int,
    num_workers: int,
    device,
    seed: int,
    max_batches: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[Any], dict[str, Any]]:
    loader = make_hlt_multiview_data_loader(
        dataset,
        specs,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        seed=seed,
    )
    logits_chunks: list[np.ndarray] = []
    labels_chunks: list[np.ndarray] = []
    jet_ids: list[Any] = []
    model.eval()
    torch = require_torch()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            batch = move_hlt_multiview_batch_to_device(batch, device)
            logits = model.forward_batch(batch)
            logits_chunks.append(logits.detach().cpu().float().numpy())
            labels_chunks.append(batch["labels"].detach().cpu().long().numpy())
            jet_ids.extend(batch["jet_ids"])
    if not logits_chunks:
        raise ValueError("No HLT multi-view prediction batches were produced")
    logits_np = np.concatenate(logits_chunks, axis=0).astype(np.float32)
    _require_finite_logits(logits_np, context=f"HLT multi-view:{dataset.split}:prediction")
    labels_np = np.concatenate(labels_chunks, axis=0).astype(np.int64)
    return logits_np, labels_np, jet_ids, pd10_prediction_metrics_from_logits(logits_np, labels_np)


def collect_and_save_hlt_multiview_predictions(
    model,
    dataset: HLTMultiViewDataset,
    *,
    config: HLTMultiViewTrainConfig,
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
    logits, labels, jet_ids, metrics = collect_hlt_multiview_outputs(
        model,
        dataset,
        config.view_specs,
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
            "contract": HLT_MULTIVIEW_PREDICTION_CONTRACT,
            "experiment_name": HLT_MULTIVIEW_EXPERIMENT_NAME,
            "experiment_step": HLT_MULTIVIEW_STEP,
            "model_name": config.model_name,
            "architecture": HLT_MULTIVIEW_MODEL_ARCHITECTURE,
            "split": split,
            "checkpoint": str(config.checkpoint_path),
            "checkpoint_sha256": sha256_file(config.checkpoint_path),
            "checkpoint_epoch": None if checkpoint_payload is None else checkpoint_payload.get("epoch"),
            "dataset": dataset.to_metadata(),
            "rich_metrics": dict(metrics),
            "allowed_inputs": HLT_SDV_ALLOWED_INPUTS,
            "deployment_inputs": HLT_MULTIVIEW_DEPLOYMENT_INPUTS,
            "requires_offline_inputs": False,
            "requires_teacher_features": False,
            "requires_deterministic_hlt2_transform": True,
            "no_final_test_used_for_selection": True,
        },
    )
    metadata = save_prediction_block(block, config.prediction_dir, overwrite=bool(config.overwrite))
    return metrics, metadata


def train_hlt_multiview_model(config: HLTMultiViewTrainConfig) -> dict[str, Any]:
    torch = require_torch()
    set_training_seed(config.seed)
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir)
    if config.checkpoint_path.exists() and not config.overwrite:
        raise FileExistsError(f"HLT multi-view checkpoint already exists: {config.checkpoint_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = load_hlt_multiview_dataset_for_config(config, config.train_split)
    val_dataset = load_hlt_multiview_dataset_for_config(config, config.val_split)
    train_loader = make_hlt_multiview_data_loader(
        train_dataset,
        config.view_specs,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        seed=int(config.seed) + 101,
    )
    val_loader = make_hlt_multiview_data_loader(
        val_dataset,
        config.view_specs,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        seed=int(config.seed) + 202,
    )

    model = HLTSelfMultiViewFusionModel(
        view_names=[spec.name for spec in config.view_specs],
        num_classes=PD10_NUM_CLASSES,
        model_size=config.model_size,
        fusion_hidden_dim=config.fusion_hidden_dim,
        representation_dim=config.representation_dim,
        dropout=config.dropout,
    ).to(device)
    branch_initialization = initialize_hlt_multiview_branches_from_checkpoints(
        model,
        config.view_specs,
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
        "contract": HLT_MULTIVIEW_FUSION_CONTRACT,
        "experiment_name": HLT_MULTIVIEW_EXPERIMENT_NAME,
        "experiment_step": HLT_MULTIVIEW_STEP,
        "model_name": config.model_name,
        "architecture": HLT_MULTIVIEW_MODEL_ARCHITECTURE,
        "config": asdict(config),
        "view_names": [spec.name for spec in config.view_specs],
        "n_views": int(len(config.view_specs)),
        "train_dataset": train_dataset.to_metadata(),
        "model_val_dataset": val_dataset.to_metadata(),
        "branch_initialization": branch_initialization,
        "allowed_inputs": HLT_SDV_ALLOWED_INPUTS,
        "deployment_inputs": HLT_MULTIVIEW_DEPLOYMENT_INPUTS,
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
            optimizer = _optimizer_for_multiview_stage(model, config, stage=stage)
            current_stage = stage
        train_metrics = run_hlt_multiview_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            amp=config.amp,
            grad_clip_norm=config.grad_clip_norm,
            max_batches=config.max_train_batches,
        )
        val_metrics = run_hlt_multiview_epoch(
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
            _multiview_checkpoint_payload(
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
                _multiview_checkpoint_payload(
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

    selected_model, selected_payload = load_hlt_multiview_model_from_checkpoint(config.checkpoint_path, device=device)
    model_val_prediction_metrics = None
    model_val_prediction_metadata = None
    validation_thresholds = None
    validation_binary_thresholds = None
    if config.evaluate_model_val_predictions:
        model_val_prediction_metrics, model_val_prediction_metadata = collect_and_save_hlt_multiview_predictions(
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
        final_test_dataset = load_hlt_multiview_dataset_for_config(config, config.final_test_split)
        final_test_metrics, final_test_prediction_metadata = collect_and_save_hlt_multiview_predictions(
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
    save_json(output_dir / HLT_MULTIVIEW_REPORT, report)
    if final_test_metrics is not None:
        save_json(output_dir / "final_test_report.json", {**report, "metrics": final_test_metrics})
    return report


__all__ = [
    "HLTMultiViewSpec",
    "HLTMultiViewTrainConfig",
    "HLTSelfMultiViewFusionModel",
    "HLT_MULTIVIEW_FUSION_CONTRACT",
    "HLT_MULTIVIEW_MODEL_ARCHITECTURE",
    "HLT_MULTIVIEW_REPORT",
    "collate_hlt_multiview_batch",
    "load_hlt_multiview_dataset",
    "load_hlt_multiview_model_from_checkpoint",
    "make_hlt_multiview_data_loader",
    "move_hlt_multiview_batch_to_device",
    "run_hlt_multiview_epoch",
    "train_hlt_multiview_model",
]
