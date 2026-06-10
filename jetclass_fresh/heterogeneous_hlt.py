"""Heterogeneous fixed-HLT taggers and fusion utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from .fusion import PredictionBlock, prediction_paths, save_prediction_block, softmax_np
from .hlt_baseline import (
    HLTBaselineTrainConfig,
    JetViewTorchDataset,
    ParticleTransformerHLTClassifier,
    default_part_config,
    make_data_loader,
    require_torch,
    resolve_device,
    save_json,
    train_hlt_baseline,
)
from .hlt_cache import load_cached_hlt_view
from .jetclass_data import LABEL_NAMES, JetView
from .part_inputs import PF_FEATURE_NAMES

try:  # Keep the module importable on machines without the training stack.
    import torch as _torch
except ImportError:  # pragma: no cover - environment dependent
    _torch = None

if _torch is None:  # pragma: no cover - environment dependent
    class _ModuleBase:
        pass
else:
    _ModuleBase = _torch.nn.Module

HETERO_HLT_ARCHITECTURES = ("part", "pn", "pfn", "pcnn")
HETERO_HLT_MODEL_NAMES = {
    "part": "hlt_part",
    "pn": "hlt_pn",
    "pfn": "hlt_pfn",
    "pcnn": "hlt_pcnn",
}


@dataclass
class HeterogeneousHLTFusionConfig:
    output_dir: str
    cache_dir: str
    checkpoint_root: str
    architectures: List[str]
    splits: List[str]
    batch_size: int = 128
    num_workers: int = 0
    device: str = "auto"
    stack_train_size: int | None = 150000
    stack_val_size: int | None = 50000
    final_test_size: int | None = 300000
    overwrite_predictions: bool = False
    skip_existing_predictions: bool = True
    confirm_final_test: bool = False
    feature_modes: List[str] | None = None
    c_grid: List[float] | None = None
    max_iter: int = 2000
    run_controls: bool = True
    control_seed: int = 12345


class ParticleNetHLTClassifier(_ModuleBase):
    """ParticleNet wrapper using the same HLT particle input tensors."""

    def __init__(self, **kwargs) -> None:
        torch = require_torch()
        super().__init__()
        try:
            from weaver.nn.model.ParticleNet import ParticleNet
        except ImportError as exc:  # pragma: no cover - depends on research env
            raise ImportError(
                "ParticleNet HLT training requires weaver-core with ParticleNet available."
            ) from exc

        clean_kwargs = dict(kwargs)
        clean_kwargs.pop("architecture", None)
        self.config = {"architecture": "pn", **clean_kwargs}
        self.mod = ParticleNet(**clean_kwargs)

    def forward(self, points, features, lorentz_vectors, mask):
        del lorentz_vectors
        return self.mod(points, features, mask)


class ParticleFlowNetworkHLTClassifier(_ModuleBase):
    """PFN implementation adapted to the local Particle Transformer input tensors."""

    def __init__(
        self,
        *,
        input_dims: int,
        num_classes: int,
        phi_sizes: Sequence[int],
        f_sizes: Sequence[int],
        use_bn: bool = False,
        **kwargs,
    ) -> None:
        torch = require_torch()
        super().__init__()
        nn = torch.nn
        self.config = {
            "architecture": "pfn",
            "input_dims": int(input_dims),
            "num_classes": int(num_classes),
            "phi_sizes": list(phi_sizes),
            "f_sizes": list(f_sizes),
            "use_bn": bool(use_bn),
            **dict(kwargs),
        }
        self.input_bn = nn.BatchNorm1d(input_dims) if use_bn else nn.Identity()
        phi_layers = []
        for index, width in enumerate(phi_sizes):
            phi_layers.append(
                nn.Sequential(
                    nn.Conv1d(input_dims if index == 0 else int(phi_sizes[index - 1]), int(width), kernel_size=1),
                    nn.BatchNorm1d(int(width)) if use_bn else nn.Identity(),
                    nn.ReLU(),
                )
            )
        self.phi = nn.Sequential(*phi_layers)
        f_layers = []
        for index, width in enumerate(f_sizes):
            f_layers.append(
                nn.Sequential(
                    nn.Linear(int(phi_sizes[-1]) if index == 0 else int(f_sizes[index - 1]), int(width)),
                    nn.ReLU(),
                )
            )
        f_layers.append(nn.Linear(int(f_sizes[-1]), int(num_classes)))
        self.fc = nn.Sequential(*f_layers)

    def forward(self, points, features, lorentz_vectors, mask):
        del points, lorentz_vectors
        x = self.input_bn(features)
        if mask is not None:
            x = x * mask.float()
        x = self.phi(x)
        if mask is not None:
            x = x * mask.float()
        return self.fc(x.sum(dim=-1))


class ResNetUnit(_ModuleBase):
    def __init__(self, in_channels: int, out_channels: int, strides: tuple[int, int] = (1, 1)) -> None:
        torch = require_torch()
        super().__init__()
        nn = torch.nn
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=strides[0], padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=strides[1], padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dim_match = in_channels == out_channels and strides == (1, 1)
        self.conv_sc = None
        if not self.dim_match:
            self.conv_sc = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=strides[0] * strides[1],
                bias=False,
            )

    def forward(self, x):
        identity = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        if self.dim_match:
            return identity + x
        return self.conv_sc(identity) + x


class PCNNHLTClassifier(_ModuleBase):
    """P-CNN/ResNet-style classifier adapted from the JetClass reference."""

    def __init__(
        self,
        *,
        input_dims: int,
        num_classes: int,
        conv_params: Sequence[Sequence[int]],
        fc_params: Sequence[tuple[int, float]],
        **kwargs,
    ) -> None:
        torch = require_torch()
        super().__init__()
        nn = torch.nn
        conv_params = [tuple(int(value) for value in row) for row in conv_params]
        fc_params = [(int(width), float(dropout)) for width, dropout in fc_params]
        self.config = {
            "architecture": "pcnn",
            "input_dims": int(input_dims),
            "num_classes": int(num_classes),
            "conv_params": [list(row) for row in conv_params],
            "fc_params": [[width, dropout] for width, dropout in fc_params],
            **dict(kwargs),
        }
        self.fts_conv = nn.Sequential(
            nn.BatchNorm1d(input_dims),
            nn.Conv1d(input_dims, conv_params[0][0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(conv_params[0][0]),
            nn.ReLU(),
        )
        self.resnet_units = nn.ModuleDict()
        for stage_index in range(len(conv_params) - 1):
            unit_layers = []
            for block_index in range(len(conv_params[stage_index + 1])):
                in_channels = conv_params[stage_index][-1] if block_index == 0 else conv_params[stage_index + 1][block_index - 1]
                out_channels = conv_params[stage_index + 1][block_index]
                strides = (2, 1) if block_index == 0 and stage_index > 0 else (1, 1)
                unit_layers.append(ResNetUnit(in_channels, out_channels, strides))
            self.resnet_units[f"resnet_unit_{stage_index}"] = nn.Sequential(*unit_layers)

        fc_layers = []
        for index, (width, dropout) in enumerate(fc_params):
            in_channels = conv_params[-1][-1] if index == 0 else fc_params[index - 1][0]
            fc_layers.append(nn.Sequential(nn.Linear(in_channels, width), nn.ReLU(), nn.Dropout(dropout)))
        fc_layers.append(nn.Linear(fc_params[-1][0], num_classes))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, points, features, lorentz_vectors, mask):
        del points, lorentz_vectors
        if mask is not None:
            features = features * mask.float()
        x = self.fts_conv(features)
        for stage_index in range(len(self.resnet_units)):
            x = self.resnet_units[f"resnet_unit_{stage_index}"](x)
        return self.fc(x.mean(dim=-1))


def _particle_net_config(*, num_classes: int, model_size: str) -> Dict[str, Any]:
    if model_size == "tiny":
        conv_params = [(8, (32, 32)), (8, (64, 64))]
        fc_params = [(128, 0.1)]
    elif model_size == "base":
        conv_params = [(16, (64, 64, 64)), (16, (128, 128, 128)), (16, (256, 256, 256))]
        fc_params = [(256, 0.1)]
    else:
        raise ValueError(f"Unknown model_size {model_size!r}")
    return {
        "architecture": "pn",
        "input_dims": len(PF_FEATURE_NAMES),
        "num_classes": int(num_classes),
        "conv_params": conv_params,
        "fc_params": fc_params,
        "use_fusion": False,
        "use_fts_bn": True,
        "use_counts": True,
        "for_inference": False,
    }


def _pfn_config(*, num_classes: int, model_size: str) -> Dict[str, Any]:
    if model_size == "tiny":
        phi_sizes = (64, 64)
        f_sizes = (64, 64)
    elif model_size == "base":
        phi_sizes = (128, 128, 128)
        f_sizes = (128, 128, 128)
    else:
        raise ValueError(f"Unknown model_size {model_size!r}")
    return {
        "input_dims": len(PF_FEATURE_NAMES),
        "num_classes": int(num_classes),
        "phi_sizes": phi_sizes,
        "f_sizes": f_sizes,
        "use_bn": False,
    }


def _pcnn_config(*, num_classes: int, model_size: str) -> Dict[str, Any]:
    if model_size == "tiny":
        conv_params = [(16,), (32, 32), (64, 64)]
        fc_params = [(128, 0.1)]
    elif model_size == "base":
        conv_params = [(32,), (64, 64), (64, 64), (128, 128)]
        fc_params = [(512, 0.2)]
    else:
        raise ValueError(f"Unknown model_size {model_size!r}")
    return {
        "input_dims": len(PF_FEATURE_NAMES),
        "num_classes": int(num_classes),
        "conv_params": conv_params,
        "fc_params": fc_params,
    }


def normalize_architecture_name(architecture: str) -> str:
    value = str(architecture).strip().lower()
    aliases = {
        "particletransformer": "part",
        "particle_transformer": "part",
        "par": "part",
        "part": "part",
        "particlenet": "pn",
        "particle_net": "pn",
        "pn": "pn",
        "particleflownetwork": "pfn",
        "particle_flow_network": "pfn",
        "pfn": "pfn",
        "p-cnn": "pcnn",
        "p_cnn": "pcnn",
        "pcnn": "pcnn",
    }
    if value not in aliases:
        raise ValueError(f"Unknown HLT architecture {architecture!r}; expected one of {HETERO_HLT_ARCHITECTURES}")
    return aliases[value]


def default_model_name_for_architecture(architecture: str) -> str:
    return HETERO_HLT_MODEL_NAMES[normalize_architecture_name(architecture)]


def build_heterogeneous_hlt_classifier(
    architecture: str,
    *,
    num_classes: int = 10,
    model_size: str = "base",
    overrides: Mapping[str, Any] | None = None,
):
    arch = normalize_architecture_name(architecture)
    if arch == "part":
        cfg = default_part_config(num_classes=num_classes, model_size=model_size)
        if overrides:
            cfg.update(dict(overrides))
        cfg.pop("architecture", None)
        model = ParticleTransformerHLTClassifier(**cfg)
        model.config = {"architecture": "part", **dict(model.config)}
        return model
    if arch == "pn":
        cfg = _particle_net_config(num_classes=num_classes, model_size=model_size)
        if overrides:
            cfg.update(dict(overrides))
        return ParticleNetHLTClassifier(**cfg)
    if arch == "pfn":
        cfg = _pfn_config(num_classes=num_classes, model_size=model_size)
        if overrides:
            cfg.update(dict(overrides))
        return ParticleFlowNetworkHLTClassifier(**cfg)
    if arch == "pcnn":
        cfg = _pcnn_config(num_classes=num_classes, model_size=model_size)
        if overrides:
            cfg.update(dict(overrides))
        return PCNNHLTClassifier(**cfg)
    raise AssertionError(f"Unhandled architecture {arch!r}")


def build_heterogeneous_hlt_classifier_from_config(model_config: Mapping[str, Any]):
    cfg = dict(model_config)
    architecture = normalize_architecture_name(cfg.pop("architecture", "part"))
    if architecture == "part":
        cfg.pop("architecture", None)
        return ParticleTransformerHLTClassifier(**cfg)
    if architecture == "pn":
        return ParticleNetHLTClassifier(**cfg)
    if architecture == "pfn":
        return ParticleFlowNetworkHLTClassifier(**cfg)
    if architecture == "pcnn":
        return PCNNHLTClassifier(**cfg)
    raise AssertionError(f"Unhandled architecture {architecture!r}")


def train_heterogeneous_hlt_model(
    config: HLTBaselineTrainConfig,
    *,
    architecture: str,
    max_train_jets: int | None = None,
    max_val_jets: int | None = None,
):
    model = build_heterogeneous_hlt_classifier(
        architecture,
        num_classes=len(LABEL_NAMES),
        model_size=config.model_size,
    )
    arch = normalize_architecture_name(architecture)
    report = train_hlt_baseline(
        config,
        model=model,
        max_train_jets=max_train_jets,
        max_val_jets=max_val_jets,
    )
    report = dict(report)
    report.update(
        {
            "experiment_step": "heterogeneous_hlt_architecture_training",
            "architecture": arch,
            "model_name": default_model_name_for_architecture(arch),
            "max_train_jets": max_train_jets,
            "max_val_jets": max_val_jets,
        }
    )
    save_json(Path(config.output_dir) / "heterogeneous_hlt_report.json", report)
    return report


def load_heterogeneous_hlt_model_from_checkpoint(path: str | Path, *, device):
    torch = require_torch()
    payload = torch.load(path, map_location=device)
    model_config = payload.get("model_config") or {}
    if not model_config:
        cfg = payload.get("config", {})
        model = build_heterogeneous_hlt_classifier(
            cfg.get("architecture", "part"),
            num_classes=len(LABEL_NAMES),
            model_size=cfg.get("model_size", "base"),
        )
    else:
        model = build_heterogeneous_hlt_classifier_from_config(model_config)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()
    return model, payload


def _maybe_limit_view(view: JetView, max_jets: int | None) -> JetView:
    if max_jets is None:
        return view
    limit = min(int(max_jets), len(view.labels))
    return JetView(
        tokens=view.tokens[:limit],
        mask=view.mask[:limit],
        labels=view.labels[:limit],
        jet_ids=view.jet_ids[:limit],
        split=view.split,
        metadata=dict(view.metadata),
    )


def evaluate_heterogeneous_hlt_model(
    model,
    view: JetView,
    *,
    model_name: str,
    architecture: str,
    batch_size: int,
    num_workers: int,
    device,
    max_jets: int | None = None,
) -> PredictionBlock:
    torch = require_torch()
    view = _maybe_limit_view(view, max_jets)
    dataset = JetViewTorchDataset(view)
    loader = make_data_loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        seed=12345,
    )
    logits_rows: list[np.ndarray] = []
    labels_rows: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            logits = model(batch["points"], batch["features"], batch["lorentz_vectors"], batch["mask"])
            logits_rows.append(logits.detach().cpu().numpy().astype(np.float32))
            labels_rows.append(batch["labels"].detach().cpu().numpy().astype(np.int64))
    logits_np = np.concatenate(logits_rows, axis=0)
    labels_np = np.concatenate(labels_rows, axis=0)
    return PredictionBlock(
        model_name=model_name,
        split=view.split,
        logits=logits_np,
        probs=softmax_np(logits_np),
        labels=labels_np,
        jet_ids=list(view.jet_ids),
        metadata={
            "model_kind": "heterogeneous_hlt",
            "hlt_architecture": normalize_architecture_name(architecture),
            "hlt_content_hash": view.metadata.get("hlt_content_hash"),
            "allowed_inputs": "cached_fixed_hlt_only",
        },
    )


def split_size_for_config(config: HeterogeneousHLTFusionConfig, split: str) -> int | None:
    if split == "stack_train":
        return config.stack_train_size
    if split == "stack_val":
        return config.stack_val_size
    if split == "final_test":
        return config.final_test_size
    return None


def collect_heterogeneous_hlt_predictions(config: HeterogeneousHLTFusionConfig) -> Dict[str, Any]:
    torch = require_torch()
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir)
    prediction_dir = output_dir / "predictions"
    reports: Dict[str, Any] = {}
    for architecture in config.architectures:
        arch = normalize_architecture_name(architecture)
        model_name = default_model_name_for_architecture(arch)
        checkpoint = Path(config.checkpoint_root) / arch / "best_model_val.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing heterogeneous HLT checkpoint: {checkpoint}")
        model, payload = load_heterogeneous_hlt_model_from_checkpoint(checkpoint, device=device)
        reports[model_name] = {}
        for split in config.splits:
            npz_path, _ = prediction_paths(prediction_dir, model_name, split)
            if npz_path.exists() and config.skip_existing_predictions and not config.overwrite_predictions:
                from .fusion import load_prediction_block

                reports[model_name][split] = load_prediction_block(prediction_dir, model_name, split).metadata
                continue
            view = load_cached_hlt_view(config.cache_dir, split)
            block = evaluate_heterogeneous_hlt_model(
                model,
                view,
                model_name=model_name,
                architecture=arch,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                device=device,
                max_jets=split_size_for_config(config, split),
            )
            block.metadata.update(
                {
                    "checkpoint": str(checkpoint),
                    "checkpoint_epoch": payload.get("epoch"),
                    "checkpoint_best_model_val_accuracy": (
                        (payload.get("metrics") or {}).get("model_val", {}) or {}
                    ).get("accuracy"),
                    "max_jets": split_size_for_config(config, split),
                }
            )
            reports[model_name][split] = save_prediction_block(
                block,
                prediction_dir,
                overwrite=config.overwrite_predictions,
            )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return reports
