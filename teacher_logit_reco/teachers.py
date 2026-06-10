"""Frozen teacher adapters for teacher-logit reconstruction.

Teacher adapters expose a single view-level API:

``teacher.forward_view(tokens, mask, weights=None) -> logits``

The teacher model is frozen and in eval mode, but the forward path intentionally
does not wrap itself in ``torch.no_grad``.  That lets future reconstructors
receive gradients through the frozen teacher with respect to reconstructed
tokens/weights while keeping all teacher parameters fixed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from jetclass_fresh.dual_view import build_part_inputs_torch
from jetclass_fresh.heterogeneous_hlt import (
    build_heterogeneous_hlt_classifier,
    build_heterogeneous_hlt_classifier_from_config,
    normalize_architecture_name,
)
from jetclass_fresh.hlt_baseline import (
    ParticleTransformerHLTClassifier,
    default_part_config,
    require_torch,
    resolve_device,
)
from jetclass_fresh.jetclass_data import LABEL_NAMES, JetView

from .views import SoftReconstructedView


TEACHER_ARCHITECTURES = ("part", "pn", "pfn", "pcnn")


@dataclass
class FrozenTeacherMetadata:
    architecture: str
    checkpoint_path: str | None
    experiment_step: str | None
    epoch: int | None
    model_config: Dict[str, Any]
    label_names: list[str]
    max_constits: int
    weight_threshold: float
    frozen: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "architecture": self.architecture,
            "checkpoint_path": self.checkpoint_path,
            "experiment_step": self.experiment_step,
            "epoch": self.epoch,
            "model_config": dict(self.model_config),
            "label_names": list(self.label_names),
            "max_constits": int(self.max_constits),
            "weight_threshold": float(self.weight_threshold),
            "frozen": bool(self.frozen),
        }


@dataclass
class FrozenTeacher:
    """A frozen view-level teacher model."""

    model: Any
    architecture: str
    device: Any
    checkpoint_path: str | None = None
    payload: Mapping[str, Any] | None = None
    max_constits: int = 128
    weight_threshold: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.architecture = normalize_teacher_architecture(self.architecture)
        self.model = self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        payload = dict(self.payload or {})
        model_config = dict(payload.get("model_config") or getattr(self.model, "config", {}) or {})
        label_names = [str(item) for item in payload.get("label_names", LABEL_NAMES)]
        self.metadata = {
            **FrozenTeacherMetadata(
                architecture=self.architecture,
                checkpoint_path=None if self.checkpoint_path is None else str(self.checkpoint_path),
                experiment_step=payload.get("experiment_step"),
                epoch=None if payload.get("epoch") is None else int(payload["epoch"]),
                model_config=model_config,
                label_names=label_names,
                max_constits=int(self.max_constits),
                weight_threshold=float(self.weight_threshold),
                frozen=self.parameters_frozen(),
            ).to_dict(),
            **dict(self.metadata),
        }

    def parameters_frozen(self) -> bool:
        return all(not bool(param.requires_grad) for param in self.model.parameters())

    def trainable_parameter_count(self) -> int:
        return int(sum(param.numel() for param in self.model.parameters() if param.requires_grad))

    def _tensor(self, value: Any, *, dtype=None):
        torch = require_torch()
        if isinstance(value, torch.Tensor):
            tensor = value.to(self.device)
            return tensor.to(dtype=dtype) if dtype is not None else tensor
        tensor = torch.as_tensor(value, device=self.device)
        return tensor.to(dtype=dtype) if dtype is not None else tensor

    def build_inputs(self, tokens: Any, mask: Any, *, weights: Any = None) -> Dict[str, Any]:
        torch = require_torch()
        token_tensor = self._tensor(tokens, dtype=torch.float32)
        mask_tensor = self._tensor(mask, dtype=torch.bool)
        weight_tensor = None if weights is None else self._tensor(weights, dtype=torch.float32)
        return build_part_inputs_torch(
            token_tensor,
            mask_tensor,
            weights=weight_tensor,
            max_constits=int(self.max_constits),
            weight_threshold=float(self.weight_threshold),
        )

    def forward_inputs(self, inputs: Mapping[str, Any]):
        logits = self.model(
            inputs["points"],
            inputs["features"],
            inputs["lorentz_vectors"],
            inputs["mask"],
        )
        if not require_torch().isfinite(logits).all():
            raise FloatingPointError("Teacher produced non-finite logits")
        return logits

    def forward_view(self, tokens: Any, mask: Any, *, weights: Any = None):
        """Forward a raw token view through the frozen teacher.

        Gradients can flow to tensor ``tokens``/``weights`` inputs. Use
        ``forward_view_no_grad`` for cached offline teacher targets.
        """

        return self.forward_inputs(self.build_inputs(tokens, mask, weights=weights))

    def forward_view_no_grad(self, tokens: Any, mask: Any, *, weights: Any = None):
        torch = require_torch()
        with torch.no_grad():
            return self.forward_view(tokens, mask, weights=weights)

    def forward_jet_view(self, view: JetView):
        return self.forward_view(view.tokens, view.mask)

    def forward_jet_view_no_grad(self, view: JetView):
        return self.forward_view_no_grad(view.tokens, view.mask)

    def forward_soft_view(self, view: SoftReconstructedView):
        return self.forward_view(view.tokens, view.mask, weights=view.weights)

    def forward_soft_view_no_grad(self, view: SoftReconstructedView):
        return self.forward_view_no_grad(view.tokens, view.mask, weights=view.weights)


def normalize_teacher_architecture(architecture: str | None) -> str:
    if architecture is None:
        return "part"
    return normalize_architecture_name(str(architecture))


def infer_teacher_architecture(
    payload: Mapping[str, Any],
    *,
    architecture: str | None = None,
) -> str:
    """Infer teacher architecture from explicit arg or checkpoint metadata."""

    if architecture:
        return normalize_teacher_architecture(architecture)
    model_config = dict(payload.get("model_config") or {})
    if model_config.get("architecture"):
        return normalize_teacher_architecture(str(model_config["architecture"]))
    cfg = dict(payload.get("config") or {})
    if cfg.get("architecture"):
        return normalize_teacher_architecture(str(cfg["architecture"]))
    # Historical offline/HLT teacher checkpoints in this repo are ParT unless
    # explicitly marked otherwise.
    return "part"


def _strip_compile_prefix(state_dict: Mapping[str, Any]) -> Dict[str, Any]:
    keys = list(state_dict.keys())
    if keys and all(str(key).startswith("_orig_mod.") for key in keys):
        return {str(key).removeprefix("_orig_mod."): value for key, value in state_dict.items()}
    return dict(state_dict)


def _build_part_teacher_from_payload(payload: Mapping[str, Any]) -> Any:
    model_config = dict(payload.get("model_config") or {})
    model_config.pop("architecture", None)
    if model_config:
        return ParticleTransformerHLTClassifier(**model_config)
    cfg = dict(payload.get("config") or {})
    model_size = str(cfg.get("model_size", "base"))
    return ParticleTransformerHLTClassifier(**default_part_config(num_classes=len(LABEL_NAMES), model_size=model_size))


def build_teacher_model_from_payload(
    payload: Mapping[str, Any],
    *,
    architecture: str | None = None,
) -> tuple[Any, str]:
    arch = infer_teacher_architecture(payload, architecture=architecture)
    if arch == "part":
        return _build_part_teacher_from_payload(payload), arch
    model_config = dict(payload.get("model_config") or {})
    if model_config:
        model_config["architecture"] = arch
        return build_heterogeneous_hlt_classifier_from_config(model_config), arch
    cfg = dict(payload.get("config") or {})
    model_size = str(cfg.get("model_size", "base"))
    return build_heterogeneous_hlt_classifier(arch, num_classes=len(LABEL_NAMES), model_size=model_size), arch


def load_frozen_teacher(
    checkpoint_path: str | Path,
    *,
    architecture: str | None = None,
    device: str = "auto",
    max_constits: int = 128,
    weight_threshold: float = 0.0,
    strict: bool = True,
) -> FrozenTeacher:
    """Load a frozen teacher checkpoint.

    Supported architectures are ParT, ParticleNet, PFN, and PCNN. The checkpoint
    must contain a ``model_state_dict`` and ideally a ``model_config``.
    """

    torch = require_torch()
    resolved_device = resolve_device(device)
    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location=resolved_device)
    if "model_state_dict" not in payload:
        raise KeyError(f"Teacher checkpoint is missing model_state_dict: {checkpoint_path}")
    model, arch = build_teacher_model_from_payload(payload, architecture=architecture)
    state_dict = _strip_compile_prefix(payload["model_state_dict"])
    model.load_state_dict(state_dict, strict=bool(strict))
    return FrozenTeacher(
        model=model,
        architecture=arch,
        device=resolved_device,
        checkpoint_path=str(checkpoint_path),
        payload=payload,
        max_constits=int(max_constits),
        weight_threshold=float(weight_threshold),
    )


def summarize_teacher_forward(
    logits: Any,
    *,
    name: str,
) -> Dict[str, Any]:
    torch = require_torch()
    if isinstance(logits, torch.Tensor):
        arr = logits.detach().cpu().numpy()
    else:
        arr = np.asarray(logits)
    return {
        "name": str(name),
        "shape": list(arr.shape),
        "finite": bool(np.isfinite(arr).all()),
        "mean": float(np.mean(arr)) if arr.size else 0.0,
        "std": float(np.std(arr)) if arr.size else 0.0,
        "min": float(np.min(arr)) if arr.size else 0.0,
        "max": float(np.max(arr)) if arr.size else 0.0,
    }


def assert_teacher_frozen(teacher: FrozenTeacher) -> None:
    if not teacher.parameters_frozen():
        raise AssertionError("Teacher has trainable parameters")
    if teacher.trainable_parameter_count() != 0:
        raise AssertionError("Teacher trainable parameter count is nonzero")
