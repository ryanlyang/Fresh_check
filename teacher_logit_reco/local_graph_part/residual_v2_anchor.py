"""Exact HLT ParT embedding anchor for residual expert V2.

The V2 residual path must consume the selected 2-class HLT ParT baseline score
and a true penultimate ParT embedding.  This module wraps an exact baseline
model and captures the input to its final classifier layer during the normal
forward pass.  If that embedding cannot be identified, V2 fails loudly instead
of falling back to the V1 widened-head proxy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from jetclass_fresh.hlt_baseline import require_torch, resolve_device

from .model import (
    LOCAL_GRAPH_HLT_PART_BASELINE_CONTRACT,
    LOCAL_GRAPH_MODEL_VARIANT_HLT_PART_BASELINE,
)
from .residual_v2_protocol import (
    LOCAL_GRAPH_RESIDUAL_V2_ANCHOR_CONTRACT,
    LOCAL_GRAPH_RESIDUAL_V2_ANCHOR_STEP,
    LOCAL_GRAPH_RESIDUAL_V2_BASELINE_VARIANT,
    LOCAL_GRAPH_RESIDUAL_V2_DISALLOWED_EMBEDDING_FALLBACKS,
    LOCAL_GRAPH_RESIDUAL_V2_POSITIVE_CLASS_INDEX,
    LOCAL_GRAPH_RESIDUAL_V2_POSITIVE_CLASS_NAME,
    LOCAL_GRAPH_RESIDUAL_V2_REQUIRED_EMBEDDING_ROLE,
)

try:  # Keep imports cheap when PyTorch is not available.
    import torch as _torch
except ImportError:  # pragma: no cover - environment dependent
    _torch = None

if _torch is None:  # pragma: no cover - environment dependent
    class _ModuleBase:
        pass
else:
    _ModuleBase = _torch.nn.Module


HLT_PART_EMBEDDING_SOURCE_FINAL_HEAD_HOOK = "final_head_forward_hook"


def strip_compile_prefix_from_state_dict(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Remove common DataParallel/torch.compile prefixes from checkpoint keys."""

    cleaned: dict[str, Any] = {}
    for key, value in dict(state_dict).items():
        clean = str(key)
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "_orig_mod."):
                if clean.startswith(prefix):
                    clean = clean[len(prefix) :]
                    changed = True
        cleaned[clean] = value
    return cleaned


def _extract_state_dict(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    state_dict = payload.get("model_state_dict", payload.get("state_dict"))
    if not isinstance(state_dict, Mapping):
        raise KeyError("HLT ParT embedding anchor checkpoint is missing model_state_dict/state_dict")
    return state_dict


def _payload_label_names(payload: Mapping[str, Any], fallback: tuple[str, ...]) -> tuple[str, ...]:
    labels = payload.get("label_names")
    if labels is None:
        model_config = payload.get("model_config")
        if isinstance(model_config, Mapping):
            labels = model_config.get("label_names")
    if labels is None:
        return fallback
    return tuple(str(item) for item in labels)


def _payload_num_classes(payload: Mapping[str, Any], fallback: int) -> int:
    for key in ("num_classes",):
        if payload.get(key) is not None:
            return int(payload[key])
    model_config = payload.get("model_config")
    if isinstance(model_config, Mapping) and model_config.get("num_classes") is not None:
        return int(model_config["num_classes"])
    labels = payload.get("label_names")
    if labels is not None:
        return len(tuple(labels))
    return int(fallback)


def _named_module(model: Any, name: str) -> Any:
    modules = dict(model.named_modules())
    if name not in modules:
        raise ValueError(f"final_head_name {name!r} was not found in the HLT ParT baseline model")
    return modules[name]


def _candidate_final_linear_modules(model: Any, *, num_classes: int) -> list[tuple[str, Any]]:
    torch = require_torch()
    candidates: list[tuple[str, Any]] = []
    for name, module in model.named_modules():
        if not name:
            continue
        if isinstance(module, torch.nn.Linear) and int(module.out_features) == int(num_classes):
            candidates.append((str(name), module))
    return candidates


def _tensor_from_model_output(output: Any) -> Any:
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, tuple) and output:
        return output[0]
    return output


def _finite_tensor(value: Any, *, name: str) -> Any:
    torch = require_torch()
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value)!r}")
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{name} contains non-finite values")
    return value


@dataclass(frozen=True)
class HLTPartEmbeddingAnchorConfig:
    """Configuration for strict V2 HLT ParT embedding extraction."""

    checkpoint_path: str | None = None
    num_classes: int = 2
    label_names: tuple[str, ...] = ("QCD", "Hgg")
    positive_class_name: str = LOCAL_GRAPH_RESIDUAL_V2_POSITIVE_CLASS_NAME
    positive_class_index: int = LOCAL_GRAPH_RESIDUAL_V2_POSITIVE_CLASS_INDEX
    baseline_variant: str = LOCAL_GRAPH_RESIDUAL_V2_BASELINE_VARIANT
    freeze_anchor: bool = True
    strict_final_head: bool = True
    final_head_name: str | None = None
    embedding_source: str = HLT_PART_EMBEDDING_SOURCE_FINAL_HEAD_HOOK
    required_embedding_role: str = LOCAL_GRAPH_RESIDUAL_V2_REQUIRED_EMBEDDING_ROLE
    max_constits: int = 128
    weight_threshold: float = 0.0
    embedding_reproduces_logits_atol: float = 1.0e-5
    embedding_reproduces_logits_rtol: float = 1.0e-4

    def __post_init__(self) -> None:
        if int(self.num_classes) != 2:
            raise ValueError("V2 HLT ParT embedding anchor is fixed to binary QCD/Hgg")
        object.__setattr__(self, "num_classes", int(self.num_classes))
        labels = tuple(str(item) for item in self.label_names)
        if labels != ("QCD", "Hgg"):
            raise ValueError("V2 HLT ParT embedding anchor label_names must be ('QCD', 'Hgg')")
        object.__setattr__(self, "label_names", labels)
        if str(self.positive_class_name) != LOCAL_GRAPH_RESIDUAL_V2_POSITIVE_CLASS_NAME:
            raise ValueError("V2 HLT ParT embedding anchor positive class must be Hgg")
        if int(self.positive_class_index) != LOCAL_GRAPH_RESIDUAL_V2_POSITIVE_CLASS_INDEX:
            raise ValueError("V2 HLT ParT embedding anchor positive class index must be 1")
        object.__setattr__(self, "positive_class_index", int(self.positive_class_index))
        if str(self.baseline_variant) != LOCAL_GRAPH_MODEL_VARIANT_HLT_PART_BASELINE:
            raise ValueError("V2 HLT ParT embedding anchor must wrap the exact hlt_part_baseline variant")
        if str(self.embedding_source) in set(LOCAL_GRAPH_RESIDUAL_V2_DISALLOWED_EMBEDDING_FALLBACKS):
            raise ValueError(f"disallowed V2 embedding fallback: {self.embedding_source}")
        if str(self.embedding_source) != HLT_PART_EMBEDDING_SOURCE_FINAL_HEAD_HOOK:
            raise ValueError("Step 2 currently supports only final_head_forward_hook embedding extraction")
        if str(self.required_embedding_role) != LOCAL_GRAPH_RESIDUAL_V2_REQUIRED_EMBEDDING_ROLE:
            raise ValueError("V2 HLT ParT embedding role changed unexpectedly")
        if int(self.max_constits) <= 0:
            raise ValueError("max_constits must be positive")
        object.__setattr__(self, "max_constits", int(self.max_constits))
        if float(self.weight_threshold) < 0.0:
            raise ValueError("weight_threshold must be nonnegative")
        object.__setattr__(self, "weight_threshold", float(self.weight_threshold))
        for field_name in ("embedding_reproduces_logits_atol", "embedding_reproduces_logits_rtol"):
            value = float(getattr(self, field_name))
            if value < 0.0:
                raise ValueError(f"{field_name} must be nonnegative")
            object.__setattr__(self, field_name, value)
        final_head_name = self.final_head_name
        if final_head_name is not None:
            final_head_name = str(final_head_name).strip()
            if not final_head_name:
                final_head_name = None
        object.__setattr__(self, "final_head_name", final_head_name)
        object.__setattr__(self, "freeze_anchor", bool(self.freeze_anchor))
        object.__setattr__(self, "strict_final_head", bool(self.strict_final_head))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["label_names"] = list(self.label_names)
        return payload


@dataclass
class HLTPartEmbeddingAnchorOutput:
    """Forward output from the exact frozen HLT ParT embedding anchor."""

    logits: Any
    embedding: Any
    final_head_output: Any
    embedding_source: str
    final_head_name: str
    config: HLTPartEmbeddingAnchorConfig
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def margin(self) -> Any:
        return self.logits[:, 1] - self.logits[:, 0]

    def summary(self) -> dict[str, Any]:
        return {
            "step": LOCAL_GRAPH_RESIDUAL_V2_ANCHOR_STEP,
            "contract": LOCAL_GRAPH_RESIDUAL_V2_ANCHOR_CONTRACT,
            "embedding_source": self.embedding_source,
            "required_embedding_role": self.config.required_embedding_role,
            "final_head_name": self.final_head_name,
            "logits_shape": list(self.logits.shape),
            "embedding_shape": list(self.embedding.shape),
            "baseline_variant": self.config.baseline_variant,
            "label_names": list(self.config.label_names),
            "positive_class_name": self.config.positive_class_name,
            "positive_class_index": int(self.config.positive_class_index),
            "true_embedding_required": True,
        }


class HLTPartEmbeddingAnchor(_ModuleBase):
    """Frozen exact HLT ParT baseline that returns logits and penultimate embedding."""

    def __init__(
        self,
        model: Any,
        *,
        config: HLTPartEmbeddingAnchorConfig | Mapping[str, Any] | None = None,
        payload: Mapping[str, Any] | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        require_torch()
        super().__init__()
        if config is None:
            config = HLTPartEmbeddingAnchorConfig(
                checkpoint_path=None if checkpoint_path is None else str(checkpoint_path)
            )
        elif isinstance(config, Mapping):
            config = HLTPartEmbeddingAnchorConfig(**dict(config))
        self.config = config
        self.model = model
        self.payload = dict(payload or {})
        self.checkpoint_path = None if checkpoint_path is None else str(checkpoint_path)
        self.final_head_name, self.final_head = self._resolve_final_head()
        if bool(config.freeze_anchor):
            self.freeze_anchor_parameters()

    def _resolve_final_head(self) -> tuple[str, Any]:
        torch = require_torch()
        if self.config.final_head_name is not None:
            module = _named_module(self.model, self.config.final_head_name)
            if not isinstance(module, torch.nn.Linear):
                raise TypeError(f"final_head_name {self.config.final_head_name!r} is not a torch.nn.Linear")
            if int(module.out_features) != int(self.config.num_classes):
                raise ValueError(
                    f"final head {self.config.final_head_name!r} has out_features={int(module.out_features)}, "
                    f"expected {int(self.config.num_classes)}"
                )
            return str(self.config.final_head_name), module

        candidates = _candidate_final_linear_modules(self.model, num_classes=int(self.config.num_classes))
        if not candidates:
            raise ValueError(
                "Could not locate a final 2-class Linear head for V2 HLT ParT embedding extraction. "
                "Pass final_head_name or update the Weaver integration; V2 may not fall back to proxy embeddings."
            )
        if len(candidates) > 1 and bool(self.config.strict_final_head):
            names = ", ".join(name for name, _ in candidates)
            raise ValueError(
                "Ambiguous final 2-class Linear heads for V2 HLT ParT embedding extraction: "
                f"{names}. Pass final_head_name explicitly."
            )
        return candidates[-1]

    @property
    def output_contract(self) -> str:
        return LOCAL_GRAPH_RESIDUAL_V2_ANCHOR_CONTRACT

    def freeze_anchor_parameters(self) -> None:
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.model.eval()

    def anchor_parameters_frozen(self) -> bool:
        return all(not bool(param.requires_grad) for param in self.model.parameters())

    def trainable_parameter_count(self) -> int:
        return int(sum(param.numel() for param in self.model.parameters() if param.requires_grad))

    def metadata(self) -> dict[str, Any]:
        return {
            "experiment_step": LOCAL_GRAPH_RESIDUAL_V2_ANCHOR_STEP,
            "output_contract": LOCAL_GRAPH_RESIDUAL_V2_ANCHOR_CONTRACT,
            "checkpoint_path": self.checkpoint_path,
            "payload_experiment_step": self.payload.get("experiment_step"),
            "payload_output_contract": self.payload.get("output_contract"),
            "payload_variant": self.payload.get("variant"),
            "config": self.config.to_dict(),
            "embedding_source": self.config.embedding_source,
            "required_embedding_role": self.config.required_embedding_role,
            "final_head_name": self.final_head_name,
            "anchor_frozen": self.anchor_parameters_frozen(),
            "disallowed_embedding_fallbacks": list(LOCAL_GRAPH_RESIDUAL_V2_DISALLOWED_EMBEDDING_FALLBACKS),
        }

    def _call_model_with_embedding_capture(self, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
        torch = require_torch()
        captured: dict[str, Any] = {}

        def pre_hook(module, inputs):
            del module
            if not inputs:
                return
            captured["embedding"] = inputs[0]

        def forward_hook(module, inputs, output):
            del module, inputs
            captured["final_head_output"] = output

        pre_handle = self.final_head.register_forward_pre_hook(pre_hook)
        forward_handle = self.final_head.register_forward_hook(forward_hook)
        try:
            with torch.no_grad():
                logits = _tensor_from_model_output(self.model(*args, **kwargs))
        finally:
            pre_handle.remove()
            forward_handle.remove()
        if "embedding" not in captured or "final_head_output" not in captured:
            raise RuntimeError(
                "V2 HLT ParT embedding hook did not fire. The selected final head was not used in forward."
            )
        return logits, captured["embedding"], captured["final_head_output"]

    def forward_outputs(self, *args: Any, **kwargs: Any) -> HLTPartEmbeddingAnchorOutput:
        torch = require_torch()
        logits, embedding, final_head_output = self._call_model_with_embedding_capture(*args, **kwargs)
        logits = _finite_tensor(logits, name="HLT ParT anchor logits")
        embedding = _finite_tensor(embedding, name="HLT ParT penultimate embedding")
        final_head_output = _finite_tensor(final_head_output, name="HLT ParT final head output")
        if int(logits.ndim) != 2 or int(logits.shape[1]) != int(self.config.num_classes):
            raise ValueError(f"anchor logits must have shape [batch, 2], got {tuple(logits.shape)}")
        if int(embedding.ndim) != 2:
            raise ValueError(
                "V2 HLT ParT embedding must be a true 2D penultimate tensor [batch, dim], "
                f"got {tuple(embedding.shape)}"
            )
        if int(embedding.shape[0]) != int(logits.shape[0]):
            raise ValueError("V2 HLT ParT embedding batch dimension does not match logits")
        if tuple(final_head_output.shape) != tuple(logits.shape):
            raise ValueError(
                f"final head output shape {tuple(final_head_output.shape)} does not match logits {tuple(logits.shape)}"
            )
        reproduces = bool(
            torch.allclose(
                final_head_output,
                logits,
                atol=float(self.config.embedding_reproduces_logits_atol),
                rtol=float(self.config.embedding_reproduces_logits_rtol),
            )
        )
        if not reproduces:
            max_abs = torch.max(torch.abs(final_head_output - logits)).detach().cpu().item()
            raise ValueError(
                "Captured final-head embedding does not reproduce the HLT ParT logits; "
                f"max_abs_diff={max_abs:.6g}. V2 requires the true penultimate embedding."
            )
        embedding = torch.nan_to_num(embedding.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        logits = torch.nan_to_num(logits.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        final_head_output = torch.nan_to_num(final_head_output.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        diagnostics = {
            "step": LOCAL_GRAPH_RESIDUAL_V2_ANCHOR_STEP,
            "contract": LOCAL_GRAPH_RESIDUAL_V2_ANCHOR_CONTRACT,
            "embedding_source": self.config.embedding_source,
            "required_embedding_role": self.config.required_embedding_role,
            "final_head_name": self.final_head_name,
            "embedding_dim": int(embedding.shape[1]),
            "embedding_norm_mean": embedding.norm(dim=1).mean(),
            "embedding_norm_std": embedding.norm(dim=1).std(unbiased=False),
            "logit_abs_mean": logits.abs().mean(),
            "embedding_reproduces_logits": reproduces,
            "anchor_parameters_frozen": self.anchor_parameters_frozen(),
        }
        return HLTPartEmbeddingAnchorOutput(
            logits=logits,
            embedding=embedding,
            final_head_output=final_head_output,
            embedding_source=str(self.config.embedding_source),
            final_head_name=str(self.final_head_name),
            config=self.config,
            diagnostics=diagnostics,
        )

    def forward(self, *args: Any, return_outputs: bool = False, **kwargs: Any):
        output = self.forward_outputs(*args, **kwargs)
        if bool(return_outputs):
            return output
        return output.logits, output.embedding


def build_hlt_part_embedding_anchor(
    model: Any,
    *,
    config: HLTPartEmbeddingAnchorConfig | Mapping[str, Any] | None = None,
    payload: Mapping[str, Any] | None = None,
    checkpoint_path: str | Path | None = None,
) -> HLTPartEmbeddingAnchor:
    """Wrap an already-built exact HLT ParT baseline as a V2 embedding anchor."""

    return HLTPartEmbeddingAnchor(model, config=config, payload=payload, checkpoint_path=checkpoint_path)


def load_hlt_part_embedding_anchor(
    checkpoint_path: str | Path,
    *,
    device: str = "auto",
    strict: bool = True,
    freeze_anchor: bool = True,
    final_head_name: str | None = None,
    strict_final_head: bool = True,
) -> HLTPartEmbeddingAnchor:
    """Load the selected local-graph ``hlt_part_baseline`` checkpoint as a V2 anchor."""

    torch = require_torch()
    resolved_device = resolve_device(device)
    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location=resolved_device)
    if not isinstance(payload, Mapping):
        raise ValueError(f"HLT ParT embedding anchor checkpoint payload must be a mapping: {checkpoint_path}")
    variant = str(payload.get("variant") or "")
    if variant != LOCAL_GRAPH_MODEL_VARIANT_HLT_PART_BASELINE:
        raise ValueError(
            f"V2 HLT ParT embedding anchor requires variant {LOCAL_GRAPH_MODEL_VARIANT_HLT_PART_BASELINE!r}, "
            f"got {variant!r}"
        )
    output_contract = str(payload.get("output_contract") or "")
    if output_contract != LOCAL_GRAPH_HLT_PART_BASELINE_CONTRACT:
        raise ValueError(
            f"V2 HLT ParT embedding anchor requires output_contract {LOCAL_GRAPH_HLT_PART_BASELINE_CONTRACT!r}, "
            f"got {output_contract!r}"
        )
    label_names = _payload_label_names(payload, fallback=("QCD", "Hgg"))
    num_classes = _payload_num_classes(payload, fallback=2)
    config = HLTPartEmbeddingAnchorConfig(
        checkpoint_path=str(checkpoint_path),
        num_classes=int(num_classes),
        label_names=tuple(label_names),
        baseline_variant=LOCAL_GRAPH_RESIDUAL_V2_BASELINE_VARIANT,
        freeze_anchor=bool(freeze_anchor),
        strict_final_head=bool(strict_final_head),
        final_head_name=final_head_name,
    )
    from .train import LocalGraphTaggerTrainConfig, build_local_graph_tagger_for_config

    train_config_payload = payload.get("config")
    if not isinstance(train_config_payload, Mapping):
        raise ValueError("V2 HLT ParT embedding anchor checkpoint does not contain a train config")
    train_config = LocalGraphTaggerTrainConfig(**dict(train_config_payload))
    if train_config.variant != LOCAL_GRAPH_MODEL_VARIANT_HLT_PART_BASELINE:
        raise ValueError("V2 HLT ParT embedding anchor train config is not hlt_part_baseline")
    model = build_local_graph_tagger_for_config(train_config)
    state_dict = strip_compile_prefix_from_state_dict(_extract_state_dict(payload))
    model.load_state_dict(state_dict, strict=bool(strict))
    model = model.to(resolved_device)
    anchor = HLTPartEmbeddingAnchor(
        model,
        config=config,
        payload=payload,
        checkpoint_path=checkpoint_path,
    )
    anchor = anchor.to(resolved_device)
    anchor.eval()
    return anchor


__all__ = [
    "HLT_PART_EMBEDDING_SOURCE_FINAL_HEAD_HOOK",
    "HLTPartEmbeddingAnchor",
    "HLTPartEmbeddingAnchorConfig",
    "HLTPartEmbeddingAnchorOutput",
    "build_hlt_part_embedding_anchor",
    "load_hlt_part_embedding_anchor",
    "strip_compile_prefix_from_state_dict",
]
