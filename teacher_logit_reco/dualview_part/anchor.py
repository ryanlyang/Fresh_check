"""Step 2 HLT Particle Transformer anchor loading.

The dual-view branch must start from the strong HLT ParT baseline.  This module
wraps that checkpoint as an anchor that can return logits and, when the wrapped
Weaver model does not expose an internal CLS token, a small parallel HLT summary
context for downstream residual heads.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping

from jetclass_fresh.dual_view import build_part_inputs_torch
from jetclass_fresh.hlt_cache import fixed_hlt_params_dict, fixed_hlt_params_from_strength, load_hlt_metadata
from jetclass_fresh.hlt_baseline import (
    ParticleTransformerHLTClassifier,
    build_particle_transformer_classifier,
    require_torch,
    resolve_device,
)
from jetclass_fresh.jetclass_data import LABEL_NAMES
from jetclass_fresh.part_inputs import PF_FEATURE_NAMES
from .config import (
    DUALVIEW_PART_ANCHOR_ARCHITECTURE,
    DUALVIEW_PART_HLT_DEGRADATION_STRENGTH,
    DUALVIEW_PART_NUM_CLASSES,
    DUALVIEW_PART_POSITIVE_CLASS_NAME,
    DUALVIEW_PART_SOURCE_LABEL_NAMES,
)

try:  # Keep module importable on machines without the training stack.
    import torch as _torch
except ImportError:  # pragma: no cover - environment dependent
    _torch = None

if _torch is None:  # pragma: no cover - environment dependent
    class _ModuleBase:
        pass
else:
    _ModuleBase = _torch.nn.Module


DUALVIEW_PART_STEP2 = "reliability_gated_dualview_part_step2_hlt_anchor_loader"
DUALVIEW_PART_ANCHOR_CONTRACT = "hlt_part_anchor_logits_plus_optional_summary_context_v1"


@dataclass(frozen=True)
class HLTPartAnchorConfig:
    """Configuration for loading and wrapping the HLT ParT anchor."""

    checkpoint_path: str | None = None
    num_classes: int = DUALVIEW_PART_NUM_CLASSES
    model_size: str = "base"
    device: str = "auto"
    strict: bool = True
    freeze_anchor: bool = True
    context_mode: str = "summary"
    context_dim: int = 128
    summary_hidden_dim: int = 128
    summary_dropout: float = 0.0
    max_constits: int = 128
    weight_threshold: float = 0.0
    label_names: tuple[str, ...] = DUALVIEW_PART_SOURCE_LABEL_NAMES
    positive_class_name: str = DUALVIEW_PART_POSITIVE_CLASS_NAME

    def validate(self) -> None:
        if int(self.num_classes) <= 1:
            raise ValueError("num_classes must be greater than 1")
        if self.model_size not in {"base", "tiny"}:
            raise ValueError("model_size must be 'base' or 'tiny'")
        if self.context_mode not in {"none", "summary"}:
            raise ValueError("context_mode must be 'none' or 'summary'")
        if int(self.context_dim) <= 0:
            raise ValueError("context_dim must be positive")
        if int(self.summary_hidden_dim) <= 0:
            raise ValueError("summary_hidden_dim must be positive")
        if float(self.summary_dropout) < 0.0 or float(self.summary_dropout) >= 1.0:
            raise ValueError("summary_dropout must be in [0, 1)")
        if int(self.max_constits) <= 0:
            raise ValueError("max_constits must be positive")
        if float(self.weight_threshold) < 0.0:
            raise ValueError("weight_threshold must be nonnegative")
        if len(tuple(self.label_names)) != int(self.num_classes):
            raise ValueError("label_names length must match num_classes")
        if self.positive_class_name not in set(self.label_names):
            raise ValueError("positive_class_name must appear in label_names")

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["label_names"] = list(self.label_names)
        return data


@dataclass
class HLTPartAnchorOutput:
    """Forward output from the HLT anchor wrapper."""

    logits: Any
    context: Any | None = None
    summary_features: Any | None = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "logits": self.logits,
            "context": self.context,
            "summary_features": self.summary_features,
            "diagnostics": dict(self.diagnostics),
        }


def strip_compile_prefix_from_state_dict(state_dict: Mapping[str, Any]) -> Dict[str, Any]:
    """Remove torch.compile ``_orig_mod.`` prefixes when present."""

    keys = list(state_dict.keys())
    if keys and all(str(key).startswith("_orig_mod.") for key in keys):
        return {str(key).removeprefix("_orig_mod."): value for key, value in state_dict.items()}
    return dict(state_dict)


def _infer_num_classes(payload: Mapping[str, Any], fallback: int) -> int:
    if payload.get("num_classes") is not None:
        return int(payload["num_classes"])
    label_names = payload.get("label_names")
    if label_names is not None:
        return int(len(label_names))
    model_config = payload.get("model_config")
    if isinstance(model_config, Mapping) and model_config.get("num_classes") is not None:
        return int(model_config["num_classes"])
    return int(fallback)


def _infer_model_size(payload: Mapping[str, Any], fallback: str) -> str:
    config = payload.get("config")
    if isinstance(config, Mapping):
        if config.get("baseline_model_size") is not None:
            return str(config["baseline_model_size"])
        if config.get("model_size") is not None:
            return str(config["model_size"])
    return str(fallback)


def _infer_label_names(payload: Mapping[str, Any], num_classes: int, fallback: tuple[str, ...]) -> tuple[str, ...]:
    label_names = payload.get("label_names")
    if label_names is not None:
        names = tuple(str(item) for item in label_names)
        if len(names) == int(num_classes):
            return names
    if len(fallback) == int(num_classes):
        return tuple(str(item) for item in fallback)
    return tuple(str(item) for item in LABEL_NAMES[: int(num_classes)])


def _iter_nested_mappings(payload: Mapping[str, Any]):
    stack = [payload]
    seen: set[int] = set()
    while stack:
        item = stack.pop()
        if not isinstance(item, Mapping):
            continue
        item_id = id(item)
        if item_id in seen:
            continue
        seen.add(item_id)
        yield item
        for key in (
            "config",
            "train_config",
            "model_config",
            "metadata",
            "hlt_metadata",
            "source_metadata",
            "run_config",
            "slurm_run_config",
        ):
            nested = item.get(key)
            if isinstance(nested, Mapping):
                stack.append(nested)


def _metadata_value(payload: Mapping[str, Any], *keys: str) -> Any:
    for mapping in _iter_nested_mappings(payload):
        for key in keys:
            if key in mapping and mapping[key] is not None:
                return mapping[key]
    return None


def _canonical_hlt_params() -> dict[str, float]:
    return fixed_hlt_params_dict(fixed_hlt_params_from_strength(DUALVIEW_PART_HLT_DEGRADATION_STRENGTH))


def _params_match_canonical(params: Mapping[str, Any]) -> bool:
    expected = _canonical_hlt_params()
    for key, expected_value in expected.items():
        if key not in params:
            return False
        if abs(float(params[key]) - float(expected_value)) > 1.0e-9:
            return False
    return True


def _check_hlt_contract_metadata(
    payload: Mapping[str, Any],
    checkpoint_path: Path,
    *,
    contract_hlt_cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "checked_label_names": True,
        "checked_num_classes": True,
        "checked_positive_class": True,
        "hlt_view_evidence": None,
        "hlt_degradation_evidence": None,
    }
    experiment_step = str(payload.get("experiment_step") or _metadata_value(payload, "experiment_step") or "")
    if "offline" in experiment_step.lower():
        raise ValueError(f"HLT anchor checkpoint looks like an offline checkpoint: experiment_step={experiment_step!r}")

    architecture = _metadata_value(payload, "architecture", "tagger_architecture", "baseline_architecture")
    if architecture is not None:
        normalized = str(architecture).strip().lower()
        if normalized not in {DUALVIEW_PART_ANCHOR_ARCHITECTURE, "particle_transformer", "hlt_part"}:
            raise ValueError(f"HLT anchor must be a ParT/particle-transformer checkpoint, got {architecture!r}")

    source_view = _metadata_value(payload, "view", "source_view", "input_view", "training_view")
    if source_view is not None:
        normalized = str(source_view).strip().lower()
        if normalized not in {"fixed_hlt", "hlt"}:
            raise ValueError(f"HLT anchor must be trained on the fixed HLT view, got view={source_view!r}")
        evidence["hlt_view_evidence"] = {"source": "payload", "value": str(source_view)}

    explicit_strength = _metadata_value(payload, "hlt_degradation_strength", "degradation_strength")
    if explicit_strength is not None:
        if abs(float(explicit_strength) - DUALVIEW_PART_HLT_DEGRADATION_STRENGTH) > 1.0e-9:
            raise ValueError(
                "HLT anchor degradation strength mismatch: "
                f"{explicit_strength} != {DUALVIEW_PART_HLT_DEGRADATION_STRENGTH}"
            )
        evidence["hlt_degradation_evidence"] = {"source": "payload_strength", "value": float(explicit_strength)}

    hlt_params = _metadata_value(payload, "hlt_params")
    if hlt_params is not None:
        if not isinstance(hlt_params, Mapping) or not _params_match_canonical(hlt_params):
            raise ValueError("HLT anchor hlt_params do not match the canonical HLT0.6 profile")
        evidence["hlt_degradation_evidence"] = {"source": "payload_hlt_params"}

    cache_candidates: list[tuple[str, Path]] = []
    if contract_hlt_cache_dir is not None:
        cache_candidates.append(("provided_hlt_cache_dir", Path(contract_hlt_cache_dir)))
    payload_cache_dir = _metadata_value(payload, "cache_dir", "hlt_cache_dir")
    if payload_cache_dir is not None:
        cache_candidates.append(("payload_cache_dir", Path(str(payload_cache_dir))))

    if evidence["hlt_degradation_evidence"] is None:
        for cache_source, cache_path in cache_candidates:
            if evidence["hlt_degradation_evidence"] is not None:
                break
            if not cache_path.exists():
                continue
            for split in ("model_train", "model_val", "stack_train", "stack_val", "final_test"):
                try:
                    metadata = load_hlt_metadata(cache_path, split)
                except FileNotFoundError:
                    continue
                params = metadata.get("hlt_params")
                if not isinstance(params, Mapping) or not _params_match_canonical(params):
                    raise ValueError(f"HLT anchor cache metadata for {split} does not match canonical HLT0.6 params")
                view = str(metadata.get("view", "")).strip().lower()
                if view and view not in {"fixed_hlt", "hlt"}:
                    raise ValueError(f"HLT anchor cache metadata for {split} is not fixed HLT: view={view!r}")
                evidence["hlt_degradation_evidence"] = {
                    "source": "cache_metadata",
                    "cache_source": cache_source,
                    "split": split,
                }
                evidence["hlt_view_evidence"] = {"source": "cache_metadata", "split": split, "value": view or "fixed_hlt"}
                break

    if evidence["hlt_degradation_evidence"] is None:
        raise ValueError(
            "Cannot verify HLT anchor degradation strength. "
            f"Checkpoint must declare hlt_degradation_strength/hlt_params or point to an accessible HLT cache: {checkpoint_path}"
        )
    if evidence["hlt_view_evidence"] is None:
        evidence["hlt_view_evidence"] = {"source": "assumed_from_hlt_degradation_metadata"}
    return evidence


def _validate_canonical_anchor_contract(
    payload: Mapping[str, Any],
    *,
    checkpoint_path: Path,
    num_classes: int,
    label_names: tuple[str, ...],
    contract_hlt_cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    if int(num_classes) != DUALVIEW_PART_NUM_CLASSES:
        raise ValueError(f"HLT anchor must have {DUALVIEW_PART_NUM_CLASSES} classes, got {num_classes}")
    if tuple(label_names) != DUALVIEW_PART_SOURCE_LABEL_NAMES:
        raise ValueError(f"HLT anchor labels must be {DUALVIEW_PART_SOURCE_LABEL_NAMES}, got {tuple(label_names)}")
    if DUALVIEW_PART_POSITIVE_CLASS_NAME not in tuple(label_names):
        raise ValueError(f"HLT anchor labels must contain positive class {DUALVIEW_PART_POSITIVE_CLASS_NAME}")
    return _check_hlt_contract_metadata(
        payload,
        checkpoint_path,
        contract_hlt_cache_dir=contract_hlt_cache_dir,
    )


def _build_hlt_part_model_from_payload(
    payload: Mapping[str, Any],
    *,
    num_classes: int,
    model_size: str,
):
    model_config = dict(payload.get("model_config") or {})
    model_config.pop("architecture", None)
    if model_config:
        return ParticleTransformerHLTClassifier(**model_config)
    return build_particle_transformer_classifier(num_classes=int(num_classes), model_size=str(model_size))


class HLTPartSummaryEncoder(_ModuleBase):
    """Small HLT summary encoder used when ParT internals are not exposed."""

    def __init__(
        self,
        *,
        input_dim: int = len(PF_FEATURE_NAMES),
        hidden_dim: int = 128,
        context_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        torch = require_torch()
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.context_dim = int(context_dim)
        self.dropout = float(dropout)
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(2 * self.input_dim + 1),
            torch.nn.Linear(2 * self.input_dim + 1, self.hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(self.hidden_dim, self.context_dim),
            torch.nn.LayerNorm(self.context_dim),
        )

    def forward(self, features: Any, mask: Any) -> tuple[Any, Any]:
        torch = require_torch()
        features = features.float()
        if features.ndim != 3:
            raise ValueError(f"features must have shape (batch, channels, tokens), got {tuple(features.shape)}")
        if mask.ndim == 3:
            token_mask = mask[:, 0, :].bool()
        elif mask.ndim == 2:
            token_mask = mask.bool()
        else:
            raise ValueError(f"mask must have shape (batch, tokens) or (batch, 1, tokens), got {tuple(mask.shape)}")
        if features.shape[1] != self.input_dim:
            raise ValueError(f"expected {self.input_dim} feature channels, got {features.shape[1]}")
        valid = token_mask[:, None, :].float()
        count = valid.sum(dim=2).clamp_min(1.0)
        mean = (features * valid).sum(dim=2) / count
        neg_inf = torch.full_like(features, -1.0e9)
        max_values = torch.where(token_mask[:, None, :], features, neg_inf).max(dim=2).values
        max_values = torch.where(torch.isfinite(max_values), max_values, torch.zeros_like(max_values))
        count_feature = torch.log1p(token_mask.float().sum(dim=1, keepdim=True)) / 5.0
        summary = torch.cat([mean, max_values, count_feature], dim=1)
        summary = torch.nan_to_num(summary, nan=0.0, posinf=0.0, neginf=0.0)
        context = self.net(summary)
        return context, summary


class HLTPartAnchor(_ModuleBase):
    """Loaded HLT ParT anchor with optional context summary."""

    def __init__(
        self,
        model: Any,
        *,
        config: HLTPartAnchorConfig,
        payload: Mapping[str, Any] | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.payload = dict(payload or {})
        self.checkpoint_path = None if checkpoint_path is None else str(checkpoint_path)
        self.model = model
        self.summary_encoder = None
        if config.context_mode == "summary":
            self.summary_encoder = HLTPartSummaryEncoder(
                input_dim=len(PF_FEATURE_NAMES),
                hidden_dim=int(config.summary_hidden_dim),
                context_dim=int(config.context_dim),
                dropout=float(config.summary_dropout),
            )
        if config.freeze_anchor:
            self.freeze_anchor_parameters()

    @property
    def context_dim(self) -> int:
        return 0 if self.summary_encoder is None else int(self.config.context_dim)

    def freeze_anchor_parameters(self) -> None:
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.model.eval()

    def unfreeze_anchor_parameters(self) -> None:
        for param in self.model.parameters():
            param.requires_grad_(True)

    def freeze_all_parameters(self) -> None:
        for param in self.parameters():
            param.requires_grad_(False)
        self.eval()

    def anchor_parameters_frozen(self) -> bool:
        return all(not bool(param.requires_grad) for param in self.model.parameters())

    def trainable_parameter_count(self) -> int:
        return int(sum(param.numel() for param in self.parameters() if param.requires_grad))

    def metadata(self) -> Dict[str, Any]:
        return {
            "experiment_step": DUALVIEW_PART_STEP2,
            "output_contract": DUALVIEW_PART_ANCHOR_CONTRACT,
            "checkpoint_path": self.checkpoint_path,
            "config": self.config.to_dict(),
            "payload_epoch": self.payload.get("epoch"),
            "payload_experiment_step": self.payload.get("experiment_step"),
            "payload_output_contract": self.payload.get("output_contract"),
            "anchor_frozen": self.anchor_parameters_frozen(),
            "context_dim": self.context_dim,
        }

    def build_inputs(self, tokens: Any, mask: Any, *, weights: Any = None) -> Dict[str, Any]:
        return build_part_inputs_torch(
            tokens,
            mask,
            weights=weights,
            max_constits=int(self.config.max_constits),
            weight_threshold=float(self.config.weight_threshold),
        )

    def forward_inputs(self, inputs: Mapping[str, Any], *, return_context: bool = True) -> HLTPartAnchorOutput:
        torch = require_torch()
        logits = self.model(
            inputs["points"],
            inputs["features"],
            inputs["lorentz_vectors"],
            inputs["mask"],
        )
        if not torch.isfinite(logits).all():
            raise FloatingPointError("HLT ParT anchor produced non-finite logits")

        context = None
        summary_features = None
        if return_context and self.summary_encoder is not None:
            context, summary_features = self.summary_encoder(inputs["features"], inputs["mask"])
            if not torch.isfinite(context).all():
                raise FloatingPointError("HLT ParT summary encoder produced non-finite context")

        diagnostics = {
            "anchor_logits_shape": list(logits.shape),
            "anchor_logits_abs_mean": logits.detach().abs().mean(),
            "anchor_context_available": context is not None,
            "anchor_context_dim": 0 if context is None else int(context.shape[-1]),
        }
        return HLTPartAnchorOutput(
            logits=logits,
            context=context,
            summary_features=summary_features,
            diagnostics=diagnostics,
        )

    def forward_tokens(self, tokens: Any, mask: Any, *, weights: Any = None, return_context: bool = True):
        return self.forward_inputs(self.build_inputs(tokens, mask, weights=weights), return_context=return_context)

    def forward(self, points: Any, features: Any, lorentz_vectors: Any, mask: Any):
        """Compatibility forward returning only logits."""

        return self.forward_inputs(
            {
                "points": points,
                "features": features,
                "lorentz_vectors": lorentz_vectors,
                "mask": mask,
            },
            return_context=False,
        ).logits


def build_hlt_part_anchor(
    model: Any,
    *,
    config: HLTPartAnchorConfig | None = None,
    payload: Mapping[str, Any] | None = None,
    checkpoint_path: str | Path | None = None,
) -> HLTPartAnchor:
    """Wrap an already-built HLT ParT-like model as an anchor."""

    cfg = config or HLTPartAnchorConfig()
    return HLTPartAnchor(model, config=cfg, payload=payload, checkpoint_path=checkpoint_path)


def load_hlt_part_anchor(
    checkpoint_path: str | Path,
    *,
    device: str = "auto",
    freeze_anchor: bool = True,
    strict: bool = True,
    context_mode: str = "summary",
    context_dim: int = 128,
    summary_hidden_dim: int = 128,
    summary_dropout: float = 0.0,
    max_constits: int = 128,
    weight_threshold: float = 0.0,
    fallback_num_classes: int = DUALVIEW_PART_NUM_CLASSES,
    fallback_model_size: str = "base",
    fallback_label_names: tuple[str, ...] = DUALVIEW_PART_SOURCE_LABEL_NAMES,
    enforce_canonical_contract: bool = True,
    contract_hlt_cache_dir: str | Path | None = None,
) -> HLTPartAnchor:
    """Load a trained QCD/Hgg HLT0.6 ParT checkpoint as the dual-view anchor."""

    torch = require_torch()
    resolved_device = resolve_device(device)
    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location=resolved_device)
    if not isinstance(payload, Mapping):
        raise ValueError(f"HLT anchor checkpoint payload must be a mapping: {checkpoint_path}")
    state_dict = payload.get("model_state_dict", payload.get("state_dict"))
    if state_dict is None:
        raise KeyError(f"HLT anchor checkpoint is missing model_state_dict/state_dict: {checkpoint_path}")
    num_classes = _infer_num_classes(payload, fallback=fallback_num_classes)
    model_size = _infer_model_size(payload, fallback=fallback_model_size)
    label_names = _infer_label_names(payload, num_classes, fallback=fallback_label_names)
    contract_evidence = {}
    if bool(enforce_canonical_contract):
        contract_evidence = _validate_canonical_anchor_contract(
            payload,
            checkpoint_path=checkpoint_path,
            num_classes=int(num_classes),
            label_names=tuple(label_names),
            contract_hlt_cache_dir=contract_hlt_cache_dir,
        )
    positive_class_name = (
        DUALVIEW_PART_POSITIVE_CLASS_NAME if DUALVIEW_PART_POSITIVE_CLASS_NAME in label_names else label_names[-1]
    )
    config = HLTPartAnchorConfig(
        checkpoint_path=str(checkpoint_path),
        num_classes=int(num_classes),
        model_size=str(model_size),
        device=str(device),
        strict=bool(strict),
        freeze_anchor=bool(freeze_anchor),
        context_mode=str(context_mode),
        context_dim=int(context_dim),
        summary_hidden_dim=int(summary_hidden_dim),
        summary_dropout=float(summary_dropout),
        max_constits=int(max_constits),
        weight_threshold=float(weight_threshold),
        label_names=tuple(label_names),
        positive_class_name=str(positive_class_name),
    )
    model = _build_hlt_part_model_from_payload(payload, num_classes=num_classes, model_size=model_size)
    model.load_state_dict(strip_compile_prefix_from_state_dict(state_dict), strict=bool(strict))
    model = model.to(resolved_device)
    anchor = HLTPartAnchor(model, config=config, payload=payload, checkpoint_path=checkpoint_path)
    if contract_evidence:
        anchor.payload["dualview_anchor_contract_evidence"] = contract_evidence
    anchor = anchor.to(resolved_device)
    anchor.eval()
    return anchor


__all__ = [
    "DUALVIEW_PART_ANCHOR_CONTRACT",
    "DUALVIEW_PART_ANCHOR_ARCHITECTURE",
    "DUALVIEW_PART_STEP2",
    "HLTPartAnchor",
    "HLTPartAnchorConfig",
    "HLTPartAnchorOutput",
    "HLTPartSummaryEncoder",
    "build_hlt_part_anchor",
    "load_hlt_part_anchor",
    "strip_compile_prefix_from_state_dict",
]
