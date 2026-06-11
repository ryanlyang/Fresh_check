"""Shared builders and checkpoint loading for teacher-logit reconstructors."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict

from jetclass_fresh.hlt_baseline import require_torch

from .global_transformer import GlobalTransformerReconstructor, GlobalTransformerReconstructorConfig
from .particle_cnn_reconstructor import ParticleCnnReconstructor, ParticleCnnReconstructorConfig
from .particle_flow_reconstructor import ParticleFlowReconstructor, ParticleFlowReconstructorConfig
from .particle_net_reconstructor import ParticleNetReconstructor, ParticleNetReconstructorConfig


GLOBAL_TRANSFORMER_RECONSTRUCTOR = "global_transformer"
PARTICLE_NET_RECONSTRUCTOR = "particle_net"
PARTICLE_FLOW_RECONSTRUCTOR = "particle_flow"
PARTICLE_CNN_RECONSTRUCTOR = "particle_cnn"
TEACHER_LOGIT_RECONSTRUCTOR_ARCHITECTURES = (
    GLOBAL_TRANSFORMER_RECONSTRUCTOR,
    PARTICLE_NET_RECONSTRUCTOR,
    PARTICLE_FLOW_RECONSTRUCTOR,
    PARTICLE_CNN_RECONSTRUCTOR,
)

_ARCHITECTURE_ALIASES = {
    "global_transformer": GLOBAL_TRANSFORMER_RECONSTRUCTOR,
    "globaltransformer": GLOBAL_TRANSFORMER_RECONSTRUCTOR,
    "gt": GLOBAL_TRANSFORMER_RECONSTRUCTOR,
    "transformer": GLOBAL_TRANSFORMER_RECONSTRUCTOR,
    "particle_net": PARTICLE_NET_RECONSTRUCTOR,
    "particlenet": PARTICLE_NET_RECONSTRUCTOR,
    "pn": PARTICLE_NET_RECONSTRUCTOR,
    "edgeconv": PARTICLE_NET_RECONSTRUCTOR,
    "particle_flow": PARTICLE_FLOW_RECONSTRUCTOR,
    "particleflow": PARTICLE_FLOW_RECONSTRUCTOR,
    "pfn": PARTICLE_FLOW_RECONSTRUCTOR,
    "pf": PARTICLE_FLOW_RECONSTRUCTOR,
    "deep_sets": PARTICLE_FLOW_RECONSTRUCTOR,
    "deepsets": PARTICLE_FLOW_RECONSTRUCTOR,
    "particle_cnn": PARTICLE_CNN_RECONSTRUCTOR,
    "particlecnn": PARTICLE_CNN_RECONSTRUCTOR,
    "pcnn": PARTICLE_CNN_RECONSTRUCTOR,
    "p_cnn": PARTICLE_CNN_RECONSTRUCTOR,
    "particle_conv": PARTICLE_CNN_RECONSTRUCTOR,
    "particleconv": PARTICLE_CNN_RECONSTRUCTOR,
}

_LEGACY_GLOBAL_TRANSFORMER_CONFIG_KEYS = (
    "hidden_dim",
    "num_layers",
    "num_heads",
    "num_extra_candidates",
    "dropout",
    "max_delta_logpt",
    "max_delta_eta",
    "max_delta_phi",
    "max_delta_loge",
    "parent_weight_bias",
    "extra_weight_bias",
    "max_total_extra_pt_fraction",
    "max_extra_delta_eta",
    "max_extra_delta_phi",
)


def normalize_reconstructor_architecture(architecture: str | None) -> str:
    """Normalize reconstructor architecture names.

    ``None`` intentionally maps to ``global_transformer`` so old Step 5
    checkpoints, written before the architecture field existed, keep loading.
    """

    if architecture is None:
        return GLOBAL_TRANSFORMER_RECONSTRUCTOR
    key = str(architecture).strip().lower().replace("-", "_").replace(" ", "_")
    if key in _ARCHITECTURE_ALIASES:
        return _ARCHITECTURE_ALIASES[key]
    compact = key.replace("_", "")
    if compact in _ARCHITECTURE_ALIASES:
        return _ARCHITECTURE_ALIASES[compact]
    expected = ", ".join(TEACHER_LOGIT_RECONSTRUCTOR_ARCHITECTURES)
    raise ValueError(f"Unknown teacher-logit reconstructor architecture {architecture!r}; expected one of: {expected}")


def _recognized_architecture(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return normalize_reconstructor_architecture(str(value))
    except ValueError:
        return None


def infer_reconstructor_architecture_from_payload(
    payload: Mapping[str, Any],
    *,
    architecture: str | None = None,
) -> str:
    """Infer reconstructor architecture from a checkpoint payload.

    New checkpoints should store ``reconstructor_architecture`` at top level.
    For compatibility this also checks nested config dictionaries, then falls
    back to ``global_transformer`` for legacy checkpoints.
    """

    if architecture is not None:
        return normalize_reconstructor_architecture(architecture)

    if payload.get("reconstructor_architecture") is not None:
        return normalize_reconstructor_architecture(str(payload["reconstructor_architecture"]))

    for key in ("model_config", "config"):
        config = payload.get(key)
        if not isinstance(config, Mapping):
            continue
        if config.get("reconstructor_architecture") is not None:
            return normalize_reconstructor_architecture(str(config["reconstructor_architecture"]))
        # Be permissive for future PN checkpoints while avoiding teacher
        # architecture fields such as ``teacher_architecture``.
        nested = _recognized_architecture(config.get("architecture"))
        if nested is not None:
            return nested

    return GLOBAL_TRANSFORMER_RECONSTRUCTOR


def build_teacher_logit_reconstructor(
    architecture: str | None,
    config: (
        Mapping[str, Any]
        | GlobalTransformerReconstructorConfig
        | ParticleNetReconstructorConfig
        | ParticleFlowReconstructorConfig
        | ParticleCnnReconstructorConfig
        | None
    ) = None,
):
    """Construct a teacher-logit reconstructor from an architecture/config pair."""

    arch = normalize_reconstructor_architecture(architecture)
    if arch == GLOBAL_TRANSFORMER_RECONSTRUCTOR:
        return GlobalTransformerReconstructor(GlobalTransformerReconstructorConfig.from_mapping(config or {}))
    if arch == PARTICLE_NET_RECONSTRUCTOR:
        return ParticleNetReconstructor(ParticleNetReconstructorConfig.from_mapping(config or {}))
    if arch == PARTICLE_FLOW_RECONSTRUCTOR:
        return ParticleFlowReconstructor(ParticleFlowReconstructorConfig.from_mapping(config or {}))
    if arch == PARTICLE_CNN_RECONSTRUCTOR:
        return ParticleCnnReconstructor(ParticleCnnReconstructorConfig.from_mapping(config or {}))
    raise AssertionError(f"Unhandled reconstructor architecture after normalization: {arch}")


def build_reconstructor_from_config(config: Mapping[str, Any]):
    """Construct a reconstructor from a config mapping containing its architecture.

    This is intentionally a thin convenience wrapper around
    :func:`build_teacher_logit_reconstructor`.  It expects a model config, not a
    full training config.
    """

    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    payload = dict(config)
    architecture = payload.pop("reconstructor_architecture", None)
    if architecture is None:
        architecture = payload.pop("architecture", None)
    else:
        payload.pop("architecture", None)
    return build_teacher_logit_reconstructor(architecture, payload)


def strip_compile_prefix_from_state_dict(state_dict: Mapping[str, Any]) -> Dict[str, Any]:
    """Remove torch.compile's ``_orig_mod.`` prefix when every key has it."""

    keys = list(state_dict.keys())
    if keys and all(str(key).startswith("_orig_mod.") for key in keys):
        return {str(key).removeprefix("_orig_mod."): value for key, value in state_dict.items()}
    return dict(state_dict)


def model_config_from_checkpoint_payload(payload: Mapping[str, Any], architecture: str) -> Mapping[str, Any]:
    """Return the model config, including legacy GT fallback extraction."""

    model_config = payload.get("model_config")
    if isinstance(model_config, Mapping) and model_config:
        return model_config

    if normalize_reconstructor_architecture(architecture) == GLOBAL_TRANSFORMER_RECONSTRUCTOR:
        config = dict(payload.get("config") or {})
        return {key: config[key] for key in _LEGACY_GLOBAL_TRANSFORMER_CONFIG_KEYS if key in config}

    return {}


def load_teacher_logit_reconstructor_checkpoint(
    path: str | Path,
    *,
    device,
    strict: bool = True,
    architecture: str | None = None,
    expected_architecture: str | None = None,
):
    """Load a teacher-logit reconstructor checkpoint and construct its model."""

    torch = require_torch()
    checkpoint_path = Path(path)
    payload = torch.load(checkpoint_path, map_location=device)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Reconstructor checkpoint payload must be a mapping: {checkpoint_path}")
    if "model_state_dict" not in payload:
        raise KeyError(f"Reconstructor checkpoint is missing model_state_dict: {checkpoint_path}")

    arch = infer_reconstructor_architecture_from_payload(payload, architecture=architecture)
    if expected_architecture is not None:
        expected = normalize_reconstructor_architecture(expected_architecture)
        if arch != expected:
            raise ValueError(
                f"Checkpoint architecture mismatch for {checkpoint_path}: expected {expected}, found {arch}"
            )

    model_config = model_config_from_checkpoint_payload(payload, arch)
    model = build_teacher_logit_reconstructor(arch, model_config)
    model.load_state_dict(strip_compile_prefix_from_state_dict(payload["model_state_dict"]), strict=bool(strict))
    model = model.to(device)
    model.eval()
    return model, dict(payload)
