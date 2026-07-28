"""Explicit pre-class Particle Transformer state and measurement-state interfaces."""

from __future__ import annotations

from typing import Any, Mapping

from .contracts import canonical_sha256, validate_content_hash, with_content_hash

try:
    import torch
except ImportError:  # pragma: no cover - environment dependent
    torch = None


PARTICLE_TAP_CONTRACT = "retb_particle_state_tap_v1"
MEASUREMENT_EMBED_CONTRACT = "retb_measurement_state_embedding_v1"
MEASUREMENT_STATE_NAMES = (
    "not_track_domain",
    "track_measurement_available",
    "track_measurement_missing",
)


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is required for the RETB particle-state tap")
    return torch


def build_particle_tap_contract() -> dict[str, Any]:
    return with_content_hash(
        {
            "contract": PARTICLE_TAP_CONTRACT,
            "schema_version": 1,
            "source_interface": {
                "encoder": "ParticleTransformer._forward_encoder",
                "aggregator": "ParticleTransformer._forward_aggregator",
            },
            "particle_state_shape": ["B", "N", 128],
            "tap_position": "after_final_particle_block_before_class_attention",
            "intermediate_tap": {
                "block_number_one_based": 4,
                "production_hooks_allowed": False,
            },
            "reference_head_parity_required": [
                "logits",
                "input_gradients",
                "parameter_gradients",
                "mask",
            ],
            "incidental_module_name_hooks_allowed": False,
        }
    )


def build_measurement_embedding_contract() -> dict[str, Any]:
    return with_content_hash(
        {
            "contract": MEASUREMENT_EMBED_CONTRACT,
            "schema_version": 1,
            "candidate_id": "V_MEASUREMENT_EMBED",
            "states": list(MEASUREMENT_STATE_NAMES),
            "state_indices": {
                name: index for index, name in enumerate(MEASUREMENT_STATE_NAMES)
            },
            "embedding_dimension": 128,
            "injection": (
                "after_initial_particle_embedding_before_first_particle_block"
            ),
            "canonical_particle_channels_changed": False,
            "derive_from_current_view_only": True,
            "padded_embedding_exact_zero": True,
            "ordinary_baseline_unchanged": True,
        }
    )


def _validate_exact_contract(
    payload: Mapping[str, Any],
    *,
    contract: str,
    expected: Mapping[str, Any],
) -> str:
    digest = validate_content_hash(payload, expected_contract=contract)
    semantic = dict(payload)
    semantic.pop("content_hash", None)
    semantic.pop("source", None)
    target = dict(expected)
    target.pop("content_hash", None)
    if canonical_sha256(semantic) != canonical_sha256(target):
        raise ValueError(f"{contract} differs from the locked contract")
    return digest


def validate_particle_tap_contract(payload: Mapping[str, Any]) -> str:
    return _validate_exact_contract(
        payload,
        contract=PARTICLE_TAP_CONTRACT,
        expected=build_particle_tap_contract(),
    )


def validate_measurement_embedding_contract(payload: Mapping[str, Any]) -> str:
    return _validate_exact_contract(
        payload,
        contract=MEASUREMENT_EMBED_CONTRACT,
        expected=build_measurement_embedding_contract(),
    )


def derive_measurement_states_torch(raw_tokens: Any, mask: Any) -> Any:
    """Derive the exact v3 three-state category from the current view only."""

    module = _require_torch()
    if (
        not isinstance(raw_tokens, module.Tensor)
        or raw_tokens.ndim != 3
        or int(raw_tokens.shape[-1]) != 14
    ):
        raise ValueError("raw_tokens must have shape [B,N,14]")
    if (
        not isinstance(mask, module.Tensor)
        or tuple(mask.shape)
        != (int(raw_tokens.shape[0]), 1, int(raw_tokens.shape[1]))
    ):
        raise ValueError("mask must have shape [B,1,N]")
    valid = mask[:, 0].bool()
    flags = raw_tokens[:, :, 5:10]
    if not bool(module.isfinite(flags[valid]).all()):
        raise FloatingPointError("valid PID flags contain NaN or infinity")
    binary_distance = module.minimum(flags.abs(), (flags - 1.0).abs())
    if bool((binary_distance[valid] > 1.0e-6).any()):
        raise ValueError("valid PID flags must be binary")
    binary = flags >= 0.5
    hot_count = binary.sum(dim=-1)
    if bool((hot_count[valid] > 1).any()):
        raise ValueError("multi-hot PID input is forbidden")
    category = binary.to(module.int64).argmax(dim=-1)
    category = module.where(
        (hot_count == 1) & valid,
        category,
        module.full_like(category, 5),
    )
    charged = valid & ((category == 0) | (category == 3) | (category == 4))
    track = raw_tokens[:, :, 10:14]
    available = (
        charged
        & module.isfinite(track).all(dim=-1)
        & (raw_tokens[:, :, 11] > 0)
        & (raw_tokens[:, :, 13] > 0)
    )
    states = module.zeros_like(category)
    states = module.where(charged, module.full_like(states, 2), states)
    states = module.where(available, module.ones_like(states), states)
    states = states.masked_fill(~valid, 0)
    return states


class MeasurementStateEmbedding(
    torch.nn.Module if torch is not None else object
):
    """Learned three-state embedding injected into 128-wide particle states."""

    def __init__(self, dimension: int = 128) -> None:
        module = _require_torch()
        super().__init__()
        if int(dimension) != 128:
            raise ValueError("V_MEASUREMENT_EMBED has fixed dimension 128")
        self.embedding = module.nn.Embedding(3, 128)

    def forward(self, states: Any, mask: Any) -> Any:
        module = _require_torch()
        if (
            not isinstance(states, module.Tensor)
            or states.ndim != 2
            or states.dtype != module.int64
        ):
            raise TypeError("measurement states must be int64 [B,N]")
        if tuple(mask.shape) != (int(states.shape[0]), 1, int(states.shape[1])):
            raise ValueError("measurement-state mask shape differs")
        if bool(((states < 0) | (states > 2)).any()):
            raise ValueError("measurement state lies outside 0..2")
        return self.embedding(states).masked_fill(
            ~mask.transpose(1, 2).bool(), 0.0
        )


class ReferenceParticleStateTap(
    torch.nn.Module if torch is not None else object
):
    """Use Weaver's explicit encoder/aggregator methods without hooks."""

    def __init__(self, transformer: Any) -> None:
        module = _require_torch()
        super().__init__()
        if not isinstance(transformer, module.nn.Module):
            raise TypeError("transformer must be a torch module")
        if not callable(getattr(transformer, "_forward_encoder", None)):
            raise RuntimeError("ParticleTransformer lacks _forward_encoder")
        if not callable(getattr(transformer, "_forward_aggregator", None)):
            raise RuntimeError("ParticleTransformer lacks _forward_aggregator")
        self.transformer = transformer

    def encode(
        self,
        features: Any,
        *,
        vectors: Any = None,
        mask: Any = None,
        pair_features: Any = None,
    ) -> dict[str, Any]:
        states, padding_mask = self.transformer._forward_encoder(
            features,
            v=vectors,
            mask=mask,
            uu=pair_features,
            uu_idx=None,
        )
        if bool(getattr(self.transformer, "include_global_token", False)):
            particle_states = states[:, 1:]
            particle_padding = padding_mask[:, 1:]
        else:
            particle_states = states
            particle_padding = padding_mask
        return {
            "particle_states": particle_states,
            "particle_mask": ~particle_padding,
            "encoder_states": states,
            "encoder_padding_mask": padding_mask,
        }

    def reference_logits(self, encoded: Mapping[str, Any]) -> Any:
        pooled = self.transformer._forward_aggregator(
            encoded["encoder_states"],
            encoded["encoder_padding_mask"],
        )
        classifier = getattr(self.transformer, "fc", None)
        return pooled if classifier is None else classifier(pooled)

    def forward(
        self,
        features: Any,
        *,
        vectors: Any = None,
        mask: Any = None,
        pair_features: Any = None,
        return_states: bool = False,
    ) -> Any:
        encoded = self.encode(
            features,
            vectors=vectors,
            mask=mask,
            pair_features=pair_features,
        )
        logits = self.reference_logits(encoded)
        if return_states:
            return {**encoded, "logits": logits}
        return logits


__all__ = [
    "MEASUREMENT_EMBED_CONTRACT",
    "MEASUREMENT_STATE_NAMES",
    "PARTICLE_TAP_CONTRACT",
    "MeasurementStateEmbedding",
    "ReferenceParticleStateTap",
    "build_measurement_embedding_contract",
    "build_particle_tap_contract",
    "derive_measurement_states_torch",
    "validate_measurement_embedding_contract",
    "validate_particle_tap_contract",
]
