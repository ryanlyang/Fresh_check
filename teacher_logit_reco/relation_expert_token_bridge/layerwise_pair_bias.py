"""Streamed dual-stem, per-layer RETB attention-bias provider."""

from __future__ import annotations

from typing import Any, Mapping

from teacher_logit_reco.relational_part.attention import (
    DirectionalPairStem,
    LayerwiseBiasProjection,
)

from .contracts import canonical_sha256, validate_content_hash, with_content_hash

try:
    import torch
except ImportError:  # pragma: no cover - environment dependent
    torch = None


LAYERWISE_PAIR_BIAS_CONTRACT = "retb_layerwise_pair_bias_v1"
DUAL_TOPOLOGIES = ("B_DUAL_FIXED", "B_DUAL_GATED")


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is required for RETB layerwise pair bias")
    return torch


def build_layerwise_pair_bias_contract(
    *,
    num_layers: int = 8,
    num_heads: int = 8,
) -> dict[str, Any]:
    if int(num_layers) <= 0 or int(num_heads) <= 0:
        raise ValueError("layer/head counts must be positive")
    return with_content_hash(
        {
            "contract": LAYERWISE_PAIR_BIAS_CONTRACT,
            "schema_version": 1,
            "topologies": list(DUAL_TOPOLOGIES),
            "shared_latents": {
                "base": ["B", "C_base", "N", "N"],
                "relation": ["B", "C_relation", "N", "N"],
            },
            "num_layers": int(num_layers),
            "num_heads": int(num_heads),
            "per_layer_output": ["B", int(num_heads), "N", "N"],
            "base_and_relation_stems_separate": True,
            "post_encoder_sum_only": True,
            "fixed_gate": 1.0,
            "gated_formula": "2*sigmoid(a_layer_head)",
            "gated_logit_initial_value": 0.0,
            "gated_scale_initial_value": 1.0,
            "gated_scale_open_interval": [0.0, 2.0],
            "one_projection_per_path_layer": True,
            "layer_projection_reuse_allowed": False,
            "materialize_B_L_H_N_N": False,
            "particle_block_activation_checkpointing": True,
            "mask_applied_after_dual_sum": True,
            "state_dictionary_compatible_with_single_bias_rpt": False,
            "migration_requires_authenticated_record": True,
            "concat_migration_may_claim_dual_path": False,
        }
    )


def validate_layerwise_pair_bias_contract(payload: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        payload, expected_contract=LAYERWISE_PAIR_BIAS_CONTRACT
    )
    layers = int(payload.get("num_layers", -1))
    heads = int(payload.get("num_heads", -1))
    semantic = dict(payload)
    semantic.pop("content_hash", None)
    semantic.pop("source", None)
    expected = build_layerwise_pair_bias_contract(
        num_layers=layers, num_heads=heads
    )
    expected.pop("content_hash")
    if canonical_sha256(semantic) != canonical_sha256(expected):
        raise ValueError("layerwise pair-bias contract differs")
    return digest


class LayerwisePairBiasProvider(
    torch.nn.Module if torch is not None else object
):
    """Retain two latents and emit exactly one layer bias on demand."""

    def __init__(
        self,
        *,
        base_stem: DirectionalPairStem,
        relation_stem: DirectionalPairStem,
        num_layers: int,
        num_heads: int,
        topology: str,
        force_zero_relation: bool = False,
    ) -> None:
        module = _require_torch()
        super().__init__()
        if topology not in DUAL_TOPOLOGIES:
            raise ValueError(f"unsupported dual topology {topology!r}")
        if int(base_stem.out_heads) != int(num_heads):
            raise ValueError("base stem head count differs")
        if int(relation_stem.out_heads) != int(num_heads):
            raise ValueError("relation stem head count differs")
        self.base_stem = base_stem
        self.relation_stem = relation_stem
        self.base_projections = LayerwiseBiasProjection(
            base_stem.reference_projection, num_layers=int(num_layers)
        )
        self.relation_projections = LayerwiseBiasProjection(
            relation_stem.reference_projection, num_layers=int(num_layers)
        )
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.topology = str(topology)
        self.force_zero_relation = bool(force_zero_relation)
        if self.topology == "B_DUAL_GATED":
            self.relation_gate_logits = module.nn.Parameter(
                module.zeros(self.num_layers, self.num_heads)
            )
            self.register_buffer("fixed_relation_scale", None)
        else:
            self.register_parameter("relation_gate_logits", None)
            self.register_buffer(
                "fixed_relation_scale",
                module.ones(self.num_layers, self.num_heads),
            )
        object.__setattr__(self, "_base_latent", None)
        object.__setattr__(self, "_relation_latent", None)
        object.__setattr__(self, "_pair_mask", None)
        object.__setattr__(self, "_emitted_layers", [])
        self._contract = build_layerwise_pair_bias_contract(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
        )

    def get_extra_state(self) -> dict[str, Any]:
        return {
            "contract": LAYERWISE_PAIR_BIAS_CONTRACT,
            "contract_sha256": self._contract["content_hash"],
            "topology": self.topology,
            "force_zero_relation": self.force_zero_relation,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
        }

    def set_extra_state(self, state: Any) -> None:
        if not isinstance(state, Mapping):
            raise RuntimeError("layerwise pair-bias state lacks version metadata")
        expected = self.get_extra_state()
        if dict(state) != expected:
            raise RuntimeError(
                "state dictionary is not retb_layerwise_pair_bias_v1 compatible"
            )

    def relation_scales(self) -> Any:
        module = _require_torch()
        if self.relation_gate_logits is None:
            return self.fixed_relation_scale
        return 2.0 * module.sigmoid(self.relation_gate_logits)

    def build_latents(
        self,
        base_pair_features: Any,
        relation_pair_features: Any,
        mask: Any,
    ) -> tuple[Any, Any]:
        base = self.base_stem(base_pair_features, mask)
        relation = self.relation_stem(relation_pair_features, mask)
        if self.force_zero_relation:
            relation = module_zeros_like(relation)
        return base, relation

    def bind(self, base_latent: Any, relation_latent: Any, mask: Any) -> None:
        if self._base_latent is not None:
            raise RuntimeError("layerwise provider is already bound")
        if base_latent.ndim != 4 or relation_latent.ndim != 4:
            raise ValueError("pair latents must be rank four")
        if (
            tuple(base_latent.shape[:1] + base_latent.shape[2:])
            != tuple(relation_latent.shape[:1] + relation_latent.shape[2:])
        ):
            raise ValueError("base and relation latent pair shapes differ")
        batch, _, query, context = map(int, base_latent.shape)
        if query != context or tuple(mask.shape) != (batch, 1, query):
            raise ValueError("layerwise provider mask shape differs")
        object.__setattr__(self, "_base_latent", base_latent)
        object.__setattr__(self, "_relation_latent", relation_latent)
        object.__setattr__(
            self,
            "_pair_mask",
            mask.bool().unsqueeze(-1) & mask.bool().unsqueeze(-2),
        )
        object.__setattr__(self, "_emitted_layers", [])

    def clear(self) -> None:
        object.__setattr__(self, "_base_latent", None)
        object.__setattr__(self, "_relation_latent", None)
        object.__setattr__(self, "_pair_mask", None)

    def _bias_from_latents(
        self,
        layer_index: int,
        base_latent: Any,
        relation_latent: Any,
        pair_mask: Any,
    ) -> Any:
        if int(layer_index) not in range(self.num_layers):
            raise IndexError("particle layer index lies outside the provider")
        base = LayerwiseBiasProjection._project(
            self.base_projections.projections[int(layer_index)],
            base_latent,
        )
        relation = LayerwiseBiasProjection._project(
            self.relation_projections.projections[int(layer_index)],
            relation_latent,
        )
        if self.force_zero_relation:
            relation = module_zeros_like(relation)
        scale = self.relation_scales()[int(layer_index)].view(
            1, self.num_heads, 1, 1
        )
        bias = base + scale.to(dtype=relation.dtype, device=relation.device) * relation
        return bias.masked_fill(~pair_mask, 0.0)

    def bias_for_layer(self, layer_index: int) -> Any:
        if self._base_latent is None:
            raise RuntimeError("layerwise provider is not bound")
        bias = self._bias_from_latents(
            layer_index,
            self._base_latent,
            self._relation_latent,
            self._pair_mask,
        )
        self._emitted_layers.append(
            {
                "layer_index": int(layer_index),
                "shape": list(bias.shape),
                "tensor_id": id(bias),
            }
        )
        return bias

    def checkpointed_bias_for_layer(
        self,
        layer_index: int,
        base_latent: Any,
        relation_latent: Any,
        pair_mask: Any,
    ) -> Any:
        """Projection entry used inside an activation-checkpointed block."""

        return self._bias_from_latents(
            layer_index, base_latent, relation_latent, pair_mask
        )

    def record_checkpointed_layer(
        self,
        layer_index: int,
        *,
        batch_size: int,
        sequence_length: int,
    ) -> None:
        """Record one streamed projection without retaining its tensor."""

        if int(layer_index) not in range(self.num_layers):
            raise IndexError("particle layer index lies outside the provider")
        self._emitted_layers.append(
            {
                "layer_index": int(layer_index),
                "shape": [
                    int(batch_size),
                    self.num_heads,
                    int(sequence_length),
                    int(sequence_length),
                ],
                "tensor_id": None,
                "activation_checkpointed": True,
            }
        )

    def diagnostics(self) -> dict[str, Any]:
        return {
            "contract": LAYERWISE_PAIR_BIAS_CONTRACT,
            "topology": self.topology,
            "force_zero_relation": self.force_zero_relation,
            "emitted_layers": list(self._emitted_layers),
            "emitted_layer_count": len(self._emitted_layers),
            "materialized_B_L_H_N_N": False,
            "shared_latents_bound": self._base_latent is not None,
            "relation_scales": self.relation_scales().detach().cpu().tolist(),
        }


def module_zeros_like(value: Any) -> Any:
    return _require_torch().zeros_like(value)


__all__ = [
    "DUAL_TOPOLOGIES",
    "LAYERWISE_PAIR_BIAS_CONTRACT",
    "LayerwisePairBiasProvider",
    "build_layerwise_pair_bias_contract",
    "validate_layerwise_pair_bias_contract",
]
