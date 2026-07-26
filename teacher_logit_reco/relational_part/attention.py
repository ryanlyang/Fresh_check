"""Confirmation-only layerwise-bias and relation edge-value attention."""

from __future__ import annotations

import copy
import inspect
import math
from typing import Any, Mapping, Sequence

from .pair_base import STANDARD_FOUR_CHANNELS, build_standard_four_pair_features
from .pair_builder import (
    RelationalPairBuilder,
    SUPPORTED_FAMILY_DIMENSIONS,
    canonical_supported_families,
)
from .relation_pt import valid_pair_mask

try:
    import torch
    from torch.nn import functional as F
except ImportError:  # pragma: no cover - contract imports without torch
    torch = None
    F = None


STEP6_ATTENTION_CONTRACT = "relational_part_step6_attention_v1"


def _require_torch():
    if torch is None:
        raise RuntimeError("PyTorch is required for relational attention")
    return torch


def efficient_edge_value_message(
    attention_weights: Any,
    relation_stem: Any,
    projection: Any,
) -> Any:
    """Project after relation aggregation without materializing pair values.

    Shapes are ``weights[B,H,Q,K]``, ``stem[B,D,Q,K]`` and
    ``projection[H,d_h,D]``.  The result is ``[B,H,Q,d_h]``.
    """

    module = _require_torch()
    if attention_weights.ndim != 4 or relation_stem.ndim != 4:
        raise ValueError("attention weights and relation stem must be rank four")
    if projection.ndim != 3:
        raise ValueError("edge-value projection must have shape [H,d_h,D]")
    batch, heads, query, context = map(int, attention_weights.shape)
    if tuple(relation_stem.shape[:1] + relation_stem.shape[2:]) != (
        batch,
        query,
        context,
    ):
        raise ValueError("attention and relation pair dimensions disagree")
    if int(projection.shape[0]) != heads:
        raise ValueError("projection head count disagrees with attention")
    if int(projection.shape[2]) != int(relation_stem.shape[1]):
        raise ValueError("projection input width disagrees with relation stem")
    if not (
        bool(module.isfinite(attention_weights).all())
        and bool(module.isfinite(relation_stem).all())
        and bool(module.isfinite(projection).all())
    ):
        raise FloatingPointError("edge-value inputs must be finite")
    aggregated = module.einsum("bhqk,bdqk->bhqd", attention_weights, relation_stem)
    return module.einsum("hod,bhqd->bhqo", projection, aggregated)


def explicit_edge_value_message(
    attention_weights: Any,
    relation_stem: Any,
    projection: Any,
) -> Any:
    """Small-fixture reference that explicitly forms pair-conditioned values."""

    module = _require_torch()
    pair_values = module.einsum("hod,bdqk->bhqko", projection, relation_stem)
    return (attention_weights.unsqueeze(-1) * pair_values).sum(dim=3)


class DirectionalPairStem(torch.nn.Module if torch is not None else object):
    """Split Weaver ``PairEmbed.fts_embed`` before its final head projection."""

    def __init__(self, pair_embed: Any, *, input_dimension: int) -> None:
        module = _require_torch()
        super().__init__()
        if int(getattr(pair_embed, "pairwise_lv_dim", -1)) != 0:
            raise RuntimeError("confirmation pair stem requires zero internal LV width")
        if int(getattr(pair_embed, "pairwise_input_dim", -1)) != int(
            input_dimension
        ):
            raise RuntimeError("confirmation pair-stem input dimension drifted")
        network = getattr(pair_embed, "fts_embed", None)
        if not isinstance(network, module.nn.Sequential):
            raise RuntimeError("Weaver fts_embed must be an nn.Sequential")
        children = list(network.children())
        if len(children) < 2:
            raise RuntimeError("Weaver fts_embed cannot be split")
        final = children[-1]
        if not isinstance(final, (module.nn.Conv1d, module.nn.Linear)):
            raise RuntimeError(
                "Weaver fts_embed final head projection is not Conv1d/Linear"
            )
        self.prefix = module.nn.Sequential(*children[:-1])
        self.reference_projection = final
        self.input_dimension = int(input_dimension)
        self.stem_width = int(
            final.in_channels if isinstance(final, module.nn.Conv1d)
            else final.in_features
        )
        self.out_heads = int(
            final.out_channels if isinstance(final, module.nn.Conv1d)
            else final.out_features
        )

    def forward(self, pair_features: Any, mask: Any) -> Any:
        module = _require_torch()
        if pair_features.ndim != 4:
            raise ValueError("pair features must have shape [B,C,N,N]")
        batch, channels, query, context = map(int, pair_features.shape)
        if channels != self.input_dimension or query != context:
            raise ValueError("pair features have an incompatible shape")
        if tuple(mask.shape) != (batch, 1, query):
            raise ValueError("pair mask has an incompatible shape")
        pair_mask = valid_pair_mask(mask)
        i0, _, i2, i3 = pair_mask.nonzero(as_tuple=True)
        if int(i0.numel()) == 0:
            raise ValueError("pair stem received an all-empty batch")
        gathered = pair_features.permute(0, 2, 3, 1)[i0, i2, i3]
        packed = self.prefix(gathered.T.unsqueeze(0)).squeeze(0).T
        output = packed.new_zeros(batch, query, query, self.stem_width)
        output[i0, i2, i3] = packed
        return output.permute(0, 3, 1, 2).contiguous()


class LayerwiseBiasProjection(torch.nn.Module if torch is not None else object):
    """One independent final PairStem projection per particle-attention layer."""

    def __init__(self, reference: Any, *, num_layers: int) -> None:
        module = _require_torch()
        super().__init__()
        if not isinstance(reference, (module.nn.Conv1d, module.nn.Linear)):
            raise TypeError("reference projection must be Conv1d or Linear")
        self.projections = module.nn.ModuleList(
            copy.deepcopy(reference) for _ in range(int(num_layers))
        )

    @staticmethod
    def _project(projection: Any, stem: Any) -> Any:
        module = _require_torch()
        batch, width, query, context = map(int, stem.shape)
        packed = stem.reshape(batch, width, query * context)
        if isinstance(projection, module.nn.Linear):
            result = projection(packed.transpose(1, 2)).transpose(1, 2)
        else:
            result = projection(packed)
        return result.reshape(batch, -1, query, context)

    def forward(self, stem: Any) -> tuple[Any, ...]:
        return tuple(self._project(projection, stem) for projection in self.projections)


def _value_tokens(attention: Any, value: Any) -> Any:
    module = _require_torch()
    weight = getattr(attention, "in_proj_weight", None)
    bias = getattr(attention, "in_proj_bias", None)
    embed_dim = int(getattr(attention, "embed_dim"))
    if weight is not None:
        value_weight = weight[2 * embed_dim :]
        value_bias = None if bias is None else bias[2 * embed_dim :]
        return F.linear(value, value_weight, value_bias)
    value_weight = getattr(attention, "v_proj_weight", None)
    if value_weight is None:
        raise RuntimeError("unsupported MultiheadAttention value projection")
    value_bias = None if bias is None else bias[2 * embed_dim :]
    return F.linear(value, value_weight, value_bias)


class EdgeValueAttention(torch.nn.Module if torch is not None else object):
    """A transparent MHA wrapper adding the locked relation-conditioned value."""

    def __init__(self, reference: Any, *, relation_width: int) -> None:
        module = _require_torch()
        super().__init__()
        if not isinstance(reference, module.nn.MultiheadAttention):
            raise TypeError("edge-value path requires nn.MultiheadAttention")
        self.reference = reference
        self.num_heads = int(reference.num_heads)
        self.head_dim = int(reference.embed_dim // reference.num_heads)
        self.relation_width = int(relation_width)
        self.edge_projection = module.nn.Parameter(
            module.empty(self.num_heads, self.head_dim, self.relation_width)
        )
        module.nn.init.xavier_uniform_(self.edge_projection)
        self._relation_stem = None
        self._query_valid = None
        self.last_diagnostics: dict[str, Any] | None = None

    def bind(self, relation_stem: Any, query_valid: Any) -> None:
        self._relation_stem = relation_stem
        self._query_valid = query_valid.bool()

    def clear(self) -> None:
        self._relation_stem = None
        self._query_valid = None

    def forward(self, query: Any, key: Any, value: Any, **kwargs: Any):
        module = _require_torch()
        relation = self._relation_stem
        query_valid = self._query_valid
        if relation is None or query_valid is None:
            raise RuntimeError("edge-value attention was not bound for this batch")
        requested_weights = bool(kwargs.get("need_weights", True))
        call = dict(kwargs)
        call["need_weights"] = True
        if "average_attn_weights" in inspect.signature(
            self.reference.forward
        ).parameters:
            call["average_attn_weights"] = False
        ordinary_output, weights = self.reference(query, key, value, **call)
        if weights.ndim == 3:
            weights = weights.unsqueeze(1)
        relation_message = efficient_edge_value_message(
            weights, relation, self.edge_projection
        )
        batch_first = bool(getattr(self.reference, "batch_first", False))
        if batch_first:
            concatenated = relation_message.permute(0, 2, 1, 3).reshape(
                relation_message.shape[0], relation_message.shape[2], -1
            )
            query_mask = query_valid.unsqueeze(-1)
        else:
            concatenated = relation_message.permute(2, 0, 1, 3).reshape(
                relation_message.shape[2], relation_message.shape[0], -1
            )
            query_mask = query_valid.T.unsqueeze(-1)
        projected = F.linear(
            concatenated,
            self.reference.out_proj.weight,
            bias=None,
        )
        projected = projected * query_mask.to(projected.dtype)
        output = ordinary_output + projected

        value_projected = _value_tokens(self.reference, value)
        if batch_first:
            value_heads = value_projected.reshape(
                value_projected.shape[0],
                value_projected.shape[1],
                self.num_heads,
                self.head_dim,
            ).permute(0, 2, 1, 3)
        else:
            value_heads = value_projected.permute(1, 0, 2).reshape(
                value_projected.shape[1],
                value_projected.shape[0],
                self.num_heads,
                self.head_dim,
            ).permute(0, 2, 1, 3)
        ordinary_heads = module.einsum("bhqk,bhkd->bhqd", weights, value_heads)
        numerator = relation_message.norm(dim=-1)
        denominator = ordinary_heads.norm(dim=-1) + 1.0e-6
        ratios = numerator / denominator
        self.last_diagnostics = {
            "per_head_mean_norm_ratio": [
                float(
                    ratios[:, head].masked_select(query_valid).mean().detach().cpu()
                )
                if bool(query_valid.any())
                else None
                for head in range(self.num_heads)
            ],
            "masked_query_count": int((~query_valid).sum().detach().cpu()),
            "materialized_pair_value_tensor": False,
        }
        returned_weights = weights if requested_weights else None
        return output, returned_weights


class ConfirmationArchitectureParticleTransformer(
    torch.nn.Module if torch is not None else object
):
    """Base4/selected relation model with layerwise bias and optional edge values."""

    def __init__(
        self,
        *,
        transformer: Any,
        weaver_module: Any,
        families: Sequence[str],
        normalization_artifact: Mapping[str, Any] | None,
        region_normalization_artifact: Mapping[str, Any] | None = None,
        edge_value: bool = False,
    ) -> None:
        module = _require_torch()
        super().__init__()
        self.mod = transformer
        self.families = (
            canonical_supported_families(families) if tuple(families) else ()
        )
        self.edge_value = bool(edge_value)
        combined = STANDARD_FOUR_CHANNELS + sum(
            SUPPORTED_FAMILY_DIMENSIONS[name] for name in self.families
        )
        pair_embed = getattr(self.mod, "pair_embed", None)
        self.pair_stem = DirectionalPairStem(
            pair_embed, input_dimension=combined
        )
        self.mod.pair_embed = None
        blocks = getattr(self.mod, "blocks", None)
        if not isinstance(blocks, module.nn.ModuleList) or len(blocks) == 0:
            raise RuntimeError("Weaver model lacks particle-attention blocks")
        self.layer_bias = LayerwiseBiasProjection(
            self.pair_stem.reference_projection, num_layers=len(blocks)
        )
        self.pair_builder = None
        if self.families:
            if normalization_artifact is None:
                raise ValueError("selected relations require a normalizer")
            self.pair_builder = RelationalPairBuilder(
                self.families,
                normalization_artifact=normalization_artifact,
                weaver_module=weaver_module,
                region_normalization_artifact=region_normalization_artifact,
            )
        object.__setattr__(self, "_weaver_module", weaver_module)
        self.edge_attention = module.nn.ModuleList()
        if self.edge_value:
            for block in blocks:
                attention = getattr(block, "attn", None)
                wrapped = EdgeValueAttention(
                    attention, relation_width=self.pair_stem.stem_width
                )
                block.attn = wrapped
                self.edge_attention.append(wrapped)

    def _pairs(
        self,
        features: Any,
        vectors: Any,
        mask: Any,
        raw_tokens: Any,
        region_trees: Any,
    ) -> Any:
        if self.pair_builder is None:
            return build_standard_four_pair_features(
                vectors, mask=mask, module=self._weaver_module
            )
        return self.pair_builder(
            features, vectors, mask, raw_tokens, region_trees
        )

    def forward(
        self,
        points: Any,
        features: Any,
        lorentz_vectors: Any,
        mask: Any,
        raw_tokens: Any | None = None,
        region_trees: Any | None = None,
    ) -> Any:
        del points
        valid = mask.bool()
        clean_features = features.masked_fill(~valid, 0)
        clean_vectors = lorentz_vectors.masked_fill(~valid, 0)
        clean_raw = raw_tokens
        if raw_tokens is not None:
            clean_raw = raw_tokens.masked_fill(~valid[:, 0].unsqueeze(-1), 0)
        pairs = self._pairs(
            clean_features, clean_vectors, valid, clean_raw, region_trees
        )
        stem = self.pair_stem(pairs, valid)
        biases = self.layer_bias(stem)
        x = self.mod.embed(clean_features)
        padding = ~valid[:, 0]
        try:
            for index, block in enumerate(self.mod.blocks):
                if self.edge_value:
                    self.edge_attention[index].bind(stem, valid[:, 0])
                x = block(x, padding_mask=padding, attn_mask=biases[index])
                if self.edge_value:
                    self.edge_attention[index].clear()
            cls = self.mod.cls_token.expand(x.shape[0], 1, -1)
            for block in self.mod.cls_blocks:
                cls = block(x, x_cls=cls, padding_mask=padding)
            return self.mod.fc(self.mod.norm(cls).squeeze(1))
        finally:
            for attention in self.edge_attention:
                attention.clear()

    def diagnostics(self) -> dict[str, Any]:
        return {
            "families": list(self.families),
            "architecture": (
                "layerwise_bias_and_edge_value"
                if self.edge_value
                else "layerwise_bias"
            ),
            "pair_stem_evaluations_per_batch": 1,
            "layer_count": len(self.layer_bias.projections),
            "edge_value": [
                attention.last_diagnostics for attention in self.edge_attention
            ],
        }


def build_step6_attention_contract(
    *,
    run_id: str,
    families: Sequence[str],
    edge_value: bool,
    model_contract_sha256: str,
) -> dict[str, Any]:
    from .contracts import require_sha256, with_content_hash

    canonical = (
        canonical_supported_families(families) if tuple(families) else ()
    )
    expected_base = run_id.startswith("RPT_BASE_")
    if expected_base != (len(canonical) == 0):
        raise ValueError("base architectural controls must contain base4 only")
    return with_content_hash(
        {
            "contract": STEP6_ATTENTION_CONTRACT,
            "schema_version": 1,
            "run_id": str(run_id),
            "model_contract_sha256": require_sha256(
                model_contract_sha256, name="model_contract_sha256"
            ),
            "enabled_relations": ["base4", *canonical],
            "new_relation_families": list(canonical),
            "pair_stem_evaluations_per_batch": 1,
            "layerwise_bias": {
                "projection_count": 8,
                "shared_final_projection": False,
            },
            "edge_value": {
                "enabled": bool(edge_value),
                "per_layer_per_head": bool(edge_value),
                "linear_bias": False,
                "aggregate_relation_before_projection": True,
                "materialize_B_H_N_N_dh": False,
                "invalid_query_and_key_relations_zero": True,
                "epsilon": 1.0e-6,
            },
            "base4_architecture_control": expected_base,
            "initialization": "from_scratch",
        }
    )


__all__ = [
    "STEP6_ATTENTION_CONTRACT",
    "ConfirmationArchitectureParticleTransformer",
    "DirectionalPairStem",
    "EdgeValueAttention",
    "LayerwiseBiasProjection",
    "build_step6_attention_contract",
    "efficient_edge_value_message",
    "explicit_edge_value_message",
]
