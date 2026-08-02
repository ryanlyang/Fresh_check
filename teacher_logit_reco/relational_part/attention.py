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


STEP6_ATTENTION_CONTRACT = "relational_part_step6_attention_v2"
ATTENTION_ANGULAR_BAND_EDGES = (0.0, 0.05, 0.10, 0.20, 0.40)


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


def _is_weaver_custom_attention(value: Any, module: Any) -> bool:
    """Recognize Weaver's batch-first Attention without importing Weaver."""

    return (
        not isinstance(value, module.nn.MultiheadAttention)
        and isinstance(getattr(value, "in_proj", None), module.nn.Linear)
        and isinstance(getattr(value, "out_proj", None), module.nn.Linear)
        and isinstance(getattr(value, "num_heads", None), int)
        and isinstance(getattr(value, "head_dim", None), int)
        and callable(getattr(value, "q_norm", None))
        and callable(getattr(value, "k_norm", None))
    )


def _weaver_custom_attention_weights(
    attention: Any,
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
) -> Any:
    """Compute Weaver's custom attention weights, including training dropout."""

    module = _require_torch()
    query = args[0] if len(args) > 0 else kwargs["query"]
    key = args[1] if len(args) > 1 else kwargs["key"]
    value = args[2] if len(args) > 2 else kwargs["value"]
    batch, query_count, _ = map(int, query.shape)
    context_count = int(key.shape[1])
    heads = int(attention.num_heads)
    head_dim = int(attention.head_dim)
    q, k, _ = F._in_projection_packed(
        query,
        key,
        value,
        attention.in_proj.weight,
        attention.in_proj.bias,
    )
    q = attention.q_norm(
        q.view(batch, query_count, heads, head_dim)
    ).transpose(1, 2)
    k = attention.k_norm(
        k.view(batch, context_count, heads, head_dim)
    ).transpose(1, 2)
    logits = (q * math.sqrt(1.0 / float(head_dim))) @ k.transpose(-2, -1)

    def additive(mask_value: Any) -> Any:
        if mask_value.dtype == module.bool:
            return module.zeros_like(
                mask_value, dtype=logits.dtype
            ).masked_fill(mask_value, -module.inf)
        return mask_value.to(dtype=logits.dtype, device=logits.device)

    attention_mask = kwargs.get(
        "attn_mask", args[4] if len(args) > 4 else None
    )
    if attention_mask is not None:
        attention_mask = additive(attention_mask.to(logits.device))
        if attention_mask.ndim == 2:
            attention_mask = attention_mask.view(
                1, 1, query_count, context_count
            )
        elif attention_mask.ndim == 3:
            if int(attention_mask.shape[0]) == batch * heads:
                attention_mask = attention_mask.view(
                    batch, heads, query_count, context_count
                )
            else:
                attention_mask = attention_mask.unsqueeze(1)
        logits = logits + attention_mask
    padding_mask = kwargs.get(
        "key_padding_mask", args[3] if len(args) > 3 else None
    )
    if padding_mask is not None:
        padding_mask = additive(padding_mask.to(logits.device))
        logits = logits + padding_mask.view(batch, 1, 1, context_count)
    probabilities = F.softmax(logits, dim=-1)
    if bool(attention.training) and float(attention.dropout) > 0:
        probabilities = F.dropout(
            probabilities,
            p=float(attention.dropout),
            training=True,
        )
    return probabilities


def capture_multihead_attention_weights(
    model: Any,
    forward_call: Any,
) -> list[Any]:
    """Capture weights with Weaver sequence trimming disabled and restored."""

    module = _require_torch()
    trimmers = [
        value
        for value in model.modules()
        if value.__class__.__name__ == "SequenceTrimmer"
        and hasattr(value, "enabled")
        and hasattr(value, "_counter")
    ]
    trimmer_states = [
        (
            value,
            bool(value.enabled),
            value._counter.detach().clone(),
        )
        for value in trimmers
    ]
    multihead_attentions = [
        value
        for value in model.modules()
        if isinstance(value, module.nn.MultiheadAttention)
    ]
    custom_attentions = [
        value
        for value in model.modules()
        if _is_weaver_custom_attention(value, module)
    ]
    originals = []
    captured: list[Any] = []
    try:
        for trimmer, _, _ in trimmer_states:
            trimmer.enabled = False
        for attention in multihead_attentions:
            original = attention.forward
            originals.append((attention, original))

            def wrapped(*args: Any, _original=original, **kwargs: Any):
                requested = bool(kwargs.get("need_weights", True))
                kwargs["need_weights"] = True
                if "average_attn_weights" in inspect.signature(
                    _original
                ).parameters:
                    kwargs["average_attn_weights"] = False
                output, weights = _original(*args, **kwargs)
                if weights is None:
                    raise RuntimeError(
                        "diagnostic MultiheadAttention returned no weights"
                    )
                if weights.ndim == 3:
                    weights = weights.unsqueeze(1)
                captured.append(weights.detach())
                return output, weights if requested else None

            attention.forward = wrapped
        for attention in custom_attentions:
            original = attention.forward
            originals.append((attention, original))

            def wrapped_custom(*args: Any, _attention=attention,
                               _original=original, **kwargs: Any):
                output = _original(*args, **kwargs)
                captured.append(
                    _weaver_custom_attention_weights(
                        _attention, args, kwargs
                    ).detach()
                )
                return output

            attention.forward = wrapped_custom
        forward_call()
    finally:
        for attention, original in originals:
            attention.forward = original
        for trimmer, enabled, counter in trimmer_states:
            trimmer.enabled = enabled
            with module.no_grad():
                trimmer._counter.copy_(counter)
    return captured


def attention_allocation_diagnostics(
    weights_by_layer: Sequence[Any],
    lorentz_vectors: Any,
    mask: Any,
    *,
    expected_particle_layer_count: int = 8,
) -> dict[str, Any]:
    """Aggregate actual attention fractions by pT group and angular band."""

    module = _require_torch()
    valid = mask[:, 0].bool()
    particle_count = int(valid.shape[1])
    pt = module.hypot(lorentz_vectors[:, 0], lorentz_vectors[:, 1])
    masked_pt = pt.masked_fill(~valid, -module.inf)
    leading_pt = masked_pt.amax(dim=1, keepdim=True)
    leading = valid & pt.eq(leading_pt)
    below_leading = masked_pt.masked_fill(leading, -module.inf)
    subleading_pt = below_leading.amax(dim=1, keepdim=True)
    has_subleading = module.isfinite(subleading_pt)
    subleading = valid & has_subleading & pt.eq(subleading_pt)
    soft = valid & ~(leading | subleading)
    safe_pt = pt.clamp_min(1.0e-30)
    eta = module.asinh(lorentz_vectors[:, 2] / safe_pt)
    phi = module.atan2(lorentz_vectors[:, 1], lorentz_vectors[:, 0])
    delta_eta = eta.unsqueeze(-1) - eta.unsqueeze(-2)
    delta_phi = module.atan2(
        module.sin(phi.unsqueeze(-1) - phi.unsqueeze(-2)),
        module.cos(phi.unsqueeze(-1) - phi.unsqueeze(-2)),
    )
    delta_r = module.hypot(delta_eta, delta_phi)
    pair_valid = valid.unsqueeze(-1) & valid.unsqueeze(-2)
    band_masks = []
    labels = []
    edges = ATTENTION_ANGULAR_BAND_EDGES
    for index, left in enumerate(edges):
        if index + 1 < len(edges):
            right = edges[index + 1]
            selected = (
                delta_r.le(right)
                if index == 0
                else delta_r.gt(left) & delta_r.le(right)
            )
            labels.append(
                f"[{left:.2f},{right:.2f}]"
                if index == 0
                else f"({left:.2f},{right:.2f}]"
            )
        else:
            selected = delta_r.gt(left)
            labels.append(f"({left:.2f},inf)")
        band_masks.append(selected & pair_valid)

    particle_weights = []
    captured_shapes = []
    for weights in weights_by_layer:
        shape = tuple(int(value) for value in weights.shape)
        captured_shapes.append(list(shape))
        if len(shape) != 4:
            continue
        if (
            shape[0] == int(valid.shape[0])
            and shape[2] == particle_count
            and shape[3] == particle_count
        ):
            particle_weights.append(weights)
    if len(particle_weights) != int(expected_particle_layer_count):
        raise RuntimeError(
            "attention diagnostics require exactly "
            f"{int(expected_particle_layer_count)} particle self-attention "
            "layers with [batch,heads,N,N] weights; captured shapes were "
            f"{captured_shapes}"
        )

    layers = []
    for layer, weights in enumerate(particle_weights):
        query_valid = valid
        per_head = []
        for head in range(int(weights.shape[1])):
            values = weights[:, head]
            query_count = int(query_valid.sum().detach().cpu())

            def context_fraction(context_mask: Any) -> tuple[float, float]:
                fraction = (
                    values * context_mask.unsqueeze(1).to(values)
                ).sum(-1)
                selected = fraction.masked_select(query_valid)
                return (
                    float(selected.mean().detach().cpu()),
                    float(selected.sum().detach().cpu()),
                )

            angular = {}
            angular_statistics = {}
            for label, selected_band in zip(labels, band_masks):
                fraction = (
                    values * selected_band.to(values)
                ).sum(-1)
                selected = fraction.masked_select(query_valid)
                angular[label] = float(selected.mean().detach().cpu())
                angular_statistics[label] = {
                    "kind": "ratio",
                    "numerator": float(selected.sum().detach().cpu()),
                    "denominator": query_count,
                }
            probability = values.clamp_min(1.0e-30)
            entropy_values = (-(probability * probability.log()).sum(-1))
            entropy_selected = entropy_values.masked_select(query_valid)
            maximum_selected = values.amax(-1).masked_select(query_valid)
            leading_value, leading_sum = context_fraction(leading)
            subleading_value, subleading_sum = context_fraction(subleading)
            soft_value, soft_sum = context_fraction(soft)
            per_head.append(
                {
                    "leading_context_fraction": leading_value,
                    "subleading_context_fraction": subleading_value,
                    "soft_context_fraction": soft_value,
                    "angular_band_fractions": {
                        **angular,
                        "_population_statistics": angular_statistics,
                    },
                    "attention_entropy": float(
                        entropy_selected.mean().detach().cpu()
                    ),
                    "maximum_attention_weight": float(
                        maximum_selected.mean().detach().cpu()
                    ),
                    "_population_statistics": {
                        "leading_context_fraction": {
                            "kind": "ratio",
                            "numerator": leading_sum,
                            "denominator": query_count,
                        },
                        "subleading_context_fraction": {
                            "kind": "ratio",
                            "numerator": subleading_sum,
                            "denominator": query_count,
                        },
                        "soft_context_fraction": {
                            "kind": "ratio",
                            "numerator": soft_sum,
                            "denominator": query_count,
                        },
                        "attention_entropy": {
                            "kind": "ratio",
                            "numerator": float(
                                entropy_selected.sum().detach().cpu()
                            ),
                            "denominator": query_count,
                        },
                        "maximum_attention_weight": {
                            "kind": "ratio",
                            "numerator": float(
                                maximum_selected.sum().detach().cpu()
                            ),
                            "denominator": query_count,
                        },
                    },
                }
            )
        layers.append({"layer": layer, "per_head": per_head})
    return {
        "angular_band_edges": list(ATTENTION_ANGULAR_BAND_EDGES),
        "angular_band_endpoint_policy": (
            "[0,0.05],(0.05,0.10],(0.10,0.20],(0.20,0.40],(0.40,inf)"
        ),
        "context_group_policy": (
            "leading=max_tied_pt; subleading=next_distinct_tied_pt; "
            "soft=remaining_valid"
        ),
        "layers": layers,
        "captured_particle_attention_layer_count": len(layers),
        "captured_attention_shapes": captured_shapes,
        "particle_attention_shape_policy": (
            "accept only [batch,heads,particle_count,particle_count]; "
            "class/cross-attention shapes are excluded"
        ),
        "expected_particle_attention_layer_count": int(
            expected_particle_layer_count
        ),
        "sequence_trimming_policy": (
            "disable only during diagnostic forward; restore enabled flag "
            "and counter exactly"
        ),
    }


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
        projection_index = next(
            (
                index
                for index in range(len(children) - 1, -1, -1)
                if isinstance(
                    children[index], (module.nn.Conv1d, module.nn.Linear)
                )
            ),
            None,
        )
        if projection_index is None:
            raise RuntimeError(
                "Weaver fts_embed has no terminal Conv1d/Linear projection"
            )
        final = children[projection_index]
        suffix = children[projection_index + 1 :]
        if any(
            not isinstance(
                child,
                (
                    module.nn.BatchNorm1d,
                    module.nn.Identity,
                    module.nn.Dropout,
                    module.nn.GELU,
                    module.nn.ReLU,
                    module.nn.SiLU,
                ),
            )
            for child in suffix
        ):
            raise RuntimeError(
                "Weaver fts_embed has an unsupported post-projection tail"
            )
        self.prefix = module.nn.Sequential(*children[:projection_index])
        projection = (
            final
            if not suffix
            else module.nn.Sequential(final, *suffix)
        )
        # Keep only an unregistered construction template.  The original
        # shared final projection (including Weaver's optional trailing
        # normalization) is deliberately absent from the active architecture;
        # every registered projection tail is layer-specific.
        object.__setattr__(
            self, "_reference_projection_template", copy.deepcopy(projection)
        )
        self.input_dimension = int(input_dimension)
        self.stem_width = int(
            final.in_channels if isinstance(final, module.nn.Conv1d)
            else final.in_features
        )
        self.out_heads = int(
            final.out_channels if isinstance(final, module.nn.Conv1d)
            else final.out_features
        )

    @property
    def reference_projection(self) -> Any:
        return self._reference_projection_template

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
        projection = self._leading_projection(reference)
        if projection is None:
            raise TypeError(
                "reference projection must begin with Conv1d or Linear"
            )
        self.projections = module.nn.ModuleList(
            copy.deepcopy(reference) for _ in range(int(num_layers))
        )

    @staticmethod
    def _leading_projection(projection: Any) -> Any | None:
        module = _require_torch()
        if isinstance(projection, (module.nn.Conv1d, module.nn.Linear)):
            return projection
        if isinstance(projection, module.nn.Sequential):
            children = list(projection.children())
            if children and isinstance(
                children[0], (module.nn.Conv1d, module.nn.Linear)
            ):
                return children[0]
        return None

    @staticmethod
    def _project(projection: Any, stem: Any) -> Any:
        module = _require_torch()
        batch, width, query, context = map(int, stem.shape)
        packed = stem.reshape(batch, width, query * context)
        leading = LayerwiseBiasProjection._leading_projection(projection)
        if leading is None:
            raise TypeError("layerwise projection layout is unsupported")
        if isinstance(leading, module.nn.Linear):
            result = projection(packed.transpose(1, 2)).transpose(1, 2)
        else:
            result = projection(packed)
        return result.reshape(batch, -1, query, context)

    def forward(self, stem: Any) -> tuple[Any, ...]:
        return tuple(self._project(projection, stem) for projection in self.projections)


def _value_tokens(attention: Any, value: Any) -> Any:
    module = _require_torch()
    packed_projection = getattr(attention, "in_proj", None)
    if isinstance(packed_projection, module.nn.Linear):
        embed_dim = int(packed_projection.out_features // 3)
        return F.linear(
            value,
            packed_projection.weight[2 * embed_dim :],
            (
                None
                if packed_projection.bias is None
                else packed_projection.bias[2 * embed_dim :]
            ),
        )
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
        native_attention = isinstance(reference, module.nn.MultiheadAttention)
        custom_attention = _is_weaver_custom_attention(reference, module)
        if not (native_attention or custom_attention):
            raise TypeError(
                "edge-value path requires nn.MultiheadAttention or Weaver Attention"
            )
        self.reference = reference
        self._weaver_custom_attention = bool(custom_attention)
        self.embed_dim = int(
            getattr(
                reference,
                "embed_dim",
                getattr(reference.out_proj, "in_features"),
            )
        )
        self.num_heads = int(reference.num_heads)
        self.head_dim = int(
            getattr(reference, "head_dim", self.embed_dim // self.num_heads)
        )
        self.batch_first = (
            True
            if self._weaver_custom_attention
            else bool(reference.batch_first)
        )
        self.dropout = float(reference.dropout)
        self.relation_width = int(relation_width)
        self.edge_projection = module.nn.Parameter(
            module.empty(self.num_heads, self.head_dim, self.relation_width)
        )
        module.nn.init.xavier_uniform_(self.edge_projection)
        self._relation_stem = None
        self._query_valid = None
        self.collect_diagnostics = False
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
        if (
            int(self.edge_projection.detach().count_nonzero()) == 0
            or int(relation.detach().count_nonzero()) == 0
        ):
            output, returned_weights = self.reference(
                query, key, value, **kwargs
            )
            if self.collect_diagnostics:
                self.last_diagnostics = {
                    "per_head_mean_norm_ratio": [0.0] * self.num_heads,
                    "masked_query_count": int(
                        (~query_valid).sum().detach().cpu()
                    ),
                    "materialized_pair_value_tensor": False,
                    "exact_zero_message_reference_path": True,
                }
            return output, returned_weights
        if self._weaver_custom_attention:
            unsupported = {"need_weights", "average_attn_weights"} & set(kwargs)
            if unsupported:
                raise TypeError(
                    "Weaver Attention does not accept "
                    + ", ".join(sorted(unsupported))
                )
            weights = _weaver_custom_attention_weights(
                self.reference,
                (query, key, value),
                kwargs,
            )
            custom_manual_attention = bool(
                self.reference.training and self.dropout > 0
            )
            if not custom_manual_attention:
                ordinary_output, returned_reference_weights = self.reference(
                    query, key, value, **kwargs
                )
                if returned_reference_weights is not None:
                    raise RuntimeError(
                        "Weaver Attention unexpectedly returned attention weights"
                    )
        else:
            custom_manual_attention = False
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
        if self._weaver_custom_attention:
            value_projected = _value_tokens(self.reference, value)
            ordinary_heads = value_projected.reshape(
                value_projected.shape[0],
                value_projected.shape[1],
                self.num_heads,
                self.head_dim,
            ).permute(0, 2, 1, 3)
            ordinary_heads = module.einsum(
                "bhqk,bhkd->bhqd", weights, ordinary_heads
            )
            relation_message = relation_message * query_valid[:, None, :, None]
            if bool(getattr(self.reference, "headwise_attn_output_gate", False)):
                gate = module.sigmoid(self.reference.gate_proj(query))
                gate = gate.permute(0, 2, 1).unsqueeze(-1)
                if custom_manual_attention:
                    ordinary_heads = ordinary_heads * gate
                relation_message = relation_message * gate
            elif bool(
                getattr(self.reference, "elementwise_attn_output_gate", False)
            ):
                gate = module.sigmoid(self.reference.gate_proj(query)).reshape(
                    query.shape[0],
                    query.shape[1],
                    self.num_heads,
                    self.head_dim,
                )
                gate = gate.permute(0, 2, 1, 3)
                if custom_manual_attention:
                    ordinary_heads = ordinary_heads * gate
                relation_message = relation_message * gate
            if custom_manual_attention:
                combined_heads = ordinary_heads + relation_message
                combined = combined_heads.permute(0, 2, 1, 3).reshape(
                    combined_heads.shape[0], combined_heads.shape[2], -1
                )
                output = self.reference.out_proj(combined)
        batch_first = self.batch_first
        if custom_manual_attention:
            pass
        elif batch_first:
            concatenated = relation_message.permute(0, 2, 1, 3).reshape(
                relation_message.shape[0], relation_message.shape[2], -1
            )
            query_mask = query_valid.unsqueeze(-1)
        else:
            concatenated = relation_message.permute(2, 0, 1, 3).reshape(
                relation_message.shape[2], relation_message.shape[0], -1
            )
            query_mask = query_valid.T.unsqueeze(-1)
        if not custom_manual_attention:
            projected = F.linear(
                concatenated,
                self.reference.out_proj.weight,
                bias=None,
            )
            projected = projected * query_mask.to(projected.dtype)
            output = ordinary_output + projected

        if not self.collect_diagnostics:
            self.last_diagnostics = None
            returned_weights = (
                None
                if self._weaver_custom_attention
                else weights if requested_weights else None
            )
            return output, returned_weights

        value_projected = _value_tokens(self.reference, value)
        if self._weaver_custom_attention:
            pass
        elif batch_first:
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
        if not self._weaver_custom_attention:
            ordinary_heads = module.einsum(
                "bhqk,bhkd->bhqd", weights, value_heads
            )
        numerator = relation_message.norm(dim=-1)
        denominator = ordinary_heads.norm(dim=-1) + 1.0e-6
        ratios = numerator / denominator
        ratio_distributions = []
        for head in range(self.num_heads):
            selected_ratios = ratios[:, head].masked_select(query_valid).float()
            ratio_distributions.append(
                {
                    "count": int(selected_ratios.numel()),
                    "mean": float(selected_ratios.mean().detach().cpu()),
                    "standard_deviation": float(
                        selected_ratios.std(unbiased=False).detach().cpu()
                    ),
                    "quantiles": {
                        str(quantile): float(
                            module.quantile(selected_ratios, quantile)
                            .detach()
                            .cpu()
                        )
                        for quantile in (0.0, 0.25, 0.5, 0.75, 1.0)
                    },
                }
            )
        self.last_diagnostics = {
            "per_head_norm_ratio_distribution": ratio_distributions,
            "masked_query_count": int((~query_valid).sum().detach().cpu()),
            "materialized_pair_value_tensor": False,
            "per_head_attention_entropy": [
                float(
                    (
                        -weights[:, head].clamp_min(1.0e-30)
                        * weights[:, head].clamp_min(1.0e-30).log()
                    )
                    .sum(-1)
                    .masked_select(query_valid)
                    .mean()
                    .detach()
                    .cpu()
                )
                for head in range(self.num_heads)
            ],
            "per_head_maximum_attention_weight": [
                float(
                    weights[:, head]
                    .amax(-1)
                    .masked_select(query_valid)
                    .mean()
                    .detach()
                    .cpu()
                )
                for head in range(self.num_heads)
            ],
        }
        returned_weights = (
            None
            if self._weaver_custom_attention
            else weights if requested_weights else None
        )
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
        object.__setattr__(self, "edge_attention", [])
        if self.edge_value:
            for block in blocks:
                attention = getattr(block, "attn", None)
                wrapped = EdgeValueAttention(
                    attention, relation_width=self.pair_stem.stem_width
                )
                block.attn = wrapped
                self.edge_attention.append(wrapped)
        self._last_pair_diagnostics: dict[str, Any] | None = None
        self.collect_diagnostics = False

    def _pairs(
        self,
        features: Any,
        vectors: Any,
        mask: Any,
        raw_tokens: Any,
        region_trees: Any,
    ) -> tuple[Any, dict[str, float | None]]:
        if self.pair_builder is None:
            return (
                build_standard_four_pair_features(
                    vectors, mask=mask, module=self._weaver_module
                ),
                {},
            )
        if not self.collect_diagnostics:
            return (
                self.pair_builder(
                    features, vectors, mask, raw_tokens, region_trees
                ),
                {},
            )
        details = self.pair_builder(
            features,
            vectors,
            mask,
            raw_tokens,
            region_trees,
            return_details=True,
        )
        pair_mask = details["pair_mask"]
        norms = {}
        for family, encoded in details["encoded"].items():
            selected = encoded.masked_select(pair_mask.expand_as(encoded))
            norms[family] = (
                float(selected.float().square().mean().sqrt().detach().cpu())
                if int(selected.numel())
                else None
            )
        return details["combined"], norms

    def forward(
        self,
        points: Any,
        features: Any,
        lorentz_vectors: Any,
        mask: Any,
        raw_tokens: Any | None = None,
        region_trees: Any | None = None,
        pair_transform: Any | None = None,
    ) -> Any:
        del points
        valid = mask.bool()
        clean_features = features.masked_fill(~valid, 0)
        clean_vectors = lorentz_vectors.masked_fill(~valid, 0)
        clean_raw = raw_tokens
        if raw_tokens is not None:
            clean_raw = raw_tokens.masked_fill(~valid[:, 0].unsqueeze(-1), 0)
        pairs, activation_norms = self._pairs(
            clean_features, clean_vectors, valid, clean_raw, region_trees
        )
        if pair_transform is not None:
            if self.training:
                raise RuntimeError(
                    "semantic pair perturbations are inference-only"
                )
            expected_shape = tuple(pairs.shape)
            pairs = pair_transform(
                pairs,
                mask=valid,
                features=clean_features,
                lorentz_vectors=clean_vectors,
                raw_tokens=clean_raw,
                region_trees=region_trees,
            )
            if tuple(pairs.shape) != expected_shape:
                raise ValueError("semantic pair transform changed tensor shape")
        stem = self.pair_stem(pairs, valid)
        biases = self.layer_bias(stem)
        if self.collect_diagnostics:
            pair_mask = valid_pair_mask(valid)[:, 0]
            bias_summaries = []
            for bias in biases:
                per_head = []
                for head in range(int(bias.shape[1])):
                    values = bias[:, head].masked_select(pair_mask)
                    per_head.append(
                        {
                            "mean": float(values.mean().detach().cpu()),
                            "standard_deviation": float(
                                values.std(unbiased=False).detach().cpu()
                            ),
                            "absolute_mean": float(
                                values.abs().mean().detach().cpu()
                            ),
                            "maximum_absolute": float(
                                values.abs().max().detach().cpu()
                            ),
                        }
                    )
                bias_summaries.append(per_head)
            selected_stem = stem.masked_select(
                pair_mask.unsqueeze(1).expand_as(stem)
            )
            self._last_pair_diagnostics = {
                "family_encoded_activation_rms": activation_norms,
                "pair_stem_activation_rms": float(
                    selected_stem.float().square().mean().sqrt().detach().cpu()
                ),
                "pair_bias_per_layer_head": bias_summaries,
            }
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

    def diagnostics(
        self,
        points: Any,
        features: Any,
        lorentz_vectors: Any,
        mask: Any,
        raw_tokens: Any | None = None,
        region_trees: Any | None = None,
    ) -> dict[str, Any]:
        was_training = bool(self.training)
        self.eval()
        self.collect_diagnostics = True
        for attention in self.edge_attention:
            attention.collect_diagnostics = True
        try:
            with _require_torch().no_grad():
                captured = capture_multihead_attention_weights(
                    self,
                    lambda: self.forward(
                        points,
                        features,
                        lorentz_vectors,
                        mask,
                        raw_tokens,
                        region_trees,
                        None,
                    ),
                )
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
                    attention.last_diagnostics
                    for attention in self.edge_attention
                ],
                "pair": self._last_pair_diagnostics,
                "attention_allocation": attention_allocation_diagnostics(
                    captured, lorentz_vectors, mask
                ),
            }
        finally:
            self.collect_diagnostics = False
            for attention in self.edge_attention:
                attention.collect_diagnostics = False
            if was_training:
                self.train()


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
            "schema_version": 2,
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
                "initialization": (
                    "independent_parameters_initialized_as_deep_copies_of_one_"
                    "Weaver_initialized_final_projection_tail"
                ),
                "split_boundary": (
                    "immediately_before_final_Conv1d_or_Linear_head_projection"
                ),
                "projection_tail": (
                    "final_head_projection_plus_supported_trailing_modules_"
                    "including_Weaver_output_BatchNorm1d"
                ),
            },
            "edge_value": {
                "enabled": bool(edge_value),
                "per_layer_per_head": bool(edge_value),
                "linear_bias": False,
                "aggregate_relation_before_projection": True,
                "materialize_B_H_N_N_dh": False,
                "invalid_query_and_key_relations_zero": True,
                "epsilon": 1.0e-6,
                "projection_initialization": "torch_xavier_uniform",
            },
            "base4_architecture_control": expected_base,
            "initialization": "from_scratch",
        }
    )


__all__ = [
    "STEP6_ATTENTION_CONTRACT",
    "ConfirmationArchitectureParticleTransformer",
    "ATTENTION_ANGULAR_BAND_EDGES",
    "attention_allocation_diagnostics",
    "capture_multihead_attention_weights",
    "DirectionalPairStem",
    "EdgeValueAttention",
    "LayerwiseBiasProjection",
    "build_step6_attention_contract",
    "efficient_edge_value_message",
    "explicit_edge_value_message",
]
