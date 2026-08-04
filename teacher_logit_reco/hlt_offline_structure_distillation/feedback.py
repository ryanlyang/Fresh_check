"""Stage-E predicted-structure feedback graphs and deterministic registry."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .auxiliary import (
    GLOBAL_PHYSICAL_TARGETS,
    PAIR_TARGETS,
    global_auxiliary_loss,
    heteroscedastic_component_mask,
    pair_auxiliary_loss,
    select_utility_row,
)
from .baselines import component_seed
from .contracts import (
    AUXILIARY_CHECKPOINT_CONTRACT,
    AUXILIARY_COMPLETION_CONTRACT,
    AUXILIARY_PREDICTION_CONTRACT,
    FEEDBACK_INTERFACE_CONTRACT,
    FEEDBACK_RESULT_CONTRACT,
    FEEDBACK_SELECTION_CONTRACT,
    SINGLE_FAMILY_SELECTION_CONTRACT,
    STAGE_E_PLAN_CONTRACT,
    canonical_sha256,
    require_sha256,
    validate_content_hash,
    with_content_hash,
)
from .heads import GlobalTargetHead, PairTargetHead
from .taps import HBaseParticleTransformer
from .target_schemas import (
    target_component_availability_groups,
    target_declarations,
)

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


class ExactHLTPairFeatureBuilder(
    torch.nn.Module if torch is not None else object
):
    """Rebuild a normalized pair target from same-event HLT inputs only.

    The identity-free train normalizer and TRACK validity conventions are
    buffers, so a selected exact-HLT reference remains self-contained after
    checkpoint/export loading.  No offline target, cached pair matrix, or
    learned side predictor is part of this path.
    """

    _SENTINEL_FIELDS = ("d0", "d0err", "dz", "dzerr")

    def __init__(self, target_id: str, component_count: int) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for exact HLT feedback")
        super().__init__()
        if target_id not in PAIR_TARGETS or int(component_count) <= 0:
            raise ValueError("exact HLT builder requires a registered pair target")
        self.target_id = str(target_id)
        self.register_buffer("configured", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("centers", torch.zeros(component_count))
        self.register_buffer("scales", torch.ones(component_count))
        self.register_buffer(
            "normalize_components", torch.zeros(component_count, dtype=torch.bool)
        )
        self.register_buffer("d0_uncertainty_floor", torch.tensor(0.0))
        self.register_buffer("dz_uncertainty_floor", torch.tensor(0.0))
        self.register_buffer("sentinel_values", torch.full((4,), float("nan")))
        self.register_buffer("target_normalizer_digest", torch.zeros(32, dtype=torch.uint8))
        self.register_buffer("relation_normalizer_digest", torch.zeros(32, dtype=torch.uint8))

    @staticmethod
    def _digest_tensor(value: str) -> Any:
        from .contracts import require_sha256

        digest = bytes.fromhex(require_sha256(value, name="normalizer.content_hash"))
        return torch.tensor(list(digest), dtype=torch.uint8)

    def configure(
        self,
        *,
        target_normalizer: Mapping[str, Any],
        relation_normalizer: Mapping[str, Any],
    ) -> None:
        from teacher_logit_reco.relational_part.relation_track import (
            normalize_track_sentinel_policy,
        )
        from .normalization import validate_target_normalizer

        validate_target_normalizer(target_normalizer)
        rows = [
            row
            for row in target_normalizer["targets"]
            if row["target_id"] == self.target_id
        ]
        if len(rows) != 1 or int(rows[0]["component_count"]) != self.centers.numel():
            raise ValueError("exact HLT target normalizer coordinate differs")
        components = sorted(
            rows[0]["components"], key=lambda row: int(row["component_index"])
        )
        if [int(row["component_index"]) for row in components] != list(
            range(self.centers.numel())
        ):
            raise ValueError("exact HLT target normalizer components differ")
        centers = torch.tensor([float(row["center"]) for row in components])
        scales = torch.tensor([float(row["scale"]) for row in components])
        normalize = torch.tensor(
            [bool(row["normalize"]) for row in components], dtype=torch.bool
        )
        if not bool(torch.isfinite(centers).all()) or not bool(
            torch.isfinite(scales).all()
        ) or bool((scales <= 0).any()):
            raise ValueError("exact HLT target normalization is nonfinite")
        floors = relation_normalizer.get("track_uncertainty_floors", {})
        d0_floor = float(floors.get("d0", {}).get("floor", 0.0))
        dz_floor = float(floors.get("dz", {}).get("floor", 0.0))
        if not all(math.isfinite(value) and value >= 0 for value in (d0_floor, dz_floor)):
            raise ValueError("exact HLT TRACK uncertainty floor differs")
        sentinel = normalize_track_sentinel_policy(
            relation_normalizer.get("track_sentinel_policy")
        )
        sentinel_values = torch.tensor(
            [
                float("nan") if sentinel[name] is None else float(sentinel[name])
                for name in self._SENTINEL_FIELDS
            ]
        )
        with torch.no_grad():
            self.centers.copy_(centers)
            self.scales.copy_(scales)
            self.normalize_components.copy_(normalize)
            self.d0_uncertainty_floor.fill_(d0_floor)
            self.dz_uncertainty_floor.fill_(dz_floor)
            self.sentinel_values.copy_(sentinel_values)
            self.target_normalizer_digest.copy_(
                self._digest_tensor(str(target_normalizer["content_hash"]))
            )
            self.relation_normalizer_digest.copy_(
                self._digest_tensor(str(relation_normalizer["content_hash"]))
            )
            self.configured.fill_(True)

    def _sentinel_policy(self) -> dict[str, float | None]:
        return {
            name: (
                None
                if math.isnan(float(value))
                else float(value)
            )
            for name, value in zip(
                self._SENTINEL_FIELDS, self.sentinel_values.detach().cpu().tolist()
            )
        }

    def forward(
        self,
        raw_tokens: Any,
        mask: Any,
        lorentz_vectors: Any,
        region_trees: Sequence[Mapping[str, Any]] | None = None,
        particle_indices: Any | None = None,
    ) -> Any:
        from .extractors import ExtractorResources, extract_registered_target
        from .normalization import NORMALIZED_CLIP

        if not bool(self.configured.item()):
            raise RuntimeError("exact HLT runtime normalizers were not configured")
        if raw_tokens is None:
            raise ValueError("exact HLT reference requires raw HLT tokens")
        target = extract_registered_target(
            self.target_id,
            raw_tokens,
            mask,
            resources=ExtractorResources(
                d0_uncertainty_floor=float(self.d0_uncertainty_floor.item()),
                dz_uncertainty_floor=float(self.dz_uncertainty_floor.item()),
                sentinel_policy=self._sentinel_policy(),
            ),
            vectors=lorentz_vectors,
            trees=region_trees,
        )
        device, dtype = raw_tokens.device, raw_tokens.dtype
        values = target.values.to(device=device, dtype=dtype)
        valid = target.loss_mask.to(device=device)
        if particle_indices is not None:
            indices = particle_indices.to(device=device, dtype=torch.long)
            if (
                indices.ndim != 2
                or int(indices.shape[0]) != int(values.shape[0])
                or bool((indices < 0).any())
                or bool((indices >= values.shape[-1]).any())
            ):
                raise ValueError("exact HLT particle-index trace differs")
            row_indices = indices[:, None, :, None].expand(
                -1, values.shape[1], -1, values.shape[-1]
            )
            values = values.gather(2, row_indices)
            valid = valid.gather(2, row_indices)
            column_indices = indices[:, None, None, :].expand(
                -1, values.shape[1], values.shape[2], -1
            )
            values = values.gather(3, column_indices)
            valid = valid.gather(3, column_indices)
        shape = (1, -1, 1, 1)
        centers = self.centers.to(device=device, dtype=dtype).view(shape)
        scales = self.scales.to(device=device, dtype=dtype).view(shape)
        normalized = ((values - centers) / scales).clamp(*NORMALIZED_CLIP)
        normalize = self.normalize_components.to(device=device).view(shape)
        output = torch.where(normalize, normalized, values).masked_fill(~valid, 0.0)
        pair_mask = valid.any(dim=1)
        if self.target_id == "T_HLT_REGION_PAIR_8":
            # The physical REGION target stores its unordered coordinate only
            # in the upper triangle; attention consumes the symmetric relation.
            output = output + output.transpose(2, 3)
            pair_mask = pair_mask | pair_mask.transpose(1, 2)
        return output.permute(0, 2, 3, 1).contiguous(), pair_mask


MANDATORY_FEEDBACK = (
    ("T_OFFLINE_TRACK_32", "FB_TOKEN"),
    ("T_OFFLINE_TRACK_32", "FB_FILM"),
    ("T_HLT_TRACK_PAIR_13", "FB_PAIR"),
    ("T_HLT_REGION_PAIR_8", "FB_PAIR"),
)
GLOBAL_CONTROLS = (
    "ZERO",
    "DISABLED_LOSS",
    "SHUFFLED_PREDICTION",
    "UNRESTRICTED",
    "MEAN_ONLY",
    "ORACLE_SUB",
    "ORACLE_TRAINED",
)
PAIR_CONTROLS = (
    "AUX_ONLY",
    "ZERO_GATE",
    "DETACHED",
    "NO_SEMANTIC_LOSS",
    "SHUFFLED",
    "UNRESTRICTED_MLP",
    "EXACT_HLT",
)


def feedback_interface_contract() -> dict[str, Any]:
    return with_content_hash(
        {
            "contract": FEEDBACK_INTERFACE_CONTRACT,
            "schema_version": 3,
            "tap": "post_block_4",
            "consumer_blocks": [5, 6, 7, 8],
            "token": {
                "slots": 4,
                "dimension": 128,
                "separate_branch": True,
                "live_particle_sequence_expanded": False,
                "gate": "2*tanh(raw_gamma)",
                "raw_gate_initialization": 0.0,
                "output_projection_bias": 0.0,
            },
            "film": {
                "scale": "1+0.1*tanh(s)",
                "shift": "0.1*tanh(b)",
                "final_projection_initialization": 0.0,
            },
            "pair": {
                "gate": "2*tanh(raw_alpha_per_head)",
                "raw_gate_initialization": 0.0,
                "warmup": "min(T,max(1,ceil(0.05*T)))",
                "invalid_and_diagonal": 0.0,
                "offline_pair_feedback_allowed": False,
                "particle_alignment": (
                    "official_weaver_trimmer_identity_trace_v1"
                ),
            },
            "gradient_paths": ["END_TO_END", "DETACHED", "AUX_ONLY"],
            "probabilistic_feedback": {
                "heteroscedastic": "mean_and_clipped_log_variance",
                "categorical": "complete_probability_vector",
                "sampling_primary": False,
                "global_packing": (
                    "availability_gated_values_then_all_availability_"
                    "probabilities_then_registered_HET_log_variances"
                ),
            },
            "oracle_primary_or_export_allowed": False,
        }
    )


def gate_warmup_updates(total_updates: int) -> int:
    total = int(total_updates)
    if total <= 0:
        raise ValueError("feedback gate schedule requires positive total updates")
    return min(total, max(1, math.ceil(0.05 * total)))


def global_feedback_layout(target_id: str, parameterization: str) -> dict[str, Any]:
    declarations = {row.target_id: row for row in target_declarations()}
    if target_id not in declarations:
        raise ValueError(f"unknown global feedback target {target_id}")
    declaration = declarations[target_id]
    groups = target_component_availability_groups(
        target_id, declaration.components
    )
    order = tuple(dict.fromkeys(groups))
    indices = tuple(order.index(group) for group in groups)
    heteroscedastic = (
        heteroscedastic_component_mask(target_id, declaration.components)
        if parameterization == "HET"
        else tuple(False for _ in declaration.components)
    )
    return {
        "component_group_ids": groups,
        "availability_group_order": order,
        "component_to_availability_index": indices,
        "heteroscedastic_component_mask": heteroscedastic,
        "packed_dimension": (
            len(declaration.components)
            + len(order)
            + sum(heteroscedastic)
        ),
    }


def pack_global_feedback(
    prediction: Mapping[str, Any],
    *,
    component_to_availability_index: Any,
    heteroscedastic_component_mask: Any,
) -> Any:
    """Canonical deployable packing of predicted values and missingness."""

    mean = prediction.get("mean", prediction["value"])
    logits = prediction.get("availability_logits")
    if logits is None or mean.ndim != 2 or logits.ndim != 2:
        raise ValueError("global feedback lacks value/availability matrices")
    indices = component_to_availability_index.to(device=mean.device)
    if int(mean.shape[1]) != int(indices.numel()):
        raise ValueError("global feedback component layout differs")
    if int(logits.shape[1]) != int(indices.max().item()) + 1:
        raise ValueError("global feedback availability layout differs")
    probabilities = torch.sigmoid(logits)
    gated = mean * probabilities.index_select(1, indices)
    pieces = [gated, probabilities]
    heteroscedastic = heteroscedastic_component_mask.to(device=mean.device)
    if bool(heteroscedastic.any()):
        log_variance = prediction.get("log_variance")
        if log_variance is None or tuple(log_variance.shape) != tuple(mean.shape):
            raise ValueError("HET feedback lacks its registered log variances")
        pieces.append(log_variance.clamp(-8.0, 5.0)[:, heteroscedastic])
    return torch.cat(pieces, dim=-1)


class ResidualStructureTokenAdapter(
    torch.nn.Module if torch is not None else object
):
    """Four structure tokens cross-attend into particles without joining them."""

    def __init__(
        self,
        target_dimension: int,
        *,
        particle_dimension: int = 128,
        heteroscedastic: bool = False,
        component_to_availability_index: Sequence[int] | None = None,
        heteroscedastic_component_mask: Sequence[bool] | None = None,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for HOSD feedback")
        super().__init__()
        component_indices = tuple(
            0 for _ in range(target_dimension)
        ) if component_to_availability_index is None else tuple(
            int(value) for value in component_to_availability_index
        )
        heteroscedastic_mask = tuple(
            bool(heteroscedastic) for _ in range(target_dimension)
        ) if heteroscedastic_component_mask is None else tuple(
            bool(value) for value in heteroscedastic_component_mask
        )
        if (
            len(component_indices) != int(target_dimension)
            or len(heteroscedastic_mask) != int(target_dimension)
            or min(component_indices, default=-1) < 0
        ):
            raise ValueError("global feedback packing layout differs")
        availability_groups = max(component_indices, default=-1) + 1
        source_dimension = (
            int(target_dimension)
            + availability_groups
            + sum(heteroscedastic_mask)
        )
        self.heteroscedastic = bool(heteroscedastic)
        self.register_buffer(
            "component_to_availability_index",
            torch.as_tensor(component_indices, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "feedback_heteroscedastic_component_mask",
            torch.as_tensor(heteroscedastic_mask, dtype=torch.bool),
            persistent=True,
        )
        self.particle_norm = torch.nn.RMSNorm(particle_dimension)
        self.query_projection = (
            torch.nn.Identity()
            if particle_dimension == 128
            else torch.nn.Linear(particle_dimension, 128, bias=False)
        )
        self.structure_projection = torch.nn.Linear(source_dimension, 4 * 128)
        self.family_embedding = torch.nn.Parameter(torch.zeros(1, 1, 128))
        self.parameterization_embedding = torch.nn.Parameter(torch.zeros(1, 1, 128))
        self.source_embedding = torch.nn.Parameter(torch.zeros(1, 1, 128))
        self.slot_embedding = torch.nn.Parameter(torch.empty(1, 4, 128))
        torch.nn.init.normal_(self.slot_embedding, std=0.02)
        self.cross_attention = torch.nn.MultiheadAttention(
            128, 8, dropout=0.0, batch_first=True
        )
        self.output_projection = (
            torch.nn.Identity()
            if particle_dimension == 128
            else torch.nn.Linear(128, particle_dimension, bias=True)
        )
        if isinstance(self.output_projection, torch.nn.Linear):
            torch.nn.init.zeros_(self.output_projection.bias)
        self.raw_gamma = torch.nn.Parameter(torch.zeros(()))

    @property
    def gamma(self) -> Any:
        return 2.0 * torch.tanh(self.raw_gamma)

    def structure_tokens(self, prediction: Mapping[str, Any]) -> Any:
        source = pack_global_feedback(
            prediction,
            component_to_availability_index=(
                self.component_to_availability_index
            ),
            heteroscedastic_component_mask=(
                self.feedback_heteroscedastic_component_mask
            ),
        )
        tokens = self.structure_projection(source).reshape(-1, 4, 128)
        return (
            tokens
            + self.family_embedding
            + self.parameterization_embedding
            + self.source_embedding
            + self.slot_embedding
        )

    def forward(
        self, particles: Any, particle_mask: Any, prediction: Mapping[str, Any]
    ) -> Any:
        tokens = self.structure_tokens(prediction)
        query = self.query_projection(self.particle_norm(particles))
        update, _ = self.cross_attention(
            query, tokens, tokens, need_weights=False
        )
        update = self.output_projection(update)
        update = update.masked_fill(~particle_mask.bool().unsqueeze(-1), 0)
        return particles + self.gamma * update


class DirectFourTokenHead(torch.nn.Module if torch is not None else object):
    """Unrestricted four-query trunk whose output is four latent tokens.

    This is deliberately not a structure predictor.  It retains the registered
    global head's four-query masked cross-attention, but emits one learned
    128-dimensional token per query instead of projecting to a named target.
    """

    def __init__(self, *, input_dimension: int = 128) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for HOSD feedback")
        super().__init__()
        self.input_projection = (
            torch.nn.Identity()
            if input_dimension == 128
            else torch.nn.Linear(input_dimension, 128)
        )
        self.input_norm = torch.nn.RMSNorm(128)
        self.queries = torch.nn.Parameter(torch.empty(4, 128))
        torch.nn.init.normal_(self.queries, std=0.02)
        self.cross_attention = torch.nn.MultiheadAttention(
            128, 8, dropout=0.0, batch_first=True
        )
        self.direct_output = torch.nn.Linear(128, 128)

    def forward(self, particle_states: Any, particle_mask: Any) -> Any:
        if particle_states.ndim != 3 or particle_mask.ndim != 2:
            raise ValueError("unrestricted token head expects [B,N,D] and [B,N]")
        if tuple(particle_states.shape[:2]) != tuple(particle_mask.shape):
            raise ValueError("unrestricted token state and mask shapes differ")
        if bool((particle_mask.bool().sum(dim=1) == 0).any()):
            raise ValueError("unrestricted token head cannot attend to an empty event")
        state = self.input_norm(self.input_projection(particle_states))
        query = self.queries.unsqueeze(0).expand(state.shape[0], -1, -1)
        tokens, _ = self.cross_attention(
            query,
            state,
            state,
            key_padding_mask=~particle_mask.bool(),
            need_weights=False,
        )
        return self.direct_output(tokens)


class DirectStructureTokenAdapter(
    torch.nn.Module if torch is not None else object
):
    """Consume unrestricted tokens through the semantic adapter's same readout."""

    def __init__(self, *, particle_dimension: int = 128) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for HOSD feedback")
        super().__init__()
        self.particle_norm = torch.nn.RMSNorm(particle_dimension)
        self.query_projection = (
            torch.nn.Identity()
            if particle_dimension == 128
            else torch.nn.Linear(particle_dimension, 128, bias=False)
        )
        self.family_embedding = torch.nn.Parameter(torch.zeros(1, 1, 128))
        self.parameterization_embedding = torch.nn.Parameter(torch.zeros(1, 1, 128))
        self.source_embedding = torch.nn.Parameter(torch.zeros(1, 1, 128))
        self.slot_embedding = torch.nn.Parameter(torch.empty(1, 4, 128))
        torch.nn.init.normal_(self.slot_embedding, std=0.02)
        self.cross_attention = torch.nn.MultiheadAttention(
            128, 8, dropout=0.0, batch_first=True
        )
        self.output_projection = (
            torch.nn.Identity()
            if particle_dimension == 128
            else torch.nn.Linear(128, particle_dimension, bias=True)
        )
        if isinstance(self.output_projection, torch.nn.Linear):
            torch.nn.init.zeros_(self.output_projection.bias)
        self.raw_gamma = torch.nn.Parameter(torch.zeros(()))

    @property
    def gamma(self) -> Any:
        return 2.0 * torch.tanh(self.raw_gamma)

    def forward(self, particles: Any, particle_mask: Any, tokens: Any) -> Any:
        if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (4, 128):
            raise ValueError("unrestricted feedback must provide [B,4,128] tokens")
        tokens = (
            tokens
            + self.family_embedding
            + self.parameterization_embedding
            + self.source_embedding
            + self.slot_embedding
        )
        query = self.query_projection(self.particle_norm(particles))
        update, _ = self.cross_attention(query, tokens, tokens, need_weights=False)
        update = self.output_projection(update)
        update = update.masked_fill(~particle_mask.bool().unsqueeze(-1), 0)
        return particles + self.gamma * update


class DirectTokenFeaturewiseConditioning(
    torch.nn.Module if torch is not None else object
):
    """Map four unrestricted latent tokens to the registered bounded FiLM path."""

    def __init__(self, *, particle_dimension: int = 128) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for HOSD feedback")
        super().__init__()
        self.projection = torch.nn.Linear(4 * 128, 2 * particle_dimension)
        torch.nn.init.zeros_(self.projection.weight)
        torch.nn.init.zeros_(self.projection.bias)

    def parameters_for(self, tokens: Any) -> tuple[Any, Any]:
        if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (4, 128):
            raise ValueError("unrestricted FiLM must receive [B,4,128] tokens")
        scale_raw, shift_raw = self.projection(tokens.flatten(1)).chunk(2, dim=-1)
        return 1.0 + 0.1 * torch.tanh(scale_raw), 0.1 * torch.tanh(shift_raw)

    def forward(self, particles: Any, particle_mask: Any, tokens: Any) -> Any:
        scale, shift = self.parameters_for(tokens)
        result = particles * scale.unsqueeze(1) + shift.unsqueeze(1)
        return result.masked_fill(~particle_mask.bool().unsqueeze(-1), 0)


class TrainableParameterPadding(torch.nn.Module if torch is not None else object):
    """Exact, inert trainable-parameter ledger for matched-capacity controls."""

    def __init__(self, count: int) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for HOSD feedback")
        super().__init__()
        if int(count) < 0:
            raise ValueError("matched-capacity padding cannot be negative")
        self.count = int(count)
        self.padding = torch.nn.Parameter(torch.zeros(self.count))

    def inert_scalar(self) -> Any:
        # Retains the parameter in the autograd graph without changing logits.
        return self.padding.sum() * 0.0


class BoundedFeaturewiseConditioning(
    torch.nn.Module if torch is not None else object
):
    def __init__(
        self,
        target_dimension: int,
        *,
        particle_dimension: int = 128,
        heteroscedastic: bool = False,
        component_to_availability_index: Sequence[int] | None = None,
        heteroscedastic_component_mask: Sequence[bool] | None = None,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for HOSD feedback")
        super().__init__()
        component_indices = tuple(
            0 for _ in range(target_dimension)
        ) if component_to_availability_index is None else tuple(
            int(value) for value in component_to_availability_index
        )
        heteroscedastic_mask = tuple(
            bool(heteroscedastic) for _ in range(target_dimension)
        ) if heteroscedastic_component_mask is None else tuple(
            bool(value) for value in heteroscedastic_component_mask
        )
        if (
            len(component_indices) != int(target_dimension)
            or len(heteroscedastic_mask) != int(target_dimension)
            or min(component_indices, default=-1) < 0
        ):
            raise ValueError("global feedback packing layout differs")
        source_dimension = (
            int(target_dimension)
            + max(component_indices, default=-1)
            + 1
            + sum(heteroscedastic_mask)
        )
        self.heteroscedastic = bool(heteroscedastic)
        self.register_buffer(
            "component_to_availability_index",
            torch.as_tensor(component_indices, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "feedback_heteroscedastic_component_mask",
            torch.as_tensor(heteroscedastic_mask, dtype=torch.bool),
            persistent=True,
        )
        self.projection = torch.nn.Linear(source_dimension, 2 * particle_dimension)
        torch.nn.init.zeros_(self.projection.weight)
        torch.nn.init.zeros_(self.projection.bias)

    def parameters_for(self, prediction: Mapping[str, Any]) -> tuple[Any, Any]:
        source = pack_global_feedback(
            prediction,
            component_to_availability_index=(
                self.component_to_availability_index
            ),
            heteroscedastic_component_mask=(
                self.feedback_heteroscedastic_component_mask
            ),
        )
        scale_raw, shift_raw = self.projection(source).chunk(2, dim=-1)
        return 1.0 + 0.1 * torch.tanh(scale_raw), 0.1 * torch.tanh(shift_raw)

    def forward(
        self, particles: Any, particle_mask: Any, prediction: Mapping[str, Any]
    ) -> Any:
        scale, shift = self.parameters_for(prediction)
        result = particles * scale.unsqueeze(1) + shift.unsqueeze(1)
        return result.masked_fill(~particle_mask.bool().unsqueeze(-1), 0)


class PredictedPairAttentionBias(
    torch.nn.Module if torch is not None else object
):
    def __init__(
        self,
        *,
        input_dimension: int,
        pair_dimension: int,
        attention_heads: int = 8,
        symmetric: bool = True,
        predictor_enabled: bool = True,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for HOSD feedback")
        super().__init__()
        self.symmetric = bool(symmetric)
        self.predictor = (
            PairTargetHead(input_dimension, pair_dimension, symmetric=symmetric)
            if predictor_enabled
            else None
        )
        self.bias_network = torch.nn.Sequential(
            torch.nn.Linear(pair_dimension, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, attention_heads),
        )
        self.raw_alpha = torch.nn.Parameter(torch.zeros(attention_heads))
        self.total_updates = 1
        self.update_ordinal = 0

    @property
    def alpha(self) -> Any:
        if self.update_ordinal <= gate_warmup_updates(self.total_updates):
            return self.raw_alpha * 0.0
        return 2.0 * torch.tanh(self.raw_alpha)

    def set_update(self, update_ordinal: int, total_updates: int) -> None:
        if not 0 <= int(update_ordinal) <= int(total_updates):
            raise ValueError("feedback update ordinal lies outside schedule")
        self.update_ordinal = int(update_ordinal)
        self.total_updates = int(total_updates)

    def forward(
        self,
        states: Any,
        mask: Any,
        *,
        direct_features: Any | None = None,
        direct_pair_mask: Any | None = None,
        detach_consumer: bool = False,
    ) -> tuple[Any, dict[str, Any]]:
        if self.predictor is None:
            if direct_features is None:
                raise ValueError("exact HLT pair feedback requires direct features")
            if direct_pair_mask is None:
                raise ValueError("exact HLT pair feedback requires applicability mask")
            predicted = direct_features
            pair_mask = direct_pair_mask.bool()
            if tuple(pair_mask.shape) != tuple(direct_features.shape[:3]):
                raise ValueError("exact HLT pair applicability shape differs")
        else:
            predicted, pair_mask = self.predictor(states, mask)
        consumed = predicted if direct_features is None else direct_features
        if detach_consumer:
            consumed = consumed.detach()
        raw = self.bias_network(consumed).permute(0, 3, 1, 2)
        bias = torch.tanh(raw) * self.alpha.view(1, -1, 1, 1)
        if self.symmetric:
            bias = 0.5 * (bias + bias.transpose(2, 3))
        valid = pair_mask.unsqueeze(1)
        bias = bias.masked_fill(~valid, 0)
        return bias, {"value": predicted, "pair_mask": pair_mask}


def _target_dimension(target_id: str) -> int:
    rows = {row.target_id: row for row in target_declarations()}
    if target_id not in rows:
        raise ValueError(f"unknown feedback target {target_id}")
    return len(rows[target_id].components)


def build_feedback_model(
    row: Mapping[str, Any],
    *,
    weaver_module: Any | None = None,
    particle_dimension: int = 128,
) -> "FeedbackHBaseClassifier":
    if str(row.get("target_id")) not in {
        declaration.target_id for declaration in target_declarations()
    }:
        raise ValueError("feedback row target is not registered")
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(row["encoder_component_seed"]))
        classifier = HBaseParticleTransformer(weaver_module=weaver_module)
        torch.manual_seed(int(row["feedback_component_seed"]))
        return FeedbackHBaseClassifier(
            classifier,
            target_id=str(row["target_id"]),
            interface=str(row["interface"]),
            gradient_path=str(row["gradient_path"]),
            parameterization=str(row["parameterization"]),
            particle_dimension=particle_dimension,
            control=row.get("control"),
            allow_oracle=row.get("control") in {"ORACLE_SUB", "ORACLE_TRAINED"},
        )


def initialize_feedback_from_auxiliary_checkpoint(
    model: "FeedbackHBaseClassifier",
    row: Mapping[str, Any],
    *,
    checkpoint_path: str | Path,
    completion: Mapping[str, Any],
    result: Mapping[str, Any],
    stage_d_plan_sha256: str,
    campaign_spec_sha256: str,
    source: Mapping[str, Any],
    checkpoint_contract: str = AUXILIARY_CHECKPOINT_CONTRACT,
    completion_contract: str = AUXILIARY_COMPLETION_CONTRACT,
    prediction_contract: str = AUXILIARY_PREDICTION_CONTRACT,
    plan_hash_field: str = "stage_d_plan_sha256",
) -> dict[str, str]:
    """Continue Stage E from the exact locked single-target A_t state."""

    if torch is None:
        raise RuntimeError("PyTorch is required for feedback initialization")
    validate_content_hash(
        completion, expected_contract=completion_contract
    )
    validate_content_hash(result, expected_contract=prediction_contract)
    selected_row_id = str(row["selected_auxiliary_row_id"])
    if (
        completion.get("row_id") != selected_row_id
        or result.get("row_id") != selected_row_id
        or result.get("target_id") != row["target_id"]
        or result.get("parameterization")
        != row["selected_auxiliary_parameterization"]
        or float(result.get("auxiliary_weight"))
        != float(row["selected_auxiliary_weight"])
        or result.get("content_hash")
        != row["selected_auxiliary_result_sha256"]
        or completion.get(plan_hash_field) != stage_d_plan_sha256
        or result.get(plan_hash_field) != stage_d_plan_sha256
        or completion.get("campaign_spec_sha256") != campaign_spec_sha256
        or result.get("campaign_spec_sha256") != campaign_spec_sha256
        or completion.get("source") != dict(source)
        or result.get("source") != dict(source)
    ):
        raise ValueError("selected auxiliary initialization lineage differs")
    checkpoint_path = Path(checkpoint_path)
    checkpoint_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if (
        checkpoint_sha != completion.get("checkpoint_sha256")
        or checkpoint_sha != result.get("checkpoint_sha256")
    ):
        raise ValueError("selected auxiliary checkpoint hash differs")
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if (
        checkpoint.get("contract") != checkpoint_contract
        or checkpoint.get("row_id") != selected_row_id
        or checkpoint.get(plan_hash_field) != stage_d_plan_sha256
        or checkpoint.get("campaign_spec_sha256") != campaign_spec_sha256
        or checkpoint.get("source") != dict(source)
    ):
        raise ValueError("selected auxiliary checkpoint metadata differs")
    destination = model.state_dict()
    copied_classifier = copied_predictor = 0
    for key, value in checkpoint["model_state_dict"].items():
        destination_key = None
        if key.startswith("classifier."):
            destination_key = key
            copied_classifier += 1
        elif key.startswith("target_head."):
            suffix = key.removeprefix("target_head.")
            if model.global_predictor is not None and model.control != "UNRESTRICTED":
                destination_key = f"global_predictor.{suffix}"
            elif (
                isinstance(model.consumer, PredictedPairAttentionBias)
                and model.consumer.predictor is not None
            ):
                destination_key = f"consumer.predictor.{suffix}"
            if destination_key is not None:
                copied_predictor += 1
        if destination_key is None:
            continue
        if destination_key not in destination:
            raise ValueError(
                f"auxiliary/feedback state shape differs: {destination_key}"
            )
        destination_value = destination[destination_key]
        if tuple(destination_value.shape) == tuple(value.shape):
            destination[destination_key] = value
        elif (
            model.control == "MEAN_ONLY"
            and row.get("selected_auxiliary_parameterization") == "HET"
            and destination_key
            in {
                "global_predictor.output.weight",
                "global_predictor.output.bias",
            }
            and value.ndim == destination_value.ndim
            and tuple(value.shape[1:]) == tuple(destination_value.shape[1:])
            and int(value.shape[0]) > int(destination_value.shape[0])
        ):
            # GlobalTargetHead packs means first and HET log variances last.
            # MEAN_ONLY inherits the exact mean projection and deliberately
            # drops the variance rows; its capacity ledger replaces them with
            # inert trainable padding.
            destination[destination_key] = value[
                : int(destination_value.shape[0])
            ]
        else:
            raise ValueError(
                f"auxiliary/feedback state shape differs: {destination_key}"
            )
    if copied_classifier == 0:
        raise ValueError("auxiliary checkpoint contains no classifier state")
    predictor_expected = (
        model.control not in {"UNRESTRICTED", "EXACT_HLT"}
    )
    if predictor_expected and copied_predictor == 0:
        raise ValueError("auxiliary checkpoint contains no reusable target head")
    model.load_state_dict(destination, strict=True)
    return {
        "selected_auxiliary_result": str(result["content_hash"]),
        "selected_auxiliary_completion": str(completion["content_hash"]),
        "selected_auxiliary_checkpoint": checkpoint_sha,
    }


class FeedbackHBaseClassifier(torch.nn.Module if torch is not None else object):
    """H_BASE with one explicitly registered HLT-predicted feedback interface."""

    def __init__(
        self,
        classifier: HBaseParticleTransformer,
        *,
        target_id: str,
        interface: str,
        gradient_path: str = "END_TO_END",
        parameterization: str = "ABS",
        particle_dimension: int = 128,
        attention_heads: int = 8,
        control: str | None = None,
        allow_oracle: bool = False,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for HOSD feedback")
        super().__init__()
        if interface not in {"FB_TOKEN", "FB_FILM", "FB_PAIR"}:
            raise ValueError("unknown feedback interface")
        if gradient_path not in {"END_TO_END", "DETACHED", "AUX_ONLY"}:
            raise ValueError("unknown feedback gradient path")
        if interface == "FB_PAIR" and target_id not in PAIR_TARGETS:
            raise ValueError("pair feedback is restricted to registered HLT targets")
        if interface != "FB_PAIR" and target_id in PAIR_TARGETS:
            raise ValueError("global feedback cannot consume a pair target")
        if control in {"ORACLE_SUB", "ORACLE_TRAINED"} and not allow_oracle:
            raise ValueError("oracle feedback requires an explicit diagnostic graph")
        self.classifier = classifier
        self.target_id = str(target_id)
        self.interface = str(interface)
        self.gradient_path = str(gradient_path)
        self.parameterization = str(parameterization)
        self.control = control
        self.allow_oracle = bool(allow_oracle)
        self.capacity_padding = None
        self.capacity_ledger = None
        self.global_availability_group_count = 0
        declarations = {row.target_id: row for row in target_declarations()}
        declaration = declarations[target_id]
        dimension = len(declaration.components)
        heteroscedastic = parameterization == "HET" and control != "MEAN_ONLY"
        if interface == "FB_PAIR":
            self.global_predictor = None
            self.exact_pair_builder = (
                ExactHLTPairFeatureBuilder(target_id, dimension)
                if control == "EXACT_HLT"
                else None
            )
            self.consumer = PredictedPairAttentionBias(
                input_dimension=particle_dimension,
                pair_dimension=dimension,
                attention_heads=attention_heads,
                symmetric=declaration.symmetry == "symmetric",
                predictor_enabled=control != "EXACT_HLT",
            )
            if control == "ZERO_GATE":
                self.consumer.raw_alpha.requires_grad_(False)
        else:
            self.exact_pair_builder = None
            layout = global_feedback_layout(target_id, parameterization)
            self.global_availability_group_count = len(
                layout["availability_group_order"]
            )
            consumer_heteroscedastic_mask = (
                layout["heteroscedastic_component_mask"]
                if heteroscedastic
                else tuple(False for _ in declaration.components)
            )
            if control == "UNRESTRICTED":
                self.global_predictor = DirectFourTokenHead(
                    input_dimension=particle_dimension
                )
                self.consumer = (
                    DirectStructureTokenAdapter(
                        particle_dimension=particle_dimension
                    )
                    if interface == "FB_TOKEN"
                    else DirectTokenFeaturewiseConditioning(
                        particle_dimension=particle_dimension
                    )
                )
                # The reference branch is instantiated only to derive its exact
                # active tensor count.  fork_rng prevents this accounting step
                # from changing the unrestricted branch initialization stream.
                devices = (
                    list(range(torch.cuda.device_count()))
                    if torch.cuda.is_available()
                    else []
                )
                with torch.random.fork_rng(devices=devices):
                    reference_predictor = GlobalTargetHead(
                        dimension,
                        input_dimension=particle_dimension,
                        availability_groups=len(
                            layout["availability_group_order"]
                        ),
                        heteroscedastic=parameterization == "HET",
                        heteroscedastic_components=layout[
                            "heteroscedastic_component_mask"
                        ],
                    )
                    reference_consumer_type = (
                        ResidualStructureTokenAdapter
                        if interface == "FB_TOKEN"
                        else BoundedFeaturewiseConditioning
                    )
                    reference_consumer = reference_consumer_type(
                        dimension,
                        particle_dimension=particle_dimension,
                        heteroscedastic=heteroscedastic,
                        component_to_availability_index=layout[
                            "component_to_availability_index"
                        ],
                        heteroscedastic_component_mask=(
                            consumer_heteroscedastic_mask
                        ),
                    )
                reference_count = sum(
                    value.numel()
                    for module in (reference_predictor, reference_consumer)
                    for value in module.parameters()
                    if value.requires_grad
                )
                unrestricted_count = sum(
                    value.numel()
                    for module in (self.global_predictor, self.consumer)
                    for value in module.parameters()
                    if value.requires_grad
                )
                difference = reference_count - unrestricted_count
                self.capacity_padding = TrainableParameterPadding(difference)
                self.capacity_ledger = with_content_hash({
                    "contract": "hosd_unrestricted_feedback_capacity_ledger_v1",
                    "schema_version": 1,
                    "target_id": self.target_id,
                    "interface": self.interface,
                    "reference_trainable_parameters": reference_count,
                    "unrestricted_pre_padding_trainable_parameters": (
                        unrestricted_count
                    ),
                    "inert_trainable_padding_parameters": difference,
                    "matched_trainable_parameters": reference_count,
                })
            else:
                self.global_predictor = GlobalTargetHead(
                    dimension,
                    input_dimension=particle_dimension,
                    availability_groups=len(layout["availability_group_order"]),
                    heteroscedastic=heteroscedastic,
                    heteroscedastic_components=layout[
                        "heteroscedastic_component_mask"
                    ],
                )
                consumer_type = (
                    ResidualStructureTokenAdapter
                    if interface == "FB_TOKEN"
                    else BoundedFeaturewiseConditioning
                )
                self.consumer = consumer_type(
                    dimension,
                    particle_dimension=particle_dimension,
                    heteroscedastic=heteroscedastic,
                    component_to_availability_index=layout[
                        "component_to_availability_index"
                    ],
                    heteroscedastic_component_mask=(
                        consumer_heteroscedastic_mask
                    ),
                )
                if control == "MEAN_ONLY":
                    if parameterization != "HET":
                        raise ValueError(
                            "MEAN_ONLY is the parameter-matched HET mean control"
                        )
                    devices = (
                        list(range(torch.cuda.device_count()))
                        if torch.cuda.is_available()
                        else []
                    )
                    with torch.random.fork_rng(devices=devices):
                        reference_predictor = GlobalTargetHead(
                            dimension,
                            input_dimension=particle_dimension,
                            availability_groups=len(
                                layout["availability_group_order"]
                            ),
                            heteroscedastic=True,
                            heteroscedastic_components=layout[
                                "heteroscedastic_component_mask"
                            ],
                        )
                        reference_consumer = consumer_type(
                            dimension,
                            particle_dimension=particle_dimension,
                            heteroscedastic=True,
                            component_to_availability_index=layout[
                                "component_to_availability_index"
                            ],
                            heteroscedastic_component_mask=layout[
                                "heteroscedastic_component_mask"
                            ],
                        )
                    reference_count = sum(
                        value.numel()
                        for module in (reference_predictor, reference_consumer)
                        for value in module.parameters()
                        if value.requires_grad
                    )
                    mean_only_count = sum(
                        value.numel()
                        for module in (self.global_predictor, self.consumer)
                        for value in module.parameters()
                        if value.requires_grad
                    )
                    difference = reference_count - mean_only_count
                    self.capacity_padding = TrainableParameterPadding(difference)
                    self.capacity_ledger = with_content_hash(
                        {
                            "contract": "hosd_mean_only_capacity_ledger_v2",
                            "schema_version": 2,
                            "target_id": self.target_id,
                            "interface": self.interface,
                            "reference_control": "HET",
                            "reference_trainable_parameters": reference_count,
                            "mean_only_pre_padding_trainable_parameters": (
                                mean_only_count
                            ),
                            "inert_trainable_padding_parameters": difference,
                            "matched_trainable_parameters": reference_count,
                        }
                    )
            if control in {"ZERO", "ZERO_GATE"} and hasattr(
                self.consumer, "raw_gamma"
            ):
                self.consumer.raw_gamma.requires_grad_(False)

    def set_update(self, update_ordinal: int, total_updates: int) -> None:
        if isinstance(self.consumer, PredictedPairAttentionBias):
            self.consumer.set_update(update_ordinal, total_updates)

    def configure_exact_hlt_runtime(
        self,
        *,
        target_normalizer: Mapping[str, Any],
        relation_normalizer: Mapping[str, Any],
    ) -> None:
        if self.control != "EXACT_HLT" or self.exact_pair_builder is None:
            raise ValueError("only an exact HLT reference accepts runtime normalization")
        self.exact_pair_builder.configure(
            target_normalizer=target_normalizer,
            relation_normalizer=relation_normalizer,
        )

    def shared_parameters(self) -> tuple[Any, ...]:
        return tuple(self.classifier.parameters())

    def head_parameters(self) -> tuple[Any, ...]:
        classifier_ids = {id(value) for value in self.classifier.parameters()}
        return tuple(
            value for value in self.parameters() if id(value) not in classifier_ids
        )

    def forward_with_feedback(
        self,
        points: Any,
        features: Any,
        lorentz_vectors: Any,
        mask: Any,
        *,
        direct_pair_features: Any | None = None,
        direct_pair_mask: Any | None = None,
        raw_tokens: Any | None = None,
        region_trees: Sequence[Mapping[str, Any]] | None = None,
        oracle_feedback: Mapping[str, Any] | None = None,
        predicted_feedback_override: Mapping[str, Any] | None = None,
    ) -> tuple[Any, Mapping[str, Any]]:
        if oracle_feedback is not None and not self.allow_oracle:
            raise ValueError("primary/deployable feedback cannot consume oracle data")
        prediction_box: dict[str, Any] = {}
        detach = self.gradient_path == "DETACHED" or self.control in {
            "DETACHED",
            "SHUFFLED_PREDICTION",
            "SHUFFLED",
        }
        active = self.gradient_path != "AUX_ONLY" and self.control != "AUX_ONLY"

        def predict_global(state: Any, active_mask: Any) -> Mapping[str, Any]:
            if self.control == "UNRESTRICTED":
                tokens = self.global_predictor(state, active_mask)
                predicted = {
                    "tokens": tokens,
                    # Semantic loss is globally fixed to zero for this row.
                    # These shape-compatible inert values keep the shared
                    # training/evaluation worker contract uniform.
                    "value": tokens.new_zeros(tokens.shape[0], _target_dimension(
                        self.target_id
                    )),
                    "availability_logits": tokens.new_zeros(
                        tokens.shape[0], self.global_availability_group_count
                    ),
                }
            else:
                predicted = self.global_predictor(state, active_mask)
                if self.control == "MEAN_ONLY":
                    mean = predicted["value"]
                    predicted = {
                        **predicted,
                        "mean": mean,
                        "log_variance": torch.zeros_like(mean),
                        "heteroscedastic_component_mask": (
                            self.consumer.feedback_heteroscedastic_component_mask
                        ),
                    }
            prediction_box.update(predicted)
            consumed = (
                oracle_feedback
                if oracle_feedback is not None
                else predicted_feedback_override
                if predicted_feedback_override is not None
                else predicted
            )
            if detach:
                consumed = {
                    key: value.detach() if hasattr(value, "detach") else value
                    for key, value in consumed.items()
                }
            return consumed

        if self.interface == "FB_TOKEN":
            def token_transform(state: Any, active_mask: Any) -> Any:
                consumed = predict_global(state, active_mask)
                if not active:
                    return state
                if self.control == "UNRESTRICTED":
                    result = self.consumer(state, active_mask, consumed["tokens"])
                else:
                    result = self.consumer(state, active_mask, consumed)
                if self.capacity_padding is not None:
                    result = result + self.capacity_padding.inert_scalar()
                return result

            result = self.classifier.forward_with_taps(
                points,
                features,
                lorentz_vectors,
                mask,
                capture=("TAP_MID",),
                post_mid_transform=token_transform,
            )
        elif self.interface == "FB_FILM":
            consumed_box: dict[str, Mapping[str, Any]] = {}

            def record(state: Any, active_mask: Any) -> Any:
                consumed_box["value"] = predict_global(state, active_mask)
                return state

            def film_transform(_block: int, state: Any, active_mask: Any) -> Any:
                result = (
                    self.consumer(
                        state,
                        active_mask,
                        consumed_box["value"]["tokens"]
                        if self.control == "UNRESTRICTED"
                        else consumed_box["value"],
                    )
                    if active
                    else state
                )
                if active and self.capacity_padding is not None:
                    result = result + self.capacity_padding.inert_scalar()
                return result

            result = self.classifier.forward_with_taps(
                points,
                features,
                lorentz_vectors,
                mask,
                capture=("TAP_MID",),
                post_mid_transform=record,
                later_block_transform=film_transform,
            )
        else:
            def pair_bias(
                state: Any, active_mask: Any, particle_indices: Any
            ) -> Any:
                exact_mask = direct_pair_mask
                if self.control == "EXACT_HLT" and direct_pair_features is None:
                    direct, exact_mask = self.exact_pair_builder(
                        raw_tokens,
                        mask,
                        lorentz_vectors,
                        region_trees,
                        particle_indices,
                    )
                else:
                    direct = (
                        direct_pair_features
                        if self.control == "EXACT_HLT"
                        else predicted_feedback_override["value"]
                        if predicted_feedback_override is not None
                        else None
                    )
                bias, predicted = self.consumer(
                    state,
                    active_mask,
                    direct_features=direct,
                    direct_pair_mask=exact_mask,
                    detach_consumer=detach,
                )
                prediction_box.update(predicted)
                return bias if active else bias * 0

            result = self.classifier.forward_with_taps(
                points,
                features,
                lorentz_vectors,
                mask,
                capture=("TAP_MID",),
                later_pair_bias=pair_bias,
            )
        if not prediction_box:
            raise RuntimeError("feedback predictor did not execute")
        return result.logits, prediction_box

    def forward(
        self,
        points: Any,
        features: Any,
        lorentz_vectors: Any,
        mask: Any,
        *,
        raw_tokens: Any | None = None,
        region_trees: Sequence[Mapping[str, Any]] | None = None,
    ):
        logits, _ = self.forward_with_feedback(
            points,
            features,
            lorentz_vectors,
            mask,
            raw_tokens=raw_tokens,
            region_trees=region_trees,
        )
        return logits

    def forward_with_aux(
        self,
        points: Any,
        features: Any,
        lorentz_vectors: Any,
        mask: Any,
        *,
        sampled_pair_indices: Mapping[str, Any] | None = None,
        raw_tokens: Any | None = None,
        region_trees: Sequence[Mapping[str, Any]] | None = None,
    ) -> tuple[Any, Mapping[str, Any]]:
        logits, prediction = self.forward_with_feedback(
            points,
            features,
            lorentz_vectors,
            mask,
            raw_tokens=raw_tokens,
            region_trees=region_trees,
        )
        if sampled_pair_indices is not None and self.target_id in PAIR_TARGETS:
            index = (
                sampled_pair_indices["event_indices"].long(),
                sampled_pair_indices["left_indices"].long(),
                sampled_pair_indices["right_indices"].long(),
            )
            prediction = {
                **prediction,
                "value": prediction["value"][index],
            }
        return logits, prediction


def _feedback_row(
    *,
    target_id: str,
    interface: str,
    gradient_path: str,
    parameterization: str,
    auxiliary_weight: float,
    row_kind: str,
    control: str | None = None,
    selected_auxiliary_row_id: str,
    selected_auxiliary_result_sha256: str,
    selected_auxiliary_parameterization: str,
    selected_auxiliary_weight: float,
    configuration_role: str = "scientific_finalist",
    semantic_loss_enabled: bool = True,
) -> dict[str, Any]:
    semantics = {
        "target_id": target_id,
        "interface": interface,
        "gradient_path": gradient_path,
        "parameterization": parameterization,
        "auxiliary_weight": float(auxiliary_weight),
        "row_kind": row_kind,
        "control": control,
        "selected_auxiliary_row_id": str(selected_auxiliary_row_id),
        "selected_auxiliary_result_sha256": require_sha256(
            selected_auxiliary_result_sha256,
            name="selected_auxiliary_result_sha256",
        ),
        "selected_auxiliary_parameterization": str(
            selected_auxiliary_parameterization
        ),
        "selected_auxiliary_weight": float(selected_auxiliary_weight),
        "configuration_role": str(configuration_role),
        "semantic_loss_enabled": bool(semantic_loss_enabled),
    }
    row_id = f"FB_{canonical_sha256(semantics)[:16]}"
    return {
        "row_id": row_id,
        **semantics,
        "pipeline_seed": 101,
        "encoder_component_seed": component_seed(101, "encoder", "H_BASE"),
        "feedback_component_seed": component_seed(101, "feedback", row_id),
        "head_type": "pair" if interface == "FB_PAIR" else "global",
        "resolved": True,
        "selection_eligible": row_kind in {"SCIENTIFIC", "REFERENCE_BASELINE"},
        "deployable": control not in {"ORACLE_SUB", "ORACLE_TRAINED"},
        "fixed_epoch_budget": 40,
        "performance_can_omit_or_cancel": False,
    }


def build_stage_e_plan(
    *,
    single_family_selection: Mapping[str, Any],
    campaign_spec_sha256: str,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(
        single_family_selection,
        expected_contract=SINGLE_FAMILY_SELECTION_CONTRACT,
    )
    selected = dict(single_family_selection["selected_row_by_target"])
    required = {
        "T_OFFLINE_TRACK_32",
        "T_HLT_TRACK_PAIR_13",
        "T_HLT_REGION_PAIR_8",
    }
    if not required.issubset(selected):
        raise ValueError("Stage-E lock lacks mandatory feedback targets")
    order = [
        str(row["target_id"])
        for row in single_family_selection.get("cross_family_order", ())
        if str(row["target_id"]) in GLOBAL_PHYSICAL_TARGETS
        and str(row["target_id"]) != "T_OFFLINE_TRACK_32"
    ]
    if not order:
        # Older Step-6 locks expose only the selected mapping. Its insertion
        # order is canonical selection order and remains deterministic.
        order = [
            target_id
            for target_id in selected
            if target_id in GLOBAL_PHYSICAL_TARGETS
            and target_id != "T_OFFLINE_TRACK_32"
        ]
    promoted = order[:2]
    base = [*MANDATORY_FEEDBACK]
    for target_id in promoted:
        base.extend(((target_id, "FB_TOKEN"), (target_id, "FB_FILM")))
    scientific = []
    for target_id, interface in base:
        selected_definition = single_family_selection[
            "selected_definition_by_target"
        ][target_id]
        selected_row_id = str(selected[target_id])
        selected_result_sha256 = single_family_selection[
            "complete_result_hashes"
        ][selected_row_id]
        parameterization = str(selected_definition["parameterization"])
        weight = float(selected_definition["auxiliary_weight"])
        for path in ("END_TO_END", "DETACHED"):
            scientific.append(
                _feedback_row(
                    target_id=target_id,
                    interface=interface,
                    gradient_path=path,
                    parameterization=parameterization,
                    auxiliary_weight=weight,
                    row_kind="SCIENTIFIC",
                    selected_auxiliary_row_id=selected_row_id,
                    selected_auxiliary_result_sha256=selected_result_sha256,
                    selected_auxiliary_parameterization=parameterization,
                    selected_auxiliary_weight=weight,
                )
            )
    controls = []
    for target_id, interface in MANDATORY_FEEDBACK:
        selected_definition = single_family_selection[
            "selected_definition_by_target"
        ][target_id]
        selected_row_id = str(selected[target_id])
        selected_result_sha256 = single_family_selection[
            "complete_result_hashes"
        ][selected_row_id]
        names = PAIR_CONTROLS if interface == "FB_PAIR" else GLOBAL_CONTROLS
        for control in names:
            reference = control == "EXACT_HLT"
            controls.append(
                _feedback_row(
                    target_id=target_id,
                    interface=interface,
                    gradient_path=(
                        "AUX_ONLY"
                        if control == "AUX_ONLY"
                        else "DETACHED"
                        if control == "DETACHED"
                        else "END_TO_END"
                    ),
                    parameterization=(
                        "HET"
                        if control == "MEAN_ONLY"
                        else str(selected_definition["parameterization"])
                    ),
                    auxiliary_weight=0.0
                    if control
                    in {"DISABLED_LOSS", "NO_SEMANTIC_LOSS", "UNRESTRICTED",
                        "UNRESTRICTED_MLP", "EXACT_HLT"}
                    else float(selected_definition["auxiliary_weight"]),
                    row_kind=("REFERENCE_BASELINE" if reference else "CONTROL"),
                    control=control,
                    selected_auxiliary_row_id=selected_row_id,
                    selected_auxiliary_result_sha256=selected_result_sha256,
                    selected_auxiliary_parameterization=str(
                        selected_definition["parameterization"]
                    ),
                    selected_auxiliary_weight=float(
                        selected_definition["auxiliary_weight"]
                    ),
                    configuration_role=(
                        "reference_baseline" if reference else "mechanism_control"
                    ),
                    semantic_loss_enabled=not reference,
                )
            )
    references = [
        row for row in controls if row["row_kind"] == "REFERENCE_BASELINE"
    ]
    controls = [row for row in controls if row["row_kind"] == "CONTROL"]
    rows = scientific + references + controls
    if len(base) > 8 or len(scientific) > 16 or len(controls) > 30 or len(rows) > 46:
        raise AssertionError("Stage-E matrix exceeds its immutable bound")
    return with_content_hash(
        {
            "contract": STAGE_E_PLAN_CONTRACT,
            "schema_version": 3,
            "source": dict(source),
            "campaign_spec_sha256": require_sha256(
                campaign_spec_sha256, name="campaign_spec_sha256"
            ),
            "single_family_selection_sha256": single_family_selection[
                "content_hash"
            ],
            "promoted_global_targets": promoted,
            "scientific_rows": scientific,
            "reference_rows": references,
            "control_rows": controls,
            "all_rows": rows,
            "row_count": len(rows),
            "hard_maximum": 46,
            "scientific_row_maximum": 16,
            "control_row_maximum": 30,
            "performance_can_cancel_or_omit": False,
        }
    )


def build_feedback_selection(
    *,
    stage_e_plan: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(stage_e_plan, expected_contract=STAGE_E_PLAN_CONTRACT)
    plan_by_id = {
        str(row["row_id"]): row for row in stage_e_plan["all_rows"]
    }
    by_id = {str(row["row_id"]): row for row in results}
    required = set(plan_by_id)
    if len(by_id) != len(results) or set(by_id) != required:
        raise ValueError("feedback selection requires complete Stage-E coverage")
    result_semantic_fields = (
        "target_id",
        "parameterization",
        "auxiliary_weight",
        "row_kind",
        "selection_eligible",
        "interface",
        "gradient_path",
        "control",
        "deployable",
    )
    for row_id, result in by_id.items():
        validate_content_hash(result, expected_contract=FEEDBACK_RESULT_CONTRACT)
        if (
            result.get("source") != dict(source)
            or result.get("stage_e_plan_sha256")
            != stage_e_plan["content_hash"]
            or result.get("campaign_spec_sha256")
            != stage_e_plan["campaign_spec_sha256"]
        ):
            raise ValueError("feedback result lineage differs")
        expected = plan_by_id[row_id]
        if any(
            result.get(key) != expected.get(key)
            for key in result_semantic_fields
        ):
            raise ValueError("feedback result semantics differ")
    scientific = [
        by_id[row["row_id"]] for row in stage_e_plan["scientific_rows"]
    ]
    references = [
        by_id[row["row_id"]]
        for row in stage_e_plan["reference_rows"]
    ]
    deployable_scientific = [
        row
        for row in scientific
        if bool(row.get("selection_eligible")) and bool(row.get("deployable"))
    ]
    winner = select_utility_row(deployable_scientific)
    by_interface = {}
    for interface in ("FB_TOKEN", "FB_FILM", "FB_PAIR"):
        candidates = [
            row for row in deployable_scientific if row["interface"] == interface
        ]
        if candidates:
            by_interface[interface] = select_utility_row(candidates)["row_id"]
    reference_definitions = {}
    definition_fields = (
        "row_id",
        "target_id",
        "interface",
        "gradient_path",
        "parameterization",
        "auxiliary_weight",
        "control",
        "deployable",
        "head_type",
        "row_kind",
        "configuration_role",
        "semantic_loss_enabled",
        "selected_auxiliary_row_id",
        "selected_auxiliary_result_sha256",
        "selected_auxiliary_parameterization",
        "selected_auxiliary_weight",
    )
    for result in references:
        definition = plan_by_id[str(result["row_id"])]
        if (
            result.get("row_kind") != "REFERENCE_BASELINE"
            or result.get("control") != "EXACT_HLT"
            or not bool(result.get("deployable"))
        ):
            raise ValueError("feedback reference baseline semantics differ")
        target = str(definition["target_id"])
        role = {
            "T_HLT_TRACK_PAIR_13": "REFERENCE_EXACT_TRACK",
            "T_HLT_REGION_PAIR_8": "REFERENCE_EXACT_REGION",
        }.get(target)
        if role is None or role in reference_definitions:
            raise ValueError("feedback reference baseline coverage differs")
        reference_definitions[role] = {
            key: definition[key] for key in definition_fields
        }
    if set(reference_definitions) != {
        "REFERENCE_EXACT_TRACK",
        "REFERENCE_EXACT_REGION",
    }:
        raise ValueError("feedback reference graph set is incomplete")
    return with_content_hash(
        {
            "contract": FEEDBACK_SELECTION_CONTRACT,
            "schema_version": 3,
            "source": dict(source),
            "stage_e_plan_sha256": stage_e_plan["content_hash"],
            "result_hashes": {
                key: by_id[key]["content_hash"] for key in sorted(by_id)
            },
            "selected_feedback_row_id": winner["row_id"],
            "selected_feedback_definition": {
                key: plan_by_id[str(winner["row_id"])][key]
                for key in definition_fields
            },
            "selected_by_interface": by_interface,
            "reference_graph_definitions": reference_definitions,
            "reference_graph_result_hashes": {
                role: by_id[definition["row_id"]]["content_hash"]
                for role, definition in sorted(reference_definitions.items())
            },
            "all_rows_completed": True,
            "negative_gain_can_still_win": True,
            "selection_split": "design_select",
            "oracle_or_control_rows_eligible": False,
            "reference_baseline_rows_eligible_for_feedback_family_selection": False,
            "reference_baseline_rows_eligible_for_later_overall_selection": True,
        }
    )


__all__ = [
    "BoundedFeaturewiseConditioning",
    "FeedbackHBaseClassifier",
    "DirectFourTokenHead",
    "DirectStructureTokenAdapter",
    "GLOBAL_CONTROLS",
    "MANDATORY_FEEDBACK",
    "PAIR_CONTROLS",
    "PredictedPairAttentionBias",
    "ResidualStructureTokenAdapter",
    "TrainableParameterPadding",
    "build_feedback_selection",
    "build_feedback_model",
    "build_stage_e_plan",
    "feedback_interface_contract",
    "gate_warmup_updates",
    "global_feedback_layout",
    "initialize_feedback_from_auxiliary_checkpoint",
    "pack_global_feedback",
]
