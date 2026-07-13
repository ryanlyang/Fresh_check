"""Reconstructor models for local per-particle residual fields."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping, Sequence

import torch

from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM


LOCAL_RESIDUAL_RECONSTRUCTOR_CONTRACT = "local_particle_residual_field_reconstructor_v1"

RECONSTRUCTOR_VARIANT_C0 = "C0_cross_attentive_local_transformer"
RECONSTRUCTOR_VARIANT_C1_NO_CONTEXT = "C1_no_global_or_state_context"
RECONSTRUCTOR_VARIANT_C2_NO_GEOMETRY = "C2_no_geometry_bias"
RECONSTRUCTOR_VARIANT_C3_DEEPSETS = "C3_deepsets_global_context"
RECONSTRUCTOR_VARIANT_C4_LOCAL_PARTICLE_ONLY = "C4_local_particle_only_transformer"
RECONSTRUCTOR_VARIANT_C5_UNCERTAINTY = "C5_uncertainty_aware"
RECONSTRUCTOR_VARIANT_C6_CONSISTENCY = "C6_jet_consistency"

LOCAL_RESIDUAL_RECONSTRUCTOR_VARIANTS: tuple[str, ...] = (
    RECONSTRUCTOR_VARIANT_C0,
    RECONSTRUCTOR_VARIANT_C1_NO_CONTEXT,
    RECONSTRUCTOR_VARIANT_C2_NO_GEOMETRY,
    RECONSTRUCTOR_VARIANT_C3_DEEPSETS,
    RECONSTRUCTOR_VARIANT_C4_LOCAL_PARTICLE_ONLY,
    RECONSTRUCTOR_VARIANT_C5_UNCERTAINTY,
    RECONSTRUCTOR_VARIANT_C6_CONSISTENCY,
)

_VARIANT_ALIASES = {
    "C0": RECONSTRUCTOR_VARIANT_C0,
    "best": RECONSTRUCTOR_VARIANT_C0,
    "cross_attentive": RECONSTRUCTOR_VARIANT_C0,
    "geometry": RECONSTRUCTOR_VARIANT_C0,
    "C1": RECONSTRUCTOR_VARIANT_C1_NO_CONTEXT,
    "no_context": RECONSTRUCTOR_VARIANT_C1_NO_CONTEXT,
    "C2": RECONSTRUCTOR_VARIANT_C2_NO_GEOMETRY,
    "no_geometry": RECONSTRUCTOR_VARIANT_C2_NO_GEOMETRY,
    "C3": RECONSTRUCTOR_VARIANT_C3_DEEPSETS,
    "deepsets": RECONSTRUCTOR_VARIANT_C3_DEEPSETS,
    "C4": RECONSTRUCTOR_VARIANT_C4_LOCAL_PARTICLE_ONLY,
    "particle_only": RECONSTRUCTOR_VARIANT_C4_LOCAL_PARTICLE_ONLY,
    "local_particle_only": RECONSTRUCTOR_VARIANT_C4_LOCAL_PARTICLE_ONLY,
    "C5": RECONSTRUCTOR_VARIANT_C5_UNCERTAINTY,
    "uncertainty": RECONSTRUCTOR_VARIANT_C5_UNCERTAINTY,
    "C6": RECONSTRUCTOR_VARIANT_C6_CONSISTENCY,
    "consistency": RECONSTRUCTOR_VARIANT_C6_CONSISTENCY,
}


def normalize_local_residual_reconstructor_variant(value: str) -> str:
    key = str(value).strip()
    if key in LOCAL_RESIDUAL_RECONSTRUCTOR_VARIANTS:
        return key
    if key in _VARIANT_ALIASES:
        return _VARIANT_ALIASES[key]
    raise ValueError(f"unknown local residual-field reconstructor variant {value!r}")


@dataclass(frozen=True)
class LocalResidualFieldReconstructorConfig:
    """Configuration for the local residual-field reconstructor family."""

    variant: str = RECONSTRUCTOR_VARIANT_C0
    particle_dim: int = RAW_TOKEN_DIM
    field_dim: int = 50
    d_model: int = 160
    num_heads: int = 5
    num_layers: int = 4
    context_dim: int | None = None
    context_layers: int = 1
    mlp_ratio: float = 2.0
    dropout: float = 0.05
    attention_dropout: float = 0.05
    max_particles: int = 256
    geometry_bias_clip: float = 8.0
    local_radius: float = 0.12
    hard_local_radius: float = 0.08
    use_zero_init_output: bool = True
    field_groups: Mapping[str, Sequence[int]] | None = None
    field_names: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        variant = normalize_local_residual_reconstructor_variant(self.variant)
        for name in ("particle_dim", "field_dim", "d_model", "num_heads", "num_layers", "max_particles"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if int(self.d_model) % int(self.num_heads) != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if self.context_dim is not None and int(self.context_dim) <= 0:
            raise ValueError("context_dim must be positive when provided")
        if int(self.context_layers) <= 0:
            raise ValueError("context_layers must be positive")
        if float(self.mlp_ratio) <= 0.0:
            raise ValueError("mlp_ratio must be positive")
        for name in ("dropout", "attention_dropout"):
            value = float(getattr(self, name))
            if not (0.0 <= value < 1.0):
                raise ValueError(f"{name} must be in [0, 1)")
        if float(self.local_radius) <= 0.0 or float(self.hard_local_radius) <= 0.0:
            raise ValueError("local radii must be positive")
        groups = self.field_groups
        if groups is None:
            groups = {"all": tuple(range(int(self.field_dim)))}
        normalized_groups: dict[str, tuple[int, ...]] = {}
        seen: set[int] = set()
        for group, indices in groups.items():
            values = tuple(int(index) for index in indices)
            if not values:
                continue
            for index in values:
                if index < 0 or index >= int(self.field_dim):
                    raise ValueError(f"field group {group!r} index {index} is outside field_dim={self.field_dim}")
                if index in seen:
                    raise ValueError(f"field index {index} appears in more than one group")
                seen.add(index)
            normalized_groups[str(group)] = values
        missing = sorted(set(range(int(self.field_dim))) - seen)
        if missing:
            normalized_groups["ungrouped"] = tuple(missing)
        names = tuple(str(name) for name in self.field_names)
        if names and len(names) != int(self.field_dim):
            raise ValueError("field_names length must equal field_dim when provided")
        object.__setattr__(self, "variant", variant)
        object.__setattr__(self, "particle_dim", int(self.particle_dim))
        object.__setattr__(self, "field_dim", int(self.field_dim))
        object.__setattr__(self, "d_model", int(self.d_model))
        object.__setattr__(self, "num_heads", int(self.num_heads))
        object.__setattr__(self, "num_layers", int(self.num_layers))
        object.__setattr__(self, "context_dim", None if self.context_dim is None else int(self.context_dim))
        object.__setattr__(self, "context_layers", int(self.context_layers))
        object.__setattr__(self, "mlp_ratio", float(self.mlp_ratio))
        object.__setattr__(self, "dropout", float(self.dropout))
        object.__setattr__(self, "attention_dropout", float(self.attention_dropout))
        object.__setattr__(self, "max_particles", int(self.max_particles))
        object.__setattr__(self, "geometry_bias_clip", float(self.geometry_bias_clip))
        object.__setattr__(self, "local_radius", float(self.local_radius))
        object.__setattr__(self, "hard_local_radius", float(self.hard_local_radius))
        object.__setattr__(self, "use_zero_init_output", bool(self.use_zero_init_output))
        object.__setattr__(self, "field_groups", normalized_groups)
        object.__setattr__(self, "field_names", names)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["contract"] = LOCAL_RESIDUAL_RECONSTRUCTOR_CONTRACT
        payload["field_groups"] = {
            str(group): [int(index) for index in indices]
            for group, indices in dict(self.field_groups or {}).items()
        }
        payload["field_names"] = list(self.field_names)
        return payload


@dataclass(frozen=True)
class LocalResidualFieldReconstructorOutput:
    """Output from a local residual-field reconstructor."""

    predicted_fields: torch.Tensor
    field_mask: torch.Tensor
    diagnostics: dict[str, Any]
    hidden: torch.Tensor
    log_sigma: torch.Tensor | None = None


def _as_bool_mask(mask: torch.Tensor | None, tokens: torch.Tensor) -> torch.Tensor:
    if mask is None:
        return torch.isfinite(tokens).all(dim=-1) & (tokens[..., 0] > 0.0)
    return mask.to(device=tokens.device, dtype=torch.bool)


def _wrap_phi(value: torch.Tensor) -> torch.Tensor:
    return torch.remainder(value + math.pi, 2.0 * math.pi) - math.pi


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=x.dtype).unsqueeze(-1)
    return (x * weights).sum(dim=1) / torch.clamp(weights.sum(dim=1), min=1.0)


class _FeedForward(torch.nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_model, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, d_model),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _GeometryBiasedSelfAttention(torch.nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        num_heads: int,
        dropout: float,
        geometry_bias_clip: float,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.head_dim = int(d_model) // int(num_heads)
        self.geometry_bias_clip = float(geometry_bias_clip)
        self.q_proj = torch.nn.Linear(d_model, d_model)
        self.k_proj = torch.nn.Linear(d_model, d_model)
        self.v_proj = torch.nn.Linear(d_model, d_model)
        self.out_proj = torch.nn.Linear(d_model, d_model)
        self.dropout = torch.nn.Dropout(float(dropout))
        self.geometry_scale = torch.nn.Parameter(torch.ones(num_heads))
        self.pt_ratio_scale = torch.nn.Parameter(torch.zeros(num_heads))

    def forward(
        self,
        x: torch.Tensor,
        *,
        mask: torch.Tensor,
        geometry_bias: torch.Tensor | None,
        pt_ratio_bias: torch.Tensor | None,
        hard_local_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, particles, _ = x.shape
        q = self.q_proj(x).reshape(batch, particles, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(batch, particles, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(batch, particles, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(float(self.head_dim))
        if geometry_bias is not None:
            scaled = geometry_bias[:, None, :, :] * torch.nn.functional.softplus(self.geometry_scale)[None, :, None, None]
            scores = scores + torch.clamp(scaled, min=-self.geometry_bias_clip, max=0.0)
        if pt_ratio_bias is not None:
            scores = scores + pt_ratio_bias[:, None, :, :] * torch.tanh(self.pt_ratio_scale)[None, :, None, None]
        valid = mask[:, None, None, :] & mask[:, None, :, None]
        if hard_local_mask is not None:
            local = hard_local_mask[:, None, :, :] & valid
            has_local = local.any(dim=-1, keepdim=True)
            valid = torch.where(has_local, local, valid)
        eye = torch.eye(particles, dtype=torch.bool, device=x.device)[None, None, :, :]
        valid = valid | (eye & mask[:, None, :, None])
        scores = scores.masked_fill(~valid, -1.0e4)
        attention = torch.softmax(scores, dim=-1)
        attention = attention * valid.to(dtype=attention.dtype)
        attention = attention / torch.clamp(attention.sum(dim=-1, keepdim=True), min=1.0e-8)
        context = torch.einsum("bhij,bhjd->bhid", self.dropout(attention), v)
        context = context.transpose(1, 2).reshape(batch, particles, self.d_model)
        return self.out_proj(context), attention.detach()


class _LocalResidualBlock(torch.nn.Module):
    def __init__(self, config: LocalResidualFieldReconstructorConfig) -> None:
        super().__init__()
        d_model = int(config.d_model)
        hidden_dim = int(round(float(config.mlp_ratio) * d_model))
        self.attn = _GeometryBiasedSelfAttention(
            d_model=d_model,
            num_heads=int(config.num_heads),
            dropout=float(config.attention_dropout),
            geometry_bias_clip=float(config.geometry_bias_clip),
        )
        self.norm_attn = torch.nn.LayerNorm(d_model)
        self.norm_ffn = torch.nn.LayerNorm(d_model)
        self.ffn = _FeedForward(d_model, hidden_dim, float(config.dropout))
        self.dropout = torch.nn.Dropout(float(config.dropout))

    def forward(
        self,
        x: torch.Tensor,
        *,
        mask: torch.Tensor,
        geometry_bias: torch.Tensor | None,
        pt_ratio_bias: torch.Tensor | None,
        hard_local_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = x
        attn_out, attention = self.attn(
            self.norm_attn(x),
            mask=mask,
            geometry_bias=geometry_bias,
            pt_ratio_bias=pt_ratio_bias,
            hard_local_mask=hard_local_mask,
        )
        x = residual + self.dropout(attn_out)
        x = x + self.ffn(self.norm_ffn(x))
        return x, attention


class _ContextBlock(torch.nn.Module):
    def __init__(self, config: LocalResidualFieldReconstructorConfig) -> None:
        super().__init__()
        d_model = int(config.d_model)
        hidden_dim = int(round(float(config.mlp_ratio) * d_model))
        self.attn = torch.nn.MultiheadAttention(
            d_model,
            int(config.num_heads),
            dropout=float(config.attention_dropout),
            batch_first=True,
        )
        self.norm_attn = torch.nn.LayerNorm(d_model)
        self.norm_ffn = torch.nn.LayerNorm(d_model)
        self.ffn = _FeedForward(d_model, hidden_dim, float(config.dropout))
        self.dropout = torch.nn.Dropout(float(config.dropout))

    def forward(
        self,
        particles: torch.Tensor,
        context: torch.Tensor,
        *,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        residual = particles
        attn_out, _ = self.attn(
            self.norm_attn(particles),
            context,
            context,
            key_padding_mask=~context_mask,
            need_weights=False,
        )
        particles = residual + self.dropout(attn_out)
        particles = particles + self.ffn(self.norm_ffn(particles))
        return particles


def _jet_summary_features(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.to(dtype=tokens.dtype)
    pt = torch.where(mask, tokens[..., 0], torch.zeros_like(tokens[..., 0]))
    eta = torch.where(mask, tokens[..., 1], torch.zeros_like(tokens[..., 1]))
    phi = torch.where(mask, tokens[..., 2], torch.zeros_like(tokens[..., 2]))
    energy = torch.where(mask, tokens[..., 3], torch.zeros_like(tokens[..., 3]))
    count = valid.sum(dim=1)
    sum_pt = pt.sum(dim=1)
    sum_energy = energy.sum(dim=1)
    eta_centroid = (pt * eta).sum(dim=1) / torch.clamp(sum_pt, min=1.0e-6)
    sin_phi = (pt * torch.sin(phi)).sum(dim=1) / torch.clamp(sum_pt, min=1.0e-6)
    cos_phi = (pt * torch.cos(phi)).sum(dim=1) / torch.clamp(sum_pt, min=1.0e-6)
    phi_centroid = torch.atan2(sin_phi, cos_phi)
    d_eta = eta - eta_centroid[:, None]
    d_phi = _wrap_phi(phi - phi_centroid[:, None])
    radial = torch.sqrt(d_eta * d_eta + d_phi * d_phi)
    radial_mean = (pt * radial).sum(dim=1) / torch.clamp(sum_pt, min=1.0e-6)
    pid = tokens[..., 5:10] if tokens.shape[-1] >= 10 else torch.zeros((*tokens.shape[:2], 5), device=tokens.device, dtype=tokens.dtype)
    pid_frac = (pid * pt[..., None]).sum(dim=1) / torch.clamp(sum_pt[:, None], min=1.0e-6)
    return torch.cat(
        [
            torch.log1p(sum_pt)[:, None],
            torch.log1p(sum_energy)[:, None],
            torch.log1p(count)[:, None],
            eta_centroid[:, None],
            phi_centroid[:, None],
            radial_mean[:, None],
            pid_frac,
        ],
        dim=1,
    )


class _FieldGroupHeads(torch.nn.Module):
    def __init__(self, config: LocalResidualFieldReconstructorConfig) -> None:
        super().__init__()
        d_model = int(config.d_model)
        hidden = int(round(float(config.mlp_ratio) * d_model))
        self.field_dim = int(config.field_dim)
        self.groups = {
            str(group): tuple(int(index) for index in indices)
            for group, indices in dict(config.field_groups or {"all": tuple(range(self.field_dim))}).items()
        }
        self.heads = torch.nn.ModuleDict()
        for group, indices in self.groups.items():
            self.heads[group] = torch.nn.Sequential(
                torch.nn.LayerNorm(d_model),
                torch.nn.Linear(d_model, hidden),
                torch.nn.GELU(),
                torch.nn.Dropout(float(config.dropout)),
                torch.nn.Linear(hidden, len(indices)),
            )
            if bool(config.use_zero_init_output):
                last = self.heads[group][-1]
                assert isinstance(last, torch.nn.Linear)
                torch.nn.init.zeros_(last.weight)
                torch.nn.init.zeros_(last.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        output = hidden.new_zeros((*hidden.shape[:2], self.field_dim))
        for group, indices in self.groups.items():
            values = self.heads[group](hidden)
            output[..., list(indices)] = values
        return output


class LocalResidualFieldTransformer(torch.nn.Module):
    """C-tier local residual-field reconstructor.

    The model predicts one residual-field vector per valid HLT particle.  It can
    run as the full geometry/context transformer or as the simpler C-tier
    ablations controlled by ``config.variant``.
    """

    def __init__(self, config: LocalResidualFieldReconstructorConfig | None = None) -> None:
        super().__init__()
        self.config = LocalResidualFieldReconstructorConfig() if config is None else config
        d_model = int(self.config.d_model)
        self.input_norm = torch.nn.LayerNorm(int(self.config.particle_dim))
        self.particle_projection = torch.nn.Linear(int(self.config.particle_dim), d_model)
        self.rank_embedding = torch.nn.Embedding(int(self.config.max_particles), d_model)
        self.jet_summary_projection = torch.nn.Sequential(
            torch.nn.Linear(11, d_model),
            torch.nn.GELU(),
            torch.nn.Linear(d_model, d_model),
        )
        self.context_projection = (
            torch.nn.Linear(int(self.config.context_dim), d_model)
            if self.config.context_dim is not None
            else None
        )
        self.context_blocks = torch.nn.ModuleList(
            [_ContextBlock(self.config) for _ in range(int(self.config.context_layers))]
        )
        self.local_blocks = torch.nn.ModuleList(
            [_LocalResidualBlock(self.config) for _ in range(int(self.config.num_layers))]
        )
        self.deepsets_particle = torch.nn.Sequential(
            torch.nn.LayerNorm(int(self.config.particle_dim)),
            torch.nn.Linear(int(self.config.particle_dim), d_model),
            torch.nn.GELU(),
            torch.nn.Linear(d_model, d_model),
        )
        self.deepsets_update = torch.nn.Sequential(
            torch.nn.LayerNorm(2 * d_model),
            torch.nn.Linear(2 * d_model, int(round(float(self.config.mlp_ratio) * d_model))),
            torch.nn.GELU(),
            torch.nn.Dropout(float(self.config.dropout)),
            torch.nn.Linear(int(round(float(self.config.mlp_ratio) * d_model)), d_model),
        )
        self.output_norm = torch.nn.LayerNorm(d_model)
        self.field_heads = _FieldGroupHeads(self.config)
        self.log_sigma_head = _FieldGroupHeads(self.config)
        if self.uses_uncertainty:
            for parameter in self.log_sigma_head.parameters():
                parameter.requires_grad = True
        self.consistency_head = torch.nn.Sequential(
            torch.nn.LayerNorm(d_model),
            torch.nn.Linear(d_model, d_model),
            torch.nn.GELU(),
            torch.nn.Linear(d_model, 4),
        )

    @property
    def uses_geometry_bias(self) -> bool:
        return self.config.variant not in {
            RECONSTRUCTOR_VARIANT_C2_NO_GEOMETRY,
            RECONSTRUCTOR_VARIANT_C3_DEEPSETS,
        }

    @property
    def uses_context(self) -> bool:
        return self.config.variant not in {
            RECONSTRUCTOR_VARIANT_C1_NO_CONTEXT,
            RECONSTRUCTOR_VARIANT_C3_DEEPSETS,
            RECONSTRUCTOR_VARIANT_C4_LOCAL_PARTICLE_ONLY,
        }

    @property
    def uses_hard_locality(self) -> bool:
        return self.config.variant == RECONSTRUCTOR_VARIANT_C4_LOCAL_PARTICLE_ONLY

    @property
    def uses_uncertainty(self) -> bool:
        return self.config.variant == RECONSTRUCTOR_VARIANT_C5_UNCERTAINTY

    @property
    def uses_consistency_head(self) -> bool:
        return self.config.variant == RECONSTRUCTOR_VARIANT_C6_CONSISTENCY

    def _geometry_terms(self, tokens: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        eta = tokens[..., 1]
        phi = tokens[..., 2]
        pt = torch.clamp(tokens[..., 0], min=1.0e-6)
        d_eta = eta[:, :, None] - eta[:, None, :]
        d_phi = _wrap_phi(phi[:, :, None] - phi[:, None, :])
        d_r = torch.sqrt(d_eta * d_eta + d_phi * d_phi + 1.0e-12)
        geometry_bias = -torch.square(d_r / max(float(self.config.local_radius), 1.0e-6))
        log_pt_ratio = torch.log(pt[:, :, None]) - torch.log(pt[:, None, :])
        hard_local_mask = d_r <= float(self.config.hard_local_radius)
        hard_local_mask = hard_local_mask & mask[:, :, None] & mask[:, None, :]
        return geometry_bias, torch.tanh(log_pt_ratio), hard_local_mask

    def _encode_particles(self, tokens: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        batch, particles, _ = tokens.shape
        rank = torch.arange(particles, device=tokens.device).clamp(max=int(self.config.max_particles) - 1)
        hidden = self.particle_projection(self.input_norm(tokens)) + self.rank_embedding(rank)[None, :, :]
        hidden = hidden * mask.unsqueeze(-1).to(dtype=hidden.dtype)
        geometry_bias, pt_ratio_bias, hard_local_mask = self._geometry_terms(tokens, mask)
        attention_means: list[torch.Tensor] = []
        for block in self.local_blocks:
            hidden, attention = block(
                hidden,
                mask=mask,
                geometry_bias=geometry_bias if self.uses_geometry_bias else None,
                pt_ratio_bias=pt_ratio_bias if self.uses_geometry_bias else None,
                hard_local_mask=hard_local_mask if self.uses_hard_locality else None,
            )
            attention_means.append(attention.mean().detach())
            hidden = hidden * mask.unsqueeze(-1).to(dtype=hidden.dtype)
        diagnostics = {
            "geometry_bias_enabled": bool(self.uses_geometry_bias),
            "context_enabled": bool(self.uses_context),
            "hard_locality_enabled": bool(self.uses_hard_locality),
            "attention_mean": torch.stack(attention_means).mean().detach() if attention_means else hidden.new_tensor(0.0),
            "valid_particles_mean": mask.sum(dim=1).float().mean().detach(),
        }
        return hidden, diagnostics

    def _apply_context(
        self,
        hidden: torch.Tensor,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        *,
        context_tokens: torch.Tensor | None,
        context_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.uses_context:
            return hidden
        context_items = [self.jet_summary_projection(_jet_summary_features(tokens, mask))[:, None, :]]
        context_masks = [torch.ones((tokens.shape[0], 1), dtype=torch.bool, device=tokens.device)]
        if context_tokens is not None:
            if self.context_projection is None:
                raise ValueError("context_tokens were provided but config.context_dim is None")
            projected = self.context_projection(context_tokens.to(device=tokens.device, dtype=tokens.dtype))
            context_items.append(projected)
            if context_mask is None:
                context_masks.append(torch.isfinite(context_tokens).all(dim=-1).to(device=tokens.device))
            else:
                context_masks.append(context_mask.to(device=tokens.device, dtype=torch.bool))
        context = torch.cat(context_items, dim=1)
        combined_mask = torch.cat(context_masks, dim=1)
        for block in self.context_blocks:
            hidden = block(hidden, context, context_mask=combined_mask)
            hidden = hidden * mask.unsqueeze(-1).to(dtype=hidden.dtype)
        return hidden

    def _forward_deepsets(self, tokens: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        local = self.deepsets_particle(tokens)
        pooled = _masked_mean(local, mask)
        hidden = self.deepsets_update(torch.cat([local, pooled[:, None, :].expand_as(local)], dim=-1))
        hidden = hidden * mask.unsqueeze(-1).to(dtype=hidden.dtype)
        return hidden, {
            "geometry_bias_enabled": False,
            "context_enabled": True,
            "hard_locality_enabled": False,
            "attention_mean": hidden.new_tensor(0.0),
            "valid_particles_mean": mask.sum(dim=1).float().mean().detach(),
        }

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        context_tokens: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
    ) -> LocalResidualFieldReconstructorOutput:
        tokens = tokens.to(dtype=torch.float32)
        if tokens.ndim != 3 or int(tokens.shape[-1]) != int(self.config.particle_dim):
            raise ValueError(f"tokens must have shape [B, P, {self.config.particle_dim}], got {tuple(tokens.shape)}")
        particle_mask = _as_bool_mask(mask, tokens)
        if particle_mask.shape != tokens.shape[:2]:
            raise ValueError("mask must match token leading shape")
        if self.config.variant == RECONSTRUCTOR_VARIANT_C3_DEEPSETS:
            hidden, diagnostics = self._forward_deepsets(tokens, particle_mask)
        else:
            hidden, diagnostics = self._encode_particles(tokens, particle_mask)
            hidden = self._apply_context(
                hidden,
                tokens,
                particle_mask,
                context_tokens=context_tokens,
                context_mask=context_mask,
            )
        hidden = self.output_norm(hidden)
        predicted = self.field_heads(hidden)
        predicted = predicted * particle_mask.unsqueeze(-1).to(dtype=predicted.dtype)
        log_sigma = None
        if self.uses_uncertainty:
            log_sigma = self.log_sigma_head(hidden).clamp(min=-8.0, max=8.0)
            log_sigma = log_sigma * particle_mask.unsqueeze(-1).to(dtype=log_sigma.dtype)
        diagnostics = {
            **diagnostics,
            "contract": LOCAL_RESIDUAL_RECONSTRUCTOR_CONTRACT,
            "variant": self.config.variant,
            "field_dim": int(self.config.field_dim),
            "predicted_abs_mean": predicted.detach().abs().mean(),
            "predicted_l2_mean": torch.sqrt(torch.sum(predicted.detach() * predicted.detach(), dim=-1) + 1.0e-12).mean(),
            "log_sigma_present": bool(log_sigma is not None),
        }
        if self.uses_consistency_head:
            pooled = _masked_mean(hidden, particle_mask)
            diagnostics["global_consistency_prediction"] = self.consistency_head(pooled)
        return LocalResidualFieldReconstructorOutput(
            predicted_fields=predicted,
            field_mask=particle_mask,
            diagnostics=diagnostics,
            hidden=hidden,
            log_sigma=log_sigma,
        )


def build_local_residual_field_reconstructor(
    config: LocalResidualFieldReconstructorConfig | Mapping[str, Any] | None = None,
) -> LocalResidualFieldTransformer:
    if config is None:
        cfg = LocalResidualFieldReconstructorConfig()
    elif isinstance(config, LocalResidualFieldReconstructorConfig):
        cfg = config
    else:
        cfg = LocalResidualFieldReconstructorConfig(**dict(config))
    return LocalResidualFieldTransformer(cfg)


__all__ = [
    "LOCAL_RESIDUAL_RECONSTRUCTOR_CONTRACT",
    "RECONSTRUCTOR_VARIANT_C0",
    "RECONSTRUCTOR_VARIANT_C1_NO_CONTEXT",
    "RECONSTRUCTOR_VARIANT_C2_NO_GEOMETRY",
    "RECONSTRUCTOR_VARIANT_C3_DEEPSETS",
    "RECONSTRUCTOR_VARIANT_C4_LOCAL_PARTICLE_ONLY",
    "RECONSTRUCTOR_VARIANT_C5_UNCERTAINTY",
    "RECONSTRUCTOR_VARIANT_C6_CONSISTENCY",
    "LOCAL_RESIDUAL_RECONSTRUCTOR_VARIANTS",
    "LocalResidualFieldReconstructorConfig",
    "LocalResidualFieldReconstructorOutput",
    "LocalResidualFieldTransformer",
    "build_local_residual_field_reconstructor",
    "normalize_local_residual_reconstructor_variant",
]
