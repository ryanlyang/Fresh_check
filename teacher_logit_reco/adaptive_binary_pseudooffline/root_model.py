"""Typed semantic-query transformer for deployable root-state prediction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping

from jetclass_fresh.hlt_baseline import require_torch

from .schemas import ABPH_MAX_PARTICLES, ABPH_PID_CATEGORIES
from .root_transforms import ROOT_SHAPE_FEATURE_NAMES


try:  # Keep cache/input utilities importable without a training environment.
    import torch as _torch
except ImportError:  # pragma: no cover - environment dependent
    _torch = None

if _torch is None:  # pragma: no cover - environment dependent
    class _ModuleBase:
        pass
else:
    _ModuleBase = _torch.nn.Module


ABPH_ROOT_PREDICTOR_CONTRACT = "adaptive_binary_pseudooffline_root_predictor_v1"
ABPH_ROOT_QUERY_NAMES: tuple[str, ...] = (
    "p4",
    "count",
    "composition",
    "shape",
    "charge",
    "uncertainty",
)
ABPH_CHARGE_SUPPORT_MIN = -ABPH_MAX_PARTICLES
ABPH_CHARGE_SUPPORT_MAX = ABPH_MAX_PARTICLES
ABPH_CHARGE_SUPPORT_SIZE = ABPH_CHARGE_SUPPORT_MAX - ABPH_CHARGE_SUPPORT_MIN + 1
ABPH_SHAPE_RAW_DIM = len(ROOT_SHAPE_FEATURE_NAMES) + 1


@dataclass(frozen=True)
class SemanticRootPredictorConfig:
    """Locked primary dimensions for the semantic root readout."""

    input_dim: int = 192
    jet_input_dim: int = 192
    d_model: int = 256
    num_heads: int = 8
    query_blocks: int = 4
    ffn_dim: int = 1024
    dropout: float = 0.10
    attention_dropout: float = 0.10
    max_particles: int = ABPH_MAX_PARTICLES
    max_count: int = ABPH_MAX_PARTICLES
    log_scale_min: float = -6.0
    log_scale_max: float = 4.0

    def __post_init__(self) -> None:
        for name in (
            "input_dim",
            "jet_input_dim",
            "d_model",
            "num_heads",
            "query_blocks",
            "ffn_dim",
            "max_particles",
            "max_count",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if int(self.d_model) % int(self.num_heads) != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if int(self.query_blocks) < 3:
            raise ValueError("semantic root prediction requires at least three query blocks")
        if int(self.max_particles) != ABPH_MAX_PARTICLES or int(self.max_count) != ABPH_MAX_PARTICLES:
            raise ValueError("primary root predictor count/particle support must be exactly 128")
        for name in ("dropout", "attention_dropout"):
            if not 0.0 <= float(getattr(self, name)) < 1.0:
                raise ValueError(f"{name} must lie in [0, 1)")
        if float(self.log_scale_min) >= float(self.log_scale_max):
            raise ValueError("log-scale bounds are reversed")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "contract": ABPH_ROOT_PREDICTOR_CONTRACT,
                "query_names": list(ABPH_ROOT_QUERY_NAMES),
                "charge_support": [ABPH_CHARGE_SUPPORT_MIN, ABPH_CHARGE_SUPPORT_MAX],
                "shape_target_names": list(ROOT_SHAPE_FEATURE_NAMES),
                "shape_raw_dim": ABPH_SHAPE_RAW_DIM,
                "input_semantics": {
                    "particle_embeddings": "final HLT ParT particle embeddings",
                    "jet_embedding": "pooled HLT ParT jet representation",
                    "offline_inputs": False,
                },
            }
        )
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        payload["config_hash"] = hashlib.sha256(encoded).hexdigest()
        return payload


@dataclass(frozen=True)
class SemanticRootPrediction:
    query_tokens: Any
    shared_context: Any
    p4_residual_mean: Any
    p4_residual_log_scale: Any
    count_logits: Any
    delta_count_mean: Any
    delta_count_log_scale: Any
    type_count_logits: Any
    type_pt_logits: Any
    type_energy_logits: Any
    composition_log_scale: Any
    scalar_pt_excess_raw: Any
    charge_logits: Any
    absolute_charge_mean: Any
    absolute_charge_log_scale: Any
    shape_raw: Any
    shape_log_scale: Any
    diagnostics: Mapping[str, Any]

    @property
    def batch_size(self) -> int:
        return int(self.p4_residual_mean.shape[0])

    def count_probabilities(self) -> Any:
        return self.count_logits.softmax(dim=-1)

    def type_count_fractions(self) -> Any:
        return self.type_count_logits.softmax(dim=-1)

    def type_pt_fractions(self) -> Any:
        return self.type_pt_logits.softmax(dim=-1)

    def type_energy_fractions(self) -> Any:
        return self.type_energy_logits.softmax(dim=-1)

    def charge_probabilities(self) -> Any:
        return self.charge_logits.softmax(dim=-1)


class _SemanticQueryBlock(_ModuleBase):
    def __init__(self, config: SemanticRootPredictorConfig) -> None:
        torch = require_torch()
        super().__init__()
        self.num_heads = int(config.num_heads)
        kwargs = {
            "embed_dim": int(config.d_model),
            "num_heads": int(config.num_heads),
            "dropout": float(config.attention_dropout),
            "batch_first": True,
        }
        self.self_attention = torch.nn.MultiheadAttention(**kwargs)
        self.cross_attention = torch.nn.MultiheadAttention(**kwargs)
        self.norm_self = torch.nn.LayerNorm(config.d_model)
        self.norm_cross = torch.nn.LayerNorm(config.d_model)
        self.norm_ffn = torch.nn.LayerNorm(config.d_model)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(config.d_model, config.ffn_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(config.dropout),
            torch.nn.Linear(config.ffn_dim, config.d_model),
            torch.nn.Dropout(config.dropout),
        )

    def forward(
        self,
        queries: Any,
        particles: Any,
        particle_mask: Any,
        attention_bias: Any | None = None,
    ) -> Any:
        normalized = self.norm_self(queries)
        update, _ = self.self_attention(normalized, normalized, normalized, need_weights=False)
        queries = queries + update
        attn_mask = None
        if attention_bias is not None:
            bias = attention_bias
            if bias.ndim == 2:
                bias = bias[:, None, :].expand(-1, queries.shape[1], -1)
            if bias.shape != (queries.shape[0], queries.shape[1], particles.shape[1]):
                raise ValueError("root cross-attention bias must have shape [B, Q, N] or [B, N]")
            attn_mask = bias.repeat_interleave(self.num_heads, dim=0)
        update, _ = self.cross_attention(
            self.norm_cross(queries),
            particles,
            particles,
            key_padding_mask=~particle_mask,
            attn_mask=attn_mask,
            need_weights=False,
        )
        queries = queries + update
        return queries + self.ffn(self.norm_ffn(queries))


class _TypedHead(_ModuleBase):
    def __init__(self, d_model: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        torch = require_torch()
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.LayerNorm(2 * d_model),
            torch.nn.Linear(2 * d_model, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, query: Any, context: Any) -> Any:
        torch = require_torch()
        return self.network(torch.cat((query, context), dim=-1))


class SemanticRootPredictor(_ModuleBase):
    """Six typed root queries cross-attending to deployable HLT evidence."""

    def __init__(self, config: SemanticRootPredictorConfig | Mapping[str, Any] | None = None) -> None:
        torch = require_torch()
        super().__init__()
        if config is None:
            resolved = SemanticRootPredictorConfig()
        elif isinstance(config, SemanticRootPredictorConfig):
            resolved = config
        else:
            resolved = SemanticRootPredictorConfig(**dict(config))
        self.config = resolved
        self.particle_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(resolved.input_dim),
            torch.nn.Linear(resolved.input_dim, resolved.d_model),
        )
        self.jet_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(resolved.jet_input_dim),
            torch.nn.Linear(resolved.jet_input_dim, resolved.d_model),
        )
        self.query_tokens = torch.nn.Parameter(
            torch.empty(1, len(ABPH_ROOT_QUERY_NAMES), resolved.d_model)
        )
        self.query_type_embedding = torch.nn.Embedding(
            len(ABPH_ROOT_QUERY_NAMES), resolved.d_model
        )
        torch.nn.init.trunc_normal_(self.query_tokens, std=0.02)
        torch.nn.init.trunc_normal_(self.query_type_embedding.weight, std=0.02)
        self.blocks = torch.nn.ModuleList(
            [_SemanticQueryBlock(resolved) for _ in range(resolved.query_blocks)]
        )
        self.query_norm = torch.nn.LayerNorm(resolved.d_model)
        self.context_token = torch.nn.Parameter(torch.zeros(1, 1, resolved.d_model))
        torch.nn.init.trunc_normal_(self.context_token, std=0.02)
        self.context_attention = torch.nn.MultiheadAttention(
            resolved.d_model,
            resolved.num_heads,
            dropout=resolved.attention_dropout,
            batch_first=True,
        )
        self.context_norm = torch.nn.LayerNorm(resolved.d_model)
        self.context_ffn = torch.nn.Sequential(
            torch.nn.Linear(resolved.d_model, resolved.ffn_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(resolved.dropout),
            torch.nn.Linear(resolved.ffn_dim, resolved.d_model),
        )
        hidden = resolved.ffn_dim
        self.p4_head = _TypedHead(resolved.d_model, hidden, 8, resolved.dropout)
        self.count_head = _TypedHead(
            resolved.d_model, hidden, resolved.max_count + 2, resolved.dropout
        )
        composition_dim = 3 * len(ABPH_PID_CATEGORIES)
        self.composition_head = _TypedHead(
            resolved.d_model,
            hidden,
            2 * composition_dim + 1,
            resolved.dropout,
        )
        self.charge_head = _TypedHead(
            resolved.d_model,
            hidden,
            ABPH_CHARGE_SUPPORT_SIZE + 2,
            resolved.dropout,
        )
        self.shape_head = _TypedHead(
            resolved.d_model,
            hidden,
            ABPH_SHAPE_RAW_DIM + len(ROOT_SHAPE_FEATURE_NAMES),
            resolved.dropout,
        )
        uncertainty_dim = (
            4
            + 1
            + 3 * len(ABPH_PID_CATEGORIES)
            + 1
            + len(ROOT_SHAPE_FEATURE_NAMES)
        )
        self.uncertainty_head = _TypedHead(
            resolved.d_model, hidden, uncertainty_dim, resolved.dropout
        )

    def _bounded_log_scale(self, values: Any) -> Any:
        midpoint = 0.5 * (self.config.log_scale_min + self.config.log_scale_max)
        half_range = 0.5 * (self.config.log_scale_max - self.config.log_scale_min)
        return midpoint + half_range * values.tanh()

    def forward(
        self,
        particle_embeddings: Any,
        particle_mask: Any,
        jet_embedding: Any | None = None,
        particle_attention_bias: Any | None = None,
    ) -> SemanticRootPrediction:
        """Predict root distributions from HLT-derived embeddings only."""

        torch = require_torch()
        particles_raw = torch.as_tensor(particle_embeddings)
        mask = torch.as_tensor(particle_mask, device=particles_raw.device).bool()
        if particles_raw.ndim != 3 or particles_raw.shape[-1] != self.config.input_dim:
            raise ValueError(
                f"particle embeddings must have shape [B, N, {self.config.input_dim}]"
            )
        if mask.shape != particles_raw.shape[:2]:
            raise ValueError("particle mask does not match particle embeddings")
        if not bool(mask.any(dim=1).all()):
            raise ValueError("semantic root predictor requires one valid HLT particle per jet")
        particles = self.particle_projection(particles_raw)
        batch = int(particles.shape[0])
        if jet_embedding is None:
            weights = mask.to(particles.dtype).unsqueeze(-1)
            jet = (particles * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        else:
            jet_raw = torch.as_tensor(jet_embedding, device=particles.device)
            if jet_raw.shape != (batch, self.config.jet_input_dim):
                raise ValueError(
                    f"jet embedding must have shape [B, {self.config.jet_input_dim}]"
                )
            jet = self.jet_projection(jet_raw)
        type_ids = torch.arange(len(ABPH_ROOT_QUERY_NAMES), device=particles.device)
        queries = self.query_tokens.expand(batch, -1, -1)
        queries = queries + self.query_type_embedding(type_ids)[None, :, :]
        for block in self.blocks:
            queries = block(queries, particles, mask, particle_attention_bias)
        queries = self.query_norm(queries)
        context_seed = self.context_token.expand(batch, -1, -1) + jet[:, None, :]
        context_update, _ = self.context_attention(
            self.context_norm(context_seed), queries, queries, need_weights=False
        )
        context = (context_seed + context_update).squeeze(1)
        context = context + self.context_ffn(self.context_norm(context))

        p4_raw = self.p4_head(queries[:, 0], context)
        count_raw = self.count_head(queries[:, 1], context)
        composition_raw = self.composition_head(queries[:, 2], context)
        shape_raw = self.shape_head(queries[:, 3], context)
        charge_raw = self.charge_head(queries[:, 4], context)
        uncertainty_raw = self.uncertainty_head(queries[:, 5], context)
        offset = 0
        uncertainty_p4 = uncertainty_raw[:, offset : offset + 4]
        offset += 4
        uncertainty_count = uncertainty_raw[:, offset : offset + 1]
        offset += 1
        composition_dim = 3 * len(ABPH_PID_CATEGORIES)
        uncertainty_composition = uncertainty_raw[:, offset : offset + composition_dim]
        offset += composition_dim
        uncertainty_charge = uncertainty_raw[:, offset : offset + 1]
        offset += 1
        uncertainty_shape = uncertainty_raw[:, offset:]

        count_logits = count_raw[:, : self.config.max_count]
        count_distribution = count_raw[:, self.config.max_count :]
        composition_logits = composition_raw[:, :composition_dim]
        composition_scale = composition_raw[:, composition_dim : 2 * composition_dim]
        scalar_pt_excess_raw = composition_raw[:, -1]
        pid_count = len(ABPH_PID_CATEGORIES)
        charge_logits = charge_raw[:, :ABPH_CHARGE_SUPPORT_SIZE]
        charge_distribution = charge_raw[:, ABPH_CHARGE_SUPPORT_SIZE:]
        shape_mean_raw = shape_raw[:, :ABPH_SHAPE_RAW_DIM]
        shape_scale_raw = shape_raw[:, ABPH_SHAPE_RAW_DIM:]
        return SemanticRootPrediction(
            query_tokens=queries,
            shared_context=context,
            p4_residual_mean=p4_raw[:, :4],
            p4_residual_log_scale=self._bounded_log_scale(
                p4_raw[:, 4:] + uncertainty_p4
            ),
            count_logits=count_logits,
            delta_count_mean=count_distribution[:, 0],
            delta_count_log_scale=self._bounded_log_scale(
                count_distribution[:, 1] + uncertainty_count[:, 0]
            ),
            type_count_logits=composition_logits[:, :pid_count],
            type_pt_logits=composition_logits[:, pid_count : 2 * pid_count],
            type_energy_logits=composition_logits[:, 2 * pid_count :],
            composition_log_scale=self._bounded_log_scale(
                composition_scale + uncertainty_composition
            ),
            scalar_pt_excess_raw=scalar_pt_excess_raw,
            charge_logits=charge_logits,
            absolute_charge_mean=charge_distribution[:, 0],
            absolute_charge_log_scale=self._bounded_log_scale(
                charge_distribution[:, 1] + uncertainty_charge[:, 0]
            ),
            shape_raw=shape_mean_raw,
            shape_log_scale=self._bounded_log_scale(
                shape_scale_raw + uncertainty_shape
            ),
            diagnostics={
                "contract": ABPH_ROOT_PREDICTOR_CONTRACT,
                "query_names": ABPH_ROOT_QUERY_NAMES,
                "offline_inputs_loaded": False,
                "teacher_logits_loaded": False,
                "particle_attention_bias_used": particle_attention_bias is not None,
                "config_hash": self.config.to_dict()["config_hash"],
            },
        )


__all__ = [
    "ABPH_CHARGE_SUPPORT_MAX",
    "ABPH_CHARGE_SUPPORT_MIN",
    "ABPH_CHARGE_SUPPORT_SIZE",
    "ABPH_ROOT_PREDICTOR_CONTRACT",
    "ABPH_ROOT_QUERY_NAMES",
    "ABPH_SHAPE_RAW_DIM",
    "SemanticRootPrediction",
    "SemanticRootPredictor",
    "SemanticRootPredictorConfig",
]
