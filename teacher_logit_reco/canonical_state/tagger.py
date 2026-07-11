"""State-conditioned ParT taggers for Canonical Multi-Scale Jet State.

Step 6 keeps the HLT particle pathway as the main classifier input.  Canonical
state tokens are encoded separately, then injected as a zero-initialized
particle-embedding residual through the same ParT embedding hook pattern used
by the AV10 adapters.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

from jetclass_fresh.hlt_baseline import ParticleTransformerHLTClassifier, build_hlt_classifier, require_torch

from teacher_logit_reco.local_compression_part.config import LocalCompressionFeatureConfig
from teacher_logit_reco.local_compression_part.features import (
    LocalCompressionCanonicalInputs,
    build_local_compression_canonical_inputs,
)

from .layout import (
    CANONICAL_STATE_TOKEN_FAMILIES,
    CanonicalJetStateLayout,
    default_canonical_jet_state_layout,
)
from .predictor import (
    CanonicalStateResidualPredictorConfig,
    CanonicalStateResidualPredictorOutput,
    GeometryBiasedStateResidualDecoder,
    build_canonical_state_residual_predictor,
)

try:  # Keep import cheap on machines without torch.
    import torch as _torch
except ImportError:  # pragma: no cover - environment dependent
    _torch = None

if _torch is None:  # pragma: no cover - environment dependent
    class _ModuleBase:
        pass
else:
    _ModuleBase = _torch.nn.Module


CANONICAL_STATE_TAGGER_CONTRACT = "canonical_state_conditioned_part_tagger_v1"

STATE_CONTEXT_PARTICLES_ONLY = "particles_only"
STATE_CONTEXT_PHI_HLT = "phi_hlt_context"
STATE_CONTEXT_DELTA_PHI = "delta_phi_context"
STATE_CONTEXT_PHI_PRED = "phi_pred_context"
STATE_CONTEXT_ALL = "phi_hlt_delta_phi_phi_pred_context"
STATE_CONTEXT_STATE_ONLY = "state_only_tagger"
STATE_CONTEXT_SHUFFLED = "shuffled_state_control"
STATE_CONTEXT_NOISE = "noise_state_control"
STATE_CONTEXT_ORACLE_PHI_OFF = "oracle_phi_off_diagnostic"
STATE_CONTEXT_FEATURE_MLP_PLUS_STATE = "feature_mlp_adapter_plus_state_context"

CANONICAL_STATE_TAGGER_MODES: tuple[str, ...] = (
    STATE_CONTEXT_PARTICLES_ONLY,
    STATE_CONTEXT_PHI_HLT,
    STATE_CONTEXT_DELTA_PHI,
    STATE_CONTEXT_PHI_PRED,
    STATE_CONTEXT_ALL,
    STATE_CONTEXT_STATE_ONLY,
    STATE_CONTEXT_SHUFFLED,
    STATE_CONTEXT_NOISE,
    STATE_CONTEXT_ORACLE_PHI_OFF,
    STATE_CONTEXT_FEATURE_MLP_PLUS_STATE,
)

_MODE_ALIASES = {
    "particles": STATE_CONTEXT_PARTICLES_ONLY,
    "baseline": STATE_CONTEXT_PARTICLES_ONLY,
    "phi_hlt": STATE_CONTEXT_PHI_HLT,
    "hlt": STATE_CONTEXT_PHI_HLT,
    "delta": STATE_CONTEXT_DELTA_PHI,
    "delta_phi": STATE_CONTEXT_DELTA_PHI,
    "phi_pred": STATE_CONTEXT_PHI_PRED,
    "pred": STATE_CONTEXT_PHI_PRED,
    "all": STATE_CONTEXT_ALL,
    "state_only": STATE_CONTEXT_STATE_ONLY,
    "shuffled": STATE_CONTEXT_SHUFFLED,
    "noise": STATE_CONTEXT_NOISE,
    "oracle": STATE_CONTEXT_ORACLE_PHI_OFF,
    "phi_off": STATE_CONTEXT_ORACLE_PHI_OFF,
    "feature_mlp_plus_state": STATE_CONTEXT_FEATURE_MLP_PLUS_STATE,
}


def normalize_state_tagger_mode(value: str) -> str:
    key = str(value).strip()
    if key in CANONICAL_STATE_TAGGER_MODES:
        return key
    if key in _MODE_ALIASES:
        return _MODE_ALIASES[key]
    raise ValueError(f"unknown canonical state tagger mode {value!r}")


def _nan_to_num_torch(value: Any, *, nan: float = 0.0, posinf: float = 0.0, neginf: float = 0.0) -> Any:
    torch = require_torch()
    if hasattr(torch, "nan_to_num"):
        return torch.nan_to_num(value, nan=float(nan), posinf=float(posinf), neginf=float(neginf))
    return torch.where(torch.isfinite(value), value, value.new_full((), float(nan)))


def _coerce_tokens_and_mask(tokens_or_batch: Any, mask: Any | None = None) -> tuple[Any, Any]:
    torch = require_torch()
    if isinstance(tokens_or_batch, Mapping):
        batch = tokens_or_batch
        token_keys = ("tokens", "hlt_tokens", "raw_tokens", "x")
        mask_keys = ("mask", "hlt_mask", "raw_mask", "particle_mask")
        tokens = None
        resolved_mask = mask
        for key in token_keys:
            if key in batch:
                tokens = batch[key]
                break
        if tokens is None:
            raise ValueError(f"batch mapping must contain one of {token_keys}")
        if resolved_mask is None:
            for key in mask_keys:
                if key in batch:
                    resolved_mask = batch[key]
                    break
        if resolved_mask is None:
            raise ValueError(f"batch mapping must contain one of {mask_keys} when mask is not provided")
    else:
        if mask is None:
            raise ValueError("mask is required when passing raw token tensors")
        tokens = tokens_or_batch
        resolved_mask = mask
    tokens = torch.as_tensor(tokens).float()
    resolved_mask = torch.as_tensor(resolved_mask, device=tokens.device).bool()
    if int(tokens.ndim) != 3:
        raise ValueError(f"tokens must have shape [batch, particles, raw_dim], got {tuple(tokens.shape)}")
    if int(resolved_mask.ndim) != 2:
        raise ValueError(f"mask must have shape [batch, particles], got {tuple(resolved_mask.shape)}")
    if tuple(tokens.shape[:2]) != tuple(resolved_mask.shape):
        raise ValueError(f"tokens/mask leading shapes differ: {tuple(tokens.shape[:2])} vs {tuple(resolved_mask.shape)}")
    return _nan_to_num_torch(tokens), resolved_mask


def _resolve_embed_module(part_model: Any) -> tuple[str, Any]:
    candidates = (
        ("mod.embed", getattr(getattr(part_model, "mod", None), "embed", None)),
        ("embed", getattr(part_model, "embed", None)),
    )
    for name, module in candidates:
        if module is not None and hasattr(module, "register_forward_hook"):
            return name, module
    if hasattr(part_model, "named_modules"):
        for name, module in part_model.named_modules():
            if str(name).endswith("embed") and hasattr(module, "register_forward_hook"):
                return str(name), module
    raise ValueError(
        "Could not locate the ParT particle embedding module. "
        "Expected part_model.mod.embed or an equivalent named module ending in 'embed'."
    )


def _delta_for_embed_output(delta_h: Any, embed_output: Any) -> Any:
    torch = require_torch()
    if not isinstance(embed_output, torch.Tensor):
        raise TypeError(f"ParT embed output must be a tensor, got {type(embed_output)!r}")
    if int(embed_output.ndim) != 3:
        raise ValueError(f"ParT embed output must be rank 3, got {tuple(embed_output.shape)}")
    batch, particles, dim = delta_h.shape
    shape = tuple(embed_output.shape)
    if int(shape[-1]) == int(dim):
        if int(shape[0]) == int(batch):
            if int(shape[1]) > int(particles):
                raise ValueError(f"embed particle dimension {shape[1]} exceeds delta_h particles {particles}")
            return delta_h[:, : int(shape[1]), :]
        if int(shape[1]) == int(batch):
            if int(shape[0]) > int(particles):
                raise ValueError(f"embed particle dimension {shape[0]} exceeds delta_h particles {particles}")
            return delta_h[:, : int(shape[0]), :].permute(1, 0, 2).contiguous()
    if int(shape[0]) == int(batch) and int(shape[1]) == int(dim):
        if int(shape[2]) > int(particles):
            raise ValueError(f"embed particle dimension {shape[2]} exceeds delta_h particles {particles}")
        return delta_h[:, : int(shape[2]), :].transpose(1, 2).contiguous()
    raise ValueError(f"Cannot align delta_h shape {tuple(delta_h.shape)} with ParT embed output shape {shape}")


def _batch_first_from_embed_output(embed_output: Any, *, batch: int, particles: int, dim: int) -> Any:
    torch = require_torch()
    if not isinstance(embed_output, torch.Tensor):
        raise TypeError(f"ParT embed output must be a tensor, got {type(embed_output)!r}")
    if int(embed_output.ndim) != 3:
        raise ValueError(f"ParT embed output must be rank 3, got {tuple(embed_output.shape)}")
    shape = tuple(embed_output.shape)
    if int(shape[-1]) == int(dim):
        if int(shape[0]) == int(batch):
            return embed_output[:, :particles, :]
        if int(shape[1]) == int(batch):
            return embed_output[:particles, :, :].permute(1, 0, 2).contiguous()
    if int(shape[0]) == int(batch) and int(shape[1]) == int(dim):
        return embed_output[:, :, :particles].transpose(1, 2).contiguous()
    raise ValueError(f"Cannot convert embed output shape {shape} to [B, P, D]")


@dataclass(frozen=True)
class CanonicalStateTaggerConfig:
    """Configuration for the Step 6 state-conditioned ParT wrapper."""

    mode: str = STATE_CONTEXT_PHI_HLT
    num_classes: int = 10
    part_model_size: str = "base"
    part_embed_dim: int = 128
    state_dim: int = 128
    state_layers: int = 2
    state_heads: int = 4
    state_mlp_ratio: float = 2.0
    dropout: float = 0.05
    attention_dropout: float = 0.05
    max_constits: int = 128
    max_state_slots: int = 32
    context_gate_bias_init: float = -6.0
    shuffle_seed: int = 1729
    allow_oracle_final_test_context: bool = False
    predictor_config: Mapping[str, Any] | None = None
    feature_config: LocalCompressionFeatureConfig | Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        mode = normalize_state_tagger_mode(self.mode)
        for name in ("num_classes", "part_embed_dim", "state_dim", "state_layers", "state_heads", "max_constits", "max_state_slots"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if int(self.state_dim) % int(self.state_heads) != 0:
            raise ValueError("state_dim must be divisible by state_heads")
        if float(self.state_mlp_ratio) <= 0.0:
            raise ValueError("state_mlp_ratio must be positive")
        for name in ("dropout", "attention_dropout"):
            value = float(getattr(self, name))
            if value < 0.0 or value >= 1.0:
                raise ValueError(f"{name} must be in [0, 1)")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "num_classes", int(self.num_classes))
        object.__setattr__(self, "part_embed_dim", int(self.part_embed_dim))
        object.__setattr__(self, "state_dim", int(self.state_dim))
        object.__setattr__(self, "state_layers", int(self.state_layers))
        object.__setattr__(self, "state_heads", int(self.state_heads))
        object.__setattr__(self, "state_mlp_ratio", float(self.state_mlp_ratio))
        object.__setattr__(self, "dropout", float(self.dropout))
        object.__setattr__(self, "attention_dropout", float(self.attention_dropout))
        object.__setattr__(self, "max_constits", int(self.max_constits))
        object.__setattr__(self, "max_state_slots", int(self.max_state_slots))
        object.__setattr__(self, "context_gate_bias_init", float(self.context_gate_bias_init))
        object.__setattr__(self, "shuffle_seed", int(self.shuffle_seed))
        object.__setattr__(self, "allow_oracle_final_test_context", bool(self.allow_oracle_final_test_context))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["contract"] = CANONICAL_STATE_TAGGER_CONTRACT
        return payload


@dataclass(frozen=True)
class CanonicalStateTaggerOutput:
    logits: Any
    canonical_inputs: LocalCompressionCanonicalInputs
    config: CanonicalStateTaggerConfig
    state_values: Any | None
    state_mask: Any | None
    state_context: Any | None
    delta_h: Any | None
    gate: Any | None
    predictor_output: CanonicalStateResidualPredictorOutput | None
    diagnostics: dict[str, Any]

    @property
    def output_contract(self) -> str:
        return CANONICAL_STATE_TAGGER_CONTRACT


class CanonicalStateTokenEncoder(_ModuleBase):
    """Encode Phi-like canonical state rows plus token metadata."""

    def __init__(self, config: CanonicalStateTaggerConfig, *, layout: CanonicalJetStateLayout) -> None:
        torch = require_torch()
        super().__init__()
        self.config = config
        self.layout = layout
        self.value_projection = torch.nn.Linear(layout.d_phi, int(config.state_dim))
        self.context_embedding = torch.nn.Embedding(4, int(config.state_dim))
        self.token_type_embedding = torch.nn.Embedding(len(CANONICAL_STATE_TOKEN_FAMILIES), int(config.state_dim))
        self.scale_embedding = torch.nn.Embedding(len(CANONICAL_STATE_TOKEN_FAMILIES), int(config.state_dim))
        self.slot_embedding = torch.nn.Embedding(int(config.max_state_slots), int(config.state_dim))
        layer = torch.nn.TransformerEncoderLayer(
            d_model=int(config.state_dim),
            nhead=int(config.state_heads),
            dim_feedforward=int(round(float(config.state_mlp_ratio) * int(config.state_dim))),
            dropout=float(config.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(layer, num_layers=int(config.state_layers))
        self.norm = torch.nn.LayerNorm(int(config.state_dim))
        self._register_layout_buffers()

    def _register_layout_buffers(self) -> None:
        torch = require_torch()
        token_type_ids = torch.tensor(self.layout.token_type_ids, dtype=torch.long)
        scale_ids = torch.tensor(self.layout.scale_ids, dtype=torch.long)
        slot_ids = torch.tensor(
            [min(int(slot), int(self.config.max_state_slots) - 1) for slot in self.layout.slot_ids],
            dtype=torch.long,
        )
        self.register_buffer("token_type_ids", token_type_ids, persistent=False)
        self.register_buffer("scale_ids", scale_ids, persistent=False)
        self.register_buffer("slot_ids", slot_ids, persistent=False)

    def forward(
        self,
        state_values: Any,
        *,
        state_mask: Any,
        context_ids: Any,
    ) -> Any:
        torch = require_torch()
        values = torch.as_tensor(state_values).float()
        mask = torch.as_tensor(state_mask, device=values.device).bool()
        context_ids = torch.as_tensor(context_ids, device=values.device).long()
        if int(values.ndim) != 3 or int(values.shape[-1]) != int(self.layout.d_phi):
            raise ValueError(f"state_values must have shape [B, T, {self.layout.d_phi}], got {tuple(values.shape)}")
        if tuple(mask.shape) != tuple(values.shape[:2]):
            raise ValueError("state_mask must match state_values leading shape")
        if tuple(context_ids.shape) != tuple(mask.shape):
            raise ValueError("context_ids must match state_mask shape")
        repeats = max(1, int(values.shape[1]) // int(self.layout.k_state))
        token_type_ids = self.token_type_ids.repeat(repeats)[: int(values.shape[1])]
        scale_ids = self.scale_ids.repeat(repeats)[: int(values.shape[1])]
        slot_ids = self.slot_ids.repeat(repeats)[: int(values.shape[1])]
        x = self.value_projection(_nan_to_num_torch(values))
        metadata = (
            self.token_type_embedding(token_type_ids)
            + self.scale_embedding(scale_ids)
            + self.slot_embedding(slot_ids)
        )
        x = x + metadata[None, :, :] + self.context_embedding(context_ids)
        padding_mask = ~mask
        all_invalid = padding_mask.all(dim=1)
        if bool(all_invalid.any()):
            padding_mask = padding_mask.clone()
            padding_mask[all_invalid, 0] = False
        encoded = self.encoder(x, src_key_padding_mask=padding_mask)
        return self.norm(encoded) * mask[:, :, None].to(dtype=encoded.dtype)


class StateToParticleCrossAttentionAdapter(_ModuleBase):
    """Particle embedding queries attend to encoded canonical state context."""

    def __init__(self, config: CanonicalStateTaggerConfig) -> None:
        torch = require_torch()
        super().__init__()
        self.config = config
        self.state_projection = torch.nn.Linear(int(config.state_dim), int(config.part_embed_dim))
        self.attn = torch.nn.MultiheadAttention(
            int(config.part_embed_dim),
            int(config.state_heads),
            dropout=float(config.attention_dropout),
            batch_first=True,
        )
        hidden = int(round(float(config.state_mlp_ratio) * int(config.part_embed_dim)))
        self.norm_particle = torch.nn.LayerNorm(int(config.part_embed_dim))
        self.norm_state = torch.nn.LayerNorm(int(config.part_embed_dim))
        self.delta_mlp = torch.nn.Sequential(
            torch.nn.Linear(int(config.part_embed_dim), hidden),
            torch.nn.GELU(),
            torch.nn.Dropout(float(config.dropout)),
            torch.nn.Linear(hidden, int(config.part_embed_dim)),
        )
        self.gate_mlp = torch.nn.Linear(int(config.part_embed_dim), 1)
        torch.nn.init.zeros_(self.delta_mlp[-1].weight)
        torch.nn.init.zeros_(self.delta_mlp[-1].bias)
        torch.nn.init.zeros_(self.gate_mlp.weight)
        torch.nn.init.constant_(self.gate_mlp.bias, float(config.context_gate_bias_init))

    def forward(self, particle_embeddings: Any, particle_mask: Any, state_context: Any, state_mask: Any) -> tuple[Any, Any, dict[str, Any]]:
        torch = require_torch()
        particles = torch.as_tensor(particle_embeddings).float()
        particle_mask = torch.as_tensor(particle_mask, device=particles.device).bool()
        state = self.state_projection(torch.as_tensor(state_context, device=particles.device).float())
        state_mask = torch.as_tensor(state_mask, device=particles.device).bool()
        if tuple(particle_mask.shape) != tuple(particles.shape[:2]):
            raise ValueError("particle_mask must match particle embedding leading shape")
        if tuple(state_mask.shape) != tuple(state.shape[:2]):
            raise ValueError("state_mask must match state context leading shape")
        state_padding_mask = ~state_mask
        all_invalid = state_padding_mask.all(dim=1)
        if bool(all_invalid.any()):
            state_padding_mask = state_padding_mask.clone()
            state_padding_mask[all_invalid, 0] = False
        query = self.norm_particle(particles)
        key_value = self.norm_state(state)
        attended, attn_weights = self.attn(
            query,
            key_value,
            key_value,
            key_padding_mask=state_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        valid = particle_mask[:, :, None].to(dtype=particles.dtype)
        adapter_output = _nan_to_num_torch(self.delta_mlp(attended)) * valid
        gate = torch.sigmoid(self.gate_mlp(attended)) * valid
        delta_h = gate * adapter_output
        diagnostics = _delta_h_diagnostics(delta_h, gate, adapter_output, particle_mask)
        diagnostics.update(
            {
                "state_cross_attention_shape": list(attn_weights.shape),
                "state_valid_count_mean": float(state_mask.sum(dim=1).float().mean().detach().cpu().item()),
            }
        )
        return delta_h, gate, diagnostics


class FeatureMLPEmbeddingAdapter(_ModuleBase):
    """AV10-style feature MLP residual used by the combined Step 6 mode."""

    def __init__(self, config: CanonicalStateTaggerConfig, *, feature_dim: int) -> None:
        torch = require_torch()
        super().__init__()
        hidden = int(round(float(config.state_mlp_ratio) * int(config.part_embed_dim)))
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(int(feature_dim)),
            torch.nn.Linear(int(feature_dim), hidden),
            torch.nn.GELU(),
            torch.nn.Dropout(float(config.dropout)),
            torch.nn.Linear(hidden, int(config.part_embed_dim)),
        )
        self.gate = torch.nn.Sequential(
            torch.nn.LayerNorm(int(feature_dim)),
            torch.nn.Linear(int(feature_dim), 1),
        )
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)
        torch.nn.init.zeros_(self.gate[-1].weight)
        torch.nn.init.constant_(self.gate[-1].bias, float(config.context_gate_bias_init))

    def forward(self, feature_rows: Any, particle_mask: Any) -> tuple[Any, Any, dict[str, Any]]:
        torch = require_torch()
        rows = torch.as_tensor(feature_rows).float()
        mask = torch.as_tensor(particle_mask, device=rows.device).bool()
        valid = mask[:, :, None].to(dtype=rows.dtype)
        adapter_output = _nan_to_num_torch(self.net(rows)) * valid
        gate = torch.sigmoid(self.gate(rows)) * valid
        delta_h = gate * adapter_output
        diagnostics = _delta_h_diagnostics(delta_h, gate, adapter_output, mask)
        diagnostics["feature_mlp_adapter_active"] = True
        return delta_h, gate, diagnostics


def _delta_h_diagnostics(delta_h: Any, gate: Any, adapter_output: Any, mask: Any) -> dict[str, Any]:
    torch = require_torch()
    mask = torch.as_tensor(mask, device=delta_h.device).bool()
    if not bool(mask.any()):
        return {
            "delta_h_norm_mean": 0.0,
            "delta_h_norm_p90": 0.0,
            "delta_h_norm_max": 0.0,
            "adapter_output_norm_mean": 0.0,
            "adapter_output_norm_max": 0.0,
            "gate_mean": 0.0,
        }
    delta_norm = delta_h.norm(dim=-1)[mask]
    adapter_norm = adapter_output.norm(dim=-1)[mask]
    gate_values = gate.squeeze(-1)[mask]
    return {
        "delta_h_norm_mean": float(delta_norm.mean().detach().cpu().item()),
        "delta_h_norm_p90": float(torch.quantile(delta_norm, 0.90).detach().cpu().item()) if int(delta_norm.numel()) else 0.0,
        "delta_h_norm_max": float(delta_norm.max().detach().cpu().item()) if int(delta_norm.numel()) else 0.0,
        "adapter_output_norm_mean": float(adapter_norm.mean().detach().cpu().item()),
        "adapter_output_norm_max": float(adapter_norm.max().detach().cpu().item()) if int(adapter_norm.numel()) else 0.0,
        "gate_mean": float(gate_values.mean().detach().cpu().item()),
    }


class CanonicalStateConditionedParT(_ModuleBase):
    """HLT ParT wrapper with canonical-state context injection."""

    def __init__(
        self,
        config: CanonicalStateTaggerConfig | Mapping[str, Any] | None = None,
        *,
        part_model: Any | None = None,
        predictor: GeometryBiasedStateResidualDecoder | None = None,
        layout: CanonicalJetStateLayout | None = None,
    ) -> None:
        require_torch()
        super().__init__()
        self.config = config if isinstance(config, CanonicalStateTaggerConfig) else CanonicalStateTaggerConfig(**dict(config or {}))
        self.layout = default_canonical_jet_state_layout() if layout is None else layout
        self.part_model = part_model or build_hlt_classifier(
            num_classes=int(self.config.num_classes),
            model_size=str(self.config.part_model_size),
        )
        self.uses_reference_part_backbone = isinstance(self.part_model, ParticleTransformerHLTClassifier)
        self.embed_module_name, self.embed_module = _resolve_embed_module(self.part_model)
        predictor_config = dict(self.config.predictor_config or {})
        self.state_predictor = predictor or build_canonical_state_residual_predictor(**predictor_config)
        self.state_encoder = CanonicalStateTokenEncoder(self.config, layout=self.layout)
        self.state_adapter = StateToParticleCrossAttentionAdapter(self.config)
        self.feature_adapter = FeatureMLPEmbeddingAdapter(
            self.config,
            feature_dim=len(LocalCompressionFeatureConfig().canonical_feature_names),
        )
        self.state_only_head = _torch.nn.Sequential(
            _torch.nn.LayerNorm(int(self.config.state_dim)),
            _torch.nn.Linear(int(self.config.state_dim), int(self.config.state_dim)),
            _torch.nn.GELU(),
            _torch.nn.Linear(int(self.config.state_dim), int(self.config.num_classes)),
        )
        generator = _torch.Generator()
        generator.manual_seed(int(self.config.shuffle_seed))
        permutation = _torch.randperm(int(self.layout.k_state), generator=generator)
        self.register_buffer("state_shuffle_permutation", permutation, persistent=False)
        self._pending_state_context: Any | None = None
        self._pending_state_mask: Any | None = None
        self._pending_feature_delta_h: Any | None = None
        self._pending_feature_gate: Any | None = None
        self._pending_particle_mask: Any | None = None
        self._last_delta_h: Any | None = None
        self._last_gate: Any | None = None
        self._last_injection_diagnostics: dict[str, Any] = {}

    @property
    def mode(self) -> str:
        return str(self.config.mode)

    def build_canonical_inputs(self, tokens: Any, mask: Any) -> LocalCompressionCanonicalInputs:
        return build_local_compression_canonical_inputs(
            tokens,
            mask,
            max_constits=int(self.config.max_constits),
            config=self.config.feature_config,
        )

    def _requires_predictor(self) -> bool:
        return self.mode in {
            STATE_CONTEXT_DELTA_PHI,
            STATE_CONTEXT_PHI_PRED,
            STATE_CONTEXT_ALL,
            STATE_CONTEXT_FEATURE_MLP_PLUS_STATE,
        }

    def _validate_phi(self, name: str, value: Any, *, device: Any) -> Any:
        torch = require_torch()
        tensor = torch.as_tensor(value, device=device).float()
        if tuple(tensor.shape[-2:]) != (self.layout.k_state, self.layout.d_phi):
            raise ValueError(f"{name} must have trailing shape {(self.layout.k_state, self.layout.d_phi)}, got {tuple(tensor.shape)}")
        return _nan_to_num_torch(tensor)

    def _validate_state_mask(self, value: Any | None, *, phi_like: Any) -> Any:
        torch = require_torch()
        if value is None:
            return torch.ones(phi_like.shape[:2], device=phi_like.device, dtype=torch.bool)
        mask = torch.as_tensor(value, device=phi_like.device).bool()
        if tuple(mask.shape) != tuple(phi_like.shape[:2]):
            raise ValueError(f"state_mask must match Phi leading shape {tuple(phi_like.shape[:2])}, got {tuple(mask.shape)}")
        return mask

    def _predict_state(
        self,
        canonical: LocalCompressionCanonicalInputs,
        phi_hlt: Any,
        *,
        state_mask: Any | None,
        delta_phi: Any | None,
        phi_pred: Any | None,
    ) -> tuple[Any | None, Any | None, CanonicalStateResidualPredictorOutput | None]:
        if delta_phi is not None and phi_pred is None:
            return delta_phi, phi_hlt + delta_phi, None
        if phi_pred is not None and delta_phi is None:
            return phi_pred - phi_hlt, phi_pred, None
        if not self._requires_predictor() and delta_phi is None and phi_pred is None:
            return delta_phi, phi_pred, None
        if delta_phi is not None and phi_pred is not None:
            return delta_phi, phi_pred, None
        predictor_output = self.state_predictor(
            canonical.selected_tokens,
            canonical.particle_mask,
            phi_hlt,
            state_mask=state_mask,
        )
        resolved_delta = predictor_output.delta_phi if delta_phi is None else delta_phi
        resolved_pred = predictor_output.phi_pred if phi_pred is None else phi_pred
        return resolved_delta, resolved_pred, predictor_output

    def _state_values_for_mode(
        self,
        *,
        canonical: LocalCompressionCanonicalInputs,
        phi_hlt: Any,
        delta_phi: Any | None,
        phi_pred: Any | None,
        phi_off: Any | None,
        state_mask: Any | None,
        delta_state_mask: Any | None,
        phi_pred_state_mask: Any | None,
        split: str | None,
        allow_oracle_context: bool,
    ) -> tuple[Any | None, Any | None, Any | None, dict[str, Any]]:
        torch = require_torch()
        device = canonical.features.device
        diagnostics: dict[str, Any] = {
            "state_context_mode": self.mode,
            "oracle_context_used": False,
            "state_semantics_broken": False,
            "state_mask_policy": "hlt_state_mask",
        }
        if self.mode == STATE_CONTEXT_PARTICLES_ONLY:
            return None, None, None, diagnostics
        if self.mode == STATE_CONTEXT_ORACLE_PHI_OFF:
            if str(split or "").lower() == "final_test" and not (allow_oracle_context or bool(self.config.allow_oracle_final_test_context)):
                raise ValueError("oracle Phi_off context is blocked for primary final_test evaluation")
            if phi_off is None:
                raise ValueError("phi_off is required for oracle Phi_off diagnostic context")
            values = self._validate_phi("phi_off", phi_off, device=device)
            diagnostics["oracle_context_used"] = True
            context_ids = torch.zeros(values.shape[:2], device=device, dtype=torch.long)
            resolved_mask = self._validate_state_mask(state_mask, phi_like=values)
        elif self.mode in {STATE_CONTEXT_PHI_HLT, STATE_CONTEXT_SHUFFLED, STATE_CONTEXT_NOISE}:
            values = self._validate_phi("phi_hlt", phi_hlt, device=device)
            context_ids = torch.zeros(values.shape[:2], device=device, dtype=torch.long)
            resolved_mask = self._validate_state_mask(state_mask, phi_like=values)
            if self.mode == STATE_CONTEXT_SHUFFLED:
                values = values[:, self.state_shuffle_permutation.to(device=values.device), :]
                resolved_mask = resolved_mask[:, self.state_shuffle_permutation.to(device=values.device)]
                diagnostics["state_semantics_broken"] = True
                diagnostics["state_control"] = "values_shuffled_against_token_metadata"
            elif self.mode == STATE_CONTEXT_NOISE:
                index = torch.arange(values.numel(), device=values.device, dtype=values.dtype).reshape_as(values)
                values = torch.sin(index * 12.9898 + 78.233)
                diagnostics["state_semantics_broken"] = True
                diagnostics["state_control"] = "deterministic_sinusoidal_noise"
        elif self.mode == STATE_CONTEXT_DELTA_PHI:
            if delta_phi is None:
                raise ValueError("delta_phi is required or must be predicted for delta_phi_context")
            pieces = [
                self._validate_phi("phi_hlt", phi_hlt, device=device),
                self._validate_phi("delta_phi", delta_phi, device=device),
            ]
            values = torch.cat(pieces, dim=1)
            ids = [
                torch.zeros(pieces[0].shape[:2], device=device, dtype=torch.long),
                torch.ones(pieces[1].shape[:2], device=device, dtype=torch.long),
            ]
            context_ids = torch.cat(ids, dim=1)
            base_mask = self._validate_state_mask(state_mask, phi_like=pieces[0])
            delta_mask = (
                self._validate_state_mask(delta_state_mask, phi_like=pieces[1])
                if delta_state_mask is not None
                else base_mask
            )
            if delta_state_mask is not None:
                diagnostics["state_mask_policy"] = "hlt_mask_for_phi_hlt__separate_mask_for_delta_phi"
            resolved_mask = torch.cat([base_mask, delta_mask], dim=1)
        elif self.mode == STATE_CONTEXT_PHI_PRED:
            if phi_pred is None:
                raise ValueError("phi_pred is required or must be predicted for phi_pred context")
            pieces = [
                self._validate_phi("phi_hlt", phi_hlt, device=device),
                self._validate_phi("phi_pred", phi_pred, device=device),
            ]
            values = torch.cat(pieces, dim=1)
            ids = [
                torch.zeros(pieces[0].shape[:2], device=device, dtype=torch.long),
                torch.full(pieces[1].shape[:2], 2, device=device, dtype=torch.long),
            ]
            context_ids = torch.cat(ids, dim=1)
            base_mask = self._validate_state_mask(state_mask, phi_like=pieces[0])
            pred_mask = (
                self._validate_state_mask(phi_pred_state_mask, phi_like=pieces[1])
                if phi_pred_state_mask is not None
                else base_mask
            )
            if phi_pred_state_mask is not None:
                diagnostics["state_mask_policy"] = "hlt_mask_for_phi_hlt__separate_mask_for_phi_pred"
            resolved_mask = torch.cat([base_mask, pred_mask], dim=1)
        elif self.mode == STATE_CONTEXT_FEATURE_MLP_PLUS_STATE:
            if phi_pred is None:
                raise ValueError("phi_pred is required or must be predicted for feature_mlp_plus_state context")
            values = self._validate_phi("phi_pred", phi_pred, device=device)
            context_ids = torch.full(values.shape[:2], 2, device=device, dtype=torch.long)
            resolved_mask = self._validate_state_mask(
                phi_pred_state_mask if phi_pred_state_mask is not None else state_mask,
                phi_like=values,
            )
            if phi_pred_state_mask is not None:
                diagnostics["state_mask_policy"] = "separate_mask_for_phi_pred"
        elif self.mode == STATE_CONTEXT_ALL:
            if delta_phi is None or phi_pred is None:
                raise ValueError("delta_phi and phi_pred are required or must be predicted for all-state context")
            pieces = [
                self._validate_phi("phi_hlt", phi_hlt, device=device),
                self._validate_phi("delta_phi", delta_phi, device=device),
                self._validate_phi("phi_pred", phi_pred, device=device),
            ]
            values = torch.cat(pieces, dim=1)
            ids = [
                torch.zeros(pieces[0].shape[:2], device=device, dtype=torch.long),
                torch.ones(pieces[1].shape[:2], device=device, dtype=torch.long),
                torch.full(pieces[2].shape[:2], 2, device=device, dtype=torch.long),
            ]
            context_ids = torch.cat(ids, dim=1)
            base_mask = self._validate_state_mask(state_mask, phi_like=pieces[0])
            delta_mask = (
                self._validate_state_mask(delta_state_mask, phi_like=pieces[1])
                if delta_state_mask is not None
                else base_mask
            )
            pred_mask = (
                self._validate_state_mask(phi_pred_state_mask, phi_like=pieces[2])
                if phi_pred_state_mask is not None
                else base_mask
            )
            if delta_state_mask is not None or phi_pred_state_mask is not None:
                diagnostics["state_mask_policy"] = "separate_masks_for_residual_state_pieces"
            resolved_mask = torch.cat([base_mask, delta_mask, pred_mask], dim=1)
        elif self.mode == STATE_CONTEXT_STATE_ONLY:
            values = self._validate_phi("phi_hlt", phi_hlt, device=device)
            context_ids = torch.zeros(values.shape[:2], device=device, dtype=torch.long)
            resolved_mask = self._validate_state_mask(state_mask, phi_like=values)
        else:  # pragma: no cover - guarded by config validation
            raise ValueError(f"unsupported state context mode {self.mode!r}")
        diagnostics["state_token_count"] = int(values.shape[1])
        diagnostics["state_valid_count_mean"] = float(resolved_mask.sum(dim=1).float().mean().detach().cpu().item())
        diagnostics["state_context_ids"] = sorted(int(x) for x in torch.unique(context_ids).detach().cpu().tolist())
        return values, resolved_mask, context_ids, diagnostics

    def _embed_injection_hook(self, module: Any, inputs: Any, output: Any) -> Any:
        del module, inputs
        torch = require_torch()
        if self._pending_state_context is None or self._pending_state_mask is None:
            return output
        particle_embeddings = _batch_first_from_embed_output(
            output,
            batch=int(self._pending_particle_mask.shape[0]),
            particles=int(self._pending_particle_mask.shape[1]),
            dim=int(self.config.part_embed_dim),
        )
        state_delta_h, state_gate, state_diag = self.state_adapter(
            particle_embeddings,
            self._pending_particle_mask,
            self._pending_state_context,
            self._pending_state_mask,
        )
        total_delta = state_delta_h
        total_gate = state_gate
        if self._pending_feature_delta_h is not None:
            total_delta = total_delta + self._pending_feature_delta_h[:, : int(total_delta.shape[1]), :]
            if self._pending_feature_gate is not None:
                total_gate = 0.5 * (total_gate + self._pending_feature_gate[:, : int(total_gate.shape[1]), :])
        total_delta = total_delta * self._pending_particle_mask[:, : int(total_delta.shape[1]), None].to(dtype=total_delta.dtype)
        self._last_delta_h = total_delta
        self._last_gate = total_gate
        self._last_injection_diagnostics = {
            "embed_module_name": str(self.embed_module_name),
            "injection_applied": True,
            "delta_h_shape": list(total_delta.shape),
            "state_adapter": state_diag,
        }
        aligned = _delta_for_embed_output(total_delta, output)
        return output + aligned.to(device=output.device, dtype=output.dtype)

    def _part_forward_with_state(
        self,
        canonical: LocalCompressionCanonicalInputs,
        state_context: Any | None,
        state_mask: Any | None,
        *,
        feature_delta_h: Any | None = None,
        feature_gate: Any | None = None,
    ) -> Any:
        self._pending_state_context = state_context
        self._pending_state_mask = state_mask
        self._pending_particle_mask = canonical.particle_mask
        self._pending_feature_delta_h = feature_delta_h
        self._pending_feature_gate = feature_gate
        self._last_delta_h = None
        self._last_gate = None
        self._last_injection_diagnostics = {"embed_module_name": str(self.embed_module_name), "injection_applied": False}
        handle = self.embed_module.register_forward_hook(self._embed_injection_hook)
        try:
            return self.part_model(canonical.points, canonical.features, canonical.lorentz_vectors, canonical.mask)
        finally:
            handle.remove()
            self._pending_state_context = None
            self._pending_state_mask = None
            self._pending_particle_mask = None
            self._pending_feature_delta_h = None
            self._pending_feature_gate = None

    def _state_only_logits(self, state_context: Any, state_mask: Any) -> Any:
        valid = state_mask[:, :, None].to(dtype=state_context.dtype)
        pooled = (state_context * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        return self.state_only_head(pooled)

    def forward(
        self,
        tokens_or_batch: Any,
        mask: Any | None = None,
        *,
        phi_hlt: Any | None = None,
        state_mask: Any | None = None,
        delta_state_mask: Any | None = None,
        phi_pred_state_mask: Any | None = None,
        delta_phi: Any | None = None,
        phi_pred: Any | None = None,
        phi_off: Any | None = None,
        split: str | None = None,
        allow_oracle_context: bool = False,
    ) -> CanonicalStateTaggerOutput:
        tokens, particle_mask = _coerce_tokens_and_mask(tokens_or_batch, mask)
        canonical = self.build_canonical_inputs(tokens, particle_mask)
        if self.mode != STATE_CONTEXT_PARTICLES_ONLY and phi_hlt is None and self.mode != STATE_CONTEXT_ORACLE_PHI_OFF:
            raise ValueError("phi_hlt is required for canonical-state context modes")
        resolved_phi_hlt = None
        if phi_hlt is not None:
            resolved_phi_hlt = self._validate_phi("phi_hlt", phi_hlt, device=canonical.features.device)
        delta_phi, phi_pred, predictor_output = self._predict_state(
            canonical,
            resolved_phi_hlt if resolved_phi_hlt is not None else canonical.features.new_zeros(
                (canonical.batch_size, self.layout.k_state, self.layout.d_phi)
            ),
            state_mask=state_mask,
            delta_phi=delta_phi,
            phi_pred=phi_pred,
        )
        state_values, state_mask, context_ids, state_diag = self._state_values_for_mode(
            canonical=canonical,
            phi_hlt=resolved_phi_hlt,
            delta_phi=delta_phi,
            phi_pred=phi_pred,
            phi_off=phi_off,
            state_mask=state_mask,
            delta_state_mask=delta_state_mask,
            phi_pred_state_mask=phi_pred_state_mask,
            split=split,
            allow_oracle_context=allow_oracle_context,
        )
        state_context = None
        feature_delta_h = None
        feature_gate = None
        feature_diag: dict[str, Any] = {}
        if state_values is not None and state_mask is not None and context_ids is not None:
            state_context = self.state_encoder(state_values, state_mask=state_mask, context_ids=context_ids)
        if self.mode == STATE_CONTEXT_FEATURE_MLP_PLUS_STATE:
            feature_delta_h, feature_gate, feature_diag = self.feature_adapter(canonical.feature_rows(), canonical.particle_mask)
        if self.mode == STATE_CONTEXT_STATE_ONLY:
            if state_context is None or state_mask is None:
                raise ValueError("state-only mode requires encoded state context")
            logits = self._state_only_logits(state_context, state_mask)
        elif self.mode == STATE_CONTEXT_PARTICLES_ONLY:
            logits = self.part_model(canonical.points, canonical.features, canonical.lorentz_vectors, canonical.mask)
        else:
            logits = self._part_forward_with_state(
                canonical,
                state_context,
                state_mask,
                feature_delta_h=feature_delta_h,
                feature_gate=feature_gate,
            )
        logits = _nan_to_num_torch(logits)
        diagnostics: dict[str, Any] = {
            "contract": CANONICAL_STATE_TAGGER_CONTRACT,
            "mode": self.mode,
            "logits_shape": list(logits.shape),
            "uses_reference_part_backbone": bool(self.uses_reference_part_backbone),
            "part_model_class": type(self.part_model).__name__,
            "embed_module_name": str(self.embed_module_name),
            "valid_particle_count": int(canonical.particle_mask.detach().cpu().sum().item()),
            **state_diag,
            "injection": dict(self._last_injection_diagnostics),
        }
        if predictor_output is not None:
            diagnostics["predictor"] = dict(predictor_output.diagnostics)
        if feature_diag:
            diagnostics["feature_mlp_adapter"] = feature_diag
        return CanonicalStateTaggerOutput(
            logits=logits,
            canonical_inputs=canonical,
            config=self.config,
            state_values=state_values,
            state_mask=state_mask,
            state_context=state_context,
            delta_h=self._last_delta_h,
            gate=self._last_gate,
            predictor_output=predictor_output,
            diagnostics=diagnostics,
        )


def build_canonical_state_conditioned_part(
    mode: str = STATE_CONTEXT_PHI_HLT,
    **kwargs: Any,
) -> CanonicalStateConditionedParT:
    config = CanonicalStateTaggerConfig(mode=mode, **kwargs)
    return CanonicalStateConditionedParT(config)
