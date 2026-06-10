"""Global Transformer reconstructor for teacher-logit reconstruction.

This module implements only the Step 3 model path:

``fixed HLT tokens -> bounded soft reconstructed particle view``

It deliberately does not know about teacher losses or training loops yet.  The
output is a differentiable ``SoftReconstructedView`` with corrected HLT-parent
particles plus a small learned set of extra candidate particles.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, Mapping

from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.jetclass_data import JetIdentity, RAW_TOKEN_DIM

from .views import SoftReconstructedView

try:  # Keep imports lightweight on machines without PyTorch.
    import torch as _torch
except ImportError:  # pragma: no cover - environment dependent
    _torch = None

if _torch is None:  # pragma: no cover - environment dependent
    class _ModuleBase:
        pass
else:
    _ModuleBase = _torch.nn.Module


EPS = 1.0e-8
ENERGY_EPS = 1.0e-4
TOKEN_EMBED_FEATURE_DIM = 15


@dataclass
class GlobalTransformerReconstructorConfig:
    """Configuration for the first clean teacher-logit reconstructor."""

    input_dim: int = RAW_TOKEN_DIM
    hidden_dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    num_extra_candidates: int = 32
    dropout: float = 0.05
    max_delta_logpt: float = 0.50
    max_delta_eta: float = 0.25
    max_delta_phi: float = 0.25
    max_delta_loge: float = 0.50
    parent_weight_bias: float = 4.0
    extra_weight_bias: float = -3.0
    max_total_extra_pt_fraction: float = 0.20
    max_extra_delta_eta: float = 1.25
    max_extra_delta_phi: float = 1.25
    eta_limit: float = 5.0
    min_pt: float = 1.0e-4
    energy_eps: float = ENERGY_EPS

    def __post_init__(self) -> None:
        if int(self.input_dim) != RAW_TOKEN_DIM:
            raise ValueError(f"input_dim must be {RAW_TOKEN_DIM}, got {self.input_dim}")
        if int(self.hidden_dim) <= 0:
            raise ValueError("hidden_dim must be positive")
        if int(self.num_layers) <= 0:
            raise ValueError("num_layers must be positive")
        if int(self.num_heads) <= 0:
            raise ValueError("num_heads must be positive")
        if int(self.hidden_dim) % int(self.num_heads) != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if int(self.num_extra_candidates) < 0:
            raise ValueError("num_extra_candidates must be non-negative")
        if float(self.max_total_extra_pt_fraction) < 0.0:
            raise ValueError("max_total_extra_pt_fraction must be non-negative")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | "GlobalTransformerReconstructorConfig" | None):
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        return cls(**dict(value))


def wrap_phi_torch(phi):
    torch = require_torch()
    return torch.remainder(phi + torch.pi, 2.0 * torch.pi) - torch.pi


def _nan_to_num_torch(value, *, nan: float = 0.0, posinf: float = 0.0, neginf: float = 0.0):
    torch = require_torch()
    if hasattr(torch, "nan_to_num"):
        return torch.nan_to_num(value, nan=float(nan), posinf=float(posinf), neginf=float(neginf))
    return torch.where(torch.isfinite(value), value, torch.zeros_like(value) + float(nan))


def physical_energy_floor(pt, eta, *, eps: float = ENERGY_EPS):
    torch = require_torch()
    return torch.clamp(pt, min=0.0) * torch.cosh(torch.clamp(eta, -5.0, 5.0)) + float(eps)


def _masked_mean(values, mask):
    torch = require_torch()
    weights = mask.float()
    denom = torch.clamp(weights.sum(dim=1, keepdim=True), min=1.0)
    return (values * weights[:, :, None]).sum(dim=1) / denom


def sanitize_hlt_tokens(tokens, mask, *, config: GlobalTransformerReconstructorConfig):
    """Return finite physical HLT tokens and a non-empty mask."""

    torch = require_torch()
    tokens = tokens.float()
    mask = mask.bool()
    finite = torch.isfinite(tokens).all(dim=-1)
    cleaned = _nan_to_num_torch(tokens).clone()
    safe_mask = mask & finite

    pt = torch.clamp(cleaned[:, :, 0], min=float(config.min_pt))
    eta = torch.clamp(cleaned[:, :, 1], -float(config.eta_limit), float(config.eta_limit))
    phi = wrap_phi_torch(cleaned[:, :, 2])
    energy = torch.maximum(
        torch.clamp(cleaned[:, :, 3], min=float(config.energy_eps)),
        physical_energy_floor(pt, eta, eps=float(config.energy_eps)),
    )
    cleaned[:, :, 0] = pt
    cleaned[:, :, 1] = eta
    cleaned[:, :, 2] = phi
    cleaned[:, :, 3] = energy
    cleaned = cleaned * safe_mask[:, :, None].float()

    empty = safe_mask.sum(dim=1) == 0
    if bool(empty.any()):
        fallback = torch.zeros_like(cleaned)
        fallback[:, 0, 0] = float(config.min_pt)
        fallback[:, 0, 1] = 0.0
        fallback[:, 0, 2] = 0.0
        fallback[:, 0, 3] = float(config.min_pt) + float(config.energy_eps)
        fallback_mask = torch.zeros_like(safe_mask)
        fallback_mask[:, 0] = True
        cleaned = torch.where(empty[:, None, None], fallback, cleaned)
        safe_mask = torch.where(empty[:, None], fallback_mask, safe_mask)

    diagnostics = {
        "nonfinite_input_token_count": int((~finite & mask).sum().detach().cpu().item()),
        "empty_input_jet_count": int(empty.sum().detach().cpu().item()),
    }
    return cleaned, safe_mask, diagnostics


def token_embedding_features(tokens, mask):
    """Build stable per-particle features for the reconstructor encoder."""

    torch = require_torch()
    pt = torch.clamp(tokens[:, :, 0], min=EPS)
    eta = torch.clamp(tokens[:, :, 1], -5.0, 5.0)
    phi = wrap_phi_torch(tokens[:, :, 2])
    energy = torch.clamp(tokens[:, :, 3], min=EPS)
    pieces = [
        0.2 * torch.log(pt),
        0.2 * torch.log(energy),
        eta / 5.0,
        torch.sin(phi),
        torch.cos(phi),
        torch.clamp(tokens[:, :, 4], -1.0, 1.0),
        torch.clamp(tokens[:, :, 5], 0.0, 1.0),
        torch.clamp(tokens[:, :, 6], 0.0, 1.0),
        torch.clamp(tokens[:, :, 7], 0.0, 1.0),
        torch.clamp(tokens[:, :, 8], 0.0, 1.0),
        torch.clamp(tokens[:, :, 9], 0.0, 1.0),
        torch.tanh(tokens[:, :, 10]),
        torch.clamp(tokens[:, :, 11], 0.0, 1.0),
        torch.tanh(tokens[:, :, 12]),
        torch.clamp(tokens[:, :, 13], 0.0, 1.0),
    ]
    features = torch.stack(pieces, dim=-1)
    return torch.where(mask[:, :, None], features, torch.zeros_like(features))


def jet_axes_from_tokens(tokens, mask):
    """Compute differentiable jet pT/eta/phi/energy from a token view."""

    torch = require_torch()
    pt = torch.where(mask, torch.clamp(tokens[:, :, 0], min=0.0), torch.zeros_like(tokens[:, :, 0]))
    eta = torch.where(mask, tokens[:, :, 1], torch.zeros_like(tokens[:, :, 1]))
    phi = torch.where(mask, tokens[:, :, 2], torch.zeros_like(tokens[:, :, 2]))
    energy = torch.where(mask, torch.clamp(tokens[:, :, 3], min=0.0), torch.zeros_like(tokens[:, :, 3]))
    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    pz = pt * torch.sinh(torch.clamp(eta, -5.0, 5.0))
    jet_px = px.sum(dim=1)
    jet_py = py.sum(dim=1)
    jet_pz = pz.sum(dim=1)
    jet_energy = energy.sum(dim=1)
    jet_pt = torch.sqrt(torch.clamp(jet_px * jet_px + jet_py * jet_py, min=0.0))
    jet_phi = torch.atan2(jet_py, jet_px)
    jet_eta = torch.asinh(jet_pz / torch.clamp(jet_pt, min=EPS))
    jet_eta = torch.where(jet_pt > EPS, jet_eta, torch.zeros_like(jet_eta))
    jet_phi = torch.where(jet_pt > EPS, jet_phi, torch.zeros_like(jet_phi))
    return {
        "pt": jet_pt,
        "eta": jet_eta,
        "phi": jet_phi,
        "energy": jet_energy,
    }


def placeholder_jet_ids(batch_size: int, labels: Any | None = None) -> list[JetIdentity]:
    if labels is not None:
        try:
            label_values = labels.detach().cpu().tolist()
        except AttributeError:
            label_values = list(labels)
    else:
        label_values = [-1] * int(batch_size)
    return [
        JetIdentity(file="in_memory_teacher_logit_reco", entry=index, label=int(label_values[index]))
        for index in range(int(batch_size))
    ]


class GlobalTransformerReconstructor(_ModuleBase):
    """ParT-like global reconstructor with bounded parent edits and extra slots."""

    def __init__(self, config: Mapping[str, Any] | GlobalTransformerReconstructorConfig | None = None) -> None:
        torch = require_torch()
        super().__init__()
        self.config = GlobalTransformerReconstructorConfig.from_mapping(config)

        self.input_proj = torch.nn.Sequential(
            torch.nn.LayerNorm(TOKEN_EMBED_FEATURE_DIM),
            torch.nn.Linear(TOKEN_EMBED_FEATURE_DIM, self.config.hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(float(self.config.dropout)),
        )
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=int(self.config.hidden_dim),
            nhead=int(self.config.num_heads),
            dim_feedforward=int(self.config.hidden_dim) * 4,
            dropout=float(self.config.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(self.config.num_layers),
            norm=torch.nn.LayerNorm(int(self.config.hidden_dim)),
        )
        self.parent_head = torch.nn.Sequential(
            torch.nn.LayerNorm(int(self.config.hidden_dim)),
            torch.nn.Linear(int(self.config.hidden_dim), int(self.config.hidden_dim)),
            torch.nn.GELU(),
            torch.nn.Linear(int(self.config.hidden_dim), 5),
        )

        self.num_extra_candidates = int(self.config.num_extra_candidates)
        if self.num_extra_candidates > 0:
            self.extra_queries = torch.nn.Parameter(
                torch.empty(self.num_extra_candidates, int(self.config.hidden_dim))
            )
            torch.nn.init.normal_(self.extra_queries, mean=0.0, std=0.02)
            self.extra_cross_attention = torch.nn.MultiheadAttention(
                embed_dim=int(self.config.hidden_dim),
                num_heads=int(self.config.num_heads),
                dropout=float(self.config.dropout),
                batch_first=True,
            )
            self.extra_norm = torch.nn.LayerNorm(int(self.config.hidden_dim))
            self.extra_head = torch.nn.Sequential(
                torch.nn.Linear(int(self.config.hidden_dim), int(self.config.hidden_dim)),
                torch.nn.GELU(),
                torch.nn.Linear(int(self.config.hidden_dim), RAW_TOKEN_DIM + 1),
            )
        else:
            self.register_parameter("extra_queries", None)
            self.extra_cross_attention = None
            self.extra_norm = None
            self.extra_head = None

    def _bounded_parent_corrections(self, raw):
        torch = require_torch()
        deltas = torch.stack(
            [
                float(self.config.max_delta_logpt) * torch.tanh(raw[:, :, 0]),
                float(self.config.max_delta_eta) * torch.tanh(raw[:, :, 1]),
                float(self.config.max_delta_phi) * torch.tanh(raw[:, :, 2]),
                float(self.config.max_delta_loge) * torch.tanh(raw[:, :, 3]),
            ],
            dim=-1,
        )
        parent_weight = torch.sigmoid(raw[:, :, 4] + float(self.config.parent_weight_bias))
        return deltas, parent_weight

    def _apply_parent_corrections(self, tokens, mask, deltas):
        torch = require_torch()
        out = tokens.clone()
        pt = torch.clamp(tokens[:, :, 0], min=float(self.config.min_pt)) * torch.exp(deltas[:, :, 0])
        eta = torch.clamp(tokens[:, :, 1] + deltas[:, :, 1], -float(self.config.eta_limit), float(self.config.eta_limit))
        phi = wrap_phi_torch(tokens[:, :, 2] + deltas[:, :, 2])
        energy = torch.clamp(tokens[:, :, 3], min=float(self.config.energy_eps)) * torch.exp(deltas[:, :, 3])
        energy = torch.maximum(energy, physical_energy_floor(pt, eta, eps=float(self.config.energy_eps)))
        out[:, :, 0] = pt
        out[:, :, 1] = eta
        out[:, :, 2] = phi
        out[:, :, 3] = energy
        return torch.where(mask[:, :, None], out, torch.zeros_like(out))

    def _make_extra_candidates(self, encoded, mask, jet_axes):
        torch = require_torch()
        batch_size = int(encoded.shape[0])
        if self.num_extra_candidates == 0:
            empty_tokens = encoded.new_zeros(batch_size, 0, RAW_TOKEN_DIM)
            empty_weights = encoded.new_zeros(batch_size, 0)
            empty_mask = torch.zeros(batch_size, 0, dtype=torch.bool, device=encoded.device)
            return empty_tokens, empty_weights, empty_mask

        queries = self.extra_queries[None, :, :].expand(batch_size, -1, -1)
        attended, _ = self.extra_cross_attention(
            query=queries,
            key=encoded,
            value=encoded,
            key_padding_mask=~mask.bool(),
            need_weights=False,
        )
        states = self.extra_norm(queries + attended)
        raw = self.extra_head(states)

        per_slot_pt_fraction = float(self.config.max_total_extra_pt_fraction) / max(self.num_extra_candidates, 1)
        pt = torch.clamp(jet_axes["pt"][:, None], min=float(self.config.min_pt)) * (
            per_slot_pt_fraction * torch.sigmoid(raw[:, :, 0])
        )
        pt = torch.clamp(pt, min=float(self.config.min_pt))
        eta = torch.clamp(
            jet_axes["eta"][:, None] + float(self.config.max_extra_delta_eta) * torch.tanh(raw[:, :, 1]),
            -float(self.config.eta_limit),
            float(self.config.eta_limit),
        )
        phi = wrap_phi_torch(jet_axes["phi"][:, None] + float(self.config.max_extra_delta_phi) * torch.tanh(raw[:, :, 2]))
        energy_scale = torch.exp(float(self.config.max_delta_loge) * torch.tanh(raw[:, :, 3]))
        energy = physical_energy_floor(pt, eta, eps=float(self.config.energy_eps)) * energy_scale
        energy = torch.maximum(energy, physical_energy_floor(pt, eta, eps=float(self.config.energy_eps)))

        tokens = raw.new_zeros(batch_size, self.num_extra_candidates, RAW_TOKEN_DIM)
        tokens[:, :, 0] = pt
        tokens[:, :, 1] = eta
        tokens[:, :, 2] = phi
        tokens[:, :, 3] = energy
        tokens[:, :, 4] = torch.tanh(raw[:, :, 4])
        tokens[:, :, 5:10] = torch.softmax(raw[:, :, 5:10], dim=-1)
        tokens[:, :, 10] = torch.tanh(raw[:, :, 10])
        tokens[:, :, 11] = torch.sigmoid(raw[:, :, 11])
        tokens[:, :, 12] = torch.tanh(raw[:, :, 12])
        tokens[:, :, 13] = torch.sigmoid(raw[:, :, 13])
        weights = torch.sigmoid(raw[:, :, 14] + float(self.config.extra_weight_bias))
        extra_mask = torch.ones(batch_size, self.num_extra_candidates, dtype=torch.bool, device=encoded.device)
        return tokens, weights, extra_mask

    def forward(
        self,
        hlt_tokens,
        hlt_mask,
        *,
        labels=None,
        jet_ids: list[JetIdentity] | None = None,
        split: str = "in_memory",
    ) -> SoftReconstructedView:
        torch = require_torch()
        hlt_tokens, hlt_mask, diagnostics = sanitize_hlt_tokens(hlt_tokens, hlt_mask, config=self.config)
        features = token_embedding_features(hlt_tokens, hlt_mask)
        encoded = self.input_proj(features)
        encoded = self.encoder(encoded, src_key_padding_mask=~hlt_mask.bool())
        encoded = torch.where(hlt_mask[:, :, None], encoded, torch.zeros_like(encoded))

        parent_raw = self.parent_head(encoded)
        parent_delta, parent_weights = self._bounded_parent_corrections(parent_raw)
        parent_tokens = self._apply_parent_corrections(hlt_tokens, hlt_mask, parent_delta)
        parent_weights = torch.where(hlt_mask, parent_weights, torch.zeros_like(parent_weights))

        jet_axes = jet_axes_from_tokens(hlt_tokens, hlt_mask)
        extra_tokens, extra_weights, extra_mask = self._make_extra_candidates(encoded, hlt_mask, jet_axes)
        tokens = torch.cat([parent_tokens, extra_tokens], dim=1)
        mask = torch.cat([hlt_mask, extra_mask], dim=1)
        weights = torch.cat([parent_weights, extra_weights], dim=1)

        batch_size = int(tokens.shape[0])
        if labels is None:
            labels = torch.full((batch_size,), -1, dtype=torch.long, device=tokens.device)
        else:
            labels = labels.to(device=tokens.device, dtype=torch.long) if isinstance(labels, torch.Tensor) else labels
        if jet_ids is None:
            jet_ids = placeholder_jet_ids(batch_size, labels=labels)

        aux = {
            "sanitized_hlt_tokens": hlt_tokens,
            "sanitized_hlt_mask": hlt_mask,
            "parent_tokens": parent_tokens,
            "parent_delta": parent_delta,
            "parent_weights": parent_weights,
            "extra_tokens": extra_tokens,
            "extra_weights": extra_weights,
            "extra_mask": extra_mask,
            "jet_axes": jet_axes,
            "diagnostics": diagnostics,
        }
        return SoftReconstructedView(
            tokens=tokens,
            mask=mask,
            weights=weights,
            labels=labels,
            jet_ids=jet_ids,
            split=split,
            metadata={
                "construction": "global_transformer_parents_plus_extras",
                "model_family": "teacher_logit_global_transformer",
                "n_parent_candidates": int(parent_tokens.shape[1]),
                "n_extra_candidates": int(extra_tokens.shape[1]),
                "config": self.config.to_dict(),
                "diagnostics": diagnostics,
            },
            aux=aux,
        )


def build_global_transformer_reconstructor(
    config: Mapping[str, Any] | GlobalTransformerReconstructorConfig | None = None,
) -> GlobalTransformerReconstructor:
    return GlobalTransformerReconstructor(config)
