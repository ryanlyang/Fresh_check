"""Shared aggressive soft-view head for teacher-logit reconstructors.

The aggressive branch keeps architecture-specific encoders separate from the
output mechanics.  Future transformer, ParticleNet, PFN, and P-CNN
reconstructors should produce per-particle embeddings and one global context
vector, then call :class:`AggressiveSoftViewHead` to build the reconstructed
soft view.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping

from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.jetclass_data import JetIdentity, RAW_TOKEN_DIM

from .global_transformer import (
    ENERGY_EPS,
    jet_axes_from_tokens,
    physical_energy_floor,
    placeholder_jet_ids,
    sanitize_hlt_tokens,
    sanitize_reconstructed_view_tensors,
    wrap_phi_torch,
)
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


AGGRESSION_LEVEL = "aggressive_v1"


@dataclass
class AggressiveSoftViewConfig:
    """Configuration for the shared aggressive edit/generate head."""

    input_dim: int = RAW_TOKEN_DIM
    embedding_dim: int = 128
    num_extra_candidates: int = 64
    dropout: float = 0.05
    max_delta_logpt: float = 1.0
    max_delta_eta: float = 0.20
    max_delta_phi: float = 0.20
    max_delta_loge: float = 1.0
    parent_weight_bias: float = 2.0
    extra_weight_bias: float = -2.0
    max_total_extra_pt_fraction: float = 0.50
    max_extra_delta_eta: float = 1.50
    max_extra_delta_phi: float = 1.50
    max_global_logpt_scale: float = 0.35
    max_global_loge_scale: float = 0.35
    max_global_eta_shift: float = 0.05
    max_global_phi_shift: float = 0.05
    extra_usage_weight_threshold: float = 0.05
    eta_limit: float = 5.0
    min_pt: float = 1.0e-4
    energy_eps: float = ENERGY_EPS
    aggression_level: str = AGGRESSION_LEVEL

    def __post_init__(self) -> None:
        if int(self.input_dim) != RAW_TOKEN_DIM:
            raise ValueError(f"input_dim must be {RAW_TOKEN_DIM}, got {self.input_dim}")
        if int(self.embedding_dim) <= 0:
            raise ValueError("embedding_dim must be positive")
        if int(self.num_extra_candidates) < 0:
            raise ValueError("num_extra_candidates must be non-negative")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        for name in (
            "max_delta_logpt",
            "max_delta_eta",
            "max_delta_phi",
            "max_delta_loge",
            "max_total_extra_pt_fraction",
            "max_extra_delta_eta",
            "max_extra_delta_phi",
            "max_global_logpt_scale",
            "max_global_loge_scale",
            "max_global_eta_shift",
            "max_global_phi_shift",
            "extra_usage_weight_threshold",
            "eta_limit",
            "min_pt",
            "energy_eps",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if float(self.extra_usage_weight_threshold) > 1.0:
            raise ValueError("extra_usage_weight_threshold must be in [0, 1]")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | "AggressiveSoftViewConfig" | None):
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        return cls(**dict(value))


@dataclass
class AggressiveSoftViewDiagnostics:
    """Scalar diagnostics summarizing how the aggressive head spent freedom."""

    parent_delta_logpt_abs_mean: float
    parent_delta_eta_abs_mean: float
    parent_delta_phi_abs_mean: float
    parent_delta_loge_abs_mean: float
    parent_weight_mean: float
    parent_weight_min: float
    parent_weight_max: float
    parent_weight_sum_mean: float
    extra_weight_mean: float
    extra_weight_min: float
    extra_weight_max: float
    extra_weight_sum_mean: float
    extra_weight_sum_min: float
    extra_weight_sum_max: float
    extra_pt_fraction_mean: float
    extra_pt_fraction_min: float
    extra_pt_fraction_max: float
    extra_slot_usage_mean: float
    extra_slot_usage_min: float
    extra_slot_usage_max: float
    extra_slot_usage_histogram: list[int]
    global_logpt_scale_mean: float
    global_loge_scale_mean: float
    global_eta_shift_mean: float
    global_phi_shift_mean: float
    global_logpt_scale_abs_mean: float
    global_loge_scale_abs_mean: float
    global_eta_shift_abs_mean: float
    global_phi_shift_abs_mean: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _masked_mean_scalar(values, mask) -> float:
    torch = require_torch()
    if int(values.numel()) == 0:
        return 0.0
    selected = values[mask] if tuple(values.shape) == tuple(mask.shape) else values
    if int(selected.numel()) == 0:
        return 0.0
    return float(torch.mean(selected.float()).detach().cpu().item())


def _safe_batch_mean(values) -> float:
    if int(values.numel()) == 0:
        return 0.0
    return float(values.float().mean().detach().cpu().item())


def _safe_batch_min(values) -> float:
    if int(values.numel()) == 0:
        return 0.0
    return float(values.float().min().detach().cpu().item())


def _safe_batch_max(values) -> float:
    if int(values.numel()) == 0:
        return 0.0
    return float(values.float().max().detach().cpu().item())


def _masked_min_scalar(values, mask) -> float:
    if int(values.numel()) == 0:
        return 0.0
    selected = values[mask] if tuple(values.shape) == tuple(mask.shape) else values
    if int(selected.numel()) == 0:
        return 0.0
    return float(selected.float().min().detach().cpu().item())


def _masked_max_scalar(values, mask) -> float:
    if int(values.numel()) == 0:
        return 0.0
    selected = values[mask] if tuple(values.shape) == tuple(mask.shape) else values
    if int(selected.numel()) == 0:
        return 0.0
    return float(selected.float().max().detach().cpu().item())


def _masked_delta_component_abs_mean(parent_delta, parent_mask, component: int) -> float:
    if int(parent_delta.numel()) == 0:
        return 0.0
    values = parent_delta[:, :, int(component)].abs()
    return _masked_mean_scalar(values, parent_mask)


def _slot_token_from_raw(raw, *, base_pt, base_eta, base_phi, config: AggressiveSoftViewConfig):
    torch = require_torch()
    pt = torch.clamp(base_pt, min=float(config.min_pt)) * torch.sigmoid(raw[:, :, 0])
    pt = torch.clamp(pt, min=float(config.min_pt))
    eta = torch.clamp(
        base_eta + float(config.max_extra_delta_eta) * torch.tanh(raw[:, :, 1]),
        -float(config.eta_limit),
        float(config.eta_limit),
    )
    phi = wrap_phi_torch(base_phi + float(config.max_extra_delta_phi) * torch.tanh(raw[:, :, 2]))
    energy = physical_energy_floor(pt, eta, eps=float(config.energy_eps)) * torch.exp(
        float(config.max_delta_loge) * torch.tanh(raw[:, :, 3])
    )
    energy = torch.maximum(energy, physical_energy_floor(pt, eta, eps=float(config.energy_eps)))
    return torch.cat(
        [
            pt[:, :, None],
            eta[:, :, None],
            phi[:, :, None],
            energy[:, :, None],
            torch.tanh(raw[:, :, 4])[:, :, None],
            torch.softmax(raw[:, :, 5:10], dim=-1),
            torch.tanh(raw[:, :, 10])[:, :, None],
            torch.sigmoid(raw[:, :, 11])[:, :, None],
            torch.tanh(raw[:, :, 12])[:, :, None],
            torch.sigmoid(raw[:, :, 13])[:, :, None],
        ],
        dim=-1,
    )


class AggressiveSoftViewHead(_ModuleBase):
    """Shared aggressive parent-edit, extra-slot, and global-calibration head."""

    def __init__(self, config: Mapping[str, Any] | AggressiveSoftViewConfig | None = None) -> None:
        torch = require_torch()
        super().__init__()
        self.config = AggressiveSoftViewConfig.from_mapping(config)
        dim = int(self.config.embedding_dim)
        self.parent_head = torch.nn.Sequential(
            torch.nn.LayerNorm(dim),
            torch.nn.Linear(dim, dim),
            torch.nn.GELU(),
            torch.nn.Dropout(float(self.config.dropout)),
            torch.nn.Linear(dim, 5),
        )
        self.global_head = torch.nn.Sequential(
            torch.nn.LayerNorm(dim),
            torch.nn.Linear(dim, dim),
            torch.nn.GELU(),
            torch.nn.Dropout(float(self.config.dropout)),
            torch.nn.Linear(dim, 4),
        )
        self.num_extra_candidates = int(self.config.num_extra_candidates)
        if self.num_extra_candidates > 0:
            self.extra_slots = torch.nn.Parameter(torch.empty(self.num_extra_candidates, dim))
            torch.nn.init.normal_(self.extra_slots, mean=0.0, std=0.02)
            self.extra_head = torch.nn.Sequential(
                torch.nn.LayerNorm(dim * 2),
                torch.nn.Linear(dim * 2, dim),
                torch.nn.GELU(),
                torch.nn.Dropout(float(self.config.dropout)),
                torch.nn.Linear(dim, RAW_TOKEN_DIM + 1),
            )
        else:
            self.register_parameter("extra_slots", None)
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
        weights = torch.sigmoid(raw[:, :, 4] + float(self.config.parent_weight_bias))
        return deltas, weights

    def _bounded_global_correction(self, raw):
        torch = require_torch()
        return {
            "logpt_scale": float(self.config.max_global_logpt_scale) * torch.tanh(raw[:, 0]),
            "loge_scale": float(self.config.max_global_loge_scale) * torch.tanh(raw[:, 1]),
            "eta_shift": float(self.config.max_global_eta_shift) * torch.tanh(raw[:, 2]),
            "phi_shift": float(self.config.max_global_phi_shift) * torch.tanh(raw[:, 3]),
        }

    def _apply_parent_corrections(self, tokens, mask, deltas, global_correction):
        torch = require_torch()
        pt = torch.clamp(tokens[:, :, 0], min=float(self.config.min_pt)) * torch.exp(deltas[:, :, 0])
        eta = torch.clamp(
            tokens[:, :, 1] + deltas[:, :, 1],
            -float(self.config.eta_limit),
            float(self.config.eta_limit),
        )
        phi = wrap_phi_torch(tokens[:, :, 2] + deltas[:, :, 2])
        energy = torch.clamp(tokens[:, :, 3], min=float(self.config.energy_eps)) * torch.exp(deltas[:, :, 3])

        pt = pt * torch.exp(global_correction["logpt_scale"][:, None])
        eta = torch.clamp(
            eta + global_correction["eta_shift"][:, None],
            -float(self.config.eta_limit),
            float(self.config.eta_limit),
        )
        phi = wrap_phi_torch(phi + global_correction["phi_shift"][:, None])
        energy = energy * torch.exp(global_correction["loge_scale"][:, None])
        energy = torch.maximum(energy, physical_energy_floor(pt, eta, eps=float(self.config.energy_eps)))
        out = torch.cat(
            [
                pt[:, :, None],
                eta[:, :, None],
                phi[:, :, None],
                energy[:, :, None],
                tokens[:, :, 4:],
            ],
            dim=-1,
        )
        return torch.where(mask[:, :, None], out, torch.zeros_like(out))

    def _make_extra_candidates(self, global_context, jet_axes, global_correction):
        torch = require_torch()
        batch_size = int(global_context.shape[0])
        if self.num_extra_candidates == 0:
            empty_tokens = global_context.new_zeros(batch_size, 0, RAW_TOKEN_DIM)
            empty_weights = global_context.new_zeros(batch_size, 0)
            empty_mask = torch.zeros(batch_size, 0, dtype=torch.bool, device=global_context.device)
            empty_raw = global_context.new_zeros(batch_size, 0, RAW_TOKEN_DIM + 1)
            return empty_tokens, empty_weights, empty_mask, empty_raw

        slots = self.extra_slots[None, :, :].expand(batch_size, -1, -1)
        context = global_context[:, None, :].expand(-1, self.num_extra_candidates, -1)
        raw = self.extra_head(torch.cat([context, slots], dim=-1))
        per_slot_pt_fraction = float(self.config.max_total_extra_pt_fraction) / max(self.num_extra_candidates, 1)
        base_pt = torch.clamp(jet_axes["pt"][:, None], min=float(self.config.min_pt)) * per_slot_pt_fraction
        base_eta = jet_axes["eta"][:, None]
        base_phi = jet_axes["phi"][:, None]
        tokens = _slot_token_from_raw(raw, base_pt=base_pt, base_eta=base_eta, base_phi=base_phi, config=self.config)

        pt = tokens[:, :, 0] * torch.exp(global_correction["logpt_scale"][:, None])
        eta = torch.clamp(
            tokens[:, :, 1] + global_correction["eta_shift"][:, None],
            -float(self.config.eta_limit),
            float(self.config.eta_limit),
        )
        phi = wrap_phi_torch(tokens[:, :, 2] + global_correction["phi_shift"][:, None])
        energy = tokens[:, :, 3] * torch.exp(global_correction["loge_scale"][:, None])
        energy = torch.maximum(energy, physical_energy_floor(pt, eta, eps=float(self.config.energy_eps)))
        tokens = torch.cat([pt[:, :, None], eta[:, :, None], phi[:, :, None], energy[:, :, None], tokens[:, :, 4:]], dim=-1)
        weights = torch.sigmoid(raw[:, :, RAW_TOKEN_DIM] + float(self.config.extra_weight_bias))
        extra_mask = torch.ones(batch_size, self.num_extra_candidates, dtype=torch.bool, device=global_context.device)
        return tokens, weights, extra_mask, raw

    def _extra_budget_tensors(self, *, extra_tokens, extra_weights, extra_mask, hlt_tokens, hlt_mask) -> Dict[str, Any]:
        torch = require_torch()
        batch_size = int(hlt_tokens.shape[0])
        n_extra = int(extra_weights.shape[1])
        parent_pt_sum = (torch.clamp(hlt_tokens[:, :, 0], min=0.0) * hlt_mask.float()).sum(dim=1)
        if n_extra == 0:
            extra_weight_sum = parent_pt_sum.new_zeros(parent_pt_sum.shape)
            extra_pt_fraction = parent_pt_sum.new_zeros(parent_pt_sum.shape)
            extra_slot_active_mask = torch.zeros(batch_size, 0, dtype=torch.bool, device=hlt_tokens.device)
            extra_slot_usage = parent_pt_sum.new_zeros(parent_pt_sum.shape)
            extra_slot_usage_histogram = [batch_size]
        else:
            bounded_weights = torch.where(extra_mask, torch.clamp(extra_weights, min=0.0, max=1.0), torch.zeros_like(extra_weights))
            extra_weight_sum = bounded_weights.sum(dim=1)
            extra_pt = (torch.clamp(extra_tokens[:, :, 0], min=0.0) * bounded_weights).sum(dim=1)
            extra_pt_fraction = extra_pt / torch.clamp(parent_pt_sum, min=float(self.config.min_pt))
            extra_slot_active_mask = extra_mask & (bounded_weights >= float(self.config.extra_usage_weight_threshold))
            extra_slot_usage = extra_slot_active_mask.sum(dim=1).float()
            usage_cpu = extra_slot_usage.to(dtype=torch.long).detach().cpu()
            extra_slot_usage_histogram = [int(v) for v in torch.bincount(usage_cpu, minlength=n_extra + 1).tolist()]
        return {
            "extra_weight_sum": extra_weight_sum,
            "extra_pt_fraction": extra_pt_fraction,
            "extra_slot_active_mask": extra_slot_active_mask,
            "extra_slot_usage": extra_slot_usage,
            "extra_slot_usage_histogram": extra_slot_usage_histogram,
        }

    def _diagnostics(
        self,
        *,
        parent_weights,
        parent_mask,
        parent_delta,
        extra_tokens,
        extra_weights,
        extra_mask,
        extra_budget,
        hlt_tokens,
        hlt_mask,
        global_correction,
    ) -> AggressiveSoftViewDiagnostics:
        extra_weight_sum = extra_budget["extra_weight_sum"]
        extra_pt_fraction = extra_budget["extra_pt_fraction"]
        extra_slot_usage = extra_budget["extra_slot_usage"]
        return AggressiveSoftViewDiagnostics(
            parent_delta_logpt_abs_mean=_masked_delta_component_abs_mean(parent_delta, parent_mask, 0),
            parent_delta_eta_abs_mean=_masked_delta_component_abs_mean(parent_delta, parent_mask, 1),
            parent_delta_phi_abs_mean=_masked_delta_component_abs_mean(parent_delta, parent_mask, 2),
            parent_delta_loge_abs_mean=_masked_delta_component_abs_mean(parent_delta, parent_mask, 3),
            parent_weight_mean=_masked_mean_scalar(parent_weights, parent_mask),
            parent_weight_min=_masked_min_scalar(parent_weights, parent_mask),
            parent_weight_max=_masked_max_scalar(parent_weights, parent_mask),
            parent_weight_sum_mean=_safe_batch_mean((parent_weights * parent_mask.float()).sum(dim=1)),
            extra_weight_mean=_masked_mean_scalar(extra_weights, extra_mask),
            extra_weight_min=_masked_min_scalar(extra_weights, extra_mask),
            extra_weight_max=_masked_max_scalar(extra_weights, extra_mask),
            extra_weight_sum_mean=_safe_batch_mean(extra_weight_sum),
            extra_weight_sum_min=_safe_batch_min(extra_weight_sum),
            extra_weight_sum_max=_safe_batch_max(extra_weight_sum),
            extra_pt_fraction_mean=_safe_batch_mean(extra_pt_fraction),
            extra_pt_fraction_min=_safe_batch_min(extra_pt_fraction),
            extra_pt_fraction_max=_safe_batch_max(extra_pt_fraction),
            extra_slot_usage_mean=_safe_batch_mean(extra_slot_usage),
            extra_slot_usage_min=_safe_batch_min(extra_slot_usage),
            extra_slot_usage_max=_safe_batch_max(extra_slot_usage),
            extra_slot_usage_histogram=extra_budget["extra_slot_usage_histogram"],
            global_logpt_scale_mean=_safe_batch_mean(global_correction["logpt_scale"]),
            global_loge_scale_mean=_safe_batch_mean(global_correction["loge_scale"]),
            global_eta_shift_mean=_safe_batch_mean(global_correction["eta_shift"]),
            global_phi_shift_mean=_safe_batch_mean(global_correction["phi_shift"]),
            global_logpt_scale_abs_mean=_safe_batch_mean(global_correction["logpt_scale"].abs()),
            global_loge_scale_abs_mean=_safe_batch_mean(global_correction["loge_scale"].abs()),
            global_eta_shift_abs_mean=_safe_batch_mean(global_correction["eta_shift"].abs()),
            global_phi_shift_abs_mean=_safe_batch_mean(global_correction["phi_shift"].abs()),
        )

    def forward(
        self,
        hlt_tokens,
        hlt_mask,
        particle_embeddings,
        global_context,
        *,
        labels=None,
        jet_ids: list[JetIdentity] | None = None,
        split: str = "in_memory",
        metadata: Mapping[str, Any] | None = None,
    ) -> SoftReconstructedView:
        torch = require_torch()
        if tuple(particle_embeddings.shape[:2]) != tuple(hlt_tokens.shape[:2]):
            raise ValueError("particle_embeddings must have shape [B, N, D] matching hlt_tokens [B, N, RAW_TOKEN_DIM]")
        if tuple(global_context.shape) != (int(hlt_tokens.shape[0]), int(self.config.embedding_dim)):
            raise ValueError(
                f"global_context must have shape [B, {self.config.embedding_dim}], got {tuple(global_context.shape)}"
            )
        if int(particle_embeddings.shape[-1]) != int(self.config.embedding_dim):
            raise ValueError(
                f"particle_embeddings last dim must be {self.config.embedding_dim}, got {particle_embeddings.shape[-1]}"
            )

        hlt_tokens, hlt_mask, diagnostics = sanitize_hlt_tokens(hlt_tokens, hlt_mask, config=self.config)
        particle_embeddings = torch.where(hlt_mask[:, :, None], particle_embeddings.float(), torch.zeros_like(particle_embeddings.float()))
        global_context = global_context.float()

        parent_raw = self.parent_head(particle_embeddings)
        parent_delta, parent_weights = self._bounded_parent_corrections(parent_raw)
        parent_weights = torch.where(hlt_mask, parent_weights, torch.zeros_like(parent_weights))
        global_raw = self.global_head(global_context)
        global_correction = self._bounded_global_correction(global_raw)
        parent_tokens = self._apply_parent_corrections(hlt_tokens, hlt_mask, parent_delta, global_correction)

        jet_axes = jet_axes_from_tokens(hlt_tokens, hlt_mask)
        extra_tokens, extra_weights, extra_mask, extra_raw = self._make_extra_candidates(global_context, jet_axes, global_correction)
        tokens = torch.cat([parent_tokens, extra_tokens], dim=1)
        mask = torch.cat([hlt_mask, extra_mask], dim=1)
        weights = torch.cat([parent_weights, extra_weights], dim=1)
        tokens, mask, weights, reco_diagnostics = sanitize_reconstructed_view_tensors(tokens, mask, weights, config=self.config)
        diagnostics = {**diagnostics, **reco_diagnostics}

        n_parent_candidates = int(parent_tokens.shape[1])
        parent_tokens = tokens[:, :n_parent_candidates, :]
        parent_weights = weights[:, :n_parent_candidates]
        extra_tokens = tokens[:, n_parent_candidates:, :]
        extra_weights = weights[:, n_parent_candidates:]
        extra_mask = mask[:, n_parent_candidates:]
        extra_budget = self._extra_budget_tensors(
            extra_tokens=extra_tokens,
            extra_weights=extra_weights,
            extra_mask=extra_mask,
            hlt_tokens=hlt_tokens,
            hlt_mask=hlt_mask,
        )
        scalar_diagnostics = self._diagnostics(
            parent_weights=parent_weights,
            parent_mask=hlt_mask,
            parent_delta=parent_delta,
            extra_tokens=extra_tokens,
            extra_weights=extra_weights,
            extra_mask=extra_mask,
            extra_budget=extra_budget,
            hlt_tokens=hlt_tokens,
            hlt_mask=hlt_mask,
            global_correction=global_correction,
        )
        diagnostics = {**diagnostics, **scalar_diagnostics.to_dict()}

        batch_size = int(tokens.shape[0])
        if labels is None:
            labels = torch.full((batch_size,), -1, dtype=torch.long, device=tokens.device)
        elif isinstance(labels, torch.Tensor):
            labels = labels.to(device=tokens.device, dtype=torch.long)
        if jet_ids is None:
            jet_ids = placeholder_jet_ids(batch_size, labels=labels)

        aux = {
            "sanitized_hlt_tokens": hlt_tokens,
            "sanitized_hlt_mask": hlt_mask,
            "parent_tokens": parent_tokens,
            "parent_raw": parent_raw,
            "parent_delta": parent_delta,
            "parent_weight_logits": parent_raw[:, :, 4],
            "parent_weights": parent_weights,
            "extra_tokens": extra_tokens,
            "extra_raw": extra_raw,
            "extra_weight_logits": extra_raw[:, :, RAW_TOKEN_DIM],
            "extra_weights": extra_weights,
            "extra_mask": extra_mask,
            "extra_weight_sum": extra_budget["extra_weight_sum"],
            "extra_pt_fraction": extra_budget["extra_pt_fraction"],
            "extra_slot_active_mask": extra_budget["extra_slot_active_mask"],
            "extra_slot_usage": extra_budget["extra_slot_usage"],
            "extra_slot_usage_histogram": extra_budget["extra_slot_usage_histogram"],
            "global_raw": global_raw,
            "global_correction": global_correction,
            "global_calibration": global_correction,
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
                "construction": "aggressive_soft_view_parents_plus_extras",
                "model_family": "teacher_logit_aggressive_reconstructor",
                "aggression_level": self.config.aggression_level,
                "parent_reweighting_enabled": True,
                "global_calibration_enabled": True,
                "n_parent_candidates": int(parent_tokens.shape[1]),
                "n_extra_candidates": int(extra_tokens.shape[1]),
                "num_extra_candidates": int(extra_tokens.shape[1]),
                "max_total_extra_pt_fraction": float(self.config.max_total_extra_pt_fraction),
                "extra_usage_weight_threshold": float(self.config.extra_usage_weight_threshold),
                "config": self.config.to_dict(),
                "diagnostics": diagnostics,
                **dict(metadata or {}),
            },
            aux=aux,
        )


__all__ = [
    "AGGRESSION_LEVEL",
    "AggressiveSoftViewConfig",
    "AggressiveSoftViewDiagnostics",
    "AggressiveSoftViewHead",
]
