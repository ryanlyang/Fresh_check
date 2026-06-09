"""Stage2 dual-view HLT + reconstructed-view tagger for Step 8."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np

from .hlt_baseline import (
    accuracy_from_logits,
    default_part_config,
    require_torch,
    resolve_device,
    save_json,
    set_training_seed,
)
from .hlt_cache import load_cached_hlt_view
from .jetclass_data import LABEL_NAMES, JetView
from .part_inputs import PF_FEATURE_NAMES
from .reconstructor import (
    M2BaseReconstructor,
    RECONSTRUCTOR_VARIANT_NAMES,
    ReconstructorVariantConfig,
    build_reconstructor,
    get_reconstructor_variant_config,
)

try:  # Keep module importable on machines without PyTorch.
    import torch as _torch
except ImportError:  # pragma: no cover - environment dependent
    _torch = None

if _torch is None:  # pragma: no cover - environment dependent
    class _ModuleBase:
        pass

    class _DatasetBase:
        pass
else:
    _ModuleBase = _torch.nn.Module
    _DatasetBase = _torch.utils.data.Dataset


EPS = 1.0e-8
ENERGY_EPS = 1.0e-4
DUAL_VIEW_EXPERIMENT_STEP = "v2_step5_cross_attention_dual_view_tagger"
CORRECTED_VIEW_BASE_FEATURE_NAMES = [
    "part_pt_log",
    "part_e_log",
    "part_logptrel",
    "part_logerel",
    "part_deltaR",
    "part_deta",
    "part_dphi",
]
CORRECTED_VIEW_SUPPORT_FEATURE_NAMES = [
    "token_weight",
    "parent_added_support",
    "budget_efficiency_share",
]
CORRECTED_VIEW_FEATURE_NAMES = CORRECTED_VIEW_BASE_FEATURE_NAMES + CORRECTED_VIEW_SUPPORT_FEATURE_NAMES


@dataclass
class CorrectedViewInputs:
    """Parent-aligned soft corrected view for the original-mechanism Step 4 path."""

    features: Any
    mask: Any
    tokens: Any
    token_weight: Any
    parent_added_support: Any
    budget_efficiency_share: Any
    support_channels: Dict[str, Any]
    feature_names: List[str]
    metadata: Dict[str, Any]


@dataclass
class DualViewTaggerTrainConfig:
    """Training configuration for one Step 9 dual-view tagger."""

    output_dir: str
    hlt_cache_dir: str
    reconstructor_checkpoint: str
    variant: str = "m2_base"
    train_split: str = "model_train"
    val_split: str = "model_val"
    seed: int = 909
    batch_size: int = 128
    epochs: int = 20
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    num_workers: int = 0
    device: str = "auto"
    amp: bool = True
    grad_clip_norm: float = 1.0
    early_stop_patience: int = 5
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    model_size: str = "base"
    compile_model: bool = False
    max_constits: int = 128
    reco_weight_threshold: float = 0.0
    hlt_baseline_report: str | None = None


def wrap_phi_torch(phi):
    torch = require_torch()
    return torch.remainder(phi + torch.pi, 2.0 * torch.pi) - torch.pi


def _manual_transform_torch(value, *, subtract: float, multiply: float, clip_min: float = -5.0, clip_max: float = 5.0):
    torch = require_torch()
    return torch.clamp((value - float(subtract)) * float(multiply), float(clip_min), float(clip_max))


def _nan_to_num_torch(value, *, nan: float = 0.0, posinf: float = 0.0, neginf: float = 0.0):
    torch = require_torch()
    if hasattr(torch, "nan_to_num"):
        return torch.nan_to_num(value, nan=float(nan), posinf=float(posinf), neginf=float(neginf))
    return torch.where(torch.isfinite(value), value, torch.zeros_like(value) + float(nan))


def _physical_energy_floor(pt, eta, *, eps: float = ENERGY_EPS):
    torch = require_torch()
    return torch.clamp(pt, min=0.0) * torch.cosh(torch.clamp(eta, -5.0, 5.0)) + float(eps)


def sanitize_tokens_for_part_inputs(tokens, mask):
    """Clamp token kinematics to finite, physical values before ParT input building."""

    torch = require_torch()
    tokens = tokens.float()
    mask = mask.bool()
    finite_tokens = torch.isfinite(tokens).all(dim=-1)
    tokens = _nan_to_num_torch(tokens)
    cleaned = tokens.clone()
    pt = torch.clamp(cleaned[:, :, 0], min=0.0)
    eta = torch.clamp(cleaned[:, :, 1], -5.0, 5.0)
    phi = wrap_phi_torch(cleaned[:, :, 2])
    energy = torch.maximum(torch.clamp(cleaned[:, :, 3], min=ENERGY_EPS), _physical_energy_floor(pt, eta))
    cleaned[:, :, 0] = pt
    cleaned[:, :, 1] = eta
    cleaned[:, :, 2] = phi
    cleaned[:, :, 3] = energy
    return cleaned, mask & finite_tokens


def _select_topk(tokens, mask, weights=None, *, max_constits: int = 128):
    """Select the top candidates by weighted pT without using offline targets."""

    torch = require_torch()
    n_constits = int(tokens.shape[1])
    if n_constits <= int(max_constits):
        return tokens, mask, weights
    score = torch.clamp(tokens[:, :, 0], min=0.0)
    if weights is not None:
        score = score * torch.clamp(weights, min=0.0)
    score = score.masked_fill(~mask.bool(), -1.0)
    _, indices = torch.topk(score, k=int(max_constits), dim=1, largest=True, sorted=True)
    token_indices = indices[:, :, None].expand(-1, -1, tokens.shape[2])
    tokens = torch.gather(tokens, dim=1, index=token_indices)
    mask = torch.gather(mask.bool(), dim=1, index=indices)
    if weights is not None:
        weights = torch.gather(weights, dim=1, index=indices)
    return tokens, mask, weights


def _force_nonempty_parent_mask(mask, fallback_mask, score):
    torch = require_torch()
    mask = mask.bool()
    fallback_mask = fallback_mask.bool()
    empty = mask.sum(dim=1) == 0
    if not bool(empty.any()):
        return mask, empty
    forced = mask.clone()
    safe_score = torch.where(fallback_mask, _nan_to_num_torch(score.float()), torch.full_like(score.float(), -1.0))
    fallback_empty = fallback_mask.sum(dim=1) == 0
    if bool(fallback_empty.any()):
        safe_score = safe_score.clone()
        safe_score[fallback_empty, 0] = 0.0
    indices = safe_score.argmax(dim=1)
    rows = torch.arange(mask.shape[0], device=mask.device)
    forced[rows[empty], indices[empty]] = True
    return forced, empty


def _parent_channel_or_zeros(value, reference, mask, *, name: str, clamp_min: float | None = 0.0):
    torch = require_torch()
    if value is None:
        channel = torch.zeros(reference.shape[:2], dtype=reference.dtype, device=reference.device)
    else:
        channel = value.float()
        if channel.ndim == 3 and channel.shape[-1] == 1:
            channel = channel.squeeze(-1)
        if tuple(channel.shape) != tuple(reference.shape[:2]):
            raise ValueError(
                f"{name} must be parent-aligned with shape {tuple(reference.shape[:2])}, "
                f"got {tuple(channel.shape)}"
            )
        channel = _nan_to_num_torch(channel)
        if clamp_min is not None:
            channel = torch.clamp(channel, min=float(clamp_min))
    return channel * mask.float()


def _corrected_parent_source(reconstruction, hlt_tokens):
    parent_tokens = getattr(reconstruction, "corrected_parent_tokens", None)
    source = "corrected_parent_tokens"
    if parent_tokens is None:
        parent_tokens = getattr(reconstruction, "edited_tokens", None)
        source = "edited_tokens"
    if parent_tokens is None:
        parent_tokens = hlt_tokens
        source = "hlt_tokens_fallback"
    return parent_tokens, source


def _corrected_parent_weight_source(reconstruction, hlt_mask):
    parent_weights = getattr(reconstruction, "corrected_parent_weights", None)
    source = "corrected_parent_weights"
    if parent_weights is None:
        parent_weights = getattr(reconstruction, "edited_weights", None)
        source = "edited_weights"
    if parent_weights is None:
        parent_weights = hlt_mask.float()
        source = "hlt_mask_fallback"
    return parent_weights, source


def build_soft_corrected_view_torch(
    hlt_tokens,
    hlt_mask,
    reconstruction,
    *,
    weight_threshold: float = 0.0,
    scale_features_by_weight: bool = True,
    force_nonempty: bool = True,
) -> CorrectedViewInputs:
    """Build the Step 4 parent-token-aligned corrected view from HLT-only outputs.

    The corrected view intentionally ignores split/generated candidates as extra
    particles. Their soft support is folded back onto the corresponding HLT
    parent as support channels, preserving one corrected token per HLT parent.
    """

    torch = require_torch()
    hlt_tokens = hlt_tokens.float()
    hlt_mask = hlt_mask.bool()
    parent_tokens, token_source = _corrected_parent_source(reconstruction, hlt_tokens)
    parent_tokens = parent_tokens.float()
    if tuple(parent_tokens.shape[:2]) != tuple(hlt_tokens.shape[:2]):
        raise ValueError(
            "Corrected parent tokens must stay aligned to the fixed-HLT parents: "
            f"expected leading shape {tuple(hlt_tokens.shape[:2])}, got {tuple(parent_tokens.shape[:2])}"
        )
    parent_tokens, finite_parent_mask = sanitize_tokens_for_part_inputs(parent_tokens, hlt_mask)

    parent_weights, weight_source = _corrected_parent_weight_source(reconstruction, hlt_mask)
    parent_weights = _parent_channel_or_zeros(
        parent_weights,
        parent_tokens,
        finite_parent_mask,
        name=weight_source,
        clamp_min=0.0,
    )
    split_support = _parent_channel_or_zeros(
        getattr(reconstruction, "split_parent_added_support", None),
        parent_tokens,
        finite_parent_mask,
        name="split_parent_added_support",
        clamp_min=0.0,
    )
    generator_support = _parent_channel_or_zeros(
        getattr(reconstruction, "generator_parent_added_support", None),
        parent_tokens,
        finite_parent_mask,
        name="generator_parent_added_support",
        clamp_min=0.0,
    )
    parent_added_support = getattr(reconstruction, "parent_added_support", None)
    if parent_added_support is None:
        parent_added_support = split_support + generator_support
    else:
        parent_added_support = _parent_channel_or_zeros(
            parent_added_support,
            parent_tokens,
            finite_parent_mask,
            name="parent_added_support",
            clamp_min=0.0,
        )
    budget_efficiency_share = getattr(reconstruction, "budget_efficiency_share", None)
    if budget_efficiency_share is None:
        added_count_pred = getattr(reconstruction, "added_count_pred", None)
        if added_count_pred is None:
            budget_efficiency_share = parent_added_support
        else:
            added_count_pred = _nan_to_num_torch(added_count_pred.float())
            budget_efficiency_share = parent_added_support / torch.clamp(added_count_pred[:, None], min=1.0)
    budget_efficiency_share = _parent_channel_or_zeros(
        budget_efficiency_share,
        parent_tokens,
        finite_parent_mask,
        name="budget_efficiency_share",
        clamp_min=0.0,
    )

    view_mask = finite_parent_mask & (parent_weights > float(weight_threshold))
    forced_empty = torch.zeros(hlt_tokens.shape[0], dtype=torch.bool, device=hlt_tokens.device)
    if force_nonempty:
        view_mask, forced_empty = _force_nonempty_parent_mask(view_mask, finite_parent_mask, parent_weights)

    prepared = parent_tokens.clone()
    if scale_features_by_weight:
        prepared[:, :, 0] = prepared[:, :, 0] * parent_weights
        prepared[:, :, 3] = prepared[:, :, 3] * parent_weights
        prepared[:, :, 3] = torch.maximum(
            prepared[:, :, 3],
            _physical_energy_floor(prepared[:, :, 0], prepared[:, :, 1]),
        )
    prepared = prepared * view_mask[:, :, None].float()

    pt = torch.where(view_mask, prepared[:, :, 0], torch.zeros_like(prepared[:, :, 0]))
    eta = torch.where(view_mask, prepared[:, :, 1], torch.zeros_like(prepared[:, :, 1]))
    phi = torch.where(view_mask, prepared[:, :, 2], torch.zeros_like(prepared[:, :, 2]))
    energy = torch.where(view_mask, prepared[:, :, 3], torch.zeros_like(prepared[:, :, 3]))

    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    pz = pt * torch.sinh(eta)
    jet_px = px.sum(dim=1)
    jet_py = py.sum(dim=1)
    jet_pz = pz.sum(dim=1)
    jet_energy = energy.sum(dim=1)
    jet_pt = torch.sqrt(torch.clamp(jet_px * jet_px + jet_py * jet_py, min=0.0))
    jet_phi = torch.atan2(jet_py, jet_px)
    jet_eta = torch.asinh(jet_pz / torch.clamp(jet_pt, min=EPS))
    jet_eta = torch.where(jet_pt > EPS, jet_eta, torch.zeros_like(jet_eta))
    jet_phi = torch.where(jet_pt > EPS, jet_phi, torch.zeros_like(jet_phi))

    eta_sign = torch.sign(jet_eta[:, None])
    eta_sign = torch.where(eta_sign == 0.0, torch.ones_like(eta_sign), eta_sign)
    part_deta = (eta - jet_eta[:, None]) * eta_sign
    part_dphi = wrap_phi_torch(phi - jet_phi[:, None])
    part_delta_r = torch.sqrt(torch.clamp(part_deta * part_deta + part_dphi * part_dphi, min=0.0))

    feature_map = {
        "part_pt_log": _manual_transform_torch(torch.log(torch.clamp(pt, min=EPS)), subtract=1.7, multiply=0.7),
        "part_e_log": _manual_transform_torch(torch.log(torch.clamp(energy, min=EPS)), subtract=2.0, multiply=0.7),
        "part_logptrel": _manual_transform_torch(
            torch.log(torch.clamp(pt / torch.clamp(jet_pt[:, None], min=EPS), min=EPS)),
            subtract=-4.7,
            multiply=0.7,
        ),
        "part_logerel": _manual_transform_torch(
            torch.log(torch.clamp(energy / torch.clamp(jet_energy[:, None], min=EPS), min=EPS)),
            subtract=-4.7,
            multiply=0.7,
        ),
        "part_deltaR": _manual_transform_torch(part_delta_r, subtract=0.2, multiply=4.0),
        "part_deta": part_deta,
        "part_dphi": part_dphi,
        "token_weight": parent_weights,
        "parent_added_support": parent_added_support,
        "budget_efficiency_share": budget_efficiency_share,
    }
    for key, value in list(feature_map.items()):
        feature_map[key] = torch.where(view_mask, _nan_to_num_torch(value), torch.zeros_like(value))
    features = torch.stack([feature_map[name] for name in CORRECTED_VIEW_FEATURE_NAMES], dim=1).float()
    features = _nan_to_num_torch(features)
    out_mask = view_mask[:, None, :].bool()
    support_channels = {
        "token_weight": features[:, CORRECTED_VIEW_FEATURE_NAMES.index("token_weight"), :],
        "parent_added_support": features[:, CORRECTED_VIEW_FEATURE_NAMES.index("parent_added_support"), :],
        "budget_efficiency_share": features[:, CORRECTED_VIEW_FEATURE_NAMES.index("budget_efficiency_share"), :],
    }
    return CorrectedViewInputs(
        features=features,
        mask=out_mask,
        tokens=parent_tokens,
        token_weight=support_channels["token_weight"],
        parent_added_support=support_channels["parent_added_support"],
        budget_efficiency_share=support_channels["budget_efficiency_share"],
        support_channels=support_channels,
        feature_names=list(CORRECTED_VIEW_FEATURE_NAMES),
        metadata={
            "parent_aligned": True,
            "uses_offline_constituents": False,
            "token_source": token_source,
            "weight_source": weight_source,
            "scale_features_by_weight": bool(scale_features_by_weight),
            "weight_threshold": float(weight_threshold),
            "force_nonempty": bool(force_nonempty),
            "forced_nonempty_count": int(forced_empty.sum().detach().cpu().item()),
        },
    )


def build_part_inputs_torch(
    tokens,
    mask,
    *,
    weights=None,
    max_constits: int = 128,
    weight_threshold: float = 0.0,
) -> Dict[str, Any]:
    """Build Particle Transformer tensors from a token view on device."""

    torch = require_torch()
    tokens, mask = sanitize_tokens_for_part_inputs(tokens, mask)
    if weights is not None:
        weights = weights.float()
        finite_weights = torch.isfinite(weights)
        weights = torch.clamp(_nan_to_num_torch(weights), min=0.0)
        mask = mask & finite_weights & (weights > float(weight_threshold))
    tokens, mask, weights = _select_topk(tokens, mask, weights, max_constits=max_constits)
    prepared = tokens.clone()
    if weights is not None:
        prepared[:, :, 0] = prepared[:, :, 0] * weights
        prepared[:, :, 3] = prepared[:, :, 3] * weights
        prepared[:, :, 3] = torch.maximum(prepared[:, :, 3], _physical_energy_floor(prepared[:, :, 0], prepared[:, :, 1]))
    prepared = prepared * mask[:, :, None].float()

    pt = torch.where(mask, prepared[:, :, 0], torch.zeros_like(prepared[:, :, 0]))
    eta = torch.where(mask, prepared[:, :, 1], torch.zeros_like(prepared[:, :, 1]))
    phi = torch.where(mask, prepared[:, :, 2], torch.zeros_like(prepared[:, :, 2]))
    energy = torch.where(mask, prepared[:, :, 3], torch.zeros_like(prepared[:, :, 3]))

    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    pz = pt * torch.sinh(eta)
    jet_px = px.sum(dim=1)
    jet_py = py.sum(dim=1)
    jet_pz = pz.sum(dim=1)
    jet_energy = energy.sum(dim=1)
    jet_pt = torch.sqrt(torch.clamp(jet_px * jet_px + jet_py * jet_py, min=0.0))
    jet_phi = torch.atan2(jet_py, jet_px)
    jet_eta = torch.asinh(jet_pz / torch.clamp(jet_pt, min=EPS))
    jet_eta = torch.where(jet_pt > EPS, jet_eta, torch.zeros_like(jet_eta))
    jet_phi = torch.where(jet_pt > EPS, jet_phi, torch.zeros_like(jet_phi))

    eta_sign = torch.sign(jet_eta[:, None])
    eta_sign = torch.where(eta_sign == 0.0, torch.ones_like(eta_sign), eta_sign)
    part_deta = (eta - jet_eta[:, None]) * eta_sign
    part_dphi = wrap_phi_torch(phi - jet_phi[:, None])
    part_delta_r = torch.sqrt(torch.clamp(part_deta * part_deta + part_dphi * part_dphi, min=0.0))

    feature_map = {
        "part_pt_log": _manual_transform_torch(torch.log(torch.clamp(pt, min=EPS)), subtract=1.7, multiply=0.7),
        "part_e_log": _manual_transform_torch(torch.log(torch.clamp(energy, min=EPS)), subtract=2.0, multiply=0.7),
        "part_logptrel": _manual_transform_torch(
            torch.log(torch.clamp(pt / torch.clamp(jet_pt[:, None], min=EPS), min=EPS)),
            subtract=-4.7,
            multiply=0.7,
        ),
        "part_logerel": _manual_transform_torch(
            torch.log(torch.clamp(energy / torch.clamp(jet_energy[:, None], min=EPS), min=EPS)),
            subtract=-4.7,
            multiply=0.7,
        ),
        "part_deltaR": _manual_transform_torch(part_delta_r, subtract=0.2, multiply=4.0),
        "part_charge": prepared[:, :, 4],
        "part_isChargedHadron": prepared[:, :, 5],
        "part_isNeutralHadron": prepared[:, :, 6],
        "part_isPhoton": prepared[:, :, 7],
        "part_isElectron": prepared[:, :, 8],
        "part_isMuon": prepared[:, :, 9],
        "part_d0": torch.tanh(prepared[:, :, 10]),
        "part_d0err": _manual_transform_torch(prepared[:, :, 11], subtract=0.0, multiply=1.0, clip_min=0.0, clip_max=1.0),
        "part_dz": torch.tanh(prepared[:, :, 12]),
        "part_dzerr": _manual_transform_torch(prepared[:, :, 13], subtract=0.0, multiply=1.0, clip_min=0.0, clip_max=1.0),
        "part_deta": part_deta,
        "part_dphi": part_dphi,
    }
    feature_order = [
        "part_pt_log",
        "part_e_log",
        "part_logptrel",
        "part_logerel",
        "part_deltaR",
        "part_charge",
        "part_isChargedHadron",
        "part_isNeutralHadron",
        "part_isPhoton",
        "part_isElectron",
        "part_isMuon",
        "part_d0",
        "part_d0err",
        "part_dz",
        "part_dzerr",
        "part_deta",
        "part_dphi",
    ]
    for key in feature_map:
        feature_map[key] = torch.where(mask, feature_map[key], torch.zeros_like(feature_map[key]))

    inputs = {
        "points": torch.stack([feature_map["part_deta"], feature_map["part_dphi"]], dim=1).float(),
        "features": torch.stack([feature_map[name] for name in feature_order], dim=1).float(),
        "lorentz_vectors": torch.stack([px, py, pz, energy], dim=1).float() * mask[:, None, :].float(),
        "mask": mask[:, None, :].bool(),
    }
    for key in ("points", "features", "lorentz_vectors"):
        inputs[key] = _nan_to_num_torch(inputs[key])
    return inputs


class HLTTokenDataset(_DatasetBase):
    """Cached fixed-HLT token dataset for dual-view classification."""

    def __init__(self, view: JetView, *, max_jets: int | None = None) -> None:
        require_torch()
        if view.metadata.get("view") not in (None, "fixed_hlt"):
            raise ValueError(f"Expected fixed_hlt view, got {view.metadata.get('view')!r}")
        limit = len(view.labels) if max_jets is None else min(int(max_jets), len(view.labels))
        self.tokens = np.asarray(view.tokens[:limit], dtype=np.float32)
        self.mask = np.asarray(view.mask[:limit], dtype=bool)
        self.labels = np.asarray(view.labels[:limit], dtype=np.int64)
        self.split = view.split

    def __len__(self) -> int:
        return int(len(self.labels))

    def __getitem__(self, index: int):
        return self.tokens[index], self.mask[index], self.labels[index]


def collate_hlt_tokens(samples):
    torch = require_torch()
    return {
        "hlt_tokens": torch.from_numpy(np.stack([row[0] for row in samples], axis=0)).float(),
        "hlt_mask": torch.from_numpy(np.stack([row[1] for row in samples], axis=0)).bool(),
        "labels": torch.from_numpy(np.asarray([row[2] for row in samples], dtype=np.int64)).long(),
    }


def make_hlt_token_loader(dataset, *, batch_size: int, shuffle: bool, num_workers: int, seed: int):
    torch = require_torch()
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_hlt_tokens,
        generator=generator,
    )


def _dual_view_size_config(model_size: str) -> Dict[str, int]:
    if model_size == "tiny":
        return {"hidden_dim": 64, "num_heads": 4, "num_layers": 2, "feedforward_dim": 160}
    if model_size == "base":
        return {"hidden_dim": 128, "num_heads": 8, "num_layers": 3, "feedforward_dim": 384}
    raise ValueError(f"Unknown dual-view model_size {model_size!r}; expected 'base' or 'tiny'")


def _input_features_and_mask(inputs):
    torch = require_torch()
    features = inputs.features if isinstance(inputs, CorrectedViewInputs) else inputs["features"]
    mask = inputs.mask if isinstance(inputs, CorrectedViewInputs) else inputs["mask"]
    features = features.float()
    mask = mask.bool()
    if features.ndim != 3:
        raise ValueError(f"Expected feature tensor [B, C, N], got shape {tuple(features.shape)}")
    if mask.ndim == 3:
        if mask.shape[1] != 1:
            raise ValueError(f"Expected mask tensor [B, 1, N], got shape {tuple(mask.shape)}")
        mask = mask[:, 0, :]
    elif mask.ndim != 2:
        raise ValueError(f"Expected mask tensor [B, N] or [B, 1, N], got shape {tuple(mask.shape)}")
    if features.shape[0] != mask.shape[0] or features.shape[2] != mask.shape[1]:
        raise ValueError(f"Feature/mask shape mismatch: features={tuple(features.shape)}, mask={tuple(mask.shape)}")
    features = _nan_to_num_torch(features)
    empty = mask.sum(dim=1) == 0
    if bool(empty.any()):
        mask = mask.clone()
        mask[empty, 0] = True
        features = features.clone()
        features[empty, :, 0] = 0.0
    return features.transpose(1, 2).contiguous(), mask


class MaskedAttentionPool(_ModuleBase):
    """Learned mask-aware attention pooling over encoded token sequences."""

    def __init__(self, hidden_dim: int, *, dropout: float = 0.0) -> None:
        require_torch()
        super().__init__()
        torch = require_torch()
        self.norm = torch.nn.LayerNorm(int(hidden_dim))
        self.score = torch.nn.Linear(int(hidden_dim), 1)
        self.dropout = torch.nn.Dropout(float(dropout))

    def forward(self, tokens, mask):
        torch = require_torch()
        tokens = self.norm(tokens)
        scores = self.score(tokens).squeeze(-1).masked_fill(~mask.bool(), -1.0e4)
        weights = torch.softmax(scores, dim=1) * mask.float()
        weights = weights / torch.clamp(weights.sum(dim=1, keepdim=True), min=1.0e-6)
        pooled = torch.einsum("bn,bnd->bd", weights, self.dropout(tokens))
        return pooled, weights


class FeatureSequenceEncoder(_ModuleBase):
    """Transformer encoder for an HLT or corrected-view feature sequence."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        feedforward_dim: int,
        dropout: float,
    ) -> None:
        require_torch()
        super().__init__()
        torch = require_torch()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.input_proj = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, self.hidden_dim),
            torch.nn.LayerNorm(self.hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(float(dropout)),
        )
        layer = torch.nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=int(num_heads),
            dim_feedforward=int(feedforward_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(layer, num_layers=int(num_layers))
        self.output_norm = torch.nn.LayerNorm(self.hidden_dim)

    def forward(self, features, mask):
        x = self.input_proj(features)
        x = self.encoder(x, src_key_padding_mask=~mask.bool())
        x = self.output_norm(x)
        return x * mask.unsqueeze(-1).float()


class DualViewCrossAttentionTagger(_ModuleBase):
    """Original-mechanism HLT/corrected-view cross-attention fusion classifier."""

    def __init__(
        self,
        *,
        num_classes: int = 10,
        model_size: str = "base",
        hidden_dim: int | None = None,
        num_heads: int | None = None,
        num_layers: int | None = None,
        feedforward_dim: int | None = None,
        dropout: float = 0.05,
    ) -> None:
        require_torch()
        super().__init__()
        torch = require_torch()
        size_cfg = _dual_view_size_config(model_size)
        hidden_dim = int(hidden_dim or size_cfg["hidden_dim"])
        num_heads = int(num_heads or size_cfg["num_heads"])
        num_layers = int(num_layers or size_cfg["num_layers"])
        feedforward_dim = int(feedforward_dim or size_cfg["feedforward_dim"])
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")

        self.hlt_encoder = FeatureSequenceEncoder(
            input_dim=len(PF_FEATURE_NAMES),
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
        )
        self.corrected_encoder = FeatureSequenceEncoder(
            input_dim=len(CORRECTED_VIEW_FEATURE_NAMES),
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
        )
        self.hlt_pool = MaskedAttentionPool(hidden_dim, dropout=dropout)
        self.corrected_pool = MaskedAttentionPool(hidden_dim, dropout=dropout)
        self.hlt_to_corrected = torch.nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.corrected_to_hlt = torch.nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.hlt_cross_norm = torch.nn.LayerNorm(hidden_dim)
        self.corrected_cross_norm = torch.nn.LayerNorm(hidden_dim)
        self.cross_dropout = torch.nn.Dropout(float(dropout))
        fused_dim = hidden_dim * 6
        self.classifier = torch.nn.Sequential(
            torch.nn.LayerNorm(fused_dim),
            torch.nn.Linear(fused_dim, hidden_dim * 2),
            torch.nn.GELU(),
            torch.nn.Dropout(float(dropout)),
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(float(dropout)),
            torch.nn.Linear(hidden_dim, int(num_classes)),
        )
        self.config = {
            "architecture": "cross_attention_fusion",
            "num_classes": int(num_classes),
            "model_size": model_size,
            "hidden_dim": int(hidden_dim),
            "num_heads": int(num_heads),
            "num_layers": int(num_layers),
            "feedforward_dim": int(feedforward_dim),
            "dropout": float(dropout),
            "hlt_feature_names": list(PF_FEATURE_NAMES),
            "corrected_view_feature_names": list(CORRECTED_VIEW_FEATURE_NAMES),
        }

    def no_weight_decay(self) -> set[str]:
        return set()

    def forward(self, hlt_inputs: Mapping[str, Any], corrected_inputs: Mapping[str, Any] | CorrectedViewInputs):
        torch = require_torch()
        hlt_features, hlt_mask = _input_features_and_mask(hlt_inputs)
        corrected_features, corrected_mask = _input_features_and_mask(corrected_inputs)
        if hlt_features.shape[-1] != len(PF_FEATURE_NAMES):
            raise ValueError(f"HLT branch expected {len(PF_FEATURE_NAMES)} features, got {hlt_features.shape[-1]}")
        if corrected_features.shape[-1] != len(CORRECTED_VIEW_FEATURE_NAMES):
            raise ValueError(
                f"Corrected branch expected {len(CORRECTED_VIEW_FEATURE_NAMES)} features, "
                f"got {corrected_features.shape[-1]}"
            )

        hlt_tokens = self.hlt_encoder(hlt_features, hlt_mask)
        corrected_tokens = self.corrected_encoder(corrected_features, corrected_mask)
        hlt_pool, _ = self.hlt_pool(hlt_tokens, hlt_mask)
        corrected_pool, _ = self.corrected_pool(corrected_tokens, corrected_mask)

        hlt_attended, _ = self.hlt_to_corrected(
            query=hlt_tokens,
            key=corrected_tokens,
            value=corrected_tokens,
            key_padding_mask=~corrected_mask.bool(),
            need_weights=False,
        )
        corrected_attended, _ = self.corrected_to_hlt(
            query=corrected_tokens,
            key=hlt_tokens,
            value=hlt_tokens,
            key_padding_mask=~hlt_mask.bool(),
            need_weights=False,
        )
        hlt_cross = self.hlt_cross_norm(hlt_tokens + self.cross_dropout(hlt_attended)) * hlt_mask.unsqueeze(-1).float()
        corrected_cross = (
            self.corrected_cross_norm(corrected_tokens + self.cross_dropout(corrected_attended))
            * corrected_mask.unsqueeze(-1).float()
        )
        hlt_cross_pool, _ = self.hlt_pool(hlt_cross, hlt_mask)
        corrected_cross_pool, _ = self.corrected_pool(corrected_cross, corrected_mask)

        fused = torch.cat(
            [
                hlt_pool,
                corrected_pool,
                hlt_cross_pool,
                corrected_cross_pool,
                torch.abs(hlt_cross_pool - corrected_cross_pool),
                hlt_cross_pool * corrected_cross_pool,
            ],
            dim=1,
        )
        logits = self.classifier(fused)
        return _nan_to_num_torch(logits)


class _ParticleTransformerEmbeddingBranch(_ModuleBase):
    """ParticleTransformer branch that returns a CLS embedding, not class logits."""

    def __init__(self, **kwargs) -> None:
        require_torch()
        super().__init__()
        try:
            from weaver.nn.model.ParticleTransformer import ParticleTransformer
        except ImportError as exc:  # pragma: no cover - depends on research env
            raise ImportError(
                "Particle Transformer dual-view loading requires weaver-core on the research compute."
            ) from exc

        branch_cfg = dict(kwargs)
        branch_cfg["num_classes"] = None
        branch_cfg["fc_params"] = None
        self.config = branch_cfg
        self.mod = ParticleTransformer(**branch_cfg)

    def no_weight_decay(self) -> set[str]:
        return {"mod.cls_token"}

    def forward(self, inputs: Mapping[str, Any]):
        return self.mod(inputs["features"], v=inputs["lorentz_vectors"], mask=inputs["mask"])


class DualViewParticleTransformerTagger(_ModuleBase):
    """Legacy-compatible two-branch ParticleTransformer dual-view classifier."""

    def __init__(
        self,
        *,
        num_classes: int = 10,
        model_size: str = "base",
        branch_config: Mapping[str, Any] | None = None,
        classifier_hidden_dim: int | None = None,
        dropout: float = 0.05,
        activation: str = "gelu",
        max_constits: int = 128,
        reco_weight_threshold: float = 0.0,
        **_: Any,
    ) -> None:
        require_torch()
        super().__init__()
        torch = require_torch()
        cfg = default_part_config(num_classes=int(num_classes), model_size=model_size)
        if branch_config:
            cfg.update(dict(branch_config))
        cfg["num_classes"] = None
        cfg["fc_params"] = None
        branch_dim = int(cfg["embed_dims"][-1])
        hidden_dim = int(classifier_hidden_dim or branch_dim)
        activation_layer = torch.nn.GELU() if str(activation).lower() == "gelu" else torch.nn.ReLU()

        self.hlt_branch = _ParticleTransformerEmbeddingBranch(**cfg)
        self.reco_branch = _ParticleTransformerEmbeddingBranch(**cfg)
        self.classifier = torch.nn.Sequential(
            torch.nn.LayerNorm(branch_dim * 2),
            torch.nn.Linear(branch_dim * 2, hidden_dim),
            activation_layer,
            torch.nn.Dropout(float(dropout)),
            torch.nn.Linear(hidden_dim, int(num_classes)),
        )
        self.max_constits = int(max_constits)
        self.reco_weight_threshold = float(reco_weight_threshold)
        self.config = {
            "architecture": "particle_transformer_concat",
            "num_classes": int(num_classes),
            "model_size": model_size,
            "branch_config": dict(cfg),
            "branch_dim": int(branch_dim),
            "classifier_hidden_dim": int(hidden_dim),
            "dropout": float(dropout),
            "activation": str(activation),
            "max_constits": int(max_constits),
            "reco_weight_threshold": float(reco_weight_threshold),
            "hlt_feature_names": list(PF_FEATURE_NAMES),
            "corrected_view_feature_names": list(PF_FEATURE_NAMES),
        }

    def no_weight_decay(self) -> set[str]:
        return {"hlt_branch.mod.cls_token", "reco_branch.mod.cls_token"}

    def _corrected_to_part_inputs(self, corrected_inputs):
        if isinstance(corrected_inputs, CorrectedViewInputs):
            mask = corrected_inputs.mask
            if mask.ndim == 3:
                mask = mask[:, 0, :]
            return build_part_inputs_torch(
                corrected_inputs.tokens,
                mask,
                weights=corrected_inputs.token_weight,
                max_constits=self.max_constits,
                weight_threshold=self.reco_weight_threshold,
            )
        return corrected_inputs

    def forward(self, hlt_inputs: Mapping[str, Any], corrected_inputs: Mapping[str, Any] | CorrectedViewInputs):
        torch = require_torch()
        reco_inputs = self._corrected_to_part_inputs(corrected_inputs)
        hlt_embedding = self.hlt_branch(hlt_inputs)
        reco_embedding = self.reco_branch(reco_inputs)
        fused = torch.cat([hlt_embedding, reco_embedding], dim=1)
        return _nan_to_num_torch(self.classifier(fused))


def detect_dual_view_architecture_from_state_dict(
    state_dict: Mapping[str, Any],
    model_config: Mapping[str, Any] | None = None,
) -> str:
    keys = tuple(state_dict.keys())
    if any(key.startswith("hlt_branch.mod.") for key in keys) and any(
        key.startswith("reco_branch.mod.") for key in keys
    ):
        return "particle_transformer_concat"
    if any(key.startswith("hlt_encoder.") for key in keys) or any(
        key.startswith("corrected_encoder.") for key in keys
    ):
        return "cross_attention_fusion"
    if model_config and model_config.get("architecture"):
        return str(model_config["architecture"])
    return "cross_attention_fusion"


def build_dual_view_tagger(
    *,
    model_size: str = "base",
    num_classes: int = 10,
    hidden_dim: int | None = None,
    num_heads: int | None = None,
    num_layers: int | None = None,
    feedforward_dim: int | None = None,
    dropout: float = 0.05,
    branch_config: Mapping[str, Any] | None = None,
    classifier_hidden_dim: int | None = None,
    activation: str = "gelu",
    max_constits: int = 128,
    reco_weight_threshold: float = 0.0,
    architecture: str | None = None,
):
    if architecture in ("particle_transformer_concat", "particle_transformer_dual_view"):
        return DualViewParticleTransformerTagger(
            num_classes=num_classes,
            model_size=model_size,
            branch_config=branch_config,
            classifier_hidden_dim=classifier_hidden_dim,
            dropout=dropout,
            activation=activation,
            max_constits=max_constits,
            reco_weight_threshold=reco_weight_threshold,
        )
    if architecture not in (None, "cross_attention_fusion"):
        raise ValueError(f"Unsupported dual-view architecture {architecture!r}")
    return DualViewCrossAttentionTagger(
        num_classes=num_classes,
        model_size=model_size,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        feedforward_dim=feedforward_dim,
        dropout=dropout,
    )


def load_stage_a_reconstructor_checkpoint(path: str | Path, *, device=None):
    torch = require_torch()
    payload = torch.load(path, map_location=device or "cpu")
    variant_payload = payload.get("variant_config", {})
    variant_config = ReconstructorVariantConfig(**variant_payload) if variant_payload else get_reconstructor_variant_config("m2_base")
    model = build_reconstructor(variant_config)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if device is not None:
        model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, payload


def move_batch_to_device(batch: Mapping[str, Any], device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def run_dual_view_epoch(
    tagger,
    reconstructor,
    loader,
    *,
    device,
    criterion,
    optimizer=None,
    scaler=None,
    amp: bool = False,
    grad_clip_norm: float = 0.0,
    max_batches: int | None = None,
    max_constits: int = 128,
    reco_weight_threshold: float = 0.0,
) -> Dict[str, float]:
    torch = require_torch()
    is_train = optimizer is not None
    tagger.train(is_train)
    reconstructor.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            batch = move_batch_to_device(batch, device)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            autocast_enabled = bool(amp and device.type == "cuda")
            with torch.cuda.amp.autocast(enabled=autocast_enabled):
                with torch.no_grad():
                    reco = reconstructor(batch["hlt_tokens"], batch["hlt_mask"])
                hlt_inputs = build_part_inputs_torch(
                    batch["hlt_tokens"],
                    batch["hlt_mask"],
                    max_constits=max_constits,
                )
                corrected_inputs = build_soft_corrected_view_torch(
                    batch["hlt_tokens"],
                    batch["hlt_mask"],
                    reco,
                    weight_threshold=reco_weight_threshold,
                )
                logits = tagger(hlt_inputs, corrected_inputs)
                loss = criterion(logits, batch["labels"])
                if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite dual-view output in batch {batch_index}: "
                        f"logits_finite={bool(torch.isfinite(logits).all())}, "
                        f"loss_finite={bool(torch.isfinite(loss))}"
                    )

            if is_train:
                if scaler is not None and autocast_enabled:
                    scaler.scale(loss).backward()
                    if grad_clip_norm and grad_clip_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(tagger.parameters(), float(grad_clip_norm))
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip_norm and grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(tagger.parameters(), float(grad_clip_norm))
                    optimizer.step()

            batch_size = int(batch["labels"].numel())
            total_loss += float(loss.detach().item()) * batch_size
            correct, seen = accuracy_from_logits(logits.detach(), batch["labels"])
            total_correct += correct
            total_seen += seen
    if total_seen == 0:
        return {"loss": float("nan"), "accuracy": 0.0, "n_jets": 0}
    return {
        "loss": total_loss / float(total_seen),
        "accuracy": total_correct / float(total_seen),
        "n_jets": int(total_seen),
    }


def dual_view_checkpoint_payload(
    tagger,
    optimizer,
    *,
    epoch: int,
    config: DualViewTaggerTrainConfig,
    metrics: Mapping[str, Any],
    reconstructor_payload: Mapping[str, Any] | None,
):
    return {
        "epoch": int(epoch),
        "model_state_dict": tagger.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(config),
        "metrics": dict(metrics),
        "label_names": list(LABEL_NAMES),
        "pf_feature_names": list(PF_FEATURE_NAMES),
        "hlt_feature_names": list(PF_FEATURE_NAMES),
        "corrected_view_feature_names": list(CORRECTED_VIEW_FEATURE_NAMES),
        "model_config": getattr(tagger, "config", {}),
        "reconstructor_checkpoint": config.reconstructor_checkpoint,
        "reconstructor_epoch": None if reconstructor_payload is None else reconstructor_payload.get("epoch"),
        "experiment_step": DUAL_VIEW_EXPERIMENT_STEP,
    }


def _load_optional_json(path: str | None) -> Dict[str, Any] | None:
    if not path:
        return None
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def train_dual_view_tagger(
    config: DualViewTaggerTrainConfig,
    *,
    tagger=None,
    reconstructor=None,
    train_view: JetView | None = None,
    val_view: JetView | None = None,
    max_train_jets: int | None = None,
    max_val_jets: int | None = None,
) -> Dict[str, Any]:
    """Train Stage2 dual-view classifier with a frozen reconstructor."""

    if config.variant not in RECONSTRUCTOR_VARIANT_NAMES:
        raise ValueError(f"Unknown dual-view variant {config.variant!r}; expected one of {RECONSTRUCTOR_VARIANT_NAMES}")
    if config.train_split != "model_train" or config.val_split != "model_val":
        raise ValueError("Dual-view tagger training may use only model_train and select only on model_val")

    torch = require_torch()
    set_training_seed(config.seed)
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_view = train_view or load_cached_hlt_view(config.hlt_cache_dir, config.train_split)
    val_view = val_view or load_cached_hlt_view(config.hlt_cache_dir, config.val_split)
    train_dataset = HLTTokenDataset(train_view, max_jets=max_train_jets)
    val_dataset = HLTTokenDataset(val_view, max_jets=max_val_jets)
    train_loader = make_hlt_token_loader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    val_loader = make_hlt_token_loader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        seed=config.seed + 1,
    )

    reconstructor_payload = None
    if reconstructor is None:
        reconstructor, reconstructor_payload = load_stage_a_reconstructor_checkpoint(
            config.reconstructor_checkpoint,
            device=device,
        )
        checkpoint_variant = reconstructor_payload.get("variant_config", {}).get("name")
        if checkpoint_variant is not None and checkpoint_variant != config.variant:
            raise ValueError(
                f"Dual-view variant {config.variant!r} does not match reconstructor checkpoint variant "
                f"{checkpoint_variant!r}"
            )
    else:
        reconstructor = reconstructor.to(device)
        reconstructor.eval()
        for param in reconstructor.parameters():
            param.requires_grad_(False)

    tagger = tagger or build_dual_view_tagger(num_classes=len(LABEL_NAMES), model_size=config.model_size)
    tagger = tagger.to(device)
    if config.compile_model and hasattr(torch, "compile"):
        tagger = torch.compile(tagger)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(tagger.parameters(), lr=float(config.lr), weight_decay=float(config.weight_decay))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(config.amp and device.type == "cuda"))
    baseline_report = _load_optional_json(config.hlt_baseline_report)

    run_metadata = {
        "config": asdict(config),
        "train_hlt_hash": train_view.metadata.get("hlt_content_hash"),
        "val_hlt_hash": val_view.metadata.get("hlt_content_hash"),
        "train_n_jets": len(train_dataset),
        "val_n_jets": len(val_dataset),
        "reconstructor_frozen": True,
        "reconstructor_checkpoint": config.reconstructor_checkpoint,
        "reconstructor_epoch": None if reconstructor_payload is None else reconstructor_payload.get("epoch"),
        "hlt_baseline_reference": baseline_report,
        "dual_view_architecture": "cross_attention_fusion",
        "corrected_view_feature_names": list(CORRECTED_VIEW_FEATURE_NAMES),
        "leakage_rule": (
            "Dual-view tagger consumes cached fixed-HLT tokens and a parent-aligned soft corrected view "
            "built from the frozen reconstructor output. Offline constituents are not loaded by Stage B training."
        ),
        "no_stack_or_final_test_partitions_loaded": True,
    }
    save_json(output_dir / "config.json", run_metadata)

    curves: List[Dict[str, Any]] = []
    best_val_accuracy = -1.0
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    for epoch in range(1, int(config.epochs) + 1):
        train_metrics = run_dual_view_epoch(
            tagger,
            reconstructor,
            train_loader,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            amp=config.amp,
            grad_clip_norm=config.grad_clip_norm,
            max_batches=config.max_train_batches,
            max_constits=config.max_constits,
            reco_weight_threshold=config.reco_weight_threshold,
        )
        val_metrics = run_dual_view_epoch(
            tagger,
            reconstructor,
            val_loader,
            device=device,
            criterion=criterion,
            amp=False,
            max_batches=config.max_val_batches,
            max_constits=config.max_constits,
            reco_weight_threshold=config.reco_weight_threshold,
        )
        row = {
            "epoch": int(epoch),
            "train": train_metrics,
            "model_val": val_metrics,
        }
        curves.append(row)
        save_json(output_dir / "training_curves.json", {"epochs": curves})

        improved = (
            val_metrics["accuracy"] > best_val_accuracy
            or (
                np.isclose(val_metrics["accuracy"], best_val_accuracy)
                and val_metrics["loss"] < best_val_loss
            )
        )
        torch.save(
            dual_view_checkpoint_payload(
                tagger,
                optimizer,
                epoch=epoch,
                config=config,
                metrics=row,
                reconstructor_payload=reconstructor_payload,
            ),
            output_dir / "last.pt",
        )
        if improved:
            best_val_accuracy = float(val_metrics["accuracy"])
            best_val_loss = float(val_metrics["loss"])
            best_epoch = int(epoch)
            epochs_without_improvement = 0
            torch.save(
                dual_view_checkpoint_payload(
                    tagger,
                    optimizer,
                    epoch=epoch,
                    config=config,
                    metrics=row,
                    reconstructor_payload=reconstructor_payload,
                ),
                output_dir / "best_model_val.pt",
            )
        else:
            epochs_without_improvement += 1
        if config.early_stop_patience >= 0 and epochs_without_improvement >= int(config.early_stop_patience):
            break

    hlt_baseline_accuracy = None
    if baseline_report:
        hlt_baseline_accuracy = baseline_report.get("best_model_val_accuracy")
    report = {
        "experiment_step": DUAL_VIEW_EXPERIMENT_STEP,
        "variant": config.variant,
        "dual_view_architecture": "cross_attention_fusion",
        "best_epoch": int(best_epoch),
        "best_model_val_accuracy": float(best_val_accuracy),
        "best_model_val_loss": float(best_val_loss),
        "hlt_baseline_model_val_accuracy": hlt_baseline_accuracy,
        "epochs_completed": len(curves),
        "final_epoch": curves[-1] if curves else None,
        "checkpoint": str(output_dir / "best_model_val.pt"),
        "last_checkpoint": str(output_dir / "last.pt"),
        "no_final_test_evaluation": True,
        "reconstructor_frozen": True,
    }
    save_json(output_dir / "model_val_report.json", report)
    return report
