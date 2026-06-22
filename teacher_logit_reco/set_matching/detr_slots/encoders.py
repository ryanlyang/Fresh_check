"""Shared encoder interface for DETR/free-slot reconstructors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util
from typing import Any, Mapping

from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM


DETR_SLOT_ENCODER_STEP = "detr_free_slot_step5_encoder_interface"
DETR_SLOT_ENCODER_OUTPUT_CONTRACT = "memory_tokens_memory_mask_global_context_v1"
DETR_SLOT_GLOBAL_TRANSFORMER_ENCODER_STEP = "detr_free_slot_step6_global_transformer_encoder"
DETR_SLOT_PARTICLE_NET_ENCODER_STEP = "detr_free_slot_step7_particle_net_encoder"
DETR_SLOT_PARTICLE_FLOW_ENCODER_STEP = "detr_free_slot_step8_particle_flow_encoder"
DETR_SLOT_PARTICLE_CNN_ENCODER_STEP = "detr_free_slot_step9_particle_cnn_encoder"
DETR_SLOT_PARTICLE_CNN_ORDERING_ASSUMPTION = "fixed_hlt_cache_order_is_canonical_rank_axis"
DETR_SLOT_TOKEN_EMBED_FEATURE_DIM = 15
DETR_SLOT_PARTICLE_NET_COORD_DIM = 3
DETR_SLOT_PARTICLE_FLOW_FEATURE_NAMES = (
    "log_pt",
    "log_energy",
    "eta_scaled",
    "sin_phi",
    "cos_phi",
    "log_pt_fraction",
    "log_energy_fraction",
    "charge",
    "is_charged_hadron",
    "is_neutral_hadron",
    "is_photon",
    "is_electron",
    "is_muon",
    "d0",
    "d0err",
    "dz",
    "dzerr",
    "valid_mask",
)
DETR_SLOT_PARTICLE_FLOW_FEATURE_DIM = len(DETR_SLOT_PARTICLE_FLOW_FEATURE_NAMES)
DETR_SLOT_PARTICLE_FLOW_SUMMARY_FEATURE_NAMES = (
    "log_total_pt",
    "log_total_energy",
    "log_valid_count",
    "pt_weighted_eta",
    "pt_weighted_sin_phi",
    "pt_weighted_cos_phi",
    "mean_abs_eta",
    "charged_hadron_pt_fraction",
    "neutral_hadron_pt_fraction",
    "photon_pt_fraction",
    "electron_pt_fraction",
    "muon_pt_fraction",
)
DETR_SLOT_PARTICLE_FLOW_SUMMARY_DIM = len(DETR_SLOT_PARTICLE_FLOW_SUMMARY_FEATURE_NAMES)
DETR_SLOT_PARTICLE_CNN_RANK_FEATURE_NAMES = (
    "rank_fraction",
    "log_rank",
    "tail_fraction",
    "is_leading",
    "is_top3",
)
DETR_SLOT_PARTICLE_CNN_RANK_FEATURE_DIM = len(DETR_SLOT_PARTICLE_CNN_RANK_FEATURE_NAMES)
DETR_SLOT_PARTICLE_CNN_FEATURE_NAMES = DETR_SLOT_PARTICLE_FLOW_FEATURE_NAMES + DETR_SLOT_PARTICLE_CNN_RANK_FEATURE_NAMES
DETR_SLOT_PARTICLE_CNN_FEATURE_DIM = len(DETR_SLOT_PARTICLE_CNN_FEATURE_NAMES)


def _maybe_torch():
    if importlib.util.find_spec("torch") is None:
        return None
    import torch

    return torch


def require_torch():
    torch = _maybe_torch()
    if torch is None:  # pragma: no cover - environment dependent
        raise ImportError("DETR slot encoder utilities require PyTorch")
    return torch


if importlib.util.find_spec("torch") is None:  # pragma: no cover - environment dependent
    class _ModuleBase:
        pass
else:
    import torch as _torch

    _ModuleBase = _torch.nn.Module


def _as_float_tensor(value, *, name: str, device=None, dtype=None):
    torch = require_torch()
    if isinstance(value, torch.Tensor):
        tensor = value
        if device is not None:
            tensor = tensor.to(device=device)
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
    else:
        tensor = torch.as_tensor(value, device=device, dtype=dtype or torch.float32)
    if not torch.is_floating_point(tensor):
        tensor = tensor.float()
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"{name} contains non-finite values")
    return tensor


def _as_bool_mask(value, *, name: str, expected_shape: tuple[int, int], device=None):
    torch = require_torch()
    if value is None:
        return torch.ones(expected_shape, dtype=torch.bool, device=device)
    tensor = value.to(device=device) if isinstance(value, torch.Tensor) else torch.as_tensor(value, device=device)
    tensor = tensor.bool()
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {tuple(tensor.shape)}")
    return tensor


def _scalar_float(value) -> float:
    torch = require_torch()
    if isinstance(value, torch.Tensor):
        if int(value.numel()) != 1:
            raise ValueError("expected scalar tensor")
        return float(value.detach().cpu().item())
    return float(value)


def _validate_positive_dim(value: int, *, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _as_positive_int_tuple(value, *, name: str) -> tuple[int, ...]:
    if isinstance(value, int):
        values = (int(value),)
    else:
        values = tuple(int(item) for item in value)
    if not values:
        raise ValueError(f"{name} must contain at least one dimension")
    if any(item <= 0 for item in values):
        raise ValueError(f"{name} must contain only positive dimensions")
    return values


def masked_mean_pool(values, mask):
    """Mean-pool ``[B, N, D]`` values with zeros for empty rows."""

    torch = require_torch()
    values = _as_float_tensor(values, name="values")
    if values.ndim != 3:
        raise ValueError(f"values must have shape [batch, tokens, dim], got {tuple(values.shape)}")
    if int(values.shape[0]) <= 0 or int(values.shape[1]) <= 0 or int(values.shape[2]) <= 0:
        raise ValueError(f"values dimensions must all be positive, got {tuple(values.shape)}")
    mask = _as_bool_mask(
        mask,
        name="mask",
        expected_shape=(int(values.shape[0]), int(values.shape[1])),
        device=values.device,
    )
    weights = mask.to(dtype=values.dtype)
    denom = torch.clamp(weights.sum(dim=1, keepdim=True), min=1.0)
    return (values * weights[:, :, None]).sum(dim=1) / denom


def _ensure_nonempty_attention_rows(values, mask):
    """Return attention-safe values/mask, forcing one dummy token for empty rows."""

    safe_values = values
    safe_mask = mask.bool()
    empty_rows = ~safe_mask.any(dim=1)
    forced_count = int(empty_rows.sum().detach().cpu().item())
    if forced_count > 0:
        safe_values = values.clone()
        safe_mask = safe_mask.clone()
        safe_values[empty_rows, 0, :] = 0.0
        safe_mask[empty_rows, 0] = True
    return safe_values, safe_mask, forced_count


def detr_slot_token_embedding_features(tokens, mask, *, max_abs_eta: float = 5.0):
    """Build stable 15-channel HLT token features for DETR-slot encoders."""

    torch = require_torch()
    tokens = _as_float_tensor(tokens, name="tokens")
    if tokens.ndim != 3:
        raise ValueError(f"tokens must have shape [batch, particles, features], got {tuple(tokens.shape)}")
    if int(tokens.shape[-1]) != RAW_TOKEN_DIM:
        raise ValueError(f"tokens feature dim must be {RAW_TOKEN_DIM}, got {int(tokens.shape[-1])}")
    mask = _as_bool_mask(
        mask,
        name="mask",
        expected_shape=(int(tokens.shape[0]), int(tokens.shape[1])),
        device=tokens.device,
    )
    max_abs_eta = float(max_abs_eta)
    if max_abs_eta <= 0.0:
        raise ValueError("max_abs_eta must be positive")

    pt = torch.clamp(tokens[:, :, 0], min=1.0e-6)
    eta = torch.clamp(tokens[:, :, 1], -max_abs_eta, max_abs_eta)
    phi = tokens[:, :, 2]
    energy = torch.clamp(tokens[:, :, 3], min=1.0e-6)
    pieces = [
        0.2 * torch.log(pt),
        0.2 * torch.log(energy),
        eta / max_abs_eta,
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


def _wrap_phi(phi):
    torch = require_torch()
    return torch.atan2(torch.sin(phi), torch.cos(phi))


def detr_slot_particle_net_coordinates(tokens, mask, *, max_abs_eta: float = 5.0):
    """Return physical kNN coordinates ``[eta, phi, log_pt]`` for PN encoders."""

    torch = require_torch()
    tokens = _as_float_tensor(tokens, name="tokens")
    if tokens.ndim != 3:
        raise ValueError(f"tokens must have shape [batch, particles, features], got {tuple(tokens.shape)}")
    if int(tokens.shape[-1]) != RAW_TOKEN_DIM:
        raise ValueError(f"tokens feature dim must be {RAW_TOKEN_DIM}, got {int(tokens.shape[-1])}")
    mask = _as_bool_mask(
        mask,
        name="mask",
        expected_shape=(int(tokens.shape[0]), int(tokens.shape[1])),
        device=tokens.device,
    )
    max_abs_eta = float(max_abs_eta)
    if max_abs_eta <= 0.0:
        raise ValueError("max_abs_eta must be positive")

    pt = torch.clamp(tokens[:, :, 0], min=1.0e-6)
    coords = torch.stack(
        [
            torch.clamp(tokens[:, :, 1], -max_abs_eta, max_abs_eta),
            _wrap_phi(tokens[:, :, 2]),
            torch.log(pt),
        ],
        dim=-1,
    )
    return torch.where(mask[:, :, None], coords, torch.zeros_like(coords))


def _pairwise_particle_net_distance(coords):
    diff = coords[:, :, None, :] - coords[:, None, :, :]
    diff = diff.clone()
    diff[:, :, :, 1] = _wrap_phi(diff[:, :, :, 1])
    return (diff * diff).sum(dim=-1)


def _masked_knn_indices(coords, mask, k: int):
    """Return nearest valid neighbor indices with shape ``[B, N, k]``."""

    torch = require_torch()
    coords = _as_float_tensor(coords, name="coords")
    if coords.ndim != 3:
        raise ValueError(f"coords must have shape [batch, particles, dims], got {tuple(coords.shape)}")
    mask = _as_bool_mask(
        mask,
        name="mask",
        expected_shape=(int(coords.shape[0]), int(coords.shape[1])),
        device=coords.device,
    )
    k = int(k)
    if k <= 0:
        raise ValueError("k must be positive")
    batch_size, num_particles, _ = coords.shape
    if int(num_particles) == 0:
        return torch.empty(batch_size, 0, k, dtype=torch.long, device=coords.device)

    finite_coords = torch.isfinite(coords).all(dim=-1)
    valid_candidates = mask & finite_coords
    distances = _pairwise_particle_net_distance(coords)
    large = torch.finfo(distances.dtype).max / 16.0
    distances = distances.masked_fill(~valid_candidates[:, None, :], large)

    topk_count = min(k, int(num_particles))
    _, indices = torch.topk(distances, k=topk_count, dim=-1, largest=False, sorted=True)
    selected_valid = torch.gather(
        valid_candidates[:, None, :].expand(-1, num_particles, -1),
        dim=2,
        index=indices,
    )
    first_index = indices[:, :, :1]
    indices = torch.where(selected_valid, indices, first_index.expand_as(indices))
    if topk_count < k:
        pad = indices[:, :, -1:].expand(-1, -1, k - topk_count)
        indices = torch.cat([indices, pad], dim=2)
    has_valid_candidate = valid_candidates.any(dim=1)
    return torch.where(has_valid_candidate[:, None, None], indices, torch.zeros_like(indices)).long()


def _gather_neighbor_features(features, indices):
    torch = require_torch()
    features = _as_float_tensor(features, name="features")
    if features.ndim != 3:
        raise ValueError(f"features must have shape [batch, particles, channels], got {tuple(features.shape)}")
    if indices.ndim != 3:
        raise ValueError(f"indices must have shape [batch, particles, neighbors], got {tuple(indices.shape)}")
    if tuple(features.shape[:2]) != tuple(indices.shape[:2]):
        raise ValueError(f"features/indices leading shapes differ: {tuple(features.shape[:2])} vs {tuple(indices.shape[:2])}")
    if bool((indices < 0).any()) or bool((indices >= int(features.shape[1])).any()):
        raise IndexError("neighbor indices are out of range for features")

    batch_size, num_particles, channels = features.shape
    _, _, num_neighbors = indices.shape
    expanded = features[:, None, :, :].expand(-1, num_particles, -1, -1)
    gather_index = indices[:, :, :, None].expand(-1, -1, -1, channels)
    return torch.gather(expanded, dim=2, index=gather_index)


def _masked_max_pool(values, mask):
    torch = require_torch()
    values = _as_float_tensor(values, name="values")
    if values.ndim != 3:
        raise ValueError(f"values must have shape [batch, tokens, dim], got {tuple(values.shape)}")
    mask = _as_bool_mask(
        mask,
        name="mask",
        expected_shape=(int(values.shape[0]), int(values.shape[1])),
        device=values.device,
    )
    very_negative = torch.finfo(values.dtype).min / 8.0
    masked = values.masked_fill(~mask[:, :, None], very_negative)
    pooled = masked.max(dim=1).values
    has_valid = mask.any(dim=1)
    return torch.where(has_valid[:, None], pooled, torch.zeros_like(pooled))


def _masked_sum_pool(values, mask):
    values = _as_float_tensor(values, name="values")
    if values.ndim != 3:
        raise ValueError(f"values must have shape [batch, tokens, dim], got {tuple(values.shape)}")
    mask = _as_bool_mask(
        mask,
        name="mask",
        expected_shape=(int(values.shape[0]), int(values.shape[1])),
        device=values.device,
    )
    weights = mask.to(dtype=values.dtype)
    return (values * weights[:, :, None]).sum(dim=1)


def detr_slot_particle_flow_features(tokens, mask, *, max_abs_eta: float = 5.0):
    """Build PFN/DeepSets per-particle energy-flow features."""

    torch = require_torch()
    tokens = _as_float_tensor(tokens, name="tokens")
    if tokens.ndim != 3:
        raise ValueError(f"tokens must have shape [batch, particles, features], got {tuple(tokens.shape)}")
    if int(tokens.shape[-1]) != RAW_TOKEN_DIM:
        raise ValueError(f"tokens feature dim must be {RAW_TOKEN_DIM}, got {int(tokens.shape[-1])}")
    mask = _as_bool_mask(
        mask,
        name="mask",
        expected_shape=(int(tokens.shape[0]), int(tokens.shape[1])),
        device=tokens.device,
    )
    max_abs_eta = float(max_abs_eta)
    if max_abs_eta <= 0.0:
        raise ValueError("max_abs_eta must be positive")

    mask_float = mask.to(dtype=tokens.dtype)
    pt = torch.clamp(tokens[:, :, 0], min=1.0e-6)
    eta = torch.clamp(tokens[:, :, 1], -max_abs_eta, max_abs_eta)
    phi = _wrap_phi(tokens[:, :, 2])
    energy = torch.clamp(tokens[:, :, 3], min=1.0e-6)
    sum_pt = torch.clamp((pt * mask_float).sum(dim=1, keepdim=True), min=1.0e-6)
    sum_energy = torch.clamp((energy * mask_float).sum(dim=1, keepdim=True), min=1.0e-6)

    pieces = [
        0.2 * torch.log(pt),
        0.2 * torch.log(energy),
        eta / max_abs_eta,
        torch.sin(phi),
        torch.cos(phi),
        torch.log(pt / sum_pt),
        torch.log(energy / sum_energy),
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
        mask_float,
    ]
    features = torch.stack(pieces, dim=-1)
    return torch.where(mask[:, :, None], features, torch.zeros_like(features))


def detr_slot_particle_flow_summary_features(tokens, mask, *, max_abs_eta: float = 5.0):
    """Build coarse permutation-invariant HLT jet summary features for PFN context."""

    torch = require_torch()
    tokens = _as_float_tensor(tokens, name="tokens")
    if tokens.ndim != 3:
        raise ValueError(f"tokens must have shape [batch, particles, features], got {tuple(tokens.shape)}")
    if int(tokens.shape[-1]) != RAW_TOKEN_DIM:
        raise ValueError(f"tokens feature dim must be {RAW_TOKEN_DIM}, got {int(tokens.shape[-1])}")
    mask = _as_bool_mask(
        mask,
        name="mask",
        expected_shape=(int(tokens.shape[0]), int(tokens.shape[1])),
        device=tokens.device,
    )
    max_abs_eta = float(max_abs_eta)
    if max_abs_eta <= 0.0:
        raise ValueError("max_abs_eta must be positive")

    mask_float = mask.to(dtype=tokens.dtype)
    pt = torch.clamp(tokens[:, :, 0], min=1.0e-6)
    eta = torch.clamp(tokens[:, :, 1], -max_abs_eta, max_abs_eta)
    phi = _wrap_phi(tokens[:, :, 2])
    energy = torch.clamp(tokens[:, :, 3], min=1.0e-6)

    valid_count = mask_float.sum(dim=1)
    has_valid = valid_count > 0
    weighted_pt = pt * mask_float
    weighted_energy = energy * mask_float
    total_pt = weighted_pt.sum(dim=1)
    total_energy = weighted_energy.sum(dim=1)
    safe_total_pt = torch.clamp(total_pt, min=1.0e-6)
    safe_total_energy = torch.clamp(total_energy, min=1.0e-6)
    safe_count = torch.clamp(valid_count, min=1.0)

    pt_weighted_eta = (weighted_pt * eta).sum(dim=1) / safe_total_pt / max_abs_eta
    pt_weighted_sin_phi = (weighted_pt * torch.sin(phi)).sum(dim=1) / safe_total_pt
    pt_weighted_cos_phi = (weighted_pt * torch.cos(phi)).sum(dim=1) / safe_total_pt
    mean_abs_eta = (mask_float * eta.abs()).sum(dim=1) / safe_count / max_abs_eta

    pid_pt_fractions = []
    for column in range(5, 10):
        pid = torch.clamp(tokens[:, :, column], 0.0, 1.0)
        pid_pt_fractions.append((weighted_pt * pid).sum(dim=1) / safe_total_pt)

    pieces = [
        torch.where(has_valid, 0.2 * torch.log(safe_total_pt), torch.zeros_like(total_pt)),
        torch.where(has_valid, 0.2 * torch.log(safe_total_energy), torch.zeros_like(total_energy)),
        torch.log1p(valid_count),
        torch.where(has_valid, pt_weighted_eta, torch.zeros_like(pt_weighted_eta)),
        torch.where(has_valid, pt_weighted_sin_phi, torch.zeros_like(pt_weighted_sin_phi)),
        torch.where(has_valid, pt_weighted_cos_phi, torch.zeros_like(pt_weighted_cos_phi)),
        torch.where(has_valid, mean_abs_eta, torch.zeros_like(mean_abs_eta)),
        *[torch.where(has_valid, fraction, torch.zeros_like(fraction)) for fraction in pid_pt_fractions],
    ]
    return torch.stack(pieces, dim=-1)


def detr_slot_particle_cnn_rank_features(mask):
    """Build fixed-cache-rank features for the DETR-slot P-CNN encoder."""

    torch = require_torch()
    if not isinstance(mask, torch.Tensor):
        mask = torch.as_tensor(mask)
    if mask.ndim != 2:
        raise ValueError(f"mask must have shape [batch, particles], got {tuple(mask.shape)}")
    mask = mask.bool()
    batch_size = int(mask.shape[0])
    num_particles = int(mask.shape[1])
    if batch_size <= 0 or num_particles <= 0:
        raise ValueError(f"mask dimensions must be positive, got {tuple(mask.shape)}")

    dtype = torch.float32
    device = mask.device
    rank = torch.arange(num_particles, dtype=dtype, device=device)
    rank_fraction = rank / max(num_particles - 1, 1)
    log_rank = torch.log1p(rank) / torch.log1p(torch.as_tensor(max(num_particles - 1, 1), dtype=dtype, device=device))
    tail_fraction = 1.0 - rank_fraction
    is_leading = (rank == 0).to(dtype)
    is_top3 = (rank < 3).to(dtype)
    features = torch.stack([rank_fraction, log_rank, tail_fraction, is_leading, is_top3], dim=-1)
    features = features[None, :, :].expand(batch_size, -1, -1)
    return torch.where(mask[:, :, None], features, torch.zeros_like(features))


def detr_slot_particle_cnn_features(tokens, mask, *, max_abs_eta: float = 5.0):
    """Build ordered rank-convolution features for the DETR-slot P-CNN encoder."""

    flow_features = detr_slot_particle_flow_features(tokens, mask, max_abs_eta=max_abs_eta)
    rank_features = detr_slot_particle_cnn_rank_features(mask).to(
        device=flow_features.device,
        dtype=flow_features.dtype,
    )
    return torch.cat([flow_features, rank_features], dim=-1)


@dataclass
class EncoderOutput:
    """Validated HLT encoder output consumed by the shared DETR decoder.

    Shapes are intentionally architecture-neutral:

    ```text
    memory_tokens:  [B, M, D]
    memory_mask:    [B, M]
    global_context: [B, C] or None
    ```

    ``memory_mask`` marks valid memory tokens.  Empty valid-memory rows are
    allowed here so the adapter can report them; the decoder later forces a
    safe dummy token for those rows to keep attention finite.
    """

    memory_tokens: Any
    memory_mask: Any
    global_context: Any | None = None
    aux: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        memory_tokens = _as_float_tensor(self.memory_tokens, name="memory_tokens")
        if memory_tokens.ndim != 3:
            raise ValueError(
                f"memory_tokens must have shape [batch, memory, dim], got {tuple(memory_tokens.shape)}"
            )
        if (
            int(memory_tokens.shape[0]) <= 0
            or int(memory_tokens.shape[1]) <= 0
            or int(memory_tokens.shape[2]) <= 0
        ):
            raise ValueError(f"memory_tokens dimensions must all be positive, got {tuple(memory_tokens.shape)}")
        memory_mask = _as_bool_mask(
            self.memory_mask,
            name="memory_mask",
            expected_shape=(int(memory_tokens.shape[0]), int(memory_tokens.shape[1])),
            device=memory_tokens.device,
        )
        global_context = None
        if self.global_context is not None:
            global_context = _as_float_tensor(
                self.global_context,
                name="global_context",
                device=memory_tokens.device,
                dtype=memory_tokens.dtype,
            )
            if global_context.ndim != 2:
                raise ValueError(
                    f"global_context must have shape [batch, dim], got {tuple(global_context.shape)}"
                )
            if int(global_context.shape[0]) != int(memory_tokens.shape[0]):
                raise ValueError(
                    "global_context batch dimension does not match memory_tokens: "
                    f"{int(global_context.shape[0])} vs {int(memory_tokens.shape[0])}"
                )
            if int(global_context.shape[1]) <= 0:
                raise ValueError(f"global_context feature dimension must be positive, got {tuple(global_context.shape)}")

        self.memory_tokens = memory_tokens
        self.memory_mask = memory_mask
        self.global_context = global_context
        self.aux = dict(self.aux or {})

    @property
    def batch_size(self) -> int:
        return int(self.memory_tokens.shape[0])

    @property
    def memory_size(self) -> int:
        return int(self.memory_tokens.shape[1])

    @property
    def memory_dim(self) -> int:
        return int(self.memory_tokens.shape[2])

    @property
    def has_global_context(self) -> bool:
        return self.global_context is not None

    @property
    def context_dim(self) -> int:
        if self.global_context is None:
            return 0
        return int(self.global_context.shape[1])

    @property
    def device(self):
        return self.memory_tokens.device

    @property
    def dtype(self):
        return self.memory_tokens.dtype

    def shape_report(self) -> dict[str, Any]:
        return {
            "memory_tokens_shape": list(self.memory_tokens.shape),
            "memory_mask_shape": list(self.memory_mask.shape),
            "global_context_shape": None if self.global_context is None else list(self.global_context.shape),
            "batch_size": self.batch_size,
            "memory_size": self.memory_size,
            "memory_dim": self.memory_dim,
            "context_dim": self.context_dim,
            "has_global_context": self.has_global_context,
            "contract": DETR_SLOT_ENCODER_OUTPUT_CONTRACT,
        }

    def detached_float_diagnostics(self) -> dict[str, float]:
        torch = require_torch()
        mask = self.memory_mask.detach()
        valid_counts = mask.sum(dim=1).to(dtype=self.memory_tokens.dtype)
        diagnostics = {
            "memory_size": float(self.memory_size),
            "memory_dim": float(self.memory_dim),
            "memory_valid_fraction": _scalar_float(mask.float().mean()),
            "memory_valid_count_mean": _scalar_float(valid_counts.mean()),
            "memory_valid_count_min": _scalar_float(valid_counts.min()),
            "memory_valid_count_max": _scalar_float(valid_counts.max()),
            "global_context_dim": float(self.context_dim),
            "has_global_context": float(self.has_global_context),
        }
        if self.aux:
            for key, value in self.aux.items():
                if isinstance(value, (int, float, bool)):
                    diagnostics[f"aux_{key}"] = float(value)
                elif isinstance(value, torch.Tensor) and int(value.numel()) == 1:
                    diagnostics[f"aux_{key}"] = _scalar_float(value)
        if not torch.isfinite(torch.as_tensor(list(diagnostics.values()), dtype=torch.float32)).all():
            raise FloatingPointError("DETR encoder diagnostics contain non-finite values")
        return diagnostics


def validate_encoder_output(output: EncoderOutput) -> EncoderOutput:
    """Return a freshly validated encoder output."""

    if isinstance(output, EncoderOutput):
        return EncoderOutput(
            memory_tokens=output.memory_tokens,
            memory_mask=output.memory_mask,
            global_context=output.global_context,
            aux=output.aux,
        )
    raise TypeError(f"Expected EncoderOutput, got {type(output).__name__}")


class BaseHLTEncoderAdapter(_ModuleBase):
    """Base interface for architecture-specific HLT encoders."""

    def __init__(self, *, input_dim: int = RAW_TOKEN_DIM, memory_dim: int = 128, context_dim: int | None = None) -> None:
        require_torch()
        super().__init__()
        self.input_dim = _validate_positive_dim(input_dim, name="input_dim")
        self.memory_dim = _validate_positive_dim(memory_dim, name="memory_dim")
        self.context_dim = self.memory_dim if context_dim is None else _validate_positive_dim(context_dim, name="context_dim")

    def validate_hlt_inputs(self, hlt_tokens, hlt_mask=None):
        tokens = _as_float_tensor(hlt_tokens, name="hlt_tokens")
        if tokens.ndim != 3:
            raise ValueError(f"hlt_tokens must have shape [batch, particles, features], got {tuple(tokens.shape)}")
        if int(tokens.shape[0]) <= 0 or int(tokens.shape[1]) <= 0 or int(tokens.shape[2]) <= 0:
            raise ValueError(f"hlt_tokens dimensions must all be positive, got {tuple(tokens.shape)}")
        if int(tokens.shape[2]) != int(self.input_dim):
            raise ValueError(f"HLT feature dim {int(tokens.shape[2])} != configured input_dim {int(self.input_dim)}")
        mask = _as_bool_mask(
            hlt_mask,
            name="hlt_mask",
            expected_shape=(int(tokens.shape[0]), int(tokens.shape[1])),
            device=tokens.device,
        )
        return tokens, mask

    def masked_mean_pool(self, values, mask):
        return masked_mean_pool(values, mask)

    def forward(self, hlt_tokens, hlt_mask=None) -> EncoderOutput:  # pragma: no cover - interface only
        raise NotImplementedError("HLT encoder adapters must implement forward")


@dataclass(frozen=True)
class DummyHLTEncoderConfig:
    """Config for a tiny test-only HLT encoder adapter."""

    input_dim: int = RAW_TOKEN_DIM
    memory_dim: int = 128
    hidden_dim: int | None = None
    dropout: float = 0.0
    use_layer_norm: bool = True

    def __post_init__(self) -> None:
        _validate_positive_dim(self.input_dim, name="input_dim")
        _validate_positive_dim(self.memory_dim, name="memory_dim")
        if self.hidden_dim is not None:
            _validate_positive_dim(self.hidden_dim, name="hidden_dim")
        if float(self.dropout) < 0.0 or float(self.dropout) >= 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @property
    def resolved_hidden_dim(self) -> int:
        return int(self.memory_dim if self.hidden_dim is None else self.hidden_dim)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["resolved_hidden_dim"] = int(self.resolved_hidden_dim)
        return payload


class DummyHLTEncoderAdapter(BaseHLTEncoderAdapter):
    """Small deterministic encoder used to test the shared DETR interface."""

    def __init__(self, config: DummyHLTEncoderConfig | Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        torch = require_torch()
        if isinstance(config, DummyHLTEncoderConfig):
            if kwargs:
                raise ValueError("do not pass kwargs when config is already a DummyHLTEncoderConfig")
            self.config = config
        else:
            payload = dict(config or {})
            payload.update(kwargs)
            self.config = DummyHLTEncoderConfig(**payload)
        super().__init__(
            input_dim=int(self.config.input_dim),
            memory_dim=int(self.config.memory_dim),
            context_dim=int(self.config.memory_dim),
        )
        layers = []
        if bool(self.config.use_layer_norm):
            layers.append(torch.nn.LayerNorm(int(self.config.input_dim)))
        layers.extend(
            [
                torch.nn.Linear(int(self.config.input_dim), int(self.config.resolved_hidden_dim)),
                torch.nn.GELU(),
            ]
        )
        if float(self.config.dropout) > 0.0:
            layers.append(torch.nn.Dropout(float(self.config.dropout)))
        layers.append(torch.nn.Linear(int(self.config.resolved_hidden_dim), int(self.config.memory_dim)))
        if bool(self.config.use_layer_norm):
            layers.append(torch.nn.LayerNorm(int(self.config.memory_dim)))
        self.projection = torch.nn.Sequential(*layers)

    def forward(self, hlt_tokens, hlt_mask=None) -> EncoderOutput:
        torch = require_torch()
        tokens, mask = self.validate_hlt_inputs(hlt_tokens, hlt_mask)
        memory_tokens = self.projection(tokens)
        memory_tokens = torch.where(mask[:, :, None], memory_tokens, torch.zeros_like(memory_tokens))
        global_context = self.masked_mean_pool(memory_tokens, mask)
        output = EncoderOutput(
            memory_tokens=memory_tokens,
            memory_mask=mask,
            global_context=global_context,
            aux={
                "encoder_step": DETR_SLOT_ENCODER_STEP,
                "dummy_encoder": True,
                "input_dim": float(self.config.input_dim),
                "memory_dim": float(self.config.memory_dim),
            },
        )
        return validate_encoder_output(output)


@dataclass(frozen=True)
class GlobalTransformerHLTEncoderConfig:
    """Configuration for the DETR-slot ParT-ish/global Transformer encoder."""

    input_dim: int = RAW_TOKEN_DIM
    memory_dim: int = 128
    context_dim: int | None = None
    num_layers: int = 4
    num_heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.05
    max_abs_eta: float = 5.0
    use_layer_norm: bool = True
    norm_first: bool = True

    def __post_init__(self) -> None:
        _validate_positive_dim(self.input_dim, name="input_dim")
        if int(self.input_dim) != RAW_TOKEN_DIM:
            raise ValueError(f"input_dim must be {RAW_TOKEN_DIM}, got {self.input_dim}")
        _validate_positive_dim(self.memory_dim, name="memory_dim")
        if self.context_dim is not None:
            _validate_positive_dim(self.context_dim, name="context_dim")
        _validate_positive_dim(self.num_layers, name="num_layers")
        _validate_positive_dim(self.num_heads, name="num_heads")
        if int(self.memory_dim) % int(self.num_heads) != 0:
            raise ValueError("memory_dim must be divisible by num_heads")
        if float(self.mlp_ratio) <= 0.0:
            raise ValueError("mlp_ratio must be positive")
        if float(self.dropout) < 0.0 or float(self.dropout) >= 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if float(self.max_abs_eta) <= 0.0:
            raise ValueError("max_abs_eta must be positive")

    @property
    def resolved_context_dim(self) -> int:
        return int(self.memory_dim if self.context_dim is None else self.context_dim)

    @property
    def dim_feedforward(self) -> int:
        return max(int(round(float(self.mlp_ratio) * int(self.memory_dim))), int(self.memory_dim))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["resolved_context_dim"] = int(self.resolved_context_dim)
        payload["dim_feedforward"] = int(self.dim_feedforward)
        payload["token_embed_feature_dim"] = int(DETR_SLOT_TOKEN_EMBED_FEATURE_DIM)
        return payload


class GlobalTransformerHLTEncoderAdapter(BaseHLTEncoderAdapter):
    """ParT-ish Transformer HLT encoder for DETR/free-slot reconstruction."""

    def __init__(self, config: GlobalTransformerHLTEncoderConfig | Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        torch = require_torch()
        if isinstance(config, GlobalTransformerHLTEncoderConfig):
            if kwargs:
                raise ValueError("do not pass kwargs when config is already a GlobalTransformerHLTEncoderConfig")
            self.config = config
        else:
            payload = dict(config or {})
            payload.update(kwargs)
            self.config = GlobalTransformerHLTEncoderConfig(**payload)
        super().__init__(
            input_dim=int(self.config.input_dim),
            memory_dim=int(self.config.memory_dim),
            context_dim=int(self.config.resolved_context_dim),
        )

        input_layers = []
        if bool(self.config.use_layer_norm):
            input_layers.append(torch.nn.LayerNorm(DETR_SLOT_TOKEN_EMBED_FEATURE_DIM))
        input_layers.extend(
            [
                torch.nn.Linear(DETR_SLOT_TOKEN_EMBED_FEATURE_DIM, int(self.config.memory_dim)),
                torch.nn.GELU(),
                torch.nn.Dropout(float(self.config.dropout)),
            ]
        )
        self.input_projection = torch.nn.Sequential(*input_layers)

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=int(self.config.memory_dim),
            nhead=int(self.config.num_heads),
            dim_feedforward=int(self.config.dim_feedforward),
            dropout=float(self.config.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=bool(self.config.norm_first),
        )
        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(self.config.num_layers),
            norm=torch.nn.LayerNorm(int(self.config.memory_dim)),
        )
        if int(self.config.resolved_context_dim) == int(self.config.memory_dim):
            self.context_projection = torch.nn.Identity()
        else:
            self.context_projection = torch.nn.Sequential(
                torch.nn.LayerNorm(int(self.config.memory_dim)),
                torch.nn.Linear(int(self.config.memory_dim), int(self.config.resolved_context_dim)),
            )

    def forward(self, hlt_tokens, hlt_mask=None) -> EncoderOutput:
        torch = require_torch()
        tokens, mask = self.validate_hlt_inputs(hlt_tokens, hlt_mask)
        features = detr_slot_token_embedding_features(tokens, mask, max_abs_eta=float(self.config.max_abs_eta))
        safe_features, safe_attention_mask, forced_count = _ensure_nonempty_attention_rows(features, mask)

        projected = self.input_projection(safe_features)
        encoded = self.encoder(projected, src_key_padding_mask=~safe_attention_mask)
        encoded = torch.where(mask[:, :, None], encoded, torch.zeros_like(encoded))
        pooled = self.masked_mean_pool(encoded, mask)
        global_context = self.context_projection(pooled)
        has_valid_token = mask.any(dim=1)
        global_context = torch.where(has_valid_token[:, None], global_context, torch.zeros_like(global_context))

        output = EncoderOutput(
            memory_tokens=encoded,
            memory_mask=mask,
            global_context=global_context,
            aux={
                "encoder_step": DETR_SLOT_GLOBAL_TRANSFORMER_ENCODER_STEP,
                "global_transformer_encoder": True,
                "input_dim": float(self.config.input_dim),
                "memory_dim": float(self.config.memory_dim),
                "context_dim": float(self.config.resolved_context_dim),
                "num_layers": float(self.config.num_layers),
                "num_heads": float(self.config.num_heads),
                "forced_nonempty_attention_rows": float(forced_count),
            },
        )
        return validate_encoder_output(output)


class DetrParticleNetEdgeConvBlock(_ModuleBase):
    """Small masked EdgeConv block for DETR-slot ParticleNet encoding."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        k: int = 16,
        hidden_dim: int | None = None,
        dropout: float = 0.05,
        residual: bool = True,
    ) -> None:
        torch = require_torch()
        super().__init__()
        self.input_dim = _validate_positive_dim(input_dim, name="input_dim")
        self.output_dim = _validate_positive_dim(output_dim, name="output_dim")
        self.k = _validate_positive_dim(k, name="k")
        self.hidden_dim = _validate_positive_dim(
            hidden_dim if hidden_dim is not None else max(self.input_dim, self.output_dim),
            name="hidden_dim",
        )
        self.dropout = float(dropout)
        self.use_residual = bool(residual)
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * self.input_dim, self.hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(self.hidden_dim, self.output_dim),
            torch.nn.GELU(),
        )
        self.residual_projection = None
        if self.use_residual:
            if self.input_dim == self.output_dim:
                self.residual_projection = torch.nn.Identity()
            else:
                self.residual_projection = torch.nn.Linear(self.input_dim, self.output_dim)
        self.output_norm = torch.nn.LayerNorm(self.output_dim)

    def forward(self, features, coords, mask):
        torch = require_torch()
        features = _as_float_tensor(features, name="features")
        coords = _as_float_tensor(coords, name="coords", device=features.device, dtype=features.dtype)
        if features.ndim != 3:
            raise ValueError(f"features must have shape [batch, particles, channels], got {tuple(features.shape)}")
        if coords.ndim != 3:
            raise ValueError(f"coords must have shape [batch, particles, dims], got {tuple(coords.shape)}")
        if tuple(features.shape[:2]) != tuple(coords.shape[:2]):
            raise ValueError(f"features/coords leading shapes differ: {tuple(features.shape[:2])} vs {tuple(coords.shape[:2])}")
        if int(features.shape[-1]) != int(self.input_dim):
            raise ValueError(f"features last dim must be {self.input_dim}, got {int(features.shape[-1])}")
        mask = _as_bool_mask(
            mask,
            name="mask",
            expected_shape=(int(features.shape[0]), int(features.shape[1])),
            device=features.device,
        )
        features = torch.where(mask[:, :, None], features, torch.zeros_like(features))

        indices = _masked_knn_indices(coords, mask, self.k)
        neighbors = _gather_neighbor_features(features, indices)
        centers = features[:, :, None, :].expand_as(neighbors)
        edge_input = torch.cat([centers, neighbors - centers], dim=-1)
        edge_features = self.edge_mlp(edge_input)
        aggregated = edge_features.max(dim=2).values
        if self.residual_projection is not None:
            aggregated = aggregated + self.residual_projection(features)
        aggregated = self.output_norm(aggregated)
        return torch.where(mask[:, :, None], aggregated, torch.zeros_like(aggregated))


@dataclass(frozen=True)
class ParticleNetHLTEncoderConfig:
    """Configuration for the DETR-slot ParticleNet/EdgeConv HLT encoder."""

    input_dim: int = RAW_TOKEN_DIM
    memory_dim: int = 128
    context_dim: int | None = None
    edgeconv_dims: tuple[int, ...] = (64, 128, 128)
    k: int = 16
    dropout: float = 0.05
    max_abs_eta: float = 5.0

    def __post_init__(self) -> None:
        _validate_positive_dim(self.input_dim, name="input_dim")
        if int(self.input_dim) != RAW_TOKEN_DIM:
            raise ValueError(f"input_dim must be {RAW_TOKEN_DIM}, got {self.input_dim}")
        _validate_positive_dim(self.memory_dim, name="memory_dim")
        if self.context_dim is not None:
            _validate_positive_dim(self.context_dim, name="context_dim")
        edgeconv_dims = tuple(int(dim) for dim in self.edgeconv_dims)
        if not edgeconv_dims:
            raise ValueError("edgeconv_dims must contain at least one dimension")
        if any(dim <= 0 for dim in edgeconv_dims):
            raise ValueError("edgeconv_dims must all be positive")
        _validate_positive_dim(self.k, name="k")
        if float(self.dropout) < 0.0 or float(self.dropout) >= 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if float(self.max_abs_eta) <= 0.0:
            raise ValueError("max_abs_eta must be positive")
        object.__setattr__(self, "edgeconv_dims", edgeconv_dims)

    @property
    def resolved_context_dim(self) -> int:
        return int(self.memory_dim if self.context_dim is None else self.context_dim)

    @property
    def edgeconv_output_dim(self) -> int:
        return int(self.edgeconv_dims[-1])

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["edgeconv_dims"] = list(self.edgeconv_dims)
        payload["edgeconv_output_dim"] = int(self.edgeconv_output_dim)
        payload["resolved_context_dim"] = int(self.resolved_context_dim)
        payload["token_embed_feature_dim"] = int(DETR_SLOT_TOKEN_EMBED_FEATURE_DIM)
        payload["coord_dim"] = int(DETR_SLOT_PARTICLE_NET_COORD_DIM)
        return payload


class ParticleNetHLTEncoderAdapter(BaseHLTEncoderAdapter):
    """ParticleNet-style masked EdgeConv HLT encoder for DETR/free slots."""

    def __init__(self, config: ParticleNetHLTEncoderConfig | Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        torch = require_torch()
        if isinstance(config, ParticleNetHLTEncoderConfig):
            if kwargs:
                raise ValueError("do not pass kwargs when config is already a ParticleNetHLTEncoderConfig")
            self.config = config
        else:
            payload = dict(config or {})
            payload.update(kwargs)
            if "edgeconv_dims" in payload:
                payload["edgeconv_dims"] = tuple(payload["edgeconv_dims"])
            self.config = ParticleNetHLTEncoderConfig(**payload)
        super().__init__(
            input_dim=int(self.config.input_dim),
            memory_dim=int(self.config.memory_dim),
            context_dim=int(self.config.resolved_context_dim),
        )

        dims = (DETR_SLOT_TOKEN_EMBED_FEATURE_DIM,) + tuple(int(dim) for dim in self.config.edgeconv_dims)
        self.edgeconv_blocks = torch.nn.ModuleList(
            [
                DetrParticleNetEdgeConvBlock(
                    dims[index],
                    dims[index + 1],
                    k=int(self.config.k),
                    hidden_dim=max(dims[index], dims[index + 1]),
                    dropout=float(self.config.dropout),
                    residual=True,
                )
                for index in range(len(self.config.edgeconv_dims))
            ]
        )
        if int(self.config.edgeconv_output_dim) == int(self.config.memory_dim):
            self.memory_projection = torch.nn.Identity()
        else:
            self.memory_projection = torch.nn.Sequential(
                torch.nn.LayerNorm(int(self.config.edgeconv_output_dim)),
                torch.nn.Linear(int(self.config.edgeconv_output_dim), int(self.config.memory_dim)),
            )
        context_input_dim = 2 * int(self.config.memory_dim)
        if int(self.config.resolved_context_dim) == context_input_dim:
            self.context_projection = torch.nn.Identity()
        else:
            self.context_projection = torch.nn.Sequential(
                torch.nn.LayerNorm(context_input_dim),
                torch.nn.Linear(context_input_dim, int(self.config.resolved_context_dim)),
            )

    def forward(self, hlt_tokens, hlt_mask=None) -> EncoderOutput:
        torch = require_torch()
        tokens, mask = self.validate_hlt_inputs(hlt_tokens, hlt_mask)
        features = detr_slot_token_embedding_features(tokens, mask, max_abs_eta=float(self.config.max_abs_eta))
        coords = detr_slot_particle_net_coordinates(tokens, mask, max_abs_eta=float(self.config.max_abs_eta))

        encoded = features
        for block in self.edgeconv_blocks:
            encoded = block(encoded, coords, mask)
        memory_tokens = self.memory_projection(encoded)
        memory_tokens = torch.where(mask[:, :, None], memory_tokens, torch.zeros_like(memory_tokens))

        pooled_mean = self.masked_mean_pool(memory_tokens, mask)
        pooled_max = _masked_max_pool(memory_tokens, mask)
        global_context = self.context_projection(torch.cat([pooled_mean, pooled_max], dim=-1))
        has_valid_token = mask.any(dim=1)
        global_context = torch.where(has_valid_token[:, None], global_context, torch.zeros_like(global_context))

        output = EncoderOutput(
            memory_tokens=memory_tokens,
            memory_mask=mask,
            global_context=global_context,
            aux={
                "encoder_step": DETR_SLOT_PARTICLE_NET_ENCODER_STEP,
                "particle_net_encoder": True,
                "input_dim": float(self.config.input_dim),
                "memory_dim": float(self.config.memory_dim),
                "context_dim": float(self.config.resolved_context_dim),
                "edgeconv_blocks": float(len(self.edgeconv_blocks)),
                "k": float(self.config.k),
            },
        )
        return validate_encoder_output(output)


def _make_feedforward_network(input_dim: int, hidden_dims: tuple[int, ...], *, dropout: float):
    torch = require_torch()
    input_dim = _validate_positive_dim(input_dim, name="input_dim")
    hidden_dims = _as_positive_int_tuple(hidden_dims, name="hidden_dims")
    dropout = float(dropout)
    if dropout < 0.0 or dropout >= 1.0:
        raise ValueError("dropout must be in [0, 1)")
    layers = [torch.nn.LayerNorm(input_dim)]
    current_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(torch.nn.Linear(current_dim, int(hidden_dim)))
        layers.append(torch.nn.GELU())
        if dropout > 0.0:
            layers.append(torch.nn.Dropout(dropout))
        current_dim = int(hidden_dim)
    layers.append(torch.nn.LayerNorm(current_dim))
    return torch.nn.Sequential(*layers)


@dataclass(frozen=True)
class ParticleFlowHLTEncoderConfig:
    """Configuration for the DETR-slot PFN/DeepSets HLT encoder."""

    input_dim: int = RAW_TOKEN_DIM
    memory_dim: int = 128
    context_dim: int | None = None
    phi_dims: tuple[int, ...] = (128, 128, 128)
    context_mlp_dims: tuple[int, ...] = (256, 256)
    dropout: float = 0.05
    max_abs_eta: float = 5.0
    broadcast_context_to_memory: bool = True

    def __post_init__(self) -> None:
        _validate_positive_dim(self.input_dim, name="input_dim")
        if int(self.input_dim) != RAW_TOKEN_DIM:
            raise ValueError(f"input_dim must be {RAW_TOKEN_DIM}, got {self.input_dim}")
        _validate_positive_dim(self.memory_dim, name="memory_dim")
        if self.context_dim is not None:
            _validate_positive_dim(self.context_dim, name="context_dim")
        phi_dims = _as_positive_int_tuple(self.phi_dims, name="phi_dims")
        context_mlp_dims = _as_positive_int_tuple(self.context_mlp_dims, name="context_mlp_dims")
        if float(self.dropout) < 0.0 or float(self.dropout) >= 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if float(self.max_abs_eta) <= 0.0:
            raise ValueError("max_abs_eta must be positive")
        object.__setattr__(self, "phi_dims", phi_dims)
        object.__setattr__(self, "context_mlp_dims", context_mlp_dims)

    @property
    def phi_output_dim(self) -> int:
        return int(self.phi_dims[-1])

    @property
    def resolved_context_dim(self) -> int:
        return int(self.memory_dim if self.context_dim is None else self.context_dim)

    @property
    def pooled_context_input_dim(self) -> int:
        return 3 * int(self.phi_output_dim) + 1 + int(DETR_SLOT_PARTICLE_FLOW_SUMMARY_DIM)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["phi_dims"] = list(self.phi_dims)
        payload["context_mlp_dims"] = list(self.context_mlp_dims)
        payload["phi_output_dim"] = int(self.phi_output_dim)
        payload["pooled_context_input_dim"] = int(self.pooled_context_input_dim)
        payload["resolved_context_dim"] = int(self.resolved_context_dim)
        payload["particle_flow_feature_dim"] = int(DETR_SLOT_PARTICLE_FLOW_FEATURE_DIM)
        payload["particle_flow_summary_dim"] = int(DETR_SLOT_PARTICLE_FLOW_SUMMARY_DIM)
        return payload


class ParticleFlowHLTEncoderAdapter(BaseHLTEncoderAdapter):
    """PFN/DeepSets HLT encoder for DETR/free-slot reconstruction."""

    def __init__(self, config: ParticleFlowHLTEncoderConfig | Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        torch = require_torch()
        if isinstance(config, ParticleFlowHLTEncoderConfig):
            if kwargs:
                raise ValueError("do not pass kwargs when config is already a ParticleFlowHLTEncoderConfig")
            self.config = config
        else:
            payload = dict(config or {})
            payload.update(kwargs)
            for key in ("phi_dims", "context_mlp_dims"):
                if key in payload:
                    payload[key] = tuple(payload[key])
            self.config = ParticleFlowHLTEncoderConfig(**payload)
        super().__init__(
            input_dim=int(self.config.input_dim),
            memory_dim=int(self.config.memory_dim),
            context_dim=int(self.config.resolved_context_dim),
        )

        self.phi_network = _make_feedforward_network(
            DETR_SLOT_PARTICLE_FLOW_FEATURE_DIM,
            tuple(int(dim) for dim in self.config.phi_dims),
            dropout=float(self.config.dropout),
        )
        self.context_network = _make_feedforward_network(
            int(self.config.pooled_context_input_dim),
            tuple(int(dim) for dim in self.config.context_mlp_dims) + (int(self.config.resolved_context_dim),),
            dropout=float(self.config.dropout),
        )
        memory_input_dim = int(self.config.phi_output_dim)
        if bool(self.config.broadcast_context_to_memory):
            memory_input_dim += int(self.config.resolved_context_dim)
        self.memory_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(memory_input_dim),
            torch.nn.Linear(memory_input_dim, int(self.config.memory_dim)),
        )

    def forward(self, hlt_tokens, hlt_mask=None) -> EncoderOutput:
        torch = require_torch()
        tokens, mask = self.validate_hlt_inputs(hlt_tokens, hlt_mask)
        features = detr_slot_particle_flow_features(tokens, mask, max_abs_eta=float(self.config.max_abs_eta))
        summary = detr_slot_particle_flow_summary_features(tokens, mask, max_abs_eta=float(self.config.max_abs_eta))

        particle_embeddings = self.phi_network(features)
        particle_embeddings = torch.where(mask[:, :, None], particle_embeddings, torch.zeros_like(particle_embeddings))
        sum_pool = _masked_sum_pool(particle_embeddings, mask)
        mean_pool = self.masked_mean_pool(particle_embeddings, mask)
        max_pool = _masked_max_pool(particle_embeddings, mask)
        valid_count = mask.sum(dim=1).to(dtype=particle_embeddings.dtype)
        pooled_context_input = torch.cat([sum_pool, mean_pool, max_pool, torch.log1p(valid_count)[:, None], summary], dim=-1)
        global_context = self.context_network(pooled_context_input)
        has_valid_token = mask.any(dim=1)
        global_context = torch.where(has_valid_token[:, None], global_context, torch.zeros_like(global_context))

        if bool(self.config.broadcast_context_to_memory):
            context_per_particle = global_context[:, None, :].expand(-1, int(tokens.shape[1]), -1)
            memory_input = torch.cat([particle_embeddings, context_per_particle], dim=-1)
        else:
            memory_input = particle_embeddings
        memory_tokens = self.memory_projection(memory_input)
        memory_tokens = torch.where(mask[:, :, None], memory_tokens, torch.zeros_like(memory_tokens))

        output = EncoderOutput(
            memory_tokens=memory_tokens,
            memory_mask=mask,
            global_context=global_context,
            aux={
                "encoder_step": DETR_SLOT_PARTICLE_FLOW_ENCODER_STEP,
                "particle_flow_encoder": True,
                "input_dim": float(self.config.input_dim),
                "memory_dim": float(self.config.memory_dim),
                "context_dim": float(self.config.resolved_context_dim),
                "phi_output_dim": float(self.config.phi_output_dim),
                "broadcast_context_to_memory": float(bool(self.config.broadcast_context_to_memory)),
            },
        )
        return validate_encoder_output(output)


def _apply_channel_mask(values, mask):
    values = _as_float_tensor(values, name="values")
    if values.ndim != 3:
        raise ValueError(f"values must have shape [batch, channels, particles], got {tuple(values.shape)}")
    mask = _as_bool_mask(
        mask,
        name="mask",
        expected_shape=(int(values.shape[0]), int(values.shape[2])),
        device=values.device,
    )
    return torch_where_mask_channel_first(values, mask)


def torch_where_mask_channel_first(values, mask):
    torch = require_torch()
    mask = mask.bool()
    return torch.where(mask[:, None, :], values, torch.zeros_like(values))


class DetrParticleCnnBlock(_ModuleBase):
    """Masked residual Conv1d block over the canonical HLT cache rank axis."""

    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.05,
    ) -> None:
        torch = require_torch()
        super().__init__()
        self.channels = _validate_positive_dim(channels, name="channels")
        self.kernel_size = _validate_positive_dim(kernel_size, name="kernel_size")
        self.dilation = _validate_positive_dim(dilation, name="dilation")
        self.dropout = float(dropout)
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd so Conv1d preserves rank alignment")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("dropout must be in [0, 1)")

        padding = self.dilation * (self.kernel_size - 1) // 2
        self.norm = torch.nn.LayerNorm(self.channels)
        self.rank_conv = torch.nn.Conv1d(
            self.channels,
            self.channels,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=padding,
        )
        self.activation = torch.nn.GELU()
        self.drop = torch.nn.Dropout(self.dropout) if self.dropout > 0.0 else torch.nn.Identity()
        self.pointwise = torch.nn.Conv1d(self.channels, self.channels, kernel_size=1)

    def forward(self, values, mask):
        values = _as_float_tensor(values, name="values")
        if values.ndim != 3:
            raise ValueError(f"values must have shape [batch, channels, particles], got {tuple(values.shape)}")
        if int(values.shape[1]) != int(self.channels):
            raise ValueError(f"channel dimension must be {self.channels}, got {int(values.shape[1])}")
        mask = _as_bool_mask(
            mask,
            name="mask",
            expected_shape=(int(values.shape[0]), int(values.shape[2])),
            device=values.device,
        )
        residual = _apply_channel_mask(values, mask)
        x = residual.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = self.rank_conv(x)
        x = _apply_channel_mask(x, mask)
        x = self.activation(x)
        x = self.drop(x)
        x = self.pointwise(x)
        x = _apply_channel_mask(x, mask)
        return _apply_channel_mask(residual + x, mask)


@dataclass(frozen=True)
class ParticleCnnHLTEncoderConfig:
    """Configuration for the DETR-slot P-CNN/local convolution HLT encoder."""

    input_dim: int = RAW_TOKEN_DIM
    memory_dim: int = 128
    context_dim: int | None = None
    hidden_channels: int = 128
    kernel_sizes: tuple[int, ...] = (5, 5, 3, 3, 3, 3)
    dilations: tuple[int, ...] = (1, 2, 4, 1, 2, 4)
    context_mlp_dims: tuple[int, ...] = (256, 256)
    dropout: float = 0.05
    max_abs_eta: float = 5.0

    def __post_init__(self) -> None:
        _validate_positive_dim(self.input_dim, name="input_dim")
        if int(self.input_dim) != RAW_TOKEN_DIM:
            raise ValueError(f"input_dim must be {RAW_TOKEN_DIM}, got {self.input_dim}")
        _validate_positive_dim(self.memory_dim, name="memory_dim")
        if self.context_dim is not None:
            _validate_positive_dim(self.context_dim, name="context_dim")
        _validate_positive_dim(self.hidden_channels, name="hidden_channels")
        kernel_sizes = _as_positive_int_tuple(self.kernel_sizes, name="kernel_sizes")
        dilations = _as_positive_int_tuple(self.dilations, name="dilations")
        context_mlp_dims = _as_positive_int_tuple(self.context_mlp_dims, name="context_mlp_dims")
        if len(kernel_sizes) != len(dilations):
            raise ValueError("kernel_sizes and dilations must have the same length")
        if any(kernel_size % 2 == 0 for kernel_size in kernel_sizes):
            raise ValueError("kernel_sizes must all be odd so Conv1d preserves rank alignment")
        if float(self.dropout) < 0.0 or float(self.dropout) >= 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if float(self.max_abs_eta) <= 0.0:
            raise ValueError("max_abs_eta must be positive")
        object.__setattr__(self, "kernel_sizes", kernel_sizes)
        object.__setattr__(self, "dilations", dilations)
        object.__setattr__(self, "context_mlp_dims", context_mlp_dims)

    @property
    def resolved_context_dim(self) -> int:
        return int(self.memory_dim if self.context_dim is None else self.context_dim)

    @property
    def num_blocks(self) -> int:
        return len(self.kernel_sizes)

    @property
    def pooled_context_input_dim(self) -> int:
        return 3 * int(self.memory_dim) + 1 + int(DETR_SLOT_PARTICLE_FLOW_SUMMARY_DIM)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["kernel_sizes"] = list(self.kernel_sizes)
        payload["dilations"] = list(self.dilations)
        payload["context_mlp_dims"] = list(self.context_mlp_dims)
        payload["num_blocks"] = int(self.num_blocks)
        payload["resolved_context_dim"] = int(self.resolved_context_dim)
        payload["pooled_context_input_dim"] = int(self.pooled_context_input_dim)
        payload["particle_cnn_feature_dim"] = int(DETR_SLOT_PARTICLE_CNN_FEATURE_DIM)
        payload["particle_cnn_rank_feature_dim"] = int(DETR_SLOT_PARTICLE_CNN_RANK_FEATURE_DIM)
        payload["ordering_assumption"] = DETR_SLOT_PARTICLE_CNN_ORDERING_ASSUMPTION
        return payload


class ParticleCnnHLTEncoderAdapter(BaseHLTEncoderAdapter):
    """P-CNN rank-convolution HLT encoder for DETR/free-slot reconstruction."""

    def __init__(self, config: ParticleCnnHLTEncoderConfig | Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        torch = require_torch()
        if isinstance(config, ParticleCnnHLTEncoderConfig):
            if kwargs:
                raise ValueError("do not pass kwargs when config is already a ParticleCnnHLTEncoderConfig")
            self.config = config
        else:
            payload = dict(config or {})
            payload.update(kwargs)
            for key in ("kernel_sizes", "dilations", "context_mlp_dims"):
                if key in payload:
                    payload[key] = tuple(payload[key])
            self.config = ParticleCnnHLTEncoderConfig(**payload)
        super().__init__(
            input_dim=int(self.config.input_dim),
            memory_dim=int(self.config.memory_dim),
            context_dim=int(self.config.resolved_context_dim),
        )

        self.input_projection = _make_feedforward_network(
            DETR_SLOT_PARTICLE_CNN_FEATURE_DIM,
            (int(self.config.hidden_channels),),
            dropout=float(self.config.dropout),
        )
        self.blocks = torch.nn.ModuleList(
            [
                DetrParticleCnnBlock(
                    int(self.config.hidden_channels),
                    kernel_size=int(kernel_size),
                    dilation=int(dilation),
                    dropout=float(self.config.dropout),
                )
                for kernel_size, dilation in zip(self.config.kernel_sizes, self.config.dilations)
            ]
        )
        self.memory_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(int(self.config.hidden_channels)),
            torch.nn.Linear(int(self.config.hidden_channels), int(self.config.memory_dim)),
        )
        self.context_network = _make_feedforward_network(
            int(self.config.pooled_context_input_dim),
            tuple(int(dim) for dim in self.config.context_mlp_dims) + (int(self.config.resolved_context_dim),),
            dropout=float(self.config.dropout),
        )

    def forward(self, hlt_tokens, hlt_mask=None) -> EncoderOutput:
        torch = require_torch()
        tokens, mask = self.validate_hlt_inputs(hlt_tokens, hlt_mask)
        features = detr_slot_particle_cnn_features(tokens, mask, max_abs_eta=float(self.config.max_abs_eta))
        summary = detr_slot_particle_flow_summary_features(tokens, mask, max_abs_eta=float(self.config.max_abs_eta))

        encoded = self.input_projection(features)
        encoded = torch.where(mask[:, :, None], encoded, torch.zeros_like(encoded))
        conv_values = encoded.transpose(1, 2)
        conv_values = _apply_channel_mask(conv_values, mask)
        for block in self.blocks:
            conv_values = block(conv_values, mask)
        particle_embeddings = conv_values.transpose(1, 2)
        particle_embeddings = torch.where(mask[:, :, None], particle_embeddings, torch.zeros_like(particle_embeddings))
        memory_tokens = self.memory_projection(particle_embeddings)
        memory_tokens = torch.where(mask[:, :, None], memory_tokens, torch.zeros_like(memory_tokens))

        sum_pool = _masked_sum_pool(memory_tokens, mask)
        mean_pool = self.masked_mean_pool(memory_tokens, mask)
        max_pool = _masked_max_pool(memory_tokens, mask)
        valid_count = mask.sum(dim=1).to(dtype=memory_tokens.dtype)
        pooled_context_input = torch.cat([sum_pool, mean_pool, max_pool, torch.log1p(valid_count)[:, None], summary], dim=-1)
        global_context = self.context_network(pooled_context_input)
        has_valid_token = mask.any(dim=1)
        global_context = torch.where(has_valid_token[:, None], global_context, torch.zeros_like(global_context))

        output = EncoderOutput(
            memory_tokens=memory_tokens,
            memory_mask=mask,
            global_context=global_context,
            aux={
                "encoder_step": DETR_SLOT_PARTICLE_CNN_ENCODER_STEP,
                "particle_cnn_encoder": True,
                "input_dim": float(self.config.input_dim),
                "memory_dim": float(self.config.memory_dim),
                "context_dim": float(self.config.resolved_context_dim),
                "hidden_channels": float(self.config.hidden_channels),
                "num_blocks": float(self.config.num_blocks),
                "ordering_assumption": DETR_SLOT_PARTICLE_CNN_ORDERING_ASSUMPTION,
            },
        )
        return validate_encoder_output(output)


__all__ = [
    "DETR_SLOT_ENCODER_OUTPUT_CONTRACT",
    "DETR_SLOT_ENCODER_STEP",
    "DETR_SLOT_GLOBAL_TRANSFORMER_ENCODER_STEP",
    "DETR_SLOT_PARTICLE_CNN_ENCODER_STEP",
    "DETR_SLOT_PARTICLE_CNN_FEATURE_DIM",
    "DETR_SLOT_PARTICLE_CNN_FEATURE_NAMES",
    "DETR_SLOT_PARTICLE_CNN_ORDERING_ASSUMPTION",
    "DETR_SLOT_PARTICLE_CNN_RANK_FEATURE_DIM",
    "DETR_SLOT_PARTICLE_CNN_RANK_FEATURE_NAMES",
    "DETR_SLOT_PARTICLE_FLOW_ENCODER_STEP",
    "DETR_SLOT_PARTICLE_FLOW_FEATURE_DIM",
    "DETR_SLOT_PARTICLE_FLOW_FEATURE_NAMES",
    "DETR_SLOT_PARTICLE_FLOW_SUMMARY_DIM",
    "DETR_SLOT_PARTICLE_FLOW_SUMMARY_FEATURE_NAMES",
    "DETR_SLOT_PARTICLE_NET_ENCODER_STEP",
    "DETR_SLOT_PARTICLE_NET_COORD_DIM",
    "DETR_SLOT_TOKEN_EMBED_FEATURE_DIM",
    "BaseHLTEncoderAdapter",
    "DetrParticleCnnBlock",
    "DetrParticleNetEdgeConvBlock",
    "DummyHLTEncoderAdapter",
    "DummyHLTEncoderConfig",
    "EncoderOutput",
    "GlobalTransformerHLTEncoderAdapter",
    "GlobalTransformerHLTEncoderConfig",
    "ParticleCnnHLTEncoderAdapter",
    "ParticleCnnHLTEncoderConfig",
    "ParticleFlowHLTEncoderAdapter",
    "ParticleFlowHLTEncoderConfig",
    "ParticleNetHLTEncoderAdapter",
    "ParticleNetHLTEncoderConfig",
    "detr_slot_particle_cnn_features",
    "detr_slot_particle_cnn_rank_features",
    "detr_slot_particle_flow_features",
    "detr_slot_particle_flow_summary_features",
    "detr_slot_particle_net_coordinates",
    "detr_slot_token_embedding_features",
    "masked_mean_pool",
    "validate_encoder_output",
]
