"""Stage A HLT-to-offline reconstructors for the reco7 family.

The Step 7 implementation started with `m2_base`; Step 9 expands the same
architecture/loss surface to the seven named reconstructor variants while
leaving HLT generation and split definitions untouched.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np

from .hlt_baseline import require_torch, resolve_device, save_json, set_training_seed
from .hlt_cache import load_cached_hlt_view
from .jetclass_data import JetView, load_offline_view, load_split_manifest, manifest_hash

try:  # Keep module importable where PyTorch is not installed.
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


RAW_DIM = 14
KIN_DIM = 4
ENERGY_EPS = 1.0e-4
SQRT_EPS = 1.0e-12
RECONSTRUCTOR_VARIANT_NAMES = [
    "m2_base",
    "m2_consstrong",
    "m2_budgetlite",
    "m2_genlow",
    "m2_genhigh",
    "m2_topk60ish",
    "m2_antioverlap",
]


@dataclass
class ReconstructorVariantConfig:
    """Architecture and loss knobs for one reconstruction variant."""

    name: str = "m2_base"
    max_generated: int = 56
    max_split_children: int = 2
    hidden_dim: int = 128
    global_dim: int = 128
    num_heads: int = 8
    num_encoder_layers: int = 6
    feedforward_dim: int = 512
    dropout: float = 0.10
    max_hlt_constits: int = 128
    set_matching_weight: float = 1.0
    budget_count_weight: float = 0.70
    sparsity_weight: float = 0.010
    locality_weight: float = 0.08
    anti_overlap_weight: float = 0.0
    pt_ratio_weight: float = 0.12
    mass_ratio_weight: float = 0.02
    energy_ratio_weight: float = 0.02
    physics_weight: float = 0.0
    split_sparsity_weight: float = 0.004
    generated_sparsity_weight: float = 0.010
    matched_weight_weight: float = 0.050
    nonfinite_penalty_weight: float = 0.10
    matching_mode: str = "hungarian"
    max_matching_candidates: int = 160
    matching_large_cost: float = 1.0e6
    target_added_particle_scale: float = 0.90
    split_locality_radius: float = 0.04
    generated_locality_radius: float = 0.20
    anti_overlap_radius: float = 0.035
    max_log_pt_shift: float = 0.60
    max_log_energy_shift: float = 0.60
    max_eta_shift: float = 0.12
    max_phi_shift: float = 0.12
    max_split_log_pt_shift: float = 0.80
    max_generated_abs_eta: float = 5.0
    description: str = "Default m2-style reconstruction objective."


def m2_base_variant_config() -> ReconstructorVariantConfig:
    return ReconstructorVariantConfig()


def all_reconstructor_variant_configs() -> Dict[str, ReconstructorVariantConfig]:
    """Return the seven Step 9 reconstructor variant configs."""

    return {
        "m2_base": ReconstructorVariantConfig(
            name="m2_base",
            description="Default m2-style reconstruction objective.",
        ),
        "m2_consstrong": ReconstructorVariantConfig(
            name="m2_consstrong",
            set_matching_weight=1.05,
            budget_count_weight=1.00,
            locality_weight=0.10,
            pt_ratio_weight=0.22,
            mass_ratio_weight=0.05,
            energy_ratio_weight=0.07,
            target_added_particle_scale=0.95,
            description="Stronger global reconstruction consistency and budget calibration.",
        ),
        "m2_budgetlite": ReconstructorVariantConfig(
            name="m2_budgetlite",
            budget_count_weight=0.25,
            sparsity_weight=0.008,
            locality_weight=0.07,
            pt_ratio_weight=0.20,
            mass_ratio_weight=0.025,
            energy_ratio_weight=0.03,
            target_added_particle_scale=0.70,
            description="Relaxed count/budget pressure with more emphasis on useful pT response.",
        ),
        "m2_genlow": ReconstructorVariantConfig(
            name="m2_genlow",
            max_generated=40,
            budget_count_weight=0.80,
            sparsity_weight=0.018,
            locality_weight=0.09,
            target_added_particle_scale=0.75,
            description="Lower generated-token capacity and stronger sparsity.",
        ),
        "m2_genhigh": ReconstructorVariantConfig(
            name="m2_genhigh",
            max_generated=72,
            budget_count_weight=0.55,
            sparsity_weight=0.004,
            locality_weight=0.06,
            target_added_particle_scale=1.05,
            description="Higher generated-token capacity with relaxed sparsity.",
        ),
        "m2_topk60ish": ReconstructorVariantConfig(
            name="m2_topk60ish",
            max_generated=60,
            budget_count_weight=0.65,
            sparsity_weight=0.008,
            locality_weight=0.08,
            target_added_particle_scale=0.90,
            description="Intermediate generated-token capacity between low and high.",
        ),
        "m2_antioverlap": ReconstructorVariantConfig(
            name="m2_antioverlap",
            budget_count_weight=0.75,
            sparsity_weight=0.010,
            locality_weight=0.14,
            anti_overlap_weight=0.060,
            split_locality_radius=0.025,
            generated_locality_radius=0.12,
            anti_overlap_radius=0.045,
            description="Locality and anti-overlap pressure to reduce redundant added candidates.",
        ),
    }


def get_reconstructor_variant_config(name: str) -> ReconstructorVariantConfig:
    configs = all_reconstructor_variant_configs()
    if name not in configs:
        raise ValueError(f"Unknown reconstructor variant {name!r}; expected one of {RECONSTRUCTOR_VARIANT_NAMES}")
    return configs[name]


@dataclass
class StageAReconstructorTrainConfig:
    """Training configuration for Stage A reconstruction."""

    output_dir: str
    manifest_path: str
    hlt_cache_dir: str
    data_dir: str | None = None
    variant: str = "m2_base"
    train_split: str = "model_train"
    val_split: str = "model_val"
    seed: int = 808
    batch_size: int = 128
    epochs: int = 20
    lr: float = 3.0e-4
    weight_decay: float = 1.0e-4
    num_workers: int = 0
    device: str = "auto"
    amp: bool = True
    grad_clip_norm: float = 1.0
    early_stop_patience: int = 5
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    verify_label_branches: bool = False
    read_chunk_size: int = 50_000
    compile_model: bool = False


@dataclass
class ReconstructionOutput:
    """Soft corrected particle view produced by the reconstructor."""

    tokens: Any
    weights: Any
    candidate_mask: Any
    edited_tokens: Any
    split_tokens: Any
    generated_tokens: Any
    edited_weights: Any
    split_weights: Any
    generated_weights: Any
    total_count_pred: Any
    added_count_pred: Any
    corrected_parent_tokens: Any = None
    corrected_parent_weights: Any = None
    split_child_tokens: Any = None
    split_child_weights: Any = None
    split_parent_probability: Any = None
    split_parent_uplift: Any = None
    split_parent_added_support: Any = None
    generator_to_parent_assignment: Any = None
    generator_parent_added_support: Any = None
    parent_added_support: Any = None
    budget_efficiency_share: Any = None
    budget_split_share: Any = None
    candidate_branch_ids: Any = None
    sanitized_hlt_tokens: Any = None
    sanitized_hlt_mask: Any = None
    diagnostics: Any = None


def wrap_phi_torch(phi):
    torch = require_torch()
    return torch.remainder(phi + torch.pi, 2.0 * torch.pi) - torch.pi


def physical_energy_floor(pt, eta, *, eps: float = ENERGY_EPS):
    """Massless-particle energy floor for stable downstream four-vectors."""

    torch = require_torch()
    return torch.clamp(pt, min=0.0) * torch.cosh(torch.clamp(eta, -5.0, 5.0)) + float(eps)


def safe_sqrt(value, *, eps: float = SQRT_EPS):
    """Sqrt with finite gradients at zero for reconstruction losses."""

    torch = require_torch()
    return torch.sqrt(torch.clamp(value, min=float(eps)))


def replace_kinematic_channels(tokens, pt, eta, phi, energy):
    """Return tokens with updated kinematic channels without in-place writes."""

    torch = require_torch()
    return torch.cat(
        [
            pt.unsqueeze(-1),
            eta.unsqueeze(-1),
            phi.unsqueeze(-1),
            energy.unsqueeze(-1),
            tokens[..., 4:14],
        ],
        dim=-1,
    )


def raw_token_features(tokens, mask):
    torch = require_torch()
    pt = torch.clamp(tokens[:, :, 0], min=1.0e-8)
    eta = tokens[:, :, 1]
    phi = tokens[:, :, 2]
    energy = torch.clamp(tokens[:, :, 3], min=1.0e-8)
    kin = torch.stack(
        [
            torch.log(pt),
            eta / 5.0,
            torch.sin(phi),
            torch.cos(phi),
            torch.log(energy),
        ],
        dim=-1,
    )
    nonkin = tokens[:, :, 4:14]
    return torch.cat([kin, nonkin, mask.unsqueeze(-1).float()], dim=-1)


def sanitize_hlt_tokens(tokens, mask):
    """Return finite, physical HLT tokens plus a non-empty valid mask."""

    torch = require_torch()
    tokens = tokens.float()
    mask = mask.bool()
    finite_tokens = torch.isfinite(tokens).all(dim=-1)
    if hasattr(torch, "nan_to_num"):
        cleaned = torch.nan_to_num(tokens, nan=0.0, posinf=0.0, neginf=0.0)
    else:  # pragma: no cover - compatibility with old torch
        cleaned = torch.where(torch.isfinite(tokens), tokens, torch.zeros_like(tokens))
    pt = torch.clamp(cleaned[:, :, 0], min=0.0)
    eta = torch.clamp(cleaned[:, :, 1], -5.0, 5.0)
    phi = wrap_phi_torch(cleaned[:, :, 2])
    energy = torch.maximum(torch.clamp(cleaned[:, :, 3], min=ENERGY_EPS), physical_energy_floor(pt, eta))
    cleaned = replace_kinematic_channels(cleaned, pt, eta, phi, energy)

    safe_mask = mask & finite_tokens
    empty = safe_mask.sum(dim=1) == 0
    if bool(empty.any()):
        token_indices = torch.arange(cleaned.shape[1], device=cleaned.device)
        first_token = token_indices[None, :].eq(0).expand_as(safe_mask)
        safe_mask = torch.where(empty[:, None], first_token, safe_mask)
        fallback_scalar = torch.zeros_like(cleaned[:, :, 0])
        fallback_pt = torch.where(first_token, torch.full_like(fallback_scalar, ENERGY_EPS), fallback_scalar)
        fallback_tokens = torch.cat(
            [
                fallback_pt.unsqueeze(-1),
                fallback_scalar.unsqueeze(-1),
                fallback_scalar.unsqueeze(-1),
                fallback_pt.unsqueeze(-1),
                torch.zeros_like(cleaned[..., 4:14]),
            ],
            dim=-1,
        )
        cleaned = torch.where(empty[:, None, None], fallback_tokens, cleaned)
    diagnostics = {
        "nonfinite_input_token_count": (~finite_tokens & mask).sum(dim=1).float(),
        "forced_nonempty_mask": empty.float(),
    }
    return cleaned, safe_mask, diagnostics


class AttentionPooling(_ModuleBase):
    """Mask-aware learned-query attention pooling."""

    def __init__(self, dim: int) -> None:
        require_torch()
        super().__init__()
        torch = require_torch()
        self.query = torch.nn.Parameter(torch.randn(int(dim)) * 0.02)
        self.norm = torch.nn.LayerNorm(int(dim))

    def forward(self, tokens, mask):
        torch = require_torch()
        tokens = self.norm(tokens)
        scores = torch.einsum("bnd,d->bn", tokens, self.query) / math.sqrt(float(tokens.shape[-1]))
        scores = scores.masked_fill(~mask.bool(), -1.0e4)
        weights = torch.softmax(scores, dim=1) * mask.float()
        weights = weights / torch.clamp(weights.sum(dim=1, keepdim=True), min=1.0e-6)
        return torch.einsum("bn,bnd->bd", weights, tokens), weights


class RelativePositionEncoderLayer(_ModuleBase):
    """Transformer encoder layer with eta/phi/dR-aware additive attention bias."""

    def __init__(self, *, dim: int, num_heads: int, feedforward_dim: int, dropout: float) -> None:
        require_torch()
        super().__init__()
        torch = require_torch()
        dim = int(dim)
        num_heads = int(num_heads)
        if dim % num_heads != 0:
            raise ValueError(f"hidden_dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = torch.nn.Linear(dim, dim * 3)
        self.out_proj = torch.nn.Linear(dim, dim)
        self.rel_bias = torch.nn.Sequential(
            torch.nn.Linear(4, max(16, num_heads * 2)),
            torch.nn.GELU(),
            torch.nn.Linear(max(16, num_heads * 2), num_heads),
        )
        self.norm1 = torch.nn.LayerNorm(dim)
        self.norm2 = torch.nn.LayerNorm(dim)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(dim, int(feedforward_dim)),
            torch.nn.GELU(),
            torch.nn.Dropout(float(dropout)),
            torch.nn.Linear(int(feedforward_dim), dim),
        )
        self.dropout = torch.nn.Dropout(float(dropout))

    def relative_features(self, raw_tokens):
        torch = require_torch()
        eta = raw_tokens[:, :, 1]
        phi = raw_tokens[:, :, 2]
        deta = torch.clamp(eta[:, :, None] - eta[:, None, :], -10.0, 10.0)
        dphi = wrap_phi_torch(phi[:, :, None] - phi[:, None, :])
        dr = safe_sqrt(deta * deta + dphi * dphi)
        return torch.stack([deta / 5.0, torch.sin(dphi), torch.cos(dphi), torch.clamp(dr, 0.0, 10.0) / 5.0], dim=-1)

    def forward(self, x, mask, raw_tokens):
        torch = require_torch()
        residual = x
        x_norm = self.norm1(x)
        batch_size, n_tokens, _ = x_norm.shape
        qkv = self.qkv(x_norm).view(batch_size, n_tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(self.head_dim))
        rel_bias = self.rel_bias(self.relative_features(raw_tokens)).permute(0, 3, 1, 2)
        scores = scores + rel_bias
        scores = scores.masked_fill(~mask[:, None, None, :].bool(), -1.0e4)
        attn = torch.softmax(scores, dim=-1)
        attn = attn * mask[:, None, None, :].float()
        attn = attn / torch.clamp(attn.sum(dim=-1, keepdim=True), min=1.0e-6)
        context = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch_size, n_tokens, self.dim)
        x = residual + self.dropout(self.out_proj(context))
        x = x * mask.unsqueeze(-1).float()
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x * mask.unsqueeze(-1).float()


class RelativePositionTokenEncoder(_ModuleBase):
    """Stack of relative-position-aware transformer encoder layers."""

    def __init__(self, config: ReconstructorVariantConfig) -> None:
        require_torch()
        super().__init__()
        torch = require_torch()
        hidden = int(config.hidden_dim)
        self.input_proj = torch.nn.Sequential(
            torch.nn.Linear(16, hidden),
            torch.nn.LayerNorm(hidden),
            torch.nn.GELU(),
        )
        self.layers = torch.nn.ModuleList(
            [
                RelativePositionEncoderLayer(
                    dim=hidden,
                    num_heads=int(config.num_heads),
                    feedforward_dim=int(config.feedforward_dim),
                    dropout=float(config.dropout),
                )
                for _ in range(int(config.num_encoder_layers))
            ]
        )
        self.final_norm = torch.nn.LayerNorm(hidden)

    def forward(self, hlt_tokens, hlt_mask):
        x = self.input_proj(raw_token_features(hlt_tokens, hlt_mask))
        x = x * hlt_mask.unsqueeze(-1).float()
        for layer in self.layers:
            x = layer(x, hlt_mask, hlt_tokens)
        return self.final_norm(x) * hlt_mask.unsqueeze(-1).float()


class GeneratorCrossAttention(_ModuleBase):
    """Learned generator queries cross-attending to encoded HLT tokens."""

    def __init__(self, *, dim: int, num_heads: int, max_generated: int, dropout: float) -> None:
        require_torch()
        super().__init__()
        torch = require_torch()
        self.queries = torch.nn.Parameter(torch.randn(int(max_generated), int(dim)) * 0.02)
        self.global_to_query = torch.nn.Linear(int(dim), int(dim))
        self.cross_attn = torch.nn.MultiheadAttention(
            embed_dim=int(dim),
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.norm = torch.nn.LayerNorm(int(dim))

    def forward(self, encoded_tokens, mask, global_context):
        batch_size = encoded_tokens.shape[0]
        if self.queries.shape[0] == 0:
            empty_states = encoded_tokens.new_zeros(batch_size, 0, encoded_tokens.shape[-1])
            empty_assignment = encoded_tokens.new_zeros(batch_size, 0, encoded_tokens.shape[1])
            return empty_states, empty_assignment
        queries = self.queries[None, :, :].expand(batch_size, -1, -1)
        queries = queries + self.global_to_query(global_context)[:, None, :]
        attended, weights = self.cross_attn(
            query=queries,
            key=encoded_tokens,
            value=encoded_tokens,
            key_padding_mask=~mask.bool(),
            need_weights=True,
            average_attn_weights=False,
        )
        assignment = weights.mean(dim=1) * mask[:, None, :].float()
        assignment = assignment / assignment.sum(dim=-1, keepdim=True).clamp(min=1.0e-6)
        return self.norm(attended + queries), assignment


class M2BaseReconstructor(_ModuleBase):
    """Original-mechanism m2-hybrid HLT-to-offline reconstructor.

    The forward path is operation-aware: it edits existing HLT parents, proposes
    split children local to each parent, generates missing-token candidates via
    learned queries that cross-attend to HLT tokens, and calibrates all soft
    supports with jet-level budget heads.
    """

    def __init__(self, config: ReconstructorVariantConfig | None = None) -> None:
        require_torch()
        super().__init__()
        torch = require_torch()
        self.config = config or m2_base_variant_config()
        hidden = int(self.config.hidden_dim)
        global_dim = int(self.config.global_dim)
        if hidden % int(self.config.num_heads) != 0:
            raise ValueError(
                f"hidden_dim={hidden} must be divisible by num_heads={int(self.config.num_heads)}"
            )
        self.token_encoder = RelativePositionTokenEncoder(self.config)
        self.pool = AttentionPooling(hidden)
        self.global_encoder = torch.nn.Sequential(
            torch.nn.Linear(hidden + 3, global_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(float(self.config.dropout)),
            torch.nn.Linear(global_dim, global_dim),
            torch.nn.GELU(),
        )
        self.global_to_hidden = torch.nn.Linear(global_dim, hidden)
        token_context_dim = hidden * 2

        self.edit_delta_head = torch.nn.Linear(token_context_dim, 4)
        self.edit_weight_head = torch.nn.Linear(token_context_dim, 1)

        self.split_parent_head = torch.nn.Linear(token_context_dim, 1)
        self.split_uplift_head = torch.nn.Linear(token_context_dim, 2)
        self.split_child_head = torch.nn.Linear(token_context_dim, int(self.config.max_split_children) * 5)

        self.generator_decoder = GeneratorCrossAttention(
            dim=hidden,
            num_heads=int(self.config.num_heads),
            max_generated=int(self.config.max_generated),
            dropout=float(self.config.dropout),
        )
        self.generated_head = torch.nn.Sequential(
            torch.nn.Linear(hidden * 2, hidden),
            torch.nn.GELU(),
            torch.nn.Dropout(float(self.config.dropout)),
            torch.nn.Linear(hidden, 15),
        )
        self.budget_head = torch.nn.Sequential(
            torch.nn.Linear(global_dim, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, 4),
        )

    def _apply_kinematic_delta(self, base_tokens, delta, *, split: bool = False):
        torch = require_torch()
        cfg = self.config
        max_log_pt = cfg.max_split_log_pt_shift if split else cfg.max_log_pt_shift
        log_pt_delta = torch.tanh(delta[..., 0]) * float(max_log_pt)
        eta_delta = torch.tanh(delta[..., 1]) * float(cfg.max_eta_shift)
        phi_delta = torch.tanh(delta[..., 2]) * float(cfg.max_phi_shift)
        log_e_delta = torch.tanh(delta[..., 3]) * float(cfg.max_log_energy_shift)

        pt = torch.clamp(base_tokens[..., 0], min=1.0e-8) * torch.exp(log_pt_delta)
        eta = torch.clamp(base_tokens[..., 1] + eta_delta, -5.0, 5.0)
        phi = wrap_phi_torch(base_tokens[..., 2] + phi_delta)
        energy = torch.clamp(base_tokens[..., 3], min=1.0e-8) * torch.exp(log_e_delta)
        energy = torch.maximum(energy, physical_energy_floor(pt, eta))
        return replace_kinematic_channels(base_tokens, pt, eta, phi, energy)

    def _uplift_parent_tokens(self, hlt_tokens, raw_uplift):
        torch = require_torch()
        uplift = torch.sigmoid(raw_uplift) * float(self.config.max_split_log_pt_shift)
        pt = torch.clamp(hlt_tokens[:, :, 0], min=1.0e-8) * torch.exp(uplift[:, :, 0])
        eta = hlt_tokens[:, :, 1]
        phi = hlt_tokens[:, :, 2]
        energy = torch.clamp(hlt_tokens[:, :, 3], min=1.0e-8) * torch.exp(uplift[:, :, 1])
        energy = torch.maximum(energy, physical_energy_floor(pt, eta))
        return replace_kinematic_channels(hlt_tokens, pt, eta, phi, energy), uplift

    @staticmethod
    def _budget_scale(raw_weights, budget, mask):
        torch = require_torch()
        raw_weights = torch.clamp(raw_weights, min=0.0)
        mask = mask.float()
        raw_weights = raw_weights * mask
        denom = torch.clamp(raw_weights.sum(dim=1, keepdim=True), min=1.0e-6)
        scaled = raw_weights * (budget[:, None] / denom)
        return torch.clamp(scaled, min=0.0, max=1.0) * mask

    def _make_generated_tokens(self, generated_raw, *, dtype, device):
        torch = require_torch()
        gen_pt = torch.nn.functional.softplus(generated_raw[:, :, 0]) + ENERGY_EPS
        gen_eta = torch.tanh(generated_raw[:, :, 1]) * float(self.config.max_generated_abs_eta)
        gen_phi = wrap_phi_torch(generated_raw[:, :, 2])
        gen_energy = physical_energy_floor(gen_pt, gen_eta) + torch.nn.functional.softplus(generated_raw[:, :, 3])
        generated_tokens = torch.cat(
            [
                gen_pt.unsqueeze(-1),
                gen_eta.unsqueeze(-1),
                gen_phi.unsqueeze(-1),
                gen_energy.unsqueeze(-1),
                torch.tanh(generated_raw[:, :, 4:14]),
            ],
            dim=-1,
        )
        return generated_tokens.to(dtype=dtype, device=device)

    def forward(self, hlt_tokens, hlt_mask) -> ReconstructionOutput:
        torch = require_torch()
        hlt_tokens, hlt_mask, input_diagnostics = sanitize_hlt_tokens(hlt_tokens, hlt_mask)
        batch_size, n_parents, _ = hlt_tokens.shape
        n_children = int(self.config.max_split_children)
        n_generated = int(self.config.max_generated)

        encoded = self.token_encoder(hlt_tokens, hlt_mask)
        pooled, pool_weights = self.pool(encoded, hlt_mask)
        hlt_count = hlt_mask.sum(dim=1).float()
        hlt_pt = (hlt_tokens[:, :, 0] * hlt_mask.float()).sum(dim=1)
        hlt_energy = (hlt_tokens[:, :, 3] * hlt_mask.float()).sum(dim=1)
        global_stats = torch.stack(
            [
                torch.log1p(hlt_count) / 5.0,
                torch.log1p(torch.clamp(hlt_pt, min=0.0)) / 8.0,
                torch.log1p(torch.clamp(hlt_energy, min=0.0)) / 8.0,
            ],
            dim=1,
        )
        global_latent = self.global_encoder(torch.cat([pooled, global_stats], dim=1))
        global_hidden = self.global_to_hidden(global_latent)
        token_context = torch.cat(
            [encoded, global_hidden[:, None, :].expand(-1, n_parents, -1)],
            dim=-1,
        )

        budget_raw = self.budget_head(global_latent)
        added_count_pred = torch.nn.functional.softplus(budget_raw[:, 1])
        total_count_pred = hlt_count + added_count_pred + 0.10 * torch.nn.functional.softplus(budget_raw[:, 0])
        budget_split_share = torch.sigmoid(budget_raw[:, 2])
        budget_efficiency_global = torch.sigmoid(budget_raw[:, 3])

        edit_delta = self.edit_delta_head(token_context)
        split_probability = torch.sigmoid(self.split_parent_head(token_context).squeeze(-1)) * hlt_mask.float()
        edited_tokens = self._apply_kinematic_delta(hlt_tokens, edit_delta, split=False)
        edit_raw_weights = torch.sigmoid(self.edit_weight_head(token_context).squeeze(-1))
        edit_raw_weights = edit_raw_weights * (1.0 - split_probability) * hlt_mask.float()

        uplifted_parent_tokens, split_parent_uplift = self._uplift_parent_tokens(
            hlt_tokens,
            self.split_uplift_head(token_context),
        )
        child_raw = self.split_child_head(token_context).view(batch_size, n_parents, n_children, 5)
        parent_for_children = uplifted_parent_tokens[:, :, None, :].expand(-1, -1, n_children, -1)
        split_child_tokens = self._apply_kinematic_delta(parent_for_children, child_raw[..., :4], split=True)
        split_child_raw_weights = (
            split_probability[:, :, None]
            * torch.sigmoid(child_raw[..., 4])
            * hlt_mask[:, :, None].float()
        )

        gen_states, generator_to_parent = self.generator_decoder(encoded, hlt_mask, global_hidden)
        generated_raw = self.generated_head(
            torch.cat([gen_states, global_hidden[:, None, :].expand(-1, n_generated, -1)], dim=-1)
        )
        generated_tokens = self._make_generated_tokens(generated_raw, dtype=hlt_tokens.dtype, device=hlt_tokens.device)
        generated_raw_weights = torch.sigmoid(generated_raw[:, :, 14])

        split_budget = added_count_pred * budget_split_share
        gen_budget = added_count_pred * (1.0 - budget_split_share)
        edit_budget = torch.clamp(total_count_pred - added_count_pred, min=1.0e-3)

        split_mask = hlt_mask[:, :, None].expand(-1, -1, n_children).reshape(batch_size, n_parents * n_children)
        split_weights = self._budget_scale(
            split_child_raw_weights.reshape(batch_size, n_parents * n_children),
            split_budget,
            split_mask,
        )
        split_child_weights = split_weights.view(batch_size, n_parents, n_children)
        generated_mask = torch.ones(batch_size, n_generated, dtype=torch.bool, device=hlt_tokens.device)
        generated_weights = self._budget_scale(generated_raw_weights, gen_budget, generated_mask)
        edited_weights = self._budget_scale(edit_raw_weights, edit_budget, hlt_mask)

        split_tokens = split_child_tokens.reshape(batch_size, n_parents * n_children, RAW_DIM)
        split_parent_added_support = split_child_weights.sum(dim=2)
        generator_parent_added_support = torch.einsum("bgn,bg->bn", generator_to_parent, generated_weights)
        parent_added_support = (split_parent_added_support + generator_parent_added_support) * hlt_mask.float()
        budget_efficiency_share = (
            parent_added_support / torch.clamp(added_count_pred[:, None], min=1.0)
        ) * budget_efficiency_global[:, None]

        tokens = torch.cat([edited_tokens, split_tokens, generated_tokens], dim=1)
        weights = torch.cat([edited_weights, split_weights, generated_weights], dim=1)
        candidate_mask = torch.cat([hlt_mask, split_mask, generated_mask], dim=1)
        branch_ids = torch.cat(
            [
                torch.zeros(batch_size, n_parents, dtype=torch.long, device=hlt_tokens.device),
                torch.ones(batch_size, n_parents * n_children, dtype=torch.long, device=hlt_tokens.device),
                torch.full((batch_size, n_generated), 2, dtype=torch.long, device=hlt_tokens.device),
            ],
            dim=1,
        )

        diagnostics = {
            **input_diagnostics,
            "attention_pool_weights": pool_weights,
            "soft_total_weight": (weights * candidate_mask.float()).sum(dim=1),
            "soft_added_weight": split_weights.sum(dim=1) + generated_weights.sum(dim=1),
            "budget_split_share": budget_split_share,
            "budget_efficiency_global": budget_efficiency_global,
        }

        return ReconstructionOutput(
            tokens=tokens,
            weights=weights,
            candidate_mask=candidate_mask,
            edited_tokens=edited_tokens,
            split_tokens=split_tokens,
            generated_tokens=generated_tokens,
            edited_weights=edited_weights,
            split_weights=split_weights,
            generated_weights=generated_weights,
            total_count_pred=total_count_pred,
            added_count_pred=added_count_pred,
            corrected_parent_tokens=edited_tokens,
            corrected_parent_weights=edited_weights,
            split_child_tokens=split_child_tokens,
            split_child_weights=split_child_weights,
            split_parent_probability=split_probability,
            split_parent_uplift=split_parent_uplift,
            split_parent_added_support=split_parent_added_support,
            generator_to_parent_assignment=generator_to_parent,
            generator_parent_added_support=generator_parent_added_support,
            parent_added_support=parent_added_support,
            budget_efficiency_share=budget_efficiency_share,
            budget_split_share=budget_split_share,
            candidate_branch_ids=branch_ids,
            sanitized_hlt_tokens=hlt_tokens,
            sanitized_hlt_mask=hlt_mask,
            diagnostics=diagnostics,
        )


def build_reconstructor(config: ReconstructorVariantConfig | None = None):
    config = config or m2_base_variant_config()
    if config.name not in RECONSTRUCTOR_VARIANT_NAMES:
        raise ValueError(f"Unknown reconstructor variant {config.name!r}")
    return M2BaseReconstructor(config)


def token_p4(tokens, weights=None, mask=None):
    torch = require_torch()
    pt = torch.clamp(tokens[:, :, 0], min=0.0)
    eta = tokens[:, :, 1]
    phi = tokens[:, :, 2]
    energy = torch.clamp(tokens[:, :, 3], min=0.0)
    if weights is not None:
        weights = weights.float()
        pt = pt * weights
        energy = energy * weights
    if mask is not None:
        mask_float = mask.float()
        pt = pt * mask_float
        energy = energy * mask_float
    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    pz = pt * torch.sinh(eta)
    return px, py, pz, energy


def jet_response(tokens, weights=None, mask=None):
    torch = require_torch()
    px, py, pz, energy = token_p4(tokens, weights=weights, mask=mask)
    jet_px = px.sum(dim=1)
    jet_py = py.sum(dim=1)
    jet_pz = pz.sum(dim=1)
    jet_energy = energy.sum(dim=1)
    jet_pt = safe_sqrt(jet_px * jet_px + jet_py * jet_py)
    mass2 = jet_energy * jet_energy - jet_px * jet_px - jet_py * jet_py - jet_pz * jet_pz
    jet_mass = safe_sqrt(mass2)
    return {
        "pt": jet_pt,
        "energy": jet_energy,
        "mass": jet_mass,
    }


def bounded_log_response_loss(pred, target, *, max_abs_log_diff: float = 6.0):
    """Stable global-response loss that cannot explode on nearly massless jets."""

    torch = require_torch()
    diff = torch.log1p(torch.clamp(pred, min=0.0)) - torch.log1p(torch.clamp(target, min=0.0))
    diff = torch.clamp(diff, -float(max_abs_log_diff), float(max_abs_log_diff))
    return (diff * diff).mean()


def matching_features(tokens):
    torch = require_torch()
    pt = torch.clamp(tokens[:, :, 0], min=1.0e-8)
    eta = tokens[:, :, 1]
    phi = tokens[:, :, 2]
    energy = torch.clamp(tokens[:, :, 3], min=1.0e-8)
    kin = torch.stack([torch.log(pt), eta, torch.sin(phi), torch.cos(phi), torch.log(energy)], dim=-1)
    charge = 0.25 * tokens[:, :, 4:5]
    pid = 0.25 * tokens[:, :, 5:10]
    tracks = 0.10 * torch.tanh(tokens[:, :, 10:14])
    return torch.cat([kin, charge, pid, tracks], dim=-1)


def pairwise_delta_r(left_tokens, right_tokens):
    torch = require_torch()
    deta = left_tokens[:, :, None, 1] - right_tokens[:, None, :, 1]
    dphi = wrap_phi_torch(left_tokens[:, :, None, 2] - right_tokens[:, None, :, 2])
    return safe_sqrt(deta * deta + dphi * dphi)


def _nan_to_num_torch(value, *, nan: float = 0.0, posinf: float = 0.0, neginf: float = 0.0):
    torch = require_torch()
    if hasattr(torch, "nan_to_num"):
        return torch.nan_to_num(value, nan=float(nan), posinf=float(posinf), neginf=float(neginf))
    return torch.where(torch.isfinite(value), value, torch.zeros_like(value) + float(nan))


def sanitize_token_weight_view(tokens, weights, mask):
    """Sanitize candidate/target tokens before matching and auxiliary losses."""

    torch = require_torch()
    tokens = tokens.float()
    mask = mask.bool()
    finite_tokens = torch.isfinite(tokens).all(dim=-1)
    cleaned = _nan_to_num_torch(tokens)
    pt = torch.clamp(cleaned[:, :, 0], min=0.0)
    eta = torch.clamp(cleaned[:, :, 1], -5.0, 5.0)
    phi = wrap_phi_torch(cleaned[:, :, 2])
    energy = torch.maximum(torch.clamp(cleaned[:, :, 3], min=ENERGY_EPS), physical_energy_floor(pt, eta))
    cleaned = replace_kinematic_channels(cleaned, pt, eta, phi, energy)

    finite_weights = torch.ones_like(mask, dtype=torch.bool)
    if weights is None:
        cleaned_weights = mask.float()
    else:
        weights = weights.float()
        finite_weights = torch.isfinite(weights)
        cleaned_weights = torch.clamp(_nan_to_num_torch(weights), min=0.0)
    safe_mask = mask & finite_tokens & finite_weights
    cleaned_weights = cleaned_weights * safe_mask.float()
    diagnostics = {
        "nonfinite_token_count": (~finite_tokens & mask).sum(dim=1).float(),
        "nonfinite_weight_count": (~finite_weights & mask).sum(dim=1).float(),
        "valid_token_count": safe_mask.sum(dim=1).float(),
    }
    return cleaned, cleaned_weights, safe_mask, diagnostics


def _linear_sum_assignment_numpy(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    """Use SciPy Hungarian assignment when available; otherwise deterministic greedy fallback."""

    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore

        rows, cols = linear_sum_assignment(cost)
        return rows.astype(np.int64), cols.astype(np.int64), True
    except Exception:  # pragma: no cover - depends on optional scipy
        work = np.array(cost, copy=True)
        rows: list[int] = []
        cols: list[int] = []
        n_matches = int(min(work.shape))
        for _ in range(n_matches):
            flat = int(np.argmin(work))
            row, col = np.unravel_index(flat, work.shape)
            if not np.isfinite(work[row, col]):
                break
            rows.append(int(row))
            cols.append(int(col))
            work[row, :] = np.inf
            work[:, col] = np.inf
        return np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64), False


def select_matching_candidates(pred_tokens, pred_weights, pred_mask, *, max_candidates: int):
    torch = require_torch()
    n_candidates = int(pred_tokens.shape[1])
    limit = int(max_candidates) if int(max_candidates) > 0 else n_candidates
    limit = min(limit, n_candidates)
    if limit == n_candidates:
        return pred_tokens, pred_weights, pred_mask
    score = pred_weights * torch.log1p(torch.clamp(pred_tokens[:, :, 0], min=0.0))
    score = score.masked_fill(~pred_mask.bool(), -1.0e9)
    _, indices = torch.topk(score, k=limit, dim=1, largest=True, sorted=False)
    token_indices = indices[:, :, None].expand(-1, -1, pred_tokens.shape[2])
    return (
        torch.gather(pred_tokens, dim=1, index=token_indices),
        torch.gather(pred_weights, dim=1, index=indices),
        torch.gather(pred_mask, dim=1, index=indices),
    )


def assignment_set_matching_loss(
    distances,
    pred_weights,
    pred_mask,
    target_mask,
    *,
    mode: str,
    large_cost: float,
) -> tuple[Any, Dict[str, Any]]:
    """Assignment-style set loss with Hungarian preferred and greedy fallback."""

    torch = require_torch()
    device = distances.device
    batch_size = int(distances.shape[0])
    row_losses = []
    weight_losses = []
    match_counts = []
    used_hungarian = []
    used_fallback = []
    mode = str(mode).lower()
    use_assignment = mode in {"hungarian", "assignment", "linear_sum_assignment"}

    if not use_assignment:
        zero = distances.sum() * 0.0
        return zero, {
            "hungarian_set_loss": zero,
            "matched_weight_loss": zero,
            "matched_count_mean": zero,
            "matching_used_hungarian": zero,
            "matching_used_greedy_fallback": zero,
        }

    for batch_index in range(batch_size):
        pred_idx = torch.nonzero(pred_mask[batch_index], as_tuple=False).flatten()
        target_idx = torch.nonzero(target_mask[batch_index], as_tuple=False).flatten()
        if pred_idx.numel() == 0 or target_idx.numel() == 0:
            row_losses.append(distances[batch_index].sum() * 0.0)
            weight_losses.append(distances[batch_index].sum() * 0.0)
            match_counts.append(0.0)
            used_hungarian.append(0.0)
            used_fallback.append(0.0)
            continue

        cost_tensor = distances[batch_index].index_select(0, pred_idx).index_select(1, target_idx)
        cost_np = cost_tensor.detach().float().cpu().numpy()
        cost_np = np.nan_to_num(
            cost_np,
            nan=float(large_cost),
            posinf=float(large_cost),
            neginf=float(large_cost),
        )
        cost_np = np.clip(cost_np, 0.0, float(large_cost))
        rows_np, cols_np, scipy_used = _linear_sum_assignment_numpy(cost_np)
        if rows_np.size == 0:
            row_losses.append(distances[batch_index].sum() * 0.0)
            weight_losses.append(distances[batch_index].sum() * 0.0)
            match_counts.append(0.0)
            used_hungarian.append(1.0 if scipy_used else 0.0)
            used_fallback.append(0.0 if scipy_used else 1.0)
            continue

        rows = torch.as_tensor(rows_np, dtype=torch.long, device=device)
        cols = torch.as_tensor(cols_np, dtype=torch.long, device=device)
        selected_pred = pred_idx.index_select(0, rows)
        selected_target = target_idx.index_select(0, cols)
        matched_cost = distances[batch_index, selected_pred, selected_target]
        matched_cost = torch.clamp(_nan_to_num_torch(matched_cost, posinf=large_cost, neginf=large_cost), 0.0, large_cost)
        matched_weights = torch.clamp(pred_weights[batch_index, selected_pred], 0.0, 1.0)
        row_losses.append(matched_cost.mean())
        weight_losses.append(((1.0 - matched_weights) ** 2).mean())
        match_counts.append(float(rows_np.size))
        used_hungarian.append(1.0 if scipy_used else 0.0)
        used_fallback.append(0.0 if scipy_used else 1.0)

    return torch.stack(row_losses).mean(), {
        "hungarian_set_loss": torch.stack(row_losses).mean(),
        "matched_weight_loss": torch.stack(weight_losses).mean(),
        "matched_count_mean": torch.tensor(float(np.mean(match_counts)), dtype=distances.dtype, device=device),
        "matching_used_hungarian": torch.tensor(float(np.mean(used_hungarian)), dtype=distances.dtype, device=device),
        "matching_used_greedy_fallback": torch.tensor(float(np.mean(used_fallback)), dtype=distances.dtype, device=device),
    }


def reconstruction_loss(
    output: ReconstructionOutput,
    *,
    hlt_tokens,
    hlt_mask,
    offline_tokens,
    offline_mask,
    config: ReconstructorVariantConfig,
) -> tuple[Any, Dict[str, Any]]:
    """Original-mechanism Stage A loss with assignment matching and branch diagnostics."""

    torch = require_torch()
    pred_tokens, pred_weights, pred_mask, pred_sanitize = sanitize_token_weight_view(
        output.tokens,
        output.weights,
        output.candidate_mask,
    )
    target_tokens, _, target_mask, target_sanitize = sanitize_token_weight_view(
        offline_tokens,
        None,
        offline_mask,
    )
    match_tokens, match_weights, match_mask = select_matching_candidates(
        pred_tokens,
        pred_weights,
        pred_mask,
        max_candidates=int(config.max_matching_candidates),
    )
    pred_feat = matching_features(pred_tokens)
    match_feat = matching_features(match_tokens)
    target_feat = matching_features(target_tokens)
    distances = torch.cdist(pred_feat, target_feat, p=2) ** 2
    match_distances = torch.cdist(match_feat, target_feat, p=2) ** 2
    large_cost = float(config.matching_large_cost)
    distances = _nan_to_num_torch(distances, nan=large_cost, posinf=large_cost, neginf=large_cost)
    match_distances = _nan_to_num_torch(match_distances, nan=large_cost, posinf=large_cost, neginf=large_cost)
    distances = distances.masked_fill(~target_mask[:, None, :], large_cost)
    distances = distances.masked_fill(~pred_mask[:, :, None], large_cost)
    match_distances = match_distances.masked_fill(~target_mask[:, None, :], large_cost)
    match_distances = match_distances.masked_fill(~match_mask[:, :, None], large_cost)

    pred_min = distances.min(dim=2).values
    pred_norm = torch.clamp(pred_weights.sum(dim=1), min=1.0)
    pred_to_target = (pred_min * pred_weights).sum(dim=1) / pred_norm

    weight_penalty = (1.0 - pred_weights).clamp(min=0.0)[:, :, None] ** 2
    target_distances = distances + weight_penalty
    target_min = target_distances.min(dim=1).values
    target_norm = torch.clamp(target_mask.sum(dim=1).float(), min=1.0)
    target_to_pred = (target_min * target_mask.float()).sum(dim=1) / target_norm
    weighted_chamfer_loss = (pred_to_target + target_to_pred).mean()
    assignment_loss, assignment_diag = assignment_set_matching_loss(
        match_distances,
        match_weights,
        match_mask,
        target_mask,
        mode=config.matching_mode,
        large_cost=large_cost,
    )
    matched_weight_loss = assignment_diag["matched_weight_loss"]
    if str(config.matching_mode).lower() in {"hungarian", "assignment", "linear_sum_assignment"}:
        set_loss = assignment_loss + 0.25 * weighted_chamfer_loss + float(config.matched_weight_weight) * matched_weight_loss
    else:
        set_loss = weighted_chamfer_loss

    pred_response = jet_response(pred_tokens, weights=pred_weights, mask=output.candidate_mask)
    target_response = jet_response(target_tokens, mask=target_mask)
    pt_ratio_loss = bounded_log_response_loss(pred_response["pt"], target_response["pt"])
    energy_ratio_loss = bounded_log_response_loss(pred_response["energy"], target_response["energy"])
    mass_ratio_loss = bounded_log_response_loss(pred_response["mass"], target_response["mass"])

    target_count = target_mask.sum(dim=1).float()
    hlt_count = hlt_mask.sum(dim=1).float()
    target_added = torch.clamp((target_count - hlt_count) * float(config.target_added_particle_scale), min=0.0)
    predicted_total = output.total_count_pred + hlt_count * 0.0
    predicted_added = output.added_count_pred
    actual_total_weight = pred_weights.sum(dim=1)
    split_weight_count = output.split_weights.shape[1]
    generated_weight_count = output.generated_weights.shape[1]
    split_weights_safe = pred_weights[
        :,
        output.edited_weights.shape[1] : output.edited_weights.shape[1] + split_weight_count,
    ].clone()
    generated_weights_safe = (
        pred_weights[:, -generated_weight_count:].clone()
        if generated_weight_count
        else pred_weights[:, :0].clone()
    )
    actual_added_weight = split_weights_safe.sum(dim=1) + generated_weights_safe.sum(dim=1)
    count_loss = (
        (torch.log1p(predicted_total) - torch.log1p(target_count)) ** 2
        + (torch.log1p(predicted_added) - torch.log1p(target_added)) ** 2
        + 0.25 * (torch.log1p(actual_total_weight) - torch.log1p(target_count)) ** 2
        + 0.25 * (torch.log1p(actual_added_weight) - torch.log1p(target_added)) ** 2
    ).mean()

    split_sparsity_loss = split_weights_safe.mean() if split_weights_safe.numel() else actual_total_weight.mean() * 0.0
    generated_sparsity_loss = generated_weights_safe.mean() if generated_weights_safe.numel() else actual_total_weight.mean() * 0.0
    sparsity_loss = split_sparsity_loss + generated_sparsity_loss

    if output.split_child_tokens is not None:
        split_tokens_for_locality, _, _, split_sanitize = sanitize_token_weight_view(
            output.split_child_tokens.reshape(output.split_tokens.shape),
            split_weights_safe,
            hlt_mask[:, :, None].expand_as(output.split_child_weights).reshape(output.split_weights.shape),
        )
        parent_tokens_for_split, _, _, _ = sanitize_token_weight_view(
            hlt_tokens[:, :, None, :].expand_as(output.split_child_tokens).reshape(output.split_tokens.shape),
            None,
            hlt_mask[:, :, None].expand_as(output.split_child_weights).reshape(output.split_weights.shape),
        )
        split_mask_for_locality = hlt_mask[:, :, None].expand_as(output.split_child_weights).reshape(output.split_weights.shape)
    else:
        split_tokens_for_locality, _, _, split_sanitize = sanitize_token_weight_view(
            output.split_tokens,
            split_weights_safe,
            hlt_mask,
        )
        parent_tokens_for_split, _, _, _ = sanitize_token_weight_view(hlt_tokens, None, hlt_mask)
        split_mask_for_locality = hlt_mask
    split_dr = safe_sqrt(
        (split_tokens_for_locality[:, :, 1] - parent_tokens_for_split[:, :, 1]) ** 2
        + wrap_phi_torch(split_tokens_for_locality[:, :, 2] - parent_tokens_for_split[:, :, 2]) ** 2
    )
    split_excess = torch.relu(split_dr - float(config.split_locality_radius)) ** 2
    split_local = (split_excess * split_weights_safe * split_mask_for_locality.float()).sum(dim=1) / torch.clamp(
        (split_weights_safe * split_mask_for_locality.float()).sum(dim=1),
        min=1.0,
    )
    if output.generated_tokens.shape[1] > 0:
        generated_tokens_safe, _, _, gen_sanitize = sanitize_token_weight_view(
            output.generated_tokens,
            generated_weights_safe,
            torch.ones_like(output.generated_weights, dtype=torch.bool),
        )
        hlt_tokens_safe, _, hlt_mask_safe, _ = sanitize_token_weight_view(hlt_tokens, None, hlt_mask)
        gen_dr = pairwise_delta_r(generated_tokens_safe, hlt_tokens_safe).masked_fill(~hlt_mask_safe[:, None, :], 1.0e3)
        gen_nearest = gen_dr.min(dim=2).values
        gen_excess = torch.relu(gen_nearest - float(config.generated_locality_radius)) ** 2
        gen_local = (gen_excess * generated_weights_safe).sum(dim=1) / torch.clamp(
            generated_weights_safe.sum(dim=1),
            min=1.0,
        )
        locality_loss = (split_local + gen_local).mean()
    else:
        generated_tokens_safe = output.generated_tokens
        gen_sanitize = {
            "nonfinite_token_count": split_local * 0.0,
            "nonfinite_weight_count": split_local * 0.0,
            "valid_token_count": split_local * 0.0,
        }
        locality_loss = split_local.mean()

    added_tokens = torch.cat([split_tokens_for_locality, generated_tokens_safe], dim=1)
    added_weights = torch.cat([split_weights_safe, generated_weights_safe], dim=1)
    if added_tokens.shape[1] > 1:
        added_dr = pairwise_delta_r(added_tokens, added_tokens)
        eye = torch.eye(added_tokens.shape[1], dtype=torch.bool, device=added_tokens.device)[None, :, :]
        pair_weights = added_weights[:, :, None] * added_weights[:, None, :]
        close_penalty = torch.relu(float(config.anti_overlap_radius) - added_dr) ** 2
        close_penalty = close_penalty.masked_fill(eye, 0.0)
        pair_weights = pair_weights.masked_fill(eye, 0.0)
        anti_overlap_loss = (close_penalty * pair_weights).sum(dim=(1, 2)) / torch.clamp(
            pair_weights.sum(dim=(1, 2)),
            min=1.0,
        )
        anti_overlap_loss = anti_overlap_loss.mean()
    else:
        anti_overlap_loss = locality_loss * 0.0

    nonfinite_candidate_fraction = (
        pred_sanitize["nonfinite_token_count"] + pred_sanitize["nonfinite_weight_count"]
    ) / torch.clamp(output.candidate_mask.sum(dim=1).float(), min=1.0)
    nonfinite_target_fraction = target_sanitize["nonfinite_token_count"] / torch.clamp(offline_mask.sum(dim=1).float(), min=1.0)
    nonfinite_branch_fraction = (
        split_sanitize["nonfinite_token_count"] + split_sanitize["nonfinite_weight_count"]
        + gen_sanitize["nonfinite_token_count"] + gen_sanitize["nonfinite_weight_count"]
    ) / torch.clamp(
        torch.tensor(
            float(max(1, output.split_weights.shape[1] + output.generated_weights.shape[1])),
            dtype=pred_weights.dtype,
            device=pred_weights.device,
        ),
        min=1.0,
    )
    nonfinite_penalty = (
        nonfinite_candidate_fraction.mean()
        + nonfinite_target_fraction.mean()
        + nonfinite_branch_fraction.mean()
    )

    total = (
        float(config.set_matching_weight) * set_loss
        + float(config.budget_count_weight) * count_loss
        + float(config.sparsity_weight) * sparsity_loss
        + float(config.split_sparsity_weight) * split_sparsity_loss
        + float(config.generated_sparsity_weight) * generated_sparsity_loss
        + float(config.locality_weight) * locality_loss
        + float(config.anti_overlap_weight) * anti_overlap_loss
        + float(config.pt_ratio_weight) * pt_ratio_loss
        + float(config.energy_ratio_weight) * energy_ratio_loss
        + float(config.mass_ratio_weight) * mass_ratio_loss
        + float(config.nonfinite_penalty_weight) * nonfinite_penalty
    )

    diagnostics = {
        "total_loss": total,
        "set_loss": set_loss,
        "weighted_chamfer_loss": weighted_chamfer_loss,
        **assignment_diag,
        "count_loss": count_loss,
        "sparsity_loss": sparsity_loss,
        "split_sparsity_loss": split_sparsity_loss,
        "generated_sparsity_loss": generated_sparsity_loss,
        "locality_loss": locality_loss,
        "anti_overlap_loss": anti_overlap_loss,
        "nonfinite_penalty": nonfinite_penalty,
        "nonfinite_candidate_count": pred_sanitize["nonfinite_token_count"].mean()
        + pred_sanitize["nonfinite_weight_count"].mean(),
        "nonfinite_target_count": target_sanitize["nonfinite_token_count"].mean(),
        "matching_candidate_count": match_mask.sum(dim=1).float().mean(),
        "matching_target_count": target_mask.sum(dim=1).float().mean(),
        "pt_ratio_loss": pt_ratio_loss,
        "energy_ratio_loss": energy_ratio_loss,
        "mass_ratio_loss": mass_ratio_loss,
        "pred_pt_mean": pred_response["pt"].mean(),
        "target_pt_mean": target_response["pt"].mean(),
        "pred_count_mean": actual_total_weight.mean(),
        "target_count_mean": target_count.mean(),
        "pred_added_mean": actual_added_weight.mean(),
        "target_added_mean": target_added.mean(),
    }
    return total, diagnostics


class PairedReconstructionDataset(_DatasetBase):
    """Paired cached-HLT/offline dataset for Stage A."""

    def __init__(self, hlt_view: JetView, offline_view: JetView, *, max_jets: int | None = None) -> None:
        require_torch()
        if hlt_view.split != offline_view.split:
            raise ValueError(f"Split mismatch: {hlt_view.split} != {offline_view.split}")
        if list(hlt_view.jet_ids) != list(offline_view.jet_ids):
            raise ValueError("HLT and offline views are not aligned by jet identity")
        if not np.array_equal(hlt_view.labels, offline_view.labels):
            raise ValueError("HLT and offline labels differ")
        limit = len(hlt_view.labels) if max_jets is None else min(int(max_jets), len(hlt_view.labels))
        self.hlt_tokens = np.asarray(hlt_view.tokens[:limit], dtype=np.float32)
        self.hlt_mask = np.asarray(hlt_view.mask[:limit], dtype=bool)
        self.offline_tokens = np.asarray(offline_view.tokens[:limit], dtype=np.float32)
        self.offline_mask = np.asarray(offline_view.mask[:limit], dtype=bool)
        self.labels = np.asarray(hlt_view.labels[:limit], dtype=np.int64)
        self.split = hlt_view.split

    def __len__(self) -> int:
        return int(len(self.labels))

    def __getitem__(self, index: int):
        return (
            self.hlt_tokens[index],
            self.hlt_mask[index],
            self.offline_tokens[index],
            self.offline_mask[index],
            self.labels[index],
        )


def collate_reconstruction_batch(samples):
    torch = require_torch()
    return {
        "hlt_tokens": torch.from_numpy(np.stack([row[0] for row in samples], axis=0)).float(),
        "hlt_mask": torch.from_numpy(np.stack([row[1] for row in samples], axis=0)).bool(),
        "offline_tokens": torch.from_numpy(np.stack([row[2] for row in samples], axis=0)).float(),
        "offline_mask": torch.from_numpy(np.stack([row[3] for row in samples], axis=0)).bool(),
        "labels": torch.from_numpy(np.asarray([row[4] for row in samples], dtype=np.int64)).long(),
    }


def make_reconstruction_loader(dataset, *, batch_size: int, shuffle: bool, num_workers: int, seed: int):
    torch = require_torch()
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_reconstruction_batch,
        generator=generator,
    )


def move_batch_to_device(batch: Mapping[str, Any], device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def summarize_loss_dict(losses: List[Dict[str, float]]) -> Dict[str, float]:
    if not losses:
        return {"n_jets": 0}
    keys = sorted({key for row in losses for key in row})
    summary = {}
    weights = np.asarray([row.get("n_jets", 0) for row in losses], dtype=np.float64)
    weights = np.maximum(weights, 1.0)
    for key in keys:
        if key == "n_jets":
            continue
        values = np.asarray([row.get(key, np.nan) for row in losses], dtype=np.float64)
        valid = np.isfinite(values)
        summary[key] = float(np.average(values[valid], weights=weights[valid])) if np.any(valid) else float("nan")
    summary["n_jets"] = int(np.sum([row.get("n_jets", 0) for row in losses]))
    return summary


def run_reconstruction_epoch(
    model,
    loader,
    *,
    device,
    variant_config: ReconstructorVariantConfig,
    optimizer=None,
    scaler=None,
    amp: bool = False,
    grad_clip_norm: float = 0.0,
    max_batches: int | None = None,
) -> Dict[str, float]:
    torch = require_torch()
    is_train = optimizer is not None
    model.train(is_train)
    rows: List[Dict[str, float]] = []
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
                output = model(batch["hlt_tokens"], batch["hlt_mask"])
                loss, diagnostics = reconstruction_loss(
                    output,
                    hlt_tokens=batch["hlt_tokens"],
                    hlt_mask=batch["hlt_mask"],
                    offline_tokens=batch["offline_tokens"],
                    offline_mask=batch["offline_mask"],
                    config=variant_config,
                )
                if not torch.isfinite(loss):
                    diag = {
                        key: float(value.detach().item())
                        for key, value in diagnostics.items()
                        if hasattr(value, "detach") and value.numel() == 1
                    }
                    raise FloatingPointError(
                        f"Non-finite Stage A reconstruction loss in batch {batch_index}: {diag}"
                    )
            if is_train:
                if scaler is not None and autocast_enabled:
                    scaler.scale(loss).backward()
                    if grad_clip_norm and grad_clip_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            float(grad_clip_norm),
                            error_if_nonfinite=True,
                        )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip_norm and grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            float(grad_clip_norm),
                            error_if_nonfinite=True,
                        )
                    optimizer.step()
            row = {key: float(value.detach().item()) for key, value in diagnostics.items()}
            row["n_jets"] = int(batch["hlt_tokens"].shape[0])
            rows.append(row)
    return summarize_loss_dict(rows)


def reconstructor_checkpoint_payload(
    model,
    optimizer,
    *,
    epoch: int,
    config: StageAReconstructorTrainConfig,
    variant_config: ReconstructorVariantConfig,
    metrics: Mapping[str, Any],
):
    return {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(config),
        "variant_config": asdict(variant_config),
        "metrics": dict(metrics),
        "experiment_step": "step7_stage_a_reconstructor",
    }


def load_stage_a_views(
    config: StageAReconstructorTrainConfig,
    split: str,
    *,
    manifest=None,
) -> tuple[JetView, JetView]:
    manifest = manifest or load_split_manifest(config.manifest_path)
    hlt_view = load_cached_hlt_view(config.hlt_cache_dir, split)
    offline_view = load_offline_view(
        manifest,
        split,
        data_dir=config.data_dir,
        verify_label_branches=config.verify_label_branches,
        read_chunk_size=config.read_chunk_size,
    )
    if list(hlt_view.jet_ids) != list(offline_view.jet_ids):
        raise ValueError(f"Cached HLT and offline views are not aligned for split {split}")
    return hlt_view, offline_view


def train_stage_a_reconstructor(
    config: StageAReconstructorTrainConfig,
    *,
    model=None,
    train_hlt_view: JetView | None = None,
    train_offline_view: JetView | None = None,
    val_hlt_view: JetView | None = None,
    val_offline_view: JetView | None = None,
    max_train_jets: int | None = None,
    max_val_jets: int | None = None,
) -> Dict[str, Any]:
    """Train Stage A HLT-to-offline reconstruction for `m2_base`."""

    if config.train_split != "model_train" or config.val_split != "model_val":
        raise ValueError("Step 7 may train only on model_train and select only on model_val")
    variant_config = get_reconstructor_variant_config(config.variant)

    torch = require_torch()
    set_training_seed(config.seed)
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = None
    manifest_sha = None
    if train_hlt_view is None or train_offline_view is None or val_hlt_view is None or val_offline_view is None:
        manifest = load_split_manifest(config.manifest_path)
        manifest_sha = manifest_hash(manifest)
    if train_hlt_view is None or train_offline_view is None:
        train_hlt_view, train_offline_view = load_stage_a_views(config, config.train_split, manifest=manifest)
    if val_hlt_view is None or val_offline_view is None:
        val_hlt_view, val_offline_view = load_stage_a_views(config, config.val_split, manifest=manifest)
    if manifest_sha is None:
        manifest_sha = train_offline_view.metadata.get("source_manifest_hash")

    train_dataset = PairedReconstructionDataset(
        train_hlt_view,
        train_offline_view,
        max_jets=max_train_jets,
    )
    val_dataset = PairedReconstructionDataset(
        val_hlt_view,
        val_offline_view,
        max_jets=max_val_jets,
    )
    train_loader = make_reconstruction_loader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    val_loader = make_reconstruction_loader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        seed=config.seed + 1,
    )

    model = model or build_reconstructor(variant_config)
    model = model.to(device)
    if config.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.lr), weight_decay=float(config.weight_decay))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(config.amp and device.type == "cuda"))

    run_metadata = {
        "config": asdict(config),
        "variant_config": asdict(variant_config),
        "manifest_hash": manifest_sha,
        "train_hlt_hash": train_hlt_view.metadata.get("hlt_content_hash"),
        "val_hlt_hash": val_hlt_view.metadata.get("hlt_content_hash"),
        "train_n_jets": len(train_dataset),
        "val_n_jets": len(val_dataset),
        "stage": "A_reconstruction_only",
        "leakage_rule": (
            "Offline constituents are supervised targets for model_train/model_val only. "
            "The reconstructor inference path consumes fixed-HLT tokens only."
        ),
        "no_stack_or_final_test_partitions_loaded": True,
    }
    save_json(output_dir / "config.json", run_metadata)

    curves: List[Dict[str, Any]] = []
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    for epoch in range(1, int(config.epochs) + 1):
        train_metrics = run_reconstruction_epoch(
            model,
            train_loader,
            device=device,
            variant_config=variant_config,
            optimizer=optimizer,
            scaler=scaler,
            amp=config.amp,
            grad_clip_norm=config.grad_clip_norm,
            max_batches=config.max_train_batches,
        )
        val_metrics = run_reconstruction_epoch(
            model,
            val_loader,
            device=device,
            variant_config=variant_config,
            amp=False,
            max_batches=config.max_val_batches,
        )
        row = {
            "epoch": int(epoch),
            "train": train_metrics,
            "model_val": val_metrics,
        }
        curves.append(row)
        save_json(output_dir / "training_curves.json", {"epochs": curves})

        val_loss = float(val_metrics["total_loss"])
        improved = np.isfinite(val_loss) and val_loss < best_val_loss
        torch.save(
            reconstructor_checkpoint_payload(
                model,
                optimizer,
                epoch=epoch,
                config=config,
                variant_config=variant_config,
                metrics=row,
            ),
            output_dir / "last.pt",
        )
        if improved:
            best_val_loss = val_loss
            best_epoch = int(epoch)
            epochs_without_improvement = 0
            torch.save(
                reconstructor_checkpoint_payload(
                    model,
                    optimizer,
                    epoch=epoch,
                    config=config,
                    variant_config=variant_config,
                    metrics=row,
                ),
                output_dir / "best_model_val.pt",
            )
        else:
            epochs_without_improvement += 1
        if config.early_stop_patience >= 0 and epochs_without_improvement >= int(config.early_stop_patience):
            break

    if best_epoch < 0 or not (output_dir / "best_model_val.pt").exists():
        raise FloatingPointError(
            "Stage A did not produce a finite model_val total_loss, so no best_model_val.pt was written"
        )

    report = {
        "experiment_step": "step7_stage_a_reconstructor",
        "variant": config.variant,
        "best_epoch": int(best_epoch),
        "best_model_val_total_loss": float(best_val_loss),
        "epochs_completed": len(curves),
        "final_epoch": curves[-1] if curves else None,
        "checkpoint": str(output_dir / "best_model_val.pt"),
        "last_checkpoint": str(output_dir / "last.pt"),
        "no_final_test_evaluation": True,
        "not_a_classifier": True,
    }
    save_json(output_dir / "model_val_reconstruction_report.json", report)
    return report
