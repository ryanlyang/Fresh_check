"""Stage A HLT-to-offline reconstructors for the reco7 family.

The Step 7 implementation started with `m2_base`; Step 9 expands the same
architecture/loss surface to the seven named reconstructor variants while
leaving HLT generation and split definitions untouched.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
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
    hidden_dim: int = 128
    global_dim: int = 128
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
    lr: float = 1.0e-3
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


def wrap_phi_torch(phi):
    torch = require_torch()
    return torch.remainder(phi + torch.pi, 2.0 * torch.pi) - torch.pi


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


class M2BaseReconstructor(_ModuleBase):
    """Operation-aware HLT-to-offline reconstructor for the `m2_base` variant."""

    def __init__(self, config: ReconstructorVariantConfig | None = None) -> None:
        require_torch()
        super().__init__()
        torch = require_torch()
        self.config = config or m2_base_variant_config()
        input_dim = 16
        hidden = int(self.config.hidden_dim)
        global_dim = int(self.config.global_dim)
        self.token_encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.GELU(),
        )
        self.global_encoder = torch.nn.Sequential(
            torch.nn.Linear(hidden + 3, global_dim),
            torch.nn.GELU(),
            torch.nn.Linear(global_dim, global_dim),
            torch.nn.GELU(),
        )
        self.edit_head = torch.nn.Linear(hidden + global_dim, 5)
        self.split_head = torch.nn.Linear(hidden + global_dim, 5)
        self.generated_query = torch.nn.Parameter(torch.randn(self.config.max_generated, global_dim) * 0.02)
        self.generated_head = torch.nn.Sequential(
            torch.nn.Linear(global_dim * 2, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, 15),
        )
        self.count_head = torch.nn.Sequential(
            torch.nn.Linear(global_dim, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, 2),
        )

    def _apply_kinematic_delta(self, base_tokens, delta, *, split: bool = False):
        torch = require_torch()
        cfg = self.config
        max_log_pt = cfg.max_split_log_pt_shift if split else cfg.max_log_pt_shift
        log_pt_delta = torch.tanh(delta[:, :, 0]) * float(max_log_pt)
        eta_delta = torch.tanh(delta[:, :, 1]) * float(cfg.max_eta_shift)
        phi_delta = torch.tanh(delta[:, :, 2]) * float(cfg.max_phi_shift)
        log_e_delta = torch.tanh(delta[:, :, 3]) * float(cfg.max_log_energy_shift)

        out = base_tokens.clone()
        pt = torch.clamp(base_tokens[:, :, 0], min=1.0e-8) * torch.exp(log_pt_delta)
        eta = torch.clamp(base_tokens[:, :, 1] + eta_delta, -5.0, 5.0)
        phi = wrap_phi_torch(base_tokens[:, :, 2] + phi_delta)
        energy = torch.clamp(base_tokens[:, :, 3], min=1.0e-8) * torch.exp(log_e_delta)
        out[:, :, 0] = pt
        out[:, :, 1] = eta
        out[:, :, 2] = phi
        out[:, :, 3] = torch.maximum(energy, pt * torch.cosh(eta) * 0.5)
        return out

    def forward(self, hlt_tokens, hlt_mask) -> ReconstructionOutput:
        torch = require_torch()
        hlt_tokens = hlt_tokens.float()
        hlt_mask = hlt_mask.bool()
        features = raw_token_features(hlt_tokens, hlt_mask)
        encoded = self.token_encoder(features) * hlt_mask.unsqueeze(-1).float()
        denom = torch.clamp(hlt_mask.sum(dim=1, keepdim=True).float(), min=1.0)
        pooled = encoded.sum(dim=1) / denom
        hlt_count = hlt_mask.sum(dim=1, keepdim=False).float()
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
        global_per_token = global_latent[:, None, :].expand(-1, hlt_tokens.shape[1], -1)
        token_context = torch.cat([encoded, global_per_token], dim=-1)

        edit_raw = self.edit_head(token_context)
        split_raw = self.split_head(token_context)
        edited_tokens = self._apply_kinematic_delta(hlt_tokens, edit_raw[:, :, :4], split=False)
        edited_weights = torch.sigmoid(edit_raw[:, :, 4]) * hlt_mask.float()

        split_tokens = self._apply_kinematic_delta(hlt_tokens, split_raw[:, :, :4], split=True)
        split_weights = torch.sigmoid(split_raw[:, :, 4]) * hlt_mask.float()

        batch_size = hlt_tokens.shape[0]
        query = self.generated_query[None, :, :].expand(batch_size, -1, -1)
        gen_global = global_latent[:, None, :].expand(-1, self.config.max_generated, -1)
        generated_raw = self.generated_head(torch.cat([query, gen_global], dim=-1))
        generated_tokens = torch.zeros(
            batch_size,
            self.config.max_generated,
            RAW_DIM,
            dtype=hlt_tokens.dtype,
            device=hlt_tokens.device,
        )
        gen_pt = torch.nn.functional.softplus(generated_raw[:, :, 0]) + 1.0e-4
        gen_eta = torch.tanh(generated_raw[:, :, 1]) * float(self.config.max_generated_abs_eta)
        gen_phi = wrap_phi_torch(generated_raw[:, :, 2])
        gen_energy = torch.nn.functional.softplus(generated_raw[:, :, 3]) + gen_pt * torch.cosh(gen_eta) * 0.5
        generated_tokens[:, :, 0] = gen_pt
        generated_tokens[:, :, 1] = gen_eta
        generated_tokens[:, :, 2] = gen_phi
        generated_tokens[:, :, 3] = gen_energy
        generated_tokens[:, :, 4:14] = torch.tanh(generated_raw[:, :, 4:14])
        generated_weights = torch.sigmoid(generated_raw[:, :, 14])

        tokens = torch.cat([edited_tokens, split_tokens, generated_tokens], dim=1)
        weights = torch.cat([edited_weights, split_weights, generated_weights], dim=1)
        candidate_mask = torch.cat(
            [
                hlt_mask,
                hlt_mask,
                torch.ones(
                    batch_size,
                    self.config.max_generated,
                    dtype=torch.bool,
                    device=hlt_tokens.device,
                ),
            ],
            dim=1,
        )
        count_raw = self.count_head(global_latent)
        total_count_pred = torch.nn.functional.softplus(count_raw[:, 0])
        added_count_pred = torch.nn.functional.softplus(count_raw[:, 1])

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
    jet_pt = torch.sqrt(torch.clamp(jet_px * jet_px + jet_py * jet_py, min=0.0))
    mass2 = jet_energy * jet_energy - jet_px * jet_px - jet_py * jet_py - jet_pz * jet_pz
    jet_mass = torch.sqrt(torch.clamp(mass2, min=0.0))
    return {
        "pt": jet_pt,
        "energy": jet_energy,
        "mass": jet_mass,
    }


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
    return torch.sqrt(torch.clamp(deta * deta + dphi * dphi, min=0.0))


def reconstruction_loss(
    output: ReconstructionOutput,
    *,
    hlt_tokens,
    hlt_mask,
    offline_tokens,
    offline_mask,
    config: ReconstructorVariantConfig,
) -> tuple[Any, Dict[str, Any]]:
    """Differentiable Stage A loss with Chamfer-style set matching."""

    torch = require_torch()
    pred_tokens = output.tokens
    pred_weights = output.weights * output.candidate_mask.float()
    target_mask = offline_mask.bool()
    pred_feat = matching_features(pred_tokens)
    target_feat = matching_features(offline_tokens)
    distances = torch.cdist(pred_feat, target_feat, p=2) ** 2
    distances = distances.masked_fill(~target_mask[:, None, :], 1.0e6)

    pred_min = distances.min(dim=2).values
    pred_norm = torch.clamp(pred_weights.sum(dim=1), min=1.0)
    pred_to_target = (pred_min * pred_weights).sum(dim=1) / pred_norm

    weight_penalty = (1.0 - pred_weights).clamp(min=0.0)[:, :, None] ** 2
    target_distances = distances + weight_penalty
    target_min = target_distances.min(dim=1).values
    target_norm = torch.clamp(target_mask.sum(dim=1).float(), min=1.0)
    target_to_pred = (target_min * target_mask.float()).sum(dim=1) / target_norm
    set_loss = (pred_to_target + target_to_pred).mean()

    pred_response = jet_response(pred_tokens, weights=pred_weights, mask=output.candidate_mask)
    target_response = jet_response(offline_tokens, mask=offline_mask)
    pt_ratio_loss = ((pred_response["pt"] / torch.clamp(target_response["pt"], min=1.0e-6) - 1.0) ** 2).mean()
    energy_ratio_loss = ((pred_response["energy"] / torch.clamp(target_response["energy"], min=1.0e-6) - 1.0) ** 2).mean()
    mass_ratio_loss = ((pred_response["mass"] / torch.clamp(target_response["mass"], min=1.0e-6) - 1.0) ** 2).mean()

    target_count = offline_mask.sum(dim=1).float()
    hlt_count = hlt_mask.sum(dim=1).float()
    target_added = torch.clamp((target_count - hlt_count) * float(config.target_added_particle_scale), min=0.0)
    predicted_total = output.total_count_pred + hlt_count * 0.0
    predicted_added = output.added_count_pred
    actual_total_weight = pred_weights.sum(dim=1)
    actual_added_weight = output.split_weights.sum(dim=1) + output.generated_weights.sum(dim=1)
    count_loss = (
        (torch.log1p(predicted_total) - torch.log1p(target_count)) ** 2
        + (torch.log1p(predicted_added) - torch.log1p(target_added)) ** 2
        + 0.25 * (torch.log1p(actual_total_weight) - torch.log1p(target_count)) ** 2
        + 0.25 * (torch.log1p(actual_added_weight) - torch.log1p(target_added)) ** 2
    ).mean()

    sparsity_loss = output.generated_weights.mean()

    split_dr = torch.sqrt(
        torch.clamp(
            (output.split_tokens[:, :, 1] - hlt_tokens[:, :, 1]) ** 2
            + wrap_phi_torch(output.split_tokens[:, :, 2] - hlt_tokens[:, :, 2]) ** 2,
            min=0.0,
        )
    )
    split_excess = torch.relu(split_dr - float(config.split_locality_radius)) ** 2
    split_local = (split_excess * output.split_weights * hlt_mask.float()).sum(dim=1) / torch.clamp(
        (output.split_weights * hlt_mask.float()).sum(dim=1),
        min=1.0,
    )
    if output.generated_tokens.shape[1] > 0:
        gen_dr = pairwise_delta_r(output.generated_tokens, hlt_tokens).masked_fill(~hlt_mask[:, None, :], 1.0e3)
        gen_nearest = gen_dr.min(dim=2).values
        gen_excess = torch.relu(gen_nearest - float(config.generated_locality_radius)) ** 2
        gen_local = (gen_excess * output.generated_weights).sum(dim=1) / torch.clamp(
            output.generated_weights.sum(dim=1),
            min=1.0,
        )
        locality_loss = (split_local + gen_local).mean()
    else:
        locality_loss = split_local.mean()

    added_tokens = torch.cat([output.split_tokens, output.generated_tokens], dim=1)
    added_weights = torch.cat([output.split_weights, output.generated_weights], dim=1)
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

    total = (
        float(config.set_matching_weight) * set_loss
        + float(config.budget_count_weight) * count_loss
        + float(config.sparsity_weight) * sparsity_loss
        + float(config.locality_weight) * locality_loss
        + float(config.anti_overlap_weight) * anti_overlap_loss
        + float(config.pt_ratio_weight) * pt_ratio_loss
        + float(config.energy_ratio_weight) * energy_ratio_loss
        + float(config.mass_ratio_weight) * mass_ratio_loss
    )

    diagnostics = {
        "total_loss": total,
        "set_loss": set_loss,
        "count_loss": count_loss,
        "sparsity_loss": sparsity_loss,
        "locality_loss": locality_loss,
        "anti_overlap_loss": anti_overlap_loss,
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
        valid = ~np.isnan(values)
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
            if is_train:
                if scaler is not None and autocast_enabled:
                    scaler.scale(loss).backward()
                    if grad_clip_norm and grad_clip_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip_norm and grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
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

        improved = val_metrics["total_loss"] < best_val_loss
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
            best_val_loss = float(val_metrics["total_loss"])
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
