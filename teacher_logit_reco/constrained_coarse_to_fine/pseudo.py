"""HLT-only pseudo-particle rendering and sharded cache support for Step 6."""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch

from jetclass_fixed_hlt import HLT_PROFILE_V2_REALISTIC, HLT_PROFILE_V2_REALISTIC_VERSION
from jetclass_fresh.hlt_baseline import amp_autocast_context, resolve_device, set_training_seed
from jetclass_fresh.hlt_cache import (
    hash_arrays,
    jet_identity_hash,
    load_cached_hlt_view,
    load_hlt_metadata,
    normalize_hlt_profile,
)
from jetclass_fresh.jetclass_data import JetIdentity, RAW_TOKEN_DIM, load_split_manifest, manifest_hash
from jetclass_fresh.part_inputs import build_particle_transformer_inputs_from_tokens

from .constraints import ACCOUNTING_INDEX, PID_PT_INDICES
from .layout import ACCOUNTING_FIELD_NAMES, MOMENT_FIELD_NAMES, PID_CATEGORY_NAMES
from .model import CoarseToFineReconstructorConfig
from .slots import (
    CTierParticleReconstructor,
    CTierReconstructorOutput,
    ParticleSlotDecoderConfig,
    build_c_tier_reconstructor,
    normalize_c_tier_variant,
)
from .targets import hlt_reference_axis, wrap_phi
from .train import COARSE_TO_FINE_TRAIN_CONTRACT


PSEUDO_PARTICLE_CACHE_CONTRACT = "constrained_coarse_to_fine_pseudo_particle_cache_v2"
PSEUDO_PARTICLE_CACHE_SET_CONTRACT = "constrained_coarse_to_fine_pseudo_particle_cache_set_v1"
PSEUDO_PARTICLE_RENDER_CONTRACT = "constrained_coarse_to_fine_pseudo_particle_render_v1"
PSEUDO_PARTICLE_ALLOWED_SPLITS = (
    "model_train",
    "model_val",
    "stack_train",
    "stack_val",
    "final_test",
)
PSEUDO_PARTICLE_DEFAULT_SPLITS = ("model_train", "model_val")
PSEUDO_PARTICLE_UNCERTAINTY_NAMES = ("log_pT", "deta", "dphi", "energy", "pid")

_PSEUDO_SHARD_HASH_KEYS = (
    "tokens",
    "mask",
    "candidate_weights",
    "existence_probability",
    "reliability",
    "expected_count",
    "slot_log_sigma",
    "uncertainty_mask",
    "rendered_accounting",
    "global_accounting",
    "global_log_sigma",
    "level1_accounting",
    "level1_log_sigma",
    "level2_accounting",
    "level2_log_sigma",
    "level3_accounting",
    "level3_log_sigma",
    "reference_eta",
    "reference_phi",
    "labels",
    "jet_file_indices",
    "jet_entries",
    "token_cell_indices",
    "token_slot_indices",
    "token_is_dust",
    "view_indices",
)


@dataclass(frozen=True)
class PseudoParticleRenderOutput:
    tokens: torch.Tensor
    mask: torch.Tensor
    candidate_weights: torch.Tensor
    existence_probability: torch.Tensor
    reliability: torch.Tensor
    expected_count: torch.Tensor
    slot_log_sigma: torch.Tensor
    uncertainty_mask: torch.Tensor
    rendered_accounting: torch.Tensor
    token_cell_indices: torch.Tensor
    token_slot_indices: torch.Tensor
    token_is_dust: torch.Tensor
    view_indices: torch.Tensor
    hierarchy: Any
    diagnostics: Mapping[str, Any]

    @property
    def num_views(self) -> int:
        return int(self.tokens.shape[1])

    @property
    def num_tokens(self) -> int:
        return int(self.tokens.shape[2])


@dataclass(frozen=True)
class PseudoParticleShard:
    arrays: Mapping[str, np.ndarray]
    labels: np.ndarray
    jet_ids: tuple[JetIdentity, ...]
    split: str
    variant: str
    shard_index: int
    start: int
    stop: int
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class PseudoParticleCache:
    arrays: Mapping[str, np.ndarray]
    labels: np.ndarray
    jet_ids: tuple[JetIdentity, ...]
    split: str
    variant: str
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class PseudoParticleCacheConfig:
    output_cache_dir: str
    manifest_path: str
    hlt_cache_dir: str
    reconstructor_checkpoint: str
    splits: tuple[str, ...] = PSEUDO_PARTICLE_DEFAULT_SPLITS
    batch_size: int = 16
    shard_size: int = 1024
    device: str = "auto"
    amp: bool = True
    cache_dtype: str = "float16"
    num_views: int | None = None
    min_particle_pt: float = 0.0
    dust_reliability: float = 0.10
    max_jets_per_split: int | None = None
    seed: int = 26061
    verify_hlt_hash: bool = True
    strict_checkpoint: bool = True
    require_model_val_checkpoint: bool = True
    overwrite: bool = False
    skip_existing: bool = True
    confirm_final_test: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "splits", tuple(str(split) for split in self.splits))
        if not self.splits or len(set(self.splits)) != len(self.splits):
            raise ValueError("splits must be nonempty and unique")
        unknown = sorted(set(self.splits) - set(PSEUDO_PARTICLE_ALLOWED_SPLITS))
        if unknown:
            raise ValueError(f"unsupported pseudo-particle splits: {unknown}")
        if "final_test" in self.splits and not bool(self.confirm_final_test):
            raise ValueError("--confirm-final-test is required to render final_test pseudo-particles")
        for name in ("batch_size", "shard_size"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.num_views is not None and int(self.num_views) <= 0:
            raise ValueError("num_views must be positive when provided")
        if self.max_jets_per_split is not None and int(self.max_jets_per_split) <= 0:
            raise ValueError("max_jets_per_split must be positive when provided")
        if str(self.cache_dtype) not in {"float16", "float32"}:
            raise ValueError("cache_dtype must be float16 or float32")
        if float(self.min_particle_pt) < 0.0:
            raise ValueError("min_particle_pt must be nonnegative")
        if not 0.0 <= float(self.dust_reliability) <= 1.0:
            raise ValueError("dust_reliability must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.__dict__,
            "splits": list(self.splits),
            "cache_contract": PSEUDO_PARTICLE_CACHE_CONTRACT,
        }


def pseudo_particle_cache_paths(
    cache_dir: str | Path,
    variant: str,
    split: str,
) -> tuple[Path, Path]:
    root = Path(cache_dir) / str(variant)
    return root / f"{split}_pseudo_particles", root / f"{split}_pseudo_particles_metadata.json"


def _json_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _torch_load(path: str | Path, device: torch.device) -> Mapping[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _filtered_dataclass_payload(cls: type, payload: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {field.name for field in fields(cls)}
    return {name: value for name, value in payload.items() if name in allowed}


def load_coarse_to_fine_reconstructor_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    strict: bool = True,
    require_model_val_checkpoint: bool = True,
) -> tuple[CTierParticleReconstructor, Mapping[str, Any]]:
    """Load a selected C-tier checkpoint without consulting target caches."""

    resolved_device = torch.device(device)
    payload = _torch_load(checkpoint_path, resolved_device)
    if payload.get("checkpoint_contract") != COARSE_TO_FINE_TRAIN_CONTRACT:
        raise ValueError("reconstructor checkpoint contract mismatch")
    if require_model_val_checkpoint and payload.get("checkpoint_role") != "best_model_val":
        raise ValueError("pseudo-particle rendering requires a best_model_val checkpoint")
    model_payload = payload.get("model")
    if not isinstance(model_payload, Mapping) or model_payload.get("family") != "C":
        raise ValueError("pseudo-particle rendering requires a C-tier particle-slot checkpoint")
    variant = normalize_c_tier_variant(str(model_payload.get("variant")))
    provenance = payload.get("provenance")
    train_provenance = provenance.get("model_train") if isinstance(provenance, Mapping) else None
    layout_payload = train_provenance.get("layout") if isinstance(train_provenance, Mapping) else None
    if not isinstance(layout_payload, Mapping):
        raise ValueError("checkpoint is missing its model_train hierarchy layout")
    from .layout import default_hierarchy_target_layout

    layout = default_hierarchy_target_layout(
        radial_boundary=float(layout_payload["radial_boundary"]),
        coordinate_extent=float(layout_payload["coordinate_extent"]),
    )
    hierarchy_payload = model_payload.get("hierarchy_config")
    slot_payload = model_payload.get("slot_config")
    if not isinstance(hierarchy_payload, Mapping) or not isinstance(slot_payload, Mapping):
        raise ValueError("checkpoint is missing hierarchy or slot model configuration")
    hierarchy_overrides = _filtered_dataclass_payload(CoarseToFineReconstructorConfig, hierarchy_payload)
    hierarchy_overrides.pop("variant", None)
    slot_overrides = _filtered_dataclass_payload(ParticleSlotDecoderConfig, slot_payload)
    slot_overrides.pop("variant", None)
    model = build_c_tier_reconstructor(
        variant,
        hierarchy_overrides=hierarchy_overrides,
        slot_overrides=slot_overrides,
        layout=layout,
    )
    result = model.load_state_dict(payload["model_state_dict"], strict=bool(strict))
    if strict and (result.missing_keys or result.unexpected_keys):
        raise ValueError(
            f"checkpoint state mismatch: missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )
    model.to(resolved_device)
    model.eval()
    return model, payload


def _cell_centers(model: CTierParticleReconstructor, level: int, reference: torch.Tensor) -> torch.Tensor:
    geometry = model.hierarchy.layout.cell_geometry(level)
    bounds = reference.new_tensor(
        [[row["eta_min"], row["eta_max"], row["phi_min"], row["phi_max"]] for row in geometry]
    )
    radial = reference.new_tensor([[row["radial_min"], row["radial_max"]] for row in geometry])
    return model.slot_decoder.coordinate_transform(
        reference.new_zeros(len(geometry), 1, 2),
        bounds[:, None, :],
        radial[:, None, :],
    ).squeeze(-2)


def render_pseudo_particle_batch(
    output: CTierReconstructorOutput,
    *,
    reference_eta: torch.Tensor,
    reference_phi: torch.Tensor,
    model: CTierParticleReconstructor,
    min_particle_pt: float = 0.0,
    dust_reliability: float = 0.10,
) -> PseudoParticleRenderOutput:
    """Convert constrained slots to a JetClass-like HLT-only pseudo view."""

    slots = output.slots
    batch, views, cells, real_slots = slots.total_pt.shape
    reference_eta = reference_eta.to(device=slots.total_pt.device, dtype=slots.total_pt.dtype)
    reference_phi = reference_phi.to(device=slots.total_pt.device, dtype=slots.total_pt.dtype)
    if tuple(reference_eta.shape) != (batch,) or tuple(reference_phi.shape) != (batch,):
        raise ValueError("reference axes must contain one value per jet")
    real_pid = slots.category_pt / slots.total_pt.unsqueeze(-1).clamp_min(1.0e-8)
    real_pid = torch.where(
        (slots.total_pt > 0.0).unsqueeze(-1),
        real_pid,
        slots.pid_probabilities,
    )
    real_charge_probability = torch.softmax(slots.charge_logits, dim=-1)
    real_existence = torch.sigmoid(slots.existence_logits)
    real_uncertainty = (
        slots.log_sigma
        if slots.log_sigma is not None
        else slots.total_pt.new_zeros(batch, views, cells, real_slots, len(PSEUDO_PARTICLE_UNCERTAINTY_NAMES))
    )
    real_uncertainty_mask = torch.full_like(slots.total_pt, slots.log_sigma is not None, dtype=torch.bool)
    if slots.dust_total_pt is not None:
        dust_pt = slots.dust_total_pt.unsqueeze(-1)
        dust_category = slots.dust_category_pt.unsqueeze(-2)
        dust_pid = dust_category / dust_pt.unsqueeze(-1).clamp_min(1.0e-8)
        uniform = torch.full_like(dust_pid, 1.0 / len(PID_CATEGORY_NAMES))
        dust_pid = torch.where((dust_pt > 0.0).unsqueeze(-1), dust_pid, uniform)
        dust_energy = slots.dust_total_energy.unsqueeze(-1)
        dust_count = torch.zeros_like(dust_pt)
        centers = _cell_centers(model, int(slots.terminal_level), slots.total_pt)
        dust_coordinates = centers[None, None, :, None, :].expand(batch, views, -1, -1, -1)
        dust_charge = slots.total_pt.new_zeros(batch, views, cells, 1, 3)
        dust_charge[..., 1] = 1.0
        dust_existence = torch.ones_like(dust_pt)
        dust_reliability_tensor = torch.full_like(dust_pt, float(dust_reliability))
        dust_uncertainty = slots.total_pt.new_zeros(
            batch, views, cells, 1, len(PSEUDO_PARTICLE_UNCERTAINTY_NAMES)
        )
        dust_uncertainty_mask = torch.zeros_like(dust_pt, dtype=torch.bool)
        total_pt = torch.cat((slots.total_pt, dust_pt), dim=-1)
        total_energy = torch.cat((slots.total_energy, dust_energy), dim=-1)
        expected_count = torch.cat((slots.expected_count, dust_count), dim=-1)
        pid = torch.cat((real_pid, dust_pid), dim=-2)
        coordinates = torch.cat((slots.local_coordinates, dust_coordinates), dim=-2)
        charge_probability = torch.cat((real_charge_probability, dust_charge), dim=-2)
        existence = torch.cat((real_existence, dust_existence), dim=-1)
        reliability = torch.cat((slots.reliability, dust_reliability_tensor), dim=-1)
        uncertainty = torch.cat((real_uncertainty, dust_uncertainty), dim=-2)
        uncertainty_mask = torch.cat((real_uncertainty_mask, dust_uncertainty_mask), dim=-1)
        slots_per_cell = real_slots + 1
        slot_indices = torch.cat(
            (
                torch.arange(real_slots, device=slots.total_pt.device, dtype=torch.int16),
                torch.full((1,), -1, device=slots.total_pt.device, dtype=torch.int16),
            )
        )
        is_dust = torch.cat(
            (
                torch.zeros(real_slots, device=slots.total_pt.device, dtype=torch.bool),
                torch.ones(1, device=slots.total_pt.device, dtype=torch.bool),
            )
        )
    else:
        total_pt = slots.total_pt
        total_energy = slots.total_energy
        expected_count = slots.expected_count
        pid = real_pid
        coordinates = slots.local_coordinates
        charge_probability = real_charge_probability
        existence = real_existence
        reliability = slots.reliability
        uncertainty = real_uncertainty
        uncertainty_mask = real_uncertainty_mask
        slots_per_cell = real_slots
        slot_indices = torch.arange(real_slots, device=slots.total_pt.device, dtype=torch.int16)
        is_dust = torch.zeros(real_slots, device=slots.total_pt.device, dtype=torch.bool)
    charge_values = total_pt.new_tensor((-1.0, 0.0, 1.0))
    charge = (charge_probability * charge_values).sum(dim=-1)
    absolute_eta = reference_eta[:, None, None, None] + coordinates[..., 0]
    absolute_phi = torch.remainder(
        reference_phi[:, None, None, None] + coordinates[..., 1] + math.pi,
        2.0 * math.pi,
    ) - math.pi
    tokens = total_pt.new_zeros(batch, views, cells, slots_per_cell, RAW_TOKEN_DIM)
    tokens[..., 0] = total_pt
    tokens[..., 1] = absolute_eta
    tokens[..., 2] = absolute_phi
    tokens[..., 3] = total_energy
    tokens[..., 4] = charge
    tokens[..., 5:10] = pid
    mask = torch.isfinite(tokens[..., :10]).all(dim=-1) & (total_pt > 0.0)
    trusted_by_pt = total_pt >= float(min_particle_pt)
    candidate_weights = existence * reliability * mask.to(dtype=existence.dtype) * trusted_by_pt.to(
        dtype=existence.dtype
    )
    token_cell_indices = torch.arange(cells, device=slots.total_pt.device, dtype=torch.int16).repeat_interleave(
        slots_per_cell
    )
    token_slot_indices = slot_indices.repeat(cells)
    token_is_dust = is_dust.repeat(cells)
    flatten = lambda value: value.reshape(batch, views, cells * slots_per_cell, *value.shape[4:])
    rendered_tokens = flatten(tokens)
    rendered_mask = flatten(mask)
    rendered_weights = flatten(candidate_weights)
    rendered_existence = flatten(existence)
    rendered_reliability = flatten(reliability)
    rendered_count = flatten(expected_count)
    rendered_uncertainty = flatten(uncertainty)
    rendered_uncertainty_mask = flatten(uncertainty_mask)
    diagnostics = {
        "contract": PSEUDO_PARTICLE_RENDER_CONTRACT,
        "variant": slots.variant,
        "terminal_level": int(slots.terminal_level),
        "num_views": int(views),
        "num_tokens": int(rendered_tokens.shape[2]),
        "num_real_slots_per_cell": int(real_slots),
        "dust_enabled": bool(slots.dust_total_pt is not None),
        "all_finite": bool(torch.isfinite(rendered_tokens).all()),
        "active_fraction": rendered_mask.float().mean().detach(),
        "candidate_weight_mean": rendered_weights.mean().detach(),
    }
    return PseudoParticleRenderOutput(
        tokens=rendered_tokens,
        mask=rendered_mask,
        candidate_weights=rendered_weights,
        existence_probability=rendered_existence,
        reliability=rendered_reliability,
        expected_count=rendered_count,
        slot_log_sigma=rendered_uncertainty,
        uncertainty_mask=rendered_uncertainty_mask,
        rendered_accounting=slots.rendered_accounting,
        token_cell_indices=token_cell_indices,
        token_slot_indices=token_slot_indices,
        token_is_dust=token_is_dust,
        view_indices=torch.arange(views, device=slots.total_pt.device, dtype=torch.int16),
        hierarchy=output.hierarchy,
        diagnostics=diagnostics,
    )


def _identity_arrays(
    jet_ids: Sequence[JetIdentity],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    files: list[str] = []
    mapping: dict[str, int] = {}
    file_indices: list[int] = []
    entries: list[int] = []
    for identity in jet_ids:
        name = str(identity.file)
        if name not in mapping:
            mapping[name] = len(files)
            files.append(name)
        file_indices.append(mapping[name])
        entries.append(int(identity.entry))
    return files, np.asarray(file_indices, dtype=np.int32), np.asarray(entries, dtype=np.int64)


def _ids_from_arrays(
    files: Sequence[str],
    file_indices: np.ndarray,
    entries: np.ndarray,
    labels: np.ndarray,
) -> tuple[JetIdentity, ...]:
    return tuple(
        JetIdentity(file=str(files[int(file_index)]), entry=int(entry), label=int(label))
        for file_index, entry, label in zip(file_indices, entries, labels)
    )


def _cache_float_dtype(config: PseudoParticleCacheConfig) -> np.dtype:
    return np.dtype(config.cache_dtype)


def _as_numpy(value: torch.Tensor, dtype: np.dtype | str | None = None) -> np.ndarray:
    array = value.detach().cpu().numpy()
    return array if dtype is None else array.astype(dtype, copy=False)


def _hierarchy_arrays(rendered: PseudoParticleRenderOutput) -> dict[str, np.ndarray]:
    hierarchy = rendered.hierarchy
    batch = int(hierarchy.global_accounting.shape[0])
    field_dim = len(ACCOUNTING_FIELD_NAMES)
    result = {
        "global_accounting": _as_numpy(hierarchy.global_accounting, np.float32),
        "global_log_sigma": _as_numpy(hierarchy.global_log_sigma, np.float32),
    }
    by_level = {int(level.level): level for level in hierarchy.levels}
    for level in (1, 2, 3):
        if level in by_level:
            result[f"level{level}_accounting"] = _as_numpy(by_level[level].accounting, np.float32)
            result[f"level{level}_log_sigma"] = _as_numpy(by_level[level].log_sigma, np.float32)
        else:
            result[f"level{level}_accounting"] = np.zeros((batch, 0, field_dim), dtype=np.float32)
            result[f"level{level}_log_sigma"] = np.zeros((batch, 0, field_dim), dtype=np.float32)
    return result


def _render_arrays(rendered: PseudoParticleRenderOutput, dtype: np.dtype) -> dict[str, np.ndarray]:
    return {
        "tokens": _as_numpy(rendered.tokens, dtype),
        "mask": _as_numpy(rendered.mask, bool),
        "candidate_weights": _as_numpy(rendered.candidate_weights, dtype),
        "existence_probability": _as_numpy(rendered.existence_probability, dtype),
        "reliability": _as_numpy(rendered.reliability, dtype),
        "expected_count": _as_numpy(rendered.expected_count, dtype),
        "slot_log_sigma": _as_numpy(rendered.slot_log_sigma, dtype),
        "uncertainty_mask": _as_numpy(rendered.uncertainty_mask, bool),
        "rendered_accounting": _as_numpy(rendered.rendered_accounting, dtype),
        "token_cell_indices": _as_numpy(rendered.token_cell_indices, np.int16),
        "token_slot_indices": _as_numpy(rendered.token_slot_indices, np.int16),
        "token_is_dust": _as_numpy(rendered.token_is_dust, bool),
        "view_indices": _as_numpy(rendered.view_indices, np.int16),
        **_hierarchy_arrays(rendered),
    }


def _combine_render_groups(groups: Sequence[PseudoParticleRenderOutput], num_views: int) -> PseudoParticleRenderOutput:
    if not groups:
        raise ValueError("at least one render group is required")
    first = groups[0]
    view_fields = (
        "tokens",
        "mask",
        "candidate_weights",
        "existence_probability",
        "reliability",
        "expected_count",
        "slot_log_sigma",
        "uncertainty_mask",
        "rendered_accounting",
    )
    combined = {
        name: torch.cat([getattr(group, name) for group in groups], dim=1)[:, :num_views]
        for name in view_fields
    }
    return PseudoParticleRenderOutput(
        **combined,
        token_cell_indices=first.token_cell_indices,
        token_slot_indices=first.token_slot_indices,
        token_is_dust=first.token_is_dust,
        view_indices=torch.arange(num_views, device=first.tokens.device, dtype=torch.int16),
        hierarchy=first.hierarchy,
        diagnostics={**dict(first.diagnostics), "num_views": int(num_views), "render_groups": len(groups)},
    )


def _generate_rendered(
    model: CTierParticleReconstructor,
    *,
    points: torch.Tensor,
    features: torch.Tensor,
    vectors: torch.Tensor,
    mask: torch.Tensor,
    reference_eta: torch.Tensor,
    reference_phi: torch.Tensor,
    num_views: int,
    generator: torch.Generator,
    min_particle_pt: float,
    dust_reliability: float,
) -> PseudoParticleRenderOutput:
    native_views = int(model.slot_decoder.spec.num_views)
    if native_views == 1 and int(num_views) != 1:
        raise ValueError("multi-view rendering requires a C6 checkpoint trained with stochastic views")
    groups: list[PseudoParticleRenderOutput] = []
    remaining = int(num_views)
    while remaining > 0:
        latent = None
        if native_views > 1:
            latent = torch.randn(
                int(points.shape[0]),
                native_views,
                int(model.slot_decoder.spec.stochastic_latent_dim),
                device=points.device,
                dtype=features.dtype,
                generator=generator,
            )
        output = model(points, features, vectors, mask, stochastic_latent=latent)
        rendered = render_pseudo_particle_batch(
            output,
            reference_eta=reference_eta,
            reference_phi=reference_phi,
            model=model,
            min_particle_pt=min_particle_pt,
            dust_reliability=dust_reliability,
        )
        groups.append(rendered)
        remaining -= native_views
    return _combine_render_groups(groups, int(num_views))


def _checkpoint_manifest_hash(payload: Mapping[str, Any]) -> str | None:
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        return None
    hashes = {
        row.get("source_manifest_hash")
        for row in provenance.values()
        if isinstance(row, Mapping)
    }
    hashes.discard(None)
    if len(hashes) != 1:
        return None
    return str(next(iter(hashes)))


def _validate_hlt_deployable_source(
    *,
    split: str,
    hlt_view: Any,
    manifest: Any,
    manifest_sha: str,
    checkpoint_payload: Mapping[str, Any],
    n_jets: int,
) -> tuple[JetIdentity, ...]:
    expected_ids = tuple(manifest.splits[split][:n_jets])
    problems: list[str] = []
    if tuple(hlt_view.jet_ids[:n_jets]) != expected_ids:
        problems.append("HLT jet identities do not match the manifest prefix")
    if hlt_view.metadata.get("source_manifest_hash") != manifest_sha:
        problems.append("HLT source_manifest_hash mismatch")
    if _checkpoint_manifest_hash(checkpoint_payload) != manifest_sha:
        problems.append("checkpoint manifest hash does not match active manifest")
    if normalize_hlt_profile(hlt_view.metadata.get("hlt_profile")) != HLT_PROFILE_V2_REALISTIC:
        problems.append("HLT profile is not fixed_hlt_v2_realistic")
    if str(hlt_view.metadata.get("hlt_profile_version") or "") != HLT_PROFILE_V2_REALISTIC_VERSION:
        problems.append("HLT profile version mismatch")
    strength = hlt_view.metadata.get("hlt_degradation_strength")
    if strength is None or abs(float(strength) - 2.5) > 1.0e-12:
        problems.append("HLT degradation strength is not 2.5")
    if not hlt_view.metadata.get("hlt_content_hash"):
        problems.append("HLT content hash is missing")
    if problems:
        raise ValueError(f"invalid deployable source for {split}: " + "; ".join(problems))
    return expected_ids


def _aggregate_hash_payload(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "cache_contract": metadata["cache_contract"],
        "split": metadata["split"],
        "variant": metadata["variant"],
        "n_jets": metadata["n_jets"],
        "num_views": metadata["num_views"],
        "num_tokens": metadata["num_tokens"],
        "source_manifest_hash": metadata["source_manifest_hash"],
        "hlt_content_hash": metadata["hlt_content_hash"],
        "hlt_profile": metadata["hlt_profile"],
        "hlt_profile_version": metadata["hlt_profile_version"],
        "hlt_degradation_strength": metadata["hlt_degradation_strength"],
        "jet_identity_hash": metadata["jet_identity_hash"],
        "source_checkpoint_sha256": metadata["source_checkpoint_sha256"],
        "source_checkpoint_role": metadata["source_checkpoint_role"],
        "source_checkpoint_manifest_hash": metadata["source_checkpoint_manifest_hash"],
        "inference_consumes_hlt_only": metadata["inference_consumes_hlt_only"],
        "offline_cache_loaded": metadata["offline_cache_loaded"],
        "offline_targets_loaded": metadata["offline_targets_loaded"],
        "shards": [
            {key: shard[key] for key in ("filename", "shard_index", "start", "stop", "content_hash")}
            for shard in metadata["shards"]
        ],
    }


def _shard_hash(arrays: Mapping[str, np.ndarray]) -> str:
    return hash_arrays({name: np.asarray(arrays[name]) for name in _PSEUDO_SHARD_HASH_KEYS})


def _validate_existing_cache(
    config: PseudoParticleCacheConfig,
    *,
    metadata: Mapping[str, Any],
    checkpoint_payload: Mapping[str, Any],
    split: str,
) -> None:
    """Fail closed instead of silently reusing a cache from another campaign."""

    manifest = load_split_manifest(config.manifest_path)
    expected_manifest_hash = manifest_hash(manifest)
    expected_checkpoint_hash = _file_sha256(config.reconstructor_checkpoint)
    expected_variant = normalize_c_tier_variant(str(checkpoint_payload["model"]["variant"]))
    expected_views = int(
        checkpoint_payload["model"]["slot_config"]["variant_spec"]["slot_spec"]["num_views"]
    )
    if config.num_views is not None:
        expected_views = int(config.num_views)
    hlt_metadata = load_hlt_metadata(config.hlt_cache_dir, split)
    problems: list[str] = []
    required = {
        "cache_contract": PSEUDO_PARTICLE_CACHE_CONTRACT,
        "split": split,
        "variant": expected_variant,
        "source_manifest_hash": expected_manifest_hash,
        "source_checkpoint_sha256": expected_checkpoint_hash,
        "source_checkpoint_role": checkpoint_payload.get("checkpoint_role"),
        "hlt_content_hash": hlt_metadata.get("hlt_content_hash"),
        "num_views": expected_views,
        "cache_dtype": str(np.dtype(config.cache_dtype)),
        "inference_consumes_hlt_only": True,
        "offline_cache_loaded": False,
        "offline_targets_loaded": False,
    }
    for name, expected in required.items():
        if metadata.get(name) != expected:
            problems.append(f"{name}={metadata.get(name)!r}, expected {expected!r}")
    if split == "final_test" and metadata.get("final_test_teacher_free") is not True:
        problems.append("final_test_teacher_free is not true")
    if problems:
        raise ValueError(
            f"refusing to reuse incompatible pseudo-particle cache for {expected_variant}/{split}: "
            + "; ".join(problems)
        )


def cache_pseudo_particle_split(
    config: PseudoParticleCacheConfig,
    *,
    model: CTierParticleReconstructor,
    checkpoint_payload: Mapping[str, Any],
    split: str,
    device: torch.device,
) -> dict[str, Any]:
    """Render one split using only its fixed HLT cache."""

    split = str(split)
    variant = normalize_c_tier_variant(str(checkpoint_payload["model"]["variant"]))
    shard_dir, metadata_path = pseudo_particle_cache_paths(config.output_cache_dir, variant, split)
    if shard_dir.exists() or metadata_path.exists():
        if config.overwrite:
            if shard_dir.exists():
                shutil.rmtree(shard_dir)
            metadata_path.unlink(missing_ok=True)
        elif config.skip_existing and shard_dir.exists() and metadata_path.exists():
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
            _validate_existing_cache(
                config,
                metadata=existing,
                checkpoint_payload=checkpoint_payload,
                split=split,
            )
            _load_metadata(config.output_cache_dir, variant, split)
            return existing
        else:
            raise FileExistsError(f"pseudo-particle cache already exists for {variant}/{split}")
    shard_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = load_split_manifest(config.manifest_path)
    manifest_sha = manifest_hash(manifest)
    hlt_view = load_cached_hlt_view(config.hlt_cache_dir, split, verify_hash=bool(config.verify_hlt_hash))
    manifest_split_size = len(manifest.splits[split])
    if config.max_jets_per_split is None and len(hlt_view.jet_ids) != manifest_split_size:
        raise ValueError(
            f"HLT cache size {len(hlt_view.jet_ids)} does not match manifest size "
            f"{manifest_split_size} for {split}"
        )
    n_jets = len(hlt_view.jet_ids)
    if config.max_jets_per_split is not None:
        n_jets = min(n_jets, int(config.max_jets_per_split))
    if n_jets <= 0:
        raise ValueError(f"HLT cache contains no rows for {split}")
    jet_ids = _validate_hlt_deployable_source(
        split=split,
        hlt_view=hlt_view,
        manifest=manifest,
        manifest_sha=manifest_sha,
        checkpoint_payload=checkpoint_payload,
        n_jets=n_jets,
    )
    labels = np.asarray(hlt_view.labels[:n_jets], dtype=np.int64)
    if not np.array_equal(labels, np.asarray([row.label for row in jet_ids], dtype=np.int64)):
        raise ValueError(f"HLT labels do not match manifest identities for {split}")
    jet_files, all_file_indices, all_entries = _identity_arrays(jet_ids)
    native_views = int(model.slot_decoder.spec.num_views)
    num_views = native_views if config.num_views is None else int(config.num_views)
    if native_views == 1 and num_views != 1:
        raise ValueError("num_views > 1 requires a trained C6 multi-view checkpoint")
    cache_dtype = _cache_float_dtype(config)
    amp_enabled = bool(config.amp and device.type == "cuda")
    generator = torch.Generator(device=device).manual_seed(int(config.seed) + PSEUDO_PARTICLE_ALLOWED_SPLITS.index(split))
    reports: list[dict[str, Any]] = []
    token_layout: dict[str, np.ndarray] | None = None
    candidate_counts: list[np.ndarray] = []
    weight_means: list[np.ndarray] = []
    for shard_index, shard_start in enumerate(range(0, n_jets, int(config.shard_size))):
        shard_stop = min(shard_start + int(config.shard_size), n_jets)
        chunks: dict[str, list[np.ndarray]] = {}
        for batch_start in range(shard_start, shard_stop, int(config.batch_size)):
            batch_stop = min(batch_start + int(config.batch_size), shard_stop)
            raw_tokens = np.asarray(hlt_view.tokens[batch_start:batch_stop], dtype=np.float32)
            raw_mask = np.asarray(hlt_view.mask[batch_start:batch_stop], dtype=bool)
            inputs = build_particle_transformer_inputs_from_tokens(
                raw_tokens,
                raw_mask,
                labels=labels[batch_start:batch_stop],
                split=split,
                source_view="fixed_hlt_v2_realistic",
            )
            points = torch.from_numpy(inputs.pf_points).to(device=device, non_blocking=True)
            features = torch.from_numpy(inputs.pf_features).to(device=device, non_blocking=True)
            vectors = torch.from_numpy(inputs.pf_vectors).to(device=device, non_blocking=True)
            mask = torch.from_numpy(inputs.pf_mask).to(device=device, non_blocking=True)
            reference_eta_np, reference_phi_np, _ = hlt_reference_axis(raw_tokens, raw_mask)
            reference_eta = torch.from_numpy(reference_eta_np).to(device=device)
            reference_phi = torch.from_numpy(reference_phi_np).to(device=device)
            with torch.no_grad(), amp_autocast_context(amp_enabled):
                rendered = _generate_rendered(
                    model,
                    points=points,
                    features=features,
                    vectors=vectors,
                    mask=mask,
                    reference_eta=reference_eta,
                    reference_phi=reference_phi,
                    num_views=num_views,
                    generator=generator,
                    min_particle_pt=float(config.min_particle_pt),
                    dust_reliability=float(config.dust_reliability),
                )
            if not bool(rendered.diagnostics["all_finite"]):
                raise FloatingPointError(f"non-finite pseudo-particles for {split} rows {batch_start}:{batch_stop}")
            arrays = _render_arrays(rendered, cache_dtype)
            arrays["reference_eta"] = reference_eta_np.astype(np.float32, copy=False)
            arrays["reference_phi"] = reference_phi_np.astype(np.float32, copy=False)
            for name, value in arrays.items():
                if name in {"token_cell_indices", "token_slot_indices", "token_is_dust", "view_indices"}:
                    if token_layout is None:
                        token_layout = {
                            key: arrays[key]
                            for key in ("token_cell_indices", "token_slot_indices", "token_is_dust", "view_indices")
                        }
                    elif not np.array_equal(value, token_layout[name]):
                        raise ValueError(f"pseudo token layout changed within {split}: {name}")
                    continue
                chunks.setdefault(name, []).append(value)
        if token_layout is None:
            raise RuntimeError(f"pseudo-particle rendering produced no batches for {split}")
        shard_arrays = {name: np.concatenate(values, axis=0) for name, values in chunks.items()}
        shard_arrays.update(token_layout)
        shard_arrays["labels"] = labels[shard_start:shard_stop]
        shard_arrays["jet_file_indices"] = all_file_indices[shard_start:shard_stop]
        shard_arrays["jet_entries"] = all_entries[shard_start:shard_stop]
        content_hash = _shard_hash(shard_arrays)
        filename = f"shard_{shard_index:05d}.npz"
        np.savez_compressed(shard_dir / filename, **shard_arrays)
        active = shard_arrays["mask"].sum(axis=-1).astype(np.float32)
        candidate_counts.append(active.reshape(-1))
        weight_means.append(shard_arrays["candidate_weights"].astype(np.float32).mean(axis=-1).reshape(-1))
        reports.append(
            {
                "filename": filename,
                "shard_index": shard_index,
                "start": shard_start,
                "stop": shard_stop,
                "n_jets": shard_stop - shard_start,
                "content_hash": content_hash,
            }
        )
    checkpoint_hash = _file_sha256(config.reconstructor_checkpoint)
    num_tokens = int(len(token_layout["token_cell_indices"])) if token_layout is not None else 0
    metadata: dict[str, Any] = {
        "cache_contract": PSEUDO_PARTICLE_CACHE_CONTRACT,
        "cache_set_contract": PSEUDO_PARTICLE_CACHE_SET_CONTRACT,
        "render_contract": PSEUDO_PARTICLE_RENDER_CONTRACT,
        "split": split,
        "variant": variant,
        "n_jets": n_jets,
        "num_views": num_views,
        "native_num_views": native_views,
        "num_tokens": num_tokens,
        "num_cells": int(model.slot_decoder.config.variant_spec.terminal_cell_count),
        "num_real_slots_per_cell": int(model.slot_decoder.spec.num_real_slots),
        "dust_enabled": bool(model.slot_decoder.spec.include_dust),
        "cache_dtype": str(cache_dtype),
        "array_directory": str(shard_dir),
        "metadata_path": str(metadata_path),
        "array_keys": list(_PSEUDO_SHARD_HASH_KEYS),
        "pseudo_token_fields": {
            "0": "pT",
            "1": "eta_absolute_from_hlt_axis_plus_local_offset",
            "2": "phi_absolute_from_hlt_axis_plus_local_offset",
            "3": "energy",
            "4": "expected_charge",
            "5": "charged_hadron_probability",
            "6": "neutral_hadron_probability",
            "7": "photon_probability",
            "8": "electron_probability",
            "9": "muon_probability",
            "10:19": "reserved_zero_raw_features",
        },
        "side_channel_fields": [
            "candidate_weights",
            "existence_probability",
            "reliability",
            "expected_count",
            "slot_log_sigma",
            "uncertainty_mask",
            "token_cell_indices",
            "token_slot_indices",
            "token_is_dust",
            "view_indices",
        ],
        "parent_accounting_fields": list(ACCOUNTING_FIELD_NAMES),
        "uncertainty_names": list(PSEUDO_PARTICLE_UNCERTAINTY_NAMES),
        "source_manifest_hash": manifest_sha,
        "hlt_content_hash": hlt_view.metadata.get("hlt_content_hash"),
        "jet_identity_hash": jet_identity_hash(jet_ids),
        "hlt_profile": normalize_hlt_profile(hlt_view.metadata.get("hlt_profile")),
        "hlt_profile_version": str(hlt_view.metadata.get("hlt_profile_version")),
        "hlt_degradation_strength": float(hlt_view.metadata.get("hlt_degradation_strength")),
        "source_checkpoint": str(Path(config.reconstructor_checkpoint)),
        "source_checkpoint_sha256": checkpoint_hash,
        "source_checkpoint_epoch": checkpoint_payload.get("epoch"),
        "source_checkpoint_role": checkpoint_payload.get("checkpoint_role"),
        "source_checkpoint_model": checkpoint_payload.get("model"),
        "source_checkpoint_manifest_hash": _checkpoint_manifest_hash(checkpoint_payload),
        "jet_files": jet_files,
        "shards": reports,
        "candidate_count_summary": _summary(np.concatenate(candidate_counts)),
        "candidate_weight_mean_summary": _summary(np.concatenate(weight_means)),
        "config": config.to_dict(),
        "inference_consumes_hlt_only": True,
        "offline_cache_loaded": False,
        "offline_targets_loaded": False,
        "final_test_teacher_free": split == "final_test",
        "leakage_rule": (
            "Pseudo-particles, hierarchy accounting, uncertainty, and reliability are produced only from the "
            "fixed HLT cache and a model_val-selected reconstructor checkpoint. Offline caches and hierarchy "
            "target caches are neither configured nor loaded by this module."
        ),
    }
    metadata["cache_content_hash"] = _json_hash(_aggregate_hash_payload(metadata))
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def _summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {"min": 0.0, "mean": 0.0, "max": 0.0, "std": 0.0}
    return {
        "min": float(array.min()),
        "mean": float(array.mean()),
        "max": float(array.max()),
        "std": float(array.std()),
    }


def _load_metadata(cache_dir: str | Path, variant: str, split: str) -> tuple[Path, dict[str, Any]]:
    shard_dir, metadata_path = pseudo_particle_cache_paths(cache_dir, variant, split)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("cache_contract") != PSEUDO_PARTICLE_CACHE_CONTRACT:
        raise ValueError(f"pseudo-particle cache contract mismatch for {variant}/{split}")
    if metadata.get("variant") != variant or metadata.get("split") != split:
        raise ValueError(f"pseudo-particle cache identity mismatch for {variant}/{split}")
    expected_hash = _json_hash(_aggregate_hash_payload(metadata))
    if metadata.get("cache_content_hash") != expected_hash:
        raise ValueError(f"pseudo-particle aggregate hash mismatch for {variant}/{split}")
    return shard_dir, metadata


def load_pseudo_particle_shard(
    cache_dir: str | Path,
    variant: str,
    split: str,
    shard_index: int,
    *,
    verify_hash: bool = True,
) -> PseudoParticleShard:
    variant = normalize_c_tier_variant(variant)
    shard_dir, metadata = _load_metadata(cache_dir, variant, split)
    index = int(shard_index)
    rows = metadata["shards"]
    if not 0 <= index < len(rows):
        raise IndexError(f"pseudo-particle shard index {index} is out of range")
    shard_metadata = rows[index]
    with np.load(shard_dir / shard_metadata["filename"], allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in _PSEUDO_SHARD_HASH_KEYS}
    if verify_hash and _shard_hash(arrays) != shard_metadata.get("content_hash"):
        raise ValueError(f"pseudo-particle shard hash mismatch for {variant}/{split}/{index}")
    labels = np.asarray(arrays["labels"], dtype=np.int64)
    jet_ids = _ids_from_arrays(
        metadata["jet_files"],
        arrays["jet_file_indices"],
        arrays["jet_entries"],
        labels,
    )
    return PseudoParticleShard(
        arrays=arrays,
        labels=labels,
        jet_ids=jet_ids,
        split=split,
        variant=variant,
        shard_index=index,
        start=int(shard_metadata["start"]),
        stop=int(shard_metadata["stop"]),
        metadata={**metadata, "active_shard": shard_metadata},
    )


def iter_pseudo_particle_shards(
    cache_dir: str | Path,
    variant: str,
    split: str,
    *,
    verify_hash: bool = True,
) -> Iterator[PseudoParticleShard]:
    normalized = normalize_c_tier_variant(variant)
    _, metadata = _load_metadata(cache_dir, normalized, split)
    for index in range(len(metadata["shards"])):
        yield load_pseudo_particle_shard(
            cache_dir, normalized, split, index, verify_hash=verify_hash
        )


def load_pseudo_particle_cache(
    cache_dir: str | Path,
    variant: str,
    split: str,
    *,
    verify_hash: bool = True,
    max_concat_jets: int = 250_000,
) -> PseudoParticleCache:
    normalized = normalize_c_tier_variant(variant)
    _, metadata = _load_metadata(cache_dir, normalized, split)
    if int(metadata["n_jets"]) > int(max_concat_jets):
        raise MemoryError("use iter_pseudo_particle_shards for high-data pseudo-particle caches")
    shards = list(iter_pseudo_particle_shards(cache_dir, normalized, split, verify_hash=verify_hash))
    row_keys = [name for name in _PSEUDO_SHARD_HASH_KEYS if name not in {
        "token_cell_indices", "token_slot_indices", "token_is_dust", "view_indices"
    }]
    arrays = {name: np.concatenate([shard.arrays[name] for shard in shards], axis=0) for name in row_keys}
    if shards:
        for name in ("token_cell_indices", "token_slot_indices", "token_is_dust", "view_indices"):
            arrays[name] = np.asarray(shards[0].arrays[name])
    labels = np.asarray(arrays["labels"], dtype=np.int64)
    jet_ids = tuple(identity for shard in shards for identity in shard.jet_ids)
    return PseudoParticleCache(arrays, labels, jet_ids, split, normalized, metadata)


def _closure_diagnostics(shard: PseudoParticleShard, max_rows: int) -> dict[str, float]:
    arrays = shard.arrays
    rows = min(int(max_rows), int(arrays["tokens"].shape[0]))
    if rows <= 0:
        return {}
    tokens = arrays["tokens"][:rows].astype(np.float64)
    mask = arrays["mask"][:rows].astype(bool)
    expected_count = arrays["expected_count"][:rows].astype(np.float64)
    cell_indices = arrays["token_cell_indices"].astype(np.int64)
    terminal_level = max(
        level for level in (1, 2, 3) if arrays[f"level{level}_accounting"].shape[1] > 0
    )
    target = arrays[f"level{terminal_level}_accounting"][:rows].astype(np.float64)
    slot_accounting = arrays["rendered_accounting"][:rows].astype(np.float64)
    batch, views, _, _ = tokens.shape
    cells = int(target.shape[1])
    reconstructed = np.zeros((batch, views, cells, len(ACCOUNTING_FIELD_NAMES)), dtype=np.float64)
    for cell in range(cells):
        selected = cell_indices == cell
        valid = mask[:, :, selected]
        rows_tokens = tokens[:, :, selected]
        pt = np.where(valid, rows_tokens[..., 0], 0.0)
        energy = np.where(valid, rows_tokens[..., 3], 0.0)
        count = np.where(valid, expected_count[:, :, selected], 0.0)
        reconstructed[:, :, cell, ACCOUNTING_INDEX["total_pT"]] = pt.sum(axis=-1)
        reconstructed[:, :, cell, ACCOUNTING_INDEX["total_energy"]] = energy.sum(axis=-1)
        reconstructed[:, :, cell, ACCOUNTING_INDEX["expected_constituent_count"]] = count.sum(axis=-1)
        for category, index in zip(PID_CATEGORY_NAMES, PID_PT_INDICES):
            category_index = PID_CATEGORY_NAMES.index(category)
            reconstructed[:, :, cell, index] = (pt * rows_tokens[..., 5 + category_index]).sum(axis=-1)
        deta = rows_tokens[..., 1] - arrays["reference_eta"][:rows, None, None]
        dphi = wrap_phi(rows_tokens[..., 2] - arrays["reference_phi"][:rows, None, None])
        radius = np.hypot(deta, dphi)
        moments = {
            "sum_pT_abs_deta_pos": pt * np.maximum(deta, 0.0),
            "sum_pT_abs_deta_neg": pt * np.maximum(-deta, 0.0),
            "sum_pT_abs_dphi_pos": pt * np.maximum(dphi, 0.0),
            "sum_pT_abs_dphi_neg": pt * np.maximum(-dphi, 0.0),
            "sum_pT_deta2": pt * deta**2,
            "sum_pT_dphi2": pt * dphi**2,
            "sum_pT_r": pt * radius,
            "sum_pT_r2": pt * radius**2,
        }
        for name in MOMENT_FIELD_NAMES:
            reconstructed[:, :, cell, ACCOUNTING_INDEX[name]] = moments[name].sum(axis=-1)
    target_expanded = target[:, None, :, :]
    primitive_indices = [ACCOUNTING_INDEX["total_pT"], ACCOUNTING_INDEX["total_energy"], ACCOUNTING_INDEX["expected_constituent_count"], *PID_PT_INDICES]
    error = np.abs(reconstructed[..., primitive_indices] - target_expanded[..., primitive_indices])
    scale = np.maximum(np.abs(target_expanded[..., primitive_indices]), 1.0)
    hard_indices = [
        ACCOUNTING_INDEX["total_pT"],
        ACCOUNTING_INDEX["total_energy"],
        ACCOUNTING_INDEX["expected_constituent_count"],
        *PID_PT_INDICES,
        *[ACCOUNTING_INDEX[f"{name}_count"] for name in PID_CATEGORY_NAMES],
    ]
    slot_error = np.abs(slot_accounting[..., hard_indices] - target_expanded[..., hard_indices])
    slot_scale = np.maximum(np.abs(target_expanded[..., hard_indices]), 1.0)
    moment_indices = [ACCOUNTING_INDEX[name] for name in MOMENT_FIELD_NAMES]
    moment_error = np.abs(reconstructed[..., moment_indices] - target_expanded[..., moment_indices])
    moment_scale = np.maximum(np.abs(target_expanded[..., moment_indices]), 1.0)
    return {
        "additive_closure_abs_max": float(error.max(initial=0.0)),
        "additive_closure_relative_max": float((error / scale).max(initial=0.0)),
        "hard_slot_accounting_closure_abs_max": float(slot_error.max(initial=0.0)),
        "hard_slot_accounting_closure_relative_max": float((slot_error / slot_scale).max(initial=0.0)),
        "moment_relative_mae": float(np.mean(moment_error / moment_scale)),
    }


def audit_pseudo_particle_cache(
    cache_dir: str | Path,
    *,
    variant: str,
    split: str,
    manifest_path: str | Path,
    hlt_cache_dir: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    verify_hash: bool = True,
    max_closure_jets: int | None = None,
) -> dict[str, Any]:
    normalized = normalize_c_tier_variant(variant)
    _, metadata = _load_metadata(cache_dir, normalized, split)
    manifest = load_split_manifest(manifest_path)
    manifest_sha = manifest_hash(manifest)
    expected_ids = tuple(manifest.splits[split][: int(metadata["n_jets"])])
    problems: list[str] = []
    if metadata.get("source_manifest_hash") != manifest_sha:
        problems.append("source_manifest_hash mismatch")
    if metadata.get("source_checkpoint_manifest_hash") != manifest_sha:
        problems.append("source checkpoint manifest hash mismatch")
    if metadata.get("jet_identity_hash") != jet_identity_hash(expected_ids):
        problems.append("jet_identity_hash mismatch")
    if metadata.get("source_checkpoint_role") != "best_model_val":
        problems.append("source checkpoint is not model_val-selected")
    if normalize_hlt_profile(metadata.get("hlt_profile")) != HLT_PROFILE_V2_REALISTIC:
        problems.append("HLT profile is not fixed_hlt_v2_realistic")
    if str(metadata.get("hlt_profile_version") or "") != HLT_PROFILE_V2_REALISTIC_VERSION:
        problems.append("HLT profile version mismatch")
    strength = metadata.get("hlt_degradation_strength")
    if strength is None or abs(float(strength) - 2.5) > 1.0e-12:
        problems.append("HLT degradation strength is not 2.5")
    if metadata.get("inference_consumes_hlt_only") is not True:
        problems.append("cache is not marked HLT-only")
    if metadata.get("offline_cache_loaded") is not False or metadata.get("offline_targets_loaded") is not False:
        problems.append("offline-dependency flags are unsafe")
    if split == "final_test" and metadata.get("final_test_teacher_free") is not True:
        problems.append("final_test cache is not marked teacher-free")
    if checkpoint_path is not None and _file_sha256(checkpoint_path) != metadata.get("source_checkpoint_sha256"):
        problems.append("source checkpoint hash mismatch")
    if hlt_cache_dir is not None:
        hlt_view = load_cached_hlt_view(hlt_cache_dir, split, verify_hash=verify_hash)
        if hlt_view.metadata.get("hlt_content_hash") != metadata.get("hlt_content_hash"):
            problems.append("HLT content hash mismatch")
        if tuple(hlt_view.jet_ids[: int(metadata["n_jets"])]) != expected_ids:
            problems.append("HLT row ordering mismatch")
    observed_ids: list[JetIdentity] = []
    expected_start = 0
    closure_rows = None if max_closure_jets is None else int(max_closure_jets)
    audited_closure_rows = 0
    closure_reports: list[Mapping[str, float]] = []
    try:
        for shard in iter_pseudo_particle_shards(
            cache_dir, normalized, split, verify_hash=verify_hash
        ):
            if shard.start != expected_start:
                problems.append(f"shard {shard.shard_index} starts at {shard.start}, expected {expected_start}")
            observed_ids.extend(shard.jet_ids)
            expected_start = shard.stop
            if closure_rows is None or closure_rows > 0:
                take = shard.stop - shard.start if closure_rows is None else min(closure_rows, shard.stop - shard.start)
                closure_reports.append(_closure_diagnostics(shard, take))
                audited_closure_rows += take
                if closure_rows is not None:
                    closure_rows -= take
    except Exception as exc:
        problems.append(str(exc))
    if tuple(observed_ids) != expected_ids:
        problems.append("shard identities do not match manifest ordering")
    max_relative = max(
        (report.get("additive_closure_relative_max", 0.0) for report in closure_reports),
        default=0.0,
    )
    max_slot_relative = max(
        (report.get("hard_slot_accounting_closure_relative_max", 0.0) for report in closure_reports),
        default=0.0,
    )
    tolerance = 0.005 if metadata.get("cache_dtype") == "float16" else 2.0e-4
    if max_relative > tolerance:
        problems.append(f"additive pseudo-particle closure {max_relative} exceeds tolerance {tolerance}")
    if max_slot_relative > tolerance:
        problems.append(f"hard slot-accounting closure {max_slot_relative} exceeds tolerance {tolerance}")
    return {
        "ok": not problems,
        "cache_contract": PSEUDO_PARTICLE_CACHE_CONTRACT,
        "variant": normalized,
        "split": split,
        "n_jets": metadata.get("n_jets"),
        "num_views": metadata.get("num_views"),
        "num_tokens": metadata.get("num_tokens"),
        "cache_content_hash": metadata.get("cache_content_hash"),
        "hard_conserved_accounting_fields": [
            "total_pT",
            "total_energy",
            "expected_constituent_count",
            *[f"{name}_pT" for name in PID_CATEGORY_NAMES],
            *[f"{name}_count" for name in PID_CATEGORY_NAMES],
        ],
        "loss_matched_moment_fields": list(MOMENT_FIELD_NAMES),
        "closure_jets_audited": audited_closure_rows,
        "full_cache_closure_audit": audited_closure_rows == int(metadata.get("n_jets", -1)),
        "closure_diagnostics": closure_reports,
        "problems": problems,
    }


def cache_pseudo_particle_views(config: PseudoParticleCacheConfig) -> dict[str, Any]:
    """Load one selected checkpoint and render all requested HLT-only splits."""

    set_training_seed(int(config.seed))
    device = resolve_device(config.device)
    model, checkpoint_payload = load_coarse_to_fine_reconstructor_checkpoint(
        config.reconstructor_checkpoint,
        device=device,
        strict=bool(config.strict_checkpoint),
        require_model_val_checkpoint=bool(config.require_model_val_checkpoint),
    )
    reports = {
        split: cache_pseudo_particle_split(
            config,
            model=model,
            checkpoint_payload=checkpoint_payload,
            split=split,
            device=device,
        )
        for split in config.splits
    }
    variant = normalize_c_tier_variant(str(checkpoint_payload["model"]["variant"]))
    report = {
        "ok": True,
        "cache_set_contract": PSEUDO_PARTICLE_CACHE_SET_CONTRACT,
        "variant": variant,
        "splits": list(config.splits),
        "output_cache_dir": str(Path(config.output_cache_dir)),
        "source_checkpoint": str(Path(config.reconstructor_checkpoint)),
        "source_checkpoint_sha256": _file_sha256(config.reconstructor_checkpoint),
        "split_reports": reports,
        "inference_consumes_hlt_only": True,
        "offline_cache_loaded": False,
        "offline_targets_loaded": False,
    }
    report_path = Path(config.output_cache_dir) / variant / "cache_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


__all__ = [
    "PSEUDO_PARTICLE_ALLOWED_SPLITS",
    "PSEUDO_PARTICLE_CACHE_CONTRACT",
    "PSEUDO_PARTICLE_CACHE_SET_CONTRACT",
    "PSEUDO_PARTICLE_DEFAULT_SPLITS",
    "PSEUDO_PARTICLE_RENDER_CONTRACT",
    "PSEUDO_PARTICLE_UNCERTAINTY_NAMES",
    "PseudoParticleCache",
    "PseudoParticleCacheConfig",
    "PseudoParticleRenderOutput",
    "PseudoParticleShard",
    "audit_pseudo_particle_cache",
    "cache_pseudo_particle_split",
    "cache_pseudo_particle_views",
    "iter_pseudo_particle_shards",
    "load_coarse_to_fine_reconstructor_checkpoint",
    "load_pseudo_particle_cache",
    "load_pseudo_particle_shard",
    "pseudo_particle_cache_paths",
    "render_pseudo_particle_batch",
]
