"""Step 5 training for constrained coarse-to-fine reconstructors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from jetclass_fixed_hlt import HLT_PROFILE_V2_REALISTIC, HLT_PROFILE_V2_REALISTIC_VERSION
from jetclass_fresh.hlt_baseline import amp_autocast_context, amp_grad_scaler, resolve_device, set_training_seed
from jetclass_fresh.hlt_cache import jet_identity_hash, load_cached_hlt_view, normalize_hlt_profile
from jetclass_fresh.jetclass_data import load_split_manifest, manifest_hash
from jetclass_fresh.part_inputs import build_particle_transformer_inputs_from_tokens
from teacher_logit_reco.architecture_view_part import load_cached_offline_view

from .cache import (
    HIERARCHY_TARGET_CACHE_CONTRACT,
    HIERARCHY_TARGET_EXPECTED_HLT_STRENGTH,
    hierarchy_target_cache_paths,
    load_hierarchy_target_shard,
)
from .layout import HierarchyTargetLayout, default_hierarchy_target_layout
from .losses import (
    HierarchyReconstructionLossConfig,
    compute_hierarchy_reconstruction_loss,
)
from .model import (
    CoarseToFineReconstructorConfig,
    CoarseToFineReconstructorOutput,
    build_coarse_to_fine_reconstructor,
    normalize_b_tier_variant,
)
from .slot_loss import ParticleSlotLossConfig, compute_particle_slot_loss, prepare_cell_slot_targets
from .slots import (
    CTierReconstructorOutput,
    build_c_tier_reconstructor,
    c_tier_variant_spec,
    normalize_c_tier_variant,
)


COARSE_TO_FINE_TRAIN_CONTRACT = "constrained_coarse_to_fine_reconstructor_training_v1"
COARSE_TO_FINE_ALLOWED_TRAIN_SPLITS = ("model_train", "model_val", "stack_val")


@dataclass(frozen=True)
class CoarseToFineTrainConfig:
    output_dir: str
    manifest_path: str
    hlt_cache_dir: str
    offline_cache_dir: str
    target_cache_dir: str
    variant: str = "C5"
    train_split: str = "model_train"
    val_split: str = "model_val"
    stack_val_split: str | None = None
    seed: int = 22031
    epochs: int = 30
    batch_size: int = 64
    eval_batch_size: int = 128
    num_workers: int = 0
    learning_rate: float = 2.0e-4
    hlt_encoder_lr_scale: float = 0.05
    weight_decay: float = 1.0e-4
    grad_clip_norm: float = 1.0
    early_stop_patience: int = 6
    max_nonfinite_batches: int = 8
    device: str = "auto"
    amp: bool = True
    verify_hash: bool = True
    pin_memory: bool = True
    max_train_jets: int | None = None
    max_val_jets: int | None = None
    max_stack_val_jets: int | None = None
    save_last_checkpoint: bool = True
    d_model: int = 256
    num_heads: int = 8
    encoder_layers: int = 8
    pool_layers: int = 2
    decoder_layers_per_level: int = 3
    ffn_multiplier: float = 4.0
    pair_hidden_dim: int = 64
    dropout: float = 0.05
    attention_dropout: float = 0.05
    slot_loss_weight: float = 1.0
    hierarchy_global_weight: float = 1.0
    hierarchy_grid_weight: float = 1.0
    hierarchy_relative_weight: float = 0.25
    hierarchy_auxiliary_weight: float = 0.25
    hierarchy_allocation_kl_weight: float = 0.10
    hierarchy_uncertainty_weight: float = 0.05

    def __post_init__(self) -> None:
        if str(self.train_split) != "model_train" or str(self.val_split) != "model_val":
            raise ValueError("Step 5 selection requires model_train and model_val")
        for split in (self.train_split, self.val_split, self.stack_val_split):
            if split is None:
                continue
            if str(split) == "final_test":
                raise ValueError("offline final_test reconstruction targets are forbidden")
            if str(split) not in COARSE_TO_FINE_ALLOWED_TRAIN_SPLITS:
                raise ValueError(f"unsupported Step 5 split {split!r}")
        for name in (
            "epochs",
            "batch_size",
            "eval_batch_size",
            "d_model",
            "num_heads",
            "encoder_layers",
            "pool_layers",
            "decoder_layers_per_level",
            "pair_hidden_dim",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if int(self.num_workers) < 0 or int(self.max_nonfinite_batches) < 0:
            raise ValueError("num_workers and max_nonfinite_batches must be nonnegative")
        for name in (
            "learning_rate",
            "hlt_encoder_lr_scale",
            "grad_clip_norm",
            "ffn_multiplier",
            "slot_loss_weight",
            "hierarchy_global_weight",
            "hierarchy_grid_weight",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if float(self.weight_decay) < 0.0:
            raise ValueError("weight_decay must be nonnegative")
        _normalize_variant(self.variant)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["contract"] = COARSE_TO_FINE_TRAIN_CONTRACT
        payload["resolved_variant"] = _normalize_variant(self.variant)[1]
        return payload


@dataclass(frozen=True)
class _SplitSource:
    split: str
    hlt_view: Any
    offline_view: Any | None
    target_metadata: Mapping[str, Any]
    layout: HierarchyTargetLayout
    provenance: Mapping[str, Any]


class _ShardDataset(Dataset):
    def __init__(
        self,
        *,
        hlt_tokens: np.ndarray,
        hlt_mask: np.ndarray,
        labels: np.ndarray,
        target_arrays: Mapping[str, np.ndarray],
        offline_tokens: np.ndarray | None,
        offline_mask: np.ndarray | None,
        split: str,
    ) -> None:
        inputs = build_particle_transformer_inputs_from_tokens(
            np.asarray(hlt_tokens, dtype=np.float32),
            np.asarray(hlt_mask, dtype=bool),
            labels=np.asarray(labels, dtype=np.int64),
            split=str(split),
            source_view="fixed_hlt_v2_realistic",
        )
        self.points = torch.from_numpy(np.asarray(inputs.pf_points, dtype=np.float32))
        self.features = torch.from_numpy(np.asarray(inputs.pf_features, dtype=np.float32))
        self.vectors = torch.from_numpy(np.asarray(inputs.pf_vectors, dtype=np.float32))
        self.mask = torch.from_numpy(np.asarray(inputs.pf_mask, dtype=bool))
        self.labels = torch.from_numpy(np.asarray(labels, dtype=np.int64))
        self.targets = {
            name: torch.from_numpy(np.asarray(value, dtype=np.float32 if name != "final_cell_indices" else np.int64))
            for name, value in target_arrays.items()
        }
        self.offline_tokens = (
            None if offline_tokens is None else torch.from_numpy(np.asarray(offline_tokens, dtype=np.float32))
        )
        self.offline_mask = None if offline_mask is None else torch.from_numpy(np.asarray(offline_mask, dtype=bool))

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = {
            "points": self.points[index],
            "features": self.features[index],
            "vectors": self.vectors[index],
            "mask": self.mask[index],
            "labels": self.labels[index],
            **{name: value[index] for name, value in self.targets.items()},
        }
        if self.offline_tokens is not None and self.offline_mask is not None:
            row["offline_tokens"] = self.offline_tokens[index]
            row["offline_mask"] = self.offline_mask[index]
        return row


class _MetricAccumulator:
    def __init__(self) -> None:
        self.weighted: dict[str, torch.Tensor | float] = {}
        self.weights: dict[str, float] = {}

    def add(self, values: Mapping[str, Any], weight: int) -> None:
        for name, value in values.items():
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    continue
                scalar: torch.Tensor | float = value.detach().float() * int(weight)
            elif isinstance(value, (int, float, np.integer, np.floating)):
                raw = float(value)
                if not math.isfinite(raw):
                    continue
                scalar = raw * int(weight)
            else:
                continue
            if name in self.weighted:
                self.weighted[name] = self.weighted[name] + scalar
            else:
                self.weighted[name] = scalar
            self.weights[name] = self.weights.get(name, 0.0) + int(weight)

    def means(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for name in sorted(self.weighted):
            mean = self.weighted[name] / max(self.weights.get(name, 0.0), 1.0)
            scalar = float(mean.detach().cpu().item()) if isinstance(mean, torch.Tensor) else float(mean)
            if math.isfinite(scalar):
                result[name] = scalar
        return result


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist() if value.numel() != 1 else value.detach().cpu().item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_variant(value: str) -> tuple[str, str]:
    try:
        return "B", normalize_b_tier_variant(value)
    except ValueError:
        return "C", normalize_c_tier_variant(value)


def _read_target_metadata(cache_dir: str, split: str) -> dict[str, Any]:
    _, metadata_path = hierarchy_target_cache_paths(cache_dir, split)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("cache_contract") != HIERARCHY_TARGET_CACHE_CONTRACT:
        raise ValueError(f"target cache contract mismatch for {split}")
    return metadata


def _load_split_source(
    config: CoarseToFineTrainConfig,
    split: str,
    *,
    require_offline_particles: bool,
) -> _SplitSource:
    if str(split) == "final_test":
        raise ValueError("final_test reconstruction supervision is forbidden")
    manifest = load_split_manifest(config.manifest_path)
    manifest_sha = manifest_hash(manifest)
    expected_ids = tuple(manifest.splits[str(split)])
    hlt_view = load_cached_hlt_view(config.hlt_cache_dir, split, verify_hash=bool(config.verify_hash))
    offline_view = (
        load_cached_offline_view(config.offline_cache_dir, split, verify_hash=bool(config.verify_hash))
        if require_offline_particles
        else None
    )
    target_metadata = _read_target_metadata(config.target_cache_dir, split)
    problems: list[str] = []
    if tuple(hlt_view.jet_ids) != expected_ids:
        problems.append("HLT identities do not match manifest ordering")
    if offline_view is not None:
        if tuple(offline_view.jet_ids) != expected_ids:
            problems.append("offline identities do not match manifest ordering")
        if not np.array_equal(hlt_view.labels, offline_view.labels):
            problems.append("HLT/offline labels differ")
    if int(target_metadata.get("n_jets", -1)) != len(expected_ids):
        problems.append("target n_jets does not match manifest")
    if target_metadata.get("jet_identity_hash") != jet_identity_hash(expected_ids):
        problems.append("target jet_identity_hash does not match manifest")
    metadata_sources = [("HLT", hlt_view.metadata), ("target", target_metadata)]
    if offline_view is not None:
        metadata_sources.append(("offline", offline_view.metadata))
    for name, metadata in metadata_sources:
        if metadata.get("source_manifest_hash") != manifest_sha:
            problems.append(f"{name} source_manifest_hash mismatch")
    hlt_hash = hlt_view.metadata.get("hlt_content_hash")
    offline_hash = target_metadata.get("offline_content_hash")
    if offline_view is not None:
        offline_hash = offline_view.metadata.get("offline_content_hash") or offline_view.metadata.get("content_hash")
    if not hlt_hash or target_metadata.get("hlt_content_hash") != hlt_hash:
        problems.append("target HLT content hash does not match loaded HLT cache")
    if not offline_hash or target_metadata.get("offline_content_hash") != offline_hash:
        problems.append("target offline content hash is missing or does not match loaded offline cache")
    profile = normalize_hlt_profile(hlt_view.metadata.get("hlt_profile"))
    if profile != HLT_PROFILE_V2_REALISTIC or target_metadata.get("hlt_profile") != HLT_PROFILE_V2_REALISTIC:
        problems.append("HLT profile is not fixed_hlt_v2_realistic")
    version = str(hlt_view.metadata.get("hlt_profile_version") or "")
    if version != HLT_PROFILE_V2_REALISTIC_VERSION or target_metadata.get("hlt_profile_version") != version:
        problems.append("HLT profile version mismatch")
    strength = hlt_view.metadata.get("hlt_degradation_strength")
    if strength is None or abs(float(strength) - HIERARCHY_TARGET_EXPECTED_HLT_STRENGTH) > 1.0e-12:
        problems.append("HLT degradation strength is not 2.5")
    if abs(float(target_metadata.get("hlt_degradation_strength", -1.0)) - HIERARCHY_TARGET_EXPECTED_HLT_STRENGTH) > 1.0e-12:
        problems.append("target HLT degradation strength is not 2.5")
    if problems:
        raise ValueError(f"invalid aligned Step 5 sources for {split}: " + "; ".join(problems))
    layout_payload = target_metadata["layout"]
    layout = default_hierarchy_target_layout(
        radial_boundary=float(layout_payload["radial_boundary"]),
        coordinate_extent=float(layout_payload["coordinate_extent"]),
    )
    provenance = {
        "split": str(split),
        "n_jets": len(expected_ids),
        "target_cache_contract": target_metadata.get("cache_contract"),
        "target_builder_version": target_metadata.get("builder_version"),
        "source_manifest_hash": manifest_sha,
        "hlt_content_hash": hlt_hash,
        "offline_content_hash": offline_hash,
        "target_content_hash": target_metadata.get("target_content_hash"),
        "jet_identity_hash": target_metadata.get("jet_identity_hash"),
        "hlt_profile": profile,
        "hlt_profile_version": version,
        "hlt_degradation_strength": float(strength),
        "layout": layout.to_dict(),
    }
    return _SplitSource(str(split), hlt_view, offline_view, target_metadata, layout, provenance)


def _target_arrays(shard: Any, stop: int) -> dict[str, np.ndarray]:
    output = shard.output
    return {
        "global_accounting": np.asarray(output.global_accounting[:stop]),
        "level1_accounting": np.asarray(output.level1_accounting[:stop]),
        "level2_accounting": np.asarray(output.level2_accounting[:stop]),
        "level3_accounting": np.asarray(output.level3_accounting[:stop]),
        "final_cell_indices": np.asarray(output.final_cell_indices[:stop], dtype=np.int64),
        "reference_eta": np.asarray(output.reference_eta[:stop], dtype=np.float32),
        "reference_phi": np.asarray(output.reference_phi[:stop], dtype=np.float32),
    }


def _iter_split_loaders(
    source: _SplitSource,
    config: CoarseToFineTrainConfig,
    *,
    train: bool,
    require_offline_particles: bool,
    max_jets: int | None,
    epoch: int,
) -> Iterator[DataLoader]:
    shard_indices = list(range(int(source.target_metadata["n_shards"])))
    # Keep the shard prefix fixed when max_jets is used; rows are shuffled
    # within each shard so pilot selection remains deterministic across epochs.
    remaining = None if max_jets is None else int(max_jets)
    for shard_index in shard_indices:
        if remaining is not None and remaining <= 0:
            break
        shard = load_hierarchy_target_shard(
            config.target_cache_dir,
            source.split,
            shard_index,
            verify_hash=bool(config.verify_hash),
        )
        count = int(shard.stop - shard.start)
        take = count if remaining is None else min(count, remaining)
        expected_ids = tuple(source.hlt_view.jet_ids[shard.start : shard.start + take])
        if tuple(shard.jet_ids[:take]) != expected_ids:
            raise ValueError(f"target shard identities do not match HLT rows for {source.split}/{shard_index}")
        if not np.array_equal(shard.labels[:take], source.hlt_view.labels[shard.start : shard.start + take]):
            raise ValueError(f"target shard labels do not match HLT rows for {source.split}/{shard_index}")
        start = int(shard.start)
        stop = start + take
        dataset = _ShardDataset(
            hlt_tokens=source.hlt_view.tokens[start:stop],
            hlt_mask=source.hlt_view.mask[start:stop],
            labels=source.hlt_view.labels[start:stop],
            target_arrays=_target_arrays(shard, take),
            offline_tokens=(
                source.offline_view.tokens[start:stop]
                if require_offline_particles and source.offline_view is not None
                else None
            ),
            offline_mask=(
                source.offline_view.mask[start:stop]
                if require_offline_particles and source.offline_view is not None
                else None
            ),
            split=source.split,
        )
        generator = torch.Generator().manual_seed(int(config.seed) + int(epoch) * 1009 + shard_index)
        yield DataLoader(
            dataset,
            batch_size=int(config.batch_size if train else config.eval_batch_size),
            shuffle=bool(train),
            num_workers=int(config.num_workers),
            pin_memory=bool(config.pin_memory),
            generator=generator,
            drop_last=False,
        )
        if remaining is not None:
            remaining -= take


def _model_config_payload(config: CoarseToFineTrainConfig, hierarchy_variant: str) -> dict[str, Any]:
    return {
        "variant": hierarchy_variant,
        "d_model": int(config.d_model),
        "num_heads": int(config.num_heads),
        "encoder_layers": int(config.encoder_layers),
        "pool_layers": int(config.pool_layers),
        "decoder_layers_per_level": int(config.decoder_layers_per_level),
        "ffn_multiplier": float(config.ffn_multiplier),
        "pair_hidden_dim": int(config.pair_hidden_dim),
        "dropout": float(config.dropout),
        "attention_dropout": float(config.attention_dropout),
    }


def _build_model(config: CoarseToFineTrainConfig, layout: HierarchyTargetLayout) -> tuple[torch.nn.Module, str, str]:
    family, variant = _normalize_variant(config.variant)
    if family == "B":
        model = build_coarse_to_fine_reconstructor(
            CoarseToFineReconstructorConfig(**_model_config_payload(config, variant)),
            layout=layout,
        )
    else:
        spec = c_tier_variant_spec(variant)
        hierarchy = _model_config_payload(config, spec.hierarchy_variant)
        model = build_c_tier_reconstructor(
            variant,
            hierarchy_overrides=hierarchy,
            slot_overrides={
                "d_model": int(config.d_model),
                "num_heads": int(config.num_heads),
                "ffn_multiplier": float(config.ffn_multiplier),
                "dropout": float(config.dropout),
                "attention_dropout": float(config.attention_dropout),
            },
            layout=layout,
        )
    return model, family, variant


def _optimizer(model: torch.nn.Module, config: CoarseToFineTrainConfig) -> torch.optim.Optimizer:
    encoder_parameters: list[torch.nn.Parameter] = []
    other_parameters: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if ".hlt_encoder." in f".{name}." or name.startswith("hlt_encoder."):
            encoder_parameters.append(parameter)
        else:
            other_parameters.append(parameter)
    groups: list[dict[str, Any]] = []
    if encoder_parameters:
        groups.append(
            {
                "params": encoder_parameters,
                "lr": float(config.learning_rate) * float(config.hlt_encoder_lr_scale),
                "group_name": "hlt_encoder",
            }
        )
    if other_parameters:
        groups.append({"params": other_parameters, "lr": float(config.learning_rate), "group_name": "reconstructor"})
    if not groups:
        raise ValueError("model has no trainable parameters")
    return torch.optim.AdamW(groups, weight_decay=float(config.weight_decay))


def _move_batch(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        name: value.to(device=device, non_blocking=True)
        for name, value in batch.items()
    }


def _all_finite(value: Any) -> bool:
    tensors: list[torch.Tensor] = []

    def collect(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            if item.is_floating_point() or item.is_complex():
                tensors.append(item)
        elif hasattr(item, "__dataclass_fields__"):
            for name in item.__dataclass_fields__:
                collect(getattr(item, name))
        elif isinstance(item, Mapping):
            for nested in item.values():
                collect(nested)
        elif isinstance(item, (tuple, list)):
            for nested in item:
                collect(nested)

    collect(value)
    if not tensors:
        return True
    finite = torch.isfinite(tensors[0]).all()
    for tensor in tensors[1:]:
        finite = finite & torch.isfinite(tensor).all()
    return bool(finite)


def _grad_norm_if_finite(model: torch.nn.Module) -> torch.Tensor | None:
    total: torch.Tensor | None = None
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        contribution = parameter.grad.detach().float().square().sum()
        total = contribution if total is None else total + contribution
    if total is None or not bool(torch.isfinite(total)):
        return None
    return torch.sqrt(total)


def _loss_configs(config: CoarseToFineTrainConfig, family: str, variant: str):
    hierarchy_config = HierarchyReconstructionLossConfig(
        global_weight=float(config.hierarchy_global_weight),
        grid_weight=float(config.hierarchy_grid_weight),
        relative_weight=float(config.hierarchy_relative_weight),
        auxiliary_weight=float(config.hierarchy_auxiliary_weight),
        allocation_kl_weight=float(config.hierarchy_allocation_kl_weight),
        uncertainty_weight=float(config.hierarchy_uncertainty_weight),
    )
    slot_config = None
    if family == "C":
        slot_config = ParticleSlotLossConfig(matching_mode=c_tier_variant_spec(variant).slot_spec.matching_mode)
    return hierarchy_config, slot_config


def _batch_losses(
    model_output: Any,
    batch: Mapping[str, torch.Tensor],
    hierarchy_config: HierarchyReconstructionLossConfig,
    slot_config: ParticleSlotLossConfig | None,
    slot_loss_weight: float,
    coordinate_extent: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if isinstance(model_output, CTierReconstructorOutput):
        hierarchy = model_output.hierarchy
        slots = model_output.slots
    else:
        hierarchy = model_output
        slots = None
    target_mapping = {
        "global_accounting": batch["global_accounting"],
        "level1_accounting": batch["level1_accounting"],
        "level2_accounting": batch["level2_accounting"],
        "level3_accounting": batch["level3_accounting"],
    }
    hierarchy_loss = compute_hierarchy_reconstruction_loss(hierarchy, target_mapping, hierarchy_config)
    total = hierarchy_loss.loss
    hierarchy_selection = (
        float(hierarchy_config.global_weight)
        * (
            hierarchy_loss.components["global_accounting"]
            + float(hierarchy_config.relative_weight) * hierarchy_loss.components["global_relative"]
            + float(hierarchy_config.auxiliary_weight) * hierarchy_loss.components["global_auxiliary"]
        )
    )
    grid_selection_rows: list[torch.Tensor] = []
    for level in hierarchy.levels:
        prefix = f"level{int(level.level)}"
        grid_selection_rows.append(
            hierarchy_loss.components[f"{prefix}_accounting"]
            + float(hierarchy_config.relative_weight) * hierarchy_loss.components[f"{prefix}_relative"]
            + float(hierarchy_config.allocation_kl_weight)
            * hierarchy_loss.components[f"{prefix}_allocation_kl"]
        )
    if grid_selection_rows:
        hierarchy_selection = hierarchy_selection + float(hierarchy_config.grid_weight) * torch.stack(
            grid_selection_rows
        ).mean()
    selection_score = hierarchy_selection
    metrics: dict[str, Any] = {
        "loss.total": total,
        "loss.hierarchy": hierarchy_loss.loss,
        "selection.reconstruction_score": selection_score,
        **{f"hierarchy.component.{name}": value for name, value in hierarchy_loss.components.items()},
        **{f"hierarchy.metric.{name}": value for name, value in hierarchy_loss.metrics.items()},
    }
    if slots is not None:
        if slot_config is None or "offline_tokens" not in batch:
            raise ValueError("C-tier training requires slot configuration and offline particle targets")
        slot_targets = prepare_cell_slot_targets(
            batch["offline_tokens"],
            batch["offline_mask"],
            batch["final_cell_indices"],
            batch["reference_eta"],
            batch["reference_phi"],
            terminal_level=int(slots.terminal_level),
            coordinate_extent=float(coordinate_extent),
        )
        slot_loss = compute_particle_slot_loss(slots, slot_targets, slot_config)
        total = total + float(slot_loss_weight) * slot_loss.loss
        slot_selection = slot_loss.loss - float(slot_config.uncertainty_weight) * slot_loss.components[
            "uncertainty_nll"
        ]
        selection_score = selection_score + float(slot_loss_weight) * slot_selection
        existence = torch.sigmoid(slots.existence_logits).clamp(1.0e-8, 1.0 - 1.0e-8)
        entropy = -(existence * torch.log(existence) + (1.0 - existence) * torch.log(1.0 - existence)).mean()
        matched_pt: list[torch.Tensor] = []
        matched_eta: list[torch.Tensor] = []
        matched_phi: list[torch.Tensor] = []
        matched_pid: list[torch.Tensor] = []
        for assignment in slot_loss.assignments:
            if int(assignment.pred_indices.numel()) == 0:
                continue
            batch_index = int(assignment.batch_index)
            view_index = int(assignment.view_index)
            cell_index = int(assignment.cell_index)
            prediction_indices = assignment.pred_indices.to(device=slots.total_pt.device)
            target_indices = assignment.target_indices.to(device=slots.total_pt.device)
            predicted_pt = slots.total_pt[batch_index, view_index, cell_index].index_select(
                0, prediction_indices
            )
            target_pt = slot_targets.pt[batch_index, cell_index].index_select(0, target_indices)
            predicted_coordinates = slots.local_coordinates[
                batch_index, view_index, cell_index
            ].index_select(0, prediction_indices)
            target_coordinates = slot_targets.local_coordinates[batch_index, cell_index].index_select(
                0, target_indices
            )
            predicted_pid = slots.pid_probabilities[
                batch_index, view_index, cell_index
            ].index_select(0, prediction_indices).argmax(dim=-1)
            target_pid = slot_targets.pid_index[batch_index, cell_index].index_select(0, target_indices)
            matched_pt.append((predicted_pt - target_pt).abs().mean())
            matched_eta.append((predicted_coordinates[:, 0] - target_coordinates[:, 0]).abs().mean())
            dphi = torch.remainder(
                predicted_coordinates[:, 1] - target_coordinates[:, 1] + math.pi,
                2.0 * math.pi,
            ) - math.pi
            matched_phi.append(dphi.abs().mean())
            matched_pid.append((predicted_pid == target_pid).float().mean())

        def assignment_mean(rows: Sequence[torch.Tensor]) -> torch.Tensor:
            return torch.stack(tuple(rows)).mean() if rows else total.new_zeros(())

        metrics.update(
            {
                "loss.total": total,
                "loss.slot": slot_loss.loss,
                "selection.reconstruction_score": selection_score,
                **{f"slot.component.{name}": value for name, value in slot_loss.components.items()},
                **{f"slot.metric.{name}": value for name, value in slot_loss.metrics.items()},
                "slot.metric.slot_usage_entropy": entropy,
                "slot.metric.matched_pT_mae": assignment_mean(matched_pt),
                "slot.metric.matched_eta_mae": assignment_mean(matched_eta),
                "slot.metric.matched_phi_mae": assignment_mean(matched_phi),
                "slot.metric.matched_pid_accuracy": assignment_mean(matched_pid),
            }
        )
    return total, metrics


def _run_epoch(
    model: torch.nn.Module,
    source: _SplitSource,
    config: CoarseToFineTrainConfig,
    *,
    family: str,
    variant: str,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: Any | None,
    amp_enabled: bool,
    max_jets: int | None,
    epoch: int,
) -> dict[str, Any]:
    train = optimizer is not None
    model.train(train)
    hierarchy_config, slot_config = _loss_configs(config, family, variant)
    accumulator = _MetricAccumulator()
    n_jets = 0
    n_batches = 0
    nonfinite_batches = 0
    for loader in _iter_split_loaders(
        source,
        config,
        train=train,
        require_offline_particles=family == "C",
        max_jets=max_jets,
        epoch=epoch,
    ):
        for cpu_batch in loader:
            batch = _move_batch(cpu_batch, device)
            batch_size = int(batch["labels"].shape[0])
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            gradients_unscaled = False
            try:
                with amp_autocast_context(bool(amp_enabled)):
                    output = model(batch["points"], batch["features"], batch["vectors"], batch["mask"])
                    if not _all_finite(output):
                        raise FloatingPointError("model output is non-finite")
                    loss, batch_metrics = _batch_losses(
                        output,
                        batch,
                        hierarchy_config,
                        slot_config,
                        float(config.slot_loss_weight),
                        float(source.layout.coordinate_extent),
                    )
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("total loss is non-finite")
                if optimizer is not None:
                    assert scaler is not None
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    gradients_unscaled = True
                    grad_norm = _grad_norm_if_finite(model)
                    if grad_norm is None:
                        raise FloatingPointError("gradients are absent or non-finite")
                    if float(config.grad_clip_norm) > 0.0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.grad_clip_norm))
                    scaler.step(optimizer)
                    scaler.update()
                    batch_metrics["train.grad_norm_before_clip"] = grad_norm
            except FloatingPointError:
                nonfinite_batches += 1
                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                    if scaler is not None and gradients_unscaled:
                        scaler.update()
                if nonfinite_batches > int(config.max_nonfinite_batches):
                    raise RuntimeError(
                        f"{source.split} exceeded max_nonfinite_batches={config.max_nonfinite_batches}"
                    )
                continue
            accumulator.add(batch_metrics, batch_size)
            n_jets += batch_size
            n_batches += 1
    metrics: dict[str, Any] = accumulator.means()
    metrics.update(
        {
            "n_jets": int(n_jets),
            "n_batches": int(n_batches),
            "nonfinite_batches_skipped": int(nonfinite_batches),
            "split": source.split,
        }
    )
    if n_jets <= 0:
        raise RuntimeError(f"no finite batches were processed for {source.split}")
    return metrics


def _flatten(prefix: str, value: Any, output: dict[str, Any]) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), item, output)
    elif isinstance(value, (str, int, float, bool)) or value is None:
        output[prefix] = value


def _write_curves_csv(path: Path, curves: Sequence[Mapping[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    keys: set[str] = set()
    for curve in curves:
        row: dict[str, Any] = {}
        _flatten("", curve, row)
        rows.append(row)
        keys.update(row)
    columns = ["epoch"] + sorted(key for key in keys if key != "epoch")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _model_description(model: torch.nn.Module, family: str, variant: str) -> dict[str, Any]:
    if family == "B":
        return {
            "family": family,
            "variant": variant,
            "hierarchy_config": model.config.to_dict(),
        }
    return {
        "family": family,
        "variant": variant,
        "hierarchy_config": model.hierarchy.config.to_dict(),
        "slot_config": model.slot_decoder.config.to_dict(),
    }


def _checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: CoarseToFineTrainConfig,
    family: str,
    variant: str,
    metrics: Mapping[str, Any],
    provenance: Mapping[str, Any],
    checkpoint_role: str,
) -> dict[str, Any]:
    hierarchy_loss_config, slot_loss_config = _loss_configs(config, family, variant)
    return {
        "checkpoint_contract": COARSE_TO_FINE_TRAIN_CONTRACT,
        "checkpoint_role": str(checkpoint_role),
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config.to_dict(),
        "model": _model_description(model, family, variant),
        "hierarchy_loss_config": hierarchy_loss_config.to_dict(),
        "slot_loss_config": None if slot_loss_config is None else slot_loss_config.to_dict(),
        "metrics": _jsonable(metrics),
        "provenance": _jsonable(provenance),
    }


def _load_checkpoint(path: Path, device: torch.device) -> Mapping[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def train_coarse_to_fine_reconstructor(config: CoarseToFineTrainConfig) -> dict[str, Any]:
    """Train one B- or C-tier reconstructor and select only on model_val."""

    set_training_seed(int(config.seed))
    output_dir = Path(config.output_dir)
    diagnostics_dir = output_dir / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config.device)
    amp_enabled = bool(config.amp and getattr(device, "type", str(device)) == "cuda")
    requested_family, _ = _normalize_variant(config.variant)
    require_offline_particles = requested_family == "C"
    train_source = _load_split_source(
        config, config.train_split, require_offline_particles=require_offline_particles
    )
    val_source = _load_split_source(
        config, config.val_split, require_offline_particles=require_offline_particles
    )
    if train_source.layout.to_dict() != val_source.layout.to_dict():
        raise ValueError("model_train and model_val hierarchy layouts differ")
    stack_source = None
    if config.stack_val_split:
        stack_source = _load_split_source(
            config,
            config.stack_val_split,
            require_offline_particles=require_offline_particles,
        )
        if train_source.layout.to_dict() != stack_source.layout.to_dict():
            raise ValueError("stack_val hierarchy layout differs from model_train")
    model, family, variant = _build_model(config, train_source.layout)
    model.to(device)
    optimizer = _optimizer(model, config)
    scaler = amp_grad_scaler(amp_enabled)
    hierarchy_loss_config, slot_loss_config = _loss_configs(config, family, variant)
    provenance = {
        "model_train": train_source.provenance,
        "model_val": val_source.provenance,
        **({"stack_val": stack_source.provenance} if stack_source is not None else {}),
    }
    source_metadata = {
        "contract": COARSE_TO_FINE_TRAIN_CONTRACT,
        "config": config.to_dict(),
        "model": _model_description(model, family, variant),
        "hierarchy_loss_config": hierarchy_loss_config.to_dict(),
        "slot_loss_config": None if slot_loss_config is None else slot_loss_config.to_dict(),
        "provenance": provenance,
        "optimizer_groups": [
            {"group_name": group.get("group_name"), "lr": group["lr"], "parameter_count": sum(p.numel() for p in group["params"])}
            for group in optimizer.param_groups
        ],
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }
    _save_json(output_dir / "source_metadata.json", source_metadata)
    _save_json(output_dir / "config.json", config.to_dict())

    curves: list[dict[str, Any]] = []
    best_epoch = -1
    best_loss = float("inf")
    best_val: Mapping[str, Any] = {}
    epochs_without_improvement = 0
    for epoch in range(int(config.epochs)):
        train_metrics = _run_epoch(
            model,
            train_source,
            config,
            family=family,
            variant=variant,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=amp_enabled,
            max_jets=config.max_train_jets,
            epoch=epoch,
        )
        with torch.no_grad():
            val_metrics = _run_epoch(
                model,
                val_source,
                config,
                family=family,
                variant=variant,
                device=device,
                optimizer=None,
                scaler=None,
                amp_enabled=amp_enabled,
                max_jets=config.max_val_jets,
                epoch=epoch,
            )
        row = {"epoch": int(epoch), "train": train_metrics, "model_val": val_metrics}
        curves.append(row)
        val_loss = float(val_metrics.get("selection.reconstruction_score", float("nan")))
        if math.isfinite(val_loss) and val_loss < best_loss:
            best_epoch = int(epoch)
            best_loss = val_loss
            best_val = dict(val_metrics)
            epochs_without_improvement = 0
            torch.save(
                _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    config=config,
                    family=family,
                    variant=variant,
                    metrics=val_metrics,
                    provenance=provenance,
                    checkpoint_role="best_model_val",
                ),
                output_dir / "best_model_val.pt",
            )
        else:
            epochs_without_improvement += 1
        if config.save_last_checkpoint:
            torch.save(
                _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    config=config,
                    family=family,
                    variant=variant,
                    metrics=val_metrics,
                    provenance=provenance,
                    checkpoint_role="last",
                ),
                output_dir / "last.pt",
            )
        _save_json(
            output_dir / "training_curves.json",
            {"selection_metric": "model_val.selection.reconstruction_score", "epochs": curves},
        )
        _write_curves_csv(diagnostics_dir / "epoch_metrics.csv", curves)
        _write_curves_csv(diagnostics_dir / "reconstruction_metrics.csv", curves)
        if int(config.early_stop_patience) >= 0 and epochs_without_improvement > int(config.early_stop_patience):
            break
    checkpoint_path = output_dir / "best_model_val.pt"
    if best_epoch < 0 or not checkpoint_path.exists():
        raise RuntimeError("training did not produce a finite model_val checkpoint")
    payload = _load_checkpoint(checkpoint_path, device)
    model.load_state_dict(payload["model_state_dict"])
    stack_metrics = None
    if stack_source is not None:
        with torch.no_grad():
            stack_metrics = _run_epoch(
                model,
                stack_source,
                config,
                family=family,
                variant=variant,
                device=device,
                optimizer=None,
                scaler=None,
                amp_enabled=amp_enabled,
                max_jets=config.max_stack_val_jets,
                epoch=best_epoch,
            )
    model_val_report = {
        "ok": True,
        "split": "model_val",
        "selection_only": True,
        "best_epoch": best_epoch,
        "metrics": best_val,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "provenance": val_source.provenance,
    }
    _save_json(output_dir / "model_val_report.json", model_val_report)
    if stack_metrics is not None:
        _save_json(
            output_dir / "stack_val_report.json",
            {
                "ok": True,
                "split": "stack_val",
                "selection_only": False,
                "best_epoch": best_epoch,
                "metrics": stack_metrics,
                "checkpoint_sha256": model_val_report["checkpoint_sha256"],
                "provenance": stack_source.provenance,
            },
        )
    report = {
        "ok": True,
        "contract": COARSE_TO_FINE_TRAIN_CONTRACT,
        "output_dir": str(output_dir),
        "family": family,
        "variant": variant,
        "best_epoch": best_epoch,
        "selection_metric": "model_val.selection.reconstruction_score",
        "best_model_val": best_val,
        "stack_val": stack_metrics,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": model_val_report["checkpoint_sha256"],
        "last_checkpoint": str(output_dir / "last.pt") if config.save_last_checkpoint else None,
        "training_curves": str(output_dir / "training_curves.json"),
        "reconstruction_diagnostics_csv": str(diagnostics_dir / "reconstruction_metrics.csv"),
        "source_metadata": str(output_dir / "source_metadata.json"),
        "model": _model_description(model, family, variant),
        "hierarchy_loss_config": hierarchy_loss_config.to_dict(),
        "slot_loss_config": None if slot_loss_config is None else slot_loss_config.to_dict(),
        "provenance": provenance,
        "final_test_loaded": False,
    }
    _save_json(output_dir / "run_report.json", report)
    return report


__all__ = [
    "COARSE_TO_FINE_ALLOWED_TRAIN_SPLITS",
    "COARSE_TO_FINE_TRAIN_CONTRACT",
    "CoarseToFineTrainConfig",
    "train_coarse_to_fine_reconstructor",
]
