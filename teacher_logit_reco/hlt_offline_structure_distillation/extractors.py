"""Numerical Step-3 HOSD physical-target extractors.

All builders are label blind.  They accept only one same-view constituent
tensor, its particle mask, and authenticated numerical resources.  Offline and
HLT-self targets deliberately call the same implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from teacher_logit_reco.relational_part.pair_base import (
    build_standard_four_pair_features,
)
from teacher_logit_reco.relational_part.region_tree import (
    build_reference_trees,
    validate_tree,
)
from teacher_logit_reco.relational_part.relation_density import (
    build_density_node_features,
)
from teacher_logit_reco.relational_part.relation_pid_charge import (
    build_charge_raw_features,
    pid_categories,
    quantize_charge,
)
from teacher_logit_reco.relational_part.relation_pt import (
    build_pt_raw_features,
)
from teacher_logit_reco.relational_part.relation_region import (
    build_batched_region_raw_features,
)
from teacher_logit_reco.relational_part.relation_track import (
    TRACK_COMPATIBILITY_FEATURE_NAMES,
    build_track_compatibility,
    build_track_node_features,
)

from .contracts import canonical_sha256, with_content_hash
from .target_schemas import (
    CA_TREE_COMPONENTS,
    COMPOSITION_COMPONENTS,
    DENSITY_COMPONENTS,
    HLT_REGION_PAIR_COMPONENTS,
    JET_COMPONENTS,
    RELATION_CHANNELS,
    RELATION_COMPONENTS,
    TRACK_COMPONENTS,
    TRACK_COMPONENT_PROXY_COMPONENTS,
    target_component_availability_groups,
    target_declarations,
)

TARGET_EXTRACTOR_MANIFEST_CONTRACT = "hosd_physical_target_extractors_v1"
EPSILON = 1.0e-6
QUANTILE_METHOD = "linear"
SUMMARY_QUANTILES = (0.10, 0.50, 0.90)
TRACK_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
_RELATION_PREFIXES = ("T_OFFLINE_RELATION_", "T_HLT_SELF_RELATION_")


@dataclass(frozen=True)
class TargetBatch:
    target_id: str
    component_names: tuple[str, ...]
    availability_groups: tuple[str, ...]
    values: torch.Tensor
    loss_mask: torch.Tensor
    diagnostics: Mapping[str, Any]

    def validate(self) -> None:
        if self.values.dtype != torch.float32:
            raise TypeError("target values must be float32")
        if self.loss_mask.dtype != torch.bool:
            raise TypeError("target loss mask must be boolean")
        if tuple(self.values.shape) != tuple(self.loss_mask.shape):
            raise ValueError("target values and loss mask shapes differ")
        if self.values.ndim not in {2, 4}:
            raise ValueError("targets must be global [B,D] or pair [B,D,N,N]")
        if int(self.values.shape[1]) != len(self.component_names):
            raise ValueError("target component axis differs from its schema")
        if len(self.availability_groups) != len(self.component_names):
            raise ValueError("target availability groups differ from its schema")
        if len(self.component_names) != len(set(self.component_names)):
            raise ValueError("target component names are not unique")
        if not bool(torch.isfinite(self.values).all()):
            raise FloatingPointError("target values contain NaN or infinity")
        if bool((self.values.masked_select(~self.loss_mask) != 0).any()):
            raise ValueError("masked target storage must be exactly zero")


@dataclass(frozen=True)
class ExtractorResources:
    d0_uncertainty_floor: float
    dz_uncertainty_floor: float
    sentinel_policy: Mapping[str, Any] | None = None
    weaver_module: Any | None = None

    def validate(self) -> None:
        for name, value in (
            ("d0_uncertainty_floor", self.d0_uncertainty_floor),
            ("dz_uncertainty_floor", self.dz_uncertainty_floor),
        ):
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")


def _physical_target_ids() -> tuple[str, ...]:
    return tuple(
        item.target_id
        for item in target_declarations()
        if item.campaign_status == "current_required"
        and item.availability_class
        in {"OFFLINE_RECO_DERIVED", "HLT_RECO_DERIVED", "DETERMINISTIC_PROXY"}
    )


PHYSICAL_TARGET_IDS = _physical_target_ids()


def build_target_extractor_manifest() -> dict[str, Any]:
    semantics = {
        "raw_layout": "[batch,particles,14]",
        "mask_layout": "[batch,particles] or [batch,1,particles]",
        "output_layouts": ["[batch,components]", "[batch,components,N,N]"],
        "dtype": "float32",
        "masked_storage": 0.0,
        "quantile_method": QUANTILE_METHOD,
        "summary_quantiles": list(SUMMARY_QUANTILES),
        "track_quantiles": list(TRACK_QUANTILES),
        "label_access": False,
        "constituent_matching_required": False,
        "target_ids": list(PHYSICAL_TARGET_IDS),
    }
    return with_content_hash(
        {
            "contract": TARGET_EXTRACTOR_MANIFEST_CONTRACT,
            "schema_version": 1,
            **semantics,
            "algorithm_semantics_sha256": canonical_sha256(semantics),
            "inherited_builders": {
                "BASE4": "build_standard_four_pair_features",
                "PT": "build_pt_raw_features",
                "TRACK": "build_track_node_features/build_track_compatibility",
                "PID": "pid_categories",
                "CHARGE": "build_charge_raw_features",
                "DENSITY": "build_density_node_features",
                "REGION": "build_batched_region_raw_features",
                "CA_TREE": "build_reference_trees",
            },
        }
    )


def _inputs(
    raw_tokens: Any,
    mask: Any,
    vectors: Any | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    raw = torch.as_tensor(raw_tokens, dtype=torch.float32).detach().cpu()
    if raw.ndim != 3 or int(raw.shape[2]) != 14:
        raise ValueError("raw_tokens must have shape [batch,particles,14]")
    valid = torch.as_tensor(mask).detach().cpu()
    if tuple(valid.shape) == (int(raw.shape[0]), int(raw.shape[1])):
        valid = valid.unsqueeze(1)
    if tuple(valid.shape) != (int(raw.shape[0]), 1, int(raw.shape[1])):
        raise ValueError("mask must have shape [batch,particles] or [batch,1,particles]")
    if valid.dtype != torch.bool:
        if not bool(((valid == 0) | (valid == 1)).all()):
            raise ValueError("mask must be boolean or binary")
        valid = valid.bool()
    if not bool(torch.isfinite(raw[valid[:, 0]]).all()):
        raise FloatingPointError("valid raw tokens contain NaN or infinity")
    if bool((raw[:, :, 0].masked_select(valid[:, 0]) < 0).any()):
        raise ValueError("valid scalar transverse momentum must be nonnegative")
    safe = raw.clone()
    safe.masked_fill_(~valid.transpose(1, 2), 0.0)
    if vectors is None:
        pt = safe[:, :, 0].double()
        eta = safe[:, :, 1].double()
        phi = safe[:, :, 2].double()
        energy = safe[:, :, 3].double()
        vectors_bn4 = torch.stack(
            (
                pt * torch.cos(phi),
                pt * torch.sin(phi),
                pt * torch.sinh(eta),
                energy,
            ),
            dim=-1,
        )
    else:
        supplied = torch.as_tensor(vectors).detach().cpu().double()
        if tuple(supplied.shape) == (
            int(raw.shape[0]),
            4,
            int(raw.shape[1]),
        ):
            supplied = supplied.transpose(1, 2)
        if tuple(supplied.shape) != (
            int(raw.shape[0]),
            int(raw.shape[1]),
            4,
        ):
            raise ValueError("vectors must have shape [B,N,4] or [B,4,N]")
        if not bool(torch.isfinite(supplied[valid[:, 0]]).all()):
            raise FloatingPointError("valid vectors contain NaN or infinity")
        vectors_bn4 = supplied.masked_fill(~valid.transpose(1, 2), 0.0)
    vectors_cf = vectors_bn4.transpose(1, 2).float().contiguous()
    return safe, valid, vectors_cf, vectors_bn4.numpy()


def _result(
    target_id: str,
    names: Sequence[str],
    values: torch.Tensor,
    loss_mask: torch.Tensor,
    diagnostics: Mapping[str, Any] | None = None,
) -> TargetBatch:
    values = values.float().masked_fill(~loss_mask.bool(), 0.0)
    result = TargetBatch(
        target_id=target_id,
        component_names=tuple(names),
        availability_groups=target_component_availability_groups(
            target_id, tuple(names)
        ),
        values=values,
        loss_mask=loss_mask.bool(),
        diagnostics=dict(diagnostics or {}),
    )
    result.validate()
    return result


def _jet_target(
    target_id: str,
    raw: torch.Tensor,
    valid: torch.Tensor,
    vectors_bn4: np.ndarray,
) -> TargetBatch:
    batch = int(raw.shape[0])
    values = np.zeros((batch, 10), dtype=np.float64)
    masks = np.zeros((batch, 10), dtype=bool)
    for row in range(batch):
        selected = valid[row, 0].numpy()
        count = int(selected.sum())
        if not count:
            continue
        vector = vectors_bn4[row, selected].astype(np.float64).sum(axis=0)
        px, py, pz, energy = map(float, vector)
        jet_pt = math.hypot(px, py)
        mass = math.sqrt(max(energy * energy - px * px - py * py - pz * pz, 0.0))
        if energy < 0:
            raise ValueError("summed valid constituent energy is negative")
        scalar_pt = raw[row, selected, 0].double().numpy()
        total_scalar_pt = float(scalar_pt.sum())
        ordered = np.sort(scalar_pt)[::-1]
        direction = jet_pt > 0
        values[row] = (
            math.log1p(jet_pt),
            math.asinh(pz / jet_pt) if direction else 0.0,
            math.sin(math.atan2(py, px)) if direction else 0.0,
            math.cos(math.atan2(py, px)) if direction else 0.0,
            math.log1p(mass),
            math.log1p(energy),
            math.log1p(count),
            float(ordered[0] / (total_scalar_pt + EPSILON)),
            float(ordered[1] / (total_scalar_pt + EPSILON)) if count > 1 else 0.0,
            float(np.sqrt(np.square(scalar_pt).sum()) / (total_scalar_pt + EPSILON)),
        )
        masks[row] = True
        masks[row, 1:4] = direction
    return _result(
        target_id,
        JET_COMPONENTS,
        torch.from_numpy(values),
        torch.from_numpy(masks),
        {"float64_vector_sums": True},
    )


def _composition_target(
    target_id: str,
    raw: torch.Tensor,
    valid: torch.Tensor,
) -> TargetBatch:
    categories = pid_categories(raw[:, :, 5:10].transpose(1, 2), valid)
    charge, _ = quantize_charge(raw[:, :, 4], valid)
    batch = int(raw.shape[0])
    values = torch.zeros(batch, 16, dtype=torch.float64)
    masks = torch.zeros(batch, 16, dtype=torch.bool)
    for row in range(batch):
        selected = valid[row, 0]
        count = int(selected.sum())
        if not count:
            continue
        pt = raw[row, :, 0].double().masked_select(selected)
        cats = categories[row].masked_select(selected)
        q = charge[row].double().masked_select(selected)
        total_pt = float(pt.sum())
        for category in range(6):
            in_category = cats == category
            values[row, category] = float(in_category.sum()) / count
            values[row, 6 + category] = float(pt.masked_select(in_category).sum()) / (
                total_pt + EPSILON
            )
        values[row, 12] = float((q < 0).sum()) / count
        values[row, 13] = float((q == 0).sum()) / count
        values[row, 14] = float((q > 0).sum()) / count
        values[row, 15] = float(q.sum()) / max(int((q != 0).sum()), 1)
        masks[row] = True
    return _result(target_id, COMPOSITION_COMPONENTS, values, masks)


def _linear_quantiles(values: np.ndarray, quantiles: Sequence[float]) -> np.ndarray:
    if values.size == 0:
        return np.zeros(len(quantiles), dtype=np.float64)
    return np.asarray(
        np.quantile(values.astype(np.float64), quantiles, method=QUANTILE_METHOD),
        dtype=np.float64,
    )


def _track_target(
    target_id: str,
    raw: torch.Tensor,
    valid: torch.Tensor,
    resources: ExtractorResources,
) -> TargetBatch:
    details = build_track_node_features(
        raw,
        valid,
        d0_uncertainty_floor=resources.d0_uncertainty_floor,
        dz_uncertainty_floor=resources.dz_uncertainty_floor,
        sentinel_policy=resources.sentinel_policy,
    )
    track_valid = details["track_valid"]
    category = pid_categories(raw[:, :, 5:10].transpose(1, 2), valid)
    charged_domain = (category == 0) | (category == 3) | (category == 4)
    batch = int(raw.shape[0])
    values = np.zeros((batch, 32), dtype=np.float64)
    masks = np.zeros((batch, 32), dtype=bool)
    for row in range(batch):
        particle = valid[row, 0]
        track = track_valid[row]
        particle_count = max(int(particle.sum()), 1)
        pt = raw[row, :, 0].double()
        particle_pt = float(pt.masked_select(particle).sum())
        track_pt = float(pt.masked_select(track).sum())
        unavailable = particle & charged_domain[row] & ~track
        values[row, :4] = (
            float(track.sum()) / particle_count,
            track_pt / (particle_pt + EPSILON),
            float(unavailable.sum()) / particle_count,
            float(pt.masked_select(unavailable).sum()) / (particle_pt + EPSILON),
        )
        masks[row, :4] = True
        if not bool(track.any()):
            continue
        d0 = details["raw_d0_significance"][row].masked_select(track).double().numpy()
        dz = details["raw_dz_significance"][row].masked_select(track).double().numpy()
        values[row, 4:9] = _linear_quantiles(d0, TRACK_QUANTILES)
        values[row, 9:14] = _linear_quantiles(dz, TRACK_QUANTILES)
        values[row, 14:19] = _linear_quantiles(np.abs(d0), TRACK_QUANTILES)
        values[row, 19:24] = _linear_quantiles(np.abs(dz), TRACK_QUANTILES)
        values[row, 24:27] = [float(np.mean(np.abs(d0) > cut)) for cut in (1, 2, 3)]
        values[row, 27:30] = [float(np.mean(np.abs(dz) > cut)) for cut in (1, 2, 3)]
        weights = pt.masked_select(track).numpy()
        weight_sum = float(weights.sum())
        values[row, 30] = float((weights * np.abs(d0)).sum()) / (
            weight_sum + EPSILON
        )
        values[row, 31] = float((weights * np.abs(dz)).sum()) / (
            weight_sum + EPSILON
        )
        masks[row, 4:] = True
    return _result(
        target_id,
        TRACK_COMPONENTS,
        torch.from_numpy(values),
        torch.from_numpy(masks),
        {
            "d0_uncertainty_floor": float(resources.d0_uncertainty_floor),
            "dz_uncertainty_floor": float(resources.dz_uncertainty_floor),
            "quantile_method": QUANTILE_METHOD,
        },
    )


def _weighted_node_means(
    descriptor: torch.Tensor,
    raw: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, channels = int(descriptor.shape[0]), int(descriptor.shape[1])
    values = torch.zeros(batch, channels, dtype=torch.float64)
    masks = torch.zeros(batch, channels, dtype=torch.bool)
    for row in range(batch):
        selected = valid[row, 0]
        count = int(selected.sum())
        if not count:
            continue
        node = descriptor[row, :, selected].double()
        weights = raw[row, selected, 0].double()
        if float(weights.sum()) > 0:
            values[row] = (node * weights.unsqueeze(0)).sum(dim=1) / weights.sum()
        else:
            values[row] = node.mean(dim=1)
        masks[row] = True
    return values, masks


def _density_target(
    target_id: str,
    raw: torch.Tensor,
    valid: torch.Tensor,
    resources: ExtractorResources,
) -> TargetBatch:
    details = build_density_node_features(
        raw,
        valid,
        d0_uncertainty_floor=resources.d0_uncertainty_floor,
        dz_uncertainty_floor=resources.dz_uncertainty_floor,
        sentinel_policy=resources.sentinel_policy,
    )
    values, masks = _weighted_node_means(details["descriptor"], raw, valid)
    return _result(target_id, DENSITY_COMPONENTS, values, masks)


def _entropy(probabilities: np.ndarray, actual_count: int) -> float:
    positive = probabilities[probabilities > 0]
    if positive.size == 0:
        return 0.0
    return float(-(positive * np.log(positive)).sum() / math.log(max(actual_count, 2)))


def _tree_target(
    target_id: str,
    raw: torch.Tensor,
    valid: torch.Tensor,
    trees: Sequence[Mapping[str, Any]],
) -> TargetBatch:
    batch = int(raw.shape[0])
    values = np.zeros((batch, 26), dtype=np.float64)
    masks = np.zeros((batch, 26), dtype=bool)
    for row, tree in enumerate(trees):
        validate_tree(tree)
        n_valid = int(tree["n_valid"])
        if not n_valid:
            continue
        particle_mask = valid[row, 0].numpy()
        leaf_nodes = np.asarray(tree["leaf_to_node"])[particle_mask]
        leaf_depth = np.asarray(tree["depth"], dtype=np.float64)[leaf_nodes]
        weights = raw[row, particle_mask, 0].double().numpy()
        if float(weights.sum()) > 0:
            mean_depth = float(np.average(leaf_depth, weights=weights))
            std_depth = float(
                np.sqrt(np.average(np.square(leaf_depth - mean_depth), weights=weights))
            )
        else:
            mean_depth = float(leaf_depth.mean())
            std_depth = float(leaf_depth.std(ddof=0))
        values[row, :5] = (
            n_valid / 128.0,
            int(tree["n_nodes"]) / 255.0,
            float(leaf_depth.max()) / 127.0,
            mean_depth / 127.0,
            std_depth / 127.0,
        )
        offset = 5
        cluster_records: dict[int, list[tuple[int, float, float, int]]] = {}
        for k in (2, 4, 8):
            assignment = np.asarray(tree["assignments"][str(k)])[particle_mask]
            nodes = sorted(int(node) for node in np.unique(assignment))
            records = [
                (
                    node,
                    float(
                        raw[row, particle_mask, 0]
                        .double()
                        .numpy()[assignment == node]
                        .sum()
                    ),
                    float(np.asarray(tree["mass"])[node]),
                    int((assignment == node).sum()),
                )
                for node in nodes
            ]
            records.sort(key=lambda item: (-item[1], item[0]))
            cluster_records[k] = records
            values[row, offset] = int(tree["actual_cluster_counts"][str(k)]) / k
            offset += 1
        for k in (2, 4, 8):
            records = cluster_records[k]
            total_pt = sum(item[1] for item in records)
            for rank in range(3):
                values[row, offset] = (
                    records[rank][1] / (total_pt + EPSILON)
                    if rank < len(records)
                    else 0.0
                )
                offset += 1
        jet_mass = float(np.asarray(tree["mass"])[int(tree["root"])])
        for k in (2, 4, 8):
            records = cluster_records[k]
            values[row, offset] = records[0][2] / jet_mass if jet_mass > 0 else 0.0
            offset += 1
        for k in (2, 4, 8):
            records = cluster_records[k]
            probs = np.asarray([item[3] / n_valid for item in records])
            values[row, offset] = _entropy(probs, len(records))
            offset += 1
        for k in (2, 4, 8):
            records = cluster_records[k]
            total_pt = sum(item[1] for item in records)
            probs = (
                np.asarray([item[1] / total_pt for item in records])
                if total_pt > 0
                else np.zeros(len(records))
            )
            values[row, offset] = _entropy(probs, len(records))
            offset += 1
        if offset != 26:
            raise AssertionError("C/A target component count drifted")
        masks[row] = True
    return _result(
        target_id,
        CA_TREE_COMPONENTS,
        torch.from_numpy(values),
        torch.from_numpy(masks),
    )


def _pair_selection(valid: torch.Tensor, channel_type: str) -> torch.Tensor:
    length = int(valid.shape[1])
    pair = valid.unsqueeze(-1) & valid.unsqueeze(-2)
    if channel_type.startswith("unordered_pair") or channel_type == "pair_binary":
        return pair & torch.triu(
            torch.ones(length, length, dtype=torch.bool), diagonal=1
        ).unsqueeze(0)
    return pair & (~torch.eye(length, dtype=torch.bool).unsqueeze(0))


def _reduce_relation_channel(
    data: torch.Tensor,
    applicability: torch.Tensor,
    *,
    channel_type: str,
    categories: Sequence[str] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = int(data.shape[0])
    dimension = 5 if channel_type.endswith("_continuous") else (
        len(categories or ()) if channel_type == "categorical" else 1
    )
    values = np.zeros((batch, dimension), dtype=np.float64)
    masks = np.zeros((batch, dimension), dtype=bool)
    for row in range(batch):
        selected = data[row].masked_select(applicability[row]).detach().cpu().numpy()
        if selected.size == 0:
            continue
        if channel_type.endswith("_continuous"):
            selected = selected.astype(np.float64)
            values[row] = (
                float(selected.mean()),
                float(selected.std(ddof=0)),
                *_linear_quantiles(selected, SUMMARY_QUANTILES),
            )
        elif channel_type.endswith("_binary"):
            values[row, 0] = float(selected.astype(np.float64).mean())
        elif channel_type == "categorical":
            indices = selected.astype(np.int64)
            if np.any(indices < 0) or np.any(indices >= dimension):
                raise ValueError("categorical relation index is out of range")
            values[row] = np.bincount(indices, minlength=dimension) / indices.size
        else:  # pragma: no cover - registry validation prevents this
            raise ValueError(f"unknown relation channel type {channel_type}")
        masks[row] = True
    return torch.from_numpy(values), torch.from_numpy(masks)


def _relation_target(
    target_id: str,
    family: str,
    raw: torch.Tensor,
    valid: torch.Tensor,
    vectors_cf: torch.Tensor,
    trees: Sequence[Mapping[str, Any]],
    resources: ExtractorResources,
) -> TargetBatch:
    sources: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    particle = valid[:, 0]
    if family == "BASE4":
        tensor = build_standard_four_pair_features(
            vectors_cf, mask=valid, module=resources.weaver_module
        )
        for index, (name, channel_type, _) in enumerate(RELATION_CHANNELS[family]):
            sources[name] = (tensor[:, index], _pair_selection(particle, channel_type))
    elif family == "PT":
        tensor = build_pt_raw_features(vectors_cf, valid)
        for index, (name, channel_type, _) in enumerate(RELATION_CHANNELS[family]):
            sources[name] = (tensor[:, index], _pair_selection(particle, channel_type))
    elif family == "TRACK":
        details = build_track_compatibility(
            raw,
            valid,
            d0_uncertainty_floor=resources.d0_uncertainty_floor,
            dz_uncertainty_floor=resources.dz_uncertainty_floor,
            sentinel_policy=resources.sentinel_policy,
        )
        track = details["track_valid"]
        for index, name in enumerate(
            name for name, kind, _ in RELATION_CHANNELS[family]
            if kind == "node_continuous"
        ):
            sources[name] = (details["continuous"][:, index], track)
        sources["track_valid"] = (track.float(), particle)
        compatibility_names = [
            name
            for name, kind, _ in RELATION_CHANNELS[family]
            if kind.endswith("pair_continuous")
        ]
        for index, name in enumerate(compatibility_names):
            kind = next(kind for n, kind, _ in RELATION_CHANNELS[family] if n == name)
            sources[name] = (
                details["compatibility"][:, index],
                _pair_selection(track, kind),
            )
        sources["pair_track_validity_state"] = (
            details["validity_index"],
            _pair_selection(particle, "ordered_pair_continuous"),
        )
    elif family == "PID":
        category = pid_categories(raw[:, :, 5:10].transpose(1, 2), valid)
        length = int(raw.shape[1])
        sources["query_pid"] = (
            category.unsqueeze(-1).expand(-1, -1, length),
            _pair_selection(particle, "ordered_pair_continuous"),
        )
        sources["context_pid"] = (
            category.unsqueeze(-2).expand(-1, length, -1),
            _pair_selection(particle, "ordered_pair_continuous"),
        )
        sources["directed_pid_pair"] = (
            category.unsqueeze(-1) * 6 + category.unsqueeze(-2),
            _pair_selection(particle, "ordered_pair_continuous"),
        )
    elif family == "CHARGE":
        tensor, _ = build_charge_raw_features(raw[:, :, 4], valid)
        for index, (name, channel_type, _) in enumerate(RELATION_CHANNELS[family]):
            sources[name] = (tensor[:, index], _pair_selection(particle, channel_type))
    elif family == "DENSITY":
        details = build_density_node_features(
            raw,
            valid,
            d0_uncertainty_floor=resources.d0_uncertainty_floor,
            dz_uncertainty_floor=resources.dz_uncertainty_floor,
            sentinel_policy=resources.sentinel_policy,
        )
        for index, (name, _, _) in enumerate(RELATION_CHANNELS[family]):
            sources[name] = (details["descriptor"][:, index], particle)
    elif family == "REGION":
        tensor = build_batched_region_raw_features(trees, raw, valid)
        for index, (name, channel_type, _) in enumerate(RELATION_CHANNELS[family]):
            sources[name] = (tensor[:, index], _pair_selection(particle, channel_type))
    else:  # pragma: no cover
        raise ValueError(f"unknown relation family {family}")

    value_parts = []
    mask_parts = []
    for name, channel_type, categories in RELATION_CHANNELS[family]:
        if name not in sources:
            raise AssertionError(f"{family} raw channel {name} was not produced")
        reduced, reduced_mask = _reduce_relation_channel(
            *sources[name],
            channel_type=channel_type,
            categories=categories,
        )
        value_parts.append(reduced)
        mask_parts.append(reduced_mask)
    values = torch.cat(value_parts, dim=1)
    masks = torch.cat(mask_parts, dim=1)
    return _result(
        target_id,
        RELATION_COMPONENTS[family],
        values,
        masks,
        {
            "family": family,
            "self_pairs_included": False,
            "quantile_method": QUANTILE_METHOD,
        },
    )


def _track_component_proxy(
    target_id: str,
    raw: torch.Tensor,
    valid: torch.Tensor,
    resources: ExtractorResources,
) -> TargetBatch:
    details = build_track_compatibility(
        raw,
        valid,
        d0_uncertainty_floor=resources.d0_uncertainty_floor,
        dz_uncertainty_floor=resources.dz_uncertainty_floor,
        sentinel_policy=resources.sentinel_policy,
    )
    batch, particles = int(raw.shape[0]), int(raw.shape[1])
    values = np.zeros((batch, 17), dtype=np.float64)
    masks = np.zeros((batch, 17), dtype=bool)
    overflow = []
    for row in range(batch):
        track_indices = [
            index for index in range(particles) if bool(details["track_valid"][row, index])
        ]
        parent = {index: index for index in track_indices}

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            left_root, right_root = find(left), find(right)
            if left_root == right_root:
                return
            if left_root < right_root:
                parent[right_root] = left_root
            else:
                parent[left_root] = right_root

        for position, left in enumerate(track_indices):
            for right in track_indices[position + 1 :]:
                if (
                    float(details["chi2"][row, left, right]) <= 4.0
                    and float(details["delta_r"][row, left, right]) <= 0.10
                ):
                    union(left, right)
        groups: dict[int, list[int]] = {}
        for index in track_indices:
            groups.setdefault(find(index), []).append(index)
        components = [sorted(group) for group in groups.values() if len(group) >= 2]
        components.sort(
            key=lambda group: (
                -float(raw[row, group, 0].double().sum()),
                min(group),
            )
        )
        overflow.append(max(0, len(components) - 8))
        values[row, 0] = min(len(components), 8) / 8.0
        masks[row, 0] = True
        jet_scalar_pt = float(raw[row, valid[row, 0], 0].double().sum())
        d0_sig = details["raw_d0_significance"][row]
        dz_sig = details["raw_dz_significance"][row]
        for rank, group in enumerate(components[:4]):
            offset = 1 + rank * 4
            values[row, offset : offset + 4] = (
                len(group) / 40.0,
                float(raw[row, group, 0].double().sum()) / (
                    jet_scalar_pt + EPSILON
                ),
                float(d0_sig[group].abs().double().mean()),
                float(dz_sig[group].abs().double().mean()),
            )
            masks[row, offset : offset + 4] = True
    return _result(
        target_id,
        TRACK_COMPONENT_PROXY_COMPONENTS,
        torch.from_numpy(values),
        torch.from_numpy(masks),
        {
            "component_cap": 8,
            "stored_component_slots": 4,
            "overflow_count_per_jet": overflow,
            "edge_rule": "track_chi2_le_4_and_delta_r_le_0p10",
        },
    )


def _track_pair_target(
    target_id: str,
    raw: torch.Tensor,
    valid: torch.Tensor,
    resources: ExtractorResources,
) -> TargetBatch:
    details = build_track_compatibility(
        raw,
        valid,
        d0_uncertainty_floor=resources.d0_uncertainty_floor,
        dz_uncertainty_floor=resources.dz_uncertainty_floor,
        sentinel_policy=resources.sentinel_policy,
    )
    pair = _pair_selection(details["track_valid"], "ordered_pair_continuous")
    mask = pair.unsqueeze(1).expand_as(details["compatibility"])
    return _result(
        target_id,
        TRACK_COMPATIBILITY_FEATURE_NAMES,
        details["compatibility"],
        mask,
        {"symmetry": "directed", "diagonal_inapplicable": True},
    )


def _lca(tree: Mapping[str, Any], left_leaf: int, right_leaf: int) -> int:
    parent = np.asarray(tree["parent"], dtype=np.int32)
    ancestors: set[int] = set()
    node = int(left_leaf)
    while node >= 0:
        ancestors.add(node)
        node = int(parent[node])
    node = int(right_leaf)
    while node not in ancestors:
        node = int(parent[node])
    return node


def _region_pair_target(
    target_id: str,
    raw: torch.Tensor,
    valid: torch.Tensor,
    trees: Sequence[Mapping[str, Any]],
) -> TargetBatch:
    batch, particles = int(raw.shape[0]), int(raw.shape[1])
    values = torch.zeros(batch, 8, particles, particles, dtype=torch.float32)
    masks = torch.zeros_like(values, dtype=torch.bool)
    for row, tree in enumerate(trees):
        validate_tree(tree)
        leaf_map = np.asarray(tree["leaf_to_node"], dtype=np.int32)
        parent_depth = np.asarray(tree["depth"], dtype=np.float64)
        root = int(tree["root"])
        jet_mass = float(np.asarray(tree["mass"])[root]) if root >= 0 else 0.0
        assignments = {
            k: np.asarray(tree["assignments"][str(k)], dtype=np.int32)
            for k in (2, 4, 8)
        }
        indices = np.flatnonzero(valid[row, 0].numpy())
        for position, left in enumerate(indices):
            for right in indices[position + 1 :]:
                ancestor = _lca(tree, int(leaf_map[left]), int(leaf_map[right]))
                row_values = (
                    float(assignments[2][left] == assignments[2][right]),
                    float(assignments[4][left] == assignments[4][right]),
                    float(assignments[8][left] == assignments[8][right]),
                    float(parent_depth[ancestor] / 127.0),
                    math.log1p(float(np.asarray(tree["merge_delta_r"])[ancestor])),
                    math.log1p(float(np.asarray(tree["merge_kt"])[ancestor])),
                    float(np.clip(np.asarray(tree["merge_z"])[ancestor], 0.0, 0.5)),
                    math.log1p(
                        float(np.asarray(tree["merge_mass"])[ancestor])
                        / max(jet_mass, EPSILON)
                    ),
                )
                values[row, :, left, right] = torch.tensor(row_values)
                masks[row, :, left, right] = True
    return _result(
        target_id,
        HLT_REGION_PAIR_COMPONENTS,
        values,
        masks,
        {
            "symmetry": "unordered_upper_triangle",
            "lca_depth_denominator": 127,
        },
    )


def _family_from_target(target_id: str) -> str | None:
    for prefix in _RELATION_PREFIXES:
        if target_id.startswith(prefix):
            return target_id.removeprefix(prefix)
    return None


def extract_registered_target(
    target_id: str,
    raw_tokens: Any,
    mask: Any,
    *,
    resources: ExtractorResources,
    vectors: Any | None = None,
    trees: Sequence[Mapping[str, Any]] | None = None,
) -> TargetBatch:
    """Extract one registered physical target without labels or matching."""

    if target_id not in PHYSICAL_TARGET_IDS:
        raise ValueError(f"{target_id!r} is not a Step-3 physical target")
    resources.validate()
    raw, valid, vectors_cf, vectors_bn4 = _inputs(raw_tokens, mask, vectors)
    base_id = target_id.replace("T_HLT_SELF_", "T_OFFLINE_", 1)
    family = _family_from_target(target_id)
    requires_trees = (
        base_id == "T_OFFLINE_CA_TREE_26"
        or family == "REGION"
        or target_id == "T_HLT_REGION_PAIR_8"
    )
    if requires_trees:
        if trees is None:
            trees = build_reference_trees(
                vectors_bn4,
                raw.numpy(),
                valid[:, 0].numpy(),
            )
        else:
            trees = tuple(trees)
            if len(trees) != int(raw.shape[0]):
                raise ValueError("tree batch size differs from raw tokens")
            for tree in trees:
                validate_tree(tree)
    if base_id == "T_OFFLINE_JET_10":
        return _jet_target(target_id, raw, valid, vectors_bn4)
    if base_id == "T_OFFLINE_COMPOSITION_16":
        return _composition_target(target_id, raw, valid)
    if base_id == "T_OFFLINE_TRACK_32":
        return _track_target(target_id, raw, valid, resources)
    if base_id == "T_OFFLINE_DENSITY_22":
        return _density_target(target_id, raw, valid, resources)
    if base_id == "T_OFFLINE_CA_TREE_26":
        return _tree_target(target_id, raw, valid, trees)
    if base_id == "T_OFFLINE_TRACK_COMPONENT_PROXY_17":
        return _track_component_proxy(target_id, raw, valid, resources)
    if family is not None:
        return _relation_target(
            target_id,
            family,
            raw,
            valid,
            vectors_cf,
            trees,
            resources,
        )
    if target_id == "T_HLT_TRACK_PAIR_13":
        return _track_pair_target(target_id, raw, valid, resources)
    if target_id == "T_HLT_REGION_PAIR_8":
        return _region_pair_target(target_id, raw, valid, trees)
    raise AssertionError(f"registered physical target {target_id} has no extractor")


__all__ = [
    "EPSILON",
    "ExtractorResources",
    "PHYSICAL_TARGET_IDS",
    "QUANTILE_METHOD",
    "SUMMARY_QUANTILES",
    "TARGET_EXTRACTOR_MANIFEST_CONTRACT",
    "TRACK_QUANTILES",
    "TargetBatch",
    "build_target_extractor_manifest",
    "extract_registered_target",
]
