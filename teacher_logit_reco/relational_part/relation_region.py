"""Beam-free exclusive angular-tree REGION relation family."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import require_sha256, validate_content_hash, with_content_hash
from .normalization import GLOBAL_EPSILON
from .pair_base import require_torch
from .relation_pt import valid_pair_mask
from .region_tree import (
    EXCLUSIVE_RESOLUTIONS,
    REFERENCE_RADIUS,
    validate_tree,
)

try:
    import torch as _torch
except ImportError:  # pragma: no cover
    _torch = None


REGION_RELATION_CONTRACT = "relational_part_region_relation_v1"
REGION_NORMALIZATION_CONTRACT = "relational_part_region_normalization_v1"
REGION_ENCODED_DIMENSION = 12
REGION_RAW_DIMENSION = 41

REGION_SAME_CLUSTER_NAMES = tuple(
    f"same_cluster_K{k}" for k in EXCLUSIVE_RESOLUTIONS
)
REGION_LCA_NAMES = (
    "lca_normalized_depth",
    "lca_log_merge_delta_r",
    "lca_log_merge_kt",
    "lca_merge_z",
    "lca_log_merge_mass_fraction",
)
REGION_ENDPOINT_DESCRIPTOR_NAMES = tuple(
    f"{endpoint}_K{k}_{quantity}"
    for k in EXCLUSIVE_RESOLUTIONS
    for endpoint in ("query", "context")
    for quantity in ("log_pt_fraction", "log_mass_fraction", "multiplicity_fraction")
)
REGION_WITHIN_CLUSTER_PT_NAMES = tuple(
    f"{endpoint}_K{k}_particle_pt_fraction"
    for k in EXCLUSIVE_RESOLUTIONS
    for endpoint in ("query", "context")
)
REGION_AXIS_DISTANCE_NAMES = tuple(
    f"{endpoint}_K{k}_axis_delta_r_over_Rref"
    for k in EXCLUSIVE_RESOLUTIONS
    for endpoint in ("query", "context")
)
REGION_RANK_DIFFERENCE_NAMES = tuple(
    f"K{k}_context_minus_query_cluster_pt_rank" for k in EXCLUSIVE_RESOLUTIONS
)
REGION_RAW_FEATURE_NAMES = (
    *REGION_SAME_CLUSTER_NAMES,
    *REGION_LCA_NAMES,
    *REGION_ENDPOINT_DESCRIPTOR_NAMES,
    *REGION_WITHIN_CLUSTER_PT_NAMES,
    *REGION_AXIS_DISTANCE_NAMES,
    *REGION_RANK_DIFFERENCE_NAMES,
)
if len(REGION_RAW_FEATURE_NAMES) != REGION_RAW_DIMENSION:
    raise AssertionError("REGION feature schema must contain 41 channels")
REGION_ROBUST_FEATURE_NAMES = tuple(
    name for name in REGION_RAW_FEATURE_NAMES
    if name not in REGION_SAME_CLUSTER_NAMES
)


if _torch is None:  # pragma: no cover
    class _ModuleBase:
        pass
else:
    _ModuleBase = _torch.nn.Module


def _eta_phi(vector: np.ndarray) -> tuple[float, float]:
    px, py, pz, _ = map(float, vector)
    pt = math.hypot(px, py)
    return math.asinh(pz / max(pt, 1e-300)), math.atan2(py, px)


def _average_cluster_ranks(nodes: Sequence[int], pt: np.ndarray) -> dict[int, float]:
    values = np.asarray([pt[node] for node in nodes], dtype=np.float64)
    output: dict[int, float] = {}
    denominator = max(len(nodes) - 1, 1)
    for index, node in enumerate(nodes):
        greater = int(np.sum(values > values[index]))
        equal = int(np.sum(values == values[index]))
        output[int(node)] = (greater + .5 * (equal - 1)) / denominator
    return output


def _lca(tree: Mapping[str, Any], left: int, right: int) -> int:
    parent = np.asarray(tree["parent"])
    depth = np.asarray(tree["depth"])
    first, second = int(left), int(right)
    while depth[first] > depth[second]:
        first = int(parent[first])
    while depth[second] > depth[first]:
        second = int(parent[second])
    while first != second:
        first = int(parent[first])
        second = int(parent[second])
    return first


def build_region_raw_features(
    tree: Mapping[str, Any],
    raw_tokens: Any,
    mask: Any,
) -> Any:
    """Materialize the transient 41-channel directed relation."""

    torch = require_torch()
    validate_tree(tree)
    if raw_tokens.ndim != 2 or tuple(raw_tokens.shape) != (
        int(tree["n_particles"]), 14
    ):
        raise ValueError("REGION raw tokens disagree with the tree")
    if tuple(mask.shape) != (int(tree["n_particles"]),):
        raise ValueError("REGION mask disagrees with the tree")
    length = int(tree["n_particles"])
    output = np.zeros((REGION_RAW_DIMENSION, length, length), dtype=np.float32)
    valid_indices = np.flatnonzero(np.asarray(mask.detach().cpu(), dtype=bool))
    if not len(valid_indices):
        return torch.from_numpy(output).to(
            device=raw_tokens.device, dtype=raw_tokens.dtype
        )
    leaf_to_node = np.asarray(tree["leaf_to_node"])
    depth = np.asarray(tree["depth"])
    node_vectors = np.asarray(tree["vectors"], dtype=np.float64)
    node_pt = np.asarray(tree["pt"], dtype=np.float64)
    node_mass = np.asarray(tree["mass"], dtype=np.float64)
    multiplicity = np.asarray(tree["multiplicity"], dtype=np.float64)
    root = int(tree["root"])
    jet_pt = float(node_pt[root])
    jet_mass = float(node_mass[root])
    maximum_leaf_depth = max(
        int(depth[int(leaf_to_node[index])]) for index in valid_indices
    )
    assignments = {
        k: np.asarray(tree["assignments"][str(k)]) for k in EXCLUSIVE_RESOLUTIONS
    }
    clusters = {
        k: tuple(sorted(set(int(assignments[k][index]) for index in valid_indices)))
        for k in EXCLUSIVE_RESOLUTIONS
    }
    ranks = {
        k: _average_cluster_ranks(clusters[k], node_pt)
        for k in EXCLUSIVE_RESOLUTIONS
    }
    tokens = raw_tokens.detach().cpu().numpy().astype(np.float64, copy=False)
    for query in valid_indices:
        for context in valid_indices:
            row: list[float] = [
                float(assignments[k][query] == assignments[k][context])
                for k in EXCLUSIVE_RESOLUTIONS
            ]
            ancestor = _lca(
                tree, int(leaf_to_node[query]), int(leaf_to_node[context])
            )
            row.append(float(depth[ancestor]) / max(maximum_leaf_depth, 1))
            if query == context:
                row.extend((0.0, 0.0, 0.0, 0.0))
            else:
                row.extend(
                    (
                        math.log(
                            float(tree["merge_delta_r"][ancestor])
                            / REFERENCE_RADIUS
                            + GLOBAL_EPSILON
                        ),
                        math.log(
                            float(tree["merge_kt"][ancestor])
                            / (jet_pt * REFERENCE_RADIUS + GLOBAL_EPSILON)
                            + GLOBAL_EPSILON
                        ),
                        float(tree["merge_z"][ancestor]),
                        math.log(
                            (float(tree["merge_mass"][ancestor]) + GLOBAL_EPSILON)
                            / (jet_mass + GLOBAL_EPSILON)
                        ),
                    )
                )
            for k in EXCLUSIVE_RESOLUTIONS:
                for endpoint in (query, context):
                    node = int(assignments[k][endpoint])
                    row.extend(
                        (
                            math.log(
                                (node_pt[node] + GLOBAL_EPSILON)
                                / (jet_pt + GLOBAL_EPSILON)
                            ),
                            math.log(
                                (node_mass[node] + GLOBAL_EPSILON)
                                / (jet_mass + GLOBAL_EPSILON)
                            ),
                            float(multiplicity[node]) / int(tree["n_valid"]),
                        )
                    )
            for k in EXCLUSIVE_RESOLUTIONS:
                for endpoint in (query, context):
                    node = int(assignments[k][endpoint])
                    row.append(
                        float(tokens[endpoint, 0])
                        / (node_pt[node] + GLOBAL_EPSILON)
                    )
            for k in EXCLUSIVE_RESOLUTIONS:
                for endpoint in (query, context):
                    node = int(assignments[k][endpoint])
                    particle_eta = float(tokens[endpoint, 1])
                    particle_phi = float(tokens[endpoint, 2])
                    axis_eta, axis_phi = _eta_phi(node_vectors[node])
                    delta_phi = math.atan2(
                        math.sin(particle_phi - axis_phi),
                        math.cos(particle_phi - axis_phi),
                    )
                    row.append(
                        math.hypot(particle_eta - axis_eta, delta_phi)
                        / REFERENCE_RADIUS
                    )
            for k in EXCLUSIVE_RESOLUTIONS:
                row.append(
                    ranks[k][int(assignments[k][context])]
                    - ranks[k][int(assignments[k][query])]
                )
            if len(row) != REGION_RAW_DIMENSION:
                raise AssertionError("REGION row dimension drifted")
            output[:, query, context] = row
    return torch.from_numpy(output).to(
        device=raw_tokens.device, dtype=raw_tokens.dtype
    )


class RegionNormalizer(_ModuleBase):
    def __init__(self, artifact: Mapping[str, Any]) -> None:
        torch = require_torch()
        super().__init__()
        validate_content_hash(
            artifact, expected_contract=REGION_NORMALIZATION_CONTRACT
        )
        self.artifact_sha256 = str(artifact["content_hash"])
        lookup = {
            str(record["feature_name"]): record
            for record in artifact["records"]
        }
        center = np.zeros(REGION_RAW_DIMENSION, dtype=np.float32)
        scale = np.ones(REGION_RAW_DIMENSION, dtype=np.float32)
        robust = np.zeros(REGION_RAW_DIMENSION, dtype=bool)
        for index, name in enumerate(REGION_RAW_FEATURE_NAMES):
            if name not in REGION_ROBUST_FEATURE_NAMES:
                continue
            record = lookup[name]
            center[index] = float(record["median"])
            scale[index] = float(record["robust_scale"])
            robust[index] = True
        self.register_buffer("center", torch.from_numpy(center).view(1, -1, 1, 1))
        self.register_buffer("scale", torch.from_numpy(scale).view(1, -1, 1, 1))
        self.register_buffer("robust", torch.from_numpy(robust).view(1, -1, 1, 1))

    def forward(self, raw: Any, mask: Any) -> Any:
        safe = _torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        transformed = _torch.clamp(
            (safe - self.center.to(safe)) / self.scale.to(safe), -8.0, 8.0
        )
        output = _torch.where(self.robust, transformed, safe)
        length = int(raw.shape[-1])
        diagonal = _torch.eye(
            length, dtype=_torch.bool, device=raw.device
        ).view(1, 1, length, length)
        output[:, 4:8] = output[:, 4:8].masked_fill(diagonal, 0.0)
        return output.masked_fill(~valid_pair_mask(mask), 0.0)


class RegionEncoder(_ModuleBase):
    raw_feature_names = REGION_RAW_FEATURE_NAMES
    encoded_dimension = REGION_ENCODED_DIMENSION

    def __init__(self, normalization_artifact: Mapping[str, Any]) -> None:
        torch = require_torch()
        super().__init__()
        self.normalizer = RegionNormalizer(normalization_artifact)
        self.normalization_sha256 = self.normalizer.artifact_sha256
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(REGION_RAW_DIMENSION, 32),
            torch.nn.GELU(),
            torch.nn.RMSNorm(32, eps=GLOBAL_EPSILON),
            torch.nn.Linear(32, REGION_ENCODED_DIMENSION),
        )

    def forward(
        self,
        raw_tokens: Any,
        mask: Any,
        trees: Sequence[Mapping[str, Any]],
        *,
        return_details: bool = False,
    ) -> Any:
        torch = require_torch()
        if len(trees) != int(raw_tokens.shape[0]):
            raise ValueError("REGION tree batch size differs")
        raw = torch.stack(
            [
                build_region_raw_features(
                    trees[row], raw_tokens[row], mask[row, 0]
                )
                for row in range(int(raw_tokens.shape[0]))
            ]
        )
        normalized = self.normalizer(raw, mask)
        encoded = self.encoder(normalized.permute(0, 2, 3, 1))
        encoded = encoded.permute(0, 3, 1, 2).contiguous()
        encoded = encoded.masked_fill(~valid_pair_mask(mask), 0.0)
        if return_details:
            return {"raw": raw, "normalized": normalized, "encoded": encoded}
        return encoded


def build_region_relation_contract(
    *,
    relation_registry_sha256: str,
    region_normalization_sha256: str,
    angular_tree_resource_sha256: str,
) -> dict[str, Any]:
    return with_content_hash(
        {
            "contract": REGION_RELATION_CONTRACT,
            "schema_version": 1,
            "relation_registry_sha256": require_sha256(
                relation_registry_sha256, name="relation_registry_sha256"
            ),
            "region_normalization_sha256": require_sha256(
                region_normalization_sha256, name="region_normalization_sha256"
            ),
            "angular_tree_resource_sha256": require_sha256(
                angular_tree_resource_sha256,
                name="angular_tree_resource_sha256",
            ),
            "family_id": "REGION",
            "raw_feature_names": list(REGION_RAW_FEATURE_NAMES),
            "raw_feature_groups": {
                "same_cluster_indicators": 3,
                "lca_depth": 1,
                "lca_merge": 4,
                "endpoint_cluster_descriptors": 18,
                "within_cluster_particle_pt_fractions": 6,
                "endpoint_to_axis_distances": 6,
                "signed_rank_differences": 3,
            },
            "raw_dimension": 41,
            "encoder": [
                "Linear(41,32)", "GELU", "RMSNorm(32,eps=1e-6)",
                "Linear(32,12)",
            ],
            "encoded_dimension": 12,
            "diagonal_merge_policy": "zero_after_normalization_and_excluded_from_fit",
            "dropout": 0.0,
        }
    )


__all__ = [
    "REGION_AXIS_DISTANCE_NAMES",
    "REGION_ENCODED_DIMENSION",
    "REGION_ENDPOINT_DESCRIPTOR_NAMES",
    "REGION_LCA_NAMES",
    "REGION_NORMALIZATION_CONTRACT",
    "REGION_RANK_DIFFERENCE_NAMES",
    "REGION_RAW_DIMENSION",
    "REGION_RAW_FEATURE_NAMES",
    "REGION_RELATION_CONTRACT",
    "REGION_ROBUST_FEATURE_NAMES",
    "REGION_SAME_CLUSTER_NAMES",
    "REGION_WITHIN_CLUSTER_PT_NAMES",
    "RegionEncoder",
    "RegionNormalizer",
    "build_region_raw_features",
    "build_region_relation_contract",
]
