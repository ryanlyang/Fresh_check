"""Beam-free exclusive angular-tree REGION relation family."""

from __future__ import annotations

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


def _pack_tree_batch(
    trees: Sequence[Mapping[str, Any]],
    *,
    particles: int,
    device: Any,
) -> dict[str, Any]:
    """Pack compact CPU sidecars into O(BN) tensors on the model device."""

    torch = require_torch()
    batch = len(trees)
    maximum_nodes = max(1, 2 * particles - 1)
    integer_fields = (
        "parent", "depth", "multiplicity", "leaf_to_node",
    )
    float_fields = (
        "pt", "mass", "merge_delta_r", "merge_kt", "merge_z", "merge_mass",
    )
    packed: dict[str, Any] = {
        name: torch.zeros(
            (batch, particles if name == "leaf_to_node" else maximum_nodes),
            dtype=torch.long,
            device=device,
        )
        for name in integer_fields
    }
    packed.update(
        {
            name: torch.zeros(
                (batch, maximum_nodes), dtype=torch.float32, device=device
            )
            for name in float_fields
        }
    )
    packed["vectors"] = torch.zeros(
        (batch, maximum_nodes, 4), dtype=torch.float32, device=device
    )
    packed["assignments"] = torch.zeros(
        (batch, len(EXCLUSIVE_RESOLUTIONS), particles),
        dtype=torch.long,
        device=device,
    )
    packed["actual_cluster_counts"] = torch.zeros(
        (batch, len(EXCLUSIVE_RESOLUTIONS)),
        dtype=torch.long,
        device=device,
    )
    packed["roots"] = torch.zeros(batch, dtype=torch.long, device=device)
    packed["n_valid"] = torch.zeros(batch, dtype=torch.long, device=device)
    packed["n_nodes"] = torch.zeros(batch, dtype=torch.long, device=device)
    for row, tree in enumerate(trees):
        validate_tree(tree)
        if int(tree["n_particles"]) != particles:
            raise ValueError("REGION tree particle dimension differs")
        nodes = int(tree["n_nodes"])
        packed["n_valid"][row] = int(tree["n_valid"])
        packed["n_nodes"][row] = nodes
        if nodes == 0:
            continue
        root = int(tree["root"])
        packed["roots"][row] = root
        for name in ("parent", "depth", "multiplicity"):
            values = torch.as_tensor(
                np.asarray(tree[name]), dtype=torch.long, device=device
            )
            packed[name][row, :nodes] = values
        # A root has persisted parent -1.  Make it its own binary-lifting
        # ancestor; padded nodes also remain self-safe at zero.
        packed["parent"][row, root] = root
        packed["leaf_to_node"][row] = torch.as_tensor(
            np.asarray(tree["leaf_to_node"]).clip(min=0),
            dtype=torch.long,
            device=device,
        )
        for name in float_fields:
            packed[name][row, :nodes] = torch.as_tensor(
                np.asarray(tree[name]), dtype=torch.float32, device=device
            )
        packed["vectors"][row, :nodes] = torch.as_tensor(
            np.asarray(tree["vectors"]), dtype=torch.float32, device=device
        )
        for index, resolution in enumerate(EXCLUSIVE_RESOLUTIONS):
            packed["assignments"][row, index] = torch.as_tensor(
                np.asarray(tree["assignments"][str(resolution)]).clip(min=0),
                dtype=torch.long,
                device=device,
            )
            packed["actual_cluster_counts"][row, index] = int(
                tree["actual_cluster_counts"][str(resolution)]
            )
    return packed


def _node_values(values: Any, indices: Any) -> Any:
    torch = require_torch()
    batch = torch.arange(
        int(values.shape[0]), device=values.device
    ).view(-1, *([1] * (indices.ndim - 1)))
    return values[batch, indices]


def _batched_lca(packed: Mapping[str, Any]) -> Any:
    """Return [B,N,N] LCAs using a batched binary-lifting jump table."""

    torch = require_torch()
    leaves = packed["leaf_to_node"]
    parent = packed["parent"]
    depth = packed["depth"]
    maximum_nodes = int(parent.shape[1])
    levels = max(1, (maximum_nodes - 1).bit_length())
    jumps = [parent]
    for _ in range(1, levels):
        jumps.append(_node_values(jumps[-1], jumps[-1]))
    left = leaves.unsqueeze(-1).expand(-1, -1, int(leaves.shape[1])).clone()
    right = leaves.unsqueeze(-2).expand(-1, int(leaves.shape[1]), -1).clone()
    difference = _node_values(depth, left) - _node_values(depth, right)
    for level in range(levels):
        bit = 1 << level
        left = torch.where(
            difference.ge(bit), _node_values(jumps[level], left), left
        )
        right = torch.where(
            difference.le(-bit), _node_values(jumps[level], right), right
        )
    for level in range(levels - 1, -1, -1):
        left_up = _node_values(jumps[level], left)
        right_up = _node_values(jumps[level], right)
        unequal = left_up.ne(right_up)
        left = torch.where(unequal, left_up, left)
        right = torch.where(unequal, right_up, right)
    return torch.where(left.eq(right), left, _node_values(parent, left))


def _leaf_cluster_ranks(
    packed: Mapping[str, Any],
    resolution_index: int,
    valid: Any,
) -> Any:
    """Average tied descending cluster-pT rank mapped back to every leaf."""

    torch = require_torch()
    assignments = packed["assignments"][:, resolution_index]
    requested = EXCLUSIVE_RESOLUTIONS[resolution_index]
    batch = int(assignments.shape[0])
    cluster_nodes = torch.zeros(
        (batch, requested), dtype=torch.long, device=assignments.device
    )
    # Packing the at-most-eight unique compact node IDs is O(BK), not O(BN²).
    for row in range(batch):
        count = int(packed["actual_cluster_counts"][row, resolution_index])
        if count:
            cluster_nodes[row, :count] = torch.unique(
                assignments[row][valid[row]],
                sorted=True,
            )[:count]
    counts = packed["actual_cluster_counts"][:, resolution_index]
    valid_cluster = (
        torch.arange(requested, device=assignments.device).unsqueeze(0)
        < counts.unsqueeze(1)
    )
    cluster_pt = _node_values(packed["pt"], cluster_nodes)
    greater = (
        cluster_pt.unsqueeze(1) > cluster_pt.unsqueeze(2)
    ) & valid_cluster.unsqueeze(1) & valid_cluster.unsqueeze(2)
    equal = (
        cluster_pt.unsqueeze(1) == cluster_pt.unsqueeze(2)
    ) & valid_cluster.unsqueeze(1) & valid_cluster.unsqueeze(2)
    ranks = (
        greater.sum(-1).float() + 0.5 * (equal.sum(-1).float() - 1.0)
    ) / (counts - 1).clamp_min(1).unsqueeze(1).float()
    ranks = ranks.masked_fill(~valid_cluster, 0.0)
    node_ranks = torch.zeros_like(packed["pt"])
    for row in range(batch):
        count = int(counts[row])
        if count:
            node_ranks[row, cluster_nodes[row, :count]] = ranks[row, :count]
    return _node_values(node_ranks, assignments)


def build_batched_region_raw_features(
    trees: Sequence[Mapping[str, Any]],
    raw_tokens: Any,
    mask: Any,
) -> Any:
    """Materialize all REGION pairs on-device with batched binary lifting."""

    torch = require_torch()
    if raw_tokens.ndim != 3 or int(raw_tokens.shape[2]) != 14:
        raise ValueError("REGION raw tokens must have shape [B,N,14]")
    batch, particles = map(int, raw_tokens.shape[:2])
    if tuple(mask.shape) == (batch, 1, particles):
        valid = mask[:, 0].bool()
    elif tuple(mask.shape) == (batch, particles):
        valid = mask.bool()
    else:
        raise ValueError("REGION mask must have shape [B,1,N] or [B,N]")
    if len(trees) != batch:
        raise ValueError("REGION tree batch size differs")
    packed = _pack_tree_batch(
        trees, particles=particles, device=raw_tokens.device
    )
    lca = _batched_lca(packed)
    pair_valid = valid.unsqueeze(-1) & valid.unsqueeze(-2)
    diagonal = torch.eye(
        particles, dtype=torch.bool, device=raw_tokens.device
    ).unsqueeze(0)
    roots = packed["roots"]
    jet_pt = _node_values(packed["pt"], roots).view(batch, 1, 1)
    jet_mass = _node_values(packed["mass"], roots).view(batch, 1, 1)
    leaf_depth = _node_values(packed["depth"], packed["leaf_to_node"])
    maximum_leaf_depth = leaf_depth.masked_fill(~valid, 0).amax(
        dim=1
    ).clamp_min(1).view(batch, 1, 1)
    channels: list[Any] = []
    for index in range(len(EXCLUSIVE_RESOLUTIONS)):
        assignment = packed["assignments"][:, index]
        channels.append(
            assignment.unsqueeze(-1).eq(assignment.unsqueeze(-2)).float()
        )
    channels.append(
        _node_values(packed["depth"], lca).float()
        / maximum_leaf_depth.float()
    )
    merge_values = (
        (
            _node_values(packed["merge_delta_r"], lca) / REFERENCE_RADIUS
            + GLOBAL_EPSILON
        ).log(),
        (
            _node_values(packed["merge_kt"], lca)
            / (jet_pt * REFERENCE_RADIUS + GLOBAL_EPSILON)
            + GLOBAL_EPSILON
        ).log(),
        _node_values(packed["merge_z"], lca),
        (
            (_node_values(packed["merge_mass"], lca) + GLOBAL_EPSILON)
            / (jet_mass + GLOBAL_EPSILON)
        ).log(),
    )
    channels.extend(value.masked_fill(diagonal, 0.0) for value in merge_values)

    token_pt = raw_tokens[:, :, 0].float()
    particle_eta = raw_tokens[:, :, 1].float()
    particle_phi = raw_tokens[:, :, 2].float()
    leaf_ranks = []
    endpoint_nodes = []
    for index in range(len(EXCLUSIVE_RESOLUTIONS)):
        nodes = packed["assignments"][:, index]
        endpoint_nodes.append(nodes)
        leaf_ranks.append(_leaf_cluster_ranks(packed, index, valid))
        for endpoint in ("query", "context"):
            values = (
                (_node_values(packed["pt"], nodes) + GLOBAL_EPSILON)
                / (jet_pt[:, :, 0] + GLOBAL_EPSILON)
            ).log()
            masses = (
                (_node_values(packed["mass"], nodes) + GLOBAL_EPSILON)
                / (jet_mass[:, :, 0] + GLOBAL_EPSILON)
            ).log()
            fractions = (
                _node_values(packed["multiplicity"], nodes).float()
                / packed["n_valid"].clamp_min(1).unsqueeze(1).float()
            )
            expand = (
                lambda value: value.unsqueeze(-1).expand(-1, -1, particles)
                if endpoint == "query"
                else value.unsqueeze(-2).expand(-1, particles, -1)
            )
            channels.extend((expand(values), expand(masses), expand(fractions)))
    for nodes in endpoint_nodes:
        cluster_pt = _node_values(packed["pt"], nodes)
        for endpoint in ("query", "context"):
            value = token_pt / (cluster_pt + GLOBAL_EPSILON)
            channels.append(
                value.unsqueeze(-1).expand(-1, -1, particles)
                if endpoint == "query"
                else value.unsqueeze(-2).expand(-1, particles, -1)
            )
    for nodes in endpoint_nodes:
        vectors = _node_values(packed["vectors"], nodes)
        axis_pt = torch.hypot(vectors[..., 0], vectors[..., 1])
        axis_eta = torch.asinh(
            vectors[..., 2] / axis_pt.clamp_min(1.0e-30)
        )
        axis_phi = torch.atan2(vectors[..., 1], vectors[..., 0])
        delta_phi = torch.atan2(
            torch.sin(particle_phi - axis_phi),
            torch.cos(particle_phi - axis_phi),
        )
        distance = torch.hypot(
            particle_eta - axis_eta, delta_phi
        ) / REFERENCE_RADIUS
        channels.extend(
            (
                distance.unsqueeze(-1).expand(-1, -1, particles),
                distance.unsqueeze(-2).expand(-1, particles, -1),
            )
        )
    for ranks in leaf_ranks:
        channels.append(ranks.unsqueeze(-2) - ranks.unsqueeze(-1))
    if len(channels) != REGION_RAW_DIMENSION:
        raise AssertionError("REGION feature channel count drifted")
    output = torch.stack(channels, dim=1)
    output = output.masked_fill(~pair_valid.unsqueeze(1), 0.0)
    return output.to(dtype=raw_tokens.dtype)


def build_batched_region_leaf_ranks(
    trees: Sequence[Mapping[str, Any]],
    mask: Any,
) -> Any:
    """Return absolute cluster-pT ranks [B,3,N] entirely on-device."""

    if mask.ndim != 3 or int(mask.shape[1]) != 1:
        raise ValueError("REGION rank mask must have shape [B,1,N]")
    batch, particles = int(mask.shape[0]), int(mask.shape[2])
    if len(trees) != batch:
        raise ValueError("REGION tree batch size differs")
    packed = _pack_tree_batch(
        trees, particles=particles, device=mask.device
    )
    valid = mask[:, 0].bool()
    return require_torch().stack(
        [
            _leaf_cluster_ranks(packed, index, valid)
            for index in range(len(EXCLUSIVE_RESOLUTIONS))
        ],
        dim=1,
    ).masked_fill(~valid.unsqueeze(1), 0.0)


def build_region_raw_features(
    tree: Mapping[str, Any],
    raw_tokens: Any,
    mask: Any,
) -> Any:
    """Materialize the transient 41-channel directed relation."""

    if raw_tokens.ndim != 2 or tuple(raw_tokens.shape) != (
        int(tree["n_particles"]), 14
    ):
        raise ValueError("REGION raw tokens disagree with the tree")
    if tuple(mask.shape) != (int(tree["n_particles"]),):
        raise ValueError("REGION mask disagrees with the tree")
    return build_batched_region_raw_features(
        [tree], raw_tokens.unsqueeze(0), mask.unsqueeze(0)
    )[0]


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
        disabled_resolutions: Sequence[int] = (),
    ) -> Any:
        torch = require_torch()
        if len(trees) != int(raw_tokens.shape[0]):
            raise ValueError("REGION tree batch size differs")
        raw = build_batched_region_raw_features(trees, raw_tokens, mask)
        disabled = tuple(sorted({int(value) for value in disabled_resolutions}))
        if any(value not in EXCLUSIVE_RESOLUTIONS for value in disabled):
            raise ValueError("REGION ablation resolution must be K=2, K=4, or K=8")
        if disabled:
            raw = raw.clone()
            for resolution in disabled:
                index = EXCLUSIVE_RESOLUTIONS.index(resolution)
                resolution_channels = (
                    index,
                    *range(8 + index * 6, 14 + index * 6),
                    *range(26 + index * 2, 28 + index * 2),
                    *range(32 + index * 2, 34 + index * 2),
                    38 + index,
                )
                raw[:, resolution_channels] = 0.0
        normalized = self.normalizer(raw, mask)
        encoded = self.encoder(normalized.permute(0, 2, 3, 1))
        encoded = encoded.permute(0, 3, 1, 2).contiguous()
        encoded = encoded.masked_fill(~valid_pair_mask(mask), 0.0)
        if return_details:
            return {
                "raw": raw,
                "normalized": normalized,
                "encoded": encoded,
                "disabled_resolutions": list(disabled),
            }
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
    "build_batched_region_raw_features",
    "build_batched_region_leaf_ranks",
    "build_region_raw_features",
    "build_region_relation_contract",
]
