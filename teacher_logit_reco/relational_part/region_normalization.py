"""Train-only robust normalization for REGION tree features."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .contracts import require_sha256, validate_content_hash, with_content_hash
from .normalization import (
    GLOBAL_EPSILON,
    NORMALIZATION_QUANTILE_METHOD,
    _fit_records,
    _identity_key,
    _identity_sequence_hash,
    select_normalization_jet_indices,
    select_normalization_pairs,
    validate_relation_normalization_artifact,
)
from .relation_region import (
    REGION_AXIS_DISTANCE_NAMES,
    REGION_ENDPOINT_DESCRIPTOR_NAMES,
    REGION_LCA_NAMES,
    REGION_NORMALIZATION_CONTRACT,
    REGION_RANK_DIFFERENCE_NAMES,
    REGION_RAW_FEATURE_NAMES,
    REGION_ROBUST_FEATURE_NAMES,
    REGION_WITHIN_CLUSTER_PT_NAMES,
)
from .region_tree import EXCLUSIVE_RESOLUTIONS, REFERENCE_RADIUS, validate_tree


_REGION_PAIR_FEATURE_NAMES = (
    REGION_LCA_NAMES[0],
    *REGION_RANK_DIFFERENCE_NAMES,
)
_REGION_MERGE_FEATURE_NAMES = REGION_LCA_NAMES[1:]
_REGION_NODE_FEATURE_NAMES = (
    *REGION_ENDPOINT_DESCRIPTOR_NAMES,
    *REGION_WITHIN_CLUSTER_PT_NAMES,
    *REGION_AXIS_DISTANCE_NAMES,
)
_REGION_DOMAIN_FEATURE_NAMES = {
    "REGION_pair": _REGION_PAIR_FEATURE_NAMES,
    "REGION_merge": _REGION_MERGE_FEATURE_NAMES,
    "REGION_node": _REGION_NODE_FEATURE_NAMES,
}
_REGION_FEATURE_DOMAIN = {
    name: domain
    for domain, names in _REGION_DOMAIN_FEATURE_NAMES.items()
    for name in names
}
if set(_REGION_FEATURE_DOMAIN) != set(REGION_ROBUST_FEATURE_NAMES):
    raise AssertionError("REGION normalization domains must cover robust features")


def _region_row_sampler(
    tree: Mapping[str, Any],
    tokens: np.ndarray,
    valid_indices: Sequence[int],
):
    """Return a direct O(tree-depth) sampled-pair feature function."""

    parent = np.asarray(tree["parent"])
    depth = np.asarray(tree["depth"])
    leaf_to_node = np.asarray(tree["leaf_to_node"])
    node_vectors = np.asarray(tree["vectors"], dtype=np.float64)
    node_pt = np.asarray(tree["pt"], dtype=np.float64)
    node_mass = np.asarray(tree["mass"], dtype=np.float64)
    multiplicity = np.asarray(tree["multiplicity"], dtype=np.float64)
    root = int(tree["root"])
    jet_pt = float(node_pt[root])
    jet_mass = float(node_mass[root])
    maximum_leaf_depth = max(
        (int(depth[int(leaf_to_node[index])]) for index in valid_indices),
        default=1,
    )
    assignments = {
        resolution: np.asarray(tree["assignments"][str(resolution)])
        for resolution in EXCLUSIVE_RESOLUTIONS
    }
    ranks: dict[int, dict[int, float]] = {}
    axes: dict[int, tuple[float, float]] = {}
    for resolution in EXCLUSIVE_RESOLUTIONS:
        nodes = tuple(
            sorted(
                {
                    int(assignments[resolution][index])
                    for index in valid_indices
                }
            )
        )
        values = np.asarray([node_pt[node] for node in nodes])
        denominator = max(len(nodes) - 1, 1)
        ranks[resolution] = {
            node: (
                int(np.sum(values > values[position]))
                + 0.5 * (int(np.sum(values == values[position])) - 1)
            )
            / denominator
            for position, node in enumerate(nodes)
        }
        for node in nodes:
            px, py, pz, _ = map(float, node_vectors[node])
            pt = math.hypot(px, py)
            axes[node] = (
                math.asinh(pz / max(pt, 1.0e-300)),
                math.atan2(py, px),
            )

    def lca(left: int, right: int) -> int:
        first = int(leaf_to_node[left])
        second = int(leaf_to_node[right])
        while depth[first] > depth[second]:
            first = int(parent[first])
        while depth[second] > depth[first]:
            second = int(parent[second])
        while first != second:
            first = int(parent[first])
            second = int(parent[second])
        return first

    def sample(query: int, context: int) -> np.ndarray:
        row: list[float] = [
            float(
                assignments[resolution][query]
                == assignments[resolution][context]
            )
            for resolution in EXCLUSIVE_RESOLUTIONS
        ]
        ancestor = lca(query, context)
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
        for resolution in EXCLUSIVE_RESOLUTIONS:
            for endpoint in (query, context):
                node = int(assignments[resolution][endpoint])
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
        for resolution in EXCLUSIVE_RESOLUTIONS:
            for endpoint in (query, context):
                node = int(assignments[resolution][endpoint])
                row.append(
                    float(tokens[endpoint, 0])
                    / (node_pt[node] + GLOBAL_EPSILON)
                )
        for resolution in EXCLUSIVE_RESOLUTIONS:
            for endpoint in (query, context):
                node = int(assignments[resolution][endpoint])
                axis_eta, axis_phi = axes[node]
                delta_phi = math.atan2(
                    math.sin(float(tokens[endpoint, 2]) - axis_phi),
                    math.cos(float(tokens[endpoint, 2]) - axis_phi),
                )
                row.append(
                    math.hypot(
                        float(tokens[endpoint, 1]) - axis_eta, delta_phi
                    )
                    / REFERENCE_RADIUS
                )
        for resolution in EXCLUSIVE_RESOLUTIONS:
            row.append(
                ranks[resolution][
                    int(assignments[resolution][context])
                ]
                - ranks[resolution][
                    int(assignments[resolution][query])
                ]
            )
        if len(row) != len(REGION_RAW_FEATURE_NAMES):
            raise AssertionError("REGION sampled row dimension drifted")
        return np.asarray(row, dtype=np.float64)

    return sample


class _RegionDomainSampler:
    """Compute only the robust feature domains used during normalization."""

    def __init__(
        self,
        tree: Mapping[str, Any],
        tokens: np.ndarray,
        valid_indices: Sequence[int],
    ) -> None:
        self.tree = tree
        self.tokens = tokens
        self.parent = np.asarray(tree["parent"])
        self.depth = np.asarray(tree["depth"])
        self.leaf_to_node = np.asarray(tree["leaf_to_node"])
        self.node_vectors = np.asarray(tree["vectors"], dtype=np.float64)
        self.node_pt = np.asarray(tree["pt"], dtype=np.float64)
        self.node_mass = np.asarray(tree["mass"], dtype=np.float64)
        self.multiplicity = np.asarray(
            tree["multiplicity"], dtype=np.float64
        )
        self.root = int(tree["root"])
        self.jet_pt = float(self.node_pt[self.root])
        self.jet_mass = float(self.node_mass[self.root])
        self.maximum_leaf_depth = max(
            (
                int(self.depth[int(self.leaf_to_node[index])])
                for index in valid_indices
            ),
            default=1,
        )
        self.assignments = {
            resolution: np.asarray(
                tree["assignments"][str(resolution)]
            )
            for resolution in EXCLUSIVE_RESOLUTIONS
        }
        self.ranks: dict[int, dict[int, float]] = {}
        self.axes: dict[int, tuple[float, float]] = {}
        for resolution in EXCLUSIVE_RESOLUTIONS:
            nodes = tuple(
                sorted(
                    {
                        int(self.assignments[resolution][index])
                        for index in valid_indices
                    }
                )
            )
            values = np.asarray([self.node_pt[node] for node in nodes])
            denominator = max(len(nodes) - 1, 1)
            self.ranks[resolution] = {
                node: (
                    int(np.sum(values > values[position]))
                    + 0.5
                    * (int(np.sum(values == values[position])) - 1)
                )
                / denominator
                for position, node in enumerate(nodes)
            }
            for node in nodes:
                px, py, pz, _ = map(float, self.node_vectors[node])
                pt = math.hypot(px, py)
                self.axes[node] = (
                    math.asinh(pz / max(pt, 1.0e-300)),
                    math.atan2(py, px),
                )

    def _lca(self, left: int, right: int) -> int:
        first = int(self.leaf_to_node[left])
        second = int(self.leaf_to_node[right])
        while self.depth[first] > self.depth[second]:
            first = int(self.parent[first])
        while self.depth[second] > self.depth[first]:
            second = int(self.parent[second])
        while first != second:
            first = int(self.parent[first])
            second = int(self.parent[second])
        return first

    def pair_and_merge(
        self,
        query: int,
        context: int,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        ancestor = self._lca(query, context)
        pair = np.asarray(
            [
                float(self.depth[ancestor])
                / max(self.maximum_leaf_depth, 1),
                *(
                    self.ranks[resolution][
                        int(self.assignments[resolution][context])
                    ]
                    - self.ranks[resolution][
                        int(self.assignments[resolution][query])
                    ]
                    for resolution in EXCLUSIVE_RESOLUTIONS
                ),
            ],
            dtype=np.float64,
        )
        if query == context:
            return pair, None
        merge = np.asarray(
            [
                math.log(
                    float(self.tree["merge_delta_r"][ancestor])
                    / REFERENCE_RADIUS
                    + GLOBAL_EPSILON
                ),
                math.log(
                    float(self.tree["merge_kt"][ancestor])
                    / (self.jet_pt * REFERENCE_RADIUS + GLOBAL_EPSILON)
                    + GLOBAL_EPSILON
                ),
                float(self.tree["merge_z"][ancestor]),
                math.log(
                    (
                        float(self.tree["merge_mass"][ancestor])
                        + GLOBAL_EPSILON
                    )
                    / (self.jet_mass + GLOBAL_EPSILON)
                ),
            ],
            dtype=np.float64,
        )
        return pair, merge

    def node(self, index: int) -> np.ndarray:
        row: list[float] = []
        for resolution in EXCLUSIVE_RESOLUTIONS:
            node = int(self.assignments[resolution][index])
            values = (
                math.log(
                    (self.node_pt[node] + GLOBAL_EPSILON)
                    / (self.jet_pt + GLOBAL_EPSILON)
                ),
                math.log(
                    (self.node_mass[node] + GLOBAL_EPSILON)
                    / (self.jet_mass + GLOBAL_EPSILON)
                ),
                float(self.multiplicity[node]) / int(self.tree["n_valid"]),
            )
            row.extend(values)
            row.extend(values)
        for resolution in EXCLUSIVE_RESOLUTIONS:
            node = int(self.assignments[resolution][index])
            value = float(self.tokens[index, 0]) / (
                self.node_pt[node] + GLOBAL_EPSILON
            )
            row.extend((value, value))
        for resolution in EXCLUSIVE_RESOLUTIONS:
            node = int(self.assignments[resolution][index])
            axis_eta, axis_phi = self.axes[node]
            delta_phi = math.atan2(
                math.sin(float(self.tokens[index, 2]) - axis_phi),
                math.cos(float(self.tokens[index, 2]) - axis_phi),
            )
            value = (
                math.hypot(
                    float(self.tokens[index, 1]) - axis_eta,
                    delta_phi,
                )
                / REFERENCE_RADIUS
            )
            row.extend((value, value))
        if len(row) != len(_REGION_NODE_FEATURE_NAMES):
            raise AssertionError("REGION node sample dimension drifted")
        return np.asarray(row, dtype=np.float64)


def _update_identity_digest(digest: Any, value: str) -> None:
    digest.update(value.encode("utf-8"))
    digest.update(b"\n")


def _collect_region_domain_samples(
    array: np.ndarray,
    valid: np.ndarray,
    identities: Sequence[Any],
    trees: Sequence[Mapping[str, Any]],
    selected: np.ndarray,
    *,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    progress_interval: int = 500,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    if int(progress_interval) <= 0:
        raise ValueError("REGION progress interval must be positive")
    chunks: dict[str, list[np.ndarray]] = {
        domain: [] for domain in _REGION_DOMAIN_FEATURE_NAMES
    }
    digests = {
        domain: hashlib.sha256()
        for domain in _REGION_DOMAIN_FEATURE_NAMES
    }
    counts = {domain: 0 for domain in _REGION_DOMAIN_FEATURE_NAMES}
    total = int(selected.size)
    for position, row_value in enumerate(selected, start=1):
        row = int(row_value)
        validate_tree(trees[row])
        identity = _identity_key(identities[row])
        valid_indices = np.flatnonzero(valid[row]).tolist()
        pairs = select_normalization_pairs(identities[row], valid_indices)
        sampler = _RegionDomainSampler(
            trees[row], array[row], valid_indices
        )

        pair_rows: list[np.ndarray] = []
        merge_rows: list[np.ndarray] = []
        for query, context in pairs:
            pair_values, merge_values = sampler.pair_and_merge(
                query, context
            )
            pair_rows.append(pair_values)
            _update_identity_digest(
                digests["REGION_pair"],
                f"{identity}#REGION_pair:{query}>{context}",
            )
            if merge_values is not None:
                merge_rows.append(merge_values)
                _update_identity_digest(
                    digests["REGION_merge"],
                    f"{identity}#REGION_merge:{query}>{context}",
                )
        if pair_rows:
            pair_chunk = np.stack(pair_rows)
            chunks["REGION_pair"].append(pair_chunk)
            counts["REGION_pair"] += int(pair_chunk.shape[0])
        if merge_rows:
            merge_chunk = np.stack(merge_rows)
            chunks["REGION_merge"].append(merge_chunk)
            counts["REGION_merge"] += int(merge_chunk.shape[0])

        if valid_indices:
            node_chunk = np.stack(
                [sampler.node(index) for index in valid_indices]
            )
            chunks["REGION_node"].append(node_chunk)
            counts["REGION_node"] += int(node_chunk.shape[0])
            for index in valid_indices:
                _update_identity_digest(
                    digests["REGION_node"],
                    f"{identity}#REGION_node:{index}>{index}",
                )

        if (
            progress_callback is not None
            and (
                position == total
                or position % int(progress_interval) == 0
            )
        ):
            progress_callback(
                {
                    "stage": "fit_region_normalization",
                    "processed_jets": position,
                    "total_jets": total,
                    "fraction_complete": (
                        float(position) / total if total else 1.0
                    ),
                    "sample_counts": dict(counts),
                }
            )

    samples = {}
    hashes = {}
    for domain, names in _REGION_DOMAIN_FEATURE_NAMES.items():
        if not chunks[domain]:
            raise ValueError(f"{domain} has no fit samples")
        samples[domain] = np.concatenate(chunks[domain], axis=0)
        if samples[domain].shape != (counts[domain], len(names)):
            raise AssertionError(f"{domain} sample matrix shape drifted")
        hashes[domain] = digests[domain].hexdigest()
    return samples, hashes


def fit_region_normalization(
    raw_tokens: np.ndarray,
    mask: np.ndarray,
    identities: Sequence[Any],
    trees: Sequence[Mapping[str, Any]],
    *,
    relation_normalization_artifact: Mapping[str, Any],
    angular_tree_resource_sha256: str,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    progress_interval: int = 500,
) -> dict[str, Any]:
    parent_sha = validate_relation_normalization_artifact(
        relation_normalization_artifact
    )
    array = np.asarray(raw_tokens)
    valid = np.asarray(mask)
    if array.dtype != np.float32 or valid.dtype != np.bool_:
        raise TypeError("REGION fit requires float32 tokens and bool masks")
    if array.ndim != 3 or array.shape[2] != 14 or valid.shape != array.shape[:2]:
        raise ValueError("REGION fit input shapes differ")
    if len(identities) != len(trees) or len(trees) != int(array.shape[0]):
        raise ValueError("REGION tree/identity counts differ")
    selected = select_normalization_jet_indices(identities)
    domain_samples, domain_hashes = _collect_region_domain_samples(
        array,
        valid,
        identities,
        trees,
        selected,
        progress_callback=progress_callback,
        progress_interval=progress_interval,
    )
    records_by_name = {}
    for domain, names in _REGION_DOMAIN_FEATURE_NAMES.items():
        for record in _fit_records(
            domain_samples[domain],
            family_id="REGION",
            feature_names=names,
            applicability_rule_id=domain,
        ):
            records_by_name[str(record["feature_name"])] = record
    records = [
        records_by_name[name] for name in REGION_ROBUST_FEATURE_NAMES
    ]
    if len(records_by_name) != len(records):
        raise AssertionError("REGION normalization record coverage drifted")
    artifact = with_content_hash(
        {
            "contract": REGION_NORMALIZATION_CONTRACT,
            "schema_version": 1,
            "fit_split": "model_train",
            "relation_normalization_sha256": parent_sha,
            "relation_registry_sha256": relation_normalization_artifact[
                "relation_registry_sha256"
            ],
            "angular_tree_resource_sha256": require_sha256(
                angular_tree_resource_sha256,
                name="angular_tree_resource_sha256",
            ),
            "selected_jet_count": int(selected.size),
            "selected_jet_identity_sha256": _identity_sequence_hash(
                [_identity_key(identities[int(index)]) for index in selected]
            ),
            "quantile_method": NORMALIZATION_QUANTILE_METHOD,
            "float_accumulation_dtype": "float64",
            "records": records,
            "feature_sample_identity_sha256": {
                name: domain_hashes[_REGION_FEATURE_DOMAIN[name]]
                for name in REGION_ROBUST_FEATURE_NAMES
            },
        }
    )
    validate_region_normalization(artifact)
    return artifact


def validate_region_normalization(
    artifact: Mapping[str, Any],
    *,
    relation_normalization_sha256: str | None = None,
    angular_tree_resource_sha256: str | None = None,
) -> str:
    digest = validate_content_hash(
        artifact, expected_contract=REGION_NORMALIZATION_CONTRACT
    )
    if artifact.get("fit_split") != "model_train":
        raise ValueError("REGION normalizer was not fitted on model_train")
    for field in (
        "relation_normalization_sha256",
        "relation_registry_sha256",
        "angular_tree_resource_sha256",
        "selected_jet_identity_sha256",
    ):
        require_sha256(artifact.get(field), name=field)
    if relation_normalization_sha256 is not None and artifact.get(
        "relation_normalization_sha256"
    ) != require_sha256(
        relation_normalization_sha256, name="relation_normalization_sha256"
    ):
        raise ValueError("REGION normalizer belongs to another base normalizer")
    if angular_tree_resource_sha256 is not None and artifact.get(
        "angular_tree_resource_sha256"
    ) != require_sha256(
        angular_tree_resource_sha256, name="angular_tree_resource_sha256"
    ):
        raise ValueError("REGION normalizer belongs to another tree resource")
    records = artifact.get("records")
    if not isinstance(records, list):
        raise ValueError("REGION normalizer records must be a list")
    actual = {str(record.get("feature_name")) for record in records}
    if actual != set(REGION_ROBUST_FEATURE_NAMES) or len(records) != len(actual):
        raise ValueError("REGION normalizer feature coverage differs")
    for record in records:
        numeric = [
            float(record.get(field))
            for field in (
                "median", "q25", "q75", "robust_scale",
                "applicable_zero_fraction", "post_normalization_clip_fraction",
            )
        ]
        if (
            record.get("family_id") != "REGION"
            or int(record.get("applicable_count", 0)) < 1
            or not np.isfinite(numeric).all()
            or float(record["robust_scale"]) < GLOBAL_EPSILON
        ):
            raise ValueError("REGION normalizer record differs")
    hashes = artifact.get("feature_sample_identity_sha256")
    if not isinstance(hashes, Mapping) or set(hashes) != set(
        REGION_ROBUST_FEATURE_NAMES
    ):
        raise ValueError("REGION sample hashes differ")
    for name, value in hashes.items():
        require_sha256(value, name=f"feature_sample_identity_sha256.{name}")
    return digest


__all__ = [
    "fit_region_normalization",
    "validate_region_normalization",
]
