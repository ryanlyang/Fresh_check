"""Train-only robust normalization for REGION tree features."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

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
    REGION_RAW_FEATURE_NAMES,
    REGION_ROBUST_FEATURE_NAMES,
    REGION_WITHIN_CLUSTER_PT_NAMES,
    build_region_raw_features,
)
from .region_tree import validate_tree

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def fit_region_normalization(
    raw_tokens: np.ndarray,
    mask: np.ndarray,
    identities: Sequence[Any],
    trees: Sequence[Mapping[str, Any]],
    *,
    relation_normalization_artifact: Mapping[str, Any],
    angular_tree_resource_sha256: str,
) -> dict[str, Any]:
    if torch is None:
        raise ImportError("REGION normalization requires PyTorch")
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
    samples: dict[str, list[float]] = {
        name: [] for name in REGION_ROBUST_FEATURE_NAMES
    }
    sample_keys: dict[str, list[str]] = {
        name: [] for name in REGION_ROBUST_FEATURE_NAMES
    }
    for row_value in selected:
        row = int(row_value)
        validate_tree(trees[row])
        identity = _identity_key(identities[row])
        valid_indices = np.flatnonzero(valid[row]).tolist()
        pairs = select_normalization_pairs(identities[row], valid_indices)
        raw = build_region_raw_features(
            trees[row],
            torch.from_numpy(array[row]),
            torch.from_numpy(valid[row]),
        ).numpy()
        for feature_index, name in enumerate(REGION_RAW_FEATURE_NAMES):
            if name not in samples:
                continue
            if name in REGION_LCA_NAMES[1:]:
                domain = [
                    (query, context)
                    for query, context in pairs
                    if query != context
                ]
                applicability = "REGION_merge"
            elif name in (
                *REGION_ENDPOINT_DESCRIPTOR_NAMES,
                *REGION_WITHIN_CLUSTER_PT_NAMES,
                *REGION_AXIS_DISTANCE_NAMES,
            ):
                domain = [(index, index) for index in valid_indices]
                applicability = "REGION_node"
            else:
                domain = list(pairs)
                applicability = "REGION_pair"
            for query, context in domain:
                samples[name].append(float(raw[feature_index, query, context]))
                sample_keys[name].append(
                    f"{identity}#{applicability}:{query}>{context}"
                )
    records: list[dict[str, Any]] = []
    for name in REGION_ROBUST_FEATURE_NAMES:
        if not samples[name]:
            raise ValueError(f"REGION feature {name} has no fit samples")
        if name in REGION_LCA_NAMES[1:]:
            applicability = "REGION_merge"
        elif name in (
            *REGION_ENDPOINT_DESCRIPTOR_NAMES,
            *REGION_WITHIN_CLUSTER_PT_NAMES,
            *REGION_AXIS_DISTANCE_NAMES,
        ):
            applicability = "REGION_node"
        else:
            applicability = "REGION_pair"
        records.extend(
            _fit_records(
                np.asarray(samples[name], dtype=np.float64).reshape(-1, 1),
                family_id="REGION",
                feature_names=(name,),
                applicability_rule_id=applicability,
            )
        )
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
                name: _identity_sequence_hash(sample_keys[name])
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
