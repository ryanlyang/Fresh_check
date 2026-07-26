"""Canonical Step-1 contracts for privileged particle-view distillation.

The particle-view campaign reuses the repository's established immutable JSON
format, but gives its coordinate system a separate, acyclic contract.  In
particular, a coordinate binding is an ancestor of consumers and predictors;
it can therefore never contain either checkpoint.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from ..bridge_contracts import (
    canonical_json_bytes,
    canonical_sha256,
    load_hashed_json,
    sha256_file,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)


PARTICLE_VIEW_CANONICAL_JSON_CONTRACT = "particle_view_canonical_json_v1"
PARTICLE_VIEW_COORDINATE_BINDING_CONTRACT = (
    "particle_view_coordinate_binding_v1"
)

SELECTED_VIEW_MATERIALIZATION_POLICY: dict[str, Any] = {
    "dtype": "float32",
    "byte_order": "little",
    "operation_order": [
        "gview_float32_forward",
        "raw_masked_centering",
        "registered_float32_normalizer",
        "standardized_clip_minus6_plus6",
        "invalid_particle_zeroing",
        "canonical_little_endian_float32",
    ],
    "clip_bounds": [-6.0, 6.0],
    "invalid_particles_exactly_zero": True,
    "live_publication_max_abs_tolerance": 1.0e-6,
}

COORDINATE_PARENT_HASH_FIELDS = (
    "source_manifest_sha256",
    "unified_split_manifest_sha256",
    "train_identity_sha256",
    "hlt_source_sha256",
    "offline_source_sha256",
    "a0_checkpoint_sha256",
    "a0_config_sha256",
    "a0_query_tap_sha256",
    "a0_input_normalization_sha256",
    "offline_teacher_checkpoint_sha256",
    "offline_teacher_config_sha256",
    "offline_tap_spec_sha256",
    "generator_checkpoint_sha256",
    "normalizer_sha256",
)

COORDINATE_DEFINITION_FIELDS = (
    "offline_tap_layer",
    "offline_tap_tensor_location",
    "cross_attention_config_sha256",
    "pair_feature_schema_sha256",
    "centering_policy",
    "bounded_coordinate_policy",
    "rate_budget_policy",
    "null_token_policy",
    "bottleneck_width",
)
PARTICLE_VIEW_COORDINATE_PARENT_HASH_FIELDS = COORDINATE_PARENT_HASH_FIELDS
PARTICLE_VIEW_COORDINATE_DEFINITION_FIELDS = COORDINATE_DEFINITION_FIELDS

_FORBIDDEN_COORDINATE_KEY_FRAGMENTS = (
    "consumer",
    "cview",
    "predictor",
    "pview",
    "target_logit",
    "deployment",
)


def require_sha256(name: str, value: Any) -> str:
    """Return *value* after validating canonical lowercase SHA-256 syntax."""

    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _contains_forbidden_descendant(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).lower()
            if any(
                fragment in normalized
                for fragment in _FORBIDDEN_COORDINATE_KEY_FRAGMENTS
            ):
                return True
            if _contains_forbidden_descendant(nested):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_forbidden_descendant(item) for item in value)
    return False


def _validate_coordinate_definition(
    coordinate_definition: Mapping[str, Any],
) -> dict[str, Any]:
    if set(coordinate_definition) != set(COORDINATE_DEFINITION_FIELDS):
        raise ValueError("coordinate definition field inventory mismatch")
    definition = dict(coordinate_definition)
    for name in ("cross_attention_config_sha256", "pair_feature_schema_sha256"):
        require_sha256(name, definition[name])
    width = definition["bottleneck_width"]
    if (
        not isinstance(width, int)
        or isinstance(width, bool)
        or width not in {1, 2, 4, 8}
    ):
        raise ValueError("bottleneck_width must be one of 1, 2, 4, or 8")
    for name in COORDINATE_DEFINITION_FIELDS:
        if name.endswith("_sha256") or name == "bottleneck_width":
            continue
        if not isinstance(definition[name], str) or not definition[name]:
            raise ValueError(f"{name} must be a non-empty string")
    return definition


def build_view_coordinate_binding(
    *,
    parent_hashes: Mapping[str, Any],
    coordinate_definition: Mapping[str, Any],
    materialization_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the immutable ancestor binding for one selected view system."""

    if set(parent_hashes) != set(COORDINATE_PARENT_HASH_FIELDS):
        raise ValueError("coordinate parent hash inventory mismatch")
    parents = {
        name: require_sha256(name, parent_hashes[name])
        for name in COORDINATE_PARENT_HASH_FIELDS
    }
    definition = _validate_coordinate_definition(coordinate_definition)
    materialization = (
        deepcopy(SELECTED_VIEW_MATERIALIZATION_POLICY)
        if materialization_policy is None
        else deepcopy(dict(materialization_policy))
    )
    if materialization != SELECTED_VIEW_MATERIALIZATION_POLICY:
        raise ValueError("selected-view materialization policy changed")
    candidate = {
        "parents": parents,
        "coordinate_definition": definition,
        "materialization": materialization,
    }
    if _contains_forbidden_descendant(candidate):
        raise ValueError(
            "coordinate binding may not reference a consumer, predictor, "
            "target-logit cache, or deployment descendant"
        )
    return with_content_hash(
        {
            "contract": PARTICLE_VIEW_COORDINATE_BINDING_CONTRACT,
            **candidate,
        }
    )


def validate_view_coordinate_binding(payload: Mapping[str, Any]) -> str:
    """Validate schema, hash, canonical materialization, and acyclic lineage."""

    validate_content_hash(
        payload, expected_contract=PARTICLE_VIEW_COORDINATE_BINDING_CONTRACT
    )
    expected_fields = {
        "contract",
        "parents",
        "coordinate_definition",
        "materialization",
        "content_hash",
    }
    if set(payload) != expected_fields:
        raise ValueError("coordinate binding field inventory mismatch")
    rebuilt = build_view_coordinate_binding(
        parent_hashes=payload["parents"],
        coordinate_definition=payload["coordinate_definition"],
        materialization_policy=payload["materialization"],
    )
    if rebuilt != dict(payload):
        raise ValueError("coordinate binding is not canonical")
    return str(payload["content_hash"])


__all__ = [
    "COORDINATE_DEFINITION_FIELDS",
    "COORDINATE_PARENT_HASH_FIELDS",
    "PARTICLE_VIEW_CANONICAL_JSON_CONTRACT",
    "PARTICLE_VIEW_COORDINATE_BINDING_CONTRACT",
    "PARTICLE_VIEW_COORDINATE_DEFINITION_FIELDS",
    "PARTICLE_VIEW_COORDINATE_PARENT_HASH_FIELDS",
    "SELECTED_VIEW_MATERIALIZATION_POLICY",
    "build_view_coordinate_binding",
    "canonical_json_bytes",
    "canonical_sha256",
    "load_hashed_json",
    "require_sha256",
    "sha256_file",
    "validate_content_hash",
    "validate_view_coordinate_binding",
    "with_content_hash",
    "write_immutable_json",
]
