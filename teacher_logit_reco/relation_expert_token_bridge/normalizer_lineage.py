"""Locked RETB relation/REGION normalizer populations and lineage checks."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from .contracts import require_sha256, validate_content_hash, with_content_hash


NORMALIZER_POPULATION_CONTRACT = "retb_normalizer_population_registry_v1"
NORMALIZER_RECIPE_CONTRACT = "retb_normalizer_population_recipe_v1"
SHARED_HLT_REPLICAS = (0, 1, 2, 3)
REALIZATION_POLICIES_SHARING_HLT_NORMALIZER = (
    "R_FIXED",
    "R_MULTI",
    "R_RANDOM",
)


def normalizer_population_rows(
    identities: Sequence[str],
    *,
    logical_domain: str,
) -> tuple[tuple[str, int | None], ...]:
    """Return the exact canonical population traversal for a fitted domain."""

    canonical = tuple(sorted(str(identity) for identity in identities))
    if len(canonical) != len(set(canonical)):
        raise ValueError("normalizer identities must be unique")
    if logical_domain in {"offline_500k", "offline_scale"}:
        return tuple((identity, None) for identity in canonical)
    if logical_domain in {"shared_hlt_500k", "shared_hlt_scale"}:
        return tuple(
            (identity, replica_id)
            for identity in canonical
            for replica_id in SHARED_HLT_REPLICAS
        )
    raise ValueError(f"unknown normalizer logical domain {logical_domain!r}")


def build_normalizer_population_recipe(
    *,
    logical_domain: str,
    identity_manifest_sha256: str,
    identity_count: int,
    raw_input_schema_sha256: str,
    hlt_v3_profile_sha256: str | None,
    inherited_estimator_contract_sha256: str,
) -> dict[str, Any]:
    if int(identity_count) <= 0:
        raise ValueError("normalizer identity_count must be positive")
    offline = logical_domain in {"offline_500k", "offline_scale"}
    hlt = logical_domain in {"shared_hlt_500k", "shared_hlt_scale"}
    if not (offline or hlt):
        raise ValueError(f"unknown normalizer logical domain {logical_domain!r}")
    scale = logical_domain.endswith("_scale")
    if hlt:
        hlt_parent = require_sha256(
            hlt_v3_profile_sha256, name="hlt_v3_profile_sha256"
        )
    elif hlt_v3_profile_sha256 is not None:
        raise ValueError("offline normalizer recipe must not bind an HLT profile")
    else:
        hlt_parent = None
    replicas = list(SHARED_HLT_REPLICAS) if hlt else []
    return with_content_hash(
        {
            "contract": NORMALIZER_RECIPE_CONTRACT,
            "schema_version": 1,
            "logical_domain": logical_domain,
            "scale_stage": "3m_scale_train" if scale else "500k_model_train",
            "view": "nominal_hlt_v3" if hlt else "offline",
            "identity_manifest_sha256": require_sha256(
                identity_manifest_sha256, name="identity_manifest_sha256"
            ),
            "identity_count": int(identity_count),
            "replica_ids": replicas,
            "replica_weighting": (
                {
                    "policy": "equal_identity_replica_weight",
                    "relative_weight_numerator": 1,
                    "relative_weight_denominator": len(SHARED_HLT_REPLICAS),
                    "population_entry_count": int(identity_count)
                    * len(SHARED_HLT_REPLICAS),
                }
                if hlt
                else {
                    "policy": "one_view_per_identity",
                    "relative_weight_numerator": 1,
                    "relative_weight_denominator": 1,
                    "population_entry_count": int(identity_count),
                }
            ),
            "canonical_traversal": (
                "ascending_identity_then_replica_0_1_2_3"
                if hlt
                else "ascending_identity"
            ),
            "raw_input_schema_sha256": require_sha256(
                raw_input_schema_sha256, name="raw_input_schema_sha256"
            ),
            "hlt_v3_profile_sha256": hlt_parent,
            "inherited_estimator_contract_sha256": require_sha256(
                inherited_estimator_contract_sha256,
                name="inherited_estimator_contract_sha256",
            ),
            "relation_families": [
                "PT",
                "TRACK",
                "PID",
                "CHARGE",
                "DENSITY",
                "REGION",
            ],
            "valid_ordered_pair_mask_inherited": True,
            "region_rebuilt_from_exact_view": True,
            "validation_stack_or_test_identity_allowed": False,
            "shared_by_realization_policies": (
                list(REALIZATION_POLICIES_SHARING_HLT_NORMALIZER) if hlt else []
            ),
            "fixed_severity_robustness_uses_shared_hlt_recipe": hlt,
        }
    )


def build_normalizer_population_registry(
    *,
    model_train_manifest_sha256: str,
    model_train_identity_count: int,
    scale_train_manifest_sha256: str,
    scale_train_identity_count: int,
    raw_input_schema_sha256: str,
    hlt_v3_profile_sha256: str,
    inherited_estimator_contract_sha256: str,
) -> dict[str, Any]:
    common = {
        "raw_input_schema_sha256": raw_input_schema_sha256,
        "inherited_estimator_contract_sha256": inherited_estimator_contract_sha256,
    }
    recipes = {
        "offline_500k": build_normalizer_population_recipe(
            logical_domain="offline_500k",
            identity_manifest_sha256=model_train_manifest_sha256,
            identity_count=model_train_identity_count,
            hlt_v3_profile_sha256=None,
            **common,
        ),
        "shared_hlt_500k": build_normalizer_population_recipe(
            logical_domain="shared_hlt_500k",
            identity_manifest_sha256=model_train_manifest_sha256,
            identity_count=model_train_identity_count,
            hlt_v3_profile_sha256=hlt_v3_profile_sha256,
            **common,
        ),
        "offline_scale": build_normalizer_population_recipe(
            logical_domain="offline_scale",
            identity_manifest_sha256=scale_train_manifest_sha256,
            identity_count=scale_train_identity_count,
            hlt_v3_profile_sha256=None,
            **common,
        ),
        "shared_hlt_scale": build_normalizer_population_recipe(
            logical_domain="shared_hlt_scale",
            identity_manifest_sha256=scale_train_manifest_sha256,
            identity_count=scale_train_identity_count,
            hlt_v3_profile_sha256=hlt_v3_profile_sha256,
            **common,
        ),
    }
    return with_content_hash(
        {
            "contract": NORMALIZER_POPULATION_CONTRACT,
            "schema_version": 1,
            "recipes": recipes,
            "recipe_hashes": {
                name: recipe["content_hash"] for name, recipe in recipes.items()
            },
            "five_hundred_k_and_scale_interchangeable": False,
            "offline_and_hlt_interchangeable": False,
            "severity_specific_hlt_normalizer_allowed": False,
        }
    )


def validate_normalizer_population_recipe(
    recipe: Mapping[str, Any],
    *,
    expected_logical_domain: str | None = None,
) -> str:
    digest = validate_content_hash(
        recipe, expected_contract=NORMALIZER_RECIPE_CONTRACT
    )
    domain = str(recipe.get("logical_domain"))
    if expected_logical_domain is not None and domain != expected_logical_domain:
        raise ValueError(
            f"normalizer domain mismatch: expected {expected_logical_domain}, got {domain}"
        )
    hlt = domain.startswith("shared_hlt_")
    if hlt and recipe.get("replica_ids") != list(SHARED_HLT_REPLICAS):
        raise ValueError("shared-HLT normalizer does not use exact replicas 0..3")
    if hlt and recipe.get("shared_by_realization_policies") != list(
        REALIZATION_POLICIES_SHARING_HLT_NORMALIZER
    ):
        raise ValueError("shared-HLT realization-policy reuse differs")
    if not hlt and recipe.get("replica_ids") != []:
        raise ValueError("offline normalizer unexpectedly contains replicas")
    return digest


def validate_normalizer_population_registry(payload: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        payload, expected_contract=NORMALIZER_POPULATION_CONTRACT
    )
    expected_domains = {
        "offline_500k",
        "shared_hlt_500k",
        "offline_scale",
        "shared_hlt_scale",
    }
    recipes = payload.get("recipes")
    if not isinstance(recipes, Mapping) or set(recipes) != expected_domains:
        raise ValueError("normalizer population registry domains differ")
    for domain in sorted(expected_domains):
        recipe_sha = validate_normalizer_population_recipe(
            recipes[domain], expected_logical_domain=domain
        )
        if payload["recipe_hashes"].get(domain) != recipe_sha:
            raise ValueError(f"normalizer recipe hash differs for {domain}")
    if (
        recipes["offline_500k"]["identity_manifest_sha256"]
        == recipes["offline_scale"]["identity_manifest_sha256"]
    ):
        raise ValueError("500k and scale normalizers must bind different manifests")
    return digest


def require_normalizer_parent(
    artifact: Mapping[str, Any],
    *,
    expected_recipe_sha256: str,
) -> None:
    expected = require_sha256(
        expected_recipe_sha256, name="expected_recipe_sha256"
    )
    if artifact.get("normalizer_population_recipe_sha256") != expected:
        raise ValueError("artifact belongs to another normalizer population recipe")


__all__ = [
    "NORMALIZER_POPULATION_CONTRACT",
    "NORMALIZER_RECIPE_CONTRACT",
    "REALIZATION_POLICIES_SHARING_HLT_NORMALIZER",
    "SHARED_HLT_REPLICAS",
    "build_normalizer_population_recipe",
    "build_normalizer_population_registry",
    "normalizer_population_rows",
    "require_normalizer_parent",
    "validate_normalizer_population_recipe",
    "validate_normalizer_population_registry",
]
