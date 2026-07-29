"""Authenticated Stage-A inputs, REGION caches, and fitted normalizers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from teacher_logit_reco.relational_part import (
    ANGULAR_TREE_SPLIT_MANIFEST_CONTRACT,
    build_angular_tree_resource_contract,
    build_normalization_contract,
    build_raw_input_schema_contract,
    build_relation_family_registry,
    validate_normalization_contract,
    validate_raw_input_schema_contract,
    validate_region_normalization,
    validate_relation_family_registry,
    validate_relation_normalization_artifact,
    unpack_tree_shard,
    validate_existing_tree_shard,
)

from .contracts import (
    bind_source,
    load_hashed_json,
    require_sha256,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from .normalizer_lineage import (
    build_normalizer_population_registry,
    validate_normalizer_population_recipe,
    validate_normalizer_population_registry,
)
from .hlt_v3 import (
    build_hlt_v3_profile_contract,
    validate_hlt_v3_profile_contract,
)


STAGE_A_CONTRACT_BUNDLE = "retb_stage_a_inherited_contract_bundle_v2"
STAGE_A_TREE_INDEX_CONTRACT = "retb_stage_a_region_tree_index_v1"
STAGE_A_NORMALIZER_BUNDLE_CONTRACT = "retb_stage_a_normalizer_bundle_v1"
STAGE_A_INPUT_AUDIT_CONTRACT = "retb_stage_a_input_audit_v2"

STAGE_A_OFFLINE_TREE_ROLES = ("model_train", "val_stop", "val_design")
STAGE_A_HLT_TREE_VIEWS = (
    *(("model_train", replica, "R_MULTI") for replica in range(4)),
    ("val_stop", 0, "R_FIXED"),
    ("val_design", 0, "R_FIXED"),
)


def build_stage_a_contract_bundle(
    *,
    campaign_spec: Mapping[str, Any],
    model_train_identity_count: int,
    scale_train_identity_count: int,
    source_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze inherited scientific contracts and RETB population recipes."""

    validate_content_hash(campaign_spec)
    parents = campaign_spec["parent_artifact_hashes"]
    relation_registry = bind_source(
        build_relation_family_registry(), source_snapshot=source_snapshot
    )
    normalization_contract = bind_source(
        build_normalization_contract(
            split_binding_sha256=parents["split_manifest"]
        ),
        source_snapshot=source_snapshot,
    )
    angular_tree_resource = bind_source(
        build_angular_tree_resource_contract(
            split_binding_sha256=parents["split_manifest"]
        ),
        source_snapshot=source_snapshot,
    )
    inherited_raw_schema = bind_source(
        build_raw_input_schema_contract(), source_snapshot=source_snapshot
    )
    hlt_v3_profile = bind_source(
        build_hlt_v3_profile_contract(
            raw_input_schema_sha256=parents["raw_input_schema"],
            hlt_replica_manifest_sha256=parents["hlt_replica_manifest"],
        ),
        source_snapshot=source_snapshot,
    )
    population_registry = bind_source(
        build_normalizer_population_registry(
            model_train_manifest_sha256=parents["split_manifest"],
            model_train_identity_count=int(model_train_identity_count),
            scale_train_manifest_sha256=parents["scale_train_manifest"],
            scale_train_identity_count=int(scale_train_identity_count),
            raw_input_schema_sha256=parents["raw_input_schema"],
            hlt_v3_profile_sha256=hlt_v3_profile["content_hash"],
            inherited_estimator_contract_sha256=normalization_contract[
                "content_hash"
            ],
        ),
        source_snapshot=source_snapshot,
    )
    artifacts = {
        "relation_family_registry": relation_registry,
        "normalization_contract": normalization_contract,
        "angular_tree_resource": angular_tree_resource,
        "inherited_raw_input_schema": inherited_raw_schema,
        "hlt_v3_profile": hlt_v3_profile,
        "normalizer_population_registry": population_registry,
    }
    bundle = bind_source(
        with_content_hash(
            {
                "contract": STAGE_A_CONTRACT_BUNDLE,
                "schema_version": 2,
                "campaign_spec_sha256": campaign_spec["content_hash"],
                "retb_raw_input_schema_sha256": parents["raw_input_schema"],
                "artifact_hashes": {
                    name: artifact["content_hash"]
                    for name, artifact in artifacts.items()
                },
                "relation_families": [
                    "PT",
                    "TRACK",
                    "PID",
                    "CHARGE",
                    "DENSITY",
                    "REGION",
                ],
                "offline_and_hlt_normalizers_separate": True,
                "shared_hlt_replica_weighting": "equal_identity_replica_weight",
                "normalizer_sampling": {
                    "maximum_fitted_views": 50_000,
                    "offline_500k": (
                        "inherited_salted_identity_sample_up_to_50000"
                    ),
                    "shared_hlt_500k": (
                        "inherited_salted_base_identity_sample_up_to_12500_"
                        "then_all_replicas_0_1_2_3"
                    ),
                    "shared_hlt_sample_is_replica_balanced": True,
                    "selected_base_identity_order": (
                        "inherited_salted_identity_order"
                    ),
                    "replica_expansion_order": "identity_major_0_1_2_3",
                },
                "region_backend": "relational_ca_tree_v1",
            }
        ),
        source_snapshot=source_snapshot,
    )
    result = {"stage_a_contract_bundle": bundle, **artifacts}
    validate_stage_a_contract_bundle(result, campaign_spec=campaign_spec)
    return result


def validate_stage_a_contract_bundle(
    bundle: Mapping[str, Any],
    *,
    campaign_spec: Mapping[str, Any],
) -> str:
    validate_content_hash(campaign_spec)
    registry_sha = validate_relation_family_registry(
        bundle["relation_family_registry"]
    )
    normalization_sha = validate_normalization_contract(
        bundle["normalization_contract"]
    )
    tree_sha = validate_content_hash(bundle["angular_tree_resource"])
    raw_sha = validate_raw_input_schema_contract(
        bundle["inherited_raw_input_schema"]
    )
    profile_sha = validate_hlt_v3_profile_contract(bundle["hlt_v3_profile"])
    population_sha = validate_normalizer_population_registry(
        bundle["normalizer_population_registry"]
    )
    artifact = bundle["stage_a_contract_bundle"]
    digest = validate_content_hash(
        artifact, expected_contract=STAGE_A_CONTRACT_BUNDLE
    )
    expected_hashes = {
        "relation_family_registry": registry_sha,
        "normalization_contract": normalization_sha,
        "angular_tree_resource": tree_sha,
        "inherited_raw_input_schema": raw_sha,
        "hlt_v3_profile": profile_sha,
        "normalizer_population_registry": population_sha,
    }
    if (
        artifact["campaign_spec_sha256"] != campaign_spec["content_hash"]
        or int(artifact.get("schema_version", -1)) != 2
        or artifact["retb_raw_input_schema_sha256"]
        != campaign_spec["parent_artifact_hashes"]["raw_input_schema"]
        or artifact["artifact_hashes"] != expected_hashes
        or artifact.get("normalizer_sampling")
        != {
            "maximum_fitted_views": 50_000,
            "offline_500k": "inherited_salted_identity_sample_up_to_50000",
            "shared_hlt_500k": (
                "inherited_salted_base_identity_sample_up_to_12500_"
                "then_all_replicas_0_1_2_3"
            ),
            "shared_hlt_sample_is_replica_balanced": True,
            "selected_base_identity_order": "inherited_salted_identity_order",
            "replica_expansion_order": "identity_major_0_1_2_3",
        }
        or any(
            child.get("source") != campaign_spec.get("source")
            for child in (
                artifact,
                bundle["relation_family_registry"],
                bundle["normalization_contract"],
                bundle["angular_tree_resource"],
                bundle["inherited_raw_input_schema"],
                bundle["hlt_v3_profile"],
                bundle["normalizer_population_registry"],
            )
        )
    ):
        raise ValueError("Stage-A inherited contract lineage differs")
    return digest


def publish_stage_a_contract_bundle(
    *,
    campaign_root: str | Path,
    bundle: Mapping[str, Any],
    campaign_spec: Mapping[str, Any],
) -> dict[str, Any]:
    validate_stage_a_contract_bundle(bundle, campaign_spec=campaign_spec)
    root = Path(campaign_root)
    paths = {
        "stage_a_contract_bundle": (
            root / "registry" / "retb_stage_a_contract_bundle.json"
        ),
        "relation_family_registry": (
            root / "registry" / "inherited_relation_family_registry.json"
        ),
        "normalization_contract": (
            root / "inputs" / "inherited_normalization_contract.json"
        ),
        "angular_tree_resource": (
            root / "inputs" / "inherited_angular_tree_resource.json"
        ),
        "inherited_raw_input_schema": (
            root / "inputs" / "inherited_relational_raw_input_schema.json"
        ),
        "hlt_v3_profile": root / "inputs" / "hlt_v3_profile.json",
        "normalizer_population_registry": (
            root / "registry" / "normalizer_population_registry.json"
        ),
    }
    return {
        name: write_immutable_json(paths[name], bundle[name])
        for name in paths
    }


def bind_fitted_normalizer(
    artifact: Mapping[str, Any],
    *,
    logical_domain: str,
    population_recipe: Mapping[str, Any],
    identity_manifest_sha256: str,
    view_content_sha256s: Sequence[str],
    campaign_spec_sha256: str,
    source_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach RETB population semantics to an inherited fitted artifact."""

    validate_relation_normalization_artifact(artifact)
    recipe_sha = validate_normalizer_population_recipe(
        population_recipe, expected_logical_domain=logical_domain
    )
    view_hashes = [
        require_sha256(value, name="view_content_sha256")
        for value in view_content_sha256s
    ]
    if logical_domain.startswith("shared_hlt_") and len(view_hashes) != 4:
        raise ValueError("shared-HLT normalizer requires exactly four views")
    if logical_domain.startswith("offline_") and len(view_hashes) != 1:
        raise ValueError("offline normalizer requires exactly one view")
    result = dict(artifact)
    result.pop("content_hash")
    result.pop("source", None)
    result.update(
        {
            "normalizer_population_recipe_sha256": recipe_sha,
            "logical_domain": logical_domain,
            "identity_manifest_sha256": require_sha256(
                identity_manifest_sha256, name="identity_manifest_sha256"
            ),
            "view_content_sha256s": view_hashes,
            "campaign_spec_sha256": require_sha256(
                campaign_spec_sha256, name="campaign_spec_sha256"
            ),
            "retb_replica_weighting": population_recipe[
                "replica_weighting"
            ]["policy"],
        }
    )
    result = bind_source(
        with_content_hash(result), source_snapshot=source_snapshot
    )
    validate_relation_normalization_artifact(result)
    return result


def bind_fitted_region_normalizer(
    artifact: Mapping[str, Any],
    *,
    relation_normalizer: Mapping[str, Any],
    logical_domain: str,
    population_recipe: Mapping[str, Any],
    tree_manifest_sha256s: Sequence[str],
    campaign_spec_sha256: str,
    source_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    relation_sha = validate_relation_normalization_artifact(
        relation_normalizer
    )
    validate_region_normalization(
        artifact, relation_normalization_sha256=relation_sha
    )
    recipe_sha = validate_normalizer_population_recipe(
        population_recipe, expected_logical_domain=logical_domain
    )
    tree_hashes = [
        require_sha256(value, name="tree_manifest_sha256")
        for value in tree_manifest_sha256s
    ]
    expected = 4 if logical_domain.startswith("shared_hlt_") else 1
    if len(tree_hashes) != expected:
        raise ValueError("REGION normalizer tree-parent multiplicity differs")
    result = dict(artifact)
    result.pop("content_hash")
    result.pop("source", None)
    result.update(
        {
            "normalizer_population_recipe_sha256": recipe_sha,
            "logical_domain": logical_domain,
            "tree_manifest_sha256s": tree_hashes,
            "campaign_spec_sha256": require_sha256(
                campaign_spec_sha256, name="campaign_spec_sha256"
            ),
        }
    )
    result = bind_source(
        with_content_hash(result), source_snapshot=source_snapshot
    )
    validate_region_normalization(
        result, relation_normalization_sha256=relation_sha
    )
    return result


def build_stage_a_normalizer_bundle(
    *,
    campaign_spec_sha256: str,
    stage_a_contract_bundle_sha256: str,
    population_registry: Mapping[str, Any],
    offline_relation: Mapping[str, Any],
    offline_region: Mapping[str, Any],
    shared_hlt_relation: Mapping[str, Any],
    shared_hlt_region: Mapping[str, Any],
    source_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    population_sha = validate_normalizer_population_registry(
        population_registry
    )
    artifacts = {
        "offline_500k_relation": offline_relation,
        "offline_500k_region": offline_region,
        "shared_hlt_500k_relation": shared_hlt_relation,
        "shared_hlt_500k_region": shared_hlt_region,
    }
    validate_relation_normalization_artifact(offline_relation)
    validate_region_normalization(
        offline_region,
        relation_normalization_sha256=offline_relation["content_hash"],
    )
    validate_relation_normalization_artifact(shared_hlt_relation)
    validate_region_normalization(
        shared_hlt_region,
        relation_normalization_sha256=shared_hlt_relation["content_hash"],
    )
    for name, artifact in artifacts.items():
        expected_domain = (
            "shared_hlt_500k"
            if name.startswith("shared_hlt")
            else "offline_500k"
        )
        if (
            artifact.get("logical_domain") != expected_domain
            or artifact.get("normalizer_population_recipe_sha256")
            != population_registry["recipe_hashes"][expected_domain]
        ):
            raise ValueError(f"{name} population lineage differs")
    artifact = bind_source(
        with_content_hash(
            {
                "contract": STAGE_A_NORMALIZER_BUNDLE_CONTRACT,
                "schema_version": 1,
                "campaign_spec_sha256": require_sha256(
                    campaign_spec_sha256, name="campaign_spec_sha256"
                ),
                "stage_a_contract_bundle_sha256": require_sha256(
                    stage_a_contract_bundle_sha256,
                    name="stage_a_contract_bundle_sha256",
                ),
                "normalizer_population_registry_sha256": population_sha,
                "artifact_hashes": {
                    name: value["content_hash"]
                    for name, value in artifacts.items()
                },
                "all_records_finite": True,
                "offline_and_hlt_interchangeable": False,
                "shared_hlt_replica_ids": [0, 1, 2, 3],
                "validation_stack_or_test_statistics_used": False,
            }
        ),
        source_snapshot=source_snapshot,
    )
    return artifact


def validate_stage_a_normalizer_bundle(
    payload: Mapping[str, Any],
    *,
    artifacts: Mapping[str, Mapping[str, Any]] | None = None,
) -> str:
    digest = validate_content_hash(
        payload, expected_contract=STAGE_A_NORMALIZER_BUNDLE_CONTRACT
    )
    for name in (
        "campaign_spec_sha256",
        "stage_a_contract_bundle_sha256",
        "normalizer_population_registry_sha256",
    ):
        require_sha256(payload.get(name), name=name)
    expected_names = {
        "offline_500k_relation",
        "offline_500k_region",
        "shared_hlt_500k_relation",
        "shared_hlt_500k_region",
    }
    if set(payload.get("artifact_hashes", {})) != expected_names:
        raise ValueError("Stage-A normalizer artifact coverage differs")
    if artifacts is not None:
        if set(artifacts) != expected_names:
            raise ValueError("Stage-A normalizer artifact inputs differ")
        for name, artifact in artifacts.items():
            validate_content_hash(artifact)
            if payload["artifact_hashes"][name] != artifact["content_hash"]:
                raise ValueError(f"Stage-A normalizer hash differs for {name}")
    if (
        payload.get("all_records_finite") is not True
        or payload.get("offline_and_hlt_interchangeable") is not False
        or payload.get("shared_hlt_replica_ids") != [0, 1, 2, 3]
        or payload.get("validation_stack_or_test_statistics_used") is not False
    ):
        raise ValueError("Stage-A normalizer scientific controls differ")
    return digest


def build_stage_a_tree_index(
    *,
    campaign_spec_sha256: str,
    backend_manifest_sha256: str,
    angular_tree_resource_sha256: str,
    views: Sequence[Mapping[str, Any]],
    source_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    rows = [dict(row) for row in views]
    identities = [str(row["view_id"]) for row in rows]
    if len(rows) != len(set(identities)):
        raise ValueError("Stage-A REGION view identities are duplicated")
    for row in rows:
        for field in (
            "view_content_sha256",
            "identity_manifest_sha256",
            "tree_manifest_sha256",
        ):
            require_sha256(row.get(field), name=f"{row['view_id']}.{field}")
        if int(row.get("jet_count", 0)) <= 0:
            raise ValueError("Stage-A REGION view is empty")
    artifact = bind_source(
        with_content_hash(
            {
                "contract": STAGE_A_TREE_INDEX_CONTRACT,
                "schema_version": 1,
                "campaign_spec_sha256": require_sha256(
                    campaign_spec_sha256, name="campaign_spec_sha256"
                ),
                "backend_manifest_sha256": require_sha256(
                    backend_manifest_sha256, name="backend_manifest_sha256"
                ),
                "angular_tree_resource_sha256": require_sha256(
                    angular_tree_resource_sha256,
                    name="angular_tree_resource_sha256",
                ),
                "views": rows,
                "view_count": len(rows),
                "identity_order_exact": True,
                "complete_stage_a_identity_coverage": True,
            }
        ),
        source_snapshot=source_snapshot,
    )
    validate_stage_a_tree_index(artifact)
    return artifact


def validate_stage_a_tree_index(payload: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        payload, expected_contract=STAGE_A_TREE_INDEX_CONTRACT
    )
    for name in (
        "campaign_spec_sha256",
        "backend_manifest_sha256",
        "angular_tree_resource_sha256",
    ):
        require_sha256(payload.get(name), name=name)
    views = payload.get("views")
    if (
        not isinstance(views, list)
        or len(views) != int(payload.get("view_count", -1))
        or len({row.get("view_id") for row in views}) != len(views)
    ):
        raise ValueError("Stage-A REGION index view coverage differs")
    if (
        payload.get("identity_order_exact") is not True
        or payload.get("complete_stage_a_identity_coverage") is not True
    ):
        raise ValueError("Stage-A REGION identity coverage differs")
    return digest


def build_stage_a_input_audit(
    *,
    campaign_spec_sha256: str,
    offline_views: Sequence[Mapping[str, Any]],
    hlt_views: Sequence[Mapping[str, Any]],
    tree_index_sha256: str,
    normalizer_bundle_sha256: str,
    hlt_v3_degradation_audit_sha256: str,
    source_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    offline = [dict(row) for row in offline_views]
    hlt = [dict(row) for row in hlt_views]
    if not offline or not hlt:
        raise ValueError("Stage-A input audit requires offline and HLT views")
    for row in (*offline, *hlt):
        for field in (
            "metadata_sha256",
            "identity_manifest_sha256",
            "identity_order_sha256",
        ):
            require_sha256(row.get(field), name=f"input_audit.{field}")
        if (
            int(row.get("event_count", 0)) <= 0
            or int(row.get("particle_capacity", -1)) != 128
            or int(row.get("raw_particle_field_count", -1)) != 14
            or row.get("tokens_dtype") != "float32"
            or row.get("mask_dtype") != "bool"
            or row.get("finite_valid_tokens") is not True
            or row.get("padding_zero_exact") is not True
            or row.get("identities_unique") is not True
        ):
            raise ValueError("Stage-A audited view invariants differ")
    artifact = bind_source(
        with_content_hash(
            {
                "contract": STAGE_A_INPUT_AUDIT_CONTRACT,
                "schema_version": 2,
                "campaign_spec_sha256": require_sha256(
                    campaign_spec_sha256, name="campaign_spec_sha256"
                ),
                "tree_index_sha256": require_sha256(
                    tree_index_sha256, name="tree_index_sha256"
                ),
                "normalizer_bundle_sha256": require_sha256(
                    normalizer_bundle_sha256, name="normalizer_bundle_sha256"
                ),
                "hlt_v3_degradation_audit_sha256": require_sha256(
                    hlt_v3_degradation_audit_sha256,
                    name="hlt_v3_degradation_audit_sha256",
                ),
                "offline_views": offline,
                "hlt_views": hlt,
                "offline_view_count": len(offline),
                "hlt_view_count": len(hlt),
                "relation_family_coverage": [
                    "base4",
                    "PT",
                    "TRACK",
                    "PID",
                    "CHARGE",
                    "DENSITY",
                    "REGION",
                ],
                "constituent_matching_fields_present": False,
                "offline_labels_authenticated": True,
                "hlt_cache_contains_labels": False,
                "validation_or_test_statistics_used_for_normalization": False,
                "identity_alignment_exact": True,
                "ok": True,
            }
        ),
        source_snapshot=source_snapshot,
    )
    validate_stage_a_input_audit(artifact)
    return artifact


def validate_stage_a_input_audit(payload: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        payload, expected_contract=STAGE_A_INPUT_AUDIT_CONTRACT
    )
    for name in (
        "campaign_spec_sha256",
        "tree_index_sha256",
        "normalizer_bundle_sha256",
        "hlt_v3_degradation_audit_sha256",
    ):
        require_sha256(payload.get(name), name=name)
    if (
        int(payload.get("schema_version", -1)) != 2
        or int(payload.get("offline_view_count", -1))
        != len(payload.get("offline_views", ()))
        or int(payload.get("hlt_view_count", -1))
        != len(payload.get("hlt_views", ()))
        or payload.get("constituent_matching_fields_present") is not False
        or payload.get("offline_labels_authenticated") is not True
        or payload.get("hlt_cache_contains_labels") is not False
        or payload.get("validation_or_test_statistics_used_for_normalization")
        is not False
        or payload.get("identity_alignment_exact") is not True
        or payload.get("ok") is not True
    ):
        raise ValueError("Stage-A input audit differs")
    return digest


def identity_newline_sha256(identities: Sequence[str]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for identity in identities:
        digest.update(str(identity).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def padding_is_exact_zero(tokens: np.ndarray, mask: np.ndarray) -> bool:
    values = np.asarray(tokens)
    valid = np.asarray(mask, dtype=bool)
    return bool(np.all(values[~valid] == 0.0))


def load_authenticated_tree_selection(
    tree_root: str | Path,
    identities: Sequence[str],
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    """Load selected trees only after validating every split/shard byte hash."""

    root = Path(tree_root)
    manifest = load_hashed_json(
        root / "manifest.json",
        expected_contract=ANGULAR_TREE_SPLIT_MANIFEST_CONTRACT,
    )
    requested = [str(value) for value in identities]
    if len(requested) != len(set(requested)):
        raise ValueError("selected REGION identities are duplicated")
    requested_set = set(requested)
    by_identity: dict[str, Mapping[str, Any]] = {}
    observed_count = 0
    for expected_index, shard_row in enumerate(manifest["shards"]):
        shard_index = int(shard_row["shard_index"])
        if shard_index != expected_index:
            raise ValueError("REGION split shard ordering differs")
        shard = root / "shards" / f"shard_{shard_index:05d}.npz"
        with np.load(shard, allow_pickle=False) as payload:
            shard_ids = [
                str(value) for value in payload["identity"].tolist()
            ]
        metadata = validate_existing_tree_shard(
            shard,
            shard_ids,
            hlt_content_sha256=manifest["parents"][
                "hlt_content_sha256"
            ],
            tree_resource_sha256=manifest["parents"][
                "tree_resource_sha256"
            ],
            backend_manifest_sha256=manifest["parents"][
                "backend_manifest_sha256"
            ],
        )
        if metadata is None:
            raise FileNotFoundError("authenticated REGION shard is absent")
        expected = {
            "metadata_sha256": metadata["content_hash"],
            "jet_count": int(metadata["jet_count"]),
            "identity_sha256": metadata["identity_sha256"],
            "npz_sha256": metadata["npz_sha256"],
        }
        if any(shard_row[name] != value for name, value in expected.items()):
            raise ValueError("REGION split/shard lineage differs")
        observed_count += len(shard_ids)
        rows = [
            index
            for index, identity in enumerate(shard_ids)
            if identity in requested_set
        ]
        if not rows:
            continue
        authenticated_ids, trees = unpack_tree_shard(shard, rows=rows)
        for row, tree in zip(rows, trees):
            identity = authenticated_ids[row]
            if identity in by_identity:
                raise ValueError("selected REGION identity is duplicated")
            by_identity[identity] = tree
    if observed_count != int(manifest["jet_count"]):
        raise ValueError("REGION split identity coverage differs")
    missing = requested_set - set(by_identity)
    if missing:
        raise ValueError(f"REGION cache lacks {len(missing)} selected identities")
    return [by_identity[value] for value in requested], manifest


__all__ = [
    "STAGE_A_CONTRACT_BUNDLE",
    "STAGE_A_HLT_TREE_VIEWS",
    "STAGE_A_INPUT_AUDIT_CONTRACT",
    "STAGE_A_NORMALIZER_BUNDLE_CONTRACT",
    "STAGE_A_OFFLINE_TREE_ROLES",
    "STAGE_A_TREE_INDEX_CONTRACT",
    "bind_fitted_normalizer",
    "bind_fitted_region_normalizer",
    "build_stage_a_contract_bundle",
    "build_stage_a_input_audit",
    "build_stage_a_normalizer_bundle",
    "build_stage_a_tree_index",
    "identity_newline_sha256",
    "load_authenticated_tree_selection",
    "padding_is_exact_zero",
    "publish_stage_a_contract_bundle",
    "validate_stage_a_contract_bundle",
    "validate_stage_a_input_audit",
    "validate_stage_a_normalizer_bundle",
    "validate_stage_a_tree_index",
]
