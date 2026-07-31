"""Fail-closed source capability audit and immutable HOSD target registry."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from teacher_logit_reco.relation_expert_token_bridge.provenance import (
    RAW_INPUT_SCHEMA_CONTRACT,
)

from .contracts import (
    SOURCE_STATUS_HASH_POLICY,
    STRUCTURE_TARGET_REGISTRY_CONTRACT,
    TARGET_CAPABILITY_AUDIT_CONTRACT,
    canonical_sha256,
    require_git_object_id,
    require_sha256,
    validate_content_hash,
    with_content_hash,
)
from .target_schemas import (
    AVAILABILITY_CLASSES,
    CURRENT_OPTIONAL,
    CURRENT_REQUIRED,
    FUTURE_ONLY,
    RELATION_CHANNELS,
    TargetDeclaration,
    target_component_availability_groups,
    target_declarations,
)

EXPECTED_RAW_FIELDS = (
    "pt",
    "eta",
    "phi",
    "energy",
    "charge",
    "is_charged_hadron",
    "is_neutral_hadron",
    "is_photon",
    "is_electron",
    "is_muon",
    "d0",
    "d0err",
    "dz",
    "dzerr",
)
EXPECTED_PARTICLE_BRANCHES = (
    "part_px",
    "part_py",
    "part_pz",
    "part_energy",
    "part_charge",
    "part_isChargedHadron",
    "part_isNeutralHadron",
    "part_isPhoton",
    "part_isElectron",
    "part_isMuon",
    "part_d0val",
    "part_d0err",
    "part_dzval",
    "part_dzerr",
)
FORBIDDEN_MATCHING_FIELD_FRAGMENTS = (
    "constituent_match",
    "matched_constituent",
    "match_index",
    "matched_index",
    "offline_index",
    "hlt_index",
    "offline_to_hlt",
    "hlt_to_offline",
    "source_index",
    "source_indices",
    "source_constituent",
    "original_constituent_index",
    "degradation_source",
    "nearest_neighbor_assignment",
    "hungarian_assignment",
    "transport_assignment",
)
FUTURE_DATA_REQUIREMENTS = {
    "T_HLT_TRACK_ORIGIN_TRUTH": {
        "required_authenticated_semantics": (
            "HLT-native truth category attached to each HLT object",
            "explicit unmatched, ambiguous, material, pileup, and merged policies",
        ),
    },
    "T_HLT_COMMON_VERTEX_TRUTH": {
        "required_authenticated_semantics": (
            "HLT-native production-vertex identity attached to each HLT object",
            "explicit unmatched, ambiguous, material, pileup, and merged policies",
        ),
    },
    "T_OFFLINE_SECONDARY_VERTEX_SET": {
        "required_authenticated_semantics": (
            "literal reconstructed offline secondary-vertex collection",
            "separately reviewed predicted-slot set-loss contract",
        ),
    },
}


def _source(value: Mapping[str, Any]) -> dict[str, Any]:
    source = {
        "commit": require_git_object_id(value.get("commit"), name="source.commit"),
        "status_sha256": require_sha256(
            value.get("status_sha256"), name="source.status_sha256"
        ),
        "dirty": bool(value.get("dirty")),
        "status_hash_policy": str(value.get("status_hash_policy")),
    }
    if source["status_hash_policy"] != SOURCE_STATUS_HASH_POLICY:
        raise ValueError("source.status_hash_policy differs from the global contract")
    return source


def _string_tuple(value: Any, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{name} must be a JSON string list")
    if len(value) != len(set(value)):
        raise ValueError(f"{name} contains duplicates")
    return tuple(value)


def _matching_field_hits(fields: Iterable[str]) -> tuple[str, ...]:
    hits = []
    for field in fields:
        normalized = (
            field.strip()
            .lower()
            .replace("-", "_")
            .replace(".", "_")
            .replace("/", "_")
        )
        if any(fragment in normalized for fragment in FORBIDDEN_MATCHING_FIELD_FRAGMENTS):
            hits.append(field)
    return tuple(sorted(set(hits)))


def inspect_raw_schema(raw_input_schema: Mapping[str, Any]) -> dict[str, Any]:
    """Inspect the authenticated reader schema without opening labels."""

    validate_content_hash(
        raw_input_schema, expected_contract=RAW_INPUT_SCHEMA_CONTRACT
    )
    raw_fields = _string_tuple(raw_input_schema.get("raw_fields"), name="raw_fields")
    branches = _string_tuple(
        raw_input_schema.get("required_particle_branches"),
        name="required_particle_branches",
    )
    forbidden_hits = _matching_field_hits((*raw_fields, *branches))
    if forbidden_hits:
        raise ValueError(
            "constituent-matching or degradation-lineage fields are forbidden: "
            f"{list(forbidden_hits)}"
        )
    if raw_input_schema.get("constituent_matching_fields_allowed") is not False:
        raise ValueError("raw schema must explicitly forbid constituent matching fields")

    fields = set(raw_fields)
    branch_set = set(branches)
    exact_current_schema = (
        raw_input_schema.get("schema_version") == 2
        and raw_input_schema.get("raw_dimension") == len(EXPECTED_RAW_FIELDS)
        and raw_input_schema.get("derived_dimension") == 17
        and raw_input_schema.get("max_constituents") == 128
        and raw_fields == EXPECTED_RAW_FIELDS
        and branches == EXPECTED_PARTICLE_BRANCHES
        and raw_input_schema.get("pid_zero_hot_policy") == "unknown_category"
        and raw_input_schema.get("pid_multi_hot_policy") == "fail_preflight"
        and raw_input_schema.get("derived_input_implementation")
        == "jetclass_fresh.part_inputs"
        and raw_input_schema.get("measurement_validity_states")
        == [
            "not_track_domain",
            "track_measurement_available",
            "track_measurement_missing",
        ]
        and raw_input_schema.get("invalid_track_measurement_sentinel")
        == {
            "d0": 0.0,
            "d0err": 0.0,
            "dz": 0.0,
            "dzerr": 0.0,
            "inferred_from_observed_zeros": False,
        }
    )

    capability_requirements = {
        "particle_four_vector": (
            {"pt", "eta", "phi", "energy"},
            {"part_px", "part_py", "part_pz", "part_energy"},
        ),
        "charge": ({"charge"}, {"part_charge"}),
        "pid_categories": (
            {
                "is_charged_hadron",
                "is_neutral_hadron",
                "is_photon",
                "is_electron",
                "is_muon",
            },
            {
                "part_isChargedHadron",
                "part_isNeutralHadron",
                "part_isPhoton",
                "part_isElectron",
                "part_isMuon",
            },
        ),
        "track_measurements": (
            {"d0", "d0err", "dz", "dzerr"},
            {"part_d0val", "part_d0err", "part_dzval", "part_dzerr"},
        ),
    }
    capabilities: dict[str, dict[str, Any]] = {}
    for capability, (needed_fields, needed_branches) in capability_requirements.items():
        missing_fields = sorted(needed_fields - fields)
        missing_branches = sorted(needed_branches - branch_set)
        capabilities[capability] = {
            "available": not missing_fields and not missing_branches,
            "raw_fields": sorted(needed_fields),
            "source_branches": sorted(needed_branches),
            "missing_raw_fields": missing_fields,
            "missing_source_branches": missing_branches,
            "evidence_sha256": canonical_sha256(
                {
                    "raw_input_schema_sha256": raw_input_schema["content_hash"],
                    "raw_fields": sorted(needed_fields),
                    "source_branches": sorted(needed_branches),
                }
            ),
        }
    capabilities.update(
        {
            "offline_teacher_checkpoint": {
                "available": True,
                "evidence_status": "scheduled_authenticated_parent_before_extraction",
                "label_access_for_target_extraction": False,
            },
            "authenticated_retb_token_parent": {
                "available": False,
                "evidence_status": "optional_parent_not_asserted_by_raw_schema",
            },
            "hlt_native_track_origin_truth": {
                "available": False,
                "missing_semantics": ["track_origin_truth_category"],
            },
            "hlt_native_production_vertex_identity": {
                "available": False,
                "missing_semantics": ["production_vertex_identity"],
            },
            "offline_reconstructed_secondary_vertices": {
                "available": False,
                "missing_semantics": [
                    "secondary_vertex_collection",
                    "secondary_vertex_mass",
                    "secondary_vertex_flight_distance",
                ],
            },
        }
    )
    return {
        "exact_current_schema": exact_current_schema,
        "schema_version": raw_input_schema.get("schema_version"),
        "raw_fields": list(raw_fields),
        "required_particle_branches": list(branches),
        "class_label_capability": {
            "available_to_label_auditor": True,
            "authenticated_source": "split_label_manifest",
            "available_to_target_builder": False,
            "copied_into_target_artifacts": False,
        },
        "forbidden_matching_field_hits": list(forbidden_hits),
        "capabilities": capabilities,
    }


def _decision(
    declaration: TargetDeclaration,
    *,
    capabilities: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    missing = tuple(
        capability
        for capability in declaration.required_capabilities
        if not bool(capabilities.get(capability, {}).get("available"))
    )
    future = declaration.campaign_status == FUTURE_ONLY
    optional_missing = declaration.campaign_status == CURRENT_OPTIONAL and bool(missing)
    admissible = not future and not missing
    implementation_status = (
        "future_data_gate"
        if future
        else "external_authenticated_parent"
        if declaration.campaign_status == CURRENT_OPTIONAL
        else "reserved_for_step_4_teacher_inference"
        if declaration.availability_class == "TEACHER_DERIVED"
        else "implemented_step_3"
    )
    row = {
        "target_id": declaration.target_id,
        "semantic_version": declaration.semantic_version,
        "availability_class": declaration.availability_class,
        "campaign_status": declaration.campaign_status,
        "admissible_current_source": admissible,
        "executable_current_source": admissible and not optional_missing,
        "required_capabilities": list(declaration.required_capabilities),
        "missing_capabilities": list(missing),
        "source_view": declaration.source_view,
        "extractor_entrypoint": declaration.extractor_entrypoint,
        "extractor_implementation_status": implementation_status,
        "label_access": False,
        "constituent_matching_required": False,
        "evidence_sha256": canonical_sha256(
            {
                "target_id": declaration.target_id,
                "required_capabilities": list(declaration.required_capabilities),
                "missing_capabilities": list(missing),
                "availability_class": declaration.availability_class,
            }
        ),
    }
    if optional_missing:
        row["not_applicable_reason"] = "missing_authenticated_optional_parent"
    if future:
        row.update(
            {
                "admissible_current_source": False,
                "executable_current_source": False,
                "future_data_gate": {
                    **FUTURE_DATA_REQUIREMENTS[declaration.target_id],
                    "new_authenticated_raw_schema_version_required": True,
                    "new_cache_contract_version_required": True,
                    "new_campaign_contract_version_required": True,
                    "repeated_capability_audit_required": True,
                    "hlt_v3_degradation_source_indices_permitted": False,
                    "offline_to_hlt_constituent_matching_permitted": False,
                },
            }
        )
    return row


def build_target_capability_audit(
    *,
    raw_input_schema: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    inspection = inspect_raw_schema(raw_input_schema)
    declarations = target_declarations()
    decisions = tuple(
        _decision(item, capabilities=inspection["capabilities"])
        for item in declarations
    )
    return with_content_hash(
        {
            "contract": TARGET_CAPABILITY_AUDIT_CONTRACT,
            "schema_version": 1,
            "source": _source(source),
            "raw_input_schema_sha256": raw_input_schema["content_hash"],
            "availability_classes": list(AVAILABILITY_CLASSES),
            "schema_inspection": inspection,
            "target_decisions": list(decisions),
            "current_required_target_count": sum(
                item.campaign_status == CURRENT_REQUIRED for item in declarations
            ),
            "current_optional_target_count": sum(
                item.campaign_status == CURRENT_OPTIONAL for item in declarations
            ),
            "future_only_target_count": sum(
                item.campaign_status == FUTURE_ONLY for item in declarations
            ),
            "all_current_required_targets_admissible": all(
                row["admissible_current_source"]
                for row in decisions
                if row["campaign_status"] == CURRENT_REQUIRED
            ),
            "label_access_for_extraction": False,
            "class_labels_in_capability_input": False,
            "constituent_matching_fields_allowed": False,
            "degradation_source_indices_are_evidence": False,
            "scientific_results_consulted": False,
        }
    )


def _component_kind(declaration: TargetDeclaration, name: str) -> str:
    if declaration.target_id == "T_HLT_REGION_PAIR_8" and name.startswith(
        "same_cluster_"
    ):
        return "binary"
    relation_prefix = next(
        (
            prefix
            for prefix in ("T_OFFLINE_RELATION_", "T_HLT_SELF_RELATION_")
            if declaration.target_id.startswith(prefix)
        ),
        None,
    )
    if relation_prefix is not None:
        family = declaration.target_id.removeprefix(relation_prefix)
        raw_name = name.split("__", 1)[0]
        raw_kind = next(
            kind
            for channel, kind, _ in RELATION_CHANNELS[family]
            if channel == raw_name
        )
        if raw_kind.endswith("_binary") or raw_kind == "categorical":
            return "bounded_frequency"
    if declaration.campaign_status == FUTURE_ONLY:
        return "future_schema_defined"
    if declaration.target_id == "T_RETB_SUMMARY_TOKENS":
        return "authenticated_parent_defined"
    return "continuous"


def _component_schema(
    declaration: TargetDeclaration,
) -> list[dict[str, Any]]:
    components = []
    groups = target_component_availability_groups(
        declaration.target_id, declaration.components
    )
    for index, name in enumerate(declaration.components):
        kind = _component_kind(declaration, name)
        continuous = kind == "continuous"
        components.append(
            {
                "index": index,
                "name": name,
                "dtype": "float32",
                "units": "dimensionless_unless_name_declares_coordinate",
                "domain": (
                    "finite_real"
                    if continuous
                    else "zero_or_one"
                    if kind == "binary"
                    else "closed_unit_interval"
                    if kind == "bounded_frequency"
                    else "resolved_by_future_or_optional_contract"
                ),
                "component_kind": kind,
                "availability_group": groups[index],
                "transform": "identity_after_registered_physical_transform",
                "clipping": (
                    "robust_normalized_to_minus12_plus12"
                    if continuous
                    else "none"
                ),
                "inverse_transform": (
                    "registered_normalizer_inverse_then_physical_inverse"
                    if continuous
                    else "identity"
                ),
                "heteroscedastic_parameterization_permitted": (
                    continuous and "HET" in declaration.allowed_parameterizations
                ),
            }
        )
    return components


def _raw_relation_channel_schema(
    declaration: TargetDeclaration,
) -> list[dict[str, Any]] | None:
    prefix = (
        "T_OFFLINE_RELATION_"
        if declaration.target_id.startswith("T_OFFLINE_RELATION_")
        else "T_HLT_SELF_RELATION_"
        if declaration.target_id.startswith("T_HLT_SELF_RELATION_")
        else None
    )
    if prefix is None:
        return None
    family = declaration.target_id.removeprefix(prefix)
    channels = RELATION_CHANNELS[family]
    return [
        {
            "index": index,
            "name": name,
            "channel_type": channel_type,
            "category_order": None if categories is None else list(categories),
            "self_pairs_included": False,
            "pair_selection": (
                "i_lt_j"
                if channel_type.startswith("unordered_pair")
                or channel_type == "pair_binary"
                else "all_valid_i_ne_j"
                if channel_type.startswith("ordered_pair")
                or channel_type == "categorical"
                else "valid_nodes"
            ),
        }
        for index, (name, channel_type, categories) in enumerate(channels)
    ]


def build_structure_target_registry(
    *,
    capability_audit: Mapping[str, Any],
    raw_input_schema: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(
        capability_audit, expected_contract=TARGET_CAPABILITY_AUDIT_CONTRACT
    )
    validate_content_hash(
        raw_input_schema, expected_contract=RAW_INPUT_SCHEMA_CONTRACT
    )
    expected_source = _source(source)
    if capability_audit.get("source") != expected_source:
        raise ValueError("capability audit source binding differs")
    if capability_audit.get("raw_input_schema_sha256") != raw_input_schema.get(
        "content_hash"
    ):
        raise ValueError("capability audit raw-schema binding differs")
    canonical_audit = build_target_capability_audit(
        raw_input_schema=raw_input_schema,
        source=expected_source,
    )
    if dict(capability_audit) != canonical_audit:
        raise ValueError("capability audit decisions differ from canonical inspection")
    if not capability_audit["schema_inspection"]["exact_current_schema"]:
        raise RuntimeError(
            "unsupported raw schema: a new authenticated schema/cache/campaign "
            "version and repeated audit are required"
        )
    if not capability_audit["all_current_required_targets_admissible"]:
        missing = sorted(
            row["target_id"]
            for row in capability_audit["target_decisions"]
            if row["campaign_status"] == CURRENT_REQUIRED
            and not row["admissible_current_source"]
        )
        raise RuntimeError(f"required current-source targets are unavailable: {missing}")

    declarations = target_declarations()
    from .extractors import build_target_extractor_manifest

    extractor_manifest = build_target_extractor_manifest()
    decisions = {
        row["target_id"]: row for row in capability_audit["target_decisions"]
    }
    if set(decisions) != {item.target_id for item in declarations}:
        raise ValueError("capability decision coverage differs from target declarations")
    targets = []
    for item in declarations:
        decision = decisions[item.target_id]
        target_family_id = (
            "T_OFFLINE_LOGITS"
            if item.target_id.startswith("T_OFFLINE_LOGITS_")
            else item.target_id
        )
        coordinate_variant = (
            item.target_id.removeprefix("T_OFFLINE_LOGITS_")
            if target_family_id == "T_OFFLINE_LOGITS"
            else None
        )
        semantic_payload = {
            "target_id": item.target_id,
            "target_family_id": target_family_id,
            "coordinate_variant": coordinate_variant,
            "semantic_version": item.semantic_version,
            "meaning": item.meaning,
            "components": list(item.components),
            "source_view": item.source_view,
            "symmetry": item.symmetry,
            "loss": item.loss,
        }
        targets.append(
            {
                "target_id": item.target_id,
                "target_family_id": target_family_id,
                "coordinate_variant": coordinate_variant,
                "semantic_version": item.semantic_version,
                "availability_class": item.availability_class,
                "campaign_status": item.campaign_status,
                "admissible_current_source": decision["admissible_current_source"],
                "executable_current_source": decision["executable_current_source"],
                "meaning": item.meaning,
                "source_view": item.source_view,
                "source_artifact": {
                    "raw_input_schema_sha256": raw_input_schema["content_hash"],
                    "teacher_or_optional_parent_binding": (
                        "required_at_cache_resolution"
                        if item.availability_class == "TEACHER_DERIVED"
                        else None
                    ),
                },
                "target_semantics_sha256": canonical_sha256(semantic_payload),
                "source_algorithm_hash": canonical_sha256(
                    {
                        "source": expected_source,
                        "extractor_entrypoint": item.extractor_entrypoint,
                        "target_semantics_sha256": canonical_sha256(
                            semantic_payload
                        ),
                    }
                ),
                "extractor_entrypoint": item.extractor_entrypoint,
                "extractor_implementation_status": decision[
                    "extractor_implementation_status"
                ],
                "physical_extractor_manifest_sha256": (
                    extractor_manifest["content_hash"]
                    if decision["extractor_implementation_status"]
                    == "implemented_step_3"
                    else None
                ),
                "shape": (
                    [len(item.components)]
                    if item.campaign_status == CURRENT_REQUIRED
                    else ["authenticated_parent_defined"]
                    if item.campaign_status == CURRENT_OPTIONAL
                    else ["future_schema_defined"]
                ),
                "component_count": (
                    len(item.components)
                    if item.campaign_status == CURRENT_REQUIRED
                    else None
                ),
                "component_names": list(item.components),
                "components": _component_schema(item),
                "raw_relation_channels": _raw_relation_channel_schema(item),
                "head_type": item.head_type,
                "symmetry": item.symmetry,
                "applicability": "target_family_registered_mask",
                "missingness_rule": (
                    "zero_storage_with_false_loss_mask_for_empty_applicable_set"
                ),
                "loss_mask": "component_availability_group_and_target_applicability",
                "event_reduction": (
                    "mean_valid_components_per_jet_then_mean_jets_with_any_valid"
                ),
                "normalizer": {
                    "population": "model_train_only",
                    "continuous_center": "finite_component_median",
                    "continuous_scale": "max((Q75-Q25)/1.349,1e-6)",
                    "normalized_clipping": [-12.0, 12.0],
                    "normalizer_hash": None,
                    "binding_status": "must_resolve_before_target_cache_publication",
                    "scale_population_rule": "refit_on_scale_train",
                },
                "allowed_parameterizations": list(item.allowed_parameterizations),
                "permitted_use_modes": [
                    "auxiliary_training",
                    *(
                        ("deployable_predicted_feedback",)
                        if item.feedback_permitted
                        else ()
                    ),
                ],
                "loss": item.loss,
                "metrics": list(item.metrics),
                "label_access": False,
                "constituent_matching_required": False,
                "feedback_permitted": item.feedback_permitted,
                "cache_contract": "hosd_target_cache_v1",
                "cache_parent_hashes": {
                    "campaign_spec": "required_at_cache_resolution",
                    "split_manifest": "required_at_cache_resolution",
                    "target_registry": "self_content_hash_after_publication",
                    "extractor_implementation": (
                        extractor_manifest["content_hash"]
                        if decision["extractor_implementation_status"]
                        == "implemented_step_3"
                        else "required_before_numeric_publication"
                    ),
                    "normalizer": "required_after_fit",
                    "teacher_checkpoint": (
                        "required_for_teacher_derived_targets"
                        if item.availability_class == "TEACHER_DERIVED"
                        else "not_applicable"
                    ),
                },
                "future_data_gate": decision.get("future_data_gate"),
                "not_applicable_reason": decision.get("not_applicable_reason"),
            }
        )
    return with_content_hash(
        {
            "contract": STRUCTURE_TARGET_REGISTRY_CONTRACT,
            "schema_version": 1,
            "source": expected_source,
            "raw_input_schema_sha256": raw_input_schema["content_hash"],
            "target_capability_audit_sha256": capability_audit["content_hash"],
            "physical_target_extractor_manifest_sha256": extractor_manifest[
                "content_hash"
            ],
            "target_order": [item.target_id for item in declarations],
            "targets": targets,
            "target_count": len(targets),
            "current_executable_target_ids": [
                row["target_id"] for row in targets if row["executable_current_source"]
            ],
            "future_unavailable_target_ids": [
                row["target_id"]
                for row in targets
                if row["campaign_status"] == FUTURE_ONLY
            ],
            "normalizer_binding_phase": (
                "registry freezes semantics now; cache manifests must resolve exact "
                "normalizer and extractor hashes before numeric publication"
            ),
            "label_access_for_extraction": False,
            "constituent_matching_fields_allowed": False,
            "scientific_results_consulted": False,
        }
    )


def validate_target_capability_audit(
    artifact: Mapping[str, Any],
    *,
    raw_input_schema: Mapping[str, Any],
    source: Mapping[str, Any],
) -> None:
    validate_content_hash(
        artifact, expected_contract=TARGET_CAPABILITY_AUDIT_CONTRACT
    )
    expected = build_target_capability_audit(
        raw_input_schema=raw_input_schema, source=source
    )
    if artifact != expected:
        raise ValueError("target capability audit differs from canonical inspection")


def validate_structure_target_registry(
    artifact: Mapping[str, Any],
    *,
    capability_audit: Mapping[str, Any],
    raw_input_schema: Mapping[str, Any],
    source: Mapping[str, Any],
) -> None:
    validate_content_hash(
        artifact, expected_contract=STRUCTURE_TARGET_REGISTRY_CONTRACT
    )
    expected = build_structure_target_registry(
        capability_audit=capability_audit,
        raw_input_schema=raw_input_schema,
        source=source,
    )
    if artifact != expected:
        raise ValueError("structure target registry differs from canonical compiler")


__all__ = [
    "EXPECTED_PARTICLE_BRANCHES",
    "EXPECTED_RAW_FIELDS",
    "FORBIDDEN_MATCHING_FIELD_FRAGMENTS",
    "FUTURE_DATA_REQUIREMENTS",
    "build_structure_target_registry",
    "build_target_capability_audit",
    "inspect_raw_schema",
    "validate_structure_target_registry",
    "validate_target_capability_audit",
]
