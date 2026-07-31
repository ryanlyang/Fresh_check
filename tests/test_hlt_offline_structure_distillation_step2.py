from __future__ import annotations

import copy

import pytest

from teacher_logit_reco.hlt_offline_structure_distillation import (
    STRUCTURE_TARGET_REGISTRY_CONTRACT,
    TARGET_CAPABILITY_AUDIT_CONTRACT,
    build_structure_target_registry,
    build_target_capability_audit,
    inspect_raw_schema,
    validate_content_hash,
    validate_structure_target_registry,
    validate_target_capability_audit,
    with_content_hash,
)
from teacher_logit_reco.hlt_offline_structure_distillation.target_schemas import (
    RELATION_COMPONENTS,
    target_declarations,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (
    build_raw_input_schema,
)


def _source(*, status: str = "b" * 64) -> dict[str, object]:
    return {
        "commit": "a" * 40,
        "status_sha256": status,
        "dirty": status != "b" * 64,
        "status_hash_policy": (
            "git_diff_binary_HEAD_plus_sorted_untracked_file_bytes_v2"
        ),
    }


def _compile() -> tuple[dict, dict, dict]:
    raw = build_raw_input_schema()
    audit = build_target_capability_audit(
        raw_input_schema=raw,
        source=_source(),
    )
    registry = build_structure_target_registry(
        capability_audit=audit,
        raw_input_schema=raw,
        source=_source(),
    )
    return raw, audit, registry


def _target(artifact: dict, target_id: str) -> dict:
    return next(row for row in artifact["targets"] if row["target_id"] == target_id)


def _decision(artifact: dict, target_id: str) -> dict:
    return next(
        row for row in artifact["target_decisions"] if row["target_id"] == target_id
    )


def test_current_jetclass_capabilities_are_authenticated_and_label_blind() -> None:
    raw, audit, registry = _compile()
    assert validate_content_hash(
        audit, expected_contract=TARGET_CAPABILITY_AUDIT_CONTRACT
    )
    assert validate_content_hash(
        registry, expected_contract=STRUCTURE_TARGET_REGISTRY_CONTRACT
    )
    inspection = audit["schema_inspection"]
    assert inspection["exact_current_schema"]
    assert inspection["forbidden_matching_field_hits"] == []
    for capability in (
        "particle_four_vector",
        "charge",
        "pid_categories",
        "track_measurements",
    ):
        assert inspection["capabilities"][capability]["available"]
    assert audit["all_current_required_targets_admissible"]
    assert audit["label_access_for_extraction"] is False
    assert audit["class_labels_in_capability_input"] is False
    assert (
        inspection["class_label_capability"]["available_to_target_builder"]
        is False
    )
    assert registry["label_access_for_extraction"] is False
    assert raw["content_hash"] == registry["raw_input_schema_sha256"]
    assert len(registry["current_executable_target_ids"]) == 31


def test_truth_and_literal_vertex_targets_fail_closed_behind_future_gates() -> None:
    _, audit, registry = _compile()
    expected = {
        "T_HLT_TRACK_ORIGIN_TRUTH",
        "T_HLT_COMMON_VERTEX_TRUTH",
        "T_OFFLINE_SECONDARY_VERTEX_SET",
    }
    assert set(registry["future_unavailable_target_ids"]) == expected
    for target_id in expected:
        row = _decision(audit, target_id)
        assert row["availability_class"] == "UNAVAILABLE"
        assert row["campaign_status"] == "future_only"
        assert row["admissible_current_source"] is False
        assert row["executable_current_source"] is False
        gate = row["future_data_gate"]
        assert gate["new_authenticated_raw_schema_version_required"]
        assert gate["new_cache_contract_version_required"]
        assert gate["new_campaign_contract_version_required"]
        assert gate["repeated_capability_audit_required"]
        assert gate["hlt_v3_degradation_source_indices_permitted"] is False
        assert gate["offline_to_hlt_constituent_matching_permitted"] is False
        registry_row = _target(registry, target_id)
        assert registry_row["future_data_gate"] == gate


def test_current_target_availability_classes_match_scientific_meaning() -> None:
    _, audit, _ = _compile()
    assert _decision(audit, "T_OFFLINE_JET_10")["availability_class"] == (
        "OFFLINE_RECO_DERIVED"
    )
    assert _decision(audit, "T_OFFLINE_CA_TREE_26")["availability_class"] == (
        "OFFLINE_RECO_DERIVED"
    )
    assert _decision(audit, "T_OFFLINE_TRACK_COMPONENT_PROXY_17")[
        "availability_class"
    ] == "DETERMINISTIC_PROXY"
    assert _decision(audit, "T_HLT_SELF_TRACK_32")["availability_class"] == (
        "HLT_RECO_DERIVED"
    )
    assert _decision(audit, "T_HLT_TRACK_PAIR_13")["availability_class"] == (
        "HLT_RECO_DERIVED"
    )
    assert _decision(audit, "T_OFFLINE_LOGITS_O_BASE")[
        "availability_class"
    ] == "TEACHER_DERIVED"


@pytest.mark.parametrize(
    "field",
    (
        "offline_source_index",
        "hlt_to_offline_constituent_match",
        "hungarian_assignment",
    ),
)
def test_matching_and_degradation_lineage_fields_are_rejected(field: str) -> None:
    raw = build_raw_input_schema()
    changed = dict(raw)
    changed.pop("content_hash")
    changed["raw_fields"] = [*changed["raw_fields"], field]
    changed["raw_dimension"] = len(changed["raw_fields"])
    changed = with_content_hash(changed)
    with pytest.raises(ValueError, match="fields are forbidden"):
        inspect_raw_schema(changed)


def test_same_contract_schema_drift_cannot_compile_current_registry() -> None:
    raw = build_raw_input_schema()
    changed = dict(raw)
    changed.pop("content_hash")
    changed["raw_fields"] = [
        name for name in changed["raw_fields"] if name != "d0err"
    ]
    changed["raw_dimension"] = len(changed["raw_fields"])
    changed = with_content_hash(changed)
    audit = build_target_capability_audit(
        raw_input_schema=changed,
        source=_source(),
    )
    assert not audit["schema_inspection"]["exact_current_schema"]
    assert not _decision(audit, "T_OFFLINE_TRACK_32")[
        "admissible_current_source"
    ]
    with pytest.raises(RuntimeError, match="new authenticated schema"):
        build_structure_target_registry(
            capability_audit=audit,
            raw_input_schema=changed,
            source=_source(),
        )


def test_target_registry_materializes_exact_dimensions_and_relation_order() -> None:
    _, _, registry = _compile()
    expected = {
        "T_OFFLINE_JET_10": 10,
        "T_OFFLINE_COMPOSITION_16": 16,
        "T_OFFLINE_TRACK_32": 32,
        "T_OFFLINE_DENSITY_22": 22,
        "T_OFFLINE_CA_TREE_26": 26,
        "T_OFFLINE_TRACK_COMPONENT_PROXY_17": 17,
        "T_HLT_TRACK_PAIR_13": 13,
        "T_HLT_REGION_PAIR_8": 8,
    }
    for target_id, dimension in expected.items():
        row = _target(registry, target_id)
        assert row["shape"] == [dimension]
        assert row["component_count"] == dimension
        assert len(row["component_names"]) == dimension
        assert len(row["component_names"]) == len(set(row["component_names"]))
    jet = _target(registry, "T_OFFLINE_JET_10")
    assert [
        jet["components"][index]["availability_group"]
        for index in (1, 2, 3)
    ] == ["jet_direction"] * 3
    track = _target(registry, "T_OFFLINE_TRACK_32")
    assert {
        component["availability_group"] for component in track["components"][:4]
    } == {"track_availability_observation"}
    assert {
        component["availability_group"] for component in track["components"][4:]
    } == {"has_valid_track"}

    for family, components in RELATION_COMPONENTS.items():
        for prefix in ("T_OFFLINE_RELATION_", "T_HLT_SELF_RELATION_"):
            row = _target(registry, f"{prefix}{family}")
            assert row["component_names"] == list(components)
            assert row["component_count"] == len(components)
            assert row["raw_relation_channels"]
            assert [channel["index"] for channel in row["raw_relation_channels"]] == list(
                range(len(row["raw_relation_channels"]))
            )
    region = _target(registry, "T_OFFLINE_RELATION_REGION")
    assert region["component_count"] == 193
    assert [
        row["channel_type"] for row in region["raw_relation_channels"][:3]
    ] == ["pair_binary"] * 3
    assert [
        row["channel_type"] for row in region["raw_relation_channels"][3:8]
    ] == ["unordered_pair_continuous"] * 5
    pid = _target(registry, "T_OFFLINE_RELATION_PID")
    assert pid["allowed_parameterizations"] == ["ABS", "RES"]
    assert not any(
        component["heteroscedastic_parameterization_permitted"]
        for component in pid["components"]
    )
    charge = _target(registry, "T_OFFLINE_RELATION_CHARGE")
    assert all(
        component["heteroscedastic_parameterization_permitted"]
        for component in charge["components"][:20]
    )
    assert not any(
        component["heteroscedastic_parameterization_permitted"]
        for component in charge["components"][20:]
    )


def test_registry_is_declarative_until_step3_and_requires_hash_resolution() -> None:
    _, _, registry = _compile()
    for row in registry["targets"]:
        if row["physical_extractor_manifest_sha256"] is not None:
            assert row["extractor_implementation_status"] == "implemented_step_3"
            assert (
                row["cache_parent_hashes"]["extractor_implementation"]
                == row["physical_extractor_manifest_sha256"]
            )
        elif row["availability_class"] == "TEACHER_DERIVED":
            assert row["extractor_implementation_status"] in {
                "reserved_for_step_4_teacher_inference",
                "external_authenticated_parent",
            }
        else:
            assert row["extractor_implementation_status"] == "future_data_gate"
        assert len(row["target_semantics_sha256"]) == 64
        assert len(row["source_algorithm_hash"]) == 64
        assert row["label_access"] is False
        assert row["constituent_matching_required"] is False
        assert row["normalizer"]["normalizer_hash"] is None
        assert (
            row["normalizer"]["binding_status"]
            == "must_resolve_before_target_cache_publication"
        )


def test_capability_and_registry_are_deterministic_tamper_evident_and_source_bound() -> None:
    raw, audit, registry = _compile()
    second_audit = build_target_capability_audit(
        raw_input_schema=raw, source=_source()
    )
    second_registry = build_structure_target_registry(
        capability_audit=second_audit,
        raw_input_schema=raw,
        source=_source(),
    )
    assert audit == second_audit
    assert registry == second_registry
    validate_target_capability_audit(
        audit, raw_input_schema=raw, source=_source()
    )
    validate_structure_target_registry(
        registry,
        capability_audit=audit,
        raw_input_schema=raw,
        source=_source(),
    )

    changed = copy.deepcopy(audit)
    changed["target_decisions"][0]["label_access"] = True
    with pytest.raises(ValueError, match="content hash mismatch"):
        validate_target_capability_audit(
            changed, raw_input_schema=raw, source=_source()
        )
    rehashed = copy.deepcopy(audit)
    rehashed.pop("content_hash")
    rehashed["target_decisions"][-1]["availability_class"] = "AUTHENTIC_TRUTH"
    rehashed = with_content_hash(rehashed)
    with pytest.raises(ValueError, match="canonical inspection"):
        build_structure_target_registry(
            capability_audit=rehashed,
            raw_input_schema=raw,
            source=_source(),
        )
    with pytest.raises(ValueError, match="source binding differs"):
        build_structure_target_registry(
            capability_audit=audit,
            raw_input_schema=raw,
            source=_source(status="c" * 64),
        )


def test_declaration_coverage_is_exact_and_optional_retb_stays_nonexecutable() -> None:
    _, audit, registry = _compile()
    declarations = target_declarations()
    assert registry["target_count"] == len(declarations) == 35
    assert registry["target_order"] == [row.target_id for row in declarations]
    for teacher_id in ("O_BASE", "O_FULLREL"):
        row = _target(registry, f"T_OFFLINE_LOGITS_{teacher_id}")
        assert row["target_family_id"] == "T_OFFLINE_LOGITS"
        assert row["coordinate_variant"] == teacher_id
    retb_decision = _decision(audit, "T_RETB_SUMMARY_TOKENS")
    assert retb_decision["availability_class"] == "TEACHER_DERIVED"
    assert retb_decision["campaign_status"] == "current_optional"
    assert retb_decision["executable_current_source"] is False
    assert (
        retb_decision["not_applicable_reason"]
        == "missing_authenticated_optional_parent"
    )


def test_step2_wrapper_uses_campaign_source_and_parent_lock() -> None:
    script = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "scripts"
        / "audit_hosd_target_capability.py"
    ).read_text(encoding="utf-8")
    assert "load_and_validate_campaign" in script
    assert "resolved_inherited_parent_lock.json" in script
    assert "require_parents_ready(parents, before_stage=\"B\")" in script
    assert "write_immutable_json" in script
