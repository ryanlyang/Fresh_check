from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from teacher_logit_reco.local_particle_residual_field.particle_view import (
    LOCKED_PARTICLE_VIEW_500K_SPLIT_CONFIG,
    PARTICLE_VIEW_TRAINABLE_COMPONENTS,
    LabelExposureRecord,
    ParticleViewRunSpec,
    SELECTED_VIEW_MATERIALIZATION_POLICY,
    audit_unified_split_manifest,
    build_label_exposure_ledger,
    build_particle_view_deployment_manifest,
    build_particle_view_registry,
    build_unified_split_manifest,
    build_view_coordinate_binding,
    load_hashed_json,
    logical_split_identities,
    miniature_parent_manifest,
    miniature_split_config,
    validate_label_exposure_ledger,
    validate_particle_view_deployment_manifest,
    validate_particle_view_registry,
    validate_view_coordinate_binding,
    with_content_hash,
    write_immutable_json,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _rehash(payload: dict) -> dict:
    unhashed = deepcopy(payload)
    unhashed.pop("content_hash", None)
    return with_content_hash(unhashed)


def _miniature():
    parent = miniature_parent_manifest(rows_per_class=4)
    config = miniature_split_config(rows_per_class=4)
    unified = build_unified_split_manifest(parent, config=config)
    return parent, config, unified


def _coordinate_binding(unified: dict) -> dict:
    parent_hashes = {
        name: _sha(name)
        for name in (
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
    }
    parent_hashes["unified_split_manifest_sha256"] = unified["content_hash"]
    parent_hashes["train_identity_sha256"] = unified["logical_splits"]["train"][
        "ordered_identity_sha256"
    ]
    definition = {
        "offline_tap_layer": "penultimate_transformer_block",
        "offline_tap_tensor_location": "after_residual_before_pooling",
        "cross_attention_config_sha256": _sha("cross-attention"),
        "pair_feature_schema_sha256": _sha("pair-features"),
        "centering_policy": "masked_per_jet_mean_v1",
        "bounded_coordinate_policy": "tanh_then_center_v1",
        "rate_budget_policy": "variance_covariance_quantization_v1",
        "null_token_policy": "learned_masked_null_token_v1",
        "bottleneck_width": 4,
    }
    return build_view_coordinate_binding(
        parent_hashes=parent_hashes,
        coordinate_definition=definition,
    )


def test_locked_inventory_is_one_500k_training_pool() -> None:
    config = LOCKED_PARTICLE_VIEW_500K_SPLIT_CONFIG
    assert config.train_count == 500_000
    assert config.model_val_count == 150_000
    assert config.stack_val_count == 150_000
    assert config.final_test_count == 150_000
    assert config.unused_parent_splits == ("stack_train",)
    assert config.to_payload()["training_topology"] == "single_pool_no_crossfit_v1"


def test_unified_manifest_has_disjoint_deterministic_children_and_one_train() -> None:
    parent, config, unified = _miniature()
    audit = audit_unified_split_manifest(unified, parent=parent, config=config)
    assert audit["single_training_pool"] is True
    assert audit["cross_fit"] is False
    assert audit["unused_parent_splits"] == ["stack_train"]
    assert audit["trainable_component_count"] == len(
        PARTICLE_VIEW_TRAINABLE_COMPONENTS
    )

    logical = unified["logical_splits"]
    assert set(logical) == {
        "train",
        "model_val_stop",
        "model_val_select",
        "stack_val",
        "final_test",
    }
    assert logical["train"]["membership_kind"] == "complete_parent_alias"
    assert "parent_row_indices" not in logical["train"]
    stop = set(logical["model_val_stop"]["parent_row_indices"])
    select = set(logical["model_val_select"]["parent_row_indices"])
    assert not stop & select
    assert stop | select == set(range(config.model_val_count))
    assert logical["model_val_stop"]["count"] == config.model_val_count // 2
    assert logical["model_val_select"]["count"] == config.model_val_count // 2

    expected_binding = {
        "logical_split": "train",
        "train_split_sha256": logical["train"]["content_hash"],
        "train_identity_sha256": logical["train"]["ordered_identity_sha256"],
    }
    assert unified["component_training_bindings"] == {
        component: expected_binding
        for component in PARTICLE_VIEW_TRAINABLE_COMPONENTS
    }
    serialized = json.dumps(unified, sort_keys=True).lower()
    for forbidden in ("train_consumer", "train_distill"):
        assert forbidden not in serialized
    assert unified["training_topology"]["fold_assignments"] is False


def test_validation_membership_is_identity_deterministic_under_parent_reorder() -> None:
    parent, config, unified = _miniature()
    reordered = deepcopy(parent)
    reordered.splits["model_val"] = list(
        reversed(reordered.splits["model_val"])
    )
    reordered_unified = build_unified_split_manifest(reordered, config=config)
    for split in ("model_val_stop", "model_val_select"):
        first = {
            identity.key()
            for identity in logical_split_identities(
                unified, parent=parent, split_name=split, config=config
            )
        }
        second = {
            identity.key()
            for identity in logical_split_identities(
                reordered_unified,
                parent=reordered,
                split_name=split,
                config=config,
            )
        }
        assert first == second


def test_unified_manifest_round_trips_sorted_json_and_fails_closed(tmp_path: Path) -> None:
    parent, config, unified = _miniature()
    path = tmp_path / "unified.json"
    write_immutable_json(path, unified)
    loaded = load_hashed_json(path)
    assert audit_unified_split_manifest(
        loaded, parent=parent, config=config
    )["ok"]

    hidden_partition = deepcopy(unified)
    hidden_partition["training_topology"]["training_subpartitions"] = [
        "train_consumer",
        "train_distill",
    ]
    with pytest.raises(ValueError, match="one unsplit pool"):
        audit_unified_split_manifest(
            _rehash(hidden_partition), parent=parent, config=config
        )

    wrong_binding = deepcopy(unified)
    wrong_binding["component_training_bindings"]["Cview_clean"][
        "train_identity_sha256"
    ] = _sha("different-train")
    with pytest.raises(ValueError, match="does not use the unified train pool"):
        audit_unified_split_manifest(
            _rehash(wrong_binding), parent=parent, config=config
        )

    changed_partition_seed = deepcopy(unified)
    stop = changed_partition_seed["logical_splits"]["model_val_stop"]
    stop["partition_seed"] += 1
    changed_partition_seed["logical_splits"]["model_val_stop"] = _rehash(stop)
    with pytest.raises(ValueError, match="deterministic membership changed"):
        audit_unified_split_manifest(
            _rehash(changed_partition_seed), parent=parent, config=config
        )

    stale_parent = deepcopy(parent)
    stale_parent.splits["model_train"] = list(
        reversed(stale_parent.splits["model_train"])
    )
    with pytest.raises(ValueError, match="stale parent"):
        audit_unified_split_manifest(
            unified, parent=stale_parent, config=config
        )


def test_coordinate_binding_is_canonical_float32_and_preconsumer() -> None:
    _, _, unified = _miniature()
    binding = _coordinate_binding(unified)
    assert validate_view_coordinate_binding(binding) == binding["content_hash"]
    assert (
        binding["materialization"]
        == SELECTED_VIEW_MATERIALIZATION_POLICY
    )
    assert binding["materialization"]["dtype"] == "float32"
    assert binding["materialization"]["byte_order"] == "little"
    serialized = json.dumps(binding, sort_keys=True).lower()
    assert "consumer_checkpoint" not in serialized
    assert "target_logits" not in serialized

    parents = dict(binding["parents"])
    parents["consumer_checkpoint_sha256"] = _sha("consumer")
    with pytest.raises(ValueError, match="parent.?hash inventory"):
        build_view_coordinate_binding(
            parent_hashes=parents,
            coordinate_definition=binding["coordinate_definition"],
        )

    changed = deepcopy(binding)
    changed["materialization"]["dtype"] = "float16"
    with pytest.raises(ValueError, match="materialization policy"):
        validate_view_coordinate_binding(_rehash(changed))


def test_registry_records_selectability_seed_scope_and_single_train_identity() -> None:
    _, _, unified = _miniature()
    train_hash = unified["logical_splits"]["train"]["ordered_identity_sha256"]
    registry = build_particle_view_registry(
        unified_split_manifest_sha256=unified["content_hash"],
        train_identity_sha256=train_hash,
        run_specs=[
            ParticleViewRunSpec(
                run_id="a0_screen",
                stage="baseline",
                scientific_role="matched_hlt_baseline",
                selection_family="pre_stage_g_deployable",
                selectable=True,
            ),
            ParticleViewRunSpec(
                run_id="view_confirmation",
                stage="confirmation",
                scientific_role="selected_privileged_replica",
                selection_family="privileged_scientific",
                seed_ids=(101, 202, 303),
                parent_run_ids=("a0_screen",),
                selectable=True,
                clean_consumer_paired=True,
                robust_consumer_paired=True,
                stack_val_eligible=True,
                final_test_eligible=True,
            ),
            ParticleViewRunSpec(
                run_id="shuffle_diagnostic",
                stage="confirmation",
                scientific_role="structural_shuffle",
                selection_family="diagnostic",
                parent_run_ids=("a0_screen",),
                diagnostic=True,
            ),
        ],
    )
    audit = validate_particle_view_registry(registry)
    assert audit["single_training_pool"] is True
    assert audit["train_identity_sha256"] == train_hash
    by_id = {row["run_id"]: row for row in registry["runs"]}
    assert by_id["a0_screen"]["single_seed_screen"] is True
    assert by_id["view_confirmation"]["three_seed_confirmation"] is True
    assert by_id["shuffle_diagnostic"]["diagnostic"] is True

    with pytest.raises(ValueError, match="cross-fit|unified train pool"):
        build_particle_view_registry(
            unified_split_manifest_sha256=unified["content_hash"],
            train_identity_sha256=train_hash,
            run_specs=[
                ParticleViewRunSpec(
                    run_id="forbidden_crossfit",
                    stage="predictor",
                    scientific_role="forbidden",
                    selection_family="privileged_scientific",
                    train_split="train_distill",
                )
            ],
        )


def test_deployment_schema_is_exactly_hlt_only() -> None:
    _, _, unified = _miniature()
    train_hash = unified["logical_splits"]["train"]["ordered_identity_sha256"]
    deployment = build_particle_view_deployment_manifest(
        bundle_id="particle_view_candidate",
        bundle_kind="frozen_consumer",
        winner_family="pre_stage_g_deployable",
        unified_split_manifest_sha256=unified["content_hash"],
        train_identity_sha256=train_hash,
        predictor_config_sha256=_sha("predictor-config"),
        predictor_checkpoint_sha256=_sha("predictor"),
        consumer_config_sha256=_sha("consumer-config"),
        consumer_checkpoint_sha256=_sha("consumer"),
        hlt_preprocessing_sha256=_sha("hlt-preprocessing"),
        hlt_schema_sha256=_sha("hlt-schema"),
        view_normalizer_sha256=_sha("normalizer"),
        coordinate_binding_sha256=_sha("coordinate"),
        resource_profile_sha256=_sha("resources"),
        source_commit="a" * 40,
        bottleneck_width=4,
    )
    audit = validate_particle_view_deployment_manifest(deployment)
    assert audit["hlt_only"] is True
    assert deployment["oracle_dependencies"] == []
    assert deployment["requires_oracle"] is False
    assert all(name.startswith("hlt_") for name in deployment["required_inputs"])

    extra_oracle = deepcopy(deployment)
    extra_oracle["offline_teacher_checkpoint_sha256"] = _sha("offline")
    with pytest.raises(ValueError, match="schema mismatch"):
        validate_particle_view_deployment_manifest(_rehash(extra_oracle))

    oracle_dependency = deepcopy(deployment)
    oracle_dependency["oracle_dependencies"] = ["offline_teacher"]
    with pytest.raises(ValueError, match="oracle dependencies"):
        validate_particle_view_deployment_manifest(_rehash(oracle_dependency))

    requires_oracle = deepcopy(deployment)
    requires_oracle["requires_oracle"] = True
    with pytest.raises(ValueError, match="cannot require an oracle"):
        validate_particle_view_deployment_manifest(_rehash(requires_oracle))


def test_label_exposure_ledger_reconciles_only_the_unified_train() -> None:
    _, _, unified = _miniature()
    train_hash = unified["logical_splits"]["train"]["ordered_identity_sha256"]
    ledger = build_label_exposure_ledger(
        unified_split_manifest=unified,
        pipeline_id="selected_privileged_scientific",
        records=[
            LabelExposureRecord(
                run_id="selected_pipeline",
                component="Cview_clean",
                stage="representation",
                seed=101,
                train_identity_sha256=train_hash,
                optimizer_steps=100,
                label_bearing_steps=80,
                labeled_examples_processed=3200,
                ce_bearing_steps=80,
                teacher_kd_steps=20,
                view_supervision_steps=0,
                training_flops=123_000,
                retained_in_deployable_path=True,
            ),
            LabelExposureRecord(
                run_id="selected_pipeline",
                component="Pview",
                stage="predictor",
                seed=101,
                train_identity_sha256=train_hash,
                optimizer_steps=50,
                label_bearing_steps=25,
                labeled_examples_processed=1000,
                ce_bearing_steps=25,
                teacher_kd_steps=50,
                view_supervision_steps=50,
                training_flops=456_000,
                retained_in_deployable_path=True,
            ),
        ],
    )
    audit = validate_label_exposure_ledger(
        ledger, unified_split_manifest=unified
    )
    assert audit["record_count"] == 2
    assert audit["totals_all_training"]["optimizer_steps"] == 150
    assert audit["totals_all_training"]["training_flops"] == 579_000

    with pytest.raises(ValueError, match="different train identity"):
        build_label_exposure_ledger(
            unified_split_manifest=unified,
            pipeline_id="bad",
            records=[
                LabelExposureRecord(
                    run_id="bad",
                    component="Pview",
                    stage="predictor",
                    seed=101,
                    train_identity_sha256=_sha("wrong"),
                    optimizer_steps=1,
                    label_bearing_steps=0,
                    labeled_examples_processed=0,
                    ce_bearing_steps=0,
                    teacher_kd_steps=1,
                    view_supervision_steps=1,
                    training_flops=1,
                    retained_in_deployable_path=False,
                )
            ],
        )


def test_immutable_publication_and_debug_cli(tmp_path: Path) -> None:
    _, _, unified = _miniature()
    immutable = tmp_path / "immutable.json"
    first = write_immutable_json(immutable, unified)
    second = write_immutable_json(immutable, unified)
    assert first["status"] == "published"
    assert second["status"] == "already_present"
    changed = deepcopy(unified)
    changed["campaign"] = "different"
    with pytest.raises(FileExistsError):
        write_immutable_json(immutable, _rehash(changed))

    output = tmp_path / "prepared"
    command = [
        sys.executable,
        "scripts/prepare_particle_view_campaign.py",
        "--debug-miniature",
        "--debug-rows-per-class",
        "4",
        "--output-dir",
        str(output),
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(completed.stdout)
    assert summary["status"] == "READY"
    assert (output / "unified_split_manifest.json").is_file()
    assert (output / "particle_view_step1_schema_catalog.json").is_file()
    report = load_hashed_json(output / "particle_view_step1_report.json")
    assert report["single_training_pool"] is True
    assert report["cross_fit"] is False
    assert report["production_eligible"] is False

    dry_output = tmp_path / "dry"
    dry = subprocess.run(
        [*command[:-1], str(dry_output), "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
    )
    dry_summary = json.loads(dry.stdout)
    assert dry_summary["manifest"]["status"] == "dry_run"
    assert not dry_output.exists()
