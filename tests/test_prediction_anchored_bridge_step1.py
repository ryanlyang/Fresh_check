from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

from jetclass_fresh.jetclass_data import (
    FILE_PREFIX_TO_LABEL,
    LABEL_NAMES,
    SPLIT_ORDER,
    JetIdentity,
    SplitManifest,
    save_split_manifest,
)
from teacher_logit_reco.local_particle_residual_field.bridge_campaign import (
    MEASUREMENT_UNMEASURED,
    NORMAL_PILOT_BUDGET_BYTES,
    PAIRED_SEED_IDS,
    POST_TEACHER_CONFIGURATION_COUNT,
    RECONSTRUCTION_BREADTH_COUNT,
    REGISTRY_CONFIGURATION_COUNT,
    RETAINED_STATE_RULE,
    build_campaign_registry,
    build_provisional_storage_projection,
    build_step1_report,
    dense_field_cache_projection,
    provisional_storage_categories,
    record_registry_measurements,
    require_production_ready,
    resolve_registry_run,
    validate_campaign_registry,
    write_step1_artifacts,
)
from teacher_logit_reco.local_particle_residual_field.bridge_contracts import (
    canonical_sha256,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.local_particle_residual_field.bridge_splits import (
    ChildSplitSpec,
    LOCKED_PILOT_SPLIT_CONFIG,
    ParentPartitionSpec,
    PredictionAnchoredSplitConfig,
    audit_child_split_manifest,
    authorize_manifest_split_access,
    authorize_split_access,
    build_child_split_manifest,
    build_manifest_validation_unlock,
    build_validation_unlock,
    claim_split_access,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _mini_parent(*, reverse_model_val: bool = False) -> SplitManifest:
    splits: dict[str, list[JetIdentity]] = {}
    for split_index, split in enumerate(SPLIT_ORDER):
        rows = [
            JetIdentity(
                file=f"{split}/class-{label}.root",
                entry=split_index * 1000 + label * 10 + local_index,
                label=label,
            )
            for label in range(len(LABEL_NAMES))
            for local_index in range(2)
        ]
        if split == "model_val" and reverse_model_val:
            rows.reverse()
        splits[split] = rows
    return SplitManifest(
        data_dir="/synthetic/jetclass",
        max_constits=128,
        class_names=list(LABEL_NAMES),
        file_prefix_to_label=dict(FILE_PREFIX_TO_LABEL),
        split_sizes={split: 20 for split in SPLIT_ORDER},
        split_seeds={split: 100 + index for index, split in enumerate(SPLIT_ORDER)},
        file_records=[],
        splits=splits,
        metadata={"file_level_separation_claimed": False},
    )


def _mini_config() -> PredictionAnchoredSplitConfig:
    return PredictionAnchoredSplitConfig(
        contract="prediction_anchored_split_config_v1_test",
        parent_split_counts=tuple((split, 20) for split in SPLIT_ORDER),
        partitions=(
            ParentPartitionSpec(
                "stack_train",
                810_101,
                (
                    ChildSplitSpec("stack_train_consumer", 10, "consumer_training"),
                    ChildSplitSpec("stack_train_distill", 10, "reconstructor_training"),
                ),
            ),
            ParentPartitionSpec(
                "model_val",
                810_202,
                (
                    ChildSplitSpec("model_val_stop", 10, "checkpoint_selection"),
                    ChildSplitSpec("model_val_select", 10, "configuration_selection"),
                ),
            ),
            ParentPartitionSpec(
                "stack_val",
                810_303,
                (
                    ChildSplitSpec(
                        "stack_val_consumer",
                        10,
                        "consumer_confirmation",
                        "consumer_preconfirmation",
                    ),
                    ChildSplitSpec(
                        "stack_val_deploy",
                        10,
                        "deployable_confirmation",
                        "deployable_preconfirmation",
                    ),
                ),
            ),
        ),
    )


def test_locked_split_contract_has_exact_production_counts() -> None:
    assert dict(LOCKED_PILOT_SPLIT_CONFIG.parent_split_counts) == {
        "model_train": 500_000,
        "model_val": 150_000,
        "stack_train": 500_000,
        "stack_val": 150_000,
        "final_test": 150_000,
    }
    children = {
        child.name: child.count
        for partition in LOCKED_PILOT_SPLIT_CONFIG.partitions
        for child in partition.children
    }
    assert children == {
        "stack_train_consumer": 250_000,
        "stack_train_distill": 250_000,
        "model_val_stop": 75_000,
        "model_val_select": 75_000,
        "stack_val_consumer": 75_000,
        "stack_val_deploy": 75_000,
    }


def test_child_splits_are_deterministic_balanced_disjoint_and_complete() -> None:
    parent = _mini_parent()
    config = _mini_config()
    first = build_child_split_manifest(parent, config=config)
    second = build_child_split_manifest(parent, config=config)
    assert first == second
    audit = audit_child_split_manifest(first, parent=parent, config=config)
    assert audit["ok"] is True
    assert audit["coverage_complete"] is True
    assert audit["overlap_count"] == 0

    for partition in config.partitions:
        child_sets = [
            set(first["children"][child.name]["parent_row_indices"])
            for child in partition.children
        ]
        assert child_sets[0].isdisjoint(child_sets[1])
        assert child_sets[0] | child_sets[1] == set(range(20))
        for child in partition.children:
            payload = first["children"][child.name]
            assert payload["count"] == 10
            assert set(payload["class_counts"].values()) == {1}


def test_child_manifest_rejects_reordered_or_tampered_parent_membership() -> None:
    config = _mini_config()
    parent = _mini_parent()
    children = build_child_split_manifest(parent, config=config)
    reordered = _mini_parent(reverse_model_val=True)
    with pytest.raises(ValueError, match="stale or reordered parent"):
        audit_child_split_manifest(children, parent=reordered, config=config)

    rebuilt = build_child_split_manifest(reordered, config=config)
    assert rebuilt["parent_manifest_sha256"] != children["parent_manifest_sha256"]
    for name in ("model_val_stop", "model_val_select"):
        old_ids = {
            parent.splits["model_val"][index].key()
            for index in children["children"][name]["parent_row_indices"]
        }
        new_ids = {
            reordered.splits["model_val"][index].key()
            for index in rebuilt["children"][name]["parent_row_indices"]
        }
        assert new_ids == old_ids

    stale = deepcopy(parent)
    stale.metadata["source_revision"] = "changed"
    with pytest.raises(ValueError, match="stale or reordered parent"):
        audit_child_split_manifest(children, parent=stale, config=config)

    tampered = deepcopy(children)
    tampered_child = dict(tampered["children"]["model_val_select"])
    tampered_child.pop("content_hash")
    tampered_child["parent_row_indices"][0] = children["children"][
        "model_val_stop"
    ]["parent_row_indices"][0]
    tampered["children"]["model_val_select"] = with_content_hash(tampered_child)
    tampered.pop("content_hash")
    tampered = with_content_hash(tampered)
    with pytest.raises(ValueError):
        audit_child_split_manifest(tampered, parent=parent, config=config)


def test_parent_count_and_stratification_mismatches_fail_closed() -> None:
    parent = _mini_parent()
    parent.splits["stack_train"] = parent.splits["stack_train"][:-1]
    with pytest.raises(ValueError, match="count mismatch"):
        build_child_split_manifest(parent, config=_mini_config())

    parent = _mini_parent()
    parent.splits["stack_train"][0] = JetIdentity(
        file="stack_train/relabelled.root", entry=9999, label=1
    )
    with pytest.raises(ValueError, match="not exactly class balanced"):
        build_child_split_manifest(parent, config=_mini_config())


def test_validation_access_is_purpose_locked_and_confirmation_is_one_shot(
    tmp_path: Path,
) -> None:
    parent_hash = "a" * 64
    split_hash = "b" * 64
    selection_hash = "c" * 64
    with pytest.raises(PermissionError, match="permits purpose"):
        authorize_split_access(
            split_name="model_val_stop",
            purpose="configuration_selection",
            parent_manifest_sha256=parent_hash,
            bound_split_sha256=split_hash,
        )
    with pytest.raises(PermissionError, match="sealed"):
        authorize_split_access(
            split_name="stack_val_consumer",
            purpose="consumer_confirmation",
            parent_manifest_sha256=parent_hash,
            bound_split_sha256=split_hash,
        )

    unlock = build_validation_unlock(
        split_name="stack_val_consumer",
        parent_manifest_sha256=parent_hash,
        bound_split_sha256=split_hash,
        selection_sha256=selection_hash,
    )
    authorization = authorize_split_access(
        split_name="stack_val_consumer",
        purpose="consumer_confirmation",
        parent_manifest_sha256=parent_hash,
        bound_split_sha256=split_hash,
        unlock=unlock,
    )
    receipt = tmp_path / "stack_val_consumer.claim.json"
    claim_split_access(receipt, authorization)
    with pytest.raises(PermissionError, match="already claimed"):
        claim_split_access(receipt, authorization)

    wrong_unlock = build_validation_unlock(
        split_name="stack_val_consumer",
        parent_manifest_sha256=parent_hash,
        bound_split_sha256="d" * 64,
        selection_sha256=selection_hash,
    )
    with pytest.raises(PermissionError, match="does not authorize"):
        authorize_split_access(
            split_name="stack_val_consumer",
            purpose="consumer_confirmation",
            parent_manifest_sha256=parent_hash,
            bound_split_sha256=split_hash,
            unlock=wrong_unlock,
        )

    parent = _mini_parent()
    config = _mini_config()
    child_manifest = build_child_split_manifest(parent, config=config)
    manifest_unlock = build_manifest_validation_unlock(
        child_manifest,
        split_name="stack_val_deploy",
        selection_sha256=selection_hash,
    )
    manifest_authorization = authorize_manifest_split_access(
        child_manifest,
        parent=parent,
        split_name="stack_val_deploy",
        purpose="deployable_confirmation",
        unlock=manifest_unlock,
        config=config,
    )
    assert manifest_authorization["status"] == "AUTHORIZED"
    with pytest.raises(ValueError, match="stale or reordered parent"):
        authorize_manifest_split_access(
            child_manifest,
            parent=_mini_parent(reverse_model_val=True),
            split_name="stack_val_deploy",
            purpose="deployable_confirmation",
            unlock=manifest_unlock,
            config=config,
        )


def test_registry_has_exact_counts_aliases_roles_seeds_and_conditional_state() -> None:
    registry = build_campaign_registry(alternate_teacher_valid=False)
    audit = validate_campaign_registry(registry)
    assert audit["configuration_count"] == REGISTRY_CONFIGURATION_COUNT == 54
    assert audit["reconstruction_breadth_count"] == RECONSTRUCTION_BREADTH_COUNT == 46
    assert audit["post_teacher_configuration_count"] == POST_TEACHER_CONFIGURATION_COUNT == 45
    assert audit["runnable_configuration_count"] == 53
    assert registry["alias_to_canonical"] == {
        "D10_A0_c0_delta": "D10_L8_full_c0",
        "D10_XA3_full_primary": "D10_A3_hlg_primary",
    }
    assert resolve_registry_run(registry, "D10_A0_c0_delta")[
        "canonical_run_id"
    ] == "D10_L8_full_c0"
    assert resolve_registry_run(registry, "D10_XA3_full_primary")[
        "canonical_run_id"
    ] == "D10_A3_hlg_primary"

    for run in registry["runs"]:
        assert isinstance(run["scientific_role"], str) and run["scientific_role"]
        assert isinstance(run["selectable_for_primary_deployment"], bool)
        assert run["paired_seed_ids"] == list(PAIRED_SEED_IDS)
        assert [row["seed_id"] for row in run["seed_replicas"]] == list(
            PAIRED_SEED_IDS
        )
        assert run["retained_state_rule"] == RETAINED_STATE_RULE
        assert run["measurement_status"] == MEASUREMENT_UNMEASURED
        assert run["measured_state_bytes"] is None
        assert run["measured_retained_bytes"] is None

    assert resolve_registry_run(registry, "D10_A3_hlg_primary")[
        "selectable_for_primary_deployment"
    ] is True
    expected_selectable = {
        "D10_L0_bridge_only",
        "D10_L1_ce_only",
        "D10_L2_kd_only",
        "D10_L3_kd_ce",
        "D10_L4_kd_bridge",
        "D10_L5_ce_bridge",
        "D10_L6_kd_ce_bridge",
        "D10_L7_plus_anchor",
        "D10_L8_full_c0",
        "D10_L9_full_true_target",
        "D10_A0M_capacity_particle",
        "D10_A1_multiscale_local",
        "D10_A1H_hard_radius",
        "D10_A2_regions_no_global",
        "D10_A3_hlg_primary",
        "D10_A4_hlg_refine",
        "D10_A6_hlg_no_pair_bias",
        "D10_A7_hlg_no_h0",
        "D10_A7F_hlg_no_f0",
        "D10_A7X_hlg_no_raw_skip",
        "D10_A8_hlg_fused_radius_heads",
        "D10_A9_hlg_group_gate",
        "D10_XA3_bridge_only",
        "D10_XA3_ce_only",
        "D10_XA3_kd_only",
        "D10_XA3_kd_bridge",
        "D10_XA3_kd_ce",
        "D10_XA3_full_no_warmup",
        "D10_XA3_full_no_smooth",
    }
    assert {
        run["canonical_run_id"]
        for run in registry["runs"]
        if run["selectable_for_primary_deployment"]
    } == expected_selectable
    for run_id in (
        "D10_L10_no_trust",
        "D10_A5_hlg_absolute_conditioned",
        "A0_CAP500_direct_hlt",
        "D10_B1_all50_fullhead",
        "D10_TALT_A3",
        "D10_N0_shuffled_logit_kd",
    ):
        assert resolve_registry_run(registry, run_id)[
            "selectable_for_primary_deployment"
        ] is False
    assert resolve_registry_run(registry, "D10_TALT_A3")[
        "execution_status"
    ] == "SKIPPED_INVALID_PARENT"

    with_alternate = build_campaign_registry(alternate_teacher_valid=True)
    assert validate_campaign_registry(with_alternate)["runnable_configuration_count"] == 54
    assert resolve_registry_run(with_alternate, "D10_TALT_A3")[
        "execution_status"
    ] == "RUNNABLE"


def test_provisional_storage_formula_and_dense_cache_projection_are_exact() -> None:
    categories = provisional_storage_categories("c0")
    weights = 8_000_000 * 4
    assert categories["retained_median_weights"] == weights
    assert categories["serialization_headroom"] == math.ceil(weights * 0.05)
    assert categories["nonmedian_replica_weights"] == 0
    assert categories["optimizer_scheduler_state"] == 0
    assert categories["generated_dense_fields"] == 0

    dense_128 = dense_field_cache_projection(
        event_count=1_300_000, particle_width=128
    )
    dense_256 = dense_field_cache_projection(
        event_count=1_300_000, particle_width=256
    )
    assert dense_128["projected_bytes"] == 16_640_000_000
    assert dense_256["projected_bytes"] == 2 * dense_128["projected_bytes"]
    assert dense_128["production_persistence_allowed"] is False


def test_step1_report_is_complete_and_production_remains_blocked() -> None:
    parent = _mini_parent()
    children = build_child_split_manifest(parent, config=_mini_config())
    registry = build_campaign_registry()
    storage = build_provisional_storage_projection(
        registry,
        children,
        particle_width=parent.max_constits,
    )
    assert storage["provisional_budget_ok"] is True
    assert storage["production_submission_allowed"] is False
    assert len(storage["unmeasured_runnable_run_ids"]) == 53
    report = build_step1_report(
        registry=registry,
        child_manifest=children,
        storage_projection=storage,
    )
    assert report["ok"] is True
    assert report["placeholder_models_instantiated"] is False
    assert report["production_submission_allowed"] is False
    assert set(report["child_splits"]["children"]) == {
        "stack_train_consumer",
        "stack_train_distill",
        "model_val_stop",
        "model_val_select",
        "stack_val_consumer",
        "stack_val_deploy",
    }
    assert len(report["runs"]) == 54
    for row in report["runs"]:
        assert len(row["seed_replicas"]) == 3
        assert row["retained_state_rule"] == RETAINED_STATE_RULE
        assert row["provisional_byte_categories"]
        assert row["measurement_status"] == MEASUREMENT_UNMEASURED

    with pytest.raises(PermissionError, match="UNMEASURED"):
        require_production_ready(
            registry,
            fixed_persistent_bytes=0,
            selected_budget_bytes=NORMAL_PILOT_BUDGET_BYTES,
        )
    measured = record_registry_measurements(
        registry,
        {
            run["canonical_run_id"]: 1024
            for run in registry["runs"]
            if run["execution_status"] == "RUNNABLE"
        },
    )
    ready = require_production_ready(
        measured,
        fixed_persistent_bytes=0,
        selected_budget_bytes=NORMAL_PILOT_BUDGET_BYTES,
    )
    assert ready["production_submission_allowed"] is True


def test_canonical_hash_and_immutable_publication_reject_replacement(
    tmp_path: Path,
) -> None:
    assert canonical_sha256({"b": 2, "a": 1}) == canonical_sha256({"a": 1, "b": 2})
    destination = tmp_path / "contract.json"
    first = with_content_hash({"contract": "test_v1", "value": 1})
    assert write_immutable_json(destination, first)["status"] == "published"
    assert write_immutable_json(destination, first)["status"] == "already_present"
    second = with_content_hash({"contract": "test_v1", "value": 2})
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_immutable_json(destination, second)


def test_step1_artifacts_publish_immutably_with_cross_bound_hashes(
    tmp_path: Path,
) -> None:
    parent = _mini_parent()
    children = build_child_split_manifest(parent, config=_mini_config())
    registry = build_campaign_registry()
    storage = build_provisional_storage_projection(
        registry, children, particle_width=parent.max_constits
    )
    report = build_step1_report(
        registry=registry,
        child_manifest=children,
        storage_projection=storage,
    )
    root = tmp_path / "campaign"
    first = write_step1_artifacts(
        root,
        child_manifest=children,
        registry=registry,
        storage_projection=storage,
        report=report,
    )
    assert {receipt["status"] for receipt in first["receipts"].values()} == {
        "published"
    }
    second = write_step1_artifacts(
        root,
        child_manifest=children,
        registry=registry,
        storage_projection=storage,
        report=report,
    )
    assert {receipt["status"] for receipt in second["receipts"].values()} == {
        "already_present"
    }
    assert (root / "contracts" / "split_children.json").is_file()
    assert (root / "contracts" / "campaign_registry.json").is_file()
    assert (root / "contracts" / "provisional_storage.json").is_file()
    assert (root / "reports" / "step1_dry_run_report.json").is_file()


def test_step1_cli_dry_run_writes_nothing_and_renders_complete_report(
    tmp_path: Path,
) -> None:
    parent_path = tmp_path / "parent.json"
    output_dir = tmp_path / "must-not-exist"
    save_split_manifest(_mini_parent(), parent_path, pretty=False)
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_prediction_anchored_bridge_campaign.py"),
            "--parent-manifest",
            str(parent_path),
            "--output-dir",
            str(output_dir),
            "--debug-miniature-splits",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["dry_run"] is True
    assert payload["debug_miniature_splits"] is True
    assert payload["report"]["counts"]["configuration_count"] == 54
    assert len(payload["report"]["runs"]) == 54
    assert payload["report"]["production_submission_allowed"] is False
    assert not output_dir.exists()
