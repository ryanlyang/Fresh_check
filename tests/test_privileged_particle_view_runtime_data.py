from __future__ import annotations

from pathlib import Path
import subprocess
import hashlib

import numpy as np
import pytest
import torch

from jetclass_fresh.hlt_cache import generate_and_cache_hlt_view
from jetclass_fresh.jetclass_data import (
    JetView,
    save_split_manifest,
)
from teacher_logit_reco.architecture_view_part import save_cached_offline_view
from teacher_logit_reco.local_particle_residual_field.particle_view import (
    ParticleViewRunSpec,
    SelectedViewLoader,
    DirectControlTrainConfig,
    build_baseline_factory,
    build_baseline_factory_config,
    build_consumer_screen_factory_config,
    build_consumer_screen_task_specs,
    build_confirmation_factory_config,
    build_confirmation_task_specs,
    build_distillation_factory_config,
    build_distillation_task_specs,
    build_focused_control_factory_config,
    build_focused_control_task_specs,
    build_fairness_factory_config,
    build_fairness_task_specs,
    build_stack_factory_config,
    build_stack_task_specs,
    canonical_sha256,
    build_direct_control_factory,
    build_direct_control_factory_config,
    build_direct_control_recipe,
    build_low_data_campaign_registry,
    build_post_target_factory_config,
    build_post_target_task_specs,
    build_particle_view_registry,
    build_runtime_data_config,
    build_runtime_task_result,
    build_scientific_task_catalog,
    build_stage_a_teacher_task_specs,
    build_stage_a_direct_resource_plan,
    build_stage_a_direct_task_specs,
    build_target_discovery_factory_config,
    build_target_discovery_task_specs,
    build_unified_split_manifest,
    execute_scientific_task,
    load_aligned_logical_jet_view,
    load_hashed_json,
    make_logical_data_loader,
    miniature_parent_manifest,
    miniature_split_config,
    resolve_parent_task_artifacts,
    register_existing_teacher_source,
    sha256_file,
    select_consumer_interface,
    with_content_hash,
    validate_runtime_data_config,
    validate_baseline_factory_config,
    validate_consumer_screen_factory_config,
    validate_confirmation_factory_config,
    validate_distillation_factory_config,
    validate_focused_control_factory_config,
    validate_fairness_factory_config,
    validate_stack_factory_config,
    validate_post_target_factory_config,
    validate_stage_a_direct_resource_plan,
    validate_target_discovery_factory_config,
    train_direct_hlt_control,
    write_immutable_json,
)
from teacher_logit_reco.local_particle_residual_field.particle_view.fairness_runtime import (
    _train_ce_trajectory,
)
from teacher_logit_reco.local_particle_residual_field.particle_view.confirmation_runtime import (
    _publish_confirmation_outputs,
    _publish_winner_selection,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _offline_view(parent, split: str) -> JetView:
    identities = list(parent.splits[split])
    count = len(identities)
    tokens = np.zeros((count, 128, 14), dtype=np.float32)
    mask = np.zeros((count, 128), dtype=bool)
    labels = np.asarray([row.label for row in identities], dtype=np.int64)
    for index in range(count):
        particles = 5 + index % 3
        mask[index, :particles] = True
        tokens[index, :particles, 0] = np.linspace(
            5.0, 1.5, particles, dtype=np.float32
        )
        tokens[index, :particles, 1] = np.linspace(
            -0.3, 0.3, particles, dtype=np.float32
        )
        tokens[index, :particles, 2] = np.linspace(
            -0.5, 0.5, particles, dtype=np.float32
        )
        tokens[index, :particles, 3] = (
            tokens[index, :particles, 0]
            * np.cosh(tokens[index, :particles, 1])
        )
        tokens[index, :particles, 5 + (index % 5)] = 1.0
    return JetView(
        tokens=tokens,
        mask=mask,
        labels=labels,
        jet_ids=identities,
        split=split,
        metadata={"view": "offline", "fixture": "particle_view_runtime_data"},
    )


def _runtime_sources(tmp_path: Path, *, reorder_offline: str | None = None):
    parent = miniature_parent_manifest(rows_per_class=2)
    split_config = miniature_split_config(rows_per_class=2)
    unified = build_unified_split_manifest(parent, config=split_config)
    parent_path = tmp_path / "parent_manifest.json"
    unified_path = tmp_path / "unified_manifest.json"
    save_split_manifest(parent, parent_path, pretty=True)
    write_immutable_json(unified_path, unified)
    hlt_root = tmp_path / "hlt"
    offline_root = tmp_path / "offline"
    used = ("model_train", "model_val", "stack_val", "final_test")
    for offset, split in enumerate(used):
        canonical = _offline_view(parent, split)
        generate_and_cache_hlt_view(
            canonical,
            hlt_root,
            seed=800 + offset,
        )
        offline = canonical
        if split == reorder_offline:
            order = np.arange(len(canonical.labels))[::-1]
            offline = JetView(
                tokens=canonical.tokens[order],
                mask=canonical.mask[order],
                labels=canonical.labels[order],
                jet_ids=[canonical.jet_ids[int(index)] for index in order],
                split=split,
                metadata=dict(canonical.metadata),
            )
        save_cached_offline_view(offline, offline_root)
    runtime_config = build_runtime_data_config(
        parent_manifest_path=parent_path,
        unified_manifest_path=unified_path,
        hlt_cache_dir=hlt_root,
        offline_cache_dir=offline_root,
    )
    return parent, unified, runtime_config


def test_runtime_data_aligns_logical_slices_and_builds_label_free_probe(tmp_path):
    _, unified, config = _runtime_sources(tmp_path)
    audit = validate_runtime_data_config(config)
    assert audit["parent_splits"] == [
        "final_test",
        "model_train",
        "model_val",
        "stack_val",
    ]

    stop = load_aligned_logical_jet_view(config, "model_val_stop")
    select = load_aligned_logical_jet_view(config, "model_val_select")
    assert len(stop) == len(select) == 10
    assert {row.key() for row in stop.identities}.isdisjoint(
        {row.key() for row in select.identities}
    )
    assert stop.logical_split_sha256 == unified["logical_splits"][
        "model_val_stop"
    ]["content_hash"]

    aligned_batch = next(
        iter(
            make_logical_data_loader(
                stop,
                mode="aligned",
                batch_size=4,
                shuffle=False,
                num_workers=0,
                seed=101,
            )
        )
    )
    assert aligned_batch["features"].shape[0] == 4
    assert aligned_batch["offline_features"].shape[0] == 4
    assert aligned_batch["labels"].shape == (4,)

    true_views = np.zeros((len(stop), 128, 2), dtype=np.float32)
    true_views[stop.hlt.mask[stop.parent_row_indices], :] = 0.25
    probe_batch = next(
        iter(
            make_logical_data_loader(
                stop,
                mode="recovery_probe",
                true_views=true_views,
                batch_size=4,
                shuffle=False,
                num_workers=0,
                seed=101,
            )
        )
    )
    assert set(probe_batch) == {"features", "mask", "true_view"}
    assert probe_batch["features"].shape[1] == 17
    assert probe_batch["mask"].shape == (4, 128)


def test_selected_view_loader_exposes_immutable_event_identity(tmp_path):
    _, _, config = _runtime_sources(tmp_path)
    aligned = load_aligned_logical_jet_view(config, "model_val_stop")
    base = make_logical_data_loader(
        aligned,
        mode="aligned",
        batch_size=4,
        shuffle=False,
        num_workers=0,
        seed=101,
    )
    views = np.zeros((len(aligned), 128, 2), dtype=np.float32)
    mask = aligned.hlt.mask[aligned.parent_row_indices]
    loader = SelectedViewLoader(
        base, aligned=aligned, views=views, mask=mask
    )
    batch = next(iter(loader))
    assert batch["event_ids"].dtype == batch["parent_indices"].dtype
    assert batch["event_ids"].tolist() == batch["parent_indices"].tolist()
    assert batch["true_view"].shape == (4, 128, 2)


def test_target_factory_promotes_complete_36_row_two_pass_screen(tmp_path):
    _, unified, runtime = _runtime_sources(tmp_path)
    config = build_target_discovery_factory_config(
        runtime_data_config=runtime,
        device="cpu",
        max_train_batches=1,
        max_val_batches=1,
    )
    audit = validate_target_discovery_factory_config(config)
    assert audit["supported_run_count"] == 36
    assert config["supported_run_ids"] == list(config["screen_recipes"])
    assert config["production_scope"] == "complete_target_screen_two_pass_v1"
    assert config["runtime_availability"]["VGEN_RECODESIGN"] == (
        "complete_two_pass"
    )
    assert config["runtime_availability"]["VGEN_TEACHER_EXISTING"] == (
        "unavailable_non_gating"
    )
    assert config["runtime_availability"]["VGEN_TAP_PENULT"] == (
        "complete_two_pass"
    )
    path = tmp_path / "target_factory.json"
    write_immutable_json(path, config)
    specs = build_target_discovery_task_specs(factory_config_path=path)
    assert set(specs) == set(config["supported_run_ids"])
    assert all(
        row["operation"] == "target_discovery"
        for row in specs.values()
    )
    registry = build_low_data_campaign_registry(
        unified_split_manifest=unified
    )
    rows = {row["run_id"]: row for row in registry["runs"]}
    assert rows["VGEN_TEACHER_LARGE"]["parent_run_ids"] == [
        "A0_VIEW",
        "TOFF_VIEW_LARGE",
    ]
    assert rows["VGEN_TEACHER_MIX2"]["parent_run_ids"] == [
        "A0_VIEW",
        "TOFF_VIEW_BASE",
        "TOFF_VIEW_LARGE",
    ]
    assert rows["VGEN_MEMORY_HLT"]["parent_run_ids"] == ["A0_VIEW"]


def test_post_target_factory_wires_selected_cache_and_robust_chain(tmp_path):
    _, _, runtime = _runtime_sources(tmp_path)
    config = build_post_target_factory_config(
        runtime_data_config=runtime,
        device="cpu",
        max_train_batches=1,
        max_val_batches=1,
    )
    assert validate_post_target_factory_config(config)["ok"]
    assert config["selected_cache_splits"] == [
        "train",
        "model_val_stop",
        "model_val_select",
    ]
    assert config["final_test_view_cache_forbidden"]
    assert not config["residual_bank_persisted"]
    path = tmp_path / "post_target_factory.json"
    write_immutable_json(path, config)
    specs = build_post_target_task_specs(factory_config_path=path)
    assert list(specs) == [
        "SELECTED_COORDINATE_BINDING",
        "SELECTED_VIEW_CACHE",
        "FINAL_CLEAN_CONSUMER",
        "PVIEW0",
        "RESIDUAL_SAMPLER",
        "ROBUST_CONSUMER",
    ]
    assert specs["FINAL_CLEAN_CONSUMER"]["operation"] == "consumer_training"
    assert specs["PVIEW0"]["operation"] == "pview0_training"
    assert specs["RESIDUAL_SAMPLER"]["operation"] == "residual_sampler_fit"
    assert (
        specs["ROBUST_CONSUMER"]["operation"]
        == "robust_consumer_training"
    )


def test_distillation_factory_wires_full_architecture_and_loss_campaign(
    tmp_path,
):
    _, unified, runtime = _runtime_sources(tmp_path)
    config = build_distillation_factory_config(
        runtime_data_config=runtime,
        device="cpu",
        max_train_batches=1,
        max_val_batches=1,
    )
    audit = validate_distillation_factory_config(config)
    assert audit["architecture_count"] == 17
    assert audit["distillation_row_count"] == 52
    assert config["target_logit_cache_policy"] == (
        "task_local_float32_exact_same_consumer"
    )
    assert config["selected_splits"] == [
        "train",
        "model_val_stop",
        "model_val_select",
    ]
    assert not config["quality_warnings_stop_execution"]
    path = tmp_path / "distillation_factory.json"
    write_immutable_json(path, config)
    specs = build_distillation_task_specs(factory_config_path=path)
    assert len(specs) == 69
    assert sum(name.startswith("ARCH_") for name in specs) == 17
    assert sum(name.startswith("DISTILL_") for name in specs) == 52
    assert {
        row["operation"] for row in specs.values()
    } == {"frozen_distillation", "joint_finetuning"}

    registry = build_low_data_campaign_registry(
        unified_split_manifest=unified
    )
    rows = {row["run_id"]: row for row in registry["runs"]}
    distillation = [
        row
        for row in rows.values()
        if row["scientific_role"].startswith("distillation:")
    ]
    primary_by_prefix = {}
    for row in distillation:
        detail = row["scientific_role"].split(":", 1)[1]
        if detail.endswith("__loss=L_PRIMARY"):
            primary_by_prefix[
                detail.removesuffix("__loss=L_PRIMARY")
            ] = row["run_id"]
    for row in distillation:
        detail = row["scientific_role"].split(":", 1)[1]
        if detail.endswith("__loss=L_CE"):
            prefix = detail.removesuffix("__loss=L_CE")
            assert primary_by_prefix[prefix] in row["parent_run_ids"]
        elif detail.endswith("__mode=joint"):
            prefix = detail.removesuffix("__mode=joint")
            assert primary_by_prefix[prefix] in row["parent_run_ids"]
        elif detail.endswith("__mode=joint_ce_control"):
            prefix = detail.removesuffix("__mode=joint_ce_control")
            assert primary_by_prefix[prefix] in row["parent_run_ids"]
            joint = next(
                candidate["run_id"]
                for candidate in distillation
                if candidate["scientific_role"].split(":", 1)[1]
                == f"{prefix}__mode=joint"
            )
            assert joint in row["parent_run_ids"]
    controls = {
        row["run_id"]: row
        for row in rows.values()
        if row["run_id"].startswith("TRAINED_CONTROL_")
    }
    assert primary_by_prefix[
        (
            "target=TARGET_ALTERNATE_SELECTED"
            "__arch=P_HIER_DECODER_REFINE"
            "__consumer=C_ROBUST_MIX"
        )
    ] in controls["TRAINED_CONTROL_IDENTICAL_CE_ONLY"][
        "parent_run_ids"
    ]
    assert len(
        controls["TRAINED_CONTROL_DVIEW_JOINT_CE_ONLY"]["parent_run_ids"]
    ) == 3


def test_consumer_interface_factory_covers_all_predeclared_rows(tmp_path):
    _, _, runtime = _runtime_sources(tmp_path)
    config = build_consumer_screen_factory_config(
        runtime_data_config=runtime,
        device="cpu",
        max_train_batches=1,
        max_val_batches=1,
    )
    assert validate_consumer_screen_factory_config(config)["ok"]
    path = tmp_path / "consumer_screen_factory.json"
    write_immutable_json(path, config)
    specs = build_consumer_screen_task_specs(factory_config_path=path)
    assert list(specs) == [
        f"SCREEN_{consumer_id}"
        for consumer_id in config["consumer_ids"]
    ]
    assert all(
        row["operation"] == "consumer_interface_screen"
        for row in specs.values()
    )


def test_focused_and_control_factory_covers_integrations_five_and_six(
    tmp_path,
):
    _, _, runtime = _runtime_sources(tmp_path)
    distillation = build_distillation_factory_config(
        runtime_data_config=runtime,
        device="cpu",
        max_train_batches=1,
        max_val_batches=1,
    )
    config = build_focused_control_factory_config(
        distillation_factory_config=distillation
    )
    audit = validate_focused_control_factory_config(config)
    assert audit["focused_interaction_count"] == 22
    assert audit["trained_control_count"] == 10
    assert not config["performance_gates"]
    assert not config["quality_warnings_stop_execution"]
    assert config["focused_interactions"]["UNCENTERED_DIM2"][
        "target_overrides"
    ] == {"center_output": False}
    assert config["focused_interactions"]["RECODESIGN_DIM8"][
        "target_overrides"
    ] == {
        "bottleneck_width": 8,
        "recoverability_codesign": True,
    }
    assert config["trained_controls"]["DVIEW_JOINT"]["deployable"]
    assert not config["trained_controls"][
        "OFFLINE_GLOBAL_LOGIT_BROADCAST"
    ]["deployable"]

    path = tmp_path / "focused_control_factory.json"
    write_immutable_json(path, config)
    specs = build_focused_control_task_specs(factory_config_path=path)
    assert len(specs) == 32
    assert sum(
        row["operation"] == "focused_composite_training"
        for row in specs.values()
    ) == 6
    assert sum(
        row["operation"] == "frozen_distillation"
        for row in specs.values()
    ) == 16
    assert sum(
        row["operation"] == "trained_control_training"
        for row in specs.values()
    ) == 10


def test_fairness_factory_covers_integration_seven(tmp_path):
    _, _, runtime = _runtime_sources(tmp_path)
    config = build_fairness_factory_config(
        runtime_data_config=runtime,
        device="cpu",
        max_train_batches=1,
        max_val_batches=1,
    )
    assert validate_fairness_factory_config(config)["ok"]
    assert config["winner_families"] == [
        "PRIVILEGED_SCIENTIFIC",
        "PRE_STAGE_G_DEPLOYABLE",
    ]
    assert config["fairness_control_ids"] == [
        "A0_VIEW_LONG_DEPLOY",
        "A0_VIEW_TOTAL_LABEL_BUDGET",
        "SELECTED_PARAMETER_MATCH",
        "SELECTED_FLOP_MATCH",
    ]
    assert len(config["direct_candidates"]) == 16
    assert config["direct_trial_count"] == 8
    assert not config["performance_gates"]
    assert not config["quality_warnings_stop_execution"]
    path = tmp_path / "fairness_factory.json"
    write_immutable_json(path, config)
    specs = build_fairness_task_specs(factory_config_path=path)
    assert len(specs) == 9
    assert "SELECTED_PATH_FAIRNESS_LEDGER" in specs
    assert all(
        row["operation"] == "fairness_closure"
        for row in specs.values()
    )


def test_stack_factory_covers_integration_eight(tmp_path):
    _, _, runtime = _runtime_sources(tmp_path)
    fairness = build_fairness_factory_config(
        runtime_data_config=runtime,
        device="cpu",
        max_train_batches=1,
        max_val_batches=1,
    )
    config = build_stack_factory_config(
        fairness_factory_config=fairness,
        device="cpu",
        max_stack_batches=1,
        bootstrap_replicates=25,
        linear_fusion_steps=10,
    )
    assert validate_stack_factory_config(config)["ok"]
    assert not config["stack_val_may_change_winner"]
    assert not config["final_test_loaded"]
    assert not config["performance_gates"]
    assert config["optional_p7b_resource"] is None
    path = tmp_path / "stack_factory.json"
    write_immutable_json(path, config)
    specs = build_stack_task_specs(factory_config_path=path)
    assert len(specs) == 19
    assert sum(
        row["operation"] == "stack_evaluation"
        for row in specs.values()
    ) == 11
    assert sum(
        row["operation"] == "fusion" for row in specs.values()
    ) == 8


def test_confirmation_factory_covers_structural_confirmation_and_selection(
    tmp_path,
):
    _, _, runtime = _runtime_sources(tmp_path)
    distillation = build_distillation_factory_config(
        runtime_data_config=runtime,
        device="cpu",
        max_train_batches=1,
        max_val_batches=1,
    )
    config = build_confirmation_factory_config(
        distillation_factory_config=distillation
    )
    audit = validate_confirmation_factory_config(config)
    assert audit["confirmation_role_count"] == 13
    assert audit["structural_control_count"] == 16
    assert config["confirmation_seeds"] == [101, 202, 303]
    assert config["selection_split"] == "model_val_select"
    assert not config["performance_gates"]
    assert not config["quality_warnings_stop_execution"]

    path = tmp_path / "confirmation_factory.json"
    write_immutable_json(path, config)
    specs = build_confirmation_task_specs(factory_config_path=path)
    assert len(specs) == 30
    assert sum(
        row["operation"] == "structural_control_evaluation"
        for row in specs.values()
    ) == 16
    assert sum(
        row["operation"] == "confirmation_training"
        for row in specs.values()
    ) == 13
    assert specs["SELECT_WINNER_FAMILIES"]["operation"] == (
        "configuration_selection"
    )


def test_confirmation_selection_publishes_exact_pv07_fairness_inputs(
    tmp_path,
):
    _, unified, _ = _runtime_sources(tmp_path / "sources")
    registry = build_low_data_campaign_registry(
        unified_split_manifest=unified
    )
    root = tmp_path / "campaign"
    diagnostic_roles = {
        "DIRECT_PARAMETER_CONTROL",
        "DIRECT_FLOP_CONTROL",
    }
    nonprivileged_roles = {
        "CE_ONLY_UPPER_BOUND",
        "HLT_MEMORY_CONTROL",
        "DVIEW_JOINT_CE_ONLY",
        *diagnostic_roles,
    }
    for role_index, role_id in enumerate(
        [
            row["scientific_role"].split(":", 1)[1]
            for row in registry["runs"]
            if row["run_id"].startswith("CONFIRM_")
        ]
    ):
        run_id = f"CONFIRM_{role_id}"
        for seed_index, seed in enumerate((101, 202, 303)):
            output = root / "runtime_tasks" / f"{run_id}__seed_{seed}"
            ledger = with_content_hash(
                {
                    "contract": "confirmation_test_ledger_v1",
                    "role_id": role_id,
                    "seed": seed,
                }
            )
            resource = with_content_hash(
                {
                    "contract": "confirmation_test_resource_v1",
                    "role_id": role_id,
                    "seed": seed,
                    "total_parameters": 1000 + role_index,
                }
            )
            ledger_path = output / "training_ledger.json"
            resource_path = output / "resource_profile.json"
            write_immutable_json(ledger_path, ledger)
            write_immutable_json(resource_path, resource)
            diagnostic = role_id in diagnostic_roles
            privileged = role_id not in nonprivileged_roles
            replica = with_content_hash(
                {
                    "contract": "particle_view_confirmation_replica_v1",
                    "configuration_id": f"configuration::{role_id}",
                    "run_id": run_id,
                    "role_id": role_id,
                    "seed": seed,
                    "split": "model_val_select",
                    "deployable_accuracy": (
                        0.70 + 0.001 * role_index + 0.0001 * seed_index
                    ),
                    "deployable_cross_entropy": 0.5,
                    "recovery_status": (
                        "finite" if privileged else "undefined"
                    ),
                    "recovery_fraction": 0.2 if privileged else None,
                    "oracle_gain": 0.01 if privileged else None,
                    "deployed_parameters": 1000 + role_index,
                    "bundle_sha256": _sha(f"{role_id}-{seed}"),
                    "bundle_path": str(output / "bundle.pt"),
                    "source_run_id": f"source::{role_id}",
                    "source_registration_sha256": _sha(
                        f"registration::{role_id}"
                    ),
                    "source_resolution_reason": "test",
                    "training_ledger_sha256": ledger["content_hash"],
                    "resource_profile_sha256": resource["content_hash"],
                    "privileged_claim_eligible": privileged,
                    "pre_stage_g_deployable_eligible": not diagnostic,
                    "diagnostic": diagnostic,
                    "stack_val_loaded": False,
                    "final_test_loaded": False,
                    "performance_gate_used": False,
                }
            )
            replica_path = output / "confirmation_replica.json"
            write_immutable_json(replica_path, replica)
            result = build_runtime_task_result(
                task_id=f"{run_id}__seed_{seed}",
                artifacts=[
                        {
                            "path": str(replica_path),
                            "sha256": sha256_file(replica_path),
                        },
                        {
                            "path": str(ledger_path),
                            "sha256": sha256_file(ledger_path),
                        },
                        {
                            "path": str(resource_path),
                            "sha256": sha256_file(resource_path),
                        },
                ],
            )
            write_immutable_json(output / "task_result.json", result)
    destination = root / "runtime_tasks" / "SELECT_WINNER_FAMILIES__seed_101"
    _publish_winner_selection(
        root=str(root),
        registry=registry,
        config={
            "flop_fixture_sha256": _sha("fixture"),
            "flop_counter_sha256": _sha("counter"),
        },
        output_dir=str(destination),
    )
    selection = load_hashed_json(destination / "winner_selection.json")
    fairness = load_hashed_json(destination / "fairness_input_index.json")
    assert selection["selection_split"] == "model_val_select"
    assert fairness["selection_sha256"] == selection["content_hash"]
    assert fairness["replica_count"] in {3, 6}
    assert not fairness["stack_val_loaded"]
    assert not fairness["final_test_loaded"]


def test_confirmation_finalizer_accounts_selected_and_total_training(
    tmp_path, monkeypatch
):
    unified = build_unified_split_manifest(
        miniature_parent_manifest(rows_per_class=2),
        config=miniature_split_config(rows_per_class=2),
    )
    root = tmp_path / "campaign"
    output = root / "runtime_tasks" / "CONFIRM_TEST__seed_101"
    output.mkdir(parents=True)
    artifacts = {}

    def publish(run_id, name, payload):
        path = root / "parents" / run_id / name
        path.parent.mkdir(parents=True, exist_ok=True)
        write_immutable_json(path, with_content_hash(payload))
        artifacts.setdefault(run_id, {})[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }
        return load_hashed_json(path)

    toff = publish(
        "TOFF",
        "teacher_registration.json",
        {
            "contract": "test_teacher",
            "optimizer_updates": 8,
            "selected_epoch": 2,
            "recipe": {"seed": 101},
        },
    )
    a0 = publish(
        "A0_VIEW",
        "teacher_registration.json",
        {
            "contract": "test_teacher",
            "optimizer_updates": 4,
            "selected_epoch": 1,
            "recipe": {"seed": 101},
        },
    )
    target = publish(
        "TARGET",
        "target_candidate_registration.json",
        {"contract": "test_target"},
    )
    publish(
        "TARGET",
        "target_discovery_recipe.json",
        {
            "contract": "test_target_recipe",
            "memory_teacher_registration_sha256": toff["content_hash"],
            "secondary_memory_teacher_registration_sha256": None,
        },
    )

    def consumer(role):
        return {
            "contract": "test_consumer",
            "role": role,
            "training_config": {"batch_size": 2, "seed": 101},
            "selected_epoch": 1,
        }

    publish("TARGET", "consumer_registration.json", consumer("discovery"))
    publish(
        "TARGET",
        "probe_consumer/consumer_registration.json",
        consumer("probe"),
    )
    publish(
        "FINAL_CLEAN_CONSUMER",
        "consumer_registration.json",
        consumer("clean"),
    )
    publish(
        "ROBUST_CONSUMER",
        "robust_consumer_registration.json",
        {
            "contract": "test_robust",
            "selected_epoch": 1,
            "optimizer_updates": 9,
            "train_config": {"seed": 101},
        },
    )
    publish(
        "ROBUST_CONSUMER",
        "robust_consumer_training_curves.json",
        {
            "contract": "test_robust_curves",
            "epochs": [{"epoch": 1, "optimizer_updates": 2}],
        },
    )
    publish(
        "PVIEW0",
        "pview0_registration.json",
        {
            "contract": "test_pview",
            "optimizer_updates": 5,
            "warmup_config": {"seed": 101},
        },
    )
    checkpoint = output / "selected_distilled_predictor.pt"
    torch.save({"predictor_state_dict": {}}, checkpoint)
    registration = with_content_hash(
        {
            "contract": "test_distillation",
            "configuration_id": "config",
            "run_id": "CONFIRM_TEST",
            "seed": 101,
            "checkpoint_file": checkpoint.name,
            "selected_epoch": 2,
            "optimizer_updates": 99,
            "label_bearing_updates": 99,
            "ce_bearing_updates": 99,
            "teacher_kd_updates": 99,
            "view_supervision_updates": 99,
            "model_val_select": {
                "deployable_accuracy": 0.7,
                "deployable_cross_entropy": 0.8,
                "recovery_status": "finite",
                "recovery_fraction": 0.2,
                "oracle_gain": 0.1,
            },
        }
    )
    write_immutable_json(output / "distillation_registration.json", registration)
    write_immutable_json(
        output / "distillation_training_curves.json",
        with_content_hash(
            {
                "contract": "test_distillation_curves",
                "rows": [
                    {"epoch": 1, "optimizer_updates": 3},
                    {"epoch": 2, "optimizer_updates": 6},
                    {"epoch": 3, "optimizer_updates": 9},
                ],
            }
        ),
    )
    resource = with_content_hash(
        {
            "contract": "test_resource",
            "total_parameters": 10,
            "forward_flops": {"exact_integer_total": 20},
        }
    )
    registry = {
        "runs": [
            {
                "run_id": "TARGET",
                "stage": "target",
                "parents": ("A0_VIEW", "TOFF"),
            },
            {"run_id": "A0_VIEW", "stage": "teacher", "parents": ()},
            {"run_id": "TOFF", "stage": "teacher", "parents": ()},
        ]
    }
    monkeypatch.setattr(
        "teacher_logit_reco.local_particle_residual_field.particle_view."
        "confirmation_runtime._task_artifacts",
        lambda _root, _registry, run_id, _seed: artifacts[run_id],
    )
    _publish_confirmation_outputs(
        output_dir=str(output),
        run_id="CONFIRM_TEST",
        seed=101,
        role_id="CANONICAL_PREDECLARED",
        source={
            "configuration_id": "config",
            "campaign_row": {
                "target_id": "TARGET_ALTERNATE_SELECTED",
                "consumer_id": "C_ROBUST_MIX",
                "mode": "frozen",
            },
            "resolved_target_run_id": "TARGET",
            "resolved_target_registration_sha256": target["content_hash"],
            "predictor_initialization": "exact_pview0_checkpoint",
            "source_run_id": "SOURCE",
            "registration_sha256": _sha("source"),
            "resolution_reason": "test",
            "privileged_claim_eligible": True,
            "pre_stage_g_deployable_eligible": True,
            "diagnostic": False,
        },
        resource_profile=resource,
        unified_manifest=unified,
        campaign_root=str(root),
        registry=registry,
        max_train_batches=None,
    )
    ledger = load_hashed_json(output / "training_ledger.json")
    by_component = {row["component"]: row for row in ledger["records"]}
    assert by_component["confirmed_hlt_only_bundle"][
        "optimizer_steps"
    ] == 6
    assert by_component["clean_particle_view_consumer"][
        "retained_in_deployable_path"
    ]
    assert by_component["robust_particle_view_consumer"][
        "retained_in_deployable_path"
    ]
    assert not by_component[
        "oracle_view_discovery_consumer_and_generator"
    ]["retained_in_deployable_path"]
    assert not by_component["offline_memory_teacher"][
        "retained_in_deployable_path"
    ]
    assert (
        ledger["totals_all_training"]["label_bearing_steps"]
        > ledger["totals_retained_deployable_path"]["label_bearing_steps"]
    )


def test_stage_g_a0_trajectory_stops_at_exact_update_budget(tmp_path):
    class TinyHLT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = torch.nn.Linear(2, 3)

        def forward(self, points, features, lorentz_vectors, mask):
            del points, lorentz_vectors, mask
            return self.classifier(features.mean(dim=2))

    batch = {
        "points": torch.zeros(4, 2, 3),
        "features": torch.randn(4, 2, 3),
        "lorentz_vectors": torch.zeros(4, 4, 3),
        "mask": torch.ones(4, 1, 3, dtype=torch.bool),
        "labels": torch.tensor([0, 1, 2, 0]),
    }
    result = _train_ce_trajectory(
        model=TinyHLT(),
        train_loader=[batch],
        stop_loader=[batch],
        output=tmp_path,
        seed=101,
        device="cpu",
        exact_updates=3,
        learning_rate=3.0e-4,
        weight_decay=1.0e-4,
        maximum_epochs=100,
        patience=None,
        max_train_batches=None,
        max_val_batches=None,
    )
    assert result["optimizer_updates"] == 3
    exact = torch.load(
        tmp_path / "exact_matched_update.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert exact["optimizer_updates"] == 3
    assert (tmp_path / "best_model_val_stop_within_budget.pt").is_file()


def test_consumer_interface_selection_is_deterministic_and_non_gating():
    config = build_low_data_campaign_registry(
        unified_split_manifest=build_unified_split_manifest(
            miniature_parent_manifest(rows_per_class=2),
            config=miniature_split_config(rows_per_class=2),
        )
    )
    consumer_ids = [
        row["run_id"][7:]
        for row in config["runs"]
        if row["run_id"].startswith("SCREEN_")
    ]
    metrics = []
    for index, consumer_id in enumerate(consumer_ids):
        recipe = {
            "consumer_id": consumer_id,
            "injection_block": 0,
            "view_path": "token_and_pair",
            "learned_trust": True,
            "augment_clean_view": False,
            "robust_probe_mixture": False,
            "training_role": "Cview_probe",
            "epochs": 12,
            "selection_split": "model_val_select",
            "quality_gate_used": False,
        }
        metrics.append(
            with_content_hash(
                {
                    "contract": "particle_view_consumer_screen_metrics_v1",
                    "consumer_id": consumer_id,
                    "run_id": f"SCREEN_{consumer_id}",
                    "recipe": recipe,
                    "recipe_sha256": _sha(f"recipe-{index}"),
                    "consumer_registration_sha256": _sha(
                        f"registration-{index}"
                    ),
                    "checkpoint_sha256": _sha(f"checkpoint-{index}"),
                    "model_val_select": {
                        "accuracy": 0.5 + index / 100,
                        "cross_entropy": 1.0 - index / 100,
                        "examples": 10.0,
                    },
                    "ranking_rule": [
                        "highest_accuracy",
                        "lowest_cross_entropy",
                        "lexicographic_consumer_id",
                    ],
                    "quality_gate_used": False,
                    "stops_execution": False,
                }
            )
        )
    # The selector validates hashes rather than re-deriving the recipe hash;
    # use the canonical hash in production-shaped rows.
    for row in metrics:
        payload = dict(row)
        payload.pop("content_hash")
        payload["recipe_sha256"] = canonical_sha256(payload["recipe"])
        row.clear()
        row.update(with_content_hash(payload))
    selected = select_consumer_interface(metrics)
    assert selected["selected_consumer_id"] == consumer_ids[-1]
    assert selected["quality_threshold_used_as_gate"] is False


def test_runtime_data_rejects_mutually_consistent_but_parent_reordered_cache(
    tmp_path,
):
    _, _, config = _runtime_sources(
        tmp_path, reorder_offline="model_val"
    )
    with pytest.raises(ValueError, match="offline cache identity order"):
        load_aligned_logical_jet_view(config, "model_val_stop")


def test_runtime_data_config_rejects_cache_changed_after_binding(tmp_path):
    _, _, config = _runtime_sources(tmp_path)
    path = Path(config["parent_cache_records"][0]["hlt_metadata"]["path"])
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash changed"):
        validate_runtime_data_config(config)


def test_parent_artifact_resolver_uses_seed_fallback_and_authenticates(tmp_path):
    registry = build_particle_view_registry(
        unified_split_manifest_sha256="a" * 64,
        train_identity_sha256="b" * 64,
        run_specs=[
            ParticleViewRunSpec(
                run_id="parent",
                stage="source",
                scientific_role="source:parent",
                selection_family="infrastructure",
                seed_ids=(101,),
                uses_labels=False,
                train_split=None,
            ),
            ParticleViewRunSpec(
                run_id="child",
                stage="baseline",
                scientific_role="baseline:child",
                selection_family="infrastructure",
                seed_ids=(101, 202, 303),
                parent_run_ids=("parent",),
            ),
        ],
        campaign_id="runtime_parent_resolution",
    )
    task_id = "parent__seed_101"
    task_root = tmp_path / "runtime_tasks" / task_id
    task_root.mkdir(parents=True)
    artifact = task_root / "source.json"
    artifact.write_text("source", encoding="utf-8")
    result = build_runtime_task_result(
        task_id=task_id,
        artifacts=[{"path": str(artifact), "sha256": sha256_file(artifact)}],
    )
    write_immutable_json(task_root / "task_result.json", result)
    resolved = resolve_parent_task_artifacts(
        registry=registry,
        artifact_root=tmp_path,
        run_id="child",
        seed=303,
    )
    assert resolved["parent"]["seed"] == 101
    assert resolved["parent"]["artifacts"]["source.json"]["sha256"] == (
        sha256_file(artifact)
    )
    artifact.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact changed"):
        resolve_parent_task_artifacts(
            registry=registry,
            artifact_root=tmp_path,
            run_id="child",
            seed=303,
        )


def test_source_preflight_scientific_factory_runs_real_cache_audit(tmp_path):
    _, unified, config = _runtime_sources(tmp_path / "sources")
    config_path = tmp_path / "runtime_data_config.json"
    write_immutable_json(config_path, config)
    registry = build_particle_view_registry(
        unified_split_manifest_sha256=unified["content_hash"],
        train_identity_sha256=unified["logical_splits"]["train"][
            "ordered_identity_sha256"
        ],
        run_specs=[
            ParticleViewRunSpec(
                run_id="PV_SOURCE_PREFLIGHT",
                stage="source",
                scientific_role="source:unified_manifest_sources_storage",
                selection_family="infrastructure",
                uses_labels=False,
                train_split=None,
            )
        ],
        campaign_id="runtime_source_factory",
    )
    catalog = build_scientific_task_catalog(
        registry=registry,
        task_specs={
            "PV_SOURCE_PREFLIGHT": {
                "operation": "source_preflight",
                "factory": (
                    "teacher_logit_reco.local_particle_residual_field."
                    "particle_view.production_factories:"
                    "build_source_preflight_factory"
                ),
                "factory_config_path": str(config_path),
                "factory_config_sha256": sha256_file(config_path),
            }
        },
    )
    output = tmp_path / "source_task"
    result = execute_scientific_task(
        catalog=catalog,
        registry=registry,
        run_id="PV_SOURCE_PREFLIGHT",
        seed=101,
        task_id="PV_SOURCE_PREFLIGHT__seed_101",
        output_dir=output,
    )
    assert result["status"] == "complete"
    audit_path = output / "source_cache_audit.json"
    assert audit_path.is_file()
    assert "source_cache_audit" in audit_path.read_text(encoding="utf-8")


def test_runtime_data_config_cli_publishes_authenticated_config(tmp_path):
    _, _, config = _runtime_sources(tmp_path / "sources")
    output = tmp_path / "cli_runtime_data_config.json"
    result = subprocess.run(
        [
            str(Path(".venv/Scripts/python.exe")),
            "scripts/build_particle_view_runtime_data_config.py",
            "--parent-manifest",
            config["parent_manifest"]["path"],
            "--unified-manifest",
            config["unified_manifest"]["path"],
            "--hlt-cache-dir",
            config["hlt_cache_dir"],
            "--offline-cache-dir",
            config["offline_cache_dir"],
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert output.is_file()
    assert "parent_splits=4" in result.stdout


def _stage_a_registry(unified):
    return build_particle_view_registry(
        unified_split_manifest_sha256=unified["content_hash"],
        train_identity_sha256=unified["logical_splits"]["train"][
            "ordered_identity_sha256"
        ],
        run_specs=[
            ParticleViewRunSpec(
                run_id="PV_SOURCE_PREFLIGHT",
                stage="source",
                scientific_role="source:unified_manifest_sources_storage",
                selection_family="infrastructure",
                uses_labels=False,
                train_split=None,
            ),
            ParticleViewRunSpec(
                run_id="A0_VIEW",
                stage="baseline",
                scientific_role="baseline:A0_VIEW",
                selection_family="infrastructure",
                parent_run_ids=("PV_SOURCE_PREFLIGHT",),
            ),
            ParticleViewRunSpec(
                run_id="TOFF_VIEW_BASE",
                stage="baseline",
                scientific_role="baseline:TOFF_VIEW_BASE",
                selection_family="infrastructure",
                parent_run_ids=("PV_SOURCE_PREFLIGHT",),
            ),
            ParticleViewRunSpec(
                run_id="TOFF_VIEW_LARGE",
                stage="baseline",
                scientific_role="baseline:TOFF_VIEW_LARGE",
                selection_family="infrastructure",
                parent_run_ids=("PV_SOURCE_PREFLIGHT",),
            ),
            ParticleViewRunSpec(
                run_id="TOFF_VIEW_EXISTING",
                stage="baseline",
                scientific_role="baseline:TOFF_VIEW_EXISTING",
                selection_family="diagnostic",
                parent_run_ids=("PV_SOURCE_PREFLIGHT",),
                diagnostic=True,
            ),
            ParticleViewRunSpec(
                run_id="STAGE_A_PARAMETER_MATCH",
                stage="baseline",
                scientific_role="baseline:STAGE_A_PARAMETER_MATCH",
                selection_family="infrastructure",
                seed_ids=(101, 202, 303),
                parent_run_ids=("PV_SOURCE_PREFLIGHT",),
            ),
            ParticleViewRunSpec(
                run_id="STAGE_A_FLOP_MATCH",
                stage="baseline",
                scientific_role="baseline:STAGE_A_FLOP_MATCH",
                selection_family="infrastructure",
                seed_ids=(101, 202, 303),
                parent_run_ids=("PV_SOURCE_PREFLIGHT",),
            ),
        ],
        campaign_id="stage_a_factory_test",
    )


def _publish_source_parent_result(artifact_root: Path):
    task_id = "PV_SOURCE_PREFLIGHT__seed_101"
    root = artifact_root / "runtime_tasks" / task_id
    root.mkdir(parents=True)
    audit = root / "source_cache_audit.json"
    audit.write_text("authenticated-source-audit", encoding="utf-8")
    result = build_runtime_task_result(
        task_id=task_id,
        artifacts=[{"path": str(audit), "sha256": sha256_file(audit)}],
    )
    write_immutable_json(root / "task_result.json", result)


def test_stage_a_teacher_factory_builds_real_hlt_and_offline_loaders(tmp_path):
    _, unified, data = _runtime_sources(tmp_path / "sources")
    artifact_root = tmp_path / "campaign"
    _publish_source_parent_result(artifact_root)
    registry = _stage_a_registry(unified)
    existing_checkpoint = tmp_path / "existing.pt"
    existing_checkpoint.write_bytes(b"existing-offline-teacher")
    config = build_baseline_factory_config(
        runtime_data_config=data,
        device="cpu",
        num_workers=0,
        max_train_batches=1,
        max_val_batches=1,
        existing_checkpoint_path=existing_checkpoint,
        existing_provenance_metadata_sha256="c" * 64,
    )
    assert validate_baseline_factory_config(config)["trained_teacher_count"] == 3

    a0 = build_baseline_factory(
        operation="teacher_training",
        config=config,
        registry=registry,
        run_id="A0_VIEW",
        seed=101,
        task_id="A0_VIEW__seed_101",
        output_dir=str(
            artifact_root / "runtime_tasks" / "A0_VIEW__seed_101"
        ),
    )
    assert a0["action"] is None
    assert a0["kwargs"]["recipe"].role == "A0_view"
    assert a0["kwargs"]["recipe"].particle_source == "fixed_hlt"
    assert a0["kwargs"]["train_loader"].batch_size == 128
    assert a0["kwargs"]["config"].max_train_batches == 1
    assert len(a0["artifact_paths"]) == 4

    offline = build_baseline_factory(
        operation="teacher_training",
        config=config,
        registry=registry,
        run_id="TOFF_VIEW_LARGE",
        seed=101,
        task_id="TOFF_VIEW_LARGE__seed_101",
        output_dir=str(
            artifact_root / "runtime_tasks" / "TOFF_VIEW_LARGE__seed_101"
        ),
    )
    assert offline["kwargs"]["recipe"].role == "Toff_view"
    assert offline["kwargs"]["recipe"].architecture == "large"
    assert offline["kwargs"]["train_loader"].batch_size == 64
    assert (
        a0["kwargs"]["recipe"].unified_split_manifest_sha256
        == offline["kwargs"]["recipe"].unified_split_manifest_sha256
    )


def test_existing_teacher_factory_is_diagnostic_and_source_bound(tmp_path):
    _, unified, data = _runtime_sources(tmp_path / "sources")
    artifact_root = tmp_path / "campaign"
    _publish_source_parent_result(artifact_root)
    registry = _stage_a_registry(unified)
    checkpoint = tmp_path / "existing.pt"
    checkpoint.write_bytes(b"existing")
    config = build_baseline_factory_config(
        runtime_data_config=data,
        device="cpu",
        existing_checkpoint_path=checkpoint,
        existing_provenance_metadata_sha256="d" * 64,
    )
    prepared = build_baseline_factory(
        operation="existing_teacher_registration",
        config=config,
        registry=registry,
        run_id="TOFF_VIEW_EXISTING",
        seed=101,
        task_id="TOFF_VIEW_EXISTING__seed_101",
        output_dir=str(
            artifact_root / "runtime_tasks" / "TOFF_VIEW_EXISTING__seed_101"
        ),
    )
    assert prepared["action"] is None
    register_existing_teacher_source(**prepared["kwargs"])
    registration = Path(prepared["artifact_paths"][0])
    lineage = Path(prepared["artifact_paths"][1])
    assert registration.is_file() and lineage.is_file()
    assert '"selectable": false' in registration.read_text(encoding="utf-8")
    assert sha256_file(
        artifact_root
        / "runtime_tasks"
        / "PV_SOURCE_PREFLIGHT__seed_101"
        / "source_cache_audit.json"
    ) in lineage.read_text(encoding="utf-8")


def test_stage_a_teacher_task_specs_exclude_unresolved_direct_controls(tmp_path):
    _, unified, data = _runtime_sources(tmp_path / "sources")
    config = build_baseline_factory_config(
        runtime_data_config=data,
        device="cpu",
    )
    config_path = tmp_path / "baseline_factory_config.json"
    write_immutable_json(config_path, config)
    specs = build_stage_a_teacher_task_specs(
        factory_config_path=config_path
    )
    assert set(specs) == {
        "A0_VIEW",
        "TOFF_VIEW_BASE",
        "TOFF_VIEW_LARGE",
        "TOFF_VIEW_EXISTING",
    }
    assert specs["A0_VIEW"]["operation"] == "teacher_training"
    assert (
        specs["TOFF_VIEW_EXISTING"]["operation"]
        == "existing_teacher_registration"
    )
    assert "STAGE_A_PARAMETER_MATCH" not in specs

    artifact_root = tmp_path / "campaign"
    _publish_source_parent_result(artifact_root)
    unavailable = build_baseline_factory(
        operation="existing_teacher_registration",
        config=config,
        registry=_stage_a_registry(unified),
        run_id="TOFF_VIEW_EXISTING",
        seed=101,
        task_id="TOFF_VIEW_EXISTING__seed_101",
        output_dir=str(
            artifact_root / "runtime_tasks" / "TOFF_VIEW_EXISTING__seed_101"
        ),
    )
    register_existing_teacher_source(**unavailable["kwargs"])
    text = Path(unavailable["artifact_paths"][0]).read_text(encoding="utf-8")
    assert '"selection_status": "diagnostic_unavailable"' in text
    assert '"warning_is_non_gating": true' in text


def test_baseline_factory_config_cli_publishes_teacher_specs(tmp_path):
    _, _, data = _runtime_sources(tmp_path / "sources")
    data_path = tmp_path / "runtime_data_config.json"
    write_immutable_json(data_path, data)
    config_path = tmp_path / "baseline_factory_config.json"
    specs_path = tmp_path / "teacher_task_specs.json"
    result = subprocess.run(
        [
            str(Path(".venv/Scripts/python.exe")),
            "scripts/build_particle_view_baseline_factory_config.py",
            "--runtime-data-config",
            str(data_path),
            "--output",
            str(config_path),
            "--task-specs-output",
            str(specs_path),
            "--device",
            "cpu",
            "--max-train-batches",
            "1",
            "--max-val-batches",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert config_path.is_file() and specs_path.is_file()
    assert "trained_teacher_roles=3" in result.stdout


def test_stage_a_direct_resource_plan_profiles_all_candidates_without_gating():
    first = build_stage_a_direct_resource_plan()
    second = build_stage_a_direct_resource_plan()
    assert first == second
    audit = validate_stage_a_direct_resource_plan(first)
    assert audit["candidate_count"] == 16
    assert first["canonical_target"]["view_dim"] == 4
    assert (
        first["canonical_target"]["predictor_architecture_id"]
        == "P_HIER_DECODER_REFINE"
    )
    assert first["canonical_target"]["deployed_parameters"] == sum(
        first["canonical_target"]["parameter_breakdown"].values()
    )
    for quantity in ("parameters", "flops"):
        selection = first["selections"][quantity]
        assert selection["selected"]["config_id"].startswith("w")
        assert selection["warning_is_non_gating"] is True
        if selection["quality_warning"] is not None:
            assert selection["quality_warning"] == "WARN_CONTROL_MATCH_TOLERANCE"


def test_stage_a_direct_factory_binds_selection_source_and_hlt_loaders(tmp_path):
    _, unified, data = _runtime_sources(tmp_path / "sources")
    artifact_root = tmp_path / "campaign"
    _publish_source_parent_result(artifact_root)
    resource_plan = build_stage_a_direct_resource_plan()
    config = build_direct_control_factory_config(
        runtime_data_config=data,
        resource_plan=resource_plan,
        device="cpu",
        num_workers=0,
        max_train_batches=1,
        max_val_batches=1,
    )
    prepared = build_direct_control_factory(
        operation="direct_control_training",
        config=config,
        registry=_stage_a_registry(unified),
        run_id="STAGE_A_PARAMETER_MATCH",
        seed=303,
        task_id="STAGE_A_PARAMETER_MATCH__seed_303",
        output_dir=str(
            artifact_root
            / "runtime_tasks"
            / "STAGE_A_PARAMETER_MATCH__seed_303"
        ),
    )
    recipe = prepared["kwargs"]["recipe"]
    assert recipe.seed == 303
    assert recipe.selection["requested_quantity"] == "parameters"
    assert recipe.resource_plan_sha256 == resource_plan["content_hash"]
    assert prepared["kwargs"]["train_loader"].batch_size == 128
    assert prepared["kwargs"]["config"].max_train_batches == 1
    assert prepared["action"] is None

    config_path = tmp_path / "direct_factory_config.json"
    write_immutable_json(config_path, config)
    specs = build_stage_a_direct_task_specs(
        factory_config_path=config_path
    )
    assert set(specs) == {
        "STAGE_A_PARAMETER_MATCH",
        "STAGE_A_FLOP_MATCH",
    }
    assert all(
        row["operation"] == "direct_control_training"
        for row in specs.values()
    )


def test_direct_control_trainer_smoke_publishes_hlt_only_registration(tmp_path):
    torch = pytest.importorskip("torch")
    parent = miniature_parent_manifest(rows_per_class=2)
    unified = build_unified_split_manifest(
        parent, config=miniature_split_config(rows_per_class=2)
    )
    plan = build_stage_a_direct_resource_plan()
    recipe = build_direct_control_recipe(
        run_id="STAGE_A_FLOP_MATCH",
        seed=101,
        resource_plan=plan,
        unified_split_manifest=unified,
        preprocessing_sha256="1" * 64,
        source_sha256="2" * 64,
        library_versions_sha256="3" * 64,
    )

    class TinyDirect(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Linear(17, 10)

        def forward(self, points, features, lorentz_vectors, mask):
            del points, lorentz_vectors
            valid = mask.float()
            pooled = (features * valid).sum(dim=2) / valid.sum(dim=2).clamp_min(1)
            return self.head(pooled)

    batch = {
        "points": torch.randn(2, 2, 5),
        "features": torch.randn(2, 17, 5),
        "lorentz_vectors": torch.randn(2, 4, 5),
        "mask": torch.ones(2, 1, 5, dtype=torch.bool),
        "labels": torch.tensor([0, 1]),
    }

    class Loader:
        batch_size = 128

        def __len__(self):
            return 1

        def __iter__(self):
            return iter((batch,))

    report = train_direct_hlt_control(
        recipe=recipe,
        train_loader=Loader(),
        model_val_stop_loader=Loader(),
        config=DirectControlTrainConfig(
            output_dir=str(tmp_path / "direct_train"),
            device="cpu",
            max_train_batches=1,
            max_val_batches=1,
            amp=False,
        ),
        model=TinyDirect(),
    )
    assert report["status"] == "COMPLETE"
    registration = (
        tmp_path / "direct_train" / "direct_control_registration.json"
    )
    text = registration.read_text(encoding="utf-8")
    assert '"hlt_only_inference": true' in text
    assert '"privileged_inputs": false' in text


def test_direct_control_factory_cli_writes_resource_plan_config_and_specs(
    tmp_path,
):
    _, _, data = _runtime_sources(tmp_path / "sources")
    data_path = tmp_path / "runtime_data_config.json"
    write_immutable_json(data_path, data)
    plan = build_stage_a_direct_resource_plan()
    input_plan = tmp_path / "input_resource_plan.json"
    write_immutable_json(input_plan, plan)
    output_plan = tmp_path / "published_resource_plan.json"
    config_path = tmp_path / "direct_factory_config.json"
    specs_path = tmp_path / "direct_task_specs.json"
    result = subprocess.run(
        [
            str(Path(".venv/Scripts/python.exe")),
            "scripts/build_particle_view_direct_control_factory_config.py",
            "--runtime-data-config",
            str(data_path),
            "--resource-plan",
            str(input_plan),
            "--resource-plan-output",
            str(output_plan),
            "--output",
            str(config_path),
            "--task-specs-output",
            str(specs_path),
            "--device",
            "cpu",
            "--max-train-batches",
            "1",
            "--max-val-batches",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert output_plan.is_file() and config_path.is_file() and specs_path.is_file()
    assert "candidates=16" in result.stdout
