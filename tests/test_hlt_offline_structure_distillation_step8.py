from __future__ import annotations

import numpy as np
import pytest
import torch

from teacher_logit_reco.hlt_offline_structure_distillation import (
    build_combination_selection,
    build_combination_beam_completion,
    build_combination_wave_completion,
    build_gradient_conflict_report,
    build_mechanism_control_plan,
    build_mechanism_summary,
    build_mechanism_result,
    build_stage_f_plan,
    expand_combination_beam,
    advance_combination_beam,
    normalize_combination_weights,
    promote_combination_beam_winners,
    pcgrad_project,
    evaluate_auxiliary_head_removal,
    eventwise_error_gain_tracking,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (
    COMBINATION_RESULT_CONTRACT,
    FEEDBACK_SELECTION_CONTRACT,
    SINGLE_FAMILY_SELECTION_CONTRACT,
    with_content_hash,
)
from teacher_logit_reco.hlt_offline_structure_distillation.combination_runtime import (
    CombinationDataset,
    CombinationHBaseClassifier,
    _native_relation_identity_indices,
    combination_losses,
)
from teacher_logit_reco.hlt_offline_structure_distillation.auxiliary_data import (
    AuxiliaryTargetDataset,
    DeterministicScaleShardSampler,
)
from teacher_logit_reco.hlt_offline_structure_distillation.taps import (
    SplitForwardResult,
)
from teacher_logit_reco.relation_expert_token_bridge.evaluation import (
    evaluate_classification,
)


SOURCE = {
    "commit": "a" * 40,
    "status_sha256": "b" * 64,
    "dirty": True,
    "status_hash_policy": "test",
}


def test_native_relation_val_design_store_is_lazily_joined_to_design_subset():
    class Base:
        logical_role = "design_select"
        identities = ("jet-c", "jet-a")
        identity_order_sha256 = "c" * 64
        replicas = {0: None}

        def __getitem__(self, index):
            return {
                "event_identity": self.identities[index],
                "replica_id": 0,
            }

    class Target:
        def __init__(self):
            self.base_dataset = Base()
            self.identities = self.base_dataset.identities
            self.target_stores = ()

        def __len__(self):
            return len(self.identities)

        def attach_target(self, index, sample):
            return {
                **sample,
                "target": np.zeros(1, dtype=np.float32),
                "target_mask": np.ones(1, dtype=bool),
            }

    source_ids = np.asarray(["jet-a", "jet-b", "jet-c"])
    source_indices = _native_relation_identity_indices(
        source_ids, Base.identities
    )
    targets = np.zeros((3, 545), dtype=np.float32)
    targets[:, 0] = (10.0, 20.0, 30.0)
    dataset = CombinationDataset(
        {"T_TEST": Target()},
        native_relation_targets={
            0: {
                "targets": targets,
                "target_mask": np.ones((3, 545), dtype=bool),
                "availability": np.ones((3, 7), dtype=np.float32),
                "source_indices": source_indices,
            }
        },
    )
    assert source_indices.tolist() == [2, 0]
    assert dataset[0]["native_relation_target"]["target"][0] == 30.0
    assert dataset[1]["native_relation_target"]["target"][0] == 10.0
    with pytest.raises(ValueError, match="absent from val_design"):
        _native_relation_identity_indices(source_ids, ("jet-missing",))
    with pytest.raises(ValueError, match="source identities are duplicated"):
        _native_relation_identity_indices(("jet-a", "jet-a"), ("jet-a",))


def test_eight_member_scale_combination_never_materializes_identity_population():
    class NoPopulationIteration:
        def __len__(self):
            return 3_000_000

        def __iter__(self):
            raise AssertionError("scale identities must not be iterated")

        def __array__(self, *args, **kwargs):
            raise AssertionError("scale identities must not be converted")

    class Base:
        def __init__(self, identities):
            self.identities = identities
            self.identity_order_sha256 = "a" * 64
            self.replicas = {0: None, 1: None, 2: None, 3: None}

        def __getitem__(self, index):
            return {"event_identity": f"jet-{index}", "replica_id": 0}

        def set_epoch(self, epoch):
            self.epoch = epoch

    class Store:
        def __init__(self):
            self.coordinator = None
            self.cached = False

        def bind_cache_coordinator(self, coordinator):
            self.coordinator = coordinator

        def clear_cached_shard(self):
            self.cached = False

        def activate(self):
            self.coordinator.activate(self)
            self.cached = True

    class Raw:
        def __init__(self, store):
            self.store = store

    class Coordinator:
        def __init__(self):
            self.active_store = None

        def activate(self, store):
            if self.active_store is not None and self.active_store is not store:
                self.active_store.clear_cached_shard()
            self.active_store = store

    class Target:
        def __init__(self, base, store):
            self.base_dataset = base
            self.identities = base.identities
            self.values = Raw(store)
            self.masks = Raw(store)
            self.store = store
            self.target_stores = (store,)
            self.target_cache_coordinator = Coordinator()
            store.bind_cache_coordinator(self.target_cache_coordinator)

        def __len__(self):
            return 3_000_000

        def attach_target(self, index, base_sample):
            self.store.activate()
            return {
                **base_sample,
                "target": np.asarray([index], dtype=np.float32),
                "target_mask": np.asarray([True]),
            }

    identities = NoPopulationIteration()
    stores = [Store() for _ in range(8)]
    datasets = {
        f"T_{index}": Target(Base(identities), store)
        for index, store in enumerate(stores)
    }
    combination = CombinationDataset(datasets)
    assert combination.identities is identities
    assert combination.target_store_count == 8
    assert len({id(row.base_dataset) for row in combination.datasets.values()}) == 1
    sample = combination[7]
    assert len(sample["combination_targets"]) == 8
    assert sum(store.cached for store in stores) == 8
    assert combination.resident_target_shard_budget == 8


def test_scale_sampler_decodes_by_shard_replica_group_not_by_event_member():
    event_count, shard_size, member_count = 64, 16, 8

    class Base:
        logical_role = "scale_train"
        realization_policy = "R_MULTI"

        def __init__(self):
            self.identities = tuple(f"jet-{index}" for index in range(event_count))
            self.identity_order_sha256 = "b" * 64
            self.labels = np.arange(event_count, dtype=np.int64) % 10
            self.replicas = {replica: None for replica in range(4)}
            self.zero_based_epoch = 0

        def __len__(self):
            return event_count

        def set_epoch(self, epoch):
            self.zero_based_epoch = int(epoch) - 1

        def replica_for_index(self, index):
            return int(index) % 4

        def locality_boundaries(self):
            return tuple(range(0, event_count + 1, shard_size))

        def __getitem__(self, index):
            return {
                "event_identity": self.identities[index],
                "replica_id": self.replica_for_index(index),
            }

    class Store:
        def __init__(self):
            self.ends = tuple(range(shard_size, event_count + 1, shard_size))
            self.cached_shard = -1
            self.decoded_shard_load_count = 0
            self.coordinator = None

        def bind_cache_coordinator(self, coordinator):
            self.coordinator = coordinator

        def clear_cached_shard(self):
            self.cached_shard = -1

        def row(self, index):
            shard = int(index) // shard_size
            if shard != self.cached_shard:
                self.coordinator.activate(self)
                self.cached_shard = shard
                self.decoded_shard_load_count += 1

    class Array:
        def __init__(self, store, *, mask):
            self.store = store
            self.mask = mask
            self.shape = (event_count, 1)

        def __getitem__(self, index):
            self.store.row(index)
            return np.asarray([True] if self.mask else [index], dtype=(
                bool if self.mask else np.float32
            ))

    base = Base()
    stores = []
    datasets = {}
    for member in range(member_count):
        replica_stores = {replica: Store() for replica in range(4)}
        stores.extend(replica_stores.values())
        datasets[f"T_{member}"] = AuxiliaryTargetDataset(
            base,
            target_id=f"T_{member}",
            target_values={
                replica: Array(store, mask=False)
                for replica, store in replica_stores.items()
            },
            target_masks={
                replica: Array(store, mask=True)
                for replica, store in replica_stores.items()
            },
            target_parent_hashes={"cache": f"{member + 1:x}" * 64},
        )
    combination = CombinationDataset(datasets)
    combination.set_epoch(1)
    sampler = DeterministicScaleShardSampler(combination, seed=101)
    sampler.set_epoch(1)
    order = list(sampler)
    assert sorted(order) == list(range(event_count))
    replay = DeterministicScaleShardSampler(combination, seed=101)
    replay.set_epoch(1)
    assert list(replay) == order
    sampler.set_epoch(2)
    second_epoch_order = list(sampler)
    assert sorted(second_epoch_order) == list(range(event_count))
    assert second_epoch_order != order
    locality_keys = [(index // shard_size, index % 4) for index in order]
    runs = 1 + sum(
        right != left for left, right in zip(locality_keys, locality_keys[1:])
    )
    assert runs == len(set(locality_keys)) == 16
    for index in order:
        combination[index]
    assert sum(store.decoded_shard_load_count for store in stores) == (
        len(set(locality_keys)) * member_count
    )
    assert all(
        dataset.target_cache_coordinator.maximum_simultaneously_resident == 1
        for dataset in combination.datasets.values()
    )


@pytest.mark.parametrize(
    ("training_role", "evaluation_role", "pipeline_seed"),
    [
        ("model_train", "design_confirm", 202),
        ("scale_train", "design_confirm", 303),
    ],
)
def test_combination_manifest_loads_authenticated_nondefault_role_seed(
    tmp_path,
    monkeypatch,
    training_role,
    evaluation_role,
    pipeline_seed,
):
    from types import SimpleNamespace

    from teacher_logit_reco.hlt_offline_structure_distillation import (
        build_combination_loader_manifest,
        build_stage_d_loader_manifest,
        data_order_seed,
        load_combination_loaders,
    )
    from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (
        write_immutable_json,
    )
    from teacher_logit_reco.hlt_offline_structure_distillation import (
        combination_runtime,
    )

    target_id = "T_OFFLINE_TRACK_32"
    roles = (training_role, "val_stop", evaluation_role)
    member_row = {
        "row_id": "TRACK_SELECTED",
        "target_id": target_id,
        "parameterization": "ABS",
        "row_kind": "SCIENTIFIC",
        "pipeline_seed": pipeline_seed,
        "resolved": True,
    }
    role_definitions = {
        role: {
            "labels": f"/labels/{role}.npz",
            "hlt_caches": {"0": f"/hlt/{role}/replica_0"},
            "target": {
                "mode": "static_cache",
                "caches": {"shared": f"/targets/{role}"},
            },
        }
        for role in roles
    }
    member_manifest = build_stage_d_loader_manifest(
        row=member_row,
        role_definitions=role_definitions,
        campaign_spec_sha256="c" * 64,
        source=SOURCE,
        training_role=training_role,
        evaluation_role=evaluation_role,
    )
    member_path = tmp_path / "member.json"
    write_immutable_json(member_path, member_manifest)
    graph = {
        "graph_id": "C_SEEDED",
        "members": [
            {
                "target_id": target_id,
                "selected_row_id": "TRACK_SELECTED",
                "parameterization": "ABS",
            }
        ],
    }
    manifest = build_combination_loader_manifest(
        graph=graph,
        member_loader_manifests={target_id: member_path},
        campaign_spec_sha256="c" * 64,
        source=SOURCE,
        evaluation_role=evaluation_role,
        training_role=training_role,
    )
    assert manifest["contract"] == "hosd_combination_loader_manifest_v6"
    assert manifest["pipeline_seed"] == pipeline_seed
    assert manifest["sampler_seed_by_role"][training_role] == data_order_seed(
        pipeline_seed, training_role
    )

    class Base:
        def __init__(self, role):
            self.logical_role = role
            self.identities = ("jet-0", "jet-1")
            self.identity_order_sha256 = "d" * 64
            self.replicas = {0: None}

        def __len__(self):
            return 2

        def locality_boundaries(self):
            return (0, 2)

        def replica_for_index(self, index):
            return 0

        def set_epoch(self, epoch):
            self.epoch = epoch

        def __getitem__(self, index):
            return {"event_identity": self.identities[index], "replica_id": 0}

    class Target:
        def __init__(self, role):
            self.base_dataset = Base(role)
            self.identities = self.base_dataset.identities
            self.target_stores = ()

        def __len__(self):
            return 2

        def locality_boundaries(self):
            return self.base_dataset.locality_boundaries()

        def attach_target(self, index, sample):
            return {
                **sample,
                "target": np.zeros(32, dtype=np.float32),
                "target_mask": np.ones(32, dtype=bool),
            }

    def fake_member_loader(**kwargs):
        assert kwargs["row"]["pipeline_seed"] == pipeline_seed
        loaders = {
            role: SimpleNamespace(dataset=Target(role), batch_size=64)
            for role in roles
        }
        return {
            "train_loader": loaders[training_role],
            "val_stop_loader": loaders["val_stop"],
            "evaluation_loader": loaders[evaluation_role],
            "lineage_hashes": {"member": "e" * 64},
        }

    monkeypatch.setattr(
        combination_runtime,
        "load_stage_d_loaders_from_manifest",
        fake_member_loader,
    )
    loaded = load_combination_loaders(
        manifest=manifest,
        graph=graph,
        campaign_root=tmp_path,
        campaign={"source": SOURCE, "content_hash": "c" * 64},
        target_registry={},
    )
    assert loaded["pipeline_seed"] == pipeline_seed
    assert loaded["train_loader"].sampler.seed == data_order_seed(
        pipeline_seed, training_role
    )
    assert loaded["train_loader"].sampler.contract == manifest[
        "sampler_contract_by_role"
    ][training_role]


def test_combination_manifest_rejects_mixed_member_pipeline_seeds(tmp_path):
    from teacher_logit_reco.hlt_offline_structure_distillation import (
        build_combination_loader_manifest,
        build_stage_d_loader_manifest,
    )
    from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (
        write_immutable_json,
    )

    targets = ("T_OFFLINE_TRACK_32", "T_OFFLINE_DENSITY_22")
    paths = {}
    for target_id, seed in zip(targets, (202, 303)):
        row = {
            "row_id": f"{target_id}_SELECTED",
            "target_id": target_id,
            "parameterization": "ABS",
            "row_kind": "SCIENTIFIC",
            "pipeline_seed": seed,
            "resolved": True,
        }
        roles = {
            role: {
                "labels": f"/labels/{role}.npz",
                "hlt_caches": {"0": f"/hlt/{role}/replica_0"},
                "target": {
                    "mode": "static_cache",
                    "caches": {"shared": f"/targets/{target_id}/{role}"},
                },
            }
            for role in ("model_train", "val_stop", "design_confirm")
        }
        artifact = build_stage_d_loader_manifest(
            row=row,
            role_definitions=roles,
            campaign_spec_sha256="c" * 64,
            source=SOURCE,
            evaluation_role="design_confirm",
        )
        path = tmp_path / f"{target_id}.json"
        write_immutable_json(path, artifact)
        paths[target_id] = path
    graph = {
        "graph_id": "C_MIXED",
        "members": [
            {
                "target_id": target,
                "selected_row_id": f"{target}_SELECTED",
            }
            for target in targets
        ],
    }
    with pytest.raises(ValueError, match="data-order contracts differ"):
        build_combination_loader_manifest(
            graph=graph,
            member_loader_manifests=paths,
            campaign_spec_sha256="c" * 64,
            source=SOURCE,
            evaluation_role="design_confirm",
        )


def _locks():
    targets = (
        "T_OFFLINE_JET_10",
        "T_OFFLINE_COMPOSITION_16",
        "T_OFFLINE_TRACK_32",
        "T_OFFLINE_DENSITY_22",
        "T_OFFLINE_CA_TREE_26",
        "T_OFFLINE_RELATION_BASE4",
        "T_OFFLINE_RELATION_PT",
        "T_OFFLINE_RELATION_PID",
        "T_OFFLINE_RELATION_CHARGE",
        "T_OFFLINE_RELATION_TRACK",
        "T_OFFLINE_RELATION_DENSITY",
        "T_OFFLINE_RELATION_REGION",
        "T_OFFLINE_TRACK_COMPONENT_PROXY_17",
        "T_OFFLINE_LOGITS_O_BASE",
        "T_OFFLINE_LOGITS_O_FULLREL",
        "T_OFFLINE_POOLED_LATENT",
        "T_HLT_TRACK_PAIR_13",
        "T_HLT_REGION_PAIR_8",
    )
    selected = {target: f"row-{index}" for index, target in enumerate(targets)}
    definitions = {
        target: {
            "target_id": target,
            "parameterization": "ABS",
            "auxiliary_weight": 0.3,
            "head_type": "global",
        }
        for target in targets
    }
    single = with_content_hash(
        {
            "contract": SINGLE_FAMILY_SELECTION_CONTRACT,
            "schema_version": 2,
            "source": SOURCE,
            "stage_d_plan_sha256": "d" * 64,
            "phase_lock_sha256": "e" * 64,
            "complete_result_hashes": {},
            "selected_row_by_target": selected,
            "selected_definition_by_target": definitions,
            "cross_family_order": [
                {"ordinal": index, "target_id": target, "row_id": selected[target]}
                for index, target in enumerate(targets)
            ],
            "global_winner_row_id": selected[targets[0]],
            "negative_results_permitted": True,
            "performance_based_termination": False,
        }
    )
    feedback = with_content_hash(
        {
            "contract": FEEDBACK_SELECTION_CONTRACT,
            "schema_version": 1,
            "source": SOURCE,
            "stage_e_plan_sha256": "f" * 64,
            "result_hashes": {},
            "selected_feedback_row_id": "fb-best",
            "selected_by_interface": {"FB_TOKEN": "fb-best"},
            "all_rows_completed": True,
            "negative_gain_can_still_win": True,
            "selection_split": "design_select",
            "oracle_or_control_rows_eligible": False,
        }
    )
    return single, feedback


def test_stage_f_mandatory_registry_bounds_and_weight_cap():
    single, feedback = _locks()
    plan = build_stage_f_plan(
        single_family_selection=single,
        feedback_selection=feedback,
        campaign_spec_sha256="c" * 64,
        source=SOURCE,
    )
    assert [row["combination_id"] for row in plan["mandatory_combinations"]] == [
        "C_PHYSICAL",
        "C_TRACK_TOPOLOGY",
        "C_PHYSICAL_KD",
        "C_PHYSICAL_LATENT",
        "C_ALL_BEST",
        "C_NATIVE_OFFLINE",
    ]
    assert plan["beam_width"] == 12
    assert plan["beam_reduced_budget_fit_hard_maximum"] == 96
    assert plan["full_fit_hard_maximum"] == 10
    assert plan["total_full_execution_hard_maximum"] == 11
    assert plan["pcgrad_control"]["weighting"] == "W_PCGRAD"
    assert plan["pcgrad_control"]["selection_eligible"] is False
    for graph in plan["mandatory_combinations"]:
        assert sum(graph["normalized_weights"].values()) <= 1.0 + 1e-12
    native = next(
        graph
        for graph in plan["mandatory_combinations"]
        if graph["combination_id"] == "C_NATIVE_OFFLINE"
    )
    assert native["native_relation_auxiliary"]["baseline_id"] == "H_NATIVE_REL_AUX"
    assert len(native["members"]) == 1
    assert "H_NATIVE_REL_AUX" in native["normalized_weights"]
    weights = normalize_combination_weights({"a": 0.3, "b": 0.9})
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["a"] / weights["b"] == pytest.approx(1 / 3)


def test_pcgrad_uses_canonical_cyclic_projection_and_averaging():
    first = [torch.tensor([1.0, 0.0])]
    second = [torch.tensor([-1.0, 1.0])]
    projected = pcgrad_project([first, second], update_ordinal=0)
    assert len(projected) == 1
    assert torch.isfinite(projected[0]).all()
    # Both task order and rotation are deterministic.
    again = pcgrad_project([first, second], update_ordinal=2)
    torch.testing.assert_close(projected[0], again[0], atol=0, rtol=0)
    rotated = pcgrad_project([first, second], update_ordinal=1)
    assert torch.isfinite(rotated[0]).all()


def test_beam_expands_add_or_omit_and_trains_only_add_branch():
    single, feedback = _locks()
    plan = build_stage_f_plan(
        single_family_selection=single,
        feedback_selection=feedback,
        campaign_spec_sha256="c" * 64,
        source=SOURCE,
    )
    expansion = expand_combination_beam(
        stage_f_plan=plan,
        family="JET",
        current_beam=[],
        completed_new_fit_count=0,
    )
    assert expansion["omit_reuse_graph_ids"] == ["H_BASE_BEAM_BUDGET"]
    assert expansion["new_fit_count"] == 1
    assert expansion["completed_new_fit_count_after"] == 1
    labels = np.tile(np.arange(10), 3)
    metrics = evaluate_classification(
        np.zeros((30, 10)), labels, split="design_select"
    )
    results = [
        with_content_hash(
            {
                "contract": COMBINATION_RESULT_CONTRACT,
                "schema_version": 1,
                "source": SOURCE,
                **candidate,
                "design_select": {"classification_metrics": metrics},
            }
        )
        for candidate in expansion["all_candidates"]
    ]
    wave = advance_combination_beam(
        stage_f_plan=plan,
        family="JET",
        expansion=expansion,
        reduced_budget_results=results,
    )
    assert wave["completed_new_fit_count"] == 1
    assert len(wave["surviving_candidates"]) == 2
    next_wave = expand_combination_beam(
        stage_f_plan=plan,
        family="COMPOSITION",
        current_beam=wave["surviving_candidates"],
        completed_new_fit_count=wave["completed_new_fit_count"],
    )
    assert next_wave["new_fit_count"] == 2
    assert next_wave["completed_new_fit_count_after"] == 3


def test_complete_beam_attests_all_eight_waves_and_promotes_distinct_full_fits():
    single, feedback = _locks()
    plan = build_stage_f_plan(
        single_family_selection=single,
        feedback_selection=feedback,
        campaign_spec_sha256="c" * 64,
        source=SOURCE,
    )
    labels = np.tile(np.arange(10), 3)
    metrics = evaluate_classification(
        np.zeros((30, 10)), labels, split="design_select"
    )
    root = with_content_hash(
        {
            "contract": COMBINATION_RESULT_CONTRACT,
            "schema_version": 1,
            "source": SOURCE,
            **plan["beam_root"],
            "design_select": {"classification_metrics": metrics},
        }
    )
    by_id = {root["graph_id"]: root}
    beam, expansions, waves = [], [], []
    completed = 0
    for family in plan["family_order"]:
        expansion = expand_combination_beam(
            stage_f_plan=plan,
            family=family,
            current_beam=beam,
            completed_new_fit_count=completed,
        )
        expansions.append(expansion)
        for candidate in expansion["new_fit_candidates"]:
            by_id[candidate["graph_id"]] = with_content_hash(
                {
                    "contract": COMBINATION_RESULT_CONTRACT,
                    "schema_version": 1,
                    "source": SOURCE,
                    **candidate,
                    "design_select": {"classification_metrics": metrics},
                }
            )
        wave = advance_combination_beam(
            stage_f_plan=plan,
            family=family,
            expansion=expansion,
            reduced_budget_results=[
                by_id[row["graph_id"]]
                for row in expansion["all_candidates"]
            ],
        )
        waves.append(wave)
        beam = wave["surviving_candidates"]
        completed = wave["completed_new_fit_count"]
    promotion = promote_combination_beam_winners(
        stage_f_plan=plan, final_wave=waves[-1]
    )
    assert promotion["winner_count"] == 4
    assert all(row["budget"] == "FULL" for row in promotion["promoted_graphs"])
    assert not (
        set(promotion["promoted_graph_ids"])
        & set(promotion["reduced_budget_graph_ids"])
    )
    nonempty = [
        candidate
        for candidate in waves[-1]["surviving_candidates"]
        if candidate.get("members")
    ]
    final_with_winning_base = with_content_hash(
        {
            key: value
            for key, value in waves[-1].items()
            if key != "content_hash"
        }
        | {
            "surviving_graph_ids": [
                "H_BASE_BEAM_BUDGET",
                *[row["graph_id"] for row in nonempty[:4]],
            ],
            "surviving_candidates": [plan["beam_root"], *nonempty[:4]],
        }
    )
    promotion_with_winning_base = promote_combination_beam_winners(
        stage_f_plan=plan,
        final_wave=final_with_winning_base,
    )
    assert promotion_with_winning_base["winner_count"] == 4
    assert promotion_with_winning_base["excluded_baseline_graph_ids"] == [
        "H_BASE_BEAM_BUDGET"
    ]
    assert all(
        row["members"] for row in promotion_with_winning_base["promoted_graphs"]
    )
    completion = build_combination_beam_completion(
        stage_f_plan=plan,
        expansions=expansions,
        waves=waves,
        reduced_budget_results=list(by_id.values()),
    )
    assert completion["all_families_completed"]
    assert completion["new_fit_count"] <= 96
    assert completion["promotion_sha256"] == promotion["content_hash"]

    full_results = [
        with_content_hash(
            {
                "contract": COMBINATION_RESULT_CONTRACT,
                "schema_version": 1,
                "source": SOURCE,
                **graph,
                "stage_f_plan_sha256": plan["content_hash"],
                "design_select": {"classification_metrics": metrics},
            }
        )
        for graph in [
            *plan["mandatory_combinations"],
            *promotion["promoted_graphs"],
        ]
    ]
    full = build_combination_wave_completion(
        stage_f_plan=plan,
        wave_kind="FULL",
        results=full_results,
        beam_completion=completion,
    )
    assert full["result_count"] == 10
    assert full["performance_based_termination"] is False


def _full_results(plan):
    labels = np.tile(np.arange(10), 3)
    logits = np.zeros((30, 10), dtype=np.float64)
    metrics = evaluate_classification(logits, labels, split="design_select")
    return [
        with_content_hash(
            {
                "contract": COMBINATION_RESULT_CONTRACT,
                "schema_version": 1,
                "source": SOURCE,
                **graph,
                "design_select": {"classification_metrics": metrics},
                "deployed_analytical_flops": 20.0,
                "deployed_parameter_count": 200,
                "training_gpu_hours": 2.0,
            }
        )
        for graph in plan["mandatory_combinations"]
    ]


def test_all_negative_combination_selection_and_design_confirm_controls():
    single, feedback = _locks()
    plan = build_stage_f_plan(
        single_family_selection=single,
        feedback_selection=feedback,
        campaign_spec_sha256="c" * 64,
        source=SOURCE,
    )
    results = _full_results(plan)
    lock = build_combination_selection(
        stage_f_plan=plan,
        full_results=results,
        beam_winner_graph_ids=[],
        source=SOURCE,
    )
    assert lock["negative_gain_can_still_win"] is True
    selected = next(
        row
        for row in plan["mandatory_combinations"]
        if row["graph_id"] == lock["selected_combination_graph_id"]
    )
    mechanism = build_mechanism_control_plan(
        combination_selection=lock,
        selected_combination=selected,
        source=SOURCE,
    )
    assert mechanism["evaluation_split"] == "design_confirm"
    assert mechanism["can_reopen_selection"] is False
    fixture = [
        build_mechanism_result(
            plan=mechanism,
            intervention_id=row["intervention_id"],
            status=(
                "not_applicable"
                if row["execution"]["worker"] == "emit_registered_not_applicable"
                else "completed"
            ),
            measurements={"accuracy_change": -0.01},
            evidence_hashes={"evidence": "a" * 64},
            source=SOURCE,
        )
        for row in mechanism["interventions"]
    ]
    summary = build_mechanism_summary(
        plan=mechanism, results=fixture, source=SOURCE
    )
    assert summary["all_interventions_complete"]
    assert summary["negative_or_null_mechanism_results_reported"]


def test_redundancy_report_has_correlations_gradients_cka_and_leave_one_out():
    identities = [f"jet-{index}" for index in range(6)]
    base = np.arange(12, dtype=np.float64).reshape(6, 2)
    report = build_gradient_conflict_report(
        identities=identities,
        residuals_by_family={"A": base, "B": base[:, ::-1]},
        target_errors_by_family={
            "A": np.arange(6, dtype=np.float64),
            "B": np.arange(6, dtype=np.float64) ** 2,
        },
        gradient_cosines={"A__B": [0.2, -0.4, 0.1]},
        representations_by_family={"A": base, "B": base + 1},
        leave_one_out_accuracy_change={"A": -0.01, "B": 0.0},
        source=SOURCE,
    )
    assert set(report["pair_diagnostics"]["A__B"]) == {
        "residual_pearson",
        "residual_spearman",
        "target_error_correlation",
        "representation_linear_cka",
    }
    assert report["gradient_diagnostics"]["A__B"]["negative_fraction"] == pytest.approx(
        1 / 3
    )
    assert report["selection_effect"] == "report_only"


class _CombinationBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(17, 128)
        self.classifier = torch.nn.Linear(128, 10)

    def forward_with_taps(
        self, points, features, vectors, mask, *, capture
    ):
        del points, vectors
        state = self.projection(features.transpose(1, 2))
        active = mask[:, 0].bool()
        pooled = (
            state.masked_fill(~active.unsqueeze(-1), 0).sum(dim=1)
            / active.sum(dim=1, keepdim=True)
        )
        return SplitForwardResult(
            logits=self.classifier(pooled),
            states={"TAP_LATE": state},
            masks={"TAP_LATE": active},
        )

    def forward(self, points, features, vectors, mask):
        return self.forward_with_taps(
            points, features, vectors, mask, capture=("TAP_LATE",)
        ).logits


def test_native_offline_combination_executes_both_auxiliary_losses():
    single, feedback = _locks()
    plan = build_stage_f_plan(
        single_family_selection=single,
        feedback_selection=feedback,
        campaign_spec_sha256="c" * 64,
        source=SOURCE,
    )
    graph = next(
        row
        for row in plan["mandatory_combinations"]
        if row["combination_id"] == "C_NATIVE_OFFLINE"
    )
    members = [
        {**member, "native_relation_auxiliary": True}
        for member in graph["members"]
    ]
    model = CombinationHBaseClassifier(_CombinationBackbone(), members)
    target = graph["members"][0]["target_id"]
    dimension = 10
    batch = {
        "points": torch.zeros(2, 2, 4),
        "features": torch.randn(2, 17, 4),
        "vectors": torch.randn(2, 4, 4),
        "mask": torch.ones(2, 1, 4, dtype=torch.bool),
        "labels": torch.tensor([0, 1]),
        "combination_targets": {
            target: {
                "target": torch.zeros(2, dimension),
                "target_mask": torch.ones(2, dimension, dtype=torch.bool),
            }
        },
        "native_relation_target": {
            "target": torch.zeros(2, 545),
            "target_mask": torch.ones(2, 545, dtype=torch.bool),
            "availability": torch.ones(2, 7),
        },
    }
    total, logits, pieces = combination_losses(
        model=model, batch=batch, graph=graph
    )
    assert logits.shape == (2, 10)
    assert torch.isfinite(total)
    assert set(pieces["auxiliaries"]) == {target, "H_NATIVE_REL_AUX"}


def test_mechanism_runtime_proves_head_removal_and_tracks_eventwise_gain():
    member = {
        "target_id": "T_OFFLINE_JET_10",
        "parameterization": "ABS",
        "auxiliary_weight": 0.3,
        "family": "JET",
    }
    model = CombinationHBaseClassifier(_CombinationBackbone(), [member])
    batch = {
        "features": torch.randn(4, 17, 3),
        "vectors": torch.randn(4, 4, 3),
        "mask": torch.ones(4, 1, 3, dtype=torch.bool),
        "event_identities": [f"jet-{index}" for index in range(4)],
    }
    parity = evaluate_auxiliary_head_removal(model, [batch], device="cpu")
    assert parity["logits_bitwise_equal"] is True
    labels = np.asarray([0, 1, 2, 3])
    baseline = np.zeros((4, 10))
    candidate = baseline.copy()
    candidate[np.arange(4), labels] = 1
    tracking = eventwise_error_gain_tracking(
        identities=batch["event_identities"],
        labels=labels,
        baseline_logits=baseline,
        candidate_logits=candidate,
        target_error=np.arange(4, dtype=float),
    )
    assert tracking["identity_count"] == 4
    assert len(tracking["deciles"]) == 10
