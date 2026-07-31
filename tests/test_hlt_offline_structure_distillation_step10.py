from __future__ import annotations

import numpy as np
import pytest
import torch

from teacher_logit_reco.hlt_offline_structure_distillation import (
    aggregate_confirmation,
    build_confirmation_plan,
    build_confirmation_result,
    build_deployable_export_audit,
    build_scale_execution_plan,
    build_scale_completion,
    build_scale_row_result,
    build_scale_shortlist,
    build_capacity_grid_artifact,
    build_graph_capacity_profile,
    compile_graph_capacity_controls,
    monolithic_grid,
    particle_net_scaled_config,
    build_capacity_control_execution_plan,
    build_capacity_control_result,
    build_target_cache_spec,
    materialize_native_relation_target_from_family_caches,
    native_relation_target_ids,
    publish_target_cache,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (
    with_content_hash,
)
from teacher_logit_reco.hlt_offline_structure_distillation.scale_runtime import (
    build_scale_target_completion,
    build_scale_target_wave_completion,
    offline_to_hlt_target,
)
from teacher_logit_reco.hlt_offline_structure_distillation.extractors import (
    TargetBatch,
)
from teacher_logit_reco.hlt_offline_structure_distillation.target_schemas import (
    target_declarations,
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


def _plan():
    graph_ids = (
        "H_BASE",
        "H_PARTICLENET",
        "H_KD_O_BASE",
        "H_KD_O_FULLREL",
        "A_PHYSICAL",
        "F_BEST",
        "C_BEST",
        "C_PHYSICAL_KD",
    )
    return build_confirmation_plan(
        h_base_graph_id="H_BASE",
        particle_net_graph_id="H_PARTICLENET",
        kd_graph_ids=["H_KD_O_BASE", "H_KD_O_FULLREL"],
        best_physical_aux_graph_id="A_PHYSICAL",
        best_feedback_graph_id="F_BEST",
        best_combination_graph_id="C_BEST",
        physical_kd_graph_id="C_PHYSICAL_KD",
        retb_comparators={"H_RETB_BRIDGE": None},
        parent_lock_hashes={"robustness": "a" * 64},
        source=SOURCE,
        graph_definitions_by_id={
            graph_id: {
                "graph_id": graph_id,
                "graph_kind": (
                    "AUXILIARY"
                    if graph_id == "A_PHYSICAL"
                    else "FEEDBACK"
                    if graph_id == "F_BEST"
                    else "COMBINATION"
                    if graph_id in {"C_BEST", "C_PHYSICAL_KD"}
                    else "BASELINE"
                ),
                **(
                    {"row": {"target_id": "T_OFFLINE_JET_10"}}
                    if graph_id in {"A_PHYSICAL", "F_BEST"}
                    else {
                        "graph": {
                            "members": [
                                {"target_id": "T_OFFLINE_JET_10"}
                            ]
                        }
                    }
                    if graph_id in {"C_BEST", "C_PHYSICAL_KD"}
                    else {"baseline_id": graph_id}
                ),
            }
            for graph_id in graph_ids
        },
    )


def _results(plan):
    labels = np.tile(np.arange(10), 3)
    logits = np.zeros((30, 10))
    metrics = evaluate_classification(logits, labels, split="design_confirm")
    training = [
        build_confirmation_result(
            plan=plan,
            row_id=row["row_id"],
            classification_metrics=metrics,
            checkpoint_sha256="a" * 64,
            prediction_sha256="b" * 64,
            training_completion_sha256="f" * 64,
            deployable_export_sha256="e" * 64,
            deployable_export_file=f"/exports/{row['row_id']}.pt",
            source=SOURCE,
        )
        for row in plan["training_rows"]
    ]
    grid = with_content_hash(
        {
            "contract": "hosd_capacity_grid_v1",
            "schema_version": 1,
            "source": SOURCE,
            "rows": [
                {
                    "kind": "MONOLITHIC",
                    "configuration": {
                        "embed_dim": 128,
                        "particle_blocks": 8,
                        "class_blocks": 2,
                        "attention_heads": 8,
                    },
                    "config_hash": "1" * 64,
                    "trainable_parameter_count": 100,
                    "analytical_flops_batch1_n128": 1000,
                    "flop_ledger": {},
                },
                {
                    "kind": "PARTICLENET",
                    "configuration": {"multiplier": 1.0},
                    "config_hash": "2" * 64,
                    "trainable_parameter_count": 100,
                    "analytical_flops_batch1_n128": 1000,
                    "flop_ledger": {},
                },
            ],
        }
    )
    parent_graphs = sorted(
        {row["parent_graph_id"] for row in plan["capacity_control_rows"]}
    )
    compilations = [
        compile_graph_capacity_controls(
            graph_profile=build_graph_capacity_profile(
                graph_id=graph,
                deployed_parameter_count=100,
                deployed_analytical_flops=1000,
                export_sha256="e" * 64,
                source=SOURCE,
            ),
            grid=grid,
            source=SOURCE,
        )
        for graph in parent_graphs
    ]
    capacity_plan = build_capacity_control_execution_plan(
        confirmation_plan=plan,
        compilations=compilations,
        source=SOURCE,
    )
    capacity = [
        build_capacity_control_result(
            execution_plan=capacity_plan,
            row_id=row["row_id"],
            classification_metrics=metrics,
            checkpoint_sha256="c" * 64,
            prediction_sha256="d" * 64,
            deployable_export_sha256="e" * 64,
            deployable_export_file=f"/exports/{row['row_id']}.pt",
            source=SOURCE,
        )
        for row in capacity_plan["rows"]
    ]
    return training, capacity, capacity_plan


def test_confirmation_requires_all_three_seeds_and_graph_specific_controls():
    plan = _plan()
    assert plan["confirmation_seeds"] == [202, 303, 404]
    assert {row["control_kind"] for row in plan["capacity_control_rows"]} == {
        "H_MONO_PARAM",
        "H_MONO_FLOP",
        "H_PARTICLENET_PARAM",
    }
    training, capacity, capacity_plan = _results(plan)
    with pytest.raises(ValueError, match="seed coverage"):
        aggregate_confirmation(
            plan=plan,
            training_results=training[:-1],
            capacity_results=capacity,
            capacity_execution_plan=capacity_plan,
            source=SOURCE,
        )
    summary = aggregate_confirmation(
        plan=plan,
        training_results=training,
        capacity_results=capacity,
        capacity_execution_plan=capacity_plan,
        source=SOURCE,
    )
    assert summary["all_required_seeds_complete"]
    assert summary["all_capacity_controls_complete"]


def test_all_negative_shortlist_still_scales_duplicate_free_then_exports():
    plan = _plan()
    training, capacity, capacity_plan = _results(plan)
    summary = aggregate_confirmation(
        plan=plan,
        training_results=training,
        capacity_results=capacity,
        capacity_execution_plan=capacity_plan,
        source=SOURCE,
    )
    roles = {
        "H_BASE": "H_BASE",
        "BEST_PHYSICAL_AUX": "A_PHYSICAL",
        "BEST_FEEDBACK": "F_BEST",
        "BEST_COMBINATION": "C_BEST",
        "H_PARTICLENET": "H_PARTICLENET",
    }
    shortlist = build_scale_shortlist(
        confirmation_summary=summary,
        role_graph_ids=roles,
        source=SOURCE,
    )
    assert shortlist["all_negative_campaign_still_shortlisted"]
    assert shortlist["duplicate_free"]
    assert shortlist["graph_count"] <= 7
    scale = build_scale_execution_plan(
        shortlist=shortlist,
        scale_train_manifest_sha256="b" * 64,
        source=SOURCE,
    )
    assert scale["all_statistics_refit_from_scale_train"]
    assert scale["student_training_before_target_completion_allowed"] is False
    scale_results = [
        build_scale_row_result(
            scale_plan=scale,
            graph_id=row["graph_id"],
            seed=row["seed"],
            checkpoint_sha256="1" * 64,
            deployable_export_sha256="2" * 64,
            classification_metrics=training[0]["classification_metrics"],
            analytical_forward_flops_by_role={
                "model_train": 30,
                "val_stop": 20,
                "design_confirm": 10,
            },
            source=SOURCE,
        )
        for row in scale["graph_rows"]
    ]
    completion = build_scale_completion(
        scale_plan=scale,
        teacher_completion_hashes={
            row["teacher_id"]: "3" * 64 for row in scale["teacher_rows"]
        },
        target_completion_hashes={
            row["target_id"]: "4" * 64 for row in scale["target_refit_rows"]
        },
        pre_student_artifact_hashes={
            "scale_inputs": "9" * 64,
            "scale_trees": "a" * 64,
            "scale_normalizers": "b" * 64,
            "teacher_lock": "5" * 64,
            "teacher_adapters": "c" * 64,
            "teacher_outputs": "6" * 64,
            "target_wave": "7" * 64,
            "scale_native_relations": "d" * 64,
            "graph_wave": "8" * 64,
        },
        graph_results=scale_results,
        source=SOURCE,
    )
    assert completion["all_shortlisted_graphs_and_seeds_complete"]
    exports = [
        with_content_hash(
            {
                "contract": "hosd_export_row_v1",
                "schema_version": 1,
                "source": SOURCE,
                "descriptor": {
                    "graph_id": row["graph_id"],
                    "seed": row["seed"],
                },
                "hlt_only": True,
                "forbidden_runtime_dependencies": [],
                "research_export_logits_parity": True,
            }
        )
        for row in scale["graph_rows"]
    ]
    audit = build_deployable_export_audit(
        scale_plan=scale, export_rows=exports, source=SOURCE
    )
    assert audit["all_shortlisted_exports_valid"]
    assert audit["hlt_only"]


def test_scale_target_mapping_and_wave_require_exact_coverage():
    assert (
        offline_to_hlt_target("T_OFFLINE_TRACK_32")
        == "T_HLT_SELF_TRACK_32"
    )
    assert (
        offline_to_hlt_target("T_OFFLINE_RELATION_TRACK")
        == "T_HLT_SELF_RELATION_TRACK"
    )
    assert offline_to_hlt_target("T_OFFLINE_LOGITS_O_BASE") is None
    plan = _plan()
    training, capacity, capacity_plan = _results(plan)
    summary = aggregate_confirmation(
        plan=plan,
        training_results=training,
        capacity_results=capacity,
        capacity_execution_plan=capacity_plan,
        source=SOURCE,
    )
    shortlist = build_scale_shortlist(
        confirmation_summary=summary,
        role_graph_ids={
            "H_BASE": "H_BASE",
            "BEST_PHYSICAL_AUX": "A_PHYSICAL",
            "BEST_FEEDBACK": "F_BEST",
            "BEST_COMBINATION": "C_BEST",
            "H_PARTICLENET": "H_PARTICLENET",
        },
        source=SOURCE,
    )
    scale = build_scale_execution_plan(
        shortlist=shortlist,
        scale_train_manifest_sha256="b" * 64,
        source=SOURCE,
    )
    completions = [
        build_scale_target_completion(
            scale_plan=scale,
            target_id=row["target_id"],
            artifact_hashes={"normalizer": "c" * 64},
            training_target_definitions={
                parameterization: {
                    "mode": "static_cache",
                    "caches": {"shared": "/target"},
                    "normalizer": "/normalizer.json",
                }
                for parameterization in row["required_parameterizations"]
            },
            source=SOURCE,
        )
        for row in scale["target_refit_rows"]
    ]
    wave = build_scale_target_wave_completion(
        scale_plan=scale, completions=completions, source=SOURCE
    )
    assert wave["coverage_exact"]
    with pytest.raises(ValueError, match="coverage"):
        build_scale_target_wave_completion(
            scale_plan=scale,
            completions=completions[:-1],
            source=SOURCE,
        )


def test_native_relation_scale_graph_refits_all_seven_same_view_families():
    shortlist = with_content_hash(
        {
            "contract": "hosd_scale_shortlist_v1",
            "schema_version": 1,
            "source": SOURCE,
            "confirmation_summary_sha256": "1" * 64,
            "graphs": [
                {
                    "graph_id": "H_NATIVE_REL_AUX",
                    "graph_definition": {
                        "graph_id": "H_NATIVE_REL_AUX",
                        "graph_kind": "BASELINE",
                        "baseline_id": "H_NATIVE_REL_AUX",
                    },
                }
            ],
            "graph_ids": ["H_NATIVE_REL_AUX"],
            "graph_count": 1,
            "maximum_graphs": 7,
            "duplicate_free": True,
            "selection_trace": [],
            "all_negative_campaign_still_shortlisted": True,
            "performance_can_disable_scale": False,
        }
    )
    scale = build_scale_execution_plan(
        shortlist=shortlist,
        scale_train_manifest_sha256="2" * 64,
        source=SOURCE,
    )
    target_ids = {row["target_id"] for row in scale["target_refit_rows"]}
    assert target_ids == set(native_relation_target_ids())
    assert all(
        row["build_hlt_analogue"] for row in scale["target_refit_rows"]
    )


def test_scale_native_relation_materialization_is_exact_and_reusable(tmp_path):
    declarations = {
        row.target_id: row for row in target_declarations()
    }
    identities = ("jet-a#0", "jet-b#1")
    roots = {}
    for ordinal, target_id in enumerate(native_relation_target_ids()):
        declaration = declarations[target_id]
        components = tuple(declaration.components)
        root = tmp_path / target_id
        spec = build_target_cache_spec(
            cache_id=f"scale-{target_id}",
            split="scale_train",
            artifact_kind="hlt_analogue",
            identities=identities,
            target_components={target_id: components},
            parent_hashes={"campaign": f"{ordinal + 1:x}" * 64},
            source=SOURCE,
            shard_size=1,
            hlt_replica_id="0",
        )

        def generator(indices, *, _target=target_id, _components=components):
            values = np.repeat(
                np.asarray(indices, dtype=np.float32)[:, None],
                len(_components),
                axis=1,
            )
            return {
                _target: TargetBatch(
                    target_id=_target,
                    component_names=_components,
                    availability_groups=("available",) * len(_components),
                    values=torch.from_numpy(values),
                    loss_mask=torch.ones_like(
                        torch.from_numpy(values), dtype=torch.bool
                    ),
                    diagnostics={},
                )
            }

        publish_target_cache(
            root,
            cache_spec=spec,
            identities=identities,
            generator=generator,
        )
        roots[target_id] = root
    output = tmp_path / "native" / "replica_0.npz"
    first = materialize_native_relation_target_from_family_caches(
        target_cache_roots=roots,
        output_path=output,
        campaign_spec_sha256="f" * 64,
        source=SOURCE,
    )
    second = materialize_native_relation_target_from_family_caches(
        target_cache_roots=roots,
        output_path=output,
        campaign_spec_sha256="f" * 64,
        source=SOURCE,
    )
    assert first == second
    with np.load(output, allow_pickle=False) as payload:
        assert payload["targets"].shape == (2, 545)
        assert payload["target_mask"].all()
        assert payload["availability"].shape == (2, 7)


class _SizedModel(torch.nn.Module):
    def __init__(self, count: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(int(count)))


def test_exact_capacity_grid_and_graph_specific_tie_rules():
    assert len(monolithic_grid()) == 144

    def factory(kind, configuration):
        if kind == "MONOLITHIC":
            count = (
                int(configuration["embed_dim"]) * 1000
                + int(configuration["particle_blocks"]) * 100
                + int(configuration["class_blocks"]) * 10
                + int(configuration["attention_heads"])
            )
        else:
            count = int(float(configuration["multiplier"]) * 100_000)
        return _SizedModel(count)

    grid = build_capacity_grid_artifact(source=SOURCE, model_factory=factory)
    assert grid["row_count"] == 150
    assert grid["monolithic_grid"]["candidate_count"] == 144
    assert all(
        row["trainable_parameter_count"] > 0
        and row["analytical_flops_batch1_n128"] > 0
        for row in grid["rows"]
    )
    profile = build_graph_capacity_profile(
        graph_id="F_BEST",
        deployed_parameter_count=128_800,
        deployed_analytical_flops=1_000_000_000,
        export_sha256="f" * 64,
        source=SOURCE,
    )
    first = compile_graph_capacity_controls(
        graph_profile=profile, grid=grid, source=SOURCE
    )
    second = compile_graph_capacity_controls(
        graph_profile=profile, grid=grid, source=SOURCE
    )
    assert first == second
    assert set(first["selected_controls"]) == {
        f"H_MONO_PARAM_{first['selected_controls'].keys().__iter__().__next__().split('_')[-1]}",
        *[
            key
            for key in first["selected_controls"]
            if key.startswith(("H_MONO_FLOP_", "H_PARTICLENET_PARAM_"))
        ],
    }
    assert first["performance_read"] is False


def test_particle_net_multiplier_rounding_is_frozen():
    half = particle_net_scaled_config(0.5)
    assert half["conv_params"] == [
        (16, (32, 32, 32)),
        (16, (64, 64, 64)),
        (16, (128, 128, 128)),
    ]
    assert half["fc_params"] == [(128, 0.1)]
