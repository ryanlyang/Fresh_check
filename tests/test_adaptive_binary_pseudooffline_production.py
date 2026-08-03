from __future__ import annotations

from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from jetclass_fresh.jetclass_data import JetIdentity

from scripts.train_adaptive_binary_pseudooffline_variant import (
    _load_selected_hlt_encoder,
    _selected_classifier_metrics,
)
from teacher_logit_reco.adaptive_binary_pseudooffline.tagger_runtime import (
    _combine_report_target_provenance,
    _joint_reconstruction_loss,
    _selected_target_provenance,
)
from teacher_logit_reco.adaptive_binary_pseudooffline import production as production_module
from teacher_logit_reco.adaptive_binary_pseudooffline.production import (
    AdaptiveBinaryTargetBatchSource,
    _concatenate_target_batches,
    _slice_target_batch,
    campaign_target_source_kwargs,
)

from teacher_logit_reco.adaptive_binary_pseudooffline import (
    AccountingState,
    AdaptiveBinaryReconstructionOutput,
    AdaptiveBinaryReconstructorModel,
    AdaptiveBinaryHierarchyLayout,
    PseudoViewInputs,
    CurriculumState,
    ReconstructorStepContext,
    CurriculumController,
    ReconstructorCurriculumConfig,
    ReconstructionLossWeights,
    build_adaptive_binary_targets,
    build_variant_hierarchy_aware_tagger,
    compose_reconstruction_loss,
    package_trainable_pseudo_views,
    reconstructor_step,
    resolve_variant_config,
)


def _hlt_batch() -> tuple[torch.Tensor, torch.Tensor]:
    tokens = torch.zeros(2, 128, 14)
    mask = torch.zeros(2, 128, dtype=torch.bool)
    for batch_index, count in enumerate((5, 7)):
        mask[batch_index, :count] = True
        tokens[batch_index, :count, 0] = torch.linspace(28.0, 4.0, count)
        tokens[batch_index, :count, 1] = torch.linspace(-0.25, 0.25, count)
        tokens[batch_index, :count, 2] = torch.linspace(-0.3, 0.3, count)
        momentum = tokens[batch_index, :count, 0] * torch.cosh(
            tokens[batch_index, :count, 1]
        )
        tokens[batch_index, :count, 3] = torch.sqrt(momentum.square() + 0.13957**2)
        tokens[batch_index, :count, 4] = 1.0
        tokens[batch_index, :count, 5] = 1.0
    return tokens, mask


def test_campaign_target_source_kwargs_bind_canonical_inputs(tmp_path) -> None:
    root = tmp_path / "campaign"
    report = root / "audits" / "target_mode_selection.json"
    report.parent.mkdir(parents=True)
    report.write_text("{}", encoding="utf-8")

    kwargs = campaign_target_source_kwargs(root)

    assert kwargs == {
        "target_mode_report": report.resolve(),
        "offline_cache_dir": (root / "inputs" / "offline_cache").resolve(),
        "manifest_path": (
            root / "inputs" / "split_manifest" / "split_manifest.json.gz"
        ).resolve(),
    }


def _reconstruction_batch(grouping: str = "exclusive_kt") -> dict:
    hlt, hlt_mask = _hlt_batch()
    offline = hlt.clone()
    offline_mask = hlt_mask.clone()
    for batch_index, start in enumerate((5, 7)):
        for particle_index in range(start, start + 3):
            offline_mask[batch_index, particle_index] = True
            offline[batch_index, particle_index, 0] = 3.0 + particle_index
            offline[batch_index, particle_index, 1] = 0.03 * particle_index
            offline[batch_index, particle_index, 2] = -0.04 * particle_index
            momentum = offline[batch_index, particle_index, 0] * torch.cosh(
                offline[batch_index, particle_index, 1]
            )
            offline[batch_index, particle_index, 3] = torch.sqrt(
                momentum.square() + 0.13957**2
            )
            offline[batch_index, particle_index, 4] = 1.0
            offline[batch_index, particle_index, 5] = 1.0
    identities = tuple(
        JetIdentity(file=f"HToBB_{index:03d}.root", entry=index, label=1)
        for index in range(2)
    )
    targets = build_adaptive_binary_targets(
        hlt.numpy(),
        hlt_mask.numpy(),
        offline.numpy(),
        offline_mask.numpy(),
        jet_ids=identities,
        layout=AdaptiveBinaryHierarchyLayout(grouping=grouping),
    )
    return {"hlt_tokens": hlt, "hlt_mask": hlt_mask, "targets": targets}


def _phase2_context(capacity: int, supervised_capacities: tuple[int, ...]):
    return ReconstructorStepContext(
        curriculum=CurriculumState(
            stage_index=1,
            stage_key=f"phase2_hierarchy_{capacity}",
            phase=2,
            phase_name="progressive_hierarchy",
            global_update=1,
            stage_update=1,
            stage_maximum_updates=2,
            active_capacity=capacity,
            stage_progress=0.5,
            teacher_forcing_probability=1.0,
            distribution_weight=0.0,
            supervised_capacities=supervised_capacities,
        ),
        split="model_train",
        mode="teacher_forced",
        validation=False,
        teacher_forcing_probability=1.0,
    )


def _phase1_context() -> ReconstructorStepContext:
    return ReconstructorStepContext(
        curriculum=CurriculumState(
            stage_index=0,
            stage_key="phase1_root",
            phase=1,
            phase_name="root_pretraining",
            global_update=1,
            stage_update=1,
            stage_maximum_updates=2,
            active_capacity=1,
            stage_progress=0.5,
            teacher_forcing_probability=1.0,
            distribution_weight=0.0,
            supervised_capacities=(),
        ),
        split="model_train",
        mode="teacher_forced",
        validation=False,
        teacher_forcing_probability=1.0,
    )


def _phase4_context() -> ReconstructorStepContext:
    return ReconstructorStepContext(
        curriculum=CurriculumState(
            stage_index=7,
            stage_key="phase4_distribution",
            phase=4,
            phase_name="probabilistic_multi_hypothesis",
            global_update=1,
            stage_update=1,
            stage_maximum_updates=2,
            active_capacity=32,
            stage_progress=0.5,
            teacher_forcing_probability=0.0,
            distribution_weight=0.25,
            supervised_capacities=(2, 4, 8, 16, 32),
        ),
        split="model_train",
        mode="rollout",
        validation=False,
        teacher_forcing_probability=0.0,
    )


def _phase4_kt8_context() -> ReconstructorStepContext:
    return ReconstructorStepContext(
        curriculum=CurriculumState(
            stage_index=5,
            stage_key="phase4_distribution",
            phase=4,
            phase_name="probabilistic_multi_hypothesis",
            global_update=1,
            stage_update=1,
            stage_maximum_updates=2,
            active_capacity=8,
            stage_progress=0.5,
            teacher_forcing_probability=0.0,
            distribution_weight=0.25,
            supervised_capacities=(2, 4, 8),
        ),
        split="model_train",
        mode="rollout",
        validation=False,
        teacher_forcing_probability=0.0,
    )


def _phase3_context() -> ReconstructorStepContext:
    return ReconstructorStepContext(
        curriculum=CurriculumState(
            stage_index=6,
            stage_key="phase3_renderer",
            phase=3,
            phase_name="particle_renderer",
            global_update=1,
            stage_update=1,
            stage_maximum_updates=2,
            active_capacity=32,
            stage_progress=0.5,
            teacher_forcing_probability=0.0,
            distribution_weight=0.0,
            supervised_capacities=(2, 4, 8, 16, 32),
        ),
        split="model_train",
        mode="rollout",
        validation=False,
        teacher_forcing_probability=0.0,
    )


def _reference_all_at_once_deployment(
    model: AdaptiveBinaryReconstructorModel,
    shared: dict,
) -> AdaptiveBinaryReconstructionOutput:
    root_state = AccountingState.from_ledger(shared["compiled_root"].root_ledger)
    hierarchy = model.hierarchy_reconstructor.deployment_rollout(
        root_state,
        shared["root_prediction"].shared_context,
        shared["root_prediction"].query_tokens,
        shared["particle_embeddings"],
        shared["mask"],
        shared["support"],
        evaluation_seed=24731,
    )
    rendered = model._render_hypotheses(
        root_prediction=shared["root_prediction"],
        hypotheses=hierarchy.hypotheses,
        particle_embeddings=shared["particle_embeddings"],
        hlt_mask=shared["mask"],
        support=shared["support"],
        axis_eta=shared["axis_eta"],
        axis_phi=shared["axis_phi"],
    )
    return AdaptiveBinaryReconstructionOutput(
        root_prediction=shared["root_prediction"],
        compiled_root=shared["compiled_root"],
        root_state=root_state,
        hlt_particle_embeddings=shared["particle_embeddings"],
        hlt_jet_embedding=shared["jet_embedding"],
        hlt_support_features=shared["support"],
        hierarchy_output=hierarchy,
        rendered_views=rendered,
    )


def test_c0_is_one_shot_and_skips_progressive_two_and_four_stages():
    model = AdaptiveBinaryReconstructorModel(
        variant_name="C0_direct_8_group_set", smoke=True
    )
    assert model.direct_group_set is True
    assert model.direct_set_decoder is not None
    config = ReconstructorCurriculumConfig(
        root_updates=1,
        hierarchy_updates_per_depth=1,
        renderer_updates=1,
        distribution_updates=1,
        evaluation_interval=1,
        maximum_capacity=8,
        hierarchy_capacities=(8,),
        renderer_enabled=False,
        distribution_enabled=False,
    )
    controller = CurriculumController(config)
    assert [stage.active_capacity for stage in controller.stages] == [1, 8]


def test_c8_has_independent_raw_child_heads_outside_compiler_parameters():
    model = AdaptiveBinaryReconstructorModel(
        variant_name="C8_unconstrained_split_control", smoke=True
    )
    assert model.constrained_hierarchy is False
    assert len(model.unconstrained_child_heads) == 5
    groups = model.module_groups()
    head_parameter = next(model.unconstrained_child_heads[2].parameters())
    assert any(parameter is head_parameter for parameter in groups["hierarchy_8"].parameters())


def test_c0_and_c8_training_losses_reach_their_distinct_heads():
    batch = _reconstruction_batch()
    direct = AdaptiveBinaryReconstructorModel(
        variant_name="C0_direct_8_group_set", smoke=True
    )
    direct_result = reconstructor_step(
        direct, batch, _phase2_context(8, (8,))
    )
    assert direct_result.metrics["mode"] == "direct_set"
    direct_result.loss_terms["group_8"].backward()
    assert any(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in direct.direct_set_decoder.parameters()
    )

    unconstrained = AdaptiveBinaryReconstructorModel(
        variant_name="C8_unconstrained_split_control", smoke=True
    )
    unconstrained_result = reconstructor_step(
        unconstrained, batch, _phase2_context(2, (2, 4, 8, 16, 32))
    )
    assert unconstrained_result.metrics["hierarchy_constraints_in_loss"] is False
    unconstrained_result.loss_terms["group_2"].backward()
    assert any(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in unconstrained.unconstrained_child_heads[0].parameters()
    )


def test_phase1_root_pretraining_reports_no_rollout_without_unbound_state():
    model = AdaptiveBinaryReconstructorModel(
        variant_name="B1_semantic_query_root", smoke=True
    ).eval()

    result = reconstructor_step(model, _reconstruction_batch(), _phase1_context())

    assert result.metrics["mode"] == "teacher_forced"
    assert result.metrics["rollout_forward_executed"] is False
    assert result.metrics["hypothesis_zero_reused"] is False
    assert set(result.loss_terms) == {"root"}


def test_teacher_forced_phase2_omits_the_inactive_rollout_forward():
    model = AdaptiveBinaryReconstructorModel(
        variant_name="C5_kt_32", smoke=True
    ).eval()
    decoder = model.hierarchy_reconstructor.decoders["exclusive_kt"]
    calls = {"teacher_forced": 0, "rollout": 0}

    def count_mode(_module, _args, kwargs, _output):
        calls[str(kwargs["mode"])] += 1

    hook = decoder.register_forward_hook(count_mode, with_kwargs=True)
    result = reconstructor_step(
        model,
        _reconstruction_batch(),
        _phase2_context(32, (2, 4, 8, 16, 32)),
    )
    hook.remove()

    assert calls == {"teacher_forced": 1, "rollout": 0}
    assert result.metrics["rollout_forward_executed"] is False
    assert "frontier" not in result.loss_terms
    assert {"root", "group_2", "group_4", "group_8", "group_16", "group_32", "topology"} <= set(
        result.loss_terms
    )


def test_phase4_rolls_and_renders_hypothesis_zero_exactly_once(monkeypatch):
    model = AdaptiveBinaryReconstructorModel(
        variant_name="D1_kt32_mh4_particles", smoke=True
    ).eval()
    model.renderer.config = replace(
        model.renderer.config, exact_nbody_projection=False
    )
    decoder = model.hierarchy_reconstructor.decoders["exclusive_kt"]
    decoder_calls = {"teacher_forced": 0, "rollout": 0}
    renderer_calls = 0
    root_calls = 0
    assembly = {}

    def count_decoder(_module, _args, kwargs, _output):
        decoder_calls[str(kwargs["mode"])] += 1

    def count_renderer(_module, _args, _output):
        nonlocal renderer_calls
        renderer_calls += 1

    def count_root(_module, _args, _output):
        nonlocal root_calls
        root_calls += 1

    original_assemble = model.assemble_deployment_output

    def capture_assembly(**kwargs):
        output = original_assemble(**kwargs)
        assembly.update(kwargs)
        assembly["output"] = output
        return output

    monkeypatch.setattr(model, "assemble_deployment_output", capture_assembly)

    handles = (
        decoder.register_forward_hook(count_decoder, with_kwargs=True),
        model.renderer.register_forward_hook(count_renderer),
        model.root_predictor.register_forward_hook(count_root),
    )
    result = reconstructor_step(model, _reconstruction_batch(), _phase4_context())
    for handle in handles:
        handle.remove()

    assert root_calls == 1
    assert decoder_calls == {"teacher_forced": 1, "rollout": 5}
    assert renderer_calls == 5
    assert result.metrics["hypothesis_zero_reused"] is True
    zero = assembly["hypothesis_zero"]
    output = assembly["output"]
    assert output.compiled_root is assembly["compiled_root"]
    assert output.hierarchy_output.hypotheses[0] is zero.hypotheses[0]
    assert output.rendered_views["exclusive_kt"][0] is zero.rendered_views[
        "exclusive_kt"
    ][0]


def test_d7_phase4_stops_and_renders_at_the_configured_kt8_frontier():
    model = AdaptiveBinaryReconstructorModel(
        variant_name="D7_kt8_mh4_particles_screen", smoke=True
    ).eval()
    model.renderer.config = replace(
        model.renderer.config, exact_nbody_projection=False
    )
    rendered_capacities = []

    def capture_frontier(_module, args):
        rendered_capacities.append(args[0].final_frontier.capacity)

    handle = model.renderer.register_forward_pre_hook(capture_frontier)
    result = reconstructor_step(
        model, _reconstruction_batch(), _phase4_kt8_context()
    )
    handle.remove()

    assert model.hierarchy_capacities == (2, 4, 8)
    assert len(model.hierarchy_reconstructor.decoders["exclusive_kt"].levels) == 3
    assert rendered_capacities == [8, 8, 8, 8, 8]
    assert result.metrics["active_capacity"] == 8
    assert {"group_2", "group_4", "group_8", "particle", "distribution"} <= set(
        result.loss_terms
    )
    assert "group_16" not in result.loss_terms
    assert "group_32" not in result.loss_terms
    module_groups = model.module_groups()
    assert list(module_groups["hierarchy_16"].parameters()) == []
    assert list(module_groups["hierarchy_32"].parameters()) == []


def test_d5_oracle_groups_share_the_offline_root_and_close_particle_counts():
    model = AdaptiveBinaryReconstructorModel(
        variant_name="D5_oracle_groups_particles", smoke=True
    ).eval()
    model.renderer.config = replace(
        model.renderer.config, exact_nbody_projection=False
    )

    result = reconstructor_step(model, _reconstruction_batch(), _phase3_context())

    assert result.metrics["oracle_groups_supplied"] is True


def test_c9_oracle_parents_use_rollout_alignment_for_predicted_children():
    model = AdaptiveBinaryReconstructorModel(
        variant_name="C9_oracle_parent_rollout", smoke=True
    ).eval()
    model.renderer.config = replace(
        model.renderer.config, exact_nbody_projection=False
    )

    result = reconstructor_step(model, _reconstruction_batch(), _phase3_context())

    assert result.metrics["oracle_parent_rollout"] is True
    assert result.metrics["rollout_forward_executed"] is True
    assert "frontier" in result.loss_terms
    assert bool(torch.isfinite(result.loss_terms["frontier"]))
    assert result.metrics["rollout_forward_executed"] is True
    assert "particle" in result.loss_terms
    assert bool(torch.isfinite(result.loss_terms["particle"]))


@pytest.mark.parametrize(
    ("variant_name", "context_factory"),
    (
        (
            "B1_semantic_query_root",
            lambda: _phase2_context(32, (2, 4, 8, 16, 32)),
        ),
        ("D1_kt32_mh4_particles", _phase4_context),
    ),
)
def test_production_reconstructor_step_is_bfloat16_autocast_safe(
    variant_name, context_factory
):
    torch.manual_seed(971)
    model = AdaptiveBinaryReconstructorModel(
        variant_name=variant_name, smoke=True
    ).train()
    if variant_name == "D1_kt32_mh4_particles":
        model.renderer.config = replace(
            model.renderer.config, phase_space_iterations=64
        )
    context = context_factory()

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        result = reconstructor_step(model, _reconstruction_batch(), context)
    loss = compose_reconstruction_loss(
        result, context, ReconstructionLossWeights()
    ).total

    assert loss.dtype == torch.float32
    assert bool(torch.isfinite(loss))
    assert all(term.dtype == torch.float32 for term in result.loss_terms.values())
    loss.backward()
    assert any(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )


def test_split_phase4_forward_matches_reference_losses_and_gradients():
    torch.manual_seed(8807)
    optimized = AdaptiveBinaryReconstructorModel(
        variant_name="D1_kt32_mh4_particles", smoke=True
    ).eval()
    reference = AdaptiveBinaryReconstructorModel(
        variant_name="D1_kt32_mh4_particles", smoke=True
    ).eval()
    reference.load_state_dict(optimized.state_dict(), strict=True)
    for model in (optimized, reference):
        model.renderer.config = replace(
            model.renderer.config, exact_nbody_projection=False
        )
    batch = _reconstruction_batch()
    context = _phase4_context()

    optimized_result = reconstructor_step(optimized, batch, context)
    reference_shared = reference.prepare_shared_reconstruction_forward(
        batch["hlt_tokens"], batch["hlt_mask"]
    )
    reference_deployment = _reference_all_at_once_deployment(
        reference, reference_shared
    )
    reference_result = reconstructor_step(
        reference,
        {
            **batch,
            "shared_reconstructor_forward": reference_shared,
            "shared_deployment_output": reference_deployment,
        },
        context,
    )

    assert optimized_result.metrics["hypothesis_zero_reused"] is True
    assert reference_result.metrics["hypothesis_zero_reused"] is True
    assert set(optimized_result.loss_terms) == set(reference_result.loss_terms)
    for name in optimized_result.loss_terms:
        torch.testing.assert_close(
            optimized_result.loss_terms[name],
            reference_result.loss_terms[name],
            rtol=2.0e-5,
            atol=2.0e-6,
        )
    optimized_loss = compose_reconstruction_loss(
        optimized_result, context, ReconstructionLossWeights()
    ).total
    reference_loss = compose_reconstruction_loss(
        reference_result, context, ReconstructionLossWeights()
    ).total
    torch.testing.assert_close(optimized_loss, reference_loss, rtol=2.0e-5, atol=2.0e-6)
    optimized_loss.backward()
    reference_loss.backward()

    reference_parameters = dict(reference.named_parameters())
    for name, parameter in optimized.named_parameters():
        reference_gradient = reference_parameters[name].grad
        if parameter.grad is None or reference_gradient is None:
            assert parameter.grad is None and reference_gradient is None, name
            continue
        assert bool(torch.isfinite(parameter.grad).all()), name
        assert bool(torch.isfinite(reference_gradient).all()), name
        torch.testing.assert_close(
            parameter.grad,
            reference_gradient,
            rtol=2.0e-3,
            atol=1.0e-3,
            msg=lambda message, name=name: f"{name}: {message}",
        )


def test_b3_draws_four_distinct_compiled_root_states_from_hlt_only():
    resolved = resolve_variant_config("B3_root_sampled_ablation")
    assert resolved["model"]["distribution"]["enabled"] is False
    assert resolved["model"]["distribution"]["sample_root"] is True
    model = AdaptiveBinaryReconstructorModel(
        variant_name="B3_root_sampled_ablation", smoke=True
    ).eval()
    tokens, mask = _hlt_batch()
    with torch.no_grad():
        roots = model.sample_compiled_roots(tokens, mask, count=4, seed=27191)
    ledgers = torch.stack([root.root_ledger for root in roots], dim=1)
    assert ledgers.shape[1] == 4
    assert float(ledgers.var(dim=1, unbiased=False).max()) > 0.0
    assert resolved["evaluation"]["sampled_root_downstream_rollout"] is False


def test_f0_joint_objective_updates_both_hierarchy_branches_from_one_model():
    resolved = resolve_variant_config("F0_ce_reco_primary")
    assert resolved["model"]["fusion"]["dual_hierarchy"] is True
    assert "E7_dual_hierarchy_dualcross" in resolved["variant"]["dependencies"]
    model = AdaptiveBinaryReconstructorModel(
        hierarchy_names=("exclusive_kt", "cambridge_aachen"),
        variant_name="F0_ce_reco_primary",
        smoke=True,
    )
    for renderer in model.renderers.values():
        renderer.config = replace(renderer.config, exact_nbody_projection=False)
    root_forward_count = 0
    renderer_forward_counts = {
        name: 0 for name in ("exclusive_kt", "cambridge_aachen")
    }

    def count_root_forward(_module, _inputs, _output):
        nonlocal root_forward_count
        root_forward_count += 1

    hook = model.root_predictor.register_forward_hook(count_root_forward)
    renderer_hooks = []
    for hierarchy_name, renderer in model.renderers.items():
        def count_renderer_forward(_module, _inputs, _output, *, name=hierarchy_name):
            renderer_forward_counts[name] += 1

        renderer_hooks.append(
            renderer.register_forward_hook(count_renderer_forward)
        )
    kt_batch = _reconstruction_batch("exclusive_kt")
    ca_batch = _reconstruction_batch("cambridge_aachen")
    shared_forward = model.prepare_shared_reconstruction_forward(
        kt_batch["hlt_tokens"], kt_batch["hlt_mask"]
    )
    shared_deployment = model.deploy_from_shared_reconstruction_forward(
        shared_forward, evaluation_seed=24731
    )
    # This is the exact pseudo-view object consumed by the F0 CE path.
    consumer_pseudo = package_trainable_pseudo_views(shared_deployment)
    consumer_pseudo.validate()
    assert consumer_pseudo.diagnostics["consumer_only_pseudo"] is True
    assert "particle__exclusive_kt__four_vector" not in consumer_pseudo.arrays
    assert "frontier__exclusive_kt__depth_00__source_child_indices" not in consumer_pseudo.arrays
    deployment_renderer_counts = dict(renderer_forward_counts)
    loss = _joint_reconstruction_loss(
        model,
        {
            **kt_batch,
            "targets_by_hierarchy": {
                "exclusive_kt": kt_batch["targets"],
                "cambridge_aachen": ca_batch["targets"],
            },
            "shared_reconstructor_forward": shared_forward,
            "shared_deployment_output": shared_deployment,
        },
        split="model_train",
        validation=False,
    )
    assert renderer_forward_counts == deployment_renderer_counts
    loss.backward()
    hook.remove()
    for renderer_hook in renderer_hooks:
        renderer_hook.remove()
    assert root_forward_count == 1
    for hierarchy_name in ("exclusive_kt", "cambridge_aachen"):
        assert any(
            parameter.grad is not None
            and bool(torch.isfinite(parameter.grad).all())
            and float(parameter.grad.abs().sum()) > 0.0
            for parameter in model.hierarchy_reconstructor.decoders[
                hierarchy_name
            ].parameters()
        )


def _target_report_provenance(*, target_hash: str, grouping_hash: str) -> dict:
    return {
        "source_manifest_hash": "manifest",
        "jet_identity_hash": "jets",
        "label_hash": "labels",
        "class_mapping_hash": "classes",
        "hlt_content_hash": "hlt",
        "hlt_profile": "fixed_hlt_v2_realistic",
        "hlt_profile_version": "2.0",
        "hlt_degradation_strength": 2.5,
        "hlt_params_hash": "hlt-params",
        "offline_cache_content_hash": "offline",
        "hierarchy_target_content_hash": target_hash,
        "hierarchy_target_schema_hash": "schema",
        "grouping_algorithm_hash": grouping_hash,
        "root_ledger_schema_hash": "root-schema",
        "normalization_hash": "normalization",
    }


def test_e7_target_provenance_selects_and_binds_both_hierarchies(tmp_path):
    branches = {
        "D1_kt32_mh4_particles": _target_report_provenance(
            target_hash="kt-target", grouping_hash="kt-grouping"
        ),
        "D2_ca32_mh4_particles": _target_report_provenance(
            target_hash="ca-target", grouping_hash="ca-grouping"
        ),
    }
    for run_name, provenance in branches.items():
        path = tmp_path / "runs" / run_name / "run_report.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"provenance": {"model_val": provenance}}),
            encoding="utf-8",
        )
    selected = _selected_target_provenance(
        tmp_path, ("E7_shared_root_dual",)
    )
    assert selected["dual_target_provenance"] is True
    assert set(selected["hierarchy_branches"]) == {
        "exclusive_kt",
        "cambridge_aachen",
    }
    assert (
        selected["hierarchy_branches"]["exclusive_kt"][
            "hierarchy_target_content_hash"
        ]
        == "kt-target"
    )
    assert (
        selected["hierarchy_branches"]["cambridge_aachen"][
            "hierarchy_target_content_hash"
        ]
        == "ca-target"
    )


def test_supplemental_kt8_target_provenance_uses_its_own_report(tmp_path):
    provenance = _target_report_provenance(
        target_hash="kt8-target", grouping_hash="kt8-grouping"
    )
    path = (
        tmp_path
        / "runs"
        / "D7_kt8_mh4_particles_screen"
        / "run_report.json"
    )
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"provenance": {"model_val": provenance}}),
        encoding="utf-8",
    )

    selected = _selected_target_provenance(
        tmp_path, ("D7_kt8_mh4_particles_screen",)
    )
    assert selected["hierarchy_target_content_hash"] == "kt8-target"
    assert selected["grouping_algorithm_hash"] == "kt8-grouping"


def test_dual_target_provenance_rejects_conflicting_offline_cache():
    kt = _target_report_provenance(
        target_hash="kt-target", grouping_hash="kt-grouping"
    )
    ca = _target_report_provenance(
        target_hash="ca-target", grouping_hash="ca-grouping"
    )
    ca["offline_cache_content_hash"] = "stale-offline"
    with pytest.raises(ValueError, match="offline_cache_content_hash"):
        _combine_report_target_provenance(
            {"exclusive_kt": kt, "cambridge_aachen": ca}
        )


def test_d3_deploys_one_true_global_particle_set():
    model = AdaptiveBinaryReconstructorModel(
        variant_name="D3_global_particle_set", smoke=True
    ).eval()
    model.renderer.config = replace(
        model.renderer.config, exact_nbody_projection=False
    )
    tokens, mask = _hlt_batch()
    with torch.no_grad():
        output = model.deploy(tokens, mask, evaluation_seed=24731)
    rendered = output.rendered_views["exclusive_kt"][0]
    assert not bool(rendered.diagnostics["group_local_self_attention"])
    assert rendered.diagnostics["renderer_grouping"] == "single_global_root_set"
    assert bool((rendered.group_indices[rendered.mask] == 0).all())


def test_e7_root_hashes_are_computed_from_each_branch_independently():
    batch = 2
    views = 5
    root = torch.randn(batch, 30)
    uncertainty = torch.zeros(batch, views, 1)
    frontier_mask = torch.ones(batch, views, 1, dtype=torch.bool)
    arrays = {"shared_root_ledger": root}
    for hierarchy_name in ("exclusive_kt", "cambridge_aachen"):
        prefix = f"frontier__{hierarchy_name}__depth_00__"
        arrays[prefix + "ledger"] = root[:, None, None].expand(
            -1, views, 1, -1
        ).contiguous()
        arrays[prefix + "uncertainty"] = uncertainty.clone()
        arrays[prefix + "mask"] = frontier_mask.clone()
    pseudo = PseudoViewInputs(
        arrays=arrays,
        view_names=tuple(f"view_{index}" for index in range(views)),
        hierarchy_names=("exclusive_kt", "cambridge_aachen"),
        frontier_depths={"exclusive_kt": 1, "cambridge_aachen": 1},
        diagnostics={},
    )
    tagger = build_variant_hierarchy_aware_tagger(
        "E7_dual_hierarchy_dualcross", smoke=True
    ).eval()
    root = tagger._root_provenance(pseudo, None, compute_hashes=True)
    assert root["branch_hashes_computed_independently"] is True
    assert root["root_hash_count"] == 1
    assert len(root["branch_root_hashes"]) == 2
    assert set(root["branch_root_hashes"].values()) == {root["root_hash"]}


def test_capacity_controls_are_distinct_and_make_no_exact_match_claim():
    a2 = resolve_variant_config("A2_hlt_capacity_control")
    a5 = resolve_variant_config("A5_hlt_part_xl")
    assert a2["model"]["hlt_part"]["num_layers"] == 20
    assert a5["model"]["hlt_part"]["num_layers"] == 16
    assert a2["model"]["hlt_part"] != a5["model"]["hlt_part"]


def test_e3_is_one_unidirectional_cross_attention_block():
    model = build_variant_hierarchy_aware_tagger(
        "E3_single_cross_attention", smoke=True
    )
    assert len(model.fusion_stacks) == 1
    assert len(model.fusion_stacks[0]) == 1
    assert model.fusion_stacks[0][0].update_pseudo_stream is False


def test_selected_baseline_metrics_are_computed_from_the_saved_checkpoint(tmp_path):
    class _FixedClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(10))

        def forward(self, points, features, vectors, mask):
            del points, vectors, mask
            raw_pt = torch.exp(features[:, 0, 0] / 0.7 + 1.7)
            labels = (raw_pt.round().long() - 1).clamp(0, 9)
            logits = torch.full(
                (features.shape[0], 10),
                -8.0,
                dtype=features.dtype,
                device=features.device,
            )
            logits.scatter_(1, labels[:, None], 8.0)
            return logits + self.bias

    model = _FixedClassifier()
    checkpoint = tmp_path / "best_model_val.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint)
    tokens = np.zeros((10, 128, 14), dtype=np.float32)
    mask = np.zeros((10, 128), dtype=np.bool_)
    mask[:, 0] = True
    # The dummy classifier exactly inverts build_part_inputs_torch's first
    # log-pT feature, making every class deterministically recoverable.
    tokens[:, 0, 0] = np.arange(1, 11, dtype=np.float32)
    tokens[:, 0, 3] = np.arange(1, 11, dtype=np.float32) + 1.0
    view = SimpleNamespace(
        tokens=tokens,
        mask=mask,
        labels=np.arange(10, dtype=np.int64),
    )
    metrics = _selected_classifier_metrics(
        model,
        checkpoint,
        view,
        device="cpu",
        batch_size=4,
        smoke=False,
    )
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_ovr_auc"] == 1.0
    assert len(metrics["per_class"]) == 10
    assert np.asarray(metrics["confusion_matrix"]).shape == (10, 10)


def test_primary_root_models_require_and_load_the_selected_a0_encoder(tmp_path):
    for name in (
        "B0_pooled_mlp_root",
        "B1_semantic_query_root",
        "B2_semantic_query_probabilistic",
    ):
        assert "A0_hlt_part" in resolve_variant_config(name)["variant"]["dependencies"]

    source = torch.nn.Linear(3, 2)
    with torch.no_grad():
        source.weight.fill_(0.25)
        source.bias.fill_(-0.5)
    checkpoint = tmp_path / "best_model_val.pt"
    torch.save({"model_state_dict": source.state_dict()}, checkpoint)
    target = torch.nn.Linear(3, 2)
    model = SimpleNamespace(
        hlt_encoder=SimpleNamespace(reference_model=target),
        smoke=False,
    )
    report = _load_selected_hlt_encoder(model, checkpoint)
    assert report["loaded"] is True
    assert report["source_variant"] == "A0_hlt_part"
    assert torch.equal(target.weight, source.weight)
    assert torch.equal(target.bias, source.bias)


def test_training_source_fills_microbatches_across_target_shards(monkeypatch):
    tokens, mask = _hlt_batch()
    tokens = torch.cat((tokens, tokens), dim=0).numpy()
    mask = torch.cat((mask, mask), dim=0).numpy()
    labels = np.arange(4, dtype=np.int64)
    jet_ids = tuple(
        JetIdentity(file="HToBB_010.root", entry=index, label=int(labels[index]))
        for index in range(4)
    )
    base_targets = _reconstruction_batch()["targets"]
    targets = _concatenate_target_batches((base_targets, base_targets))
    shards = (
        SimpleNamespace(
            targets=_slice_target_batch(targets, 0, 3),
            jet_ids=jet_ids[:3],
            start=0,
        ),
        SimpleNamespace(
            targets=_slice_target_batch(targets, 3, 4),
            jet_ids=jet_ids[3:],
            start=3,
        ),
    )
    hlt_view = SimpleNamespace(
        tokens=tokens,
        mask=mask,
        labels=labels,
        jet_ids=jet_ids,
        metadata={"hlt_content_hash": "hlt-hash"},
    )
    metadata = {
        "n_shards": 2,
        "shards": [
            {"shard_index": 0, "start": 0, "stop": 3},
            {"shard_index": 1, "start": 3, "stop": 4},
        ],
        "hlt_content_hash": "hlt-hash",
        "jet_identity_hash": production_module.jet_identity_hash(jet_ids),
    }
    monkeypatch.setattr(
        production_module, "load_cached_hlt_view", lambda *_args, **_kwargs: hlt_view
    )
    monkeypatch.setattr(
        production_module,
        "load_adaptive_binary_target_cache_metadata",
        lambda *_args, **_kwargs: metadata,
    )
    monkeypatch.setattr(
        production_module,
        "load_adaptive_binary_target_shard",
        lambda _root, _split, _grouping, index, **_kwargs: shards[index],
    )

    training_source = AdaptiveBinaryTargetBatchSource(
        hlt_cache_dir="unused",
        target_cache_dir="unused",
        split="model_train",
        grouping="exclusive_kt",
        batch_size=4,
        shuffle_shards=False,
        seed=24731,
    )
    batch = training_source.next_batch()
    assert batch["hlt_tokens"].shape[0] == 4
    assert batch["targets"].n_jets == 4
    assert batch["indices"].tolist() == [0, 1, 2, 3]
    assert batch["targets"].diagnostics["assembled_target_chunks"] == 2

    validation_source = AdaptiveBinaryTargetBatchSource(
        hlt_cache_dir="unused",
        target_cache_dir="unused",
        split="model_val",
        grouping="exclusive_kt",
        batch_size=4,
        shuffle_shards=False,
        seed=24731,
    )
    validation_batches = list(validation_source.iter_epoch())
    assert [row["hlt_tokens"].shape[0] for row in validation_batches] == [4]
    assert sum(row["hlt_tokens"].shape[0] for row in validation_batches) == 4
    assert validation_source.last_validation_range.n_jets == 4

    bounded_validation_source = AdaptiveBinaryTargetBatchSource(
        hlt_cache_dir="unused",
        target_cache_dir="unused",
        split="model_val",
        grouping="exclusive_kt",
        batch_size=4,
        shuffle_shards=False,
        seed=24731,
        validation_maximum_jets=2,
    )
    bounded_batches = list(bounded_validation_source.iter_epoch())
    assert [row["indices"].tolist() for row in bounded_batches] == [[0, 1]]
    assert bounded_validation_source.validation_expected_jet_ids == jet_ids[:2]
    assert bounded_validation_source.last_validation_range.n_jets == 2
