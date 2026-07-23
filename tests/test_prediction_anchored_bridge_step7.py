from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from tests.test_prediction_anchored_bridge_step6 import _fixture, _tensor_inputs
from teacher_logit_reco.local_particle_residual_field import (
    ARCH_A0M_CAPACITY_PARTICLE,
    ARCH_A2_REGIONS_NO_GLOBAL,
    ARCH_A3_HLG_PRIMARY,
    ARCH_A4_HLG_REFINE,
    ARCH_A5_HLG_ABSOLUTE,
    ARCH_A5S_HLG_SCRATCH,
    ARCH_A6_HLG_NO_PAIR,
    ARCH_A7_HLG_NO_H0,
    ARCH_A7F_HLG_NO_F0,
    ARCH_A7X_HLG_NO_RAW,
    ARCH_A8_HLG_FUSED_HEAD,
    ARCH_A9_HLG_GROUP_GATE,
    ARCH_AFIX_HLG_FIXED_ASSIGNMENT,
    ARCH_AGLOBAL_HLG_ONE_GLOBAL,
    ARCH_AL_HLG_REGIONS_8_8_4,
    ARCH_ASAME_HLG_SAME_SCALE,
    ARCH_AS_HLG_REGIONS_2_2_1,
    ASSIGNMENT_FIXED,
    DIRECT_HLT,
    DIRECT_R0REP,
    GATE_INITIAL_BIAS,
    GATE_LOSS_COEFFICIENT,
    HLGCorrectionConfig,
    REGION_POOL_CANONICAL_INPUT_DIM,
    STEP7_DIRECT_CONTROL_IDS,
    STEP7_HIERARCHY_ARCHITECTURE_IDS,
    STEP7_MEASURED_ARCHITECTURE_IDS,
    DeployedBundleResourceReference,
    DirectHLGClassifier,
    DirectHLGConfig,
    DirectHLGTrainConfig,
    build_campaign_registry,
    build_capacity_matched_direct_hlg,
    build_step6_correction_model,
    build_step7_hlg_correction_model,
    fit_absolute_output_scaler,
    measure_correction_resources,
    measure_step7_registry_states,
    measure_step7_resources,
    particle_capacity_match,
    run_step7_paired_seed_miniature,
    step7_gate_regularization,
    train_step7_direct_hlg,
    tiny_train_reload_step7_direct,
    tiny_train_reload_step7_hierarchy,
)


def _absolute(batch, scaler, source_hash="d" * 64):
    return fit_absolute_output_scaler(
        [(batch["bridge_fields"], batch["mask"])],
        source_manifest_sha256=source_hash,
        bridge_recipe_sha256="b" * 64,
        epsilon=scaler["epsilon"],
    )


def _hierarchy_inputs(batch, architecture_id):
    tokens, mask, f0, h0 = _tensor_inputs(batch)
    if architecture_id == ARCH_A5S_HLG_SCRATCH:
        return tokens, mask, f0[..., 45:], None
    return tokens, mask, f0, h0


def _synthetic_direct_reference(width=4):
    return DeployedBundleResourceReference(
        particle_width=width,
        valid_particles=width,
        r0_parameters=500_000,
        r0_forward_flops=10_000_000,
        a3_parameters=1_000_000,
        a3_forward_flops=15_000_000,
        t10_parameters=500_000,
        t10_forward_flops=15_000_000,
        r0_checkpoint_sha256="a" * 64,
        a3_config_sha256="c" * 64,
        t10_checkpoint_sha256="e" * 64,
        physical45_scaler_sha256="f" * 64,
        r0_registration_sha256="1" * 64,
        execution_spec_sha256="2" * 64,
        child_manifest_sha256="3" * 64,
        selected_consumer_sha256="4" * 64,
        physical45_recipe_sha256="5" * 64,
        source_manifest_sha256="d" * 64,
    )


def _verified_reference(scaler, width=4):
    model = build_step7_hlg_correction_model(
        ARCH_A3_HLG_PRIMARY, scaler_artifact=scaler, dropout=0.05
    )
    profile = measure_step7_resources(model, particle_width=width)
    remaining_parameters = 2_000_000 - profile.total_parameters
    assert remaining_parameters > 1
    return DeployedBundleResourceReference(
        particle_width=width,
        valid_particles=width,
        r0_parameters=remaining_parameters // 2,
        r0_forward_flops=1_750_000,
        a3_parameters=profile.total_parameters,
        a3_forward_flops=profile.forward_flops,
        t10_parameters=remaining_parameters - remaining_parameters // 2,
        t10_forward_flops=1_750_000,
        r0_checkpoint_sha256="a" * 64,
        a3_config_sha256=model.config.to_artifact()["content_hash"],
        t10_checkpoint_sha256="e" * 64,
        physical45_scaler_sha256=scaler["content_hash"],
        r0_registration_sha256="1" * 64,
        execution_spec_sha256="2" * 64,
        child_manifest_sha256="3" * 64,
        selected_consumer_sha256="4" * 64,
        physical45_recipe_sha256="5" * 64,
        source_manifest_sha256="d" * 64,
    )


def test_step7_variant_matrix_is_complete_locked_and_hashed():
    assert len(STEP7_HIERARCHY_ARCHITECTURE_IDS) == 16
    assert len(STEP7_MEASURED_ARCHITECTURE_IDS) == 18
    configs = {
        architecture_id: HLGCorrectionConfig.for_architecture(architecture_id)
        for architecture_id in STEP7_HIERARCHY_ARCHITECTURE_IDS
    }
    assert configs[ARCH_A2_REGIONS_NO_GLOBAL].use_global_transformer is False
    assert configs[ARCH_A4_HLG_REFINE].refinement_passes == 1
    assert configs[ARCH_A5_HLG_ABSOLUTE].output_mode == "bounded_absolute_physical45"
    assert configs[ARCH_A5S_HLG_SCRATCH].use_f0_conditioning is False
    assert configs[ARCH_A5S_HLG_SCRATCH].use_h0_conditioning is False
    assert configs[ARCH_A6_HLG_NO_PAIR].use_pair_bias is False
    assert configs[ARCH_A7_HLG_NO_H0].use_h0_conditioning is False
    assert configs[ARCH_A7F_HLG_NO_F0].use_f0_conditioning is False
    assert configs[ARCH_A7X_HLG_NO_RAW].use_raw_conditioning is False
    assert configs[ARCH_A8_HLG_FUSED_HEAD].fused_radius_head is True
    assert configs[ARCH_A9_HLG_GROUP_GATE].group_gate is True
    assert configs[ARCH_AS_HLG_REGIONS_2_2_1].region_counts == (2, 2, 1)
    assert configs[ARCH_AL_HLG_REGIONS_8_8_4].region_counts == (8, 8, 4)
    assert configs[ARCH_AFIX_HLG_FIXED_ASSIGNMENT].assignment_mode == ASSIGNMENT_FIXED
    assert configs[ARCH_ASAME_HLG_SAME_SCALE].same_scale_only is True
    assert configs[ARCH_AGLOBAL_HLG_ONE_GLOBAL].one_global_token is True
    canonical = configs[ARCH_A3_HLG_PRIMARY].to_artifact()
    assert canonical["region_pool_input_dim"] == REGION_POOL_CANONICAL_INPUT_DIM
    assert canonical["hlt_only_region_provenance"] is True
    assert canonical["oracle_or_bridge_input_present"] is False
    assert set(canonical["upstream_contract_versions"]) == {"seed", "assignment", "transformer"}
    with pytest.raises(ValueError, match="changed locked fields"):
        HLGCorrectionConfig(architecture_id=ARCH_A3_HLG_PRIMARY, region_counts=(2, 2, 1))


def test_canonical_389_pool_matches_hand_assignment_weighted_equation():
    batch, scaler = _fixture(n=2, p=5)
    model = build_step7_hlg_correction_model(
        ARCH_A3_HLG_PRIMARY, scaler_artifact=scaler, dropout=0.0
    ).eval()
    tokens, mask, f0, h0 = _tensor_inputs(batch)
    standardized = model.scalers.conditioning_standardize(f0, mask, input_space="physical_field")
    with torch.no_grad():
        encoding = model.encoder(tokens, mask, standardized, h0)
    region = encoding.region_output
    assert region.pool_inputs.shape[-1] == 389
    weights = region.assignment_output.cluster_weights
    mass = weights.sum(-1)
    denominator = mass.clamp_min(1e-12).unsqueeze(-1)
    scale = region.assignment_output.scale_index
    local = torch.stack(encoding.radius_streams, dim=1).index_select(1, scale)
    local_pool = (weights.unsqueeze(-1) * local).sum(2) / denominator
    f0_pool = torch.einsum("brp,bpd->brd", weights, standardized) / denominator
    h0_pool = torch.einsum("brp,bpd->brd", weights, h0) / denominator
    raw_pool = torch.einsum("brp,bpd->brd", weights, tokens) / denominator
    seed_pt = region.assignment_output.seed_output.seed_pt_fraction.unsqueeze(-1)
    one_hot = torch.nn.functional.one_hot(scale, 3).float()[None].expand(tokens.shape[0], -1, -1)
    expected = torch.cat(
        (local_pool, f0_pool, h0_pool, raw_pool, torch.log1p(mass).unsqueeze(-1), seed_pt, one_hot),
        dim=-1,
    ).masked_fill(~region.region_mask.unsqueeze(-1), 0.0)
    assert torch.allclose(region.pool_inputs, expected, atol=1e-6)
    assert region.diagnostics["assignment_mass_semantics"] == "scale_wise_cluster_membership_sum"
    assert region.diagnostics["padded_particle_mass_exact_zero"] is True


def test_empty_regions_and_empty_jet_rows_are_finite_zero_and_mask_safe():
    batch, scaler = _fixture(n=2, p=4)
    for key in ("hlt_tokens", "f0", "h0", "bridge_fields"):
        batch[key][0] = 0
    batch["mask"][0] = False
    model = build_step7_hlg_correction_model(
        ARCH_A3_HLG_PRIMARY, scaler_artifact=scaler, dropout=0.0
    ).eval()
    with torch.no_grad():
        output = model(*_tensor_inputs(batch), need_attention_weights=True)
    assert torch.isfinite(output.f_hat).all()
    assert torch.count_nonzero(output.f_hat[0]).item() == 0
    assert not output.reasoning_state.region_mask[0].any()
    assert torch.count_nonzero(output.reasoning_state.region_tokens[0]).item() == 0
    assert output.diagnostics["global"]["empty_batch_rows_safely_masked"] == 1
    assert output.diagnostics["readback"]["empty_region_rows_safely_masked"] == 1


def test_pair_bias_transformer_and_readback_have_exact_shapes_and_equation():
    batch, scaler = _fixture(n=2, p=5)
    model = build_step7_hlg_correction_model(
        ARCH_A3_HLG_PRIMARY, scaler_artifact=scaler, dropout=0.0
    ).eval()
    tokens, mask, f0, h0 = _tensor_inputs(batch)
    standardized = model.scalers.conditioning_standardize(f0, mask, input_space="physical_field")
    with torch.no_grad():
        encoding = model.encoder(tokens, mask, standardized, h0, need_attention_weights=True)
    global_output = encoding.global_output
    assert global_output.pair_features.shape == (2, 10, 10, 19)
    assert global_output.pair_bias.shape == (2, 4, 10, 10)
    assert global_output.attention_weights.shape == (2, 2, 4, 10, 10)
    readback = model.encoder.readback
    with torch.no_grad():
        safe_mask = encoding.region_mask.clone()
        context, _ = readback.attention(
            readback.query_norm(encoding.base_hidden),
            encoding.region_tokens,
            encoding.region_tokens,
            key_padding_mask=~safe_mask,
            need_weights=False,
        )
        expected = readback.output_norm(
            encoding.base_hidden + readback.output_projection(context)
        ).masked_fill(~mask.unsqueeze(-1), 0.0)
    assert torch.allclose(encoding.readback, expected, atol=1e-6)


def test_a2_a3_a4_region_counts_one_global_and_exact_single_refinement():
    batch, scaler = _fixture(n=2, p=5)
    expected_regions = {
        ARCH_A2_REGIONS_NO_GLOBAL: 10,
        ARCH_A3_HLG_PRIMARY: 10,
        ARCH_A4_HLG_REFINE: 10,
        ARCH_AS_HLG_REGIONS_2_2_1: 5,
        ARCH_AL_HLG_REGIONS_8_8_4: 20,
        ARCH_AGLOBAL_HLG_ONE_GLOBAL: 1,
    }
    for architecture_id, count in expected_regions.items():
        model = build_step7_hlg_correction_model(
            architecture_id, scaler_artifact=scaler, dropout=0.0
        )
        output = model(*_hierarchy_inputs(batch, architecture_id))
        assert output.reasoning_state.region_tokens.shape == (2, count, 160)
        assert output.diagnostics["refinement_pass_count"] == (1 if architecture_id == ARCH_A4_HLG_REFINE else 0)
        assert output.diagnostics["seeds_recomputed_during_refinement"] is False
        assert output.diagnostics["assignments_recomputed_during_refinement"] is False
        if architecture_id == ARCH_A2_REGIONS_NO_GLOBAL:
            assert model.encoder.region_transformer is None
            assert output.diagnostics["global"] is None
        if architecture_id == ARCH_A4_HLG_REFINE:
            assert model.encoder.refinement_transformer is not None
            assert model.encoder.refinement_readback is not None


def test_zero_initialization_input_removals_and_a5s_has_no_privileged_trainable_path():
    batch, scaler = _fixture(n=2, p=5)
    absolute = _absolute(batch, scaler)
    f0 = torch.as_tensor(batch["f0"])
    for architecture_id in STEP7_HIERARCHY_ARCHITECTURE_IDS:
        model = build_step7_hlg_correction_model(
            architecture_id,
            scaler_artifact=scaler,
            absolute_scaler_artifact=absolute if architecture_id in {ARCH_A5_HLG_ABSOLUTE, ARCH_A5S_HLG_SCRATCH} else None,
            dropout=0.0,
        )
        output = model(*_hierarchy_inputs(batch, architecture_id))
        assert torch.equal(output.f_hat[..., 45:], f0[..., 45:])
        if architecture_id not in {ARCH_A5_HLG_ABSOLUTE, ARCH_A5S_HLG_SCRATCH}:
            assert torch.equal(output.f_hat, f0)
    assert build_step7_hlg_correction_model(ARCH_A7_HLG_NO_H0, scaler_artifact=scaler).encoder.base_fusion.h0_projection is None
    assert build_step7_hlg_correction_model(ARCH_A7F_HLG_NO_F0, scaler_artifact=scaler).encoder.base_fusion.f0_projection is None
    assert build_step7_hlg_correction_model(ARCH_A7X_HLG_NO_RAW, scaler_artifact=scaler).encoder.base_fusion.raw_projection is None
    scratch = build_step7_hlg_correction_model(
        ARCH_A5S_HLG_SCRATCH, scaler_artifact=scaler, absolute_scaler_artifact=absolute, dropout=0.0
    ).eval()
    tokens, mask, anchor, h0 = _tensor_inputs(batch)
    with torch.no_grad():
        first = scratch(tokens, mask, anchor[..., 45:], None)
        second = scratch(tokens, mask, anchor[..., 45:].clone(), None)
    with pytest.raises(ValueError, match="forbids h0"):
        scratch(tokens, mask, anchor, h0)
    assert torch.equal(first.f_hat, second.f_hat)
    assert first.diagnostics["a5s_f0_physical_or_h0_entered_trainable_path"] is False


def test_absolute_output_initializes_to_center_and_keeps_reliability_passthrough():
    batch, scaler = _fixture(n=2, p=5)
    absolute = _absolute(batch, scaler)
    center = torch.as_tensor(absolute["center"])
    mask = torch.as_tensor(batch["mask"])
    for architecture_id in (ARCH_A5_HLG_ABSOLUTE, ARCH_A5S_HLG_SCRATCH):
        model = build_step7_hlg_correction_model(
            architecture_id, scaler_artifact=scaler, absolute_scaler_artifact=absolute, dropout=0.0
        )
        output = model(*_hierarchy_inputs(batch, architecture_id))
        assert torch.allclose(output.f_hat[..., :45][mask], center.expand(mask.sum(), -1))
        assert torch.equal(output.f_hat[..., 45:], torch.as_tensor(batch["f0"])[..., 45:])
        assert output.saturation_mask is None
        assert output.diagnostics["trust_bound_enabled"] is False


def test_no_pair_fixed_same_scale_fused_head_and_gate_contracts():
    batch, scaler = _fixture(n=2, p=5)
    no_pair = build_step7_hlg_correction_model(ARCH_A6_HLG_NO_PAIR, scaler_artifact=scaler, dropout=0.0)
    assert no_pair(*_tensor_inputs(batch)).diagnostics["global"]["pair_bias_present"] is False
    fixed = build_step7_hlg_correction_model(ARCH_AFIX_HLG_FIXED_ASSIGNMENT, scaler_artifact=scaler, dropout=0.0)
    assert fixed.encoder.assignment is None
    fixed_output = fixed(*_tensor_inputs(batch))
    cluster = fixed_output.reasoning_state.diagnostics["region_pool"]
    assert cluster["hlt_only_provenance"] is True
    same = build_step7_hlg_correction_model(ARCH_ASAME_HLG_SAME_SCALE, scaler_artifact=scaler, dropout=0.0)
    tokens, mask, f0, h0 = _tensor_inputs(batch)
    standardized = same.scalers.conditioning_standardize(f0, mask, input_space="physical_field")
    encoding = same.encoder(tokens, mask, standardized, h0)
    scale = encoding.assignment_output.scale_index
    cross = scale[:, None] != scale[None, :]
    assert torch.all(encoding.global_output.pair_bias[:, :, cross] == -1.0e4)
    fused = build_step7_hlg_correction_model(ARCH_A8_HLG_FUSED_HEAD, scaler_artifact=scaler)
    assert len(fused.radius_heads) == 1
    assert fused.radius_heads[0].output.out_features == 45
    gated = build_step7_hlg_correction_model(ARCH_A9_HLG_GROUP_GATE, scaler_artifact=scaler, dropout=0.0)
    assert all(not parameter.requires_grad for parameter in gated.gate_heads.parameters())
    for head in gated.radius_heads:
        torch.nn.init.constant_(head.output.bias, 0.1)
    gated_output = gated(*_tensor_inputs(batch))
    valid_gates = gated_output.gate_values[mask]
    assert gated_output.gate_values.shape == (2, 5, 3, 4)
    assert torch.allclose(valid_gates, torch.full_like(valid_gates, 0.95), atol=1e-7)
    assert torch.allclose(gated_output.physical_correction, 0.95 * gated_output.pre_gate_physical_correction, atol=1e-6)
    assert gated_output.gate_loss.item() == pytest.approx(0.0025, rel=1e-5)
    _, warmup_coefficient = step7_gate_regularization(gated_output, phase="field_warmup")
    _, phase2_coefficient = step7_gate_regularization(gated_output, phase="distillation")
    assert warmup_coefficient == 0.0
    assert phase2_coefficient == GATE_LOSS_COEFFICIENT
    assert all(value == pytest.approx(GATE_INITIAL_BIAS) for value in gated.gate_heads[0].bias.tolist())
    gated.set_training_phase("distillation")
    assert all(parameter.requires_grad for parameter in gated.gate_heads.parameters())


def test_every_hierarchy_variant_has_finite_gradients():
    batch, scaler = _fixture(n=2, p=4)
    absolute = _absolute(batch, scaler)
    target = torch.as_tensor(batch["bridge_fields"])
    for architecture_id in STEP7_HIERARCHY_ARCHITECTURE_IDS:
        model = build_step7_hlg_correction_model(
            architecture_id,
            scaler_artifact=scaler,
            absolute_scaler_artifact=absolute if architecture_id in {ARCH_A5_HLG_ABSOLUTE, ARCH_A5S_HLG_SCRATCH} else None,
            dropout=0.0,
        )
        output = model(*_hierarchy_inputs(batch, architecture_id))
        loss = torch.nn.functional.mse_loss(output.f_hat[output.mask], target[output.mask])
        loss.backward()
        gradients = [value.grad for value in model.parameters() if value.requires_grad and value.grad is not None]
        assert gradients
        assert all(torch.isfinite(value).all() for value in gradients)


def test_executed_a3_matches_particle_capacity_control_and_direct_controls_match_bundle():
    batch, scaler = _fixture(n=2, p=4)
    a3 = measure_step7_resources(
        build_step7_hlg_correction_model(ARCH_A3_HLG_PRIMARY, scaler_artifact=scaler),
        particle_width=32,
    )
    a0m = measure_correction_resources(
        build_step6_correction_model(ARCH_A0M_CAPACITY_PARTICLE, scaler_artifact=scaler),
        particle_width=32,
    )
    from teacher_logit_reco.local_particle_residual_field import CorrectionResourceProfile
    reference = CorrectionResourceProfile(
        a3.architecture_id, a3.trainable_parameters, a3.total_parameters, a3.forward_flops,
        a3.batch_size, a3.particle_width, a3.valid_particles, a3.method,
    )
    assert particle_capacity_match(a0m, reference)["passed"] is True
    direct_reference = _synthetic_direct_reference()
    for run_id in STEP7_DIRECT_CONTROL_IDS:
        direct, profile, match = build_capacity_matched_direct_hlg(
            run_id,
            scaler_artifact=scaler if run_id == DIRECT_R0REP else None,
            reference=direct_reference,
            dropout=0.0,
        )
        assert isinstance(direct, DirectHLGClassifier)
        assert profile.total_parameters == direct_reference.total_parameters
        assert match["passed"] is True
        assert not any("field_head" in name or "t10" in name.lower() for name, _ in direct.named_modules())


def test_direct_controls_forbid_wrong_inputs_and_train_reload_exactly():
    batch, scaler = _fixture(n=2, p=4)
    batch["labels"] = np.asarray([0, 1], dtype=np.int64)
    reference = _synthetic_direct_reference()
    raw = DirectHLGClassifier(DirectHLGConfig(DIRECT_HLT, 4, dropout=0.0))
    tokens, mask, f0, h0 = _tensor_inputs(batch)
    with pytest.raises(ValueError, match="forbids f0/h0"):
        raw(tokens, mask, f0, h0)
    rep = DirectHLGClassifier(DirectHLGConfig(DIRECT_R0REP, 4, dropout=0.0), scaler_artifact=scaler)
    with pytest.raises(ValueError, match="requires f0 and h0"):
        rep(tokens, mask)
    for run_id in STEP7_DIRECT_CONTROL_IDS:
        result = tiny_train_reload_step7_direct(
            run_id,
            scaler_artifact=scaler,
            deployed_reference=reference,
            batch=deepcopy(batch),
        )
        assert result["strict_reload"] is True
        assert result["reload_exact_output"] is True
        assert result["capacity_match"]["passed"] is True
        assert result["objective"] == "cross_entropy_only"


def test_full_union_direct_training_contract_is_ce_only_and_manifest_bound():
    batch, scaler = _fixture(n=2, p=4)
    batch["labels"] = np.asarray([0, 1], dtype=np.int64)
    reference = _synthetic_direct_reference()
    for run_id in STEP7_DIRECT_CONTROL_IDS:
        model, _, _ = build_capacity_matched_direct_hlg(
            run_id,
            scaler_artifact=scaler if run_id == DIRECT_R0REP else None,
            reference=reference,
            dropout=0.0,
        )
        train_batch = {
            key: value for key, value in batch.items()
            if key in ({"hlt_tokens", "mask", "labels", "f0", "h0"} if run_id == DIRECT_R0REP else {"hlt_tokens", "mask", "labels"})
        }
        config = DirectHLGTrainConfig(
            run_id,
            stack_train_consumer_manifest_sha256="1" * 64,
            stack_train_distill_manifest_sha256="2" * 64,
            union_manifest_sha256="3" * 64,
            optimizer_steps=1,
        )
        result = train_step7_direct_hlg(model, [train_batch], config)
        assert result["optimizer_steps_completed"] == 1
        assert result["objective"] == "cross_entropy_only"
        assert result["field_output_present"] is False
        assert result["kd_present"] is False
        assert result["persistent_batch_or_field_tensor_written"] is False
    with pytest.raises(ValueError, match="500k"):
        DirectHLGTrainConfig(
            DIRECT_HLT, "1" * 64, "2" * 64, "3" * 64, optimizer_steps=1,
            unique_training_jets=499_999,
        )


def test_hierarchy_tiny_train_reload_representative_paths():
    batch, scaler = _fixture(n=2, p=4)
    absolute = _absolute(batch, scaler)
    for architecture_id in (
        ARCH_A3_HLG_PRIMARY,
        ARCH_A4_HLG_REFINE,
        ARCH_A5_HLG_ABSOLUTE,
        ARCH_AFIX_HLG_FIXED_ASSIGNMENT,
        ARCH_A9_HLG_GROUP_GATE,
    ):
        result = tiny_train_reload_step7_hierarchy(
            architecture_id,
            scaler_artifact=scaler,
            absolute_scaler_artifact=absolute,
            batch=deepcopy(batch),
        )
        assert result["strict_reload"] is True
        assert result["reload_exact_output"] is True
        assert result["all_gradients_finite"] is True


def test_step7_registry_measurement_marks_18_rows_and_leaves_step8_open():
    batch, scaler = _fixture(n=2, p=4)
    absolute = _absolute(batch, scaler)
    a3 = measure_step7_resources(
        build_step7_hlg_correction_model(ARCH_A3_HLG_PRIMARY, scaler_artifact=scaler),
        particle_width=32,
    )
    reference = _verified_reference(scaler, width=32)
    updated, artifact = measure_step7_registry_states(
        build_campaign_registry(),
        scaler_artifact=scaler,
        absolute_scaler_artifact=absolute,
        source_manifest_sha256="d" * 64,
        deployed_reference=reference,
    )
    rows = {row["canonical_run_id"]: row for row in updated["runs"]}
    assert artifact["newly_measured_configuration_count"] == 18
    assert all(rows[run_id]["measurement_status"] == "MEASURED" for run_id in STEP7_MEASURED_ARCHITECTURE_IDS)
    assert all(rows[run_id]["measurement_status"] == "UNMEASURED" for run_id in artifact["step8_deferred_unmeasured_run_ids"])
    assert artifact["particle_capacity_match_to_executed_a3"]["passed"] is True
    assert all(value["passed"] for value in artifact["direct_capacity_matches"].values())


def test_paired_seed_miniature_covers_every_hierarchy_variant():
    batch, scaler = _fixture(n=2, p=4)
    artifact = run_step7_paired_seed_miniature(
        scaler_artifact=scaler,
        absolute_scaler_artifact=_absolute(batch, scaler),
        batch=batch,
    )
    assert artifact["complete_hierarchy_matrix"] is True
    assert len(artifact["rows"]) == 3 * len(STEP7_HIERARCHY_ARCHITECTURE_IDS)
    assert set(artifact["aggregates"]) == set(STEP7_HIERARCHY_ARCHITECTURE_IDS)
    assert all(value["paired_seed_ids"] == [101, 202, 303] for value in artifact["aggregates"].values())
    assert artifact["scientific_results_allowed"] is False


def test_step7_operator_plan_cli_is_dry_run_safe(tmp_path):
    batch, scaler = _fixture(n=2, p=4)
    absolute = _absolute(batch, scaler)
    reference = _verified_reference(scaler).to_artifact()
    scaler_path = tmp_path / "scaler.json"
    absolute_path = tmp_path / "absolute.json"
    reference_path = tmp_path / "reference.json"
    scaler_path.write_text(json.dumps(scaler), encoding="utf-8")
    absolute_path.write_text(json.dumps(absolute), encoding="utf-8")
    reference_path.write_text(json.dumps(reference), encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts" / "measure_prediction_anchored_bridge_step7.py"
    completed = subprocess.run(
        [
            sys.executable, str(script), "--mode", "plan", "--scaler", str(scaler_path),
            "--absolute-scaler", str(absolute_path), "--deployed-resource-reference", str(reference_path),
            "--particle-width", "4", "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["dry_run"] is True
    assert payload["plan"]["hierarchy_architecture_ids"] == list(STEP7_HIERARCHY_ARCHITECTURE_IDS)
    assert payload["plan"]["direct_control_ids"] == [DIRECT_HLT, DIRECT_R0REP]
    assert payload["plan"]["all_direct_capacity_tolerances_passed"] is True
    assert payload["plan"]["oracle_inference_input_present"] is False
