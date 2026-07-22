from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from teacher_logit_reco.local_particle_residual_field import (
    ARCH_A0M_CAPACITY_PARTICLE,
    ARCH_A1H_HARD_RADIUS,
    ARCH_A1_MULTISCALE_LOCAL,
    KERNEL_GAUSSIAN,
    KERNEL_HARD_RADIUS,
    LOCAL_EDGE_FEATURE_NAMES,
    LOCAL_RADII,
    STEP6_ARCHITECTURE_IDS,
    STEP7_DEFERRED_ARCHITECTURE_IDS,
    LocalCorrectionConfig,
    SharedLocalMessageLayer,
    TargetKernelLocalProcessor,
    build_campaign_registry,
    build_directed_neighbor_graph,
    build_local_graph_features,
    build_step6_correction_model,
    canonical_a3_resource_reference,
    fit_bridge_scalers,
    measure_correction_resources,
    measure_step6_registry_states,
    particle_capacity_match,
    tiny_train_reload_step6_model,
)


def _fixture(n=4, p=6):
    rng = np.random.default_rng(606)
    mask = np.ones((n, p), dtype=bool)
    if p > 2:
        mask[::2, -1] = False
    tokens = rng.normal(scale=0.1, size=(n, p, 14)).astype(np.float32)
    tokens[..., 0] = rng.uniform(0.5, 3.0, size=(n, p))
    tokens[..., 1] = rng.uniform(-0.12, 0.12, size=(n, p))
    tokens[..., 2] = rng.uniform(-0.12, 0.12, size=(n, p))
    tokens[..., 3] = rng.uniform(1.0, 4.0, size=(n, p))
    f0 = rng.normal(scale=0.3, size=(n, p, 50)).astype(np.float32)
    true = f0 + rng.normal(scale=0.2, size=f0.shape).astype(np.float32)
    h0 = rng.normal(scale=0.2, size=(n, p, 160)).astype(np.float32)
    tokens[~mask] = 0
    f0[~mask] = 0
    true[~mask] = 0
    h0[~mask] = 0
    bridge = f0.copy()
    bridge[..., :45] += np.float32(0.1) * (true - f0)[..., :45]
    bridge[~mask] = 0
    scaler = fit_bridge_scalers(
        [(f0, true, mask)],
        parent_hashes={"source": "a" * 64, "r0": "b" * 64},
        channel_policy="physical45",
    ).to_artifact()
    return {
        "hlt_tokens": tokens,
        "mask": mask,
        "f0": f0,
        "h0": h0,
        "bridge_fields": bridge,
    }, scaler


def _tensor_inputs(batch):
    return tuple(
        torch.as_tensor(batch[name]) for name in ("hlt_tokens", "mask", "f0", "h0")
    )


def test_locked_step6_configs_and_future_global_boundary():
    gaussian = LocalCorrectionConfig.for_architecture(ARCH_A1_MULTISCALE_LOCAL)
    hard = LocalCorrectionConfig.for_architecture(ARCH_A1H_HARD_RADIUS)
    capacity = LocalCorrectionConfig.for_architecture(ARCH_A0M_CAPACITY_PARTICLE)
    assert gaussian.kernel_mode == KERNEL_GAUSSIAN
    assert hard.kernel_mode == KERNEL_HARD_RADIUS
    assert gaussian.local_layers == hard.local_layers == 2
    assert capacity.local_layers == 6
    assert capacity.capacity_particle_blocks == 2
    assert capacity.capacity_hidden_dim == 288
    for config in (gaussian, hard, capacity):
        artifact = config.to_artifact()
        assert artifact["message_mlp"] == [326, 160, 160]
        assert artifact["update_mlp"] == [320, 160, 160]
        assert artifact["radius_head"] == [480, 160, 128, 64, 15]
        assert artifact["global_or_region_module_present"] is False
    with pytest.raises(ValueError):
        LocalCorrectionConfig(
            architecture_id=ARCH_A1_MULTISCALE_LOCAL,
            kernel_mode=KERNEL_GAUSSIAN,
            graph_cap=31,
        )


def test_gaussian_and_hard_weights_match_analytical_formula_at_all_radii():
    tokens = torch.zeros((1, 2, 14))
    tokens[..., 0] = torch.tensor([[1.0, 2.0]])
    tokens[..., 3] = torch.tensor([[2.0, 4.0]])
    tokens[0, 1, 1] = 0.03
    mask = torch.ones((1, 2), dtype=torch.bool)
    graph = build_directed_neighbor_graph(tokens, mask, cap=32, support=0.30)
    gaussian = build_local_graph_features(
        tokens, mask, kernel_mode=KERNEL_GAUSSIAN, graph=graph
    )
    hard = build_local_graph_features(
        tokens, mask, kernel_mode=KERNEL_HARD_RADIUS, graph=graph
    )
    expected = torch.tensor(
        [np.exp(-0.5 * 0.03**2 / radius**2) for radius in LOCAL_RADII],
        dtype=torch.float32,
    )
    assert torch.allclose(gaussian.kernel_weights[0, 0, 0], expected, atol=1e-6)
    assert torch.equal(hard.kernel_weights[0, 0, 0], torch.tensor([0.0, 1.0, 1.0]))
    assert torch.all(gaussian.kernel_weights[graph.edge_valid] > 0)
    assert tuple(LOCAL_EDGE_FEATURE_NAMES) == (
        "source_minus_target_delta_eta",
        "sin_source_minus_target_delta_phi",
        "cos_source_minus_target_delta_phi",
        "delta_r",
        "clipped_log_source_target_pt_ratio",
        "clipped_log_source_target_energy_ratio",
    )


def test_capped_graph_has_deterministic_source_index_ties_and_no_self_edges():
    particles = 35
    tokens = torch.zeros((1, particles, 14))
    tokens[..., 0] = 1.0
    tokens[..., 3] = 2.0
    tokens[0, 1:, 1] = 0.10
    mask = torch.ones((1, particles), dtype=torch.bool)
    graph = build_directed_neighbor_graph(tokens, mask, cap=32, support=0.30)
    selected = graph.neighbor_indices[0, 0][graph.edge_valid[0, 0]]
    assert torch.equal(selected, torch.arange(1, 33))
    target_indices = torch.arange(particles)[:, None].expand_as(graph.neighbor_indices[0])
    assert not torch.any(
        (graph.neighbor_indices[0] == target_indices) & graph.edge_valid[0]
    )
    assert graph.edge_valid[0, 0].sum().item() == 32
    assert bool(((graph.neighbor_indices[0, 0] == 1) & graph.edge_valid[0, 0]).any())
    assert not bool(((graph.neighbor_indices[0, 1] == 0) & graph.edge_valid[0, 1]).any())


def test_edge_features_use_locked_source_minus_target_convention_and_clips():
    tokens = torch.zeros((1, 2, 14))
    tokens[0, :, 0] = torch.tensor([1.0e-6, 1.0e6])
    tokens[0, :, 3] = torch.tensor([1.0e-6, 1.0e6])
    tokens[0, 0, 1:3] = torch.tensor([0.0, 0.0])
    tokens[0, 1, 1:3] = torch.tensor([0.04, 0.05])
    mask = torch.ones((1, 2), dtype=torch.bool)
    values = build_local_graph_features(tokens, mask, kernel_mode=KERNEL_GAUSSIAN)
    edge = values.edge_features[0, 0, 0]
    assert edge[0].item() == pytest.approx(0.04)
    assert edge[1].item() == pytest.approx(np.sin(0.05))
    assert edge[2].item() == pytest.approx(np.cos(0.05))
    assert edge[3].item() == pytest.approx(np.hypot(0.04, 0.05))
    assert edge[4].item() == pytest.approx(8.0)
    assert edge[5].item() == pytest.approx(8.0)


def test_weighted_mean_and_message_update_match_direct_hand_computation():
    torch.manual_seed(77)
    config = LocalCorrectionConfig.for_architecture(
        ARCH_A1_MULTISCALE_LOCAL, dropout=0.0
    )
    layer = SharedLocalMessageLayer(config).eval()
    hidden = torch.randn(1, 3, 160)
    indices = torch.tensor([[[1, 2], [0, 2], [0, 1]]])
    features = torch.randn(1, 3, 2, 6)
    weights = torch.tensor([[[0.25, 0.75], [1.0, 0.0], [0.4, 0.6]]])
    mask = torch.ones((1, 3), dtype=torch.bool)
    aggregate, normalized = layer.aggregate_messages(hidden, features, indices, weights)
    source = normalized[torch.arange(1)[:, None, None], indices]
    target = normalized[:, :, None, :].expand_as(source)
    messages = layer.message_mlp(torch.cat((target, source, features), dim=-1))
    expected_aggregate = (weights[..., None] * messages).sum(2) / (
        weights.sum(2, keepdim=True) + 1.0e-6
    )
    expected = hidden + layer.update_mlp(
        torch.cat((normalized, expected_aggregate), dim=-1)
    )
    actual = layer(
        hidden,
        edge_features=features,
        neighbor_indices=indices,
        weights=weights,
        mask=mask,
    )
    assert torch.allclose(aggregate, expected_aggregate, atol=1e-7, rtol=1e-6)
    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-6)


def test_local_layer_parameters_share_across_radii_but_not_depth_and_heads_route_15():
    batch, scaler = _fixture(n=2, p=4)
    model = build_step6_correction_model(
        ARCH_A1_MULTISCALE_LOCAL, scaler_artifact=scaler, dropout=0.0
    )
    assert isinstance(model.local_processor, TargetKernelLocalProcessor)
    assert len(model.local_processor.layers) == 2
    assert model.local_processor.layers[0] is not model.local_processor.layers[1]
    assert len(model.radius_heads) == 3
    for layer in model.local_processor.layers:
        assert layer.message_mlp[0].in_features == 326
        assert layer.message_mlp[0].out_features == 160
        assert layer.update_mlp[0].in_features == 320
        assert layer.update_mlp[-1].out_features == 160
    for head in model.radius_heads:
        linears = [module for module in head.modules() if isinstance(module, torch.nn.Linear)]
        assert [(item.in_features, item.out_features) for item in linears] == [
            (480, 160),
            (160, 128),
            (128, 64),
            (64, 15),
        ]
    output = model(*_tensor_inputs(batch))
    assert [tuple(value.shape) for value in output.reasoning_state.radius_streams] == [
        (2, 4, 160),
        (2, 4, 160),
        (2, 4, 160),
    ]


def test_zero_initialization_pass_through_spaces_and_shared_global_interface():
    batch, scaler = _fixture(n=3, p=5)
    for architecture_id in STEP6_ARCHITECTURE_IDS:
        model = build_step6_correction_model(
            architecture_id, scaler_artifact=scaler, dropout=0.0
        )
        output = model(*_tensor_inputs(batch))
        f0 = torch.as_tensor(batch["f0"])
        assert torch.equal(output.f_hat, f0)
        assert torch.count_nonzero(output.standardized_raw_correction).item() == 0
        assert torch.equal(output.f_hat[..., 45:], f0[..., 45:])
        assert output.reasoning_state.region_tokens is None
        assert output.reasoning_state.region_mask is None
        assert torch.equal(
            output.reasoning_state.readback, output.reasoning_state.base_hidden
        )
        assert output.diagnostics["learned_gate_present"] is False
        assert output.diagnostics["consumer_output_space"] == "physical_field"
        assert output.diagnostics["conditioning_space"] == "conditioning_standardized"
        assert output.diagnostics["correction_space"] == "correction_standardized"
        assert output.diagnostics["loss_space"] == "loss_standardized"
        with pytest.raises(ValueError, match="physical-field"):
            model(*_tensor_inputs(batch), f0_space="conditioning_standardized")


def test_a1_and_a1h_are_identical_except_for_kernel_policy():
    batch, scaler = _fixture(n=2, p=5)
    torch.manual_seed(909)
    gaussian = build_step6_correction_model(
        ARCH_A1_MULTISCALE_LOCAL, scaler_artifact=scaler, dropout=0.0
    )
    torch.manual_seed(909)
    hard = build_step6_correction_model(
        ARCH_A1H_HARD_RADIUS, scaler_artifact=scaler, dropout=0.0
    )
    assert gaussian.state_dict().keys() == hard.state_dict().keys()
    assert all(
        torch.equal(gaussian.state_dict()[name], hard.state_dict()[name])
        for name in gaussian.state_dict()
    )
    gaussian_output = gaussian(*_tensor_inputs(batch))
    hard_output = hard(*_tensor_inputs(batch))
    assert any(
        not torch.allclose(left, right)
        for left, right in zip(
            gaussian_output.reasoning_state.radius_streams,
            hard_output.reasoning_state.radius_streams,
        )
    )
    gaussian_config = gaussian.config.to_artifact()
    hard_config = hard.config.to_artifact()
    differing = {
        key
        for key in gaussian_config
        if gaussian_config.get(key) != hard_config.get(key)
    }
    assert differing == {"architecture_id", "kernel_mode", "content_hash"}


def test_particle_capacity_control_matches_locked_a3_parameters_and_flops():
    _, scaler = _fixture(n=2, p=4)
    control = measure_correction_resources(
        build_step6_correction_model(
            ARCH_A0M_CAPACITY_PARTICLE, scaler_artifact=scaler
        ),
        particle_width=128,
    )
    reference = canonical_a3_resource_reference(
        scaler_artifact=scaler,
        particle_width=128,
    )
    match = particle_capacity_match(control, reference)
    assert match["parameter_tolerance_passed"] is True
    assert match["flop_tolerance_passed"] is True
    assert match["passed"] is True
    assert match["parameter_relative_error"] <= 0.05
    assert match["flop_relative_error"] <= 0.10


def test_step6_registry_measurement_covers_c0_and_new_models_but_not_hlg():
    _, scaler = _fixture(n=2, p=4)
    registry = build_campaign_registry()
    updated, artifact = measure_step6_registry_states(
        registry,
        scaler_artifact=scaler,
        particle_width=128,
        source_manifest_sha256="c" * 64,
    )
    assert artifact["implemented_configuration_count"] == 14
    assert set(artifact["new_step6_state_bytes"]) == set(STEP6_ARCHITECTURE_IDS)
    rows = {row["canonical_run_id"]: row for row in updated["runs"]}
    assert all(rows[run_id]["measurement_status"] == "MEASURED" for run_id in STEP6_ARCHITECTURE_IDS)
    assert all(
        rows[run_id]["measurement_status"] == "UNMEASURED"
        for run_id in STEP7_DEFERRED_ARCHITECTURE_IDS
    )
    assert artifact["particle_capacity_match"]["passed"] is True


def test_tiny_batch_train_and_strict_reload_for_every_step6_architecture():
    batch, scaler = _fixture(n=3, p=4)
    for architecture_id in STEP6_ARCHITECTURE_IDS:
        result = tiny_train_reload_step6_model(
            architecture_id,
            scaler_artifact=scaler,
            batch=deepcopy(batch),
        )
        assert result["strict_reload"] is True
        assert result["reload_exact_f_hat"] is True
        assert result["gradient_tensor_count"] > 0
        assert result["loss_coefficients"] == {
            "kd": 0.0,
            "ce": 0.0,
            "bridge": 0.2,
            "true": 0.0,
            "anchor": 0.02,
            "smooth": 0.01,
            "gate": 0.0,
        }
        assert result["serialized_state_bytes"] > 0
        assert result["optimizer_state_persisted"] is False
        assert result["scientific_results_allowed"] is False


def test_step6_operator_plan_cli_is_dry_run_safe(tmp_path):
    _, scaler = _fixture(n=2, p=4)
    scaler_path = tmp_path / "scaler.json"
    scaler_path.write_text(json.dumps(scaler), encoding="utf-8")
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "measure_prediction_anchored_bridge_step6.py"
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--mode",
            "plan",
            "--scaler",
            str(scaler_path),
            "--particle-width",
            "128",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["dry_run"] is True
    assert payload["plan"]["architecture_ids"] == list(STEP6_ARCHITECTURE_IDS)
    assert payload["plan"]["particle_capacity_match"]["passed"] is True
    assert payload["plan"]["full_hlg_instantiated"] is False
