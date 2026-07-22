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
    C0_CANONICAL_RUN_IDS,
    C0CampaignConfig,
    C0CorrectionConfig,
    C0ReplicaResult,
    C0TrainConfig,
    FrozenLiveBridgeConsumer,
    PredictionAnchoredC0Correction,
    aggregate_c0_replicas,
    bridge_reachability_metrics,
    build_campaign_registry,
    build_c0_campaign_manifest,
    build_directed_neighbor_graph,
    c0_loss_recipes,
    capture_c0_ram_snapshot,
    compute_c0_objective,
    distillation_kl_loss,
    fit_bridge_scalers,
    local_smoothness_loss,
    masked_group_balanced_huber,
    measure_c0_registry_states,
    publish_c0_paired_replicas,
    resolve_c0_loss_recipe,
    restore_c0_ram_snapshot,
    run_c0_cpu_miniature,
    select_l0_checkpoint,
    select_postteacher_checkpoint,
    train_c0_replica,
    validate_c0_teacher_lineage,
    validate_model_val_stop_access,
)
from teacher_logit_reco.local_particle_residual_field.bridge_contracts import (
    with_content_hash,
)


def _fixture(n=12, p=5):
    rng = np.random.default_rng(51)
    mask = np.ones((n, p), dtype=bool)
    mask[::3, -1] = False
    tokens = rng.normal(size=(n, p, 14)).astype(np.float32)
    tokens[..., 0] = rng.uniform(0.5, 4.0, size=(n, p))
    tokens[..., 1] = rng.uniform(-0.3, 0.3, size=(n, p))
    tokens[..., 2] = rng.uniform(-0.4, 0.4, size=(n, p))
    tokens[..., 3] = tokens[..., 0] * np.cosh(tokens[..., 1])
    f0 = rng.normal(scale=0.4, size=(n, p, 50)).astype(np.float32)
    true = f0 + rng.normal(scale=0.25, size=f0.shape).astype(np.float32)
    h0 = rng.normal(scale=0.2, size=(n, p, 160)).astype(np.float32)
    tokens[~mask] = 0
    f0[~mask] = 0
    true[~mask] = 0
    h0[~mask] = 0
    bridge = f0.copy()
    bridge[..., :45] += np.float32(0.1) * (true - f0)[..., :45]
    bridge[~mask] = 0
    scalers = fit_bridge_scalers(
        [(f0, true, mask)],
        parent_hashes={"source": "a" * 64, "r0": "b" * 64},
        channel_policy="physical45",
    )
    labels = (np.arange(n) % 3).astype(np.int64)
    return {
        "hlt_tokens": tokens,
        "mask": mask,
        "f0": f0,
        "h0": h0,
        "true_fields": true,
        "bridge_fields": bridge,
        "labels": labels,
    }, scalers.to_artifact()


class _ToyT10(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(50, 3, bias=False)
        torch.manual_seed(98)
        torch.nn.init.normal_(self.projection.weight, std=0.15)

    def forward(self, fields, mask):
        weights = mask.to(fields.dtype).unsqueeze(-1)
        pooled = (fields * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)
        return self.projection(pooled)


def _live_consumer():
    return FrozenLiveBridgeConsumer(
        _ToyT10(),
        checkpoint_sha256="c" * 64,
        forward_adapter=lambda consumer, batch, fields: consumer(fields, batch["mask"]),
    )


def _add_target_logits(batch, live):
    copied = deepcopy(batch)
    bridge = torch.as_tensor(copied["bridge_fields"], dtype=torch.float32)
    mask = torch.as_tensor(copied["mask"], dtype=torch.bool)
    with torch.no_grad():
        copied["target_logits"] = live.consumer(bridge, mask).detach().numpy()
    return copied


def _access_receipt(split="model_val_stop", purpose="checkpoint_selection"):
    return with_content_hash(
        {
            "contract": "prediction_anchored_split_access_receipt_v1",
            "status": "AUTHORIZED",
            "split_name": split,
            "purpose": purpose,
            "parent_manifest_sha256": "d" * 64,
            "bound_split_sha256": "e" * 64,
            "seal_kind": None,
            "one_shot": False,
            "unlock_sha256": None,
            "selection_sha256": None,
        }
    )


def _teacher_artifacts(live, *, include_cache=True):
    selected = with_content_hash(
        {
            "contract": "selected_bridge_consumer_v2",
            "status": "CONFIRMED_LOCKED",
            "checkpoint_sha256": live.checkpoint_sha256,
            "selected_rho_endpoint": 0.10,
            "bridge_channel_policy": "physical45",
            "stack_val_consumer_opened": True,
            "refit_performed": False,
        }
    )
    live_config = with_content_hash(
        {
            "contract": "prediction_anchored_live_teacher_config_v1",
            "teacher_binding_sha256": "f" * 64,
            "binding_kind": "primary",
            "checkpoint_sha256": live.checkpoint_sha256,
            "channel_policy": "physical45",
            "parameters_frozen": True,
            "input_gradient_enabled": True,
            "checkpoint_refit_forbidden": True,
        }
    )
    cache = None
    if include_cache:
        cache = with_content_hash(
            {
                "contract": "prediction_anchored_teacher_logit_cache_v1",
                "cache_namespace": "physical45_selected_bridge_teacher",
                "teacher_binding_kind": "primary",
                "teacher_binding_sha256": live_config["teacher_binding_sha256"],
                "checkpoint_sha256": live.checkpoint_sha256,
                "live_checkpoint_sha256": live.checkpoint_sha256,
                "channel_policy": "physical45",
                "field_condition": "bridge_0.100",
                "rho_endpoint": 0.10,
                "target_logits_detached": True,
                "same_checkpoint_target_and_live": True,
                "checkpoint_refit_forbidden": True,
            }
        )
    return selected, live_config, cache


def test_c0_zero_initialization_numerical_spaces_and_five_channel_pass_through():
    batch, scaler = _fixture()
    model = PredictionAnchoredC0Correction(
        scaler, C0CorrectionConfig(d_model=32, particle_mlp_layers=1, dropout=0)
    )
    output = model(
        torch.as_tensor(batch["hlt_tokens"]),
        torch.as_tensor(batch["mask"]),
        torch.as_tensor(batch["f0"]),
        torch.as_tensor(batch["h0"]),
    )
    assert torch.equal(output.f_hat, torch.as_tensor(batch["f0"]))
    assert torch.count_nonzero(output.standardized_raw_correction).item() == 0
    assert torch.equal(output.f_hat[..., 45:], torch.as_tensor(batch["f0"])[..., 45:])
    assert output.diagnostics["learned_gate_present"] is False
    assert not any("gate" in name.lower() for name, _ in model.named_parameters())
    with pytest.raises(ValueError, match="physical-field"):
        model(
            torch.as_tensor(batch["hlt_tokens"]),
            torch.as_tensor(batch["mask"]),
            torch.as_tensor(batch["f0"]),
            torch.as_tensor(batch["h0"]),
            f0_space="conditioning_standardized",
        )
    bad_f0 = torch.as_tensor(batch["f0"]).clone()
    padding = ~torch.as_tensor(batch["mask"])
    bad_f0[padding, 0] = 1.0
    with pytest.raises(ValueError, match="padded"):
        model(
            torch.as_tensor(batch["hlt_tokens"]),
            torch.as_tensor(batch["mask"]),
            bad_f0,
            torch.as_tensor(batch["h0"]),
        )


def test_ce_only_objective_does_not_require_unused_privileged_bridge_fields():
    batch, scaler = _fixture(n=4, p=3)
    batch.pop("bridge_fields")
    batch.pop("true_fields")
    model = PredictionAnchoredC0Correction(
        scaler, C0CorrectionConfig(d_model=24, particle_mlp_layers=1, dropout=0)
    )
    live = _live_consumer()
    output = model(
        torch.as_tensor(batch["hlt_tokens"]),
        torch.as_tensor(batch["mask"]),
        torch.as_tensor(batch["f0"]),
        torch.as_tensor(batch["h0"]),
    )
    tensor_batch = {
        key: torch.as_tensor(value) if isinstance(value, np.ndarray) else value
        for key, value in batch.items()
    }
    logits = live(tensor_batch, output.f_hat)
    loss, diagnostics = compute_c0_objective(
        output,
        tensor_batch,
        resolve_c0_loss_recipe("D10_L1_ce_only"),
        model.scalers,
        phase="distillation",
        live_logits=logits,
    )
    assert torch.isfinite(loss)
    assert diagnostics["coefficients"]["bridge"] == 0


def test_componentwise_trust_saturation_and_declarative_no_trust():
    batch, scaler = _fixture(n=4, p=3)
    bounded = PredictionAnchoredC0Correction(
        scaler,
        C0CorrectionConfig(d_model=24, particle_mlp_layers=1, head_hidden_dim=16, dropout=0),
    )
    unbounded = PredictionAnchoredC0Correction(
        scaler,
        C0CorrectionConfig(
            d_model=24,
            particle_mlp_layers=1,
            head_hidden_dim=16,
            dropout=0,
            trust_bound_enabled=False,
        ),
    )
    for model in (bounded, unbounded):
        for head in model.radius_heads:
            final = head.net[-1]
            final.bias.data.fill_(100.0)
    inputs = tuple(
        torch.as_tensor(batch[name]) for name in ("hlt_tokens", "mask", "f0", "h0")
    )
    bounded_output = bounded(*inputs)
    unbounded_output = unbounded(*inputs)
    trust = bounded.scalers.trust_scale[:45]
    assert torch.all(bounded_output.physical_correction.abs() <= trust + 1e-6)
    assert bounded_output.saturation_mask is not None
    assert bounded_output.saturation_mask.any()
    assert unbounded_output.saturation_mask is None
    assert unbounded_output.diagnostics["saturation_definition"] == "not_applicable"
    assert torch.any(unbounded_output.physical_correction.abs() > trust)
    assert torch.equal(bounded_output.f_hat[..., 45:], inputs[2][..., 45:])


def test_group_balanced_standardized_huber_and_smoothness_match_hand_math():
    prediction = torch.full((1, 3, 50), 2.0)
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[True, True, False]])
    prediction[:, 2] = 1_000.0
    huber, groups = masked_group_balanced_huber(
        prediction, target, mask, torch.ones(50)
    )
    assert len(groups) == 12
    assert float(huber) == pytest.approx(1.5, abs=1e-7)

    tokens = torch.zeros((1, 2, 14))
    tokens[..., 0] = 1
    tokens[0, :, 1] = torch.tensor([0.0, 0.01])
    valid = torch.ones((1, 2), dtype=torch.bool)
    graph = build_directed_neighbor_graph(tokens, valid)
    assert graph.edge_valid.sum().item() == 2
    assert torch.all(graph.neighbor_indices[0, :, 0] == torch.tensor([1, 0]))
    correction = torch.zeros((1, 2, 45))
    correction[:, 1] = 1.0
    smooth, smooth_groups = local_smoothness_loss(
        correction, tokens, valid, torch.ones(50), graph=graph
    )
    assert len(smooth_groups) == 12
    assert float(smooth) == pytest.approx(1.0, abs=2e-4)


def test_kd_direction_matches_hand_computation():
    live = torch.tensor([[0.2, -0.1, 0.4]], dtype=torch.float64)
    target = torch.tensor([[0.4, 0.0, -0.2]], dtype=torch.float64)
    tau = 2.0
    actual = distillation_kl_loss(live, target, temperature=tau)
    p = torch.softmax(target / tau, dim=-1)
    q = torch.softmax(live / tau, dim=-1)
    expected = tau**2 * torch.sum(p * (torch.log(p) - torch.log(q)))
    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_exact_l0_through_l10_mapping_and_l6_l7_l8_factor_isolation():
    recipes = c0_loss_recipes()
    assert tuple(recipes) == C0_CANONICAL_RUN_IDS
    assert resolve_c0_loss_recipe("D10_A0_c0_delta").run_id == "D10_L8_full_c0"
    assert recipes["D10_L6_kd_ce_bridge"].anchor == 0
    assert recipes["D10_L6_kd_ce_bridge"].smooth == 0
    assert recipes["D10_L7_plus_anchor"].anchor == 0.02
    assert recipes["D10_L7_plus_anchor"].smooth == 0
    assert recipes["D10_L8_full_c0"].anchor == 0.02
    assert recipes["D10_L8_full_c0"].smooth == 0.01
    assert recipes["D10_L9_full_true_target"].true == 0.05
    assert recipes["D10_L10_no_trust"].trust_bound_enabled is False
    assert recipes["D10_L10_no_trust"].selectable_for_primary_deployment is False
    assert recipes["D10_L0_bridge_only"].requires_selected_teacher is False
    assert recipes["D10_L1_ce_only"].field_warmup is False
    assert recipes["D10_L2_kd_only"].field_warmup is False


def test_campaign_manifest_launches_only_l0_early_and_declares_all_paired_runs():
    _, scaler = _fixture(n=4, p=3)
    manifest = build_c0_campaign_manifest(
        C0CampaignConfig(
            field_warmup_steps=3,
            phase2_epochs=2,
            model_width=24,
            particle_mlp_layers=1,
            head_hidden_dim=16,
            dropout=0,
        ),
        scaler_artifact=scaler,
    )
    assert manifest["canonical_configuration_count"] == 11
    assert manifest["paired_replica_count"] == 33
    rows = {row["run_id"]: row for row in manifest["runs"]}
    assert rows["D10_L0_bridge_only"]["stage"] == "B3"
    assert rows["D10_L0_bridge_only"]["launch_group"] == "parallel_with_consumer_training"
    assert rows["D10_L0_bridge_only"]["requires_selected_bridge_consumer_json"] is False
    assert all(
        row["stage"] == "B6" and row["requires_selected_bridge_consumer_json"]
        for run_id, row in rows.items()
        if run_id != "D10_L0_bridge_only"
    )
    assert rows["D10_L1_ce_only"]["requires_primary_target_logit_cache"] is False
    assert rows["D10_L2_kd_only"]["requires_primary_target_logit_cache"] is True
    assert manifest["launch_contract"]["guessed_consumer_allowed"] is False


def test_teacher_lineage_requires_locked_selection_and_same_target_live_checkpoint():
    live = _live_consumer()
    selected, live_config, cache = _teacher_artifacts(live)
    recipe = resolve_c0_loss_recipe("D10_L2_kd_only")
    lineage = validate_c0_teacher_lineage(
        recipe,
        live_consumer=live,
        selected_bridge_consumer=selected,
        live_teacher_config=live_config,
        target_cache_manifest=cache,
    )
    assert lineage["same_checkpoint_selection_target_live"] is True
    changed_cache = with_content_hash(
        {
            **{key: value for key, value in cache.items() if key != "content_hash"},
            "checkpoint_sha256": "9" * 64,
        }
    )
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        validate_c0_teacher_lineage(
            recipe,
            live_consumer=live,
            selected_bridge_consumer=selected,
            live_teacher_config=live_config,
            target_cache_manifest=changed_cache,
        )
    with pytest.raises(ValueError, match="selected_bridge_consumer"):
        validate_c0_teacher_lineage(
            recipe,
            live_consumer=live,
            selected_bridge_consumer=None,
            live_teacher_config=live_config,
            target_cache_manifest=cache,
        )
    with pytest.raises(ValueError, match="without a guessed"):
        validate_c0_teacher_lineage(
            resolve_c0_loss_recipe("D10_L0_bridge_only"),
            live_consumer=live,
            selected_bridge_consumer=selected,
            live_teacher_config=live_config,
            target_cache_manifest=cache,
        )


def test_frozen_live_t10_has_field_gradient_but_no_parameter_gradient():
    batch, scaler = _fixture(n=6, p=4)
    live = _live_consumer()
    batch = _add_target_logits(batch, live)
    model = PredictionAnchoredC0Correction(
        scaler, C0CorrectionConfig(d_model=32, particle_mlp_layers=1, dropout=0)
    )
    output = model(
        torch.as_tensor(batch["hlt_tokens"]),
        torch.as_tensor(batch["mask"]),
        torch.as_tensor(batch["f0"]),
        torch.as_tensor(batch["h0"]),
    )
    tensor_batch = {key: torch.as_tensor(value) if isinstance(value, np.ndarray) else value for key, value in batch.items()}
    logits = live(tensor_batch, output.f_hat)
    loss, diagnostics = compute_c0_objective(
        output,
        tensor_batch,
        resolve_c0_loss_recipe("D10_L8_full_c0"),
        model.scalers,
        phase="distillation",
        live_logits=logits,
        target_logits=torch.as_tensor(batch["target_logits"]),
    )
    loss.backward()
    final_gradients = [head.net[-1].weight.grad for head in model.radius_heads]
    assert all(value is not None and torch.isfinite(value).all() for value in final_gradients)
    assert any(torch.count_nonzero(value).item() > 0 for value in final_gradients)
    assert all(parameter.grad is None for parameter in live.consumer.parameters())
    assert live.consumer.training is False
    assert diagnostics["coefficients"]["gate"] == 0


def test_l0_reachability_metrics_promote_every_radius_and_group():
    rng = np.random.default_rng(7)
    target = rng.normal(size=(3, 4, 45)).astype(np.float32)
    mask = np.ones((3, 4), dtype=bool)
    metrics = bridge_reachability_metrics(target.copy(), target, mask)
    assert len(metrics["by_radius"]) == 3
    assert len(metrics["by_radius_semantic_group"]) == 12
    assert metrics["overall"]["explained_variance"] == pytest.approx(1.0)
    assert metrics["overall"]["normalized_mse"] == pytest.approx(0.0)
    assert metrics["overall"]["pearson"] == pytest.approx(1.0)
    assert metrics["overall"]["cosine"] == pytest.approx(1.0)
    assert metrics["selection_threshold_applied"] is False


def test_deterministic_checkpoint_rules_include_the_l0_exception():
    state = lambda value: {"weight": torch.tensor([value])}
    post = select_postteacher_checkpoint(
        [
            {"epoch": 1, "accuracy": 0.80000, "cross_entropy": 0.70, "model_state_dict": state(1)},
            {"epoch": 2, "accuracy": 0.80005, "cross_entropy": 0.69, "model_state_dict": state(2)},
            {"epoch": 3, "accuracy": 0.80005, "cross_entropy": 0.6900005, "model_state_dict": state(3)},
        ]
    )
    assert post["artifact"]["selected_epoch"] == 2
    l0 = select_l0_checkpoint(
        [
            {"epoch": 1, "bridge_loss": 1.00000, "normalized_bridge_mse": 0.8, "model_state_dict": state(1)},
            {"epoch": 2, "bridge_loss": 1.00005, "normalized_bridge_mse": 0.7, "model_state_dict": state(2)},
            {"epoch": 3, "bridge_loss": 1.00005, "normalized_bridge_mse": 0.7, "model_state_dict": state(3)},
        ]
    )
    assert l0["artifact"]["selected_epoch"] == 2
    assert l0["artifact"]["selected_teacher_required"] is False


def test_ram_resume_is_same_allocation_only_and_access_is_fail_closed():
    batch, scaler = _fixture(n=4, p=3)
    model = PredictionAnchoredC0Correction(
        scaler, C0CorrectionConfig(d_model=24, particle_mlp_layers=1, dropout=0)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    config = C0TrainConfig(
        run_id="D10_L0_bridge_only",
        seed_id=101,
        field_warmup_steps=1,
        phase2_epochs=1,
        allocation_id="allocation-1",
    )
    parents = {"r0": "a" * 64, "scaler": "b" * 64}
    snapshot = capture_c0_ram_snapshot(
        model=model,
        optimizer=optimizer,
        config=config,
        phase="zero_initialization",
        completed_steps=0,
        parent_hashes=parents,
    )
    original = {key: value.clone() for key, value in model.state_dict().items()}
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1)
    restore_c0_ram_snapshot(
        snapshot,
        model=model,
        optimizer=optimizer,
        allocation_id="allocation-1",
        parent_hashes=parents,
    )
    assert all(torch.equal(model.state_dict()[key], value) for key, value in original.items())
    with pytest.raises(PermissionError, match="cross-allocation"):
        restore_c0_ram_snapshot(
            snapshot,
            model=model,
            optimizer=optimizer,
            allocation_id="allocation-2",
            parent_hashes=parents,
        )
    assert validate_model_val_stop_access(_access_receipt())["ok"]
    with pytest.raises(PermissionError):
        validate_model_val_stop_access(_access_receipt("stack_val_consumer", "consumer_confirmation"))


def test_two_phase_l0_trains_without_guessed_teacher_and_postteacher_fails_without_one():
    batch, scaler = _fixture(n=6, p=4)
    model = PredictionAnchoredC0Correction(
        scaler, C0CorrectionConfig(d_model=24, particle_mlp_layers=1, dropout=0)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    config = C0TrainConfig(
        run_id="D10_L0_bridge_only",
        seed_id=101,
        field_warmup_steps=2,
        phase2_epochs=2,
        early_stop_patience=-1,
        allocation_id="l0-allocation",
    )

    def factories(phase, index):
        return [batch]

    result = train_c0_replica(
        config,
        model=model,
        optimizer=optimizer,
        train_batches=factories,
        model_val_stop_batches=factories,
        model_val_stop_access_receipt=_access_receipt(),
        parent_hashes={"r0": "a" * 64, "scaler": "b" * 64},
    )
    assert result["audit"]["warmup_steps_completed"] == 2
    assert result["audit"]["warmup_validation_could_stop"] is False
    assert result["candidate_weights"]["selected_teacher_checkpoint_sha256"] is None
    post_config = C0TrainConfig(
        run_id="D10_L2_kd_only",
        seed_id=101,
        field_warmup_steps=0,
        phase2_epochs=1,
        allocation_id="post-allocation",
    )
    post_model = PredictionAnchoredC0Correction(
        scaler, C0CorrectionConfig(d_model=24, particle_mlp_layers=1, dropout=0)
    )
    with pytest.raises(ValueError, match="selected_bridge_consumer"):
        train_c0_replica(
            post_config,
            model=post_model,
            optimizer=torch.optim.AdamW(post_model.parameters(), lr=1e-3),
            train_batches=factories,
            model_val_stop_batches=factories,
            model_val_stop_access_receipt=_access_receipt(),
            parent_hashes={"r0": "a" * 64},
        )


def test_postteacher_training_records_verified_selection_cache_live_lineage():
    batch, scaler = _fixture(n=6, p=4)
    live = _live_consumer()
    batch = _add_target_logits(batch, live)
    selected, live_config, cache = _teacher_artifacts(live)
    model = PredictionAnchoredC0Correction(
        scaler, C0CorrectionConfig(d_model=24, particle_mlp_layers=1, dropout=0)
    )
    config = C0TrainConfig(
        run_id="D10_L2_kd_only",
        seed_id=101,
        field_warmup_steps=0,
        phase2_epochs=1,
        early_stop_patience=-1,
        allocation_id="post-allocation",
    )

    def batches(phase, index):
        return [batch]

    result = train_c0_replica(
        config,
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        train_batches=batches,
        model_val_stop_batches=batches,
        model_val_stop_access_receipt=_access_receipt(),
        parent_hashes={"r0": "a" * 64, "scaler": "b" * 64},
        live_consumer=live,
        selected_bridge_consumer=selected,
        live_teacher_config=live_config,
        target_cache_manifest=cache,
    )
    lineage = result["audit"]["teacher_lineage"]
    assert lineage["teacher_checkpoint_sha256"] == live.checkpoint_sha256
    assert lineage["same_checkpoint_selection_target_live"] is True
    assert result["candidate_weights"]["target_cache_sha256"] == cache["content_hash"]


def test_all_eleven_cpu_miniatures_have_finite_backward_and_exact_declared_phases():
    batch, scaler = _fixture(n=5, p=4)
    live = _live_consumer()
    batch = _add_target_logits(batch, live)
    result = run_c0_cpu_miniature(
        scaler_artifact=scaler,
        batch=batch,
        live_consumer=live,
        debug_width=24,
    )
    assert result["configuration_count"] == 11
    assert tuple(result["run_ids"]) == C0_CANONICAL_RUN_IDS
    assert result["scientific_results_allowed"] is False
    for run_id, row in result["runs"].items():
        assert set(row["phases"]) == (
            {"field_warmup", "distillation"}
            if resolve_c0_loss_recipe(run_id).field_warmup
            else {"distillation"}
        )
        assert all(phase["gradient_norm"] >= 0 for phase in row["phases"].values())


def test_measured_registry_and_median_only_publication(tmp_path):
    batch, scaler = _fixture(n=4, p=3)
    registry = build_campaign_registry()
    updated, measurement = measure_c0_registry_states(
        registry, scaler_artifact=scaler, model_width=24
    )
    assert set(measurement["measured_state_bytes"]) == set(C0_CANONICAL_RUN_IDS)
    c0_rows = {
        row["canonical_run_id"]: row
        for row in updated["runs"]
        if row["canonical_run_id"] in C0_CANONICAL_RUN_IDS
    }
    assert len(c0_rows) == 11
    assert all(row["measurement_status"] == "MEASURED" for row in c0_rows.values())

    model = PredictionAnchoredC0Correction(
        scaler, C0CorrectionConfig(d_model=24, particle_mlp_layers=1, dropout=0)
    )
    replicas = []
    for seed, loss, nmse in ((101, 0.8, 0.7), (202, 0.7, 0.6), (303, 0.9, 0.8)):
        replicas.append(
            C0ReplicaResult(
                run_id="D10_L0_bridge_only",
                seed_id=seed,
                metrics={"model_val_stop": {"bridge_loss": loss, "normalized_bridge_mse": nmse}},
                weights_payload={
                    "checkpoint_contract": "prediction_anchored_c0_replica_v1",
                    "run_id": "D10_L0_bridge_only",
                    "seed_id": seed,
                    "epoch": 1,
                    "model_config": model.config.to_artifact(),
                    "model_state_dict": deepcopy(model.state_dict()),
                    "scaler_sha256": model.scaler_sha256,
                    "parent_hashes": {"r0": "a" * 64},
                    "optimizer_state_dict": {"forbidden": True},
                    "generated_fields": torch.ones(2),
                },
            )
        )
    aggregate = aggregate_c0_replicas(replicas)
    assert aggregate["median_seed_id"] == 101
    assert aggregate["best_seed_id"] == 202
    publication = publish_c0_paired_replicas(replicas, output_dir=tmp_path / "published")
    assert publication["persistent_artifacts"] == [
        "aggregate_metrics.json",
        "median_weights.pt",
        "publication.json",
    ]
    loaded = torch.load(publication["checkpoint"], map_location="cpu", weights_only=False)
    assert loaded["seed_id"] == 101
    assert "optimizer_state_dict" not in loaded
    assert "generated_fields" not in loaded
    assert loaded["frozen_parent_weights_persisted"] is False


def test_step5_plan_cli_dry_run_is_operator_usable(tmp_path):
    _, scaler = _fixture(n=4, p=3)
    scaler_path = tmp_path / "scaler.json"
    scaler_path.write_text(json.dumps(scaler), encoding="utf-8")
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "train_prediction_anchored_bridge_reconstructor.py"
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--mode",
            "plan",
            "--scaler",
            str(scaler_path),
            "--field-warmup-steps",
            "2",
            "--phase2-epochs",
            "1",
            "--model-width",
            "24",
            "--particle-mlp-layers",
            "1",
            "--head-hidden-dim",
            "16",
            "--dropout",
            "0",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["dry_run"] is True
    assert payload["manifest"]["canonical_configuration_count"] == 11
    assert payload["manifest"]["launch_contract"]["guessed_consumer_allowed"] is False
