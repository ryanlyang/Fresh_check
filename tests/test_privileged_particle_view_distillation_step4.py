from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch import nn

from teacher_logit_reco.local_particle_residual_field.particle_view import (
    OFFLINE_KD_SCREEN,
    PARTICLE_VIEW_COORDINATE_PARENT_HASH_FIELDS,
    OracleObjectiveConfig,
    ParticleViewConsumer,
    ParticleViewConsumerConfig,
    ParticleViewConsumerTrainConfig,
    RecoverabilityCoDesignConfig,
    RecoverabilityCoDesignProjection,
    RecoveryProbeConfig,
    TargetCandidateMetrics,
    audit_live_selected_view_equivalence,
    audit_zero_scaled_a0_endpoint,
    build_codesign_ledger,
    build_clean_consumer_registration,
    build_scientific_warnings,
    build_target_metrics_artifact,
    build_target_metrics_from_counterfactual,
    build_two_pass_candidate_artifact,
    co_design_projection_loss,
    co_design_schedule,
    consumer_diagnostics,
    evaluate_view_counterfactuals,
    finalize_selected_view_coordinate,
    fit_particle_view_normalizer,
    load_particle_view_normalizer,
    load_selected_view_cache,
    normalize_particle_view,
    oracle_discovery_loss,
    prepare_clean_consumer_view,
    publish_selected_view_cache,
    quantized_view_diagnostics,
    rank_target_candidates,
    recovery_probe_losses,
    select_codesign_cycle,
    select_target_candidates,
    train_particle_view_consumer,
    train_recovery_probe,
    validate_content_hash,
    view_relational_loss,
    write_particle_view_normalizer,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class _FakeEmbed(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.linear = nn.Linear(17, hidden)

    def forward(self, features):
        return self.linear(features.transpose(1, 2)).transpose(1, 2)


class _FakePair(nn.Module):
    def __init__(self, heads: int) -> None:
        super().__init__()
        self.heads = heads

    def forward(self, vectors):
        pt = vectors[:, 0]
        pair = (pt[:, :, None] - pt[:, None, :]).abs()
        return pair[:, None].repeat(1, self.heads, 1, 1)


class _FakeBlock(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden, hidden)

    def forward(self, values):
        return values + torch.tanh(self.linear(values))


class _FakePart(nn.Module):
    def __init__(self, hidden: int = 16, heads: int = 4) -> None:
        super().__init__()
        self.embed = _FakeEmbed(hidden)
        self.pair_embed = _FakePair(heads)
        self.blocks = nn.ModuleList([_FakeBlock(hidden), _FakeBlock(hidden)])
        self.head = nn.Linear(hidden, 3)

    def forward(self, features, *, v, mask):
        pair = self.pair_embed(v)
        values = self.embed(features).permute(2, 0, 1)
        pair_context = pair.mean(dim=(1, 3)).transpose(0, 1)[..., None]
        values = values + pair_context
        for block in self.blocks:
            values = block(values)
        valid = mask[:, 0].transpose(0, 1)[..., None]
        pooled = (values * valid).sum(dim=0) / valid.sum(dim=0).clamp_min(1)
        return self.head(pooled)


class _FakeA0(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mod = _FakePart()

    def forward(self, points, features, lorentz_vectors, mask):
        del points
        return self.mod(features, v=lorentz_vectors, mask=mask)


class _FakeTrimPart(_FakePart):
    """Mimic Weaver's batch-local trim before pair/embed transformer work."""

    def forward(self, features, *, v, mask):
        particles = int(mask[:, 0].sum(dim=1).max().item())
        features = features[:, :, :particles]
        v = v[:, :, :particles]
        trimmed_mask = mask[:, :, :particles]
        pair = self.pair_embed(v)
        values = self.embed(features).permute(2, 0, 1)
        pair_context = pair.mean(dim=(1, 3)).transpose(0, 1)[..., None]
        values = values + pair_context
        for block in self.blocks:
            values = block(values)
        valid = trimmed_mask[:, 0].transpose(0, 1)[..., None]
        pooled = (values * valid).sum(dim=0) / valid.sum(dim=0).clamp_min(1)
        return self.head(pooled)


class _FakeTrimA0(_FakeA0):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.mod = _FakeTrimPart()


class _FakePermutingTrimmer(nn.Module):
    """Drop and reorder active particles exactly as Weaver training can."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "indices", torch.tensor([4, 1, 3], dtype=torch.int64)
        )

    def forward(self, features, vectors, mask, pair_inputs=None):
        indices = self.indices.to(features.device)
        return (
            features.index_select(2, indices),
            vectors.index_select(2, indices),
            mask.index_select(2, indices),
            pair_inputs,
        )


class _FakePermutingPart(_FakePart):
    def __init__(self, hidden: int = 16, heads: int = 4) -> None:
        super().__init__(hidden=hidden, heads=heads)
        self.trimmer = _FakePermutingTrimmer()

    def forward(self, features, *, v, mask):
        features, v, mask, _ = self.trimmer(features, v, mask, None)
        pair = self.pair_embed(v)
        values = self.embed(features).permute(2, 0, 1)
        pair_context = pair.mean(dim=(1, 3)).transpose(0, 1)[..., None]
        values = values + pair_context
        for block in self.blocks:
            values = block(values)
        valid = mask[:, 0].transpose(0, 1)[..., None]
        pooled = (values * valid).sum(dim=0) / valid.sum(dim=0).clamp_min(1)
        return self.head(pooled)


class _FakePermutingA0(_FakeA0):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.mod = _FakePermutingPart()


class _FakeWeaverBatchFirstEmbed(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.linear = nn.Linear(17, hidden)

    def forward(self, features):
        return self.linear(features.transpose(1, 2))


class _FakeWeaverBatchFirstPart(_FakePart):
    def __init__(self) -> None:
        super().__init__(hidden=5, heads=1)
        self.embed = _FakeWeaverBatchFirstEmbed(5)
        self.trimmer = _FakePermutingTrimmer()

    def forward(self, features, *, v, mask):
        features, v, mask, _ = self.trimmer(features, v, mask, None)
        pair = self.pair_embed(v)
        values = self.embed(features)
        values = values + pair.mean(dim=(1, 3))[..., None]
        for block in self.blocks:
            values = block(values)
        valid = mask[:, 0, :, None]
        pooled = (values * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        return self.head(pooled)


class _FakeWeaverBatchFirstA0(_FakeA0):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.mod = _FakeWeaverBatchFirstPart()


def _consumer_batch(batch: int = 3, particles: int = 5, dim: int = 4):
    torch.manual_seed(13)
    mask = torch.ones(batch, 1, particles, dtype=torch.bool)
    mask[0, 0, -1] = False
    return {
        "points": torch.randn(batch, 2, particles),
        "features": torch.randn(batch, 17, particles),
        "lorentz_vectors": torch.randn(batch, 4, particles),
        "mask": mask,
        "labels": torch.tensor([0, 1, 2])[:batch],
        "true_view": torch.randn(batch, particles, dim),
        "predicted_view": torch.randn(batch, particles, dim),
    }


def _consumer(path: str = "token_and_pair") -> ParticleViewConsumer:
    return ParticleViewConsumer(
        _FakeA0(),
        ParticleViewConsumerConfig(
            view_dim=4,
            hidden_dim=16,
            num_heads=4,
            view_path=path,
        ),
    )


@pytest.mark.parametrize(
    ("view_path", "injection_block"),
    [("raw_projected", -1), ("token_only", -1)],
)
def test_raw_and_post_embedding_consumer_interfaces_keep_exact_a0_endpoint(
    view_path,
    injection_block,
):
    batch = _consumer_batch()
    model = ParticleViewConsumer(
        _FakeA0(),
        ParticleViewConsumerConfig(
            view_dim=4,
            hidden_dim=16,
            num_heads=4,
            view_path=view_path,
            injection_block=injection_block,
        ),
    )
    audit = audit_zero_scaled_a0_endpoint(
        model,
        points=batch["points"],
        features=batch["features"],
        lorentz_vectors=batch["lorentz_vectors"],
        mask=batch["mask"],
        view=batch["true_view"],
    )
    assert audit["ok"]
    assert audit["maximum_absolute_logit_difference"] == 0.0


def test_zero_scaled_warm_start_and_two_step_gradient_reachability():
    batch = _consumer_batch()
    model = _consumer()
    audit = audit_zero_scaled_a0_endpoint(
        model,
        points=batch["points"],
        features=batch["features"],
        lorentz_vectors=batch["lorentz_vectors"],
        mask=batch["mask"],
        view=batch["true_view"],
    )
    assert audit["maximum_absolute_logit_difference"] == 0.0

    view = batch["true_view"].clone().requires_grad_(True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    first = model(
        batch["points"],
        batch["features"],
        batch["lorentz_vectors"],
        batch["mask"],
        view,
    )
    torch.nn.functional.cross_entropy(first.logits, batch["labels"]).backward()
    assert model.raw_token_scale.grad.abs().item() > 0
    assert model.raw_pair_scale.grad.abs().item() > 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    view.grad = None
    second = model(
        batch["points"],
        batch["features"],
        batch["lorentz_vectors"],
        batch["mask"],
        view,
    )
    (
        torch.nn.functional.cross_entropy(second.logits, batch["labels"])
        + 0.01 * second.trust_loss
    ).backward()
    assert view.grad is not None and view.grad.abs().sum().item() > 0
    assert model.view_adapter[0].weight.grad.abs().sum().item() > 0
    assert sum(
        parameter.grad.abs().sum().item()
        for parameter in model.gate.parameters()
        if parameter.grad is not None
    ) > 0
    with torch.no_grad():
        model.raw_token_scale.fill_(100)
        model.raw_pair_scale.fill_(-100)
    bounded = model(
        batch["points"],
        batch["features"],
        batch["lorentz_vectors"],
        batch["mask"],
        batch["true_view"],
    )
    assert bounded.effective_token_scale.item() == 1
    assert bounded.effective_pair_scale.item() == -1


def test_consumer_accepts_weaver_trimmed_token_and_pair_layouts():
    batch = _consumer_batch(batch=3, particles=7)
    batch["mask"][:, :, -2:] = False
    model = ParticleViewConsumer(
        _FakeTrimA0(),
        ParticleViewConsumerConfig(
            view_dim=4,
            hidden_dim=16,
            num_heads=4,
            view_path="token_and_pair",
        ),
    )
    audit = audit_zero_scaled_a0_endpoint(
        model,
        points=batch["points"],
        features=batch["features"],
        lorentz_vectors=batch["lorentz_vectors"],
        mask=batch["mask"],
        view=batch["true_view"],
    )
    assert audit["ok"]
    with torch.no_grad():
        model.raw_token_scale.fill_(0.3)
        model.raw_pair_scale.fill_(0.2)
    output = model(
        batch["points"],
        batch["features"],
        batch["lorentz_vectors"],
        batch["mask"],
        batch["true_view"],
    )
    assert output.logits.shape == (3, 3)
    assert output.token_correction.shape == (3, 7, 16)
    assert output.pair_bias.shape == (3, 4, 7, 7)
    assert torch.count_nonzero(output.token_correction[:, -2:]) == 0
    assert torch.count_nonzero(output.pair_bias[:, :, -2:]) == 0


def test_consumer_tracks_weaver_training_permutation_and_active_subsample():
    batch = _consumer_batch(batch=3, particles=5)
    batch["mask"][:] = True
    model = ParticleViewConsumer(
        _FakePermutingA0(),
        ParticleViewConsumerConfig(
            view_dim=4,
            hidden_dim=16,
            num_heads=4,
            view_path="token_and_pair",
        ),
    )
    assert audit_zero_scaled_a0_endpoint(
        model,
        points=batch["points"],
        features=batch["features"],
        lorentz_vectors=batch["lorentz_vectors"],
        mask=batch["mask"],
        view=batch["true_view"],
    )["ok"]
    with torch.no_grad():
        model.raw_token_scale.fill_(0.3)
        model.raw_pair_scale.fill_(0.2)
    output = model(
        batch["points"],
        batch["features"],
        batch["lorentz_vectors"],
        batch["mask"],
        batch["true_view"],
    )
    selected = torch.tensor([False, True, False, True, True])
    assert output.logits.shape == (3, 3)
    assert torch.count_nonzero(output.token_correction[:, ~selected]) == 0
    pair_selected = selected[None, :, None] & selected[None, None, :]
    pair_selected = pair_selected[:, None].expand_as(output.pair_bias)
    assert torch.count_nonzero(output.pair_bias[~pair_selected]) == 0
    assert torch.count_nonzero(output.token_correction[:, selected]) > 0


def test_consumer_prefers_weaver_batch_first_when_hidden_equals_particles():
    batch = _consumer_batch(batch=3, particles=5)
    batch["mask"][:] = True
    model = ParticleViewConsumer(
        _FakeWeaverBatchFirstA0(),
        ParticleViewConsumerConfig(
            view_dim=4,
            hidden_dim=5,
            num_heads=1,
            view_path="token_and_pair",
        ),
    )
    assert audit_zero_scaled_a0_endpoint(
        model,
        points=batch["points"],
        features=batch["features"],
        lorentz_vectors=batch["lorentz_vectors"],
        mask=batch["mask"],
        view=batch["true_view"],
    )["ok"]
    with torch.no_grad():
        model.raw_token_scale.fill_(0.3)
        model.raw_pair_scale.fill_(0.2)
    assert model(
        batch["points"],
        batch["features"],
        batch["lorentz_vectors"],
        batch["mask"],
        batch["true_view"],
    ).logits.shape == (3, 3)


@pytest.mark.parametrize("path", ["token_only", "pair_only", "token_and_pair"])
def test_consumer_injection_paths_masks_and_diagnostics(path):
    batch = _consumer_batch()
    model = _consumer(path)
    with torch.no_grad():
        model.raw_token_scale.fill_(0.3)
        model.raw_pair_scale.fill_(0.2)
    output = model(
        batch["points"],
        batch["features"],
        batch["lorentz_vectors"],
        batch["mask"],
        batch["true_view"],
    )
    invalid = ~batch["mask"][:, 0]
    assert torch.count_nonzero(output.token_correction[invalid]) == 0
    pair_invalid = invalid[:, None, :, None] | invalid[:, None, None, :]
    assert torch.count_nonzero(
        output.pair_bias[pair_invalid.expand_as(output.pair_bias)]
    ) == 0
    diagnostics = consumer_diagnostics(output, batch["mask"])
    assert diagnostics["gate_p50"] == pytest.approx(0.5)
    assert "effective_pair_bias_head0_rms" in diagnostics


def test_clean_view_augmentation_order_preserves_exact_eval_cache():
    view = torch.tensor(
        [[[1.0, 3.0], [2.0, 5.0], [99.0, 99.0]]]
    )
    mask = torch.tensor([[True, True, False]])
    evaluated = prepare_clean_consumer_view(view, mask, training=False)
    assert torch.equal(
        evaluated, torch.tensor([[[1.0, 3.0], [2.0, 5.0], [0.0, 0.0]]])
    )
    trained = prepare_clean_consumer_view(
        view,
        mask,
        training=True,
        coordinate_dropout=0.0,
        noise_sigma=0.0,
    )
    assert torch.allclose(trained[:, :2].mean(dim=1), torch.zeros(1, 2))
    assert torch.count_nonzero(trained[:, 2]) == 0


def test_normalizer_cache_publication_reload_and_stale_rejection(tmp_path):
    rng = np.random.default_rng(4)
    raw = rng.normal(size=(4, 5, 4)).astype(np.float32)
    mask = np.ones((4, 5), dtype=bool)
    mask[0, -1] = False
    raw[~mask] = 0
    normalizer = fit_particle_view_normalizer(
        raw,
        mask,
        train_split_sha256=_sha("train"),
        generator_checkpoint_sha256=_sha("generator"),
    )
    artifact = write_particle_view_normalizer(
        tmp_path / "normalizer.json", normalizer
    )
    assert load_particle_view_normalizer(
        tmp_path / "normalizer.json"
    ) == normalizer
    manifest = publish_selected_view_cache(
        tmp_path / "cache",
        split="train",
        split_sha256=_sha("train"),
        ordered_identity_sha256=_sha("identity"),
        raw_view=raw,
        mask=mask,
        normalizer=normalizer,
        coordinate_binding_sha256=_sha("coordinate"),
        target_id="VGEN_DIM4",
    )
    cached, cached_mask, loaded = load_selected_view_cache(
        tmp_path / "cache" / "train_selected_views.json",
        expected_coordinate_binding_sha256=_sha("coordinate"),
        expected_split_sha256=_sha("train"),
        expected_normalizer_sha256=artifact["content_hash"],
    )
    assert cached.dtype.str == "<f4"
    assert loaded == manifest
    assert audit_live_selected_view_equivalence(
        raw_view=raw,
        mask=mask,
        cached_view=cached,
        cached_mask=cached_mask,
        normalizer=normalizer,
    )["ok"]
    with pytest.raises(ValueError, match="split_sha256 mismatch"):
        load_selected_view_cache(
            tmp_path / "cache" / "train_selected_views.json",
            expected_coordinate_binding_sha256=_sha("coordinate"),
            expected_split_sha256=_sha("other"),
            expected_normalizer_sha256=artifact["content_hash"],
        )
    with pytest.raises(ValueError, match="forbidden"):
        publish_selected_view_cache(
            tmp_path / "bad",
            split="final_test",
            split_sha256=_sha("final"),
            ordered_identity_sha256=_sha("identity"),
            raw_view=raw,
            mask=mask,
            normalizer=normalizer,
            coordinate_binding_sha256=_sha("coordinate"),
            target_id="VGEN_DIM4",
        )
    assert quantized_view_diagnostics(
        cached, cached_mask, normalizer, bits=8
    )["diagnostic_only"]


def test_final_selected_coordinate_recomputes_and_audits(tmp_path):
    rng = np.random.default_rng(8)
    splits = ("train", "model_val_stop", "model_val_select")
    views = {
        split: rng.normal(size=(3, 4, 4)).astype(np.float32)
        for split in splits
    }
    masks = {
        split: np.ones((3, 4), dtype=bool)
        for split in splits
    }
    parents = {
        name: _sha(name) for name in PARTICLE_VIEW_COORDINATE_PARENT_HASH_FIELDS
    }
    definition = {
        "offline_tap_layer": "penultimate",
        "offline_tap_tensor_location": "post_block_pre_pool",
        "cross_attention_config_sha256": _sha("xattn"),
        "pair_feature_schema_sha256": _sha("pair"),
        "centering_policy": "masked_particle_mean_v1",
        "bounded_coordinate_policy": "tanh_then_center_minus2_plus2",
        "rate_budget_policy": "variance_rate_covariance_v1",
        "null_token_policy": "learned_null_v1",
        "bottleneck_width": 4,
    }
    publication = finalize_selected_view_coordinate(
        tmp_path,
        target_id="VGEN_DIM4",
        target_selection_sha256=_sha("selection"),
        raw_views_by_split=views,
        masks_by_split=masks,
        split_sha256_by_split={name: _sha(name) for name in splits},
        ordered_identity_sha256_by_split={
            name: _sha(f"{name}-identity") for name in splits
        },
        coordinate_parent_hashes=parents,
        coordinate_definition=definition,
    )
    validate_content_hash(publication)
    assert publication["normalizer_recomputed_after_selection"]
    assert not publication["final_test_materialized"]
    assert set(publication["cache_manifest_sha256_by_split"]) == set(splits)

    checkpoint = tmp_path / "clean_consumer.pt"
    checkpoint.write_bytes(b"registered-clean-consumer")
    clean = build_clean_consumer_registration(
        consumer_config=ParticleViewConsumerConfig(
            view_dim=4, hidden_dim=16, num_heads=4
        ),
        checkpoint_path=checkpoint,
        a0_registration_sha256=_sha("a0-registration"),
        coordinate_binding_sha256=publication[
            "coordinate_binding_sha256"
        ],
        selected_view_publication_sha256=publication["content_hash"],
        normalizer_sha256=publication["normalizer_sha256"],
        target_selection_sha256=_sha("selection"),
        train_identity_sha256=_sha("train-identity"),
        selected_epoch=7,
    )
    validate_content_hash(clean)
    assert clean["initialized_from_exact_a0"]
    assert clean["trained_from_epoch_zero_in_final_coordinates"]
    assert clean["raw_discovery_coordinates_seen"] is False
    assert clean["live_generator_used"] is False


def test_probe_losses_are_label_free_event_weighted_and_degenerate_safe():
    target = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0]], [[2.0, 0.0], [0.0, 0.0]]]
    )
    prediction = torch.zeros_like(target)
    mask = torch.tensor([[True, True], [True, False]])
    losses = recovery_probe_losses(prediction, target, mask)
    assert all(torch.isfinite(value) for value in losses.values())
    assert view_relational_loss(
        prediction[:, :1], target[:, :1], mask[:, :1]
    ).item() == 0
    empty = torch.zeros(1, 2, 2)
    empty_mask = torch.zeros(1, 2, dtype=torch.bool)
    assert recovery_probe_losses(empty, empty, empty_mask)["total"].item() == 0


def test_fixed_recovery_probe_completes_eight_epochs_and_binds_lineage(tmp_path):
    torch.manual_seed(3)
    batch = {
        "features": torch.randn(2, 17, 3),
        "mask": torch.ones(2, 3, dtype=torch.bool),
        "true_view": torch.randn(2, 3, 1),
    }
    hashes = {name: _sha(name) for name in (
        "target_registration_sha256",
        "normalizer_sha256",
        "train_identity_sha256",
        "model_val_stop_split_sha256",
        "hlt_preprocessing_sha256",
    )}
    registration = train_recovery_probe(
        config=RecoveryProbeConfig(view_dim=1),
        train_loader=[batch],
        model_val_stop_loader=[batch],
        output_dir=tmp_path,
        **hashes,
    )
    validate_content_hash(registration)
    assert registration["epochs_completed"] == 8
    assert registration["fixed_budget_completed"]
    assert registration["labels_exposed"] is False
    with pytest.raises(ValueError, match="labels"):
        train_recovery_probe(
            config=RecoveryProbeConfig(view_dim=1),
            train_loader=[{**batch, "labels": torch.zeros(2, dtype=torch.long)}],
            model_val_stop_loader=[batch],
            output_dir=tmp_path / "bad",
            **hashes,
        )


def _candidate(
    target_id: str,
    predicted_gain: float,
    oracle_gain: float,
    *,
    width: int = 4,
) -> TargetCandidateMetrics:
    return TargetCandidateMetrics(
        run_id=f"run_{target_id}",
        target_id=target_id,
        bottleneck_width=width,
        predicted_view_gain=predicted_gain,
        oracle_gain=oracle_gain,
        predicted_view_cross_entropy=0.6,
        zero_view_accuracy=0.70,
        predicted_view_accuracy=0.70 + predicted_gain,
        oracle_accuracy=0.70 + oracle_gain,
        a0_accuracy=0.70,
        target_registration_sha256=_sha(target_id),
    )


def test_target_ranking_forwards_two_plus_canonical_and_never_quality_gates():
    candidates = [
        _candidate("best", 0.003, 0.004),
        _candidate("second", 0.002, -0.001, width=2),
        _candidate("canonical", -0.01, -0.02),
    ]
    ranked = rank_target_candidates(candidates)
    assert [row["target_id"] for row in ranked] == [
        "best",
        "second",
        "canonical",
    ]
    selected = select_target_candidates(
        candidates, canonical_target_id="canonical"
    )
    assert selected["forwarded_target_ids"] == [
        "best",
        "second",
        "canonical",
    ]
    assert selected["quality_threshold_used_as_gate"] is False
    warning_metric = build_target_metrics_artifact(candidates[-1])
    warnings = build_scientific_warnings(
        candidates[-1],
        graph_node="stage_b",
        configuration_id="canonical",
        seed=101,
        split="model_val_select",
        supporting_metric_sha256=warning_metric["content_hash"],
        source_commit="abc123",
    )
    assert {row["warning_code"] for row in warnings} >= {
        "WARN_ORACLE_GAIN_BELOW_005",
        "WARN_ORACLE_GAIN_NONPOSITIVE",
    }
    assert all(not row["stops_execution"] for row in warnings)


def test_oracle_objective_screen_rate_control_and_codesign_contracts():
    assert OFFLINE_KD_SCREEN == (0.0, 0.25, 0.5, 1.0)
    logits = torch.randn(2, 3, requires_grad=True)
    labels = torch.tensor([0, 2])
    view = torch.randn(2, 4, 4, requires_grad=True)
    mask = torch.ones(2, 4, dtype=torch.bool)
    objective = oracle_discovery_loss(
        consumer_logits=logits,
        labels=labels,
        offline_logits=torch.randn(2, 3),
        raw_centered_view=view,
        mask=mask,
        trust_loss=torch.tensor(0.5),
        config=OracleObjectiveConfig(),
    )
    objective["total"].backward()
    assert view.grad is not None
    no_rate = OracleObjectiveConfig(
        rate_weight=0.0,
        covariance_weight=0.0,
        rate_budget_enabled=False,
    )
    assert no_rate.to_payload()["rate_budget_enabled"] is False

    config = RecoverabilityCoDesignConfig(view_dim=4)
    projection = RecoverabilityCoDesignProjection(config)
    rich = torch.randn(2, 5, 160)
    projected = projection(rich, mask=torch.ones(2, 5, dtype=torch.bool))
    assert projected.abs().max().item() <= 2.0
    assert torch.allclose(projected.mean(dim=1), torch.zeros(2, 4), atol=1e-6)
    schedule = co_design_schedule(config)
    assert len(schedule) == 12
    assert all(
        row["probe_optimizer_steps"]
        == 4 * row["projection_consumer_optimizer_steps"]
        for row in schedule
    )


def test_codesign_selection_ledger_two_pass_and_agreement_loss():
    rows = [
        {
            "cycle": cycle,
            "predicted_view_accuracy": 0.70 + cycle / 10_000,
            "predicted_gain": cycle / 10_000,
            "oracle_gain": 0.01,
            "oracle_accuracy": 0.71,
            "cross_entropy": 0.5,
        }
        for cycle in range(1, 13)
    ]
    assert select_codesign_cycle(rows)["cycle"] == 12
    config = RecoverabilityCoDesignConfig(view_dim=4)
    cycles = [
        {
            "cycle": cycle,
            "projection_checkpoint_sha256": _sha(f"p{cycle}"),
            "consumer_checkpoint_sha256": _sha(f"c{cycle}"),
            "probe_checkpoint_sha256": _sha(f"r{cycle}"),
            "probe_optimizer_steps": 2_000,
            "projection_consumer_optimizer_steps": 500,
        }
        for cycle in range(1, 13)
    ]
    ledger = build_codesign_ledger(
        config=config,
        rich_context_registration_sha256=_sha("rich"),
        provisional_head_registration_sha256=_sha("seed"),
        cycles=cycles,
        selected_cycle=12,
        final_projection_checkpoint_sha256=_sha("final"),
    )
    validate_content_hash(ledger)
    two_pass = build_two_pass_candidate_artifact(
        target_registration_sha256=_sha("target"),
        discovery_consumer_checkpoint_sha256=_sha("discovery"),
        frozen_generator_checkpoint_sha256=_sha("generator"),
        provisional_normalizer_sha256=_sha("normalizer"),
        probe_consumer_checkpoint_sha256=_sha("consumer"),
        recovery_probe_registration_sha256=_sha("probe"),
        model_val_select_metrics_sha256=_sha("metrics"),
    )
    assert two_pass["probe_consumer_reinitialized_from_a0"]

    view = torch.randn(2, 3, 4, requires_grad=True)
    loss = co_design_projection_loss(
        consumer_logits=torch.randn(2, 3, requires_grad=True),
        labels=torch.tensor([0, 1]),
        offline_logits=torch.randn(2, 3),
        raw_centered_view=view,
        probe_prediction=torch.randn(2, 3, 4),
        mask=torch.ones(2, 3, dtype=torch.bool),
        trust_loss=torch.tensor(0.5),
        oracle_config=OracleObjectiveConfig(),
    )
    loss["total"].backward()
    assert view.grad is not None and view.grad.abs().sum().item() > 0


def test_counterfactuals_use_one_consumer_and_training_budgets_are_locked():
    assert ParticleViewConsumerTrainConfig.for_role(
        "Cview_discovery"
    ).maximum_epochs == 30
    assert ParticleViewConsumerTrainConfig.for_role(
        "Cview_probe"
    ).early_stop_patience is None
    assert ParticleViewConsumerTrainConfig.for_role(
        "Cview_clean"
    ).maximum_epochs == 40
    batch = _consumer_batch()
    model = _consumer()
    with torch.no_grad():
        model.raw_token_scale.fill_(0.2)
        model.raw_pair_scale.fill_(0.1)
    metrics = evaluate_view_counterfactuals(model, [batch])
    validate_content_hash(metrics)
    assert metrics["same_consumer_checkpoint"]
    assert set(metrics) >= {"zero_view", "true_view", "predicted_view"}
    target_metrics = build_target_metrics_from_counterfactual(
        counterfactual_metrics=metrics,
        run_id="VGEN_DIM4_seed101",
        target_id="VGEN_DIM4",
        bottleneck_width=4,
        a0_accuracy=0.70,
        target_registration_sha256=_sha("target-registration"),
        selection_status="canonical_selectable",
    )
    assert target_metrics["same_consumer_counterfactual"]
    assert target_metrics["a0_accuracy"] == 0.70


def test_probe_consumer_training_completes_fixed_twelve_epochs(tmp_path):
    batch = _consumer_batch()
    lineage = {
        name: _sha(name)
        for name in (
            "a0_registration_sha256",
            "target_registration_sha256",
            "train_identity_sha256",
            "model_val_stop_split_sha256",
            "normalizer_sha256",
        )
    }
    registration = train_particle_view_consumer(
        model=_consumer(),
        train_loader=[batch],
        model_val_stop_loader=[batch],
        config=ParticleViewConsumerTrainConfig.for_role("Cview_probe"),
        output_dir=tmp_path,
        lineage=lineage,
    )
    validate_content_hash(registration)
    assert registration["role"] == "Cview_probe"
    assert registration["epochs_completed"] == 12
    assert registration["deployable_registration"] is False


def test_discovery_consumer_jointly_checkpoints_gview(tmp_path):
    batch = _consumer_batch()
    batch["offline_logits"] = torch.randn(3, 3)
    generator = nn.Linear(4, 4)

    def provide_view(current):
        generated = generator(current["true_view"])
        return {"view": generated, "raw_centered_view": generated}

    lineage = {
        name: _sha(name)
        for name in (
            "a0_registration_sha256",
            "target_registration_sha256",
            "train_identity_sha256",
            "model_val_stop_split_sha256",
        )
    }
    registration = train_particle_view_consumer(
        model=_consumer(),
        train_loader=[batch],
        model_val_stop_loader=[batch],
        config=ParticleViewConsumerTrainConfig.for_role("Cview_discovery"),
        output_dir=tmp_path,
        lineage=lineage,
        view_provider=provide_view,
        oracle_config=OracleObjectiveConfig(),
        joint_trainable_modules={"Gview": generator},
    )
    assert registration["joint_checkpointed_module_names"] == ["Gview"]
    checkpoint = torch.load(
        tmp_path / "best_model_val_stop.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert set(checkpoint["joint_model_state_dicts"]) == {"Gview"}


def test_target_selection_cli_writes_selection_and_non_gating_warnings(tmp_path):
    paths = []
    for candidate in (
        _candidate("best", 0.002, 0.003),
        _candidate("second", 0.001, 0.002),
        _candidate("canonical", -0.001, -0.002),
    ):
        artifact = build_target_metrics_artifact(candidate)
        path = tmp_path / f"{candidate.target_id}.json"
        path.write_text(json.dumps(artifact), encoding="utf-8")
        paths.extend(["--metrics", str(path)])
    output = tmp_path / "selection.json"
    warnings = tmp_path / "warnings.jsonl"
    command = [
        sys.executable,
        "scripts/select_particle_view_targets.py",
        *paths,
        "--canonical-target-id",
        "canonical",
        "--output",
        str(output),
        "--warnings-jsonl",
        str(warnings),
        "--source-commit",
        "test-commit",
    ]
    completed = subprocess.run(
        command, check=False, capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(output.read_text())["forwarded_target_ids"] == [
        "best",
        "second",
        "canonical",
    ]
    warning_rows = [
        json.loads(line) for line in warnings.read_text().splitlines()
    ]
    assert warning_rows and all(
        row["stops_execution"] is False for row in warning_rows
    )
