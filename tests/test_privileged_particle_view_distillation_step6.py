from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest
import torch
from torch import nn

from teacher_logit_reco.local_particle_residual_field.particle_view import (
    PVIEW0_LINEAGE_FIELDS,
    ROBUST_CONSUMER_LINEAGE_FIELDS,
    ROBUST_VIEW_PROBABILITIES,
    DeterministicRobustViewMixer,
    HierarchicalParticleViewPredictor,
    ParticleViewConsumer,
    ParticleViewConsumerConfig,
    ParticleViewPredictorConfig,
    ParticleViewPredictorOutput,
    ParticleViewWarmupConfig,
    RobustViewMixtureConfig,
    build_uncertainty_calibration_report,
    build_paired_consumer_metrics,
    collect_pview_predictions,
    evaluate_view_counterfactuals,
    fit_correlated_residual_sampler,
    heteroscedastic_huber_loss,
    load_registered_pview0,
    load_registered_robust_consumer,
    load_snapshot_dropout_prediction_bank,
    particle_view_representation_losses,
    publish_correlated_residual_sampler,
    run_residual_sampler_fit,
    train_pview0,
    train_robust_consumer,
    uncertainty_calibration_metrics,
    validate_content_hash,
    validate_pview0_registration,
    with_content_hash,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _predictor(view_dim: int = 2):
    config = ParticleViewPredictorConfig(
        view_dim=view_dim,
        architecture_id="P_LOCAL",
        width=16,
        num_heads=4,
        local_blocks=1,
        global_region_blocks=0,
        particle_global_blocks=0,
        decoder_blocks=0,
        final_refinement_blocks=0,
        hierarchy_sizes=(),
        use_hierarchy=False,
        decoder_mode="none",
        use_pair_bias=True,
        use_balance_loss=False,
        balance_weight=0.0,
        dropout=0.05,
    )
    return HierarchicalParticleViewPredictor(config)


def _predictor_batch(
    batch: int = 3, particles: int = 5, view_dim: int = 2
):
    generator = torch.Generator().manual_seed(66)
    features = torch.randn(batch, 17, particles, generator=generator)
    momentum = torch.randn(batch, 3, particles, generator=generator)
    energy = torch.sqrt(momentum.square().sum(dim=1, keepdim=True) + 1)
    vectors = torch.cat((momentum, energy), dim=1)
    mask = torch.ones(batch, 1, particles, dtype=torch.bool)
    mask[0, 0, -1] = False
    view = torch.randn(batch, particles, view_dim, generator=generator)
    valid = mask[:, 0]
    counts = valid.sum(dim=1, keepdim=True).clamp_min(1)
    mean = (view * valid[:, :, None]).sum(dim=1) / counts
    view = view - mean[:, None]
    view = torch.where(valid[:, :, None], view, torch.zeros_like(view))
    return {
        "features": features,
        "lorentz_vectors": vectors,
        "mask": mask,
        "true_view": view,
    }


def _pview_lineage():
    return {name: _sha(name) for name in PVIEW0_LINEAGE_FIELDS}


def test_production_residual_sampler_wrapper_publishes_metadata_only(tmp_path):
    true = np.zeros((3, 4, 2), dtype=np.float32)
    predicted = np.zeros_like(true)
    mask = np.ones((3, 4), dtype=bool)
    true[:, :, 0] = 0.2
    stop_true = true.copy()
    stop_true[:, :, 0] = 0.3
    run_residual_sampler_fit(
        train_true_view=true,
        train_prediction=predicted,
        train_mask=mask,
        model_val_stop_true_view=stop_true,
        model_val_stop_prediction=predicted,
        model_val_stop_mask=mask,
        train_identity_sha256=_sha("train"),
        model_val_stop_split_sha256=_sha("stop"),
        coordinate_binding_sha256=_sha("coordinate"),
        pview0_checkpoint_sha256=_sha("pview0"),
        snapshot_sha256=[
            _sha("snapshot-2"),
            _sha("snapshot-3"),
            _sha("snapshot-4"),
        ],
        output_dir=str(tmp_path),
    )
    registration = json.loads(
        (tmp_path / "correlated_residual_sampler.json").read_text()
    )
    publication = json.loads(
        (tmp_path / "residual_sampler_publication.json").read_text()
    )
    validate_content_hash(registration)
    validate_content_hash(publication)
    assert registration["residual_events_persisted_to_disk"] is False
    assert publication["ram_resident_training_resource"] is True
    assert not list(tmp_path.glob("*.npz"))


def test_step6_representation_and_uncertainty_losses_match_contract():
    prediction = torch.tensor(
        [[[0.0, 0.0], [0.2, -0.2]], [[1.0, 1.0], [0.0, 0.0]]]
    )
    target = torch.tensor(
        [[[0.1, 0.0], [0.0, 0.0]], [[0.5, 1.0], [0.0, 0.0]]]
    )
    mask = torch.tensor([[True, True], [True, False]])
    log_variance = torch.zeros_like(target)
    uncertainty = heteroscedastic_huber_loss(
        prediction, target, log_variance, mask
    )
    absolute = (prediction - target).abs()
    penalty = torch.where(
        absolute <= 0.1,
        absolute.square() / 0.2,
        absolute - 0.05,
    )
    expected = 0.5 * penalty[
        mask[:, :, None].expand_as(penalty)
    ].mean()
    assert uncertainty.item() == pytest.approx(expected.item())

    output = ParticleViewPredictorOutput(
        mean=prediction,
        log_variance=log_variance,
        trust=None,
        balance_loss=torch.tensor(0.03),
        hierarchy=None,
        local_embeddings=torch.empty(0),
        decoded_embeddings=torch.empty(0),
    )
    losses = particle_view_representation_losses(output, target, mask)
    assert losses["total"] == pytest.approx(
        losses["huber"]
        + 0.25 * losses["cosine"]
        + 0.15 * losses["relational"]
        + 0.05 * losses["uncertainty"]
        + 0.03
    )
    empty = torch.zeros(1, 2, 2)
    empty_mask = torch.zeros(1, 2, dtype=torch.bool)
    assert heteroscedastic_huber_loss(
        empty, empty, empty, empty_mask
    ).item() == 0
    metrics = uncertainty_calibration_metrics(
        prediction, target, log_variance, mask
    )
    assert metrics["valid_entries"] == 6
    assert 0 <= metrics["coverage_1sigma"] <= 1
    assert -1 <= metrics["absolute_error_variance_spearman"] <= 1


def test_pview0_fixed_four_epoch_registration_snapshots_and_reload(tmp_path):
    batch = _predictor_batch()
    model = _predictor()
    lineage = _pview_lineage()
    registration = train_pview0(
        model=model,
        train_loader=[batch],
        model_val_stop_loader=[batch],
        output_dir=tmp_path,
        lineage=lineage,
        config=ParticleViewWarmupConfig(amp=False),
    )
    validate_content_hash(registration)
    assert registration["epochs_completed"] == 4
    assert registration["snapshot_epochs"] == [2, 3, 4]
    assert len(registration["snapshot_sha256"]) == 3
    assert registration["labels_exposed"] is False
    validated = validate_pview0_registration(
        registration, root=tmp_path, expected_lineage=lineage
    )
    assert len(validated["snapshot_paths"]) == 3
    reloaded = load_registered_pview0(
        _predictor(),
        registration_path=tmp_path / "pview0_registration.json",
        expected_lineage=lineage,
    )
    assert not reloaded.training
    assert not any(parameter.requires_grad for parameter in reloaded.parameters())
    collected = collect_pview_predictions(model, [batch])
    assert collected["prediction"].dtype.str == "<f4"
    assert collected["mask"].dtype == np.bool_
    calibration = registration["model_val_stop"][
        "uncertainty_calibration"
    ]
    calibration_report = build_uncertainty_calibration_report(
        model_val_stop_metrics=calibration,
        model_val_select_metrics=calibration,
        pview0_checkpoint_sha256=registration["checkpoint_sha256"],
        coordinate_binding_sha256=lineage[
            "coordinate_binding_sha256"
        ],
        model_val_stop_split_sha256=lineage[
            "model_val_stop_split_sha256"
        ],
        model_val_select_split_sha256=_sha("model-val-select"),
    )
    validate_content_hash(calibration_report)
    with pytest.raises(ValueError, match="forbidden"):
        train_pview0(
            model=_predictor(),
            train_loader=[{**batch, "labels": torch.arange(3)}],
            model_val_stop_loader=[batch],
            output_dir=tmp_path / "bad",
            lineage=lineage,
            config=ParticleViewWarmupConfig(amp=False),
        )


def test_correlated_sampler_tail_inflation_warning_and_whole_events(tmp_path):
    train_true = np.array(
        [
            [[0.1, 0.2], [0.2, 0.4], [0.0, 0.0]],
            [[-0.2, 0.1], [-0.4, 0.2], [0.0, 0.0]],
            [[0.3, -0.1], [0.6, -0.2], [0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    train_mask = np.array(
        [[True, True, False], [True, True, False], [True, True, False]]
    )
    stop_true = train_true * 4
    sampler, warning = fit_correlated_residual_sampler(
        train_true_view=train_true,
        train_prediction=np.zeros_like(train_true),
        train_mask=train_mask,
        model_val_stop_true_view=stop_true,
        model_val_stop_prediction=np.zeros_like(stop_true),
        model_val_stop_mask=train_mask,
        train_identity_sha256=_sha("train"),
        model_val_stop_split_sha256=_sha("stop"),
        coordinate_binding_sha256=_sha("coordinate"),
        pview0_checkpoint_sha256=_sha("pview"),
        snapshot_sha256=[_sha(f"snapshot-{index}") for index in range(3)],
    )
    assert sampler.inflation_factor == 2.0
    assert warning["warning_code"] == "WARN_PVIEW_HELDOUT_TAIL_RATIO_ABOVE_2"
    assert warning["stops_execution"] is False
    publication = publish_correlated_residual_sampler(
        tmp_path, sampler=sampler, warning=warning
    )
    assert publication["residual_events_persisted_to_disk"] is False
    assert not (tmp_path / "residual_events.npz").exists()
    current_mask = torch.tensor(
        [[True, True, True], [True, True, False]]
    )
    sampled, indices = sampler.sample(current_mask)
    expected = torch.from_numpy(sampler.residuals[indices]) * 2
    source_mask = torch.from_numpy(sampler.masks[indices])
    expected = torch.where(
        (current_mask & source_mask)[:, :, None],
        expected,
        torch.zeros_like(expected),
    )
    assert torch.equal(sampled, expected)
    ratios = sampled[0, 1] / sampled[0, 0].clamp_min(1.0e-12)
    original_ratios = expected[0, 1] / expected[0, 0].clamp_min(1.0e-12)
    assert torch.equal(ratios, original_ratios)


def test_deterministic_robust_mixture_has_exact_cycle_counts():
    residual = np.arange(3 * 4 * 2, dtype=np.float32).reshape(3, 4, 2) / 100
    mask_np = np.ones((3, 4), dtype=bool)
    sampler, warning = fit_correlated_residual_sampler(
        train_true_view=residual,
        train_prediction=np.zeros_like(residual),
        train_mask=mask_np,
        model_val_stop_true_view=residual,
        model_val_stop_prediction=np.zeros_like(residual),
        model_val_stop_mask=mask_np,
        train_identity_sha256=_sha("train"),
        model_val_stop_split_sha256=_sha("stop"),
        coordinate_binding_sha256=_sha("coordinate"),
        pview0_checkpoint_sha256=_sha("pview"),
        snapshot_sha256=[_sha(f"snapshot-{index}") for index in range(3)],
    )
    assert warning is None
    first = DeterministicRobustViewMixer(sampler.clone())
    second = DeterministicRobustViewMixer(sampler.clone())
    true = torch.ones(2, 4, 2)
    mask = torch.ones(2, 1, 4, dtype=torch.bool)
    predictions = {
        "snapshot_epoch_2": torch.full_like(true, 0.2),
        "snapshot_epoch_3": torch.full_like(true, 0.3),
        "snapshot_epoch_4": torch.full_like(true, 0.4),
        "mc_dropout_epoch_4": torch.full_like(true, 0.5),
    }
    sources = []
    variants = []
    for _ in range(20):
        left = first.next(true_view=true, predictions=predictions, mask=mask)
        right = second.next(true_view=true, predictions=predictions, mask=mask)
        assert left.source == right.source
        assert left.prediction_variant == right.prediction_variant
        assert torch.equal(left.view, right.view)
        sources.append(left.source)
        if left.prediction_variant:
            variants.append(left.prediction_variant)
    assert {name: sources.count(name) / 20 for name in ROBUST_VIEW_PROBABILITIES} == (
        ROBUST_VIEW_PROBABILITIES
    )
    assert max(variants.count(name) for name in predictions) - min(
        variants.count(name) for name in predictions
    ) <= 1


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
        pair = (vectors[:, 0, :, None] - vectors[:, 0, None, :]).abs()
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
        values = values + pair.mean(dim=(1, 3)).transpose(0, 1)[..., None]
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


def _consumer():
    return ParticleViewConsumer(
        _FakeA0(),
        ParticleViewConsumerConfig(
            view_dim=2,
            hidden_dim=16,
            num_heads=4,
        ),
    )


def _consumer_batch_from_predictor(batch):
    return {
        **batch,
        "points": torch.zeros(
            batch["features"].shape[0], 2, batch["features"].shape[2]
        ),
        "labels": torch.tensor([0, 1, 2]),
    }


def test_snapshot_dropout_bank_robust_training_reload_and_paired_metrics(tmp_path):
    pview_root = tmp_path / "pview"
    batch = _predictor_batch()
    lineage = _pview_lineage()
    clean = _consumer()
    for parameter in clean.parameters():
        parameter.requires_grad_(False)
    clean_checkpoint = tmp_path / "clean_consumer.pt"
    torch.save({"model_state_dict": clean.state_dict()}, clean_checkpoint)
    lineage["clean_consumer_checkpoint_sha256"] = hashlib.sha256(
        clean_checkpoint.read_bytes()
    ).hexdigest()
    pview_registration = train_pview0(
        model=_predictor(),
        train_loader=[batch],
        model_val_stop_loader=[batch],
        output_dir=pview_root,
        lineage=lineage,
        config=ParticleViewWarmupConfig(amp=False),
    )
    bank = load_snapshot_dropout_prediction_bank(
        _predictor(),
        pview0_registration_path=pview_root / "pview0_registration.json",
        expected_lineage=lineage,
    )
    bank.reset()
    first = bank.predict(
        batch["features"], batch["lorentz_vectors"], batch["mask"]
    )
    bank.reset()
    second = bank.predict(
        batch["features"], batch["lorentz_vectors"], batch["mask"]
    )
    assert tuple(first) == (
        "snapshot_epoch_2",
        "snapshot_epoch_3",
        "snapshot_epoch_4",
        "mc_dropout_epoch_4",
    )
    assert all(torch.equal(first[name], second[name]) for name in first)
    assert not torch.equal(
        first["mc_dropout_epoch_4"], first["snapshot_epoch_4"]
    )

    true = batch["true_view"].numpy()
    final_prediction = first["snapshot_epoch_4"].numpy()
    mask = batch["mask"][:, 0].numpy()
    sampler, warning = fit_correlated_residual_sampler(
        train_true_view=true,
        train_prediction=final_prediction,
        train_mask=mask,
        model_val_stop_true_view=true,
        model_val_stop_prediction=final_prediction,
        model_val_stop_mask=mask,
        train_identity_sha256=lineage["train_identity_sha256"],
        model_val_stop_split_sha256=lineage[
            "model_val_stop_split_sha256"
        ],
        coordinate_binding_sha256=lineage[
            "coordinate_binding_sha256"
        ],
        pview0_checkpoint_sha256=pview_registration[
            "checkpoint_sha256"
        ],
        snapshot_sha256=pview_registration["snapshot_sha256"],
    )
    assert warning is None
    consumer_batch = _consumer_batch_from_predictor(batch)
    robust_lineage = {
        name: _sha(name) for name in ROBUST_CONSUMER_LINEAGE_FIELDS
    }
    robust_lineage.update(
        {
            "source_manifest_sha256": lineage[
                "source_manifest_sha256"
            ],
            "train_identity_sha256": lineage["train_identity_sha256"],
            "model_val_stop_split_sha256": lineage[
                "model_val_stop_split_sha256"
            ],
            "target_selection_sha256": lineage[
                "target_selection_sha256"
            ],
            "coordinate_binding_sha256": lineage[
                "coordinate_binding_sha256"
            ],
            "selected_view_publication_sha256": lineage[
                "selected_view_publication_sha256"
            ],
            "train_view_cache_manifest_sha256": lineage[
                "train_view_cache_manifest_sha256"
            ],
            "model_val_stop_view_cache_manifest_sha256": lineage[
                "model_val_stop_view_cache_manifest_sha256"
            ],
            "pview0_registration_sha256": pview_registration[
                "content_hash"
            ],
            "pview0_checkpoint_sha256": pview_registration[
                "checkpoint_sha256"
            ],
            "residual_sampler_registration_sha256": sampler.registration[
                "content_hash"
            ],
        }
    )
    robust_lineage["clean_consumer_checkpoint_sha256"] = hashlib.sha256(
        clean_checkpoint.read_bytes()
    ).hexdigest()
    robust_lineage["clean_consumer_registration_sha256"] = lineage[
        "clean_consumer_registration_sha256"
    ]
    robust, registration = train_robust_consumer(
        clean_consumer=clean,
        train_loader=[consumer_batch],
        model_val_stop_loader=[consumer_batch],
        prediction_bank=bank,
        sampler=sampler,
        output_dir=tmp_path / "robust",
        lineage=robust_lineage,
        clean_consumer_checkpoint_path=clean_checkpoint,
    )
    validate_content_hash(registration)
    assert registration["pview0_frozen"]
    assert registration["snapshot_sha256"] == pview_registration[
        "snapshot_sha256"
    ]
    assert not any(parameter.requires_grad for parameter in robust.parameters())

    reloaded = load_registered_robust_consumer(
        _consumer(),
        registration_path=tmp_path
        / "robust"
        / "robust_consumer_registration.json",
        expected_lineage=robust_lineage,
        expected_snapshot_sha256=pview_registration["snapshot_sha256"],
        expected_sampler_sha256=sampler.registration["content_hash"],
        expected_pview_architecture_config_sha256=bank.architecture_config_sha256,
    )
    assert not reloaded.training
    with pytest.raises(ValueError, match="sampler lineage"):
        load_registered_robust_consumer(
            _consumer(),
            registration_path=tmp_path
            / "robust"
            / "robust_consumer_registration.json",
            expected_lineage=robust_lineage,
            expected_snapshot_sha256=pview_registration[
                "snapshot_sha256"
            ],
            expected_sampler_sha256=_sha("stale-sampler"),
            expected_pview_architecture_config_sha256=bank.architecture_config_sha256,
        )
    stale = deepcopy(registration)
    stale.pop("content_hash")
    stale["mixture_config"]["mixture_seed"] = 999
    stale = with_content_hash(stale)
    stale_path = tmp_path / "robust" / "stale_registration.json"
    stale_path.write_text(json.dumps(stale), encoding="utf-8")
    with pytest.raises(ValueError, match="mixture contract"):
        load_registered_robust_consumer(
            _consumer(),
            registration_path=stale_path,
            expected_lineage=robust_lineage,
            expected_snapshot_sha256=pview_registration[
                "snapshot_sha256"
            ],
            expected_sampler_sha256=sampler.registration["content_hash"],
            expected_pview_architecture_config_sha256=bank.architecture_config_sha256,
        )

    counterfactual_batch = {
        **consumer_batch,
        "predicted_view": first["snapshot_epoch_4"],
    }
    clean_metrics = evaluate_view_counterfactuals(
        clean, [counterfactual_batch]
    )
    robust_metrics = evaluate_view_counterfactuals(
        robust, [counterfactual_batch]
    )
    paired = build_paired_consumer_metrics(
        clean_metrics=clean_metrics,
        robust_metrics=robust_metrics,
        clean_consumer_checkpoint_sha256=robust_lineage[
            "clean_consumer_checkpoint_sha256"
        ],
        robust_consumer_checkpoint_sha256=registration[
            "checkpoint_sha256"
        ],
        pview0_checkpoint_sha256=pview_registration[
            "checkpoint_sha256"
        ],
        coordinate_binding_sha256=lineage[
            "coordinate_binding_sha256"
        ],
        split_sha256=_sha("model-val-select"),
    )
    validate_content_hash(paired)
    assert paired["paired_consumer_requirement_satisfied"]
