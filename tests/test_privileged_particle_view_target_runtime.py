from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from teacher_logit_reco.local_particle_residual_field.particle_view import (
    CANONICAL_TARGET_DISCOVERY_RUN_ID,
    CONSUMER_SCREEN_IDS,
    CandidateViewSet,
    IndexedCandidateViewProvider,
    MatchingFreeParticleViewGenerator,
    ParticleViewGeneratorConfig,
    RecoveryProbeConfig,
    FixedCapacityRecoveryProbe,
    StagedContextualMemory,
    StagedDiscoveryViewProvider,
    TARGET_SCREEN_IDS,
    TargetCandidateMetrics,
    TwoTeacherTokenMixture,
    build_target_metrics_artifact,
    build_tap_stage_reservation,
    build_target_screen_recipe,
    canonical_sha256,
    fit_particle_view_normalizer,
    lorentz_vectors_to_particle_geometry,
    particle_view_generator_config_from_payload,
    predict_recovery_probe_views,
    run_recoverability_codesign,
    run_target_discovery_operation,
    run_target_selection,
    load_hashed_json,
    stage_teacher_tap_float16,
    with_content_hash,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _geometry(
    pt: torch.Tensor,
    eta: torch.Tensor,
    phi: torch.Tensor,
    mass: torch.Tensor,
) -> torch.Tensor:
    return torch.stack((pt, eta, phi, mass), dim=-1)


def test_target_runtime_compiles_every_declared_screen_deterministically():
    recipes = {
        run_id: build_target_screen_recipe(run_id)
        for run_id in TARGET_SCREEN_IDS
    }
    assert len(recipes) == 36
    assert len({recipe.content_hash for recipe in recipes.values()}) == 36
    canonical = recipes[CANONICAL_TARGET_DISCOVERY_RUN_ID]
    assert canonical.query_tap_choice == "penultimate"
    assert canonical.memory_tap_choice == "penultimate"
    assert canonical.generator_config.num_cross_attention_blocks == 2
    assert canonical.generator_config.bottleneck_width == 4
    assert canonical.selection_status == "canonical_selectable"

    assert recipes["VGEN_QUERY_RAW"].generator_config.query_dim == 17
    assert recipes["VGEN_TAP_MIX3"].memory_tap_choice == "mix_last3"
    assert recipes["VGEN_XATTN4"].generator_config.num_cross_attention_blocks == 4
    assert recipes["VGEN_NO_PAIR"].generator_config.use_pair_bias is False
    assert recipes["VGEN_LOCAL02"].generator_config.hard_local_radius == 0.2
    assert recipes["VGEN_LOCAL04"].generator_config.hard_local_radius == 0.4
    assert recipes["VGEN_UNCENTERED"].generator_config.center_output is False
    assert recipes["VGEN_DIM8"].generator_config.bottleneck_width == 8
    assert recipes["VGEN_KD100"].oracle_objective.offline_kd_weight == 1.0
    assert recipes["VGEN_NO_RATE"].oracle_objective.rate_budget_enabled is False
    assert recipes["VGEN_TEACHER_LARGE"].generator_config.memory_dim == 192
    assert recipes["VGEN_TEACHER_MIX2"].generator_config.memory_dim == 160
    assert recipes["VGEN_MEMORY_HLT"].generator_config.memory_source == "hlt"
    assert (
        recipes["VGEN_MEMORY_HLT_SELFMASK"].generator_config.self_mask_same_particle
        is True
    )
    assert (
        recipes["VGEN_MEMORY_HLT"].oracle_objective.offline_kd_weight == 0.0
    )


def test_generator_config_round_trip_and_local_radius_masks_only_far_keys():
    config = ParticleViewGeneratorConfig(
        query_dim=3,
        memory_dim=3,
        width=8,
        num_heads=2,
        num_cross_attention_blocks=1,
        bottleneck_width=2,
        hard_local_radius=0.2,
    )
    assert particle_view_generator_config_from_payload(
        config.to_payload()
    ) == config
    with pytest.raises(ValueError, match="0.2/0.4"):
        ParticleViewGeneratorConfig(
            query_dim=3,
            memory_dim=3,
            hard_local_radius=0.3,
        )

    query = torch.randn(1, 1, 3)
    memory = torch.randn(1, 2, 3)
    query_geometry = _geometry(
        torch.tensor([[10.0]]),
        torch.tensor([[0.0]]),
        torch.tensor([[0.0]]),
        torch.tensor([[0.0]]),
    )
    memory_geometry = _geometry(
        torch.tensor([[8.0, 7.0]]),
        torch.tensor([[0.1, 1.0]]),
        torch.tensor([[0.0, 0.0]]),
        torch.zeros(1, 2),
    )
    output = MatchingFreeParticleViewGenerator(config).eval()(
        query,
        memory,
        query_geometry=query_geometry,
        memory_geometry=memory_geometry,
        query_mask=torch.ones(1, 1, dtype=torch.bool),
        memory_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    # The final attention key is the learned null token.
    assert torch.all(output.attention[..., 0] > 0)
    assert torch.count_nonzero(output.attention[..., 1]) == 0
    assert torch.all(output.attention[..., 2] > 0)


def test_lorentz_geometry_is_physical_masked_and_finite():
    vectors = torch.tensor(
        [
            [
                [3.0, 0.0],
                [4.0, 0.0],
                [0.0, 0.0],
                [5.0, 0.0],
            ]
        ]
    )
    mask = torch.tensor([[[True, False]]])
    geometry = lorentz_vectors_to_particle_geometry(vectors, mask)
    assert geometry.shape == (1, 2, 4)
    assert geometry[0, 0].tolist() == pytest.approx([5.0, 0.0, 0.9272952, 0.0])
    assert torch.count_nonzero(geometry[0, 1]) == 0
    assert torch.isfinite(geometry).all()


class _FrozenQuery:
    def __call__(self, points, features, lorentz_vectors, mask):
        del points, lorentz_vectors
        tokens = features.transpose(1, 2).detach()
        return SimpleNamespace(
            single_layer_tokens=tokens,
            particle_tokens=tokens[:, None],
            particle_mask=mask[:, 0],
            logits=torch.zeros(features.shape[0], 3, device=features.device),
        )


def test_staged_provider_reorders_parent_rows_and_injects_offline_logits():
    tokens = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    mask = torch.ones(2, 2, dtype=torch.bool)
    reservation = build_tap_stage_reservation(
        source_role="offline_teacher",
        source_manifest_sha256=_sha("source"),
        logical_split_sha256=_sha("split"),
        ordered_identity_sha256=_sha("identity"),
        teacher_checkpoint_sha256=_sha("teacher"),
        tap_spec_sha256=_sha("tap"),
        jets=2,
        max_particles=2,
        token_width=3,
        identity_columns=2,
    )
    staged = stage_teacher_tap_float16(
        tokens,
        mask,
        torch.tensor([[8, 0], [4, 1]], dtype=torch.int64),
        reservation=reservation,
    )
    memory = StagedContextualMemory(
        staged_tap=staged,
        logits=torch.tensor([[8.0, 0.0, 0.0], [4.0, 0.0, 0.0]]),
        parent_indices=torch.tensor([8, 4], dtype=torch.int64),
    )
    generator = MatchingFreeParticleViewGenerator(
        ParticleViewGeneratorConfig(
            query_dim=3,
            memory_dim=3,
            width=8,
            num_heads=2,
            num_cross_attention_blocks=1,
            bottleneck_width=2,
        )
    )
    provider = StagedDiscoveryViewProvider(
        generator=generator,
        query_teacher=_FrozenQuery(),
        staged_memory=memory,
        query_tap_choice="penultimate",
    )
    momentum = torch.tensor(
        [
            [[1.0, 0.5], [0.0, 0.0], [0.0, 0.0]],
            [[1.5, 0.7], [0.0, 0.0], [0.0, 0.0]],
        ]
    )
    energy = torch.sqrt(momentum.square().sum(dim=1, keepdim=True) + 1.0)
    lorentz = torch.cat((momentum, energy), dim=1)
    batch = {
        "points": torch.zeros(2, 2, 2),
        "features": torch.randn(2, 3, 2),
        "lorentz_vectors": lorentz,
        "mask": torch.ones(2, 1, 2, dtype=torch.bool),
        "offline_lorentz_vectors": lorentz.clone(),
        "offline_mask": torch.ones(2, 1, 2, dtype=torch.bool),
        "parent_indices": torch.tensor([4, 8]),
    }
    provided = provider(batch)
    assert provided["view"].shape == (2, 2, 2)
    assert provided["raw_centered_view"].shape == (2, 2, 2)
    assert batch["offline_logits"][:, 0].tolist() == [4.0, 8.0]
    provided["view"].square().sum().backward()
    assert generator.memory_projection.weight.grad is not None


def test_candidate_view_set_normalizes_train_only_and_reorders_by_parent():
    view_set = CandidateViewSet(
        views=torch.tensor(
            [
                [[1.0, -1.0], [3.0, 1.0], [0.0, 0.0]],
                [[5.0, 3.0], [7.0, 5.0], [9.0, 7.0]],
            ],
            dtype=torch.float32,
        ),
        mask=torch.tensor(
            [[True, True, False], [True, True, True]],
            dtype=torch.bool,
        ),
        parent_indices=torch.tensor([8, 4], dtype=torch.int64),
        logical_split_sha256=_sha("train-split"),
        ordered_identity_sha256=_sha("train-order"),
    )
    normalizer = fit_particle_view_normalizer(
        view_set.views,
        view_set.mask,
        train_split_sha256=_sha("train-split"),
        generator_checkpoint_sha256=_sha("generator"),
    )
    normalized = view_set.normalized(normalizer)
    assert normalized.views[normalized.mask].mean(dim=0).tolist() == pytest.approx(
        [0.0, 0.0], abs=1.0e-6
    )
    assert torch.count_nonzero(normalized.views[~normalized.mask]) == 0
    provider = IndexedCandidateViewProvider(normalized)
    provided = provider(
        {
            "features": torch.zeros(2, 17, 3),
            "mask": torch.tensor(
                [
                    [[True, True, True]],
                    [[True, True, False]],
                ]
            ),
            "parent_indices": torch.tensor([4, 8]),
        }
    )
    assert torch.equal(provided["view"][0], normalized.views[1])
    assert torch.equal(provided["view"][1], normalized.views[0])
    assert provided["raw_centered_view"] is provided["view"]


def test_recovery_prediction_loader_is_label_free_and_preserves_order(
    monkeypatch,
):
    import teacher_logit_reco.local_particle_residual_field.particle_view.target_runtime as runtime

    truth = CandidateViewSet(
        views=torch.randn(2, 3, 1),
        mask=torch.ones(2, 3, dtype=torch.bool),
        parent_indices=torch.tensor([5, 9], dtype=torch.int64),
        logical_split_sha256=_sha("select-split"),
        ordered_identity_sha256=_sha("select-order"),
    )
    batch = {
        "features": torch.randn(2, 17, 3),
        "mask": truth.mask.clone(),
        "true_view": truth.views.clone(),
    }
    observed = {}

    def loader(*args, **kwargs):
        observed["mode"] = kwargs["mode"]
        observed["true_views"] = kwargs["true_views"]
        return [batch]

    monkeypatch.setattr(runtime, "make_logical_data_loader", loader)
    config = RecoveryProbeConfig(view_dim=1)
    model = FixedCapacityRecoveryProbe(config).eval()
    aligned = SimpleNamespace(
        logical_split_sha256=truth.logical_split_sha256,
    )
    prediction = predict_recovery_probe_views(
        model=model,
        aligned=aligned,
        true_views=truth,
        device="cpu",
        num_workers=0,
    )
    assert observed["mode"] == "recovery_probe"
    assert "labels" not in batch
    assert prediction.parent_indices.tolist() == [5, 9]
    assert torch.equal(prediction.mask, truth.mask)

    batch["labels"] = torch.tensor([0, 1])
    with pytest.raises(ValueError, match="unexpected fields"):
        predict_recovery_probe_views(
            model=model,
            aligned=aligned,
            true_views=truth,
            device="cpu",
            num_workers=0,
        )


def test_two_teacher_mixture_is_masked_trainable_and_dimension_locked():
    mixture = TwoTeacherTokenMixture()
    base = torch.randn(2, 4, 128, requires_grad=True)
    large = torch.randn(2, 4, 192, requires_grad=True)
    mask = torch.tensor(
        [[True, True, False, False], [True, True, True, False]]
    )
    mixed = mixture(base, large, mask)
    assert mixed.shape == (2, 4, 160)
    assert torch.count_nonzero(mixed[~mask]) == 0
    mixed.square().sum().backward()
    assert mixture.source_logits.grad is not None
    assert mixture.base_projection.weight.grad is not None
    with pytest.raises(ValueError, match="dimensions"):
        TwoTeacherTokenMixture(output_dim=128)


def test_full_target_selection_accepts_unavailable_diagnostic_and_never_gates(
    tmp_path,
):
    unavailable_path = tmp_path / "unavailable.json"
    run_target_discovery_operation(
        unavailable={
            "run_id": "VGEN_TEACHER_EXISTING",
            "seed": 101,
            "reason": "fixture_unavailable",
            "source_registration_sha256": _sha("unavailable-source"),
            "output_path": str(unavailable_path),
        }
    )
    unavailable = load_hashed_json(unavailable_path)
    candidates = []
    for index, run_id in enumerate(TARGET_SCREEN_IDS):
        if run_id == "VGEN_TEACHER_EXISTING":
            continue
        recipe = build_target_screen_recipe(run_id)
        predicted_gain = (
            0.010
            if run_id == "VGEN_TAP_RAW"
            else 0.009
            if run_id == "VGEN_TAP_MID"
            else -0.001 - index * 1.0e-5
        )
        candidates.append(
            build_target_metrics_artifact(
                TargetCandidateMetrics(
                    run_id=run_id,
                    target_id=run_id,
                    bottleneck_width=(
                        recipe.generator_config.bottleneck_width
                    ),
                    predicted_view_gain=predicted_gain,
                    oracle_gain=-0.002,
                    predicted_view_cross_entropy=0.7 + index * 1.0e-4,
                    zero_view_accuracy=0.70,
                    predicted_view_accuracy=0.70 + predicted_gain,
                    oracle_accuracy=0.698,
                    a0_accuracy=0.70,
                    target_registration_sha256=_sha(run_id),
                    selection_status=recipe.selection_status,
                )
            )
        )
    selection_path = tmp_path / "selection.json"
    warnings_path = tmp_path / "warnings.jsonl"
    result_path = tmp_path / "result.json"
    consumer_metrics = []
    for index, consumer_id in enumerate(CONSUMER_SCREEN_IDS):
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
        consumer_metrics.append(
            with_content_hash(
                {
                    "contract": "particle_view_consumer_screen_metrics_v1",
                    "consumer_id": consumer_id,
                    "run_id": f"SCREEN_{consumer_id}",
                    "recipe": recipe,
                    "recipe_sha256": canonical_sha256(recipe),
                    "consumer_registration_sha256": _sha(
                        f"consumer-{index}"
                    ),
                    "checkpoint_sha256": _sha(
                        f"consumer-checkpoint-{index}"
                    ),
                    "model_val_select": {
                        "accuracy": 0.7,
                        "cross_entropy": 0.8 + index * 1.0e-4,
                        "examples": 100.0,
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
    run_target_selection(
        candidates=candidates,
        consumer_metrics=consumer_metrics,
        unavailable_targets=[unavailable],
        source_commit="abc123",
        output_path=str(selection_path),
        consumer_output_path=str(tmp_path / "consumer_selection.json"),
        warnings_path=str(warnings_path),
        result_path=str(result_path),
    )
    selection = load_hashed_json(selection_path)
    assert selection["forwarded_target_ids"][:2] == [
        "VGEN_TAP_RAW",
        "VGEN_TAP_MID",
    ]
    assert "VGEN_TAP_PENULT" in selection["forwarded_target_ids"]
    assert "VGEN_MEMORY_HLT" in selection["forwarded_target_ids"]
    assert "VGEN_TEACHER_EXISTING" not in selection[
        "forwarded_target_ids"
    ]
    assert selection["quality_threshold_used_as_gate"] is False
    result = load_hashed_json(result_path)
    assert result["warnings_stop_execution"] is False
    assert set(result["unavailable_target_sha256_by_run"]) == {
        "VGEN_TEACHER_EXISTING"
    }


class _TinyCodesignConfig:
    def __init__(self, view_dim: int, seed: int) -> None:
        self.view_dim = view_dim
        self.rich_dim = 160
        self.cycles = 12
        self.probe_steps_per_cycle = 1
        self.projection_steps_per_cycle = 1
        self.probe_learning_rate = 3.0e-4
        self.projection_learning_rate = 1.0e-4
        self.consumer_learning_rate = 3.0e-5
        self.probe_weight_decay = 1.0e-4
        self.projection_weight_decay = 1.0e-5
        self.consumer_weight_decay = 1.0e-4
        self.batch_size = 128
        self.gradient_clip = 1.0
        self.seed = seed

    def to_payload(self):
        return {
            "contract": "tiny_codesign_test_v1",
            "view_dim": self.view_dim,
            "rich_dim": self.rich_dim,
            "cycles": self.cycles,
            "probe_steps_per_cycle": self.probe_steps_per_cycle,
            "projection_steps_per_cycle": self.projection_steps_per_cycle,
            "seed": self.seed,
            "performance_early_termination": False,
        }


class _TinyCodesignProvider:
    def __init__(self, generator):
        self.generator = generator
        self.codesign_projection = None

    def __call__(self, batch):
        mask = batch["mask"][:, 0]
        rich = torch.nn.functional.pad(
            batch["features"].transpose(1, 2),
            (0, 160 - batch["features"].shape[1]),
        )
        deterministic = self.codesign_projection(rich, mask)
        view = (
            self.generator._augment(
                deterministic, mask, generator=None
            )
            if self.codesign_projection.training
            else deterministic
        )
        batch["offline_logits"] = torch.zeros(
            batch["features"].shape[0],
            3,
            device=batch["features"].device,
        )
        return {
            "view": view,
            "raw_centered_view": deterministic,
        }


class _TinyConsumerConfig:
    def to_payload(self):
        return {"contract": "tiny_consumer_test_v1"}


class _TinyCodesignConsumer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _TinyConsumerConfig()
        self.classifier = torch.nn.Linear(21, 3)

    def forward(
        self,
        points,
        features,
        lorentz_vectors,
        mask,
        view,
        *,
        augment_clean_view,
    ):
        del points, lorentz_vectors, augment_clean_view
        valid = mask[:, 0, :, None].to(features.dtype)
        joined = torch.cat((features.transpose(1, 2), view), dim=-1)
        pooled = (joined * valid).sum(dim=1) / valid.sum(dim=1)
        logits = self.classifier(pooled)
        return SimpleNamespace(
            logits=logits,
            trust_loss=logits.square().mean() * 0.0,
        )


def test_codesign_runtime_runs_all_cycles_freezes_rview_and_selects_projection(
    tmp_path,
    monkeypatch,
):
    import teacher_logit_reco.local_particle_residual_field.particle_view.target_runtime as runtime

    monkeypatch.setattr(
        runtime,
        "RecoverabilityCoDesignConfig",
        _TinyCodesignConfig,
    )
    generator = MatchingFreeParticleViewGenerator(
        ParticleViewGeneratorConfig(
            query_dim=17,
            memory_dim=17,
            width=160,
            num_heads=8,
            num_cross_attention_blocks=1,
            bottleneck_width=4,
        )
    )
    train_provider = _TinyCodesignProvider(generator)
    stop_provider = _TinyCodesignProvider(generator)
    momentum = torch.randn(2, 3, 4)
    energy = (
        momentum.square().sum(dim=1, keepdim=True) + 1.0
    ).sqrt()
    batch = {
        "points": torch.zeros(2, 2, 4),
        "features": torch.randn(2, 17, 4),
        "lorentz_vectors": torch.cat((momentum, energy), dim=1),
        "mask": torch.ones(2, 1, 4, dtype=torch.bool),
        "labels": torch.tensor([0, 1]),
    }
    result = run_recoverability_codesign(
        generator=generator,
        consumer=_TinyCodesignConsumer(),
        train_provider=train_provider,
        stop_provider=stop_provider,
        train_loader=[batch],
        stop_loader=[batch],
        oracle_config=build_target_screen_recipe(
            "VGEN_RECODESIGN"
        ).oracle_objective,
        provisional_generator_checkpoint_sha256=_sha("provisional-g"),
        provisional_consumer_registration_sha256=_sha("provisional-c"),
        output_dir=tmp_path,
        seed=101,
        device=torch.device("cpu"),
    )
    ledger = result["ledger"]
    assert len(ledger["cycles"]) == 12
    assert ledger["performance_early_termination"] is False
    assert 1 <= result["selected_cycle"] <= 12
    assert all(
        not parameter.requires_grad
        for parameter in generator.parameters()
    )
    assert train_provider.codesign_projection is result["projection"]
    assert Path(
        result["generator_checkpoint_path"]
    ).is_file()
    assert not (
        tmp_path / "recoverability_codesign" / "cycle_checkpoints"
    ).exists()
