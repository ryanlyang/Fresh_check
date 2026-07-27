from __future__ import annotations

from dataclasses import replace

import pytest

from jetclass_fresh.hlt_baseline import require_torch
from teacher_logit_reco.local_particle_residual_field.particle_view import (
    FrozenContextualParticleTeacher,
    FrozenTokenLayerMixture,
    ParticleTokenTapSpec,
    ParticleViewTeacherTrainConfig,
    audit_frozen_teacher,
    build_predeclared_direct_control_grid,
    build_existing_teacher_source_registration,
    build_teacher_checkpoint_payload,
    build_teacher_recipe,
    build_teacher_registration,
    build_token_tap_registration,
    build_unified_split_manifest,
    miniature_parent_manifest,
    miniature_split_config,
    reload_registered_teacher,
    select_teacher_checkpoint,
    teacher_learning_rate,
    token_tap_block_indices,
    train_particle_view_teacher,
    validate_matched_teacher_recipes,
    validate_token_tap_registration,
)


torch = require_torch()


class _FakeParticleBlock(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(width, width)
        self.activation = torch.nn.GELU()

    def forward(self, values):
        return values + self.activation(self.linear(values))


class _FakeMod(torch.nn.Module):
    def __init__(self, *, width: int = 128, layers: int = 8) -> None:
        super().__init__()
        self.embed = torch.nn.Sequential(
            torch.nn.Linear(17, width), torch.nn.GELU()
        )
        self.blocks = torch.nn.ModuleList(
            [_FakeParticleBlock(width) for _ in range(layers)]
        )
        self.head = torch.nn.Linear(width, 10)

    def forward(self, features, mask):
        values = self.embed(features.transpose(1, 2))
        for block in self.blocks:
            values = block(values)
        valid = mask[:, 0].unsqueeze(-1)
        pooled = (values * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        return self.head(pooled)


class _FakeTeacher(torch.nn.Module):
    def __init__(self, *, width: int = 128, layers: int = 8) -> None:
        super().__init__()
        self.mod = _FakeMod(width=width, layers=layers)

    def forward(self, points, features, lorentz_vectors, mask):
        del points, lorentz_vectors
        return self.mod(features, mask)


class _FakeTrimmedMod(_FakeMod):
    """Mimic Weaver's batch-local removal of fully padded suffix columns."""

    def forward(self, features, mask):
        particles = int(mask[:, 0].sum(dim=1).max().item())
        values = self.embed(features[:, :, :particles].transpose(1, 2))
        for block in self.blocks:
            values = block(values)
        valid = mask[:, 0, :particles].unsqueeze(-1)
        pooled = (values * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        return self.head(pooled)


class _FakeTrimmedTeacher(_FakeTeacher):
    def __init__(self, *, width: int = 128, layers: int = 8) -> None:
        torch.nn.Module.__init__(self)
        self.mod = _FakeTrimmedMod(width=width, layers=layers)


class _FakeOverTrimmedMod(_FakeMod):
    def forward(self, features, mask):
        particles = int(mask[:, 0].sum(dim=1).max().item()) - 1
        values = self.embed(features[:, :, :particles].transpose(1, 2))
        for block in self.blocks:
            values = block(values)
        valid = mask[:, 0, :particles].unsqueeze(-1)
        pooled = (values * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        return self.head(pooled)


class _FakeOverTrimmedTeacher(_FakeTeacher):
    def __init__(self, *, width: int = 128, layers: int = 8) -> None:
        torch.nn.Module.__init__(self)
        self.mod = _FakeOverTrimmedMod(width=width, layers=layers)


def _manifest():
    parent = miniature_parent_manifest(rows_per_class=4)
    config = miniature_split_config(rows_per_class=4)
    return build_unified_split_manifest(parent, config=config)


def _recipe(role="A0_view", architecture="base", seed=101):
    return build_teacher_recipe(
        role=role,
        architecture=architecture,
        seed=seed,
        unified_split_manifest=_manifest(),
        preprocessing_sha256="1" * 64,
        source_sha256=("2" if role == "A0_view" else "3") * 64,
        initialization_implementation_sha256="4" * 64,
        library_versions_sha256="5" * 64,
    )


def _batch(batch=3, particles=7):
    torch.manual_seed(17)
    features = torch.randn(batch, 17, particles)
    vectors = torch.randn(batch, 4, particles)
    points = torch.randn(batch, 2, particles)
    mask = torch.ones(batch, 1, particles, dtype=torch.bool)
    mask[0, 0, -2:] = False
    labels = torch.arange(batch) % 10
    return {
        "points": points,
        "features": features,
        "lorentz_vectors": vectors,
        "mask": mask,
        "labels": labels,
    }


def test_locked_base_and_large_recipes_and_matched_identity():
    hlt = _recipe("A0_view", "base")
    offline = _recipe("Toff_view", "base")
    common_hash = validate_matched_teacher_recipes(hlt, offline)
    assert len(common_hash) == 64
    base = hlt.to_payload()
    assert base["architecture"]["embed_dims"] == [128, 512, 128]
    assert base["architecture"]["num_layers"] == 8
    assert base["physical_batch_size"] == 128
    assert base["gradient_accumulation_steps"] == 1
    assert base["schedule"]["warmup_updates"] == 2_000
    assert base["schedule"]["minimum_learning_rate"] == 3.0e-6
    large = _recipe("Toff_view", "large").to_payload()
    assert large["architecture"]["embed_dims"] == [192, 768, 192]
    assert large["architecture"]["num_layers"] == 12
    assert large["architecture"]["num_cls_layers"] == 3
    assert large["physical_batch_size"] == 64
    assert large["gradient_accumulation_steps"] == 2


def test_recipe_drift_and_wrong_source_fail_closed():
    with pytest.raises(ValueError, match="must use"):
        replace(_recipe(), particle_source="offline").to_payload()
    with pytest.raises(ValueError, match="locked base"):
        _recipe("A0_view", "large").to_payload()


def test_contextual_tap_shape_mask_freeze_and_permutation():
    teacher = _FakeTeacher()
    tap = FrozenContextualParticleTeacher(
        teacher,
        ParticleTokenTapSpec(
            particle_source="fixed_hlt",
            architecture="base",
            tap_choice="penultimate",
        ),
    )
    batch = _batch()
    output = tap(
        batch["points"],
        batch["features"],
        batch["lorentz_vectors"],
        batch["mask"],
    )
    assert output.particle_tokens.shape == (3, 1, 7, 128)
    assert output.logits.shape == (3, 10)
    assert torch.count_nonzero(output.particle_tokens[0, :, -2:]) == 0
    assert output.particle_tokens.requires_grad is False
    assert output.logits.requires_grad is False
    assert audit_frozen_teacher(teacher)["trainable_parameter_count"] == 0
    tap.train(True)
    assert not tap.training and not teacher.training

    permutation = torch.tensor([3, 0, 6, 2, 1, 5, 4])
    permuted = tap(
        batch["points"][:, :, permutation],
        batch["features"][:, :, permutation],
        batch["lorentz_vectors"][:, :, permutation],
        batch["mask"][:, :, permutation],
    )
    assert torch.allclose(output.logits, permuted.logits, atol=1e-6)
    assert torch.allclose(
        output.particle_tokens[:, :, permutation],
        permuted.particle_tokens,
        atol=1e-6,
    )


def test_all_tap_locations_and_learned_last_three_mixture():
    assert token_tap_block_indices(8, "middle") == (3,)
    assert token_tap_block_indices(8, "penultimate") == (6,)
    assert token_tap_block_indices(8, "final") == (7,)
    assert token_tap_block_indices(8, "mix_last3") == (5, 6, 7)
    batch = _batch()
    teacher = _FakeTeacher()
    tapped = FrozenContextualParticleTeacher(
        teacher,
        ParticleTokenTapSpec(
            particle_source="offline",
            architecture="base",
            tap_choice="mix_last3",
        ),
    )(
        batch["points"],
        batch["features"],
        batch["lorentz_vectors"],
        batch["mask"],
    )
    mixture = FrozenTokenLayerMixture()
    mixed = mixture(tapped.particle_tokens, tapped.particle_mask)
    assert mixed.shape == (3, 7, 128)
    mixed.sum().backward()
    assert mixture.logits.grad is not None
    assert all(parameter.grad is None for parameter in teacher.parameters())


def test_contextual_tap_restores_weaver_trimmed_padding_suffix():
    batch = _batch(batch=3, particles=7)
    batch["mask"][:, :, -2:] = False
    tapped = FrozenContextualParticleTeacher(
        _FakeTrimmedTeacher(),
        ParticleTokenTapSpec(
            particle_source="fixed_hlt",
            architecture="base",
            tap_choice="penultimate",
        ),
    )(
        batch["points"],
        batch["features"],
        batch["lorentz_vectors"],
        batch["mask"],
    )
    assert tapped.particle_tokens.shape == (3, 1, 7, 128)
    assert torch.count_nonzero(tapped.particle_tokens[:, :, -2:]) == 0
    assert torch.count_nonzero(tapped.particle_tokens[:, :, :5]) > 0
    assert torch.equal(tapped.particle_mask, batch["mask"][:, 0])


def test_contextual_tap_rejects_trimming_an_active_particle():
    batch = _batch(batch=3, particles=7)
    batch["mask"][:, :, -2:] = False
    tapped = FrozenContextualParticleTeacher(
        _FakeOverTrimmedTeacher(),
        ParticleTokenTapSpec(
            particle_source="fixed_hlt",
            architecture="base",
            tap_choice="penultimate",
        ),
    )
    with pytest.raises(ValueError, match="trimmed an active external particle"):
        tapped(
            batch["points"],
            batch["features"],
            batch["lorentz_vectors"],
            batch["mask"],
        )


def test_checkpoint_order_is_accuracy_tolerance_then_ce_ece_epoch():
    selected = select_teacher_checkpoint(
        [
            {"epoch": 1, "accuracy": 0.80000, "cross_entropy": 0.5, "ece": 0.04},
            {"epoch": 2, "accuracy": 0.80005, "cross_entropy": 0.6, "ece": 0.01},
            {"epoch": 3, "accuracy": 0.79980, "cross_entropy": 0.1, "ece": 0.01},
        ]
    )
    assert selected["epoch"] == 1
    assert teacher_learning_rate(update_index=0, total_updates=4_000) > 0
    assert teacher_learning_rate(
        update_index=3_999, total_updates=4_000
    ) == pytest.approx(3.0e-6)


def test_registration_and_reload_are_hash_bound_and_deterministic(tmp_path):
    recipe = _recipe()
    model = _FakeTeacher()
    checkpoint = tmp_path / "teacher.pt"
    torch.save(
        build_teacher_checkpoint_payload(
            recipe=recipe,
            model=model,
            epoch=4,
            optimizer_updates=12,
            model_val_stop_metrics={
                "accuracy": 0.7,
                "cross_entropy": 0.8,
                "ece": 0.1,
            },
        ),
        checkpoint,
    )
    registration = build_teacher_registration(
        recipe=recipe, checkpoint_path=checkpoint
    )
    tap_registration = build_token_tap_registration(
        teacher_registration=registration,
        tap_spec=ParticleTokenTapSpec(
            particle_source="fixed_hlt",
            architecture="base",
            tap_choice="penultimate",
        ),
        input_normalization_sha256="a" * 64,
    )
    assert (
        validate_token_tap_registration(tap_registration)
        == tap_registration["content_hash"]
    )
    reloaded = reload_registered_teacher(
        registration=registration,
        checkpoint_path=checkpoint,
        model_factory=lambda _payload: _FakeTeacher(),
    )
    batch = _batch()
    model.eval()
    reloaded.eval()
    expected = model(
        batch["points"],
        batch["features"],
        batch["lorentz_vectors"],
        batch["mask"],
    )
    actual = reloaded(
        batch["points"],
        batch["features"],
        batch["lorentz_vectors"],
        batch["mask"],
    )
    for name, value in model.state_dict().items():
        assert torch.equal(value, reloaded.state_dict()[name])
    torch.testing.assert_close(expected, actual, rtol=0.0, atol=1.0e-6)
    assert audit_frozen_teacher(reloaded)["frozen"]

    checkpoint.write_bytes(checkpoint.read_bytes() + b"stale")
    with pytest.raises(ValueError, match="hash differs"):
        reload_registered_teacher(
            registration=registration,
            checkpoint_path=checkpoint,
            model_factory=lambda _payload: _FakeTeacher(),
        )


def test_existing_teacher_is_selectable_only_with_exact_recipe_and_train_hash(
    tmp_path,
):
    checkpoint = tmp_path / "existing.pt"
    checkpoint.write_bytes(b"existing-checkpoint")
    recipe = _recipe("Toff_view", "base")
    compatible = build_existing_teacher_source_registration(
        checkpoint_path=checkpoint,
        canonical_train_identity_sha256=recipe.train_identity_sha256,
        observed_train_identity_sha256=recipe.train_identity_sha256,
        serialized_recipe=recipe.to_payload(),
        recipe_reproduced_exactly=True,
        provenance_metadata_sha256="b" * 64,
        description="compatible existing offline teacher",
    )
    assert compatible["selectable"] is True
    diagnostic = build_existing_teacher_source_registration(
        checkpoint_path=checkpoint,
        canonical_train_identity_sha256=recipe.train_identity_sha256,
        observed_train_identity_sha256="c" * 64,
        serialized_recipe=recipe.to_payload(),
        recipe_reproduced_exactly=True,
        provenance_metadata_sha256="b" * 64,
        description="different training identities",
    )
    assert diagnostic["selectable"] is False
    assert diagnostic["selection_status"] == "diagnostic_nonselectable"


def test_predeclared_direct_control_grid_is_deterministic():
    first = build_predeclared_direct_control_grid()
    second = build_predeclared_direct_control_grid()
    assert first == second
    assert len(first["candidates"]) == 16
    assert all(row["hlt_only"] for row in first["candidates"])


class _OneBatchLoader:
    batch_size = 128

    def __init__(self, batch):
        self.batch = batch

    def __len__(self):
        return 1

    def __iter__(self):
        return iter((self.batch,))


def test_training_smoke_publishes_selected_checkpoint_without_other_splits(
    tmp_path,
):
    batch = _batch(batch=2, particles=5)
    loader = _OneBatchLoader(batch)
    report = train_particle_view_teacher(
        recipe=_recipe(),
        train_loader=loader,
        model_val_stop_loader=loader,
        config=ParticleViewTeacherTrainConfig(
            output_dir=str(tmp_path), device="cpu", max_train_batches=1
        ),
        model=_FakeTeacher(),
    )
    assert report["status"] == "COMPLETE"
    assert report["model_val_select_evaluated"] is False
    assert report["stack_val_loaded"] is False
    assert report["final_test_loaded"] is False
    assert (tmp_path / "teacher_registration.json").is_file()
