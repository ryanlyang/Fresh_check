from __future__ import annotations

from pathlib import Path

import pytest
import torch

from jetclass_fresh.part_inputs import PF_FEATURE_NAMES
from teacher_logit_reco.local_particle_residual_field import (
    LocalResidualFieldAugmentedParT,
    LocalResidualFieldControlConfig,
    LocalResidualFieldControlGenerator,
    LocalResidualFieldTaggerConfig,
    LocalResidualFieldTaggerTrainConfig,
    RESIDUAL_FIELD_SOURCE_CROSS_JET_SHUFFLE,
    RESIDUAL_FIELD_SOURCE_LEARNED_NO_TARGET,
    RESIDUAL_FIELD_SOURCE_RADIUS_PERMUTED,
    RESIDUAL_FIELD_SOURCE_WITHIN_JET_SHUFFLE,
    apply_local_residual_field_control,
    collate_local_particle_residual_field_batch,
    radius_field_groups,
    train_local_residual_field_tagger,
)
from tests.test_local_particle_residual_field_step5_tagger import (
    FIELD_GROUPS,
    FakePart,
    _dataset,
    fake_build_hlt_classifier,
)
import teacher_logit_reco.local_particle_residual_field.tagger as tagger_module
import teacher_logit_reco.local_particle_residual_field.tagger_train as tagger_train_module


RADIUS_FIELD_NAMES = (
    "r0p02.delta_log_pt_sum",
    "r0p02.delta_pt_frac",
    "r0p05.delta_log_pt_sum",
    "r0p05.delta_pt_frac",
    "flag.is_merged_token",
)


def test_step6_cross_jet_shuffle_preserves_shape_and_current_mask():
    fields = torch.arange(3 * 4 * 2, dtype=torch.float32).reshape(3, 4, 2)
    mask = torch.tensor(
        [
            [True, True, False, False],
            [True, False, False, False],
            [True, True, True, False],
        ]
    )

    output = apply_local_residual_field_control(
        source=RESIDUAL_FIELD_SOURCE_CROSS_JET_SHUFFLE,
        target_fields=fields,
        mask=mask,
        field_names=("a", "b"),
        config=LocalResidualFieldControlConfig(seed=123),
        indices=torch.arange(3),
    )

    assert output.fields.shape == fields.shape
    assert torch.all(output.fields[~mask] == 0.0)
    assert output.diagnostics["control_source"] == RESIDUAL_FIELD_SOURCE_CROSS_JET_SHUFFLE
    assert not torch.allclose(output.fields[mask], fields[mask])


def test_step6_within_jet_shuffle_keeps_values_inside_each_jet():
    fields = torch.arange(2 * 5 * 1, dtype=torch.float32).reshape(2, 5, 1)
    mask = torch.tensor([[True, True, True, False, False], [True, True, False, False, False]])

    output = apply_local_residual_field_control(
        source=RESIDUAL_FIELD_SOURCE_WITHIN_JET_SHUFFLE,
        target_fields=fields,
        mask=mask,
        field_names=("x",),
        config=LocalResidualFieldControlConfig(seed=321),
        indices=torch.arange(2),
    )

    assert output.fields.shape == fields.shape
    for jet in range(2):
        valid = mask[jet]
        assert sorted(output.fields[jet, valid, 0].tolist()) == sorted(fields[jet, valid, 0].tolist())
    assert torch.all(output.fields[~mask] == 0.0)


def test_step6_radius_permutation_swaps_radius_channels_only():
    groups = radius_field_groups(RADIUS_FIELD_NAMES)
    assert groups == ((0, 1), (2, 3))
    fields = torch.zeros((1, 2, len(RADIUS_FIELD_NAMES)), dtype=torch.float32)
    fields[..., 0] = 1.0
    fields[..., 1] = 2.0
    fields[..., 2] = 10.0
    fields[..., 3] = 20.0
    fields[..., 4] = 99.0

    output = apply_local_residual_field_control(
        source=RESIDUAL_FIELD_SOURCE_RADIUS_PERMUTED,
        target_fields=fields,
        mask=torch.ones((1, 2), dtype=torch.bool),
        field_names=RADIUS_FIELD_NAMES,
        config=LocalResidualFieldControlConfig(seed=7),
    )

    assert output.fields.shape == fields.shape
    assert torch.allclose(output.fields[..., 4], fields[..., 4])
    assert output.diagnostics["radius_permutation"] in ([1, 0], [0, 1])
    assert torch.allclose(output.fields[..., :4].sort(dim=-1).values, fields[..., :4].sort(dim=-1).values)


def test_step6_learned_no_target_generator_masks_invalid_particles():
    generator = LocalResidualFieldControlGenerator(field_dim=4, hidden_dim=16, dropout=0.0)
    tokens = torch.randn(2, 5, 14)
    mask = torch.tensor([[True, True, False, False, False], [True, False, False, False, False]])

    output = generator(tokens, mask)

    assert output.fields.shape == (2, 5, 4)
    assert torch.all(output.fields[~mask] == 0.0)
    assert output.diagnostics["learned_no_target"] is True


def test_step6_tagger_accepts_radius_permuted_control():
    dataset = _dataset("model_train", n_jets=3)
    batch = collate_local_particle_residual_field_batch([dataset[0], dataset[1]])
    model = LocalResidualFieldAugmentedParT(
        LocalResidualFieldTaggerConfig(
            num_classes=3,
            field_dim=len(dataset.field_names),
            field_source=RESIDUAL_FIELD_SOURCE_RADIUS_PERMUTED,
            field_names=dataset.field_names,
            field_groups=FIELD_GROUPS,
            control_config={"seed": 13},
        ),
        part_model=FakePart(input_dim=len(PF_FEATURE_NAMES) + len(dataset.field_names), num_classes=3),
    )

    output = model(
        batch["points"],
        batch["features"],
        batch["lorentz_vectors"],
        batch["mask"],
        tokens=batch["tokens"],
        raw_mask=batch["raw_mask"],
        indices=batch["indices"],
        target_fields=batch["target_fields"],
        return_outputs=True,
    )

    assert output.logits.shape == (2, 3)
    assert output.residual_fields.shape == batch["target_fields"].shape
    assert output.control_diagnostics is not None
    assert output.diagnostics["control_contract"]


def test_step6_learned_no_target_tagger_train_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    datasets = {
        "model_train": _dataset("model_train", n_jets=9),
        "model_val": _dataset("model_val", n_jets=6),
        "stack_val": _dataset("stack_val", n_jets=6),
    }

    def fake_load_dataset(config, split, *, max_jets):
        del config, max_jets
        return datasets[str(split)]

    monkeypatch.setattr(tagger_module, "build_hlt_classifier", fake_build_hlt_classifier)
    monkeypatch.setattr(tagger_train_module, "_load_tagger_dataset", fake_load_dataset)
    config = LocalResidualFieldTaggerTrainConfig(
        output_dir=str(tmp_path / "learned_no_target"),
        hlt_cache_dir="unused_hlt",
        target_cache_dir="unused_targets",
        field_source=RESIDUAL_FIELD_SOURCE_LEARNED_NO_TARGET,
        num_classes=3,
        label_names=("a", "b", "c"),
        model_size="tiny",
        batch_size=3,
        eval_batch_size=3,
        epochs=1,
        part_lr=1.0e-3,
        reconstructor_lr=1.0e-3,
        learned_control_hidden_dim=16,
        learned_control_dropout=0.0,
        device="cpu",
        amp=False,
        early_stop_patience=-1,
    )

    report = train_local_residual_field_tagger(config)

    assert report["ok"] is True
    assert report["field_source"] == RESIDUAL_FIELD_SOURCE_LEARNED_NO_TARGET
    assert (tmp_path / "learned_no_target" / "best_model_val.pt").exists()
