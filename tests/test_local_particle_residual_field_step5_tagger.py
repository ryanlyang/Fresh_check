from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from jetclass_fresh.jetclass_data import JetIdentity, JetView
from jetclass_fresh.part_inputs import PF_FEATURE_NAMES
from teacher_logit_reco.local_particle_residual_field import (
    LocalParticleResidualFieldCache,
    LocalParticleResidualFieldDataset,
    LocalParticleResidualFieldDatasetConfig,
    LocalParticleResidualFieldHLTOnlyDataset,
    LocalResidualFieldAugmentedParT,
    LocalResidualFieldTaggerConfig,
    LocalResidualFieldTaggerTrainConfig,
    RESIDUAL_FIELD_SOURCE_HLT_ONLY,
    RESIDUAL_FIELD_SOURCE_ORACLE,
    RESIDUAL_FIELD_SOURCE_ORACLE_DROPOUT,
    RESIDUAL_FIELD_SOURCE_ORACLE_FIELD_SUBSET,
    RESIDUAL_FIELD_SOURCE_ORACLE_NOISY,
    RESIDUAL_FIELD_SOURCE_ORACLE_SCALED,
    RESIDUAL_FIELD_SOURCE_ZERO,
    collate_local_particle_residual_field_batch,
    train_local_residual_field_tagger,
    warm_start_local_residual_field_tagger_part,
)
import teacher_logit_reco.local_particle_residual_field.tagger as tagger_module
import teacher_logit_reco.local_particle_residual_field.tagger_train as tagger_train_module


FIELD_NAMES = (
    "r0p02.delta_log_pt_sum",
    "r0p02.delta_pt_frac",
    "r0p02.delta_log_n",
    "r0p02.missing_n_frac",
    "flag.is_merged_token",
    "local_reliability_score",
)
FIELD_GROUPS = {
    "pt_density": [0, 1],
    "multiplicity": [2, 3],
    "reliability": [4, 5],
}


class FakePart(torch.nn.Module):
    def __init__(self, *, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.config = {"input_dim": int(input_dim), "num_classes": int(num_classes)}
        self.proj = torch.nn.Conv1d(int(input_dim), int(num_classes), kernel_size=1)

    def forward(self, points, features, lorentz_vectors, mask):
        del points, lorentz_vectors
        logits_per_particle = self.proj(features)
        weights = mask.to(dtype=features.dtype)
        return (logits_per_particle * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)


def fake_build_hlt_classifier(*, num_classes: int, model_size: str = "base", overrides=None):
    del model_size
    input_dim = int((overrides or {}).get("input_dim", len(PF_FEATURE_NAMES)))
    return FakePart(input_dim=input_dim, num_classes=int(num_classes))


def _dataset(split: str, *, n_jets: int = 8) -> LocalParticleResidualFieldDataset:
    tokens = np.zeros((n_jets, 5, 14), dtype=np.float32)
    mask = np.zeros((n_jets, 5), dtype=bool)
    labels = np.arange(n_jets, dtype=np.int64) % 3
    mask[:, :3] = True
    for jet in range(n_jets):
        tokens[jet, :3, 0] = np.asarray([10.0 + jet, 5.0, 2.0], dtype=np.float32)
        tokens[jet, :3, 1] = np.asarray([0.0, 0.04, 0.08], dtype=np.float32)
        tokens[jet, :3, 2] = np.asarray([0.0, 0.03, 0.10], dtype=np.float32)
        tokens[jet, :3, 3] = tokens[jet, :3, 0] * 1.1
        tokens[jet, :3, 5] = 1.0
    fields = np.zeros((n_jets, 5, len(FIELD_NAMES)), dtype=np.float32)
    fields[:, :3, 0] = 0.1
    fields[:, :3, 1] = tokens[:, :3, 0] / 100.0
    fields[:, :3, 2] = 0.2
    fields[:, :3, 3] = 0.05
    fields[:, :3, 4] = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    fields[:, :3, 5] = 0.8
    jet_ids = tuple(
        JetIdentity(file=f"{split}_{index}.root", entry=index, label=int(label))
        for index, label in enumerate(labels)
    )
    view = JetView(
        tokens=tokens,
        mask=mask,
        labels=labels,
        jet_ids=jet_ids,
        split=split,
        metadata={
            "view": "fixed_hlt",
            "hlt_content_hash": f"hlt_{split}",
            "jet_identity_hash": f"identity_{split}",
            "source_manifest_hash": "manifest_hash",
        },
    )
    cache = LocalParticleResidualFieldCache(
        target_fields=fields,
        target_mask=mask.copy(),
        labels=labels.copy(),
        jet_ids=jet_ids,
        field_names=FIELD_NAMES,
        field_groups={key: list(value) for key, value in FIELD_GROUPS.items()},
        radii=(0.02,),
        split=split,
        metadata={
            "target_content_hash": f"target_{split}",
            "hlt_content_hash": f"hlt_{split}",
            "offline_content_hash": f"offline_{split}",
            "jet_identity_hash": f"identity_{split}",
            "source_manifest_hash": "manifest_hash",
            "zero_baseline_metrics": {"mae": 0.1},
        },
    )
    return LocalParticleResidualFieldDataset(
        view,
        cache,
        config=LocalParticleResidualFieldDatasetConfig(
            hlt_cache_dir="unused",
            target_cache_dir="unused",
            split=split,
            include_oracle_fields=True,
        ),
    )


def _hlt_only_dataset(split: str, *, n_jets: int = 8) -> LocalParticleResidualFieldHLTOnlyDataset:
    tokens = np.zeros((n_jets, 5, 14), dtype=np.float32)
    mask = np.zeros((n_jets, 5), dtype=bool)
    labels = np.arange(n_jets, dtype=np.int64) % 3
    mask[:, :3] = True
    for jet in range(n_jets):
        tokens[jet, :3, 0] = np.asarray([10.0 + jet, 5.0, 2.0], dtype=np.float32)
        tokens[jet, :3, 1] = np.asarray([0.0, 0.04, 0.08], dtype=np.float32)
        tokens[jet, :3, 2] = np.asarray([0.0, 0.03, 0.10], dtype=np.float32)
        tokens[jet, :3, 3] = tokens[jet, :3, 0] * 1.1
        tokens[jet, :3, 5] = 1.0
    jet_ids = tuple(
        JetIdentity(file=f"{split}_{index}.root", entry=index, label=int(label))
        for index, label in enumerate(labels)
    )
    view = JetView(
        tokens=tokens,
        mask=mask,
        labels=labels,
        jet_ids=jet_ids,
        split=split,
        metadata={
            "view": "fixed_hlt",
            "hlt_content_hash": f"hlt_{split}",
            "jet_identity_hash": f"identity_{split}",
            "source_manifest_hash": "manifest_hash",
        },
    )
    return LocalParticleResidualFieldHLTOnlyDataset(
        view,
        config=LocalParticleResidualFieldDatasetConfig(
            hlt_cache_dir="unused",
            target_cache_dir="missing_targets",
            split=split,
        ),
    )


def test_step5_oracle_fields_are_concatenated_to_part_features():
    dataset = _dataset("model_train", n_jets=3)
    batch = collate_local_particle_residual_field_batch([dataset[0], dataset[1]])
    model = LocalResidualFieldAugmentedParT(
        LocalResidualFieldTaggerConfig(
            num_classes=3,
            field_dim=len(FIELD_NAMES),
            field_source=RESIDUAL_FIELD_SOURCE_ORACLE,
            field_names=FIELD_NAMES,
            field_groups=FIELD_GROUPS,
        ),
        part_model=FakePart(input_dim=len(PF_FEATURE_NAMES) + len(FIELD_NAMES), num_classes=3),
    )

    output = model(
        batch["points"],
        batch["features"],
        batch["lorentz_vectors"],
        batch["mask"],
        tokens=batch["tokens"],
        raw_mask=batch["raw_mask"],
        target_fields=batch["target_fields"],
        return_outputs=True,
    )

    assert output.logits.shape == (2, 3)
    assert output.augmented_features.shape[1] == len(PF_FEATURE_NAMES) + len(FIELD_NAMES)
    assert torch.allclose(output.residual_fields, batch["target_fields"])
    assert output.diagnostics["field_source"] == RESIDUAL_FIELD_SOURCE_ORACLE


def test_step5_residual_feature_channels_are_finite_clipped_before_part():
    dataset = _dataset("model_train", n_jets=3)
    batch = collate_local_particle_residual_field_batch([dataset[0], dataset[1]])
    bad_fields = batch["target_fields"].clone()
    bad_fields[0, 0, 0] = float("nan")
    bad_fields[0, 1, 1] = float("inf")
    bad_fields[1, 0, 2] = -float("inf")
    bad_fields[1, 1, 3] = 1234.0
    model = LocalResidualFieldAugmentedParT(
        LocalResidualFieldTaggerConfig(
            num_classes=3,
            field_dim=len(FIELD_NAMES),
            field_source=RESIDUAL_FIELD_SOURCE_ORACLE,
            residual_field_clip_value=4.0,
            field_names=FIELD_NAMES,
            field_groups=FIELD_GROUPS,
        ),
        part_model=FakePart(input_dim=len(PF_FEATURE_NAMES) + len(FIELD_NAMES), num_classes=3),
    )

    output = model(
        batch["points"],
        batch["features"],
        batch["lorentz_vectors"],
        batch["mask"],
        tokens=batch["tokens"],
        raw_mask=batch["raw_mask"],
        target_fields=bad_fields,
        return_outputs=True,
    )

    residual_channels = output.augmented_features[:, -len(FIELD_NAMES) :, :]
    assert torch.isfinite(residual_channels).all()
    assert float(residual_channels.abs().max().item()) <= 4.0
    assert residual_channels[0, 0, 0].item() == 0.0
    assert residual_channels[0, 1, 1].item() == 4.0
    assert residual_channels[1, 2, 0].item() == -4.0


def test_step5_hlt_only_uses_clean_part_input_without_zero_field_channels():
    dataset = _dataset("model_train", n_jets=3)
    batch = collate_local_particle_residual_field_batch([dataset[0], dataset[1]])
    model = LocalResidualFieldAugmentedParT(
        LocalResidualFieldTaggerConfig(
            num_classes=3,
            field_dim=len(FIELD_NAMES),
            field_source=RESIDUAL_FIELD_SOURCE_HLT_ONLY,
            field_names=FIELD_NAMES,
            field_groups=FIELD_GROUPS,
        ),
        part_model=FakePart(input_dim=len(PF_FEATURE_NAMES), num_classes=3),
    )

    output = model(
        batch["points"],
        batch["features"],
        batch["lorentz_vectors"],
        batch["mask"],
        tokens=batch["tokens"],
        raw_mask=batch["raw_mask"],
        target_fields=batch["target_fields"],
        return_outputs=True,
    )

    assert output.logits.shape == (2, 3)
    assert output.augmented_features.shape[1] == len(PF_FEATURE_NAMES)
    assert output.residual_fields.shape[-1] == 0
    assert output.diagnostics["field_source"] == RESIDUAL_FIELD_SOURCE_HLT_ONLY
    assert output.diagnostics["hlt_only"] is True
    assert model.part_model.config["input_dim"] == len(PF_FEATURE_NAMES)


def test_step5_field_subset_changes_actual_part_input_tensor():
    dataset = _dataset("model_train", n_jets=3)
    batch = collate_local_particle_residual_field_batch([dataset[0], dataset[1]])
    model = LocalResidualFieldAugmentedParT(
        LocalResidualFieldTaggerConfig(
            num_classes=3,
            field_dim=2,
            field_source=RESIDUAL_FIELD_SOURCE_ORACLE,
            field_names=FIELD_NAMES[:2],
            field_groups={"pt_density": (0, 1)},
            source_field_indices=(0, 1),
        ),
        part_model=FakePart(input_dim=len(PF_FEATURE_NAMES) + 2, num_classes=3),
    )

    output = model(
        batch["points"],
        batch["features"],
        batch["lorentz_vectors"],
        batch["mask"],
        tokens=batch["tokens"],
        raw_mask=batch["raw_mask"],
        target_fields=batch["target_fields"],
        return_outputs=True,
    )

    assert output.augmented_features.shape[1] == len(PF_FEATURE_NAMES) + 2
    assert output.residual_fields.shape[-1] == 2
    assert torch.allclose(output.residual_fields, batch["target_fields"][..., :2])


def test_curriculum_step1_oracle_scaled_source_applies_alpha_before_part_input():
    dataset = _dataset("model_train", n_jets=3)
    batch = collate_local_particle_residual_field_batch([dataset[0], dataset[1]])
    model = LocalResidualFieldAugmentedParT(
        LocalResidualFieldTaggerConfig(
            num_classes=3,
            field_dim=len(FIELD_NAMES),
            field_source=RESIDUAL_FIELD_SOURCE_ORACLE_SCALED,
            oracle_field_alpha=0.25,
            field_names=FIELD_NAMES,
            field_groups=FIELD_GROUPS,
        ),
        part_model=FakePart(input_dim=len(PF_FEATURE_NAMES) + len(FIELD_NAMES), num_classes=3),
    )

    output = model(
        batch["points"],
        batch["features"],
        batch["lorentz_vectors"],
        batch["mask"],
        tokens=batch["tokens"],
        raw_mask=batch["raw_mask"],
        target_fields=batch["target_fields"],
        return_outputs=True,
    )

    expected = batch["target_fields"] * 0.25
    assert torch.allclose(output.residual_fields, expected)
    assert torch.allclose(output.augmented_features[:, -len(FIELD_NAMES) :, :], expected.transpose(1, 2))
    assert output.diagnostics["field_source"] == RESIDUAL_FIELD_SOURCE_ORACLE_SCALED
    assert output.diagnostics["oracle_field_transform"]["oracle_field_alpha"] == 0.25


def test_curriculum_step1_alpha_endpoints_preserve_masks_and_sanitize_nonfinite_fields():
    dataset = _dataset("model_train", n_jets=3)
    batch = collate_local_particle_residual_field_batch([dataset[0], dataset[1]])
    target = batch["target_fields"].clone()
    target[:, 3:, :] = 123.0
    target[0, 0, 0] = float("nan")
    target[0, 1, 1] = float("inf")

    outputs = {}
    for alpha in (0.0, 1.0):
        model = LocalResidualFieldAugmentedParT(
            LocalResidualFieldTaggerConfig(
                num_classes=3,
                field_dim=len(FIELD_NAMES),
                field_source=RESIDUAL_FIELD_SOURCE_ORACLE_SCALED,
                oracle_field_alpha=alpha,
                field_names=FIELD_NAMES,
                field_groups=FIELD_GROUPS,
            ),
            part_model=FakePart(input_dim=len(PF_FEATURE_NAMES) + len(FIELD_NAMES), num_classes=3),
        )
        model.eval()
        outputs[alpha] = model(
            batch["points"],
            batch["features"],
            batch["lorentz_vectors"],
            batch["mask"],
            raw_mask=batch["raw_mask"],
            target_fields=target,
            return_outputs=True,
        )

    blank_model = LocalResidualFieldAugmentedParT(
        LocalResidualFieldTaggerConfig(
            num_classes=3,
            field_dim=len(FIELD_NAMES),
            field_source=RESIDUAL_FIELD_SOURCE_ZERO,
            field_names=FIELD_NAMES,
            field_groups=FIELD_GROUPS,
        ),
        part_model=FakePart(input_dim=len(PF_FEATURE_NAMES) + len(FIELD_NAMES), num_classes=3),
    )
    blank_model.eval()
    blank = blank_model(
        batch["points"],
        batch["features"],
        batch["lorentz_vectors"],
        batch["mask"],
        raw_mask=batch["raw_mask"],
        return_outputs=True,
    )
    assert torch.equal(outputs[0.0].residual_fields, blank.residual_fields)
    assert torch.equal(outputs[0.0].residual_features, blank.residual_features)
    valid = batch["raw_mask"][:, :, None].expand_as(target)
    expected_alpha_one = torch.nan_to_num(target, nan=0.0, posinf=8.0, neginf=-8.0).clamp(-8.0, 8.0)
    expected_alpha_one = expected_alpha_one * valid.to(dtype=target.dtype)
    assert torch.allclose(outputs[1.0].residual_fields, expected_alpha_one)
    assert torch.count_nonzero(outputs[1.0].residual_fields[~valid]) == 0
    diagnostics = outputs[1.0].diagnostics["oracle_field_transform"]
    assert diagnostics["oracle_field_input_nonfinite_count"] == 2
    assert diagnostics["oracle_field_pre_sanitize_nonfinite_count"] == 2


def test_curriculum_step1_oracle_subset_selects_noncontiguous_physical_columns():
    dataset = _dataset("model_train", n_jets=3)
    batch = collate_local_particle_residual_field_batch([dataset[0], dataset[1]])
    selected_indices = (0, 3, 5)
    selected_names = tuple(FIELD_NAMES[index] for index in selected_indices)
    model = LocalResidualFieldAugmentedParT(
        LocalResidualFieldTaggerConfig(
            num_classes=3,
            field_dim=len(selected_indices),
            field_source=RESIDUAL_FIELD_SOURCE_ORACLE_FIELD_SUBSET,
            field_names=selected_names,
            field_groups={"selected": tuple(range(len(selected_indices)))},
            source_field_indices=selected_indices,
        ),
        part_model=FakePart(input_dim=len(PF_FEATURE_NAMES) + len(selected_indices), num_classes=3),
    )

    output = model(
        batch["points"],
        batch["features"],
        batch["lorentz_vectors"],
        batch["mask"],
        raw_mask=batch["raw_mask"],
        target_fields=batch["target_fields"],
        return_outputs=True,
    )

    expected = batch["target_fields"].index_select(-1, torch.tensor(selected_indices))
    assert torch.allclose(output.residual_fields, expected)
    assert output.diagnostics["oracle_field_transform"]["oracle_field_selected_indices"] == list(selected_indices)
    assert output.diagnostics["oracle_field_transform"]["oracle_field_selected_names"] == list(selected_names)


def test_curriculum_step1_noisy_dropout_is_seeded_and_eval_is_clean():
    dataset = _dataset("model_train", n_jets=3)
    batch = collate_local_particle_residual_field_batch([dataset[0], dataset[1]])
    model = LocalResidualFieldAugmentedParT(
        LocalResidualFieldTaggerConfig(
            num_classes=3,
            field_dim=len(FIELD_NAMES),
            field_source=RESIDUAL_FIELD_SOURCE_ORACLE_NOISY,
            oracle_field_alpha=1.0,
            oracle_field_noise_std=0.2,
            oracle_field_dropout=0.25,
            oracle_field_group_dropout=0.5,
            field_names=FIELD_NAMES,
            field_groups=FIELD_GROUPS,
        ),
        part_model=FakePart(input_dim=len(PF_FEATURE_NAMES) + len(FIELD_NAMES), num_classes=3),
    )

    def forward_fields(seed: int):
        torch.manual_seed(seed)
        return model(
            batch["points"],
            batch["features"],
            batch["lorentz_vectors"],
            batch["mask"],
            raw_mask=batch["raw_mask"],
            target_fields=batch["target_fields"],
            return_outputs=True,
        )

    model.train()
    first = forward_fields(31415)
    repeated = forward_fields(31415)
    changed = forward_fields(27182)
    assert torch.equal(first.residual_fields, repeated.residual_fields)
    assert not torch.equal(first.residual_fields, changed.residual_fields)
    assert first.diagnostics["oracle_field_transform"]["oracle_field_corruption_active"] is True

    model.eval()
    clean = forward_fields(31415)
    assert torch.allclose(clean.residual_fields, batch["target_fields"])
    assert clean.diagnostics["oracle_field_transform"]["oracle_field_corruption_active"] is False


def test_curriculum_step1_named_modes_validate_and_record_distinct_contracts():
    configs = (
        LocalResidualFieldTaggerConfig(
            num_classes=3,
            field_dim=len(FIELD_NAMES),
            field_source=RESIDUAL_FIELD_SOURCE_ORACLE_SCALED,
            oracle_field_alpha=0.5,
            field_names=FIELD_NAMES,
            field_groups=FIELD_GROUPS,
        ),
        LocalResidualFieldTaggerConfig(
            num_classes=3,
            field_dim=len(FIELD_NAMES),
            field_source=RESIDUAL_FIELD_SOURCE_ORACLE_NOISY,
            oracle_field_noise_std=0.1,
            oracle_field_dropout=0.2,
            field_names=FIELD_NAMES,
            field_groups=FIELD_GROUPS,
        ),
        LocalResidualFieldTaggerConfig(
            num_classes=3,
            field_dim=len(FIELD_NAMES),
            field_source=RESIDUAL_FIELD_SOURCE_ORACLE_DROPOUT,
            oracle_field_dropout=0.2,
            oracle_field_group_dropout=0.3,
            field_names=FIELD_NAMES,
            field_groups=FIELD_GROUPS,
        ),
    )
    assert [config.to_dict()["field_source"] for config in configs] == [
        RESIDUAL_FIELD_SOURCE_ORACLE_SCALED,
        RESIDUAL_FIELD_SOURCE_ORACLE_NOISY,
        RESIDUAL_FIELD_SOURCE_ORACLE_DROPOUT,
    ]
    with pytest.raises(ValueError, match="positive oracle_field_noise_std"):
        LocalResidualFieldTaggerConfig(field_source=RESIDUAL_FIELD_SOURCE_ORACLE_NOISY)
    with pytest.raises(ValueError, match="does not apply Gaussian noise"):
        LocalResidualFieldTaggerConfig(
            field_source=RESIDUAL_FIELD_SOURCE_ORACLE_DROPOUT,
            oracle_field_dropout=0.1,
            oracle_field_noise_std=0.1,
        )
    with pytest.raises(ValueError, match="finite"):
        LocalResidualFieldTaggerConfig(
            field_source=RESIDUAL_FIELD_SOURCE_ORACLE_SCALED,
            oracle_field_alpha=float("nan"),
        )


def test_step5_joint_reconstructor_fields_feed_part_features():
    dataset = _dataset("model_train", n_jets=3)
    batch = collate_local_particle_residual_field_batch([dataset[0], dataset[1]])
    model = LocalResidualFieldAugmentedParT(
        LocalResidualFieldTaggerConfig(
            num_classes=3,
            field_dim=len(FIELD_NAMES),
            field_source="joint_reconstructor",
            field_names=FIELD_NAMES,
            field_groups=FIELD_GROUPS,
            reconstructor_config={
                "variant": "C0",
                "field_dim": len(FIELD_NAMES),
                "d_model": 24,
                "num_heads": 3,
                "num_layers": 1,
                "field_names": FIELD_NAMES,
                "field_groups": FIELD_GROUPS,
            },
        ),
        part_model=FakePart(input_dim=len(PF_FEATURE_NAMES) + len(FIELD_NAMES), num_classes=3),
    )

    output = model(
        batch["points"],
        batch["features"],
        batch["lorentz_vectors"],
        batch["mask"],
        tokens=batch["tokens"],
        raw_mask=batch["raw_mask"],
        return_outputs=True,
    )

    assert output.logits.shape == (2, 3)
    assert output.reconstructor_output is not None
    assert output.residual_fields.shape == batch["target_fields"].shape
    assert output.diagnostics["reconstructor_contract"]


def test_step5_warm_start_partially_copies_expanded_input_projection(tmp_path: Path):
    source = FakePart(input_dim=len(PF_FEATURE_NAMES), num_classes=3)
    target = LocalResidualFieldAugmentedParT(
        LocalResidualFieldTaggerConfig(
            num_classes=3,
            field_dim=len(FIELD_NAMES),
            field_source="zero",
            field_names=FIELD_NAMES,
            field_groups=FIELD_GROUPS,
        ),
        part_model=FakePart(input_dim=len(PF_FEATURE_NAMES) + len(FIELD_NAMES), num_classes=3),
    )
    checkpoint = tmp_path / "baseline.pt"
    torch.save({"model_state_dict": source.state_dict()}, checkpoint)

    report = warm_start_local_residual_field_tagger_part(target, checkpoint, require=True)

    assert report["partial_loaded_key_count"] >= 1
    source_weight = source.proj.weight.detach()
    target_weight = target.part_model.proj.weight.detach()
    assert torch.allclose(target_weight[:, : len(PF_FEATURE_NAMES), :], source_weight)
    assert torch.allclose(target_weight[:, len(PF_FEATURE_NAMES) :, :], torch.zeros_like(target_weight[:, len(PF_FEATURE_NAMES) :, :]))


def test_step5_tagger_train_smoke_writes_checkpoint_and_reports(
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
        output_dir=str(tmp_path / "tagger"),
        hlt_cache_dir="unused_hlt",
        target_cache_dir="unused_targets",
        field_source=RESIDUAL_FIELD_SOURCE_ORACLE,
        num_classes=3,
        label_names=("a", "b", "c"),
        model_size="tiny",
        batch_size=3,
        eval_batch_size=3,
        epochs=2,
        part_lr=1.0e-3,
        device="cpu",
        amp=False,
        early_stop_patience=-1,
    )

    report = train_local_residual_field_tagger(config)

    output_dir = tmp_path / "tagger"
    assert report["ok"] is True
    assert report["best_epoch"] >= 0
    assert (output_dir / "best_model_val.pt").exists()
    assert (output_dir / "last.pt").exists()
    teacher_config = json.loads((output_dir / "teacher_config.json").read_text(encoding="utf-8"))
    assert teacher_config["contract"] == "local_residual_field_oracle_teacher_config_v1"
    assert teacher_config["role"] == "oracle_teacher_candidate"
    assert teacher_config["field_source"] == RESIDUAL_FIELD_SOURCE_ORACLE
    assert teacher_config["best_epoch"] == report["best_epoch"]
    assert teacher_config["reuse_contract"]["contract"] == "local_residual_field_oracle_teacher_reuse_v1"
    assert teacher_config["reuse_contract"]["reuse_contract_hash"] == report["teacher_reuse_contract_hash"]
    assert set(teacher_config["reuse_contract"]["split_provenance"]) == {
        "model_train",
        "model_val",
        "stack_val",
    }


def test_step5_tagger_training_field_subset_reports_selected_shape(
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
        output_dir=str(tmp_path / "tagger_subset"),
        hlt_cache_dir="unused_hlt",
        target_cache_dir="unused_targets",
        field_source=RESIDUAL_FIELD_SOURCE_ORACLE_FIELD_SUBSET,
        field_subset=("pt_density",),
        num_classes=3,
        label_names=("a", "b", "c"),
        model_size="tiny",
        batch_size=3,
        eval_batch_size=3,
        epochs=1,
        part_lr=1.0e-3,
        device="cpu",
        amp=False,
        early_stop_patience=-1,
    )

    report = train_local_residual_field_tagger(config)

    assert report["ok"] is True
    assert report["selected_field_names"] == list(FIELD_NAMES[:2])
    assert report["model_config"]["field_dim"] == 2
    assert report["model_config"]["augmented_feature_dim"] == len(PF_FEATURE_NAMES) + 2
    output_dir = tmp_path / "tagger_subset"
    assert (output_dir / "run_report.json").exists()
    assert (output_dir / "training_curves.json").exists()
    assert (output_dir / "diagnostics" / "epoch_metrics.csv").exists()
    saved = json.loads((output_dir / "run_report.json").read_text(encoding="utf-8"))
    assert saved["field_source"] == RESIDUAL_FIELD_SOURCE_ORACLE_FIELD_SUBSET


def test_step5_hlt_only_training_does_not_require_target_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    datasets = {
        "model_train": _hlt_only_dataset("model_train", n_jets=9),
        "model_val": _hlt_only_dataset("model_val", n_jets=6),
        "stack_val": _hlt_only_dataset("stack_val", n_jets=6),
    }

    def fake_load_dataset(config, split, *, max_jets):
        del config, max_jets
        return datasets[str(split)]

    monkeypatch.setattr(tagger_module, "build_hlt_classifier", fake_build_hlt_classifier)
    monkeypatch.setattr(tagger_train_module, "_load_tagger_dataset", fake_load_dataset)
    config = LocalResidualFieldTaggerTrainConfig(
        output_dir=str(tmp_path / "tagger_hlt_only"),
        hlt_cache_dir="unused_hlt",
        target_cache_dir=str(tmp_path / "does_not_exist"),
        field_source=RESIDUAL_FIELD_SOURCE_HLT_ONLY,
        num_classes=3,
        label_names=("a", "b", "c"),
        model_size="tiny",
        batch_size=3,
        eval_batch_size=3,
        epochs=1,
        part_lr=1.0e-3,
        device="cpu",
        amp=False,
        early_stop_patience=-1,
    )

    report = train_local_residual_field_tagger(config)

    assert report["ok"] is True
    assert report["field_source"] == RESIDUAL_FIELD_SOURCE_HLT_ONLY
    assert report["selected_field_names"] == []
    assert report["model_config"]["field_dim"] == 0
    assert report["model_config"]["augmented_feature_dim"] == len(PF_FEATURE_NAMES)


def test_step5_kd_requested_fails_when_teacher_logits_are_missing(tmp_path: Path):
    config = LocalResidualFieldTaggerTrainConfig(
        output_dir=str(tmp_path / "tagger_kd_missing"),
        hlt_cache_dir="unused_hlt",
        target_cache_dir="unused_targets",
        field_source=RESIDUAL_FIELD_SOURCE_HLT_ONLY,
        teacher_logits_dir=str(tmp_path / "teacher_logits"),
        kd_loss_weight=0.25,
        num_classes=3,
        label_names=("a", "b", "c"),
        epochs=1,
        device="cpu",
        amp=False,
    )

    with pytest.raises(FileNotFoundError, match="teacher logits are missing"):
        train_local_residual_field_tagger(config)


def test_step5_selection_rejects_tiny_finite_validation_subset():
    ok, reason = tagger_train_module._metrics_valid_for_selection(
        {"n_jets": 1024, "loss": 0.5, "accuracy": 0.9},
        expected_n_jets=1_000_000,
        min_valid_fraction=0.99,
    )

    assert ok is False
    assert "finite validation coverage 1024/1000000" in reason

    ok, reason = tagger_train_module._metrics_valid_for_selection(
        {"n_jets": 990_000, "loss": 0.5, "accuracy": 0.9},
        expected_n_jets=1_000_000,
        min_valid_fraction=0.99,
    )

    assert ok is True
    assert reason == ""

    ok, reason = tagger_train_module._metrics_valid_for_selection(
        {"n_jets": 1_000_000, "loss": float("nan"), "accuracy": 0.9},
        expected_n_jets=1_000_000,
        min_valid_fraction=0.99,
    )

    assert ok is False
    assert reason == "loss is not finite"


def test_step5_teacher_logits_resolver_accepts_pd10_prediction_cache_layout(tmp_path: Path):
    root = tmp_path / "teacher_logits"
    offline_dir = root / "offline_part_teacher_10class"
    offline_dir.mkdir(parents=True)
    expected = offline_dir / "model_train_predictions.npz"
    np.savez_compressed(
        expected,
        logits=np.zeros((2, 3), dtype=np.float32),
        labels=np.zeros((2,), dtype=np.int64),
    )
    config = LocalResidualFieldTaggerTrainConfig(
        output_dir=str(tmp_path / "tagger_kd"),
        hlt_cache_dir="unused_hlt",
        target_cache_dir="unused_targets",
        field_source=RESIDUAL_FIELD_SOURCE_HLT_ONLY,
        teacher_logits_dir=str(root),
        kd_loss_weight=0.25,
        num_classes=3,
        label_names=("a", "b", "c"),
        epochs=1,
        device="cpu",
        amp=False,
    )

    assert tagger_train_module._teacher_logits_path(config, "model_train") == str(expected)
