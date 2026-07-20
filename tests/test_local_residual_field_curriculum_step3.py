from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from jetclass_fresh.jetclass_data import JetIdentity, JetView
from jetclass_fresh.part_inputs import PF_FEATURE_NAMES
from teacher_logit_reco.local_particle_residual_field import (
    FrozenLocalResidualFieldOracleConsumer,
    FrozenOracleConsumerConfig,
    LocalParticleResidualFieldCache,
    LocalParticleResidualFieldDataset,
    LocalParticleResidualFieldDatasetConfig,
    LocalResidualFieldAugmentedParT,
    LocalResidualFieldTaggerConfig,
    collate_local_particle_residual_field_batch,
)
import teacher_logit_reco.local_particle_residual_field.tagger as tagger_module


FIELD_NAMES = (
    "r0p02.delta_log_pt_sum",
    "r0p02.delta_pt_frac",
    "r0p02.delta_log_n",
    "flag.is_merged_token",
)
FIELD_GROUPS = {
    "pt_density": [0, 1],
    "multiplicity": [2],
    "reliability": [3],
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


def _dataset(split: str, *, n_jets: int = 4) -> LocalParticleResidualFieldDataset:
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
    fields[:, :3, 3] = 0.8
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


def _write_fake_oracle_checkpoint(path: Path, *, field_names=FIELD_NAMES, field_groups=FIELD_GROUPS) -> None:
    model = LocalResidualFieldAugmentedParT(
        LocalResidualFieldTaggerConfig(
            num_classes=3,
            field_dim=len(field_names),
            field_source="oracle_scaled",
            oracle_field_alpha=1.0,
            field_names=field_names,
            field_groups=field_groups,
        ),
        part_model=FakePart(input_dim=len(PF_FEATURE_NAMES) + len(field_names), num_classes=3),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model.config.to_dict(),
            "config": {
                "train_split": "model_train",
                "field_subset": ["pt_density"] if len(field_names) == 2 else [],
            },
            "metrics": {"accuracy": 0.73},
            "selected_field_indices": [0, 1] if len(field_names) == 2 else list(range(len(field_names))),
            "selected_field_names": list(field_names),
        },
        path,
    )
    (path.parent / "teacher_config.json").write_text(
        json.dumps(
            {
                "contract": "local_residual_field_oracle_teacher_config_v1",
                "teacher_id": "Ofull",
                "field_source": "oracle_scaled",
                "oracle_field_alpha": 1.0,
                "field_subset": ["pt_density"] if len(field_names) == 2 else [],
                "selected_field_indices": [0, 1] if len(field_names) == 2 else list(range(len(field_names))),
                "selected_field_names": list(field_names),
            }
        ),
        encoding="utf-8",
    )


def test_step3_frozen_oracle_consumer_detaches_true_logits_and_keeps_predicted_field_gradients(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(tagger_module, "build_hlt_classifier", fake_build_hlt_classifier)
    checkpoint = tmp_path / "Ofull" / "best_model_val.pt"
    _write_fake_oracle_checkpoint(checkpoint)
    dataset = _dataset("model_train", n_jets=4)
    batch = collate_local_particle_residual_field_batch([dataset[0], dataset[1]])
    predicted_fields = batch["target_fields"].detach().clone().requires_grad_(True)

    wrapper = FrozenLocalResidualFieldOracleConsumer(
        FrozenOracleConsumerConfig(checkpoint=str(checkpoint), consumer_id="Ofull", alpha=0.5),
        device="cpu",
    )
    wrapper.model.train()
    output = wrapper(
        points=batch["points"],
        features=batch["features"],
        lorentz_vectors=batch["lorentz_vectors"],
        mask=batch["mask"],
        tokens=batch["tokens"],
        raw_mask=batch["raw_mask"],
        indices=batch["indices"],
        true_fields=batch["target_fields"],
        predicted_fields=predicted_fields,
    )

    assert wrapper.parameters_frozen()
    assert output.teacher_logits_true is not None
    assert output.teacher_logits_pred is not None
    assert output.teacher_logits_true.requires_grad is False
    assert output.teacher_logits_pred.requires_grad is True
    output.teacher_logits_pred.sum().backward()
    assert predicted_fields.grad is not None
    assert float(predicted_fields.grad.abs().sum().item()) > 0.0
    assert wrapper.model.training is False
    assert all(parameter.grad is None for parameter in wrapper.model.parameters())
    assert output.diagnostics["consumer_id"] == "Ofull"
    assert output.diagnostics["alpha"] == 0.5
    assert len(output.diagnostics["teacher_checkpoint_hash"]) == 64
    assert output.diagnostics["teacher_train_split"] == "model_train"
    assert output.diagnostics["teacher_model_val_accuracy"] == 0.73
    assert output.diagnostics["oracle_logit_only_fallback"] is False
    assert output.diagnostics["oracle_input_gradient_distillation_enabled"] is True
    assert output.diagnostics["teacher_logits_true_source"] == "live_true_fields"


def test_step3_frozen_oracle_consumer_selects_teacher_field_subset_from_full_fields(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(tagger_module, "build_hlt_classifier", fake_build_hlt_classifier)
    checkpoint = tmp_path / "Ofull_subset" / "best_model_val.pt"
    _write_fake_oracle_checkpoint(
        checkpoint,
        field_names=FIELD_NAMES[:2],
        field_groups={"pt_density": [0, 1]},
    )
    dataset = _dataset("model_train", n_jets=4)
    batch = collate_local_particle_residual_field_batch([dataset[0], dataset[1]])
    predicted_fields = batch["target_fields"].detach().clone().requires_grad_(True)

    wrapper = FrozenLocalResidualFieldOracleConsumer(
        FrozenOracleConsumerConfig(checkpoint=str(checkpoint), consumer_id="Ofull_subset", alpha=0.25),
        device="cpu",
    )
    output = wrapper(
        points=batch["points"],
        features=batch["features"],
        lorentz_vectors=batch["lorentz_vectors"],
        mask=batch["mask"],
        tokens=batch["tokens"],
        raw_mask=batch["raw_mask"],
        indices=batch["indices"],
        true_fields=batch["target_fields"],
        predicted_fields=predicted_fields,
    )

    assert output.teacher_logits_true.shape == (2, 3)
    assert output.teacher_logits_pred.shape == (2, 3)
    output.teacher_logits_pred.sum().backward()
    assert predicted_fields.grad is not None
    assert float(predicted_fields.grad[..., :2].abs().sum().item()) > 0.0
    assert output.diagnostics["teacher_selected_field_indices"] == [0, 1]
    assert output.diagnostics["teacher_field_subset"] == ["pt_density"]


def test_step3_oracle_forward_microbatching_preserves_logits_and_field_gradients(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(tagger_module, "build_hlt_classifier", fake_build_hlt_classifier)
    checkpoint = tmp_path / "Ofull" / "best_model_val.pt"
    _write_fake_oracle_checkpoint(checkpoint)
    dataset = _dataset("model_train", n_jets=4)
    batch = collate_local_particle_residual_field_batch([dataset[index] for index in range(4)])
    predicted_full = batch["target_fields"].detach().clone().requires_grad_(True)
    predicted_micro = batch["target_fields"].detach().clone().requires_grad_(True)
    full = FrozenLocalResidualFieldOracleConsumer(
        FrozenOracleConsumerConfig(checkpoint=str(checkpoint), alpha=0.75),
        device="cpu",
    )
    micro = FrozenLocalResidualFieldOracleConsumer(
        FrozenOracleConsumerConfig(
            checkpoint=str(checkpoint),
            alpha=0.75,
            oracle_forward_microbatch_size=2,
        ),
        device="cpu",
    )
    common = {
        "points": batch["points"],
        "features": batch["features"],
        "lorentz_vectors": batch["lorentz_vectors"],
        "mask": batch["mask"],
        "tokens": batch["tokens"],
        "raw_mask": batch["raw_mask"],
        "indices": batch["indices"],
    }

    full_output = full(**common, predicted_fields=predicted_full)
    micro_output = micro(**common, predicted_fields=predicted_micro)

    assert torch.allclose(full_output.teacher_logits_pred, micro_output.teacher_logits_pred)
    full_output.teacher_logits_pred.sum().backward()
    micro_output.teacher_logits_pred.sum().backward()
    assert torch.allclose(predicted_full.grad, predicted_micro.grad)
    assert micro_output.diagnostics["oracle_forward_microbatch_size"] == 2


def test_step3_logit_only_fallback_uses_detached_cache_and_disables_predicted_branch(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(tagger_module, "build_hlt_classifier", fake_build_hlt_classifier)
    checkpoint = tmp_path / "Ofull" / "best_model_val.pt"
    _write_fake_oracle_checkpoint(checkpoint)
    dataset = _dataset("model_train", n_jets=4)
    batch = collate_local_particle_residual_field_batch([dataset[0], dataset[1]])
    cached = torch.randn(2, 3, requires_grad=True)
    predicted = batch["target_fields"].detach().clone().requires_grad_(True)
    wrapper = FrozenLocalResidualFieldOracleConsumer(
        FrozenOracleConsumerConfig(
            checkpoint=str(checkpoint),
            oracle_logit_only_fallback=True,
        ),
        device="cpu",
    )

    output = wrapper(
        points=batch["points"],
        features=batch["features"],
        lorentz_vectors=batch["lorentz_vectors"],
        mask=batch["mask"],
        true_fields=batch["target_fields"],
        predicted_fields=predicted,
        cached_true_logits=cached,
        cached_true_logits_metadata={
            "checkpoint_hash": wrapper.checkpoint_hash,
            "teacher_id": "Ofull",
            "model_config": {"oracle_field_alpha": 1.0},
        },
    )

    assert wrapper.model is None
    assert wrapper.parameters_frozen()
    assert torch.equal(output.teacher_logits_true, cached.detach())
    assert output.teacher_logits_true.requires_grad is False
    assert output.teacher_logits_pred is None
    assert output.diagnostics["oracle_logit_only_fallback"] is True
    assert output.diagnostics["oracle_input_gradient_distillation_enabled"] is False
    assert output.diagnostics["oracle_pred_logits_disabled_by_fallback"] is True
    assert output.diagnostics["teacher_logits_true_source"] == "cached"
    assert output.diagnostics["cached_true_logits_metadata_validated"] is True
    with pytest.raises(ValueError, match="requires cached_true_logits"):
        wrapper(
            points=batch["points"],
            features=batch["features"],
            lorentz_vectors=batch["lorentz_vectors"],
            mask=batch["mask"],
        )


def test_step3_rejects_nonfinite_alpha_nonoracle_checkpoint_and_attached_true_branch(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(tagger_module, "build_hlt_classifier", fake_build_hlt_classifier)
    with pytest.raises(ValueError, match="finite and non-negative"):
        FrozenOracleConsumerConfig(checkpoint="unused", alpha=float("nan"))

    checkpoint = tmp_path / "not_oracle" / "best_model_val.pt"
    _write_fake_oracle_checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["model_config"]["field_source"] = "zero"
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="requires an oracle field source"):
        FrozenLocalResidualFieldOracleConsumer(
            FrozenOracleConsumerConfig(checkpoint=str(checkpoint)),
            device="cpu",
        )

    valid_checkpoint = tmp_path / "Ofull" / "best_model_val.pt"
    _write_fake_oracle_checkpoint(valid_checkpoint)
    wrapper = FrozenLocalResidualFieldOracleConsumer(
        FrozenOracleConsumerConfig(checkpoint=str(valid_checkpoint)),
        device="cpu",
    )
    dataset = _dataset("model_train", n_jets=2)
    batch = collate_local_particle_residual_field_batch([dataset[0], dataset[1]])
    with pytest.raises(ValueError, match="must always be detached"):
        wrapper(
            points=batch["points"],
            features=batch["features"],
            lorentz_vectors=batch["lorentz_vectors"],
            mask=batch["mask"],
            true_fields=batch["target_fields"],
            detach_true=False,
        )
