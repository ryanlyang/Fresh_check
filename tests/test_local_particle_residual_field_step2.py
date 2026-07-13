from __future__ import annotations

import numpy as np
import pytest

from jetclass_fresh.jetclass_data import JetIdentity, JetView
from teacher_logit_reco.local_particle_residual_field import (
    LocalParticleResidualFieldCache,
    LocalParticleResidualFieldDataset,
    LocalParticleResidualFieldDatasetConfig,
    TeacherLogitBlock,
    collate_local_particle_residual_field_batch,
)


def _view_and_cache(*, include_oracle_fields: bool = False):
    tokens = np.zeros((3, 4, 14), dtype=np.float32)
    mask = np.zeros((3, 4), dtype=bool)
    labels = np.asarray([0, 1, 2], dtype=np.int64)
    mask[:, :2] = True
    tokens[:, :2, 0] = np.asarray([[10.0, 5.0], [8.0, 4.0], [6.0, 3.0]], dtype=np.float32)
    tokens[:, :2, 1] = 0.1
    tokens[:, :2, 2] = 0.2
    tokens[:, :2, 3] = tokens[:, :2, 0] * 1.1
    jet_ids = tuple(JetIdentity(file=f"file_{idx}.root", entry=idx, label=int(label)) for idx, label in enumerate(labels))
    view = JetView(
        tokens=tokens,
        mask=mask,
        labels=labels,
        jet_ids=jet_ids,
        split="model_train",
        metadata={
            "view": "fixed_hlt",
            "hlt_content_hash": "hlt_hash",
            "jet_identity_hash": "identity_hash",
            "source_manifest_hash": "manifest_hash",
        },
    )
    fields = np.zeros((3, 4, 5), dtype=np.float32)
    fields[:, :2, :] = 0.25
    fields[:, 2:, :] = 99.0
    cache = LocalParticleResidualFieldCache(
        target_fields=fields,
        target_mask=mask.copy(),
        labels=labels.copy(),
        jet_ids=jet_ids,
        field_names=tuple(f"field_{idx}" for idx in range(5)),
        field_groups={"all": [0, 1, 2, 3, 4]},
        radii=(0.02,),
        split="model_train",
        metadata={
            "target_content_hash": "target_hash",
            "hlt_content_hash": "hlt_hash",
            "offline_content_hash": "offline_hash",
            "jet_identity_hash": "identity_hash",
            "source_manifest_hash": "manifest_hash",
            "zero_baseline_metrics": {"mae": 0.25},
        },
    )
    config = LocalParticleResidualFieldDatasetConfig(
        hlt_cache_dir="unused",
        target_cache_dir="unused",
        split="model_train",
        include_oracle_fields=include_oracle_fields,
    )
    return view, cache, config


def test_step2_dataset_masks_invalid_residual_fields_and_returns_oracle_alias():
    view, cache, config = _view_and_cache(include_oracle_fields=True)
    dataset = LocalParticleResidualFieldDataset(view, cache, config=config)

    sample = dataset[0]

    assert len(dataset) == 3
    assert sample["target_fields"].shape == (4, 5)
    assert np.all(sample["target_fields"][2:] == 0.0)
    assert "oracle_fields" in sample
    assert np.array_equal(sample["oracle_fields"], sample["target_fields"])
    assert dataset.metadata["target_mask_matches_hlt_mask"] is True


def test_step2_dataset_rejects_mask_mismatch():
    view, cache, config = _view_and_cache()
    bad_mask = cache.target_mask.copy()
    bad_mask[0, 2] = True
    bad_cache = LocalParticleResidualFieldCache(
        target_fields=cache.target_fields,
        target_mask=bad_mask,
        labels=cache.labels,
        jet_ids=cache.jet_ids,
        field_names=cache.field_names,
        field_groups=cache.field_groups,
        radii=cache.radii,
        split=cache.split,
        metadata=cache.metadata,
    )

    with pytest.raises(ValueError, match="particle mask"):
        LocalParticleResidualFieldDataset(view, bad_cache, config=config)


def test_step2_collate_returns_part_inputs_targets_and_teacher_logits():
    view, cache, config = _view_and_cache(include_oracle_fields=True)
    teacher = TeacherLogitBlock(logits=np.ones((3, 10), dtype=np.float32), labels=view.labels)
    dataset = LocalParticleResidualFieldDataset(view, cache, config=config, teacher_logits=teacher)

    batch = collate_local_particle_residual_field_batch([dataset[0], dataset[1]])

    assert batch["tokens"].shape == (2, 4, 14)
    assert batch["features"].shape[0] == 2
    assert batch["target_fields"].shape == (2, 4, 5)
    assert batch["target_features"].shape == (2, 5, 4)
    assert batch["target_mask"].shape == (2, 4)
    assert batch["oracle_fields"].shape == (2, 4, 5)
    assert batch["teacher_logits"].shape == (2, 10)
    assert bool((batch["target_fields"][:, 2:, :] == 0.0).all().item())
