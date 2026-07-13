from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from jetclass_fresh.jetclass_data import JetIdentity, JetView
from teacher_logit_reco.local_particle_residual_field import (
    LocalParticleResidualFieldCache,
    LocalParticleResidualFieldDataset,
    LocalParticleResidualFieldDatasetConfig,
    LocalResidualFieldReconstructorConfig,
    LocalResidualReconstructorTrainConfig,
    build_local_residual_field_reconstructor,
    compute_local_residual_reconstruction_loss,
    resolve_local_residual_field_indices,
    train_local_residual_reconstructor,
)
import teacher_logit_reco.local_particle_residual_field.train as train_module


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
        ),
    )


def test_step4_field_subset_resolves_groups_and_names():
    selected = resolve_local_residual_field_indices(
        field_names=FIELD_NAMES,
        field_groups=FIELD_GROUPS,
        subset=("pt_density", "local_reliability_score"),
    )

    assert selected == (0, 1, 5)
    radius_selected = resolve_local_residual_field_indices(
        field_names=FIELD_NAMES,
        field_groups=FIELD_GROUPS,
        subset=("r0p02.*",),
    )
    assert radius_selected == (0, 1, 2, 3)
    radius_prefix_selected = resolve_local_residual_field_indices(
        field_names=FIELD_NAMES,
        field_groups=FIELD_GROUPS,
        subset=("r0p02",),
    )
    assert radius_prefix_selected == (0, 1, 2, 3)
    with pytest.raises(ValueError, match="unknown field subset"):
        resolve_local_residual_field_indices(
            field_names=FIELD_NAMES,
            field_groups=FIELD_GROUPS,
            subset=("not_a_field",),
        )


def test_step4_loss_uses_mask_field_subset_uncertainty_and_consistency():
    model = build_local_residual_field_reconstructor(
        LocalResidualFieldReconstructorConfig(
            variant="C6",
            field_dim=len(FIELD_NAMES),
            d_model=24,
            num_heads=3,
            num_layers=1,
            field_names=FIELD_NAMES,
            field_groups=FIELD_GROUPS,
        )
    )
    dataset = _dataset("model_train", n_jets=3)
    from teacher_logit_reco.local_particle_residual_field import collate_local_particle_residual_field_batch

    batch = collate_local_particle_residual_field_batch([dataset[0], dataset[1]])
    output = model(batch["tokens"], batch["raw_mask"])

    loss, metrics = compute_local_residual_reconstruction_loss(
        output,
        batch,
        field_names=FIELD_NAMES,
        field_groups=FIELD_GROUPS,
        selected_indices=(0, 1, 2, 3),
        field_group_weights={"pt_density": 2.0},
        consistency_loss_weight=0.5,
    )

    assert torch.isfinite(loss)
    assert metrics["n_selected_fields"] == 4
    assert metrics["n_valid_particles"] == 6
    assert "pt_density" in metrics["per_group"]
    assert "reliability" not in metrics["per_group"]
    assert metrics["consistency_loss"] >= 0.0


def test_step4_train_smoke_writes_checkpoint_and_reports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    datasets = {
        "model_train": _dataset("model_train", n_jets=8),
        "model_val": _dataset("model_val", n_jets=4),
        "stack_val": _dataset("stack_val", n_jets=4),
    }

    def fake_load_dataset(config, split, *, max_jets):
        dataset = datasets[str(split)]
        if max_jets is None:
            return dataset
        return _dataset(str(split), n_jets=min(int(max_jets), len(dataset)))

    monkeypatch.setattr(train_module, "_load_dataset", fake_load_dataset)
    config = LocalResidualReconstructorTrainConfig(
        output_dir=str(tmp_path / "run"),
        hlt_cache_dir="unused_hlt",
        target_cache_dir="unused_targets",
        variant="C0",
        d_model=24,
        num_heads=3,
        num_layers=1,
        context_layers=1,
        batch_size=4,
        eval_batch_size=4,
        epochs=2,
        lr=1.0e-3,
        num_workers=0,
        device="cpu",
        amp=False,
        early_stop_patience=-1,
        field_subset=("pt_density", "multiplicity"),
        field_group_weights={"pt_density": 1.5},
    )

    report = train_local_residual_reconstructor(config)

    output_dir = tmp_path / "run"
    assert report["ok"] is True
    assert report["best_epoch"] >= 0
    assert (output_dir / "best_model_val.pt").exists()
    assert (output_dir / "last.pt").exists()
    assert (output_dir / "run_report.json").exists()
    assert (output_dir / "training_curves.json").exists()
    assert (output_dir / "diagnostics" / "epoch_metrics.csv").exists()
    saved = json.loads((output_dir / "run_report.json").read_text(encoding="utf-8"))
    assert saved["selected_field_names"] == [
        "r0p02.delta_log_pt_sum",
        "r0p02.delta_pt_frac",
        "r0p02.delta_log_n",
        "r0p02.missing_n_frac",
    ]
