from __future__ import annotations

from pathlib import Path

from jetclass_fresh.jetclass_data import JetIdentity, SplitManifest, save_split_manifest
from teacher_logit_reco.set_matching.label_filters import label_names_to_manifest_indices


def test_label_filter_names_use_compact_manifest_labels(tmp_path: Path) -> None:
    manifest_path = tmp_path / "split_manifest.json.gz"
    manifest = SplitManifest(
        data_dir=str(tmp_path),
        max_constits=128,
        class_names=["QCD", "Hgg"],
        file_prefix_to_label={"ZJetsToNuNu": 0, "HToGG": 1},
        split_sizes={"model_train": 2, "model_val": 2, "stack_train": 2, "stack_val": 2, "final_test": 2},
        split_seeds={"model_train": 1, "model_val": 2, "stack_train": 3, "stack_val": 4, "final_test": 5},
        file_records=[],
        splits={
            split: [
                JetIdentity(file="ZJetsToNuNu_000.root", entry=0, label=0),
                JetIdentity(file="HToGG_000.root", entry=0, label=1),
            ]
            for split in ("model_train", "model_val", "stack_train", "stack_val", "final_test")
        },
        metadata={"source_to_filtered_label": {"0": 0, "3": 1}},
    )
    save_split_manifest(manifest, manifest_path)

    assert label_names_to_manifest_indices(["QCD", "Hgg"], manifest_path=manifest_path) == (0, 1)


def test_label_filter_names_fallback_to_global_jetclass_labels() -> None:
    assert label_names_to_manifest_indices(["QCD", "Hgg"], manifest_path=None) == (0, 3)
