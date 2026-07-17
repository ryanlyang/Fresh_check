from __future__ import annotations

import pytest

from jetclass_fresh.jetclass_data import (
    BALANCED_SPLIT_ROW_ORDERING,
    FileRecord,
    JetIdentity,
    SPLIT_ORDER,
    SplitManifest,
    manifest_hash,
    save_split_manifest,
)
from scripts.build_constrained_coarse_to_fine_calibration_slice import (
    CALIBRATION_SLICE_CONTRACT,
    build_calibration_manifest,
)
from scripts.validate_constrained_coarse_to_fine_calibration_slice import _validate_manifest


def _parent_manifest() -> SplitManifest:
    splits: dict[str, list[JetIdentity]] = {split: [] for split in SPLIT_ORDER}
    for split, rows_per_label in (("model_train", 10), ("model_val", 6)):
        for index in range(rows_per_label):
            for label in range(2):
                splits[split].append(JetIdentity(file=f"class{label}_{split}.root", entry=index, label=label))
    records = [
        FileRecord(path=f"class{label}_{split}.root", label=label, num_entries=20)
        for split in ("model_train", "model_val")
        for label in range(2)
    ]
    return SplitManifest(
        data_dir="toy",
        max_constits=8,
        class_names=["class0", "class1"],
        file_prefix_to_label={"class0": 0, "class1": 1},
        split_sizes={split: len(rows) for split, rows in splits.items()},
        split_seeds={split: index for index, split in enumerate(SPLIT_ORDER)},
        file_records=records,
        splits=splits,
        metadata={"row_ordering": BALANCED_SPLIT_ROW_ORDERING},
    )


def test_calibration_manifest_is_balanced_and_hash_bound(tmp_path) -> None:
    parent = _parent_manifest()
    calibration = build_calibration_manifest(parent, model_train_size=12, model_val_size=8, seed=9)

    assert calibration.split_sizes == {
        "model_train": 12,
        "model_val": 8,
        "stack_train": 0,
        "stack_val": 0,
        "final_test": 0,
    }
    assert calibration.metadata["contract"] == CALIBRATION_SLICE_CONTRACT
    assert calibration.metadata["parent_manifest_hash"] == manifest_hash(parent)
    assert calibration.metadata["selected_label_counts"] == {
        "model_train": {"0": 6, "1": 6},
        "model_val": {"0": 4, "1": 4},
    }
    assert calibration.splits["stack_train"] == []
    assert calibration.splits["final_test"] == []

    parent_path = tmp_path / "parent.json.gz"
    calibration_path = tmp_path / "calibration.json.gz"
    save_split_manifest(parent, parent_path)
    save_split_manifest(calibration, calibration_path)
    _, validated, calibration_sha = _validate_manifest(parent_path, calibration_path)
    assert calibration_sha == manifest_hash(validated)


def test_calibration_manifest_rejects_unbalanced_request_or_parent_drift(tmp_path) -> None:
    parent = _parent_manifest()
    with pytest.raises(ValueError, match="divisible"):
        build_calibration_manifest(parent, model_train_size=11, model_val_size=8)

    calibration = build_calibration_manifest(parent, model_train_size=12, model_val_size=8)
    parent_path = tmp_path / "parent.json.gz"
    calibration_path = tmp_path / "calibration.json.gz"
    save_split_manifest(parent, parent_path)
    calibration.metadata["parent_manifest_hash"] = "stale"
    save_split_manifest(calibration, calibration_path)
    with pytest.raises(ValueError, match="parent manifest hash"):
        _validate_manifest(parent_path, calibration_path)
