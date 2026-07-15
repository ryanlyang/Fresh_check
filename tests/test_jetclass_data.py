from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from jetclass_fresh.jetclass_data import (
    DEFAULT_SPLIT_SEEDS,
    FileRecord,
    LABEL_NAMES,
    audit_split_manifest,
    build_split_manifest_from_records,
    discover_file_records,
    label_from_filename,
    load_split_manifest,
    manifest_hash,
    save_split_manifest,
    split_summary,
)


class JetClassDataStep2Tests(unittest.TestCase):
    def test_filename_mapping_handles_ttbarlep_before_ttbar(self):
        self.assertEqual(label_from_filename("TTBar_001.root"), 8)
        self.assertEqual(label_from_filename("TTBarLep_001.root"), 9)
        self.assertEqual(label_from_filename("/data/ZJetsToNuNu_000.root"), 0)

    def test_balanced_split_manifest_has_no_identity_overlap(self):
        prefixes = [
            "ZJetsToNuNu",
            "HToBB",
            "HToCC",
            "HToGG",
            "HToWW4Q",
            "HToWW2Q1L",
            "ZToQQ",
            "WToQQ",
            "TTBar",
            "TTBarLep",
        ]
        records = [
            FileRecord(path=f"{prefix}_000.root", label=label, num_entries=20)
            for label, prefix in enumerate(prefixes)
        ]
        split_sizes = {
            "model_train": 20,
            "model_val": 10,
            "stack_train": 10,
            "stack_val": 10,
            "final_test": 20,
        }

        manifest = build_split_manifest_from_records(
            records,
            data_dir="/tmp/jetclass",
            split_sizes=split_sizes,
            split_seeds=DEFAULT_SPLIT_SEEDS,
        )
        audit = audit_split_manifest(manifest)
        self.assertTrue(audit["ok"])
        self.assertEqual(audit["cross_split_overlap_count"], 0)
        self.assertEqual(audit["duplicate_within_split_count"], 0)
        self.assertEqual(audit["split_counts"], split_sizes)

        summary = split_summary(manifest)
        for split, counts in summary["class_counts"].items():
            expected_per_class = split_sizes[split] // len(LABEL_NAMES)
            self.assertTrue(all(value == expected_per_class for value in counts.values()))

    def test_manifest_save_load_roundtrip_json_gz(self):
        records = [
            FileRecord(path=f"class{label}.root", label=label, num_entries=10)
            for label in range(len(LABEL_NAMES))
        ]
        split_sizes = {
            "model_train": 10,
            "model_val": 10,
            "stack_train": 10,
            "stack_val": 10,
            "final_test": 10,
        }
        manifest = build_split_manifest_from_records(
            records,
            data_dir="/tmp/jetclass",
            split_sizes=split_sizes,
            split_seeds=DEFAULT_SPLIT_SEEDS,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "split_manifest.json.gz"
            save_split_manifest(manifest, path)
            loaded = load_split_manifest(path)
        self.assertEqual(manifest_hash(manifest), manifest_hash(loaded))

    def test_discover_file_records_retries_transient_root_open_failure(self):
        class FakeTree:
            num_entries = 12

        class FakeHandle:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def __getitem__(self, _tree_name):
                return FakeTree()

        class FakeUproot:
            attempts = 0

            @classmethod
            def open(cls, _path):
                cls.attempts += 1
                if cls.attempts == 1:
                    raise OverflowError("transient incomplete ROOT read")
                return FakeHandle()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = root / "ZJetsToNuNu_good.root"
            good.touch()
            with patch.dict(sys.modules, {"uproot": FakeUproot}):
                with patch("jetclass_fresh.jetclass_data.time.sleep") as sleep:
                    records = discover_file_records(root, require_all_classes=False)
        self.assertEqual(len(records), 1)
        self.assertEqual(FakeUproot.attempts, 2)
        sleep.assert_called_once()

    def test_discover_file_records_reports_unreadable_root_path(self):
        class FakeUproot:
            @staticmethod
            def open(path):
                raise OverflowError("fake corrupted ROOT payload")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad = root / "ZJetsToNuNu_bad.root"
            bad.touch()
            with patch.dict(sys.modules, {"uproot": FakeUproot}):
                with patch("jetclass_fresh.jetclass_data.time.sleep"):
                    with self.assertRaisesRegex(RuntimeError, "ZJetsToNuNu_bad.root"):
                        discover_file_records(root, require_all_classes=False)

    def test_discover_file_records_can_skip_unreadable_roots_explicitly(self):
        class FakeTree:
            num_entries = 12

        class FakeHandle:
            def __init__(self, path):
                self.path = Path(path)

            def __enter__(self):
                if "bad" in self.path.name:
                    raise OverflowError("fake corrupted ROOT payload")
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def __getitem__(self, _tree_name):
                return FakeTree()

        class FakeUproot:
            @staticmethod
            def open(path):
                return FakeHandle(path)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "ZJetsToNuNu_bad.root").touch()
            (root / "HToBB_good.root").touch()
            with patch.dict(sys.modules, {"uproot": FakeUproot}):
                with self.assertWarnsRegex(RuntimeWarning, "ZJetsToNuNu_bad.root"):
                    records = discover_file_records(
                        root,
                        require_all_classes=False,
                        skip_unreadable=True,
                    )
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].path, "HToBB_good.root")


if __name__ == "__main__":
    unittest.main()
