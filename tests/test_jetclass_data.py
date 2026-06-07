from pathlib import Path
import tempfile
import unittest

from jetclass_fresh.jetclass_data import (
    DEFAULT_SPLIT_SEEDS,
    FileRecord,
    LABEL_NAMES,
    audit_split_manifest,
    build_split_manifest_from_records,
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


if __name__ == "__main__":
    unittest.main()
