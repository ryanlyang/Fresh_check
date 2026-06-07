from pathlib import Path
import tempfile
import unittest

import numpy as np

from jetclass_fresh.audits import (
    AuditRunConfig,
    audit_file_split,
    audit_fusion_source,
    audit_hlt_sharing,
    audit_jet_identity_splits,
    block_shuffle_audit,
    holdout_stack_audit,
    permutation_label_audit,
    run_audit_suite,
)
from jetclass_fresh.fusion import PredictionBlock, softmax_np, save_prediction_block
from jetclass_fresh.hlt_baseline import save_json
from jetclass_fresh.jetclass_data import (
    FILE_PREFIX_TO_LABEL,
    LABEL_NAMES,
    FileRecord,
    JetIdentity,
    SplitManifest,
    save_split_manifest,
)


STACK_SPLITS = ["stack_train", "stack_val", "final_test"]
ALL_SPLITS = ["model_train", "model_val", "stack_train", "stack_val", "final_test"]


def make_manifest(*, shared_files: bool = True) -> SplitManifest:
    splits = {}
    file_records = []
    for split_index, split in enumerate(ALL_SPLITS):
        rows = []
        for entry in range(6):
            label = entry % 3
            file_name = f"class{label}_shared.root" if shared_files else f"{split}_class{label}.root"
            rows.append(JetIdentity(file=file_name, entry=split_index * 100 + entry, label=label))
            file_records.append(FileRecord(path=file_name, label=label, num_entries=1000))
        splits[split] = rows
    unique_records = {(record.path, record.label): record for record in file_records}
    return SplitManifest(
        data_dir="/tmp/jetclass",
        max_constits=128,
        class_names=list(LABEL_NAMES),
        file_prefix_to_label=dict(FILE_PREFIX_TO_LABEL),
        split_sizes={split: 6 for split in ALL_SPLITS},
        split_seeds={split: 100 + index for index, split in enumerate(ALL_SPLITS)},
        file_records=list(unique_records.values()),
        splits=splits,
        metadata={"file_level_separation_claimed": not shared_files},
    )


def make_block(model_name, split, labels, *, strength=4.0, offset=0, hlt_hash=None):
    labels = np.asarray(labels, dtype=np.int64)
    n_classes = int(np.max(labels)) + 1
    logits = np.full((len(labels), n_classes), -1.0, dtype=np.float32)
    for row, label in enumerate(labels):
        logits[row, int(label)] = strength
        logits[row] += 0.03 * np.sin(offset + row + np.arange(n_classes))
    jet_ids = [JetIdentity(file=f"{split}.root", entry=i, label=int(label)) for i, label in enumerate(labels)]
    return PredictionBlock(
        model_name=model_name,
        split=split,
        logits=logits,
        probs=softmax_np(logits),
        labels=labels,
        jet_ids=jet_ids,
        metadata={
            "model_kind": "synthetic",
            "hlt_content_hash": hlt_hash or f"hlt_hash_{split}",
            "allowed_inputs": "cached_fixed_hlt_only",
        },
    )


def write_hlt_metadata(cache_dir, split):
    save_json(
        Path(cache_dir) / f"{split}_fixed_hlt_metadata.json",
        {
            "split": split,
            "seed": 1055,
            "hlt_content_hash": f"hlt_hash_{split}",
        },
    )


def write_prediction_fixture(root, model_names=("model_a", "model_b"), n_rows=120):
    labels = np.asarray([0, 1, 2] * (n_rows // 3), dtype=np.int64)
    for split in STACK_SPLITS:
        for offset, model_name in enumerate(model_names):
            save_prediction_block(
                make_block(model_name, split, labels, strength=4.0 - offset, offset=offset * 11),
                root,
            )


def write_fusion_report(path, model_names):
    metrics = {
        split: {"accuracy": 1.0, "cross_entropy": 0.01, "n_jets": 120}
        for split in STACK_SPLITS
    }
    save_json(
        Path(path),
        {
            "model_names": list(model_names),
            "splits": list(STACK_SPLITS),
            "final_test_evaluated": True,
            "stacked_logistic_regression": {"metrics": metrics},
        },
    )


class AuditStep12Tests(unittest.TestCase):
    def test_file_and_jet_identity_audits_distinguish_file_overlap(self):
        manifest = make_manifest(shared_files=True)

        strict_file_audit = audit_file_split(manifest, require_disjoint=True)
        relaxed_file_audit = audit_file_split(manifest, require_disjoint=False)
        identity_audit = audit_jet_identity_splits(manifest)

        self.assertFalse(strict_file_audit["ok"])
        self.assertTrue(relaxed_file_audit["ok"])
        self.assertTrue(identity_audit["ok"])
        self.assertEqual(identity_audit["cross_split_overlap_count"], 0)

    def test_hlt_sharing_and_fusion_source_pass_on_allowed_prediction_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            prediction_dir = Path(tmp) / "predictions"
            cache_dir = Path(tmp) / "hlt_cache"
            for split in STACK_SPLITS:
                write_hlt_metadata(cache_dir, split)
            write_prediction_fixture(prediction_dir)

            sharing = audit_hlt_sharing(prediction_dir, ["model_a", "model_b"], hlt_cache_dir=cache_dir)
            source = audit_fusion_source(prediction_dir, ["model_a", "model_b"])

        self.assertTrue(sharing["ok"])
        self.assertTrue(source["ok"])
        self.assertEqual(source["feature_mode"], "logits_probs")

    def test_stack_sanity_audits_run_from_saved_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            prediction_dir = Path(tmp) / "predictions"
            write_prediction_fixture(prediction_dir)

            permutation = permutation_label_audit(
                prediction_dir,
                ["model_a", "model_b"],
                C_grid=[0.1],
                max_iter=80,
                accuracy_slack=0.20,
            )
            holdout = holdout_stack_audit(
                prediction_dir,
                ["model_a", "model_b"],
                C_grid=[0.1],
                max_iter=80,
                max_accuracy_gap=0.20,
            )
            shuffled = block_shuffle_audit(
                prediction_dir,
                ["model_a", "model_b"],
                C_grid=[0.1],
                max_iter=80,
            )

        self.assertIn("final_test", permutation["metrics"])
        self.assertTrue(holdout["ok"])
        self.assertIn("accuracy_delta_shuffled_minus_reference", shuffled)

    def test_run_audit_suite_writes_report(self):
        manifest = make_manifest(shared_files=True)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            manifest_path = tmp_path / "manifest.json"
            prediction_dir = tmp_path / "fusion" / "predictions"
            fusion_report = tmp_path / "fusion" / "fusion_report.json"
            cache_dir = tmp_path / "hlt_cache"
            output_dir = tmp_path / "audits"
            save_split_manifest(manifest, manifest_path, pretty=True)
            for split in STACK_SPLITS:
                write_hlt_metadata(cache_dir, split)
            write_prediction_fixture(prediction_dir)
            write_fusion_report(fusion_report, ["model_a", "model_b"])

            report = run_audit_suite(
                AuditRunConfig(
                    manifest_path=str(manifest_path),
                    prediction_dir=str(prediction_dir),
                    output_dir=str(output_dir),
                    hlt_cache_dir=str(cache_dir),
                    fusion_report_path=str(fusion_report),
                    require_file_disjoint=False,
                    C_grid=[0.1],
                    max_iter=80,
                    permutation_accuracy_slack=0.20,
                    holdout_max_accuracy_gap=0.20,
                )
            )
            self.assertTrue((output_dir / "audit_report.json").exists())

        self.assertIn("fusion_source", report["audits"])
        self.assertTrue(report["audits"]["file_split"]["ok"])


if __name__ == "__main__":
    unittest.main()
