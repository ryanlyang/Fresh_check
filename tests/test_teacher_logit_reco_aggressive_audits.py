import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from jetclass_fresh.fusion import PredictionBlock, save_prediction_block, softmax_np
from jetclass_fresh.fusion import STACK_SPLITS, prediction_paths
from jetclass_fresh.jetclass_data import JetIdentity
from teacher_logit_reco.aggressive_audits import (
    AggressiveAuditConfig,
    AggressiveReconstructionDiagnosticsAccumulator,
    run_aggressive_audit,
    summarize_prediction_arrays,
)
from teacher_logit_reco.crossarch_experiment import (
    aggressive_reco_domain_tagger_model_name,
    aggressive_reco_model_name,
    hlt_model_name,
)


class DummySoftView:
    def __init__(self):
        self.labels = np.asarray([0, 1], dtype=np.int64)
        self.aux = {
            "parent_weights": np.asarray([[1.0, 0.5, 0.0], [0.8, 0.2, 0.0]], dtype=np.float32),
            "sanitized_hlt_mask": np.asarray([[True, True, False], [True, True, False]]),
            "parent_delta": np.asarray(
                [
                    [[0.1, -0.2, 0.3, -0.4], [0.2, 0.1, -0.1, 0.0], [0.0, 0.0, 0.0, 0.0]],
                    [[-0.3, 0.0, 0.2, 0.1], [0.0, -0.1, 0.0, -0.2], [0.0, 0.0, 0.0, 0.0]],
                ],
                dtype=np.float32,
            ),
            "extra_weights": np.asarray([[0.1, 0.0], [0.4, 0.2]], dtype=np.float32),
            "extra_mask": np.asarray([[True, False], [True, True]]),
            "extra_weight_sum": np.asarray([0.1, 0.6], dtype=np.float32),
            "extra_pt_fraction": np.asarray([0.02, 0.12], dtype=np.float32),
            "extra_slot_usage": np.asarray([1, 2], dtype=np.float32),
            "extra_slot_usage_histogram": np.asarray([0, 1, 1], dtype=np.int64),
            "extra_slot_active_mask": np.asarray([[True, False], [True, True]]),
            "global_correction": {
                "logpt_scale": np.asarray([0.1, -0.2], dtype=np.float32),
                "loge_scale": np.asarray([0.05, -0.05], dtype=np.float32),
                "eta_shift": np.asarray([0.01, 0.02], dtype=np.float32),
                "phi_shift": np.asarray([-0.01, 0.03], dtype=np.float32),
            },
        }


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def write_prediction_metadata(root: Path, model_name: str, split: str, n_jets: int, *, diagnostics=True):
    _, metadata_path = prediction_paths(root, model_name, split)
    payload = {
        "n_jets": int(n_jets),
        "metrics": {"accuracy": 0.5},
        "hlt_content_hash": f"hlt-{split}",
    }
    if diagnostics:
        payload["reconstruction_diagnostics"] = {"batch_count": 1, "jet_count": int(n_jets)}
    write_json(metadata_path, payload)


class AggressiveAuditTests(unittest.TestCase):
    def test_accumulator_summarizes_parent_extra_and_global_controls(self):
        acc = AggressiveReconstructionDiagnosticsAccumulator()
        acc.update_from_soft_view(DummySoftView())
        report = acc.to_dict()

        self.assertEqual(report["batch_count"], 1)
        self.assertEqual(report["jet_count"], 2)
        self.assertEqual(report["parent_weight"]["count"], 4)
        self.assertAlmostEqual(report["extra_pt_fraction"]["mean"], 0.07, places=6)
        self.assertEqual(report["extra_slot_usage_histogram"], [0, 1, 1])
        self.assertEqual(report["extra_slot_activation_fraction_by_slot"], [1.0, 0.5])
        self.assertAlmostEqual(report["global_correction_abs"]["logpt_scale"]["mean"], 0.15, places=6)

    def test_run_aggressive_audit_accepts_hlt_without_reco_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prediction_dir = root / "predictions"
            reco_model_dir = root / "reco_models"
            adapted_dir = root / "adapted"
            output_dir = root / "audit"
            expected = {"stack_train": 3, "stack_val": 2, "final_test": 4}

            for arch in ("part", "pn", "pfn", "pcnn"):
                for split, n_jets in expected.items():
                    write_prediction_metadata(prediction_dir, hlt_model_name(arch), split, n_jets, diagnostics=False)

            frozen_name = aggressive_reco_model_name("aggt", "part")
            adapted_name = aggressive_reco_domain_tagger_model_name("aggt", "part")
            for split, n_jets in expected.items():
                write_prediction_metadata(prediction_dir, frozen_name, split, n_jets)
                write_prediction_metadata(prediction_dir, adapted_name, split, n_jets)

            write_json(
                reco_model_dir / "aggt" / "part" / "run_report.json",
                {
                    "best_epoch": 1,
                    "checkpoint": "best_model_val.pt",
                    "no_final_test_evaluation": True,
                    "config": {"train_split": "model_train", "val_split": "model_val"},
                },
            )
            write_json(
                adapted_dir / "aggt" / "part" / "run_report.json",
                {
                    "best_epoch": 1,
                    "checkpoint": "best_model_val.pt",
                    "no_final_test_evaluation": True,
                },
            )
            fusion_report = root / "fusion" / "fusion_report.json"
            write_json(fusion_report, {"groups": {"hlt4": {"ok": True}}})

            report = run_aggressive_audit(
                AggressiveAuditConfig(
                    prediction_dir=str(prediction_dir),
                    reco_model_dir=str(reco_model_dir),
                    adapted_tagger_dir=str(adapted_dir),
                    output_dir=str(output_dir),
                    fusion_report=str(fusion_report),
                    reconstructors=("aggt",),
                    teachers=("part",),
                    fusion_groups=("hlt4",),
                    expected_split_sizes=expected,
                )
            )

            self.assertTrue(report["ok"], report["flags"])
            self.assertEqual(report["error_count"], 0)
            self.assertEqual(report["warning_count"], 0)
            self.assertTrue((output_dir / "aggressive_audit_report.json").exists())
            self.assertTrue((output_dir / "aggressive_audit_summary.md").exists())

    def test_prediction_array_audit_loads_npz_and_checks_shapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logits = np.asarray([[1.0, 0.0], [0.2, 0.8]], dtype=np.float32)
            labels = np.asarray([0, 1], dtype=np.int64)
            save_prediction_block(
                PredictionBlock(
                    model_name="demo",
                    split="stack_val",
                    logits=logits,
                    probs=softmax_np(logits),
                    labels=labels,
                    jet_ids=[
                        JetIdentity(file="a.root", entry=0, label=0),
                        JetIdentity(file="a.root", entry=1, label=1),
                    ],
                    metadata={},
                ),
                root,
            )

            report, flags = summarize_prediction_arrays(
                prediction_dir=root,
                model_names=("demo",),
                splits=("stack_val",),
            )

            self.assertEqual(flags, [])
            self.assertTrue(report["demo"]["stack_val"]["loaded"])
            self.assertEqual(report["demo"]["stack_val"]["n_jets"], 2)
            self.assertTrue(report["demo"]["stack_val"]["logits_all_finite"])


if __name__ == "__main__":
    unittest.main()
