import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from jetclass_fresh.fusion import PredictionBlock, save_prediction_block, softmax_np
from jetclass_fresh.jetclass_data import JetIdentity
from teacher_logit_reco.hlt_self_dualview import (
    HLTSDVReportConfig,
    HLT_SDV_BINARY_TABLE_CSV,
    HLT_SDV_COMPARISON_TABLE_CSV,
    HLT_SDV_METRIC_TABLE_CSV,
    HLT_SDV_REPORT_CONTRACT,
    HLT_SDV_REPORT_JSON,
    HLT_SDV_REPORT_MD,
    HLT_SDV_REPORT_RUN_JSON,
    HLT_SDV_REPORT_SUMMARY_JSON,
    HLT_SDV_VARIANT_HLT2_ONLY,
    HLT_SDV_VARIANT_SAME_VIEW,
    HLT_SDV_VARIANT_TTA,
    build_hlt_sdv_required_variants,
    hlt_sdv_dual_hlt2_variant_name,
    write_hlt_sdv_report,
)
from teacher_logit_reco.privileged_distill_10class.config import (
    PD10_DEFAULT_ALPHA,
    PD10_DEFAULT_TEMPERATURE,
    PD10_STUDENT_INIT_WARM_START,
    PD10_TARGET_FULL_LOGITS,
    PD10_TEACHER_DUAL_VIEW,
    PD10_TEACHER_HLT,
    PD10_TEACHER_NONE,
    pd10_extended_teacher_model_name,
    pd10_student_variant_name,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_SCRIPT = REPO_ROOT / "scripts" / "write_pd10_hlt_self_dualview_report.py"


def _labels(n: int = 50) -> np.ndarray:
    return np.asarray([index % 10 for index in range(n)], dtype=np.int64)


def _jet_ids(labels: np.ndarray, split: str) -> list[JetIdentity]:
    return [
        JetIdentity(file=f"{split}_class{int(label)}.root", entry=index, label=int(label))
        for index, label in enumerate(labels)
    ]


def _logits(labels: np.ndarray, *, correct: int, margin: float) -> np.ndarray:
    logits = np.full((len(labels), 10), -1.5, dtype=np.float32)
    for row, label in enumerate(labels):
        pred = int(label) if row < int(correct) else (int(label) + 1) % 10
        logits[row, pred] = float(margin)
        logits[row, int(label)] += 0.2
    return logits


def _write_block(prediction_dir: Path, model_name: str, split: str, *, correct: int, margin: float) -> None:
    labels = _labels()
    logits = _logits(labels, correct=correct, margin=margin)
    block = PredictionBlock(
        model_name=model_name,
        split=split,
        logits=logits,
        probs=softmax_np(logits),
        labels=labels,
        jet_ids=_jet_ids(labels, split),
        metadata={
            "model_name": model_name,
            "split": split,
            "unit_test": True,
            "requires_offline_inputs": False,
            "requires_teacher_features": False,
        },
    )
    save_prediction_block(block, prediction_dir, overwrite=True)


def _write_run_report(path: Path, *, variant: str, final_accuracy_hint: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "ok": True,
                "variant_name": variant,
                "model_name": variant,
                "selection_metric": "model_val_cross_entropy",
                "best_epoch": 2,
                "requires_offline_inputs": False,
                "requires_teacher_features": False,
                "requires_deterministic_hlt2_transform": True,
                "no_final_test_used_for_selection": True,
                "final_test_metrics": {"accuracy": final_accuracy_hint},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_sdv_variant(root: Path, variant: str, *, val_correct: int, final_correct: int, val_margin: float) -> None:
    variant_dir = root / "hlt_self_dualview" / "models" / variant
    _write_run_report(variant_dir / "run_report.json", variant=variant, final_accuracy_hint=final_correct / 50.0)
    _write_block(variant_dir / "predictions", variant, "model_val", correct=val_correct, margin=val_margin)
    _write_block(variant_dir / "predictions", variant, "final_test", correct=final_correct, margin=2.0)


def _write_teacher_anchor(root: Path) -> None:
    model_name = pd10_extended_teacher_model_name(PD10_TEACHER_HLT)
    teacher_dir = root / "teachers" / model_name
    teacher_dir.mkdir(parents=True, exist_ok=True)
    (teacher_dir / "run_report.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    _write_block(root / "teacher_logits", model_name, "model_val", correct=32, margin=1.8)
    _write_block(root / "teacher_logits", model_name, "final_test", correct=32, margin=1.8)


def _write_student_anchor(root: Path, alias: str, variant: str, *, val_correct: int, final_correct: int) -> None:
    student_dir = root / "students" / variant
    _write_run_report(student_dir / "run_report.json", variant=variant, final_accuracy_hint=final_correct / 50.0)
    _write_block(student_dir / "student_predictions", variant, "model_val", correct=val_correct, margin=1.8)
    _write_block(student_dir / "student_predictions", variant, "final_test", correct=final_correct, margin=1.8)
    del alias


def _populate_report_inputs(root: Path) -> tuple[str, ...]:
    _write_teacher_anchor(root)
    warm_ce = pd10_student_variant_name(PD10_STUDENT_INIT_WARM_START, PD10_TEACHER_NONE)
    warm_dual = pd10_student_variant_name(
        PD10_STUDENT_INIT_WARM_START,
        PD10_TEACHER_DUAL_VIEW,
        PD10_TARGET_FULL_LOGITS,
        temperature=PD10_DEFAULT_TEMPERATURE,
        kd_alpha=PD10_DEFAULT_ALPHA,
    )
    _write_student_anchor(root, "warm_start_ce_only", warm_ce, val_correct=34, final_correct=34)
    _write_student_anchor(root, "v1_warm_dual_view_kd", warm_dual, val_correct=36, final_correct=36)
    variants = build_hlt_sdv_required_variants((0.00, 0.10, 0.20, 0.35, 1.00))
    for variant in variants:
        if variant == HLT_SDV_VARIANT_SAME_VIEW:
            _write_sdv_variant(root, variant, val_correct=35, final_correct=35, val_margin=1.5)
        elif variant == hlt_sdv_dual_hlt2_variant_name(0.10):
            _write_sdv_variant(root, variant, val_correct=37, final_correct=37, val_margin=1.7)
        elif variant == hlt_sdv_dual_hlt2_variant_name(0.20):
            _write_sdv_variant(root, variant, val_correct=39, final_correct=39, val_margin=2.4)
        elif variant == hlt_sdv_dual_hlt2_variant_name(0.35):
            _write_sdv_variant(root, variant, val_correct=38, final_correct=38, val_margin=1.9)
        elif variant == hlt_sdv_dual_hlt2_variant_name(1.00):
            _write_sdv_variant(root, variant, val_correct=36, final_correct=36, val_margin=1.6)
        elif variant == HLT_SDV_VARIANT_HLT2_ONLY:
            _write_sdv_variant(root, variant, val_correct=33, final_correct=33, val_margin=1.3)
        elif variant == HLT_SDV_VARIANT_TTA:
            _write_sdv_variant(root, variant, val_correct=34, final_correct=34, val_margin=1.4)
    return variants


def load_script(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class HLTSDVStep7ReportTest(unittest.TestCase):
    def test_final_test_guard(self):
        with self.assertRaisesRegex(ValueError, "requires confirm_final_test"):
            HLTSDVReportConfig(pd10_root="pd10", confirm_final_test=False)

    def test_report_answers_headline_questions_and_writes_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "pd10"
            variants = _populate_report_inputs(root)

            report = write_hlt_sdv_report(
                HLTSDVReportConfig(
                    pd10_root=str(root),
                    variants=variants,
                    confirm_final_test=True,
                    require_anchors=True,
                )
            )

            self.assertTrue(report["ok"], report.get("problems"))
            self.assertEqual(report["contract"], HLT_SDV_REPORT_CONTRACT)
            answers = report["answers"]
            self.assertEqual(answers["best_sdv_selected_by_model_val_ce"], hlt_sdv_dual_hlt2_variant_name(0.20))
            self.assertEqual(answers["best_sdv_hlt2_strength_by_model_val_ce"], 0.20)
            self.assertAlmostEqual(answers["best_sdv_final_test_accuracy"], 39 / 50)
            self.assertTrue(answers["did_hlt_hlt2_beat_hlt_part"])
            self.assertTrue(answers["did_hlt_hlt2_beat_warm_start_ce_only"])
            self.assertTrue(answers["did_hlt_hlt2_beat_hlt_hlt_same_view"])
            self.assertTrue(answers["did_hlt_hlt2_beat_tta_averaging"])
            self.assertFalse(answers["did_final_test_winner_use_final_test_information_for_selection"])
            self.assertGreater(len(report["binary_projection_rows"]), 0)
            self.assertGreaterEqual(len(report["comparison_rows"]), len(variants))
            self.assertTrue(
                any(
                    row["candidate_name"] == HLT_SDV_VARIANT_HLT2_ONLY
                    and row["baseline_name"] == "warm_start_ce_only"
                    for row in report["comparison_rows"]
                )
            )
            selected_row = next(
                row
                for row in report["metric_rows"]
                if row["name"] == hlt_sdv_dual_hlt2_variant_name(0.20) and row["split"] == "final_test"
            )
            self.assertIn("per_class_metrics", selected_row["metrics"])
            self.assertIn("confusion_matrix", selected_row["metrics"])
            output_dir = root / "hlt_self_dualview" / "final_report"
            for filename in (
                HLT_SDV_REPORT_SUMMARY_JSON,
                HLT_SDV_REPORT_JSON,
                HLT_SDV_REPORT_MD,
                HLT_SDV_REPORT_RUN_JSON,
                HLT_SDV_METRIC_TABLE_CSV,
                HLT_SDV_COMPARISON_TABLE_CSV,
                HLT_SDV_BINARY_TABLE_CSV,
            ):
                self.assertTrue((output_dir / filename).exists(), filename)
            markdown = (output_dir / HLT_SDV_REPORT_MD).read_text(encoding="utf-8")
            self.assertIn("Did HLT+HLT2 beat HLT ParT?", markdown)

    def test_cli_build_config_defaults(self):
        module = load_script(REPORT_SCRIPT, "write_pd10_hlt_self_dualview_report")
        with tempfile.TemporaryDirectory() as tmp:
            pd10_root = Path(tmp) / "pd10"
            args = module.parse_args(["--pd10-root", str(pd10_root), "--confirm-final-test"])
            config = module.build_config(args)
            self.assertEqual(Path(config.pd10_root), pd10_root)
            self.assertTrue(config.confirm_final_test)
            self.assertIn(hlt_sdv_dual_hlt2_variant_name(0.20), config.variants)


if __name__ == "__main__":
    unittest.main()
