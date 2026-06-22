import json
import tempfile
import unittest
from pathlib import Path

from teacher_logit_reco.set_matching.detr_slots.audit_report import (
    DETR_SLOT_AUDIT_REPORT_STEP,
    DetrSlotAuditReportConfig,
    build_detr_slot_audit_final_report,
    render_detr_slot_final_report_markdown,
)
from teacher_logit_reco.set_matching.detr_slots.experiment import DETR_SLOT_ENCODER_ARCHITECTURES
from teacher_logit_reco.set_matching.detr_slots.five_view import DETR_SLOT_FIVE_VIEW_TAGGER_VARIANTS


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def binary_metrics(auc, fpr30, fpr50):
    return {
        "auc": auc,
        "fpr_at_signal_eff_0p30": fpr30,
        "fpr_at_signal_eff_0p50": fpr50,
        "background_rejection_at_signal_eff_0p30": 1.0 / fpr30 if fpr30 else None,
        "background_rejection_at_signal_eff_0p50": 1.0 / fpr50 if fpr50 else None,
    }


def metric_block(acc, auc, fpr30=None, fpr50=None):
    if fpr30 is None:
        fpr30 = max(1.0 - auc, 0.001)
    if fpr50 is None:
        fpr50 = max(1.0 - auc + 0.01, 0.001)
    return {
        "accuracy": acc,
        "binary_metrics": binary_metrics(auc=auc, fpr30=fpr30, fpr50=fpr50),
    }


def write_reconstructor_reports(root: Path):
    for offset, architecture in enumerate(DETR_SLOT_ENCODER_ARCHITECTURES):
        total = 1.0 - 0.05 * offset
        write_json(
            root / "reconstructors" / architecture / "run_report.json",
            {
                "architecture": architecture,
                "best_epoch": 3 + offset,
                "epochs_completed": 5,
                "best_model_val_total_loss": total,
                "best_model_val_metrics": {
                    "total": total,
                    "matched_core_loss": total * 0.5,
                    "matched_aux_loss": total * 0.2,
                    "existence_loss": total * 0.1,
                    "count_loss": total * 0.05,
                    "metric_count_mae": 2.0 - 0.1 * offset,
                    "metric_existence_precision": 0.7 + 0.02 * offset,
                    "metric_existence_recall": 0.8 + 0.01 * offset,
                    "metric_matched_delta_r_p90": 0.4 - 0.02 * offset,
                    "metric_jet_sum_pt_relative_error_mean": 0.1,
                    "metric_jet_sum_energy_relative_error_mean": 0.2,
                },
                "uses_aux_logits_for_bce": True,
                "checkpoint": f"reconstructors/{architecture}/best_model_val.pt",
            },
        )


def write_cache_reports(root: Path):
    for offset, architecture in enumerate(DETR_SLOT_ENCODER_ARCHITECTURES):
        split_reports = {}
        for split in ("stack_train", "stack_val", "final_test"):
            split_reports[split] = {
                "array_path": f"reconstructed_views/{architecture}/{split}_reconstructed_view.npz",
                "n_jets": 1000,
                "n_candidates": 128,
                "candidate_count_summary": {"mean": 128.0},
                "exported_tokens_summary": {"mean": 80.0 + offset},
                "top_existence_score_mean": 0.9,
                "nonfinite_count": 0,
                "heldout_detr_slot_metrics": {
                    "total": 1.1 - 0.04 * offset,
                    "matched_core_loss": 0.5 - 0.02 * offset,
                    "matched_aux_loss": 0.2 - 0.01 * offset,
                    "metric_count_mae": 2.2 - 0.1 * offset,
                    "metric_existence_precision": 0.75 + 0.02 * offset,
                    "metric_existence_recall": 0.82 + 0.01 * offset,
                    "metric_matched_delta_r_p90": 0.45 - 0.02 * offset,
                },
            }
        write_json(
            root / "reconstructed_views" / architecture / "cache_report.json",
            {
                "architecture": architecture,
                "splits": list(split_reports),
                "split_reports": split_reports,
            },
        )


def write_tagger_reports(root: Path):
    scores = {
        "hlt_only": 0.70,
        "hlt_plus_gt": 0.72,
        "hlt_plus_pn": 0.71,
        "hlt_plus_pfn": 0.69,
        "hlt_plus_pcnn": 0.74,
        "five_view_plain": 0.76,
        "five_view_geometry": 0.78,
        "five_view_no_confidence": 0.75,
        "view_label_shuffle_control": 0.79,
    }
    fpr50 = {
        "hlt_only": 0.20,
        "hlt_plus_gt": 0.14,
        "hlt_plus_pn": 0.15,
        "hlt_plus_pfn": 0.18,
        "hlt_plus_pcnn": 0.12,
        "five_view_plain": 0.08,
        "five_view_geometry": 0.10,
        "five_view_no_confidence": 0.13,
        "view_label_shuffle_control": 0.01,
    }
    for variant in DETR_SLOT_FIVE_VIEW_TAGGER_VARIANTS:
        auc = scores[variant]
        fpr = fpr50[variant]
        write_json(
            root / "taggers" / variant / "run_report.json",
            {
                "best_epoch": 4,
                "selection_metric": "fpr_at_signal_eff_0p50",
                "selection_metric_direction": "lower",
                "best_model_selection_metric_value": fpr,
                "best_stack_val_metrics": metric_block(acc=auc - 0.03, auc=auc - 0.01, fpr50=fpr + 0.01),
                "final_test_metrics": metric_block(acc=auc - 0.02, auc=auc, fpr50=fpr),
                "final_test_evaluated": True,
                "config": {
                    "drop_views": [],
                    "use_confidence": variant != "five_view_no_confidence",
                    "use_geometry_attention": variant == "five_view_geometry",
                    "shuffle_view_labels": variant == "view_label_shuffle_control",
                },
            },
        )


def write_offline_reference(root: Path):
    write_json(
        root / "offline_teacher_reference" / "offline_part" / "run_report.json",
        {
            "best_epoch": 10,
            "best_model_val_accuracy": 0.95,
            "evaluations": {
                "stack_val": {"metrics": metric_block(acc=0.9, auc=0.96, fpr50=0.04)},
                "final_test": {"metrics": metric_block(acc=0.91, auc=0.97, fpr50=0.03)},
            },
        },
    )


def write_audit_report(root: Path):
    write_json(
        root / "ablations" / "five_view_ablation_eval" / "run_report.json",
        {
            "ok": True,
            "summary_csv": str(root / "ablations" / "five_view_ablation_eval" / "summary.csv"),
            "skipped": [{"variant": "missing_debug_checkpoint"}],
        },
    )
    summary = root / "ablations" / "five_view_ablation_eval" / "summary.csv"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text("variant,split,accuracy\nfive_view_plain,final_test,0.9\n", encoding="utf-8")


def write_full_fixture(root: Path):
    write_reconstructor_reports(root)
    write_cache_reports(root)
    write_tagger_reports(root)
    write_offline_reference(root)
    write_audit_report(root)


class DetrFreeSlotStep15AuditReportTests(unittest.TestCase):
    def test_final_report_answers_core_questions_and_writes_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "experiment"
            write_full_fixture(root)

            report = build_detr_slot_audit_final_report(
                DetrSlotAuditReportConfig(
                    output_dir=str(root / "final_report"),
                    experiment_dir=str(root),
                    confirm_final_test=True,
                )
            )

            self.assertTrue(report["ok"])
            self.assertEqual(report["experiment_step"], DETR_SLOT_AUDIT_REPORT_STEP)
            self.assertTrue(report["comparison_summary"]["free_slot_improved_over_hlt_only"])
            self.assertEqual(report["comparison_summary"]["comparison_metric"], "final_test_fpr_at_signal_eff_0p50")
            self.assertEqual(report["comparison_summary"]["comparison_direction"], "lower")
            self.assertEqual(report["comparison_summary"]["answered_questions"]["which_encoder_helped_most"], "hlt_plus_pcnn")
            self.assertTrue(report["comparison_summary"]["five_view_beat_every_single_view"])
            self.assertEqual(report["comparison_summary"]["best_overall_tagger"]["variant"], "five_view_plain")
            self.assertEqual(report["comparison_summary"]["best_five_view"]["variant"], "five_view_plain")
            self.assertAlmostEqual(report["comparison_summary"]["best_overall_tagger"]["comparison_value"], 0.08)
            self.assertAlmostEqual(report["offline_reference_comparison"][0]["final_test_auc"], 0.97)
            self.assertAlmostEqual(report["offline_reference_comparison"][0]["final_test_fpr_at_signal_eff_0p50"], 0.03)
            self.assertTrue(report["five_view_audit_summary"]["exists"])
            self.assertEqual(report["five_view_audit_summary"]["n_summary_rows"], 1)
            self.assertEqual(report["five_view_audit_summary"]["skipped_specs"][0]["variant"], "missing_debug_checkpoint")
            self.assertEqual(len(report["reconstructor_summary"]), 4)
            self.assertEqual(len(report["tagger_summary"]), len(DETR_SLOT_FIVE_VIEW_TAGGER_VARIANTS))
            self.assertTrue((root / "final_report" / "detr_slot_final_report.json").exists())
            self.assertTrue((root / "final_report" / "detr_slot_final_report.md").exists())
            self.assertTrue((root / "final_report" / "tagger_summary.csv").exists())
            self.assertTrue((root / "final_report" / "binary_operating_points.csv").exists())
            self.assertIn("Did free-slot reconstruction improve over HLT-only?", render_detr_slot_final_report_markdown(report))

    def test_missing_required_artifacts_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "experiment"
            write_full_fixture(root)
            (root / "taggers" / "hlt_plus_pn" / "run_report.json").unlink()

            report = build_detr_slot_audit_final_report(
                DetrSlotAuditReportConfig(
                    output_dir=str(root / "final_report"),
                    experiment_dir=str(root),
                    confirm_final_test=True,
                )
            )

            self.assertFalse(report["ok"])
            self.assertTrue(any("hlt_plus_pn" in problem for problem in report["problems"]))


if __name__ == "__main__":
    unittest.main()
