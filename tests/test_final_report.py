from pathlib import Path
import tempfile
import unittest

from jetclass_fresh.final_report import (
    FinalReportConfig,
    build_final_report_summary,
    render_final_report_markdown,
    write_final_report,
)
from jetclass_fresh.hlt_baseline import save_json


def model_val_report(step, *, accuracy, loss, role=None):
    report = {
        "experiment_step": step,
        "best_epoch": 3,
        "best_model_val_accuracy": accuracy,
        "best_model_val_loss": loss,
        "epochs_completed": 3,
        "checkpoint": "best_model_val.pt",
        "no_final_test_evaluation": True,
    }
    if role:
        report["reference_role"] = role
    return report


def metric(accuracy, cross_entropy, n_jets=100):
    return {"accuracy": accuracy, "cross_entropy": cross_entropy, "n_jets": n_jets}


def fusion_report(model_names, *, stack_accuracy, stack_ce):
    return {
        "model_names": list(model_names),
        "splits": ["stack_train", "stack_val", "final_test"],
        "final_test_evaluated": True,
        "single_models": {
            "final_test": {
                "hlt_baseline": metric(stack_accuracy - 0.08, stack_ce + 0.18),
            }
        },
        "uniform_probability_average": {
            "final_test": metric(stack_accuracy - 0.03, stack_ce + 0.06),
        },
        "weighted_probability_average": {
            "metrics": {
                "final_test": metric(stack_accuracy - 0.02, stack_ce + 0.04),
            }
        },
        "weighted_logit_average": {
            "metrics": {
                "final_test": metric(stack_accuracy - 0.01, stack_ce + 0.02),
            }
        },
        "stacked_logistic_regression": {
            "metrics": {
                "stack_val": metric(stack_accuracy - 0.01, stack_ce + 0.03, n_jets=40),
                "final_test": metric(stack_accuracy, stack_ce),
            }
        },
    }


def audit_report(ok=True):
    return {
        "ok": ok,
        "audits": {
            "file_split": {"ok": ok},
            "jet_identity": {"ok": ok},
            "hlt_sharing": {"ok": ok},
            "fusion_source": {"ok": ok},
        },
    }


def write_fixture(root, *, audits_ok=True):
    root = Path(root)
    paths = {
        "hlt": root / "hlt.json",
        "offline": root / "offline.json",
        "reco7": root / "reco7_fusion.json",
        "hlt5": root / "hlt5_fusion.json",
        "reco7_audit": root / "reco7_audit.json",
        "hlt5_audit": root / "hlt5_audit.json",
    }
    save_json(paths["hlt"], model_val_report("step5_single_hlt_baseline", accuracy=0.62, loss=1.1))
    save_json(
        paths["offline"],
        model_val_report(
            "step6_offline_teacher_reference",
            accuracy=0.84,
            loss=0.55,
            role="offline_upper_reference_only",
        ),
    )
    save_json(paths["hlt5"], fusion_report(["hlt_seed101", "hlt_seed202"], stack_accuracy=0.70, stack_ce=0.90))
    save_json(paths["reco7"], fusion_report(["hlt_baseline", "m2_base"], stack_accuracy=0.725, stack_ce=0.84))
    save_json(paths["reco7_audit"], audit_report(audits_ok))
    save_json(paths["hlt5_audit"], audit_report(audits_ok))
    return paths


class FinalReportStep13Tests(unittest.TestCase):
    def test_summary_states_reco7_substantially_stronger_when_audits_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_fixture(tmp, audits_ok=True)
            summary = build_final_report_summary(
                FinalReportConfig(
                    output_dir=str(Path(tmp) / "out"),
                    hlt_baseline_report=str(paths["hlt"]),
                    offline_teacher_report=str(paths["offline"]),
                    reco7_fusion_report=str(paths["reco7"]),
                    hlt5_fusion_report=str(paths["hlt5"]),
                    reco7_audit_report=str(paths["reco7_audit"]),
                    hlt5_audit_report=str(paths["hlt5_audit"]),
                    substantial_accuracy_delta=0.01,
                )
            )

        self.assertEqual(summary["interpretation"]["state"], "supports_reco7_stronger")
        self.assertAlmostEqual(summary["interpretation"]["accuracy_delta_reco7_minus_hlt5"], 0.025)
        self.assertTrue(summary["audit_outcomes"]["all_configured_audits_ok"])
        self.assertEqual(summary["model_val_references"]["offline_teacher"]["reference_role"], "offline_upper_reference_only")

    def test_failed_audit_prevents_interpretation(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_fixture(tmp, audits_ok=False)
            summary = build_final_report_summary(
                FinalReportConfig(
                    output_dir=str(Path(tmp) / "out"),
                    hlt_baseline_report=str(paths["hlt"]),
                    offline_teacher_report=str(paths["offline"]),
                    reco7_fusion_report=str(paths["reco7"]),
                    hlt5_fusion_report=str(paths["hlt5"]),
                    reco7_audit_report=str(paths["reco7_audit"]),
                    hlt5_audit_report=str(paths["hlt5_audit"]),
                )
            )

        self.assertEqual(summary["interpretation"]["state"], "audit_failed")
        self.assertEqual(summary["interpretation"]["claim"], "not_interpretable_yet")

    def test_write_final_report_saves_markdown_and_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            paths = write_fixture(tmp_path, audits_ok=True)
            result = write_final_report(
                FinalReportConfig(
                    output_dir=str(tmp_path / "final"),
                    hlt_baseline_report=str(paths["hlt"]),
                    offline_teacher_report=str(paths["offline"]),
                    reco7_fusion_report=str(paths["reco7"]),
                    hlt5_fusion_report=str(paths["hlt5"]),
                    reco7_audit_report=str(paths["reco7_audit"]),
                    hlt5_audit_report=str(paths["hlt5_audit"]),
                )
            )
            markdown = Path(result["markdown_path"]).read_text(encoding="utf-8")

            self.assertTrue(Path(result["json_path"]).exists())
            self.assertIn("JetClass Same-HLT Fresh-Start Report", markdown)
            self.assertIn("HLT5 stacked logistic final_test", markdown)

    def test_missing_inputs_can_write_incomplete_draft(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = FinalReportConfig(
                output_dir=str(Path(tmp) / "out"),
                hlt_baseline_report=str(Path(tmp) / "missing_hlt.json"),
                offline_teacher_report=str(Path(tmp) / "missing_offline.json"),
                reco7_fusion_report=str(Path(tmp) / "missing_reco7.json"),
                hlt5_fusion_report=str(Path(tmp) / "missing_hlt5.json"),
                reco7_audit_report=str(Path(tmp) / "missing_audit.json"),
                hlt5_audit_report=None,
                allow_missing=True,
            )
            summary = build_final_report_summary(cfg)
            markdown = render_final_report_markdown(summary)

        self.assertEqual(summary["interpretation"]["state"], "incomplete")
        self.assertIn("Missing Inputs", markdown)


if __name__ == "__main__":
    unittest.main()
