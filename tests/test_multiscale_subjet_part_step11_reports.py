import json
import tempfile
import unittest
from pathlib import Path

from jetclass_fresh.hlt_cache import fixed_hlt_params_dict, fixed_hlt_params_from_strength

from teacher_logit_reco.multiscale_subjet_part import (
    MULTISCALE_SUBJET_BASELINE_VARIANT,
    MULTISCALE_SUBJET_DEFAULT_VARIANTS,
    MULTISCALE_SUBJET_PRIMARY_METRIC,
    MULTISCALE_SUBJET_PRIMARY_VARIANT,
    MULTISCALE_SUBJET_REPORT_CONTRACT,
    MULTISCALE_SUBJET_REPORT_STEP,
    MultiScaleSubjetReportConfig,
    build_multiscale_subjet_part_report,
)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def hlt_metadata(n_jets=12):
    return {
        "n_jets": int(n_jets),
        "source_view": "hlt",
        "label_counts": {"QCD": n_jets // 2, "Hgg": n_jets - n_jets // 2},
        "hlt_content_hash": "hlt-hash",
        "jet_identity_hash": "jet-hash",
        "hlt_params": fixed_hlt_params_dict(fixed_hlt_params_from_strength(0.6)),
        "hlt_protocol_audit": {"ok": True, "problems": []},
    }


def metrics(*, fpr50, auc=0.96, accuracy=0.88):
    return {
        "loss": 0.3,
        "accuracy": float(accuracy),
        "n_jets": 12,
        "binary_metrics": {
            "auc": float(auc),
            "fpr_at_signal_eff_0p30": float(fpr50) / 2.0,
            "fpr_at_signal_eff_0p50": float(fpr50),
            "background_rejection_at_signal_eff_0p50": 1.0 / max(float(fpr50), 1.0e-12),
            "validation_threshold_final_test_fpr": float(fpr50) * 1.05,
            "validation_threshold_final_test_signal_efficiency": 0.5,
        },
        "diagnostics": {
            "subjet_assignment_entropy_mean": 1.2,
            "subjet_valid_fraction": 0.95,
            "readback_gamma_F": 0.1,
            "readback_feature_delta_norm_mean": 0.03,
            "subjet_pair_bias_abs_mean": 0.2,
            "fusion_attention_entropy": 0.4,
        },
        "hlt_degradation_slice_metrics": [
            {
                "slice": "high_particle_loss",
                "n_jets": 4,
                "fpr_at_signal_eff_0p50": float(fpr50) * 1.2,
            }
        ],
    }


def child_report(variant, *, fpr50):
    return {
        "experiment_step": "multiscale_subjet_part_step10_train",
        "variant": variant,
        "best_epoch": 7,
        "epochs_completed": 10,
        "selection_metric": MULTISCALE_SUBJET_PRIMARY_METRIC,
        "selection_metric_direction": "minimize",
        "best_model_selection_metric_value": fpr50,
        "checkpoint": "best_model_val.pt",
        "num_classes": 2,
        "label_names": ["QCD", "Hgg"],
        "label_filter": [0, 1],
        "inference_consumes_hlt_only": True,
        "model_config": {
            "uses_reference_part_backbone": variant != "subjet_branch_only",
            "baseline_recoverable_at_zero_gamma": variant == MULTISCALE_SUBJET_PRIMARY_VARIANT,
            "query_mode": "seeded",
        },
        "best_model_val_metrics": metrics(fpr50=fpr50 * 1.1, auc=0.95, accuracy=0.86),
        "stack_val_metrics": metrics(fpr50=fpr50 * 1.03, auc=0.955, accuracy=0.87),
        "final_test_metrics": metrics(fpr50=fpr50, auc=0.97, accuracy=0.89),
        "train_dataset": hlt_metadata(),
        "val_dataset": hlt_metadata(),
        "stack_val_dataset": hlt_metadata(),
        "final_test_dataset": hlt_metadata(),
        "runtime": {"elapsed_seconds": 123.0, "elapsed_minutes": 2.05, "seconds_per_completed_epoch": 12.3},
        "walltime_seconds": 123.0,
    }


class MultiscaleSubjetPartStep11ReportTests(unittest.TestCase):
    def test_report_ranks_by_final_test_fpr50_and_compares_primary_to_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fprs = {
                MULTISCALE_SUBJET_BASELINE_VARIANT: 0.042,
                MULTISCALE_SUBJET_PRIMARY_VARIANT: 0.036,
                "pure_perceiver_latent_control": 0.050,
                "part_plus_random_subjet_control": 0.061,
            }
            for variant in MULTISCALE_SUBJET_DEFAULT_VARIANTS:
                write_json(root / variant / "run_report.json", child_report(variant, fpr50=fprs[variant]))

            report = build_multiscale_subjet_part_report(
                MultiScaleSubjetReportConfig(
                    experiment_dir=str(root),
                    output_dir=str(root / "final_report"),
                    include_parameter_counts=False,
                    confirm_final_test=True,
                )
            )

            self.assertTrue(report["ok"], report["problems"])
            self.assertEqual(report["contract"], MULTISCALE_SUBJET_REPORT_CONTRACT)
            self.assertEqual(report["experiment_step"], MULTISCALE_SUBJET_REPORT_STEP)
            summary = report["comparison_summary"]
            self.assertEqual(summary["primary_metric"], MULTISCALE_SUBJET_PRIMARY_METRIC)
            self.assertEqual(summary["primary_metric_direction"], "minimize")
            self.assertEqual(summary["best_variant"], MULTISCALE_SUBJET_PRIMARY_VARIANT)
            self.assertTrue(summary["primary_beats_baseline"])
            self.assertAlmostEqual(summary["primary_improvement_vs_baseline"], 0.006)
            self.assertTrue((root / "final_report" / "metric_table.csv").exists())
            self.assertTrue((root / "final_report" / "diagnostics.csv").exists())
            self.assertGreater(report["diagnostic_summary"]["assignment_rows"], 0)
            self.assertGreater(report["diagnostic_summary"]["residual_readback_rows"], 0)
            self.assertGreater(report["hlt_degradation_summary"]["behavioral_slice_rows"], 0)

    def test_report_supports_nested_evaluations_metrics_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = child_report(MULTISCALE_SUBJET_PRIMARY_VARIANT, fpr50=0.025)
            nested.pop("final_test_metrics")
            nested["evaluations"] = {"final_test": {"metrics": metrics(fpr50=0.025)}}
            write_json(root / MULTISCALE_SUBJET_PRIMARY_VARIANT / "run_report.json", nested)

            report = build_multiscale_subjet_part_report(
                MultiScaleSubjetReportConfig(
                    experiment_dir=str(root),
                    output_dir=str(root / "final_report"),
                    variants=(MULTISCALE_SUBJET_PRIMARY_VARIANT,),
                    baseline_variant=MULTISCALE_SUBJET_PRIMARY_VARIANT,
                    primary_variant=MULTISCALE_SUBJET_PRIMARY_VARIANT,
                    include_parameter_counts=False,
                    strict_protocol=False,
                    require_all_default_variants=False,
                    confirm_final_test=True,
                )
            )

            self.assertTrue(report["ok"], report["problems"])
            self.assertEqual(report["comparison_summary"]["best_metric_value"], 0.025)
            row = report["metric_table"][0]
            self.assertEqual(row["final_test_fpr_at_signal_eff_0p50"], 0.025)

    def test_strict_report_fails_when_required_controls_are_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root / MULTISCALE_SUBJET_BASELINE_VARIANT / "run_report.json", child_report(MULTISCALE_SUBJET_BASELINE_VARIANT, fpr50=0.05))
            write_json(root / MULTISCALE_SUBJET_PRIMARY_VARIANT / "run_report.json", child_report(MULTISCALE_SUBJET_PRIMARY_VARIANT, fpr50=0.04))

            report = build_multiscale_subjet_part_report(
                MultiScaleSubjetReportConfig(
                    experiment_dir=str(root),
                    output_dir=str(root / "final_report"),
                    include_parameter_counts=False,
                    confirm_final_test=True,
                )
            )

            self.assertFalse(report["ok"])
            self.assertTrue(any("missing required protocol variants" in problem for problem in report["problems"]))


if __name__ == "__main__":
    unittest.main()
