import json
from pathlib import Path
import tempfile
import unittest

from teacher_logit_reco.privileged_distill_10class import (
    PD10_DEFAULT_ALPHA,
    PD10_DEFAULT_REPRESENTATION_BETA,
    PD10_DEFAULT_TEMPERATURE,
    PD10_REPRESENTATION_DIM,
    PD10_REPRESENTATION_MODE_COSINE,
    PD10_REPORT_TABLES,
    PD10_STUDENT_INIT_WARM_START,
    PD10_TARGET_FULL_LOGITS,
    PD10_TARGET_FULL_LOGITS_PLUS_REP,
    PD10_TARGET_REP_ONLY,
    PD10_TEACHER_DUAL_VIEW,
    PD10_TEACHER_HLT,
    PD10_TEACHER_NONE,
    PD10_TEACHER_OFFLINE,
    PD10_TEACHER_PARTICLE_DUAL_VIEW,
    PD10ReportConfig,
    pd10_extended_student_variant_name,
    pd10_extended_teacher_model_name,
    write_pd10_report,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_teacher_report(root: Path, target: str, *, final_acc: float) -> None:
    model_name = pd10_extended_teacher_model_name(target)
    write_json(
        root / "teachers" / model_name / "run_report.json",
        {
            "ok": True,
            "teacher_target": target,
            "model_name": model_name,
            "best_model_val_metrics": {
                "accuracy": final_acc - 0.01,
                "cross_entropy": 1.01 - final_acc,
                "n_jets": 100,
            },
            "final_test_metrics": {
                "accuracy": final_acc,
                "cross_entropy": 1.0 - final_acc,
                "macro_ovr_auc": final_acc + 0.10,
                "n_jets": 100,
                "validation_threshold_fpr_at_signal_eff_0p50_macro": 0.20 - final_acc / 10.0,
            },
        },
    )


def write_student_report(
    root: Path,
    *,
    name: str,
    init_mode: str,
    teacher_target: str,
    target_mode: str,
    final_acc: float,
    uses_representations: bool = False,
    uses_logit_teacher: bool | None = None,
) -> None:
    if uses_logit_teacher is None:
        uses_logit_teacher = teacher_target != PD10_TEACHER_NONE and target_mode != PD10_TARGET_REP_ONLY
    metrics = {
        "accuracy": final_acc,
        "ce_loss": 1.0 - final_acc,
        "macro_ovr_auc": final_acc + 0.10,
        "kd_loss": 0.03 if uses_logit_teacher else 0.0,
        "rep_loss": 0.11 if uses_representations else 0.0,
        "effective_kd_alpha": PD10_DEFAULT_ALPHA if uses_logit_teacher else 0.0,
        "effective_representation_beta": PD10_DEFAULT_REPRESENTATION_BETA if uses_representations else 0.0,
        "n_jets": 100,
        "validation_threshold_fpr_at_signal_eff_0p50_macro": 0.18 - final_acc / 10.0,
    }
    write_json(
        root / "students" / name / "run_report.json",
        {
            "ok": True,
            "variant_name": name,
            "student_init": init_mode,
            "teacher_target": teacher_target,
            "target_mode": target_mode,
            "temperature": PD10_DEFAULT_TEMPERATURE,
            "kd_alpha": PD10_DEFAULT_ALPHA if uses_logit_teacher else 0.0,
            "top_k": 3,
            "representation_beta": PD10_DEFAULT_REPRESENTATION_BETA if uses_representations else 0.0,
            "representation_dim": PD10_REPRESENTATION_DIM,
            "representation_mode": PD10_REPRESENTATION_MODE_COSINE if uses_representations else "none",
            "uses_logit_teacher": bool(uses_logit_teacher),
            "uses_representations": bool(uses_representations),
            "best_model_val_metrics": {**metrics, "accuracy": final_acc - 0.01},
            "final_test_metrics": metrics,
            "teacher_logits_train_time_only": True,
            "teacher_representations_train_time_only": True,
            "inference_requires_teacher_logits": False,
            "inference_requires_teacher_representations": False,
            "inference_requires_offline_inputs": False,
            "inference_export_requires_teacher_features": False,
        },
    )


class PD10V2Step5ReportTests(unittest.TestCase):
    def test_report_discovers_v2_rows_and_compares_against_v04_anchors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for target, acc in (
                (PD10_TEACHER_HLT, 0.720),
                (PD10_TEACHER_OFFLINE, 0.820),
                (PD10_TEACHER_DUAL_VIEW, 0.815),
                (PD10_TEACHER_PARTICLE_DUAL_VIEW, 0.830),
            ):
                write_teacher_report(root, target, final_acc=acc)

            warm_ce = pd10_extended_student_variant_name(
                PD10_STUDENT_INIT_WARM_START,
                PD10_TEACHER_NONE,
                PD10_TARGET_FULL_LOGITS,
            )
            warm_dual = pd10_extended_student_variant_name(
                PD10_STUDENT_INIT_WARM_START,
                PD10_TEACHER_DUAL_VIEW,
                PD10_TARGET_FULL_LOGITS,
            )
            particle_logit = pd10_extended_student_variant_name(
                PD10_STUDENT_INIT_WARM_START,
                PD10_TEACHER_PARTICLE_DUAL_VIEW,
                PD10_TARGET_FULL_LOGITS,
            )
            particle_rep = pd10_extended_student_variant_name(
                PD10_STUDENT_INIT_WARM_START,
                PD10_TEACHER_PARTICLE_DUAL_VIEW,
                PD10_TARGET_REP_ONLY,
            )
            particle_logit_rep = pd10_extended_student_variant_name(
                PD10_STUDENT_INIT_WARM_START,
                PD10_TEACHER_PARTICLE_DUAL_VIEW,
                PD10_TARGET_FULL_LOGITS_PLUS_REP,
            )
            logit_fusion_rep = pd10_extended_student_variant_name(
                PD10_STUDENT_INIT_WARM_START,
                PD10_TEACHER_DUAL_VIEW,
                PD10_TARGET_FULL_LOGITS_PLUS_REP,
            )

            write_student_report(
                root,
                name=warm_ce,
                init_mode=PD10_STUDENT_INIT_WARM_START,
                teacher_target=PD10_TEACHER_NONE,
                target_mode=PD10_TARGET_FULL_LOGITS,
                final_acc=0.740,
                uses_logit_teacher=False,
            )
            write_student_report(
                root,
                name=warm_dual,
                init_mode=PD10_STUDENT_INIT_WARM_START,
                teacher_target=PD10_TEACHER_DUAL_VIEW,
                target_mode=PD10_TARGET_FULL_LOGITS,
                final_acc=0.760,
            )
            write_student_report(
                root,
                name=particle_logit,
                init_mode=PD10_STUDENT_INIT_WARM_START,
                teacher_target=PD10_TEACHER_PARTICLE_DUAL_VIEW,
                target_mode=PD10_TARGET_FULL_LOGITS,
                final_acc=0.768,
            )
            write_student_report(
                root,
                name=particle_rep,
                init_mode=PD10_STUDENT_INIT_WARM_START,
                teacher_target=PD10_TEACHER_PARTICLE_DUAL_VIEW,
                target_mode=PD10_TARGET_REP_ONLY,
                final_acc=0.755,
                uses_representations=True,
                uses_logit_teacher=False,
            )
            write_student_report(
                root,
                name=particle_logit_rep,
                init_mode=PD10_STUDENT_INIT_WARM_START,
                teacher_target=PD10_TEACHER_PARTICLE_DUAL_VIEW,
                target_mode=PD10_TARGET_FULL_LOGITS_PLUS_REP,
                final_acc=0.785,
                uses_representations=True,
            )
            write_student_report(
                root,
                name=logit_fusion_rep,
                init_mode=PD10_STUDENT_INIT_WARM_START,
                teacher_target=PD10_TEACHER_DUAL_VIEW,
                target_mode=PD10_TARGET_FULL_LOGITS_PLUS_REP,
                final_acc=0.772,
                uses_representations=True,
            )

            cfg = PD10ReportConfig(
                output_dir=str(root / "final_report"),
                teachers_dir=str(root / "teachers"),
                students_dir=str(root / "students"),
                include_priority_students=False,
                require_core_students=False,
                require_teacher_reports=False,
                require_audit=False,
                include_prediction_metrics=False,
                confirm_final_test=True,
            )
            report = write_pd10_report(cfg)

            self.assertTrue(report["ok"], report["problems"])
            self.assertIn("v2_comparisons", PD10_REPORT_TABLES)
            self.assertTrue((root / "final_report" / PD10_REPORT_TABLES["v2_comparisons"]).exists())
            self.assertIn(PD10_TEACHER_PARTICLE_DUAL_VIEW, {row["teacher_target"] for row in report["teacher_metric_rows"]})

            v2_student_rows = [
                row for row in report["student_metric_rows"] if row.get("split") == "final_test" and row.get("group") == "v2"
            ]
            self.assertEqual({row["variant"] for row in v2_student_rows}, {particle_logit, particle_rep, particle_logit_rep, logit_fusion_rep})
            logit_rep_row = next(row for row in v2_student_rows if row["variant"] == particle_logit_rep)
            self.assertTrue(logit_rep_row["uses_representations"])
            self.assertFalse(logit_rep_row["inference_requires_teacher_representations"])
            self.assertAlmostEqual(logit_rep_row["rep_loss"], 0.11)

            self.assertEqual(report["answers"]["best_v2_student_variant"], particle_logit_rep)
            self.assertTrue(report["answers"]["did_best_v2_beat_warm_start_ce_only"])
            self.assertTrue(report["answers"]["did_best_v2_beat_v04_warm_dual_view_kd"])
            self.assertAlmostEqual(report["answers"]["particle_dual_view_teacher_final_test_accuracy"], 0.830)

            anchor_rows = [
                row
                for row in report["v2_comparison_rows"]
                if row.get("candidate_source") == particle_logit_rep
                and row.get("comparison_type") == "student_vs_v04_anchor"
            ]
            self.assertEqual({row["baseline_source"] for row in anchor_rows}, {warm_ce, warm_dual})
            self.assertTrue(all(row["beats_baseline"] for row in anchor_rows))
            self.assertTrue(
                any(
                    row.get("comparison_type") == "teacher_vs_anchor"
                    and row.get("candidate_source") == PD10_TEACHER_PARTICLE_DUAL_VIEW
                    and row.get("baseline_source") == PD10_TEACHER_DUAL_VIEW
                    for row in report["v2_comparison_rows"]
                )
            )

            markdown = (root / "final_report" / "pd10_report.md").read_text(encoding="utf-8")
            self.assertIn("## V2", markdown)
            self.assertIn("Best V2 student", markdown)


if __name__ == "__main__":
    unittest.main()
