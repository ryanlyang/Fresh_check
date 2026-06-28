import tempfile
import unittest
from pathlib import Path

import numpy as np

from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.hlt_cache import fixed_hlt_params_dict, fixed_hlt_params_from_strength
from jetclass_fresh.jetclass_data import JetIdentity, JetView, RAW_TOKEN_DIM

from teacher_logit_reco.multiscale_subjet_part import (
    MULTISCALE_SUBJET_HLT_DEGRADATION_STRENGTH,
    MULTISCALE_SUBJET_PRIMARY_METRIC,
    MULTISCALE_SUBJET_TRAIN_STEP,
    MultiScaleSubjetTrainConfig,
    compare_hlt_params_for_multiscale_subjet,
    train_multiscale_subjet_tagger,
)
from teacher_logit_reco.subtoken_part.train import SubtokenHLTJetDataset

torch = require_torch()


class TinyRawTokenClassifier(torch.nn.Module):
    """Small model with the raw-token API used by the Step 10 trainer."""

    output_contract = "tiny_raw_token_classifier_for_tests"

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)

    def to_config_dict(self):
        return {"kind": "tiny_raw_token_classifier_for_tests"}

    def forward(self, tokens, mask, *, return_outputs=False, return_diagnostics=False):
        mask_f = mask.to(dtype=tokens.dtype)
        denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean_pt = (tokens[:, :, 0] * mask_f).sum(dim=1, keepdim=True) / denom
        mean_energy = (tokens[:, :, 3] * mask_f).sum(dim=1, keepdim=True) / denom
        valid_frac = mask_f.mean(dim=1, keepdim=True)
        logits = self.linear(torch.cat([mean_pt, mean_energy, valid_frac], dim=1))
        if return_outputs:
            return TinyOutput(logits)
        if return_diagnostics:
            return logits, TinyOutput(logits).diagnostics()
        return logits


class TinyOutput:
    def __init__(self, logits):
        self.logits = logits

    def diagnostics(self):
        return {"tiny_logit_abs_mean": self.logits.detach().abs().mean()}


def make_view(split: str, *, n_per_class: int = 8, hlt_strength: float = MULTISCALE_SUBJET_HLT_DEGRADATION_STRENGTH):
    labels = np.asarray([0] * n_per_class + [1] * n_per_class, dtype=np.int64)
    n_jets = int(labels.shape[0])
    tokens = np.zeros((n_jets, 6, RAW_TOKEN_DIM), dtype=np.float32)
    mask = np.zeros((n_jets, 6), dtype=bool)
    for index, label in enumerate(labels):
        mask[index, :3] = True
        base_pt = 6.0 if int(label) == 0 else 24.0
        tokens[index, :3, 0] = base_pt + np.asarray([0.0, 1.0, 2.0], dtype=np.float32)
        tokens[index, :3, 1] = np.asarray([0.0, 0.05, -0.04], dtype=np.float32)
        tokens[index, :3, 2] = np.asarray([0.0, 0.02, -0.02], dtype=np.float32)
        tokens[index, :3, 3] = tokens[index, :3, 0] + 2.0
        tokens[index, :3, 5] = 1.0
    jet_ids = [JetIdentity(file=f"{split}.root", entry=index, label=int(label)) for index, label in enumerate(labels)]
    return JetView(
        tokens=tokens,
        mask=mask,
        labels=labels,
        jet_ids=jet_ids,
        split=split,
        metadata={
            "view": "fixed_hlt",
            "hlt_params": fixed_hlt_params_dict(fixed_hlt_params_from_strength(float(hlt_strength))),
            "hlt_content_hash": f"{split}_content",
            "jet_identity_hash": f"{split}_identity",
        },
    )


def make_dataset(split: str):
    return SubtokenHLTJetDataset(
        make_view(split),
        label_filter=(0, 1),
        label_names=("QCD", "Hgg"),
    )


class MultiscaleSubjetPartStep10TrainTests(unittest.TestCase):
    def test_config_enforces_frozen_protocol(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "confirm_split_settings"):
                MultiScaleSubjetTrainConfig(output_dir=tmpdir, hlt_cache_dir="cache")
            with self.assertRaisesRegex(ValueError, "model_train"):
                MultiScaleSubjetTrainConfig(
                    output_dir=tmpdir,
                    hlt_cache_dir="cache",
                    train_split="stack_train",
                    confirm_split_settings=True,
                )
            with self.assertRaisesRegex(ValueError, "selects checkpoints"):
                MultiScaleSubjetTrainConfig(
                    output_dir=tmpdir,
                    hlt_cache_dir="cache",
                    selection_metric="accuracy",
                    confirm_split_settings=True,
                )

    def test_hlt_param_audit_accepts_0p6_and_rejects_1p0(self):
        ok = compare_hlt_params_for_multiscale_subjet(
            fixed_hlt_params_dict(fixed_hlt_params_from_strength(0.6)),
            expected_strength=0.6,
        )
        bad = compare_hlt_params_for_multiscale_subjet(
            fixed_hlt_params_dict(fixed_hlt_params_from_strength(1.0)),
            expected_strength=0.6,
        )

        self.assertTrue(ok["ok"])
        self.assertFalse(bad["ok"])
        self.assertGreater(len(bad["problems"]), 0)

    def test_training_loop_selects_model_val_then_evaluates_guarded_splits(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MultiScaleSubjetTrainConfig(
                output_dir=str(Path(tmpdir) / "run"),
                hlt_cache_dir="unused_in_injected_test",
                variant="subjet_branch_only",
                confirm_split_settings=True,
                confirm_final_test=True,
                seed=123,
                batch_size=4,
                eval_batch_size=4,
                epochs=3,
                lr=1.0e-2,
                early_stop_patience=-1,
                device="cpu",
                amp=False,
                max_train_batches=2,
                max_val_batches=2,
                max_stack_val_batches=2,
                max_final_test_batches=2,
            )
            report = train_multiscale_subjet_tagger(
                config,
                model=TinyRawTokenClassifier(),
                train_dataset=make_dataset("model_train"),
                val_dataset=make_dataset("model_val"),
                stack_val_dataset=make_dataset("stack_val"),
                final_test_dataset=make_dataset("final_test"),
            )

            output_dir = Path(config.output_dir)
            diagnostics_dir = output_dir / "diagnostics"
            self.assertEqual(report["experiment_step"], MULTISCALE_SUBJET_TRAIN_STEP)
            self.assertEqual(report["selection_metric"], MULTISCALE_SUBJET_PRIMARY_METRIC)
            self.assertEqual(report["selection_metric_direction"], "minimize")
            self.assertTrue(report["final_test_evaluated"])
            self.assertFalse(report["final_test_loaded_during_training"])
            self.assertIn("fpr_at_signal_eff_0p50", report["best_model_val_metrics"]["binary_metrics"])
            self.assertIn("fpr_at_signal_eff_0p50", report["stack_val_metrics"]["binary_metrics"])
            self.assertIn("fpr_at_signal_eff_0p50", report["final_test_metrics"]["binary_metrics"])
            self.assertIn("validation_threshold_final_test_fpr", report["final_test_metrics"]["binary_metrics"])
            self.assertIn("validation_threshold_final_test_signal_efficiency", report["final_test_metrics"]["binary_metrics"])
            self.assertIn("validation_threshold", report["final_test_metrics"]["binary_metrics"])
            self.assertIn("validation_threshold_final_test_fpr", report["final_test_metrics"])
            self.assertTrue((output_dir / "best_model_val.pt").exists())
            self.assertTrue((output_dir / "last.pt").exists())
            self.assertTrue((output_dir / "run_report.json").exists())
            self.assertTrue((diagnostics_dir / "run_report.json").exists())
            self.assertTrue((diagnostics_dir / "training_curves.json").exists())
            self.assertTrue((diagnostics_dir / "summary_metrics.csv").exists())
            self.assertTrue((diagnostics_dir / "best_metrics.csv").exists())


if __name__ == "__main__":
    unittest.main()
