import unittest

from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.jetclass_data import JetIdentity, JetView, RAW_TOKEN_DIM

from teacher_logit_reco.subtoken_part import (
    SUBTOKEN_PART_COMPAT_STEP,
    SUBTOKEN_PART_GATE_CONTEXT_SOFTMAX,
    SUBTOKEN_PART_GATE_LOCAL_SOFTMAX,
    SUBTOKEN_PART_GATE_NONE,
    SUBTOKEN_PART_STEP13_VARIANTS,
    SUBTOKEN_PART_VARIANT_CONTEXT_GATE,
    SUBTOKEN_PART_VARIANT_HLT_PART_BASELINE,
    SUBTOKEN_PART_VARIANT_LOCAL_GATE,
    SUBTOKEN_PART_VARIANT_NO_GATE,
    SubtokenHLTJetDataset,
    SubtokenPartCompatibilityConfig,
    build_subtoken_part_compat_report,
    collate_hlt_part_baseline_batch,
)


class SubtokenPartStep13CompatibilityTests(unittest.TestCase):
    def make_view(self, *, split="model_train", n_jets=6):
        torch = require_torch()
        tokens = torch.zeros((n_jets, 5, RAW_TOKEN_DIM), dtype=torch.float32)
        mask = torch.ones((n_jets, 5), dtype=torch.bool)
        labels = torch.tensor([0, 3] * ((n_jets + 1) // 2), dtype=torch.long)[:n_jets]
        for index in range(n_jets):
            label = int(labels[index].item())
            base = 35.0 + 8.0 * label + float(index)
            tokens[index, :, 0] = base - torch.arange(5, dtype=torch.float32)
            tokens[index, :, 1] = torch.linspace(-0.4, 0.4, 5) + 0.05 * label
            tokens[index, :, 2] = torch.linspace(-1.0, 1.0, 5)
            tokens[index, :, 3] = tokens[index, :, 0] + 12.0
            tokens[index, :, 4] = 1.0 if label == 3 else -1.0
            tokens[index, :, 5 + (label % 5)] = 1.0
            tokens[index, :, 10] = 0.05 * (label + 1)
            tokens[index, :, 11] = 0.2
            tokens[index, :, 12] = -0.05 * (label + 1)
            tokens[index, :, 13] = 0.3
        mask[1, 4:] = False
        tokens = tokens * mask[:, :, None]
        jet_ids = [
            JetIdentity(file=f"{split}_{index}.root", entry=index, label=int(labels[index].item()))
            for index in range(n_jets)
        ]
        return JetView(
            tokens=tokens.numpy(),
            mask=mask.numpy(),
            labels=labels.numpy(),
            jet_ids=jet_ids,
            split=split,
            metadata={"view": "fixed_hlt", "hlt_content_hash": f"{split}_hash", "jet_identity_hash": f"{split}_ids"},
        )

    def make_dataset(self, split="model_train", n_jets=6):
        return SubtokenHLTJetDataset(
            self.make_view(split=split, n_jets=n_jets),
            label_filter=(0, 3),
            label_names=("QCD", "Hgg"),
        )

    def make_config(self, **kwargs):
        payload = {
            "output_dir": "unused",
            "hlt_cache_dir": "unused",
            "confirm_split_settings": True,
            "confirm_final_test": True,
            "num_classes": 2,
            "label_names": ("QCD", "Hgg"),
            "label_filter": (0, 3),
            "selection_metric": "fpr_at_signal_eff_0p50",
            "variants": SUBTOKEN_PART_STEP13_VARIANTS,
        }
        payload.update(kwargs)
        return SubtokenPartCompatibilityConfig(**payload)

    def test_variant_mapping_uses_planned_gate_modes(self):
        config = self.make_config()

        no_gate = config.to_subtoken_train_config(SUBTOKEN_PART_VARIANT_NO_GATE)
        local_gate = config.to_subtoken_train_config(SUBTOKEN_PART_VARIANT_LOCAL_GATE)
        context_gate = config.to_subtoken_train_config(SUBTOKEN_PART_VARIANT_CONTEXT_GATE)

        self.assertEqual(no_gate.variant, SUBTOKEN_PART_VARIANT_NO_GATE)
        self.assertEqual(no_gate.gate_mode, SUBTOKEN_PART_GATE_NONE)
        self.assertTrue(no_gate.use_pairwise_bias)
        self.assertEqual(local_gate.gate_mode, SUBTOKEN_PART_GATE_LOCAL_SOFTMAX)
        self.assertEqual(context_gate.gate_mode, SUBTOKEN_PART_GATE_CONTEXT_SOFTMAX)
        with self.assertRaises(ValueError):
            config.to_subtoken_train_config(SUBTOKEN_PART_VARIANT_HLT_PART_BASELINE)

    def test_config_rejects_non_step13_variants(self):
        with self.assertRaises(ValueError):
            self.make_config(variants=("dual_part_subtoken_cross_attention",))

    def test_hlt_part_baseline_collate_uses_same_raw_dataset(self):
        torch = require_torch()
        dataset = self.make_dataset(n_jets=4)
        batch = collate_hlt_part_baseline_batch([dataset[0], dataset[1]])

        self.assertEqual(tuple(batch["points"].shape), (2, 2, 5))
        self.assertEqual(tuple(batch["features"].shape), (2, 17, 5))
        self.assertEqual(tuple(batch["lorentz_vectors"].shape), (2, 4, 5))
        self.assertEqual(tuple(batch["mask"].shape), (2, 1, 5))
        self.assertEqual(tuple(batch["labels"].shape), (2,))
        self.assertTrue(bool(torch.isfinite(batch["features"]).all()))

    def test_report_compares_identical_splits_and_prefers_lower_fpr(self):
        config = self.make_config(
            variants=(
                SUBTOKEN_PART_VARIANT_HLT_PART_BASELINE,
                SUBTOKEN_PART_VARIANT_CONTEXT_GATE,
            )
        )
        datasets = {
            "model_train": self.make_dataset("model_train", n_jets=4),
            "model_val": self.make_dataset("model_val", n_jets=4),
            "stack_val": self.make_dataset("stack_val", n_jets=4),
            "final_test": self.make_dataset("final_test", n_jets=4),
        }
        reports = {
            SUBTOKEN_PART_VARIANT_HLT_PART_BASELINE: {
                "experiment_step": SUBTOKEN_PART_COMPAT_STEP,
                "output_contract": "baseline",
                "best_epoch": 2,
                "epochs_completed": 2,
                "selection_metric": "fpr_at_signal_eff_0p50",
                "checkpoint": "baseline/best_model_val.pt",
                "best_model_val_metrics": {"accuracy": 0.8, "loss": 0.4, "n_jets": 4},
                "stack_val_metrics": {"accuracy": 0.8, "loss": 0.4, "n_jets": 4},
                "final_test_metrics": {
                    "accuracy": 0.8,
                    "loss": 0.4,
                    "n_jets": 4,
                    "binary_metrics": {"fpr_at_signal_eff_0p50": 0.30},
                },
            },
            SUBTOKEN_PART_VARIANT_CONTEXT_GATE: {
                "experiment_step": "subtoken",
                "output_contract": "subtoken",
                "best_epoch": 3,
                "epochs_completed": 3,
                "selection_metric": "fpr_at_signal_eff_0p50",
                "checkpoint": "subtoken/best_model_val.pt",
                "best_model_val_metrics": {"accuracy": 0.9, "loss": 0.3, "n_jets": 4},
                "stack_val_metrics": {"accuracy": 0.9, "loss": 0.3, "n_jets": 4},
                "final_test_metrics": {
                    "accuracy": 0.9,
                    "loss": 0.3,
                    "n_jets": 4,
                    "binary_metrics": {"fpr_at_signal_eff_0p50": 0.20},
                },
            },
        }

        report = build_subtoken_part_compat_report(config=config, child_reports=reports, datasets=datasets)

        self.assertEqual(report["experiment_step"], SUBTOKEN_PART_COMPAT_STEP)
        self.assertEqual(report["comparison_split"], "final_test")
        self.assertEqual(report["primary_metric_direction"], "minimize")
        self.assertEqual(report["best_variant"], SUBTOKEN_PART_VARIANT_CONTEXT_GATE)
        self.assertEqual(report["shared_datasets"]["model_train"]["hlt_content_hash"], "model_train_hash")
        self.assertEqual(len(report["comparison_rows"]), 2)


if __name__ == "__main__":
    unittest.main()
