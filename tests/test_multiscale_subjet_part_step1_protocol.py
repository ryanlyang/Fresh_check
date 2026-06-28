from pathlib import Path
import unittest

from teacher_logit_reco.multiscale_subjet_part import (
    MULTISCALE_SUBJET_BINARY_LABEL_FILTER,
    MULTISCALE_SUBJET_CONTRACT,
    MULTISCALE_SUBJET_DEFAULT_VARIANTS,
    MULTISCALE_SUBJET_HLT_DEGRADATION_STRENGTH,
    MULTISCALE_SUBJET_PRIMARY_METRIC,
    MULTISCALE_SUBJET_SOURCE_LABEL_INDICES,
    MULTISCALE_SUBJET_SOURCE_LABEL_NAMES,
    default_multiscale_subjet_part_protocol,
    multiscale_subjet_part_protocol_manifest,
)


class MultiscaleSubjetPartStep1ProtocolTests(unittest.TestCase):
    def test_protocol_freezes_qcd_hgg_hlt06_binary_task(self):
        protocol = default_multiscale_subjet_part_protocol()

        self.assertEqual(protocol.contract, MULTISCALE_SUBJET_CONTRACT)
        self.assertEqual(protocol.source_label_names, MULTISCALE_SUBJET_SOURCE_LABEL_NAMES)
        self.assertEqual(protocol.source_label_indices, MULTISCALE_SUBJET_SOURCE_LABEL_INDICES)
        self.assertEqual(protocol.source_label_names, ("QCD", "Hgg"))
        self.assertEqual(protocol.source_label_indices, (0, 3))
        self.assertEqual(protocol.binary_label_filter, MULTISCALE_SUBJET_BINARY_LABEL_FILTER)
        self.assertEqual(protocol.binary_label_filter, (0, 1))
        self.assertEqual(protocol.num_classes, 2)
        self.assertEqual(protocol.inference_view, "hlt")
        self.assertFalse(protocol.offline_view_allowed_at_inference)
        self.assertAlmostEqual(protocol.hlt_degradation_strength, MULTISCALE_SUBJET_HLT_DEGRADATION_STRENGTH)
        self.assertAlmostEqual(protocol.hlt_degradation_strength, 0.6)

    def test_protocol_freezes_splits_metrics_and_required_variants(self):
        protocol = default_multiscale_subjet_part_protocol()

        self.assertEqual(
            protocol.split_size_by_name,
            {
                "model_train": 500_000,
                "model_val": 150_000,
                "stack_train": 500_000,
                "stack_val": 150_000,
                "final_test": 500_000,
            },
        )
        self.assertEqual(protocol.primary_metric, MULTISCALE_SUBJET_PRIMARY_METRIC)
        self.assertEqual(protocol.primary_metric, "fpr_at_signal_eff_0p50")
        self.assertEqual(protocol.selection_metric, protocol.primary_metric)
        self.assertEqual(protocol.metric_direction_by_name[protocol.primary_metric], "minimize")
        self.assertEqual(protocol.metric_direction_by_name["background_rejection_at_signal_eff_0p50"], "maximize")
        self.assertEqual(protocol.metric_direction_by_name["accuracy"], "maximize")
        self.assertEqual(protocol.comparison_split, "final_test")
        self.assertTrue(protocol.confirm_final_test)
        self.assertEqual(protocol.required_variant_names, MULTISCALE_SUBJET_DEFAULT_VARIANTS)
        self.assertIn("hlt_part_baseline", protocol.required_variant_names)
        self.assertIn("multiscale_subjet_residual_part_adapter", protocol.required_variant_names)
        self.assertIn("pure_perceiver_latent_control", protocol.required_variant_names)
        self.assertIn("part_plus_random_subjet_control", protocol.required_variant_names)
        self.assertIn("larger_hlt_part_control", protocol.optional_variant_names)
        self.assertIn("two_part_ensemble_control", protocol.optional_variant_names)

    def test_manifest_is_json_ready_and_validated(self):
        manifest = multiscale_subjet_part_protocol_manifest()

        self.assertEqual(manifest["contract"], MULTISCALE_SUBJET_CONTRACT)
        self.assertEqual(manifest["source_label_names"], ["QCD", "Hgg"])
        self.assertEqual(manifest["source_label_indices"], [0, 3])
        self.assertEqual(manifest["binary_label_filter"], [0, 1])
        self.assertEqual(manifest["hlt_degradation_strength"], 0.6)
        self.assertEqual(manifest["primary_metric"], "fpr_at_signal_eff_0p50")
        self.assertEqual(manifest["split_specs"][0]["name"], "model_train")
        self.assertEqual(manifest["split_specs"][-1]["name"], "final_test")
        self.assertEqual(manifest["variant_specs"][1]["name"], "multiscale_subjet_residual_part_adapter")

    def test_protocol_note_contains_the_frozen_contract(self):
        text = Path("teacher_logit_reco/multiscale_subjet_part/PROTOCOL.md").read_text(encoding="utf-8")

        self.assertIn("QCD vs Hgg", text)
        self.assertIn("HLT degradation strength: `0.6`", text)
        self.assertIn("HLT only", text)
        self.assertIn("fpr_at_signal_eff_0p50", text)
        self.assertIn("final_test", text)
        self.assertIn("hlt_part_baseline", text)
        self.assertIn("multiscale_subjet_residual_part_adapter", text)


if __name__ == "__main__":
    unittest.main()
