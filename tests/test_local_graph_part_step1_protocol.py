from pathlib import Path
import unittest

from teacher_logit_reco.local_graph_part import (
    LOCAL_GRAPH_PART_BINARY_LABEL_FILTER,
    LOCAL_GRAPH_PART_CONTRACT,
    LOCAL_GRAPH_PART_DEFAULT_VARIANTS,
    LOCAL_GRAPH_PART_HLT_DEGRADATION_STRENGTH,
    LOCAL_GRAPH_PART_PRIMARY_METRIC,
    LOCAL_GRAPH_PART_SOURCE_LABEL_INDICES,
    LOCAL_GRAPH_PART_SOURCE_LABEL_NAMES,
    default_local_graph_part_protocol,
    local_graph_part_protocol_manifest,
)


class LocalGraphPartStep1ProtocolTests(unittest.TestCase):
    def test_protocol_freezes_qcd_hgg_hlt06_binary_task(self):
        protocol = default_local_graph_part_protocol()

        self.assertEqual(protocol.contract, LOCAL_GRAPH_PART_CONTRACT)
        self.assertEqual(protocol.source_label_names, LOCAL_GRAPH_PART_SOURCE_LABEL_NAMES)
        self.assertEqual(protocol.source_label_indices, LOCAL_GRAPH_PART_SOURCE_LABEL_INDICES)
        self.assertEqual(protocol.source_label_names, ("QCD", "Hgg"))
        self.assertEqual(protocol.source_label_indices, (0, 3))
        self.assertEqual(protocol.binary_label_filter, LOCAL_GRAPH_PART_BINARY_LABEL_FILTER)
        self.assertEqual(protocol.binary_label_filter, (0, 1))
        self.assertEqual(protocol.num_classes, 2)
        self.assertEqual(protocol.inference_view, "hlt")
        self.assertFalse(protocol.offline_view_allowed_at_inference)
        self.assertAlmostEqual(protocol.hlt_degradation_strength, LOCAL_GRAPH_PART_HLT_DEGRADATION_STRENGTH)
        self.assertAlmostEqual(protocol.hlt_degradation_strength, 0.6)

    def test_protocol_freezes_splits_metrics_and_variants(self):
        protocol = default_local_graph_part_protocol()

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
        self.assertEqual(protocol.primary_metric, LOCAL_GRAPH_PART_PRIMARY_METRIC)
        self.assertEqual(protocol.primary_metric, "fpr_at_signal_eff_0p50")
        self.assertEqual(protocol.selection_metric, protocol.primary_metric)
        self.assertEqual(protocol.metric_direction_by_name[protocol.primary_metric], "minimize")
        self.assertEqual(protocol.metric_direction_by_name["background_rejection_at_signal_eff_0p50"], "maximize")
        self.assertEqual(protocol.metric_direction_by_name["accuracy"], "maximize")
        self.assertEqual(protocol.comparison_split, "final_test")
        self.assertTrue(protocol.confirm_final_test)
        self.assertEqual(protocol.required_variant_names, LOCAL_GRAPH_PART_DEFAULT_VARIANTS)
        self.assertIn("hlt_part_baseline", protocol.required_variant_names)
        self.assertIn("local_edgeconv_adapter", protocol.required_variant_names)
        self.assertIn("local_point_attention_adapter", protocol.required_variant_names)

    def test_manifest_is_json_ready_and_validated(self):
        manifest = local_graph_part_protocol_manifest()

        self.assertEqual(manifest["contract"], LOCAL_GRAPH_PART_CONTRACT)
        self.assertEqual(manifest["source_label_names"], ["QCD", "Hgg"])
        self.assertEqual(manifest["source_label_indices"], [0, 3])
        self.assertEqual(manifest["binary_label_filter"], [0, 1])
        self.assertEqual(manifest["hlt_degradation_strength"], 0.6)
        self.assertEqual(manifest["primary_metric"], "fpr_at_signal_eff_0p50")
        self.assertEqual(manifest["split_specs"][0]["name"], "model_train")
        self.assertEqual(manifest["split_specs"][-1]["name"], "final_test")

    def test_protocol_note_contains_the_frozen_contract(self):
        text = Path("teacher_logit_reco/local_graph_part/PROTOCOL.md").read_text(encoding="utf-8")

        self.assertIn("QCD vs Hgg", text)
        self.assertIn("HLT degradation strength: `0.6`", text)
        self.assertIn("HLT only", text)
        self.assertIn("fpr_at_signal_eff_0p50", text)
        self.assertIn("final_test", text)
        self.assertIn("hlt_part_baseline", text)
        self.assertIn(LOCAL_GRAPH_PART_CONTRACT, text)


if __name__ == "__main__":
    unittest.main()
