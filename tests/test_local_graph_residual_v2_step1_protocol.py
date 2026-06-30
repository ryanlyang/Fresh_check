import unittest

from teacher_logit_reco.local_graph_part import (
    LOCAL_GRAPH_MODEL_VARIANT_HLT_PART_BASELINE,
    LOCAL_GRAPH_PART_CONTRACT,
    LOCAL_GRAPH_PART_HLT_DEGRADATION_STRENGTH,
    LOCAL_GRAPH_PART_PRIMARY_METRIC,
    LOCAL_GRAPH_PART_PROTOCOL_STEP,
    LOCAL_GRAPH_PART_SOURCE_LABEL_INDICES,
    LOCAL_GRAPH_PART_SOURCE_LABEL_NAMES,
    LOCAL_GRAPH_RESIDUAL_V2_ANCHOR_CONTRACT,
    LOCAL_GRAPH_RESIDUAL_V2_CACHE_CONTRACT,
    LOCAL_GRAPH_RESIDUAL_V2_LOSS_BOUNDARY_PAIRWISE,
    LOCAL_GRAPH_RESIDUAL_V2_LOSS_BOUNDARY_PAIRWISE_BCE_ANCHOR,
    LOCAL_GRAPH_RESIDUAL_V2_LOSS_BOUNDARY_PAIRWISE_SOFT_FPR_BCE_ANCHOR,
    LOCAL_GRAPH_RESIDUAL_V2_LOSS_WEIGHTED_BCE,
    LOCAL_GRAPH_RESIDUAL_V2_MODEL_CONTRACT,
    LOCAL_GRAPH_RESIDUAL_V2_REPORT_CONTRACT,
    LOCAL_GRAPH_RESIDUAL_V2_REQUIRED_EMBEDDING_ROLE,
    LOCAL_GRAPH_RESIDUAL_V2_TRAIN_CONTRACT,
    LocalGraphResidualV2Protocol,
    default_local_graph_residual_v2_protocol,
    local_graph_residual_v2_protocol_manifest,
)


class LocalGraphResidualV2Step1ProtocolTest(unittest.TestCase):
    def test_default_protocol_is_qcd_hgg_hlt06_fpr50(self):
        protocol = default_local_graph_residual_v2_protocol()

        self.assertEqual(protocol.base_protocol_step, LOCAL_GRAPH_PART_PROTOCOL_STEP)
        self.assertEqual(protocol.base_protocol_contract, LOCAL_GRAPH_PART_CONTRACT)
        self.assertEqual(protocol.inference_view, "hlt")
        self.assertEqual(protocol.hlt_degradation_strength, LOCAL_GRAPH_PART_HLT_DEGRADATION_STRENGTH)
        self.assertEqual(protocol.label_names, LOCAL_GRAPH_PART_SOURCE_LABEL_NAMES)
        self.assertEqual(protocol.source_label_indices, LOCAL_GRAPH_PART_SOURCE_LABEL_INDICES)
        self.assertEqual(protocol.positive_class_name, "Hgg")
        self.assertEqual(protocol.positive_class_index, 1)
        self.assertEqual(protocol.num_classes, 2)
        self.assertEqual(protocol.primary_metric, LOCAL_GRAPH_PART_PRIMARY_METRIC)
        self.assertEqual(protocol.primary_metric_direction, "minimize")
        self.assertEqual(protocol.selection_metric, "fpr_at_signal_eff_0p50")
        self.assertEqual(protocol.selection_split, "model_val")
        self.assertEqual(protocol.train_splits, ("model_train",))
        self.assertEqual(protocol.baseline_variant, LOCAL_GRAPH_MODEL_VARIANT_HLT_PART_BASELINE)

    def test_protocol_requires_true_part_embedding_and_disallows_v1_proxy(self):
        protocol = default_local_graph_residual_v2_protocol()

        self.assertTrue(protocol.true_embedding_required)
        self.assertEqual(protocol.required_embedding_role, LOCAL_GRAPH_RESIDUAL_V2_REQUIRED_EMBEDDING_ROLE)
        self.assertIn("widened_classifier_logits", protocol.disallowed_embedding_fallbacks)
        self.assertIn("num_classes_embedding_proxy", protocol.disallowed_embedding_fallbacks)

        with self.assertRaisesRegex(ValueError, "true HLT ParT embedding"):
            LocalGraphResidualV2Protocol(true_embedding_required=False).validate()
        with self.assertRaisesRegex(ValueError, "widened-head embedding proxies"):
            LocalGraphResidualV2Protocol(disallowed_embedding_fallbacks=("raw_hlt_summary_only",)).validate()

    def test_contract_names_are_manifested(self):
        protocol = default_local_graph_residual_v2_protocol()

        self.assertEqual(protocol.contract_by_name["anchor"], LOCAL_GRAPH_RESIDUAL_V2_ANCHOR_CONTRACT)
        self.assertEqual(protocol.contract_by_name["cache"], LOCAL_GRAPH_RESIDUAL_V2_CACHE_CONTRACT)
        self.assertEqual(protocol.contract_by_name["model"], LOCAL_GRAPH_RESIDUAL_V2_MODEL_CONTRACT)
        self.assertEqual(protocol.contract_by_name["train"], LOCAL_GRAPH_RESIDUAL_V2_TRAIN_CONTRACT)
        self.assertEqual(protocol.contract_by_name["report"], LOCAL_GRAPH_RESIDUAL_V2_REPORT_CONTRACT)

        manifest = local_graph_residual_v2_protocol_manifest()
        self.assertEqual(manifest["contract_specs"][0]["name"], "anchor")
        self.assertEqual(manifest["contract_specs"][0]["contract"], LOCAL_GRAPH_RESIDUAL_V2_ANCHOR_CONTRACT)
        self.assertEqual(manifest["primary_metric"], "fpr_at_signal_eff_0p50")

    def test_default_train_modes_are_a_c_d_not_duplicate_e(self):
        protocol = default_local_graph_residual_v2_protocol()

        self.assertEqual(
            set(protocol.train_loss_modes),
            {
                LOCAL_GRAPH_RESIDUAL_V2_LOSS_WEIGHTED_BCE,
                LOCAL_GRAPH_RESIDUAL_V2_LOSS_BOUNDARY_PAIRWISE_BCE_ANCHOR,
                LOCAL_GRAPH_RESIDUAL_V2_LOSS_BOUNDARY_PAIRWISE_SOFT_FPR_BCE_ANCHOR,
            },
        )
        self.assertNotIn(LOCAL_GRAPH_RESIDUAL_V2_LOSS_BOUNDARY_PAIRWISE, protocol.train_loss_modes)
        self.assertIn(LOCAL_GRAPH_RESIDUAL_V2_LOSS_BOUNDARY_PAIRWISE, protocol.ablation_loss_modes)
        self.assertEqual(protocol.alpha_shrinkage_policy, "validation_shrinkage_over_learned_correction")

        with self.assertRaisesRegex(ValueError, "A/C/D only"):
            LocalGraphResidualV2Protocol(
                train_loss_modes=(
                    LOCAL_GRAPH_RESIDUAL_V2_LOSS_WEIGHTED_BCE,
                    LOCAL_GRAPH_RESIDUAL_V2_LOSS_BOUNDARY_PAIRWISE,
                )
            ).validate()


if __name__ == "__main__":
    unittest.main()
