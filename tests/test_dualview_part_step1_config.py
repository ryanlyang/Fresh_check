import unittest

from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM

from teacher_logit_reco.dualview_part import (
    DUALVIEW_PART_BINARY_LABEL_FILTER_NAMES,
    DUALVIEW_PART_CONTRACT,
    DUALVIEW_PART_HLT_DEGRADATION_STRENGTH,
    DUALVIEW_PART_PRIMARY_METRIC,
    DUALVIEW_PART_REQUIRED_VIEWS,
    DUALVIEW_PART_SOURCE_LABEL_NAMES,
    DUALVIEW_PART_SPLIT_SIZES,
    DUALVIEW_PART_VARIANT_FROZEN_CROSS_ATTENTION,
    DUALVIEW_PART_VARIANT_FROZEN_RESIDUAL,
    DUALVIEW_PART_VARIANT_FROZEN_RESIDUAL_CROSS_ATTENTION,
    DUALVIEW_PART_VARIANT_HLT_PART_BASELINE,
    DUALVIEW_PART_VARIANT_SHUFFLED_PN_CONTROL,
    DUALVIEW_PART_VARIANT_WARM_RESIDUAL,
    DUALVIEW_PART_VIEW_HLT,
    DUALVIEW_PART_VIEW_PN_RECO,
    DualViewPartExperimentConfig,
    DualViewPartExperimentLayout,
    canonical_dualview_part_tag,
    default_dualview_part_config,
    default_dualview_part_layout,
    dualview_metric_direction,
    normalize_dualview_part_variant,
    normalize_dualview_split_name,
)


class DualViewPartStep1ConfigTests(unittest.TestCase):
    def test_default_config_declares_canonical_qcd_hgg_hlt0p6_contract(self):
        cfg = default_dualview_part_config()
        payload = cfg.to_dict()

        self.assertEqual(cfg.source_label_names, DUALVIEW_PART_SOURCE_LABEL_NAMES)
        self.assertEqual(cfg.label_names, ("QCD", "Hgg"))
        self.assertEqual(cfg.downstream_label_filter_names, DUALVIEW_PART_BINARY_LABEL_FILTER_NAMES)
        self.assertEqual(cfg.source_label_indices, (0, 3))
        self.assertEqual(cfg.num_classes, 2)
        self.assertEqual(cfg.positive_class_name, "Hgg")
        self.assertEqual(cfg.positive_class_index, 1)
        self.assertEqual(cfg.hlt_degradation_strength, DUALVIEW_PART_HLT_DEGRADATION_STRENGTH)
        self.assertEqual(cfg.required_views, (DUALVIEW_PART_VIEW_HLT, DUALVIEW_PART_VIEW_PN_RECO))
        self.assertEqual(cfg.pn_reconstructor_architecture, "pn")
        self.assertEqual(cfg.anchor_architecture, "part")
        self.assertEqual(cfg.offline_reference_architecture, "part")
        self.assertEqual(cfg.primary_metric, DUALVIEW_PART_PRIMARY_METRIC)
        self.assertEqual(cfg.selection_metric, DUALVIEW_PART_PRIMARY_METRIC)
        self.assertEqual(cfg.primary_metric_direction, "minimize")
        self.assertEqual(cfg.selection_metric_direction, "minimize")
        self.assertEqual(cfg.raw_token_dim, RAW_TOKEN_DIM)
        self.assertEqual(cfg.split_sizes, DUALVIEW_PART_SPLIT_SIZES)
        self.assertTrue(cfg.confirm_final_test)
        self.assertEqual(payload["contract"], DUALVIEW_PART_CONTRACT)
        self.assertEqual(payload["experiment_tag"], "dualview_part_qcd_hgg_binary_hlt0p6_true500k")

    def test_default_layout_uses_canonical_root_and_dualview_paths(self):
        layout = default_dualview_part_layout(output_root="checkpoints")

        self.assertEqual(layout.root.as_posix(), "checkpoints/dualview_part_qcd_hgg_binary_hlt0p6_true500k")
        self.assertEqual(
            layout.hlt_cache_dir.as_posix(),
            "checkpoints/dualview_part_qcd_hgg_binary_hlt0p6_true500k/binary_inputs/hlt_cache",
        )
        self.assertEqual(
            layout.pn_reconstructed_view_dir.as_posix(),
            "checkpoints/dualview_part_qcd_hgg_binary_hlt0p6_true500k/reconstructed_views/pn",
        )
        self.assertIn(DUALVIEW_PART_VARIANT_FROZEN_RESIDUAL, layout.tagger_dir("residual").as_posix())
        report = layout.to_dict()
        self.assertIn("offline_teacher_reference", report["offline_reference_dir"])
        self.assertIn("final_report", report["final_report_dir"])

    def test_normalizers_cover_expected_variants_and_splits(self):
        self.assertEqual(normalize_dualview_part_variant("part"), DUALVIEW_PART_VARIANT_HLT_PART_BASELINE)
        self.assertEqual(normalize_dualview_part_variant("residual"), DUALVIEW_PART_VARIANT_FROZEN_RESIDUAL)
        self.assertEqual(normalize_dualview_part_variant("cross-attention"), DUALVIEW_PART_VARIANT_FROZEN_CROSS_ATTENTION)
        self.assertEqual(
            normalize_dualview_part_variant("residual_cross"),
            DUALVIEW_PART_VARIANT_FROZEN_RESIDUAL_CROSS_ATTENTION,
        )
        self.assertEqual(normalize_dualview_part_variant("warm"), DUALVIEW_PART_VARIANT_WARM_RESIDUAL)
        self.assertEqual(normalize_dualview_part_variant("shuffled"), DUALVIEW_PART_VARIANT_SHUFFLED_PN_CONTROL)
        self.assertEqual(normalize_dualview_split_name("final_test"), "final_test")
        self.assertEqual(dualview_metric_direction("fpr_at_signal_eff_0p50"), "minimize")
        self.assertEqual(dualview_metric_direction("accuracy"), "maximize")

    def test_canonical_config_rejects_accidental_setup_drift(self):
        bad_configs = [
            {"source_label_names": ("QCD", "Tbqq")},
            {"label_names": ("QCD", "Tbqq")},
            {"downstream_label_filter_names": ("0", "3")},
            {"positive_class_name": "QCD"},
            {"positive_class_index": 0},
            {"hlt_degradation_strength": 1.0},
            {"split_sizes": {**DUALVIEW_PART_SPLIT_SIZES, "stack_train": 250_000}},
            {"required_views": ("hlt",)},
            {"primary_metric": "accuracy"},
            {"selection_metric": "accuracy"},
        ]
        for kwargs in bad_configs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    DualViewPartExperimentConfig(**kwargs)

    def test_noncanonical_override_is_explicit(self):
        cfg = DualViewPartExperimentConfig(
            source_label_names=("QCD", "Tbqq"),
            label_names=("QCD", "Tbqq"),
            positive_class_name="Tbqq",
            positive_class_index=1,
            hlt_degradation_strength=1.0,
            allow_noncanonical=True,
        )

        self.assertEqual(cfg.label_names, ("QCD", "Tbqq"))
        self.assertEqual(cfg.hlt_degradation_strength, 1.0)
        self.assertTrue(cfg.allow_noncanonical)

    def test_from_mapping_rejects_unknown_keys_and_round_trips(self):
        cfg = DualViewPartExperimentConfig.from_mapping(default_dualview_part_config().to_dict())

        self.assertEqual(cfg.label_names, ("QCD", "Hgg"))
        with self.assertRaises(ValueError):
            DualViewPartExperimentConfig.from_mapping({"not_a_real_key": 1})

    def test_canonical_tag_formats_hlt_strength(self):
        self.assertEqual(canonical_dualview_part_tag(), "dualview_part_qcd_hgg_binary_hlt0p6_true500k")
        self.assertEqual(
            canonical_dualview_part_tag(hlt_degradation_strength=1.0),
            "dualview_part_qcd_hgg_binary_hlt1p0_true500k",
        )


if __name__ == "__main__":
    unittest.main()
