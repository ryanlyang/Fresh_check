import unittest

from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM

from teacher_logit_reco.set_matching.detr_slots import (
    DETR_SLOT_ENCODER_ARCHITECTURES,
    DETR_SLOT_RECONSTRUCTED_VIEW_NAMES,
    DETR_SLOT_RECONSTRUCTOR_CONTRACT,
    DETR_SLOT_VIEW_NAMES,
    DetrSlotExperimentConfig,
    DetrSlotExperimentLayout,
    build_detr_slot_five_view_group,
    build_detr_slot_view_specs,
    default_detr_slot_experiment_config,
    detr_slot_model_name,
    detr_slot_view_name_for_encoder,
    normalize_detr_slot_encoder_architecture,
    normalize_detr_slot_view_name,
)


class DetrFreeSlotStep1Tests(unittest.TestCase):
    def test_architecture_aliases_normalize_to_four_encoder_families(self):
        self.assertEqual(DETR_SLOT_ENCODER_ARCHITECTURES, ("gt", "pn", "pfn", "pcnn"))
        self.assertEqual(normalize_detr_slot_encoder_architecture("part"), "gt")
        self.assertEqual(normalize_detr_slot_encoder_architecture("ParticleNet"), "pn")
        self.assertEqual(normalize_detr_slot_encoder_architecture("particle_flow"), "pfn")
        self.assertEqual(normalize_detr_slot_encoder_architecture("particle_cnn"), "pcnn")

    def test_detr_view_names_are_separate_from_parent_aligned_set_matching_names(self):
        self.assertEqual(DETR_SLOT_RECONSTRUCTED_VIEW_NAMES, ("detr_gt", "detr_pn", "detr_pfn", "detr_pcnn"))
        self.assertEqual(DETR_SLOT_VIEW_NAMES, ("hlt", "detr_gt", "detr_pn", "detr_pfn", "detr_pcnn"))
        self.assertEqual(detr_slot_view_name_for_encoder("part"), "detr_gt")
        self.assertEqual(normalize_detr_slot_view_name("pn"), "detr_pn")
        self.assertEqual(normalize_detr_slot_view_name("detr_pcnn"), "detr_pcnn")
        with self.assertRaises(ValueError):
            normalize_detr_slot_view_name("gt_reco")

    def test_default_config_declares_free_slot_contract(self):
        cfg = default_detr_slot_experiment_config()
        self.assertIsInstance(cfg, DetrSlotExperimentConfig)
        self.assertEqual(cfg.num_slots, 160)
        self.assertEqual(cfg.export_max_tokens, 128)
        self.assertEqual(cfg.particle_feature_dim, RAW_TOKEN_DIM)
        self.assertEqual(cfg.encoder_architectures, ("gt", "pn", "pfn", "pcnn"))
        self.assertEqual(cfg.to_dict()["contract"], DETR_SLOT_RECONSTRUCTOR_CONTRACT)

    def test_invalid_config_rejects_export_more_than_available_slots(self):
        with self.assertRaises(ValueError):
            DetrSlotExperimentConfig(num_slots=64, export_max_tokens=128)

    def test_invalid_config_rejects_bad_token_threshold_and_decoder_values(self):
        bad_configs = [
            {"min_tokens_per_view": -1},
            {"min_tokens_per_view": 129, "export_max_tokens": 128},
            {"confidence_threshold": -0.1},
            {"confidence_threshold": 1.1},
            {"embed_dim": 0},
            {"decoder_layers": 0},
            {"decoder_heads": 0},
            {"embed_dim": 10, "decoder_heads": 4},
            {"split_sizes": {**{split: 1 for split in ("model_train", "model_val", "stack_train", "stack_val", "final_test")}, "model_train": 0}},
        ]
        for kwargs in bad_configs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    DetrSlotExperimentConfig(**kwargs)

    def test_view_specs_and_five_view_group_are_ordered(self):
        specs = build_detr_slot_view_specs()
        self.assertEqual([spec.name for spec in specs], list(DETR_SLOT_VIEW_NAMES))
        self.assertEqual([spec.encoder_architecture for spec in specs[1:]], ["gt", "pn", "pfn", "pcnn"])
        group = build_detr_slot_five_view_group()
        self.assertEqual(group.view_names, DETR_SLOT_VIEW_NAMES)

    def test_layout_uses_isolated_detr_slot_directories(self):
        layout = DetrSlotExperimentLayout(output_root="checkpoints", experiment_name="demo")
        self.assertEqual(layout.root.as_posix(), "checkpoints/demo")
        self.assertIn("detr_slot_reconstructors", layout.reconstructor_dir("gt").as_posix())
        self.assertIn("detr_slot_reconstructed_views", layout.reconstructed_view_dir("pn").as_posix())
        self.assertEqual(detr_slot_model_name("pcnn"), "detr_slot_pcnn_reco")


if __name__ == "__main__":
    unittest.main()
