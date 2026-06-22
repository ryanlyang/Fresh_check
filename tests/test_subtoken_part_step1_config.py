import unittest

from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM
from jetclass_fresh.part_inputs import PF_FEATURE_NAMES

from teacher_logit_reco.subtoken_part import (
    SUBTOKEN_MODALITY_IDENTITY,
    SUBTOKEN_MODALITY_KINEMATICS,
    SUBTOKEN_MODALITY_TRACK,
    SUBTOKEN_PART_CONTRACT,
    SUBTOKEN_PART_GATE_CONTEXT_SOFTMAX,
    SUBTOKEN_PART_GATE_LOCAL_SOFTMAX,
    SUBTOKEN_PART_GATE_NONE,
    SUBTOKEN_PART_PARTICLE_FEATURE_DIM,
    SUBTOKEN_PART_VARIANT_CONTEXT_GATE,
    SUBTOKEN_PART_VARIANT_HLT_PART_BASELINE,
    SUBTOKEN_PART_VARIANT_LOCAL_GATE,
    SUBTOKEN_PART_VARIANT_NO_GATE,
    SUBTOKEN_PART_VARIANT_SCALAR_LOCAL,
    SUBTOKEN_PART_VERSION_A,
    SUBTOKEN_PART_VERSION_B,
    SubtokenExperimentLayout,
    SubtokenFeatureConfig,
    SubtokenModalitySpec,
    SubtokenPartConfig,
    SubtokenTrainingConfig,
    build_subtoken_variant_config,
    build_subtoken_variant_configs,
    default_subtoken_part_config,
    default_subtoken_training_config,
    normalize_subtoken_gate_mode,
    normalize_subtoken_part_variant,
    normalize_subtoken_part_version,
    subtoken_part_model_name,
)


class SubtokenPartStep1ConfigTests(unittest.TestCase):
    def test_default_feature_config_declares_three_physics_modalities(self):
        cfg = SubtokenFeatureConfig()

        self.assertEqual(cfg.raw_token_dim, RAW_TOKEN_DIM)
        self.assertEqual(cfg.particle_feature_dim, len(PF_FEATURE_NAMES))
        self.assertEqual(SUBTOKEN_PART_PARTICLE_FEATURE_DIM, len(PF_FEATURE_NAMES))
        self.assertEqual(cfg.modality_names, ("kinematics", "identity", "track"))
        self.assertEqual(cfg.modalities[0].raw_indices, (0, 1, 2, 3))
        self.assertEqual(cfg.modalities[1].raw_indices, (4, 5, 6, 7, 8, 9))
        self.assertEqual(cfg.modalities[2].raw_indices, (10, 11, 12, 13))
        self.assertEqual(
            [m.name for m in cfg.modalities],
            [SUBTOKEN_MODALITY_KINEMATICS, SUBTOKEN_MODALITY_IDENTITY, SUBTOKEN_MODALITY_TRACK],
        )

    def test_feature_config_rejects_bad_groupings(self):
        bad_modalities = [
            (SubtokenModalitySpec("kin", (0, 1)), SubtokenModalitySpec("pid", (1, 2))),
            (SubtokenModalitySpec("kin", (0, 1)),),
            (SubtokenModalitySpec("kin", (0, 99)),),
        ]
        for modalities in bad_modalities:
            with self.subTest(modalities=modalities):
                with self.assertRaises(ValueError):
                    SubtokenFeatureConfig(modalities=modalities)

    def test_alias_normalizers_cover_versions_variants_and_gates(self):
        self.assertEqual(normalize_subtoken_part_version("A"), SUBTOKEN_PART_VERSION_A)
        self.assertEqual(normalize_subtoken_part_version("privileged"), SUBTOKEN_PART_VERSION_B)
        self.assertEqual(normalize_subtoken_part_variant("part"), SUBTOKEN_PART_VARIANT_HLT_PART_BASELINE)
        self.assertEqual(normalize_subtoken_part_variant("local_gate"), SUBTOKEN_PART_VARIANT_LOCAL_GATE)
        self.assertEqual(normalize_subtoken_part_variant("full"), SUBTOKEN_PART_VARIANT_CONTEXT_GATE)
        self.assertEqual(normalize_subtoken_gate_mode("disabled"), SUBTOKEN_PART_GATE_NONE)
        self.assertEqual(normalize_subtoken_gate_mode("context-softmax"), SUBTOKEN_PART_GATE_CONTEXT_SOFTMAX)

    def test_default_model_config_is_hlt_only_context_gated_subtoken_part(self):
        cfg = default_subtoken_part_config(num_classes=2)
        report = cfg.to_dict()

        self.assertIsInstance(cfg, SubtokenPartConfig)
        self.assertEqual(cfg.version, SUBTOKEN_PART_VERSION_A)
        self.assertEqual(cfg.variant, SUBTOKEN_PART_VARIANT_CONTEXT_GATE)
        self.assertEqual(cfg.gate_mode, SUBTOKEN_PART_GATE_CONTEXT_SOFTMAX)
        self.assertTrue(cfg.use_pairwise_bias)
        self.assertEqual(report["contract"], SUBTOKEN_PART_CONTRACT)
        self.assertEqual(report["feature_config"]["num_modalities"], 3)
        self.assertTrue(report["feature_config"]["include_part_style_derived_features"])

    def test_invalid_model_config_rejects_bad_dimensions_heads_and_probabilities(self):
        bad_configs = [
            {"num_classes": 1},
            {"num_classes": 2, "embed_dim": 0},
            {"num_classes": 2, "local_layers": 0},
            {"num_classes": 2, "context_layers": 0},
            {"num_classes": 2, "global_layers": 0},
            {"num_classes": 2, "embed_dim": 130, "local_heads": 8},
            {"num_classes": 2, "embed_dim": 130, "context_heads": 8},
            {"num_classes": 2, "embed_dim": 130, "global_heads": 8},
            {"num_classes": 2, "modality_dropout": -0.1},
            {"num_classes": 2, "modality_dropout": 1.0},
            {"num_classes": 2, "dropout": 1.1},
            {"num_classes": 2, "attention_dropout": -0.1},
            {"num_classes": 2, "variant": SUBTOKEN_PART_VARIANT_HLT_PART_BASELINE},
        ]
        for kwargs in bad_configs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    SubtokenPartConfig(**kwargs)

    def test_part_feature_anchor_requires_derived_features(self):
        for anchor_source in ("part_features", "raw_and_part_features"):
            with self.subTest(anchor_source=anchor_source):
                with self.assertRaises(ValueError):
                    SubtokenFeatureConfig(anchor_source=anchor_source, include_part_style_derived_features=False)

    def test_derived_feature_flag_keeps_legacy_alias_but_rejects_conflicts(self):
        canonical = SubtokenFeatureConfig(include_part_style_derived_features=False)
        legacy = SubtokenFeatureConfig(include_derived_kinematics=False)

        self.assertFalse(canonical.include_part_style_derived_features)
        self.assertFalse(canonical.include_derived_kinematics)
        self.assertFalse(legacy.include_part_style_derived_features)
        self.assertFalse(legacy.include_derived_kinematics)
        with self.assertRaises(ValueError):
            SubtokenFeatureConfig(include_part_style_derived_features=True, include_derived_kinematics=False)

    def test_variant_configs_match_planned_ablations(self):
        baseline = build_subtoken_variant_config("hlt_part")
        no_gate = build_subtoken_variant_config(SUBTOKEN_PART_VARIANT_NO_GATE)
        local_gate = build_subtoken_variant_config(SUBTOKEN_PART_VARIANT_LOCAL_GATE)
        full = build_subtoken_variant_config(SUBTOKEN_PART_VARIANT_CONTEXT_GATE)
        scalar = build_subtoken_variant_config(SUBTOKEN_PART_VARIANT_SCALAR_LOCAL)

        self.assertFalse(baseline.use_subtoken_encoder)
        self.assertTrue(baseline.use_standard_part_branch)
        self.assertEqual(baseline.gate_mode, SUBTOKEN_PART_GATE_NONE)
        self.assertEqual(no_gate.gate_mode, SUBTOKEN_PART_GATE_NONE)
        self.assertEqual(local_gate.gate_mode, SUBTOKEN_PART_GATE_LOCAL_SOFTMAX)
        self.assertFalse(local_gate.use_context_stage)
        self.assertEqual(full.gate_mode, SUBTOKEN_PART_GATE_CONTEXT_SOFTMAX)
        self.assertTrue(full.use_context_stage)
        self.assertTrue(scalar.use_scalar_tokens)

        variants = build_subtoken_variant_configs()
        self.assertGreaterEqual(len(variants), 6)

    def test_training_config_distinguishes_hlt_only_and_privileged_versions(self):
        hlt_only = default_subtoken_training_config()
        privileged = SubtokenTrainingConfig(
            version=SUBTOKEN_PART_VERSION_B,
            teacher_distill_weight=0.3,
            residual_weight=0.2,
            masked_subtoken_weight=0.1,
        )

        self.assertEqual(hlt_only.version, SUBTOKEN_PART_VERSION_A)
        self.assertEqual(privileged.version, SUBTOKEN_PART_VERSION_B)
        self.assertEqual(privileged.teacher_distill_weight, 0.3)
        with self.assertRaises(ValueError):
            SubtokenTrainingConfig(version=SUBTOKEN_PART_VERSION_A, teacher_distill_weight=0.1)

    def test_layout_and_model_names_are_isolated(self):
        layout = SubtokenExperimentLayout(output_root="checkpoints", experiment_name="demo")

        self.assertEqual(layout.root.as_posix(), "checkpoints/demo")
        self.assertEqual(layout.hlt_cache_dir.as_posix(), "checkpoints/demo/binary_inputs/hlt_cache")
        self.assertIn(SUBTOKEN_PART_VARIANT_CONTEXT_GATE, layout.tagger_dir("full").as_posix())
        self.assertEqual(subtoken_part_model_name("full"), SUBTOKEN_PART_VARIANT_CONTEXT_GATE)
        self.assertEqual(
            subtoken_part_model_name("full", version=SUBTOKEN_PART_VERSION_B),
            f"{SUBTOKEN_PART_VARIANT_CONTEXT_GATE}_{SUBTOKEN_PART_VERSION_B}",
        )


if __name__ == "__main__":
    unittest.main()
