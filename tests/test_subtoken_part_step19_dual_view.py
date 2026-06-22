import unittest

from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM

from teacher_logit_reco.subtoken_part import (
    SUBTOKEN_PART_DUAL_FUSION_CONCAT,
    SUBTOKEN_PART_DUAL_FUSION_CROSS_ATTENTION,
    SUBTOKEN_PART_DUAL_FUSION_LATE_LOGITS,
    SUBTOKEN_PART_DUAL_VIEW_CLASSIFIER_CONTRACT,
    SUBTOKEN_PART_VARIANT_DUAL_CROSS_ATTENTION,
    SubtokenPartConfig,
    SubtokenParticleTransformerClassifier,
    SubtokenTaggerTrainConfig,
    normalize_subtoken_dual_fusion_mode,
)


class SubtokenPartStep19DualViewTests(unittest.TestCase):
    def make_config(self, **kwargs):
        defaults = {
            "num_classes": 2,
            "variant": SUBTOKEN_PART_VARIANT_DUAL_CROSS_ATTENTION,
            "embed_dim": 16,
            "local_layers": 1,
            "local_heads": 4,
            "context_layers": 1,
            "context_heads": 4,
            "global_layers": 1,
            "global_heads": 4,
            "standard_branch_layers": 1,
            "dropout": 0.0,
            "attention_dropout": 0.0,
            "use_pairwise_bias": True,
            "standard_branch_use_pairwise_bias": True,
        }
        defaults.update(kwargs)
        return SubtokenPartConfig(**defaults)

    def make_tokens(self):
        torch = require_torch()
        tokens = torch.zeros((2, 6, RAW_TOKEN_DIM), dtype=torch.float32)
        mask = torch.tensor(
            [
                [True, True, True, False, True, False],
                [True, False, True, True, False, True],
            ],
            dtype=torch.bool,
        )
        tokens[:, :, 0] = torch.tensor(
            [
                [80.0, 45.0, 20.0, 900.0, 12.0, 900.0],
                [60.0, 800.0, 18.0, 10.0, 700.0, 8.0],
            ]
        )
        tokens[:, :, 1] = torch.tensor(
            [
                [0.1, -0.2, 0.7, 9.0, -1.1, 9.0],
                [-0.5, 8.0, 0.3, 1.2, -7.0, -0.9],
            ]
        )
        tokens[:, :, 2] = torch.tensor(
            [
                [0.2, 2.4, -2.8, -8.0, 1.3, -8.0],
                [-2.5, 8.0, 0.6, -0.4, 7.0, 1.8],
            ]
        )
        tokens[:, :, 3] = tokens[:, :, 0] + 20.0
        tokens[:, :, 4] = torch.tensor(
            [
                [1.0, -1.0, 0.0, 3.0, 1.0, 3.0],
                [-1.0, 5.0, 0.0, 1.0, 5.0, 0.0],
            ]
        )
        pid_rows = torch.tensor(
            [
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [9, 9, 9, 9, 9],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
            ],
            dtype=torch.float32,
        )
        tokens[0, :, 5:10] = pid_rows
        tokens[1, :, 5:10] = pid_rows.roll(1, dims=0)
        tokens[:, :, 10] = torch.tensor(
            [
                [0.2, -0.3, 0.4, 99.0, -0.1, 99.0],
                [0.5, 8.0, -0.2, 0.1, 7.0, -0.4],
            ]
        )
        tokens[:, :, 11] = torch.tensor(
            [
                [0.1, 0.2, 0.3, 99.0, 0.4, 99.0],
                [0.4, 8.0, 0.5, 0.6, 7.0, 0.2],
            ]
        )
        tokens[:, :, 12] = torch.tensor(
            [
                [-0.1, 0.2, -0.4, 99.0, 0.1, 99.0],
                [0.7, 8.0, 0.3, -0.2, 7.0, -0.3],
            ]
        )
        tokens[:, :, 13] = torch.tensor(
            [
                [0.6, 0.7, 0.8, 99.0, 0.2, 99.0],
                [0.9, 8.0, 1.0, 0.1, 7.0, 0.5],
            ]
        )
        return tokens * mask[:, :, None], mask

    def test_fusion_modes_return_fused_logits_and_diagnostics(self):
        torch = require_torch()
        tokens, mask = self.make_tokens()
        for mode in (
            SUBTOKEN_PART_DUAL_FUSION_LATE_LOGITS,
            SUBTOKEN_PART_DUAL_FUSION_CONCAT,
            SUBTOKEN_PART_DUAL_FUSION_CROSS_ATTENTION,
        ):
            with self.subTest(mode=mode):
                model = SubtokenParticleTransformerClassifier(self.make_config(dual_fusion_mode=mode))
                output = model(tokens, mask, return_outputs=True)
                summary = output.summary()
                diagnostics = output.diagnostics()

                self.assertEqual(summary["contract"], SUBTOKEN_PART_DUAL_VIEW_CLASSIFIER_CONTRACT)
                self.assertEqual(tuple(output.logits.shape), (2, 2))
                self.assertIsNotNone(output.dual_view)
                self.assertEqual(output.dual_view.fusion_mode, mode)
                self.assertEqual(tuple(output.dual_view.standard.logits.shape), (2, 2))
                self.assertEqual(tuple(output.dual_view.subtoken_logits.shape), (2, 2))
                self.assertEqual(tuple(output.dual_view.fused_embedding.shape), (2, 32))
                self.assertTrue(bool(torch.isfinite(output.logits).all()))
                self.assertTrue(bool(torch.isfinite(diagnostics["dual_branch_logit_delta_abs_mean"]).all()))
                self.assertTrue(summary["use_dual_view_fusion"])

    def test_late_logit_fusion_is_exact_average_of_branches(self):
        torch = require_torch()
        tokens, mask = self.make_tokens()
        model = SubtokenParticleTransformerClassifier(
            self.make_config(dual_fusion_mode=SUBTOKEN_PART_DUAL_FUSION_LATE_LOGITS)
        )

        output = model(tokens, mask, return_outputs=True)
        expected = 0.5 * (output.dual_view.subtoken_logits + output.dual_view.standard.logits)

        self.assertTrue(bool(torch.allclose(output.logits, expected, atol=1.0e-6)))

    def test_cross_attention_mode_returns_attention_weights(self):
        tokens, mask = self.make_tokens()
        model = SubtokenParticleTransformerClassifier(
            self.make_config(dual_fusion_mode=SUBTOKEN_PART_DUAL_FUSION_CROSS_ATTENTION)
        )

        output = model(tokens, mask, return_outputs=True)

        self.assertIsNotNone(output.dual_view.subtoken_to_standard_attention)
        self.assertIsNotNone(output.dual_view.standard_to_subtoken_attention)
        self.assertEqual(tuple(output.dual_view.standard.sequence_mask[:, 1:].shape), tuple(mask.shape))

    def test_training_config_builds_dual_variant(self):
        config = SubtokenTaggerTrainConfig(
            output_dir="unused",
            hlt_cache_dir="unused",
            confirm_split_settings=True,
            num_classes=2,
            label_names=("QCD", "Hgg"),
            variant=SUBTOKEN_PART_VARIANT_DUAL_CROSS_ATTENTION,
            dual_fusion_mode="crossvit",
            standard_branch_layers=1,
        )

        model_config = config.model_config()

        self.assertEqual(model_config.variant, SUBTOKEN_PART_VARIANT_DUAL_CROSS_ATTENTION)
        self.assertEqual(model_config.dual_fusion_mode, SUBTOKEN_PART_DUAL_FUSION_CROSS_ATTENTION)
        self.assertEqual(model_config.standard_branch_layers, 1)
        self.assertEqual(normalize_subtoken_dual_fusion_mode("concat"), SUBTOKEN_PART_DUAL_FUSION_CONCAT)


if __name__ == "__main__":
    unittest.main()
