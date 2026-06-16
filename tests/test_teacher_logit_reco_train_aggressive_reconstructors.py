import unittest

from teacher_logit_reco.train_aggressive_reconstructors import (
    AGGRESSIVE_RECONSTRUCTOR_ARCHITECTURES,
    EXPERIMENT_STEP,
    TeacherLogitAggressiveReconstructorTrainConfig,
)


def make_config(**overrides):
    payload = {
        "output_dir": "tmp/aggressive_reco",
        "manifest_path": "checkpoints/jetclass_fresh_splits/split_manifest.json.gz",
        "hlt_cache_dir": "checkpoints/jetclass_fresh_hlt_cache",
        "teacher_checkpoint": "checkpoints/offline_teacher/best_model_val.pt",
    }
    payload.update(overrides)
    return TeacherLogitAggressiveReconstructorTrainConfig(**payload)


class TeacherLogitAggressiveTrainConfigTests(unittest.TestCase):
    def test_step_metadata_and_architecture_set(self):
        self.assertEqual(EXPERIMENT_STEP, "teacher_logit_reco_step8_aggressive_train")
        self.assertEqual(
            AGGRESSIVE_RECONSTRUCTOR_ARCHITECTURES,
            (
                "aggressive_global_transformer",
                "aggressive_particle_net",
                "aggressive_particle_flow",
                "aggressive_particle_cnn",
            ),
        )

    def test_accepts_aggressive_aliases_but_not_conservative_aliases(self):
        self.assertEqual(make_config(reco_architecture="aggressive_gt").normalized_reco_architecture(), "aggressive_global_transformer")
        self.assertEqual(make_config(reco_architecture="agpn").normalized_reco_architecture(), "aggressive_particle_net")
        self.assertEqual(make_config(reco_architecture="aggressive-pfn").normalized_reco_architecture(), "aggressive_particle_flow")
        self.assertEqual(make_config(reco_architecture="agpcnn").normalized_reco_architecture(), "aggressive_particle_cnn")
        with self.assertRaises(ValueError):
            make_config(reco_architecture="pn")

    def test_global_transformer_model_config_uses_hidden_dim_for_head(self):
        cfg = make_config(
            reco_architecture="aggressive_gt",
            hidden_dim=64,
            num_heads=4,
            num_layers=2,
            num_extra_candidates=7,
            max_global_logpt_scale=0.11,
        )
        model_config = cfg.model_config()
        payload = model_config.to_dict()
        self.assertEqual(payload["reconstructor_architecture"], "aggressive_global_transformer")
        self.assertEqual(payload["hidden_dim"], 64)
        self.assertEqual(payload["aggressive_head_config"]["embedding_dim"], 64)
        self.assertEqual(payload["aggressive_head_config"]["num_extra_candidates"], 7)
        self.assertEqual(payload["aggressive_head_config"]["max_global_logpt_scale"], 0.11)

    def test_particle_net_model_config_routes_encoder_and_head_fields(self):
        cfg = make_config(
            reco_architecture="aggressive_pn",
            edgeconv_dims=(16, 32),
            embedding_dim=40,
            k=5,
            num_extra_candidates=9,
            extra_weight_bias=-1.5,
        )
        payload = cfg.model_config().to_dict()
        self.assertEqual(payload["reconstructor_architecture"], "aggressive_particle_net")
        self.assertEqual(payload["edgeconv_dims"], [16, 32])
        self.assertEqual(payload["k"], 5)
        self.assertEqual(payload["embedding_dim"], 40)
        self.assertEqual(payload["aggressive_head_config"]["embedding_dim"], 40)
        self.assertEqual(payload["aggressive_head_config"]["num_extra_candidates"], 9)
        self.assertEqual(payload["aggressive_head_config"]["extra_weight_bias"], -1.5)

    def test_particle_flow_model_config_routes_encoder_and_head_fields(self):
        cfg = make_config(
            reco_architecture="aggressive_pfn",
            phi_dims=(24, 48),
            context_dim=64,
            context_mlp_dims=(80,),
            embedding_dim=36,
        )
        payload = cfg.model_config().to_dict()
        self.assertEqual(payload["reconstructor_architecture"], "aggressive_particle_flow")
        self.assertEqual(payload["phi_dims"], [24, 48])
        self.assertEqual(payload["context_dim"], 64)
        self.assertEqual(payload["context_mlp_dims"], [80])
        self.assertEqual(payload["aggressive_head_config"]["embedding_dim"], 36)

    def test_particle_cnn_model_config_routes_encoder_and_head_fields(self):
        cfg = make_config(
            reco_architecture="aggressive_pcnn",
            hidden_channels=32,
            num_blocks=3,
            kernel_sizes=(5, 3, 3),
            dilations=(1, 2, 4),
            context_dim=64,
            context_mlp_dims=(80,),
            embedding_dim=36,
        )
        payload = cfg.model_config().to_dict()
        self.assertEqual(payload["reconstructor_architecture"], "aggressive_particle_cnn")
        self.assertEqual(payload["hidden_channels"], 32)
        self.assertEqual(payload["num_blocks"], 3)
        self.assertEqual(payload["kernel_sizes"], [5, 3, 3])
        self.assertEqual(payload["dilations"], [1, 2, 4])
        self.assertEqual(payload["aggressive_head_config"]["embedding_dim"], 36)

    def test_loss_config_enables_aggressive_terms(self):
        cfg = make_config()
        loss = cfg.loss_config()
        self.assertEqual(loss.teacher_kl_weight, 1.0)
        self.assertEqual(loss.ce_weight, 0.5)
        self.assertGreater(loss.aggressive_extra_budget_weight, 0.0)
        self.assertGreater(loss.aggressive_parent_weight_budget_weight, 0.0)
        self.assertGreater(loss.aggressive_global_calibration_budget_weight, 0.0)
        self.assertEqual(loss.max_total_extra_pt_fraction, cfg.max_total_extra_pt_fraction)

    def test_validates_split_and_pcnn_kernel_lengths(self):
        with self.assertRaises(ValueError):
            make_config(train_split="stack_train")
        with self.assertRaises(ValueError):
            make_config(reco_architecture="aggressive_pcnn", num_blocks=2, kernel_sizes=(5,), dilations=(1, 2))
        with self.assertRaises(ValueError):
            make_config(reco_architecture="aggressive_pcnn", num_blocks=1, kernel_sizes=(4,), dilations=(1,))


if __name__ == "__main__":
    unittest.main()
