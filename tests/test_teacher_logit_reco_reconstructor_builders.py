import importlib.util
import tempfile
import unittest
from pathlib import Path

from teacher_logit_reco.reconstructor_builders import (
    TEACHER_LOGIT_RECONSTRUCTOR_ARCHITECTURES,
    build_reconstructor_from_config,
    build_teacher_logit_reconstructor,
    infer_reconstructor_architecture_from_payload,
    load_teacher_logit_reconstructor_checkpoint,
    normalize_reconstructor_architecture,
)
from teacher_logit_reco.particle_cnn_reconstructor import ParticleCnnReconstructorConfig
from teacher_logit_reco.particle_flow_reconstructor import ParticleFlowReconstructorConfig

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch

    from teacher_logit_reco.global_transformer import (
        GlobalTransformerReconstructor,
        GlobalTransformerReconstructorConfig,
    )
    from teacher_logit_reco.particle_flow_reconstructor import ParticleFlowReconstructor
    from teacher_logit_reco.particle_cnn_reconstructor import ParticleCnnReconstructor
    from teacher_logit_reco.particle_net_reconstructor import ParticleNetReconstructor


class TeacherLogitReconstructorBuilderTests(unittest.TestCase):
    def test_normalizes_architecture_aliases(self):
        self.assertIn("global_transformer", TEACHER_LOGIT_RECONSTRUCTOR_ARCHITECTURES)
        self.assertIn("particle_net", TEACHER_LOGIT_RECONSTRUCTOR_ARCHITECTURES)
        self.assertIn("particle_flow", TEACHER_LOGIT_RECONSTRUCTOR_ARCHITECTURES)
        self.assertIn("particle_cnn", TEACHER_LOGIT_RECONSTRUCTOR_ARCHITECTURES)
        self.assertEqual(normalize_reconstructor_architecture(None), "global_transformer")
        self.assertEqual(normalize_reconstructor_architecture("gt"), "global_transformer")
        self.assertEqual(normalize_reconstructor_architecture("global-transformer"), "global_transformer")
        self.assertEqual(normalize_reconstructor_architecture("ParticleNet"), "particle_net")
        self.assertEqual(normalize_reconstructor_architecture("pn"), "particle_net")
        self.assertEqual(normalize_reconstructor_architecture("particle-flow"), "particle_flow")
        self.assertEqual(normalize_reconstructor_architecture("PFN"), "particle_flow")
        self.assertEqual(normalize_reconstructor_architecture("deep sets"), "particle_flow")
        self.assertEqual(normalize_reconstructor_architecture("P-CNN"), "particle_cnn")
        self.assertEqual(normalize_reconstructor_architecture("particle cnn"), "particle_cnn")
        with self.assertRaises(ValueError):
            normalize_reconstructor_architecture("unknown_reco")

    def test_infers_architecture_from_payload(self):
        self.assertEqual(infer_reconstructor_architecture_from_payload({}), "global_transformer")
        self.assertEqual(
            infer_reconstructor_architecture_from_payload({"reconstructor_architecture": "pn"}),
            "particle_net",
        )
        self.assertEqual(
            infer_reconstructor_architecture_from_payload({"reconstructor_architecture": "pfn"}),
            "particle_flow",
        )
        self.assertEqual(
            infer_reconstructor_architecture_from_payload({"reconstructor_architecture": "pcnn"}),
            "particle_cnn",
        )
        self.assertEqual(
            infer_reconstructor_architecture_from_payload(
                {"model_config": {"reconstructor_architecture": "global_transformer"}}
            ),
            "global_transformer",
        )
        self.assertEqual(
            infer_reconstructor_architecture_from_payload(
                {"config": {"teacher_architecture": "pn", "reconstructor_architecture": "gt"}}
            ),
            "global_transformer",
        )
        self.assertEqual(
            infer_reconstructor_architecture_from_payload({}, architecture="particle-net"),
            "particle_net",
        )
        self.assertEqual(
            infer_reconstructor_architecture_from_payload({}, architecture="particle flow"),
            "particle_flow",
        )
        self.assertEqual(
            infer_reconstructor_architecture_from_payload({}, architecture="particle-cnn"),
            "particle_cnn",
        )
        with self.assertRaises(ValueError):
            infer_reconstructor_architecture_from_payload({}, architecture="bad_architecture")
        with self.assertRaises(ValueError):
            infer_reconstructor_architecture_from_payload({"reconstructor_architecture": "bad_architecture"})

    def test_particle_flow_config_accepts_architecture_keys_and_alias_dims(self):
        config = ParticleFlowReconstructorConfig.from_mapping(
            {
                "reconstructor_architecture": "particle_flow",
                "architecture": "pfn",
                "phi_dims": [16, 32],
                "context_dim": 48,
                "context_dims": [64],
                "decoder_dims": [24],
                "num_extra_candidates": 2,
                "dropout": 0.0,
            }
        )
        self.assertEqual(config.phi_dims, (16, 32))
        self.assertEqual(config.context_mlp_dims, (64,))
        self.assertEqual(config.decoder_dims, (24,))
        self.assertEqual(config.to_dict()["reconstructor_architecture"], "particle_flow")

    def test_particle_cnn_config_accepts_architecture_keys_and_alias_dims(self):
        config = ParticleCnnReconstructorConfig.from_mapping(
            {
                "reconstructor_architecture": "particle_cnn",
                "architecture": "pcnn",
                "hidden_channels": 32,
                "num_blocks": 3,
                "kernel_sizes": [5, 3, 3],
                "dilations": [1, 2, 4],
                "context_dim": 48,
                "context_dims": [64],
                "decoder_dims": [24],
                "num_extra_candidates": 2,
                "dropout": 0.0,
            }
        )
        self.assertEqual(config.kernel_sizes, (5, 3, 3))
        self.assertEqual(config.dilations, (1, 2, 4))
        self.assertEqual(config.context_mlp_dims, (64,))
        self.assertEqual(config.decoder_dims, (24,))
        self.assertEqual(config.to_dict()["reconstructor_architecture"], "particle_cnn")

    def test_build_reconstructor_from_config_rejects_unknown_architecture_before_torch(self):
        with self.assertRaises(ValueError):
            build_reconstructor_from_config({"reconstructor_architecture": "not_a_reco"})


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class TeacherLogitReconstructorCheckpointTests(unittest.TestCase):
    def make_model(self):
        torch.manual_seed(11)
        return GlobalTransformerReconstructor(
            GlobalTransformerReconstructorConfig(
                hidden_dim=32,
                num_heads=4,
                num_layers=1,
                num_extra_candidates=2,
                dropout=0.0,
            )
        )

    def test_loads_legacy_global_transformer_checkpoint_without_architecture_field(self):
        model = self.make_model()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy_best_model_val.pt"
            torch.save(
                {
                    "epoch": 3,
                    "model_state_dict": model.state_dict(),
                    "model_config": model.config.to_dict(),
                },
                path,
            )
            loaded, payload = load_teacher_logit_reconstructor_checkpoint(path, device=torch.device("cpu"))
            self.assertIsInstance(loaded, GlobalTransformerReconstructor)
            self.assertEqual(payload["epoch"], 3)

    def test_loads_new_global_transformer_checkpoint_with_architecture_field(self):
        model = self.make_model()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best_model_val.pt"
            torch.save(
                {
                    "epoch": 4,
                    "reconstructor_architecture": "global_transformer",
                    "model_state_dict": model.state_dict(),
                    "model_config": model.config.to_dict(),
                },
                path,
            )
            loaded, payload = load_teacher_logit_reconstructor_checkpoint(
                path,
                device=torch.device("cpu"),
                expected_architecture="gt",
            )
            self.assertIsInstance(loaded, GlobalTransformerReconstructor)
            self.assertEqual(payload["reconstructor_architecture"], "global_transformer")

    def test_builds_particle_net_reconstructor_after_step4(self):
        model = build_teacher_logit_reconstructor(
            "particle_net",
            {"edgeconv_dims": [16, 16], "k": 4, "num_extra_candidates": 2, "dropout": 0.0},
        )
        self.assertIsInstance(model, ParticleNetReconstructor)
        self.assertEqual(model.config.edgeconv_dims, (16, 16))

    def test_builds_particle_flow_reconstructor_after_step1(self):
        model = build_reconstructor_from_config(
            {
                "reconstructor_architecture": "particle_flow",
                "phi_dims": [16, 16],
                "context_dim": 32,
                "context_mlp_dims": [32],
                "decoder_dims": [16],
                "num_extra_candidates": 2,
                "dropout": 0.0,
            }
        )
        self.assertIsInstance(model, ParticleFlowReconstructor)
        self.assertEqual(model.config.phi_dims, (16, 16))
        self.assertEqual(model.config.context_dim, 32)

    def test_builds_particle_cnn_reconstructor_after_step1(self):
        model = build_reconstructor_from_config(
            {
                "reconstructor_architecture": "particle_cnn",
                "hidden_channels": 16,
                "num_blocks": 2,
                "kernel_sizes": [5, 3],
                "dilations": [1, 2],
                "context_dim": 32,
                "context_mlp_dims": [32],
                "decoder_dims": [16],
                "num_extra_candidates": 2,
                "dropout": 0.0,
            }
        )
        self.assertIsInstance(model, ParticleCnnReconstructor)
        self.assertEqual(model.config.hidden_channels, 16)
        self.assertEqual(model.config.kernel_sizes, (5, 3))

    def test_loads_particle_flow_checkpoint_through_shared_loader(self):
        model = build_teacher_logit_reconstructor(
            "pfn",
            {
                "phi_dims": [16],
                "context_dim": 32,
                "context_mlp_dims": [32],
                "decoder_dims": [16],
                "num_extra_candidates": 1,
                "dropout": 0.0,
            },
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pfn_best_model_val.pt"
            torch.save(
                {
                    "epoch": 1,
                    "reconstructor_architecture": "particle_flow",
                    "model_state_dict": model.state_dict(),
                    "model_config": model.config.to_dict(),
                },
                path,
            )
            loaded, payload = load_teacher_logit_reconstructor_checkpoint(
                path,
                device=torch.device("cpu"),
                expected_architecture="pfn",
            )
            self.assertIsInstance(loaded, ParticleFlowReconstructor)
            self.assertEqual(payload["epoch"], 1)

    def test_loads_particle_cnn_checkpoint_through_shared_loader(self):
        model = build_teacher_logit_reconstructor(
            "pcnn",
            {
                "hidden_channels": 16,
                "num_blocks": 1,
                "kernel_sizes": [3],
                "dilations": [1],
                "context_dim": 32,
                "context_mlp_dims": [32],
                "decoder_dims": [16],
                "num_extra_candidates": 1,
                "dropout": 0.0,
            },
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pcnn_best_model_val.pt"
            torch.save(
                {
                    "epoch": 1,
                    "reconstructor_architecture": "particle_cnn",
                    "model_state_dict": model.state_dict(),
                    "model_config": model.config.to_dict(),
                },
                path,
            )
            loaded, payload = load_teacher_logit_reconstructor_checkpoint(
                path,
                device=torch.device("cpu"),
                expected_architecture="pcnn",
            )
            self.assertIsInstance(loaded, ParticleCnnReconstructor)
            self.assertEqual(payload["epoch"], 1)

    def test_expected_architecture_mismatch_raises_before_model_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pn_checkpoint.pt"
            torch.save({"reconstructor_architecture": "particle_net", "model_state_dict": {}}, path)
            with self.assertRaises(ValueError):
                load_teacher_logit_reconstructor_checkpoint(
                    path,
                    device=torch.device("cpu"),
                    expected_architecture="global_transformer",
                )


if __name__ == "__main__":
    unittest.main()
