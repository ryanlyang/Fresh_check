import math
import tempfile
import unittest
from pathlib import Path

from jetclass_fresh.hlt_baseline import ParticleTransformerHLTClassifier, require_torch
from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM

from teacher_logit_reco.local_compression_part import (
    LOCAL_COMPRESSION_CANONICAL_FEATURE_NAMES,
    LOCAL_COMPRESSION_CHECKPOINT_CONTRACT,
    LOCAL_COMPRESSION_PART_HLT_DEGRADATION_STRENGTH,
    LOCAL_COMPRESSION_PRIMARY_METRIC,
    LocalCompressionFeatureAdapterParT,
    LocalCompressionPartConfig,
    compute_init_logit_diff_vs_baseline,
    load_hlt_part_baseline_checkpoint,
    sha256_file,
    warm_start_local_compression_part_model,
)


torch = require_torch()


class DummyReferencePart(ParticleTransformerHLTClassifier):
    """Small is-a ParticleTransformerHLTClassifier for checkpoint tests."""

    def __init__(self, num_classes: int = 2):
        torch.nn.Module.__init__(self)
        self.config = {"dummy_reference_part": True, "num_classes": int(num_classes)}
        self.linear = torch.nn.Linear(len(LOCAL_COMPRESSION_CANONICAL_FEATURE_NAMES), int(num_classes))
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.bias.zero_()
            self.linear.weight[0, 0] = 0.7
            self.linear.weight[0, 5] = -0.2
            self.linear.weight[1, 1] = -0.4
            self.linear.weight[1, 7] = 0.3

    def forward(self, points, features, lorentz_vectors, mask):
        del points, lorentz_vectors
        rows = features.transpose(1, 2).contiguous()
        particle_mask = mask.squeeze(1).to(dtype=rows.dtype)
        pooled = (rows * particle_mask[:, :, None]).sum(dim=1) / particle_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return self.linear(pooled)


def small_config(**kwargs):
    payload = {
        "embed_dim": 16,
        "local_layers": 1,
        "local_heads": 4,
        "context_layers": 1,
        "context_heads": 4,
        "dropout": 0.0,
        "attention_dropout": 0.0,
    }
    payload.update(kwargs)
    return LocalCompressionPartConfig(**payload)


def make_tokens(num_particles: int = 6):
    tokens = torch.zeros((2, num_particles, RAW_TOKEN_DIM), dtype=torch.float32)
    mask = torch.zeros((2, num_particles), dtype=torch.bool)
    mask[0, :5] = True
    mask[1, :4] = True
    for batch_index in range(2):
        for particle_index in range(int(mask[batch_index].sum().item())):
            pt = 16.0 + 3.0 * particle_index + 0.8 * batch_index
            eta = -0.2 + 0.08 * particle_index
            phi = -math.pi + 0.16 * particle_index
            tokens[batch_index, particle_index, 0] = pt
            tokens[batch_index, particle_index, 1] = eta
            tokens[batch_index, particle_index, 2] = phi
            tokens[batch_index, particle_index, 3] = pt * math.cosh(eta) + 0.5
            tokens[batch_index, particle_index, 4] = 1.0 if particle_index % 2 == 0 else -1.0
            tokens[batch_index, particle_index, 5 + (particle_index % 5)] = 1.0
            tokens[batch_index, particle_index, 10] = 0.015 * particle_index
            tokens[batch_index, particle_index, 11] = 0.05 + 0.01 * particle_index
            tokens[batch_index, particle_index, 12] = -0.025 * particle_index
            tokens[batch_index, particle_index, 13] = 0.08 + 0.01 * batch_index
    return tokens, mask


def write_checkpoint(path: Path, model: torch.nn.Module, *, prefixed: bool = False, hlt_strength: float = 0.6):
    state = model.state_dict()
    if prefixed:
        state = {f"part_model.{key}": value.clone() for key, value in state.items()}
    torch.save(
        {
            "model_state_dict": state,
            "selection_metric": LOCAL_COMPRESSION_PRIMARY_METRIC,
            "hlt_degradation_strength": float(hlt_strength),
            "split_manifest_hash": "split-hash-123",
            "model_config": dict(getattr(model, "config", {})),
        },
        path,
    )


class LocalCompressionStep11CheckpointTests(unittest.TestCase):
    def test_loads_direct_checkpoint_into_part_model_and_records_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best_model_val.pt"
            source = DummyReferencePart()
            write_checkpoint(path, source)
            target = DummyReferencePart()
            with torch.no_grad():
                target.linear.weight.fill_(9.0)
                target.linear.bias.fill_(-4.0)

            report = load_hlt_part_baseline_checkpoint(
                path,
                target,
                expected_split_manifest_hash="split-hash-123",
                require_metadata=True,
            )

            self.assertEqual(report.contract, LOCAL_COMPRESSION_CHECKPOINT_CONTRACT)
            self.assertEqual(report.baseline_checkpoint_hash, sha256_file(path))
            self.assertEqual(report.baseline_checkpoint_selection_metric, LOCAL_COMPRESSION_PRIMARY_METRIC)
            self.assertEqual(
                report.baseline_checkpoint_hlt_degradation_strength,
                LOCAL_COMPRESSION_PART_HLT_DEGRADATION_STRENGTH,
            )
            self.assertEqual(report.baseline_checkpoint_split_manifest_hash, "split-hash-123")
            self.assertEqual(report.missing_key_count, 0)
            self.assertEqual(report.loaded_key_count, len(source.state_dict()))
            for key, value in source.state_dict().items():
                self.assertTrue(torch.allclose(target.state_dict()[key], value))

    def test_loads_prefixed_wrapper_checkpoint_into_inner_part_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best_model_val.pt"
            source = DummyReferencePart()
            write_checkpoint(path, source, prefixed=True)
            model = LocalCompressionFeatureAdapterParT(small_config(), part_model=DummyReferencePart())
            with torch.no_grad():
                model.part_model.linear.weight.fill_(3.0)
                model.part_model.linear.bias.fill_(2.0)

            report = warm_start_local_compression_part_model(
                model,
                path,
                expected_split_manifest_hash="split-hash-123",
                require_metadata=True,
            )

            self.assertEqual(report.missing_key_count, 0)
            self.assertEqual(model.baseline_checkpoint_report["baseline_checkpoint_path"], str(path))
            for key, value in source.state_dict().items():
                self.assertTrue(torch.allclose(model.part_model.state_dict()[key], value))

    def test_metadata_mismatch_raises_before_training_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best_model_val.pt"
            write_checkpoint(path, DummyReferencePart(), hlt_strength=1.0)

            with self.assertRaisesRegex(ValueError, "hlt_degradation_strength mismatch"):
                load_hlt_part_baseline_checkpoint(
                    path,
                    DummyReferencePart(),
                    require_metadata=True,
                )

    def test_init_logit_diff_is_zero_for_zero_delta_adapter_and_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best_model_val.pt"
            write_checkpoint(path, DummyReferencePart(), prefixed=True)
            tokens, mask = make_tokens()
            model = LocalCompressionFeatureAdapterParT(small_config(), part_model=DummyReferencePart())

            warm_start_local_compression_part_model(model, path, expected_split_manifest_hash="split-hash-123")
            diff = compute_init_logit_diff_vs_baseline(model, tokens, mask, max_constits=tokens.shape[1])
            output = model(tokens, mask, return_outputs=True, max_constits=tokens.shape[1])
            diagnostics = output.diagnostics()
            config_dict = model.to_config_dict()

            self.assertTrue(diff["allclose_atol_1e_6"])
            self.assertLessEqual(diff["max_abs_logit_diff"], 1.0e-6)
            self.assertEqual(
                diagnostics["baseline_checkpoint_selection_metric"],
                LOCAL_COMPRESSION_PRIMARY_METRIC,
            )
            self.assertEqual(
                diagnostics["baseline_checkpoint_split_manifest_hash"],
                "split-hash-123",
            )
            self.assertTrue(diagnostics["init_logit_diff_vs_baseline"]["allclose_atol_1e_6"])
            self.assertTrue(config_dict["init_logit_diff_vs_baseline"]["allclose_atol_1e_6"])


if __name__ == "__main__":
    unittest.main()
