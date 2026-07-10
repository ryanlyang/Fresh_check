import tempfile
import unittest
from pathlib import Path

from jetclass_fresh.hlt_baseline import require_torch
from teacher_logit_reco.target_denoising_part import (
    TARGET_DENOISING_MODEL_CONTRACT,
    TARGET_DENOISING_TAGGER_CONTRACT,
    TARGET_DENOISING_TRAINING_CONTRACT,
    TARGET_DENOISING_VARIANT_DENOISER_FEATURES_FROZEN,
    TARGET_DENOISING_VARIANT_DENOISER_LOCAL_KERNEL_ONLY,
    TARGET_DENOISING_VARIANT_DENOISER_NO_PAIR_BIAS,
    TARGET_DENOISING_VARIANT_FEATURE_MLP_ADAPTER_TAG_ONLY,
    TARGET_DENOISING_VARIANT_HLT_PART_BASELINE,
    TargetConditionedDenoiserConfig,
    TargetDenoisingAugmentedParT,
    TargetDenoisingAugmentedParTConfig,
    load_target_denoising_pretrained_checkpoint,
)
from teacher_logit_reco.target_denoising_part.tagger import _tensor_quantile


class FakeParticleTransformerPart(require_torch().nn.Module):
    def __init__(self, *, embed_dim=12, num_classes=3):
        torch = require_torch()
        super().__init__()
        self.config = {"fake_part": True, "embed_dim": int(embed_dim), "num_classes": int(num_classes)}
        self.embed = torch.nn.Linear(17, int(embed_dim))
        self.classifier = torch.nn.Linear(int(embed_dim), int(num_classes))

    def no_weight_decay(self):
        return {"fake_no_decay"}

    def forward(self, points, features, lorentz_vectors, mask):
        del points, lorentz_vectors
        valid = mask[:, 0, :].bool()
        rows = features.transpose(1, 2).contiguous()
        embedded = self.embed(rows)
        embedded = embedded * valid[:, :, None].to(dtype=embedded.dtype)
        denom = valid.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=embedded.dtype)
        pooled = embedded.sum(dim=1) / denom
        return self.classifier(pooled)


def _tokens(batch=2, particles=5):
    torch = require_torch()
    tokens = torch.zeros((batch, particles, 14), dtype=torch.float32)
    mask = torch.ones((batch, particles), dtype=torch.bool)
    mask[:, -1] = False
    for row in range(batch):
        for col in range(particles):
            tokens[row, col, 0] = 15.0 + row + col
            tokens[row, col, 1] = -0.3 + 0.02 * row + 0.05 * col
            tokens[row, col, 2] = -2.0 + 0.4 * col
            tokens[row, col, 3] = tokens[row, col, 0] + 3.0
            tokens[row, col, 4] = 1.0 if col % 2 == 0 else -1.0
            tokens[row, col, 5 + (col % 5)] = 1.0
    tokens = tokens * mask[:, :, None].to(dtype=tokens.dtype)
    return tokens, mask


def _denoiser_config():
    return TargetConditionedDenoiserConfig(
        embed_dim=16,
        num_heads=4,
        pair_hidden_dim=8,
        head_hidden_dim=16,
        dropout=0.0,
        attention_dropout=0.0,
        zero_init=True,
    )


def _tagger_config(variant, **kwargs):
    values = {
        "variant": variant,
        "num_classes": 3,
        "model_size": "tiny",
        "part_embed_dim": 12,
        "max_constits": 5,
        "adapter_hidden_dim": 16,
        "adapter_dropout": 0.0,
        "denoiser_config": _denoiser_config(),
    }
    values.update(kwargs)
    return TargetDenoisingAugmentedParTConfig(**values)


class TargetDenoisingPartStep4TaggerTests(unittest.TestCase):
    def test_tensor_quantile_accepts_amp_half_tensors(self):
        torch = require_torch()
        values = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float16)

        quantile = _tensor_quantile(values, 0.5)

        self.assertEqual(quantile.dtype, torch.float32)
        self.assertAlmostEqual(float(quantile.detach().cpu().item()), 1.5, places=5)

    def test_hlt_part_baseline_matches_direct_part_forward(self):
        torch = require_torch()
        torch.manual_seed(11)
        tokens, mask = _tokens()
        part = FakeParticleTransformerPart()
        model = TargetDenoisingAugmentedParT(
            _tagger_config(TARGET_DENOISING_VARIANT_HLT_PART_BASELINE),
            part_model=part,
        )

        output = model.forward_outputs(tokens, mask)
        direct = part(
            output.canonical_inputs.points,
            output.canonical_inputs.features,
            output.canonical_inputs.lorentz_vectors,
            output.canonical_inputs.mask,
        )

        self.assertEqual(output.output_contract, TARGET_DENOISING_TAGGER_CONTRACT)
        self.assertTrue(bool(torch.allclose(output.logits, direct)))
        self.assertIsNone(output.denoiser_output)
        self.assertIsNone(output.delta_h)
        self.assertFalse(output.diagnostics()["uses_denoiser"])

    def test_feature_mlp_adapter_zero_init_recovers_baseline_then_can_inject(self):
        torch = require_torch()
        torch.manual_seed(13)
        tokens, mask = _tokens()
        part = FakeParticleTransformerPart()
        model = TargetDenoisingAugmentedParT(
            _tagger_config(TARGET_DENOISING_VARIANT_FEATURE_MLP_ADAPTER_TAG_ONLY),
            part_model=part,
        )

        zero_output = model.forward_outputs(tokens, mask)
        baseline = part(
            zero_output.canonical_inputs.points,
            zero_output.canonical_inputs.features,
            zero_output.canonical_inputs.lorentz_vectors,
            zero_output.canonical_inputs.mask,
        )
        self.assertTrue(bool(torch.allclose(zero_output.logits, baseline)))
        self.assertEqual(tuple(zero_output.delta_h.shape), (2, 5, 12))
        self.assertEqual(zero_output.injection_summary["injection_applied"], True)

        with torch.no_grad():
            model.embedding_adapter.network[-1].bias.fill_(0.2)
        changed = model.forward_outputs(tokens, mask)
        self.assertGreater(float((changed.logits - baseline).abs().sum().detach().cpu().item()), 1.0e-6)

    def test_frozen_denoiser_variant_uses_denoiser_outputs_without_changing_inputs(self):
        torch = require_torch()
        torch.manual_seed(17)
        tokens, mask = _tokens()
        model = TargetDenoisingAugmentedParT(
            _tagger_config(TARGET_DENOISING_VARIANT_DENOISER_FEATURES_FROZEN),
            part_model=FakeParticleTransformerPart(),
        )

        self.assertTrue(all(not parameter.requires_grad for parameter in model.denoiser.parameters()))
        output = model.forward_outputs(tokens, mask)
        diagnostics = output.diagnostics()

        self.assertIsNotNone(output.denoiser_output)
        self.assertEqual(tuple(output.delta_h.shape), (2, 5, 12))
        self.assertTrue(diagnostics["original_hlt_part_inputs_unchanged"])
        self.assertTrue(diagnostics["uses_denoiser"])
        self.assertEqual(model.variant_behavior()["freezes_denoiser"], True)
        self.assertEqual(output.denoiser_output.summary()["contract"], TARGET_DENOISING_MODEL_CONTRACT)

    def test_pair_bias_ablation_variants_resolve_denoiser_config(self):
        no_pair = TargetDenoisingAugmentedParT(
            _tagger_config(TARGET_DENOISING_VARIANT_DENOISER_NO_PAIR_BIAS),
            part_model=FakeParticleTransformerPart(),
        )
        local_only = TargetDenoisingAugmentedParT(
            _tagger_config(TARGET_DENOISING_VARIANT_DENOISER_LOCAL_KERNEL_ONLY),
            part_model=FakeParticleTransformerPart(),
        )

        self.assertFalse(no_pair.denoiser.config.use_pair_bias)
        self.assertFalse(no_pair.denoiser.config.use_local_kernel)
        self.assertFalse(local_only.denoiser.config.use_pair_bias)
        self.assertTrue(local_only.denoiser.config.use_local_kernel)

    def test_loads_step3_denoiser_checkpoint_into_wrapper(self):
        torch = require_torch()
        torch.manual_seed(23)
        model = TargetDenoisingAugmentedParT(
            _tagger_config(TARGET_DENOISING_VARIANT_DENOISER_FEATURES_FROZEN),
            part_model=FakeParticleTransformerPart(),
        )
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "best_denoiser_model_val.pt"
            torch.save(
                {
                    "output_contract": TARGET_DENOISING_TRAINING_CONTRACT,
                    "model_contract": TARGET_DENOISING_MODEL_CONTRACT,
                    "epoch": 3,
                    "metrics": {"normalized_rmse": 0.12},
                    "model_state_dict": model.denoiser.state_dict(),
                },
                checkpoint,
            )

            report = load_target_denoising_pretrained_checkpoint(checkpoint, model)

        self.assertEqual(report["checkpoint_epoch"], 3)
        self.assertEqual(report["missing_keys"], [])
        self.assertEqual(report["unexpected_keys"], [])
        self.assertEqual(model.denoiser_checkpoint_report["metrics"]["normalized_rmse"], 0.12)


if __name__ == "__main__":
    unittest.main()
