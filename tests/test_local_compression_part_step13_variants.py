import math
import unittest

from jetclass_fresh.hlt_baseline import ParticleTransformerHLTClassifier, require_torch
from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM

from teacher_logit_reco.local_compression_part import (
    LOCAL_COMPRESSION_CANONICAL_FEATURE_NAMES,
    LOCAL_COMPRESSION_GATE_CONTEXT_SIGMOID,
    LOCAL_COMPRESSION_GATE_NONE,
    LOCAL_COMPRESSION_MODALITIES,
    LOCAL_COMPRESSION_PRIMARY_METRIC,
    LOCAL_COMPRESSION_VARIANT_BASELINE_RECHECK,
    LOCAL_COMPRESSION_VARIANT_CONTEXT_DELTA_NO_MODALITIES,
    LOCAL_COMPRESSION_VARIANT_CONTEXT_GATED,
    LOCAL_COMPRESSION_VARIANT_LOCAL_NO_CONTEXT,
    LOCAL_COMPRESSION_VARIANT_MLP_DELTA,
    LOCAL_COMPRESSION_VARIANT_RANDOM_GROUPING,
    LocalCompressionFeatureAdapterParT,
    LocalCompressionFeatureConfig,
    LocalCompressionPartConfig,
    LocalCompressionTaggerTrainConfig,
    default_local_compression_modality_specs,
    local_compression_tagger_checkpoint_payload,
    randomized_local_compression_modality_specs,
)


torch = require_torch()


class DummyReferencePart(ParticleTransformerHLTClassifier):
    def __init__(self, num_classes: int = 2):
        torch.nn.Module.__init__(self)
        self.config = {"dummy_reference_part": True, "num_classes": int(num_classes)}
        self.linear = torch.nn.Linear(len(LOCAL_COMPRESSION_CANONICAL_FEATURE_NAMES), int(num_classes))
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.bias.zero_()
            self.linear.weight[0, 0] = 0.35
            self.linear.weight[0, 5] = -0.15
            self.linear.weight[1, 0] = -0.25
            self.linear.weight[1, 7] = 0.45

    def forward(self, points, features, lorentz_vectors, mask):
        del points, lorentz_vectors
        rows = features.transpose(1, 2).contiguous()
        particle_mask = mask.squeeze(1).to(dtype=rows.dtype)
        pooled = (rows * particle_mask[:, :, None]).sum(dim=1) / particle_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return self.linear(pooled)


def make_tokens(num_particles: int = 6):
    tokens = torch.zeros((2, num_particles, RAW_TOKEN_DIM), dtype=torch.float32)
    mask = torch.zeros((2, num_particles), dtype=torch.bool)
    mask[0, :5] = True
    mask[1, :4] = True
    for batch_index in range(2):
        for particle_index in range(int(mask[batch_index].sum().item())):
            pt = 18.0 + 2.5 * particle_index + 0.7 * batch_index
            eta = -0.25 + 0.07 * particle_index
            phi = -math.pi + 0.18 * particle_index
            tokens[batch_index, particle_index, 0] = pt
            tokens[batch_index, particle_index, 1] = eta
            tokens[batch_index, particle_index, 2] = phi
            tokens[batch_index, particle_index, 3] = pt * math.cosh(eta) + 0.4
            tokens[batch_index, particle_index, 4] = 1.0 if particle_index % 2 == 0 else -1.0
            tokens[batch_index, particle_index, 5 + (particle_index % 5)] = 1.0
            tokens[batch_index, particle_index, 10] = 0.02 * particle_index
            tokens[batch_index, particle_index, 11] = 0.04 + 0.01 * particle_index
            tokens[batch_index, particle_index, 12] = -0.02 * particle_index
            tokens[batch_index, particle_index, 13] = 0.06 + 0.01 * batch_index
    return tokens, mask


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


def spec_signature(specs):
    return {
        spec.name: (
            tuple(spec.raw_feature_names),
            tuple(spec.pf_feature_names),
            tuple(spec.derived_feature_names),
            int(spec.field_count),
        )
        for spec in specs
    }


class LocalCompressionStep13VariantTests(unittest.TestCase):
    def test_baseline_recheck_forces_zero_delta_even_if_adapter_changes(self):
        tokens, mask = make_tokens()
        part_model = DummyReferencePart()
        model = LocalCompressionFeatureAdapterParT(
            small_config(variant=LOCAL_COMPRESSION_VARIANT_BASELINE_RECHECK, gate_mode=LOCAL_COMPRESSION_GATE_NONE),
            part_model=part_model,
        )
        model.eval()
        canonical = model.build_canonical_inputs(tokens, mask, max_constits=tokens.shape[1])

        with torch.no_grad():
            baseline_logits = part_model(
                canonical.points,
                canonical.features,
                canonical.lorentz_vectors,
                canonical.mask,
            )
            model.adapter.projector[-1].bias.fill_(0.75)
            output = model(tokens, mask, return_outputs=True, max_constits=tokens.shape[1])

        self.assertEqual(output.diagnostics()["variant"], LOCAL_COMPRESSION_VARIANT_BASELINE_RECHECK)
        self.assertTrue(output.diagnostics()["variant_behavior"]["forces_zero_delta"])
        self.assertEqual(float(output.delta_output.delta_F_rows.abs().sum().item()), 0.0)
        self.assertTrue(torch.allclose(output.logits, baseline_logits, atol=1.0e-6))

    def test_variant_paths_route_modalities_context_and_gates(self):
        tokens, mask = make_tokens()
        cases = (
            (LOCAL_COMPRESSION_VARIANT_MLP_DELTA, LOCAL_COMPRESSION_GATE_NONE, False, False, False, False),
            (LOCAL_COMPRESSION_VARIANT_LOCAL_NO_CONTEXT, LOCAL_COMPRESSION_GATE_NONE, True, True, False, False),
            (
                LOCAL_COMPRESSION_VARIANT_CONTEXT_DELTA_NO_MODALITIES,
                LOCAL_COMPRESSION_GATE_NONE,
                False,
                False,
                True,
                False,
            ),
            (
                LOCAL_COMPRESSION_VARIANT_CONTEXT_GATED,
                LOCAL_COMPRESSION_GATE_CONTEXT_SIGMOID,
                True,
                True,
                True,
                True,
            ),
        )
        for variant, gate_mode, uses_modalities, uses_local, uses_context, uses_gates in cases:
            with self.subTest(variant=variant):
                model = LocalCompressionFeatureAdapterParT(
                    small_config(variant=variant, gate_mode=gate_mode),
                    part_model=DummyReferencePart(),
                )
                model.eval()
                with torch.no_grad():
                    output = model(tokens, mask, return_outputs=True, max_constits=tokens.shape[1])
                behavior = output.diagnostics()["variant_behavior"]

                self.assertEqual(behavior["uses_modalities"], uses_modalities)
                self.assertEqual(behavior["uses_local_compressor"], uses_local)
                self.assertEqual(behavior["uses_particle_context"], uses_context)
                self.assertEqual(behavior["uses_context_gates"], uses_gates)
                if not uses_local:
                    self.assertTrue(
                        all(name.startswith("feature_anchor") for name in output.compressor_output.modality_names)
                    )
                if not uses_context:
                    self.assertTrue(
                        torch.allclose(
                            output.context_output.context_tokens,
                            output.pool_output.local_particle_token,
                            atol=1.0e-6,
                        )
                    )

    def test_random_grouping_preserves_modality_names_and_sizes_but_changes_sources(self):
        default_specs = default_local_compression_modality_specs()
        random_specs = randomized_local_compression_modality_specs(seed=2907)
        default_signature = spec_signature(default_specs)
        random_signature = spec_signature(random_specs)

        self.assertEqual(tuple(spec.name for spec in random_specs), LOCAL_COMPRESSION_MODALITIES)
        self.assertEqual(
            {name: signature[-1] for name, signature in random_signature.items()},
            {name: signature[-1] for name, signature in default_signature.items()},
        )
        changed_sources = [
            name
            for name in LOCAL_COMPRESSION_MODALITIES
            if random_signature[name][:-1] != default_signature[name][:-1]
        ]
        self.assertTrue(changed_sources)

        train_config = LocalCompressionTaggerTrainConfig(
            output_dir="out",
            manifest_path="split_manifest.json.gz",
            hlt_cache_dir="hlt_cache",
            baseline_checkpoint="baseline.pt",
            confirm_split_settings=True,
            confirm_final_test=True,
            variant=LOCAL_COMPRESSION_VARIANT_RANDOM_GROUPING,
            random_grouping_seed=2907,
            label_names=("QCD", "Hgg"),
            label_filter=(0, 1),
            embed_dim=16,
            local_heads=4,
            context_heads=4,
            dropout=0.0,
            attention_dropout=0.0,
        )
        model_config = train_config.model_config()

        self.assertEqual(model_config.gate_mode, LOCAL_COMPRESSION_GATE_CONTEXT_SIGMOID)
        self.assertEqual(spec_signature(model_config.feature_config.modalities), random_signature)

    def test_checkpoint_payload_allows_no_optimizer_for_baseline_recheck(self):
        model = LocalCompressionFeatureAdapterParT(
            small_config(variant=LOCAL_COMPRESSION_VARIANT_BASELINE_RECHECK, gate_mode=LOCAL_COMPRESSION_GATE_NONE),
            part_model=DummyReferencePart(),
        )
        train_config = LocalCompressionTaggerTrainConfig(
            output_dir="out",
            manifest_path="split_manifest.json.gz",
            hlt_cache_dir="hlt_cache",
            baseline_checkpoint="baseline.pt",
            confirm_split_settings=True,
            confirm_final_test=True,
            variant=LOCAL_COMPRESSION_VARIANT_BASELINE_RECHECK,
            label_names=("QCD", "Hgg"),
            label_filter=(0, 1),
            embed_dim=16,
            local_heads=4,
            context_heads=4,
            dropout=0.0,
            attention_dropout=0.0,
        )
        payload = local_compression_tagger_checkpoint_payload(
            model,
            None,
            epoch=0,
            config=train_config,
            metrics={"model_val": {"accuracy": 1.0}},
            source={"test": True},
            baseline_load_report={"baseline_checkpoint_selection_metric": LOCAL_COMPRESSION_PRIMARY_METRIC},
        )

        self.assertIsNone(payload["optimizer_state_dict"])
        self.assertEqual(payload["variant"], LOCAL_COMPRESSION_VARIANT_BASELINE_RECHECK)
        self.assertTrue(payload["variant_behavior"]["forces_zero_delta"])

    def test_invalid_random_grouping_feature_config_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must use context_sigmoid"):
            small_config(
                variant=LOCAL_COMPRESSION_VARIANT_RANDOM_GROUPING,
                gate_mode=LOCAL_COMPRESSION_GATE_NONE,
                feature_config=LocalCompressionFeatureConfig(),
            )


if __name__ == "__main__":
    unittest.main()
