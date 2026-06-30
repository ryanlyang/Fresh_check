import unittest

import numpy as np

from jetclass_fresh.hlt_baseline import require_torch
from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM

from teacher_logit_reco.local_graph_part import (
    LOCAL_GRAPH_RESIDUAL_V2_CONDITION_CONTROL_SHUFFLED,
    LOCAL_GRAPH_RESIDUAL_V2_INPUT_EMBEDDING_ONLY,
    LOCAL_GRAPH_RESIDUAL_V2_INPUT_LOCAL_ONLY,
    LOCAL_GRAPH_RESIDUAL_V2_LABEL_CONTROL_SHUFFLED,
    LocalGraphResidualExpertV2Config,
    LocalGraphResidualExpertV2TrainConfig,
    LocalGraphResidualV2BaselineEmbeddingBlock,
    build_local_graph_residual_expert_v2,
)
from teacher_logit_reco.local_graph_part.residual_v2_train import _v2_cache_arrays_for_batch


def _tokens(batch_size: int = 4, particles: int = 7):
    torch = require_torch()
    tokens = np.zeros((batch_size, particles, RAW_TOKEN_DIM), dtype=np.float32)
    mask = np.zeros((batch_size, particles), dtype=bool)
    for batch in range(batch_size):
        mask[batch, :5] = True
        for particle in range(5):
            tokens[batch, particle, 0] = 18.0 + batch + particle
            tokens[batch, particle, 1] = -0.3 + 0.05 * particle
            tokens[batch, particle, 2] = 0.1 * batch + 0.04 * particle
            tokens[batch, particle, 3] = tokens[batch, particle, 0] * np.cosh(tokens[batch, particle, 1])
            tokens[batch, particle, 5 + (particle % 5)] = 1.0
    return torch.from_numpy(tokens), torch.from_numpy(mask)


def _baseline_inputs(batch_size: int = 4, embedding_dim: int = 8):
    torch = require_torch()
    z_base = torch.linspace(-1.0, 1.0, steps=batch_size, dtype=torch.float32)
    embedding = torch.randn(batch_size, embedding_dim, generator=torch.Generator().manual_seed(713))
    condition = torch.stack(
        (
            z_base,
            torch.sigmoid(z_base),
            z_base - 0.1,
            torch.abs(z_base - 0.1),
            z_base - 0.7,
            torch.exp(-torch.abs(z_base - 0.1)),
        ),
        dim=1,
    )
    return z_base, embedding, condition


def _config(mode: str = "full") -> LocalGraphResidualExpertV2Config:
    return LocalGraphResidualExpertV2Config(
        baseline_embedding_dim=8,
        max_constits=7,
        k=3,
        local_embed_dim=16,
        local_heads=4,
        local_context_dim=12,
        condition_embed_dim=5,
        residual_hidden_dim=18,
        dropout=0.0,
        attention_dropout=0.0,
        residual_dropout=0.0,
        gamma_learnable=False,
        residual_input_mode=mode,
    )


def _block() -> LocalGraphResidualV2BaselineEmbeddingBlock:
    labels = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64)
    margin = np.asarray([-0.8, 0.7, -0.4, 0.5, -0.2, 0.3], dtype=np.float32)
    logits = np.stack((-0.5 * margin, 0.5 * margin), axis=1)
    embedding = np.arange(labels.shape[0] * 8, dtype=np.float32).reshape(labels.shape[0], 8) / 10.0
    condition = np.stack(
        (
            margin,
            1.0 / (1.0 + np.exp(-margin)),
            margin,
            np.abs(margin),
            margin - 0.6,
            np.exp(-np.abs(margin) / 0.25),
        ),
        axis=1,
    ).astype(np.float32)
    return LocalGraphResidualV2BaselineEmbeddingBlock(
        split="model_train",
        logits=logits.astype(np.float32),
        embedding=embedding,
        labels=labels,
        indices=np.arange(labels.shape[0], dtype=np.int64),
        metadata={
            "condition_reference": {
                "source_split": "model_train",
                "tau50": 0.0,
                "tau30": 0.6,
                "near_tau50_scale": 0.25,
                "feature_names": [],
            },
            "label_names": ["QCD", "Hgg"],
            "positive_class_name": "Hgg",
            "positive_class_index": 1,
            "required_embedding_role": "true_hlt_part_penultimate_embedding",
            "embedding_dim": 8,
        },
        condition_features_array=condition,
    )


class LocalGraphResidualV2Step13ControlsTest(unittest.TestCase):
    def test_embedding_only_control_ignores_local_particle_tokens(self):
        torch = require_torch()
        torch.manual_seed(811)
        model = build_local_graph_residual_expert_v2(_config(LOCAL_GRAPH_RESIDUAL_V2_INPUT_EMBEDDING_ONLY))
        model.eval()
        tokens, mask = _tokens()
        z_base, embedding, condition = _baseline_inputs()
        with torch.no_grad():
            first = model(
                tokens.float(),
                mask.bool(),
                baseline_logit=z_base,
                baseline_embedding=embedding,
                baseline_condition_features=condition,
                return_outputs=True,
            ).fused_logits
            second = model(
                (tokens + 100.0).float(),
                mask.bool(),
                baseline_logit=z_base,
                baseline_embedding=embedding,
                baseline_condition_features=condition,
                return_outputs=True,
            ).fused_logits
        torch.testing.assert_close(first, second)

    def test_local_only_control_ignores_baseline_embedding_vector(self):
        torch = require_torch()
        torch.manual_seed(812)
        model = build_local_graph_residual_expert_v2(_config(LOCAL_GRAPH_RESIDUAL_V2_INPUT_LOCAL_ONLY))
        model.eval()
        tokens, mask = _tokens()
        z_base, embedding, condition = _baseline_inputs()
        with torch.no_grad():
            first = model(
                tokens.float(),
                mask.bool(),
                baseline_logit=z_base,
                baseline_embedding=embedding,
                baseline_condition_features=condition,
                return_outputs=True,
            ).fused_logits
            second = model(
                tokens.float(),
                mask.bool(),
                baseline_logit=z_base,
                baseline_embedding=embedding + 50.0,
                baseline_condition_features=condition,
                return_outputs=True,
            ).fused_logits
        torch.testing.assert_close(first, second)

    def test_shuffled_condition_control_changes_condition_features_only(self):
        torch = require_torch()
        block = _block()
        indices = torch.arange(6, dtype=torch.long)
        labels = torch.from_numpy(block.labels)
        z_normal, embedding_normal, condition_normal = _v2_cache_arrays_for_batch(
            block,
            indices,
            labels=labels,
            device=torch.device("cpu"),
        )
        z_shuf, embedding_shuf, condition_shuf = _v2_cache_arrays_for_batch(
            block,
            indices,
            labels=labels,
            device=torch.device("cpu"),
            condition_control_mode=LOCAL_GRAPH_RESIDUAL_V2_CONDITION_CONTROL_SHUFFLED,
            condition_shuffle_seed=1234,
        )
        torch.testing.assert_close(z_normal, z_shuf)
        torch.testing.assert_close(embedding_normal, embedding_shuf)
        self.assertGreater(float((condition_normal - condition_shuf).detach().abs().sum()), 0.0)

    def test_train_config_accepts_control_modes_and_rejects_bad_modes(self):
        config = LocalGraphResidualExpertV2TrainConfig(
            output_dir="out",
            hlt_cache_dir="hlt",
            baseline_embedding_cache_dir="emb",
            confirm_split_settings=True,
            residual_input_mode=LOCAL_GRAPH_RESIDUAL_V2_INPUT_EMBEDDING_ONLY,
            condition_control_mode=LOCAL_GRAPH_RESIDUAL_V2_CONDITION_CONTROL_SHUFFLED,
            label_control_mode=LOCAL_GRAPH_RESIDUAL_V2_LABEL_CONTROL_SHUFFLED,
        )
        self.assertEqual(config.residual_input_mode, LOCAL_GRAPH_RESIDUAL_V2_INPUT_EMBEDDING_ONLY)
        self.assertEqual(config.condition_control_mode, LOCAL_GRAPH_RESIDUAL_V2_CONDITION_CONTROL_SHUFFLED)
        self.assertEqual(config.label_control_mode, LOCAL_GRAPH_RESIDUAL_V2_LABEL_CONTROL_SHUFFLED)

        with self.assertRaisesRegex(ValueError, "residual_input_mode"):
            LocalGraphResidualExpertV2TrainConfig(
                output_dir="out",
                hlt_cache_dir="hlt",
                baseline_embedding_cache_dir="emb",
                confirm_split_settings=True,
                residual_input_mode="bad",
            )


if __name__ == "__main__":
    unittest.main()
